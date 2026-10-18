"""Small, fully controlled worlds for the realism layer: an E. coli-like organism on a six-channel field."""
import dataclasses

import numpy as np

from microbiome.sim.model import AgentState, DT_S, NutrientEnvironment, Phenotype, make_rngs
from microbiome.sim.realism import ChannelMap, Physiology, RealismConfig, RealisticSimulation

K = 6
CH = ChannelMap()
GLC, LAC, O2, ACE, TOX, SIG = CH.glucose, CH.lactose, CH.oxygen, CH.acetate, CH.toxin, CH.signal
MOLAR = np.array([180.156, 342.296, 31.998, 60.052, 270.28, 134.13])
DIFFUSION = np.array([600.0, 450.0, 1400.0, 1100.0, 400.0, 500.0])
GENOME = 1111


def traits(**changes):
    values = dict(
        max_uptake_mmol_s=np.array([8.88e-16, 4.67e-16, 3.0e-15, 6.0e-16, 0.0, 0.0]),
        storage_fraction=np.array([0.1, 0.1, 0.0, 0.1, 0.0, 0.0]),
        biomass_fraction=np.array([0.75, 0.75, 0.0, 0.5, 0.0, 0.0]),
        energy_yield_pj_per_mmol=np.array([1e15, 1e15, 0.0, 3.3e14, 0.0, 0.0]),
        motility_um_s=10.0, chemotaxis_sensitivity=0.85, random_motility_fraction=0.15,
        division_rate_s=2e-4, division_mass_pg=0.6, death_rate_s=1e-5, max_age_s=86400.0,
        antagonism_radius_um=0.0, antagonism_rate_s=0.0,
    )
    values.update(changes)
    return Phenotype(**values)


def physiology(**changes):
    values = dict(aerobic_capacity=1.0, fermentation_capacity=1.0, acetate_overflow=0.4, ccr_strength=1.0,
                  stress_resistance=1.0, sporulation_capacity=0.0, competence_capacity=0.0, biofilm_capacity=1.0,
                  qs_capacity=1.0, ph_min=4.4, toxin_production=0.0, toxin_susceptibility=1.0,
                  half_saturation_mmol_l=(0.01, 0.2, 0.002, 0.3, 1.0, 1.0))
    values.update(changes)
    return Physiology(**values)


MAINTENANCE_MMOL_GLC_GDW_H = 0.4      # a typical Pirt coefficient (see data/calibration_targets.json); tests use it where cells must starve


def maintaining(**changes):
    """A physiology that pays maintenance (so it can starve); the default ``physiology()`` has none, which keeps rate tests exact."""
    return physiology(maintenance_mmol_glc_gdw_h=MAINTENANCE_MMOL_GLC_GDW_H, **changes)


def maintenance_pj_s(mass_pg, m=MAINTENANCE_MMOL_GLC_GDW_H):
    """Maintenance power of a cell of this dry mass: glucose equivalents per gram per hour through the glucose energy yield."""
    return m / 3600.0 * mass_pg * 1e-12 * 1e15


def environment(nx=20, ny=20, dx=10.0, depth=10.0, **initial):
    """Closed periodic slab; oxygen starts at saturation. ``initial`` is channel-name -> uniform mmol/L or array."""
    names = {"glucose": GLC, "lactose": LAC, "oxygen": O2, "acetate": ACE, "toxin": TOX, "signal": SIG}
    conc = np.zeros((K, ny, nx))
    conc[O2] = 0.21
    for name, value in initial.items():
        conc[names[name]] = value
    return NutrientEnvironment(nutrient_id=np.arange(K, dtype=np.uint32), concentration_mmol_l=conc,
                               diffusion_um2_s=DIFFUSION, molar_mass_g_mol=MOLAR, dx_um=dx, dy_um=dx, depth_um=depth,
                               width_um=nx * dx, height_um=ny * dx)


def cells(positions, *, mass=0.3, genome=GENOME, energy=None, first_id=1, species=0):
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    n = len(positions)
    biomass = np.zeros((n, K))
    biomass[:, GLC] = mass / (MOLAR[GLC] * 1e9)
    mass_pg = biomass @ (MOLAR * 1e9)
    return AgentState(
        agent_id=np.arange(first_id, first_id + n, dtype=np.uint64), positions_um=positions,
        velocities_um_s=np.zeros((n, 2)), mass_pg=mass_pg, radius_um=np.cbrt(3 * mass_pg / (4 * np.pi)),
        energy_pj=np.zeros(n) if energy is None else np.broadcast_to(np.asarray(energy, dtype=np.float64), (n,)).copy(),
        species_id=np.full(n, species, dtype=np.uint32), genome_id=np.full(n, genome, dtype=np.uint64),
        age_s=np.zeros(n), nutrient_store_mmol=np.zeros((n, K)), biomass_mmol=biomass)


class World:
    """state + environment + simulation + rngs, advanced one realistic tick at a time."""

    def __init__(self, state, env, sim, rngs, phenotypes):
        self.state, self.env, self.sim, self.rngs, self.phenotypes = state, env, sim, rngs, phenotypes

    def step(self, ticks=1):
        for _ in range(ticks):
            self.sim.step(self.state, self.env, self.phenotypes, self.rngs)

    @property
    def n(self):
        return self.state.n_agents

    def set_field(self, channel, values):
        """Replace a channel's concentration the lawful way: the added (or removed) mass is booked as a source (sink)."""
        old = float(self.env.concentration_mmol_l[channel].sum()) * self.env.cell_volume_l
        self.env.concentration_mmol_l[channel] = values
        new = float(self.env.concentration_mmol_l[channel].sum()) * self.env.cell_volume_l
        if not self.state.ledger_initialized:   # the first tick captures the field as the initial total; booking now would double count
            return
        if new >= old:
            self.state.source_cum_mmol[channel] += new - old
        else:
            self.state.sink_cum_mmol[channel] += old - new

    def field_mmol(self, channel):
        return float(self.env.concentration_mmol_l[channel].sum()) * self.env.cell_volume_l


def world(positions=((100.0, 100.0),), *, nx=20, ny=20, seed=7, config=None, trait=None, phys=None, mass=0.3,
          energy=None, genome=GENOME, **field):
    env = environment(nx, ny, **field)
    state = cells(positions, mass=mass, energy=energy, genome=genome)
    sim = RealisticSimulation({genome: phys or physiology()}, config or RealismConfig())
    return World(state, env, sim, make_rngs(seed), {genome: trait or traits()})


def config(**changes):
    return dataclasses.replace(RealismConfig(), **changes)


def quiet(**changes):
    """A config with the random and mechanical extras off, for tests that isolate one mechanism."""
    base = dict(collisions=False, translational_diffusion_um2_s=0.0, rotational_diffusion_rad2_s=0.0)
    base.update(changes)
    return config(**base)
