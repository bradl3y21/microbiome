"""Opt-in realism layer: physics and physiology the frozen D1 step does not model.

The default D1 step is untouched. ``RealisticSimulation`` is a drop-in alternative per-tick implementation
(``engine.step(..., simulation=...)``) that reuses the engine's allocator, ledger, division and detritus code, so
every mechanism below runs under the same per-tick, per-nutrient mass-conservation assertion.

Mechanisms (details, equations and sources in docs/REALISM.md):

* movement: run-and-tumble chemotaxis with temporal sensing and adaptation, rotational and translational Brownian
  motion, optional uniform flow, steric collisions and EPS-mediated cohesion
* chemistry: Michaelis-Menten uptake, catabolite repression, oxygen with gas exchange, aerobic/anaerobic yield,
  acetate overflow and re-uptake, pH from acetate, a diffusing toxin that decays, an AI-2-like signal, necromass
  recycling
* physiology: lag, maintenance energy and starvation, stationary-phase resistance, sporulation and germination,
  quorum-triggered biofilm matrix, toxin/acid damage and protection

Every numeric parameter here is an explicit calibration, not a measurement; ``RealismConfig`` states each one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping

import numpy as np

from .contracts import SimulationState
from .engine import (
    _add_dead_to_detritus,
    _contract_step,
    _derive_mass,
    _divide,
    _register_variant_phenotype,
    _field_amounts,
    _initialize_ledger,
    _internal_ledger,
    _proportional_grants,
    _wrapped_cell_indices,
    diffuse,
)
from .model import (
    AGENT_ARRAY_FIELDS,
    DT_S,
    RNG_ANTAGONISM,
    RNG_DEATH,
    RNG_DIVISION,
    RNG_MOTILITY,
    RNG_MUTATION,
    AgentState,
    NutrientEnvironment,
    PhenotypeArrays,
    phenotype_arrays,
    validate_rngs,
)
from .spatial import SpatialHash

CELL_DENSITY_PG_UM3 = 1.0  # the engine's own mass-to-radius convention (see SimulationController._introduce)


# ------------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class ChannelMap:
    """Which field channel plays which chemical role. The demo uses ids 0..5 in this order."""

    glucose: int = 0
    lactose: int = 1
    oxygen: int = 2
    acetate: int = 3
    toxin: int = 4
    signal: int = 5
    toxin2: int | None = None      # a second diffusible toxin (pyocyanin); None in worlds that do not have it

    @property
    def carbon(self) -> tuple[int, int, int]:
        return (self.glucose, self.lactose, self.acetate)

    @property
    def size(self) -> int:
        extra = () if self.toxin2 is None else (self.toxin2,)
        return max(self.glucose, self.lactose, self.oxygen, self.acetate, self.toxin, self.signal, *extra) + 1


CARBON_PER_MMOL = np.array([6.0, 12.0, 2.0])   # glucose, lactose, acetate
O2_PER_MMOL = np.array([6.0, 12.0, 2.0])       # complete oxidation: C6H12O6 + 6 O2, C12H22O11 + 12 O2, C2H4O2 + 2 O2


@dataclass(frozen=True)
class RealismConfig:
    """Calibration of the realism layer. None of these are fitted to data; see docs/REALISM.md."""

    # --- environment
    oxygen_saturation_mmol_l: float = 0.21        # air-saturated water near 37 C
    gas_exchange_per_s: float = 0.05              # relaxation of oxygen toward saturation (thin film, open to air)
    toxin_decay_per_s: float = math.log(2.0) / 1800.0
    signal_decay_per_s: float = math.log(2.0) / 3600.0
    detritus_mineralization_per_s: float = 2.0e-5  # necromass returns to the field with a ~14 h time constant
    flow_um_s: tuple[float, float] = (0.0, 0.0)
    ph_neutral: float = 7.0
    ph_buffer_mmol_l: float = 20.0                # pH = neutral - log10(1 + acetate / buffer)
    ph_optimum: float = 6.8
    # --- movement
    swim_speed_scale: float = 3.000001  # D2 motility (10 um/s) x this = 30 um/s, the modern tracking value; fitted, see docs/CALIBRATION.md
    tumble_rate_per_s: float = 1.5625    # 1 / the 0.64 s mean run (Taute 2015)
    tumble_angle_mean_deg: float = 60.666173  # underlying normal; the clipped angle has mean 62.5 deg
    tumble_angle_sd_deg: float = 39.405613  # underlying normal; the clipped angle has sd 36 deg
    chemotaxis_gain: float = 9.179488  # fitted: peak drift 30 % of swim speed
    adaptation_time_s: float = 2.0
    sensing_half_saturation_mmol_l: float = 0.005
    translational_diffusion_um2_s: float = 0.4
    rotational_diffusion_rad2_s: float = 0.15
    # --- contact mechanics
    cell_width_um: float = 0.8                    # rod width; E. coli and B. subtilis are both ~0.7-1 um wide
    cell_volume_um3_per_pg: float = 3.3           # wet volume per pg of dry mass (a 0.3 pg cell is ~1 um^3)
    collisions: bool = True
    collision_stiffness: float = 0.5
    collision_interval_ticks: int = 2             # contacts are resolved every n-th tick, n times as firmly (same stability, 1/n the cost)
    cohesion_range_um: float = 0.3
    cohesion_strength: float = 0.1
    cluster_separation_rate_per_s: float = 2.0e-5  # chance per second that a bond between sister cells is cut (autolysin separation; assumed, 14 h on average)
    cluster_stiffness: float = 0.5                # how fast a stretched bond pulls its two cells back to contact
    contact_signal_hill: float = 4.0              # steepness of the contact response (assumed; a switch-like response is what lets slow cells gather)
    contact_signal_half_neighbours: float = 4.0   # a starving, signalling cell with this many touching neighbours is half-way to full contact signalling (assumed)
    # --- metabolism
    anaerobic_yield_ratio: float = 0.3
    acetate_carbon_recovery: float = 0.7
    overflow_half_saturation_mmol_l: float = 0.5
    ccr_half_saturation_mmol_l: float = 0.05
    derepression_rise_tau_s: float = 600.0        # enzyme synthesis after glucose is gone: the diauxic lag
    derepression_fall_tau_s: float = 30.0         # inducer exclusion shuts the alternative sugar off quickly
    reference_mass_pg: float = 0.3                # the D2 per-cell uptake rates are for a cell of this dry mass; uptake scales with mass
    initial_energy_reserve_s: float = 7200.0      # an energy reserve of this many seconds of maintenance
    # --- lifecycle
    lag_initial: float = 0.05
    lag_recovery_tau_s: float = 1200.0
    lag_decay_tau_s: float = 10800.0
    lag_carbon_threshold_mmol_l: float = 1.0e-3
    starvation_tau_s: float = 1800.0
    starvation_hazard_per_s: float = 5.0e-5
    division_stress_limit: float = 0.5
    division_commitment_s: float = 30.0
    sporulation_stress: float = 0.6
    sporulation_time_s: float = 28800.0
    germination_rate_per_s: float = 1.0 / 600.0
    germination_threshold_mmol_l: float = 0.01
    spore_hazard_factor: float = 1.0e-3
    # --- signalling, biofilm, toxin, acid
    signal_production_mmol_s: float = 1.0e-18
    quorum_half_saturation_mmol_l: float = 2.0e-4
    eps_rise_tau_s: float = 900.0
    eps_fall_tau_s: float = 7200.0
    toxin_production_mmol_s: float = 5.0e-17
    toxin_half_saturation_mmol_l: float = 1.0e-4
    toxin_max_hazard_per_s: float = 0.01
    toxin2_decay_per_s: float = math.log(2.0) / 3600.0   # the second toxin (pyocyanin): a redox-active phenazine, more persistent
    toxin2_production_mmol_s: float = 5.0e-17
    toxin2_half_saturation_mmol_l: float = 1.0e-4
    toxin2_max_hazard_per_s: float = 0.01
    acid_hazard_per_s_per_ph: float = 1.0e-3
    # --- gliding, predation and starvation signalling
    reversal_rate_per_s: float = 0.002222         # a gliding cell reverses about every 7.5 min (fitted, see data/calibration_fit.json)
    reversal_suppression: float = 0.0             # how much activation suppresses reversals; 0 because published reversal periods inside and outside aggregates are equal
    development_speed_factor: float = 3.130435      # a fully starved developing glider moves this much faster than a growing one (0.12 against 0.0383 um/s)
    aggregate_speed_ratio: float = 0.191667         # and this fraction as fast when fully activated by contact, inside an aggregate (0.023 / 0.12)
    predation_reach_um: float = 0.5               # a predator reaches this far beyond touching
    predation_rate_per_s: float = 1.0 / 450.0     # chance per second that a predator in contact kills prey: a kill takes 5-10 min
    predation_assimilation: float = 0.135302      # share of a killed cell's biomass the predator takes up; the rest is released (fitted, data/calibration_fit.json)
    predation_max_intake_per_s: float = 1.8e-3    # a predator can take in at most this fraction of its own mass per second of feeding (assumed)
    predation_feeding_window_s: float = 281.0     # seconds a predator stays on a prey cell: the measured residence time (Zhang 2020)
    spore_predation_factor: float = 0.05          # spores are killed this much as readily as vegetative cells
    # --- natural transformation (needs mutation on: variants are what it moves)
    transformation_rate_per_s: float = 2.0e-3     # chance per second that a competent cell takes up DNA lying in its grid cell
    competence_stress: float = 0.2                # starvation stress that makes a competence-capable cell competent ...
    competence_signal: float = 0.1                # ... or a quorum activation this high
    dna_lifetime_s: float = 1800.0                # free DNA is degraded by nucleases
    transformation_window_bp: int = 50_000        # length of the recombination tract
    dna_pool_limit: int = 20_000


@dataclass(frozen=True)
class Physiology:
    """One genome's realism traits (derived from annotated genes by ``genome.physiology``)."""

    aerobic_capacity: float = 0.0
    fermentation_capacity: float = 0.0
    acetate_overflow: float = 0.0
    acetate_uptake_capacity: float = 0.0
    ccr_strength: float = 0.0
    stress_resistance: float = 0.0
    sporulation_capacity: float = 0.0
    competence_capacity: float = 0.0
    biofilm_capacity: float = 0.0
    qs_capacity: float = 0.0
    ph_min: float = 5.8
    toxin_production: float = 0.0
    toxin_susceptibility: float = 1.0
    toxin2_production: float = 0.0
    toxin2_susceptibility: float = 1.0
    predation_capacity: float = 0.0               # 0 none, 1 a full predator (gene-derived)
    gliding: float = 0.0                          # 1: a surface glider that reverses instead of tumbling
    starvation_signalling: float = 0.0            # 1: signal production rises with starvation (a developmental signal)
    width_um: float = 0.0                         # cell width; 0 = the configured default (organism profile, not genes)
    round_cell: float = 0.0                       # 1: a coccus (length = width = diameter of the sphere of the cell's volume)
    imaging_style: float = 0.0                    # index into the microscope's colour tables (organism profile, not genes)
    run_time_s: float = 0.0                       # own mean run time (calibration); 0 = the shared tumble rate
    turn_mean_deg: float = 0.0                    # own mean reorientation angle; 0 = the shared distribution
    turn_sd_deg: float = 0.0
    cluster: float = 0.0                          # 1: daughters stay attached to the mother (staphylococci); organism profile, not genes
    maintenance_mmol_glc_gdw_h: float = 0.0       # Pirt non-growth maintenance, glucose equivalents (0 = none)
    half_saturation_mmol_l: tuple[float, ...] = ()


_SCALAR_FIELDS = tuple(f for f in Physiology.__dataclass_fields__ if f != "half_saturation_mmol_l")


@dataclass
class RealismSetup:
    """Everything the controller needs to run the realistic world, bundled so it never learns about genes."""

    simulation: "RealisticSimulation"
    channel_names: tuple[str, ...]
    widen_table: Callable[[object], object]       # D2 two-nutrient PhenotypeTable -> all realism channels
    founder_channels: tuple[int, ...] = (0, 1)    # which channels a newly introduced cell's biomass is made of


# ------------------------------------------------------------------------------------------ pure kinetics
def hill(x: np.ndarray, half: float, n: float = 2.0) -> np.ndarray:
    """Hill activation x^n / (K^n + x^n), 0 at x = 0 and 1/2 at x = K."""
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    xn = x ** n
    return xn / (half ** n + xn)


def michaelis_menten(s: np.ndarray, km: np.ndarray) -> np.ndarray:
    """Saturating uptake fraction s / (Km + s)."""
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    return s / (km + s)


def ph_from_acetate(acetate_mmol_l: np.ndarray, config: RealismConfig) -> np.ndarray:
    """pH of a buffered medium that accumulates a weak organic acid: neutral - log10(1 + [acid] / buffer)."""
    return config.ph_neutral - np.log10(1.0 + np.maximum(acetate_mmol_l, 0.0) / config.ph_buffer_mmol_l)


def ph_growth_factor(ph: np.ndarray, ph_min: np.ndarray, ph_optimum: float, steepness: float = 3.0) -> np.ndarray:
    """Relative growth at a given pH: 0 at ``ph_min``, 1 at the optimum and above, saturating in between.

    f(x) = (1 - exp(-s x)) / (1 - exp(-s)) with x = (pH - pH_min) / (pH_opt - pH_min). With s = 3 an organism whose
    minimum is 4.4 grows at ~90% of its maximum at pH 6, ~55% at pH 5 and ~10% at pH 4.5, the shape of measured
    E. coli pH profiles; a linear ramp would wrongly halve growth at a mildly acidic pH.
    """
    x = np.clip((np.asarray(ph, dtype=np.float64) - ph_min) / (ph_optimum - ph_min), 0.0, 1.0)
    return (1.0 - np.exp(-steepness * x)) / (1.0 - math.exp(-steepness))


def sample_bilinear(field: np.ndarray, positions_um: np.ndarray, env: NutrientEnvironment) -> np.ndarray:
    """Periodic bilinear interpolation of a [ny, nx] cell-centred field, or of a [m, ny, nx] stack (result [m, N]), at (N, 2) points."""
    fx = positions_um[:, 0] / env.dx_um - 0.5
    fy = positions_um[:, 1] / env.dy_um - 0.5
    x0 = np.floor(fx)
    y0 = np.floor(fy)
    wx = fx - x0
    wy = fy - y0
    ix0 = x0.astype(np.int64) % env.nx
    iy0 = y0.astype(np.int64) % env.ny
    ix1 = (ix0 + 1) % env.nx
    iy1 = (iy0 + 1) % env.ny
    return ((1.0 - wy) * ((1.0 - wx) * field[..., iy0, ix0] + wx * field[..., iy0, ix1])
            + wy * ((1.0 - wx) * field[..., iy1, ix0] + wx * field[..., iy1, ix1]))


def rod_length_um(mass_pg: np.ndarray, config: "RealismConfig") -> np.ndarray:
    """Total length (um) of a capsule of fixed width whose volume is the cell's: a growing rod elongates, it does not widen.

    A 0.3 pg cell is ~2.2 um long and a 0.6 pg cell about to divide ~4.2 um, as measured for E. coli. Below the volume of a
    sphere of the cell's width it is a sphere.
    """
    width = config.cell_width_um
    volume = config.cell_volume_um3_per_pg * np.maximum(np.asarray(mass_pg, dtype=np.float64), 0.0)
    cylinder = np.maximum(volume - np.pi * width ** 3 / 6.0, 0.0) / (np.pi * width ** 2 / 4.0)
    return cylinder + width


def cell_dimensions(mass_pg: np.ndarray, config: "RealismConfig", width_um: np.ndarray | None = None,
                    round_cell: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(length, width) in um per cell: a capsule of the organism's width, or for a coccus a sphere of the cell's volume."""
    volume = config.cell_volume_um3_per_pg * np.maximum(np.asarray(mass_pg, dtype=np.float64), 0.0)
    width = np.full(volume.shape, config.cell_width_um) if width_um is None else np.where(width_um > 0.0, width_um, config.cell_width_um)
    cylinder = np.maximum(volume - np.pi * width ** 3 / 6.0, 0.0) / (np.pi * width ** 2 / 4.0)
    length = cylinder + width
    if round_cell is not None:
        diameter = np.cbrt(6.0 * volume / np.pi)
        sphere = round_cell > 0.5
        length = np.where(sphere, diameter, length)
        width = np.where(sphere, diameter, width)
    return length, width


def segment_closest_points(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points of the segments ``p1 + s d1`` and ``p2 + t d2`` (s, t in [0, 1]), vectorised over rows of (M, 2) arrays."""
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    r = p1 - p2
    a, e, f, c, b = dot(d1, d1), dot(d2, d2), dot(d2, r), dot(d1, r), dot(d1, d2)
    eps = 1e-12
    safe = lambda x: np.where(np.abs(x) > eps, x, 1.0)
    denom = a * e - b * b
    s = np.where(denom > eps, np.clip((b * f - c * e) / safe(denom), 0.0, 1.0), 0.0)
    t = np.where(e > eps, (b * s + f) / safe(e), 0.0)
    to_start = (e <= eps) | (t < 0.0)
    to_end = (e > eps) & (t > 1.0)
    t = np.clip(t, 0.0, 1.0)
    s_at_start = np.where(a > eps, np.clip(-c / safe(a), 0.0, 1.0), 0.0)
    s_at_end = np.where(a > eps, np.clip((b - c) / safe(a), 0.0, 1.0), 0.0)
    s = np.where(to_start, s_at_start, np.where(to_end, s_at_end, s))
    return p1 + s[:, None] * d1, p2 + t[:, None] * d2


def radius_from_mass(mass_pg: np.ndarray) -> np.ndarray:
    return np.cbrt(3.0 * np.maximum(mass_pg, 0.0) / (4.0 * np.pi * CELL_DENSITY_PG_UM3))


def advect_upwind(field: np.ndarray, ux_um_s: float, uy_um_s: float, env: NutrientEnvironment, dt_s: float) -> np.ndarray:
    """Conservative first-order upwind advection by a uniform flow on the periodic grid (x then y)."""
    cx = ux_um_s * dt_s / env.dx_um
    cy = uy_um_s * dt_s / env.dy_um
    if abs(cx) > 1.0 or abs(cy) > 1.0:
        raise ValueError("flow violates the advective CFL limit |u| dt / dx <= 1")
    out = field
    if cx:
        out = out - cx * (out - np.roll(out, 1, axis=-1)) if cx > 0 else out - cx * (np.roll(out, -1, axis=-1) - out)
    if cy:
        out = out - cy * (out - np.roll(out, 1, axis=-2)) if cy > 0 else out - cy * (np.roll(out, -1, axis=-2) - out)
    return out


# ------------------------------------------------------------------------------------------ life state
_LIFE_FIELDS = ("heading", "memory", "lag", "stress_s", "spore", "eps", "derepression", "contact")


@dataclass
class LifeState:
    """Per-agent realism state, aligned to agent rows by ``agent_id`` (rows are sorted by id every tick)."""

    ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint64))
    heading: np.ndarray = field(default_factory=lambda: np.empty(0))      # swimming direction, radians
    memory: np.ndarray = field(default_factory=lambda: np.empty(0))       # adapted attractant level; <0 means unset
    lag: np.ndarray = field(default_factory=lambda: np.empty(0))          # 0..1 growth readiness
    stress_s: np.ndarray = field(default_factory=lambda: np.empty(0))     # seconds of recent starvation
    spore: np.ndarray = field(default_factory=lambda: np.empty(0))        # 0 vegetative .. 1 dormant spore
    eps: np.ndarray = field(default_factory=lambda: np.empty(0))          # 0..1 biofilm matrix level
    derepression: np.ndarray = field(default_factory=lambda: np.empty(0))  # 0..1 expression of alternative-sugar genes; <0 unset
    contact: np.ndarray = field(default_factory=lambda: np.empty(0))      # touching neighbours of the same genome at the last contact pass

    def aligned(self, agent_id: np.ndarray, config: RealismConfig) -> tuple["LifeState", np.ndarray]:
        """A copy aligned to ``agent_id``; rows never seen before (e.g. introduced by the user) get defaults."""
        n = agent_id.size
        if self.ids.size:
            idx = np.minimum(np.searchsorted(self.ids, agent_id), self.ids.size - 1)
            known = self.ids[idx] == agent_id
        else:
            idx = np.zeros(n, dtype=np.intp)
            known = np.zeros(n, dtype=bool)
        hashed = ((agent_id * np.uint64(2654435761)) % np.uint64(3600)).astype(np.float64) * (2.0 * np.pi / 3600.0)
        defaults = {"heading": hashed, "memory": np.full(n, -1.0), "lag": np.full(n, config.lag_initial),
                    "stress_s": np.zeros(n), "spore": np.zeros(n), "eps": np.zeros(n),
                    "derepression": np.full(n, -1.0), "contact": np.zeros(n)}
        out = LifeState(ids=agent_id.copy())
        for name in _LIFE_FIELDS:
            current = getattr(self, name)
            value = defaults[name].copy()
            if known.any() and current.size == self.ids.size:       # a field a caller did not supply keeps its default
                value[known] = current[idx[known]]
            setattr(out, name, value)
        return out, ~known

    def keep(self, mask: np.ndarray) -> None:
        self.ids = self.ids[mask]
        for name in _LIFE_FIELDS:
            setattr(self, name, getattr(self, name)[mask])


# ------------------------------------------------------------------------------------------ the simulation
class RealisticSimulation:
    """One 1/60-second realistic tick. Same signature as ``engine.Simulation.step``."""

    dt_s = DT_S

    def __init__(self, physiology: Mapping[int, Physiology], config: RealismConfig | None = None,
                 channels: ChannelMap | None = None) -> None:
        self.config = config or RealismConfig()
        self.channels = channels or ChannelMap()
        self.flow_um_s = tuple(float(v) for v in self.config.flow_um_s)
        self.life = LifeState()
        self.stats: dict[str, int] = {}
        self.transformer = None     # an object with create_transformant (see genome.variants); None disables transformation
        self.dna = _empty_pool()
        self.bond_a = np.empty(0, dtype=np.uint64)       # sister cells that have not separated: agent ids of each bond's two ends
        self.bond_b = np.empty(0, dtype=np.uint64)
        self._set_physiology(physiology)

    # -------------------------------------------------------------------------- setup
    def _set_physiology(self, physiology: Mapping[int, Physiology]) -> None:
        if not physiology:
            raise ValueError("at least one genome physiology is required")
        k = self.channels.size
        ids = np.asarray(sorted(int(i) for i in physiology), dtype=np.uint64)
        rows = [physiology[int(i)] for i in ids]
        for row in rows:
            if len(row.half_saturation_mmol_l) < k or min(row.half_saturation_mmol_l) <= 0:
                raise ValueError(f"half_saturation_mmol_l must give a positive value for each of the {k} channels")
        self._phys_ids = ids
        self._phys = {name: np.asarray([getattr(r, name) for r in rows], dtype=np.float64) for name in _SCALAR_FIELDS}
        self._half = np.asarray([r.half_saturation_mmol_l[:k] for r in rows], dtype=np.float64)

    def _gather(self, genome_id: np.ndarray) -> dict[str, np.ndarray]:
        rows = np.searchsorted(self._phys_ids, genome_id)
        if genome_id.size and (np.any(rows >= self._phys_ids.size) or np.any(self._phys_ids[rows] != genome_id)):
            raise ValueError("no physiology for a genome present in the population")
        out = {name: values[rows] for name, values in self._phys.items()}
        out["half"] = self._half[rows]
        return out

    def imaging_styles(self, genome_id: np.ndarray) -> np.ndarray:
        return self._gather(np.asarray(genome_id, dtype=np.uint64))["imaging_style"].astype(np.uint8)

    def cell_shape(self, genome_id: np.ndarray, mass_pg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(length, width) in um for cells of these genomes and dry masses."""
        phys = self._gather(np.asarray(genome_id, dtype=np.uint64))
        return cell_dimensions(mass_pg, self.config, phys["width_um"], phys["round_cell"])

    # -------------------------------------------------------------------------- the tick
    def step(self, state, env, phenotypes, rngs, mutation_provider=None) -> None:
        if isinstance(state, SimulationState):
            _contract_step(state, env, phenotypes, rngs, mutation_provider, simulation=self)
            return None
        cfg, ch = self.config, self.channels
        self._rngs = rngs
        validate_rngs(rngs)
        state.canonicalize()
        env.validate(DT_S)
        state.validate(env)
        if env.n_nutrients < ch.size:
            raise ValueError(f"the realism layer needs {ch.size} field channels, the environment has {env.n_nutrients}")
        n0 = state.n_agents
        traits = phenotype_arrays(phenotypes, state.genome_id, env.n_nutrients, state.variant_id)
        _initialize_ledger(state, env)
        phys = self._gather(state.genome_id)
        life, fresh = self.life.aligned(state.agent_id, cfg)

        division_draw = rngs[RNG_DIVISION].random(n0)
        division_angle = rngs[RNG_DIVISION].random(n0) * (2.0 * np.pi)
        death_draw = rngs[RNG_DEATH].random(n0)
        life_draw = rngs[RNG_DEATH].random(n0)
        motility = rngs[RNG_MOTILITY]
        noise = {"brown": motility.standard_normal((n0, 2)), "rot": motility.standard_normal(n0),
                 "tumble_u": motility.random(n0), "tumble_z": motility.standard_normal(n0),
                 "tumble_s": motility.random(n0)}

        stats: dict[str, int] = {}
        diffuse(env, DT_S)
        self._advect(env)
        self._environment_sources(state, env)

        div_gate = np.zeros(n0, dtype=bool)
        if n0:
            div_gate = self._cells_act(state, env, traits, phys, life, fresh, noise, life_draw, stats, phenotypes)
            state.age_s += DT_S
            dead = self._death(state, env, traits, phys, life, div_gate, death_draw, stats)
            self._release_dna(state, env, dead & (life.spore < 1.0))
            _add_dead_to_detritus(state, env, dead)
            if np.any(dead):
                survivors = ~dead
                for name in AGENT_ARRAY_FIELDS:
                    setattr(state, name, np.ascontiguousarray(getattr(state, name)[survivors]))
                life.keep(survivors)
                division_draw, division_angle = division_draw[survivors], division_angle[survivors]
                div_gate = div_gate[survivors]
                traits = PhenotypeArrays(**{name: getattr(traits, name)[survivors]
                                            for name in PhenotypeArrays.__dataclass_fields__})
            traits = replace(traits, division_rate_s=np.where(div_gate, 1.0 / cfg.division_commitment_s, 0.0))
            before = state.positions_um.copy()
            parents = _divide(state, env, traits, division_draw, division_angle, phenotypes=phenotypes,
                              mutation_provider=mutation_provider,
                              mutation_rng=rngs[RNG_MUTATION] if mutation_provider is not None else None,
                              origin_tick=state.tick)
            if parents.size:
                stats["births"] = int(parents.size)
                life.ids = np.concatenate((life.ids, state.agent_id[-parents.size:]))
                for name in _LIFE_FIELDS:
                    value = getattr(life, name)
                    child = value[parents]
                    if name == "heading":                                # a daughter keeps the axis, with a hair of noise
                        child = child + 0.02 * (division_angle[parents] - np.pi)
                    setattr(life, name, np.concatenate((value, child)))
                kids = np.arange(state.n_agents - parents.size, state.n_agents)
                round_p = self._gather(state.genome_id[parents])["round_cell"] > 0.5
                if round_p.any():                                   # cocci divide in alternating perpendicular planes
                    life.heading[parents[round_p]] += math.pi / 2.0
                    life.heading[kids[round_p]] += math.pi / 2.0
                self._lay_daughters_end_to_end(state, env, life, before, parents)
                sticky = self._gather(state.genome_id[parents])["cluster"] > 0.5
                if sticky.any():
                    self.bond_a = np.concatenate((self.bond_a, state.agent_id[parents[sticky]]))
                    self.bond_b = np.concatenate((self.bond_b, state.agent_id[kids[sticky]]))

        _derive_mass(state, env)
        state.radius_um = np.ascontiguousarray(radius_from_mass(state.mass_pg))
        state.tick += 1
        state.canonicalize()
        life.ids = state.agent_id.copy()
        self.life = life
        self.stats = stats
        state.validate(env)
        _internal_ledger(state, env).assert_conserved()

    def _lay_daughters_end_to_end(self, state, env, life, before: np.ndarray, parents: np.ndarray) -> None:
        """A dividing rod splits across its short axis: the two daughters sit end to end along the mother's axis."""
        _derive_mass(state, env)
        count = parents.size
        children = np.arange(state.n_agents - count, state.n_agents)
        length, _ = self.cell_shape(state.genome_id[parents], state.mass_pg[parents])      # the daughters' length (the mother's mass is halved)
        axis = np.column_stack((np.cos(life.heading[parents]), np.sin(life.heading[parents])))
        centre = before[parents]
        shift = 0.5 * length[:, None] * axis
        state.positions_um[parents] = env.wrap_positions(centre - shift)
        state.positions_um[children] = env.wrap_positions(centre + shift)

    # -------------------------------------------------------------------------- environment phases
    def _advect(self, env: NutrientEnvironment) -> None:
        ux, uy = self.flow_um_s
        if ux or uy:
            env.concentration_mmol_l = np.ascontiguousarray(
                advect_upwind(env.concentration_mmol_l, ux, uy, env, DT_S))

    def _environment_sources(self, state: AgentState, env: NutrientEnvironment) -> None:
        """Gas exchange, first-order decay and necromass recycling, each booked on the ledger."""
        cfg, ch, dt = self.config, self.channels, DT_S
        v = env.cell_volume_l
        conc = env.concentration_mmol_l
        delta = cfg.gas_exchange_per_s * dt * (cfg.oxygen_saturation_mmol_l - conc[ch.oxygen])
        conc[ch.oxygen] = conc[ch.oxygen] + delta
        state.source_cum_mmol[ch.oxygen] += float(np.clip(delta, 0.0, None).sum()) * v
        state.sink_cum_mmol[ch.oxygen] += float(np.clip(-delta, 0.0, None).sum()) * v
        decaying = [(ch.toxin, cfg.toxin_decay_per_s), (ch.signal, cfg.signal_decay_per_s)]
        if ch.toxin2 is not None:
            decaying.append((ch.toxin2, cfg.toxin2_decay_per_s))
        for channel, rate in decaying:
            lost = conc[channel] * (-math.expm1(-rate * dt))
            conc[channel] = conc[channel] - lost
            state.sink_cum_mmol[channel] += float(lost.sum()) * v
        for channel in ch.carbon:
            moved = env.detritus_mmol[channel] * (-math.expm1(-cfg.detritus_mineralization_per_s * dt))
            env.detritus_mmol[channel] = env.detritus_mmol[channel] - moved
            conc[channel] = conc[channel] + moved / v

    # -------------------------------------------------------------------------- the cells
    def _cells_act(self, state, env, traits, phys, life, fresh, noise, life_draw, stats, phenotypes=None) -> np.ndarray:
        """Physiology, metabolism and movement for one tick. Returns which cells may divide."""
        cfg, ch, dt = self.config, self.channels, DT_S
        n = state.n_agents
        v = env.cell_volume_l
        conc = env.concentration_mmol_l
        cell_id, ix, iy = _wrapped_cell_indices(env, state.positions_um)
        local = conc[:, iy, ix].T                                             # (N, K) concentrations at each cell
        g, l, o, a = ch.glucose, ch.lactose, ch.oxygen, ch.acetate
        carbon = list(ch.carbon)
        yield_c = traits.energy_yield_pj_per_mmol[:, carbon]
        # Pirt maintenance: glucose equivalents per gram dry mass per hour, converted to pJ/s through the glucose energy yield
        maintenance = (phys["maintenance_mmol_glc_gdw_h"] / 3600.0) * (state.mass_pg * 1e-12) * traits.energy_yield_pj_per_mmol[:, g]
        reserve = fresh & (state.energy_pj <= 0.0)
        state.energy_pj[reserve] = maintenance[reserve] * cfg.initial_energy_reserve_s

        # signalling, biofilm matrix, acidity
        sampled = sample_bilinear(conc[[ch.signal, ch.glucose, ch.lactose]], state.positions_um, env)   # (3, N), one pass
        signal_here = sampled[0]
        activation = hill(signal_here, cfg.quorum_half_saturation_mmol_l, 2.0) * phys["qs_capacity"]
        # a starving cell that signals by contact (myxobacterial C-signalling) is activated by touching neighbours, not by a diffusing level
        stress_before = -np.expm1(-life.stress_s / cfg.starvation_tau_s)
        by_contact = hill(life.contact, cfg.contact_signal_half_neighbours, cfg.contact_signal_hill) * stress_before * (phys["starvation_signalling"] > 0.5)
        activation = np.maximum(activation, by_contact * phys["qs_capacity"])
        target = phys["biofilm_capacity"] * activation
        tau = np.where(target > life.eps, cfg.eps_rise_tau_s, cfg.eps_fall_tau_s)
        life.eps = np.clip(life.eps + (target - life.eps) * dt / tau, 0.0, 1.0)
        ph = ph_from_acetate(local[:, a], cfg)
        ph_factor = ph_growth_factor(ph, phys["ph_min"], cfg.ph_optimum)

        # starvation, sporulation, germination, lag
        dormant = life.spore >= 1.0
        starving = (state.energy_pj <= 1e-12) & (maintenance > 0.0)
        life.stress_s = np.where(dormant, 0.0, np.where(starving, life.stress_s + dt, np.maximum(0.0, life.stress_s - 3.0 * dt)))
        stress = -np.expm1(-life.stress_s / cfg.starvation_tau_s)
        committing = ((phys["sporulation_capacity"] > 0) & ~dormant & (stress > cfg.sporulation_stress)
                      & ((activation > 0.2) | ((stress > 0.9) & (phys["starvation_signalling"] <= 0.5))))   # a developing cell sporulates only inside an aggregate
        progressing = (life.spore > 0) | committing
        life.spore = np.where(progressing & (life.spore < 1.0), np.minimum(1.0, life.spore + dt / cfg.sporulation_time_s), life.spore)
        dormant = life.spore >= 1.0
        germinate = (dormant & (local[:, g] > cfg.germination_threshold_mmol_l)
                     & (life_draw < -math.expm1(-cfg.germination_rate_per_s * dt)))
        if germinate.any():
            life.spore = np.where(germinate, 0.0, life.spore)
            life.lag = np.where(germinate, cfg.lag_initial, life.lag)
            life.stress_s = np.where(germinate, 0.0, life.stress_s)
            state.energy_pj = np.where(germinate, maintenance * 600.0, state.energy_pj)
            dormant = life.spore >= 1.0
            stats["germinations"] = int(germinate.sum())
        food = np.sum(local[:, carbon], axis=1) > cfg.lag_carbon_threshold_mmol_l
        life.lag = np.clip(np.where(food & ~dormant, life.lag + (1.0 - life.lag) * dt / cfg.lag_recovery_tau_s,
                                    life.lag - life.lag * dt / cfg.lag_decay_tau_s), 0.0, 1.0)
        gate = life.lag * ph_factor * (1.0 - life.spore)

        maintenance = np.where(dormant, 0.0, maintenance)
        capture, self._eaten = self._predation(state, env, phys, dormant, stats)
        stress_now = -np.expm1(-life.stress_s / cfg.starvation_tau_s)
        signal_factor = np.where(phys["starvation_signalling"] > 0.5, 0.05 + 0.95 * stress_now, 1.0)   # a developmental signal
        shortage = self._metabolise(state, env, traits, phys, life, local, cell_id, gate, activation, dormant, v, stats,
                                    maintenance * dt, capture, signal_factor)
        self._maintain(state, shortage, maintenance * cfg.initial_energy_reserve_s, yield_c)
        _derive_mass(state, env)
        state.radius_um = np.ascontiguousarray(radius_from_mass(state.mass_pg))
        self._move(state, env, traits, phys, life, noise, dormant, stats, sampled[1:], activation)
        interval = max(1, int(cfg.collision_interval_ticks))
        if cfg.collisions and state.tick % interval == 0:
            self._collide(state, env, life, float(interval), phys)
            self._pull_bonds(state, env, phys, float(interval))
        stress = -np.expm1(-life.stress_s / cfg.starvation_tau_s)
        stats["spores"] = int(dormant.sum())
        stats["sporulating"] = int(((life.spore > 0) & ~dormant).sum())
        stats["starving"] = int((stress > 0.5).sum())
        stats["biofilm"] = int((life.eps > 0.5).sum())
        self._transform(state, env, phys, life_draw, activation, stress, dormant, phenotypes, stats)
        return (~dormant) & (life.spore == 0.0) & (life.lag >= 0.5) & (stress < cfg.division_stress_limit) & (ph_factor > 0.3)

    def _maintain(self, state, shortage_pj, reserve_cap_pj, yield_c) -> None:
        """Pay the maintenance energy that uptake did not cover from the reserve, then from stored substrate (a ledger term).

        The reserve is capped (``reserve_cap_pj``, zero meaning no maintenance and no cap): surplus energy is dissipated.
        """
        carbon = list(self.channels.carbon)
        energy = state.energy_pj - shortage_pj
        short = energy < 0.0
        if short.any():
            store = state.nutrient_store_mmol[:, carbon]
            available = np.sum(store * yield_c, axis=1)
            need = np.where(short, -energy, 0.0)
            frac = np.where(available > 0.0, np.minimum(1.0, need / np.where(available > 0.0, available, 1.0)), 0.0)
            frac = np.minimum(frac, 1.0)
            burn = store * frac[:, None]
            state.nutrient_store_mmol[:, carbon] = store - burn
            state.respiration_loss_mmol[carbon] += burn.sum(axis=0)
            energy = energy + np.sum(burn * yield_c, axis=1)
        energy = np.maximum(energy, 0.0)
        energy = np.where(reserve_cap_pj > 0.0, np.minimum(energy, reserve_cap_pj), energy)
        state.energy_pj = np.ascontiguousarray(energy)

    def _metabolise(self, state, env, traits, phys, life, local, cell_id, gate, activation, dormant, v, stats,
                    maintenance_pj, capture, signal_factor) -> np.ndarray:
        """Uptake with saturation, repression and an oxygen budget; overflow, secretion and the ledger entries.

        Uptake is proportional to dry mass (the D2 rate is per ``reference_mass_pg``), so growth is exponential. Before
        anything is stored or built, a cell's maintenance energy is taken from what it ate and respired (Pirt); the
        returned array is the maintenance energy per cell that uptake could not cover this tick.
        """
        cfg, ch, dt = self.config, self.channels, DT_S
        g, l, o, a = ch.glucose, ch.lactose, ch.oxygen, ch.acetate
        carbon = list(ch.carbon)
        n, k = state.n_agents, env.n_nutrients
        ncell = env.nx * env.ny
        conc = env.concentration_mmol_l
        half = phys["half"]

        mm = michaelis_menten(local[:, carbon], half[:, carbon])
        repression = np.ones((n, 3))
        glucose_sat = local[:, g] / (local[:, g] + cfg.ccr_half_saturation_mmol_l)
        # Alternative sugars are expressed in proportion to how un-repressed the cell is, but the cell's enzyme level
        # lags the glucose level: shut off fast (inducer exclusion), switched on slowly (synthesis): the diauxic lag.
        target = 1.0 - phys["ccr_strength"] * glucose_sat
        level = np.where(life.derepression < 0.0, target, life.derepression)
        tau = np.where(target > level, cfg.derepression_rise_tau_s, cfg.derepression_fall_tau_s)
        life.derepression = np.clip(level + (target - level) * dt / tau, 0.0, 1.0)
        repression[:, 1] = repression[:, 2] = life.derepression
        mass_scale = state.mass_pg / cfg.reference_mass_pg
        request = traits.max_uptake_mmol_s[:, carbon] * dt * mm * repression * gate[:, None] * mass_scale[:, None]
        res_frac = np.clip(1.0 - traits.storage_fraction[:, carbon] - traits.biomass_fraction[:, carbon], 0.0, 1.0)

        yield_c = traits.energy_yield_pj_per_mmol[:, carbon]
        income = np.sum((request + capture) * yield_c, axis=1)     # captured prey is eaten like uptake, but costs the field nothing
        wanted = np.where(income > 0.0, np.minimum(1.0, maintenance_pj / np.where(income > 0.0, income, 1.0)), 0.0)
        # the oxygen the cell will need: maintenance and the respired share of growth, less what overflows as acetate
        saturation = local[:, [g, l]] / (local[:, [g, l]] + cfg.overflow_half_saturation_mmol_l)
        expected_overflow = np.concatenate(
            (np.clip(phys["acetate_overflow"][:, None] * saturation, 0.0, 1.0), np.zeros((n, 1))), axis=1)
        demand = np.sum(O2_PER_MMOL[None, :] * (wanted[:, None] + (1.0 - wanted[:, None]) * res_frac * (1.0 - expected_overflow))
                        * (request + capture), axis=1)
        o2_request = np.minimum(demand, traits.max_uptake_mmol_s[:, o] * dt * mass_scale
                                * michaelis_menten(local[:, o], half[:, o]) * (1.0 - life_spore_gate(dormant)))
        o2_available = conc[o].reshape(-1) * v
        o2_grant, _ = _proportional_grants(o2_request, o2_available, cell_id)
        have_demand = demand > 1e-300
        aerobic = np.where(have_demand, np.minimum(1.0, o2_grant / np.where(have_demand, demand, 1.0)), 1.0)
        ferment = phys["fermentation_capacity"]
        request = request * (aerobic + (1.0 - aerobic) * ferment)[:, None]

        available = (conc[carbon].transpose(1, 2, 0).reshape(-1)) * v
        keys = (cell_id[:, None] * 3 + np.arange(3, dtype=np.int64)[None, :]).reshape(-1)
        grants, granted = _proportional_grants(request.reshape(-1), available, keys)
        grant = grants.reshape(n, 3) + capture

        yield_factor = aerobic + (1.0 - aerobic) * cfg.anaerobic_yield_ratio
        energy_in = np.sum(grant * yield_c, axis=1) * yield_factor
        covered = np.where(energy_in > 0.0, np.minimum(1.0, maintenance_pj / np.where(energy_in > 0.0, energy_in, 1.0)), 0.0)
        kept = grant * covered[:, None]                                         # eaten to pay maintenance: all respired
        grant = grant - kept
        store = grant * traits.storage_fraction[:, carbon]
        biomass_raw = np.minimum(grant * traits.biomass_fraction[:, carbon], grant - store)
        biomass = biomass_raw * yield_factor[:, None]
        respired = grant - store - biomass
        overflow = np.clip(phys["acetate_overflow"][:, None] * saturation + ((1.0 - aerobic) * ferment)[:, None], 0.0, 1.0)
        acetate_made = cfg.acetate_carbon_recovery * (3.0 * respired[:, 0] * overflow[:, 0] + 6.0 * respired[:, 1] * overflow[:, 1])
        oxidised = np.stack([respired[:, 0] * (1.0 - overflow[:, 0]), respired[:, 1] * (1.0 - overflow[:, 1]), respired[:, 2]],
                            axis=1) + kept
        respired = respired + kept
        o2_used = np.minimum(np.sum(O2_PER_MMOL[None, :] * oxidised, axis=1), o2_grant)

        state.nutrient_store_mmol[:, carbon] += store
        state.biomass_mmol[:, carbon] += biomass
        state.energy_pj += (1.0 - covered) * energy_in
        state.respiration_loss_mmol[carbon] += respired.sum(axis=0)
        state.respiration_loss_mmol[o] += float(o2_used.sum())

        remaining = available - granted
        if np.any(remaining < -1e-18):
            raise FloatingPointError("uptake would produce a negative nutrient amount")
        conc[carbon] = np.maximum(remaining, 0.0).reshape(env.ny, env.nx, 3).transpose(2, 0, 1) / v
        used_cell = np.bincount(cell_id, weights=o2_used, minlength=ncell)
        o2_left = o2_available - used_cell
        if np.any(o2_left < -1e-18):
            raise FloatingPointError("oxygen uptake would produce a negative amount")
        conc[o] = np.maximum(o2_left, 0.0).reshape(env.ny, env.nx) / v

        awake = (~dormant) & ((state.energy_pj > 0.0) | (phys["maintenance_mmol_glc_gdw_h"] <= 0.0))
        secretions = [(a, acetate_made),
                      (ch.toxin, cfg.toxin_production_mmol_s * phys["toxin_production"] * (0.2 + 0.8 * activation) * dt * awake),
                      (ch.signal, cfg.signal_production_mmol_s * phys["qs_capacity"] * signal_factor * dt * awake)]
        if ch.toxin2 is not None:
            secretions.append((ch.toxin2, cfg.toxin2_production_mmol_s * phys["toxin2_production"] * (0.2 + 0.8 * activation) * dt * awake))
        for channel, amount in secretions:
            if np.any(amount > 0.0):
                per_cell = np.bincount(cell_id, weights=amount, minlength=ncell)
                conc[channel] = conc[channel] + per_cell.reshape(env.ny, env.nx) / v
                state.source_cum_mmol[channel] += float(amount.sum())
        stats["aerobic_fraction_x1000"] = int(1000 * float(np.mean(aerobic))) if n else 1000
        return np.maximum(maintenance_pj - covered * energy_in, 0.0)

    # -------------------------------------------------------------------------- movement and contact
    def _move(self, state, env, traits, phys, life, noise, dormant, stats, sugars, activation) -> None:
        cfg, ch, dt = self.config, self.channels, DT_S
        n = state.n_agents
        sensed = np.zeros(n)
        for row, channel in enumerate((ch.glucose, ch.lactose)):   # a sugar is sensed only if the cell can use it
            sensed += (traits.max_uptake_mmol_s[:, channel] > 0.0) * sugars[row]
        unset = life.memory < 0.0
        life.memory = np.where(unset, sensed, life.memory)
        kd = cfg.sensing_half_saturation_mmol_l
        sigma = np.log(sensed + kd) - np.log(life.memory + kd)
        life.memory = life.memory + (sensed - life.memory) * dt / cfg.adaptation_time_s
        base_rate = np.where(phys["run_time_s"] > 0.0, 1.0 / np.maximum(phys["run_time_s"], 1e-9), cfg.tumble_rate_per_s)
        rate = base_rate * np.clip(np.exp(-cfg.chemotaxis_gain * traits.chemotaxis_sensitivity * sigma), 0.05, 5.0)
        glider = phys["gliding"] > 0.5
        tumble = (noise["tumble_u"] < -np.expm1(-rate * dt)) & (traits.motility_um_s > 0.0) & ~glider   # only swimmers tumble
        reversal_rate = cfg.reversal_rate_per_s * (1.0 - cfg.reversal_suppression * np.clip(activation, 0.0, 1.0))
        reverse = glider & (traits.motility_um_s > 0.0) & (noise["tumble_u"] < -np.expm1(-reversal_rate * dt))
        own = phys["turn_mean_deg"] > 0.0                  # an organism with its own statistics (P. aeruginosa reverses by 150-180 degrees)
        mean = np.where(own, phys["turn_mean_deg"], cfg.tumble_angle_mean_deg)
        sd = np.where(own, phys["turn_sd_deg"], cfg.tumble_angle_sd_deg)
        turn = np.deg2rad(np.clip(mean + sd * noise["tumble_z"], 10.0, np.where(own, 180.0, 170.0)))
        turn = np.where(noise["tumble_s"] < 0.5, -turn, turn)
        life.heading = (life.heading + np.where(tumble, turn, 0.0) + np.where(reverse, math.pi, 0.0)
                        + math.sqrt(2.0 * cfg.rotational_diffusion_rad2_s * dt) * noise["rot"] * ~glider)    # a glider is held to its surface
        moving = (~dormant) & ((state.energy_pj > 0.0) | (phys["maintenance_mmol_glc_gdw_h"] <= 0.0))
        developing = phys["starvation_signalling"] > 0.5
        stress_now = -np.expm1(-life.stress_s / cfg.starvation_tau_s)
        development = (1.0 + (cfg.development_speed_factor - 1.0) * stress_now) * (1.0 - (1.0 - cfg.aggregate_speed_ratio) * np.clip(activation, 0.0, 1.0))
        crowd = np.where(developing, development, 1.0 - 0.85 * life.eps)
        speed = traits.motility_um_s * np.where(glider, 1.0, cfg.swim_speed_scale) * crowd * moving
        swim = speed[:, None] * np.column_stack((np.cos(life.heading), np.sin(life.heading)))
        drift = np.array(self.flow_um_s)[None, :]
        state.velocities_um_s = np.ascontiguousarray(swim + drift)
        jitter = math.sqrt(2.0 * cfg.translational_diffusion_um2_s * dt) * noise["brown"] * (~glider)[:, None]
        state.positions_um = env.wrap_positions(state.positions_um + (swim + drift) * dt + jitter)
        stats["tumbles"] = int(tumble.sum())
        stats["reversals"] = int(reverse.sum())

    def _collide(self, state, env, life, factor: float = 1.0, phys=None) -> None:
        """Soft-sphere overlap relaxation plus a weak attraction between matrix-producing cells (``factor``: ticks skipped + 1)."""
        cfg = self.config
        n = state.n_agents
        life.contact = np.zeros(n)
        if n < 2:
            return
        if phys is None:
            length, width = cell_dimensions(state.mass_pg, cfg)
        else:
            length, width = cell_dimensions(state.mass_pg, cfg, phys["width_um"], phys["round_cell"])
        reach = float(length.max()) + float(width.max()) + cfg.cohesion_range_um
        pairs = SpatialHash(env.width_um, env.height_um, reach).pairs(
            state.positions_um, state.agent_id, reach, ordered=False)
        if pairs.size == 0:
            return
        a, b = pairs[:, 0], pairs[:, 1]
        size = np.array([env.width_um, env.height_um])
        offset = state.positions_um[b] - state.positions_um[a]
        offset -= size * np.rint(offset / size)
        axis = np.column_stack((np.cos(life.heading), np.sin(life.heading)))
        half = (0.5 * np.maximum(length - width, 0.0))[:, None] * axis        # half of each rod's straight section
        near_a, near_b = segment_closest_points(-half[a], 2.0 * half[a], offset - half[b], 2.0 * half[b])
        delta = near_b - near_a                                               # between the rods' closest points
        dist = np.hypot(delta[:, 0], delta[:, 1])
        contact = 0.5 * (width[a] + width[b])                                 # two rods touch when their axes are one mean width apart
        same = state.genome_id[a] == state.genome_id[b]
        touching = same & (dist < contact + cfg.cohesion_range_um)
        life.contact = (np.bincount(a[touching], minlength=n) + np.bincount(b[touching], minlength=n)).astype(np.float64)
        crossing = dist < 1e-9
        sideways = np.column_stack((-axis[a, 1], axis[a, 0]))
        apart = np.hypot(offset[:, 0], offset[:, 1])
        # rods that cross or lie overlapping in line have coincident closest points: separate them along the line
        # between their centres, and only sideways when even the centres coincide
        along_centres = np.where((apart > 1e-9)[:, None], offset / np.where(apart > 1e-9, apart, 1.0)[:, None], sideways)
        unit = np.where(crossing[:, None], along_centres, delta / np.where(crossing, 1.0, dist)[:, None])
        overlap = contact - dist
        push = np.where(overlap > 0.0, 0.5 * min(1.0, cfg.collision_stiffness * factor) * overlap, 0.0)
        sticky = life.eps[a] * life.eps[b]
        gap = dist - contact
        pull = np.where((gap > 0.0) & (gap < cfg.cohesion_range_um) & (sticky > 0.09),
                        0.5 * min(1.0, cfg.cohesion_strength * factor) * sticky * gap, 0.0)
        move = (pull - push)[:, None] * unit               # displacement of cell a toward b (negative = away)
        shift = np.zeros_like(state.positions_um)
        np.add.at(shift, a, move)
        np.add.at(shift, b, -move)
        limit = 0.25 * width
        length = np.hypot(shift[:, 0], shift[:, 1])
        shift *= np.minimum(1.0, limit / np.where(length > 0.0, length, 1.0))[:, None]
        state.positions_um = env.wrap_positions(state.positions_um + shift)

    def _pull_bonds(self, state, env, phys, factor: float) -> None:
        """Sister cells that have not separated are pulled back into contact; each bond is cut at random (autolysin)."""
        if self.bond_a.size == 0:
            return
        cfg = self.config
        ids = state.agent_id
        n = ids.size
        if n < 2:
            self.bond_a = self.bond_a[:0]
            self.bond_b = self.bond_b[:0]
            return
        ia = np.minimum(np.searchsorted(ids, self.bond_a), n - 1)
        ib = np.minimum(np.searchsorted(ids, self.bond_b), n - 1)
        alive = (ids[ia] == self.bond_a) & (ids[ib] == self.bond_b)
        cut = self._rngs[RNG_ANTAGONISM].random(self.bond_a.size) < -np.expm1(-cfg.cluster_separation_rate_per_s * DT_S * factor)
        keep = alive & ~cut
        self.bond_a, self.bond_b, ia, ib = self.bond_a[keep], self.bond_b[keep], ia[keep], ib[keep]
        if ia.size == 0:
            return
        length, width = cell_dimensions(state.mass_pg, cfg, phys["width_um"], phys["round_cell"])
        size = np.array([env.width_um, env.height_um])
        offset = state.positions_um[ib] - state.positions_um[ia]
        offset -= size * np.rint(offset / size)
        dist = np.hypot(offset[:, 0], offset[:, 1])
        rest = 0.5 * (width[ia] + width[ib])      # touching: one mean width between centres
        unit = offset / np.where(dist > 1e-12, dist, 1.0)[:, None]
        pull = np.where(dist > rest, 0.5 * min(1.0, cfg.cluster_stiffness * factor) * (dist - rest), 0.0)
        shift = np.zeros_like(state.positions_um)
        np.add.at(shift, ia, pull[:, None] * unit)
        np.add.at(shift, ib, -pull[:, None] * unit)
        limit = 0.25 * width
        norm = np.hypot(shift[:, 0], shift[:, 1])
        shift *= np.minimum(1.0, limit / np.where(norm > 0.0, norm, 1.0))[:, None]
        state.positions_um = env.wrap_positions(state.positions_um + shift)

    # -------------------------------------------------------------------------- natural transformation
    def _release_dna(self, state, env, lysed: np.ndarray) -> None:
        """A lysed cell leaves its chromosome (genome and variant) in its grid cell for a while."""
        if self.transformer is None or not lysed.any():
            return
        cell, _, _ = _wrapped_cell_indices(env, state.positions_um[lysed])
        added = {"cell": cell.astype(np.int64), "genome": state.genome_id[lysed].copy(), "variant": state.variant_id[lysed].copy(),
                 "donor": state.agent_id[lysed].copy(), "tick": np.full(int(lysed.sum()), state.tick, dtype=np.int64)}
        limit = self.config.dna_pool_limit
        self.dna = {name: np.concatenate((self.dna[name], added[name]))[-limit:] for name in added}

    def _transform(self, state, env, phys, life_draw, activation, stress, dormant, phenotypes, stats) -> None:
        """Competent cells recombine a tract of nearby same-species DNA into their chromosome (at most one fragment each)."""
        cfg = self.config
        if self.transformer is None or self.dna["cell"].size == 0 or phenotypes is None:
            return
        pool = self.dna
        fresh_dna = (state.tick - pool["tick"]) * DT_S <= cfg.dna_lifetime_s
        if not fresh_dna.all():
            self.dna = pool = {name: values[fresh_dna] for name, values in pool.items()}
            if pool["cell"].size == 0:
                return
        competent = ((phys["competence_capacity"] > 0) & ~dormant
                     & ((activation > cfg.competence_signal) | (stress > cfg.competence_stress))
                     & (life_draw < -math.expm1(-cfg.transformation_rate_per_s * DT_S)))
        rows = np.flatnonzero(competent)
        if rows.size == 0:
            return
        cells, _, _ = _wrapped_cell_indices(env, state.positions_um[rows])
        consumed = np.zeros(pool["cell"].size, dtype=bool)
        done = 0
        for row, cell in zip(rows.tolist(), cells.tolist()):
            genome = state.genome_id[row]
            match = np.flatnonzero((pool["cell"] == cell) & (pool["genome"] == genome)
                                   & (pool["donor"] != state.agent_id[row]) & ~consumed)
            if match.size == 0:
                continue
            j = int(match[0])
            consumed[j] = True
            result = self.transformer.create_transformant(
                int(genome), int(state.variant_id[row]), int(pool["variant"][j]), origin_tick=state.tick,
                recipient_agent_id=int(state.agent_id[row]), donor_agent_id=int(pool["donor"][j]),
                window_bp=cfg.transformation_window_bp)
            if result is None:
                continue
            variant, phenotype = result
            _register_variant_phenotype(phenotypes, phenotype, int(genome), int(variant.variant_id))
            state.variant_id[row] = np.uint64(variant.variant_id)
            done += 1
        if consumed.any():
            self.dna = {name: values[~consumed] for name, values in pool.items()}
        if done:
            stats["transformations"] = done

    # -------------------------------------------------------------------------- death
    def _death(self, state, env, traits, phys, life, div_gate, death_draw, stats) -> np.ndarray:
        cfg, ch = self.config, self.channels
        cell_id, ix, iy = _wrapped_cell_indices(env, state.positions_um)
        conc = env.concentration_mmol_l
        dormant = life.spore >= 1.0
        stress = -np.expm1(-life.stress_s / cfg.starvation_tau_s)
        h_background = traits.death_rate_s
        h_starve = cfg.starvation_hazard_per_s * stress * (1.0 - 0.8 * phys["stress_resistance"])
        toxin_here = conc[ch.toxin][iy, ix]
        h_toxin = (cfg.toxin_max_hazard_per_s * hill(toxin_here, cfg.toxin_half_saturation_mmol_l, 2.0)
                   * phys["toxin_susceptibility"] * (1.0 - 0.7 * life.eps))
        if ch.toxin2 is not None:
            h_toxin2 = (cfg.toxin2_max_hazard_per_s * hill(conc[ch.toxin2][iy, ix], cfg.toxin2_half_saturation_mmol_l, 2.0)
                        * phys["toxin2_susceptibility"] * (1.0 - 0.7 * life.eps))
        else:
            h_toxin2 = np.zeros_like(h_toxin)
        ph = ph_from_acetate(conc[ch.acetate][iy, ix], cfg)
        h_acid = cfg.acid_hazard_per_s_per_ph * np.clip(phys["ph_min"] - ph, 0.0, 3.0) * (1.0 - 0.5 * life.eps)
        vegetative = np.where(dormant, 0.0, 1.0)
        components = np.stack([h_background, h_starve * vegetative, h_toxin * vegetative, h_acid * vegetative, h_toxin2 * vegetative], axis=0)
        total = components.sum(axis=0) * np.where(dormant, cfg.spore_hazard_factor, 1.0)
        aged = (state.age_s >= traits.max_age_s) & ~dormant
        natural = (death_draw < -np.expm1(-total * DT_S)) | aged
        eaten = getattr(self, "_eaten", None)
        if eaten is None or eaten.shape != natural.shape:
            eaten = np.zeros_like(natural)
        dead = natural | eaten
        if dead.any():
            cause = np.where(aged, 5, np.argmax(components, axis=0))
            names = ("background", "starvation", "toxin", "acid", "toxin2", "age")
            for index, name in enumerate(names):
                count = int(np.sum(natural & ~eaten & (cause == index)))
                if count:
                    stats[f"deaths_{name}"] = count
            if eaten.any():
                stats["deaths_predation"] = int(eaten.sum())
        return dead

    def _predation(self, state, env, phys, dormant, stats):
        """Predators in contact with prey of another genome kill it and take up part of its biomass.

        Returns ``(capture, eaten)``: the glucose-equivalent mmol each predator takes up this tick, per carbon channel, and
        which rows were killed. The share the predator takes is removed from the prey's arrays here and goes through the
        predator's own allocation (maintenance, storage, growth, respiration), so mass is conserved; what remains of the
        dead prey goes to detritus with every other death. A kill needs contact (centres closer than half the summed
        lengths plus ``predation_reach_um``), is a random event with the predator's capacity as its rate, and a predator
        can take in at most ``predation_max_intake_per_s`` x ``predation_feeding_window_s`` of its own mass per kill; what it\n        cannot take stays with the dead prey and goes to detritus (the real predator often leaves lysed prey unconsumed).
        """
        cfg = self.config
        n = state.n_agents
        carbon = list(self.channels.carbon)
        capture = np.zeros((n, 3))
        eaten = np.zeros(n, dtype=bool)
        capacity = phys["predation_capacity"]
        if n < 2 or not (capacity > 0.0).any():
            return capture, eaten
        length, _ = cell_dimensions(state.mass_pg, cfg, phys["width_um"], phys["round_cell"])
        reach = float(length.max()) + cfg.predation_reach_um
        pairs = SpatialHash(env.width_um, env.height_um, reach).pairs(state.positions_um, state.agent_id, reach, ordered=False)
        if pairs.size == 0:
            return capture, eaten
        both = np.concatenate((pairs, pairs[:, ::-1]))
        p, q = both[:, 0], both[:, 1]
        keep = (capacity[p] > 0.0) & (capacity[q] == 0.0) & (state.genome_id[p] != state.genome_id[q]) & ~dormant[p]
        p, q = p[keep], q[keep]
        if p.size == 0:
            return capture, eaten
        size = np.array([env.width_um, env.height_um])
        offset = state.positions_um[q] - state.positions_um[p]
        offset -= size * np.rint(offset / size)
        touching = np.hypot(offset[:, 0], offset[:, 1]) <= 0.5 * (length[p] + length[q]) + cfg.predation_reach_um
        p, q = p[touching], q[touching]
        if p.size == 0:
            return capture, eaten
        rate = cfg.predation_rate_per_s * capacity[p] * np.where(dormant[q], cfg.spore_predation_factor, 1.0)
        hit = self._rngs[RNG_ANTAGONISM].random(p.size) < -np.expm1(-rate * DT_S)
        p, q = p[hit], q[hit]
        if p.size == 0:
            return capture, eaten
        molar_glucose = float(env.molar_mass_g_mol[self.channels.glucose]) * 1e9
        allowance = cfg.predation_max_intake_per_s * cfg.predation_feeding_window_s * state.mass_pg / molar_glucose   # mmol glucose-equivalent per kill
        taken = np.zeros(n)
        eta = cfg.predation_assimilation
        for predator, prey in zip(p.tolist(), q.tolist()):
            if eaten[prey]:
                continue
            prey_amount = state.biomass_mmol[prey, carbon] + state.nutrient_store_mmol[prey, carbon]
            share = eta * float(prey_amount.sum())
            room = allowance[predator] - taken[predator]
            if room <= 0.0 or share <= 0.0:
                continue                                       # a predator that is full does not kill
            fraction = min(1.0, room / share)
            gained = eta * fraction * prey_amount
            capture[predator] += gained
            taken[predator] += float(gained.sum())
            eaten[prey] = True
            state.biomass_mmol[prey, carbon] *= 1.0 - eta * fraction
            state.nutrient_store_mmol[prey, carbon] *= 1.0 - eta * fraction
        stats["predation_kills"] = int(eaten.sum())
        return capture, eaten

def _empty_pool() -> dict[str, np.ndarray]:
    return {"cell": np.empty(0, dtype=np.int64), "genome": np.empty(0, dtype=np.uint64), "variant": np.empty(0, dtype=np.uint64),
            "donor": np.empty(0, dtype=np.uint64), "tick": np.empty(0, dtype=np.int64)}


def life_spore_gate(dormant: np.ndarray) -> np.ndarray:
    return dormant.astype(np.float64)
