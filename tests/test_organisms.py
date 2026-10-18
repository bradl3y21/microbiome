"""The organisms added after E. coli and B. subtilis: P. aeruginosa PAO1, S. aureus Newman and M. xanthus DK 1622.

What is checked: each is a real manifest genome, its physiology comes from annotated genes (and goes when a required gene is
removed), its shape and look follow its profile, and the new mechanisms they need (a second toxin, gliding with reversals,
contact predation on another genome) behave and conserve mass. None of this validates the ecology.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from microbiome.genome.organisms import BS, EC, MX, NEW_ORGANISMS, PA, PROFILES, SA
from microbiome.genome.parser import derive_genome_id, load_genome
from microbiome.genome.physiology import (
    CHANNEL_NAMES, DIFFUSIVITY_UM2_S, MOLAR_MASS_G_MOL, REALISM_CHANNELS, derive_physiology, load_physiology_tables,
)
from microbiome.render import microscope as M
from microbiome.sim.model import AgentState, NutrientEnvironment, Phenotype, make_rngs
from microbiome.sim.realism import Physiology, RealismConfig, RealisticSimulation, cell_dimensions

ROOT = Path(__file__).resolve().parent.parent
K = len(CHANNEL_NAMES)
CH = REALISM_CHANNELS
MOLAR = np.asarray(MOLAR_MASS_G_MOL)
TABLES = load_physiology_tables(ROOT)
GENOMES = {accession: load_genome(accession, ROOT) for accession in (EC, BS, PA, SA, MX)}
PHYS = {accession: derive_physiology(genome, TABLES) for accession, genome in GENOMES.items()}


# ===================================================================== data and profiles
def test_the_manifest_holds_exactly_the_five_profiled_organisms_each_a_complete_assembly():
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert {entry["accession"] for entry in manifest["genomes"]} == set(PROFILES) == {EC, BS, PA, SA, MX}
    assert all(entry["assembly_level"] == "Complete Genome" for entry in manifest["genomes"])
    ids = {derive_genome_id(accession) for accession in PROFILES}
    assert len(ids) == 5


def test_each_profile_has_its_own_imaging_style_and_the_microscope_width_table_agrees_with_it():
    styles = [profile.style for profile in PROFILES.values()]
    assert len(set(styles)) == len(styles) and max(styles) < M.SPORE_STYLE
    for accession, profile in PROFILES.items():
        expected = 0.0 if profile.shape == "coccus" else profile.width_um
        assert M.STYLE_WIDTH_UM[profile.style] == expected, accession


def test_gram_stain_colours_follow_the_gram_reaction():
    negative = {style: np.asarray(M.STAIN[style], dtype=float) for style in (p.style for p in PROFILES.values() if p.gram == "negative")}
    positive = {style: np.asarray(M.STAIN[style], dtype=float) for style in (p.style for p in PROFILES.values() if p.gram == "positive")}
    assert negative and positive
    for colour in negative.values():
        assert colour[0] > colour[2], "Gram-negative cells take the red/pink counterstain"
    for colour in positive.values():
        assert colour[2] > colour[0], "Gram-positive cells keep the blue/violet crystal violet"


# ===================================================================== physiology from genes
def test_physiology_traits_come_from_the_annotated_genes():
    assert PHYS[PA].aerobic_capacity == pytest.approx(1.0) and PHYS[PA].toxin2_production == 1.0
    assert PHYS[PA].toxin2_susceptibility < 0.2, "a phenazine producer is mostly immune to its own toxin"
    assert PHYS[MX].gliding == 1.0 and PHYS[MX].predation_capacity == 1.0 and PHYS[MX].starvation_signalling == 1.0
    assert PHYS[MX].sporulation_capacity == 1.0
    assert PHYS[SA].round_cell == 1.0 and PHYS[SA].biofilm_capacity == 1.0 and PHYS[SA].qs_capacity == 1.0
    for accession in (EC, BS, SA, PA):
        assert PHYS[accession].gliding == 0.0 and PHYS[accession].predation_capacity == 0.0
    for accession in (EC, BS, SA, MX):
        assert PHYS[accession].toxin2_production == 0.0


def test_removing_one_required_gene_removes_the_trait_it_belongs_to():
    def without(genome, locus):
        genes = tuple(gene for gene in genome.genes if gene.locus_tag != locus)
        assert len(genes) == len(genome.genes) - 1
        return dataclasses.replace(genome, genes=genes)

    mgl_a = next(g.locus_tag for g in GENOMES[MX].genes if g.gene_symbol == "mglA")
    assert derive_physiology(without(GENOMES[MX], mgl_a), TABLES).gliding == 0.0
    assert derive_physiology(without(GENOMES[MX], mgl_a), TABLES).predation_capacity == 0.0
    phz = next(g.locus_tag for g in GENOMES[PA].genes if g.gene_symbol == "phzM")
    assert derive_physiology(without(GENOMES[PA], phz), TABLES).toxin2_production == 0.0


def test_the_profile_shapes_reach_the_physiology():
    assert PHYS[PA].width_um == 0.75 and PHYS[MX].width_um == 0.5 and PHYS[SA].round_cell == 1.0
    assert {a: int(PHYS[a].imaging_style) for a in PHYS} == {a: p.style for a, p in PROFILES.items()}


# ===================================================================== shape
def test_a_coccus_is_a_sphere_of_the_cells_volume_and_a_rod_keeps_its_width():
    config = RealismConfig()
    mass = np.array([0.15, 0.3, 0.6])
    length, width = cell_dimensions(mass, config, np.full(3, 1.0), np.ones(3))
    assert np.allclose(length, width)
    assert np.allclose(np.pi / 6.0 * width ** 3, config.cell_volume_um3_per_pg * mass)
    length, width = cell_dimensions(mass, config, np.full(3, 0.5), np.zeros(3))
    assert np.allclose(width, 0.5) and (np.diff(length) > 0).all()
    volume = np.pi * 0.25 / 4.0 * (length - 0.5) + np.pi * 0.125 / 6.0
    assert np.allclose(volume, config.cell_volume_um3_per_pg * mass)


def test_a_myxococcus_cell_of_the_fitted_mean_mass_is_as_long_as_the_published_range():
    config = RealismConfig()
    length, _ = cell_dimensions(np.array([0.3]), config, np.array([0.5]), np.zeros(1))
    assert 4.5 < float(length[0]) < 6.5          # newborn: at the short end of the 5-7 um range


# ===================================================================== a mixed world
def phenotype(accession, **changes):
    vals = dict(max_uptake_mmol_s=np.zeros(K), storage_fraction=np.zeros(K), biomass_fraction=np.zeros(K),
                energy_yield_pj_per_mmol=np.zeros(K), motility_um_s=0.0, chemotaxis_sensitivity=0.0, random_motility_fraction=0.15,
                division_rate_s=0.0, division_mass_pg=10.0, death_rate_s=0.0, max_age_s=1e9, antagonism_radius_um=0.0, antagonism_rate_s=0.0)
    vals["max_uptake_mmol_s"][CH.glucose] = 8.88e-16
    vals["storage_fraction"][[CH.glucose, CH.acetate]] = 0.1
    vals["biomass_fraction"][[CH.glucose, CH.acetate]] = 0.75
    vals["energy_yield_pj_per_mmol"][[CH.glucose, CH.acetate]] = (1e15, 3.3e14)
    vals.update(changes)
    return Phenotype(**vals)


class Mixed:
    """Cells of several genomes on one seven-channel slab, advanced by the realistic tick."""

    def __init__(self, groups, config, *, seed=3, nx=20, ny=20, **field):
        self.genome_ids = {name: derive_genome_id(name) for name in groups}
        physiology = {self.genome_ids[name]: dataclasses.replace(PHYS[name]) for name in groups}
        for name, (_, changes) in groups.items():
            if changes:
                physiology[self.genome_ids[name]] = dataclasses.replace(physiology[self.genome_ids[name]], **changes)
        conc = np.zeros((K, ny, nx))
        conc[CH.oxygen] = 0.21
        for channel, value in field.items():
            conc[getattr(CH, channel)] = value
        self.env = NutrientEnvironment(nutrient_id=np.arange(K, dtype=np.uint32), concentration_mmol_l=conc,
                                       diffusion_um2_s=np.asarray(DIFFUSIVITY_UM2_S, dtype=np.float64), molar_mass_g_mol=MOLAR.copy(),
                                       dx_um=10.0, dy_um=10.0, depth_um=10.0, width_um=nx * 10.0, height_um=ny * 10.0)
        rows, genome = [], []
        for name, (positions, _) in groups.items():
            positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
            rows.append(positions)
            genome.extend([self.genome_ids[name]] * len(positions))
        positions = np.concatenate(rows)
        n = len(positions)
        biomass = np.zeros((n, K))
        biomass[:, CH.glucose] = 0.3 / (MOLAR[CH.glucose] * 1e9)
        mass = biomass @ (MOLAR * 1e9)
        self.state = AgentState(
            agent_id=np.arange(1, n + 1, dtype=np.uint64), positions_um=positions, velocities_um_s=np.zeros((n, 2)), mass_pg=mass,
            radius_um=np.cbrt(3 * mass / (4 * np.pi)), energy_pj=np.zeros(n), species_id=np.zeros(n, dtype=np.uint32),
            genome_id=np.asarray(genome, dtype=np.uint64), age_s=np.zeros(n), nutrient_store_mmol=np.zeros((n, K)), biomass_mmol=biomass)
        self.sim = RealisticSimulation(physiology, config, CH)
        self.rngs = make_rngs(seed)
        self.table = {self.genome_ids[name]: phenotype(name) for name in groups}

    def step(self, ticks=1):
        for _ in range(ticks):
            self.sim.step(self.state, self.env, self.table, self.rngs)

    def count(self, name):
        return int(np.sum(self.state.genome_id == np.uint64(self.genome_ids[name])))

    def ledger_residual(self):
        s, v, worst = self.state, self.env.cell_volume_l, 0.0
        for k in range(K):
            field = float(self.env.concentration_mmol_l[k].sum()) * v
            agents = float((s.nutrient_store_mmol[:, k] + s.biomass_mmol[:, k]).sum())
            lhs = s.initial_mmol[k] + s.source_cum_mmol[k]
            rhs = field + agents + float(self.env.detritus_mmol[k].sum()) + s.sink_cum_mmol[k] + s.respiration_loss_mmol[k]
            worst = max(worst, abs(rhs - lhs) / max(abs(lhs), 1e-18))
        return worst


def quiet(**changes):
    base = dict(collisions=False, translational_diffusion_um2_s=0.0, rotational_diffusion_rad2_s=0.0, lag_initial=1.0)
    base.update(changes)
    return dataclasses.replace(RealismConfig(), **base)


# ===================================================================== predation
def test_a_predator_in_contact_kills_and_eats_prey_of_another_genome_and_mass_is_conserved():
    prey = [(100.0, 100.0), (102.0, 100.0), (100.0, 102.0)]
    far = [(30.0, 150.0)]
    w = Mixed({MX: ([(101.0, 101.0)], None), EC: (prey + far, None)}, quiet(predation_rate_per_s=2.0, predation_max_intake_per_s=1.0))
    w.step(1)
    before = float(w.state.biomass_mmol[:, CH.glucose].sum() + w.state.nutrient_store_mmol[:, CH.glucose].sum())
    mass_before = float(w.state.mass_pg[w.state.genome_id == np.uint64(w.genome_ids[MX])].sum())
    w.step(300)
    assert w.count(EC) == 1, "the three prey in contact are eaten, the one 80 um away is not"
    assert w.count(MX) == 1
    assert float(w.state.mass_pg[w.state.genome_id == np.uint64(w.genome_ids[MX])].sum()) > mass_before, "the predator took up biomass"
    assert w.ledger_residual() < 1e-9
    assert before > 0


def test_predators_leave_their_own_kind_and_non_predators_alone():
    cluster = [(100.0, 100.0), (101.0, 100.0), (100.0, 101.0)]
    w = Mixed({MX: (cluster, None)}, quiet(predation_rate_per_s=5.0))
    w.step(300)
    assert w.count(MX) == 3, "a predator never kills a cell of its own genome"
    w = Mixed({EC: (cluster, None), BS: ([(100.5, 100.5), (101.5, 100.5)], None)}, quiet(predation_rate_per_s=5.0))
    w.step(300)
    assert (w.count(EC), w.count(BS)) == (3, 2), "without the predation genes nothing is eaten"


def test_a_predator_takes_up_at_most_its_intake_limit_per_kill_and_the_rest_of_the_prey_becomes_detritus():
    prey = [(100.0 + 0.4 * i, 100.0) for i in range(4)]

    def run(intake):
        w = Mixed({MX: ([(101.0, 100.5)], None), EC: (prey, None)}, quiet(predation_rate_per_s=5.0, predation_max_intake_per_s=intake))
        w.step(240)
        predator = w.state.genome_id == np.uint64(w.genome_ids[MX])
        return w, float(w.state.mass_pg[predator].sum()), float(w.env.detritus_mmol[CH.glucose].sum())

    slow, slow_mass, slow_detritus = run(2e-5)
    fast, fast_mass, fast_detritus = run(1.0)
    assert slow.count(EC) == fast.count(EC) == 0, "the intake limit does not stop the kill"
    assert fast_mass > slow_mass > 0.3, "a hungrier predator takes up more of each prey"
    assert slow_detritus > fast_detritus
    assert slow.ledger_residual() < 1e-9 and fast.ledger_residual() < 1e-9


def test_a_mixed_world_with_predation_is_deterministic_and_conserves_every_channel():
    def run(seed):
        rng = np.random.default_rng(1)
        w = Mixed({MX: (rng.random((6, 2)) * 40 + 80, None), EC: (rng.random((12, 2)) * 40 + 80, None), PA: (rng.random((6, 2)) * 40 + 80, None)},
                  quiet(predation_rate_per_s=0.5, collisions=True), seed=seed, glucose=0.3)
        w.step(240)
        return w

    a, b, c = run(3), run(3), run(4)
    for name in ("positions_um", "biomass_mmol", "genome_id"):
        assert getattr(a.state, name).tobytes() == getattr(b.state, name).tobytes()
    assert a.state.positions_um.tobytes() != c.state.positions_um.tobytes()
    assert a.ledger_residual() < 1e-9


# ===================================================================== gliding
def test_gliders_reverse_instead_of_tumbling_at_the_fitted_rate_and_keep_a_constant_speed():
    rng = np.random.default_rng(2)
    n, ticks = 150, 7200
    w = Mixed({MX: (rng.random((n, 2)) * 180 + 10, {"qs_capacity": 0.0})}, quiet(), seed=5)
    w.table[w.genome_ids[MX]] = phenotype(MX, motility_um_s=glide_speed_um_s())
    reversals = tumbles = 0
    speeds = []
    for _ in range(ticks):
        w.step()
        reversals += int(w.sim.stats.get("reversals", 0))
        tumbles += int(w.sim.stats.get("tumbles", 0))
        speeds.append(float(np.mean(np.hypot(*w.state.velocities_um_s.T))))
    expected = n * ticks / 60.0 * RealismConfig().reversal_rate_per_s
    assert tumbles == 0
    assert reversals == pytest.approx(expected, rel=0.25)
    assert np.mean(speeds) == pytest.approx(glide_speed_um_s(), rel=0.05)


def glide_speed_um_s() -> float:
    """The M. xanthus coefficient in the realism phenotype table (gene-derived D2 sum times the calibration factor)."""
    from microbiome.genome.models import NutrientChannel
    from microbiome.genome.phenotype import build_phenotype_table
    from microbiome.genome.physiology import extend_phenotype_table

    base = build_phenotype_table(list(GENOMES.values()), (NutrientChannel(0, "glucose"), NutrientChannel(1, "lactose")), ROOT)
    table = extend_phenotype_table(base, list(GENOMES.values()), TABLES)
    return float(table.motility_um_s[int(np.flatnonzero(table.genome_id == np.uint64(derive_genome_id(MX)))[0])])


def test_the_glider_speed_is_the_published_one_after_calibration():
    assert glide_speed_um_s() * 60.0 == pytest.approx(2.3, rel=0.02)


# ===================================================================== the second toxin
def test_pyocyanin_is_made_by_the_phenazine_producer_and_kills_a_susceptible_neighbour_not_the_producer():
    producers = [(100.0 + 3 * i, 100.0) for i in range(4)]
    w = Mixed({PA: (producers, None)}, quiet(toxin2_production_mmol_s=5e-13), seed=2, glucose=0.5)
    w.step(120)
    assert w.env.concentration_mmol_l[CH.toxin2].max() > 0.0
    assert w.count(PA) == 4

    soaked = Mixed({EC: ([(30.0 + 7 * i, 100.0) for i in range(20)], None), PA: ([(30.0 + 7 * i, 140.0) for i in range(20)], None)},
                   quiet(toxin2_max_hazard_per_s=2.0, toxin2_half_saturation_mmol_l=1e-6), seed=2, glucose=0.5)
    soaked.env.concentration_mmol_l[CH.toxin2] = 1e-3
    soaked.step(180)
    assert soaked.count(EC) < 10 and soaked.count(PA) > soaked.count(EC)


# ===================================================================== what the UI receives
def test_the_snapshot_tells_the_renderer_each_genomes_style_and_the_true_cell_lengths():
    from microbiome.sim.controller import SimulationController

    c = SimulationController.create_demo(seed=4, realism=True)
    by = {g.accession: int(g.genome_id) for g in c._genomes}
    assert set(by) == set(PROFILES)
    from microbiome.ui.app import dispatch_command

    dispatch_command(c, "advance_ticks", count=1)
    for accession, genome_id in by.items():
        dispatch_command(c, "introduce_organism", genome_id=genome_id, position_um=(300.0 + 20 * PROFILES[accession].style, 320.0))
    dispatch_command(c, "advance_ticks", count=2)
    s = c.render_snapshot()
    styles = {int(g): int(st) for g, st in zip(s.phenotype_genome_id, s.genome_style)}
    assert styles == {by[a]: p.style for a, p in PROFILES.items()}
    sa_cells = s.genome_id == np.uint64(by[SA])
    mx_cells = s.genome_id == np.uint64(by[MX])
    assert s.length_um[mx_cells].min() > 3.0 and s.length_um[sa_cells].max() < 1.6
    from microbiome.render.view import WorldView

    assert {int(i) for i in WorldView.from_snapshot(s).colour_index} == {p.style for p in PROFILES.values()}


# ===================================================================== swimmers that reverse, developing gliders, clusters
def test_p_aeruginosa_reverses_after_about_two_seconds_by_about_165_degrees():
    rng = np.random.default_rng(9)
    n, seconds = 300, 60
    w = Mixed({PA: (rng.random((n, 2)) * 180 + 10, {"qs_capacity": 0.0})}, quiet(), seed=6)
    w.table[w.genome_ids[PA]] = phenotype(PA, motility_um_s=10.0)
    w.step(1)
    turns, tumbles = [], 0
    previous_ids, previous = w.state.agent_id.copy(), w.sim.life.heading.copy()
    for _ in range(seconds * 60):
        w.step()
        ids, heading = w.state.agent_id, w.sim.life.heading
        at = np.minimum(np.searchsorted(previous_ids, ids), previous_ids.size - 1)
        same = previous_ids[at] == ids                                  # a cell that died is skipped
        change = np.abs(np.angle(np.exp(1j * (heading[same] - previous[at[same]]))))
        turns.extend(change[change > 1e-9].tolist())
        tumbles += int(w.sim.stats.get("tumbles", 0))
        previous_ids, previous = ids.copy(), heading.copy()
    assert abs(len(turns) - tumbles) <= 3
    assert n * seconds / tumbles == pytest.approx(2.18, rel=0.1), "mean time between reversals"
    assert np.degrees(np.mean(turns)) == pytest.approx(165.0, abs=2.0)
    assert np.degrees(np.std(turns)) < 14.0


def test_a_starved_glider_moves_faster_and_a_fully_activated_one_slows_to_the_published_ratio():
    rng = np.random.default_rng(4)
    n = 60
    base = glide_speed_um_s()

    def speed(stress_s, contact):
        w = Mixed({MX: (rng.random((n, 2)) * 180 + 10, None)}, quiet(), seed=2)
        w.table[w.genome_ids[MX]] = phenotype(MX, motility_um_s=base)
        w.step(1)
        w.sim.life.stress_s[:] = stress_s
        w.sim.life.contact[:] = contact
        w.step(3)
        return float(np.mean(np.hypot(*w.state.velocities_um_s.T)))

    assert speed(0.0, 0.0) == pytest.approx(base, rel=0.03), "a growing cell glides at its calibrated speed"
    outside = speed(1e7, 0.0)
    assert outside == pytest.approx(0.12, rel=0.05), "a starved cell outside an aggregate: 0.12 um/s (Sliusarenko 2007)"
    assert speed(1e7, 1e3) / outside == pytest.approx(0.023 / 0.12, rel=0.05), "fully activated by contact: inside an aggregate"


def test_cocci_stay_attached_to_their_sisters_and_a_control_without_bonds_does_not():
    def run(separation):
        w = Mixed({SA: (np.array([(100.0, 100.0), (60.0, 60.0), (140.0, 60.0), (60.0, 140.0)]), None)},
                  quiet(cluster_separation_rate_per_s=separation, translational_diffusion_um2_s=0.4, collisions=True, collision_interval_ticks=1),
                  seed=3, glucose=0.5)
        w.table[w.genome_ids[SA]] = phenotype(SA, division_mass_pg=0.25)
        w.step(1800)
        return w

    bonded, control = run(0.0), run(1.0)
    for w in (bonded, control):
        assert w.count(SA) >= 6, "the cells divided"
    assert bonded.sim.bond_a.size >= 3 and control.sim.bond_a.size == 0
    s = bonded.state

    def sister_gaps(w):
        a, b = w.sim.bond_a, w.sim.bond_b
        ids = w.state.agent_id
        ia, ib = np.searchsorted(ids, a), np.searchsorted(ids, b)
        offset = w.state.positions_um[ib] - w.state.positions_um[ia]
        offset -= 200.0 * np.rint(offset / 200.0)
        return np.hypot(offset[:, 0], offset[:, 1])

    _, width = cell_dimensions(s.mass_pg, quiet(), np.full(s.n_agents, 1.0), np.ones(s.n_agents))
    assert (sister_gaps(bonded) < 1.3 * width.max()).all(), "bonded sisters stay in contact"


def test_a_coccus_divides_in_alternating_planes():
    w = Mixed({SA: ([(100.0, 100.0)], None)}, quiet(), seed=3, glucose=0.5)
    w.table[w.genome_ids[SA]] = phenotype(SA, division_mass_pg=0.25)
    w.step(1)
    before = float(w.sim.life.heading[0])
    for _ in range(3000):
        w.step()
        if w.state.n_agents > 1:
            break
    assert w.state.n_agents == 2
    turned = np.abs(np.angle(np.exp(1j * (w.sim.life.heading - before))))
    assert np.allclose(turned, np.pi / 2, atol=0.1), "the division axis turns a right angle each generation"
