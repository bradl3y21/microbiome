"""The realism layer through the real controller: real genomes, real tables, the D3 facade, and the default path untouched."""
import dataclasses

import numpy as np
import pytest

from microbiome.core.ui_api import RealismSnapshot, RenderSnapshot
from microbiome.sim.controller import SimulationController
from microbiome.sim.realism import RealismConfig

CHANNELS = ("glucose", "lactose", "oxygen", "acetate", "bacilysin", "ai2", "pyocyanin")


def ids(controller):
    return {o.accession_version: int(o.genome_id) for o in controller.list_genomes()}


def populated(seed=11, realism=True, n=10, mutation=False):
    c = SimulationController.create_demo(seed=seed, realism=realism, mutation=mutation)
    g = ids(c)
    c.advance_ticks(1)
    for i in range(n):
        c.introduce_organism(g["GCF_000005845.2"], (150.0 + 4 * i, 200.0))
        c.introduce_organism(g["GCF_000009045.1"], (150.0 + 4 * i, 230.0))
    c.advance_ticks(2)
    return c, g


def test_the_default_controller_is_still_the_plain_two_channel_d1_world():
    c = SimulationController.create_demo(seed=5)
    assert not c.realism_enabled() and c.channel_names() == ("glucose", "lactose")
    c.advance_ticks(1)
    snap = c.render_snapshot()
    assert type(snap) is RenderSnapshot and snap.concentration_mmol_l.shape[0] == 2
    assert c.realism_stats() == {} and c.life_of(1) is None


def test_the_realistic_world_has_seven_named_channels_with_oxygen_at_saturation():
    c = SimulationController.create_demo(seed=5, realism=True)
    snap = c.render_snapshot()
    assert isinstance(snap, RealismSnapshot) and isinstance(snap, RenderSnapshot)
    assert snap.channel_names == CHANNELS and snap.concentration_mmol_l.shape[0] == 7
    assert snap.concentration_mmol_l[2].min() == pytest.approx(0.21) and snap.concentration_mmol_l[[0, 1, 3, 4, 5, 6]].max() == 0.0
    assert snap.phenotype_max_uptake_mmol_s.shape == (len(c.list_genomes()), 7)


def test_a_custom_config_reaches_the_simulation():
    cfg = dataclasses.replace(RealismConfig(), oxygen_saturation_mmol_l=0.1, swim_speed_scale=1.0)
    c = SimulationController.create_demo(seed=5, realism=cfg)
    assert c.render_snapshot().concentration_mmol_l[2].max() == pytest.approx(0.1)
    assert c._realism.simulation.config.swim_speed_scale == 1.0


def test_real_genomes_run_in_the_realistic_world_and_conserve_every_channel():
    c, _ = populated(n=15)
    rng = np.random.default_rng(2)
    for _ in range(40):
        c.place_nutrient(0, (float(rng.uniform(100, 300)), float(rng.uniform(150, 300))), 3e-12)
    c.advance_ticks(600)                         # the engine raises on any per-channel conservation failure
    snap = c.render_snapshot()
    assert isinstance(snap, RealismSnapshot) and len(snap.agent_id) >= 30
    assert np.isfinite(snap.concentration_mmol_l).all() and (snap.concentration_mmol_l >= 0).all()


def test_the_two_real_genomes_get_different_physiology_from_their_own_genes():
    c, g = populated()
    sim = c._realism.simulation
    e, b = (sim._phys["ph_min"][np.searchsorted(sim._phys_ids, g[a])] for a in ("GCF_000005845.2", "GCF_000009045.1"))
    assert (e, b) == pytest.approx((4.4, 5.8))
    spores = sim._phys["sporulation_capacity"]
    assert spores[np.searchsorted(sim._phys_ids, g["GCF_000009045.1"])] == 1.0
    assert spores[np.searchsorted(sim._phys_ids, g["GCF_000005845.2"])] == 0.0


def test_only_bsubtilis_charges_the_toxin_field_and_ecoli_pays_the_price():
    c = SimulationController.create_demo(seed=21, realism=True)
    g = ids(c)
    c.advance_ticks(1)
    for i in range(40):
        c.introduce_organism(g["GCF_000009045.1"], (300.0 + (i % 8) * 2, 300.0 + (i // 8) * 2))
    c.advance_ticks(1200)
    snap = c.render_snapshot()
    toxin = snap.concentration_mmol_l[4]
    assert toxin.max() > 0 and toxin.argmax() // 80 in range(28, 36), "the producers' toxin is centred on them"
    only_ecoli = SimulationController.create_demo(seed=21, realism=True)
    only_ecoli.advance_ticks(1)
    for i in range(40):
        only_ecoli.introduce_organism(g["GCF_000005845.2"], (300.0 + (i % 8) * 2, 300.0 + (i // 8) * 2))
    only_ecoli.advance_ticks(1200)
    assert only_ecoli.render_snapshot().concentration_mmol_l[4].max() == 0.0


def test_the_extra_readouts_report_chemistry_stats_and_cell_state():
    c, _ = populated(n=5)
    snap = c.render_snapshot()
    here = c.chemistry_at((150.0, 200.0))
    assert set(here) == set(CHANNELS) | {"pH"} and here["pH"] == pytest.approx(7.0) and here["oxygen"] == pytest.approx(0.21, rel=0.05)
    life = c.life_of(int(snap.agent_id[0]))
    assert set(life) == {"lag", "stress", "spore", "matrix", "energy_pj", "alt_sugar_expression"}
    assert 0.04 < life["lag"] < 0.1 and life["spore"] == 0.0 and life["energy_pj"] > 0
    assert c.life_of(10**9) is None
    assert {"tumbles", "spores", "sporulating", "starving", "biofilm"} <= set(c.realism_stats())
    assert snap.cell_state.dtype == np.uint8 and snap.eps.shape == snap.agent_id.shape


def test_inspecting_a_cell_and_previewing_a_genome_still_work_with_seven_channels():
    c, g = populated(n=3)
    snap = c.render_snapshot()
    inspection = c.inspect_cell(int(snap.agent_id[0]))
    assert inspection is not None and len(inspection.nutrient_id) == 7 and len(inspection.nutrient_store_mmol) == 7
    preview = c.preview_phenotype(g["GCF_000005845.2"])
    assert len(preview.nutrient_id) == 2, "the D2 preview describes the two sugars the gene table maps"
    assert c.inspect_genome(g["GCF_000009045.1"]) is not None


def test_a_founder_is_built_from_the_sugars_not_from_oxygen_or_acetate():
    c, _ = populated(n=2)
    biomass = c._state.agents.biomass_mmol
    assert (biomass[:, 0] > 0).all() and (biomass[:, 2:] == 0).all()
    assert c._state.agents.mass_pg[0] == pytest.approx(0.3, rel=1e-6)


def test_the_realistic_controller_is_deterministic_from_its_seed():
    def run(seed):
        c, _ = populated(seed=seed, n=8)
        c.place_nutrient(0, (160.0, 210.0), 5e-12)
        c.advance_ticks(300)
        a, snap = c._state.agents, c.render_snapshot()
        return (a.agent_id.tobytes(), a.position_um.tobytes(), a.biomass_mmol.tobytes(), snap.concentration_mmol_l.tobytes(),
                c._realism.simulation.life.heading.tobytes())
    assert run(4) == run(4)
    assert run(4) != run(5)


def test_mutation_and_realism_work_together_and_variant_rows_have_all_seven_channels():
    c = SimulationController.create_demo(seed=2, realism=True, mutation=True)
    g = ids(c)
    c.advance_ticks(1)
    c.introduce_organism(g["GCF_000005845.2"], (200.0, 300.0))
    c.advance_ticks(2)
    from microbiome.genome import tables as T, variants as V
    provider = c._mutation
    result = provider.create_offspring_variant(g["GCF_000005845.2"], 0, mutation_rng=_ForceOne(1), origin_tick=3,
                                               parent_agent_id=1, daughter_agent_id=2)
    variant, phenotype = result
    assert len(phenotype.max_uptake_mmol_s) == 7 and phenotype.variant_id == variant.variant_id
    assert phenotype.max_uptake_mmol_s[2] == pytest.approx(3.0e-15), "the root genome's oxygen capacity is carried over"


class _ForceOne:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def poisson(self, lam):
        return 1

    def __getattr__(self, name):
        return getattr(self.rng, name)


def test_introduced_cells_start_in_lag_and_the_snapshot_never_exposes_live_arrays():
    c, _ = populated(n=3)
    snap = c.render_snapshot()
    before = snap.eps.copy()
    c._realism.simulation.life.eps[:] = 0.9
    assert np.array_equal(snap.eps, before)
    for name in ("agent_id", "position_um", "cell_state", "eps", "concentration_mmol_l"):
        assert not getattr(snap, name).flags.writeable, name
