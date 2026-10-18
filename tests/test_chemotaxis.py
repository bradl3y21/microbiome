"""README must-do #2 / item 225: chemotaxis moves a population up a nutrient gradient.

Experiment (same seed, same start positions, same field for every run; only the traits differ):

  field       : 32 x 32 cells of 10 um, closed (diffusion = 0), glucose = c0 * (1 + 0.9 cos(2 pi x / W)),
                so the nutrient peaks at x = 0 (mod W) and is lowest at x = W/2
  uptake      : switched off in every run, so the gradient is a fixed stimulus and cannot be eaten away
  populations : E. coli rows with their frozen traits (chemotaxis_sensitivity 0.85, random fraction 0.15,
                motility 10 um/s) versus the same row with chemotaxis_sensitivity = 0
  outcome     : mean of cos(2 pi x / W) over the population, which is +1 at the nutrient peak, -1 at the
                trough and ~0 for agents spread uniformly in x
  prediction  : chemotactic agents drift up the gradient, so the mean rises well above 0; agents that
                cannot sense it (sensitivity 0) random-walk and stay near 0; and in a flat field even the
                chemotactic agents have no gradient to follow and stay near 0
"""
import numpy as np
import pytest

import sim_support as S
import test_competition as tc
import microbiome.sim.engine as engine

CELLS = 32
N = 400
TICKS = 900              # 15 s at 10 um/s = up to 150 um of travel on a 320 um field
AMPLITUDE = 0.9


@pytest.fixture(scope="module")
def frozen():
    genomes = {a: tc.load_genome(a) for a in (tc.EC, tc.BS)}
    table = tc.build_phenotype_table(list(genomes.values()), tc.CHANNELS)
    row = {a: int(np.flatnonzero(table.genome_id == np.uint64(g.genome_id))[0]) for a, g in genomes.items()}
    inert = tc.edit(table, **{
        "max_uptake_mmol_s": (row[tc.EC], tc.GLC, 0.0),
        "antagonism_rate_s": (row[tc.EC], 0.0),
        "antagonism_radius_um": (row[tc.EC], 0.0),
    })
    inert = tc.edit(inert, max_uptake_mmol_s=(row[tc.EC], tc.LAC, 0.0))
    assert float(inert.max_uptake_mmol_s[row[tc.EC]].sum()) == 0.0
    blind = tc.edit(inert, chemotaxis_sensitivity=(row[tc.EC], 0.0))
    return inert, blind, row[tc.EC]


def cosine_mean(state, width):
    return float(np.cos(2.0 * np.pi * state.agents.position_um[:, 0] / width).mean())


def drift(monkeypatch, tbl, ec_row, amplitude=AMPLITUDE, ticks=TICKS):
    monkeypatch.setattr(tc, "CELLS", CELLS)
    assign = np.full(N, ec_row, dtype=np.int64)
    state, env, rngs = tc.world(tc.SEED, assign, tbl, prime=(tc.CONC, 0.0))
    x_centres = (np.arange(CELLS) + 0.5) * tc.DX
    profile = tc.CONC * (1.0 + amplitude * np.cos(2.0 * np.pi * x_centres / env.width_um))
    env.field.concentration_mmol_l[tc.GLC] = np.tile(profile, (CELLS, 1))
    width = float(env.width_um)
    start = cosine_mean(state, width)
    step = S.contract_attr("step")
    for _ in range(ticks):
        step(state, env, tbl, rngs)
    return start, cosine_mean(state, width), len(state.agents.agent_id)


def test_chemotaxis_moves_up_gradient(frozen, monkeypatch):
    inert, blind, ec = frozen
    start, end, n = drift(monkeypatch, inert, ec)
    b_start, b_end, _ = drift(monkeypatch, blind, ec)
    print(f"\nchemotactic: mean cos {start:+.3f} -> {end:+.3f} (n={n});  blind: {b_start:+.3f} -> {b_end:+.3f}")
    assert start == pytest.approx(b_start, abs=1e-12), "both runs must start from identical positions"
    assert abs(start) < 0.15, "agents must start spread uniformly in x"
    assert end > start + 0.25, f"chemotactic agents must climb the gradient: {start:+.3f} -> {end:+.3f}"
    assert abs(b_end - b_start) < 0.12, f"agents that cannot sense it must not climb: {b_start:+.3f} -> {b_end:+.3f}"
    assert end > b_end + 0.2, "the climb must be caused by sensitivity, not by motility alone"


def test_null_control_a_flat_field_gives_no_drift(frozen, monkeypatch):
    inert, _, ec = frozen
    start, end, _ = drift(monkeypatch, inert, ec, amplitude=0.0)
    assert abs(end - start) < 0.12, f"no gradient, so no net drift: {start:+.3f} -> {end:+.3f}"


def test_an_engine_that_descends_the_gradient_is_caught(frozen, monkeypatch):
    """Mutation: the movement step sees the negated field, i.e. a reversed gradient. The climb must fail."""
    inert, blind, ec = frozen
    original = engine._chemotaxis_and_movement

    def reversed_gradient(state, env, traits, motility_rng):
        saved = env.concentration_mmol_l
        env.concentration_mmol_l = -saved
        try:
            original(state, env, traits, motility_rng)
        finally:
            env.concentration_mmol_l = saved

    monkeypatch.setattr(engine, "_chemotaxis_and_movement", reversed_gradient)
    with pytest.raises(AssertionError, match="chemotactic agents must climb the gradient"):
        test_chemotaxis_moves_up_gradient(frozen, monkeypatch)
    start, end, _ = drift(monkeypatch, inert, ec)
    assert end < start - 0.25, f"reversed gradient should send agents to the trough: {start:+.3f} -> {end:+.3f}"


def test_chemotaxis_is_deterministic(frozen, monkeypatch):
    inert, _, ec = frozen
    assert drift(monkeypatch, inert, ec, ticks=120) == drift(monkeypatch, inert, ec, ticks=120)
