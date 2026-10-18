"""Mutation receipts against the REAL engine: break step() from outside, never edit src/.

Each case wraps microbiome's step (or initialize_rngs) with one deliberate defect and
asserts the matching QA check goes red. If any of these passed on a broken engine,
the determinism / invariant / ledger suite would be decoration.
"""
import time

import numpy as np
import pytest

import sim_support
import test_sim_contract as tsc
from sim_support import run_checked

REAL = sim_support.contract_attr


def break_step(monkeypatch, post=None, init=None):
    real_step = REAL("step")

    def step(state, env, ph, rng):
        real_step(state, env, ph, rng)
        if post:
            post(state, env)

    def attr(name):
        if name == "step":
            return step
        if name == "initialize_rngs" and init:
            return init
        return REAL(name)

    monkeypatch.setattr(sim_support, "contract_attr", attr)


# ----------------------------------------------------------- determinism cheats
def test_global_rng_leak_is_caught(monkeypatch):
    def post(s, e):
        s.agents.velocity_um_s[:, 0] += np.random.random(len(s.agents.agent_id)) * 1e-9

    break_step(monkeypatch, post)
    with pytest.raises(AssertionError, match="global RNG"):
        tsc.test_immune_to_global_rng_and_wall_clock()


def test_wall_clock_leak_is_caught(monkeypatch):
    def post(s, e):
        # perf_counter_ns: time.time_ns() ticks too coarsely on Windows for back-to-back runs to differ
        s.agents.velocity_um_s[:, 0] += (time.perf_counter_ns() % 1000) * 1e-15

    break_step(monkeypatch, post)
    with pytest.raises(AssertionError, match="diverge"):
        tsc.test_same_seed_same_digest_every_tick()
    with pytest.raises(AssertionError, match="differs between runs"):
        tsc.test_agent_arrays_identical_at_every_tick_including_after_division_and_death()


def test_ignored_seed_is_caught(monkeypatch):
    real_init = REAL("initialize_rngs")
    break_step(monkeypatch, init=lambda seed: real_init(0))
    with pytest.raises(AssertionError, match="seed is ignored"):
        tsc.test_engine_seed_changes_the_physics_not_just_the_rng_state()


# ------------------------------------------------------- invariant / ledger cheats
@pytest.mark.parametrize("name,post,needle", [
    ("nan", lambda s, e: s.agents.energy_pj.__setitem__(0, np.nan), "non-finite"),
    ("edge_x", lambda s, e: s.agents.position_um.__setitem__((0, 0), e.width_um), "position outside"),
    ("neg_concentration", lambda s, e: e.field.concentration_mmol_l.__setitem__((0, 0, 0), -1e-30),
     "negative concentration"),
    ("id_reuse", lambda s, e: setattr(s, "next_agent_id", np.uint64(1)), "id would be reused"),
    ("unsorted_ids", lambda s, e: s.agents.agent_id.__setitem__(slice(0, 2), s.agents.agent_id[1::-1].copy()),
     "strictly increasing"),
    ("negative_mass", lambda s, e: s.agents.mass_pg.__setitem__(0, -1.0), "negative mass"),
])
def test_each_impossible_state_is_caught_on_the_real_engine(monkeypatch, name, post, needle):
    # One step only: the engine validates its input and would raise on a corrupted state at the
    # next step, so the QA checker must see the corruption on the tick it appears.
    break_step(monkeypatch, post)
    log = run_checked(tsc.SEED, 1)
    assert any(needle in v for r in log for v in r["violations"]), (name, log[0]["violations"])


def test_nutrient_leak_is_caught_by_the_ledger(monkeypatch):
    break_step(monkeypatch, lambda s, e: e.field.concentration_mmol_l.__imul__(1 - 1e-3))
    log = run_checked(tsc.SEED, 1)
    assert log[0]["ledger_excess"] > 0 and log[0]["ledger_tight_excess"] > 0


def test_small_leak_hides_inside_d1_atol_but_not_the_scale_aware_check(monkeypatch):
    """FINDING: a 1e-9 relative leak (~5e-19 mmol here) is inside D1's atol=1e-15 mmol, so the
    D1-tolerance ledger cannot see it. Only the scale-aware 1e-12*inventory check does."""
    break_step(monkeypatch, lambda s, e: e.field.concentration_mmol_l.__imul__(1 - 1e-9))
    r = run_checked(tsc.SEED, 1)[0]
    assert r["ledger_excess"] <= 0, "D1 tolerance now sees a 1e-9 leak: tolerance was tightened, update this test"
    assert r["ledger_tight_excess"] > 0


def test_nutrient_creation_is_caught_by_the_ledger(monkeypatch):
    break_step(monkeypatch, lambda s, e: e.field.detritus_mmol.__iadd__(1e-12))
    assert any(r["ledger_excess"] > 0 for r in run_checked(tsc.SEED, 1))


def test_skipped_tick_is_caught(monkeypatch):
    break_step(monkeypatch, lambda s, e: setattr(s, "tick_index", np.uint64(int(s.tick_index) + 1)))
    assert any(r["tick_delta"] != 1 for r in run_checked(tsc.SEED, 1))


def test_unmutated_engine_is_clean_for_the_same_one_step_checks():
    # Control for the mutation tests above: with no defect the same checks report nothing.
    log = run_checked(tsc.SEED, 1)
    assert log[0]["violations"] == [] and log[0]["ledger_excess"] <= 0 and log[0]["tick_delta"] == 1
