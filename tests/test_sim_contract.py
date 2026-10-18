"""Black-box acceptance tests for Simulation.step against frozen D1.

Reproduce:   python -m pytest -q tests/test_sim_contract.py
They fail today with ImportError because microbiome.sim.engine / core.contracts do
not exist yet. That is the intended starting state for backlog #5.
"""
import json
import random
import time

import numpy as np
import pytest

import sim_support
from sim_support import (
    build_world, collect, contract_attr, find, ledger_ok, ledger_residuals, run_checked,
    run_trace, violations,
)

TICKS = 40
SEED = 20261017


# ------------------------------------------------------------------ determinism
def test_same_seed_same_digest_every_tick():
    a = run_trace(SEED, TICKS)
    b = run_trace(SEED, TICKS)
    first_bad = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    assert first_bad is None, f"runs diverge at tick {first_bad}"
    assert len(set(a)) == len(a), "state never changed between ticks: the trace proves nothing"


def test_agent_arrays_identical_at_every_tick_including_after_division_and_death():
    """Two runs, same seed and same initial state, stepped in lockstep: every agent array
    must be bitwise equal at every tick, and that must include ticks after a birth and a death."""
    from sim_support import AGENT_ROWS

    step = sim_support.contract_attr("step")  # looked up on the module so mutation tests can patch it
    sa, ea, pa, ra = build_world(SEED)
    sb, eb, pb, rb = build_world(SEED)
    ids = lambda s: find(collect(s, "s"), "agent_id")  # noqa: E731
    assert collect_equal(sa, sb), "the two initial states differ: test premise broken"
    prev, births_seen, deaths_seen, checked_after_birth, checked_after_death = ids(sa).copy(), 0, 0, 0, 0
    for t in range(1, 301):
        step(sa, ea, pa, ra)
        step(sb, eb, pb, rb)
        ca, cb = collect(sa, "s"), collect(sb, "s")
        for name in AGENT_ROWS:
            xa, xb = find(ca, name), find(cb, name)
            assert xa.dtype == xb.dtype and xa.shape == xb.shape, f"tick {t}: {name} dtype/shape differ"
            assert xa.tobytes() == xb.tobytes(), f"tick {t}: agent array '{name}' differs between runs"
        cur = ids(sa)
        born, died = np.setdiff1d(cur, prev).size, np.setdiff1d(prev, cur).size
        births_seen += born
        deaths_seen += died
        checked_after_birth += births_seen > 0
        checked_after_death += deaths_seen > 0
        prev = cur.copy()
    assert births_seen > 0 and deaths_seen > 0, "no division/death happened: the claim was not exercised"
    assert checked_after_birth > 10 and checked_after_death > 10, "too few ticks compared after birth/death"


def collect_equal(a, b):
    ca, cb = collect(a, "s"), collect(b, "s")
    return ca.keys() == cb.keys() and all(ca[k].tobytes() == cb[k].tobytes() for k in ca)


def test_engine_seed_changes_the_physics_not_just_the_rng_state():
    # Identical world, only the engine seed differs; RNG states are excluded from
    # the digest so a difference can only come from the simulation actually using them.
    a = run_trace(SEED, TICKS, include_rngs=False, engine_seed=1)
    b = run_trace(SEED, TICKS, include_rngs=False, engine_seed=2)
    assert a[-1] != b[-1], "different engine seeds gave identical state: seed is ignored"


def test_immune_to_global_rng_and_wall_clock():
    def noise():
        np.random.seed(int(time.time_ns() % 2**31))
        random.seed(time.time_ns())
        np.random.random(1000)
        random.random()
        time.sleep(0.001)

    clean = run_trace(SEED, 25)
    noisy = run_trace(SEED, 25, between=noise)
    first_bad = next((i for i, (x, y) in enumerate(zip(clean, noisy)) if x != y), None)
    assert first_bad is None, f"global RNG / wall clock leaks into step(): diverges at tick {first_bad}"


def _trace_code(extra=""):
    return (
        "import json;from sim_support import run_trace;"
        f"print(json.dumps(run_trace({SEED}, 25{extra})))"
    )


def test_fresh_processes_with_different_hashseeds_reproduce(fresh_python):
    here = run_trace(SEED, 25)
    for hs in ("0", "12345"):
        there = json.loads(fresh_python(_trace_code(), hashseed=hs))
        first_bad = next((i for i, (x, y) in enumerate(zip(here, there)) if x != y), None)
        assert first_bad is None, f"PYTHONHASHSEED={hs}: diverges from in-process at tick {first_bad}"


def test_thread_count_does_not_change_results(fresh_python):
    one = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    a = json.loads(fresh_python(_trace_code(), extra_env=one))
    b = json.loads(fresh_python(_trace_code()))
    assert a == b


def test_row_order_at_step_entry_does_not_matter():
    base = run_trace(SEED, 20)
    shuffled = run_trace(SEED, 20, permute_rows=True)
    assert base[0] != shuffled[0], "permutation was a no-op; test is vacuous"
    # D1: rows may be permuted at entry, committed rows are canonical. From tick 1 on, identical.
    first_bad = next((i for i in range(1, 21) if base[i] != shuffled[i]), None)
    assert first_bad is None, f"row order leaks into the result at tick {first_bad}"


# ------------------------------------------------------------- invariants/ledger
@pytest.fixture(scope="module")
def mixed_log():
    return run_checked(SEED, 300)


def test_scenario_exercises_birth_and_death(mixed_log):
    # Guards the two tests below against passing on a world where nothing happens.
    assert sum(r["births"] for r in mixed_log) > 0, "no division occurred in 300 ticks"
    assert sum(r["deaths"] for r in mixed_log) > 0, "no death occurred in 300 ticks"


def test_no_impossible_state_on_any_tick(mixed_log):
    bad = [(r["tick"], r["violations"]) for r in mixed_log if r["violations"]]
    assert not bad, f"first violation: {bad[0]}"


def test_tick_advances_by_exactly_one(mixed_log):
    assert [r["tick"] for r in mixed_log] == list(range(1, 301))
    assert all(r["tick_delta"] == 1 for r in mixed_log)


def test_nutrient_ledger_closes_every_tick_per_species(mixed_log):
    bad = [(r["tick"], r["ledger_resid"]) for r in mixed_log if r["ledger_excess"] > 0]
    assert not bad, f"ledger open at tick {bad[0][0]} by {bad[0][1]:.3e} mmol (rtol 1e-12/atol 1e-15)"


def test_ledger_closes_to_1e12_of_the_nutrient_inventory(mixed_log):
    """Stricter than D1's atol 1e-15 mmol, which is ~2e-6 of this world's field inventory
    (cells are ~1e-13 L) and would hide a leak of that size. Scale-aware: 1e-12*(initial+source)."""
    bad = [(r["tick"], r["ledger_tight_excess"]) for r in mixed_log if r["ledger_tight_excess"] > 0]
    assert not bad, f"ledger open beyond 1e-12 of inventory at tick {bad[0][0]} (excess {bad[0][1]:.3e} mmol)"


def test_measured_ledger_error_is_reported(mixed_log, record_property):
    worst = max(r["ledger_resid"] for r in mixed_log)
    record_property("max_ledger_residual_mmol_300_ticks", worst)
    print(f"max |ledger residual| over 300 ticks = {worst:.3e} mmol")


def test_rng_streams_use_the_d1_purpose_tags():
    """D1 conformance (item 210): the purpose tags are frozen at motility=1, division=2, mutation=3,
    death=4, antagonism=5. Source now conforms, so this is a normal regression test."""
    d1 = {"motility": 1, "division": 2, "mutation": 3, "death": 4, "antagonism": 5}
    rngs = contract_attr("initialize_rngs")(SEED)
    for purpose, tag in d1.items():
        assert hasattr(rngs, purpose), f"no '{purpose}' stream"
        want = np.random.PCG64(np.random.SeedSequence(SEED, spawn_key=(tag,))).state
        assert getattr(rngs, purpose).bit_generator.state == want, f"'{purpose}' is not spawn_key=({tag},)"


# ----------------------------------------------- strict concentration (D1 + QA M1)
def test_full_cell_uptake_never_makes_concentration_negative():
    """Every cell is fully consumed in one tick. With c_new = c - grant/V about 4.7% of such
    cells go ~1 ulp negative (measured, 1e6 cells). The exact form (c*V - grant)/V gives 0."""
    step, ledger = sim_support.contract_attr("step"), sim_support.contract_attr("ledger")
    state, env, pheno, rngs = build_world(SEED, scenario="full_cell")
    before = find(collect(env, "env"), "concentration_mmol_l").copy()
    step(state, env, pheno, rngs)
    after = find(collect(env, "env"), "concentration_mmol_l")
    assert (after < before).all(), "nothing was consumed: scenario is vacuous"
    assert (after == 0).mean() > 0.95, "cells were not fully consumed: scenario is vacuous"
    assert after.min() >= 0.0, f"negative concentration {after.min():.3e} mmol/L"
    assert violations(state, env) == []
    assert ledger_ok(ledger(state, env)), ledger_residuals(ledger(state, env))
