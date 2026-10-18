"""Test the tests: the contract suite must pass on a correct toy engine and go red on each cheat.

Uses tests/toy_engine.py (NOT the simulation). Mutation receipts for the QA suite.
"""
import pytest

import sim_support
import test_sim_contract as tsc
import toy_engine
from sim_support import build_world, collect, find, ledger_ok, run_checked, violations


@pytest.fixture
def toy(monkeypatch):
    def attr(name):
        return getattr(toy_engine, name)

    monkeypatch.setattr(sim_support, "contract_attr", attr)
    monkeypatch.setattr(sim_support, "_contract_modules", lambda: [toy_engine])
    yield toy_engine
    toy_engine.MODE = "good"


@pytest.mark.parametrize("scenario", ["mixed", "full_cell"])
def test_built_initial_world_is_legal_and_ledger_closed(toy, scenario):
    state, env, pheno, rngs = build_world(7, scenario=scenario)
    assert violations(state, env) == []
    assert ledger_ok(toy_engine.ledger(state, env))
    assert isinstance(env.field, toy_engine.NutrientField)  # nested construction worked


def test_permute_helper_really_shuffles_every_agent_array_together(toy):
    state, *_ = build_world(7)
    before = {k: v.copy() for k, v in collect(state, "s").items()}
    perm = sim_support.permute_agent_rows(state, __import__("numpy").random.default_rng(3))
    after = collect(state, "s")
    ids_b, ids_a = find(before, "agent_id"), find(after, "agent_id")
    assert (ids_a != ids_b).any()
    assert (ids_a == ids_b[perm]).all()
    assert (find(after, "position_um") == find(before, "position_um")[perm]).all()


def test_good_toy_passes_the_determinism_tests(toy):
    tsc.test_same_seed_same_digest_every_tick()
    tsc.test_engine_seed_changes_the_physics_not_just_the_rng_state()
    tsc.test_immune_to_global_rng_and_wall_clock()
    tsc.test_row_order_at_step_entry_does_not_matter()


def test_good_toy_passes_invariants_and_ledger(toy):
    log = run_checked(11, 60)
    assert all(not r["violations"] for r in log)
    assert all(r["ledger_excess"] <= 0 for r in log)
    assert [r["tick"] for r in log] == list(range(1, 61))


def test_toy_has_no_births_so_the_vacuity_guard_fires(toy):
    log = run_checked(11, 60)
    assert sum(r["births"] for r in log) == 0 and sum(r["deaths"] for r in log) == 0


@pytest.mark.parametrize("mode,test", [
    ("global_rng", "test_immune_to_global_rng_and_wall_clock"),
    ("wallclock", "test_immune_to_global_rng_and_wall_clock"),
    ("wallclock", "test_same_seed_same_digest_every_tick"),
    ("no_canon", "test_row_order_at_step_entry_does_not_matter"),
    ("ignore_seed", "test_engine_seed_changes_the_physics_not_just_the_rng_state"),
])
def test_each_cheat_turns_a_determinism_test_red(toy, mode, test):
    toy_engine.MODE = mode
    with pytest.raises(AssertionError):
        getattr(tsc, test)()


def test_leaky_ledger_is_caught(toy):
    toy_engine.MODE = "leaky_ledger"
    log = run_checked(11, 30)
    assert any(r["ledger_excess"] > 0 for r in log)
