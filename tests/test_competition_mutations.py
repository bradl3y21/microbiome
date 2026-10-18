"""Mutation receipts for tests/test_competition.py, against the REAL pipeline (src/ untouched).

Each case breaks exactly one thing the competition test claims to prove and requires that test to go red on the
assertion meant to catch it (matched by message). If the competition test still passed on a broken engine, its
half-harvest claim would be decoration.
"""
import numpy as np
import pytest

import microbiome.sim.engine as engine
import test_competition as tc


def greedy_grants(requests, available, keys):
    """Mutation: first-come-first-served uptake instead of proportional sharing.

    Lower agent IDs drain a cell before anyone else is served, so in a co-culture the resident population (rows
    first) monopolises the pool and is not suppressed at all.
    """
    grants = np.zeros(requests.shape, dtype=np.float64)
    available = np.asarray(available, dtype=np.float64)
    remaining = available.copy()
    for i in range(requests.size):
        grant = min(float(requests[i]), max(float(remaining[keys[i]]), 0.0))
        grants[i] = grant
        remaining[keys[i]] -= grant
    # Report the field loss as (available - remaining) so the engine's own subtraction cannot go negative,
    # which keeps this mutation a model change rather than a conservation crash.
    return grants, np.minimum(available, available - remaining)


def equal_share_grants(requests, available, keys):
    """Mutation: ignore the request entirely and give every agent in a cell an equal share.

    A neighbour with zero uptake then still takes half the nutrient, so the null control no longer holds.
    """
    available = np.asarray(available, dtype=np.float64)
    counts = np.zeros_like(available)
    np.add.at(counts, keys, 1.0)
    per_agent = np.divide(available, np.maximum(counts, 1.0), out=np.zeros_like(counts), where=counts > 0.0)
    per_agent = np.nextafter(per_agent, 0.0)
    grants = per_agent[keys]
    granted = np.zeros_like(per_agent)
    np.add.at(granted, keys, grants)
    return grants, np.minimum(granted, available)


def red(fn, *args, match):
    with pytest.raises(AssertionError, match=match):
        fn(*args)


@pytest.fixture(scope="module")
def tables():
    """The same frozen table + rows + antagonism-off copy the competition tests use."""
    genomes = {a: tc.load_genome(a) for a in (tc.EC, tc.BS)}
    table = tc.build_phenotype_table(list(genomes.values()), tc.CHANNELS)
    rows = {a: int(np.flatnonzero(table.genome_id == np.uint64(g.genome_id))[0]) for a, g in genomes.items()}
    quiet = tc.edit(table, antagonism_rate_s=(rows[tc.BS], 0.0), antagonism_radius_um=(rows[tc.BS], 0.0))
    return table, rows, quiet


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason=(
    "MEASURED TEST-POWER GAP (not an engine bug): at the current calibration the pool drains over 75-149 ticks "
    "and both species draw at equal rates until it is empty, so first-come-first-served shifts E's harvest by "
    "~0.7% (0.016329 vs 0.016214 pg), inside test_competition's 5% tolerance. The competition test cannot "
    "distinguish proportional sharing from greedy. The allocator is covered where it matters, a pool scarce "
    "within one tick with 3:1 demands, by tests/test_uptake_sharing.py (both mutations caught there). "
    "Goes red if test_competition ever regains the power to catch this; then delete this mark."))
def test_an_engine_without_proportional_uptake_is_caught(tables, monkeypatch):
    """No sharing -> the resident keeps the whole pool -> 'competition must suppress both populations' fails."""
    table, row, quiet = tables
    monkeypatch.setattr(engine, "_proportional_grants", greedy_grants)
    red(tc.test_competition_changes_the_outcome, table, row, quiet,
        match="competition must suppress both populations")


def test_uptake_that_ignores_the_request_is_caught_by_the_null_control(tables, monkeypatch):
    """If a zero-uptake neighbour still takes nutrient, the null control must go red, not silently pass."""
    table, row, quiet = tables
    monkeypatch.setattr(engine, "_proportional_grants", equal_share_grants)
    red(tc.test_the_null_control_a_non_competing_neighbour_changes_nothing, table, row, quiet,
        match="a neighbour with no uptake must harvest nothing")


def test_the_half_harvest_measurement_rejects_a_run_with_no_competition(tables):
    """The main assertion must be false for a world in which the neighbour cannot take up the shared nutrient."""
    table, row, quiet = tables
    alone = tc.run(tc.SEED, [(row[tc.EC], tc.N)], quiet)["per"][row[tc.EC]]["gain_pg"]
    inert = tc.edit(quiet, max_uptake_mmol_s=(row[tc.BS], tc.GLC, 0.0))
    unopposed = tc.run(tc.SEED, [(row[tc.EC], tc.N), (row[tc.BS], tc.N)], inert)["per"][row[tc.EC]]["gain_pg"]
    assert unopposed == pytest.approx(alone, rel=1e-12)
    with pytest.raises(AssertionError):
        assert unopposed < alone
    with pytest.raises(AssertionError):
        assert unopposed == pytest.approx(0.5 * alone, rel=0.05)
