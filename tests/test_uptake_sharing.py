"""D1 uptake allocation: a pool that is scarce *within one tick* is shared in proportion to demand.

tests/test_competition.py measures competition in the rate-limited regime, where the pool drains over
75-149 ticks and any allocator that serves equal requests equally gives the same 50/50 split. It therefore
cannot tell proportional sharing from first-come-first-served (see the xfail in
tests/test_competition_mutations.py). This file uses the regime where the allocator is the only thing that
matters: the pool in a cell is smaller than that cell's demand for the very first tick, and the two
species' demands differ 3:1.

Prediction (D1, "proportional-shared"): in every saturated cell, every agent is granted the same *fraction*
of its request, whatever its species and agent_id. Both null mutations break that: first-come-first-served
serves low agent_ids in full and starves the rest; equal shares ignore the request.
"""
import numpy as np
import pytest

import sim_support as S
import test_competition as tc
import microbiome.sim.engine as engine
from test_competition_mutations import equal_share_grants, greedy_grants

DEMAND_RATIO = 3.0
SCARCITY = 0.25  # pool in a cell = this fraction of the average cell's one-tick demand


@pytest.fixture(scope="module")
def scenario(table_row):
    table, row = table_row
    quiet = tc.edit(table, antagonism_rate_s=(row[tc.BS], 0.0), antagonism_radius_um=(row[tc.BS], 0.0))
    base = float(quiet.max_uptake_mmol_s[row[tc.EC], tc.GLC])
    assert base > 0.0
    return tc.edit(quiet, max_uptake_mmol_s=(row[tc.BS], tc.GLC, DEMAND_RATIO * base)), row


@pytest.fixture(scope="module")
def table_row():
    genomes = {a: tc.load_genome(a) for a in (tc.EC, tc.BS)}
    table = tc.build_phenotype_table(list(genomes.values()), tc.CHANNELS)
    return table, {a: int(np.flatnonzero(table.genome_id == np.uint64(g.genome_id))[0]) for a, g in genomes.items()}


def one_tick(scenario):
    """Step one tick on a scarce pool; return per-agent granted fraction and the saturated-cell mask."""
    tbl, row = scenario
    assign = np.concatenate([np.full(tc.N, row[tc.EC]), np.full(tc.N, row[tc.BS])]).astype(np.int64)
    request = tbl.max_uptake_mmol_s[assign, tc.GLC] / 60.0
    tick_demand = request.sum() / (tc.CELLS * tc.CELLS)
    conc = SCARCITY * tick_demand / tc.VCELL_L
    state, env, rngs = tc.world(tc.SEED, assign, tbl, prime=(conc, 0.0))
    pos = state.agents.position_um.copy()
    cell = np.floor(pos[:, 1] / tc.DX).astype(int) * tc.CELLS + np.floor(pos[:, 0] / tc.DX).astype(int)
    before = state.agents.biomass_mmol[:, tc.GLC].copy()
    S.contract_attr("step")(state, env, tbl, rngs)
    assert len(state.agents.agent_id) == len(assign) and (np.diff(state.agents.agent_id.astype(np.int64)) > 0).all()
    keep = np.minimum(tbl.biomass_fraction[assign, tc.GLC], 1.0 - tbl.storage_fraction[assign, tc.GLC])
    granted = (state.agents.biomass_mmol[:, tc.GLC] - before) / keep
    pool = conc * tc.VCELL_L
    demand = np.bincount(cell, weights=request, minlength=tc.CELLS * tc.CELLS)
    saturated = demand[cell] > pool
    return granted / request, cell, saturated, assign, row, S.violations(state, env)


def assert_proportional(result):
    fraction, cell, saturated, assign, row, violations = result
    assert violations == []
    cells = [c for c in np.unique(cell[saturated])
             if len({int(a) for a in assign[cell == c]}) == 2]
    assert len(cells) >= 6, "scenario must put both species in several saturated cells"
    for c in cells:
        f = fraction[cell == c]
        assert f.max() == pytest.approx(f.min(), rel=1e-9), \
            f"cell {c}: agents must receive the same fraction of their request, got {f.min():.6g}..{f.max():.6g}"
    assert (fraction[saturated] < 1.0).all(), "saturated cells must actually be rationed"
    assert (fraction[~saturated] == pytest.approx(1.0, rel=1e-9)), "unsaturated cells must be served in full"


def test_a_scarce_pool_is_shared_in_proportion_to_demand(scenario):
    assert_proportional(one_tick(scenario))


def test_first_come_first_served_is_caught(scenario, monkeypatch):
    monkeypatch.setattr(engine, "_proportional_grants", greedy_grants)
    with pytest.raises(AssertionError, match="same fraction of their request"):
        assert_proportional(one_tick(scenario))


def test_equal_shares_that_ignore_demand_are_caught(scenario, monkeypatch):
    monkeypatch.setattr(engine, "_proportional_grants", equal_share_grants)
    with pytest.raises(AssertionError, match="same fraction of their request"):
        assert_proportional(one_tick(scenario))
