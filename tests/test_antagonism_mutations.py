"""Mutation receipts for tests/test_antagonism.py, against the REAL pipeline (src/ untouched).

Each case wraps Simulation.step with one defect, or breaks the test's own control, and requires the antagonism
tests to go red on the assertion meant to catch it (matched by message). If any passed on a broken engine the
antagonism proof would be decoration."""
import dataclasses

import numpy as np
import pytest

import sim_support as S
import test_antagonism as ta
import test_two_genomes as tt

REAL = S.contract_attr


@pytest.fixture(scope="module")
def good():
    genomes = {a: tt.load_genome(a) for a in (tt.EC, tt.BS)}
    table = tt.build_phenotype_table(list(genomes.values()), tt.CHANNELS)
    rows = {a: int(np.flatnonzero(table.genome_id == np.uint64(g.genome_id))[0]) for a, g in genomes.items()}
    return genomes, table, rows


def wrap_step(monkeypatch, table_map=None, pre=None):
    real = REAL("step")

    def step(state, env, ph, rngs):
        if pre:
            pre(state, env)
        real(state, env, table_map(ph) if table_map else ph, rngs)

    monkeypatch.setattr(S, "contract_attr", lambda name: step if name == "step" else REAL(name))


def red(fn, *a, match):
    with pytest.raises(AssertionError, match=match):
        fn(*a)


def test_an_engine_without_antagonism_is_caught(good, monkeypatch):
    genomes, table, rows = good
    wrap_step(monkeypatch, table_map=lambda t: dataclasses.replace(t, antagonism_rate_s=np.zeros_like(t.antagonism_rate_s)))
    red(ta.test_bacilysin_suppresses_a_competitor, genomes, table, rows, match="competitor not suppressed")


def test_a_doubled_hazard_is_caught_by_the_per_agent_oracle_and_the_white_box_check(good, monkeypatch):
    _, table, rows = good
    wrap_step(monkeypatch, table_map=lambda t: dataclasses.replace(t, antagonism_rate_s=t.antagonism_rate_s * 2.0))
    red(ta.test_each_ecoli_dies_at_the_rate_its_own_neighbours_imply, table, rows, match="per-agent expectation")
    red(ta.test_engine_pressure_on_every_agent_equals_rate_times_its_neighbours, table, rows,
        match="engine hazard != rate")


def test_an_attack_radius_twice_too_large_is_caught_by_locality_checks(good, monkeypatch):
    _, table, rows = good
    wrap_step(monkeypatch, table_map=lambda t: dataclasses.replace(t, antagonism_radius_um=t.antagonism_radius_um * 2.0))
    red(ta.test_the_attack_radius_is_respected_to_two_tenths_of_a_micrometre, table, rows,
        match="E. coli died although")
    red(ta.test_engine_pressure_on_every_agent_equals_rate_times_its_neighbours, table, rows,
        match="engine hazard != rate|received hazard")


def test_a_radius_one_micrometre_too_small_is_caught_by_the_boundary_test(good, monkeypatch):
    _, table, rows = good
    shrink = lambda t: dataclasses.replace(  # noqa: E731  only rows that attack get a smaller radius
        t, antagonism_radius_um=np.where(t.antagonism_radius_um > 0, t.antagonism_radius_um - 1.0, t.antagonism_radius_um))
    wrap_step(monkeypatch, table_map=shrink)
    red(ta.test_the_attack_radius_is_respected_to_two_tenths_of_a_micrometre, table, rows,
        match="inside-radius deaths 0 vs expected")


def test_friendly_fire_between_attackers_is_caught(good, monkeypatch):
    genomes, table, rows = good

    def everyone_distinct(state, env):  # every agent a different 'species': attackers now hit each other too
        a = state.agents
        a.species_id = (np.arange(len(a.agent_id)) + 1).astype(np.uint32)

    wrap_step(monkeypatch, pre=everyone_distinct)
    red(ta.test_antagonism_is_specific_no_friendly_fire_and_the_attacker_is_unharmed, table, rows,
        match="no deaths expected")


def test_an_attacker_that_is_itself_harmed_is_caught(good, monkeypatch):
    genomes, table, rows = good
    e, b = rows[tt.EC], rows[tt.BS]
    both = tt.edit(table, antagonism_rate_s=(e, float(table.antagonism_rate_s[b])),
                   antagonism_radius_um=(e, float(table.antagonism_radius_um[b])))
    # E. coli attacks back, but only in the real run (the control, with B. subtilis's rate at 0, is left alone so
    # the failure must come from the 'attacker must not be harmed' assertion and not from the control check)
    wrap_step(monkeypatch, table_map=lambda t: both if t.antagonism_rate_s.max() > 0 else t)
    red(ta.test_bacilysin_suppresses_a_competitor, genomes, table, rows, match="attacker must not be harmed")


def test_a_control_that_changes_more_than_the_antagonism_trait_is_caught(good, monkeypatch):
    _, table, rows = good

    def dirty(t, bb):
        return tt.edit(tt.edit(t, antagonism_rate_s=(bb, 0.0)), death_rate_s=(bb, 2e-5))

    monkeypatch.setattr(ta, "control_table", dirty)
    red(ta.test_the_control_differs_from_the_real_run_only_in_the_antagonism_trait, table, rows,
        match="changes more than the antagonism trait")
