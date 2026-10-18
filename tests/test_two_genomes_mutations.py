"""Mutation receipts for tests/test_two_genomes.py, against the REAL pipeline (src/ untouched).

Each case wraps Simulation.step (or edits the table the test sees) with one defect and requires the
two-genome tests to go red. If any passed on a broken pipeline the proof would be decoration."""
import dataclasses

import numpy as np
import pytest

import sim_support as S
import test_two_genomes as tt

REAL = S.contract_attr


@pytest.fixture(scope="module")
def good():
    genomes = {a: tt.load_genome(a) for a in (tt.EC, tt.BS)}
    table = tt.build_phenotype_table(list(genomes.values()), tt.CHANNELS)
    row = {a: int(np.flatnonzero(table.genome_id == np.uint64(g.genome_id))[0]) for a, g in genomes.items()}
    return genomes, table, row


def wrap_step(monkeypatch, table_map=None, post=None):
    real = REAL("step")

    def step(state, env, ph, rngs):
        real(state, env, table_map(ph) if table_map else ph, rngs)
        if post:
            post(state, env)

    monkeypatch.setattr(S, "contract_attr", lambda name: step if name == "step" else REAL(name))


def red(fn, *a, match=None):
    with pytest.raises(AssertionError, match=match):
        fn(*a)


# (The unmutated control is test_two_genomes.py itself: the same functions run there against the real pipeline.)


def test_an_engine_that_ignores_genome_differences_is_caught(good, monkeypatch):
    genomes, table, row = good
    wrap_step(monkeypatch, table_map=lambda t: tt.copy_row(t, row[tt.BS], row[tt.EC]))  # B. subtilis run as E. coli
    red(tt.test_pure_cultures_diverge_in_lactose_consumption_exactly_as_the_table_predicts, table, row,
        match="E. coli must consume and B. subtilis must not")
    red(tt.test_two_genomes_produce_distinguishable_populations, genomes, table, row)


def test_an_engine_whose_consumption_is_one_percent_off_the_table_is_caught(good, monkeypatch):
    _, table, row = good
    scaled = lambda t: dataclasses.replace(t, max_uptake_mmol_s=t.max_uptake_mmol_s * 1.01)  # noqa: E731
    wrap_step(monkeypatch, table_map=scaled)
    red(tt.test_pure_cultures_diverge_in_lactose_consumption_exactly_as_the_table_predicts, table, row,
        match="!= table prediction")


def test_an_engine_without_antagonism_is_caught(good, monkeypatch):
    genomes, table, row = good
    none = lambda t: dataclasses.replace(t, antagonism_rate_s=np.zeros_like(t.antagonism_rate_s))  # noqa: E731
    wrap_step(monkeypatch, table_map=none)
    red(tt.test_coculture_diverges_in_survival_and_matches_the_antagonism_prediction, table, row, match="'dead'")
    red(tt.test_two_genomes_produce_distinguishable_populations, genomes, table, row)


def test_mortality_that_hits_both_species_equally_is_not_mistaken_for_antagonism(good, monkeypatch):
    """A ledger-clean defect: every genome gets a high death rate, so both species die in equal numbers whatever
    the antagonism traits say. The survival tests must refuse it (B. subtilis must not die; switching the
    antagonism off must stop the deaths)."""
    _, table, row = good
    wrap_step(monkeypatch, table_map=lambda t: dataclasses.replace(t, death_rate_s=np.full_like(t.death_rate_s, 0.01)))
    red(tt.test_the_survival_divergence_follows_the_named_antagonism_traits, table, row,
        match="E. coli must die and B. subtilis must not")
    red(tt.test_coculture_diverges_in_survival_and_matches_the_antagonism_prediction, table, row, match="'dead'")


def test_a_hidden_dependence_on_genome_identity_is_caught_by_the_null_control(good, monkeypatch):
    _, table, row = good
    b_id = table.genome_id[row[tt.BS]]

    def post(state, env):
        a = state.agents
        a.position_um[a.genome_id == b_id, 0] += 1e-9  # a label-dependent nudge, identical traits otherwise

    wrap_step(monkeypatch, post=post)
    red(tt.test_null_control_the_engine_has_no_hidden_dependence_on_genome_identity, table, row, match="differs")


def test_a_table_that_no_longer_differs_in_the_named_cells_is_caught(good):
    genomes, table, row = good
    same = tt.copy_row(table, row[tt.BS], row[tt.EC])
    red(tt.test_the_frozen_table_differs_in_exactly_three_named_traits, same, row, match="missing")
    extra = tt.edit(table, max_age_s=(row[tt.BS], 1.0))  # an unnamed extra difference
    red(tt.test_the_frozen_table_differs_in_exactly_three_named_traits, extra, row, match="max_age_s")


def test_a_difference_with_no_explaining_gene_rule_is_caught(good):
    genomes, table, row = good
    drift = tt.edit(table, antagonism_rate_s=(row[tt.BS], 0.002))  # table value no longer equals the rule sum
    red(tt.test_each_difference_is_the_sum_of_named_gene_rules_and_the_genes_really_differ, genomes, drift, row, match="rule sum")
