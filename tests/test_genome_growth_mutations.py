"""Mutation receipts for tests/test_genome_growth.py, against the REAL pipeline (src/ untouched).

Each case injects one defect into the engine the growth test sees and requires the population-growth
test to go red. If the test passed on a broken engine it would be decoration, not verification.
The unmutated control is test_genome_growth.py itself, which runs the same functions on the real step.
"""
import numpy as np
import pytest

import sim_support as S
import test_genome_growth as tg
import test_two_genomes as tt

REAL = S.contract_attr
EC, BS = tt.EC, tt.BS


@pytest.fixture(scope="module")
def good():
    genomes = {a: tt.load_genome(a) for a in (EC, BS)}
    table = tt.build_phenotype_table(list(genomes.values()), tt.CHANNELS)
    row = {a: int(np.flatnonzero(table.genome_id == np.uint64(g.genome_id))[0]) for a, g in genomes.items()}
    return table, row


def wrap_step(monkeypatch, table_map=None, post=None):
    real = REAL("step")

    def step(state, env, ph, rngs):
        real(state, env, table_map(ph) if table_map else ph, rngs)
        if post:
            post(state, env)

    monkeypatch.setattr(S, "contract_attr", lambda name: step if name == "step" else REAL(name))


def red(fn, *args):
    with pytest.raises(AssertionError):
        fn(*args)


def test_an_engine_that_gives_both_genomes_the_same_growth_traits_is_caught(good, monkeypatch):
    """Run B. subtilis with E. coli's whole trait row: it then grows too, and the required
    'B. subtilis produces zero births' assertion must fire."""
    table, row = good
    wrap_step(monkeypatch, table_map=lambda t: tt.copy_row(t, row[BS], row[EC]))
    red(tg.test_two_genomes_diverge_in_population_growth, table, row)


def test_an_engine_that_ignores_lactose_uptake_is_caught(good, monkeypatch):
    """Zero the lactose column for every genome: the documented trait can no longer drive growth, so
    E. coli produces no births and the test must reject the engine."""
    table, row = good
    wrap_step(monkeypatch, table_map=lambda t: tt.edit(t, max_uptake_mmol_s=(slice(None), tg.LAC, 0.0)))
    red(tg.test_two_genomes_diverge_in_population_growth, table, row)


def test_an_engine_that_grows_mass_but_never_divides_is_caught(good, monkeypatch):
    """Keep uptake, remove the division hazard: mass is gained but no population grows, which is
    exactly the 'uptake is not population growth' failure this item exists to catch."""
    table, row = good
    wrap_step(monkeypatch, table_map=lambda t: tt.edit(t, division_rate_s=(slice(None), 0.0)))
    red(tg.test_two_genomes_diverge_in_population_growth, table, row)


def test_a_label_dependent_nudge_is_caught_by_the_null_control(good, monkeypatch):
    table, row = good
    b_id = table.genome_id[row[BS]]

    def post(state, env):
        agents = state.agents
        agents.position_um[agents.genome_id == b_id, 0] += 1e-9

    wrap_step(monkeypatch, post=post)
    red(tg.test_null_control_the_divergence_is_the_lactose_trait_not_the_genome_label, table, row)
