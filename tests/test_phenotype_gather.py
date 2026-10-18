"""Per-agent trait gather: values, validation and cache invalidation of ``phenotype_arrays``.

The gather prepares each table once (sort, uniqueness, defaults, per-genome validity mask) and then does one
row lookup plus takes per call. These pin that this is observationally the same as indexing the table
directly, and that validation still fires exactly when a used row is bad.
"""
import dataclasses

import numpy as np
import pytest

import test_competition as tc
from microbiome.sim.model import phenotype_arrays

K = 2
FIELDS = ("max_uptake_mmol_s", "storage_fraction", "biomass_fraction", "energy_yield_pj_per_mmol",
          "motility_um_s", "chemotaxis_sensitivity", "random_motility_fraction", "division_rate_s",
          "division_mass_pg", "death_rate_s", "max_age_s", "antagonism_radius_um", "antagonism_rate_s")


@pytest.fixture(scope="module")
def base():
    genomes = {a: tc.load_genome(a) for a in (tc.EC, tc.BS)}
    table = tc.build_phenotype_table(list(genomes.values()), tc.CHANNELS)
    return table, [int(i) for i in table.genome_id]


def agents(ids, rng, n=64):
    return np.asarray(ids, dtype=np.uint64)[rng.integers(0, len(ids), n)]


def test_gathered_values_are_the_table_rows_bit_for_bit(base):
    table, ids = base
    g = agents(ids, np.random.default_rng(1))
    out = phenotype_arrays(table, g, K)
    rows = np.searchsorted(table.genome_id, g)
    for name in FIELDS:
        assert np.array_equal(getattr(out, name), getattr(table, name)[rows]), name
        assert getattr(out, name).flags.c_contiguous


def test_an_invalid_row_raises_only_if_an_agent_uses_it(base):
    table, ids = base
    bad = tc.edit(table, death_rate_s=(1, -1.0))
    only_first = np.full(8, ids[0], dtype=np.uint64)
    phenotype_arrays(bad, only_first, K)                                   # row 1 unused: fine
    with pytest.raises(ValueError, match="death_rate_s must be nonnegative"):
        phenotype_arrays(bad, np.asarray([ids[0], ids[1]], dtype=np.uint64), K)
    nan = tc.edit(table, max_uptake_mmol_s=(1, 0, float("nan")))
    phenotype_arrays(nan, only_first, K)
    with pytest.raises(ValueError, match="max_uptake_mmol_s contains invalid values"):
        phenotype_arrays(nan, np.asarray([ids[1]], dtype=np.uint64), K)
    over = tc.edit(table, storage_fraction=(0, 0, 0.9), biomass_fraction=(0, 0, 0.9))
    with pytest.raises(ValueError, match="storage_fraction \\+ biomass_fraction"):
        phenotype_arrays(over, np.asarray([ids[0]], dtype=np.uint64), K)


def test_a_genome_missing_from_the_table_is_an_error(base):
    table, ids = base
    with pytest.raises(ValueError, match="missing a \\(genome_id, variant_id\\)"):
        phenotype_arrays(table, np.asarray([ids[0], max(ids) + 1], dtype=np.uint64), K)
    with pytest.raises(ValueError, match="missing a \\(genome_id, variant_id\\)"):
        phenotype_arrays(table, np.asarray([ids[0]], dtype=np.uint64), K, np.asarray([1], dtype=np.uint64))


def test_variant_keyed_tables_select_the_right_row(base):
    table, ids = base
    d = {f.name: getattr(table, f.name) for f in dataclasses.fields(table)}
    d["genome_id"] = np.repeat(table.genome_id, 2)
    d["variant_id"] = np.tile(np.asarray([0, 7], dtype=np.uint64), len(ids))
    for name in FIELDS:
        d[name] = np.repeat(d[name], 2, axis=0)
    d["motility_um_s"] = d["motility_um_s"].copy()
    d["motility_um_s"][1::2] = 99.0                                       # every variant 7 row is distinctive
    table2 = type(table)(**d)
    g = np.asarray([ids[0], ids[0], ids[1]], dtype=np.uint64)
    v = np.asarray([0, 7, 7], dtype=np.uint64)
    out = phenotype_arrays(table2, g, K, v)
    assert out.motility_um_s.tolist() == [float(table.motility_um_s[0]), 99.0, 99.0]


def test_in_place_edits_to_a_table_are_seen_on_the_next_gather(base):
    table, ids = base
    local = tc.edit(table)                       # private copy; edit() copies every array
    g = np.asarray([ids[0]], dtype=np.uint64)
    before = float(phenotype_arrays(local, g, K).motility_um_s[0])
    local.motility_um_s[0] = before + 5.0
    assert float(phenotype_arrays(local, g, K).motility_um_s[0]) == before + 5.0


def test_repeat_calls_with_changing_populations_stay_correct(base):
    table, ids = base
    rng = np.random.default_rng(9)
    for n in (1, 7, 200, 3, 64):
        g = agents(ids, rng, n)
        out = phenotype_arrays(table, g, K)
        rows = np.searchsorted(table.genome_id, g)
        assert np.array_equal(out.motility_um_s, table.motility_um_s[rows])
