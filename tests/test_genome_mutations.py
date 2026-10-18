"""Mutation receipts for tests/test_genome.py: corrupt the module's OUTPUT and require the independent
oracle tests to go red. If any of these passed on broken output, the genome suite would be decoration.
Nothing under src/ or data/ is edited; defects are injected in memory."""
import copy
import dataclasses

import numpy as np
import pytest

import test_genome as tg

A = tg.ACCESSIONS


@pytest.fixture(scope="module")
def good():
    genomes = {a: tg.load_genome(a) for a in A}
    oracles = {e["accession"]: tg.oracle_gff(tg.ROOT / e["annotation_file"]["path"]) for e in tg.MANIFEST["genomes"]}
    return genomes, oracles


def red(fn, *a):
    with pytest.raises(AssertionError):
        fn(*a)


def test_control_unmutated_output_passes_the_same_checks(good):
    genomes, oracles = good
    for a in A:
        tg.test_parsed_genes_match_an_independent_gff3_parse(a, genomes, oracles)
        tg.test_pseudogene_and_symbol_flags_match_the_independent_parse(a, genomes, oracles)
        tg.test_every_trait_equals_an_independent_reimplementation_of_the_rule(a, genomes, oracles)


def test_a_dropped_gene_is_caught(good):
    genomes, oracles = good
    bad = {a: dataclasses.replace(g, genes=g.genes[:-1]) for a, g in genomes.items()}
    red(tg.test_parsed_genes_match_an_independent_gff3_parse, A[0], bad, oracles)


def test_a_shifted_coordinate_is_caught(good):
    genomes, oracles = good
    g = genomes[A[0]]
    genes = (dataclasses.replace(g.genes[10], start=g.genes[10].start + 1),)
    bad = {A[0]: dataclasses.replace(g, genes=g.genes[:10] + genes + g.genes[11:])}
    red(tg.test_parsed_genes_match_an_independent_gff3_parse, A[0], bad, oracles)


def test_a_flipped_pseudogene_flag_is_caught(good):
    genomes, oracles = good
    g = genomes[A[0]]
    flipped = dataclasses.replace(g.genes[5], pseudogene=not g.genes[5].pseudogene)
    bad = {A[0]: dataclasses.replace(g, genes=g.genes[:5] + (flipped,) + g.genes[6:])}
    red(tg.test_pseudogene_and_symbol_flags_match_the_independent_parse, A[0], bad, oracles)


def test_a_perturbed_trait_value_is_caught(good, monkeypatch):
    genomes, oracles = good
    real = tg.derive_phenotype
    monkeypatch.setattr(tg, "derive_phenotype",
                        lambda g, c, *a: dataclasses.replace(real(g, c, *a), motility_um_s=real(g, c, *a).motility_um_s + 1e-3))
    red(tg.test_every_trait_equals_an_independent_reimplementation_of_the_rule, A[0], genomes, oracles)


def test_a_rule_that_fires_when_it_should_not_is_caught(good, monkeypatch):
    genomes, oracles = good
    real = tg.derive_phenotype
    monkeypatch.setattr(tg, "derive_phenotype",
                        lambda g, c, *a: dataclasses.replace(real(g, c, *a), applied_rule_ids=real(g, c, *a).applied_rule_ids[:-1]))
    red(tg.test_every_trait_equals_an_independent_reimplementation_of_the_rule, A[0], genomes, oracles)


def _table(genomes):
    return tg.build_phenotype_table(list(genomes.values()), tg.CHANNELS)


def test_unsorted_genome_ids_are_caught(good):
    genomes, _ = good
    t = _table(genomes)
    bad = dataclasses.replace(t, genome_id=t.genome_id[::-1].copy())
    red(tg.test_phenotype_table_is_a_valid_d1_table, bad, genomes)


def test_storage_plus_biomass_over_one_is_caught(good):
    genomes, _ = good
    t = _table(genomes)
    bad = dataclasses.replace(t, storage_fraction=t.storage_fraction + 0.5)
    red(tg.test_phenotype_table_is_a_valid_d1_table, bad, genomes)


def test_wrong_dtype_nan_and_missing_field_are_caught(good):
    genomes, _ = good
    t = _table(genomes)
    red(tg.test_phenotype_table_is_a_valid_d1_table, dataclasses.replace(t, motility_um_s=t.motility_um_s.astype(np.float32)), genomes)
    nan = t.max_age_s.copy()
    nan[0] = np.nan
    red(tg.test_phenotype_table_is_a_valid_d1_table, dataclasses.replace(t, max_age_s=nan), genomes)
    red(tg.test_phenotype_table_is_a_valid_d1_table, dataclasses.replace(t, antagonism_rate_s=None), genomes)


def test_a_value_beyond_the_schema_bound_is_caught(good):
    genomes, _ = good
    t = _table(genomes)
    bad = dataclasses.replace(t, motility_um_s=t.motility_um_s + 1000.0)
    red(tg.test_every_value_is_inside_the_schema_bounds, bad)


def test_nondeterministic_table_building_is_caught(good, monkeypatch):
    genomes, _ = good
    t = _table(genomes)
    real = tg.build_phenotype_table

    def noisy(gs, ch, *a):
        r = real(gs, ch, *a)
        return dataclasses.replace(r, motility_um_s=r.motility_um_s + np.random.random(r.motility_um_s.shape) * 1e-9)

    monkeypatch.setattr(tg, "build_phenotype_table", noisy)
    red(tg.test_repeated_builds_are_bitwise_identical, t, genomes)


def test_input_order_dependence_is_caught(good, monkeypatch):
    genomes, _ = good
    t = _table(genomes)
    real = tg.build_phenotype_table
    monkeypatch.setattr(tg, "build_phenotype_table",
                        lambda gs, ch, *a: dataclasses.replace(real(gs, ch, *a), motility_um_s=real(gs, ch, *a).motility_um_s * (1 + 1e-9 * len(gs[0].accession))
                                                               + (1e-6 if gs[0] is not list(genomes.values())[0] else 0)))
    red(tg.test_table_rows_agree_with_derive_phenotype_and_ignore_input_order, t, genomes)


def test_a_tampered_manifest_hash_size_and_provenance_are_caught(monkeypatch):
    entry = copy.deepcopy(tg.MANIFEST["genomes"][0])
    bad = copy.deepcopy(entry)
    bad["annotation_file"]["sha256"] = "0" * 64
    red(tg.test_every_manifest_artifact_exists_with_the_recorded_size_and_sha256, bad)
    bad = copy.deepcopy(entry)
    bad["sequence_file"]["size_bytes"] += 1
    red(tg.test_every_manifest_artifact_exists_with_the_recorded_size_and_sha256, bad)
    for field in ("fetched_at_utc", "license_or_terms", "source_url", "tax_id"):
        bad = copy.deepcopy(entry)
        bad.pop(field)
        red(tg.test_manifest_records_the_provenance_the_readme_requires, bad)
    bad = copy.deepcopy(entry)
    bad["source_url"] = bad["source_url"].replace(entry["accession"], "GCF_999999999.9")
    red(tg.test_manifest_records_the_provenance_the_readme_requires, bad)


def test_the_wrong_organism_is_caught_by_the_public_length_anchor(good, monkeypatch):
    genomes, _ = good
    monkeypatch.setitem(tg.KNOWN_LENGTH_BP, A[0], tg.KNOWN_LENGTH_BP[A[0]] + 1)
    red(tg.test_fasta_length_and_record_count_match_independent_count_and_public_facts, A[0], genomes)


def test_a_rule_pointing_at_a_gene_that_does_not_exist_is_caught(good, monkeypatch):
    genomes, _ = good
    bad = copy.deepcopy(tg.TABLE)
    bad["rules"][0]["gene_key"] = bad["rules"][0]["genome_accession"] + "|locus_tag:b9999"
    k = bad["rules"][0]["gene_key"]
    bad["rules"][0]["gene_symbols"] = {k: "nope"}
    monkeypatch.setattr(tg, "TABLE", bad)
    red(tg.test_trait_rules_cite_real_annotated_genes, genomes)
