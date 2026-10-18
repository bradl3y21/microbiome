"""D2 producer API: bundle loader, versioned TSV tables, TSV-driven phenotype table, evidence, feature rows.

The anchor is equivalence: building both real genomes through ``trait_schema.tsv`` / ``gene_trait_map.tsv`` gives
the same ``PhenotypeTable`` bits as the original JSON path, so the new API is the old biology in D2's clothes.
The rest pins each D2 sentence on small synthetic annotations, where the expected answer is obvious by hand.
"""
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from microbiome.genome import (
    GenomeRecord,
    NutrientChannel,
    build_phenotype_table,
    derive_phenotype,
    explain_phenotype,
    gene_feature_rows,
    load_gene_trait_map,
    load_genome,
    load_genome_bundle,
    load_trait_schema,
    normalize_gene_features,
    write_gene_features_tsv,
)
from microbiome.genome import tables as T
from microbiome.genome.models import AnnotatedGene
from microbiome.genome.parser import derive_genome_id

ROOT = Path(__file__).resolve().parent.parent
EC, BS = "GCF_000005845.2", "GCF_000009045.1"
CHANNELS = (NutrientChannel(0, "glucose"), NutrientChannel(1, "lactose"))
FIELDS = ("max_uptake_mmol_s", "storage_fraction", "biomass_fraction", "energy_yield_pj_per_mmol", "motility_um_s",
          "chemotaxis_sensitivity", "random_motility_fraction", "division_rate_s", "division_mass_pg",
          "death_rate_s", "max_age_s", "antagonism_radius_um", "antagonism_rate_s")


@pytest.fixture(scope="module")
def real():
    genomes = [load_genome_bundle(ROOT / "data" / "genomes" / a) for a in (EC, BS)]
    schema = load_trait_schema(ROOT / "data" / "trait_schema.tsv")
    gene_map = load_gene_trait_map(ROOT / "data" / "gene_trait_map.tsv")
    return genomes, schema, gene_map


# ============================================================ equivalence with the JSON path
def test_the_tsv_path_reproduces_the_json_path_bit_for_bit(real):
    genomes, schema, gene_map = real
    new = build_phenotype_table(genomes, schema, gene_map)
    old = build_phenotype_table(genomes, CHANNELS)
    assert np.array_equal(new.genome_id, old.genome_id)
    for name in FIELDS:
        a, b = getattr(new, name), getattr(old, name)
        assert a.dtype == b.dtype == np.float64 and a.shape == b.shape, name
        assert a.tobytes() == b.tobytes(), f"{name} differs between the TSV and JSON paths"


def test_explained_rules_are_exactly_the_rules_the_json_path_applied(real):
    genomes, schema, gene_map = real
    for genome in genomes:
        want = derive_phenotype(genome, CHANNELS).applied_rule_ids
        got = tuple(sorted({e.rule_id for e in explain_phenotype(genome, schema, gene_map)}))
        assert got == tuple(sorted(want)), genome.accession


def test_evidence_exposes_gene_keys_rule_ids_references_and_the_approximation(real):
    genomes, schema, gene_map = real
    bacillus = next(g for g in genomes if g.accession == BS)
    evidence = explain_phenotype(bacillus, schema, gene_map)
    operon = [e for e in evidence if e.rule_id == "bacillus-bacilysin-antagonism-rate"]
    assert len(operon) == 1 and len(operon[0].gene_keys) == 7, "an all-of rule reports its whole gene set"
    assert all(k.startswith(f"{BS}|") for e in evidence for k in e.gene_keys)
    for e in evidence:
        assert e.evidence_refs and any(r.startswith("http") for r in e.evidence_refs)
        assert e.approximation_note and e.unit


def test_the_committed_tsvs_are_exactly_what_the_exporter_produces(tmp_path):
    (tmp_path / "data").mkdir()
    shutil.copy(ROOT / "data" / "gene_trait_table.json", tmp_path / "data" / "gene_trait_table.json")
    schema, gene_map = T.export_tables_from_json(tmp_path)
    assert schema.read_bytes() == (ROOT / "data" / "trait_schema.tsv").read_bytes()
    assert gene_map.read_bytes() == (ROOT / "data" / "gene_trait_map.tsv").read_bytes()


def test_the_legacy_call_shape_still_works_and_a_half_new_one_is_rejected(real):
    genomes, schema, _ = real
    assert len(build_phenotype_table(genomes, CHANNELS).genome_id) == 2
    with pytest.raises(TypeError, match="GeneTraitMap"):
        build_phenotype_table(genomes, schema)


def test_the_table_feeds_the_simulation_gather(real):
    from microbiome.sim.model import phenotype_arrays

    genomes, schema, gene_map = real
    table = build_phenotype_table(genomes, schema, gene_map)
    out = phenotype_arrays(table, table.genome_id, 2)
    assert np.array_equal(out.motility_um_s, table.motility_um_s)


# ============================================================ semantics on synthetic annotations
ACC = "GCF_999999999.1"


def gene(tag, *, ftypes=("gene", "CDS"), pseudo=False, partial=False, **attributes):
    attrs = {"locus_tag": tag, **attributes}
    if partial:
        attrs["partial"] = "true"
    return AnnotatedGene(
        accession=ACC, gene_key=f"{ACC}|locus_tag:{tag}", locus_tag=tag, gene_symbol=attributes.get("gene"),
        product=None, seqid="chr", start=1, end=100, strand="+", feature_type="gene",
        source_feature_types=tuple(ftypes), source_ids=(), parent_ids=(), pseudogene=pseudo, attributes=attrs)


def record(*genes):
    return GenomeRecord(accession=ACC, genome_id=derive_genome_id(ACC), organism="synthetic", tax_id=1,
                        assembly_level="Complete Genome", sequence_path="-", sequence_length_bp=1,
                        sequence_record_count=1, sequence_sha256="0" * 64, annotation_path="-",
                        annotation_sha256="0" * 64, genes=tuple(genes))


def schema_for(real, **overrides):
    """The real schema with some specs replaced, e.g. a wider max to test clamping."""
    base = real[1]
    specs = []
    for spec in base.traits:
        change = overrides.get(spec.trait_id)
        specs.append(dataclasses.replace(spec, **change) if change else spec)
    return dataclasses.replace(base, traits=tuple(specs))


def rule(mapping_id, namespace, value, *, trait="motility_um_s", effect=2.0, priority=100, nutrient=None,
         feature_type="CDS", genome_id=None, extra=()):
    matches = [T.MatchTerm(namespace, value, feature_type)] + [T.MatchTerm(*item) for item in extra]
    return T.GeneTraitRule(mapping_id, genome_id, tuple(matches), trait, nutrient, effect, "um/s", priority,
                           "calibrated_simplification", ("synthetic",), ("https://example.org",), "why", "approx")


def gmap(*rules):
    return T.GeneTraitMap(1, tuple(rules), "0" * 64)


def motility(real, genome, *rules):
    table = build_phenotype_table([genome], real[1], gmap(*rules))
    return float(table.motility_um_s[0])


def test_an_exact_match_adds_its_effect_once_to_the_baseline(real):
    base = real[1].spec("motility_um_s").baseline
    assert motility(real, record(gene("A")), rule("r", "locus_tag", "A")) == base + 2.0


def test_absent_partial_pseudo_and_wrong_feature_type_genes_contribute_zero(real):
    base = real[1].spec("motility_um_s").baseline
    r = rule("r", "locus_tag", "A")
    assert motility(real, record(gene("B")), r) == base
    assert motility(real, record(gene("A", pseudo=True)), r) == base
    assert motility(real, record(gene("A", partial=True)), r) == base
    assert motility(real, record(gene("A", ftypes=("gene",))), r) == base


def test_distinct_paralogs_count_separately(real):
    base = real[1].spec("motility_um_s").baseline
    genome = record(gene("A", ec_number="1.1.1.1"), gene("B", ec_number="1.1.1.1"), gene("C"))
    assert motility(real, genome, rule("r", "EC", "1.1.1.1")) == base + 4.0
    assert len(explain_phenotype(genome, real[1], gmap(rule("r", "EC", "1.1.1.1")))) == 2


def test_every_namespace_matches_exactly_and_nothing_else(real):
    base = real[1].spec("motility_um_s").baseline
    g = gene("A", Dbxref="GeneID:945395,KO:K03406,GO:0006935,EC:3.1.1.61", ec_number="2.7.13.3")
    genome = record(g)
    for namespace, value in (("gene_key", g.gene_key), ("locus_tag", "A"), ("GeneID", "945395"), ("EC", "2.7.13.3"),
                             ("EC", "3.1.1.61"), ("KO", "K03406"), ("GO", "0006935")):
        assert motility(real, genome, rule("r", namespace, value)) == base + 2.0, (namespace, value)
    for namespace, value in (("GeneID", "945"), ("EC", "2.7.13"), ("KO", "k03406"), ("locus_tag", "a")):
        assert motility(real, genome, rule("r", namespace, value)) == base, "no prefix, case or fuzzy matching"


def test_the_lowest_priority_wins_and_a_tie_fails(real):
    base = real[1].spec("motility_um_s").baseline
    genome = record(gene("A"))
    low = rule("low", "locus_tag", "A", effect=1.0, priority=5)
    high = rule("high", "locus_tag", "A", effect=50.0, priority=9)
    assert motility(real, genome, low, high) == base + 1.0
    with pytest.raises(ValueError, match="priority tie"):
        motility(real, genome, rule("a", "locus_tag", "A", priority=5), rule("b", "locus_tag", "A", priority=5))
    # different traits on one gene never compete
    other = rule("other", "locus_tag", "A", trait="chemotaxis_sensitivity", effect=0.1, priority=5)
    assert motility(real, genome, low, other) == base + 1.0


def test_an_all_of_mapping_needs_every_gene(real):
    base = real[1].spec("motility_um_s").baseline
    operon = rule("op", "locus_tag", "A", effect=3.0, extra=[("locus_tag", "B", "CDS"), ("locus_tag", "C", "CDS")])
    assert motility(real, record(gene("A"), gene("B"), gene("C")), operon) == base + 3.0
    assert motility(real, record(gene("A"), gene("B")), operon) == base, "a partial operon gives nothing"
    assert motility(real, record(gene("A"), gene("B"), gene("C", pseudo=True)), operon) == base


def test_a_genome_scoped_mapping_applies_only_to_that_genome(real):
    base = real[1].spec("motility_um_s").baseline
    assert motility(real, record(gene("A")), rule("r", "locus_tag", "A", genome_id=derive_genome_id(ACC))) == base + 2.0
    assert motility(real, record(gene("A")), rule("r", "locus_tag", "A", genome_id=123)) == base


def test_effects_clamp_to_the_schema_bounds_and_cross_trait_violations_raise(real):
    high = real[1].spec("motility_um_s").maximum
    assert motility(real, record(gene("A")), rule("r", "locus_tag", "A", effect=1e6)) == high
    over = [rule(f"s{n}", "locus_tag", "A", trait=t, nutrient=0, effect=0.4)
            for n, t in enumerate(("storage_fraction", "biomass_fraction"))]
    with pytest.raises(ValueError, match="storage_fraction \\+ biomass_fraction"):
        build_phenotype_table([record(gene("A"))], real[1], gmap(*over))


def test_a_mapping_to_a_nutrient_the_schema_lacks_is_rejected(real):
    bad = rule("r", "locus_tag", "A", trait="max_uptake_mmol_s", nutrient=7)
    with pytest.raises(ValueError, match="absent from the schema"):
        build_phenotype_table([record(gene("A"))], real[1], gmap(bad))


def test_genome_id_collisions_are_rejected(real):
    a = record(gene("A"))
    with pytest.raises(ValueError, match="collision"):
        build_phenotype_table([a, a], real[1], gmap())


# ============================================================ loaders reject malformed tables
def edited(tmp_path, source, transform, name):
    text = (ROOT / "data" / source).read_text(encoding="utf-8")
    path = tmp_path / name
    path.write_text(transform(text), encoding="utf-8", newline="\n")
    return path


@pytest.mark.parametrize("transform, match", [
    (lambda t: t.replace("#schema_version=1\n", ""), "schema_version=1"),
    (lambda t: t.replace("float64", "float32", 1), "float64"),
    (lambda t: t.replace("sum_clip", "mean", 1), "sum_clip"),
    (lambda t: t.replace("max_uptake_mmol_s\tnutrient\t0", "max_uptake_mmol_s\tscalar\t0", 1), "scope"),
    (lambda t: t.replace("trait_id\tscope", "trait\tscope"), "exactly the columns"),
    (lambda t: t.replace("\tmotility_um_s\tscalar", "\tmotility_xx\tscalar") if False else t.replace("motility_um_s\tscalar\t\tfloat64\tum/s\t0.0\t0.0\t50.0", "motility_um_s\tscalar\t\tfloat64\tum/s\t99.0\t0.0\t50.0"), "within ordered"),
])
def test_the_schema_loader_rejects_bad_tables(tmp_path, transform, match):
    path = edited(tmp_path, "trait_schema.tsv", transform, "bad.tsv")
    with pytest.raises(ValueError, match=match):
        load_trait_schema(path)


@pytest.mark.parametrize("transform, match", [
    (lambda t: t.replace("\tgene_key\t", "\tproduct\t", 1), "namespace"),
    (lambda t: t.replace("\tCDS\t", "\texon\t", 1), "feature_type"),
    (lambda t: t.replace("\t100\t", "\thigh\t"), "priority"),
    (lambda t: t.replace("\tmotility_um_s\t", "\tspeed\t", 1), "unknown trait"),
    (lambda t: t.replace("#schema_version=1\n", ""), "schema_version=1"),
])
def test_the_map_loader_rejects_bad_tables(tmp_path, transform, match):
    path = edited(tmp_path, "gene_trait_map.tsv", transform, "bad.tsv")
    with pytest.raises(ValueError, match=match):
        load_gene_trait_map(path)


def test_rows_of_one_mapping_may_differ_only_in_their_match_columns(tmp_path):
    def transform(text):
        lines = text.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("bacillus-bacilysin-antagonism-rate\t"))
        cells = lines[index].split("\t")
        cells[7] = "0.5"                                  # effect differs inside one all-of mapping
        lines[index] = "\t".join(cells)
        return "\n".join(lines) + "\n"
    with pytest.raises(ValueError, match="differ only in their match columns"):
        load_gene_trait_map(edited(tmp_path, "gene_trait_map.tsv", transform, "bad.tsv"))


def test_the_loaded_tables_carry_their_own_hashes(real):
    import hashlib

    _, schema, gene_map = real
    assert schema.sha256 == hashlib.sha256((ROOT / "data" / "trait_schema.tsv").read_bytes()).hexdigest()
    assert gene_map.sha256 == hashlib.sha256((ROOT / "data" / "gene_trait_map.tsv").read_bytes()).hexdigest()
    assert schema.nutrient_ids == (0, 1) and len(gene_map.rules) == len(json.loads((ROOT / "data" / "gene_trait_table.json").read_text(encoding="utf-8"))["rules"])


# ============================================================ gene_features.tsv rows
def test_gene_feature_rows_follow_the_d2_column_list_and_order(real, tmp_path):
    genomes, _, _ = real
    genome = genomes[0]
    rows = gene_feature_rows(genome)
    assert tuple(rows[0]) == T.FEATURE_COLUMNS
    assert len(rows) >= len(genome.genes)
    keys = [(r["seqid"], int(r["start_1based"]), int(r["end_1based"]), r["strand"], r["feature_type"], r["source_id"]) for r in rows]
    assert keys == sorted(keys)
    assert {r["gene_key"] for r in rows} == {g.gene_key for g in genome.genes}
    assert {r["source_annotation_sha256"] for r in rows} == {genome.annotation_sha256}
    assert {r["genome_id"] for r in rows} == {str(genome.genome_id)}
    assert {r["feature_type"] for r in rows} <= {"gene", "CDS"}
    che_a = [r for r in rows if r["gene_key"] == f"{EC}|locus_tag:b1888"]
    assert {r["feature_type"] for r in che_a} == {"gene", "CDS"} and che_a[0]["gene_symbol"] == "cheA"
    out = write_gene_features_tsv(genome, tmp_path / "gene_features.tsv")
    header = out.read_text(encoding="utf-8").splitlines()[3].split("\t")
    assert tuple(header) == T.FEATURE_COLUMNS


def test_normalize_gene_features_returns_the_genomes_ordered_genes(real):
    genome = real[0][1]
    genes = normalize_gene_features(genome)
    order = [(g.seqid, g.start, g.end, g.strand, g.feature_type, g.gene_key) for g in genes]
    assert order == sorted(order) and len(genes) == len(genome.genes)


# ============================================================ bundle loader
@pytest.fixture()
def bundle_copy(tmp_path):
    (tmp_path / "data" / "genomes").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "manifest.json", tmp_path / "data" / "manifest.json")
    target = tmp_path / "data" / "genomes" / EC
    shutil.copytree(ROOT / "data" / "genomes" / EC, target)
    return target


def test_a_clean_bundle_loads_and_matches_load_genome(bundle_copy):
    record_ = load_genome_bundle(bundle_copy)
    reference = load_genome(EC)
    assert record_.genome_id == reference.genome_id == derive_genome_id(EC)
    assert record_.genes == reference.genes


def test_a_bundle_in_the_wrong_place_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_genome_bundle(tmp_path / "nope")
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(ValueError, match="must live at"):
        load_genome_bundle(tmp_path / "elsewhere")


def flip(path):
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("victim, match", [
    ("ncbi_dataset/data/GCF_000005845.2/genomic.gff", "SHA-256 mismatch"),
    ("ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna", "SHA-256 mismatch"),
    ("ncbi_datasets.zip", "SHA-256 mismatch"),
    ("ncbi_dataset/data/assembly_data_report.jsonl", "mismatch"),
    ("md5sum.txt", "mismatch"),
])
def test_tampering_with_any_retained_artifact_is_detected(bundle_copy, victim, match):
    flip(bundle_copy / victim)
    with pytest.raises(ValueError, match=match):
        load_genome_bundle(bundle_copy)


def test_a_missing_retained_artifact_is_detected(bundle_copy):
    (bundle_copy / "README.md").unlink()
    with pytest.raises(FileNotFoundError):
        load_genome_bundle(bundle_copy)


def test_the_manifest_must_pin_every_required_role(bundle_copy):
    manifest = bundle_copy.parent.parent / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entry = next(e for e in data["genomes"] if e["accession"] == EC)
    entry["artifacts"] = [a for a in entry["artifacts"] if a["role"] != "original_ncbi_datasets_zip"]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="missing retained artifacts"):
        load_genome_bundle(bundle_copy)
