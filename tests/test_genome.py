"""Independent verification of the staged genome module (backlog #4).

SCOPE: written against the module AS STAGED (load_genome, derive_phenotype, build_phenotype_table,
data/gene_trait_table.json). D2 is NOT frozen and D2's text names a different API/layout
(load_genome_bundle, TSV tables), so API-shaped tests may need adapting once the operator decides.
The checks that matter most use independent oracles that do not call the module:
  * file size/sha256 and NCBI's own md5sum.txt, computed here
  * a separate FASTA length/record count and GFF3 locus_tag/pseudogene parse, computed here
  * a second implementation of the expression rule (baseline + sum of matching effects, clamped)
Tests marked xfail(strict=True) record a measured deviation; they go red when it is fixed.
"""
import copy
import dataclasses
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest

from conftest import ROOT, digest_arrays
from microbiome.genome.models import GenomeRecord, NutrientChannel
from microbiome.genome.parser import derive_genome_id, load_genome, parse_gff3
from microbiome.genome.phenotype import build_phenotype_table, derive_phenotype, phenotype_demo
from microbiome.sim.contracts import PhenotypeTable

DATA = ROOT / "data"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
TABLE = json.loads((DATA / "gene_trait_table.json").read_text(encoding="utf-8"))
ACCESSIONS = [g["accession"] for g in MANIFEST["genomes"]]
CHANNELS = (NutrientChannel(0, "glucose"), NutrientChannel(1, "lactose"))
NUTRIENT_FIELDS = ("max_uptake_mmol_s", "storage_fraction", "biomass_fraction", "energy_yield_pj_per_mmol")
SCALAR_FIELDS = ("motility_um_s", "chemotaxis_sensitivity", "random_motility_fraction", "division_rate_s",
                 "division_mass_pg", "death_rate_s", "max_age_s", "antagonism_radius_um", "antagonism_rate_s")
# Public facts about the organisms, an anchor that is independent of every file we ship.
KNOWN_LENGTH_BP = {"GCF_000005845.2": 4_641_652, "GCF_000009045.1": 4_215_606,
                   "GCF_000006765.1": 6_264_404,   # P. aeruginosa PAO1, NCBI Datasets assembly report ASM676v1
                   "GCF_000010465.1": 2_878_897,   # S. aureus Newman, ASM1046v1
                   "GCF_000012685.1": 9_139_763}   # M. xanthus DK 1622, ASM1268v1


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@pytest.fixture(scope="module")
def genomes():
    return {a: load_genome(a) for a in ACCESSIONS}


# ====================================================================== provenance
@pytest.mark.parametrize("entry", MANIFEST["genomes"], ids=lambda e: e["accession"])
def test_every_manifest_artifact_exists_with_the_recorded_size_and_sha256(entry):
    for key in ("sequence_file", "annotation_file", "source_package"):
        rec = entry[key]
        p = ROOT / rec["path"]
        assert p.is_file(), f"{key} missing: {rec['path']}"
        assert p.stat().st_size == rec["size_bytes"], f"{key} size differs from manifest"
        assert sha256(p) == rec["sha256"].lower(), f"{key} sha256 differs from manifest"
    assert entry["sha256"] == entry["sequence_file"]["sha256"]  # top-level fields describe the sequence file


@pytest.mark.parametrize("entry", MANIFEST["genomes"], ids=lambda e: e["accession"])
def test_manifest_records_the_provenance_the_readme_requires(entry):
    for field in ("accession", "organism", "tax_id", "source_url", "fetched_at_utc", "fetch_date_utc",
                  "sha256", "size_bytes", "license_or_terms", "assembly_level"):
        assert entry.get(field) not in (None, "", 0), f"manifest entry lacks {field}"
    acc = entry["accession"]
    assert acc.startswith("GCF_") and "." in acc and acc.split(".")[1].isdigit(), "accession.version required"
    assert entry["fetch_date_utc"] == entry["fetched_at_utc"][:10]
    assert entry["source_url"].startswith("https://")
    assert acc in entry["source_url"], "source_url must be for this accession"
    for top in ("terms_url", "terms_notice", "retrieval_tool", "checksum_algorithm"):
        assert MANIFEST.get(top), f"manifest lacks {top}"


@pytest.mark.parametrize("accession", ACCESSIONS)
def test_ncbi_own_md5sum_file_agrees_with_the_files_on_disk(accession):
    """Cross-check that does not trust our manifest: NCBI shipped md5sum.txt inside the package."""
    base = DATA / "genomes" / accession
    lines = [ln.split(None, 1) for ln in (base / "md5sum.txt").read_text().splitlines() if ln.strip()]
    assert lines
    checked = 0
    for md5, rel in lines:
        f = base / rel.strip()
        assert f.is_file(), f"md5sum.txt lists {rel.strip()} but it is absent"
        assert hashlib.md5(f.read_bytes()).hexdigest() == md5, f"{rel.strip()} md5 differs from NCBI's"
        checked += 1
    assert checked >= 5


def test_downloaded_zip_contains_the_extracted_files_byte_for_byte():
    import zipfile

    for entry in MANIFEST["genomes"]:
        with zipfile.ZipFile(ROOT / entry["source_package"]["path"]) as z:
            for key in ("sequence_file", "annotation_file"):
                path = ROOT / entry[key]["path"]
                member = "ncbi_dataset" + path.as_posix().split("/ncbi_dataset", 1)[1]
                assert hashlib.sha256(z.read(member)).hexdigest() == sha256(path), f"{member} differs from the zip"


def test_manifest_has_no_duplicate_accessions_and_genome_ids_do_not_collide():
    assert len(set(ACCESSIONS)) == len(ACCESSIONS) >= 2
    ids = [derive_genome_id(a) for a in ACCESSIONS]
    assert len(set(ids)) == len(ids)
    assert all(0 <= i < 2**64 for i in ids)


# ===================================================================== FASTA oracle
def oracle_fasta(path):
    n, bp = 0, 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                n += 1
            else:
                bp += len(line.strip())
    return n, bp


@pytest.mark.parametrize("accession", ACCESSIONS)
def test_fasta_length_and_record_count_match_independent_count_and_public_facts(accession, genomes):
    entry = next(e for e in MANIFEST["genomes"] if e["accession"] == accession)
    n, bp = oracle_fasta(ROOT / entry["sequence_file"]["path"])
    g = genomes[accession]
    assert (g.sequence_record_count, g.sequence_length_bp) == (n, bp)
    assert (entry["sequence_count"], entry["sequence_length_bp"]) == (n, bp)
    assert bp == KNOWN_LENGTH_BP[accession], "not the genome this accession names"


# ===================================================================== GFF3 oracle
def oracle_gff(path):
    """Independent minimal parse: rows by locus_tag for gene/CDS features."""
    rows = defaultdict(list)
    n_features = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if c[2] not in ("gene", "CDS"):
                continue
            attrs = {}
            for f in c[8].split(";"):
                k, _, v = f.partition("=")
                if _:
                    attrs[unquote(k)] = unquote(v)
            n_features += 1
            rows[attrs.get("locus_tag")].append((c[2], int(c[3]), int(c[4]), c[6], attrs))
    return rows, n_features


@pytest.fixture(scope="module")
def oracles():
    out = {}
    for e in MANIFEST["genomes"]:
        out[e["accession"]] = oracle_gff(ROOT / e["annotation_file"]["path"])
    return out


@pytest.mark.parametrize("accession", ACCESSIONS)
def test_parsed_genes_match_an_independent_gff3_parse(accession, genomes, oracles):
    rows, n_features = oracles[accession]
    g = genomes[accession]
    assert None not in rows, "this assembly has gene/CDS rows without locus_tag; extend the oracle"
    assert {x.locus_tag for x in g.genes} == set(rows), "locus_tag sets differ"
    assert len(g.genes) == len(rows)
    assert len({x.gene_key for x in g.genes}) == len(g.genes)
    for x in g.genes:
        r = rows[x.locus_tag]
        assert x.gene_key == f"{accession}|locus_tag:{x.locus_tag}"
        assert x.start == min(s for _, s, _, _, _ in r) and x.end == max(e for _, _, e, _, _ in r)
        assert x.strand in {s for _, _, _, s, _ in r}
        assert set(x.source_feature_types) == {t for t, *_ in r}
        assert 1 <= x.start <= x.end


@pytest.mark.parametrize("accession", ACCESSIONS)
def test_pseudogene_and_symbol_flags_match_the_independent_parse(accession, genomes, oracles):
    rows, _ = oracles[accession]
    g = genomes[accession]
    for x in g.genes:
        r = rows[x.locus_tag]
        want_pseudo = any(a.get("pseudo", "").lower() == "true" or a.get("gene_biotype") == "pseudogene"
                          for *_, a in r)
        assert x.pseudogene == want_pseudo, x.locus_tag
        symbols = {a["gene"] for *_, a in r if a.get("gene")}
        if symbols:
            assert x.gene_symbol in symbols, x.locus_tag
    assert sum(x.pseudogene for x in g.genes) == sum(
        any(a.get("pseudo", "").lower() == "true" or a.get("gene_biotype") == "pseudogene" for *_, a in r)
        for r in rows.values())


def test_gene_order_is_canonical(genomes):
    for g in genomes.values():
        keys = [(x.seqid, x.start, x.end, x.strand, x.feature_type, x.gene_key) for x in g.genes]
        assert keys == sorted(keys)


# ======================================================== genome_id (D2 text deviation)
def test_genome_id_is_a_stable_uint64_independent_of_hash_randomisation(fresh_python):
    code = ("from microbiome.genome.parser import derive_genome_id as d;"
            "print(d('GCF_000005845.2'), d('GCF_000009045.1'))")
    a, b = fresh_python(code, hashseed="0"), fresh_python(code, hashseed="12345")
    assert a == b == " ".join(str(derive_genome_id(x)) for x in ("GCF_000005845.2", "GCF_000009045.1"))


def test_genome_id_follows_the_d2_formula_exactly():
    for acc in ACCESSIONS:
        want = int.from_bytes(hashlib.sha256(acc.encode("ascii")).digest()[:8], "big")
        assert derive_genome_id(acc) == want


@pytest.mark.parametrize("bad", ["", " GCF_1.1", "GCF_000005845", "GCF_000005845.2 "])
def test_genome_id_rejects_non_canonical_accessions(bad):
    with pytest.raises(ValueError):
        derive_genome_id(bad)


# ================================================== phenotype table (D1 shape + values)
@pytest.fixture(scope="module")
def table(genomes):
    return build_phenotype_table(list(genomes.values()), CHANNELS)


def test_phenotype_table_is_a_valid_d1_table(table, genomes):
    assert isinstance(table, PhenotypeTable)
    G, K = len(genomes), len(CHANNELS)
    assert table.genome_id.dtype == np.uint64 and table.genome_id.shape == (G,)
    assert (np.diff(table.genome_id.astype(object)) > 0).all(), "genome_id must be strictly increasing"
    assert sorted(int(i) for i in table.genome_id) == sorted(g.genome_id for g in genomes.values())
    for f in NUTRIENT_FIELDS:
        a = getattr(table, f)
        assert a is not None and a.dtype == np.float64 and a.shape == (G, K), f
        assert a.flags.c_contiguous
    for f in SCALAR_FIELDS:
        a = getattr(table, f)
        assert a is not None and a.dtype == np.float64 and a.shape == (G,), f
    for f in NUTRIENT_FIELDS + SCALAR_FIELDS:
        a = getattr(table, f)
        assert np.isfinite(a).all() and (a >= 0).all(), f
    assert (table.storage_fraction + table.biomass_fraction <= 1.0).all()
    assert (table.division_mass_pg > 0).all() and (table.max_age_s > 0).all()


def test_every_value_is_inside_the_schema_bounds(table):
    for f in NUTRIENT_FIELDS + SCALAR_FIELDS:
        lo, hi = TABLE["bounds"][f]
        a = getattr(table, f)
        assert (a >= lo).all() and (a <= hi).all(), f"{f} outside [{lo},{hi}]"


def test_table_rows_agree_with_derive_phenotype_and_ignore_input_order(table, genomes):
    shuffled = build_phenotype_table(list(reversed(list(genomes.values()))), CHANNELS)
    for f in NUTRIENT_FIELDS + SCALAR_FIELDS + ("genome_id",):
        assert getattr(shuffled, f).tobytes() == getattr(table, f).tobytes(), f"input order leaked into {f}"
    for row, gid in enumerate(table.genome_id):
        g = next(x for x in genomes.values() if x.genome_id == int(gid))
        p = derive_phenotype(g, CHANNELS)
        for f in NUTRIENT_FIELDS:
            assert tuple(getattr(table, f)[row]) == getattr(p, f)
        for f in SCALAR_FIELDS:
            assert getattr(table, f)[row] == getattr(p, f)


def test_repeated_builds_are_bitwise_identical(table, genomes):
    again = build_phenotype_table([load_genome(a) for a in ACCESSIONS], CHANNELS)
    for f in NUTRIENT_FIELDS + SCALAR_FIELDS + ("genome_id",):
        assert getattr(again, f).tobytes() == getattr(table, f).tobytes(), f


def _table_digest(t):
    return digest_arrays({f: getattr(t, f) for f in NUTRIENT_FIELDS + SCALAR_FIELDS + ("genome_id",)})


def test_fresh_processes_with_different_hashseeds_build_the_same_table(table, fresh_python):
    code = ("from conftest import digest_arrays;"
            "from microbiome.genome.parser import load_genome;"
            "from microbiome.genome.models import NutrientChannel as N;"
            "from microbiome.genome.phenotype import build_phenotype_table as b;"
            f"A={ACCESSIONS!r};"
            "t=b([load_genome(a) for a in A],(N(0,'glucose'),N(1,'lactose')));"
            f"F={NUTRIENT_FIELDS + SCALAR_FIELDS + ('genome_id',)!r};"
            "print(digest_arrays({f:getattr(t,f) for f in F}))")
    want = _table_digest(table)
    assert fresh_python(code, hashseed="0") == want
    assert fresh_python(code, hashseed="12345") == want


# ===================================== second implementation of the expression rule
def expected_phenotype(accession, oracle_rows, channel_names):
    """Re-derive every trait from the JSON table + the independent GFF3 parse (never the module)."""
    rows = oracle_rows
    base = {f: [TABLE["baselines"][f][n] for n in channel_names] for f in NUTRIENT_FIELDS}
    base.update({f: TABLE["baselines"][f] for f in SCALAR_FIELDS})
    applied = []
    for rule in sorted(TABLE["rules"], key=lambda r: r["rule_id"]):
        if rule["genome_accession"] != accession:
            continue
        keys = rule.get("requires_gene_keys", [rule["gene_key"]])
        ok = True
        for k in keys:
            tag = k.split("locus_tag:", 1)[1]
            r = rows.get(tag)
            if not r:
                ok = False
                break
            pseudo = any(a.get("pseudo", "").lower() == "true" or a.get("gene_biotype") == "pseudogene"
                         for *_, a in r)
            symbols = {a["gene"] for *_, a in r if a.get("gene")} | {a["Name"] for *_, a in r if a.get("Name")}
            if (pseudo or rule["source_feature_type"] not in {t for t, *_ in r}
                    or rule["gene_symbols"][k] not in symbols):
                ok = False
                break
        if not ok:
            continue
        applied.append(rule["rule_id"])
        if rule["trait"] in NUTRIENT_FIELDS:
            base[rule["trait"]][channel_names.index(rule["nutrient_name"])] += rule["value"]
        else:
            base[rule["trait"]] += rule["value"]
    out = {}
    for f in NUTRIENT_FIELDS:
        lo, hi = TABLE["bounds"][f]
        out[f] = [min(hi, max(lo, v)) for v in base[f]]
    for f in SCALAR_FIELDS:
        lo, hi = TABLE["bounds"][f]
        out[f] = min(hi, max(lo, base[f]))
    return out, applied


@pytest.mark.parametrize("accession", ACCESSIONS)
def test_every_trait_equals_an_independent_reimplementation_of_the_rule(accession, genomes, oracles):
    want, want_rules = expected_phenotype(accession, oracles[accession][0], [c.name for c in CHANNELS])
    p = derive_phenotype(genomes[accession], CHANNELS)
    assert list(p.applied_rule_ids) == want_rules
    for f in NUTRIENT_FIELDS:
        assert np.allclose(getattr(p, f), want[f], rtol=1e-12, atol=0), f
    for f in SCALAR_FIELDS:
        assert np.isclose(getattr(p, f), want[f], rtol=1e-12, atol=0), f
    assert want_rules, "no rule applied: nothing was verified"


def test_documented_demo_values_are_reproduced(genomes):
    by_acc = {p["accession"]: p for p in phenotype_demo()["phenotypes"]}
    e, b = by_acc["GCF_000005845.2"], by_acc["GCF_000009045.1"]
    # Exact uptake numbers are pinned by the independent-reimplementation test above (the table was
    # rescaled on 2026-10-17 09:18; the old 2.2e-4 snapshot was retired). Here: the documented, qualitative claims.
    assert e["antagonism_radius_um"] == 0.0 and e["antagonism_rate_s"] == 0.0
    assert b["antagonism_radius_um"] == 12.0 and b["antagonism_rate_s"] == 0.001
    assert e["max_uptake_mmol_s"][1] > 0.0 and b["max_uptake_mmol_s"][1] == 0.0  # lacY: lactose uptake in E. coli only
    assert e["max_uptake_mmol_s"][0] == b["max_uptake_mmol_s"][0] > 0.0          # ptsG: glucose uptake in both


def test_different_genomes_differ_for_reasons_you_can_point_at(table, genomes):
    """README requirement 1: each differing trait must trace to rules that fire in one genome only."""
    ps = {a: derive_phenotype(g, CHANNELS) for a, g in genomes.items()}
    e, b = ps["GCF_000005845.2"], ps["GCF_000009045.1"]
    rule = {r["rule_id"]: r for r in TABLE["rules"]}
    differing = [f for f in SCALAR_FIELDS if getattr(e, f) != getattr(b, f)]
    differing += [f for f in NUTRIENT_FIELDS if getattr(e, f) != getattr(b, f)]
    assert differing, "two genomes produced identical phenotypes"
    for f in differing:
        owners = {rule[r]["genome_accession"] for r in e.applied_rule_ids + b.applied_rule_ids
                  if rule[r]["trait"] == f}
        assert owners, f"{f} differs but no mapped rule explains it"
    assert b.antagonism_rate_s > 0 == e.antagonism_rate_s  # only B. subtilis carries the bacilysin cluster rule
    assert all(rule[r]["genome_accession"] == "GCF_000009045.1" for r in b.applied_rule_ids if "bacilysin" in r)


def test_trait_rules_cite_real_annotated_genes(genomes):
    """Every rule's gene_key must exist in its genome with the stated symbol; otherwise it can never fire."""
    for r in TABLE["rules"]:
        g = genomes[r["genome_accession"]]
        index = {x.gene_key: x for x in g.genes}
        for k in r.get("requires_gene_keys", [r["gene_key"]]):
            assert k in index, f"{r['rule_id']}: {k} is not an annotated gene of {g.accession}"
            assert index[k].gene_symbol == r["gene_symbols"][k], f"{r['rule_id']}: symbol mismatch for {k}"
            assert not index[k].pseudogene, f"{r['rule_id']}: {k} is a pseudogene and can never fire"


def test_uptake_scale_is_consistent_with_the_stated_division_rate(table):
    """FIXED 2026-10-17 09:18 (was a strict xfail: uptake 2.2e-4 mmol/s implied a ~2e-8 s doubling against a
    ~4000 s division_rate_s, 2e11x apart). The table now gives 8.88e-16 mmol/s, uptake-limited growth equal to the
    stated division rate (ratio 1.0000). Kept as a regression test: compares the table with ITSELF, no literature."""
    molar_mass = 180.16  # g/mol, glucose; the sim derives mass_pg = biomass_mmol*molar_mass*1e9
    for row in range(len(table.genome_id)):
        if table.max_uptake_mmol_s[row, 0] == 0.0:
            continue          # M. xanthus has no sugar uptake: it grows on prey, not glucose (Bretscher and Kaiser 1978)
        gain_pg_s = (table.max_uptake_mmol_s[row, 0] * table.biomass_fraction[row, 0] * molar_mass * 1e9)
        doubling_s = table.division_mass_pg[row] / gain_pg_s
        implied_rate = 1.0 / doubling_s
        ratio = implied_rate / table.division_rate_s[row]
        assert 1e-3 <= ratio <= 1e3, f"uptake-limited growth is {ratio:.1e}x the stated division_rate_s"


# ================================================ synthetic projects: malformed input
ACC = "GCF_900000001.1"
SEQ = ">NC_TEST.1 Test organism, complete genome\n" + ("ACGT" * 250 + "\n")  # 1000 bp
GOOD_GFF = (
    "##gff-version 3\n"
    "NC_TEST.1\tRefSeq\tgene\t1\t300\t.\t+\t.\tID=gene-T_1;Name=cheA;gene=cheA;locus_tag=T_1\n"
    "NC_TEST.1\tRefSeq\tCDS\t1\t300\t.\t+\t0\tID=cds-1;Parent=gene-T_1;gene=cheA;locus_tag=T_1;product=kinase\n"
    "NC_TEST.1\tRefSeq\tgene\t400\t600\t.\t-\t.\tID=gene-T_2;Name=fliC;gene=fliC;locus_tag=T_2\n"
    "NC_TEST.1\tRefSeq\tCDS\t400\t600\t.\t-\t0\tID=cds-2;Parent=gene-T_2;gene=fliC;locus_tag=T_2\n"
)


def rule(rid, key_tags, trait, value, symbol_for, nutrient=None, ftype="CDS"):
    keys = [f"{ACC}|locus_tag:{t}" for t in key_tags]
    r = {"rule_id": rid, "genome_accession": ACC, "gene_key": keys[0],
         "gene_symbols": {k: symbol_for[t] for k, t in zip(keys, key_tags)}, "source_feature_type": ftype,
         "trait": trait, "operation": "add", "value": value, "unit": "u", "approximation_status": "test",
         "justification": "synthetic", "evidence": [{"url": "https://example.org", "feature": "synthetic"}]}
    if len(keys) > 1:
        r["requires_gene_keys"] = keys
    if nutrient:
        r["nutrient_name"] = nutrient
    return r


SYMBOLS = {"T_1": "cheA", "T_2": "fliC"}


def make_project(tmp_path, fasta=SEQ, gff=GOOD_GFF, rules=(), level="Complete Genome", edit_manifest=None):
    root = tmp_path / "proj"
    gdir = root / "data" / "genomes" / ACC
    gdir.mkdir(parents=True)
    (gdir / "seq.fna").write_text(fasta, encoding="ascii", newline="")
    (gdir / "ann.gff").write_text(gff, encoding="utf-8", newline="")
    seq_len = sum(len(l.strip()) for l in fasta.splitlines() if l and not l.startswith(">"))

    def rec(name, role):
        p = gdir / name
        return {"path": f"data/genomes/{ACC}/{name}", "role": role, "size_bytes": p.stat().st_size, "sha256": sha256(p)}

    entry = {"accession": ACC, "organism": "Testus synthetica", "tax_id": 1, "assembly_level": level,
             "sequence_count": sum(l.startswith(">") for l in fasta.splitlines()), "sequence_length_bp": seq_len,
             "sequence_file": rec("seq.fna", "genome_sequence_fasta"),
             "annotation_file": rec("ann.gff", "genome_annotation_gff3")}
    manifest = {"schema_version": 1, "genomes": [entry]}
    if edit_manifest:
        edit_manifest(manifest, root)
    (root / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    t = copy.deepcopy(TABLE)
    t["rules"] = list(rules)
    t["demo_accessions"] = [ACC]
    (root / "data" / "gene_trait_table.json").write_text(json.dumps(t), encoding="utf-8")
    return root


def test_a_minimal_valid_project_loads_and_gets_baselines_only(tmp_path):
    root = make_project(tmp_path)
    g = load_genome(ACC, root)
    assert [x.locus_tag for x in g.genes] == ["T_1", "T_2"] and g.sequence_length_bp == 1000
    p = derive_phenotype(g, CHANNELS, root)
    assert p.applied_rule_ids == ()
    for f in NUTRIENT_FIELDS:
        assert getattr(p, f) == tuple(TABLE["baselines"][f][c.name] for c in CHANNELS)
    for f in SCALAR_FIELDS:
        assert getattr(p, f) == TABLE["baselines"][f]


@pytest.mark.parametrize("name,edit", [
    ("sha256", lambda m, r: m["genomes"][0]["annotation_file"].__setitem__("sha256", "0" * 64)),
    ("size", lambda m, r: m["genomes"][0]["sequence_file"].__setitem__("size_bytes", 1)),
    ("missing file", lambda m, r: (r / "data/genomes" / ACC / "ann.gff").unlink()),
    ("absolute path", lambda m, r: m["genomes"][0]["sequence_file"].__setitem__("path", str(r / "data/x"))),
    ("path escapes root", lambda m, r: m["genomes"][0]["sequence_file"].__setitem__("path", "../outside.fna")),
    ("seq count", lambda m, r: m["genomes"][0].__setitem__("sequence_count", 2)),
    ("seq length", lambda m, r: m["genomes"][0].__setitem__("sequence_length_bp", 999)),
])
def test_tampered_or_missing_artifacts_are_refused(tmp_path, name, edit):
    root = make_project(tmp_path, edit_manifest=edit)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_genome(ACC, root)


def test_incomplete_assembly_unknown_and_duplicate_accessions_are_refused(tmp_path):
    with pytest.raises(ValueError, match="complete genome"):
        load_genome(ACC, make_project(tmp_path / "a", level="Scaffold"))
    root = make_project(tmp_path / "b")
    with pytest.raises(ValueError, match="exactly one manifest entry"):
        load_genome("GCF_900000002.1", root)

    def dup(m, r):
        m["genomes"].append(copy.deepcopy(m["genomes"][0]))

    with pytest.raises(ValueError, match="exactly one manifest entry"):
        load_genome(ACC, make_project(tmp_path / "c", edit_manifest=dup))


@pytest.mark.parametrize("name,fasta", [
    ("non-nucleotide symbol", ">x\nACGTXACGT\n"),
    ("sequence before header", "ACGT\n>x\nACGT\n"),
    ("empty record", ">x\n>y\nACGT\n"),
    ("empty header", ">\nACGT\n"),
    ("no records", "\n\n"),
])
def test_malformed_fasta_is_refused(tmp_path, name, fasta):
    def keep_consistent(m, r):
        m["genomes"][0]["sequence_count"] = max(1, sum(l.startswith(">") for l in fasta.splitlines()))
        m["genomes"][0]["sequence_length_bp"] = sum(len(l) for l in fasta.splitlines() if not l.startswith(">"))

    root = make_project(tmp_path, fasta=fasta, edit_manifest=keep_consistent)
    with pytest.raises(ValueError):
        load_genome(ACC, root)


@pytest.mark.parametrize("name,gff", [
    ("no gff-version header", GOOD_GFF.replace("##gff-version 3\n", "")),
    ("wrong gff version", GOOD_GFF.replace("##gff-version 3", "##gff-version 2")),
    ("8 columns", GOOD_GFF + "NC_TEST.1\tRefSeq\tgene\t1\t2\t.\t+\tID=x\n"),
    ("start after end", GOOD_GFF + "NC_TEST.1\tRefSeq\tgene\t50\t10\t.\t+\t.\tID=g;locus_tag=T_9\n"),
    ("start below 1", GOOD_GFF + "NC_TEST.1\tRefSeq\tgene\t0\t10\t.\t+\t.\tID=g;locus_tag=T_9\n"),
    ("bad strand", GOOD_GFF + "NC_TEST.1\tRefSeq\tgene\t1\t10\t.\tX\t.\tID=g;locus_tag=T_9\n"),
    ("non-integer coordinate", GOOD_GFF + "NC_TEST.1\tRefSeq\tgene\ta\t10\t.\t+\t.\tID=g;locus_tag=T_9\n"),
    ("no gene or CDS at all", "##gff-version 3\nNC_TEST.1\tRefSeq\tregion\t1\t1000\t.\t+\t.\tID=r\n"),
    ("empty file", ""),
    ("seqid not in the FASTA", GOOD_GFF.replace("NC_TEST.1\tRefSeq\tgene\t1\t300", "NC_OTHER.9\tRefSeq\tgene\t1\t300", 1)),
])
def test_malformed_or_inconsistent_annotation_is_refused(tmp_path, name, gff):
    root = make_project(tmp_path, gff=gff)
    with pytest.raises((ValueError,)):
        load_genome(ACC, root)


def test_a_gene_without_locus_tag_gets_a_coordinate_key_and_its_cds_joins_it(tmp_path):
    gff = ("##gff-version 3\n"
           "NC_TEST.1\tRefSeq\tgene\t10\t90\t.\t+\t.\tID=gene-a;gene=abcD\n"
           "NC_TEST.1\tRefSeq\tCDS\t10\t90\t.\t+\t0\tID=cds-a;Parent=gene-a;gene=abcD;product=thing\n")
    g = load_genome(ACC, make_project(tmp_path, gff=gff))
    assert len(g.genes) == 1
    x = g.genes[0]
    assert x.locus_tag is None and x.gene_key == f"{ACC}|NC_TEST.1:10-90:+:gene"
    assert set(x.source_feature_types) == {"gene", "CDS"} and x.product == "thing"


def test_percent_encoded_and_empty_attributes_are_decoded_not_crashed_on(tmp_path):
    gff = GOOD_GFF.replace("product=kinase", "product=chemotaxis%20protein%3B%20CheA;Note=")
    g = load_genome(ACC, make_project(tmp_path, gff=gff))
    assert next(x for x in g.genes if x.locus_tag == "T_1").product == "chemotaxis protein; CheA"


def test_crlf_line_endings_parse_identically(tmp_path):
    a = load_genome(ACC, make_project(tmp_path / "lf"))
    b = load_genome(ACC, make_project(tmp_path / "crlf", fasta=SEQ.replace("\n", "\r\n"),
                                      gff=GOOD_GFF.replace("\n", "\r\n")))
    assert [dataclasses.replace(x, attributes={}) for x in a.genes] == [dataclasses.replace(x, attributes={}) for x in b.genes]


def test_two_distinct_genes_sharing_a_locus_tag_are_not_silently_merged(tmp_path):
    gff = ("##gff-version 3\n"
           "NC_TEST.1\tRefSeq\tgene\t1\t100\t.\t+\t.\tID=g1;gene=aaaA;locus_tag=DUP\n"
           "NC_TEST.1\tRefSeq\tgene\t500\t600\t.\t+\t.\tID=g2;gene=bbbB;locus_tag=DUP\n")
    try:
        g = load_genome(ACC, make_project(tmp_path, gff=gff))
    except ValueError:
        return  # refusing is acceptable
    assert len(g.genes) == 2, f"fused into {[(x.start, x.end, x.gene_key) for x in g.genes]}"


# ----------------------------------------------------- rule semantics (synthetic table)
def pheno(tmp_path, gff, rules, channels=CHANNELS):
    root = make_project(tmp_path, gff=gff, rules=rules)
    return derive_phenotype(load_genome(ACC, root), channels, root)


def test_matching_rule_applies_exactly_once_and_adds_its_effect(tmp_path):
    r = rule("r1", ["T_1"], "motility_um_s", 4.0, SYMBOLS)
    p = pheno(tmp_path, GOOD_GFF, [r])
    assert p.applied_rule_ids == ("r1",) and p.motility_um_s == TABLE["baselines"]["motility_um_s"] + 4.0


def test_split_cds_rows_do_not_multiply_an_effect(tmp_path):
    gff = GOOD_GFF + "NC_TEST.1\tRefSeq\tCDS\t1\t300\t.\t+\t0\tID=cds-1b;Parent=gene-T_1;gene=cheA;locus_tag=T_1\n"
    p = pheno(tmp_path, gff, [rule("r1", ["T_1"], "motility_um_s", 4.0, SYMBOLS)])
    assert p.motility_um_s == TABLE["baselines"]["motility_um_s"] + 4.0


def test_missing_gene_pseudogene_wrong_symbol_and_wrong_feature_type_do_not_fire(tmp_path):
    base = TABLE["baselines"]["motility_um_s"]
    gone = GOOD_GFF.replace("locus_tag=T_1", "locus_tag=T_77")
    assert pheno(tmp_path / "a", gone, [rule("r", ["T_1"], "motility_um_s", 4.0, SYMBOLS)]).motility_um_s == base
    pseudo = GOOD_GFF.replace("ID=gene-T_1;", "ID=gene-T_1;pseudo=true;")
    assert pheno(tmp_path / "b", pseudo, [rule("r", ["T_1"], "motility_um_s", 4.0, SYMBOLS)]).motility_um_s == base
    renamed = GOOD_GFF.replace("gene=cheA", "gene=cheZ").replace("Name=cheA", "Name=cheZ")
    assert pheno(tmp_path / "c", renamed, [rule("r", ["T_1"], "motility_um_s", 4.0, SYMBOLS)]).motility_um_s == base
    cds_only_rule = rule("r", ["T_1"], "motility_um_s", 4.0, SYMBOLS, ftype="CDS")
    gene_only = "\n".join(l for l in GOOD_GFF.splitlines() if "\tCDS\t" not in l) + "\n"
    assert pheno(tmp_path / "d", gene_only, [cds_only_rule]).motility_um_s == base


def test_a_grouped_rule_needs_every_required_gene(tmp_path):
    both = rule("g", ["T_1", "T_2"], "motility_um_s", 3.0, SYMBOLS)
    base = TABLE["baselines"]["motility_um_s"]
    assert pheno(tmp_path / "a", GOOD_GFF, [both]).motility_um_s == base + 3.0
    one_missing = GOOD_GFF.replace("locus_tag=T_2", "locus_tag=T_88")
    assert pheno(tmp_path / "b", one_missing, [both]).motility_um_s == base


def test_effects_clamp_to_schema_bounds_and_storage_plus_biomass_over_one_is_an_error(tmp_path):
    hi = TABLE["bounds"]["motility_um_s"][1]
    big = rule("big", ["T_1"], "motility_um_s", 1e6, SYMBOLS)
    assert pheno(tmp_path / "a", GOOD_GFF, [big]).motility_um_s == hi
    neg = rule("neg", ["T_1"], "motility_um_s", -1e6, SYMBOLS)
    assert pheno(tmp_path / "b", GOOD_GFF, [neg]).motility_um_s == TABLE["bounds"]["motility_um_s"][0]
    over = [rule("s", ["T_1"], "storage_fraction", 0.4, SYMBOLS, nutrient="glucose"),
            rule("b", ["T_2"], "biomass_fraction", 0.15, SYMBOLS, nutrient="glucose")]
    with pytest.raises(ValueError, match="<= 1"):
        pheno(tmp_path / "c", GOOD_GFF, over)  # 0.1+0.4 + 0.75+0.15 = 1.4


def test_nutrient_rule_targets_only_its_own_channel_and_is_skipped_when_channel_absent(tmp_path):
    r = rule("u", ["T_1"], "max_uptake_mmol_s", 1e-4, SYMBOLS, nutrient="lactose")
    p = pheno(tmp_path / "a", GOOD_GFF, [r])
    base = TABLE["baselines"]["max_uptake_mmol_s"]
    assert p.max_uptake_mmol_s == (base["glucose"], base["lactose"] + 1e-4)
    only_glucose = pheno(tmp_path / "b", GOOD_GFF, [r], channels=(NutrientChannel(0, "glucose"),))
    assert only_glucose.max_uptake_mmol_s == (base["glucose"],) and only_glucose.applied_rule_ids == ()


def test_rules_for_another_genome_never_apply(tmp_path):
    r = rule("other", ["T_1"], "motility_um_s", 4.0, SYMBOLS)
    r["genome_accession"] = "GCF_900000009.1"
    r["gene_key"] = "GCF_900000009.1|locus_tag:T_1"
    r["gene_symbols"] = {"GCF_900000009.1|locus_tag:T_1": "cheA"}
    assert pheno(tmp_path, GOOD_GFF, [r]).applied_rule_ids == ()


# --------------------------------------------------- table / channel / collision refusals
@pytest.mark.parametrize("name,edit", [
    ("missing baseline field", lambda t: t["baselines"].pop("motility_um_s")),
    ("missing bounds field", lambda t: t["bounds"].pop("max_age_s")),
    ("baseline outside bounds", lambda t: t["baselines"].__setitem__("motility_um_s", 1e9)),
    ("inverted bounds", lambda t: t["bounds"].__setitem__("motility_um_s", [5.0, 1.0])),
    ("NaN bound", lambda t: t["bounds"].__setitem__("motility_um_s", [0.0, float("nan")])),
    ("wrong schema_version", lambda t: t.__setitem__("schema_version", 2)),
    ("nutrient baseline missing a channel", lambda t: t["baselines"]["max_uptake_mmol_s"].pop("lactose")),
])
def test_incomplete_or_inconsistent_trait_table_is_refused(tmp_path, name, edit):
    root = make_project(tmp_path, rules=[rule("r1", ["T_1"], "motility_um_s", 1.0, SYMBOLS)])
    path = root / "data" / "gene_trait_table.json"
    t = json.loads(path.read_text())
    edit(t)
    path.write_text(json.dumps(t, allow_nan=True))
    with pytest.raises(ValueError):
        derive_phenotype(load_genome(ACC, root), CHANNELS, root)


@pytest.mark.parametrize("name,edit", [
    ("duplicate rule_id", lambda rs: rs.append(copy.deepcopy(rs[0]))),
    ("no evidence", lambda rs: rs[0].__setitem__("evidence", [])),
    ("evidence without url", lambda rs: rs[0].__setitem__("evidence", [{"feature": "x"}])),
    ("empty justification", lambda rs: rs[0].__setitem__("justification", "  ")),
    ("key outside its accession", lambda rs: rs[0].__setitem__("gene_key", "GCF_1.1|locus_tag:T_1")),
    ("unknown trait", lambda rs: rs[0].__setitem__("trait", "speed_of_light")),
    ("unsupported operation", lambda rs: rs[0].__setitem__("operation", "multiply")),
    ("NaN effect", lambda rs: rs[0].__setitem__("value", float("nan"))),
    ("missing field", lambda rs: rs[0].pop("unit")),
])
def test_undocumented_or_malformed_rules_are_refused(tmp_path, name, edit):
    rules = [rule("r1", ["T_1"], "motility_um_s", 1.0, SYMBOLS)]
    edit(rules)
    root = make_project(tmp_path, rules=rules)
    with pytest.raises(ValueError):
        derive_phenotype(load_genome(ACC, root), CHANNELS, root)


def test_bad_nutrient_channels_are_refused(genomes):
    g = genomes[ACCESSIONS[0]]
    for bad in ((), (NutrientChannel(1, "glucose"), NutrientChannel(0, "lactose")),
                (NutrientChannel(0, "glucose"), NutrientChannel(0, "lactose")),
                (NutrientChannel(0, "glucose"), NutrientChannel(1, "glucose")),
                (NutrientChannel(0, "unobtainium"),), (NutrientChannel(0, ""),),
                (NutrientChannel(-1, "glucose"),), (NutrientChannel(2**32, "glucose"),)):
        with pytest.raises((ValueError, TypeError)):
            derive_phenotype(g, bad)
    with pytest.raises(TypeError):
        derive_phenotype(g, ("glucose",))


def test_empty_genome_list_and_genome_id_collisions_are_refused(genomes):
    with pytest.raises(ValueError):
        build_phenotype_table([], CHANNELS)
    g = genomes[ACCESSIONS[0]]
    clone = dataclasses.replace(g, accession="GCF_900000007.1")  # keeps g.genome_id
    with pytest.raises(ValueError, match="collision"):
        build_phenotype_table([g, clone], CHANNELS)


def test_channel_order_is_the_column_order(genomes):
    g = genomes[ACCESSIONS[0]]
    a = derive_phenotype(g, CHANNELS)
    b = derive_phenotype(g, (NutrientChannel(5, "glucose"), NutrientChannel(9, "lactose")))
    assert a.max_uptake_mmol_s == b.max_uptake_mmol_s, "values follow channel NAME, columns follow nutrient_id order"
