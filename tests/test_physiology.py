"""Genes -> physiology: real annotations, exact rules, and what each organism's genome therefore supports."""
import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pytest

from microbiome.genome import NutrientChannel, build_phenotype_table, load_genome_bundle, load_gene_trait_map
from microbiome.genome import physiology as P
from microbiome.genome import tables as T
from microbiome.genome.calibration import Calibration
from microbiome.sim.realism import Physiology

ROOT = Path(__file__).resolve().parent.parent
EC, BS, PA, SA, MX = P.EC, P.BS, P.PA, P.SA, P.MX
ALL = (EC, BS, PA, SA, MX)


@pytest.fixture(scope="module")
def genomes():
    return {a: load_genome_bundle(ROOT / "data" / "genomes" / a) for a in ALL}


@pytest.fixture(scope="module")
def tables():
    """Gene-derived physiology only: the calibration (data/calibration_fit.json) is tested in test_calibration.py."""
    return dataclasses.replace(P.load_physiology_tables(ROOT), calibration=Calibration())


@pytest.fixture(scope="module")
def derived(genomes, tables):
    return {a: P.derive_physiology(g, tables) for a, g in genomes.items()}


def test_the_committed_tables_are_exactly_what_the_exporter_produces(tmp_path):
    (tmp_path / "data" / "genomes").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "manifest.json", tmp_path / "data" / "manifest.json")
    for acc in ALL:
        shutil.copytree(ROOT / "data" / "genomes" / acc, tmp_path / "data" / "genomes" / acc)
    schema, gene_map = P.export_physiology_tables(tmp_path)
    assert schema.read_bytes() == (ROOT / "data" / "physiology_schema.tsv").read_bytes()
    assert gene_map.read_bytes() == (ROOT / "data" / "physiology_map.tsv").read_bytes()


def test_every_rule_names_genes_that_exist_with_that_symbol_in_its_own_genome(genomes):
    seen = 0
    for rule in P.RULES:
        genome = genomes[rule.accession]
        for symbol in rule.symbols:
            gene = P._resolve(genome, symbol, rule.loci)
            named_by_locus = bool(rule.loci) and symbol in rule.loci          # annotations without a symbol are named by locus tag
            assert (gene.locus_tag == rule.loci[symbol] if named_by_locus else gene.gene_symbol == symbol)
            assert "CDS" in gene.source_feature_types and not gene.pseudogene
            seen += 1
    assert seen > 90


def test_a_symbol_that_means_different_genes_in_different_organisms_is_not_confused():
    ec, bs = (load_genome_bundle(ROOT / "data" / "genomes" / a) for a in (EC, BS))
    e = next(g for g in ec.genes if g.gene_symbol == "bacA")
    b = next(g for g in bs.genes if g.gene_symbol == "bacA")
    assert e.locus_tag == "b3057" and b.locus_tag == "BSU_37740"
    assert "undecaprenyl" in (e.product or "").lower() and "undecaprenyl" not in (b.product or "").lower()
    ecoli_rules = [r for r in P.RULES if r.accession == EC]
    assert not any(r.trait in ("toxin_production",) for r in ecoli_rules), "E. coli's bacA is not bacilysin synthesis"


def test_the_rule_set_has_no_gene_effect_outside_its_genome_and_no_duplicate_ids():
    ids = [r.rule_id for r in P.RULES]
    assert len(ids) == len(set(ids))
    assert all(r.rule_id.startswith("ecoli-") == (r.accession == EC) for r in P.RULES)


def test_the_loaded_schema_and_map_carry_their_hashes_and_cover_every_channel(tables):
    import hashlib

    assert tables.schema.sha256 == hashlib.sha256((ROOT / "data" / "physiology_schema.tsv").read_bytes()).hexdigest()
    assert tables.gene_map.sha256 == hashlib.sha256((ROOT / "data" / "physiology_map.tsv").read_bytes()).hexdigest()
    assert tables.schema.nutrient_ids == tuple(range(len(P.CHANNEL_NAMES))) and len(P.CHANNEL_NAMES) == 7
    assert len(tables.gene_map.rules) == len(P.RULES)


def test_ecoli_is_a_facultative_aerobe_with_overflow_strong_repression_and_acid_resistance(derived):
    e = derived[EC]
    assert (e.aerobic_capacity, e.fermentation_capacity, e.acetate_overflow, e.ccr_strength) == pytest.approx((1.0, 1.0, 0.4, 1.0))
    assert e.stress_resistance == pytest.approx(1.0) and e.biofilm_capacity == pytest.approx(1.0) and e.qs_capacity == 1.0
    assert e.ph_min == pytest.approx(4.4), "gad + adi + cad lower the growth pH minimum from 5.8 to 4.4"
    assert (e.sporulation_capacity, e.competence_capacity, e.toxin_production) == (0.0, 0.0, 0.0)
    assert e.toxin_susceptibility == 1.0


def test_bsubtilis_sporulates_is_competent_makes_bacilysin_and_is_immune_to_it(derived):
    b = derived[BS]
    assert b.sporulation_capacity == pytest.approx(1.0) and b.competence_capacity == pytest.approx(1.0)
    assert b.toxin_production == pytest.approx(1.0) and b.toxin_susceptibility == pytest.approx(0.1)
    assert b.stress_resistance == pytest.approx(0.6) and b.ph_min == pytest.approx(5.8), "no acid-resistance systems"
    assert 0.3 < b.fermentation_capacity < 0.4 and b.aerobic_capacity == pytest.approx(1.0)
    assert b.biofilm_capacity == pytest.approx(1.0) and b.ccr_strength == pytest.approx(1.0)


def test_the_two_organisms_now_differ_in_many_independent_ways(derived):
    e, b = (dataclasses.asdict(derived[a]) for a in (EC, BS))
    differing = [k for k in e if k != "half_saturation_mmol_l" and not np.isclose(e[k], b[k])]
    assert set(differing) >= {"fermentation_capacity", "stress_resistance", "sporulation_capacity",
                              "competence_capacity", "ph_min", "toxin_production", "toxin_susceptibility"}


def test_an_incomplete_operon_contributes_nothing_but_a_complete_one_counts_once(genomes, tables):
    ec = genomes[EC]
    key = lambda sym: next(g.gene_key for g in ec.genes if g.gene_symbol == sym and "CDS" in g.source_feature_types)
    full, _ = T._derive(ec, tables.schema, tables.gene_map)
    without_cyoA, _ = T._derive(ec, tables.schema, tables.gene_map, frozenset({key("cyoA")}))
    assert full[("aerobic_capacity", None)] == pytest.approx(1.0)
    assert without_cyoA[("aerobic_capacity", None)] == pytest.approx(0.3), "losing one cyo subunit removes the whole bo3 oxidase"
    without_gadC, _ = T._derive(ec, tables.schema, tables.gene_map, frozenset({key("gadC")}))
    assert without_gadC[("ph_min", None)] == pytest.approx(4.4 + 0.7), "the gad system needs its antiporter"
    bs = genomes[BS]
    bkey = lambda sym: next(g.gene_key for g in bs.genes if g.gene_symbol == sym and "CDS" in g.source_feature_types)
    no_spo0A, _ = T._derive(bs, tables.schema, tables.gene_map, frozenset({bkey("spo0A")}))
    assert no_spo0A[("sporulation_capacity", None)] == 0.0
    no_bacG, _ = T._derive(bs, tables.schema, tables.gene_map, frozenset({bkey("bacG")}))
    assert (no_bacG[("toxin_production", None)], no_bacG[("toxin_susceptibility", None)]) == (0.0, 1.0), "no operon, no toxin and no immunity"


def test_evidence_for_every_row_cites_a_reference_url_and_the_approximation(genomes, tables):
    for acc, genome in genomes.items():
        for item in T.explain_phenotype(genome, tables.schema, tables.gene_map):
            assert item.approximation_note and any(r.startswith("http") for r in item.evidence_refs)
            assert any("doi.org" in r for r in item.evidence_refs) or item.rule_id.split("-")[1] in (
                "cyo", "cyd", "pfl", "ldhA", "adhE", "frd", "qox", "cta", "ldh", "als", "acs", "actP", "pts", "cya",
                "cco1", "cio", "aa3", "aa3b", "cbb3", "arc")      # the added organisms' oxidases and arginine fermentation: annotation by product only


def test_the_tsv_loaders_reject_a_trait_that_belongs_to_the_other_family(tmp_path):
    text = (ROOT / "data" / "physiology_map.tsv").read_text(encoding="utf-8").replace("aerobic_capacity", "motility_um_s")
    bad = tmp_path / "bad.tsv"
    bad.write_text(text, encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="unknown trait"):
        load_gene_trait_map(bad, P.PHYSIOLOGY_TRAITS)
    with pytest.raises(ValueError, match="unknown trait"):
        load_gene_trait_map(ROOT / "data" / "physiology_map.tsv")        # the D1 vocabulary does not know these traits


def test_the_widened_d1_table_adds_the_four_channels_without_touching_the_original_columns(genomes, tables):
    base = build_phenotype_table(list(genomes.values()), (NutrientChannel(0, "glucose"), NutrientChannel(1, "lactose")))
    wide = P.extend_phenotype_table(base, list(genomes.values()), tables)
    assert wide.max_uptake_mmol_s.shape == (len(ALL), 7) and wide.storage_fraction.shape == (len(ALL), 7)
    for name in ("max_uptake_mmol_s", "storage_fraction", "biomass_fraction", "energy_yield_pj_per_mmol"):
        assert np.array_equal(getattr(wide, name)[:, :2], getattr(base, name))
    assert np.array_equal(wide.motility_um_s, base.motility_um_s) and np.array_equal(wide.genome_id, base.genome_id)
    row = {int(g): i for i, g in enumerate(wide.genome_id)}
    ec, bs = row[genomes[EC].genome_id], row[genomes[BS].genome_id]
    assert wide.max_uptake_mmol_s[ec, 2] == pytest.approx(P.O2_VMAX_MMOL_S) and wide.max_uptake_mmol_s[bs, 2] == pytest.approx(P.O2_VMAX_MMOL_S)
    assert wide.max_uptake_mmol_s[ec, 3] == pytest.approx(0.7 * 6e-16 + 0.3 * 6e-16) and wide.max_uptake_mmol_s[bs, 3] == pytest.approx(6e-16)
    assert (wide.max_uptake_mmol_s[:, 4:] == 0).all()
    assert (wide.storage_fraction + wide.biomass_fraction <= 1.0).all()
