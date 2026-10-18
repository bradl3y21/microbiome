"""D2 versioned tables: ``trait_schema.tsv``, ``gene_trait_map.tsv`` and ``gene_features.tsv``.

Producer signatures frozen by D2 (docs/ARCHITECTURE.md):

* ``load_trait_schema(Path) -> TraitSchema``
* ``load_gene_trait_map(Path) -> GeneTraitMap``
* ``normalize_gene_features(GenomeRecord) -> tuple[AnnotatedGene, ...]``
* ``build_phenotype_table(Sequence[GenomeRecord], TraitSchema, GeneTraitMap) -> PhenotypeTable``
* ``explain_phenotype(...) -> tuple[TraitEvidence, ...]``

Semantics (D2): only complete, non-pseudogene annotated features explicitly matched by a rule are eligible;
presence weight is 1.0; ``value = clamp(baseline + sum(selected effects), min, max)``; match namespaces are the
exact set gene_key / locus_tag / GeneID / EC / KO / GO (no fuzzy or motif matching); the lowest integer priority
wins and ties fail; a gene contributes at most once per trait/channel while distinct paralogs count separately;
cross-trait constraints are validated, never silently normalised.

Interpretation recorded here, not in the D2 text: rows that share one ``mapping_id`` form an all-of rule (one
row per required gene), applied once and only if every row matches. D2 has no other way to say "this effect needs
the whole operon", which the bacilysin antagonism rules need.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.ui_api import TraitEvidence
from ..sim.contracts import PhenotypeTable
from .models import AnnotatedGene, GenomePhenotype, GenomeRecord
from .parser import _PROJECT_ROOT, _normalized_groups, _raw_gff3, derive_genome_id

NUTRIENT_TRAITS = ("max_uptake_mmol_s", "storage_fraction", "biomass_fraction", "energy_yield_pj_per_mmol")
SCALAR_TRAITS = (
    "motility_um_s", "chemotaxis_sensitivity", "random_motility_fraction", "division_rate_s",
    "division_mass_pg", "death_rate_s", "max_age_s", "antagonism_radius_um", "antagonism_rate_s",
)


@dataclass(frozen=True, slots=True)
class TraitSet:
    """The trait ids a table family may use. D1 phenotype tables and physiology tables share all other rules."""

    nutrient: tuple[str, ...]
    scalar: tuple[str, ...]

    @property
    def every(self) -> tuple[str, ...]:
        return self.nutrient + self.scalar


NAMESPACES = ("gene_key", "locus_tag", "GeneID", "EC", "KO", "GO")
FEATURE_TYPES = ("gene", "CDS")
UNITS = {
    "max_uptake_mmol_s": "mmol/s", "storage_fraction": "dimensionless", "biomass_fraction": "dimensionless",
    "energy_yield_pj_per_mmol": "pJ/mmol", "motility_um_s": "um/s", "chemotaxis_sensitivity": "dimensionless",
    "random_motility_fraction": "dimensionless", "division_rate_s": "1/s", "division_mass_pg": "pg",
    "death_rate_s": "1/s", "max_age_s": "s", "antagonism_radius_um": "um", "antagonism_rate_s": "1/s",
}
D1_TRAITS = TraitSet(NUTRIENT_TRAITS, SCALAR_TRAITS)
SCHEMA_COLUMNS = ("trait_id", "scope", "nutrient_id", "dtype", "unit", "baseline", "min", "max", "aggregation",
                  "baseline_evidence", "approximation_note")
MAP_COLUMNS = ("mapping_id", "genome_id", "match_namespace", "match_value", "feature_type", "trait_id",
               "nutrient_id", "effect", "unit", "priority", "evidence_class", "evidence_ref", "evidence_url",
               "rationale", "approximation_note")
FEATURE_COLUMNS = (
    "genome_id", "assembly_accession_version", "gene_key", "seqid", "feature_type", "source_id", "parent_id",
    "start_1based", "end_1based", "strand", "phase", "locus_tag", "ncbi_gene_id", "gene_symbol", "product",
    "protein_id_version", "dbxrefs", "pseudo", "partial", "attributes_json", "source_annotation_sha256",
)
PARSER_VERSION = "microbiome-gff3-parser/2"


# ------------------------------------------------------------------------------------------ records
@dataclass(frozen=True, slots=True)
class TraitSpec:
    trait_id: str
    scope: str                       # "scalar" or "nutrient"
    nutrient_id: int | None
    dtype: str
    unit: str
    baseline: float
    minimum: float
    maximum: float
    aggregation: str
    baseline_evidence: str
    approximation_note: str


@dataclass(frozen=True, slots=True)
class TraitSchema:
    version: int
    traits: tuple[TraitSpec, ...]
    sha256: str

    @property
    def nutrient_ids(self) -> tuple[int, ...]:
        return tuple(sorted({spec.nutrient_id for spec in self.traits if spec.nutrient_id is not None}))

    def spec(self, trait_id: str, nutrient_id: int | None = None) -> TraitSpec:
        for spec in self.traits:
            if spec.trait_id == trait_id and spec.nutrient_id == nutrient_id:
                return spec
        raise KeyError((trait_id, nutrient_id))


@dataclass(frozen=True, slots=True)
class MatchTerm:
    namespace: str
    value: str
    feature_type: str


@dataclass(frozen=True, slots=True)
class GeneTraitRule:
    mapping_id: str
    genome_id: int | None
    matches: tuple[MatchTerm, ...]   # all-of; one entry unless the mapping is a conjunction
    trait_id: str
    nutrient_id: int | None
    effect: float
    unit: str
    priority: int
    evidence_class: str
    evidence_refs: tuple[str, ...]
    evidence_urls: tuple[str, ...]
    rationale: str
    approximation_note: str


@dataclass(frozen=True, slots=True)
class GeneTraitMap:
    version: int
    rules: tuple[GeneTraitRule, ...]
    sha256: str


# ------------------------------------------------------------------------------------------ TSV io
def _cell(value: object) -> str:
    return re.sub(r"[\t\r\n]+", " ", str(value))


def _read_tsv(path: str | Path, columns: Sequence[str]) -> tuple[dict[str, str], list[dict[str, str]], str]:
    data = Path(path).read_bytes()
    text = data.decode("utf-8")
    meta: dict[str, str] = {}
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].strip().partition("=")
            meta[key.strip()] = value.strip()
        elif line.strip():
            lines.append(line)
    reader = csv.reader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)
    header = next(reader, None)
    if header is None or tuple(header) != tuple(columns):
        raise ValueError(f"{Path(path).name} must have exactly the columns {list(columns)}; found {header}")
    rows = []
    for number, cells in enumerate(reader, start=2):
        if len(cells) != len(columns):
            raise ValueError(f"{Path(path).name} row {number} has {len(cells)} cells, expected {len(columns)}")
        rows.append(dict(zip(columns, cells)))
    return meta, rows, hashlib.sha256(data).hexdigest()


def _write_tsv(path: Path, meta: dict[str, object], columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    out = [f"#{key}={_cell(value)}" for key, value in meta.items()]
    out.append("\t".join(columns))
    out.extend("\t".join(_cell(row[column]) for column in columns) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(out) + "\n").encode("utf-8"))


def _finite(text: str, what: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{what} is not a number: {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite")
    return value


def _optional_int(text: str, what: str) -> int | None:
    if text == "":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{what} is not an integer: {text!r}") from exc


# ------------------------------------------------------------------------------------------ loaders
def load_trait_schema(path: str | Path, traits: TraitSet = D1_TRAITS) -> TraitSchema:
    meta, rows, digest = _read_tsv(path, SCHEMA_COLUMNS)
    if meta.get("schema_version") != "1":
        raise ValueError("trait_schema.tsv must declare '#schema_version=1'")
    specs: list[TraitSpec] = []
    seen: set[tuple[str, int | None]] = set()
    for row in rows:
        trait, scope = row["trait_id"], row["scope"]
        nutrient_id = _optional_int(row["nutrient_id"], f"{trait} nutrient_id")
        if trait not in traits.every:
            raise ValueError(f"unknown trait_id {trait!r}")
        if scope not in {"scalar", "nutrient"} or (scope == "nutrient") != (trait in traits.nutrient):
            raise ValueError(f"{trait} has scope {scope!r}, which does not match its D1 shape")
        if (nutrient_id is None) != (scope == "scalar"):
            raise ValueError(f"{trait}: nutrient_id is required for nutrient scope and forbidden for scalar scope")
        if row["dtype"] != "float64":
            raise ValueError(f"{trait} dtype must be float64")
        if row["aggregation"] != "sum_clip":
            raise ValueError(f"{trait} aggregation must be sum_clip")
        if not row["unit"] or not row["baseline_evidence"].strip() or not row["approximation_note"].strip():
            raise ValueError(f"{trait} must state a unit, its baseline evidence and an approximation note")
        baseline, low, high = (_finite(row[name], f"{trait} {name}") for name in ("baseline", "min", "max"))
        if low > high or not low <= baseline <= high:
            raise ValueError(f"{trait} baseline must lie within ordered min/max bounds")
        if (trait, nutrient_id) in seen:
            raise ValueError(f"duplicate trait row {(trait, nutrient_id)}")
        seen.add((trait, nutrient_id))
        specs.append(TraitSpec(trait, scope, nutrient_id, "float64", row["unit"], baseline, low, high,
                               "sum_clip", row["baseline_evidence"], row["approximation_note"]))
    nutrient_ids = sorted({spec.nutrient_id for spec in specs if spec.nutrient_id is not None})
    if not nutrient_ids:
        raise ValueError("trait schema must define at least one nutrient channel")
    for trait in traits.nutrient:
        if {spec.nutrient_id for spec in specs if spec.trait_id == trait} != set(nutrient_ids):
            raise ValueError(f"{trait} must be defined for exactly the same nutrient ids as every other nutrient trait")
    for trait in traits.scalar:
        if (trait, None) not in seen:
            raise ValueError(f"trait schema lacks scalar trait {trait}")
    specs.sort(key=lambda spec: (traits.every.index(spec.trait_id), spec.nutrient_id or 0))
    return TraitSchema(1, tuple(specs), digest)


def load_gene_trait_map(path: str | Path, traits: TraitSet = D1_TRAITS) -> GeneTraitMap:
    meta, rows, digest = _read_tsv(path, MAP_COLUMNS)
    if meta.get("schema_version") != "1":
        raise ValueError("gene_trait_map.tsv must declare '#schema_version=1'")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row["mapping_id"]:
            raise ValueError("every gene_trait_map row needs a mapping_id")
        grouped[row["mapping_id"]].append(row)
    shared = ("genome_id", "trait_id", "nutrient_id", "effect", "unit", "priority", "evidence_class",
              "evidence_ref", "evidence_url", "rationale", "approximation_note")
    rules: list[GeneTraitRule] = []
    for mapping_id in sorted(grouped):
        group = grouped[mapping_id]
        first = group[0]
        for row in group[1:]:
            if any(row[name] != first[name] for name in shared):
                raise ValueError(f"rows of mapping {mapping_id} may differ only in their match columns")
        terms = []
        for row in group:
            if row["match_namespace"] not in NAMESPACES:
                raise ValueError(f"mapping {mapping_id}: namespace must be one of {NAMESPACES}, not {row['match_namespace']!r}")
            if row["feature_type"] not in FEATURE_TYPES or not row["match_value"]:
                raise ValueError(f"mapping {mapping_id}: needs a gene/CDS feature_type and a match_value")
            terms.append(MatchTerm(row["match_namespace"], row["match_value"], row["feature_type"]))
        if len(set(terms)) != len(terms):
            raise ValueError(f"mapping {mapping_id} repeats a match row")
        trait = first["trait_id"]
        nutrient_id = _optional_int(first["nutrient_id"], f"{mapping_id} nutrient_id")
        if trait not in traits.every:
            raise ValueError(f"mapping {mapping_id} targets unknown trait {trait!r}")
        if (nutrient_id is None) == (trait in traits.nutrient):
            raise ValueError(f"mapping {mapping_id}: nutrient_id is required exactly for nutrient-scoped traits")
        try:
            priority = int(first["priority"])
        except ValueError as exc:
            raise ValueError(f"mapping {mapping_id}: priority must be an integer") from exc
        if not first["unit"] or not first["rationale"].strip() or not first["approximation_note"].strip():
            raise ValueError(f"mapping {mapping_id} must state unit, rationale and approximation note")
        urls = tuple(part.strip() for part in first["evidence_url"].split(" | ") if part.strip())
        refs = tuple(part.strip() for part in first["evidence_ref"].split(" | ") if part.strip())
        if not urls or not refs or not first["evidence_class"]:
            raise ValueError(f"mapping {mapping_id} must cite an evidence class, reference and URL")
        rules.append(GeneTraitRule(
            mapping_id, _optional_int(first["genome_id"], f"{mapping_id} genome_id"), tuple(terms), trait,
            nutrient_id, _finite(first["effect"], f"{mapping_id} effect"), first["unit"], priority,
            first["evidence_class"], refs, urls, first["rationale"], first["approximation_note"],
        ))
    return GeneTraitMap(1, tuple(rules), digest)


# ------------------------------------------------------------------------------------------ matching
def gene_identifiers(gene: AnnotatedGene) -> dict[str, frozenset[str]]:
    """The exact identifiers a gene offers each D2 namespace; nothing is guessed from names or sequence."""
    attributes = gene.attributes
    dbxrefs = [token.strip() for token in attributes.get("Dbxref", "").split(",") if token.strip()]

    def tagged(prefix: str) -> set[str]:
        return {token[len(prefix):] for token in dbxrefs if token.startswith(prefix)}

    ec = {item.strip() for item in attributes.get("ec_number", "").split(",") if item.strip()} | tagged("EC:")
    go = {item.strip() for item in attributes.get("Ontology_term", "").split(",") if item.strip()} | tagged("GO:")
    return {
        "gene_key": frozenset({gene.gene_key}),
        "locus_tag": frozenset({gene.locus_tag} if gene.locus_tag else ()),
        "GeneID": frozenset(tagged("GeneID:")),
        "EC": frozenset(ec),
        "KO": frozenset(tagged("KO:")),
        "GO": frozenset(go),
    }


def _eligible(gene: AnnotatedGene) -> bool:
    partial = gene.attributes.get("partial", "false").lower() in {"true", "1", "yes"}
    return not gene.pseudogene and not partial


_MATCH_INDEX_CACHE: dict[int, tuple[tuple[AnnotatedGene, ...], dict[tuple[str, str], tuple[AnnotatedGene, ...]]]] = {}


def _match_index(genome: GenomeRecord) -> dict[tuple[str, str], tuple[AnnotatedGene, ...]]:
    """``(namespace, value) -> eligible genes``, built once per gene tuple.

    Keyed by the identity of the immutable ``genes`` tuple, and the entry holds that tuple so its id cannot be
    reused while cached; two records that merely share an accession or hash never share an index.
    """
    entry = _MATCH_INDEX_CACHE.get(id(genome.genes))
    if entry is not None and entry[0] is genome.genes:
        return entry[1]
    building: dict[tuple[str, str], list[AnnotatedGene]] = defaultdict(list)
    for gene in genome.genes:
        if _eligible(gene):
            for namespace, values in gene_identifiers(gene).items():
                for value in values:
                    building[(namespace, value)].append(gene)
    index = {pair: tuple(genes) for pair, genes in building.items()}
    if len(_MATCH_INDEX_CACHE) >= 16:
        _MATCH_INDEX_CACHE.clear()
    _MATCH_INDEX_CACHE[id(genome.genes)] = (genome.genes, index)
    return index


@dataclass(frozen=True, slots=True)
class _Fired:
    rule: GeneTraitRule
    genes: tuple[AnnotatedGene, ...]


def _select(genome: GenomeRecord, gene_map: GeneTraitMap, knockouts: frozenset[str] = frozenset()) -> list[_Fired]:
    """Rules that fire; ``knockouts`` are gene keys treated exactly like pseudogenes (D4 stop-gains)."""
    index = _match_index(genome)

    def matching(term: MatchTerm) -> list[AnnotatedGene]:
        return sorted(
            (gene for gene in index.get((term.namespace, term.value), ())
             if term.feature_type in gene.source_feature_types and gene.gene_key not in knockouts),
            key=lambda gene: gene.gene_key,
        )

    fired: list[_Fired] = []
    for rule in sorted(gene_map.rules, key=lambda item: item.mapping_id):
        if rule.genome_id is not None and rule.genome_id != genome.genome_id:
            continue
        if len(rule.matches) == 1:
            fired.extend(_Fired(rule, (gene,)) for gene in matching(rule.matches[0]))
            continue
        per_term = [matching(term) for term in rule.matches]
        if any(not genes for genes in per_term):
            continue
        if any(len(genes) > 1 for genes in per_term):
            raise ValueError(f"all-of mapping {rule.mapping_id} matches several genes for one required row")
        fired.append(_Fired(rule, tuple(genes[0] for genes in per_term)))

    claims: dict[tuple[str, str, int | None], list[_Fired]] = defaultdict(list)
    for item in fired:
        for gene in item.genes:
            claims[(gene.gene_key, item.rule.trait_id, item.rule.nutrient_id)].append(item)
    losers: set[int] = set()
    for (gene_key, trait, nutrient), contenders in claims.items():
        if len(contenders) < 2:
            continue
        best = min(item.rule.priority for item in contenders)
        winners = [item for item in contenders if item.rule.priority == best]
        if len(winners) > 1:
            names = sorted(item.rule.mapping_id for item in winners)
            raise ValueError(f"priority tie for gene {gene_key}, trait {trait}, channel {nutrient}: {names}")
        losers.update(id(item) for item in contenders if item is not winners[0])
    return [item for item in fired if id(item) not in losers]


# ------------------------------------------------------------------------------------------ phenotype
def _derive(genome: GenomeRecord, schema: TraitSchema, gene_map: GeneTraitMap, knockouts: frozenset[str] = frozenset()):
    for rule in gene_map.rules:
        if rule.nutrient_id is not None and rule.nutrient_id not in schema.nutrient_ids:
            raise ValueError(f"mapping {rule.mapping_id} targets nutrient {rule.nutrient_id}, absent from the schema")
    values: dict[tuple[str, int | None], float] = {(s.trait_id, s.nutrient_id): s.baseline for s in schema.traits}
    selected = _select(genome, gene_map, knockouts)
    for item in selected:
        values[(item.rule.trait_id, item.rule.nutrient_id)] += item.rule.effect
    for spec in schema.traits:
        key = (spec.trait_id, spec.nutrient_id)
        values[key] = min(spec.maximum, max(spec.minimum, values[key]))
    for nutrient in schema.nutrient_ids:
        if ("storage_fraction", nutrient) in values and values[("storage_fraction", nutrient)] + values[("biomass_fraction", nutrient)] > 1.0:
            raise ValueError("storage_fraction + biomass_fraction must be <= 1 for every channel")
    return values, selected


def derive_trait_values(
    genome: GenomeRecord, schema: TraitSchema, gene_map: GeneTraitMap
) -> tuple[dict[tuple[str, int | None], float], list[_Fired]]:
    """Every trait of any table family for one genome: ``{(trait_id, nutrient_id|None): value}`` and the rules fired."""
    return _derive(genome, schema, gene_map)


def build_phenotype_table(
    genomes: Sequence[GenomeRecord], schema: TraitSchema, gene_map: GeneTraitMap
) -> PhenotypeTable:
    """The D1 ``PhenotypeTable`` derived from real annotations through the versioned TSV tables."""
    if not genomes:
        raise ValueError("at least one parsed genome is required")
    ordered = sorted(genomes, key=lambda genome: (genome.genome_id, genome.accession))
    ids = [genome.genome_id for genome in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("uint64 genome_id collision; accession strings remain distinct")
    derived = [_derive(genome, schema, gene_map)[0] for genome in ordered]
    nutrients = schema.nutrient_ids

    def nutrient(trait: str) -> np.ndarray:
        return np.ascontiguousarray(np.asarray([[row[(trait, n)] for n in nutrients] for row in derived], dtype=np.float64))

    def scalar(trait: str) -> np.ndarray:
        return np.ascontiguousarray(np.asarray([row[(trait, None)] for row in derived], dtype=np.float64))

    return PhenotypeTable(
        genome_id=np.ascontiguousarray(np.asarray(ids, dtype=np.uint64)),
        max_uptake_mmol_s=nutrient("max_uptake_mmol_s"),
        storage_fraction=nutrient("storage_fraction"),
        biomass_fraction=nutrient("biomass_fraction"),
        energy_yield_pj_per_mmol=nutrient("energy_yield_pj_per_mmol"),
        motility_um_s=scalar("motility_um_s"),
        division_mass_pg=scalar("division_mass_pg"),
        max_age_s=scalar("max_age_s"),
        division_rate_s=scalar("division_rate_s"),
        death_rate_s=scalar("death_rate_s"),
        antagonism_radius_um=scalar("antagonism_radius_um"),
        antagonism_rate_s=scalar("antagonism_rate_s"),
        chemotaxis_sensitivity=scalar("chemotaxis_sensitivity"),
        random_motility_fraction=scalar("random_motility_fraction"),
    )


def derive_genome_phenotype(
    genome: GenomeRecord,
    schema: TraitSchema,
    gene_map: GeneTraitMap,
    *,
    knockouts: frozenset[str] = frozenset(),
    variant_id: int = 0,
) -> GenomePhenotype:
    """One genome's phenotype record (D4 ``GenomePhenotype``), optionally with knocked-out genes removed."""
    values, selected = _derive(genome, schema, gene_map, frozenset(knockouts))
    nutrients = schema.nutrient_ids

    def per_nutrient(trait: str) -> tuple[float, ...]:
        return tuple(values[(trait, n)] for n in nutrients)

    return GenomePhenotype(
        genome_id=genome.genome_id,
        accession=genome.accession,
        organism=genome.organism,
        max_uptake_mmol_s=per_nutrient("max_uptake_mmol_s"),
        storage_fraction=per_nutrient("storage_fraction"),
        biomass_fraction=per_nutrient("biomass_fraction"),
        energy_yield_pj_per_mmol=per_nutrient("energy_yield_pj_per_mmol"),
        motility_um_s=values[("motility_um_s", None)],
        division_mass_pg=values[("division_mass_pg", None)],
        max_age_s=values[("max_age_s", None)],
        chemotaxis_sensitivity=values[("chemotaxis_sensitivity", None)],
        random_motility_fraction=values[("random_motility_fraction", None)],
        division_rate_s=values[("division_rate_s", None)],
        death_rate_s=values[("death_rate_s", None)],
        antagonism_radius_um=values[("antagonism_radius_um", None)],
        antagonism_rate_s=values[("antagonism_rate_s", None)],
        applied_rule_ids=tuple(sorted({item.rule.mapping_id for item in selected})),
        variant_id=int(variant_id),
    )


def mapped_genes(genome: GenomeRecord, gene_map: GeneTraitMap) -> tuple[AnnotatedGene, ...]:
    """Genes some mapping row selects in the unmutated genome: the only ones a mutation can matter for."""
    unique: dict[str, AnnotatedGene] = {}
    for item in _select(genome, gene_map):
        for gene in item.genes:
            unique[gene.gene_key] = gene
    return tuple(unique[key] for key in sorted(unique))


def explain_phenotype(
    genome: GenomeRecord,
    schema: TraitSchema,
    gene_map: GeneTraitMap,
    knockouts: frozenset[str] = frozenset(),
) -> tuple[TraitEvidence, ...]:
    """Every selected effect with its gene keys, rule id, unit, evidence and approximation note."""
    _, selected = _derive(genome, schema, gene_map, frozenset(knockouts))
    return tuple(
        TraitEvidence(
            rule_id=item.rule.mapping_id,
            gene_keys=tuple(gene.gene_key for gene in item.genes),
            trait_id=item.rule.trait_id,
            nutrient_id=item.rule.nutrient_id,
            effect_value=item.rule.effect,
            unit=item.rule.unit,
            evidence_refs=item.rule.evidence_refs + item.rule.evidence_urls,
            approximation_note=item.rule.approximation_note,
        )
        for item in sorted(selected, key=lambda i: (i.rule.mapping_id, i.genes[0].gene_key))
    )


# ------------------------------------------------------------------------------------------ gene features
def normalize_gene_features(genome: GenomeRecord) -> tuple[AnnotatedGene, ...]:
    """The genome's normalised genes in D2 order (seqid, start, end, strand, type, then stable key)."""
    return tuple(genome.genes)


def gene_feature_rows(genome: GenomeRecord, project_root: str | Path | None = None) -> list[dict[str, str]]:
    """``gene_features.tsv`` rows, one per GFF3 feature part; multipart rows share a ``gene_key``."""
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    raw = _raw_gff3(root / genome.annotation_path, genome.accession)
    rows: list[dict[str, str]] = []
    for key, parts, merged in _normalized_groups(raw, genome.accession):
        pseudo = merged.get("pseudo", "false").lower() in {"true", "1", "yes"} or merged.get("gene_biotype", "").lower() == "pseudogene"
        partial = merged.get("partial", "false").lower() in {"true", "1", "yes"}
        dbxrefs = merged.get("Dbxref", "")
        gene_ids = sorted(token.split(":", 1)[1] for token in dbxrefs.split(",") if token.startswith("GeneID:"))
        for part in parts:
            rows.append({
                "genome_id": str(genome.genome_id),
                "assembly_accession_version": genome.accession,
                "gene_key": key,
                "seqid": part.seqid,
                "feature_type": part.feature_type,
                "source_id": part.attributes.get("ID", ""),
                "parent_id": part.attributes.get("Parent", ""),
                "start_1based": str(part.start),
                "end_1based": str(part.end),
                "strand": part.strand,
                "phase": part.phase,
                "locus_tag": merged.get("locus_tag", ""),
                "ncbi_gene_id": ",".join(gene_ids),
                "gene_symbol": merged.get("gene") or merged.get("Name") or "",
                "product": merged.get("product", ""),
                "protein_id_version": merged.get("protein_id", ""),
                "dbxrefs": dbxrefs,
                "pseudo": "true" if pseudo else "false",
                "partial": "true" if partial else "false",
                "attributes_json": json.dumps(part.attributes, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
                "source_annotation_sha256": genome.annotation_sha256,
            })
    rows.sort(key=lambda r: (r["seqid"], int(r["start_1based"]), int(r["end_1based"]), r["strand"],
                             r["feature_type"], r["source_id"]))
    return rows


def write_gene_features_tsv(genome: GenomeRecord, path: str | Path, project_root: str | Path | None = None) -> Path:
    target = Path(path)
    _write_tsv(target, {"schema_version": 1, "parser_version": PARSER_VERSION,
                        "source_annotation_sha256": genome.annotation_sha256},
               FEATURE_COLUMNS, gene_feature_rows(genome, project_root))
    return target


# ------------------------------------------------------------------------------------------ export
def export_tables_from_json(project_root: str | Path | None = None) -> tuple[Path, Path]:
    """Write ``data/trait_schema.tsv`` and ``data/gene_trait_map.tsv`` from ``data/gene_trait_table.json``.

    The JSON stays the human-curated source; the TSVs are its D2-shaped, hashable, versioned rendering.
    """
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    table = json.loads((root / "data" / "gene_trait_table.json").read_text(encoding="utf-8"))
    nutrient_ids = {str(entry["name"]): int(entry["nutrient_id"]) for entry in table["demo_nutrient_channels"]}
    notice = str(table["approximation_notice"])
    meta = {"schema_version": 1, "table_revision": table["table_revision"], "source": "data/gene_trait_table.json"}

    schema_rows = []
    for trait in NUTRIENT_TRAITS + SCALAR_TRAITS:
        low, high = table["bounds"][trait]
        pairs = ([(nutrient_ids[name], value) for name, value in sorted(table["baselines"][trait].items(), key=lambda kv: nutrient_ids[kv[0]])]
                 if trait in NUTRIENT_TRAITS else [(None, table["baselines"][trait])])
        for nutrient_id, baseline in pairs:
            schema_rows.append({
                "trait_id": trait, "scope": "nutrient" if nutrient_id is not None else "scalar",
                "nutrient_id": "" if nutrient_id is None else nutrient_id, "dtype": "float64", "unit": UNITS[trait],
                "baseline": repr(float(baseline)), "min": repr(float(low)), "max": repr(float(high)),
                "aggregation": "sum_clip",
                "baseline_evidence": "none: a calibrated default, not a measurement (see uptake_division_calibration in data/gene_trait_table.json)",
                "approximation_note": notice,
            })

    map_rows = []
    for rule in sorted(table["rules"], key=lambda r: str(r["rule_id"])):
        keys = [str(key) for key in rule.get("requires_gene_keys", [rule["gene_key"]])]
        nutrient_id = nutrient_ids[str(rule["nutrient_name"])] if rule["trait"] in NUTRIENT_TRAITS else ""
        for key in sorted(keys):
            map_rows.append({
                "mapping_id": rule["rule_id"], "genome_id": derive_genome_id(str(rule["genome_accession"])),
                "match_namespace": "gene_key", "match_value": key, "feature_type": rule["source_feature_type"],
                "trait_id": rule["trait"], "nutrient_id": nutrient_id, "effect": repr(float(rule["value"])),
                "unit": rule["unit"], "priority": 100, "evidence_class": rule["approximation_status"],
                "evidence_ref": " | ".join(f"{item['source']}: {item['feature']}" for item in rule["evidence"]),
                "evidence_url": " | ".join(item["url"] for item in rule["evidence"]),
                "rationale": rule["justification"],
                "approximation_note": f"{rule['approximation_status']}: genome potential, not a measured rate",
            })
    schema_path, map_path = root / "data" / "trait_schema.tsv", root / "data" / "gene_trait_map.tsv"
    _write_tsv(schema_path, meta, SCHEMA_COLUMNS, schema_rows)
    _write_tsv(map_path, meta, MAP_COLUMNS, map_rows)
    return schema_path, map_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m microbiome.genome.tables")
    parser.add_argument("command", choices=["export", "features"])
    parser.add_argument("--accession", help="assembly accession.version, for 'features'")
    parser.add_argument("--out", help="output path, for 'features'")
    args = parser.parse_args(argv)
    if args.command == "export":
        for path in export_tables_from_json():
            print(path)
        return 0
    from .bundle import load_genome_bundle

    if not args.accession or not args.out:
        parser.error("'features' needs --accession and --out")
    print(write_gene_features_tsv(load_genome_bundle(_PROJECT_ROOT / "data" / "genomes" / args.accession), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
