"""Apply the versioned, evidence-bearing exact gene-to-trait mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..sim.contracts import PhenotypeTable
from .models import GenomePhenotype, GenomeRecord, NutrientChannel
from .parser import _PROJECT_ROOT, load_genome
from .tables import GeneTraitMap, TraitSchema, build_phenotype_table as build_table_from_tsv

_NUTRIENT_FIELDS = (
    "max_uptake_mmol_s",
    "storage_fraction",
    "biomass_fraction",
    "energy_yield_pj_per_mmol",
)
_SCALAR_FIELDS = (
    "motility_um_s",
    "chemotaxis_sensitivity",
    "random_motility_fraction",
    "division_rate_s",
    "division_mass_pg",
    "death_rate_s",
    "max_age_s",
    "antagonism_radius_um",
    "antagonism_rate_s",
)


def _table(project_root: str | Path | None = None) -> dict[str, object]:
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    value = json.loads((root / "data" / "gene_trait_table.json").read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported gene trait table schema_version")
    _validate_trait_table(value)
    return value


def _validate_trait_table(table: dict[str, object]) -> None:
    """Reject incomplete baselines, bounds, or undocumented mapping rows."""
    all_fields = set(_NUTRIENT_FIELDS + _SCALAR_FIELDS)
    baselines = table.get("baselines")
    bounds = table.get("bounds")
    if not isinstance(baselines, dict) or set(baselines) != all_fields:
        raise ValueError("trait table must give one explicit baseline for every D2 phenotype field")
    if not isinstance(bounds, dict) or set(bounds) != all_fields:
        raise ValueError("trait table must give explicit numeric bounds for every D2 phenotype field")
    nutrient_names = set(table.get("nutrient_channels", {}))
    if not nutrient_names:
        raise ValueError("trait table must define at least one named nutrient channel")
    for field in _NUTRIENT_FIELDS:
        if not isinstance(baselines[field], dict) or set(baselines[field]) != nutrient_names:
            raise ValueError(f"{field} must give a baseline for every and only defined nutrient channel")
    for field in all_fields:
        try:
            low, high = map(float, bounds[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} bounds must be a two-value numeric pair") from exc
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError(f"{field} bounds must be finite and ordered")
        baseline_values = baselines[field].values() if field in _NUTRIENT_FIELDS else (baselines[field],)
        for baseline in baseline_values:
            numeric = float(baseline)
            if not np.isfinite(numeric) or not low <= numeric <= high:
                raise ValueError(f"{field} baseline must be finite and within its explicit bounds")
    rules = table.get("rules")
    if not isinstance(rules, list):
        raise ValueError("trait table rules must be an explicit list")
    rule_ids: set[str] = set()
    required_fields = {
        "rule_id", "genome_accession", "gene_key", "gene_symbols", "source_feature_type",
        "trait", "operation", "value", "unit", "approximation_status", "justification", "evidence",
    }
    for rule in rules:
        if not isinstance(rule, dict) or required_fields - set(rule):
            raise ValueError("every trait rule must record its key, operation, value, unit, rationale, and evidence")
        rule_id = str(rule["rule_id"])
        if not rule_id or rule_id in rule_ids:
            raise ValueError(f"trait rule_id must be non-empty and unique: {rule_id!r}")
        rule_ids.add(rule_id)
        accession = str(rule["genome_accession"])
        anchor = str(rule["gene_key"])
        required_keys = tuple(str(key) for key in rule.get("requires_gene_keys", [anchor]))
        if not accession or not required_keys or anchor not in required_keys:
            raise ValueError(f"trait rule {rule_id} has invalid assembly/gene keys")
        if any(not key.startswith(f"{accession}|") for key in required_keys):
            raise ValueError(f"trait rule {rule_id} contains a key outside its accession namespace")
        if set(rule["gene_symbols"]) != set(required_keys):
            raise ValueError(f"trait rule {rule_id} must state the exact symbol for every stable gene key")
        if rule["source_feature_type"] not in {"gene", "CDS"}:
            raise ValueError(f"trait rule {rule_id} must identify an annotated gene or CDS source feature")
        if rule["trait"] not in all_fields or rule["operation"] != "add":
            raise ValueError(f"trait rule {rule_id} uses an unsupported trait or operation")
        if not str(rule["unit"]) or not str(rule["approximation_status"]) or not str(rule["justification"]).strip():
            raise ValueError(f"trait rule {rule_id} must document units, approximation status, and justification")
        if not np.isfinite(float(rule["value"])):
            raise ValueError(f"trait rule {rule_id} effect must be finite")
        evidence = rule["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"trait rule {rule_id} must have at least one evidence source")
        if any(not item.get("url") or not item.get("feature") for item in evidence):
            raise ValueError(f"trait rule {rule_id} evidence must cite a URL and supporting feature")
        if rule["trait"] in _NUTRIENT_FIELDS and rule.get("nutrient_name") not in nutrient_names:
            raise ValueError(f"nutrient trait rule {rule_id} must target a defined nutrient channel")


def _validate_channels(channels: Sequence[NutrientChannel], table: dict[str, object]) -> tuple[NutrientChannel, ...]:
    result = tuple(channels)
    if not result:
        raise ValueError("at least one explicit nutrient channel is required")
    if any(not isinstance(channel, NutrientChannel) for channel in result):
        raise TypeError("nutrient_channels must contain NutrientChannel records")
    ids = [int(channel.nutrient_id) for channel in result]
    names = [channel.name for channel in result]
    if any(value < 0 or value > np.iinfo(np.uint32).max for value in ids):
        raise ValueError("nutrient_id must fit uint32")
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("nutrient channels must have strictly increasing nutrient_id")
    if len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("nutrient channel names must be non-empty and unique")
    known = set(table["nutrient_channels"])
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"no explicit schema baseline for nutrient channels: {unknown}")
    return result


def _feature_index(genome: GenomeRecord) -> dict[str, object]:
    return {gene.gene_key: gene for gene in genome.genes if not gene.pseudogene}


def derive_phenotype(
    genome: GenomeRecord,
    nutrient_channels: Sequence[NutrientChannel],
    project_root: str | Path | None = None,
) -> GenomePhenotype:
    """Derive one genome's explicit phenotype; absent mappings never fuzzy-match.

    Effects are summed in stable rule-ID order, and grouped rules fire once only
    when every required annotated gene/CDS is present. The output values are
    bounded by the versioned table. No mutation or expression inference occurs.
    """
    table = _table(project_root)
    channels = _validate_channels(nutrient_channels, table)
    channel_names = [channel.name for channel in channels]
    feature_index = _feature_index(genome)
    baselines = table["baselines"]
    bounds = table["bounds"]
    values: dict[str, object] = {}
    for field in _NUTRIENT_FIELDS:
        field_baselines = baselines[field]
        if set(channel_names) - set(field_baselines):
            raise ValueError(f"{field} lacks explicit baselines for requested nutrient channels")
        values[field] = [float(field_baselines[name]) for name in channel_names]
    for field in _SCALAR_FIELDS:
        if field not in baselines:
            raise ValueError(f"missing explicit baseline for {field}")
        values[field] = float(baselines[field])

    applied: list[str] = []
    rules = sorted(table["rules"], key=lambda rule: str(rule["rule_id"]))
    seen_rule_ids: set[str] = set()
    for rule in rules:
        rule_id = str(rule["rule_id"])
        if rule_id in seen_rule_ids:
            raise ValueError(f"duplicate trait rule_id: {rule_id}")
        seen_rule_ids.add(rule_id)
        if rule.get("genome_accession") != genome.accession:
            continue
        anchor_key = str(rule["gene_key"])
        required_keys = tuple(str(key) for key in rule.get("requires_gene_keys", [anchor_key]))
        if anchor_key not in required_keys:
            raise ValueError(f"rule {rule_id} anchor is not in its required gene set")
        matched = [feature_index.get(key) for key in required_keys]
        if any(feature is None for feature in matched):
            continue
        if any(feature.accession != genome.accession for feature in matched if feature is not None):
            raise ValueError(f"rule {rule_id} contains a cross-assembly gene key")
        expected_feature_type = str(rule["source_feature_type"])
        if any(expected_feature_type not in feature.source_feature_types for feature in matched if feature is not None):
            continue
        expected_symbols = rule.get("gene_symbols", {})
        if any(
            expected_symbols.get(key) != feature.gene_symbol
            for key, feature in zip(required_keys, matched, strict=True)
            if feature is not None
        ):
            continue
        trait = str(rule["trait"])
        operation = str(rule["operation"])
        if operation != "add":
            raise ValueError(f"unsupported operation {operation!r} in rule {rule_id}")
        if trait in _NUTRIENT_FIELDS:
            nutrient_name = str(rule["nutrient_name"])
            if nutrient_name not in channel_names:
                continue
            values[trait][channel_names.index(nutrient_name)] += float(rule["value"])
        elif trait in _SCALAR_FIELDS:
            values[trait] += float(rule["value"])
        else:
            raise ValueError(f"rule {rule_id} targets unknown trait {trait!r}")
        applied.append(rule_id)

    for field in _NUTRIENT_FIELDS:
        low, high = map(float, bounds[field])
        values[field] = tuple(min(high, max(low, value)) for value in values[field])
    for field in _SCALAR_FIELDS:
        low, high = map(float, bounds[field])
        values[field] = min(high, max(low, values[field]))
    if any(a + b > 1.0 for a, b in zip(values["storage_fraction"], values["biomass_fraction"], strict=True)):
        raise ValueError("storage_fraction + biomass_fraction must be <= 1 for every channel")

    return GenomePhenotype(
        genome_id=genome.genome_id,
        accession=genome.accession,
        organism=genome.organism,
        max_uptake_mmol_s=values["max_uptake_mmol_s"],
        storage_fraction=values["storage_fraction"],
        biomass_fraction=values["biomass_fraction"],
        energy_yield_pj_per_mmol=values["energy_yield_pj_per_mmol"],
        motility_um_s=values["motility_um_s"],
        chemotaxis_sensitivity=values["chemotaxis_sensitivity"],
        random_motility_fraction=values["random_motility_fraction"],
        division_rate_s=values["division_rate_s"],
        division_mass_pg=values["division_mass_pg"],
        death_rate_s=values["death_rate_s"],
        max_age_s=values["max_age_s"],
        antagonism_radius_um=values["antagonism_radius_um"],
        antagonism_rate_s=values["antagonism_rate_s"],
        applied_rule_ids=tuple(applied),
    )


def build_phenotype_table(
    genomes: Sequence[GenomeRecord],
    nutrient_channels,
    project_root=None,
) -> PhenotypeTable:
    """Create the frozen D2 ``PhenotypeTable`` with genome-ID sorted rows.

    Two call shapes share this name. The D2 producer signature is
    ``build_phenotype_table(genomes, TraitSchema, GeneTraitMap)`` (versioned TSV tables, see
    :mod:`microbiome.genome.tables`). The original ``build_phenotype_table(genomes, nutrient_channels,
    project_root=None)`` reads ``data/gene_trait_table.json`` and is kept for existing callers.
    """
    if isinstance(nutrient_channels, TraitSchema):
        if not isinstance(project_root, GeneTraitMap):
            raise TypeError("build_phenotype_table(genomes, schema, gene_map) needs a GeneTraitMap as its third argument")
        return build_table_from_tsv(genomes, nutrient_channels, project_root)
    if not genomes:
        raise ValueError("at least one parsed genome is required")
    ordered = sorted(genomes, key=lambda genome: (genome.genome_id, genome.accession))
    ids = [genome.genome_id for genome in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("uint64 genome_id collision; accession strings remain distinct")
    channels = tuple(nutrient_channels)
    rows = [derive_phenotype(genome, channels, project_root) for genome in ordered]
    def nutrient(field: str) -> np.ndarray:
        return np.ascontiguousarray(
            np.asarray([getattr(row, field) for row in rows], dtype=np.float64)
        )

    def scalar(field: str) -> np.ndarray:
        return np.ascontiguousarray(
            np.asarray([getattr(row, field) for row in rows], dtype=np.float64)
        )
    table = PhenotypeTable(
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
    return table


def phenotype_demo(project_root: str | Path | None = None) -> dict[str, object]:
    """Return reproducible trait values for both complete assemblies in data/."""
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    table = _table(root)
    channels = tuple(NutrientChannel(int(entry["nutrient_id"]), str(entry["name"])) for entry in table["demo_nutrient_channels"])
    genomes = sorted(
        (load_genome(accession, root) for accession in table["demo_accessions"]),
        key=lambda genome: (genome.genome_id, genome.accession),
    )
    phenotypes = [derive_phenotype(genome, channels, root) for genome in genomes]
    return {
        "gene_trait_table_version": table["schema_version"],
        "approximation_notice": table["approximation_notice"],
        "nutrient_channels": [{"nutrient_id": channel.nutrient_id, "name": channel.name} for channel in channels],
        "phenotypes": [phenotype.as_dict() for phenotype in phenotypes],
    }
