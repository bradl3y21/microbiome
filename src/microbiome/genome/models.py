"""Immutable records used by the genome parser and phenotype derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _display_float(value: float) -> float:
    """Remove binary floating-point noise from JSON/demo output only."""
    return float(format(float(value), ".15g"))


@dataclass(frozen=True, slots=True)
class AnnotatedGene:
    """One normalized gene with its source GFF3 gene/CDS features.

    Coordinates retain GFF3's 1-based inclusive convention. ``gene_key`` is
    accession-version scoped and uses ``locus_tag`` where available; otherwise
    it is derived from the canonical feature coordinates and type.
    """

    accession: str
    gene_key: str
    locus_tag: str | None
    gene_symbol: str | None
    product: str | None
    seqid: str
    start: int
    end: int
    strand: str
    feature_type: str
    source_feature_types: tuple[str, ...]
    source_ids: tuple[str, ...]
    parent_ids: tuple[str, ...]
    pseudogene: bool
    attributes: Mapping[str, str]
    # (start_1based, end_1based, phase, cds_id) of every GFF3 CDS row, for in-frame stop-codon checks (D4).
    # Rows sharing a cds_id are the parts of one multipart CDS; distinct ids are alternative products.
    cds_parts: tuple[tuple[int, int, int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class GenomeRecord:
    """A validated assembly and normalized annotation from the data manifest."""

    accession: str
    genome_id: int
    organism: str
    tax_id: int
    assembly_level: str
    sequence_path: str
    sequence_length_bp: int
    sequence_record_count: int
    sequence_sha256: str
    annotation_path: str
    annotation_sha256: str
    genes: tuple[AnnotatedGene, ...]


@dataclass(frozen=True, slots=True)
class NutrientChannel:
    """Simulation nutrient ID and the exact substrate name used by the map."""

    nutrient_id: int
    name: str


@dataclass(frozen=True, slots=True)
class GenomePhenotype:
    """Single genome's values in the frozen D2 phenotype schema."""

    genome_id: int
    accession: str
    organism: str
    max_uptake_mmol_s: tuple[float, ...]
    storage_fraction: tuple[float, ...]
    biomass_fraction: tuple[float, ...]
    energy_yield_pj_per_mmol: tuple[float, ...]
    motility_um_s: float
    division_mass_pg: float
    max_age_s: float
    chemotaxis_sensitivity: float
    random_motility_fraction: float
    division_rate_s: float
    death_rate_s: float
    antagonism_radius_um: float
    antagonism_rate_s: float
    applied_rule_ids: tuple[str, ...]
    variant_id: int = 0  # 0 is the unmutated accession genome (D4)

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic, JSON-ready report without internal IDs."""
        return {
            "genome_id": self.genome_id,
            "accession": self.accession,
            "organism": self.organism,
            "max_uptake_mmol_s": [_display_float(value) for value in self.max_uptake_mmol_s],
            "storage_fraction": [_display_float(value) for value in self.storage_fraction],
            "biomass_fraction": [_display_float(value) for value in self.biomass_fraction],
            "energy_yield_pj_per_mmol": [_display_float(value) for value in self.energy_yield_pj_per_mmol],
            "motility_um_s": _display_float(self.motility_um_s),
            "division_mass_pg": _display_float(self.division_mass_pg),
            "max_age_s": _display_float(self.max_age_s),
            "chemotaxis_sensitivity": _display_float(self.chemotaxis_sensitivity),
            "random_motility_fraction": _display_float(self.random_motility_fraction),
            "division_rate_s": _display_float(self.division_rate_s),
            "death_rate_s": _display_float(self.death_rate_s),
            "antagonism_radius_um": _display_float(self.antagonism_radius_um),
            "antagonism_rate_s": _display_float(self.antagonism_rate_s),
            "applied_rule_ids": list(self.applied_rule_ids),
        }
