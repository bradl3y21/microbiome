"""Frozen D3 UI/controller/render boundary types.

Core owns these protocol and DTO declarations. The simulation implementation
owns the mutable world; consumers receive detached, immutable value records.
Units, ranges, and field names follow frozen D1 and D2.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

GenomeId: TypeAlias = np.uint64
AgentId: TypeAlias = np.uint64
CommandId: TypeAlias = np.uint64
NutrientId: TypeAlias = np.uint32
Vec2Um: TypeAlias = tuple[float, float]
CommandStatus: TypeAlias = Literal["pending", "applied", "rejected"]

if TYPE_CHECKING:
    from microbiome.genome.models import AnnotatedGene


def _detached(value: object, dtype: type, ndim: int, name: str) -> np.ndarray:
    """Return an owned, C-order, read-only array with the contract dtype."""
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.ndim}")
    if np.issubdtype(result.dtype, np.floating) and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class GenomeOption:
    """One real, manifest-listed D2 assembly available to the UI."""

    genome_id: GenomeId
    accession_version: str
    organism: str
    tax_id: int
    assembly_level: str
    sequence_sha256: str
    annotation_sha256: str
    source_url: str
    fetched_at_utc: str
    terms_url: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "genome_id", np.uint64(self.genome_id))


@dataclass(frozen=True, slots=True)
class TraitEvidence:
    """One applied D2 rule with its gene basis, units, evidence, and limit."""

    rule_id: str
    gene_keys: tuple[str, ...]
    trait_id: str
    nutrient_id: NutrientId | None
    effect_value: float
    unit: str
    evidence_refs: tuple[str, ...]
    approximation_note: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "gene_keys", tuple(self.gene_keys))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "effect_value", float(self.effect_value))
        if self.nutrient_id is not None:
            object.__setattr__(self, "nutrient_id", np.uint32(self.nutrient_id))


@dataclass(frozen=True, slots=True)
class PhenotypePreview:
    """One genome's D2 phenotype in ascending nutrient_id order.

    D2 bounds: max_uptake_mmol_s [0, 0.001] mmol/s; storage_fraction [0, 0.5]
    and biomass_fraction [0, 0.9], dimensionless, with their sum <= 1;
    energy_yield_pj_per_mmol [0, 1e16] pJ/mmol; motility_um_s [0, 50] µm/s;
    chemotaxis_sensitivity [0, 2] and random_motility_fraction [0, 1],
    dimensionless; division_rate_s and death_rate_s [0, 0.01] s^-1;
    division_mass_pg [0.1, 10] pg; max_age_s [1, 1e9] s;
    antagonism_radius_um [0, 100] µm; antagonism_rate_s [0, 0.01] s^-1.
    """

    genome_id: GenomeId
    nutrient_id: NDArray[np.uint32]
    max_uptake_mmol_s: NDArray[np.float64]
    storage_fraction: NDArray[np.float64]
    biomass_fraction: NDArray[np.float64]
    energy_yield_pj_per_mmol: NDArray[np.float64]
    motility_um_s: float
    chemotaxis_sensitivity: float
    random_motility_fraction: float
    division_rate_s: float
    division_mass_pg: float
    death_rate_s: float
    max_age_s: float
    antagonism_radius_um: float
    antagonism_rate_s: float
    applied_rule_ids: tuple[str, ...]
    evidence: tuple[TraitEvidence, ...]

    def __post_init__(self) -> None:
        nutrient_id = _detached(self.nutrient_id, np.uint32, 1, "nutrient_id")
        object.__setattr__(self, "nutrient_id", nutrient_id)
        k = len(nutrient_id)
        nutrient_fields = (
            "max_uptake_mmol_s",
            "storage_fraction",
            "biomass_fraction",
            "energy_yield_pj_per_mmol",
        )
        for name in nutrient_fields:
            values = _detached(getattr(self, name), np.float64, 1, name)
            if values.shape != (k,):
                raise ValueError(f"{name} must have shape (K,)")
            object.__setattr__(self, name, values)
        bounds = {
            "max_uptake_mmol_s": (0.0, 0.001),
            "storage_fraction": (0.0, 0.5),
            "biomass_fraction": (0.0, 0.9),
            "energy_yield_pj_per_mmol": (0.0, 1e16),
        }
        for name, (low, high) in bounds.items():
            values = getattr(self, name)
            if np.any(values < low) or np.any(values > high):
                raise ValueError(f"{name} is outside D2 bounds [{low}, {high}]")
        if np.any(self.storage_fraction + self.biomass_fraction > 1.0):
            raise ValueError("storage_fraction + biomass_fraction must be <= 1")
        scalar_bounds = {
            "motility_um_s": (0.0, 50.0),
            "chemotaxis_sensitivity": (0.0, 2.0),
            "random_motility_fraction": (0.0, 1.0),
            "division_rate_s": (0.0, 0.01),
            "division_mass_pg": (0.1, 10.0),
            "death_rate_s": (0.0, 0.01),
            "max_age_s": (1.0, 1e9),
            "antagonism_radius_um": (0.0, 100.0),
            "antagonism_rate_s": (0.0, 0.01),
        }
        for name, (low, high) in scalar_bounds.items():
            value = float(getattr(self, name))
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} is outside D2 bounds [{low}, {high}]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "applied_rule_ids", tuple(self.applied_rule_ids))
        object.__setattr__(self, "evidence", tuple(self.evidence))


@dataclass(frozen=True, slots=True)
class CellInspection:
    """Detached cell state keyed by persistent agent_id, never row index."""

    agent_id: AgentId
    species_id: np.uint32
    genome_id: GenomeId
    accession_version: str
    organism: str
    tax_id: int
    position_um: Vec2Um
    velocity_um_s: Vec2Um
    mass_pg: float
    radius_um: float
    energy_pj: float
    age_s: float
    nutrient_id: NDArray[np.uint32]
    nutrient_store_mmol: NDArray[np.float64]
    biomass_mmol: NDArray[np.float64]
    phenotype: PhenotypePreview

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", np.uint64(self.agent_id))
        object.__setattr__(self, "species_id", np.uint32(self.species_id))
        object.__setattr__(self, "genome_id", np.uint64(self.genome_id))
        for name in ("position_um", "velocity_um_s"):
            vector = tuple(float(value) for value in getattr(self, name))
            if len(vector) != 2 or not all(np.isfinite(value) for value in vector):
                raise ValueError(f"{name} must be a finite 2D world vector")
            object.__setattr__(self, name, vector)
        for name in ("mass_pg", "radius_um", "energy_pj", "age_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        nutrient_id = _detached(self.nutrient_id, np.uint32, 1, "nutrient_id")
        object.__setattr__(self, "nutrient_id", nutrient_id)
        k = len(nutrient_id)
        for name in ("nutrient_store_mmol", "biomass_mmol"):
            values = _detached(getattr(self, name), np.float64, 1, name)
            if values.shape != (k,):
                raise ValueError(f"{name} must have shape (K,)")
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class GenomeInspection:
    """Manifest provenance and normalized D2 gene records; no mutable sequence."""

    genome: GenomeOption
    sequence_fasta_path: str
    annotation_gff3_path: str
    genes: tuple[AnnotatedGene, ...]
    phenotype_evidence: tuple[TraitEvidence, ...]

    def __post_init__(self) -> None:
        # D2 AnnotatedGene is frozen; freeze its attributes mapping too, so a
        # nested DTO cannot mutate the genome catalogue through a dict.
        genes = []
        for gene in self.genes:
            attributes = getattr(gene, "attributes", None)
            if attributes is not None:
                gene = replace(gene, attributes=MappingProxyType(dict(attributes)))
            genes.append(gene)
        object.__setattr__(self, "genes", tuple(genes))
        object.__setattr__(self, "phenotype_evidence", tuple(self.phenotype_evidence))


@dataclass(frozen=True, slots=True)
class CommandTicket:
    """Receipt for a validated command queued for target_tick."""

    command_id: CommandId
    target_tick: np.uint64

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", np.uint64(self.command_id))
        object.__setattr__(self, "target_tick", np.uint64(self.target_tick))


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Command status; a successful introduction includes its persistent id."""

    command_id: CommandId
    status: CommandStatus
    target_tick: np.uint64
    applied_tick: np.uint64 | None = None
    agent_id: AgentId | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", np.uint64(self.command_id))
        object.__setattr__(self, "target_tick", np.uint64(self.target_tick))
        if self.applied_tick is not None:
            object.__setattr__(self, "applied_tick", np.uint64(self.applied_tick))
        if self.agent_id is not None:
            object.__setattr__(self, "agent_id", np.uint64(self.agent_id))


@dataclass(frozen=True, slots=True)
class RenderSnapshot:
    """Detached D1/D2 render projection with owned, read-only arrays."""

    tick_index: np.uint64
    width_um: float
    height_um: float
    agent_id: NDArray[np.uint64]
    position_um: NDArray[np.float64]
    radius_um: NDArray[np.float64]
    genome_id: NDArray[np.uint64]
    nutrient_id: NDArray[np.uint32]
    concentration_mmol_l: NDArray[np.float64]
    phenotype_genome_id: NDArray[np.uint64]
    phenotype_motility_um_s: NDArray[np.float64]
    phenotype_max_uptake_mmol_s: NDArray[np.float64]

    def __post_init__(self) -> None:
        tick = int(self.tick_index)
        width, height = float(self.width_um), float(self.height_um)
        if tick < 0 or not np.isfinite(width) or not np.isfinite(height) or width <= 0.0 or height <= 0.0:
            raise ValueError("snapshot tick and world dimensions must be valid")
        object.__setattr__(self, "tick_index", np.uint64(tick))
        object.__setattr__(self, "width_um", width)
        object.__setattr__(self, "height_um", height)
        fields = {
            "agent_id": (np.uint64, 1),
            "position_um": (np.float64, 2),
            "radius_um": (np.float64, 1),
            "genome_id": (np.uint64, 1),
            "nutrient_id": (np.uint32, 1),
            "concentration_mmol_l": (np.float64, 3),
            "phenotype_genome_id": (np.uint64, 1),
            "phenotype_motility_um_s": (np.float64, 1),
            "phenotype_max_uptake_mmol_s": (np.float64, 2),
        }
        for name, (dtype, ndim) in fields.items():
            object.__setattr__(self, name, _detached(getattr(self, name), dtype, ndim, name))
        n, k, g = len(self.agent_id), len(self.nutrient_id), len(self.phenotype_genome_id)
        if self.position_um.shape != (n, 2) or self.radius_um.shape != (n,) or self.genome_id.shape != (n,):
            raise ValueError("render agent arrays must align as N rows")
        if self.concentration_mmol_l.shape[0] != k:
            raise ValueError("nutrient_id must match the field K axis")
        if self.phenotype_motility_um_s.shape != (g,) or self.phenotype_max_uptake_mmol_s.shape != (g, k):
            raise ValueError("phenotype arrays must align as [G] and [G,K]")


@dataclass(frozen=True, slots=True)
class RealismSnapshot(RenderSnapshot):
    """A RenderSnapshot plus per-cell realism state, produced only when the realism layer is on (D5 extension).

    ``cell_state``: 0 vegetative, 1 sporulating, 2 dormant spore, 3 matrix-producing (biofilm); ``eps`` is the matrix
    level. The plain ``RenderSnapshot`` keeps exactly its frozen D3 fields; this is a subclass, so everything that
    consumes the base class keeps working unchanged.
    """

    channel_names: tuple[str, ...] = ()
    cell_state: NDArray[np.uint8] = None  # type: ignore[assignment]
    eps: NDArray[np.float64] = None  # type: ignore[assignment]
    length_um: NDArray[np.float64] = None  # type: ignore[assignment]   # true rod length of each cell
    cell_width_um: float = 0.8
    heading_rad: NDArray[np.float64] = None  # type: ignore[assignment]   # axis direction of each rod, radians
    genome_style: NDArray[np.uint8] = None  # type: ignore[assignment]   # imaging style of each phenotype row (aligned with phenotype_genome_id)

    def __post_init__(self) -> None:
        RenderSnapshot.__post_init__(self)
        object.__setattr__(self, "channel_names", tuple(str(name) for name in self.channel_names))
        n = len(self.agent_id)
        object.__setattr__(self, "cell_state", _detached(self.cell_state, np.uint8, 1, "cell_state"))
        object.__setattr__(self, "eps", _detached(self.eps, np.float64, 1, "eps"))
        if self.length_um is not None:
            object.__setattr__(self, "length_um", _detached(self.length_um, np.float64, 1, "length_um"))
            if self.length_um.shape != (n,):
                raise ValueError("length_um must align as N rows")
        if self.heading_rad is not None:
            object.__setattr__(self, "heading_rad", _detached(self.heading_rad, np.float64, 1, "heading_rad"))
            if self.heading_rad.shape != (n,):
                raise ValueError("heading_rad must align as N rows")
        if self.genome_style is not None:
            object.__setattr__(self, "genome_style", _detached(self.genome_style, np.uint8, 1, "genome_style"))
            if self.genome_style.shape != self.phenotype_genome_id.shape:
                raise ValueError("genome_style must align with phenotype_genome_id")
        object.__setattr__(self, "cell_width_um", float(self.cell_width_um))
        if self.cell_state.shape != (n,) or self.eps.shape != (n,):
            raise ValueError("realism per-cell arrays must align as N rows")
        if self.channel_names and len(self.channel_names) != len(self.nutrient_id):
            raise ValueError("channel_names must name every field channel")


@runtime_checkable
class SimulationController(Protocol):
    """D3 UI facade; the implementation in sim owns mutable world state."""

    def list_genomes(self) -> tuple[GenomeOption, ...]: ...

    def preview_phenotype(self, genome_id: GenomeId) -> PhenotypePreview: ...

    def place_nutrient(self, nutrient_id: NutrientId, position_um: Vec2Um, amount_mmol: float) -> CommandTicket: ...

    def introduce_organism(self, genome_id: GenomeId, position_um: Vec2Um) -> CommandTicket: ...

    def command_result(self, command_id: CommandId) -> CommandResult | None: ...

    def inspect_cell(self, agent_id: AgentId) -> CellInspection | None: ...

    def inspect_genome(self, genome_id: GenomeId) -> GenomeInspection | None: ...

    def set_paused(self, paused: bool) -> None: ...

    def advance_ticks(self, count: int) -> np.uint64: ...

    def single_step(self) -> np.uint64: ...

    def render_snapshot(self) -> RenderSnapshot: ...
