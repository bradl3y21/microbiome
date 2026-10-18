"""D4: accession-preserving derived variants (SNVs) of real genomes, and their re-derived phenotypes.

Frozen producer signatures (docs/ARCHITECTURE.md, D4): ``create_derived_variant``, ``derive_variant_phenotype``
and ``build_variant_phenotype_table``. The raw genome package is never touched: a variant is a hashed record of
single-nucleotide substitutions against the *inherited* real sequence, and it can change a phenotype in exactly
one way, an in-frame premature stop codon in a complete, D2-mapped CDS. That gene is then removed from the D2
derivation (as a pseudogene would be) and everything else is the unchanged D2 aggregation.

Choices D4 leaves open, fixed here so runs are reproducible:

* RNG draws per daughter division, in order: ``poisson(L * 2e-10)``; if K > 0, ``choice(L, K, replace=False)``
  then ``integers(0, 3, K)`` (index into the three other bases in A,C,G,T order). K = 0 draws nothing more.
* Positions index the FASTA records concatenated in file order; a sequence digest is the SHA-256 of that
  concatenation as codes 0..3 (A,C,G,T), not of the file bytes (``root_sequence_sha256`` is the file hash).
* ``phenotype_sha256`` hashes the 13 D1 traits plus the root ``genome_id``; ``variant_id`` is excluded because it
  is derived from the record that contains this hash.
* An isoform is a CDS id; rows sharing an id are one multipart CDS. A gene is knocked out only if every isoform
  is translatable (length divisible by 3, ending in a stop) and carries an earlier in-frame stop.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..sim.contracts import PhenotypeTable
from .models import AnnotatedGene, GenomePhenotype, GenomeRecord
from .parser import _PROJECT_ROOT
from .tables import GeneTraitMap, TraitSchema, derive_genome_phenotype, mapped_genes

MUTATION_MODEL_ID = "snv-poisson-2e-10-v1"
TRANSFORMATION_MODEL_ID = "transformation-snv-v1"   # a recombination tract imported from another cell of the same species
_MODEL_IDS = (MUTATION_MODEL_ID, TRANSFORMATION_MODEL_ID)
_MASK64 = 2**64 - 1
SUBSTITUTIONS_PER_BASE = 2.0e-10
BASES = "ACGT"
_UINT64_MAX = 2**64 - 1
_STOP_INDEX = np.asarray([48, 50, 56])  # TAA, TAG, TGA with A=0,C=1,G=2,T=3 and index 16a+4b+c
_LUT = np.full(256, 255, dtype=np.uint8)
for _code, _base in enumerate(BASES):
    _LUT[ord(_base)] = _code
_TRAITS = (
    "max_uptake_mmol_s", "storage_fraction", "biomass_fraction", "energy_yield_pj_per_mmol", "motility_um_s",
    "chemotaxis_sensitivity", "random_motility_fraction", "division_rate_s", "division_mass_pg", "death_rate_s",
    "max_age_s", "antagonism_radius_um", "antagonism_rate_s",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


# ------------------------------------------------------------------------------------------ records
@dataclass(frozen=True, slots=True)
class NucleotideSubstitution:
    position_0based: int
    ref: str
    alt: str

    def __post_init__(self) -> None:
        if self.ref not in BASES or self.alt not in BASES or len(self.ref) != 1 or len(self.alt) != 1:
            raise ValueError("substitution bases must each be one of A, C, G, T")
        if self.ref == self.alt:
            raise ValueError("a substitution must change the base (ref != alt)")
        if not 0 <= int(self.position_0based) <= _UINT64_MAX:
            raise ValueError("position_0based must be a uint64")
        object.__setattr__(self, "position_0based", int(self.position_0based))

    def to_json_dict(self) -> dict[str, object]:
        return {"position_0based": self.position_0based, "ref": self.ref, "alt": self.alt}


@dataclass(frozen=True, slots=True)
class DerivedVariant:
    """Immutable lineage/provenance record; ``variant_id`` and ``record_sha256`` are derived from the payload."""

    variant_id: int
    record_sha256: str
    root_accession_version: str
    root_genome_id: int
    root_sequence_sha256: str
    root_annotation_sha256: str
    parent_variant_id: int
    parent_record_sha256: str | None
    parent_sequence_sha256: str
    derived_sequence_sha256: str
    sequence_length_bp: int
    origin_tick: int
    parent_agent_id: int
    daughter_agent_id: int
    substitutions: tuple[NucleotideSubstitution, ...]
    trait_schema_sha256: str
    gene_trait_map_sha256: str
    phenotype_sha256: str
    mutation_model_id: str = MUTATION_MODEL_ID
    schema_version: int = 1

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "root_accession_version": self.root_accession_version,
            "root_genome_id": str(self.root_genome_id),
            "root_sequence_sha256": self.root_sequence_sha256,
            "root_annotation_sha256": self.root_annotation_sha256,
            "parent_variant_id": str(self.parent_variant_id),
            "parent_record_sha256": self.parent_record_sha256,
            "parent_sequence_sha256": self.parent_sequence_sha256,
            "derived_sequence_sha256": self.derived_sequence_sha256,
            "sequence_length_bp": str(self.sequence_length_bp),
            "origin_tick": str(self.origin_tick),
            "parent_agent_id": str(self.parent_agent_id),
            "daughter_agent_id": str(self.daughter_agent_id),
            "mutation_model_id": self.mutation_model_id,
            "substitutions": [item.to_json_dict() for item in self.substitutions],
            "trait_schema_sha256": self.trait_schema_sha256,
            "gene_trait_map_sha256": self.gene_trait_map_sha256,
            "phenotype_sha256": self.phenotype_sha256,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {**self.payload(), "variant_id": str(self.variant_id), "record_sha256": self.record_sha256}

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "DerivedVariant":
        variant = cls(
            variant_id=int(data["variant_id"]), record_sha256=str(data["record_sha256"]),
            root_accession_version=str(data["root_accession_version"]), root_genome_id=int(data["root_genome_id"]),
            root_sequence_sha256=str(data["root_sequence_sha256"]), root_annotation_sha256=str(data["root_annotation_sha256"]),
            parent_variant_id=int(data["parent_variant_id"]),
            parent_record_sha256=None if data["parent_record_sha256"] is None else str(data["parent_record_sha256"]),
            parent_sequence_sha256=str(data["parent_sequence_sha256"]),
            derived_sequence_sha256=str(data["derived_sequence_sha256"]),
            sequence_length_bp=int(data["sequence_length_bp"]), origin_tick=int(data["origin_tick"]),
            parent_agent_id=int(data["parent_agent_id"]), daughter_agent_id=int(data["daughter_agent_id"]),
            substitutions=tuple(NucleotideSubstitution(int(s["position_0based"]), s["ref"], s["alt"]) for s in data["substitutions"]),
            trait_schema_sha256=str(data["trait_schema_sha256"]), gene_trait_map_sha256=str(data["gene_trait_map_sha256"]),
            phenotype_sha256=str(data["phenotype_sha256"]), mutation_model_id=str(data["mutation_model_id"]),
            schema_version=int(data["schema_version"]),
        )
        variant.verify()
        return variant

    def verify(self) -> None:
        """Re-derive the record hash and id from the payload; any edit to any field fails here."""
        if self.schema_version != 1 or self.mutation_model_id not in _MODEL_IDS:
            raise ValueError("unsupported DerivedVariant schema_version or mutation_model_id")
        digest = _sha256(_canonical(self.payload()))
        if digest != self.record_sha256:
            raise ValueError("record_sha256 does not match the canonical payload")
        if self.variant_id != int.from_bytes(bytes.fromhex(digest)[:8], "big") or self.variant_id == 0:
            raise ValueError("variant_id is not the first eight bytes of record_sha256 (or is zero)")
        positions = [item.position_0based for item in self.substitutions]
        if positions != sorted(set(positions)):
            raise ValueError("substitutions must be sorted by position with no duplicates")


def _seal(**fields: object) -> DerivedVariant:
    draft = DerivedVariant(variant_id=0, record_sha256="", **fields)  # type: ignore[arg-type]
    digest = _sha256(_canonical(draft.payload()))
    return dataclasses.replace(draft, record_sha256=digest, variant_id=int.from_bytes(bytes.fromhex(digest)[:8], "big"))


def phenotype_sha256(phenotype: GenomePhenotype) -> str:
    """SHA-256 of the 13 D1 traits (float64 as ``float.hex()``) plus the root genome id; never the variant id."""
    row: dict[str, object] = {"genome_id": str(phenotype.genome_id)}
    for name in _TRAITS:
        value = getattr(phenotype, name)
        row[name] = [float(v).hex() for v in value] if isinstance(value, tuple) else float(value).hex()
    return _sha256(_canonical(row))


# ------------------------------------------------------------------------------------------ sequence
@dataclass(frozen=True, slots=True)
class RootSequence:
    codes: np.ndarray                 # uint8 0..3, FASTA records concatenated in file order
    offsets: Mapping[str, int]
    sha256: str                       # of ``codes``

    @classmethod
    def load(cls, genome: GenomeRecord, project_root: Path) -> "RootSequence":
        records: dict[str, list[bytes]] = {}
        current: list[bytes] | None = None
        for line in (project_root / genome.sequence_path).read_bytes().split(b"\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith(b">"):
                current = records.setdefault(line[1:].split()[0].decode("ascii"), [])
            elif current is not None:
                current.append(line.upper())
        offsets: dict[str, int] = {}
        pieces = []
        total = 0
        for seqid, chunks in records.items():
            offsets[seqid] = total
            piece = np.frombuffer(b"".join(chunks), dtype=np.uint8)
            pieces.append(piece)
            total += piece.size
        codes = _LUT[np.concatenate(pieces)]
        if np.any(codes == 255):
            raise ValueError("mutation requires an unambiguous A/C/G/T sequence; this assembly has other symbols")
        if total != genome.sequence_length_bp:
            raise ValueError("FASTA length disagrees with the genome record")
        codes.setflags(write=False)
        return cls(codes, offsets, _sha256(codes.tobytes()))


def _apply(codes: np.ndarray, substitutions: Sequence[NucleotideSubstitution]) -> np.ndarray:
    out = np.array(codes, copy=True)
    for item in substitutions:
        if item.position_0based >= out.size:
            raise ValueError("substitution lies beyond the sequence")
        if out[item.position_0based] != BASES.index(item.ref):
            raise ValueError(f"ref allele at {item.position_0based} does not match the inherited sequence")
        out[item.position_0based] = BASES.index(item.alt)
    out.setflags(write=False)
    return out


class VariantStore:
    """All variant records of one session, plus verified sequence reconstruction from root + chain."""

    def __init__(self, project_root: str | Path | None = None, cache: int = 3) -> None:
        self._root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
        self._variants: dict[int, DerivedVariant] = {}
        self._roots: dict[str, RootSequence] = {}
        self._reserved: set[int] = set()
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._cache_size = cache

    def reserve_genome_ids(self, ids: Iterable[int]) -> None:
        self._reserved.update(int(i) for i in ids)

    def root(self, genome: GenomeRecord) -> RootSequence:
        found = self._roots.get(genome.accession)
        if found is None:
            found = self._roots[genome.accession] = RootSequence.load(genome, self._root)
        return found

    def register(self, variant: DerivedVariant) -> None:
        variant.verify()
        if variant.variant_id in self._reserved:
            raise ValueError("variant_id collides with an accession genome id")
        existing = self._variants.get(variant.variant_id)
        if existing is not None and existing != variant:
            raise ValueError("variant_id collision: refusing to overwrite a different record")
        self._variants[variant.variant_id] = variant

    def get(self, variant_id: int) -> DerivedVariant:
        try:
            return self._variants[int(variant_id)]
        except KeyError:
            raise ValueError(f"unknown variant_id {int(variant_id)}") from None

    def __len__(self) -> int:
        return len(self._variants)

    def chain(self, variant: DerivedVariant | None) -> list[DerivedVariant]:
        links: list[DerivedVariant] = []
        node = variant
        while node is not None:
            links.append(node)
            if node.parent_variant_id == 0:
                if node.parent_record_sha256 is not None:
                    raise ValueError("a direct child of the accession genome has no parent record")
                break
            parent = self.get(node.parent_variant_id)
            if parent.record_sha256 != node.parent_record_sha256:
                raise ValueError("parent_record_sha256 does not match the stored parent record")
            node = parent
        links.reverse()
        return links

    def sequence(self, genome: GenomeRecord, variant: DerivedVariant | None) -> np.ndarray:
        """Inherited sequence of ``variant`` with every ref allele and both digests verified at each link."""
        root = self.root(genome)
        if variant is None or variant.variant_id == 0:
            return root.codes
        key = (genome.accession, variant.variant_id)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        codes = root.codes
        digest = root.sha256
        for link in self.chain(variant):
            if (link.root_genome_id != genome.genome_id or link.root_accession_version != genome.accession
                    or link.root_sequence_sha256 != genome.sequence_sha256
                    or link.root_annotation_sha256 != genome.annotation_sha256):
                raise ValueError("variant record does not belong to this accession's immutable sources")
            if link.sequence_length_bp != codes.size or link.parent_sequence_sha256 != digest:
                raise ValueError("parent sequence digest does not match the reconstructed parent")
            codes = _apply(codes, link.substitutions)
            digest = _sha256(codes.tobytes())
            if digest != link.derived_sequence_sha256:
                raise ValueError("derived sequence digest does not match the reconstruction")
        self._cache[key] = codes
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return codes


_DEFAULT_STORE: VariantStore | None = None


def _store_or_default(store: VariantStore | None) -> VariantStore:
    global _DEFAULT_STORE
    if store is not None:
        return store
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = VariantStore()
    return _DEFAULT_STORE


# ------------------------------------------------------------------------------------------ stop-gains
def _isoform_premature_stop(gene: AnnotatedGene, parts: list[tuple[int, int, int]], codes: np.ndarray,
                            offsets: Mapping[str, int]) -> bool | None:
    """True/False for a translatable isoform, None if it cannot be judged (so it is never called knocked out)."""
    if gene.strand not in {"+", "-"} or gene.seqid not in offsets:
        return None
    base = offsets[gene.seqid]
    ordered = sorted(parts, key=lambda part: part[0], reverse=gene.strand == "-")
    pieces = []
    for start, end, _phase in ordered:
        piece = codes[base + start - 1: base + end]
        pieces.append((3 - piece)[::-1] if gene.strand == "-" else piece)
    coding = np.concatenate(pieces)[ordered[0][2]:]
    if coding.size == 0 or coding.size % 3:
        return None
    triplets = coding.reshape(-1, 3).astype(np.int16)
    stops = np.isin(triplets[:, 0] * 16 + triplets[:, 1] * 4 + triplets[:, 2], _STOP_INDEX)
    if not stops[-1]:
        return None
    return bool(stops[:-1].any())


def stop_gain_knockouts(genes: Iterable[AnnotatedGene], codes: np.ndarray, offsets: Mapping[str, int]) -> frozenset[str]:
    """Gene keys whose every CDS isoform now has an in-frame stop before its terminal stop."""
    knocked: set[str] = set()
    for gene in genes:
        isoforms: dict[str, list[tuple[int, int, int]]] = {}
        for number, (start, end, phase, cds_id) in enumerate(gene.cds_parts):
            isoforms.setdefault(cds_id or f"#{number}", []).append((start, end, phase))
        if not isoforms:
            continue
        verdicts = [_isoform_premature_stop(gene, parts, codes, offsets) for parts in isoforms.values()]
        if all(verdict is True for verdict in verdicts):
            knocked.add(gene.gene_key)
    return frozenset(knocked)


def _knockouts(genome: GenomeRecord, gene_map: GeneTraitMap, store: VariantStore, codes: np.ndarray) -> frozenset[str]:
    return stop_gain_knockouts(mapped_genes(genome, gene_map), codes, store.root(genome).offsets)


# ------------------------------------------------------------------------------------------ producers
def _check_inputs(schema: TraitSchema, gene_map: GeneTraitMap, schema_sha256: str, map_sha256: str) -> None:
    if schema_sha256 != schema.sha256 or map_sha256 != gene_map.sha256:
        raise ValueError("the supplied table hashes are not the hashes of the supplied D2 tables")


def create_derived_variant(
    genome: GenomeRecord,
    parent_variant: DerivedVariant | None,
    *,
    mutation_rng: np.random.Generator,
    origin_tick: int,
    parent_agent_id: int,
    daughter_agent_id: int,
    schema: TraitSchema,
    gene_trait_map: GeneTraitMap,
    trait_schema_sha256: str,
    gene_trait_map_sha256: str,
    store: VariantStore | None = None,
) -> tuple[DerivedVariant, GenomePhenotype] | None:
    """One daughter division: ``None`` iff K = 0, else a registered variant and its re-derived phenotype."""
    length = int(genome.sequence_length_bp)
    count = int(mutation_rng.poisson(length * SUBSTITUTIONS_PER_BASE))
    if count == 0:
        return None
    _check_inputs(schema, gene_trait_map, trait_schema_sha256, gene_trait_map_sha256)
    store = _store_or_default(store)
    if parent_variant is not None and parent_variant.root_genome_id != genome.genome_id:
        raise ValueError("parent variant belongs to a different accession genome")
    parent_codes = store.sequence(genome, parent_variant)

    positions = np.sort(mutation_rng.choice(length, size=count, replace=False))
    picks = mutation_rng.integers(0, 3, size=count)
    substitutions = []
    for position, pick in zip(positions.tolist(), picks.tolist()):
        ref = BASES[int(parent_codes[position])]
        substitutions.append(NucleotideSubstitution(position, ref, [b for b in BASES if b != ref][pick]))
    return variant_from_substitutions(
        genome, parent_variant, substitutions, origin_tick=origin_tick, parent_agent_id=parent_agent_id,
        daughter_agent_id=daughter_agent_id, schema=schema, gene_trait_map=gene_trait_map,
        trait_schema_sha256=trait_schema_sha256, gene_trait_map_sha256=gene_trait_map_sha256, store=store,
    )


def variant_from_substitutions(
    genome: GenomeRecord,
    parent_variant: DerivedVariant | None,
    substitutions: Sequence[NucleotideSubstitution],
    *,
    origin_tick: int,
    parent_agent_id: int,
    daughter_agent_id: int,
    schema: TraitSchema,
    gene_trait_map: GeneTraitMap,
    trait_schema_sha256: str,
    gene_trait_map_sha256: str,
    store: VariantStore | None = None,
    mutation_model_id: str = MUTATION_MODEL_ID,
) -> tuple[DerivedVariant, GenomePhenotype]:
    """Seal, verify and register the variant that applies ``substitutions`` to the inherited sequence.

    This is the deterministic half of :func:`create_derived_variant` (which only chooses the substitutions
    from the mutation RNG); tests and tools use it to construct a specific mutation.
    """
    _check_inputs(schema, gene_trait_map, trait_schema_sha256, gene_trait_map_sha256)
    store = _store_or_default(store)
    if parent_variant is not None and parent_variant.root_genome_id != genome.genome_id:
        raise ValueError("parent variant belongs to a different accession genome")
    substitutions = tuple(sorted(substitutions, key=lambda item: item.position_0based))
    if len({item.position_0based for item in substitutions}) != len(substitutions):
        raise ValueError("substitutions must not repeat a position")
    parent_codes = store.sequence(genome, parent_variant)
    derived = _apply(parent_codes, substitutions)

    phenotype = derive_genome_phenotype(genome, schema, gene_trait_map,
                                        knockouts=_knockouts(genome, gene_trait_map, store, derived))
    variant = _seal(
        root_accession_version=genome.accession, root_genome_id=genome.genome_id,
        root_sequence_sha256=genome.sequence_sha256, root_annotation_sha256=genome.annotation_sha256,
        parent_variant_id=0 if parent_variant is None else parent_variant.variant_id,
        parent_record_sha256=None if parent_variant is None else parent_variant.record_sha256,
        parent_sequence_sha256=store.root(genome).sha256 if parent_variant is None else parent_variant.derived_sequence_sha256,
        derived_sequence_sha256=_sha256(derived.tobytes()), sequence_length_bp=int(genome.sequence_length_bp),
        origin_tick=int(origin_tick), parent_agent_id=int(parent_agent_id), daughter_agent_id=int(daughter_agent_id),
        substitutions=substitutions, trait_schema_sha256=trait_schema_sha256,
        gene_trait_map_sha256=gene_trait_map_sha256, phenotype_sha256=phenotype_sha256(phenotype),
        mutation_model_id=mutation_model_id,
    )
    store.register(variant)
    return variant, dataclasses.replace(phenotype, variant_id=variant.variant_id)


def _mix(value: int) -> int:
    """splitmix64: a fixed, seedless 64-bit mixer, so a window derived from ids never depends on a random stream."""
    z = (value + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def transformation_window_start(recipient_agent_id: int, donor_agent_id: int, tick: int, length: int) -> int:
    return _mix(int(recipient_agent_id) ^ _mix(int(donor_agent_id) ^ _mix(int(tick)))) % int(length)


def create_transformant(
    genome: GenomeRecord,
    recipient: DerivedVariant | None,
    donor: DerivedVariant | None,
    *,
    window_start: int,
    window_length: int,
    origin_tick: int,
    recipient_agent_id: int,
    donor_agent_id: int,
    schema: TraitSchema,
    gene_trait_map: GeneTraitMap,
    trait_schema_sha256: str,
    gene_trait_map_sha256: str,
    store: VariantStore | None = None,
) -> tuple[DerivedVariant, GenomePhenotype] | None:
    """Natural transformation: the recipient's chromosome takes the donor's alleles over one tract.

    The tract is ``window_length`` bases from ``window_start`` on the circular chromosome. Wherever the donor's base
    differs from the recipient's, the recipient takes the donor's (a donor mutation is gained; a donor wild-type base
    repairs a recipient mutation). Returns ``None`` when the two sequences agree over the tract. Both cells must be
    the same species; the record has the recipient as the daughter and the donor as the parent agent.
    """
    for cell in (recipient, donor):
        if cell is not None and cell.root_genome_id != genome.genome_id:
            raise ValueError("transformation needs recipient and donor of the same accession genome")
    store = _store_or_default(store)
    mine, theirs = store.sequence(genome, recipient), store.sequence(genome, donor)
    length = mine.size
    span = min(int(window_length), length)
    positions = (int(window_start) + np.arange(span, dtype=np.int64)) % length
    differing = np.sort(positions[mine[positions] != theirs[positions]])
    if differing.size == 0:
        return None
    substitutions = [NucleotideSubstitution(int(p), BASES[int(mine[p])], BASES[int(theirs[p])]) for p in differing]
    return variant_from_substitutions(
        genome, recipient, substitutions, origin_tick=origin_tick, parent_agent_id=donor_agent_id,
        daughter_agent_id=recipient_agent_id, schema=schema, gene_trait_map=gene_trait_map,
        trait_schema_sha256=trait_schema_sha256, gene_trait_map_sha256=gene_trait_map_sha256, store=store,
        mutation_model_id=TRANSFORMATION_MODEL_ID,
    )


def derive_variant_phenotype(
    genome: GenomeRecord,
    variant: DerivedVariant,
    schema: TraitSchema,
    gene_trait_map: GeneTraitMap,
    *,
    store: VariantStore | None = None,
) -> GenomePhenotype:
    """Re-derive a recorded variant's phenotype and check it against the hash the record stored."""
    variant.verify()
    if variant.root_genome_id != genome.genome_id:
        raise ValueError("variant belongs to a different accession genome")
    _check_inputs(schema, gene_trait_map, variant.trait_schema_sha256, variant.gene_trait_map_sha256)
    store = _store_or_default(store)
    codes = store.sequence(genome, variant)
    phenotype = derive_genome_phenotype(genome, schema, gene_trait_map,
                                        knockouts=_knockouts(genome, gene_trait_map, store, codes))
    if phenotype_sha256(phenotype) != variant.phenotype_sha256:
        raise ValueError("re-derived phenotype does not match the phenotype_sha256 recorded in the variant")
    return dataclasses.replace(phenotype, variant_id=variant.variant_id)


def build_variant_phenotype_table(
    genomes: Sequence[GenomeRecord],
    variants: Sequence[DerivedVariant],
    schema: TraitSchema,
    gene_trait_map: GeneTraitMap,
    *,
    store: VariantStore | None = None,
) -> PhenotypeTable:
    """Pair-keyed D1 table: every accession genome (variant 0) plus every variant, sorted by the pair."""
    if not genomes:
        raise ValueError("at least one parsed genome is required")
    by_id = {genome.genome_id: genome for genome in genomes}
    if len(by_id) != len(genomes):
        raise ValueError("uint64 genome_id collision; accession strings remain distinct")
    rows = [derive_genome_phenotype(genome, schema, gene_trait_map) for genome in genomes]
    for variant in variants:
        if variant.root_genome_id not in by_id:
            raise ValueError("variant refers to a genome that is not in the catalogue")
        rows.append(derive_variant_phenotype(by_id[variant.root_genome_id], variant, schema, gene_trait_map, store=store))
    rows.sort(key=lambda row: (row.genome_id, row.variant_id))
    keys = [(row.genome_id, row.variant_id) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("phenotype (genome_id, variant_id) keys must be unique")

    def column(name: str) -> np.ndarray:
        return np.ascontiguousarray(np.asarray([getattr(row, name) for row in rows], dtype=np.float64))

    return PhenotypeTable(
        genome_id=np.ascontiguousarray(np.asarray([row.genome_id for row in rows], dtype=np.uint64)),
        variant_id=np.ascontiguousarray(np.asarray([row.variant_id for row in rows], dtype=np.uint64)),
        **{name: column(name) for name in _TRAITS},
    )


class VariantMutationProvider:
    """The object the engine calls once per daughter, in ascending parent agent_id order (D1/D4)."""

    def __init__(self, genomes: Sequence[GenomeRecord], schema: TraitSchema, gene_trait_map: GeneTraitMap,
                 *, project_root: str | Path | None = None, store: VariantStore | None = None) -> None:
        self.genomes = {int(genome.genome_id): genome for genome in genomes}
        self.schema = schema
        self.gene_trait_map = gene_trait_map
        self.store = store or VariantStore(project_root)
        self.store.reserve_genome_ids(self.genomes)

    def warm(self) -> None:
        """Load each root sequence and gene index now, so the first real mutation does not stall a frame."""
        for genome in self.genomes.values():
            self.store.root(genome)
            mapped_genes(genome, self.gene_trait_map)

    def create_transformant(self, genome_id, recipient_variant_id, donor_variant_id, *, origin_tick, recipient_agent_id,
                            donor_agent_id, window_bp, window_start=None):
        """Natural transformation between two cells of one genome; ``(DerivedVariant, GenomePhenotype)`` or ``None``."""
        genome = self.genomes[int(genome_id)]
        if window_start is None:
            window_start = transformation_window_start(recipient_agent_id, donor_agent_id, origin_tick, genome.sequence_length_bp)
        recipient = None if int(recipient_variant_id) == 0 else self.store.get(int(recipient_variant_id))
        donor = None if int(donor_variant_id) == 0 else self.store.get(int(donor_variant_id))
        return create_transformant(
            genome, recipient, donor, window_start=window_start, window_length=int(window_bp), origin_tick=int(origin_tick),
            recipient_agent_id=int(recipient_agent_id), donor_agent_id=int(donor_agent_id), schema=self.schema,
            gene_trait_map=self.gene_trait_map, trait_schema_sha256=self.schema.sha256,
            gene_trait_map_sha256=self.gene_trait_map.sha256, store=self.store,
        )

    def create_offspring_variant(self, genome_id, parent_variant_id, *, mutation_rng, origin_tick,
                                 parent_agent_id, daughter_agent_id):
        parent = None if int(parent_variant_id) == 0 else self.store.get(int(parent_variant_id))
        return create_derived_variant(
            self.genomes[int(genome_id)], parent, mutation_rng=mutation_rng, origin_tick=int(origin_tick),
            parent_agent_id=int(parent_agent_id), daughter_agent_id=int(daughter_agent_id),
            schema=self.schema, gene_trait_map=self.gene_trait_map,
            trait_schema_sha256=self.schema.sha256, gene_trait_map_sha256=self.gene_trait_map.sha256,
            store=self.store,
        )
