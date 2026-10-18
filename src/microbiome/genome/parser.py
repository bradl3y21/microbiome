"""Checksum-verifying parser for the project's NCBI Datasets source packages."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from .models import AnnotatedGene, GenomeRecord

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class _RawFeature:
    accession: str
    seqid: str
    feature_type: str
    start: int
    end: int
    strand: str
    attributes: dict[str, str]
    phase: str = "."


def derive_genome_id(accession: str) -> int:
    """Return the D2 uint64 ID: big-endian first 8 bytes of versioned SHA-256.

    The exact accession.version remains the canonical identity in every
    :class:`GenomeRecord` and manifest entry. Consumers must reject collisions
    rather than silently aliasing two assemblies.
    """
    if not accession or accession.strip() != accession:
        raise ValueError("accession.version must be a non-empty exact string")
    if "." not in accession:
        raise ValueError("genome accession must include its assembly version")
    digest = hashlib.sha256(accession.encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_artifact(project_root: Path, record: dict[str, object]) -> Path:
    relative = Path(str(record["path"]))
    if relative.is_absolute():
        raise ValueError(f"manifest path must be project-relative: {relative}")
    resolved = (project_root / relative).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes project root: {relative}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    expected_size = int(record["size_bytes"])
    if resolved.stat().st_size != expected_size:
        raise ValueError(f"manifest size mismatch for {relative}")
    expected_hash = str(record["sha256"]).lower()
    if _sha256(resolved) != expected_hash:
        raise ValueError(f"manifest SHA-256 mismatch for {relative}")
    return resolved


def _parse_attributes(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in text.split(";"):
        if not field:
            continue
        key, separator, value = field.partition("=")
        if separator:
            result[unquote(key)] = unquote(value)
    return result


def parse_fasta(path: str | Path) -> tuple[int, int, tuple[str, ...]]:
    """Parse FASTA headers and sequence lengths; return count, bp, headers.

    Sequence letters are streamed from the actual FASTA (not inferred from
    metadata). Empty records, data before the first header, and non-IUPAC
    nucleotide symbols are rejected.
    """
    headers: list[str] = []
    lengths: list[int] = []
    with Path(path).open("rt", encoding="ascii", newline=None) as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if headers and lengths[-1] == 0:
                    raise ValueError(f"empty FASTA record before line {line_number}")
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"empty FASTA header at line {line_number}")
                headers.append(header)
                lengths.append(0)
                continue
            if not headers:
                raise ValueError(f"FASTA sequence before first header at line {line_number}")
            sequence = line.upper()
            if any(base not in "ACGTUNRYKMSWBDHV.-" for base in sequence):
                raise ValueError(f"invalid FASTA nucleotide at line {line_number}")
            lengths[-1] += len(sequence)
    if not headers or any(length == 0 for length in lengths):
        raise ValueError("FASTA must contain at least one non-empty sequence record")
    return len(headers), sum(lengths), tuple(headers)


def _raw_gff3(path: Path, accession: str) -> tuple[_RawFeature, ...]:
    features: list[_RawFeature] = []
    version_seen = False
    with path.open("rt", encoding="utf-8", newline=None) as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("##gff-version"):
                version_seen = line == "##gff-version 3"
                continue
            if line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) != 9:
                raise ValueError(f"GFF3 line {line_number} has {len(columns)} columns, expected 9")
            seqid, _source, feature_type, start, end, _score, strand, phase, attributes = columns
            if feature_type not in {"gene", "CDS"}:
                continue
            start_i, end_i = int(start), int(end)
            if start_i < 1 or end_i < start_i or strand not in {"+", "-", ".", "?"}:
                raise ValueError(f"invalid GFF3 coordinates/strand at line {line_number}")
            features.append(
                _RawFeature(
                    accession=accession,
                    seqid=seqid,
                    feature_type=feature_type,
                    start=start_i,
                    end=end_i,
                    strand=strand,
                    attributes=_parse_attributes(attributes),
                    phase=phase,
                )
            )
    if not version_seen:
        raise ValueError("annotation is not an NCBI GFF3 version 3 file")
    if not features:
        raise ValueError("GFF3 contains no annotated gene or CDS features")
    return tuple(features)


def _group_rows_by_gene(raw_features: tuple[_RawFeature, ...]) -> list[list[_RawFeature]]:
    """Group rows into distinct genes using source GFF3 identity, never locus_tag alone.

    Each ``gene`` row (or set of gene rows sharing one GFF3 ``ID``) is one gene.
    A CDS joins the gene named by its ``Parent``; failing that, the single gene
    carrying its ``locus_tag``; failing that, CDS rows sharing a GFF3 ``ID``
    (multipart features) form one gene. Anything else stands alone.
    """
    groups: list[list[_RawFeature]] = []
    gene_by_id: dict[str, int] = {}
    genes_by_tag: dict[str, set[int]] = defaultdict(set)
    for feature in raw_features:
        if feature.feature_type != "gene":
            continue
        feature_id = feature.attributes.get("ID")
        index = gene_by_id.get(feature_id) if feature_id else None
        if index is None:
            index = len(groups)
            groups.append([])
            if feature_id:
                gene_by_id[feature_id] = index
        groups[index].append(feature)
        if feature.attributes.get("locus_tag"):
            genes_by_tag[feature.attributes["locus_tag"]].add(index)

    cds_by_id: dict[str, int] = {}
    for feature in raw_features:
        if feature.feature_type != "CDS":
            continue
        parents = feature.attributes.get("Parent", "").split(",")
        parent_genes = sorted(gene_by_id[parent] for parent in parents if parent in gene_by_id)
        tag_genes = genes_by_tag.get(feature.attributes.get("locus_tag", ""), set())
        feature_id = feature.attributes.get("ID")
        if parent_genes:
            index = parent_genes[0]
        elif len(tag_genes) == 1:
            index = next(iter(tag_genes))
        elif feature_id and feature_id in cds_by_id:
            index = cds_by_id[feature_id]
        else:
            index = len(groups)
            groups.append([])
            if feature_id:
                cds_by_id[feature_id] = index
        groups[index].append(feature)
    return groups


def parse_gff3_parts(path: str | Path, accession: str) -> list[tuple[str, _RawFeature]]:
    """Every GFF3 gene/CDS feature part with the stable ``gene_key`` its gene was given.

    Multipart features share a key only when GFF3 identity puts them in one gene; this is the row view
    D2's ``gene_features.tsv`` is built from, and :func:`parse_gff3` is the per-gene view of the same groups.
    """
    genes = _normalized_groups(_raw_gff3(Path(path), accession), accession)
    return [(key, row) for key, rows, _merged in genes for row in rows]


def parse_gff3(path: str | Path, accession: str) -> tuple[AnnotatedGene, ...]:
    """Normalize annotated gene/CDS rows, coalescing the rows of each distinct gene.

    No sequence-motif scans or product-name fuzzy matching are performed.
    Rows are coalesced only when GFF3 identity (``ID``/``Parent``) shows they
    belong to one gene. The key is ``accession|locus_tag:<tag>`` only when that
    ``locus_tag`` belongs to exactly one gene; otherwise the exact-coordinate
    fallback key is used, so distinct genes are never fused by a shared tag.
    """
    raw_features = _raw_gff3(Path(path), accession)
    return _genes_from_groups(_normalized_groups(raw_features, accession), accession)


def _normalized_groups(
    raw_features: tuple[_RawFeature, ...], accession: str
) -> list[tuple[str, list[_RawFeature], dict[str, str]]]:
    """Group rows into genes and assign each gene its stable key; shared by the gene and row views."""
    grouped_rows = [rows for rows in _group_rows_by_gene(raw_features) if rows]
    merged_by_group: list[dict[str, str]] = []
    for rows in grouped_rows:
        rows.sort(key=lambda row: (row.seqid, row.start, row.end, row.strand, row.feature_type))
        merged: dict[str, str] = {}
        for row in rows:
            for name, value in sorted(row.attributes.items()):
                if value and (name not in merged or name in {"product", "protein_id", "Dbxref"}):
                    merged[name] = value
        merged_by_group.append(merged)
    tag_counts: dict[str, int] = defaultdict(int)
    for merged in merged_by_group:
        if merged.get("locus_tag"):
            tag_counts[merged["locus_tag"]] += 1

    groups: list[tuple[str, list[_RawFeature], dict[str, str]]] = []
    for rows, merged in zip(grouped_rows, merged_by_group):
        canonical = _canonical_row(rows)
        locus_tag = merged.get("locus_tag")
        if locus_tag and tag_counts[locus_tag] == 1:
            key = f"{accession}|locus_tag:{locus_tag}"
        else:
            key = (
                f"{accession}|{canonical.seqid}:{canonical.start}-{canonical.end}:"
                f"{canonical.strand}:{canonical.feature_type}"
            )
        groups.append((key, rows, merged))
    return groups


def _canonical_row(rows: list[_RawFeature]) -> _RawFeature:
    return next((row for row in rows if row.feature_type == "gene"), rows[0])


def _genes_from_groups(
    groups: list[tuple[str, list[_RawFeature], dict[str, str]]], accession: str
) -> tuple[AnnotatedGene, ...]:
    genes: list[AnnotatedGene] = []
    for key, rows, merged in groups:
        canonical = _canonical_row(rows)
        locus_tag = merged.get("locus_tag")
        pseudogene = (
            merged.get("pseudo", "false").lower() in {"true", "1", "yes"}
            or merged.get("gene_biotype", "").lower() == "pseudogene"
        )
        genes.append(
            AnnotatedGene(
                accession=accession,
                gene_key=key,
                locus_tag=locus_tag,
                gene_symbol=merged.get("gene") or merged.get("Name"),
                product=merged.get("product"),
                seqid=canonical.seqid,
                start=min(row.start for row in rows),
                end=max(row.end for row in rows),
                strand=canonical.strand,
                feature_type=canonical.feature_type,
                source_feature_types=tuple(sorted({row.feature_type for row in rows})),
                source_ids=tuple(sorted({row.attributes["ID"] for row in rows if row.attributes.get("ID")})),
                parent_ids=tuple(sorted({parent for row in rows for parent in row.attributes.get("Parent", "").split(",") if parent})),
                pseudogene=pseudogene,
                attributes=merged,
                cds_parts=tuple(sorted(
                    (row.start, row.end, int(row.phase) if row.phase.isdigit() else 0,
                     row.attributes.get("ID", ""))
                    for row in rows if row.feature_type == "CDS"
                )),
            )
        )
    genes.sort(key=lambda gene: (gene.seqid, gene.start, gene.end, gene.strand, gene.feature_type, gene.gene_key))
    keys = [gene.gene_key for gene in genes]
    if len(keys) != len(set(keys)):
        raise ValueError("normalized annotation contains duplicate stable gene keys")
    return tuple(genes)


def load_genome(accession: str, project_root: str | Path | None = None) -> GenomeRecord:
    """Load one manifest-listed genome and verify FASTA/GFF3 bytes before use."""
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    manifest_path = root / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [entry for entry in manifest.get("genomes", []) if entry.get("accession") == accession]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest entry for {accession}; found {len(matches)}")
    record = matches[0]
    if record.get("assembly_level") != "Complete Genome":
        raise ValueError(f"{accession} is not recorded as a complete genome")
    sequence_record = record["sequence_file"]
    annotation_record = record["annotation_file"]
    sequence_path = _resolve_artifact(root, sequence_record)
    annotation_path = _resolve_artifact(root, annotation_record)
    count, sequence_length, headers = parse_fasta(sequence_path)
    if count != int(record["sequence_count"]):
        raise ValueError(f"FASTA record-count mismatch for {accession}: {count} != {record['sequence_count']}")
    expected_length = int(record["sequence_length_bp"])
    if sequence_length != expected_length:
        raise ValueError(f"FASTA length mismatch for {accession}: {sequence_length} != {expected_length}")
    genes = parse_gff3(annotation_path, accession)
    sequence_ids = {header.split(maxsplit=1)[0] for header in headers}
    annotation_ids = {gene.seqid for gene in genes}
    if annotation_ids - sequence_ids:
        raise ValueError(
            f"GFF3 sequence IDs do not all occur in the genome FASTA for {accession}: "
            f"{sorted(annotation_ids - sequence_ids)}"
        )
    return GenomeRecord(
        accession=accession,
        genome_id=derive_genome_id(accession),
        organism=str(record["organism"]),
        tax_id=int(record["tax_id"]),
        assembly_level=str(record["assembly_level"]),
        sequence_path=str(sequence_record["path"]),
        sequence_length_bp=sequence_length,
        sequence_record_count=count,
        sequence_sha256=str(sequence_record["sha256"]),
        annotation_path=str(annotation_record["path"]),
        annotation_sha256=str(annotation_record["sha256"]),
        genes=genes,
    )
