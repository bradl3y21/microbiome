"""D2 producer ``load_genome_bundle``: verify one immutable NCBI Datasets package, then parse it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import GenomeRecord
from .parser import _resolve_artifact, load_genome, parse_fasta

REQUIRED_ROLES = frozenset({
    "genome_sequence_fasta",
    "genome_annotation_gff3",
    "assembly_metadata_jsonl",
    "sequence_metadata_jsonl",
    "ncbi_dataset_catalog",
    "ncbi_md5_checksum_list",
    "ncbi_package_readme",
    "original_ncbi_datasets_zip",
})


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_genome_bundle(path: str | Path) -> GenomeRecord:
    """Load ``<root>/data/genomes/<accession.version>/`` after verifying the whole package.

    Beyond the FASTA/GFF3 checks of :func:`load_genome`, every artifact the manifest retains for the assembly
    must exist inside the bundle with its pinned size, role and SHA-256; NCBI's own ``md5sum.txt`` must agree
    with the files on disk; and the package's assembly and sequence reports must name this accession and the
    FASTA's actual sequence. Nothing is rewritten; any mismatch raises ``ValueError``.
    """
    bundle = Path(path).resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(bundle)
    if bundle.parent.name != "genomes" or bundle.parent.parent.name != "data":
        raise ValueError("a genome bundle must live at <project>/data/genomes/<accession.version>/")
    accession = bundle.name
    root = bundle.parent.parent.parent

    manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest.get("genomes", []) if entry.get("accession") == accession]
    if len(entries) != 1:
        raise ValueError(f"expected exactly one manifest entry for {accession}; found {len(entries)}")
    entry = entries[0]

    artifacts = entry.get("artifacts", [])
    roles = {artifact["role"] for artifact in artifacts}
    if REQUIRED_ROLES - roles:
        raise ValueError(f"bundle {accession} is missing retained artifacts: {sorted(REQUIRED_ROLES - roles)}")
    by_role: dict[str, Path] = {}
    for artifact in artifacts:
        resolved = _resolve_artifact(root, artifact)
        try:
            resolved.relative_to(bundle)
        except ValueError as exc:
            raise ValueError(f"artifact {artifact['path']} is outside bundle {accession}") from exc
        by_role[artifact["role"]] = resolved
    for key, role in (("sequence_file", "genome_sequence_fasta"), ("annotation_file", "genome_annotation_gff3")):
        if entry[key]["sha256"] != next(a["sha256"] for a in artifacts if a["role"] == role):
            raise ValueError(f"manifest {key} disagrees with its retained {role} artifact")

    listed = 0
    for line in by_role["ncbi_md5_checksum_list"].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, _, relative = line.partition("  ")
        target = bundle / relative.strip()
        if not target.is_file():
            raise ValueError(f"md5sum.txt lists a file that is not in the bundle: {relative.strip()}")
        if _md5(target) != expected.lower():
            raise ValueError(f"md5sum.txt disagrees with the bytes of {relative.strip()}")
        listed += 1
    if not listed:
        raise ValueError("md5sum.txt lists no files")

    reports = _jsonl(by_role["assembly_metadata_jsonl"])
    if len(reports) != 1 or reports[0].get("accession") != accession:
        raise ValueError(f"assembly report does not describe exactly {accession}")
    if reports[0].get("organism", {}).get("organismName") != entry["organism"]:
        raise ValueError("assembly report organism disagrees with the manifest")

    sequences = _jsonl(by_role["sequence_metadata_jsonl"])
    count, length, headers = parse_fasta(by_role["genome_sequence_fasta"])
    if len(sequences) != count or sum(int(item["length"]) for item in sequences) != length:
        raise ValueError("sequence report disagrees with the FASTA record count or length")
    reported = {item["refseqAccession"] for item in sequences}
    if {header.split(maxsplit=1)[0] for header in headers} != reported:
        raise ValueError("sequence report accessions disagree with the FASTA headers")

    return load_genome(accession, root)
