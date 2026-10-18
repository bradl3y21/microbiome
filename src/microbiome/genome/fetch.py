"""Add a real genome to the manifest: download it from NCBI Datasets, hash every file, write the manifest entry.

``python -m microbiome.genome.fetch GCF_000006765.1`` downloads the RefSeq assembly package (FASTA, GFF3, sequence
report), extracts it under ``data/genomes/<accession>/``, and appends an entry in exactly the format of the existing ones
(sizes and SHA-256 of every artifact, source URL, fetch time, NCBI's terms notice). Only assemblies NCBI records as
``Complete Genome`` are accepted, which is the project's data policy (see ``load_genome``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .parser import _PROJECT_ROOT

API = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{accession}"
DOWNLOAD = (API + "/download?include_annotation_type=GENOME_FASTA&include_annotation_type=GENOME_GFF"
            "&include_annotation_type=SEQUENCE_REPORT")
ROLES = {"md5sum.txt": "ncbi_md5_checksum_list", "assembly_data_report.jsonl": "assembly_metadata_jsonl",
         "dataset_catalog.json": "ncbi_dataset_catalog", "genomic.gff": "genome_annotation_gff3",
         "sequence_report.jsonl": "sequence_metadata_jsonl", "ncbi_datasets.zip": "original_ncbi_datasets_zip",
         "README.md": "ncbi_package_readme"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(url: str, timeout: int = 300) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "microbiome-genome-fetch/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _artifact(root: Path, path: Path, role: str) -> dict[str, object]:
    return {"path": path.relative_to(root).as_posix(), "role": role, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def fetch_genome(accession: str, project_root: str | Path | None = None, *, replace: bool = False) -> dict[str, object]:
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    manifest_path = root / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(entry["accession"] == accession for entry in manifest["genomes"]) and not replace:
        raise ValueError(f"{accession} is already in the manifest")
    template = manifest["genomes"][0]
    report = json.loads(_get(API.format(accession=accession) + "/dataset_report"))["reports"][0]
    level = report["assembly_info"]["assembly_level"]
    if level != "Complete Genome":
        raise ValueError(f"{accession} is a {level!r} assembly; only Complete Genome assemblies are accepted")

    target = root / "data" / "genomes" / accession
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    zip_path = target / "ncbi_datasets.zip"
    zip_path.write_bytes(_get(DOWNLOAD.format(accession=accession)))
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    data_dir = target / "ncbi_dataset" / "data"
    sequence = next((data_dir / accession).glob("*_genomic.fna"))
    annotation = data_dir / accession / "genomic.gff"
    artifacts = []
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        role = "genome_sequence_fasta" if path == sequence else ROLES.get(path.name)
        if role is None:
            raise ValueError(f"unexpected file in the NCBI package: {path}")
        artifacts.append(_artifact(root, path, role))
    now = datetime.now(timezone.utc)
    by_role = {a["role"]: a for a in artifacts}
    count = sum(1 for line in sequence.open("rt", encoding="ascii") if line.startswith(">"))
    stats = report["assembly_stats"]
    entry = {
        "accession": accession,
        "organism": report["organism"]["organism_name"],
        "tax_id": int(report["organism"]["tax_id"]),
        "assembly_name": report["assembly_info"]["assembly_name"],
        "assembly_level": level,
        "sequence_count": count,
        "sequence_length_bp": int(stats["total_sequence_length"]),
        "source_url": DOWNLOAD.format(accession=accession),
        "assembly_record_url": f"https://www.ncbi.nlm.nih.gov/assembly/{accession}/",
        "fetched_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetch_date_utc": now.strftime("%Y-%m-%d"),
        "size_bytes": by_role["genome_sequence_fasta"]["size_bytes"],
        "sha256": by_role["genome_sequence_fasta"]["sha256"],
        "sequence_file": {k: by_role["genome_sequence_fasta"][k] for k in ("path", "role", "size_bytes", "sha256")},
        "annotation_file": {k: by_role["genome_annotation_gff3"][k] for k in ("path", "role", "size_bytes", "sha256")},
        "source_package": {k: by_role["original_ncbi_datasets_zip"][k] for k in ("path", "role", "size_bytes", "sha256")},
        "license_or_terms": template["license_or_terms"],
        "terms_url": template["terms_url"],
        "additional_terms_reference": template["additional_terms_reference"],
        "artifacts": [a for a in artifacts if a["role"] not in ("genome_sequence_fasta", "genome_annotation_gff3")],
    }
    entry["artifacts"] = sorted(artifacts, key=lambda a: a["path"])
    # the existing entries list every file including the sequence and annotation; keep the same convention
    manifest["genomes"] = sorted([e for e in manifest["genomes"] if e["accession"] != accession] + [entry], key=lambda e: e["accession"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return entry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m microbiome.genome.fetch")
    parser.add_argument("accessions", nargs="+")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    for accession in args.accessions:
        entry = fetch_genome(accession, replace=args.replace)
        print(f"{accession}: {entry['organism']}, {entry['sequence_count']} sequence(s), {entry['sequence_length_bp']:,} bp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
