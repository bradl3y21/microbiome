"""Offline tests for genome fetching helpers and validation."""

import hashlib
import json

import pytest

from microbiome.genome import fetch


def test_sha256_matches_hashlib_for_a_small_file(tmp_path):
    contents = b"small genome fixture\n"
    path = tmp_path / "genome.fna"
    path.write_bytes(contents)

    assert fetch._sha256(path) == hashlib.sha256(contents).hexdigest()


def test_artifact_records_relative_path_role_size_and_sha256(tmp_path):
    contents = b">sequence\nACGT\n"
    path = tmp_path / "data" / "genomes" / "sample.fna"
    path.parent.mkdir(parents=True)
    path.write_bytes(contents)

    artifact = fetch._artifact(tmp_path, path, "genome_sequence_fasta")

    assert set(artifact) == {"path", "role", "size_bytes", "sha256"}
    assert artifact == {
        "path": "data/genomes/sample.fna",
        "role": "genome_sequence_fasta",
        "size_bytes": len(contents),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def test_fetch_rejects_accession_already_in_manifest_without_network(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    manifest = {
        "genomes": [{
            "accession": "GCF_X.1",
            "license_or_terms": "terms",
            "terms_url": "https://example.test/terms",
            "additional_terms_reference": "reference",
        }]
    }
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(fetch, "_get", lambda *_args, **_kwargs: pytest.fail("unexpected network access"))

    with pytest.raises(ValueError, match="already in the manifest"):
        fetch.fetch_genome("GCF_X.1", tmp_path)


def test_fetch_rejects_non_complete_assembly_level(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    manifest = {
        "genomes": [{
            "accession": "GCF_TEMPLATE.1",
            "license_or_terms": "terms",
            "terms_url": "https://example.test/terms",
            "additional_terms_reference": "reference",
        }]
    }
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = {"reports": [{"assembly_info": {"assembly_level": "Contig"}}]}
    monkeypatch.setattr(fetch, "_get", lambda *_args, **_kwargs: json.dumps(report).encode("utf-8"))

    with pytest.raises(ValueError, match="Contig"):
        fetch.fetch_genome("GCF_NEW.1", tmp_path)
