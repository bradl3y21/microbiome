"""Loading and comparison behaviour for published calibration targets."""

import hashlib
import json

import pytest

from microbiome.calibration.targets import Target, load_targets


def _target_data(target_id, role="fit", **values):
    return {
        "id": target_id,
        "role": role,
        "quantity": "growth rate",
        "unit": "1/h",
        "value": 1.0,
        **values,
    }


def _write_project(root, targets, schema_version=1):
    data_dir = root / "data"
    data_dir.mkdir()
    payload = {
        "schema_version": schema_version,
        "sources": {"example": {"citation": "Example", "read_via": "test"}},
        "targets": targets,
    }
    path = data_dir / "calibration_targets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_target(value, sd=None, range_=None):
    return Target(
        id="example",
        role="fit",
        organism=None,
        quantity="growth rate",
        unit="1/h",
        value=value,
        sd=sd,
        range=range_,
        raw={},
    )


def test_loads_targets_and_exposes_lookup_roles_and_file_hash(tmp_path):
    path = _write_project(
        tmp_path,
        [
            _target_data("fit-target", "fit", value=2.5, sd=0.1),
            _target_data("held-out-target", "held_out", value=3.0),
        ],
    )

    targets = load_targets(tmp_path)

    assert targets["fit-target"].value == 2.5
    assert targets["fit-target"].sd == 0.1
    assert targets.with_role("fit") == (targets["fit-target"],)
    with pytest.raises(KeyError):
        targets["missing"]
    assert targets.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_unknown_role_raises_value_error(tmp_path):
    _write_project(tmp_path, [_target_data("unknown-role", "mystery")])

    with pytest.raises(ValueError, match="unknown role"):
        load_targets(tmp_path)


def test_duplicate_target_ids_raise_value_error(tmp_path):
    _write_project(
        tmp_path,
        [_target_data("duplicate"), _target_data("duplicate", "held_out")],
    )

    with pytest.raises(ValueError, match="duplicate target ids"):
        load_targets(tmp_path)


def test_wrong_schema_version_raises_value_error(tmp_path):
    _write_project(tmp_path, [_target_data("target")], schema_version=2)

    with pytest.raises(ValueError, match="schema_version"):
        load_targets(tmp_path)


def test_contains_uses_two_standard_deviations(tmp_path):
    target = _make_target(value=10.0, sd=1.0)

    assert target.contains(12.0)
    assert not target.contains(12.01)


def test_contains_uses_range_when_there_is_no_sd():
    target = _make_target(value=None, range_=(2.0, 5.0))

    assert target.contains(3.0)
    assert not target.contains(5.01)


def test_contains_without_sd_or_range_only_accepts_exact_value():
    target = _make_target(value=4.0)

    assert target.contains(4.0)
    assert not target.contains(4.0001)


def test_tolerance_uses_sd_then_half_range_width_then_zero():
    assert _make_target(value=4.0, sd=0.3, range_=(1.0, 9.0)).tolerance == 0.3
    assert _make_target(value=None, range_=(2.0, 8.0)).tolerance == 3.0
    assert _make_target(value=4.0).tolerance == 0.0
