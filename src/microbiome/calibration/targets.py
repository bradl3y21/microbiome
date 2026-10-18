"""The published targets the realism layer is calibrated and validated against (``data/calibration_targets.json``)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..genome.parser import _PROJECT_ROOT

TARGETS_FILE = Path("data") / "calibration_targets.json"
ROLES = ("fit", "held_out", "range", "assumed", "reference")


@dataclass(frozen=True)
class Target:
    id: str
    role: str
    organism: str | None
    quantity: str
    unit: str
    value: float | None
    sd: float | None
    range: tuple[float, float] | None
    raw: dict[str, Any]

    @property
    def tolerance(self) -> float:
        """Absolute tolerance used when comparing a simulated value with this target: the reported spread if any."""
        if self.sd is not None:
            return float(self.sd)
        if self.range is not None:
            return 0.5 * (self.range[1] - self.range[0])
        return 0.0

    def contains(self, measured: float) -> bool:
        if self.range is not None and self.sd is None:
            return self.range[0] <= measured <= self.range[1]
        return self.value is not None and abs(measured - self.value) <= 2.0 * self.tolerance


@dataclass(frozen=True)
class TargetSet:
    targets: tuple[Target, ...]
    sources: dict[str, dict[str, Any]]
    sha256: str

    def __getitem__(self, target_id: str) -> Target:
        for target in self.targets:
            if target.id == target_id:
                return target
        raise KeyError(target_id)

    def with_role(self, role: str) -> tuple[Target, ...]:
        return tuple(t for t in self.targets if t.role == role)


def load_targets(project_root: str | Path | None = None) -> TargetSet:
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    raw_bytes = (root / TARGETS_FILE).read_bytes()
    data = json.loads(raw_bytes.decode("utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported calibration_targets.json schema_version")
    targets = []
    for item in data["targets"]:
        if item["role"] not in ROLES:
            raise ValueError(f"target {item['id']}: unknown role {item['role']!r}")
        rng = item.get("range")
        targets.append(Target(
            id=item["id"], role=item["role"], organism=item.get("organism"), quantity=item["quantity"], unit=item["unit"],
            value=None if item.get("value") is None else float(item["value"]), sd=None if item.get("sd") is None else float(item["sd"]),
            range=None if rng is None else (float(rng[0]), float(rng[1])), raw=item))
    ids = [t.id for t in targets]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate target ids")
    return TargetSet(tuple(targets), data["sources"], hashlib.sha256(raw_bytes).hexdigest())
