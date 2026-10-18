"""Per-organism calibration of the realism layer against published measurements.

The D2 gene-to-trait table says in its own notes that its numeric baselines are not measured rates. In the realism layer
(and only there: the plain D1 world keeps the D2 numbers) each organism's sugar uptake, biomass yield, storage, division
mass and swimming speed are multiplied by the factors in ``data/calibration_fit.json``, and acetate overflow and the Pirt
maintenance coefficient are set from it. Factors are multiplicative so gene-derived differences (a knocked-out transporter,
a variant) still show through.

The file is produced by ``python -m microbiome.calibration fit`` from ``data/calibration_targets.json``; nothing in it is
typed in by hand. ``IDENTITY`` (all factors 1, no maintenance) is what an organism without an entry gets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping

from .parser import _PROJECT_ROOT

CALIBRATION_FILE = Path("data") / "calibration_fit.json"


@dataclass(frozen=True)
class OrganismCalibration:
    uptake_scale: float = 1.0                 # multiplies the D2 glucose and lactose uptake per reference-mass cell
    biomass_scale: float = 1.0                # multiplies the D2 biomass fraction (mass yield) of the sugars
    storage_scale: float = 1.0                # multiplies the D2 storage fraction of the sugars
    overflow_scale: float = 1.0               # multiplies the gene-derived acetate overflow capacity
    division_mass_scale: float = 1.0          # multiplies the D2 division mass
    motility_scale: float = 1.0               # multiplies the D2 swimming coefficient
    maintenance_mmol_glc_gdw_h: float = 0.0   # Pirt maintenance, glucose equivalents per gram dry mass per hour
    run_time_s: float = 0.0                   # mean run before a reorientation; 0 = the shared (E. coli) tumble rate
    turn_mean_deg: float = 0.0                # mean reorientation angle; 0 = the shared (E. coli) distribution
    turn_sd_deg: float = 0.0                  # its spread

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not value >= 0.0 or value != value:
                raise ValueError(f"calibration {item.name} must be a finite non-negative number, got {value!r}")


IDENTITY = OrganismCalibration()


@dataclass(frozen=True)
class Calibration:
    organisms: Mapping[str, OrganismCalibration] = field(default_factory=dict)
    shared: Mapping[str, float] = field(default_factory=dict)
    targets_sha256: str = ""

    def for_accession(self, accession: str) -> OrganismCalibration:
        return self.organisms.get(accession, IDENTITY)


def load_calibration(project_root: str | Path | None = None) -> Calibration:
    """The committed fit, or an identity calibration if the file does not exist."""
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    path = root / CALIBRATION_FILE
    if not path.is_file():
        return Calibration()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported calibration_fit.json schema_version")
    known = {item.name for item in fields(OrganismCalibration)}
    organisms = {}
    for accession, values in data["organisms"].items():
        organisms[accession] = OrganismCalibration(**{k: float(v) for k, v in values.items() if k in known})
    return Calibration(organisms, {k: float(v) for k, v in data.get("shared", {}).items()}, data.get("targets_sha256", ""))
