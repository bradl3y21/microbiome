"""``python -m microbiome.calibration fit`` solves the parameters and writes ``data/calibration_fit.json``;
``... check`` evaluates the committed fit against every target without changing anything."""

from __future__ import annotations

import argparse
import json
import sys

from ..genome.calibration import load_calibration
from .rig import default_config
from . import fit as F


def _print(rows) -> None:
    width = max(len(r["id"]) for r in rows)
    for r in rows:
        measured = "      -" if r["measured"] is None else f"{r['measured']:10.4g}"
        target = "-" if r["target"] is None else f"{r['target']:.4g}"
        mark = {True: "ok", False: "FAIL", None: "--"}[r["ok"]]
        print(f"{r['id']:{width}s}  {r['role']:9s} target {target:>9s}  simulated {measured}  {mark:4s} {r['note']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m microbiome.calibration")
    parser.add_argument("command", choices=["fit", "check", "cycle", "report"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "fit":
        calibration, config = F.fit(verbose=not args.quiet)
        rows = F.evaluate(calibration, config)
        path = F.write_fit(calibration, rows)
        _print(rows)
        print("wrote", path)
        return 0
    if args.command == "cycle":
        F.cycle()
        return 0
    if args.command == "report":
        from . import report

        print("wrote", report.write())
        return 0
    calibration = load_calibration()
    rows = F.evaluate(calibration, default_config())
    _print(rows)
    return 1 if any(r["ok"] is False for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
