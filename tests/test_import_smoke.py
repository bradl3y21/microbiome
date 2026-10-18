"""Import smoke test — the package must import in ANY order (Assay/QA).

Regression (2026-10-17): after the D3 controller landed, `sim/controller.py`
imported `microbiome.genome.phenotype` at module scope while `sim/__init__.py`
eagerly imported `.controller`. Importing `microbiome.genome` first therefore
raised:

    ImportError: cannot import name '_table' from partially initialized module
    'microbiome.genome.phenotype' (most likely due to a circular import)

This made the whole suite fail at collection, so it is a build-breaking defect,
not a nuance. These tests fail until the cycle is broken.

Run just this file:
    .venv\\Scripts\\python -m pytest tests/test_import_smoke.py -q
"""

from __future__ import annotations

import importlib
import subprocess
import sys


def test_genome_package_imports_in_a_fresh_interpreter():
    """`import microbiome.genome` must succeed on its own (exit 0)."""
    proc = subprocess.run(
        [sys.executable, "-c", "import microbiome.genome"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "importing microbiome.genome raised a circular-import error:\n" + proc.stderr
    )


def test_genome_then_sim_then_core_import_in_one_process():
    """The order that exposes the cycle: genome, then sim, then core DTOs."""
    for module in ("microbiome.genome", "microbiome.sim", "microbiome.core.ui_api"):
        importlib.import_module(module)
