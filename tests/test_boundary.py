"""Item 16 — the D3 boundary holds, mechanically.

Two claims are proven here:

1. Rendering never mutates simulation state. A world is built, snapshotted, and a
   full render frame is run through ``WorldView``/``PygameRenderer``; afterwards
   every byte of the session state (D1 arrays, environment field, RNG streams,
   command log/results, phenotype catalogue) must be identical, and the snapshot
   must still be detached and read-only.

2. ``dispatch_command`` is the UI's only mutation path. A static scan of every
   module under ``src/microbiome/ui`` proves that no module other than
   ``commands.py`` calls a mutating facade method, and that the dispatcher covers
   every mutating method the D3 protocol exposes.

The state digest reuses the canonical harness (``sim_support.collect`` +
``conftest.digest_arrays``) rather than inventing a weaker one, so a silently
skipped field would fail here as it does in the determinism tests.

Run just this file:
    .venv\\Scripts\\python -m pytest tests/test_boundary.py -q
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, digest_arrays
from sim_support import collect

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# The D3 protocol's mutating facade methods (core/ui_api.SimulationController).
# Everything else on the facade (list_genomes, preview_phenotype, command_result,
# inspect_cell, inspect_genome, render_snapshot) is read-only by contract.
MUTATING_FACADE_METHODS = frozenset(
    {"place_nutrient", "introduce_organism", "set_paused", "advance_ticks", "single_step"}
)


# --------------------------------------------------------------------------- helpers
def _digest_session(controller) -> str:
    """sha256 over every mutable byte reachable from the session, including RNGs.

    Reading the private session fields is deliberate: this test's whole point is to
    pin the mutable state that the render path must not touch, so if those fields
    are renamed the boundary test should fail loudly rather than quietly weaken.
    """
    parts: dict[str, np.ndarray] = {}
    for label, obj in (
        ("state", controller._state),
        ("environment", controller._environment),
        ("rngs", controller._rngs),
        ("phenotypes", controller._phenotypes),
        ("command_log", controller._command_log),
        ("results", controller._results),
    ):
        parts.update(collect(obj, label))
    parts["paused"] = np.asarray(controller._paused, dtype=np.bool_)
    parts["next_command_id"] = np.asarray(controller._next_command_id, dtype=np.int64)
    return digest_arrays(parts)


def _first_demo_nutrient_id() -> int:
    table = json.loads((ROOT / "data" / "gene_trait_table.json").read_text(encoding="utf-8"))
    channels = sorted(table["demo_nutrient_channels"], key=lambda row: int(row["nutrient_id"]))
    return int(channels[0]["nutrient_id"])


def _build_world():
    """A real, non-trivial session: nutrient added, two organisms introduced, ticks run."""
    from microbiome.sim.controller import SimulationController

    controller = SimulationController.create_demo(seed=2026)
    genomes = controller.list_genomes()
    assert genomes, "the shipped catalogue must offer at least one genome"
    nutrient_id = _first_demo_nutrient_id()

    controller.place_nutrient(nutrient_id, (200.0, 300.0), 1.0)
    controller.introduce_organism(genomes[0].genome_id, (100.0, 100.0))
    if len(genomes) > 1:
        controller.introduce_organism(genomes[1].genome_id, (300.0, 200.0))
    controller.advance_ticks(30)
    return controller


@pytest.fixture
def pygame_screen():
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((960, 640))
    yield screen
    pygame.quit()


def _ui_module_paths() -> list[Path]:
    import microbiome.ui as ui_package

    return sorted(Path(ui_package.__file__).parent.glob("*.py"))


def _mutating_call_sites(tree: ast.AST) -> list[tuple[str, str]]:
    """(enclosing function name, facade method) for every mutating facade call."""
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    sites: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in MUTATING_FACADE_METHODS:
            continue
        enclosing = "<module>"
        cursor: ast.AST = node
        while cursor in parents:
            cursor = parents[cursor]
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                enclosing = cursor.name
                break
        sites.append((enclosing, func.attr))
    return sites


# ----------------------------------------------------------------------- the tests
def test_render_never_mutates_sim(pygame_screen):
    """A full render frame through WorldView leaves the simulation byte-identical."""
    from microbiome.core.ui_api import RenderSnapshot
    from microbiome.render.renderer import PygameRenderer
    from microbiome.render.view import WorldView

    controller = _build_world()
    snapshot = controller.render_snapshot()
    assert isinstance(snapshot, RenderSnapshot)

    # The boundary is only real if the snapshot is detached and unwritable.
    sim = controller._state.agents
    assert not np.shares_memory(snapshot.position_um, sim.position_um), "snapshot shares sim memory"
    assert not np.shares_memory(snapshot.radius_um, sim.radius_um), "snapshot shares sim memory"
    assert not np.shares_memory(snapshot.genome_id, sim.genome_id), "snapshot shares sim memory"
    for name in (
        "agent_id",
        "position_um",
        "radius_um",
        "genome_id",
        "nutrient_id",
        "concentration_mmol_l",
        "phenotype_genome_id",
        "phenotype_motility_um_s",
        "phenotype_max_uptake_mmol_s",
    ):
        array = getattr(snapshot, name)
        assert not array.flags.writeable, f"snapshot.{name} is writable"
        with pytest.raises((ValueError, TypeError)):
            array.flat[0] = array.flat[0]

    snapshot_bytes_before = {
        name: getattr(snapshot, name).tobytes() for name in (
            "agent_id", "position_um", "radius_um", "genome_id", "nutrient_id",
            "concentration_mmol_l", "phenotype_genome_id", "phenotype_motility_um_s",
            "phenotype_max_uptake_mmol_s",
        )
    }
    before = _digest_session(controller)

    # Run a real frame: the projection the renderer consumes, then the renderer.
    view = WorldView.from_snapshot(snapshot)
    assert not view.position_um.flags.writeable
    assert not view.field_density.flags.writeable

    stats = PygameRenderer().draw(pygame_screen, snapshot)
    assert stats is not None, "draw() must report render stats"
    assert stats.visible_cells >= 0

    after = _digest_session(controller)
    assert after == before, "rendering mutated simulation state"
    for name, blob in snapshot_bytes_before.items():
        assert getattr(snapshot, name).tobytes() == blob, f"rendering mutated snapshot.{name}"


def test_ui_mutations_only_go_through_dispatch_command():
    """No UI module mutates the simulation except through commands.dispatch_command."""
    offenders: list[str] = []
    dispatcher_calls: set[str] = set()

    for path in _ui_module_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for enclosing, method in _mutating_call_sites(tree):
            if path.name == "commands.py" and enclosing == "dispatch_command":
                dispatcher_calls.add(method)
            else:
                offenders.append(f"{path.name}:{enclosing} calls {method}() directly")

    assert not offenders, (
        "the UI must mutate the simulation only through dispatch_command:\n  "
        + "\n  ".join(offenders)
    )
    assert dispatcher_calls == set(MUTATING_FACADE_METHODS), (
        "dispatch_command must cover every mutating facade method; missing: "
        f"{sorted(set(MUTATING_FACADE_METHODS) - dispatcher_calls)}"
    )


def test_dispatch_command_is_the_single_runtime_gate():
    """The dispatcher is the one gate: it delegates and rejects unknown actions."""
    from microbiome.ui import app as ui_app
    from microbiome.ui.commands import dispatch_command

    controller = _build_world()
    assert controller._paused is False

    # The UI-facing wrapper in app.py must route into the one dispatcher.
    ui_app.dispatch_command(controller, "set_paused", paused=True)
    assert controller._paused is True, "UI wrapper did not reach the controller via dispatch_command"

    # And it must refuse anything outside the frozen action set.
    with pytest.raises(ValueError):
        dispatch_command(controller, "definitely_not_an_action")
