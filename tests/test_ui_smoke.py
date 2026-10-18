"""The interactive UI must build against the D3 controller and draw a real frame.

History: until 2026-10-17 no test touched src/microbiome/ui, so the suite stayed
green while the UI broke — the renderer moved to draw(target, RenderSnapshot)
while ui/app.py still called the old draw(target, state, environment, phenotypes).
That strict xfail has been retired now that Vial's migration landed: MicrobiomeUI
holds a D3 controller, reads state only through render_snapshot(), and draws.

Item 222 (independently rechecking item 15's board status) is pinned by
``test_ui_uses_the_d3_controller``: the UI may import only the D3 facade
(``microbiome.sim.controller``), must not name a D1 private attribute, calls the
controller directly only for the two read-only D3 reads, arrives at the sim as
dispatcher actions, leaves the D1 state byte-identical for every non-mutating
interaction, and renders a real frame. ``tests/test_ui_smoke_mutations.py`` shows
that each of those guards actually fails when the property is broken.

Run just this file:
    .venv\\Scripts\\python -m pytest tests/test_ui_smoke.py -q
"""
import ast
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "microbiome"

# The frozen D3 facade (microbiome.core.ui_api.SimulationController). These are
# the only simulation methods a UI or renderer may call.
D3_READ_METHODS = frozenset({
    "list_genomes", "preview_phenotype", "command_result",
    "inspect_cell", "inspect_genome", "render_snapshot",
})
D3_MUTATING_METHODS = frozenset({
    "place_nutrient", "introduce_organism", "set_paused", "advance_ticks", "single_step",
})
D3_METHODS = D3_READ_METHODS | D3_MUTATING_METHODS

def _absolute_imports(path: Path, package: str) -> set[str]:
    """Every module imported by ``path``, with relative imports resolved."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                modules.add(".".join(filter(None, (base, node.module))))
            elif node.module:
                modules.add(node.module)
    return modules


def _sim_imports(path: Path, package: str) -> set[str]:
    """Simulation modules imported by ``path`` (D3 facade or D1 internals).

    ``microbiome.sim`` re-exports the D1 internals (``step``, ``AgentState``,
    ``SimulationState``, ...), so importing the package itself is already a leak;
    only ``microbiome.sim.controller`` — the D3 facade — may be imported, and only
    by the UI.
    """
    return {
        module
        for module in _absolute_imports(path, package)
        if module == "microbiome.sim" or module.startswith("microbiome.sim.")
    }


def _rgb_bytes(surface) -> bytes:
    """Frame bytes for a Surface, across pygame / pygame-ce versions."""
    import pygame

    to_bytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
    return to_bytes(surface, "RGB")


D1_PRIVATE_NAMES = ("_state", "_environment", "_rngs", "_phenotypes", "_command_log")


def _ui_source_leaks(source: str) -> dict[str, object]:
    """What a single UI source file leaks: D1 private names and direct controller calls."""
    return {
        "d1_named": [token for token in D1_PRIVATE_NAMES if token in source],
        "calls": set(re.findall(r"controller\.(\w+)\(", source)),
    }


def _fingerprint(controller) -> str:
    """Byte-exact digest of the D1 state visible through the D3 projection."""
    snapshot = controller.render_snapshot()
    digest = hashlib.sha256(str(int(snapshot.tick_index)).encode())
    for name in ("agent_id", "position_um", "radius_um", "genome_id",
                 "concentration_mmol_l", "phenotype_genome_id"):
        digest.update(np.ascontiguousarray(getattr(snapshot, name)).tobytes())
    return digest.hexdigest()


def _post(ui, kind, **kwargs):
    import pygame

    ui.handle_event(pygame.event.Event(kind, **kwargs))


def _key(ui, key):
    import pygame

    _post(ui, pygame.KEYDOWN, key=key)


class RecordingD3Facade:
    """Expose *only* the frozen D3 facade and record every call the UI makes.

    Any attempt to reach past the facade (``controller._state``, ``controller._rngs``,
    a D1 verb such as ``controller.step``) raises AttributeError instead of silently
    working, so the UI cannot mutate or step D1 state behind the contract.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "calls", [])
        for name in D3_METHODS:
            object.__setattr__(self, name, self._forwarder(name))

    def _forwarder(self, name):
        def call(**payload):
            self.calls.append((name, tuple(sorted(payload))))
            return getattr(self._inner, name)(**payload)

        return call

    def __getattr__(self, name):  # only reached when normal lookup fails
        raise AttributeError(
            f"UI reached for {name!r}, which is not part of the frozen D3 facade"
        )

    def recorded(self) -> list[str]:
        return [name for name, _ in self.calls]


def _non_mutating_script(ui):
    """Every interaction that must NOT change the world: panels, previews, view."""
    import pygame

    for key in (pygame.K_h, pygame.K_h,            # help open/close
                pygame.K_p, pygame.K_p,            # paint mode on/off (no click)
                pygame.K_1, pygame.K_2,            # select a nutrient
                pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET,  # brush dose
                pygame.K_o,                        # organism preview...
                pygame.K_RIGHT, pygame.K_LEFT,     # ...cycle genomes
                pygame.K_ESCAPE,                   # ...cancel, no commit
                pygame.K_PAGEUP, pygame.K_PAGEDOWN):
        _key(ui, key)
    _post(ui, pygame.MOUSEWHEEL, y=1)
    _post(ui, pygame.MOUSEWHEEL, y=-1)
    _post(ui, pygame.MOUSEMOTION, pos=(300, 300), buttons=(0, 0, 0), rel=(5, 5))
    _post(ui, pygame.MOUSEBUTTONDOWN, button=2, pos=(300, 300))   # middle-drag pan
    _post(ui, pygame.MOUSEMOTION, pos=(330, 312), buttons=(0, 1, 0), rel=(30, 12))
    _post(ui, pygame.MOUSEBUTTONUP, button=2, pos=(330, 312))


@pytest.fixture
def pygame_screen():
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((960, 640))
    yield screen
    pygame.quit()


def test_ui_uses_the_d3_controller(pygame_screen):
    """Item 222: the UI talks to the sim only through the frozen D3 facade.

    Reproduced independently: (1) no UI or renderer source imports D1 internals;
    (2) the UI can only reach the D3 methods, and never steps D1 itself; (3) every
    non-mutating interaction leaves the D1 state byte-identical; (4) the mutating
    interactions arrive at the controller as D3 calls; (5) a real frame renders.
    """
    import pygame

    from microbiome.sim.controller import SimulationController
    from microbiome.ui.app import MicrobiomeUI, dispatch_command
    from microbiome.ui.commands import dispatch_command as command_module_dispatch

    # (1) Static layering: the UI may import the D3 facade, nothing D1. Renderers
    #     may import no sim module at all; they consume only RenderSnapshot.
    for path in sorted((SRC / "ui").glob("*.py")):
        leaked = _sim_imports(path, "microbiome.ui") - {"microbiome.sim.controller"}
        assert not leaked, f"{path.name} imports simulation internals: {sorted(leaked)}"
    for path in sorted((SRC / "render").glob("*.py")):
        leaked = _sim_imports(path, "microbiome.render")
        assert not leaked, f"{path.name} imports simulation modules: {sorted(leaked)}"

    # …and no UI source names a D1 private attribute. Outside the single dispatcher,
    # the only direct controller calls allowed are the two read-only D3 reads.
    dispatcher_calls: set[str] = set()
    for path in sorted((SRC / "ui").glob("*.py")):
        leaks = _ui_source_leaks(path.read_text(encoding="utf-8"))
        assert not leaks["d1_named"], f"{path.name} names D1 internals: {leaks['d1_named']}"
        calls = leaks["calls"]
        if path.name == "commands.py":
            dispatcher_calls |= calls
        else:
            assert calls <= {"list_genomes", "render_snapshot"}, (
                f"{path.name} calls the controller outside the read-only D3 reads: "
                f"{sorted(calls - {'list_genomes', 'render_snapshot'})}"
            )
    assert dispatcher_calls == set(D3_MUTATING_METHODS) | {
        "inspect_cell", "inspect_genome", "preview_phenotype", "command_result",
    }, f"the dispatcher's action surface drifted: {sorted(dispatcher_calls)}"

    # (2) The UI has exactly one command path: app.py's public alias is a thin
    #     wrapper around microbiome.ui.commands.dispatch_command (not the same object),
    #     and that dispatcher carries *actions* only.
    import microbiome.ui.app as ui_app

    assert ui_app._dispatch_command is command_module_dispatch, (
        "ui.app must route through the single ui.commands dispatcher"
    )
    real = SimulationController.create_demo(seed=2026)
    # The dispatcher carries *actions* only: D1 verbs are refused, and so are the
    # read-projection methods the UI calls directly on the facade.
    for not_an_action in ("step", "diffuse", "ledger", "state_digest",
                          "render_snapshot", "list_genomes"):
        with pytest.raises(ValueError):
            dispatch_command(real, not_an_action)
    assert dispatch_command(real, "command_result", command_id=0) is None

    # (3) Drive the whole UI through a facade that exposes nothing else.
    facade = RecordingD3Facade(SimulationController.create_demo(seed=2026))
    inner = facade._inner
    ui = MicrobiomeUI(controller=facade, window_size=(960, 640), project_root=REPO_ROOT)
    with pytest.raises(AttributeError):
        facade._state  # noqa: B018 - the D1 world must be unreachable

    before = _fingerprint(inner)
    facade.calls.clear()
    _non_mutating_script(ui)
    ui.draw(pygame_screen)

    assert not [name for name in facade.recorded() if name in D3_MUTATING_METHODS], (
        f"a read-only interaction mutated the simulation: {facade.calls}"
    )
    assert _fingerprint(inner) == before, "the UI mutated D1 state without dispatching a D3 command"

    # Drawing a frame must not step or mutate anything either.
    facade.calls.clear()
    for _ in range(5):
        ui.draw(pygame_screen)
    assert facade.recorded() == ["render_snapshot"] * 5, facade.calls

    # (4) Mutating interactions arrive as D3 facade calls and only then move the tick.
    facade.calls.clear()
    ui.paused = False
    _key(ui, pygame.K_PERIOD)                       # '.' -> exactly one fixed tick
    assert [name for name, _ in facade.calls] == ["single_step"], facade.calls
    assert int(inner.render_snapshot().tick_index) == 1
    # The fixed timestep lives in the simulation, not the UI: one key press is one tick.
    facade.calls.clear()
    ui.advance_frame()                              # the run loop asks for `speed` ticks
    assert [name for name, _ in facade.calls] == ["advance_ticks"], facade.calls
    assert int(inner.render_snapshot().tick_index) == 2

    facade.calls.clear()
    _key(ui, pygame.K_SPACE)                        # pause
    assert [name for name, _ in facade.calls] == ["set_paused"], facade.calls
    _key(ui, pygame.K_SPACE)
    assert "set_paused" in facade.recorded()

    facade.calls.clear()
    _key(ui, pygame.K_1)                            # paint mode, glucose
    _post(ui, pygame.MOUSEBUTTONDOWN, button=1, pos=(200, 400))
    ui.advance_frame()
    assert "place_nutrient" in facade.recorded(), facade.calls
    assert "advance_ticks" in facade.recorded(), facade.calls

    facade.calls.clear()
    _key(ui, pygame.K_o)                            # introduce a real genome
    _key(ui, pygame.K_RETURN)
    ui.advance_frame()
    assert "introduce_organism" in facade.recorded(), facade.calls
    assert int(inner.render_snapshot().tick_index) >= 2
    assert len(inner.render_snapshot().agent_id) == 1

    # (5) A real frame renders, and the field region changes once a cell exists.
    #     Both frames are at the same tick with no nutrients, so the only
    #     difference between them is the live cell the renderer must draw.
    field = pygame.Rect(0, 50, 600, 540)

    def fresh_driven_ui(ticks, introduce):
        driver = RecordingD3Facade(SimulationController.create_demo(seed=2026))
        driven = MicrobiomeUI(controller=driver, window_size=(960, 640), project_root=REPO_ROOT)
        if introduce:
            _key(driven, pygame.K_o)
            _key(driven, pygame.K_RETURN)
        for _ in range(ticks):
            driven.advance_frame()
        return driven

    empty_frame = fresh_driven_ui(2, introduce=False)
    empty_frame.draw(pygame_screen)
    without_cell = _rgb_bytes(pygame_screen.subsurface(field))
    cell_frame = fresh_driven_ui(2, introduce=True)
    cell_frame.draw(pygame_screen)
    with_cell = _rgb_bytes(pygame_screen.subsurface(field))

    status_row = {pygame_screen.get_at((x, 20))[:3] for x in range(0, 400, 2)}
    assert len(status_row) > 3, "the status bar text was not rendered"
    assert with_cell != without_cell, "the field did not change when a real cell was present"

    # The UI holds no hidden world state: same seed + same script -> same D1 digest.
    assert _fingerprint(fresh_driven_ui(30, introduce=True).controller) == _fingerprint(
        fresh_driven_ui(30, introduce=True).controller
    ), "the UI added non-deterministic state outside the facade"


def test_the_demo_ui_constructs_against_the_d3_controller(pygame_screen):
    from microbiome.core.ui_api import RenderSnapshot
    from microbiome.ui.app import MicrobiomeUI

    ui = MicrobiomeUI.from_demo(seed=2026)

    controller = ui.controller
    assert controller is not None, "MicrobiomeUI must hold a D3 SimulationController"
    for method in (
        "list_genomes",
        "preview_phenotype",
        "place_nutrient",
        "introduce_organism",
        "command_result",
        "inspect_cell",
        "inspect_genome",
        "set_paused",
        "advance_ticks",
        "single_step",
        "render_snapshot",
    ):
        assert callable(getattr(controller, method, None)), f"D3 facade is missing {method}()"

    # The UI must read simulation state only through the detached snapshot.
    assert isinstance(controller.render_snapshot(), RenderSnapshot)


def test_the_demo_ui_draws_one_frame_against_the_real_renderer(pygame_screen):
    from microbiome.ui.app import MicrobiomeUI

    ui = MicrobiomeUI.from_demo(seed=2026)
    ui.draw(pygame_screen)  # must not raise; was a TypeError before the D3 migration
