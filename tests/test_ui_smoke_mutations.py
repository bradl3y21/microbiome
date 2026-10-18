"""Mutation checks for the D3-routing guards in ``tests/test_ui_smoke.py``.

A guard that cannot fail proves nothing. Each test here breaks exactly one thing
the item-222 guard claims to catch and shows that the guard's predicate rejects it.
"""
import pytest

from microbiome.sim.controller import SimulationController
from test_ui_smoke import (
    RecordingD3Facade,
    _fingerprint,
    _sim_imports,
    _ui_source_leaks,
)


def test_the_layering_guard_rejects_a_ui_that_imports_d1(tmp_path):
    """A UI that imported a D1 verb would be caught by the static audit."""
    leaking = tmp_path / "leaking_ui.py"
    leaking.write_text(
        "import microbiome.sim\n"
        "from microbiome.sim.engine import step\n"
        "from microbiome.sim.controller import SimulationController\n",
        encoding="utf-8",
    )
    leaked = _sim_imports(leaking, "microbiome.ui") - {"microbiome.sim.controller"}
    assert leaked == {"microbiome.sim", "microbiome.sim.engine"}


def test_the_state_fingerprint_is_sensitive_to_an_out_of_band_step():
    """The fingerprint used above must change when D1 is stepped, or it proves nothing."""
    controller = SimulationController.create_demo(seed=2026)
    before = _fingerprint(controller)
    controller.advance_ticks(1)
    assert _fingerprint(controller) != before


@pytest.mark.parametrize("name", ["_state", "_environment", "_rngs", "step"])
def test_the_recording_facade_hides_everything_outside_d3(name):
    """Reading or stepping D1 through the UI's controller reference must raise."""
    facade = RecordingD3Facade(SimulationController.create_demo(seed=2026))
    with pytest.raises(AttributeError):
        getattr(facade, name)


def test_the_direct_call_guard_rejects_a_ui_that_steps_the_controller_itself():
    """A UI that called controller.advance_ticks() directly would be caught."""
    leaks = _ui_source_leaks(
        "def advance(self):\n"
        "    self.controller.advance_ticks(count=1)\n"
        "    return self.controller._state.tick_index\n"
    )
    assert leaks["calls"] == {"advance_ticks"}
    assert leaks["calls"] - {"list_genomes", "render_snapshot"} == {"advance_ticks"}
    assert leaks["d1_named"] == ["_state"]


def test_the_direct_call_guard_accepts_the_real_dispatcher_and_ui():
    """The guard must not fire on the code that legitimately exists today."""
    from test_ui_smoke import SRC

    for path in sorted((SRC / "ui").glob("*.py")):
        leaks = _ui_source_leaks(path.read_text(encoding="utf-8"))
        assert not leaks["d1_named"], path.name
        if path.name != "commands.py":
            assert leaks["calls"] <= {"list_genomes", "render_snapshot"}, path.name
