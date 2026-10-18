"""Item 214: ``python -m microbiome`` must run the application, not print the scaffold message.

History: the root entry point used to print ``microbiome: scaffold only - no simulation yet.`` and
open no window. ``src/microbiome/__main__.py`` now imports ``run_demo`` from ``microbiome.ui.app``
and calls it, returning 0. This file pins that wiring, the exit code, and the absence of the old
scaffold message, and it demonstrates that the same contract rejects the retired stub shape.

The window loop is replaced by a recorder in every test: the real ``run_demo`` blocks on the pygame
event loop, so calling it for real would hang. Nothing else about the entry point is mocked.
"""
import contextlib
import io

import pytest

import microbiome.__main__ as entry
import microbiome.ui.app as ui_app
from conftest import run_python

SCAFFOLD = "scaffold only"


def run_and_check(main_fn):
    """Call ``main_fn`` with the UI run path replaced by a recorder; enforce the entry contract.

    A passing run means: exit code 0, the UI run path called exactly once with no arguments, and
    neither stdout nor stderr contains the old scaffold message.
    """
    calls = []
    original = entry.run_demo
    entry.run_demo = lambda *args, **kwargs: calls.append((args, kwargs))
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main_fn()
    finally:
        entry.run_demo = original
    text = (out.getvalue() + err.getvalue()).lower()
    assert rc == 0, f"entry point returned {rc!r}, want 0"
    assert calls == [((), {})], f"UI run path called {len(calls)} time(s), want exactly once"
    assert SCAFFOLD not in text, f"entry point still printed the scaffold message: {text!r}"
    return rc


def test_root_entry_point_imports_the_ui_run_path():
    """The root module must bind the UI's run_demo itself, not a local placeholder."""
    assert callable(entry.main)
    assert entry.run_demo is ui_app.run_demo, (
        "root __main__ must bind microbiome.ui.app.run_demo, not a local stub")


def test_root_entry_point_runs_the_app():
    """The registered contract test: documented entry point -> UI run path -> exit 0, no stub text."""
    assert run_and_check(entry.main) == 0


def test_python_m_microbiome_exits_zero_and_does_not_print_the_scaffold():
    """Execute the real module file in a fresh interpreter, replacing only the blocking window loop."""
    code = "\n".join([
        "import runpy",
        "import microbiome.ui.app as app",
        "calls = []",
        "app.run_demo = lambda *a, **k: calls.append((a, k))",
        "try:",
        "    runpy.run_module('microbiome', run_name='__main__')",
        "except SystemExit as exc:",
        "    assert exc.code in (0, None), exc.code",
        "print('RUN_DEMO_CALLS=%d' % len(calls))",
        "assert calls == [((), {})], calls",
    ])
    out = run_python(code)
    assert "RUN_DEMO_CALLS=1" in out, out
    assert SCAFFOLD not in out.lower(), out


def test_the_old_scaffold_stub_is_rejected():
    """The retired stub shape must fail the contract above; that is the defect item 214 closes."""
    def old_stub_main():
        print("microbiome: scaffold only - no simulation yet.")
        return 0

    with pytest.raises(AssertionError):
        run_and_check(old_stub_main)
