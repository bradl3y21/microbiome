"""Mutation receipt for tests/test_entry_point.py (item 214).

The registered test must go red if the root entry point reverts to the scaffold stub. We replace
``microbiome.__main__.main`` with the retired stub and call the registered test directly; it has no
pytest fixtures, so it can be called as a plain function. src/ is untouched.
"""
import pytest

import microbiome.__main__ as entry
import test_entry_point as ep


def test_the_registered_entry_point_test_rejects_the_old_stub(monkeypatch):
    def old_stub_main():
        print("microbiome: scaffold only - no simulation yet.")
        return 0

    monkeypatch.setattr(entry, "main", old_stub_main)
    with pytest.raises(AssertionError):
        ep.test_root_entry_point_runs_the_app()
