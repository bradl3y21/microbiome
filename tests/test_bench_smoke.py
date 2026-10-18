"""The benchmark harness must measure honestly before its numbers are quoted.

Fast (small N). The real measurement is: .venv\\Scripts\\python tests\\bench_perf.py
"""
import time

import pytest

import bench_perf
from sim_support import contract_attr


def test_measure_returns_consistent_split_for_a_small_world():
    r = bench_perf.measure(300, frames=6, warmup=2)
    assert r["n_start"] == 300 and abs(r["n_median"] - 300) <= 3
    assert r["sim_ms"]["median"] > 0 and r["render_ms"]["median"] > 0 and r["snapshot_ms"]["median"] > 0
    # a frame is sim + D3 snapshot build + render (+flip, off here): never shorter than any part
    assert r["frame_ms"]["median"] >= max(r["sim_ms"]["median"], r["render_ms"]["median"])
    parts = r["sim_ms"]["median"] + r["snapshot_ms"]["median"] + r["render_ms"]["median"]
    assert r["frame_ms"]["median"] == pytest.approx(parts, rel=0.25)
    assert r["display_driver"] == "dummy"
    assert r["visible_cells_median"] > 0


def test_harness_sees_an_injected_slowdown_in_sim_and_not_in_render():
    real = contract_attr("step")

    def slow_step(*a):
        time.sleep(0.01)
        return real(*a)

    base = bench_perf.measure(300, frames=6, warmup=2)
    slow = bench_perf.measure(300, frames=6, warmup=2, step_fn=slow_step)
    assert slow["sim_ms"]["median"] >= base["sim_ms"]["median"] + 8.0
    assert slow["render_ms"]["median"] < base["render_ms"]["median"] * 2 + 1.0  # the delay landed in sim only


def test_steady_scenario_keeps_the_population_constant():
    r = bench_perf.measure(1500, frames=20, warmup=2)
    assert r["n_max"] - r["n_min"] <= 0.01 * 1500, (r["n_min"], r["n_max"])


def test_mixed_scenario_reports_its_drift_instead_of_hiding_it():
    r = bench_perf.measure(300, frames=40, warmup=2, scenario="mixed")
    assert r["n_max"] != r["n_min"], "mixed scenario did not change population; drift is not being reported"


@pytest.mark.parametrize("antagonism,expect_calls", [(True, True), (False, False)])
def test_antagonism_phase_runs_in_the_benchmark_only_when_requested(monkeypatch, antagonism, expect_calls):
    """The engine silently skips SpatialHash when no genome has antagonism_rate_s > 0. A benchmark
    that did not turn it on would under-report step cost by ~45% at 10k (measured)."""
    import microbiome.sim.engine as eng

    calls = {"pairs": 0}
    real_hash = eng.SpatialHash

    class Counting(real_hash):
        def pairs(self, *a, **k):
            calls["pairs"] += 1
            return super().pairs(*a, **k)

    monkeypatch.setattr(eng, "SpatialHash", Counting)
    bench_perf.measure(400, frames=3, warmup=1, antagonism=antagonism)
    assert (calls["pairs"] > 0) is expect_calls


def test_phase_breakdown_accounts_for_the_step_and_restores_the_engine():
    import microbiome.sim.engine as eng

    before = (eng.diffuse, eng._antagonism_pressure)
    ph = bench_perf.phase_breakdown(500, frames=3)
    assert (eng.diffuse, eng._antagonism_pressure) == before, "phase wrappers leaked out of the benchmark"
    assert ph["sim: antagonism (SpatialHash pairs)"] > 0
    assert ph["sim: Simulation.step (inner total)"] <= ph["sim: step() total"] + 1e-6
    assert ph["render: _cell_blits (sprite lookup + command list)"] <= ph["render: draw() total"]
    assert ph["sim/controller: D3 snapshot build (detached copies)"] > 0
    # the phase labels must account for the whole frame: nothing may be silently dropped by a renamed helper
    accounted = (ph["render: _cell_blits (sprite lookup + command list)"] + ph["render: nutrient density layer"]
                 + ph["render: WorldView projection (phenotype lookup, density sum)"] + ph["render: visibility cull"]
                 + ph["render: native Surface.blits + field edge + rest"])
    assert accounted == pytest.approx(ph["render: draw() total"], rel=1e-9)


def test_the_benchmark_snapshot_is_a_valid_d3_snapshot_detached_from_live_state():
    """D3 (frozen): exact field list; arrays detached, C-contiguous, non-writeable. Mutating the live state after
    the snapshot is taken must not change it, and the renderer must accept it and not write to it."""
    import numpy as np
    import pygame
    from microbiome.render.renderer import PygameRenderer
    from sim_support import collect, find

    state, env, pheno, rngs = bench_perf.world_for(400)
    snap = bench_perf.make_snapshot(state, env, pheno)
    assert set(f.name for f in bench_perf.dataclasses.fields(snap)) == set(bench_perf.D3_SNAPSHOT_FIELDS)
    expected = {"agent_id": np.uint64, "position_um": np.float64, "radius_um": np.float64, "genome_id": np.uint64,
                "nutrient_id": np.uint32, "concentration_mmol_l": np.float64, "phenotype_genome_id": np.uint64,
                "phenotype_motility_um_s": np.float64, "phenotype_max_uptake_mmol_s": np.float64}
    for name, dtype in expected.items():
        a = getattr(snap, name)
        assert a.dtype == np.dtype(dtype) and a.flags.c_contiguous and not a.flags.writeable, name
    assert snap.position_um.shape == (400, 2) and snap.concentration_mmol_l.ndim == 3
    before = {n: getattr(snap, n).copy() for n in expected}
    state.agents.position_um[:] += 1.0                     # live state moves on...
    env.field.concentration_mmol_l[:] *= 0.5
    for n in expected:                                      # ...the snapshot does not
        assert np.array_equal(getattr(snap, n), before[n]), f"{n} is not detached from the live arrays"
    pygame.init()
    surface = pygame.display.set_mode((320, 240))
    stats = PygameRenderer().draw(surface, snap)
    pygame.quit()
    assert stats.visible_cells > 0 and stats.blit_calls == 1
    for n in expected:
        assert np.array_equal(getattr(snap, n), before[n]), f"the renderer modified {n}"
