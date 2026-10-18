"""Measured 10k-agent benchmark: how many agents can be stepped AND rendered at 60 fps.

Run (from the Microbiome dir, project venv):
    .venv\\Scripts\\python tests\\bench_perf.py                       # sweep, headless
    .venv\\Scripts\\python tests\\bench_perf.py --sizes 10000 --profile
    .venv\\Scripts\\python tests\\bench_perf.py --real-window --sizes 10000   # includes display.flip()

Not named test_*.py on purpose: it is a measurement, not a pass/fail gate. The pytest
smoke test (tests/test_bench_smoke.py) checks that the harness itself measures honestly.

One frame = one fixed D1 tick (1/60 s) of Simulation step() + building one D3 RenderSnapshot (the
detached, read-only copy the sim/controller hands to render) + one PygameRenderer.draw(target, snapshot).
That is the real-time case: 60 ticks of simulated time per second of wall time.

D3 (frozen) made draw() take a RenderSnapshot. Core's DTO class (microbiome.core.ui_api.RenderSnapshot) does
not exist yet, so a stand-in with exactly D3's field list is used; the real class is picked up automatically
once Core publishes it. Building the snapshot is a NEW per-frame cost on the sim side and is timed separately.
"""
import argparse
import cProfile
import dataclasses
import gc
import io
import json
import os
import platform
import pstats
import sys
import time

# Must precede the conftest/pygame import: conftest only setdefault()s the dummy driver.
if "--real-window" in sys.argv:
    os.environ["SDL_VIDEODRIVER"] = "windows" if sys.platform == "win32" else os.environ.get("SDL_VIDEODRIVER", "")
else:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_support import build_world, contract_attr  # noqa: E402

BUDGET_MS = 1000.0 / 60.0
DENSITY_CELLS_PER_AGENT = 150 / (12 * 12)  # matches the contract-test world: 150 agents on a 12x12 grid

try:  # the frozen D3 DTO, owned by Core; absent until Core implements src/microbiome/core/ui_api.py
    from microbiome.core.ui_api import RenderSnapshot as CoreRenderSnapshot
except ImportError:
    CoreRenderSnapshot = None

D3_SNAPSHOT_FIELDS = (
    "tick_index", "width_um", "height_um", "agent_id", "position_um", "radius_um", "genome_id",
    "nutrient_id", "concentration_mmol_l", "phenotype_genome_id", "phenotype_motility_um_s",
    "phenotype_max_uptake_mmol_s",
)


@dataclasses.dataclass(frozen=True)
class StandInRenderSnapshot:
    """Exactly D3's RenderSnapshot field list (decision kb_mvc746dnDCSQvHtSJL-6), used until Core's DTO exists."""

    tick_index: np.uint64
    width_um: float
    height_um: float
    agent_id: np.ndarray
    position_um: np.ndarray
    radius_um: np.ndarray
    genome_id: np.ndarray
    nutrient_id: np.ndarray
    concentration_mmol_l: np.ndarray
    phenotype_genome_id: np.ndarray
    phenotype_motility_um_s: np.ndarray
    phenotype_max_uptake_mmol_s: np.ndarray


def _detached(value, dtype):
    """D3: detached copy, C-contiguous, non-writeable."""
    a = np.array(value, dtype=dtype, order="C", copy=True)
    a.setflags(write=False)
    return a


def make_snapshot(state, env, table):
    """What sim/controller does between ticks under D3: copy the live arrays into a read-only snapshot."""
    a, f = state.agents, env.field
    fields = dict(
        tick_index=np.uint64(int(state.tick_index)), width_um=float(env.width_um), height_um=float(env.height_um),
        agent_id=_detached(a.agent_id, np.uint64), position_um=_detached(a.position_um, np.float64),
        radius_um=_detached(a.radius_um, np.float64), genome_id=_detached(a.genome_id, np.uint64),
        nutrient_id=_detached(f.nutrient_id, np.uint32),
        concentration_mmol_l=_detached(f.concentration_mmol_l, np.float64),
        phenotype_genome_id=_detached(table.genome_id, np.uint64),
        phenotype_motility_um_s=_detached(table.motility_um_s, np.float64),
        phenotype_max_uptake_mmol_s=_detached(table.max_uptake_mmol_s, np.float64),
    )
    assert set(fields) == set(D3_SNAPSHOT_FIELDS)
    return (CoreRenderSnapshot or StandInRenderSnapshot)(**fields)


def world_for(n, seed=20261017, scenario="steady", density=1.0, antagonism=True):
    """density=1 is 0.0242 agents/um^2 (the contract-test world); 4 means 4x as crowded."""
    g = max(12, int(round(12 * (n / 150.0) ** 0.5 / density ** 0.5)))
    return build_world(seed, n_agents=n, nx=g, ny=g, scenario=scenario, antagonism=antagonism)


def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {"median": float(np.median(a)), "mean": float(a.mean()), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "min": float(a.min())}


def measure(n, frames=60, warmup=10, viewport=(1280, 720), zoom=None, flip=False,
            ticks_per_frame=1, step_fn=None, seed=20261017, scenario="steady", density=1.0,
            antagonism=True):
    """Time step() and draw() per frame for a world that starts with n agents."""
    from microbiome.render.camera import Camera2D
    from microbiome.render.renderer import PygameRenderer

    step = step_fn or contract_attr("step")
    state, env, pheno, rngs = world_for(n, seed, scenario, density, antagonism)
    pygame.init()
    surface = pygame.display.set_mode(viewport)
    camera = Camera2D(zoom=zoom) if zoom else Camera2D()
    renderer = PygameRenderer(camera=camera)

    for _ in range(warmup):
        step(state, env, pheno, rngs)
        renderer.draw(surface, make_snapshot(state, env, pheno))

    gc.collect()
    sim_ms, snap_ms, draw_ms, frame_ms, counts, visible = [], [], [], [], [], []
    for _ in range(frames):
        t0 = time.perf_counter_ns()
        for _t in range(ticks_per_frame):
            step(state, env, pheno, rngs)
        t1 = time.perf_counter_ns()
        snap = make_snapshot(state, env, pheno)
        t1b = time.perf_counter_ns()
        rs = renderer.draw(surface, snap)
        t2 = time.perf_counter_ns()
        if flip:
            pygame.display.flip()
        t3 = time.perf_counter_ns()
        sim_ms.append((t1 - t0) / 1e6)
        snap_ms.append((t1b - t1) / 1e6)
        draw_ms.append((t2 - t1b) / 1e6)
        frame_ms.append((t3 - t0) / 1e6)
        counts.append(len(state.agents.agent_id))
        visible.append(rs.visible_cells)
    driver = pygame.display.get_driver()
    pygame.quit()
    f = stats(frame_ms)
    return {
        "n_start": n, "n_min": int(min(counts)), "n_max": int(max(counts)), "n_median": int(np.median(counts)),
        "visible_cells_median": int(np.median(visible)), "frames": frames, "viewport": list(viewport),
        "zoom": zoom, "flip": flip, "display_driver": driver, "ticks_per_frame": ticks_per_frame,
        "scenario": scenario, "density_mult": density, "antagonism": antagonism,
        "snapshot_class": (CoreRenderSnapshot or StandInRenderSnapshot).__module__ + "."
                          + (CoreRenderSnapshot or StandInRenderSnapshot).__name__,
        "sim_ms": stats(sim_ms), "snapshot_ms": stats(snap_ms), "render_ms": stats(draw_ms), "frame_ms": f,
        "frac_frames_over_budget": float(np.mean(np.asarray(frame_ms) > BUDGET_MS)),
        "fps_median": 1000.0 / f["median"], "fps_p95_worst": 1000.0 / f["p95"],
        "holds_60fps_median": f["median"] <= BUDGET_MS, "holds_60fps_p95": f["p95"] <= BUDGET_MS,
    }


def profile_phases(n, ticks=30, viewport=(1280, 720), density=1.0, antagonism=True, scenario="steady"):
    from microbiome.render.renderer import PygameRenderer

    step = contract_attr("step")
    state, env, pheno, rngs = world_for(n, scenario=scenario, density=density, antagonism=antagonism)
    pygame.init()
    surface = pygame.display.set_mode(viewport)
    renderer = PygameRenderer()
    for _ in range(5):
        step(state, env, pheno, rngs)
        renderer.draw(surface, make_snapshot(state, env, pheno))
    out = {}
    snap = make_snapshot(state, env, pheno)
    snap_counter = {"n": 0}

    base_tick = int(snap.tick_index)

    def render_once():
        # advance the public tick each call, as a running sim does, so the renderer's per-tick density cache
        # cannot hit on every frame and hide its cost
        snap_counter["n"] += 1
        s = dataclasses.replace(snap, tick_index=np.uint64(base_tick + snap_counter["n"]))
        return renderer.draw(surface, s)

    for label, fn in (("step", lambda: step(state, env, pheno, rngs)),
                      ("snapshot", lambda: make_snapshot(state, env, pheno)),
                      ("render", render_once)):
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(ticks):
            fn()
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
        out[label] = s.getvalue()
    pygame.quit()
    return out


def phase_breakdown(n, frames=60, viewport=(1280, 720), density=1.0, antagonism=True, scenario="steady"):
    """Unprofiled per-phase ms/frame: thin perf_counter wrappers around the engine's phase functions
    and the renderer's helpers. cProfile inflates call-heavy code, so this is the number to quote."""
    from collections import defaultdict

    import microbiome.sim.engine as eng
    import microbiome.sim.model as mod
    from microbiome.render.renderer import PygameRenderer
    from microbiome.render.view import WorldView

    acc = defaultdict(int)
    undo = []

    def wrap(owner, name, label):
        orig = getattr(owner, name)

        def w(*a, **k):
            t = time.perf_counter_ns()
            try:
                return orig(*a, **k)
            finally:
                acc[label] += time.perf_counter_ns() - t

        setattr(owner, name, w)
        undo.append((owner, name, orig))

    for name, label in (("diffuse", "sim: diffusion"), ("_update_uptake", "sim: uptake"),
                        ("_derive_mass", "sim: derive mass (x2)"),
                        ("_chemotaxis_and_movement", "sim: chemotaxis+motion"),
                        ("_antagonism_pressure", "sim: antagonism (SpatialHash pairs)"),
                        ("_add_dead_to_detritus", "sim: death->detritus"), ("_divide", "sim: division"),
                        ("phenotype_arrays", "sim: phenotype gather"),
                        ("_internal_ledger", "sim: ledger recompute (assert)")):
        wrap(eng, name, label)
    wrap(mod.AgentState, "validate", "sim: state.validate (x2)")
    wrap(mod.AgentState, "canonicalize", "sim: canonicalize (x2)")
    wrap(mod.NutrientEnvironment, "validate", "sim: env.validate")
    wrap(eng.Simulation, "step", "sim: Simulation.step (inner total)")
    wrap(PygameRenderer, "_cell_blits", "render: _cell_blits (sprite lookup + command list)")
    wrap(PygameRenderer, "_draw_density", "render: nutrient density layer")
    wrap(PygameRenderer, "_visible_indices", "render: visibility cull")
    wrap(WorldView, "from_snapshot", "render: WorldView projection (phenotype lookup, density sum)")

    step = contract_attr("step")
    state, env, pheno, rngs = world_for(n, scenario=scenario, density=density, antagonism=antagonism)
    pygame.init()
    surface = pygame.display.set_mode(viewport)
    renderer = PygameRenderer()
    per = defaultdict(list)
    try:
        for i in range(frames + 10):
            acc.clear()
            t0 = time.perf_counter_ns()
            step(state, env, pheno, rngs)
            t1 = time.perf_counter_ns()
            snap = make_snapshot(state, env, pheno)
            t1b = time.perf_counter_ns()
            renderer.draw(surface, snap)
            t2 = time.perf_counter_ns()
            if i < 10:
                continue
            per["sim: step() total"].append((t1 - t0) / 1e6)
            per["sim/controller: D3 snapshot build (detached copies)"].append((t1b - t1) / 1e6)
            per["render: draw() total"].append((t2 - t1b) / 1e6)
            for k, v in acc.items():
                per[k].append(v / 1e6)
    finally:
        for owner, name, orig in undo:
            setattr(owner, name, orig)
        pygame.quit()
    med = {k: float(np.median(v)) for k, v in per.items()}
    med["sim: adapter overhead (contract<->internal copy)"] = (
        med["sim: step() total"] - med.get("sim: Simulation.step (inner total)", 0.0))
    med["render: native Surface.blits + field edge + rest"] = (
        med["render: draw() total"] - med.get("render: _cell_blits (sprite lookup + command list)", 0.0)
        - med.get("render: nutrient density layer", 0.0)
        - med.get("render: WorldView projection (phenotype lookup, density sum)", 0.0)
        - med.get("render: visibility cull", 0.0))
    return med


def machine():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pygame": pygame.version.ver,
            "sdl": ".".join(map(str, pygame.get_sdl_version())), "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count()}


def ceiling(rows):
    """Largest measured N whose p95 / median frame time fits 16.67 ms, plus a labelled linear extrapolation."""
    held_p95 = [r["n_median"] for r in rows if r["holds_60fps_p95"]]
    held_med = [r["n_median"] for r in rows if r["holds_60fps_median"]]
    xs = np.array([r["n_median"] for r in rows], dtype=float)
    ys = np.array([r["frame_ms"]["median"] for r in rows], dtype=float)
    k = min(len(xs), 4)
    slope, icpt = np.polyfit(xs[-k:], ys[-k:], 1)
    extrap = (BUDGET_MS - icpt) / slope if slope > 0 else None
    return {"max_measured_n_holding_median": max(held_med) if held_med else None,
            "max_measured_n_holding_p95": max(held_p95) if held_p95 else None,
            "linear_fit_ms_per_1000_agents_last4": float(slope * 1000), "linear_fit_intercept_ms": float(icpt),
            "extrapolated_n_at_16.67ms_median (NOT measured)": None if extrap is None else int(extrap)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[500, 1000, 2000, 4000, 6000, 8000, 10000, 12500, 15000, 20000])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--viewport", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--zoom", type=float, default=None, help="px per um; default = fit whole field (worst case)")
    ap.add_argument("--real-window", action="store_true", help="real display driver and display.flip() each frame")
    ap.add_argument("--scenario", default="steady", choices=["steady", "mixed"],
                    help="steady = constant N (division/death off); mixed = births+deaths, N drifts (reported)")
    ap.add_argument("--density", type=float, default=1.0, help="crowding multiplier vs the contract-test world")
    ap.add_argument("--no-antagonism", action="store_true", help="skip the SpatialHash/contact-antagonism phase")
    ap.add_argument("--phases", action="store_true", help="unprofiled per-phase ms/frame at the largest size")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    print("machine:", json.dumps(machine()))
    rows = []
    hdr = f"{'N start':>8} {'N med':>7} {'vis':>7} | {'sim med':>8} {'sim p95':>8} | {'snap med':>8} | {'draw med':>8} {'draw p95':>8} | {'frame med':>9} {'frame p95':>9} {'fps med':>8} | {'%>16.7ms':>8} | 60fps?"
    print(hdr)
    for n in args.sizes:
        r = measure(n, frames=args.frames, viewport=tuple(args.viewport), zoom=args.zoom,
                    flip=args.real_window, scenario=args.scenario, density=args.density,
                    antagonism=not args.no_antagonism)
        rows.append(r)
        print(f"{r['n_start']:>8} {r['n_median']:>7} {r['visible_cells_median']:>7} | "
              f"{r['sim_ms']['median']:>8.2f} {r['sim_ms']['p95']:>8.2f} | "
              f"{r['snapshot_ms']['median']:>8.2f} | "
              f"{r['render_ms']['median']:>8.2f} {r['render_ms']['p95']:>8.2f} | "
              f"{r['frame_ms']['median']:>9.2f} {r['frame_ms']['p95']:>9.2f} {r['fps_median']:>8.1f} | "
              f"{100 * r['frac_frames_over_budget']:>7.1f}% | "
              f"{'YES' if r['holds_60fps_p95'] else ('median only' if r['holds_60fps_median'] else 'NO')}", flush=True)
    c = ceiling(rows)
    print("budget_ms_per_frame:", round(BUDGET_MS, 3))
    print("ceiling:", json.dumps(c, indent=1))
    if args.phases:
        n = args.sizes[-1]
        ph = phase_breakdown(n, viewport=tuple(args.viewport), density=args.density,
                             antagonism=not args.no_antagonism, scenario=args.scenario)
        print(f"\nphase breakdown at N={n} (median ms/frame, perf_counter wrappers, no profiler):")
        for k in sorted(ph, key=lambda k: (not k.startswith("sim"), -ph[k])):
            print(f"  {ph[k]:7.3f} ms  {k}")
    prof = profile_phases(args.sizes[min(len(args.sizes) - 1, args.sizes.index(10000) if 10000 in args.sizes else 0)],
                          viewport=tuple(args.viewport), density=args.density,
                          antagonism=not args.no_antagonism, scenario=args.scenario) if args.profile else None
    if prof:
        for k, v in prof.items():
            print(f"\n--- cProfile {k} (top by tottime) ---\n{v}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"machine": machine(), "rows": rows, "ceiling": c, "budget_ms": BUDGET_MS}, fh, indent=1)
    return rows


if __name__ == "__main__":
    main()
