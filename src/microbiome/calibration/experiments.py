"""Virtual experiments: each reproduces, in the simulation, how a published quantity was measured.

Every function builds a ``Rig`` (real genome, real tables, the realistic tick) and returns the measured quantity in the
units of the literature, so it can be set directly against a target in ``data/calibration_targets.json``.
"""

from __future__ import annotations

import math

import numpy as np

from ..genome.calibration import Calibration
from ..sim.model import DT_S
from .rig import ACE, GLC, GLUCOSE_G_PER_MMOL, O2, SIG, TOX, TOX2, Rig, context, default_config

GRID_SIDE = 6


def _grid_positions(n: int, dx: float = 10.0) -> np.ndarray:
    """One cell per grid cell, so cells never compete for the same field cell."""
    side = int(math.ceil(math.sqrt(n)))
    return np.array([((i % side) + 0.5) * dx for i in range(n)]), np.array([((i // side) + 0.5) * dx for i in range(n)])


# ------------------------------------------------------------------------------------------ growth and uptake
def steady_growth(accession: str, glucose_mmol_l: float, *, seconds: float = 120.0, n_cells: int = 9,
                  calibration: Calibration | None = None, config=None, seed: int = 1, project_root=None,
                  oxygen_mmol_l: float | None = None) -> dict[str, float]:
    """Exponential growth with glucose and oxygen held constant: growth rate, specific uptake rates and acetate output.

    Rates are per gram of dry mass per hour, as in the literature (mmol/gDW/h); the growth rate is in 1/h.
    """
    cfg = config or default_config()
    xs, ys = _grid_positions(n_cells)
    rig = Rig(accession, np.column_stack([xs, ys]), nx=GRID_SIDE, ny=GRID_SIDE, seed=seed, config=cfg,
              calibration=calibration, project_root=project_root)
    oxygen = cfg.oxygen_saturation_mmol_l if oxygen_mmol_l is None else oxygen_mmol_l
    rig.clamp(GLC, glucose_mmol_l)
    rig.clamp(O2, oxygen)
    v = rig.volume_l
    ticks = int(round(seconds / DT_S))
    mass0 = rig.mass_g
    glucose = o2 = acetate = mass_integral = 0.0
    for _ in range(ticks):
        rig.step()
        glucose += float(np.sum(glucose_mmol_l - rig.env.concentration_mmol_l[GLC])) * v
        o2 += float(np.sum(oxygen - rig.env.concentration_mmol_l[O2])) * v
        acetate += rig.total_mmol(ACE)
        mass_integral += rig.mass_g * DT_S
        rig.clamp(GLC, glucose_mmol_l)
        rig.clamp(O2, oxygen)
        rig.clamp(ACE, 0.0)
        rig.clamp(TOX, 0.0)
        rig.clamp(TOX2, 0.0)
        rig.clamp(SIG, 0.0)
    hours = ticks * DT_S / 3600.0
    mean_mass = mass_integral / (ticks * DT_S)
    q_glc = glucose / (mean_mass * hours)
    return {
        "glucose_mmol_l": glucose_mmol_l,
        "growth_rate_per_h": math.log(rig.mass_g / mass0) / hours,
        "q_glucose": q_glc,
        "q_oxygen": o2 / (mean_mass * hours),
        "q_acetate": acetate / (mean_mass * hours),
        "acetate_carbon_fraction": (2.0 * acetate) / (6.0 * glucose) if glucose > 0 else 0.0,
        "yield_g_per_g": math.log(rig.mass_g / mass0) / hours / (q_glc * GLUCOSE_G_PER_MMOL) if q_glc > 0 else 0.0,
        "divided": float(rig.state.n_agents > n_cells),
    }


def division_timing(accession: str, *, start_fraction: float = 0.9, n_cells: int = 9, glucose_mmol_l: float = 22.0,
                    calibration: Calibration | None = None, config=None, seed: int = 5, project_root=None,
                    max_seconds: float = 5400.0) -> dict[str, float]:
    """Time from a cell of ``start_fraction`` x division mass to its division, in growth medium.

    ``start_fraction = 0.5`` is a whole cell cycle (a newborn); the default starts close to division so the check is cheap.
    The expected time is ln(1 / start_fraction) / mu plus the mean division commitment time.
    """
    cfg = config or default_config()
    xs, ys = _grid_positions(n_cells)
    rig = Rig(accession, np.column_stack([xs, ys]), nx=GRID_SIDE, ny=GRID_SIDE, seed=seed, config=cfg, mass_fraction=start_fraction,
              calibration=calibration, project_root=project_root)
    growth = steady_growth(accession, glucose_mmol_l, seconds=60.0, calibration=calibration, config=cfg, project_root=project_root)
    mu = growth["growth_rate_per_h"]
    expected = math.log(1.0 / start_fraction) / mu * 3600.0 + cfg.division_commitment_s
    horizon = min(max_seconds, 1.4 * expected)       # first generation only: a cell that died must not let a daughter be counted instead
    oxygen = cfg.oxygen_saturation_mmol_l
    rig.clamp(GLC, glucose_mmol_l)
    births, times, tick = 0, [], 0
    while births < n_cells and tick * DT_S < horizon:
        rig.step()
        tick += 1
        new = int(rig.sim.stats.get("births", 0))
        if new:
            births += new
            times.extend([tick * DT_S] * new)
        rig.clamp(GLC, glucose_mmol_l)
        rig.clamp(O2, oxygen)
        rig.clamp(ACE, 0.0)
        rig.clamp(TOX, 0.0)
        rig.clamp(TOX2, 0.0)
        rig.clamp(SIG, 0.0)
    return {"mean_division_time_s": float(np.mean(times)) if times else float("nan"), "divisions": births,
            "expected_s": expected, "growth_rate_per_h": mu,
            "doubling_time_min_equivalent": math.log(2.0) / mu * 60.0}


def saturating_growth(accession: str, **kwargs) -> dict[str, float]:
    """The growth-medium conditions of the source experiments: 22 mM glucose (0.4 %), air-saturated."""
    return steady_growth(accession, kwargs.pop("glucose_mmol_l", 22.0), **kwargs)


def monod_curve(accession: str, concentrations_mmol_l, *, seconds: float = 60.0, **kwargs) -> list[dict[str, float]]:
    return [steady_growth(accession, float(s), seconds=seconds, **kwargs) for s in concentrations_mmol_l]


def half_saturation(curve: list[dict[str, float]], mu_max: float) -> float:
    """Glucose concentration at which growth reaches half of ``mu_max`` (log-linear interpolation), in mmol/L."""
    s = np.array([c["glucose_mmol_l"] for c in curve])
    mu = np.array([c["growth_rate_per_h"] for c in curve])
    order = np.argsort(s)
    return float(np.exp(np.interp(0.5 * mu_max, mu[order], np.log(s[order]))))


def pirt_line(accession: str, *, mu_range=(0.05, 0.4), points: int = 12, seconds: float = 60.0, **kwargs) -> dict[str, float]:
    """q = mu / Y + m from glucose-limited steady states: in a chemostat the growth rate equals the dilution rate, so the
    pairs (mu, q) of clamped-glucose cultures are the chemostat's steady states. Returns the intercept (maintenance) and 1/Y."""
    curve = monod_curve(accession, np.geomspace(2e-5, 3e-2, points), seconds=seconds, **kwargs)
    mu = np.array([c["growth_rate_per_h"] for c in curve])
    q = np.array([c["q_glucose"] for c in curve])
    keep = (mu >= mu_range[0]) & (mu <= mu_range[1])
    if keep.sum() < 3:
        raise RuntimeError("fewer than three steady states fall inside the dilution-rate window")
    slope, intercept = np.polyfit(mu[keep], q[keep], 1)
    return {"maintenance_mmol_glc_gdw_h": float(intercept), "inverse_yield_mmol_per_gdw": float(slope),
            "yield_g_per_g": float(1.0 / (slope * GLUCOSE_G_PER_MMOL)), "points": int(keep.sum())}


# ------------------------------------------------------------------------------------------ motility
def random_walk(accession: str, *, n_cells: int = 120, seconds: float = 150.0, calibration: Calibration | None = None,
                seed: int = 2, project_root=None, config=None) -> dict[str, float]:
    """Swimming with no gradient: mean speed, tumble rate and the 2D random-motility coefficient from the MSD slope."""
    cfg = config or default_config()
    rng = np.random.default_rng(seed)
    size = 200.0
    rig = Rig(accession, rng.random((n_cells, 2)) * size, nx=20, ny=20, seed=seed, config=cfg, calibration=calibration,
              project_root=project_root)
    ticks = int(round(seconds / DT_S))
    sample_every = int(round(1.0 / DT_S))
    ids0 = rig.state.agent_id.copy()
    prev = rig.state.positions_um.copy()
    unwrapped = prev.copy()
    track = [unwrapped.copy()]
    speeds, tumbles = [], 0
    for tick in range(ticks):
        rig.step()
        ids = rig.state.agent_id
        at = np.minimum(np.searchsorted(ids0, ids), ids0.size - 1)
        alive = ids0[at] == ids                          # a cell that died drops out of the track (NaN) instead of shifting the rows
        step = np.full_like(prev, np.nan)
        step[at[alive]] = rig.state.positions_um[alive] - prev[at[alive]]
        step -= size * np.rint(step / size)
        unwrapped += step
        prev = np.full_like(prev, np.nan)
        prev[at[alive]] = rig.state.positions_um[alive]
        speeds.append(float(np.mean(np.hypot(*rig.state.velocities_um_s.T))))
        tumbles += int(rig.sim.stats.get("tumbles", 0))
        for channel in (TOX, TOX2, SIG):                 # a producer must not poison the small closed box it is measured in
            rig.clamp(channel, 0.0)
        if (tick + 1) % sample_every == 0:
            track.append(unwrapped.copy())
    track = np.asarray(track)
    longest = min(20, int(0.7 * seconds))          # 5 to 20 s: several run-and-tumble persistence times (shorter runs: shorter window)
    lags = np.arange(5, longest + 1)
    msd = np.array([np.nanmean(np.sum((track[lag:] - track[:-lag]) ** 2, axis=2)) for lag in lags])
    slope = float(np.polyfit(lags.astype(float), msd, 1)[0])
    d2 = slope / 4.0
    return {"speed_um_s": float(np.mean(speeds)), "tumble_rate_per_s": tumbles / (n_cells * seconds),
            "diffusivity_2d_um2_s": d2, "diffusivity_3d_equivalent_um2_s": d2 * 2.0 / 3.0}


def gliding(accession: str, *, n_cells: int = 60, seconds: float = 900.0, calibration: Calibration | None = None, seed: int = 2,
            project_root=None, config=None) -> dict[str, float]:
    """A surface glider alone: mean speed and the mean time between reversals (reversal events per cell and second, inverted)."""
    cfg = config or default_config()
    rng = np.random.default_rng(seed)
    rig = Rig(accession, rng.random((n_cells, 2)) * 200.0, nx=20, ny=20, seed=seed, config=cfg, calibration=calibration,
              project_root=project_root)
    ticks = int(round(seconds / DT_S))
    speeds, reversals = [], 0
    for _ in range(ticks):
        rig.step()
        speeds.append(float(np.mean(np.hypot(*rig.state.velocities_um_s.T))))
        reversals += int(rig.sim.stats.get("reversals", 0))
    rate = reversals / (n_cells * seconds)
    return {"speed_um_s": float(np.mean(speeds)), "reversal_rate_per_s": rate, "reversal_period_min": (1.0 / rate / 60.0) if rate > 0 else float("inf")}


def predator_growth(predator: str, prey: str, *, prey_per_um2: float = 0.0625, n_predators: int = 12, box_um: float = 80.0,
                    seconds: float = 3600.0, calibration: Calibration | None = None, seed: int = 4, project_root=None,
                    config=None) -> dict[str, float]:
    """A predator among an excess of non-growing prey (a lawn on buffer): the predator's exponential growth rate (per hour).

    The prey are fed nothing, as on starvation agar, so they are food and nothing else; the predator's mass growth over
    the hour is its prey-limited growth rate. Only the predator's own mass is followed.
    """
    cfg = config or default_config()
    rng = np.random.default_rng(seed)
    side = int(round(box_um / 10.0))
    size = side * 10.0
    rig = Rig(predator, rng.random((n_predators, 2)) * size, nx=side, ny=side, seed=seed, config=cfg, calibration=calibration,
              project_root=project_root)
    n_prey = int(round(prey_per_um2 * size * size))
    rig.add_cells(prey, rng.random((n_prey, 2)) * size, mass_pg=0.12)       # below any division mass: prey are food, they neither grow nor divide
    rig.clamp(O2, cfg.oxygen_saturation_mmol_l)
    own = rig.state.genome_id == np.uint64(rig.genome.genome_id)
    start = float(rig.state.mass_pg[own].sum())
    kills = 0
    mass_integral = 0.0
    ticks = int(round(seconds / DT_S))
    for _ in range(ticks):
        rig.step()
        kills += int(rig.sim.stats.get("predation_kills", 0))
        own = rig.state.genome_id == np.uint64(rig.genome.genome_id)
        mass_integral += float(rig.state.mass_pg[own].sum()) * DT_S
        rig.clamp(O2, cfg.oxygen_saturation_mmol_l)
        rig.clamp(ACE, 0.0)
        rig.clamp(TOX, 0.0)
        rig.clamp(TOX2, 0.0)
        rig.clamp(SIG, 0.0)
    own = rig.state.genome_id == np.uint64(rig.genome.genome_id)
    end = float(rig.state.mass_pg[own].sum())
    hours = seconds / 3600.0
    prey_left = int(np.sum(rig.state.genome_id == np.uint64(context(project_root).genomes[prey].genome_id)))
    return {"growth_rate_per_h": math.log(end / start) / hours, "kills": kills, "prey_start": n_prey, "prey_end": prey_left,
            "predators_end": int(own.sum())}


def tumble_angles(accession: str, *, n_cells: int = 200, seconds: float = 60.0, calibration: Calibration | None = None,
                  seed: int = 3, project_root=None, config=None) -> dict[str, float]:
    """Reorientation at tumbles, measured from heading changes. Rotational diffusion is off so a heading changes only at a tumble."""
    import dataclasses

    cfg = dataclasses.replace(config or default_config(), rotational_diffusion_rad2_s=0.0)
    rng = np.random.default_rng(seed)
    rig = Rig(accession, rng.random((n_cells, 2)) * 200.0, nx=20, ny=20, seed=seed, config=cfg, calibration=calibration,
              project_root=project_root)
    rig.step()
    heading = rig.sim.life.heading.copy()
    angles = []
    for _ in range(int(round(seconds / DT_S))):
        rig.step()
        for channel in (TOX, TOX2, SIG):
            rig.clamp(channel, 0.0)
        now = rig.sim.life.heading
        change = np.abs(np.angle(np.exp(1j * (now - heading))))
        angles.extend(np.degrees(change[change > 1e-9]).tolist())
        heading = now.copy()
    angles = np.asarray(angles)
    return {"mean_deg": float(angles.mean()), "sd_deg": float(angles.std()), "samples": int(angles.size),
            "mean_cosine": float(np.mean(np.cos(np.radians(angles))))}


def chemotactic_velocity(accession: str, *, period_um: float, mean_mmol_l: float = 0.02, amplitude: float = 0.9,
                         n_cells: int = 600, warmup_s: float = 3.0, seconds: float | None = None,
                         calibration: Calibration | None = None, seed: int = 4, project_root=None, config=None,
                         gradient: bool = True) -> dict[str, float]:
    """Up-gradient drift as a fraction of swimming speed, in a sinusoidal glucose profile held constant.

    Cells start uniformly distributed and the drift is measured in the transient that follows (after ``warmup_s``, long
    enough for the receptor memory to adapt), as in the microfluidic experiments: left alone, a chemotactic population
    soon piles up at the peak and the steep regions empty. Only cells in the steep part of the profile (|sin| > 0.7) count;
    each tick contributes its displacement along the local gradient. ``gradient=False`` gives the no-gradient control.
    """
    cfg = config or default_config()
    if seconds is None:
        seconds = float(np.clip(period_um / (6.0 * 30.0), 5.0, 20.0))     # well inside the time a cell needs to cross a slope
    nx = int(round(period_um / 10.0))
    ny = 6
    rng = np.random.default_rng(seed)
    rig = Rig(accession, rng.random((n_cells, 2)) * [period_um, 60.0], nx=nx, ny=ny, seed=seed, config=cfg,
              calibration=calibration, project_root=project_root)
    centres = (np.arange(nx) + 0.5) * 10.0
    profile = mean_mmol_l * (1.0 + amplitude * np.cos(2.0 * np.pi * centres / period_um)) if gradient else np.full(nx, mean_mmol_l)
    field = np.tile(profile, (ny, 1))
    rig.clamp(GLC, field)
    prev = rig.state.positions_um.copy()
    size = np.array([period_um, 60.0])
    drift, speed = [], []
    ticks = int(round((warmup_s + seconds) / DT_S))
    for tick in range(ticks):
        phase = np.sin(2.0 * np.pi * prev[:, 0] / period_um)
        rig.step()
        step = rig.state.positions_um - prev
        step -= size * np.rint(step / size)
        prev = rig.state.positions_um.copy()
        if tick * DT_S >= warmup_s:
            steep = np.abs(phase) > 0.7
            if steep.any():
                drift.append(float(np.mean(-np.sign(phase[steep]) * step[steep, 0])) / DT_S)
                speed.append(float(np.mean(np.hypot(*rig.state.velocities_um_s[steep].T))))
        rig.clamp(GLC, field)
        rig.clamp(ACE, 0.0)
    v_c, v = float(np.mean(drift)), float(np.mean(speed))
    return {"period_um": period_um, "drift_um_s": v_c, "speed_um_s": v, "drift_fraction": v_c / v if v > 0 else 0.0}


def chemotaxis_peak(accession: str, *, periods=(200.0, 500.0, 1200.0), **kwargs) -> dict[str, float]:
    """The largest drift fraction over a scan of gradient steepness (the published maximum is a best case, not a typical one)."""
    runs = [chemotactic_velocity(accession, period_um=p, **kwargs) for p in periods]
    best = max(runs, key=lambda r: r["drift_fraction"])
    return {"peak_drift_fraction": best["drift_fraction"], "peak_period_um": best["period_um"],
            "by_period": {r["period_um"]: r["drift_fraction"] for r in runs}}
