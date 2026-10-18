"""Solve the realism layer's free parameters against the published targets, then evaluate every target.

Each parameter is solved against the virtual experiment that measures the corresponding published quantity, in this order:
tumble-angle distribution, tumble rate, swimming speed, then each organism's uptake, biomass yield, acetate overflow, then
the chemotactic gain (which depends on the swimming parameters). Targets with role ``fit`` are matched by construction;
targets with role ``held_out`` are measured afterwards and are the only ones that can disagree for a reason the fit did not
already absorb. ``evaluate`` reports every role honestly, including the ones that fail.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from ..genome.calibration import CALIBRATION_FILE, Calibration, OrganismCalibration
from ..genome.parser import _PROJECT_ROOT
from ..sim.realism import RealismConfig
from . import experiments as X
from ..genome.organisms import MX, PA, SA
from .rig import BS, EC, context, default_config
from .targets import TargetSet, load_targets

DIVISION_MASS_PER_MEAN_MASS = 2.0 / (2.0 * math.log(2.0))     # division mass / mean mass of a steadily growing population
ASSUMED_STORAGE_SCALE = 0.2                                    # storage fraction 0.02 of the D2 0.1: no target, see the file
GROWTH_MEDIUM_MMOL_L = 22.0


def solve_tumble_angle(mean_deg: float, sd_deg: float) -> tuple[float, float]:
    """Underlying (mean, sd) of the normal angle whose clipped (10-170 degrees) realisation has the requested mean and sd."""
    z = np.random.default_rng(0).standard_normal(400_000)
    mu, sd = mean_deg, sd_deg
    for _ in range(80):
        realised = np.clip(mu + sd * z, 10.0, 170.0)
        mu += mean_deg - realised.mean()
        sd *= sd_deg / realised.std()
    return float(mu), float(sd)


def _calibration(ec: OrganismCalibration, bs: OrganismCalibration, shared: dict[str, float], sha: str = "",
                 extra: dict[str, OrganismCalibration] | None = None) -> Calibration:
    return Calibration({EC: ec, BS: bs, **(extra or {})}, dict(shared), sha)


def _growth(accession: str, calibration: Calibration, config: RealismConfig, **kwargs) -> dict[str, float]:
    return X.steady_growth(accession, GROWTH_MEDIUM_MMOL_L, seconds=60.0, calibration=calibration, config=config, **kwargs)


def _bisect(function, target: float, low: float, high: float, increasing: bool, iterations: int = 14) -> float:
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        above = function(mid) > target
        if above == increasing:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def fit(project_root: str | Path | None = None, *, verbose: bool = False) -> tuple[Calibration, RealismConfig]:
    ts = load_targets(project_root)
    say = print if verbose else (lambda *a, **k: None)
    ctx = context(project_root)

    # 1-2. tumbling, from the tracking papers
    angle_mean, angle_sd = solve_tumble_angle(ts["ecoli.tumble_angle"].value, ts["ecoli.tumble_angle_sd"].value)
    tumble_rate = 1.0 / ts["ecoli.run_time"].value
    base = dataclasses.replace(default_config(), tumble_rate_per_s=tumble_rate, tumble_angle_mean_deg=angle_mean,
                               tumble_angle_sd_deg=angle_sd)
    say(f"tumble angle underlying mean {angle_mean:.2f} sd {angle_sd:.2f}; tumble rate {tumble_rate:.4f}/s")

    # 3. swimming speed (E. coli through the global scale, B. subtilis through its own factor)
    ident = OrganismCalibration()
    swim = base.swim_speed_scale
    for _ in range(3):
        cfg = dataclasses.replace(base, swim_speed_scale=swim)
        speed = X.random_walk(EC, n_cells=60, seconds=20.0, config=cfg, calibration=_calibration(ident, ident, {}))["speed_um_s"]
        swim *= ts["ecoli.swim_speed"].value / speed
    base = dataclasses.replace(base, swim_speed_scale=swim)
    bs_motility = ts["bsub.swim_speed"].value / ts["ecoli.swim_speed"].value
    say(f"swim_speed_scale {swim:.4f}; B. subtilis motility_scale {bs_motility:.4f}")

    # 4. physiology of each organism
    division_base = {acc: float(ctx_row) for acc, ctx_row in
                     ((acc, _division_mass(ctx, acc)) for acc in (EC, BS))}
    maintenance = {EC: ts["ecoli.maintenance"].value, BS: ts["bsub.maintenance"].value}
    cals = {
        EC: OrganismCalibration(division_mass_scale=ts["ecoli.dry_mass.mean_at_mu_0.65"].value * 1e-3 * DIVISION_MASS_PER_MEAN_MASS / division_base[EC],
                                storage_scale=ASSUMED_STORAGE_SCALE, maintenance_mmol_glc_gdw_h=maintenance[EC]),
        BS: OrganismCalibration(motility_scale=bs_motility, storage_scale=ASSUMED_STORAGE_SCALE, maintenance_mmol_glc_gdw_h=maintenance[BS]),
    }
    goals = {EC: ("ecoli.q_glucose.glucose_M9", "ecoli.growth_rate.glucose_M9", "q_oxygen", "ecoli.q_oxygen.glucose_M9"),
             BS: ("bsub.q_glucose.glucose_M9", "bsub.growth_rate.glucose_M9", "q_acetate", "bsub.q_acetate.glucose_M9")}
    for acc in (EC, BS):
        q_id, mu_id, third, third_id = goals[acc]
        for sweep in range(4):
            def measured(cal, key):
                shared = _calibration(cals[EC] if acc != EC else cal, cals[BS] if acc != BS else cal, {})
                return _growth(acc, shared, base)[key]

            cal = cals[acc]
            cal = dataclasses.replace(cal, uptake_scale=cal.uptake_scale * ts[q_id].value / measured(cal, "q_glucose"))
            cal = dataclasses.replace(cal, biomass_scale=cal.biomass_scale * ts[mu_id].value / measured(cal, "growth_rate_per_h"))
            key = third
            increasing = key == "q_acetate"
            target = ts[third_id].value
            cal = dataclasses.replace(cal, overflow_scale=_bisect(lambda v: measured(dataclasses.replace(cal, overflow_scale=v), key),
                                                                  target, 0.0, 1.5, increasing))
            cals[acc] = cal
            say(f"{acc} sweep {sweep}: uptake {cal.uptake_scale:.4f} biomass {cal.biomass_scale:.4f} overflow {cal.overflow_scale:.4f}")

    # 4b. the organisms added later: growth by the same loop; speed, size, reversals from their own targets
    base = dataclasses.replace(base, reversal_rate_per_s=round(1.0 / (ts["mx.reversal_period"].value * 60.0), 6))
    say(f"reversal rate {base.reversal_rate_per_s:.6f}/s")
    base = dataclasses.replace(base,
                               development_speed_factor=round(ts["mx.dev_speed_outside"].value / (ts["mx.glide_speed"].value / 60.0), 6),
                               aggregate_speed_ratio=round(ts["mx.dev_speed_inside"].value / ts["mx.dev_speed_outside"].value, 6))
    extra = _fit_added(ts, ctx, base, cals, say)
    base = dataclasses.replace(base, predation_assimilation=_fit_assimilation(ts, base, _calibration(cals[EC], cals[BS], {}, extra=extra), say))

    # 5. chemotactic gain, last: it depends on the swimming parameters
    calibration = _calibration(cals[EC], cals[BS], {}, extra=extra)
    target = ts["ecoli.chemotactic_velocity"].value

    def peak(gain: float) -> float:
        cfg = dataclasses.replace(base, chemotaxis_gain=gain)
        return X.chemotaxis_peak(EC, config=cfg, calibration=calibration)["peak_drift_fraction"]

    low, high = 2.0, 80.0
    for _ in range(9):
        mid = math.sqrt(low * high)
        low, high = (mid, high) if peak(mid) < target else (low, mid)
    gain = math.sqrt(low * high)
    say(f"chemotaxis_gain {gain:.3f}")
    config = dataclasses.replace(base, chemotaxis_gain=gain)
    shared = {"swim_speed_scale": swim, "tumble_rate_per_s": tumble_rate, "tumble_angle_mean_deg": angle_mean,
              "tumble_angle_sd_deg": angle_sd, "chemotaxis_gain": gain, "reversal_rate_per_s": base.reversal_rate_per_s,
              "predation_assimilation": base.predation_assimilation, "development_speed_factor": base.development_speed_factor,
              "aggregate_speed_ratio": base.aggregate_speed_ratio}
    return _calibration(cals[EC], cals[BS], shared, ts.sha256, extra=extra), config


def _fit_added(ts: TargetSet, ctx, base: RealismConfig, cals: dict, say) -> dict[str, OrganismCalibration]:
    """P. aeruginosa and S. aureus (uptake, yield, size, speed) and M. xanthus (gliding speed, length)."""
    volume_per_pg = base.cell_volume_um3_per_pg
    maintenance = ts["ecoli.maintenance"].value             # the placeholder used for organisms with no measured coefficient
    added: dict[str, OrganismCalibration] = {
        PA: OrganismCalibration(storage_scale=ASSUMED_STORAGE_SCALE, maintenance_mmol_glc_gdw_h=maintenance,
                                run_time_s=ts["pa.run_reverse"].value, turn_mean_deg=ts["pa.turn_angle"].value,
                                turn_sd_deg=ts["pa.turn_angle"].sd),
        SA: OrganismCalibration(storage_scale=ASSUMED_STORAGE_SCALE, maintenance_mmol_glc_gdw_h=maintenance,
                                division_mass_scale=ts["sa.division_volume"].value / volume_per_pg / _division_mass(ctx, SA)),
        MX: OrganismCalibration(storage_scale=ASSUMED_STORAGE_SCALE, maintenance_mmol_glc_gdw_h=maintenance,
                                division_mass_scale=_mass_of_rod(ts["mx.cell_length"].value, 0.5, volume_per_pg)
                                * DIVISION_MASS_PER_MEAN_MASS / _division_mass(ctx, MX)),
    }
    ec, bs = cals[EC], cals[BS]

    def world(**overrides):
        return _calibration(ec, bs, {}, extra={**added, **overrides})

    for acc, q_id, mu_id in ((PA, "pa.q_glucose.glucose", "pa.growth_rate.glucose"), (SA, "sa.q_glucose.glucose", "sa.growth_rate.glucose")):
        for sweep in range(4):
            def measured(cal, key, acc=acc):
                return _growth(acc, world(**{acc: cal}), base)[key]

            cal = added[acc]
            cal = dataclasses.replace(cal, uptake_scale=cal.uptake_scale * ts[q_id].value / measured(cal, "q_glucose"))
            cal = dataclasses.replace(cal, biomass_scale=cal.biomass_scale * ts[mu_id].value / measured(cal, "growth_rate_per_h"))
            added[acc] = cal
            say(f"{acc} sweep {sweep}: uptake {cal.uptake_scale:.4f} biomass {cal.biomass_scale:.4f}")

    for acc, target in ((PA, ts["pa.swim_speed"].value), (MX, ts["mx.glide_speed"].value / 60.0)):
        for _ in range(3):
            cal = added[acc]
            speed = X.random_walk(acc, n_cells=40, seconds=20.0, config=base, calibration=world())["speed_um_s"]
            added[acc] = dataclasses.replace(cal, motility_scale=(cal.motility_scale if cal.motility_scale else 1.0) * target / speed)
        say(f"{acc} motility_scale {added[acc].motility_scale:.4f}")
    return added


def _fit_assimilation(ts: TargetSet, base: RealismConfig, calibration: Calibration, say, seconds: float = 2400.0) -> float:
    """The share of a killed cell's biomass the predator takes up, solved so M. xanthus among excess E. coli grows at the measured rate."""
    target = ts["mx.predator_growth"].value

    def growth(eta: float) -> float:
        config = dataclasses.replace(base, predation_assimilation=eta)
        return X.predator_growth(MX, EC, seconds=seconds, calibration=calibration, config=config)["growth_rate_per_h"]

    low_eta, high_eta = 0.02, 0.4
    low, high = growth(low_eta), growth(high_eta)
    say(f"predator growth {low:.3f}/h at assimilation {low_eta}, {high:.3f}/h at {high_eta}")
    eta = low_eta + (target - low) * (high_eta - low_eta) / (high - low)
    for _ in range(2):                                           # secant refinement; growth is close to linear in the share
        value = growth(eta)
        say(f"assimilation {eta:.4f} -> {value:.3f}/h")
        if abs(value - target) < 0.01:
            break
        slope = (high - low) / (high_eta - low_eta)
        low_eta, low, high_eta, high = eta, value, high_eta, high
        eta = eta + (target - value) / slope
    return round(float(min(max(eta, 0.001), 1.0)), 6)


def _mass_of_rod(length_um: float, width_um: float, volume_per_pg: float) -> float:
    """Dry mass (pg) of a capsule of this length and width."""
    volume = math.pi * width_um ** 2 / 4.0 * max(length_um - width_um, 0.0) + math.pi * width_um ** 3 / 6.0
    return volume / volume_per_pg


def _evaluate_added(ts: TargetSet, ctx, calibration: Calibration, config: RealismConfig) -> list[dict]:
    """Every target of P. aeruginosa, S. aureus and M. xanthus, measured in the simulation."""
    from ..genome.organisms import PROFILES
    from ..sim.realism import cell_dimensions

    rows: list[dict] = []

    def add(target_id: str, measured: float | None, note: str = "", ok: bool | None = None):
        target = ts[target_id]
        if ok is None and measured is not None:
            ok = target.contains(measured)
        rows.append({"id": target_id, "role": target.role, "quantity": target.quantity, "unit": target.unit,
                     "target": target.value, "sd": target.sd, "range": target.range,
                     "measured": None if measured is None else float(measured), "ok": None if ok is None else bool(ok), "note": note})

    pa = _growth(PA, calibration, config)
    add("pa.growth_rate.glucose", pa["growth_rate_per_h"])
    add("pa.q_glucose.glucose", pa["q_glucose"])
    add("pa.growth_rate.glucose_dolan", pa["growth_rate_per_h"], f"held out: a second laboratory; yield {pa['yield_g_per_g']:.2f} g/g against the 0.53 of the first")
    add("pa.swim_speed", X.random_walk(PA, n_cells=40, seconds=20.0, calibration=calibration, config=config)["speed_um_s"])
    pa_walk = X.random_walk(PA, n_cells=60, seconds=60.0, calibration=calibration, config=config)
    run = 1.0 / pa_walk["tumble_rate_per_s"]
    add("pa.run_reverse", run, "measured reversal rate inverted", ok=abs(run - ts["pa.run_reverse"].value) < 0.25)
    angles = X.tumble_angles(PA, calibration=calibration, config=config)
    add("pa.turn_angle", angles["mean_deg"], f"sd {angles['sd_deg']:.1f} degrees")
    add("pa.width", PROFILES[PA].width_um)
    line = X.pirt_line(PA, calibration=calibration, config=config)
    add("pa.maintenance", line["maintenance_mmol_glc_gdw_h"], f"Pirt intercept; yield {line['yield_g_per_g']:.3f} g/g from the slope")

    sa = _growth(SA, calibration, config)
    add("sa.growth_rate.glucose", sa["growth_rate_per_h"])
    add("sa.q_glucose.glucose", sa["q_glucose"])
    division_pg = _division_mass(ctx, SA) * calibration.for_accession(SA).division_mass_scale
    add("sa.division_volume", division_pg * config.cell_volume_um3_per_pg, "analytic: division mass x the model's 3.3 um^3/pg")
    line = X.pirt_line(SA, calibration=calibration, config=config)
    add("sa.maintenance", line["maintenance_mmol_glc_gdw_h"], f"Pirt intercept; yield {line['yield_g_per_g']:.3f} g/g from the slope")
    add("sa.products", None, f"the model makes acetate only ({sa['q_acetate']:.1f} mmol/gDW/h) and no lactate")

    predation = X.predator_growth(MX, EC, seconds=2400.0, calibration=calibration, config=config)
    add("mx.predator_growth", predation["growth_rate_per_h"],
        f"{predation['kills']} kills of {predation['prey_start']} prey in 40 min; the 0.03 tolerance is this work's, the source gives no error",
        ok=abs(predation["growth_rate_per_h"] - ts["mx.predator_growth"].value) < 0.03)
    glide = X.gliding(MX, calibration=calibration, config=config)
    add("mx.glide_speed", glide["speed_um_s"] * 60.0)
    add("mx.reversal_period", glide["reversal_period_min"])
    add("mx.dev_speed_outside", glide["speed_um_s"] * config.development_speed_factor, "a fully starved cell: gliding speed x the development factor")
    add("mx.dev_speed_inside", glide["speed_um_s"] * config.development_speed_factor * config.aggregate_speed_ratio, "the same cell fully activated by contact")
    mean_pg = _division_mass(ctx, MX) * calibration.for_accession(MX).division_mass_scale / DIVISION_MASS_PER_MEAN_MASS
    length, _ = cell_dimensions(np.array([mean_pg]), config, np.array([PROFILES[MX].width_um]))
    add("mx.cell_length", float(length[0]), "analytic: length of the mean-mass cell of a growing population")
    add("mx.kill_time", 1.0 / config.predation_rate_per_s, "1 / predation rate (a fully capable predator in contact)")
    add("mx.maintenance", None, "no sugar growth: not measurable; the placeholder is set directly")
    add("mx.generation_time", None, "the model grows M. xanthus on prey only")
    add("mx.development", None, "see docs/ORGANISMS.md for what the simulated development shows")
    return rows


def _division_mass(ctx, accession: str) -> float:
    table = ctx.base_table
    row = int(np.flatnonzero(table.genome_id == np.uint64(ctx.genomes[accession].genome_id))[0])
    return float(table.division_mass_pg[row])


# ------------------------------------------------------------------------------------------ evaluation
def evaluate(calibration: Calibration, config: RealismConfig, ts: TargetSet | None = None, project_root=None,
             slow: bool = False) -> list[dict]:
    """Measure every target in the simulation. Rows: id, role, target, measured, ok, note."""
    ts = ts or load_targets(project_root)
    ctx = context(project_root)
    rows: list[dict] = []

    def add(target_id: str, measured: float | None, note: str = "", ok: bool | None = None):
        target = ts[target_id]
        if ok is None and measured is not None:
            ok = target.contains(measured)
        rows.append({"id": target_id, "role": target.role, "quantity": target.quantity, "unit": target.unit,
                     "target": target.value, "sd": target.sd, "range": target.range,
                     "measured": None if measured is None else float(measured), "ok": None if ok is None else bool(ok), "note": note})

    ec = _growth(EC, calibration, config)
    bs = _growth(BS, calibration, config)
    add("ecoli.growth_rate.glucose_M9", ec["growth_rate_per_h"])
    add("ecoli.q_glucose.glucose_M9", ec["q_glucose"])
    add("ecoli.q_oxygen.glucose_M9", ec["q_oxygen"], f"acetate carbon fraction {ec['acetate_carbon_fraction']:.2f} (inferred, not measured)")
    add("ecoli.dry_mass.mean_at_mu_0.65", _division_mass(ctx, EC) * calibration.for_accession(EC).division_mass_scale * 1000.0 / DIVISION_MASS_PER_MEAN_MASS,
        "analytic: division mass / 1.443")
    add("bsub.growth_rate.glucose_M9", bs["growth_rate_per_h"])
    add("bsub.q_glucose.glucose_M9", bs["q_glucose"])
    add("bsub.q_acetate.glucose_M9", bs["q_acetate"])
    add("bsub.yield.glucose_M9", bs["yield_g_per_g"], "held out")

    curve = X.monod_curve(EC, np.geomspace(1e-5, 30.0, 14), seconds=60.0, calibration=calibration, config=config)
    mu_max = ec["growth_rate_per_h"]
    add("ecoli.Ks.glucose", X.half_saturation(curve, mu_max), "interpolated half-saturation of the simulated Monod curve")
    mu = np.array([c["growth_rate_per_h"] for c in curve])
    fraction = np.array([c["acetate_carbon_fraction"] for c in curve])
    order = np.argsort(mu)
    at_03 = float(np.interp(0.3, mu[order], fraction[order]))
    add("ecoli.acetate.none_below_D_0.3", at_03, "acetate carbon fraction at mu = 0.3/h", ok=at_03 < 0.02)
    rows.append({"id": "ecoli.acetate.batch_threshold", "role": "reference", "quantity": ts["ecoli.acetate.batch_threshold"].quantity,
                 "unit": "1/h", "target": 0.76, "sd": None, "range": None, "measured": None, "ok": None,
                 "note": "not reproduced: overflow depends on sugar saturation, not on growth rate"})

    for accession, tids in ((EC, ("ecoli.maintenance",)), (BS, ("bsub.maintenance.chemostat_check", "bsub.maintenance"))):
        line = X.pirt_line(accession, calibration=calibration, config=config)
        for tid in tids:
            add(tid, line["maintenance_mmol_glc_gdw_h"], f"Pirt intercept; yield {line['yield_g_per_g']:.3f} g/g from the slope")
    walk = X.random_walk(EC, calibration=calibration, config=config)
    add("ecoli.swim_speed", walk["speed_um_s"])
    add("ecoli.run_time", 1.0 / walk["tumble_rate_per_s"], "measured tumble rate inverted", ok=abs(1.0 / walk["tumble_rate_per_s"] - ts["ecoli.run_time"].value) < 0.05)
    add("ecoli.random_motility", walk["diffusivity_3d_equivalent_um2_s"], f"2D MSD diffusivity {walk['diffusivity_2d_um2_s']:.0f} um^2/s x 2/3")
    bs_walk = X.random_walk(BS, n_cells=60, seconds=20.0, calibration=calibration, config=config)
    add("bsub.swim_speed", bs_walk["speed_um_s"], ok=abs(bs_walk["speed_um_s"] - ts["bsub.swim_speed"].value) < 1.0)
    angles = X.tumble_angles(EC, calibration=calibration, config=config)
    add("ecoli.tumble_angle", angles["mean_deg"])
    add("ecoli.tumble_angle_sd", angles["sd_deg"], ok=abs(angles["sd_deg"] - 36.0) < 2.0)
    peak = X.chemotaxis_peak(EC, calibration=calibration, config=config)
    add("ecoli.chemotactic_velocity", peak["peak_drift_fraction"], f"peak at a {peak['peak_period_um']:.0f} um period", ok=0.15 <= peak["peak_drift_fraction"] <= 0.35)
    rows.append({"id": "ecoli.tumble_time", "role": "reference", "quantity": ts["ecoli.tumble_time"].quantity, "unit": "s", "target": 0.19,
                 "sd": None, "range": None, "measured": 0.0, "ok": None, "note": "tumbles are instantaneous in the model"})

    rows.extend(_evaluate_added(ts, ctx, calibration, config))
    from ..genome import variants as V
    add("mutation.snv_rate", float(V.SUBSTITUTIONS_PER_BASE), "frozen D4 rate")
    add("bsub.sporulation_time", float(config.sporulation_time_s))
    from ..genome.physiology import DIFFUSIVITY_UM2_S
    add("diffusion.glucose", DIFFUSIVITY_UM2_S[0])
    add("diffusion.lactose", DIFFUSIVITY_UM2_S[1])
    rows.append({"id": "diffusion.oxygen", "role": "reference", "quantity": ts["diffusion.oxygen"].quantity, "unit": "um^2/s", "target": 2000.0,
                 "sd": None, "range": None, "measured": DIFFUSIVITY_UM2_S[2], "ok": None, "note": "capped by the explicit scheme's stability"})
    return rows


def cycle(project_root=None, *, verbose: bool = True) -> dict[str, dict]:
    """Whole-cell-cycle validation: a newborn simulated until it divides. Merged into data/calibration_fit.json."""
    from ..genome.calibration import load_calibration

    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    calibration = load_calibration(root)
    results = {}
    for accession in (EC, BS, PA, SA):
        results[accession] = {k: float(v) for k, v in X.division_timing(accession, start_fraction=0.5, calibration=calibration,
                                                                        project_root=root).items()}
        if verbose:
            print(accession, results[accession], flush=True)
    path = root / CALIBRATION_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data["validation_cycle"] = results
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return results


# ------------------------------------------------------------------------------------------ files
def write_fit(calibration: Calibration, rows: list[dict], project_root=None) -> Path:
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    path = root / CALIBRATION_FILE
    organisms = {acc: {k: round(v, 6) for k, v in dataclasses.asdict(cal).items()} for acc, cal in calibration.organisms.items()}
    payload = {
        "schema_version": 1,
        "generated_by": "python -m microbiome.calibration fit",
        "targets_sha256": calibration.targets_sha256,
        "note": "Produced by the fit; do not edit by hand. Fit targets are matched by construction; see docs/CALIBRATION.md for what is and is not validated.",
        "organisms": organisms,
        "shared": {k: round(v, 6) for k, v in calibration.shared.items()},
        "assumptions": {"storage_scale": "0.2 of the D2 storage fraction (0.1 -> 0.02): no measurement of glycogen storage was found; it only sets how much substrate a starving cell can burn",
                        "ecoli.maintenance": "NGAM 8.39 mmol ATP/gDW/h divided by an assumed 20-26 ATP per glucose",
                        "bsub.dry_mass": "not found; the D2 division mass is kept",
                        "maintenance of P. aeruginosa, S. aureus, M. xanthus": "no coefficient was found; E. coli's assumed 0.37 mmol glucose/gDW/h is a placeholder",
                        "pa.dry_mass, sa.dry_mass": "not found (searched); the D2 division mass is kept for P. aeruginosa, and S. aureus is sized by cell volume instead"},
        "report": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path
