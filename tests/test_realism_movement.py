"""Realistic movement against closed-form expectations: run-and-tumble, adaptation, Brownian motion, flow, contact."""
import math

import numpy as np
import pytest

import realism_support as R
from microbiome.sim.realism import LifeState, rod_length_um

DT = 1.0 / 60.0
SENSE_ONLY = np.array([1e-30, 0, 0, 0, 0, 0])   # can sense glucose without eating a measurable amount


def swimmers(n, *, seed=3, nx=40, ny=40, config=None, trait=None, field=None, **kw):
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)) * [nx * 10.0, ny * 10.0]
    t = trait or R.traits(max_uptake_mmol_s=SENSE_ONLY, division_rate_s=0.0, death_rate_s=0.0)
    return R.world(pos, nx=nx, ny=ny, seed=seed, config=config or R.quiet(), trait=t, energy=1e9, **(field or {}), **kw)


def test_the_tumble_rate_without_a_gradient_is_the_configured_baseline():
    n, ticks = 1000, 600
    w = swimmers(n, trait=R.traits(chemotaxis_sensitivity=0.0, max_uptake_mmol_s=SENSE_ONLY, division_rate_s=0.0, death_rate_s=0.0))
    total = 0
    for _ in range(ticks):
        w.step()
        total += w.sim.stats["tumbles"]
    rate = R.quiet().tumble_rate_per_s
    p = -math.expm1(-rate * DT)
    expected = n * ticks * p
    assert total == pytest.approx(expected, rel=0.05), f"{total} tumbles vs {expected:.0f} expected at {rate:.3f} per second"


def test_a_tumble_turns_by_the_configured_angle_distribution():
    cfg = R.quiet(tumble_rate_per_s=30.0)
    w = swimmers(500, config=cfg, trait=R.traits(chemotaxis_sensitivity=0.0, max_uptake_mmol_s=SENSE_ONLY, division_rate_s=0.0, death_rate_s=0.0))
    w.step()
    angles = []
    for _ in range(200):
        before = w.sim.life.heading.copy()
        w.step()
        change = np.abs(np.angle(np.exp(1j * (w.sim.life.heading - before))))
        angles.extend(change[change > 0])
    draw = np.deg2rad(np.clip(cfg.tumble_angle_mean_deg + cfg.tumble_angle_sd_deg * np.random.default_rng(0).standard_normal(400_000),
                              10.0, 170.0))
    assert len(angles) > 5000
    assert np.mean(angles) == pytest.approx(draw.mean(), rel=0.03)
    assert np.std(angles) == pytest.approx(draw.std(), rel=0.08)
    assert min(angles) >= math.radians(10.0) - 1e-9 and max(angles) <= math.radians(170.0) + 1e-9


def toward_peak_speed(w, ticks, width):
    start = w.state.positions_um[:, 0].copy()
    unwrapped, prev = start.copy(), start.copy()
    for _ in range(ticks):
        w.step()
        now = w.state.positions_um[:, 0]
        d = now - prev
        d -= width * np.rint(d / width)
        unwrapped += d
        prev = now.copy()
    sign = -np.sign(np.sin(2 * np.pi * start / width))          # which way is up the cosine profile
    return float((sign * (unwrapped - start)).mean() / (ticks * DT))


def cosine_world(sensitivity, n=600, nx=80, ny=12, seed=3):
    width = nx * 10.0
    profile = 0.05 * (1.0 + 0.9 * np.cos(2 * np.pi * (np.arange(nx) + 0.5) * 10.0 / width))
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)) * [width, ny * 10.0]
    t = R.traits(chemotaxis_sensitivity=sensitivity, max_uptake_mmol_s=SENSE_ONLY, division_rate_s=0.0, death_rate_s=0.0)
    return R.world(pos, nx=nx, ny=ny, seed=seed, config=R.quiet(), trait=t, energy=1e9, glucose=np.tile(profile, (ny, 1))), width


def test_run_and_tumble_climbs_a_gradient_and_a_blind_cell_does_not():
    sighted, width = cosine_world(0.85)
    blind, _ = cosine_world(0.0)
    up = toward_peak_speed(sighted, 3600, width)
    flat = toward_peak_speed(blind, 3600, width)
    assert up > 0.8, f"drift up the gradient {up:.2f} um/s (a few um/s is typical of measured E. coli)"
    assert abs(flat) < 0.35, f"no sensing, no drift: {flat:.2f} um/s"
    assert up < 0.5 * 10.0 * 2.0, "drift stays a fraction of swim speed (chemotactic index well below 1)"


def test_a_cell_that_cannot_use_the_sugar_does_not_chemotax_toward_it():
    deaf, width = cosine_world(0.85)
    deaf.phenotypes = {R.GENOME: R.traits(chemotaxis_sensitivity=0.85, max_uptake_mmol_s=np.zeros(6), division_rate_s=0.0, death_rate_s=0.0)}
    assert abs(toward_peak_speed(deaf, 3600, width)) < 0.35


def test_chemotaxis_adapts_a_step_change_suppresses_tumbling_then_it_recovers():
    n = 800
    w = swimmers(n, field={"glucose": 0.05})
    w.step(30)
    base = np.mean([ (w.step(), w.sim.stats["tumbles"])[1] for _ in range(300)]) / n / DT
    w.set_field(R.GLC, 0.10)                                       # a sudden doubling of the attractant
    first = np.mean([(w.step(), w.sim.stats["tumbles"])[1] for _ in range(60)]) / n / DT
    w.step(600)                                                    # ~10 s: several adaptation times
    late = np.mean([(w.step(), w.sim.stats["tumbles"])[1] for _ in range(600)]) / n / DT
    assert first < 0.3 * base, f"tumbling should collapse after an up-step ({first:.2f} vs baseline {base:.2f} per s)"
    assert late == pytest.approx(base, rel=0.2), "perfect adaptation: back to baseline in a constant new background"


def test_brownian_motion_gives_the_einstein_mean_squared_displacement():
    n, ticks, D = 2000, 600, 0.4
    cfg = R.config(collisions=False, rotational_diffusion_rad2_s=0.0, translational_diffusion_um2_s=D)
    rng = np.random.default_rng(5)
    pos = 400.0 + rng.random((n, 2)) * 200.0
    w = R.world(pos, nx=100, ny=100, seed=5, config=cfg, energy=1e9,
                trait=R.traits(motility_um_s=0.0, max_uptake_mmol_s=np.zeros(6), division_rate_s=0.0, death_rate_s=0.0))
    start = w.state.positions_um.copy()
    w.step(ticks)
    msd = float(np.mean(np.sum((w.state.positions_um - start) ** 2, axis=1)))
    assert msd == pytest.approx(4 * D * ticks * DT, rel=0.08), f"MSD {msd:.2f} um^2 vs 4Dt = {4 * D * ticks * DT:.2f}"


def test_rotational_diffusion_spreads_the_heading_by_two_d_t():
    n, ticks, Dr = 3000, 120, 0.15
    cfg = R.config(collisions=False, translational_diffusion_um2_s=0.0, rotational_diffusion_rad2_s=Dr, tumble_rate_per_s=0.0)
    w = swimmers(n, nx=100, ny=100, config=cfg, trait=R.traits(chemotaxis_sensitivity=0.0, motility_um_s=0.0, max_uptake_mmol_s=np.zeros(6), division_rate_s=0.0, death_rate_s=0.0))
    w.step()
    start = w.sim.life.heading.copy()
    w.step(ticks)
    spread = float(np.var(w.sim.life.heading - start))
    assert spread == pytest.approx(2 * Dr * (ticks + 0) * DT, rel=0.1)


def test_swim_speed_is_the_d2_coefficient_times_the_documented_scale_and_cells_without_energy_stop():
    w = swimmers(50, nx=60, ny=60, trait=R.traits(chemotaxis_sensitivity=0.0, motility_um_s=10.0, max_uptake_mmol_s=np.zeros(6), division_rate_s=0.0, death_rate_s=0.0))
    w.step()
    assert np.hypot(*w.state.velocities_um_s.T) == pytest.approx(10.0 * R.quiet().swim_speed_scale)
    w.state.energy_pj[:] = 0.0
    w2 = R.world(((300.0, 300.0), (310.0, 310.0)), nx=60, ny=60, config=R.quiet(initial_energy_reserve_s=0.0),
                 trait=R.traits(max_uptake_mmol_s=np.zeros(6), division_rate_s=0.0, death_rate_s=0.0), energy=0.0, phys=R.maintaining())
    w2.step(3)
    assert np.allclose(w2.state.velocities_um_s, 0.0), "no energy, no swimming"


def test_uniform_flow_carries_cells_and_the_solute_together_and_conserves_mass():
    w = swimmers(300, nx=60, ny=20, config=R.quiet(), trait=R.traits(motility_um_s=0.0, max_uptake_mmol_s=np.zeros(6), division_rate_s=0.0, death_rate_s=0.0))
    blob = np.zeros((20, 60)); blob[:, 10] = 0.1
    w.set_field(R.GLC, blob)
    w.sim.flow_um_s = (30.0, 0.0)
    start = w.state.positions_um.copy()
    glucose0 = w.field_mmol(R.GLC)
    w.step(60)                                                     # one second of 30 um/s flow
    moved = w.state.positions_um[:, 0] - start[:, 0]
    moved -= 600.0 * np.rint(moved / 600.0)
    assert moved.mean() == pytest.approx(30.0, abs=0.5)
    assert w.field_mmol(R.GLC) == pytest.approx(glucose0, rel=1e-9)
    column = w.env.concentration_mmol_l[R.GLC].mean(axis=0)
    centroid = float((column * (np.arange(60) + 0.5) * 10.0).sum() / column.sum())
    assert centroid == pytest.approx(10.5 * 10.0 + 30.0, abs=4.0), "the solute moves with the water"


WIDTH = 0.8
ROD = float(rod_length_um(np.array([0.3]), R.config())[0])        # a 0.3 pg founder: ~2.24 um long, 0.8 um wide


def rods(positions, headings, *, eps=0.0, nx=20):
    """Immobile rods at given positions and headings; collisions every tick so a few hundred ticks fully relax them."""
    cfg = R.quiet(collisions=True, collision_interval_ticks=1)
    positions = np.asarray(positions, dtype=np.float64)
    w = R.world(positions, nx=nx, ny=20, config=cfg, energy=1e9,
                trait=R.traits(motility_um_s=0.0, max_uptake_mmol_s=np.zeros(6), division_rate_s=0.0, death_rate_s=0.0))
    n = len(positions)
    w.sim.life = LifeState(ids=w.state.agent_id.copy(), heading=np.asarray(headings, dtype=float), memory=np.full(n, -1.0),
                           lag=np.ones(n), stress_s=np.zeros(n), spore=np.zeros(n), eps=np.full(n, eps),
                           derepression=np.full(n, -1.0))
    return w


def rod_gap(w, i, j):
    """Brute-force surface distance between two rods: the smallest centre-line distance, found by dense sampling."""
    size = np.array([w.env.width_um, w.env.height_um])
    t = np.linspace(-0.5, 0.5, 241)[:, None]
    def line(k, shift):
        h = w.sim.life.heading[k]
        half = 0.5 * max(float(rod_length_um(w.state.mass_pg[k:k + 1], w.sim.config)[0]) - WIDTH, 0.0)
        return w.state.positions_um[k] + shift + 2 * half * t * np.array([np.cos(h), np.sin(h)])
    offset = w.state.positions_um[j] - w.state.positions_um[i]
    shift = -size * np.rint(offset / size)
    a, b = line(i, 0.0), line(j, shift)
    return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min()) - WIDTH


def test_side_by_side_rods_that_overlap_are_pushed_apart_to_exactly_one_width():
    w = rods([(100.0, 100.0), (100.0, 100.3)], [0.0, 0.0])
    assert rod_gap(w, 0, 1) < -0.4
    w.step(300)
    assert rod_gap(w, 0, 1) == pytest.approx(0.0, abs=0.03), "rods rest on each other's surface"
    assert w.state.positions_um[1, 1] - w.state.positions_um[0, 1] == pytest.approx(WIDTH, rel=0.03)
    assert w.state.positions_um[1, 0] == pytest.approx(w.state.positions_um[0, 0], abs=1e-6), "they only separate, they do not slide"
    assert w.n == 2


def test_rods_in_line_stop_when_their_tips_touch_one_length_apart():
    w = rods([(100.0, 100.0), (101.0, 100.0)], [0.0, 0.0])
    w.step(400)
    assert w.state.positions_um[1, 0] - w.state.positions_um[0, 0] == pytest.approx(ROD, rel=0.03)
    assert rod_gap(w, 0, 1) == pytest.approx(0.0, abs=0.03)


def test_crossed_rods_and_coincident_rods_are_separated_without_numerical_trouble():
    for headings in ([0.0, np.pi / 2], [0.3, 0.3]):
        w = rods([(100.0, 100.0), (100.0, 100.0)], headings)
        w.step(400)
        assert np.isfinite(w.state.positions_um).all()
        assert rod_gap(w, 0, 1) > -0.05, headings


def test_contact_works_across_the_periodic_boundary():
    w = rods([(199.95, 100.0), (0.05, 100.0)], [np.pi / 2, np.pi / 2])        # vertical rods 0.1 um apart through the edge
    assert rod_gap(w, 0, 1) < -0.5
    w.step(300)
    assert rod_gap(w, 0, 1) == pytest.approx(0.0, abs=0.03)
    assert (w.state.positions_um >= 0).all() and (w.state.positions_um[:, 0] < 200.0).all()


def test_a_dense_clump_of_rods_relaxes_to_a_non_overlapping_packing():
    rng = np.random.default_rng(9)
    w = rods(100.0 + rng.random((40, 2)) * 14.0, rng.random(40) * np.pi)
    def worst():
        return max(-rod_gap(w, i, j) for i in range(40) for j in range(i + 1, 40)) / WIDTH
    before = worst()
    w.step(1200)
    after = worst()
    assert before > 0.8, "the clump starts with rods crossing each other"
    assert after < 0.3, f"after relaxing, the worst overlap is {after:.2f} of a rod width"
    assert w.n == 40


def test_matrix_producing_rods_attract_when_just_apart_and_plain_rods_do_not():
    gap = 0.2
    sticky = rods([(100.0, 100.0), (100.0, 100.0 + WIDTH + gap)], [0.0, 0.0], eps=1.0)
    plain = rods([(100.0, 100.0), (100.0, 100.0 + WIDTH + gap)], [0.0, 0.0], eps=0.0)
    s0, p0 = rod_gap(sticky, 0, 1), rod_gap(plain, 0, 1)
    sticky.step(300)
    plain.step(300)
    assert rod_gap(plain, 0, 1) == pytest.approx(p0, abs=1e-9), "no matrix, no attraction"
    assert rod_gap(sticky, 0, 1) < s0 - 0.05, "matrix cells draw together"
    assert rod_gap(sticky, 0, 1) > -0.05, "but never overlap"


def test_rods_far_outside_cohesion_range_feel_nothing():
    far = rods([(100.0, 100.0), (100.0, 100.0 + WIDTH + 2.0)], [0.0, 0.0], eps=1.0)
    gap = rod_gap(far, 0, 1)
    far.step(100)
    assert rod_gap(far, 0, 1) == pytest.approx(gap, abs=1e-9)


def test_cells_without_flagella_never_tumble_so_a_still_rod_keeps_its_orientation():
    cfg = R.quiet(tumble_rate_per_s=30.0)
    w = swimmers(200, config=cfg, trait=R.traits(motility_um_s=0.0, max_uptake_mmol_s=SENSE_ONLY, division_rate_s=0.0, death_rate_s=0.0))
    w.step()
    before = w.sim.life.heading.copy()
    total = 0
    for _ in range(120):
        w.step()
        total += w.sim.stats["tumbles"]
    assert total == 0 and np.array_equal(w.sim.life.heading, before)
