"""Pure kinetics of the realism layer, each checked against a hand calculation."""
import math

import numpy as np
import pytest

from microbiome.sim.realism import (RealismConfig, advect_upwind, hill, michaelis_menten, ph_from_acetate,
                                    ph_growth_factor, radius_from_mass, rod_length_um, sample_bilinear,
                                    segment_closest_points)
import realism_support as R


def test_michaelis_menten_is_half_saturated_at_km_and_monotone():
    km = np.array(0.01)
    assert michaelis_menten(np.array(0.01), km) == pytest.approx(0.5)
    assert michaelis_menten(np.array(0.0), km) == 0.0
    s = np.logspace(-6, 1, 50)
    v = michaelis_menten(s, km)
    assert (np.diff(v) > 0).all() and v[-1] < 1.0 and v[-1] > 0.99
    assert michaelis_menten(np.array(1e-4), km) == pytest.approx(1e-4 / (0.01 + 1e-4)), "linear at low concentration"


def test_hill_is_half_at_k_and_steeper_with_larger_n():
    assert hill(np.array(2e-4), 2e-4, 2.0) == pytest.approx(0.5)
    assert hill(np.array(0.0), 2e-4, 2.0) == 0.0
    x = np.array(4e-4)
    assert hill(x, 2e-4, 4.0) > hill(x, 2e-4, 2.0) > 0.5
    assert hill(np.array(-1.0), 1.0) == 0.0, "negative concentrations are clamped, never amplified"


def test_ph_follows_the_buffered_weak_acid_formula():
    cfg = RealismConfig()
    assert ph_from_acetate(np.array(0.0), cfg) == pytest.approx(7.0)
    assert ph_from_acetate(np.array(20.0), cfg) == pytest.approx(7.0 - math.log10(2.0))
    assert ph_from_acetate(np.array(180.0), cfg) == pytest.approx(7.0 - 1.0)
    acid = np.linspace(0, 200, 40)
    assert (np.diff(ph_from_acetate(acid, cfg)) < 0).all()


def test_growth_versus_ph_is_a_saturating_curve_with_the_documented_anchor_points():
    ph = np.array([4.4, 4.5, 5.0, 6.0, 6.8, 7.4])
    f = ph_growth_factor(ph, np.array(4.4), 6.8)
    assert f[0] == 0.0 and f[4] == pytest.approx(1.0) and f[5] == pytest.approx(1.0), "zero at the minimum, 1 at the optimum"
    assert (np.diff(f) > 0).all() or f[-1] == f[-2]
    assert 0.05 < f[1] < 0.2 and 0.45 < f[2] < 0.65 and f[3] > 0.88, "~10% at 4.5, ~55% at 5, ~90% at 6 for E. coli"
    sensitive = ph_growth_factor(ph, np.array(5.8), 6.8)
    assert (sensitive <= f + 1e-12).all() and sensitive[3] < f[3], "a higher pH minimum is strictly worse"


def test_bilinear_sampling_is_exact_at_centres_linear_between_and_periodic():
    env = R.environment(nx=8, ny=6, dx=10.0)
    ramp = np.tile(np.arange(8, dtype=float), (6, 1))                  # c = ix
    at = lambda x, y: float(sample_bilinear(ramp, np.array([[x, y]]), env)[0])
    assert at(25.0, 25.0) == pytest.approx(2.0)                        # the centre of cell ix=2
    assert at(30.0, 25.0) == pytest.approx(2.5)                        # halfway to the next centre
    assert at(5.0, 25.0) == pytest.approx(0.0)
    # x = 1 um lies between the last cell centre (75 um, c = 7, seen at -5 um after wrapping) and the first (5 um, c = 0):
    # weights 0.4 and 0.6, so the periodic value is 0.4 * 7 + 0.6 * 0 = 2.8, not an edge-clamped 0
    assert at(1.0, 25.0) == pytest.approx(2.8)
    assert sample_bilinear(np.full((6, 8), 3.5), np.random.default_rng(0).random((50, 2)) * [80, 60], env) == pytest.approx(3.5)


def test_radius_matches_the_engines_mass_convention():
    assert radius_from_mass(np.array(0.3)) == pytest.approx(0.4153, abs=1e-3), "the 0.415 um of an introduced founder"
    m = np.array([0.1, 0.3, 0.6])
    assert (np.diff(radius_from_mass(m)) > 0).all()
    assert radius_from_mass(np.array(0.0)) == 0.0


def test_advection_conserves_mass_and_moves_a_blob_at_the_flow_speed():
    env = R.environment(nx=40, ny=10, dx=10.0)
    field = np.zeros((1, 10, 40))
    field[0, :, 5] = 1.0
    ux = 120.0                                                         # um/s, so 2 um per tick: CFL 0.2
    total = field.sum()
    steps = 60                                                         # 1 s -> the blob moves 120 um = 12 cells
    for _ in range(steps):
        field = advect_upwind(field, ux, 0.0, env, 1 / 60)
        assert field.sum() == pytest.approx(total, rel=1e-12), "advection is conservative"
        assert field.min() >= -1e-15, "upwind never makes a negative concentration"
    centre = float((field[0, 0] * np.arange(40)).sum() / field[0, 0].sum())
    assert centre == pytest.approx(5 + 12, abs=0.3)
    left = advect_upwind(np.eye(10, 40)[None], -ux, 0.0, env, 1 / 60)
    assert left.sum() == pytest.approx(np.eye(10, 40).sum(), rel=1e-12)
    with pytest.raises(ValueError, match="CFL"):
        advect_upwind(field, 700.0, 0.0, env, 1 / 60)


def test_advection_in_y_conserves_too():
    env = R.environment(nx=10, ny=30, dx=10.0)
    field = np.random.default_rng(3).random((2, 30, 10))
    out = advect_upwind(field, 0.0, -90.0, env, 1 / 60)
    assert out.sum(axis=(1, 2)) == pytest.approx(field.sum(axis=(1, 2)), rel=1e-12)
    assert not np.allclose(out, field)


def test_a_rod_elongates_as_it_grows_and_its_volume_is_the_cells_volume():
    cfg = RealismConfig()
    mass = np.array([0.3, 0.45, 0.6])
    length = rod_length_um(mass, cfg)
    w = cfg.cell_width_um
    assert length == pytest.approx([2.236, 3.22, 4.206], abs=0.01), "a 0.3 pg founder ~2.2 um, a cell about to divide ~4.2 um"
    volume = np.pi * w ** 2 / 4 * (length - w) + np.pi * w ** 3 / 6                     # cylinder plus two hemispherical caps
    assert volume == pytest.approx(cfg.cell_volume_um3_per_pg * mass, rel=1e-9)
    assert (np.diff(length) > 0).all()
    tiny = rod_length_um(np.array([0.0, 0.01, 0.05]), cfg)
    assert (tiny == w).all(), "below the volume of a sphere of the cell's width it is a sphere, never shorter than it is wide"


def test_the_closest_points_of_two_segments_agree_with_brute_force_in_every_configuration():
    rng = np.random.default_rng(1)
    m = 600
    p1, d1, p2, d2 = (rng.normal(size=(m, 2)) for _ in range(4))
    d1[:40] = 0.0                                              # a point against a segment
    d2[40:80] = 0.0
    d2[80:160] = d1[80:160] * rng.uniform(0.3, 2.0, (80, 1))   # parallel
    p2[160:220] = p1[160:220] + d1[160:220] * rng.uniform(0.2, 1.5, (60, 1))
    d2[160:220] = d1[160:220] * rng.uniform(0.3, 2.0, (60, 1))  # collinear
    near1, near2 = segment_closest_points(p1, d1, p2, d2)
    t = np.linspace(0.0, 1.0, 1201)[:, None]
    for i in range(0, m, 3):
        a, b = p1[i] + t * d1[i], p2[i] + t * d2[i]
        brute = float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())
        mine = float(np.hypot(*(near2[i] - near1[i])))
        assert mine <= brute + 1e-9 and mine == pytest.approx(brute, abs=3e-3), i
    assert (np.abs(near1 - p1) <= np.abs(d1) + 1e-12).all(), "the points stay on their segments"
