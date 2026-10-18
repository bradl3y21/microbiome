"""The realistic tick as a whole: it runs, conserves every channel's mass, is deterministic and leaves D1 alone."""
import numpy as np
import pytest

import realism_support as R
from microbiome.sim.model import DT_S


def fingerprint(w):
    s = w.state
    return (s.agent_id.tobytes(), s.positions_um.tobytes(), s.biomass_mmol.tobytes(), s.energy_pj.tobytes(),
            w.env.concentration_mmol_l.tobytes(), s.respiration_loss_mmol.tobytes(), s.source_cum_mmol.tobytes(),
            s.sink_cum_mmol.tobytes(), w.sim.life.heading.tobytes(), w.sim.life.lag.tobytes())


def busy_world(seed=7):
    rng = np.random.default_rng(1)
    positions = rng.random((40, 2)) * [80.0, 80.0] + 60.0
    w = R.world(positions, nx=20, ny=20, seed=seed, glucose=0.5, lactose=0.2, mass=0.5)
    return w


def test_a_busy_world_runs_and_the_engines_own_ledger_assertion_never_fires():
    w = busy_world()
    w.step(300)                                  # the engine raises RuntimeError from assert_conserved on any leak
    assert w.n >= 40
    assert np.isfinite(w.env.concentration_mmol_l).all() and (w.env.concentration_mmol_l >= 0).all()
    led = w.state.respiration_loss_mmol
    assert led[R.GLC] > 0 and led[R.O2] > 0, "glucose is oxidised and oxygen consumed"


def test_same_seed_same_world_bit_for_bit_and_a_different_seed_differs():
    a, b, c = busy_world(7), busy_world(7), busy_world(8)
    for w in (a, b, c):
        w.step(120)
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)


def test_every_channel_closes_its_ledger_to_float_precision():
    w = busy_world()
    w.step(240)
    s, v = w.state, w.env.cell_volume_l
    for k in range(R.K):
        field = float(w.env.concentration_mmol_l[k].sum()) * v
        agents = float((s.nutrient_store_mmol[:, k] + s.biomass_mmol[:, k]).sum())
        detritus = float(w.env.detritus_mmol[k].sum())
        lhs = s.initial_mmol[k] + s.source_cum_mmol[k]
        rhs = field + agents + detritus + s.sink_cum_mmol[k] + s.respiration_loss_mmol[k]
        assert rhs == pytest.approx(lhs, rel=1e-9, abs=1e-18), f"channel {k}"


def test_with_no_cells_the_environment_still_evolves_and_conserves():
    w = R.world(((50.0, 50.0),), glucose=0.0)
    w.state = R.cells(np.empty((0, 2)))
    w.env.concentration_mmol_l[R.O2] = 0.05                        # below saturation: gas exchange must refill it
    w.step(600)
    assert w.n == 0
    assert w.env.concentration_mmol_l[R.O2].mean() > 0.05
    assert w.env.concentration_mmol_l[R.O2].max() <= 0.21 + 1e-12


def test_the_frozen_d1_step_is_not_affected_by_importing_the_realism_layer():
    from microbiome.sim.engine import Simulation

    w = R.world(((100.0, 100.0),), glucose=0.5)
    assert type(w.sim) is not Simulation
    assert Simulation.dt_s == DT_S
