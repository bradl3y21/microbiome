"""Realistic chemistry: every number below is derived by hand from the documented equations, then compared."""
import math

import numpy as np
import pytest

import realism_support as R

DT = 1.0 / 60.0
T = R.traits()
VMAX = {k: float(T.max_uptake_mmol_s[i]) for k, i in (("glc", R.GLC), ("lac", R.LAC), ("o2", R.O2), ("ace", R.ACE))}


def one_cell(*, phys=None, trait=None, **field):
    """One awake, fully 'learned' cell (lag 1), no gas exchange, nothing random: a clean uptake experiment."""
    cfg = R.quiet(lag_initial=1.0, gas_exchange_per_s=0.0)
    return R.world(((100.0, 100.0),), config=cfg, phys=phys, trait=trait, **field)


def lost(w, channel, ticks=1):
    before = w.field_mmol(channel)
    w.step(ticks)
    return before - w.field_mmol(channel)


def mm(s, km):
    return s / (km + s)


def test_uptake_saturates_as_michaelis_menten_half_rate_at_km():
    km = 0.01
    at_km = lost(one_cell(glucose=km), R.GLC)
    assert at_km == pytest.approx(VMAX["glc"] * DT * 0.5, rel=1e-9)
    low = lost(one_cell(glucose=km / 100), R.GLC)
    assert low == pytest.approx(VMAX["glc"] * DT * mm(km / 100, km), rel=1e-9)
    high = lost(one_cell(glucose=10.0), R.GLC)
    assert high == pytest.approx(VMAX["glc"] * DT * mm(10.0, km), rel=1e-9) and high < VMAX["glc"] * DT
    assert low < at_km < high, "uptake rises with concentration but flattens"
    assert lost(one_cell(glucose=0.1), R.GLC) / at_km == pytest.approx(mm(0.1, km) / 0.5, rel=1e-9)


def test_taken_up_glucose_is_split_into_store_biomass_and_respiration_exactly():
    w = one_cell(glucose=0.5)
    g = lost(w, R.GLC)
    overflow = 0.4 * (0.5 / (0.5 + 0.5))                           # acetate_overflow * sugar saturation
    assert g > 0
    assert w.state.nutrient_store_mmol[0, R.GLC] == pytest.approx(0.1 * g, rel=1e-9)
    gained = w.state.biomass_mmol[0, R.GLC] - 0.3 / (R.MOLAR[R.GLC] * 1e9)
    assert gained == pytest.approx(0.75 * g, rel=1e-9)
    assert w.state.respiration_loss_mmol[R.GLC] == pytest.approx(0.15 * g, rel=1e-9)
    assert w.field_mmol(R.ACE) == pytest.approx(0.7 * 3.0 * 0.15 * g * overflow, rel=1e-9), "aerobic acetate overflow"
    assert w.state.respiration_loss_mmol[R.O2] == pytest.approx(6.0 * 0.15 * g * (1 - overflow), rel=1e-9), "6 O2 per oxidised glucose"


def test_catabolite_repression_switches_lactose_off_while_glucose_is_present():
    alone = lost(one_cell(lactose=0.2), R.LAC)
    assert alone == pytest.approx(VMAX["lac"] * DT * mm(0.2, 0.2), rel=1e-9)
    with_glucose = lost(one_cell(lactose=0.2, glucose=0.5), R.LAC)
    repression = 1.0 - 1.0 * (0.5 / (0.5 + 0.05))
    assert with_glucose == pytest.approx(alone * repression, rel=1e-9)
    assert with_glucose < 0.1 * alone
    no_ccr = lost(one_cell(lactose=0.2, glucose=0.5, phys=R.physiology(ccr_strength=0.0)), R.LAC)
    assert no_ccr == pytest.approx(alone, rel=1e-9), "an organism without the regulator is not repressed"
    trace = lost(one_cell(lactose=0.2, glucose=1e-5), R.LAC)
    assert trace > 0.99 * alone, "traces of glucose do not repress"


def test_anoxic_growth_of_a_fermenter_has_low_yield_and_excretes_acetate():
    w = one_cell(glucose=0.5, oxygen=0.0)
    g = lost(w, R.GLC)
    respired = g - 0.1 * g - 0.75 * g * 0.3                         # yield falls to 30% of aerobic
    assert g == pytest.approx(VMAX["glc"] * DT * mm(0.5, 0.01), rel=1e-9), "a fermenter eats at full rate without oxygen"
    gained = w.state.biomass_mmol[0, R.GLC] - 0.3 / (R.MOLAR[R.GLC] * 1e9)
    assert gained == pytest.approx(0.75 * g * 0.3, rel=1e-9)
    assert w.state.respiration_loss_mmol[R.GLC] == pytest.approx(respired, rel=1e-9)
    assert w.field_mmol(R.ACE) == pytest.approx(0.7 * 3.0 * respired * 1.0, rel=1e-9), "all respired carbon is fermented"
    assert w.state.respiration_loss_mmol[R.O2] == 0.0


def test_a_strict_aerobe_cannot_grow_without_oxygen():
    w = one_cell(glucose=0.5, oxygen=0.0, phys=R.physiology(fermentation_capacity=0.0))
    assert lost(w, R.GLC, 30) == 0.0
    assert w.field_mmol(R.ACE) == 0.0


def test_more_oxygen_means_more_biomass_per_glucose_and_less_acetate():
    biomass_per_g, acetate_per_g, aerobic = [], [], []
    for o2 in (0.0, 1e-5, 1e-4, 1e-3, 0.21):
        w = one_cell(glucose=0.5, oxygen=o2)
        g = lost(w, R.GLC)
        gained = w.state.biomass_mmol[0, R.GLC] - 0.3 / (R.MOLAR[R.GLC] * 1e9)
        biomass_per_g.append(gained / g)
        acetate_per_g.append(w.field_mmol(R.ACE) / g)
        aerobic.append(w.sim.stats["aerobic_fraction_x1000"] / 1000)
    assert aerobic == sorted(aerobic) and aerobic[0] == 0.0 and aerobic[-1] == 1.0
    assert biomass_per_g == sorted(biomass_per_g) and acetate_per_g == sorted(acetate_per_g, reverse=True)
    assert biomass_per_g[0] == pytest.approx(0.75 * 0.3, rel=1e-9) and biomass_per_g[-1] == pytest.approx(0.75, rel=1e-9)


def chamber(n, *, exchange, seed=2):
    """A small box (4x4 grid cells of 10 um) full of cells, fully adapted, with plenty of glucose."""
    rng = np.random.default_rng(seed)
    cfg = R.quiet(lag_initial=1.0, gas_exchange_per_s=exchange)
    return R.world(rng.random((n, 2)) * 40.0, nx=4, ny=4, config=cfg, energy=1e9, glucose=5.0)


def test_a_sealed_chamber_runs_out_of_oxygen_and_the_cells_switch_to_fermentation():
    sealed = chamber(120, exchange=0.0)
    sealed.step(4000)
    assert sealed.env.concentration_mmol_l[R.O2].mean() < 0.02, "the closed oxygen pool is used up"
    assert sealed.sim.stats["aerobic_fraction_x1000"] < 500, "most cells can no longer respire"
    assert sealed.field_mmol(R.ACE) > 0
    open_ = chamber(120, exchange=0.0)
    open_.step(10)
    assert open_.sim.stats["aerobic_fraction_x1000"] == 1000, "at the start, with oxygen, everyone respires"


def test_an_open_thin_film_stays_aerobic_because_oxygen_is_replenished_and_diffuses_fast():
    film = chamber(40, exchange=0.05)
    film.step(4000)
    assert film.env.concentration_mmol_l[R.O2].mean() > 0.05
    assert film.sim.stats["aerobic_fraction_x1000"] == 1000
    lone = R.world(((105.0, 105.0),), config=R.quiet(lag_initial=1.0), energy=1e9, glucose=0.5)
    lone.step(1500)
    assert lone.env.concentration_mmol_l[R.O2][10, 10] > 0.2


def test_acetate_is_eaten_only_when_glucose_is_gone_the_diauxic_shift():
    alone = lost(one_cell(acetate=1.0), R.ACE)
    assert alone == pytest.approx(VMAX["ace"] * DT * mm(1.0, 0.3), rel=1e-9)
    with_glucose = lost(one_cell(acetate=1.0, glucose=0.5), R.ACE)
    assert with_glucose == pytest.approx(alone * (1.0 - 0.5 / 0.55), rel=1e-9)
    w = one_cell(acetate=1.0)
    g = lost(w, R.ACE)
    assert w.state.biomass_mmol[0, R.ACE] == pytest.approx(0.5 * g, rel=1e-9)
    assert w.state.respiration_loss_mmol[R.O2] == pytest.approx(2.0 * (g - 0.1 * g - 0.5 * g), rel=1e-9), "2 O2 per acetate"


def test_acetate_never_holds_more_carbon_than_the_cells_respired():
    rng = np.random.default_rng(4)
    for o2 in (0.0, 0.01, 0.21):
        w = R.world(100.0 + rng.random((50, 2)) * 20.0, energy=1e9, glucose=0.5, lactose=0.2, oxygen=o2,
                    config=R.quiet(gas_exchange_per_s=0.0, lag_initial=1.0))
        w.step(300)
        s = w.state
        made = s.source_cum_mmol[R.ACE]
        assert made > 0
        assert 2.0 * made <= 6.0 * s.respiration_loss_mmol[R.GLC] + 12.0 * s.respiration_loss_mmol[R.LAC] + 1e-30, f"o2={o2}"


def test_acid_slows_growth_for_a_sensitive_organism_in_proportion():
    neutral = lost(one_cell(glucose=0.5), R.GLC)
    tolerant = lost(one_cell(glucose=0.5, acetate=180.0), R.GLC)
    sensitive = lost(one_cell(glucose=0.5, acetate=180.0, phys=R.physiology(ph_min=5.8)), R.GLC)
    assert 7.0 - math.log10(1 + 180.0 / 20.0) == pytest.approx(6.0)
    f = lambda x: (1 - math.exp(-3 * x)) / (1 - math.exp(-3))
    assert tolerant / neutral == pytest.approx(f((6.0 - 4.4) / (6.8 - 4.4)), rel=1e-6)
    assert sensitive / neutral == pytest.approx(f((6.0 - 5.8) / (6.8 - 5.8)), rel=1e-6)
    assert tolerant / neutral > 0.9, "an acid-tolerant organism barely notices pH 6"
    assert sensitive < tolerant


def dead_fraction(w, ticks):
    n = w.n
    w.step(ticks)
    return 1.0 - w.n / n


def crowd(n, *, seed=6, phys=None, config=None, **field):
    rng = np.random.default_rng(seed)
    cfg = config or R.quiet(lag_initial=1.0)
    return R.world(100.0 + rng.random((n, 2)) * 60.0, config=cfg, phys=phys, energy=1e9,
                   trait=R.traits(division_rate_s=0.0, max_uptake_mmol_s=np.zeros(6), death_rate_s=0.0), **field)


def test_strong_acid_kills_a_sensitive_organism_and_spares_an_acid_resistant_one():
    cfg = R.quiet(acid_hazard_per_s_per_ph=0.05)
    ph = 7.0 - math.log10(1 + 2000.0 / 20.0)
    hazard = 0.05 * (5.8 - ph)
    sensitive = crowd(400, phys=R.physiology(ph_min=5.8), config=cfg, acetate=2000.0)
    resistant = crowd(400, phys=R.physiology(ph_min=4.4), config=cfg, acetate=2000.0)
    assert dead_fraction(sensitive, 600) == pytest.approx(1 - math.exp(-hazard * 10.0), abs=0.07)
    assert dead_fraction(resistant, 600) < 0.01
    assert sensitive.sim.stats.get("deaths_acid", 0) >= 0


def test_toxin_kills_at_a_hill_dose_response_and_immunity_protects():
    cfg = R.quiet(toxin_max_hazard_per_s=0.2)
    sus = crowd(400, config=cfg, toxin=1e-3)
    imm = crowd(400, config=cfg, toxin=1e-3, phys=R.physiology(toxin_susceptibility=0.1))
    none = crowd(400, config=cfg)
    h = 0.2 * (10.0 ** 2 / (1 + 10.0 ** 2))
    assert dead_fraction(sus, 300) == pytest.approx(1 - math.exp(-h * 5.0), abs=0.08)
    assert dead_fraction(imm, 300) == pytest.approx(1 - math.exp(-0.1 * h * 5.0), abs=0.05)
    assert dead_fraction(none, 300) < 0.01
    half = crowd(400, config=cfg, toxin=1e-4)                     # at K the hazard is exactly half of the maximum
    assert dead_fraction(half, 300) == pytest.approx(1 - math.exp(-0.1 * 5.0), abs=0.08)


def test_a_toxin_producer_charges_the_field_and_quorum_signal_boosts_production():
    n, ticks = 20, 60
    base = R.world(100.0 + np.random.default_rng(1).random((n, 2)) * 5.0, config=R.quiet(collisions=False),
                   energy=1e9, trait=R.traits(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6)),
                   phys=R.physiology(toxin_production=1.0, toxin_susceptibility=0.0, qs_capacity=0.0))
    base.step(ticks)
    assert base.state.source_cum_mmol[R.TOX] == pytest.approx(n * 5e-17 * 0.2 * ticks * DT, rel=1e-9)
    assert base.field_mmol(R.TOX) == pytest.approx(base.state.source_cum_mmol[R.TOX] - base.state.sink_cum_mmol[R.TOX], rel=1e-9)
    boosted = R.world(100.0 + np.random.default_rng(1).random((n, 2)) * 5.0, config=R.quiet(collisions=False),
                      energy=1e9, signal=2e-3, trait=R.traits(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6)),
                      phys=R.physiology(toxin_production=1.0, toxin_susceptibility=0.0, qs_capacity=1.0))
    boosted.step(ticks)
    activation = (2e-3 / 2e-4) ** 2 / (1 + (2e-3 / 2e-4) ** 2)
    assert boosted.state.source_cum_mmol[R.TOX] == pytest.approx(n * 5e-17 * (0.2 + 0.8 * activation) * ticks * DT, rel=2e-3)


def test_toxin_signal_and_oxygen_follow_their_first_order_time_constants_exactly():
    w = R.world(((50.0, 50.0),), config=R.quiet(), toxin=1e-3, signal=5e-4)
    w.state = R.cells(np.empty((0, 2)))
    cfg = w.sim.config
    w.env.concentration_mmol_l[R.O2] = 0.05
    w.step(600)                                                   # 10 s
    decay = lambda rate: math.exp(-rate * 10.0)
    assert w.env.concentration_mmol_l[R.TOX].mean() == pytest.approx(1e-3 * decay(cfg.toxin_decay_per_s), rel=1e-9)
    assert w.env.concentration_mmol_l[R.SIG].mean() == pytest.approx(5e-4 * decay(cfg.signal_decay_per_s), rel=1e-9)
    k = cfg.gas_exchange_per_s * DT
    assert w.env.concentration_mmol_l[R.O2].mean() == pytest.approx(0.21 - (0.21 - 0.05) * (1 - k) ** 600, rel=1e-9)
    assert w.state.sink_cum_mmol[R.TOX] == pytest.approx((1e-3 - w.env.concentration_mmol_l[R.TOX].mean()) * 400 * w.env.cell_volume_l, rel=1e-6)


def test_dead_cells_return_to_the_field_as_recyclable_nutrient():
    w = R.world(((50.0, 50.0),), config=R.quiet(), glucose=0.0)
    w.state = R.cells(np.empty((0, 2)))
    w.step(1)
    w.env.detritus_mmol[R.GLC][5, 5] = 1e-12
    w.state.initial_mmol[R.GLC] += 1e-12                          # the necromass entered through the ledger's initial total
    total = 1e-12
    w.step(3600)                                                  # one minute
    kept = total * math.exp(-w.sim.config.detritus_mineralization_per_s * 60.0)
    assert w.env.detritus_mmol[R.GLC].sum() == pytest.approx(kept, rel=1e-9)
    assert w.field_mmol(R.GLC) == pytest.approx(total - kept, rel=1e-6)


def test_the_quorum_signal_builds_the_biofilm_matrix_at_the_documented_rate():
    cells = np.array([[100.0, 100.0]])
    w = R.world(cells, config=R.quiet(), signal=2e-4, energy=1e9,
                trait=R.traits(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6)))
    w.step(600)
    assert w.sim.life.eps[0] == pytest.approx(0.5 * (1 - math.exp(-10.0 / 900.0)), rel=0.01)
    none = R.world(cells, config=R.quiet(), energy=1e9, trait=R.traits(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6)))
    none.step(600)
    assert none.sim.life.eps[0] < 1e-6, "a lone cell's own trace of signal builds no matrix"
    deaf = R.world(cells, config=R.quiet(), signal=2e-4, energy=1e9, phys=R.physiology(qs_capacity=0.0),
                   trait=R.traits(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6)))
    deaf.step(600)
    assert deaf.sim.life.eps[0] == 0.0, "an organism without the sensing genes does not respond"


def test_cells_make_signal_in_proportion_to_their_number_and_awake_time():
    for n in (5, 20):
        w = R.world(100.0 + np.random.default_rng(1).random((n, 2)) * 5.0, config=R.quiet(), energy=1e9,
                    trait=R.traits(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6)))
        w.step(120)
        assert w.state.source_cum_mmol[R.SIG] == pytest.approx(n * 1e-18 * 120 * DT, rel=1e-9)


def test_matrix_protects_against_toxin():
    cfg = R.quiet(toxin_max_hazard_per_s=0.2, eps_fall_tau_s=1e9)
    bare, coated = crowd(400, config=cfg, toxin=1e-3), crowd(400, config=cfg, toxin=1e-3)
    for w in (bare, coated):
        w.step(1)
    coated.sim.life.eps[:] = 1.0
    h = 0.2 * (100.0 / 101.0)
    assert dead_fraction(bare, 300) == pytest.approx(1 - math.exp(-h * 5.0), abs=0.08)
    assert dead_fraction(coated, 300) == pytest.approx(1 - math.exp(-0.3 * h * 5.0), abs=0.06)
