"""Cell lifecycle: lag, maintenance, starvation, sporulation and germination, growth and division."""
import math

import numpy as np
import pytest

import realism_support as R

DT = 1.0 / 60.0
T = R.traits()
YIELD = [1e15, 1e15, 3.3e14]
MAINTENANCE = R.maintenance_pj_s(0.3)        # pJ/s for the 0.3 pg cell these tests use (0.4 mmol glucose/gDW/h)


def inert(**changes):
    base = dict(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6))
    base.update(changes)
    return R.traits(**base)


def test_lag_recovers_exponentially_toward_full_readiness_when_food_is_present():
    tau = 20.0
    w = R.world(((100.0, 100.0),), config=R.quiet(lag_recovery_tau_s=tau), glucose=0.5, energy=1e9, trait=inert())
    w.step(600)
    assert w.sim.life.lag[0] == pytest.approx(1.0 - 0.95 * (1.0 - DT / tau) ** 600, rel=1e-9)
    w.step(3000)                                                   # 60 s in all: three time constants
    assert w.sim.life.lag[0] == pytest.approx(1.0 - 0.95 * (1.0 - DT / tau) ** 3600, rel=1e-9)
    assert w.sim.life.lag[0] > 0.95


def test_lag_is_slowly_lost_again_without_food():
    tau = 10800.0
    w = R.world(((100.0, 100.0),), config=R.quiet(lag_initial=1.0), glucose=0.0, energy=1e9, trait=inert())
    w.step(600)
    assert w.sim.life.lag[0] == pytest.approx((1.0 - DT / tau) ** 600, rel=1e-9)


def test_a_cell_in_lag_takes_up_proportionally_less_than_an_adapted_one():
    def first_tick_uptake(lag):
        w = R.world(((100.0, 100.0),), config=R.quiet(lag_initial=lag, gas_exchange_per_s=0.0), glucose=0.5)
        before = w.field_mmol(R.GLC)
        w.step()
        return before - w.field_mmol(R.GLC)
    ratio = first_tick_uptake(0.05) / first_tick_uptake(1.0)
    assert ratio == pytest.approx(0.05 + 0.95 * DT / 1200.0, rel=1e-6)


def test_a_new_cell_gets_an_energy_reserve_and_pays_maintenance_from_it():
    w = R.world(((100.0, 100.0),), config=R.quiet(), glucose=0.0, phys=R.maintaining())
    w.step()
    assert w.state.energy_pj[0] == pytest.approx(MAINTENANCE * 7200.0 - MAINTENANCE * DT, rel=1e-9)
    w.step(599)
    assert w.state.energy_pj[0] == pytest.approx(MAINTENANCE * (7200.0 - 600 * DT), rel=1e-9)
    kept = R.world(((100.0, 100.0),), config=R.quiet(), glucose=0.0, energy=123.0, phys=R.maintaining())
    kept.step()
    assert kept.state.energy_pj[0] == pytest.approx(123.0 - MAINTENANCE * DT, rel=1e-9), "an explicit energy is respected"


def test_when_the_reserve_runs_out_stored_substrate_is_burned_and_booked_as_respiration():
    w = R.world(((100.0, 100.0),), config=R.quiet(initial_energy_reserve_s=0.0), glucose=0.0, energy=0.0,
                phys=R.maintaining(), trait=R.traits(division_rate_s=0.0, death_rate_s=0.0))
    w.state.nutrient_store_mmol[0, R.GLC] = 1e-15
    w.step()
    burned = MAINTENANCE * DT / 1e15
    assert w.state.energy_pj[0] == pytest.approx(0.0, abs=1e-12), "the burned store exactly covers the shortfall"
    assert w.state.energy_pj[0] >= 0.0
    assert w.state.nutrient_store_mmol[0, R.GLC] == pytest.approx(1e-15 - burned, rel=1e-9)
    assert w.state.respiration_loss_mmol[R.GLC] == pytest.approx(burned, rel=1e-9)


def test_starvation_builds_stress_one_second_per_second_and_is_relieved_by_food():
    w = R.world(((100.0, 100.0),), config=R.quiet(initial_energy_reserve_s=0.0), glucose=0.0, energy=0.0, trait=inert(),
                phys=R.maintaining())
    w.step(300)
    assert w.sim.life.stress_s[0] == pytest.approx(300 * DT, rel=1e-9)
    w.state.energy_pj[:] = 5.0                                    # fed: stress decays three times faster than it grew
    w.step(30)
    assert w.sim.life.stress_s[0] == pytest.approx(300 * DT - 3 * 30 * DT, rel=1e-6)


def survival(resistance, n=400, seconds=20.0):
    cfg = R.quiet(initial_energy_reserve_s=0.0, starvation_tau_s=10.0, starvation_hazard_per_s=0.5)
    rng = np.random.default_rng(8)
    w = R.world(100.0 + rng.random((n, 2)) * 60.0, config=cfg, glucose=0.0, energy=0.0,
                phys=R.maintaining(stress_resistance=resistance), trait=R.traits(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6)))
    w.step(int(seconds * 60))
    return w.n / n, w


def test_starvation_death_follows_the_analytic_curve_and_stress_resistance_protects():
    tolerant, w = survival(1.0)
    fragile, _ = survival(0.0)
    cumulative = 0.5 * 0.2 * (20.0 - 10.0 * (1.0 - math.exp(-2.0)))      # integral of hazard*(1-0.8*resistance)*(1-e^(-t/tau))
    assert tolerant == pytest.approx(math.exp(-cumulative), abs=0.07)
    assert fragile < 0.03
    assert w.sim.stats.get("starving", 0) > 0 or tolerant < 1.0


def sporulators(n=100, **cfg):
    config = R.quiet(initial_energy_reserve_s=0.0, starvation_tau_s=5.0, sporulation_time_s=30.0,
                     germination_rate_per_s=2.0, **cfg)
    rng = np.random.default_rng(3)
    return R.world(100.0 + rng.random((n, 2)) * 60.0, config=config, glucose=0.0, energy=0.0,
                   phys=R.maintaining(sporulation_capacity=1.0), trait=inert())


def test_starving_cells_that_can_sporulate_become_dormant_spores_on_the_documented_schedule():
    w = sporulators()
    w.step(600)                                                    # 10 s: stress ~0.86, still below the 0.9 commitment gate
    assert w.sim.life.spore.max() == 0.0
    w.step(2400)                                                   # by ~41.5 s they are spores
    assert w.sim.stats["spores"] == w.n == 100
    assert (w.sim.life.spore == 1.0).all()


def test_quorum_signal_brings_commitment_forward():
    quorum = sporulators(); quorum.set_field(R.SIG, 2e-3)
    alone = sporulators()
    quorum.step(500); alone.step(500)                              # 8.3 s: stress is 0.81, past 0.6 but short of 0.9
    assert quorum.sim.life.spore.max() > 0.0 and alone.sim.life.spore.max() == 0.0


def test_a_spore_is_immobile_does_not_eat_does_not_divide_and_survives_what_kills_vegetative_cells():
    w = sporulators()
    w.step(3000)
    assert w.sim.stats["spores"] == 100
    control = R.world(w.state.positions_um.copy(), config=R.quiet(initial_energy_reserve_s=0.0, starvation_tau_s=5.0), glucose=0.0, energy=0.0,
                      phys=R.maintaining(sporulation_capacity=0.0), trait=inert())
    lethal = R.traits(division_rate_s=0.0, death_rate_s=0.5, max_uptake_mmol_s=R.traits().max_uptake_mmol_s, division_mass_pg=0.01)
    w.phenotypes = {R.GENOME: lethal}
    control.phenotypes = {R.GENOME: lethal}
    w.set_field(R.GLC, 0.005)                                      # food, but below the germination threshold
    before_positions = w.state.positions_um.copy()
    before_biomass = w.state.biomass_mmol.copy()
    before_sugar = w.field_mmol(R.GLC)
    w.step(600)
    control.step(600)
    assert np.allclose(w.state.positions_um, before_positions), "spores do not swim"
    # exact comparisons: np.allclose's default atol (1e-8) is far larger than these ~1e-12 mmol quantities
    assert np.array_equal(w.state.biomass_mmol, before_biomass), "spores do not eat or grow"
    assert w.field_mmol(R.GLC) == pytest.approx(before_sugar, rel=1e-9), "and the sugar around them is untouched"
    assert 97 <= w.n <= 100, "spores do not divide, even above the division mass, and shrug off the hazard (x0.001)"
    assert control.n < 5, "the same hazard (0.5 per second) wipes out vegetative cells"


def test_a_spore_germinates_when_sugar_appears_and_restarts_in_lag():
    w = sporulators()
    w.step(3000)
    assert w.sim.stats["spores"] == 100
    w.phenotypes = {R.GENOME: R.traits(division_rate_s=0.0, death_rate_s=0.0)}   # cells that can use sugar, so they have a maintenance budget
    w.set_field(R.GLC, 0.5)
    w.step(300)
    assert (w.sim.life.spore == 0.0).all(), "at 2 per second every spore has germinated within 5 s"
    assert w.sim.life.lag.max() < 0.2, "germinated cells begin in lag"
    assert (w.state.energy_pj > 0).all()


def test_an_organism_without_the_sporulation_genes_never_sporulates():
    w = R.world(100.0 + np.random.default_rng(3).random((50, 2)) * 60.0,
                config=R.quiet(initial_energy_reserve_s=0.0, starvation_tau_s=5.0, sporulation_time_s=30.0),
                glucose=0.0, energy=0.0, phys=R.maintaining(sporulation_capacity=0.0), trait=inert())
    w.step(3000)
    assert w.sim.life.spore.max() == 0.0 and w.sim.stats["spores"] == 0


def test_growth_raises_mass_and_radius_and_the_first_division_waits_for_the_threshold():
    w = R.world(((100.0, 100.0),), config=R.quiet(lag_initial=1.0), glucose=0.5, mass=0.59,
                trait=R.traits(death_rate_s=0.0))
    rate = float(T.max_uptake_mmol_s[R.GLC]) * (0.5 / 0.51) * 0.75 * R.MOLAR[R.GLC] * 1e9 / 0.3   # pg per pg per second
    radius0 = float(w.state.radius_um[0])
    reach = math.log(0.6 / 0.59) / rate                                  # exponential growth: uptake is proportional to mass
    ticks, previous = 0, None
    while w.n == 1 and ticks < 30000:
        previous = float(w.state.biomass_mmol.sum())
        w.step()
        ticks += 1
    assert w.n == 2
    assert ticks * DT >= 0.97 * reach, "no division before the mass threshold is reached"
    assert ticks * DT <= reach + 8 * 30.0, "and it follows within a few commitment times"
    after = float(w.state.biomass_mmol.sum())
    assert 0.0 < after - previous < 3e-17, "the division tick conserved the parent's biomass up to one tick of uptake"
    assert w.state.radius_um[0] == pytest.approx(float(np.cbrt(3 * w.state.mass_pg[0] / (4 * np.pi))))
    assert w.state.mass_pg.sum() > 0.59


def test_daughters_inherit_the_parents_physiological_state_and_the_arrays_stay_aligned():
    w = R.world(((100.0, 100.0),), config=R.quiet(lag_initial=1.0, division_commitment_s=0.5), glucose=0.5, mass=0.7,
                trait=R.traits(death_rate_s=0.0))
    w.step(1)
    w.sim.life.eps[:] = 0.4
    w.sim.life.stress_s[:] = 0.0
    while w.n == 1:
        w.step()
    life = w.sim.life
    assert len(life.ids) == w.n == 2 and (life.ids == w.state.agent_id).all()
    for name in ("heading", "memory", "lag", "stress_s", "spore", "eps"):
        assert getattr(life, name).shape == (2,), name
    assert life.eps[0] == pytest.approx(life.eps[1]) and life.lag[0] == pytest.approx(life.lag[1])
    assert life.heading[0] != life.heading[1], "the daughter faces a new direction"


def test_a_cell_still_in_lag_does_not_divide_even_above_the_division_mass():
    def divisions(lag):
        w = R.world(((100.0, 100.0),), config=R.quiet(lag_initial=lag, division_commitment_s=0.5), glucose=0.5, mass=0.7,
                    trait=R.traits(death_rate_s=0.0))
        w.step(600)
        return w.n
    assert divisions(1.0) == 2 and divisions(0.05) == 1


def diauxic_cell(tau_rise=20.0, glucose=0.2):
    cfg = R.quiet(lag_initial=1.0, gas_exchange_per_s=0.0, derepression_rise_tau_s=tau_rise)
    return R.world(((100.0, 100.0),), config=cfg, glucose=glucose, lactose=0.2, energy=1e9,
                   trait=R.traits(division_rate_s=0.0, death_rate_s=0.0))


def test_diauxic_lag_lactose_genes_switch_on_slowly_after_glucose_is_gone():
    tau = 20.0
    w = diauxic_cell(tau)
    w.step(1)
    d0 = 1.0 - 0.2 / (0.2 + 0.05)                                  # pre-adapted to glucose: 20% expressed
    assert w.sim.life.derepression[0] == pytest.approx(d0, rel=1e-9)
    w.set_field(R.GLC, 0.0)                                       # the glucose is exhausted
    potential = float(T.max_uptake_mmol_s[R.LAC]) * DT * 0.5      # full lactose uptake at 0.2 mmol/L
    seen = {}
    for n in (1, 1200, 6000):                                     # right away, one time constant, five time constants
        w.step(n - sum(seen) if seen else n)
        seen[n] = w.sim.life.derepression[0]
    assert seen[1] == pytest.approx(1.0 - (1.0 - d0) * (1.0 - DT / tau), rel=1e-9)
    assert seen[1200] == pytest.approx(1.0 - (1.0 - d0) * (1.0 - DT / tau) ** 1200, rel=1e-9)
    assert seen[1] < 0.25 and 0.65 < seen[1200] < 0.75 and seen[6000] > 0.99, "a delay of order the time constant, then full expression"
    before = w.field_mmol(R.LAC)
    w.step()
    uptake = before - w.field_mmol(R.LAC)
    assert uptake == pytest.approx(potential * w.sim.life.derepression[0], rel=1e-3), "uptake tracks enzyme level, not glucose"


def test_a_new_dose_of_glucose_shuts_the_alternative_sugar_off_much_faster_than_it_switched_on():
    w = diauxic_cell(20.0, glucose=0.0)
    w.step(1)
    assert w.sim.life.derepression[0] == pytest.approx(1.0)
    w.set_field(R.GLC, 5.0)
    target = 1.0 - 5.0 / (5.0 + 0.05)
    w.step(600)                                                    # 10 s with the default 30 s fall constant
    expected = target + (1.0 - target) * (1.0 - DT / 30.0) ** 600
    assert w.sim.life.derepression[0] == pytest.approx(expected, rel=2e-3)
    assert w.sim.life.derepression[0] < 0.75, "already substantially repressed after 10 s, versus ~10 min to switch on"


def test_an_organism_without_a_catabolite_regulator_has_no_diauxic_lag():
    w = R.world(((100.0, 100.0),), config=R.quiet(lag_initial=1.0, gas_exchange_per_s=0.0), glucose=5.0, lactose=0.2, energy=1e9,
                phys=R.physiology(ccr_strength=0.0), trait=R.traits(division_rate_s=0.0, death_rate_s=0.0))
    w.step(60)
    assert w.sim.life.derepression[0] == pytest.approx(1.0)


def test_daughters_lie_end_to_end_along_the_mothers_axis_each_the_length_of_half_the_cell():
    from microbiome.sim.realism import rod_length_um

    w = R.world(((100.0, 100.0),), config=R.quiet(lag_initial=1.0, division_commitment_s=0.5, collisions=False), glucose=0.5, mass=0.7,
                trait=R.traits(death_rate_s=0.0, motility_um_s=0.0))
    w.step(1)
    w.sim.life.heading[:] = 0.7
    mother = w.state.positions_um[0].copy()
    while w.n == 1:
        w.step()
    a, b = w.state.positions_um
    daughter = float(rod_length_um(w.state.mass_pg[:1], w.sim.config)[0])
    separation = b - a
    assert np.hypot(*separation) == pytest.approx(daughter, rel=1e-6), "end to end: centres one daughter-length apart"
    assert np.arctan2(separation[1], separation[0]) % np.pi == pytest.approx(0.7, abs=0.05), "along the mother's axis"
    assert ((a + b) / 2) == pytest.approx(mother, abs=0.05), "centred where the mother was"
    assert np.allclose(w.sim.life.heading, 0.7, atol=0.07), "both keep the axis"
    assert daughter < float(rod_length_um(np.array([0.7]), w.sim.config)[0]) * 0.6
