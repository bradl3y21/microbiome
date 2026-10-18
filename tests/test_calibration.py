"""The realism layer against published single-organism data: integrity of the targets, live re-measurement of every one in the
simulation, the mechanisms that make those measurements possible (mass-proportional uptake, Pirt maintenance), and the
generated documentation.

The 'fit' targets agree by construction, so passing them only shows the committed fit still reproduces the code; the
'held_out' tests are the ones that could have failed.
"""
import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

import realism_support as R
from microbiome.calibration import experiments as X
from microbiome.calibration import report
from microbiome.calibration.rig import BS, EC, GLC, MX, PA, SA, context, default_config
from microbiome.calibration.targets import ROLES, load_targets
from microbiome.genome import physiology as P
from microbiome.genome.calibration import IDENTITY, Calibration, OrganismCalibration, load_calibration
from microbiome.genome.phenotype import derive_phenotype
from microbiome.genome.models import NutrientChannel
from microbiome.sim.engine import diffuse
from microbiome.sim.model import DT_S
from microbiome.sim.realism import RealismConfig

ROOT = Path(__file__).resolve().parent.parent
TS = load_targets(ROOT)
CAL = load_calibration(ROOT)
FIT = json.loads((ROOT / "data" / "calibration_fit.json").read_text(encoding="utf-8"))
CFG = default_config()
SHARED_KEYS = {"swim_speed_scale", "tumble_rate_per_s", "tumble_angle_mean_deg", "tumble_angle_sd_deg", "chemotaxis_gain",
               "reversal_rate_per_s", "predation_assimilation", "development_speed_factor", "aggregate_speed_ratio"}


def target(target_id):
    return TS[target_id]


def near(measured, target_id, relative=None):
    t = target(target_id)
    if relative is not None:
        return abs(measured - t.value) <= relative * abs(t.value)
    return abs(measured - t.value) <= t.tolerance


# ------------------------------------------------------------------------------------------ the targets file
def test_every_target_names_a_known_role_a_real_source_and_its_evidence():
    ids = [t.id for t in TS.targets]
    assert len(ids) == len(set(ids)) and len(ids) >= 25
    for t in TS.targets:
        assert t.role in ROLES, t.id
        assert t.raw.get("unit") and t.raw.get("quantity"), t.id
        if t.role == "assumed":
            assert t.raw.get("note") or t.raw.get("derivation"), f"{t.id}: an assumption must say what it assumes"
            continue
        assert t.raw["source"] in TS.sources, f"{t.id}: source missing"
        assert len(t.raw.get("evidence", "")) >= 10, f"{t.id}: no evidence recorded"
        assert t.value is not None or t.range is not None, t.id


def test_every_source_is_cited_and_says_how_it_was_read_and_any_doi_is_well_formed():
    for key, source in TS.sources.items():
        assert source["citation"] and source["read_via"], key
        if source.get("doi"):
            assert re.fullmatch(r"10\.\d{4,9}/\S+", source["doi"]), key


def test_every_fit_target_names_the_parameter_it_is_solved_for():
    parameters = {f.name for f in dataclasses.fields(OrganismCalibration)} | SHARED_KEYS
    for t in TS.with_role("fit"):
        assert t.raw.get("fits", "").split()[0] in parameters, f"{t.id}: {t.raw.get('fits')}"
    fitted = {t.raw["fits"].split()[0] for t in TS.with_role("fit")}
    assert {"uptake_scale", "biomass_scale", "overflow_scale", "division_mass_scale", "motility_scale", "chemotaxis_gain",
            "swim_speed_scale", "tumble_angle_mean_deg", "maintenance_mmol_glc_gdw_h"} <= fitted


def test_derived_targets_show_their_arithmetic_and_there_are_held_out_checks_of_every_kind():
    for t in TS.targets:
        if t.raw.get("derived"):
            assert len(t.raw["derivation"]) > 80, t.id
    assert len(TS.with_role("held_out")) >= 4 and len(TS.with_role("reference")) >= 3


def test_the_committed_fit_was_made_from_exactly_this_targets_file():
    assert FIT["targets_sha256"] == TS.sha256, "targets changed after the fit: run python -m microbiome.calibration fit"


def test_the_committed_fit_has_all_five_organisms_and_the_shared_parameters_and_the_code_defaults_equal_it():
    assert set(CAL.organisms) == {EC, BS, PA, SA, MX} and set(CAL.shared) == SHARED_KEYS
    for key, value in CAL.shared.items():
        assert getattr(RealismConfig(), key) == pytest.approx(value, rel=1e-3), f"RealismConfig.{key} is out of step with the fit"
    for cal in CAL.organisms.values():
        assert cal.uptake_scale > 0 and cal.biomass_scale > 0 and cal.maintenance_mmol_glc_gdw_h > 0
    assert all(r["ok"] is not False for r in FIT["report"]), [r["id"] for r in FIT["report"] if r["ok"] is False]


# ------------------------------------------------------------------------------------------ re-measured in the simulation
@pytest.fixture(scope="module")
def growth():
    return {acc: X.saturating_growth(acc, calibration=CAL, config=CFG, seconds=60.0) for acc in (EC, BS)}


def test_fit_targets_growth_rate_and_glucose_uptake_are_reproduced_for_both_organisms(growth):
    assert near(growth[EC]["growth_rate_per_h"], "ecoli.growth_rate.glucose_M9", 0.02)
    assert near(growth[EC]["q_glucose"], "ecoli.q_glucose.glucose_M9", 0.02)
    assert near(growth[BS]["growth_rate_per_h"], "bsub.growth_rate.glucose_M9", 0.02)
    assert near(growth[BS]["q_glucose"], "bsub.q_glucose.glucose_M9", 0.02)


def test_fit_targets_oxygen_use_of_e_coli_and_acetate_output_of_b_subtilis(growth):
    assert near(growth[EC]["q_oxygen"], "ecoli.q_oxygen.glucose_M9", 0.03)
    assert near(growth[BS]["q_acetate"], "bsub.q_acetate.glucose_M9", 0.03)


def test_held_out_the_b_subtilis_biomass_yield_agrees_with_the_measured_yield(growth):
    t = target("bsub.yield.glucose_M9")
    assert abs(growth[BS]["yield_g_per_g"] - t.value) <= 2 * t.sd, growth[BS]["yield_g_per_g"]


def test_the_mass_yield_on_glucose_is_physically_possible_where_the_old_value_was_not(growth):
    for acc in (EC, BS):
        assert 0.25 < growth[acc]["yield_g_per_g"] < 0.5, "no organism makes more than about half its weight in dry mass per gram of glucose"


def test_the_carbon_balance_closes_for_both_organisms_using_only_the_simulated_rates(growth):
    """Glucose carbon = biomass + acetate + the CO2 that goes with acetate + fully oxidised carbon (one O2 per carbon).

    The model's biomass is glucose-equivalent, so it is 40 % carbon by mass (6 C in 180.156 g/mol), not the 48 % of real dry
    matter; it recovers 70 % of overflow carbon as acetate and the rest leaves as CO2; about 2 % more is stored.
    """
    recovery = RealismConfig().acetate_carbon_recovery
    for acc in (EC, BS):
        g = growth[acc]
        biomass_c = g["growth_rate_per_h"] * (6.0 * 12.011 / 180.156) / 12.011 * 1000.0   # mmol C per gDW per h
        acetate_c = 2.0 * g["q_acetate"]
        accounted = biomass_c + acetate_c / recovery + g["q_oxygen"]
        assert accounted == pytest.approx(6.0 * g["q_glucose"], rel=0.05), (acc, accounted, 6.0 * g["q_glucose"])


def test_held_out_e_coli_makes_no_acetate_in_slow_glucose_limited_growth():
    curve = X.monod_curve(EC, [2e-4, 5e-4, 1e-3, 3e-3, 1e-2], seconds=30.0, calibration=CAL, config=CFG)
    mu = np.array([c["growth_rate_per_h"] for c in curve])
    fraction = np.array([c["acetate_carbon_fraction"] for c in curve])
    assert mu[0] < 0.3 < mu[-1]
    assert float(np.interp(0.3, mu, fraction)) < 0.02, "Kayser et al. 2005: no acetate below D = 0.3/h"
    assert (np.diff(mu) > 0).all() and (np.diff(fraction) >= -1e-9).all(), "growth and overflow rise with the sugar level"


def test_the_monod_half_saturation_lies_inside_the_published_range_and_follows_the_model_constant(growth):
    curve = X.monod_curve(EC, np.geomspace(1e-5, 30.0, 14), seconds=30.0, calibration=CAL, config=CFG)
    half = X.half_saturation(curve, growth[EC]["growth_rate_per_h"])
    lo, hi = target("ecoli.Ks.glucose").range
    assert lo <= half <= hi
    assert half == pytest.approx(P.KS_BASELINES[0], rel=0.25), "with maintenance the apparent Ks shifts a little above the transporter constant"


def test_held_out_pirt_line_the_simulated_chemostat_has_the_measured_maintenance_and_yield():
    line = X.pirt_line(BS, calibration=CAL, config=CFG, seconds=45.0)
    t = target("bsub.maintenance.chemostat_check")
    assert abs(line["maintenance_mmol_glc_gdw_h"] - t.value) <= t.sd
    yield_t = target("bsub.yield.glucose_M9")
    assert abs(line["yield_g_per_g"] - yield_t.value) <= 2 * yield_t.sd, "the slope of q against mu is 1/Y, as in Tannler et al."


@pytest.fixture(scope="module")
def walk():
    return X.random_walk(EC, calibration=CAL, config=CFG)


def test_fit_targets_swimming_speed_and_run_time_of_e_coli(walk):
    assert abs(walk["speed_um_s"] - target("ecoli.swim_speed").value) < 1.5
    assert 1.0 / walk["tumble_rate_per_s"] == pytest.approx(target("ecoli.run_time").value, rel=0.06)


def test_held_out_random_motility_follows_from_speed_run_time_and_angle(walk):
    t = target("ecoli.random_motility")
    assert abs(walk["diffusivity_3d_equivalent_um2_s"] - t.value) <= 2 * t.sd, walk
    assert walk["diffusivity_2d_um2_s"] * 2.0 / 3.0 == pytest.approx(walk["diffusivity_3d_equivalent_um2_s"])


def test_fit_targets_tumble_angle_distribution_and_the_b_subtilis_speed():
    angles = X.tumble_angles(EC, calibration=CAL, config=CFG)
    assert angles["mean_deg"] == pytest.approx(target("ecoli.tumble_angle").value, abs=1.5)
    assert angles["sd_deg"] == pytest.approx(target("ecoli.tumble_angle_sd").value, abs=1.5)
    speed = X.random_walk(BS, n_cells=40, seconds=15.0, calibration=CAL, config=CFG)["speed_um_s"]
    assert speed == pytest.approx(target("bsub.swim_speed").value, abs=0.8)


def test_chemotactic_drift_peaks_inside_the_published_range_and_vanishes_without_a_gradient():
    peak = X.chemotaxis_peak(EC, calibration=CAL, config=CFG)
    low, high = target("ecoli.chemotactic_velocity").range
    assert low <= peak["peak_drift_fraction"] <= high
    assert peak["peak_drift_fraction"] == pytest.approx(target("ecoli.chemotactic_velocity").value, abs=0.04)
    flat = X.chemotactic_velocity(EC, period_um=500.0, gradient=False, calibration=CAL, config=CFG)
    assert abs(flat["drift_fraction"]) < 0.04
    steady = X.chemotactic_velocity(EC, period_um=500.0, calibration=CAL, config=CFG)
    assert steady["drift_um_s"] > 3.0


@pytest.mark.parametrize("accession", [EC, BS])
def test_a_cell_close_to_division_divides_when_growth_and_commitment_predict(accession):
    r = X.division_timing(accession, start_fraction=0.9, n_cells=4, calibration=CAL, config=CFG)
    assert r["divisions"] == 4
    assert r["mean_division_time_s"] == pytest.approx(r["expected_s"], rel=0.06), r


# ------------------------------------------------------------------------------------------ the mechanisms behind them
def quiet_world(**kw):
    kw.setdefault("config", R.quiet(lag_initial=1.0, gas_exchange_per_s=0.0))
    return R.world(**kw)


def test_uptake_is_proportional_to_dry_mass():
    eaten = {}
    for mass in (0.15, 0.3, 0.6):
        w = quiet_world(positions=((100.0, 100.0),), glucose=0.5, mass=mass)
        before = w.field_mmol(R.GLC)
        w.step()
        eaten[mass] = before - w.field_mmol(R.GLC)
    assert eaten[0.6] / eaten[0.3] == pytest.approx(2.0, rel=1e-6)          # the totals are ~1e-10 and the difference ~1e-17: 1e-9 is float noise
    assert eaten[0.15] / eaten[0.3] == pytest.approx(0.5, rel=1e-6)


def test_relative_growth_rate_does_not_depend_on_size_so_growth_is_exponential():
    w = quiet_world(positions=((50.0, 50.0), (150.0, 150.0)), glucose=100.0, mass=0.1)
    w.state.mass_pg[1] *= 4.0
    w.state.biomass_mmol[1] *= 4.0
    start = w.state.mass_pg.copy()
    w.step(600)
    relative = (w.state.mass_pg - start) / start
    assert relative[0] > 1e-3
    assert relative[1] == pytest.approx(relative[0], rel=1e-6), "a four times larger cell grows four times faster in absolute terms"


AMPLE_OXYGEN = R.traits(max_uptake_mmol_s=np.array([8.88099467e-16, 4.67e-16, 3.0e-13, 6.0e-16, 0.0, 0.0]))   # never oxygen-limited


def pirt_tick(m, mass=0.3, energy=None):
    w = quiet_world(positions=((100.0, 100.0),), glucose=100.0, mass=mass, energy=energy, trait=AMPLE_OXYGEN,
                    phys=R.physiology(maintenance_mmol_glc_gdw_h=m))
    initial = float(w.state.biomass_mmol[0, R.GLC])
    w.step()
    request = float(AMPLE_OXYGEN.max_uptake_mmol_s[R.GLC]) * DT_S * (100.0 / 100.01) * (mass / 0.3)
    energy_in = request * 1e15
    need = R.maintenance_pj_s(mass, m) * DT_S
    return w, initial, request, energy_in, need


def test_pirt_maintenance_is_taken_from_what_the_cell_eats_before_it_grows():
    w, initial, request, energy_in, need = pirt_tick(m=0.4)
    covered = min(1.0, need / energy_in)
    assert 0.0 < covered < 1.0
    assert w.state.biomass_mmol[0, R.GLC] - initial == pytest.approx(0.75 * request * (1.0 - covered), rel=1e-6)
    assert w.state.nutrient_store_mmol[0, R.GLC] == pytest.approx(0.1 * request * (1.0 - covered), rel=1e-6)
    assert w.state.respiration_loss_mmol[R.GLC] > request * covered, "the maintenance share is respired, on top of the growth respiration"


def test_with_more_maintenance_than_income_there_is_no_growth_and_the_reserve_pays_the_difference():
    w, initial, request, energy_in, need = pirt_tick(m=60.0, energy=100.0)
    assert need > energy_in
    assert w.state.biomass_mmol[0, R.GLC] == initial and w.state.nutrient_store_mmol[0, R.GLC] == 0.0
    assert w.state.respiration_loss_mmol[R.GLC] == pytest.approx(request, rel=1e-9), "everything eaten was respired"
    assert w.state.energy_pj[0] == pytest.approx(100.0 - (need - energy_in), rel=1e-9)


def test_the_pirt_relation_q_equals_mu_over_y_plus_m_holds_exactly_in_the_model():
    """q = growth/(yield) + m for any uptake level: derived from the allocation above, here checked at three uptake levels."""
    rows = []
    for glucose in (0.002, 0.01, 0.5):
        w = quiet_world(positions=((100.0, 100.0),), glucose=glucose, mass=0.3, trait=AMPLE_OXYGEN,
                        phys=R.physiology(maintenance_mmol_glc_gdw_h=0.4))
        before_g, before_m = w.field_mmol(R.GLC), float(w.state.mass_pg[0])
        w.step(120)
        eaten = before_g - w.field_mmol(R.GLC)
        mu = math.log(float(w.state.mass_pg[0]) / before_m) / (120 * DT_S / 3600.0)
        q = eaten / (before_m * 1e-12 * 1.0 * (120 * DT_S / 3600.0))
        rows.append((mu, q))
    mu, q = np.array(rows).T
    slope, intercept = np.polyfit(mu, q, 1)
    assert intercept == pytest.approx(0.4, rel=0.05)
    assert 1.0 / (slope * 0.180156) == pytest.approx(0.75, rel=0.05), "the yield is the D2 biomass fraction (0.75) here"


def test_a_cell_with_no_maintenance_coefficient_is_unchanged_and_never_starves():
    w = quiet_world(positions=((100.0, 100.0),), glucose=0.0, energy=0.0)
    w.step(600)
    assert w.sim.life.stress_s[0] == 0.0 and w.state.energy_pj[0] == 0.0


def test_the_mass_ledger_still_closes_with_maintenance_on():
    rng = np.random.default_rng(3)
    w = R.world(100.0 + rng.random((30, 2)) * 60.0, config=R.quiet(lag_initial=1.0), glucose=0.2, lactose=0.1, mass=0.4,
                phys=R.physiology(maintenance_mmol_glc_gdw_h=0.4))
    w.step(900)          # the engine raises from assert_conserved on any leak
    assert w.n >= 30


# ------------------------------------------------------------------------------------------ the overlay on the D2 numbers
@pytest.fixture(scope="module")
def ctx():
    return context(ROOT)


def widen(ctx, calibration):
    tables = dataclasses.replace(ctx.tables, calibration=calibration)
    return tables, P.extend_phenotype_table(ctx.base_table, list(ctx.genomes.values()), tables)


def test_identity_calibration_leaves_the_d2_numbers_exactly_and_the_fit_changes_them_by_the_stated_factors(ctx):
    base = ctx.base_table
    _, same = widen(ctx, Calibration())
    for name in ("max_uptake_mmol_s", "storage_fraction", "biomass_fraction"):
        assert np.array_equal(getattr(same, name)[:, :2], getattr(base, name))
    assert np.array_equal(same.motility_um_s, base.motility_um_s) and np.array_equal(same.division_mass_pg, base.division_mass_pg)
    _, cal = widen(ctx, CAL)
    for i, genome_id in enumerate(base.genome_id):
        accession = next(a for a, g in ctx.genomes.items() if np.uint64(g.genome_id) == genome_id)
        f = CAL.for_accession(accession)
        assert np.allclose(cal.max_uptake_mmol_s[i, :2], base.max_uptake_mmol_s[i] * f.uptake_scale)
        assert np.allclose(cal.biomass_fraction[i, :2], base.biomass_fraction[i] * f.biomass_scale)
        assert np.allclose(cal.storage_fraction[i, :2], base.storage_fraction[i] * f.storage_scale)
        assert cal.motility_um_s[i] == pytest.approx(base.motility_um_s[i] * f.motility_scale)
        assert cal.division_mass_pg[i] == pytest.approx(base.division_mass_pg[i] * f.division_mass_scale)
    assert (cal.storage_fraction + cal.biomass_fraction <= 1.0).all()


def test_a_gene_derived_difference_survives_the_calibration_because_the_factors_multiply(ctx):
    """B. subtilis has no lactose uptake in D2; any factor must leave it at zero while E. coli's stays positive."""
    _, wide = widen(ctx, CAL)
    row = {int(g): i for i, g in enumerate(wide.genome_id)}
    assert wide.max_uptake_mmol_s[row[ctx.genomes[BS].genome_id], 1] == 0.0
    assert wide.max_uptake_mmol_s[row[ctx.genomes[EC].genome_id], 1] > 0.0


def test_a_variants_phenotype_row_is_calibrated_the_same_way_as_its_root_genomes(ctx):
    channels = (NutrientChannel(0, "glucose"), NutrientChannel(1, "lactose"))
    tables, table = widen(ctx, CAL)
    for accession, genome in ctx.genomes.items():
        phenotype = P.widen_genome_phenotype(derive_phenotype(genome, channels, ROOT), genome, tables)
        i = int(np.flatnonzero(table.genome_id == np.uint64(genome.genome_id))[0])
        assert np.allclose(phenotype.max_uptake_mmol_s, table.max_uptake_mmol_s[i])
        assert np.allclose(phenotype.biomass_fraction, table.biomass_fraction[i])
        assert phenotype.motility_um_s == pytest.approx(table.motility_um_s[i])
        assert phenotype.division_mass_pg == pytest.approx(table.division_mass_pg[i])


def test_physiology_takes_maintenance_and_overflow_from_the_calibration_per_organism(ctx):
    for accession, genome in ctx.genomes.items():
        calibrated = P.derive_physiology(genome, dataclasses.replace(ctx.tables, calibration=CAL))
        plain = P.derive_physiology(genome, dataclasses.replace(ctx.tables, calibration=Calibration()))
        f = CAL.for_accession(accession)
        assert calibrated.maintenance_mmol_glc_gdw_h == pytest.approx(f.maintenance_mmol_glc_gdw_h) and plain.maintenance_mmol_glc_gdw_h == 0.0
        assert calibrated.acetate_overflow == pytest.approx(min(0.6, plain.acetate_overflow * f.overflow_scale))
        assert calibrated.sporulation_capacity == plain.sporulation_capacity, "gene effects not named in the calibration are untouched"


def test_a_calibration_value_must_be_a_finite_non_negative_number():
    with pytest.raises(ValueError):
        OrganismCalibration(uptake_scale=-1.0)
    with pytest.raises(ValueError):
        OrganismCalibration(biomass_scale=float("nan"))
    assert IDENTITY.uptake_scale == 1.0 and IDENTITY.maintenance_mmol_glc_gdw_h == 0.0


def test_the_controller_world_uses_the_calibrated_numbers():
    from microbiome.sim.controller import SimulationController

    c = SimulationController.create_demo(seed=3, realism=True)
    genome = next(int(o.genome_id) for o in c.list_genomes() if o.accession_version == EC)
    row = int(np.flatnonzero(c._phenotypes.genome_id == np.uint64(genome))[0])
    base = context(ROOT).base_table
    i = int(np.flatnonzero(base.genome_id == np.uint64(genome))[0])
    f = CAL.for_accession(EC)
    assert c._phenotypes.max_uptake_mmol_s[row, 0] == pytest.approx(base.max_uptake_mmol_s[i, 0] * f.uptake_scale)
    assert c._realism.simulation._phys["maintenance_mmol_glc_gdw_h"][np.searchsorted(c._realism.simulation._phys_ids, genome)] == pytest.approx(
        f.maintenance_mmol_glc_gdw_h)
    plain = SimulationController.create_demo(seed=3)
    assert plain._phenotypes.max_uptake_mmol_s[0, 0] == pytest.approx(base.max_uptake_mmol_s[0, 0]), "the plain D1 world keeps the D2 numbers"


# ------------------------------------------------------------------------------------------ numerical and constant checks
def test_the_diffusion_scheme_reproduces_the_gaussian_spreading_it_is_given():
    env = R.environment(nx=80, ny=64)
    env.concentration_mmol_l[R.GLC, 32, 40] = 1.0
    seconds = 5.0
    for _ in range(int(seconds / DT_S)):
        diffuse(env, DT_S)
    profile = env.concentration_mmol_l[R.GLC].sum(axis=0)
    x = (np.arange(80) + 0.5) * 10.0
    mean = float((profile * x).sum() / profile.sum())
    variance = float((profile * (x - mean) ** 2).sum() / profile.sum())
    expected = 2.0 * R.DIFFUSION[R.GLC] * seconds + 10.0 ** 2 / 12.0
    assert variance == pytest.approx(expected, rel=0.02)
    assert env.concentration_mmol_l[R.GLC].sum() == pytest.approx(1.0, rel=1e-12), "diffusion conserves the solute"


def test_the_field_diffusivities_agree_with_the_world_the_controller_builds_and_with_the_published_values():
    from microbiome.sim.controller import SimulationController

    field = SimulationController.create_demo(seed=1, realism=True)._environment.field
    assert tuple(field.diffusion_um2_s) == pytest.approx(P.DIFFUSIVITY_UM2_S)
    assert tuple(field.molar_mass_g_mol) == pytest.approx(P.MOLAR_MASS_G_MOL)
    for name, index in (("diffusion.glucose", 0), ("diffusion.lactose", 1)):
        low, high = target(name).range
        assert low <= P.DIFFUSIVITY_UM2_S[index] <= high, name
    assert P.DIFFUSIVITY_UM2_S[2] < target("diffusion.oxygen").value, "oxygen is capped below the published value, and the documentation says so"


def test_the_frozen_mutation_rate_and_the_sporulation_time_lie_in_their_published_ranges():
    from microbiome.genome import variants

    low, high = target("mutation.snv_rate").range
    assert low <= variants.SUBSTITUTIONS_PER_BASE <= high
    low, high = target("bsub.sporulation_time").range
    assert low <= RealismConfig().sporulation_time_s <= high


def test_the_default_cell_dry_mass_matches_the_literature_growth_law_at_the_fitted_growth_rate(ctx):
    division = float(ctx.base_table.division_mass_pg[np.flatnonzero(ctx.base_table.genome_id == np.uint64(ctx.genomes[EC].genome_id))[0]])
    division *= CAL.for_accession(EC).division_mass_scale
    mean_fg = division * 1000.0 / (2.0 / (2.0 * math.log(2.0)))
    assert abs(mean_fg - target("ecoli.dry_mass.mean_at_mu_0.65").value) <= target("ecoli.dry_mass.mean_at_mu_0.65").sd
    # the 188 fg is an interpolation, not an extrapolation, between two published points (100 and 40 min doublings)
    growth_rate = math.log(2.0) * 60.0 / np.array([100.0, 40.0])
    assert growth_rate[0] < 0.65 < growth_rate[1]
    slope = np.polyfit(growth_rate, np.log([148.0, 280.0]), 1)
    assert math.exp(np.polyval(slope, 0.65)) == pytest.approx(188.0, rel=0.03)


# ------------------------------------------------------------------------------------------ the generated documentation
def test_the_calibration_document_is_exactly_what_the_generator_produces_from_the_data():
    on_disk = (ROOT / "docs" / "CALIBRATION.md").read_text(encoding="utf-8")
    assert on_disk == report.render(ROOT), "run python -m microbiome.calibration report"


def test_the_document_lists_every_target_and_says_which_agreements_are_by_construction():
    text = (ROOT / "docs" / "CALIBRATION.md").read_text(encoding="utf-8")
    for t in TS.targets:
        assert f"`{t.id}`" in text, t.id
    assert "Matched by construction" in text and "Held out" in text and "Known mismatches" in text and "Not calibrated at all" in text
    assert "weak" in text, "the document must not oversell the held-out checks"
