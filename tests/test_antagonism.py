"""README must-do #2: at least one form of antagonism, population behaviour EMERGING from per-agent rules.

The frozen table gives B. subtilis 168 a bacilysin-cluster rule (bacA-bacG -> antagonism_rate_s 1e-3 /s,
antagonism_radius_um 12) and gives E. coli K-12 none. The engine applies it as a contact hazard on agents of
OTHER species inside the attacker's radius. This file proves, with the real genome module and the real
Simulation.step, that:
  1. B. subtilis measurably suppresses a co-cultured E. coli versus a no-antagonism control on the same seed;
  2. the population effect is exactly what the per-agent rule implies (an independent per-agent oracle on a fixed
     geometry), is local (nothing dies beyond the radius, to 0.2 um), and rises with rate, radius and density;
  3. it is specific (no friendly fire, the attacker is not harmed) and the control differs ONLY in that trait.
"""
import math

import numpy as np
import pytest

import sim_support as S
import test_two_genomes as tt
from conftest import ROOT  # noqa: F401  (keeps the project root importable like the other test files)

EC, BS = tt.EC, tt.BS
DT = tt.DT
SEED = 20261017


# ------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def genomes():
    return {a: tt.load_genome(a) for a in (EC, BS)}


@pytest.fixture(scope="module")
def table(genomes):
    return tt.build_phenotype_table(list(genomes.values()), tt.CHANNELS)


@pytest.fixture(scope="module")
def rows(genomes, table):
    return {a: int(np.flatnonzero(table.genome_id == np.uint64(g.genome_id))[0]) for a, g in genomes.items()}


# ------------------------------------------------------------------------------------------- helpers
def mix(seed, table, e, b, n_e, n_b, ticks, n_cells=8):
    """n_e E. coli + n_b B. subtilis, shuffled, seeded placement (identical for every table variant)."""
    step = S.contract_attr("step")
    assign = np.r_[np.full(n_e, e), np.full(n_b, b)]
    np.random.default_rng(seed + 77).shuffle(assign)
    state, env, rngs = tt.world(seed, assign, table, n_cells=n_cells)
    return finish(state, env, rngs, table, assign, e, b, ticks, step)


def show(r):
    """Counts only, so assertion messages stay readable."""
    return {"E": r["E"], "B": r["B"], "n_end": r["n_end"]}


def finish(state, env, rngs, table, assign, e, b, ticks, step):
    ids0 = state.agents.agent_id.copy()
    for _ in range(ticks):
        step(state, env, table, rngs)
    assert S.violations(state, env) == []
    alive = np.isin(ids0, state.agents.agent_id)
    return {"E": {"n": int((assign == e).sum()), "dead": int((~alive[assign == e]).sum())},
            "B": {"n": int((assign == b).sum()), "dead": int((~alive[assign == b]).sum())},
            "alive": alive, "n_end": int(len(state.agents.agent_id))}


def immobile(table, e, b):
    return tt.edit(tt.edit(table, motility_um_s=(e, 0.0)), motility_um_s=(b, 0.0))


def neighbours_in_radius(pos_target, pos_attacker, width, height, radius):
    """Independent oracle: periodic pairwise distance, same <= rule as the engine, O(N*M) numpy."""
    d = pos_target[:, None, :] - pos_attacker[None, :, :]
    d[..., 0] -= np.rint(d[..., 0] / width) * width
    d[..., 1] -= np.rint(d[..., 1] / height) * height
    return ((d ** 2).sum(-1) <= radius ** 2).sum(1)


def poisson_sf(k, lam):
    """P(X >= k) for X ~ Poisson(lam), summed in log space (no scipy)."""
    return sum(math.exp(-lam + j * math.log(lam) - math.lgamma(j + 1)) for j in range(k, k + 400))


def control_table(table, b):
    """The no-antagonism control: B. subtilis's attack rate set to zero. Nothing else changes."""
    return tt.edit(table, antagonism_rate_s=(b, 0.0))


# ===================================================================== 1. the contract test
def test_bacilysin_suppresses_a_competitor(genomes, table, rows):
    """Same seed, same placement, same everything; B. subtilis's bacilysin rate on vs off. E. coli, the
    competitor, loses members only when it is on; B. subtilis does not; the loss matches the table-derived
    contact-hazard prediction and is overwhelmingly unlikely to be the baseline death rate."""
    e, b = rows[EC], rows[BS]
    phen = {a: tt.derive_phenotype(g, tt.CHANNELS) for a, g in genomes.items()}
    assert "bacillus-bacilysin-antagonism-rate" in phen[BS].applied_rule_ids       # B. subtilis carries the rule
    assert not any("bacilysin" in r for r in phen[EC].applied_rule_ids)            # E. coli does not
    ctrl_table, ticks, n = control_table(table, b), 300, 320
    real_dead = ctrl_dead = 0
    for seed in (SEED, SEED + 1, SEED + 2):
        real = mix(seed, table, e, b, n, n, ticks)
        ctrl = mix(seed, ctrl_table, e, b, n, n, ticks)
        assert ctrl["E"]["dead"] <= 1 and ctrl["B"]["dead"] <= 1, f"control must not kill (seed {seed}): {show(ctrl)}"
        assert real["E"]["dead"] > ctrl["E"]["dead"] + 10, f"seed {seed}: competitor not suppressed: {show(real)} vs {show(ctrl)}"
        assert real["B"]["dead"] <= 1, f"the attacker must not be harmed (seed {seed}): {show(real)}"
        e_share_real = (n - real["E"]["dead"]) / (real["n_end"])
        e_share_ctrl = (n - ctrl["E"]["dead"]) / (ctrl["n_end"])
        assert e_share_real < 0.48 <= 0.4985 < e_share_ctrl + 1e-12 <= 0.5 + 1e-12, (e_share_real, e_share_ctrl)
        real_dead += real["E"]["dead"]
        ctrl_dead += ctrl["E"]["dead"]
    frac = real_dead / (3 * n)
    pred, neighbours = tt.predicted_death_fraction(table, b, n, 8, ticks)
    assert 0.65 * pred <= frac <= 1.35 * pred, f"suppression {frac:.3f} vs table-derived prediction {pred:.3f}"
    # significance against the null "only baseline death": lambda0 = agents * death_rate * t
    lam0 = 3 * n * float(table.death_rate_s[e]) * ticks * DT
    assert poisson_sf(real_dead, lam0 + ctrl_dead) < 1e-12, (real_dead, lam0)
    print(f"\nE. coli dead of {3 * n}: bacilysin on {real_dead} ({100 * frac:.1f}%), control {ctrl_dead};"
          f" predicted {100 * pred:.1f}% from rate x {neighbours:.1f} neighbours x {ticks * DT:.1f} s")


# =========================================== 2. the population effect IS the per-agent rule
def fixed_geometry(seed, n_e=600, n_b=450, cells=10, dx=10.0):
    rng = np.random.default_rng(seed + 5)
    W = H = cells * dx
    pos_b = np.column_stack([rng.random(n_b) * 40.0, rng.random(n_b) * H])   # B. subtilis only in x < 40 um
    pos_e = rng.random((n_e, 2)) * [W, H]                                      # E. coli everywhere
    return pos_e, pos_b, W, H


def test_each_ecoli_dies_at_the_rate_its_own_neighbours_imply(table, rows):
    """Immobile agents on a fixed geometry. Per agent, the engine's hazard is death_rate + rate * (B. subtilis
    inside the radius), so survival over t is exp(-hazard * t) exactly. Expected deaths are computed here from
    the geometry alone and compared with what the engine produced."""
    e, b = rows[EC], rows[BS]
    ticks = 600
    pos_e, pos_b, W, H = fixed_geometry(SEED)
    tab = immobile(table, e, b)
    assign = np.r_[np.full(len(pos_e), e), np.full(len(pos_b), b)]
    state, env, rngs = tt.world(SEED, assign, tab, positions=np.vstack([pos_e, pos_b]))
    out = finish(state, env, rngs, tab, assign, e, b, ticks, S.contract_attr("step"))
    rate, radius = float(tab.antagonism_rate_s[b]), float(tab.antagonism_radius_um[b])
    n_i = neighbours_in_radius(pos_e, pos_b, W, H, radius)
    p = 1.0 - np.exp(-(float(tab.death_rate_s[e]) + rate * n_i) * ticks * DT)
    dead = ~out["alive"][:len(pos_e)]
    expected, sd = p.sum(), math.sqrt((p * (1 - p)).sum())
    assert (n_i > 0).sum() > 300 and (n_i == 0).sum() > 150, "geometry must contain exposed and unexposed E. coli"
    assert abs(dead.sum() - expected) <= 4 * sd, f"deaths {dead.sum()} vs per-agent expectation {expected:.1f} +- {sd:.1f}"
    assert dead.sum() > 40 and out["B"]["dead"] <= 2   # attackers die only at baseline (expect ~0.05)
    # locality: E. coli with no B. subtilis inside the radius die only at the baseline rate. Expected number
    # here is death_rate * t * count ~ 0.02, so one chance death is possible (it happened for this seed: an agent
    # 24 um from any B. subtilis, pressure exactly 0.0, see the white-box test below); several would not be.
    unexposed = n_i == 0
    base = (1 - math.exp(-float(tab.death_rate_s[e]) * ticks * DT)) * unexposed.sum()
    assert dead[unexposed].sum() <= 2, f"{int(dead[unexposed].sum())} unexposed E. coli died; baseline expects {base:.3f}"
    # dose-response by neighbour count: the most exposed bin carries the deaths, and matches its own expectation
    hi = n_i >= 15
    exp_hi, sd_hi = p[hi].sum(), math.sqrt((p[hi] * (1 - p[hi])).sum())
    assert abs(dead[hi].sum() - exp_hi) <= 4 * sd_hi, (int(dead[hi].sum()), exp_hi, sd_hi)
    low = (n_i >= 1) & (n_i < 15)
    assert dead[hi].mean() > 2 * dead[low].mean()


def test_engine_pressure_on_every_agent_equals_rate_times_its_neighbours(table, rows):
    """WHITE-BOX, deterministic (no statistics): spy on the engine's contact-hazard function and compare, for
    every E. coli on the fixed geometry, the hazard it applied with rate * (B. subtilis within the radius)
    computed independently here. Exposed agents must match to rounding; unexposed agents must never get any.
    This also checks Cilia's half-stencil spatial hash against brute force on 600 agents."""
    import microbiome.sim.engine as eng

    if not hasattr(eng, "_antagonism_pressure"):
        pytest.skip("engine no longer exposes _antagonism_pressure; the statistical tests above still apply")
    e, b = rows[EC], rows[BS]
    tab = immobile(table, e, b)
    pos_e, pos_b, W, H = fixed_geometry(SEED)
    assign = np.r_[np.full(len(pos_e), e), np.full(len(pos_b), b)]
    seen, real = {}, eng._antagonism_pressure

    def spy(state, env, traits):
        p = real(state, env, traits)
        for aid, v in zip(state.agent_id.tolist(), p.tolist()):
            seen[aid] = max(seen.get(aid, 0.0), v)
        return p

    eng._antagonism_pressure = spy
    try:
        state, env, rngs = tt.world(SEED, assign, tab, positions=np.vstack([pos_e, pos_b]))
        finish(state, env, rngs, tab, assign, e, b, 120, S.contract_attr("step"))
    finally:
        eng._antagonism_pressure = real
    rate, radius = float(tab.antagonism_rate_s[b]), float(tab.antagonism_radius_um[b])
    n_i = neighbours_in_radius(pos_e, pos_b, W, H, radius)
    got = np.array([seen.get(i + 1, 0.0) for i in range(len(pos_e))])
    assert (n_i > 0).sum() > 300 and (n_i == 0).sum() > 150
    assert np.abs(got - rate * n_i).max() < 1e-12, "engine hazard != rate * independently counted neighbours"
    assert (got[n_i == 0] == 0.0).all(), "an E. coli with no attacker in range received hazard"
    assert all(seen.get(len(pos_e) + j + 1, 0.0) == 0.0 for j in range(len(pos_b))), "an attacker received hazard"


def test_one_attacker_one_victim_dies_at_exactly_the_table_rate(table, rows):
    """The cleanest possible calibration: 400 isolated (B. subtilis, E. coli) pairs, 6 um apart and 50 um from
    any other pair, immobile. Each E. coli has exactly one attacker, so its survival over t is
    exp(-(rate + baseline) * t) with no mean-field model. Rate raised to 0.1 /s so 400 pairs resolve it.
    (Checked offline on 1,200 pairs over 3 seeds: 0.642 dead vs 0.632 expected, z = +0.7: no bias.)"""
    e, b = rows[EC], rows[BS]
    rate, ticks, side = 0.1, 600, 20
    tab = tt.edit(immobile(table, e, b), antagonism_rate_s=(b, rate))
    sites = np.array([(25.0 + 50.0 * i, 25.0 + 50.0 * j) for i in range(side) for j in range(side)])
    assign = np.r_[np.full(len(sites), e), np.full(len(sites), b)]
    state, env, rngs = tt.world(SEED, assign, tab, n_cells=100, positions=np.vstack([sites + [6.0, 0.0], sites]))
    out = finish(state, env, rngs, tab, assign, e, b, ticks, S.contract_attr("step"))
    dead = int((~out["alive"][:len(sites)]).sum())
    p = 1.0 - math.exp(-(rate + float(tab.death_rate_s[e])) * ticks * DT)
    n = len(sites)
    sd = math.sqrt(n * p * (1 - p))
    assert abs(dead - n * p) <= 4 * sd, f"{dead}/{n} dead vs exact expectation {n * p:.1f} +- {sd:.1f}"
    assert out["B"]["dead"] <= 2   # attackers die only at baseline: 400 x 1e-5 /s x 10 s = 0.04 expected


def test_the_attack_radius_is_respected_to_two_tenths_of_a_micrometre(table, rows):
    """Pairs of (B. subtilis, E. coli) far apart from each other, E. coli 11.9 um from its B. subtilis on half
    the sites and 12.1 um on the other half. Attack rate raised to 0.1 /s so a few dozen pairs show the effect.
    Inside: deaths at the predicted rate. Outside: none."""
    e, b = rows[EC], rows[BS]
    radius = float(table.antagonism_radius_um[b])
    rate = 0.1
    tab = tt.edit(immobile(table, e, b), antagonism_rate_s=(b, rate))
    sites = [(25.0 + 50.0 * i, 25.0 + 50.0 * j) for i in range(8) for j in range(8)]
    pos_b = np.array(sites)
    d = np.array([radius - 0.1 if k % 2 == 0 else radius + 0.1 for k in range(len(sites))])
    pos_e = pos_b + np.column_stack([d, np.zeros(len(d))])
    assign = np.r_[np.full(len(sites), e), np.full(len(sites), b)]
    state, env, rngs = tt.world(SEED, assign, tab, n_cells=40, positions=np.vstack([pos_e, pos_b]))
    ticks = 600
    out = finish(state, env, rngs, tab, assign, e, b, ticks, S.contract_attr("step"))
    dead = ~out["alive"][:len(sites)]
    inside = np.arange(len(sites)) % 2 == 0
    p = 1.0 - math.exp(-(rate + float(tab.death_rate_s[e])) * ticks * DT)
    expected, sd = inside.sum() * p, math.sqrt(inside.sum() * p * (1 - p))
    assert abs(dead[inside].sum() - expected) <= 4 * sd, \
        f"inside-radius deaths {int(dead[inside].sum())} vs expected {expected:.1f} +- {sd:.1f}"
    assert dead[~inside].sum() <= 1, f"{int(dead[~inside].sum())} E. coli died although {radius + 0.1} um > radius {radius}"
    assert out["B"]["dead"] <= 1   # baseline only (32 x 1e-5 /s x 10 s = 0.003 expected)


# ======================================================================== dose-response
def within_binomial_band(dead, trials, p, systematic=0.15, sigmas=4.0):
    """|dead - trials*p| <= sigmas * binomial sd + a mean-field model allowance (a fraction of the expectation)."""
    expect = trials * p
    return abs(dead - expect) <= sigmas * math.sqrt(trials * p * (1 - p)) + systematic * expect


def test_suppression_rises_with_attack_rate_radius_and_attacker_density(table, rows):
    e, b = rows[EC], rows[BS]
    rate, radius = float(table.antagonism_rate_s[b]), float(table.antagonism_radius_um[b])
    ticks, n, seeds = 300, 320, (SEED, SEED + 1)
    area = 80.0 ** 2

    def deaths(tab, n_b):
        return sum(mix(s, tab, e, b, n, n_b, ticks)["E"]["dead"] for s in seeds)

    def p_of(mult, n_b):
        return 1 - math.exp(-rate * mult * n_b / area * math.pi * radius ** 2 * ticks * DT)

    # rate: 0.5x, 1x, 2x -> each within its binomial band of the table-derived prediction; strictly increasing
    by_rate = [deaths(tt.edit(table, antagonism_rate_s=(b, rate * m)), n) for m in (0.5, 1.0, 2.0)]
    for m, d in zip((0.5, 1.0, 2.0), by_rate):
        assert within_binomial_band(d, len(seeds) * n, p_of(m, n)), (m, d, len(seeds) * n * p_of(m, n))
    assert by_rate[0] < by_rate[1] < by_rate[2], by_rate
    # radius: 0, 6, 12 um -> none, some, more
    by_radius = [mix(SEED, tt.edit(table, antagonism_radius_um=(b, rr)), e, b, n, n, ticks)["E"]["dead"]
                 for rr in (0.0, radius / 2, radius)]
    assert by_radius[0] == 0 and 0 < by_radius[1] < by_radius[2], by_radius
    # attacker density: emergent, density-dependent suppression from a contact rule (160 / 320 / 640 B. subtilis)
    by_density = [deaths(table, nb) for nb in (160, 320, 640)]
    for nb, d in zip((160, 320, 640), by_density):
        assert within_binomial_band(d, len(seeds) * n, p_of(1.0, nb)), (nb, d, len(seeds) * n * p_of(1.0, nb))
    assert by_density[0] < by_density[1] < by_density[2], by_density


# ======================================================================= specificity
def test_antagonism_is_specific_no_friendly_fire_and_the_attacker_is_unharmed(table, rows):
    e, b = rows[EC], rows[BS]
    ticks, n = 300, 640
    for label, r in (("B. subtilis alone", mix(SEED, table, e, b, 0, n, ticks)),
                     ("E. coli alone", mix(SEED, table, e, b, n, 0, ticks))):
        # no antagonism applies within one species; only baseline death (640 x 1e-5 /s x 5 s = 0.03 expected)
        assert r["E"]["dead"] <= 2 and r["B"]["dead"] <= 2, f"{label}: no deaths expected, got {r['E']['dead'] + r['B']['dead']}"
    both = mix(SEED, table, e, b, 320, 320, ticks)
    assert both["E"]["dead"] > 10 and both["B"]["dead"] <= 1, show(both)


def test_the_control_differs_from_the_real_run_only_in_the_antagonism_trait(table, rows):
    e, b = rows[EC], rows[BS]
    ctrl = control_table(table, b)
    changed = []
    import dataclasses
    for f in dataclasses.fields(table):
        x, y = getattr(table, f.name), getattr(ctrl, f.name)
        if x is not None and not np.array_equal(x, y):
            changed.append(f.name)
    assert changed == ["antagonism_rate_s"], f"control changes more than the antagonism trait: {changed}"
    assert (np.flatnonzero(table.antagonism_rate_s != ctrl.antagonism_rate_s) == [b]).all()
    # identical initial worlds: same arrays, same engine seed state
    assign = np.where(np.arange(640) % 2 == 0, e, b)
    s1, e1, r1 = tt.world(SEED, assign, table, n_cells=8)
    s2, e2, r2 = tt.world(SEED, assign, ctrl, n_cells=8)
    assert S.snapshot_digest(s1, e1, r1) == S.snapshot_digest(s2, e2, r2)
