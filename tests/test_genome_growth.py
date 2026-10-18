"""Item 216: two real genomes diverge in POPULATION GROWTH, not only in nutrient uptake.

E. coli K-12 MG1655 (GCF_000005845.2) carries lacY and the frozen table gives it lactose uptake;
B. subtilis 168 (GCF_000009045.1) has no lacY and no lactose uptake. Both genomes share the same
division-rate calibration and the same division mass. The claim tested here is that the documented
lacY/lactose difference is enough for the two populations to grow at different rates, not merely for
one to take up a different amount of substrate.

Measured quantity: births and population size in identical lactose-only pure cultures (same seed,
same grid, same inoculum, same number of agents; only the genome's trait row differs). The prediction
comes from the documented values, not from the engine: at saturating lactose, a cell gains

    uptake_lactose * biomass_fraction * molar_mass_lactose * 1e9   pg/s

crosses the shared division mass after (division_mass - inoculum_mass) / that rate seconds, and then
divides with the table's division hazard.

Why a growth clock: the frozen calibration is slow by design (~2e-4 /s division hazard, 833 s to close
a 0.1 pg gap). Observing births at 1x would need ~300,000 ticks (~100 s wall clock) per culture. The
population experiment therefore multiplies BOTH rows' uptake and division hazard by the same factor
GROWTH_CLOCK, which is exactly a time compression of the documented growth law: the documented
crossing time is divided by the factor, and the documented division hazard is multiplied by it. The
edits are symmetric and asserted to be symmetric; division rates stay identical, so the factor cannot
create a difference between genomes. A separate test at 1x verifies the documented biomass growth
exactly. Antagonism and death are set to zero for both rows in these pure cultures (both documented
rates are identical anyway), so the only remaining difference is the lactose-uptake cell.

Null controls: (1) give B. subtilis only E. coli's lactose-uptake value - the run becomes identical to
the E. coli run in every non-identity array; (2) remove E. coli's lactose uptake - it stops growing,
exactly like B. subtilis; (3) remove lactose from the medium - neither genome grows.
"""
import dataclasses

import numpy as np
import pytest

import sim_support as S
import test_two_genomes as tt
from conftest import ROOT  # noqa: F401  (keeps the project root importable like the other test files)

EC, BS = tt.EC, tt.BS
GLC, LAC = 0, 1
MOLAR = tt.MOLAR
DT = tt.DT
SEED = tt.SEED

N = 400                     # agents per pure culture
N_CELLS = 16                # 160 x 160 um periodic grid, dx = dy = 10 um
LACTOSE_MM = 50.0           # replete: measured >= 40 mM remains at the end of every run
TICKS = 4800                # 80 s of compressed time = 3200 s of documented time at GROWTH_CLOCK
GROWTH_CLOCK = 40.0         # test instrument: both rows' uptake and division hazard x 40
START_MASS_PG = 0.5         # tt.world inoculates 0.5 pg; the shared division mass is 0.6 pg


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
def row_differences(table, e, b):
    """Every (field, channel-or-None) in which table row e differs from row b."""
    out = set()
    for f in dataclasses.fields(table):
        if f.name == "genome_id":  # identity column, not a trait
            continue
        a = getattr(table, f.name)
        if a is None:
            continue
        if a.ndim == 2:
            out |= {(f.name, k) for k in range(a.shape[1]) if a[e, k] != a[b, k]}
        elif a[e] != a[b]:
            out.add((f.name, None))
    return out


def growth_table(table, clock):
    """Pure-culture growth table. Every edit is applied to BOTH rows and is symmetric.

    uptake and division hazard are multiplied by the same clock (a time compression of the documented
    growth law); antagonism and death are zeroed in both rows because these are single-genome cultures
    and neither trait is part of the growth comparison. The lactose-uptake cell is the only asymmetry.
    """
    return dataclasses.replace(
        table,
        max_uptake_mmol_s=table.max_uptake_mmol_s * clock,
        division_rate_s=table.division_rate_s * clock,
        antagonism_rate_s=np.zeros_like(table.antagonism_rate_s),
        antagonism_radius_um=np.zeros_like(table.antagonism_radius_um),
        death_rate_s=np.zeros_like(table.death_rate_s),
    )


def with_lactose_uptake(table, row_index, value):
    uptake = table.max_uptake_mmol_s.copy()
    uptake[row_index, LAC] = value
    return dataclasses.replace(table, max_uptake_mmol_s=uptake)


def world(seed, assign, table, n=N, n_cells=N_CELLS, dx=10.0, depth=3.0, concentrations=(0.0, LACTOSE_MM)):
    """Same construction as tt.world, but with one concentration per nutrient channel (glucose 0,
    lactose replete) and a ledger-consistent initial amount for that field."""
    k = 2
    rng = np.random.default_rng(seed)
    width, height = n_cells * dx, n_cells * dx
    positions = np.minimum(rng.random((n, 2)) * [width, height], np.nextafter([width, height], 0))
    biomass = np.tile((START_MASS_PG / 2) / (MOLAR * 1e9), (n, 1))
    cell_volume_l = dx * dx * depth * 1e-15
    field = np.empty((k, n_cells, n_cells))
    for channel, value in enumerate(concentrations):
        field[channel] = value
    pool = dict(
        agent_id=np.arange(1, n + 1, dtype=np.uint64), position_um=positions, velocity_um_s=np.zeros((n, 2)),
        mass_pg=(biomass * MOLAR * 1e9).sum(1), radius_um=np.full(n, 0.5), energy_pj=np.full(n, 1e3),
        species_id=(assign + 1).astype(np.uint32), genome_id=table.genome_id[assign],
        age_ticks=np.zeros(n, np.uint64), nutrient_store_mmol=np.zeros((n, k)), biomass_mmol=biomass,
        tick_index=np.uint64(0), next_agent_id=np.uint64(n + 1), width_um=width, height_um=height, depth_um=depth,
        dx_um=dx, dy_um=dx, nx=n_cells, ny=n_cells, nutrient_id=np.arange(k, dtype=np.uint32),
        concentration_mmol_l=field, diffusion_um2_s=np.full(k, 10.0), molar_mass_g_mol=MOLAR,
        detritus_mmol=np.zeros((k, n_cells, n_cells)),
        initial_mmol=field.sum(axis=(1, 2)) * cell_volume_l + biomass.sum(0),
        source_cum_mmol=np.zeros(k), sink_cum_mmol=np.zeros(k), respiration_loss_mmol=np.zeros(k),
    )
    return (S.build_instance(S.contract_attr("SimulationState"), pool),
            S.build_instance(S.contract_attr("Environment"), pool),
            S.contract_attr("initialize_rngs")(seed))


def culture(seed, table, row_index, ticks=TICKS, lactose_mM=LACTOSE_MM):
    """One pure culture of a single genome row; returns births, population trace, mass gain and the
    final per-agent arrays (used by the bit-for-bit null control)."""
    assign = np.full(N, row_index)
    state, env, rngs = world(seed, assign, table, concentrations=(0.0, lactose_mM))
    ids0 = state.agents.agent_id.copy()
    mass0 = float(state.agents.mass_pg.sum())
    assert abs(mass0 / N - START_MASS_PG) < 1e-12, "inoculum mass changed; update START_MASS_PG and the prediction"
    counts = []
    step = S.contract_attr("step")
    for _ in range(ticks):
        step(state, env, table, rngs)
        counts.append(len(state.agents.agent_id))
    assert S.violations(state, env) == []
    ids1 = state.agents.agent_id.copy()
    collected = S.collect(state, "s")
    after = {name: S.find(collected, name).copy() for name in S.AGENT_ROWS}
    return {
        "births": int(np.setdiff1d(ids1, ids0).size),
        "deaths": int(np.setdiff1d(ids0, ids1).size),
        "n_end": int(ids1.size),
        "counts": counts,
        "mass_gain_pg": float(state.agents.mass_pg.sum()) - mass0,
        "min_lactose_mM": float(env.field.concentration_mmol_l[LAC].min()),
        "after": after,
    }


def growth_prediction(table, row_index, ticks=TICKS):
    """Expected births from the DOCUMENTED table values alone.

    A cell closes the gap to the shared division mass at a constant rate (lactose is replete), then has
    the documented division hazard for the rest of the window. Daughters are half the parent's mass and
    cannot reach the threshold again inside the window (asserted by the caller), so one birth per cell
    is the exact model under the null of no interactions.
    """
    gap_pg = float(table.division_mass_pg[row_index]) - START_MASS_PG
    rate_pg_s = (float(table.max_uptake_mmol_s[row_index, LAC])
                 * float(table.biomass_fraction[row_index, LAC]) * MOLAR[LAC] * 1e9)
    t_cross_s = gap_pg / rate_pg_s
    documented_window_s = ticks * DT * GROWTH_CLOCK - t_cross_s
    p = float(-np.expm1(-float(table.division_rate_s[row_index]) * documented_window_s))
    return {
        "p": p,
        "expected_births": N * p,
        "sd": float(np.sqrt(N * p * (1.0 - p))),
        "t_cross_s": t_cross_s,
        "t_cross_ticks": t_cross_s / GROWTH_CLOCK / DT,
        "rate_pg_s": rate_pg_s,
        "documented_window_s": documented_window_s,
        "daughter_regrow_s": (float(table.division_mass_pg[row_index]) / 2.0) / rate_pg_s,
    }


# =========================================================== 1. the premise: only lactose uptake differs
def test_the_shared_division_calibration_leaves_lactose_uptake_as_the_only_growth_difference(genomes, table, rows):
    e, b = rows[EC], rows[BS]
    # documented calibration: identical division hazard and threshold, identical glucose uptake
    assert table.division_rate_s[e] == table.division_rate_s[b] > 0.0
    assert table.division_mass_pg[e] == table.division_mass_pg[b] > 0.0
    assert table.max_uptake_mmol_s[e, GLC] == table.max_uptake_mmol_s[b, GLC] > 0.0
    # the one documented growth asymmetry: E. coli takes up lactose, B. subtilis does not
    assert table.max_uptake_mmol_s[e, LAC] > 0.0 == table.max_uptake_mmol_s[b, LAC]
    assert row_differences(table, e, b) == tt.EXPECTED_DIFFERENCES
    # and it is the lacY rule, present only in the E. coli annotation
    phenotype = {a: tt.derive_phenotype(g, tt.CHANNELS) for a, g in genomes.items()}
    symbols = {a: {g.gene_symbol for g in genomes[a].genes if g.gene_symbol} for a in genomes}
    assert "ecoli-lacy-lactose-uptake" in phenotype[EC].applied_rule_ids
    assert not any("lactose" in rule for rule in phenotype[BS].applied_rule_ids)
    assert "lacY" in symbols[EC] and "lacY" not in symbols[BS]
    # the growth-clock edit is symmetric: division rates stay identical, and the only row difference
    # left in the growth table is the lactose-uptake cell
    fast = growth_table(table, GROWTH_CLOCK)
    assert np.array_equal(fast.division_rate_s, table.division_rate_s * GROWTH_CLOCK)
    assert fast.division_rate_s[e] == fast.division_rate_s[b]
    assert np.array_equal(fast.max_uptake_mmol_s, table.max_uptake_mmol_s * GROWTH_CLOCK)
    assert np.array_equal(fast.death_rate_s, np.zeros_like(table.death_rate_s))
    assert np.array_equal(fast.antagonism_rate_s, np.zeros_like(table.antagonism_rate_s))
    assert np.array_equal(fast.antagonism_radius_um, np.zeros_like(table.antagonism_radius_um))
    assert row_differences(fast, e, b) == {("max_uptake_mmol_s", LAC)}


# ============================================ 2. the required measurement: population growth diverges
def test_two_genomes_diverge_in_population_growth(table, rows):
    """Same seed, same grid, same 400-agent inoculum, same division calibration. The only difference is
    the genome's lactose-uptake trait. E. coli's population grows; B. subtilis's does not; the number of
    births matches the prediction from the documented uptake, biomass fraction, division mass and
    division hazard. Two seeds, so the result is not one lucky draw."""
    e, b = rows[EC], rows[BS]
    fast = growth_table(table, GROWTH_CLOCK)
    prediction = growth_prediction(table, e)
    assert 0.0 < prediction["p"] < 1.0
    # a daughter (half the division mass) cannot regrow to the threshold inside the observation window,
    # so each initial cell can divide at most once and the binomial prediction is exact for this design
    assert prediction["documented_window_s"] < prediction["daughter_regrow_s"], prediction
    observed = {}
    for seed in (SEED, SEED + 1):
        ecoli = culture(seed, fast, e)
        bacillus = culture(seed, fast, b)
        observed[seed] = (ecoli, bacillus)
        # B. subtilis never accumulates lactose and never closes the 0.1 pg gap: zero births, static N
        assert bacillus["births"] == 0 and bacillus["n_end"] == N and bacillus["deaths"] == 0
        assert all(count == N for count in bacillus["counts"])
        # E. coli does grow, and the births match the table-derived expectation
        assert ecoli["births"] > 0 and ecoli["deaths"] == 0
        assert ecoli["n_end"] == N + ecoli["births"]
        assert abs(ecoli["births"] - prediction["expected_births"]) <= 4.0 * prediction["sd"], (
            seed, ecoli["births"], prediction)
        # the population divergence itself, not just the uptake difference
        assert ecoli["n_end"] >= bacillus["n_end"] + 100, (seed, ecoli["n_end"], bacillus["n_end"])
        assert ecoli["mass_gain_pg"] > 0.0 == bacillus["mass_gain_pg"]
        # the substrate stayed replete, so growth was not limited by the field
        assert ecoli["min_lactose_mM"] > 0.5 * LACTOSE_MM
    e1, b1 = observed[SEED]
    e2, b2 = observed[SEED + 1]
    print(f"\nlactose-only pure cultures, {N} agents, {TICKS} ticks ({TICKS * DT * GROWTH_CLOCK:.0f} s documented):"
          f"\n  E. coli      births {e1['births']}, {e2['births']}  -> N {e1['n_end']}, {e2['n_end']}"
          f"   (predicted {prediction['expected_births']:.1f} +- {prediction['sd']:.1f})"
          f"\n  B. subtilis  births {b1['births']}, {b2['births']}  -> N {b1['n_end']}, {b2['n_end']}"
          f"\n  documented crossing time {prediction['t_cross_s']:.0f} s; compressed {prediction['t_cross_ticks']:.0f} ticks")


# =============================================================== 3. null controls
def test_null_control_the_divergence_is_the_lactose_trait_not_the_genome_label(table, rows):
    """Give B. subtilis ONLY E. coli's lactose-uptake cell. Every other trait stays B. subtilis's.
    The resulting run must equal the E. coli run bit for bit in every non-identity array."""
    e, b = rows[EC], rows[BS]
    fast = growth_table(table, GROWTH_CLOCK)
    reference = culture(SEED, fast, e)
    gained_table = with_lactose_uptake(fast, b, float(fast.max_uptake_mmol_s[e, LAC]))
    # the intervention changes exactly one table cell
    changed = np.argwhere(gained_table.max_uptake_mmol_s != fast.max_uptake_mmol_s)
    assert changed.shape == (1, 2) and tuple(changed[0]) == (b, LAC)
    gained = culture(SEED, gained_table, b)
    for name in S.AGENT_ROWS:
        if name in ("genome_id", "species_id"):
            continue
        assert gained["after"][name].tobytes() == reference["after"][name].tobytes(), \
            f"'{name}' differs although the only trait change was lactose uptake"
    assert not np.array_equal(gained["after"]["genome_id"], reference["after"]["genome_id"])
    assert gained["births"] == reference["births"] > 0
    # removing only E. coli's lactose uptake stops its growth, exactly like B. subtilis
    lost = culture(SEED, with_lactose_uptake(fast, e, 0.0), e)
    assert lost["births"] == 0 and lost["n_end"] == N and lost["deaths"] == 0
    assert lost["mass_gain_pg"] == 0.0
    print(f"\nnull (B. subtilis + E. coli lacY uptake): {gained['births']} births, identical non-identity state;"
          f"\nnull (E. coli lacY uptake removed): {lost['births']} births")


def test_null_control_without_lactose_neither_genome_grows(table, rows):
    """The same two genomes in the same medium with lactose removed: zero births and zero mass gain for
    both. So the growth above requires the substrate, not just the presence of a genome label."""
    e, b = rows[EC], rows[BS]
    fast = growth_table(table, GROWTH_CLOCK)
    for accession, row_index in ((EC, e), (BS, b)):
        out = culture(SEED, fast, row_index, lactose_mM=0.0)
        assert out["births"] == 0 and out["n_end"] == N and out["deaths"] == 0, (accession, out["births"])
        assert out["mass_gain_pg"] == 0.0, accession
        assert out["min_lactose_mM"] == 0.0, accession


# ============================================ 4. the same claim at the documented (1x) scale
def test_at_the_documented_calibration_ecoli_gains_biomass_and_b_subtilis_does_not(table, rows):
    """No clock: the frozen uptake and division values. Over 120 ticks E. coli's mass gain is exactly
    uptake x biomass_fraction x molar_mass x time, B. subtilis's is exactly zero, and the documented
    crossing time is ~833 s - which is why the population experiment needs the clock."""
    e, b = rows[EC], rows[BS]
    slow = growth_table(table, 1.0)
    ecoli = culture(SEED, slow, e, ticks=120)
    bacillus = culture(SEED, slow, b, ticks=120)
    expected = (N * float(table.max_uptake_mmol_s[e, LAC]) * float(table.biomass_fraction[e, LAC])
                * MOLAR[LAC] * 1e9 * 120 * DT)
    assert ecoli["mass_gain_pg"] == pytest.approx(expected, rel=1e-9)
    assert ecoli["mass_gain_pg"] > 0.0 and bacillus["mass_gain_pg"] == 0.0
    assert growth_prediction(table, e)["t_cross_s"] > 800.0
    print(f"\nat 1x: E. coli mass +{ecoli['mass_gain_pg']:.6f} pg in 2 s (exact {expected:.6f}); "
          f"B. subtilis +{bacillus['mass_gain_pg']:.6f} pg; documented crossing {growth_prediction(table, e)['t_cross_s']:.0f} s")
