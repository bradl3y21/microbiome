"""README must-do #2: competition. Two species competing for ONE nutrient measurably change the outcome.

The claim under test is that exploitation competition is real in D1: when two populations draw on the same
finite nutrient pool in the same field, each population's outcome differs from what it achieves alone on the
same seed, and the difference is caused by *sharing the nutrient* — not merely by a second population existing.

Experiment (every run: same seed, same positions, same field, same total nutrient; only the occupants differ).
The two demo genomes share the frozen glucose trait exactly (``max_uptake_mmol_s[glucose]`` is equal and > 0),
and lactose is a channel only E. coli can take up, which gives a clean resource-segregation null.

  shared nutrient : glucose, the only primed channel; field is closed (diffusion = 0), so the pool is finite
  populations     : mono E alone, mono B alone, and E + B together, at identical seeded positions
  outcome         : the population-level harvest — summed biomass gained by the population, pg —
                    plus the tick at which the shared pool is exhausted
  prediction      : the pool can only be converted once, so mono harvest = pool_mmol * biomass_fraction *
                    molar_mass * 1e9 pg, and in a symmetric co-culture each population takes about half of it
  null controls   : (a) a neighbour that cannot take up the shared nutrient, (b) a nutrient only one species
                    can use, (c) two monocultures of *different* species with identical traits and positions

Isolation step: B. subtilis antagonism (bacilysin) is switched off in every run, because killing is a different
interaction mechanism (it has its own tests in tests/test_two_genomes.py and tests/test_antagonism.py). Apart
from that, every number below is the frozen data/gene_trait_table.json value.

Calibration limit, asserted rather than ignored: with the frozen table the division rate (2e-4 /s) and the
uptake/division-mass ratio put a division ~3e5 ticks away, so population *size* cannot move inside any sane
test horizon. The outcome measured here is therefore the population's harvested resource, and the counts are
asserted to stay constant so a future recalibration forces this file to be revisited.
"""
import dataclasses

import numpy as np
import pytest

import sim_support as S
from microbiome.genome.models import NutrientChannel
from microbiome.genome.parser import load_genome
from microbiome.genome.phenotype import build_phenotype_table
from microbiome.sim.contracts import Environment, SimulationState

EC, BS = "GCF_000005845.2", "GCF_000009045.1"
CHANNELS = (NutrientChannel(0, "glucose"), NutrientChannel(1, "lactose"))
GLC, LAC = 0, 1
MOLAR = np.array([180.156, 342.297])  # g/mol, glucose and lactose (the sim derives mass from these)

SEED = 20261017
CELLS, DX, DEPTH = 4, 10.0, 3.0
N = 128                      # agents per population (each cell starts with 8 of each)
CONC = 0.05                  # mmol/L of the primed channel
TICKS = 240                  # the pool is exhausted by tick 149 alone / 75 together, so 240 is ample
VCELL_L = DX * DX * DEPTH * 1e-15
POOL_MMOL = CONC * VCELL_L * CELLS * CELLS


# --------------------------------------------------------------------------------------- frozen table
@pytest.fixture(scope="module")
def genomes():
    return {a: load_genome(a) for a in (EC, BS)}


@pytest.fixture(scope="module")
def table(genomes):
    return build_phenotype_table(list(genomes.values()), CHANNELS)


@pytest.fixture(scope="module")
def row(genomes, table):
    return {a: int(np.flatnonzero(table.genome_id == np.uint64(g.genome_id))[0]) for a, g in genomes.items()}


@pytest.fixture(scope="module")
def quiet(table, row):
    """The frozen table with B. subtilis antagonism switched off (this file is about nutrient competition)."""
    out = edit(table, antagonism_rate_s=(row[BS], 0.0), antagonism_radius_um=(row[BS], 0.0))
    assert float(out.antagonism_rate_s[row[EC]]) == 0.0 and float(out.antagonism_rate_s[row[BS]]) == 0.0
    return out


def edit(table, **changes):
    """Copy of the table with {field: (row, value) or (row, column, value)} applied; nothing else touched."""
    d = {f.name: getattr(table, f.name) for f in dataclasses.fields(table)}
    for field, spec in changes.items():
        a = d[field].copy()
        if len(spec) == 2:
            a[spec[0]] = spec[1]
        else:
            a[spec[0], spec[1]] = spec[2]
        d[field] = a
    return type(table)(**d)


# ------------------------------------------------------------------------------------------------ world
def world(seed, assign, tbl, prime=(CONC, 0.0)):
    """Closed field of CELLS x CELLS 10 um cells; `prime` is the per-channel mmol/L.

    Positions, masses, field and engine seed depend only on `seed`, so two runs with the same seed differ
    only in `assign` (which table row each agent carries) and in how many agents occupy the rows.
    """
    n, K = len(assign), 2
    rng = np.random.default_rng(seed)
    width = height = CELLS * DX
    pos = np.minimum(rng.random((n, 2)) * [width, height], np.nextafter([width, height], 0))
    bio = np.tile((0.5 / K) / (MOLAR * 1e9), (n, 1))  # 0.5 pg per agent, below the 0.6 pg division mass
    c0 = np.zeros((K, CELLS, CELLS))
    for k, value in enumerate(prime):
        c0[k] = value
    pool = dict(
        agent_id=np.arange(1, n + 1, dtype=np.uint64), position_um=pos, velocity_um_s=np.zeros((n, 2)),
        mass_pg=(bio * MOLAR * 1e9).sum(1), radius_um=np.full(n, 0.5), energy_pj=np.full(n, 1e3),
        species_id=(assign + 1).astype(np.uint32), genome_id=tbl.genome_id[assign],
        age_ticks=np.zeros(n, np.uint64), nutrient_store_mmol=np.zeros((n, K)), biomass_mmol=bio,
        tick_index=np.uint64(0), next_agent_id=np.uint64(n + 1), width_um=width, height_um=height,
        depth_um=DEPTH, dx_um=DX, dy_um=DX, nx=CELLS, ny=CELLS, nutrient_id=np.arange(K, dtype=np.uint32),
        concentration_mmol_l=c0, diffusion_um2_s=np.zeros(K), molar_mass_g_mol=MOLAR,
        detritus_mmol=np.zeros((K, CELLS, CELLS)),
        initial_mmol=c0.sum(axis=(1, 2)) * VCELL_L + bio.sum(0),
        source_cum_mmol=np.zeros(K), sink_cum_mmol=np.zeros(K), respiration_loss_mmol=np.zeros(K),
    )
    return (S.build_instance(SimulationState, pool), S.build_instance(Environment, pool),
            S.contract_attr("initialize_rngs")(seed))


def agent_arrays(state):
    """Canonical per-agent arrays as copies (the D1 names, wherever the contract nests them)."""
    collected = S.collect(state, "state")
    return {name: S.find(collected, name).copy() for name in S.AGENT_ROWS}


def run(seed, groups, tbl, prime=(CONC, 0.0), ticks=TICKS):
    """Step a world occupied by `groups` = [(table_row, count), ...] and measure each population's harvest."""
    assign = np.concatenate([np.full(count, r, dtype=np.int64) for r, count in groups])
    state, env, rngs = world(seed, assign, tbl, prime=prime)
    step = S.contract_attr("step")
    mass0 = (state.agents.biomass_mmol * MOLAR * 1e9).copy()
    exhausted_at = None
    for i in range(ticks):
        step(state, env, tbl, rngs)
        if exhausted_at is None and float(env.field.concentration_mmol_l.sum()) <= 0.0:
            exhausted_at = i + 1
    mass1 = state.agents.biomass_mmol * MOLAR * 1e9
    per = {}
    for r in sorted({int(r) for r, _ in groups}):
        m = assign == r
        per[r] = {"n": int(m.sum()), "gain_pg": float(mass1[m].sum() - mass0[m].sum())}
    return {
        "per": per,
        "gain_pg": float(mass1.sum() - mass0.sum()),
        "exhausted_at": exhausted_at,
        "n_final": int(len(state.agents.agent_id)),
        "violations": S.violations(state, env),
        "digest": S.snapshot_digest(state, env, rngs),
        "arrays": agent_arrays(state),
    }


def _report(mono_e, mono_b, co, row):
    return (
        f"\nshared pool {POOL_MMOL:.6g} mmol glucose; harvest of {N} agents in {TICKS} ticks\n"
        f"  alone  : E {mono_e['per'][row[EC]]['gain_pg']:.8g} pg (empty @{mono_e['exhausted_at']})  "
        f"B {mono_b['per'][row[BS]]['gain_pg']:.8g} pg (empty @{mono_b['exhausted_at']})\n"
        f"  together: E {co['per'][row[EC]]['gain_pg']:.8g} pg  B {co['per'][row[BS]]['gain_pg']:.8g} pg "
        f"(empty @{co['exhausted_at']}), total {co['gain_pg']:.8g} pg"
    )


# ============================================ 1. competition changes each population's outcome
def test_competition_changes_the_outcome(table, row, quiet):
    """Contract symbol for item 224: alone vs together, on the same seed, with the pool split as predicted."""
    mono_e = run(SEED, [(row[EC], N)], quiet)
    mono_b = run(SEED, [(row[BS], N)], quiet)
    co = run(SEED, [(row[EC], N), (row[BS], N)], quiet)
    print(_report(mono_e, mono_b, co, row))

    for run_result in (mono_e, mono_b, co):
        assert run_result["violations"] == [], run_result["violations"]
        assert run_result["exhausted_at"] is not None, "each run must drain the shared pool within TICKS"

    e_alone = mono_e["per"][row[EC]]["gain_pg"]
    b_alone = mono_b["per"][row[BS]]["gain_pg"]
    e_together = co["per"][row[EC]]["gain_pg"]
    b_together = co["per"][row[BS]]["gain_pg"]

    # (a) each population is measurably worse off than on its own, on the same seed
    assert e_together < e_alone and b_together < b_alone, "competition must suppress both populations"
    # (b) the suppression is the predicted one: the population harvests about *half* the pool it had to itself
    assert e_together == pytest.approx(0.5 * e_alone, rel=0.05), f"E {e_together:.8g} vs half of {e_alone:.8g}"
    assert b_together == pytest.approx(0.5 * b_alone, rel=0.05), f"B {b_together:.8g} vs half of {b_alone:.8g}"
    # (c) the pool is split, not enlarged: the combined harvest equals the single-population harvest
    assert co["gain_pg"] == pytest.approx(e_alone, rel=1e-9), \
        f"together {co['gain_pg']:.10g} vs alone {e_alone:.10g}: competition must not create nutrient"
    # (d) that total is exactly what the frozen table says the pool is worth
    predicted = POOL_MMOL * float(quiet.biomass_fraction[row[EC], GLC]) * MOLAR[GLC] * 1e9
    assert e_alone == pytest.approx(predicted, rel=1e-9), f"harvest {e_alone:.10g} != table {predicted:.10g}"
    # (e) the resource is drawn down about twice as fast with two populations on it
    assert co["exhausted_at"] < mono_e["exhausted_at"], "two populations must exhaust the pool sooner"
    assert co["exhausted_at"] == pytest.approx(0.5 * mono_e["exhausted_at"], rel=0.15)
    # (f) known calibration limit, asserted so a recalibration cannot silently pass here
    assert co["n_final"] == 2 * N and mono_e["n_final"] == N and mono_b["n_final"] == N, \
        "population size moved: the division calibration changed, so this file's framing needs revisiting"


# ============================================ 2. null controls: sharing the nutrient is the cause
def test_the_null_control_a_non_competing_neighbour_changes_nothing(table, row, quiet):
    """Same seed, same positions, same field: if the neighbour cannot use the pool, nothing changes.

    Three separate nulls, each removing a different possible explanation for the divergence above:
      (a) identity     — two monocultures of *different* species with equal traits and equal positions agree
      (b) inert        — the same neighbour, with its uptake of the shared nutrient set to zero, suppresses nothing
      (c) segregation  — a nutrient only one species can take up (lactose) is not competed for at all

    (b) and (c) compare population harvests, not per-agent trajectories: a second population shifts the engine's
    RNG stream, so the neighbour perturbs *who* gets what even when it eats nothing. What must not move is how
    much the resident population harvests in total, which is the quantity competition is about.
    """
    mono_e = run(SEED, [(row[EC], N)], quiet)
    e_alone = mono_e["per"][row[EC]]["gain_pg"]

    # (a) identity: the frozen table gives E. coli and B. subtilis the same glucose uptake and the same
    #     division & motility numbers, so with the same positions the two monocultures must agree exactly.
    mono_b = run(SEED, [(row[BS], N)], quiet)
    assert float(quiet.max_uptake_mmol_s[row[EC], GLC]) == float(quiet.max_uptake_mmol_s[row[BS], GLC])
    assert mono_b["per"][row[BS]]["gain_pg"] == e_alone, \
        "different species labels must not change a monoculture whose traits are identical"
    for name in S.AGENT_ROWS:
        if name in ("genome_id", "species_id"):
            continue
        assert mono_b["arrays"][name].tobytes() == mono_e["arrays"][name].tobytes(), \
            f"'{name}' differs although every trait and every position is identical"
    assert not np.array_equal(mono_b["arrays"]["genome_id"], mono_e["arrays"]["genome_id"])  # identities differ

    # (b) inert neighbour: same competitor, same positions, but it cannot take up the shared nutrient
    inert = edit(quiet, max_uptake_mmol_s=(row[BS], GLC, 0.0))
    with_inert = run(SEED, [(row[EC], N), (row[BS], N)], inert)
    assert with_inert["per"][row[BS]]["gain_pg"] == 0.0, "a neighbour with no uptake must harvest nothing"
    assert with_inert["exhausted_at"] is not None, "the pool must still be fully drained"
    assert with_inert["per"][row[EC]]["gain_pg"] == pytest.approx(e_alone, rel=1e-12), \
        "the resident must be back to the whole pool: no uptake means no competition"
    assert with_inert["per"][row[EC]]["gain_pg"] > 1.9 * run(
        SEED, [(row[EC], N), (row[BS], N)], quiet)["per"][row[EC]]["gain_pg"]

    # (c) resource segregation: lactose is primed instead; B. subtilis cannot take up lactose (frozen table).
    #     Lactose uptake is ~half of glucose's, so give these runs enough ticks to drain the pool completely;
    #     then "no competition" means both runs harvest the whole pool.
    assert float(quiet.max_uptake_mmol_s[row[EC], LAC]) > 0.0 == float(quiet.max_uptake_mmol_s[row[BS], LAC])
    lactose_alone = run(SEED, [(row[EC], N)], quiet, prime=(0.0, CONC), ticks=600)
    lactose_shared = run(SEED, [(row[EC], N), (row[BS], N)], quiet, prime=(0.0, CONC), ticks=600)
    assert lactose_alone["exhausted_at"] is not None and lactose_shared["exhausted_at"] is not None
    assert lactose_shared["per"][row[BS]]["gain_pg"] == 0.0
    assert lactose_shared["per"][row[EC]]["gain_pg"] == pytest.approx(
        lactose_alone["per"][row[EC]]["gain_pg"], rel=1e-9), \
        "a neighbour that cannot use the primed nutrient must not compete for it"
    assert lactose_alone["per"][row[EC]]["gain_pg"] > e_alone, \
        "the lactose run is a different resource, so it must not be bit-identical to the glucose run"


# ============================================ 3. the effect is repeatable and deterministic
def test_the_competitive_outcome_holds_across_seeds_and_repeats(table, row, quiet):
    """Same seed twice -> identical run; the ~half-harvest outcome holds for other placements too."""
    first = run(SEED, [(row[EC], N), (row[BS], N)], quiet)
    again = run(SEED, [(row[EC], N), (row[BS], N)], quiet)
    assert first["digest"] == again["digest"], "the same seed must reproduce the run exactly"

    for seed in (SEED, SEED + 1, 2026):
        alone = run(seed, [(row[EC], N)], quiet)["per"][row[EC]]["gain_pg"]
        together = run(seed, [(row[EC], N), (row[BS], N)], quiet)
        e_together = together["per"][row[EC]]["gain_pg"]
        b_together = together["per"][row[BS]]["gain_pg"]
        total = e_together + b_together
        assert e_together < alone and b_together < alone
        assert e_together == pytest.approx(0.5 * alone, rel=0.05), (seed, e_together, alone)
        assert total == pytest.approx(alone, rel=1e-9), (seed, total, alone)
