"""README must-do #1: two organisms with different genomes behave differently, for reasons you can point at.

E. coli K-12 MG1655 (GCF_000005845.2) and B. subtilis 168 (GCF_000009045.1), the two demo genomes, run through
the real pipeline: genome module -> PhenotypeTable -> Simulation.step. Same seed, same environment, same initial
agents; only the genome (its table row) differs. Two measured quantities diverge, and each divergence is
  (1) predicted from the table's numbers, and
  (2) shown to FOLLOW from one named trait by intervention: change only that trait and the divergence follows it.

The frozen table currently differs in exactly three cells (asserted below, so a table change forces a review):
  max_uptake_mmol_s[lactose]   E. coli > 0, B. subtilis == 0    rule ecoli-lacy-lactose-uptake      gene lacY (b0343)
  antagonism_rate_s / _radius  B. subtilis > 0, E. coli == 0    rules bacillus-bacilysin-*          genes bacA-bacG
Measured quantity 1: lactose taken from the field by a pure culture (exact prediction: sum over ticks of N * uptake * dt).
Measured quantity 2: survivors of a 50/50 co-culture (prediction: mean-field contact hazard from rate, radius, density).
"""
import dataclasses
import json
from collections import defaultdict

import numpy as np
import pytest

import sim_support as S
from conftest import ROOT
from microbiome.genome.models import NutrientChannel
from microbiome.genome.parser import load_genome
from microbiome.genome.phenotype import build_phenotype_table, derive_phenotype
from microbiome.sim.contracts import Environment, SimulationState

EC, BS = "GCF_000005845.2", "GCF_000009045.1"
CHANNELS = (NutrientChannel(0, "glucose"), NutrientChannel(1, "lactose"))
GLC, LAC = 0, 1
MOLAR = np.array([180.156, 342.297])  # g/mol glucose, lactose (standard values; the sim derives mass from these)
DT = 1.0 / 60.0
SEED = 20261017
TABLE_JSON = json.loads((ROOT / "data" / "gene_trait_table.json").read_text(encoding="utf-8"))
RULES = {r["rule_id"]: r for r in TABLE_JSON["rules"]}

# The three cells that differ in the frozen table. If this set changes the experiments below no longer
# describe the table: update them deliberately.
EXPECTED_DIFFERENCES = {("max_uptake_mmol_s", LAC), ("antagonism_radius_um", None), ("antagonism_rate_s", None)}
NUTRIENT_FIELDS = ("max_uptake_mmol_s", "storage_fraction", "biomass_fraction", "energy_yield_pj_per_mmol")
SCALAR_FIELDS = ("motility_um_s", "chemotaxis_sensitivity", "random_motility_fraction", "division_rate_s",
                 "division_mass_pg", "death_rate_s", "max_age_s", "antagonism_radius_um", "antagonism_rate_s")


# ------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def genomes():
    return {a: load_genome(a) for a in (EC, BS)}


@pytest.fixture(scope="module")
def table(genomes):
    return build_phenotype_table(list(genomes.values()), CHANNELS)


@pytest.fixture(scope="module")
def row(genomes, table):
    return {a: int(np.flatnonzero(table.genome_id == np.uint64(g.genome_id))[0]) for a, g in genomes.items()}


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


def copy_row(table, dst, src):
    """Give table row `dst` every trait of row `src` (all 13 fields); genome_id stays dst's."""
    d = {f.name: getattr(table, f.name) for f in dataclasses.fields(table)}
    for name, a in d.items():
        if name != "genome_id" and a is not None:
            a = a.copy()
            a[dst] = a[src]
            d[name] = a
    return type(table)(**d)


# ------------------------------------------------------------------------------------------- world
def world(seed, assign, table, n_cells=10, dx=10.0, dy=10.0, depth=3.0, conc=5.0, diffusion=10.0, positions=None):
    """Identical initial state for every compared run: positions, masses, field and engine seed depend only on
    `seed`; `assign` (agent -> table row) is the ONLY thing that varies between genomes.
    `positions` (um, shape [n,2]) overrides the seeded uniform placement (used for fixed geometries)."""
    n, K = len(assign), 2
    rng = np.random.default_rng(seed)
    W, H = n_cells * dx, n_cells * dy
    pos = np.minimum(rng.random((n, 2)) * [W, H], np.nextafter([W, H], 0))
    if positions is not None:
        pos = np.asarray(positions, dtype=np.float64).copy()
    bio = np.tile((0.5 / 2) / (MOLAR * 1e9), (n, 1))  # 0.5 pg, below the 0.6 pg division mass
    vcell = dx * dy * depth * 1e-15
    c0 = np.full((K, n_cells, n_cells), conc)
    pool = dict(
        agent_id=np.arange(1, n + 1, dtype=np.uint64), position_um=pos, velocity_um_s=np.zeros((n, 2)),
        mass_pg=(bio * MOLAR * 1e9).sum(1), radius_um=np.full(n, 0.5), energy_pj=np.full(n, 1e3),
        species_id=(assign + 1).astype(np.uint32), genome_id=table.genome_id[assign],
        age_ticks=np.zeros(n, np.uint64), nutrient_store_mmol=np.zeros((n, K)), biomass_mmol=bio,
        tick_index=np.uint64(0), next_agent_id=np.uint64(n + 1), width_um=W, height_um=H, depth_um=depth,
        dx_um=dx, dy_um=dy, nx=n_cells, ny=n_cells, nutrient_id=np.arange(K, dtype=np.uint32),
        concentration_mmol_l=c0, diffusion_um2_s=np.full(K, diffusion), molar_mass_g_mol=MOLAR,
        detritus_mmol=np.zeros((K, n_cells, n_cells)), initial_mmol=c0.sum(axis=(1, 2)) * vcell + bio.sum(0),
        source_cum_mmol=np.zeros(K), sink_cum_mmol=np.zeros(K), respiration_loss_mmol=np.zeros(K),
    )
    return (S.build_instance(SimulationState, pool), S.build_instance(Environment, pool),
            S.contract_attr("initialize_rngs")(seed))


def agent_arrays(state):
    c = S.collect(state, "s")
    return {n: S.find(c, n).copy() for n in S.AGENT_ROWS}


def pure_culture(seed, table, row_index, ticks=120, n=1000):
    """All agents carry table row `row_index`. Returns what was taken from the field and what the agents gained."""
    step, ledger = S.contract_attr("step"), S.contract_attr("ledger")
    state, env, rngs = world(seed, np.full(n, row_index), table)
    before, L0, counts = agent_arrays(state), ledger(state, env), []
    for _ in range(ticks):
        counts.append(len(state.agents.agent_id))
        step(state, env, table, rngs)
    L1, after = ledger(state, env), agent_arrays(state)
    assert S.violations(state, env) == [] and S.ledger_ok(L1) and S.ledger_tight_excess(L1) <= 0
    consumed = L0.field_mmol - L1.field_mmol
    predicted = np.array([sum(counts) * table.max_uptake_mmol_s[row_index, k] * DT for k in range(2)])
    return {"consumed": consumed, "predicted": predicted, "counts": counts, "before": before, "after": after,
            "mass_gain_pg": float(after["mass_pg"].sum() - before["mass_pg"].sum())}


def coculture(seed, table, row_e, row_b, ticks=400, n=640, n_cells=8):
    """Half E. coli, half B. subtilis, shuffled, same placement for every table variant."""
    step = S.contract_attr("step")
    assign = np.where(np.arange(n) % 2 == 0, row_e, row_b)
    np.random.default_rng(seed + 77).shuffle(assign)
    state, env, rngs = world(seed, assign, table, n_cells=n_cells)
    ids0 = state.agents.agent_id.copy()
    for _ in range(ticks):
        step(state, env, table, rngs)
    assert S.violations(state, env) == []
    alive = np.isin(ids0, state.agents.agent_id)
    return {name: {"n": int((assign == r).sum()), "dead": int((~alive[assign == r]).sum())}
            for name, r in (("E", row_e), ("B", row_b))}


def predicted_death_fraction(table, attacker_row, n_attackers, n_cells, ticks, dx=10.0):
    """Mean-field contact hazard on the other species: an attacker kills at antagonism_rate_s per second per
    contact inside antagonism_radius_um; neighbours = density * pi * r^2. 1 - exp(-rate * neighbours * t)."""
    r = float(table.antagonism_radius_um[attacker_row])
    rate = float(table.antagonism_rate_s[attacker_row])
    neighbours = n_attackers / (n_cells * dx) ** 2 * np.pi * r * r
    return 1.0 - np.exp(-rate * neighbours * ticks * DT), neighbours


# =========================================================== 1. what differs, in the frozen table
def table_differences(table, row):
    e, b = row[EC], row[BS]
    out = set()
    for f in NUTRIENT_FIELDS:
        a = getattr(table, f)
        out |= {(f, k) for k in range(a.shape[1]) if a[e, k] != a[b, k]}
    for f in SCALAR_FIELDS:
        a = getattr(table, f)
        if a[e] != a[b]:
            out.add((f, None))
    return out


def test_the_frozen_table_differs_in_exactly_three_named_traits(table, row):
    diff = table_differences(table, row)
    assert diff == EXPECTED_DIFFERENCES, f"table differences changed: extra {diff - EXPECTED_DIFFERENCES}, missing {EXPECTED_DIFFERENCES - diff}"
    e, b = row[EC], row[BS]
    assert table.max_uptake_mmol_s[e, LAC] > 0 == table.max_uptake_mmol_s[b, LAC]
    assert table.antagonism_rate_s[b] > 0 == table.antagonism_rate_s[e]
    assert table.antagonism_radius_um[b] > 0 == table.antagonism_radius_um[e]
    # shared traits are genuinely shared, so nothing else can explain a divergence
    assert table.max_uptake_mmol_s[e, GLC] == table.max_uptake_mmol_s[b, GLC] > 0


def test_each_difference_is_the_sum_of_named_gene_rules_and_the_genes_really_differ(genomes, table, row):
    e, b = row[EC], row[BS]
    phen = {EC: derive_phenotype(genomes[EC], CHANNELS), BS: derive_phenotype(genomes[BS], CHANNELS)}
    symbols = {a: {g.gene_symbol for g in genomes[a].genes if g.gene_symbol} for a in genomes}
    index = {a: {g.gene_key: g for g in genomes[a].genes} for a in genomes}
    for field, k in sorted(EXPECTED_DIFFERENCES, key=str):
        def effect(acc):
            total = 0.0
            for rid in phen[acc].applied_rule_ids:
                r = RULES[rid]
                if r["trait"] == field and (k is None or r.get("nutrient_name") == CHANNELS[k].name):
                    total += r["value"]
            return total
        a = getattr(table, field)
        got = (a[e] - a[b]) if k is None else (a[e, k] - a[b, k])
        assert got == pytest.approx(effect(EC) - effect(BS), rel=1e-12, abs=0), f"{field}[{k}] is not the rule sum"
        explaining = [rid for acc in (EC, BS) for rid in phen[acc].applied_rule_ids
                      if RULES[rid]["trait"] == field and (k is None or RULES[rid].get("nutrient_name") == CHANNELS[k].name)]
        assert explaining, f"{field}[{k}] differs but no applied rule explains it"
        for rid in explaining:
            r = RULES[rid]
            owner, other = (r["genome_accession"], EC if r["genome_accession"] == BS else BS)
            keys = r.get("requires_gene_keys", [r["gene_key"]])
            assert all(key in index[owner] for key in keys), f"{rid}: required genes are not all in {owner}"
            need = {r["gene_symbols"][key] for key in keys}
            assert not need <= symbols[other], f"{rid}: {other} also has every required gene symbol"
    # the named genes, concretely
    assert "lacY" in symbols[EC] and "lacY" not in symbols[BS]
    assert {"bacA", "bacB", "bacC", "bacD", "bacE", "bacF", "bacG"} <= symbols[BS]
    assert not {"bacB", "bacC", "bacD", "bacE", "bacF", "bacG"} & symbols[EC]
    # E. coli does carry a gene SYMBOLLED bacA (an unrelated gene); the rule needs all seven exact locus tags
    assert "ecoli-lacy-lactose-uptake" in phen[EC].applied_rule_ids
    assert "bacillus-bacilysin-antagonism-rate" in phen[BS].applied_rule_ids
    assert not any("bacilysin" in r for r in phen[EC].applied_rule_ids)


# ============================================ 2. measured quantity 1: lactose taken from the field
def test_pure_cultures_diverge_in_lactose_consumption_exactly_as_the_table_predicts(table, row):
    e, b = pure_culture(SEED, table, row[EC]), pure_culture(SEED, table, row[BS])
    # E. coli takes up lactose; B. subtilis takes up none; both take up glucose identically
    assert e["consumed"][LAC] > 0 and b["consumed"][LAC] == 0.0, \
        f"lactose: E. coli must consume and B. subtilis must not, got {e['consumed'][LAC]:.3e} / {b['consumed'][LAC]:.3e}"
    assert e["consumed"][GLC] == pytest.approx(b["consumed"][GLC], rel=1e-12), "glucose uptake is shared (ptsG)"
    # the engine's measured consumption equals sum(N * uptake * dt) from the table
    for run in (e, b):
        for k in (GLC, LAC):
            assert run["consumed"][k] == pytest.approx(run["predicted"][k], rel=1e-9, abs=0), \
                f"nutrient {k}: consumption {run['consumed'][k]:.6e} != table prediction {run['predicted'][k]:.6e}"
    # and the biomass the agents gained is that uptake times biomass_fraction times molar mass
    for run, r in ((e, row[EC]), (b, row[BS])):
        expect = sum(run["predicted"][k] * table.biomass_fraction[r, k] * MOLAR[k] * 1e9 for k in (GLC, LAC))
        assert run["mass_gain_pg"] == pytest.approx(expect, rel=1e-6)
    assert e["mass_gain_pg"] > b["mass_gain_pg"], "lactose uptake must make E. coli gain more mass"


def test_the_lactose_divergence_follows_the_lacy_trait_in_both_directions(table, row):
    e_row, b_row = row[EC], row[BS]
    real_e, real_b = pure_culture(SEED, table, e_row), pure_culture(SEED, table, b_row)
    lac_e = float(table.max_uptake_mmol_s[e_row, LAC])
    # give B. subtilis only E. coli's lactose uptake: it now consumes exactly what E. coli does
    gained = pure_culture(SEED, edit(table, max_uptake_mmol_s=(b_row, LAC, lac_e)), b_row)
    assert gained["consumed"][LAC] == pytest.approx(real_e["consumed"][LAC], rel=1e-9)
    assert gained["consumed"][GLC] == pytest.approx(real_b["consumed"][GLC], rel=1e-12)  # nothing else moved
    # remove only E. coli's lactose uptake: it now consumes none, like B. subtilis
    lost = pure_culture(SEED, edit(table, max_uptake_mmol_s=(e_row, LAC, 0.0)), e_row)
    assert lost["consumed"][LAC] == 0.0
    assert lost["consumed"][GLC] == pytest.approx(real_e["consumed"][GLC], rel=1e-12)


def test_null_control_the_engine_has_no_hidden_dependence_on_genome_identity(table, row):
    """Give B. subtilis agents E. coli's entire table row: the run must equal the E. coli run bit for bit
    except for the identity columns. So any divergence above comes through trait values, not labels."""
    e_row, b_row = row[EC], row[BS]
    ref = pure_culture(SEED, table, e_row)["after"]
    same = pure_culture(SEED, copy_row(table, b_row, e_row), b_row)["after"]
    for name in S.AGENT_ROWS:
        if name in ("genome_id", "species_id"):
            continue
        assert ref[name].tobytes() == same[name].tobytes(), f"'{name}' differs although every trait is identical"
    assert not np.array_equal(ref["genome_id"], same["genome_id"])  # the identities really did differ


# ============================================== 3. measured quantity 2: who survives a co-culture
def test_coculture_diverges_in_survival_and_matches_the_antagonism_prediction(table, row):
    e_row, b_row = row[EC], row[BS]
    for seed in (SEED, SEED + 1):
        r = coculture(seed, table, e_row, b_row)
        pred, neighbours = predicted_death_fraction(table, b_row, r["B"]["n"], 8, 400)
        e_frac, b_frac = r["E"]["dead"] / r["E"]["n"], r["B"]["dead"] / r["B"]["n"]
        assert r["E"]["n"] == r["B"]["n"] == 320
        assert e_frac > 0.05 and r["B"]["dead"] <= 2, (seed, r)  # E. coli is killed, B. subtilis is not
        assert 0.65 * pred <= e_frac <= 1.35 * pred, f"seed {seed}: E. coli died {e_frac:.3f}, predicted {pred:.3f}"
        assert e_frac > 20 * max(b_frac, 1 / 320)


def test_the_survival_divergence_follows_the_named_antagonism_traits(table, row):
    e_row, b_row = row[EC], row[BS]
    real = coculture(SEED, table, e_row, b_row)
    assert real["E"]["dead"] > 20 and real["B"]["dead"] <= 2, f"E. coli must die and B. subtilis must not; dead: {real}"
    # necessary: remove ONLY the attack rate, or ONLY the attack radius, of B. subtilis -> nobody dies
    for field in ("antagonism_rate_s", "antagonism_radius_um"):
        off = coculture(SEED, edit(table, **{field: (b_row, 0.0)}), e_row, b_row)
        assert off["E"]["dead"] <= 2 and off["B"]["dead"] <= 2, (field, off)
    # sufficient: give E. coli ONLY those two traits and the damage becomes symmetric
    both = coculture(SEED, edit(table, antagonism_rate_s=(e_row, float(table.antagonism_rate_s[b_row])),
                                antagonism_radius_um=(e_row, float(table.antagonism_radius_um[b_row]))), e_row, b_row)
    assert both["E"]["dead"] > 20 and both["B"]["dead"] > 20
    assert abs(both["E"]["dead"] - both["B"]["dead"]) <= 0.35 * max(both["E"]["dead"], both["B"]["dead"])


# =========================================================== 4. the contract symbol for item 19
def test_two_genomes_produce_distinguishable_populations(genomes, table, row):
    """Umbrella: same seed + same environment, two real genomes -> populations that differ in two measured
    quantities, each differing for a named, table-level, gene-attributed reason, and with a null control."""
    e_row, b_row = row[EC], row[BS]
    # (a) the reasons: three table cells, nothing else
    assert table_differences(table, row) == EXPECTED_DIFFERENCES
    # (b) quantity 1: lactose taken from the field, exact to the table
    e, b = pure_culture(SEED, table, e_row), pure_culture(SEED, table, b_row)
    assert e["consumed"][LAC] > 0 and b["consumed"][LAC] == 0.0
    assert e["consumed"][LAC] == pytest.approx(e["predicted"][LAC], rel=1e-9)
    # (c) quantity 2: who survives living together
    r = coculture(SEED, table, e_row, b_row)
    assert r["E"]["dead"] > 20 and r["B"]["dead"] <= 2
    # (d) causal: the divergence moves with the trait, not with the label
    off = coculture(SEED, edit(table, antagonism_rate_s=(b_row, 0.0)), e_row, b_row)
    assert off["E"]["dead"] <= 2
    gained = pure_culture(SEED, edit(table, max_uptake_mmol_s=(b_row, LAC, float(table.max_uptake_mmol_s[e_row, LAC]))), b_row)
    assert gained["consumed"][LAC] == pytest.approx(e["consumed"][LAC], rel=1e-9)
    # (e) null control: identical traits -> identical run (labels alone change nothing)
    same = pure_culture(SEED, copy_row(table, b_row, e_row), b_row)["after"]
    assert same["position_um"].tobytes() == e["after"]["position_um"].tobytes()
    assert same["biomass_mmol"].tobytes() == e["after"]["biomass_mmol"].tobytes()
    print(f"\nlactose taken (mmol): E. coli {e['consumed'][LAC]:.3e}  B. subtilis {b['consumed'][LAC]:.3e}"
          f"\nco-culture deaths of 320 each: E. coli {r['E']['dead']}  B. subtilis {r['B']['dead']}"
          f"   (B. subtilis antagonism off: E. coli {off['E']['dead']})")
