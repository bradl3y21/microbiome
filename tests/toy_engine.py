"""A deliberately tiny stand-in for the D1 engine, used ONLY to test the tests.

It is not the simulation. It has the D1 names and shapes, a 'good' mode that obeys
the determinism/ledger/canonical-row rules, and cheating modes that each break one
rule. tests/test_sim_contract_selftest.py proves the contract tests pass on 'good'
and go red on every cheat.
"""
import dataclasses
import time

import numpy as np

MODE = "good"  # good | global_rng | no_canon | ignore_seed | wallclock | leaky_ledger


@dataclasses.dataclass
class AgentArrays:
    agent_id: np.ndarray
    position_um: np.ndarray
    velocity_um_s: np.ndarray
    mass_pg: np.ndarray
    radius_um: np.ndarray
    energy_pj: np.ndarray
    species_id: np.ndarray
    genome_id: np.ndarray
    age_ticks: np.ndarray
    nutrient_store_mmol: np.ndarray
    biomass_mmol: np.ndarray


@dataclasses.dataclass
class SimulationState:
    agents: AgentArrays
    tick_index: np.uint64
    next_agent_id: np.uint64


@dataclasses.dataclass
class NutrientField:
    nutrient_id: np.ndarray
    concentration_mmol_l: np.ndarray
    diffusion_um2_s: np.ndarray
    molar_mass_g_mol: np.ndarray
    detritus_mmol: np.ndarray
    initial_mmol: np.ndarray
    source_cum_mmol: np.ndarray
    sink_cum_mmol: np.ndarray
    respiration_loss_mmol: np.ndarray


@dataclasses.dataclass
class Environment:
    width_um: float
    height_um: float
    depth_um: float
    dx_um: float
    dy_um: float
    nx: int
    ny: int
    field: NutrientField


@dataclasses.dataclass
class PhenotypeTable:
    genome_id: np.ndarray
    max_uptake_mmol_s: np.ndarray
    storage_fraction: np.ndarray
    biomass_fraction: np.ndarray
    energy_yield_pj_per_mmol: np.ndarray
    motility_um_s: np.ndarray
    division_mass_pg: np.ndarray
    max_age_s: np.ndarray


@dataclasses.dataclass
class RNGStreams:
    motility: np.random.Generator


@dataclasses.dataclass
class NutrientLedger:
    initial_mmol: np.ndarray
    field_mmol: np.ndarray
    agents_mmol: np.ndarray
    detritus_mmol: np.ndarray
    source_cum_mmol: np.ndarray
    sink_cum_mmol: np.ndarray
    respiration_loss_mmol: np.ndarray


ROWS = [f.name for f in dataclasses.fields(AgentArrays)]


def initialize_rngs(root_seed):
    seed = 0 if MODE == "ignore_seed" else root_seed
    ss = np.random.SeedSequence(seed, spawn_key=(1,))
    return RNGStreams(motility=np.random.Generator(np.random.PCG64(ss)))


def step(state, env, phenotypes, rngs):
    a = state.agents
    if MODE != "no_canon":
        order = np.argsort(a.agent_id, kind="stable")
        for name in ROWS:
            arr = getattr(a, name)
            arr[:] = arr[order]
    n = len(a.agent_id)
    jitter = rngs.motility.normal(size=(n, 2)) * 0.1
    if MODE == "global_rng":
        jitter += np.random.random((n, 2)) * 1e-6
    if MODE == "wallclock":
        jitter += (time.time_ns() % 1000) * 1e-12
    W, H = env.width_um, env.height_um
    pos = np.mod(a.position_um + jitter, np.array([W, H]))
    pos[pos[:, 0] >= W, 0] = 0.0
    pos[pos[:, 1] >= H, 1] = 0.0
    a.position_um[:] = pos
    a.age_ticks += np.uint64(1)
    f = env.field
    vcell = env.dx_um * env.dy_um * env.depth_um * 1e-15
    loss = 0.01 * f.concentration_mmol_l
    f.concentration_mmol_l -= loss
    if MODE != "leaky_ledger":
        f.respiration_loss_mmol += loss.sum(axis=(1, 2)) * vcell
    state.tick_index = np.uint64(int(state.tick_index) + 1)


def ledger(state, env):
    f, a = env.field, state.agents
    vcell = env.dx_um * env.dy_um * env.depth_um * 1e-15
    return NutrientLedger(
        initial_mmol=f.initial_mmol.copy(),
        field_mmol=f.concentration_mmol_l.sum(axis=(1, 2)) * vcell,
        agents_mmol=(a.nutrient_store_mmol + a.biomass_mmol).sum(axis=0),
        detritus_mmol=f.detritus_mmol.sum(axis=(1, 2)),
        source_cum_mmol=f.source_cum_mmol.copy(),
        sink_cum_mmol=f.sink_cum_mmol.copy(),
        respiration_loss_mmol=f.respiration_loss_mmol.copy(),
    )
