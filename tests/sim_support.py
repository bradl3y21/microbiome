"""Black-box support for testing Simulation.step against the frozen D1 contract.

Everything here uses D1's *names* (agent_id, concentration_mmol_l, ...) and finds
them wherever the contract dataclasses nest them, so tests do not depend on how
core/contracts.py groups fields. Construction is the only place that touches the
real classes (build_instance); a mismatch raises naming the exact missing field.
"""
import dataclasses
import importlib
import json
import typing

import numpy as np

from conftest import digest_arrays

DT_S = 1.0 / 60.0
AGENT_ROWS = (
    "agent_id", "position_um", "velocity_um_s", "mass_pg", "radius_um", "energy_pj",
    "species_id", "genome_id", "age_ticks", "nutrient_store_mmol", "biomass_mmol",
)
LEDGER_TERMS = (
    "initial_mmol", "field_mmol", "agents_mmol", "detritus_mmol",
    "source_cum_mmol", "sink_cum_mmol", "respiration_loss_mmol",
)
RTOL, ATOL = 1e-12, 1e-15


# ---------------------------------------------------------------- canonical walk
def collect(obj, path="", out=None):
    """Flatten every piece of state reachable from obj into {dotted_path: ndarray}.

    Unknown types raise instead of being skipped: a digest that silently ignores
    part of the state would pass determinism tests it should fail.
    """
    out = {} if out is None else out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            collect(getattr(obj, f.name), f"{path}.{f.name}" if path else f.name, out)
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError(f"object-dtype array at {path}")
        out[path] = obj
    elif isinstance(obj, np.generic):
        out[path] = np.asarray(obj)
    elif isinstance(obj, np.random.Generator):
        collect(obj.bit_generator, path, out)
    elif isinstance(obj, np.random.BitGenerator):
        blob = json.dumps(obj.state, sort_keys=True).encode()
        out[path + ".rng_state_json"] = np.frombuffer(blob, dtype=np.uint8)
    elif isinstance(obj, bool):
        out[path] = np.asarray(obj, dtype=np.bool_)
    elif isinstance(obj, int):
        out[path] = np.asarray(obj, dtype=np.int64 if obj < 2**63 else np.uint64)
    elif isinstance(obj, float):
        out[path] = np.asarray(obj, dtype=np.float64)
    elif isinstance(obj, str) or obj is None:
        out[path] = np.frombuffer(repr(obj).encode(), dtype=np.uint8)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            collect(obj[k], f"{path}[{k!r}]", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            collect(v, f"{path}[{i}]", out)
    else:
        raise TypeError(f"digest cannot see {type(obj).__name__} at {path!r}")
    return out


def snapshot_digest(state, env, rngs, ledger_obj=None):
    """rngs=None leaves RNG states out (used to ask whether the *seed* changed the physics)."""
    parts = {}
    for label, obj in (("state", state), ("env", env), ("rngs", rngs), ("ledger", ledger_obj)):
        if obj is not None:
            parts.update(collect(obj, label))
    return digest_arrays(parts)


def find(collected, name):
    """The one collected array whose leaf name is `name`."""
    hits = [k for k in collected if k == name or k.endswith("." + name)]
    if len(hits) != 1:
        raise LookupError(f"expected exactly one '{name}' in state, found {hits}")
    return collected[hits[0]]


# --------------------------------------------------------------------- checkers
def violations(state, env):
    """Every impossible-state check from D1; returns a list of human-readable strings."""
    c = collect(state, "state")
    c.update(collect(env, "env"))
    bad = []
    for path, a in c.items():
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            bad.append(f"non-finite value in {path}")
    g = lambda n: find(c, n)  # noqa: E731
    ids = g("agent_id")
    n = len(ids)
    for name in AGENT_ROWS:
        if len(g(name)) != n:
            bad.append(f"{name} has {len(g(name))} rows, agent_id has {n}")
    if ids.dtype != np.uint64:
        bad.append(f"agent_id dtype {ids.dtype}, want uint64")
    if n > 1 and not (ids[1:] > ids[:-1]).all():
        bad.append("agent_id not strictly increasing")
    if n and int(g("next_agent_id")) <= int(ids.max()):
        bad.append("next_agent_id <= max(agent_id): an id would be reused")
    if (g("mass_pg") < 0).any():
        bad.append("negative mass_pg")
    if (g("radius_um") <= 0).any():
        bad.append("non-positive radius_um")
    if (g("energy_pj") < 0).any():
        bad.append("negative energy_pj")
    pos = g("position_um")
    w, h = float(g("width_um")), float(g("height_um"))
    if n and not ((pos[:, 0] >= 0).all() and (pos[:, 0] < w).all()
                  and (pos[:, 1] >= 0).all() and (pos[:, 1] < h).all()):
        bad.append(f"position outside [0,{w})x[0,{h})")
    for name in ("nutrient_store_mmol", "biomass_mmol", "detritus_mmol"):
        if (g(name) < 0).any():
            bad.append(f"negative {name}")
    if (g("concentration_mmol_l") < 0).any():
        bad.append(f"negative concentration_mmol_l (min {g('concentration_mmol_l').min():.3e})")
    derived = (g("biomass_mmol") * g("molar_mass_g_mol")[None, :] * 1e9).sum(axis=1)
    if not np.allclose(g("mass_pg"), derived, rtol=1e-12, atol=1e-300):
        bad.append("mass_pg != sum_k(biomass_mmol*molar_mass*1e9) at rtol 1e-12")
    return bad


def ledger_residuals(ledger_obj):
    """Per species: (lhs - rhs, allowed). lhs=initial+source, rhs=field+agents+detritus+sink+respiration."""
    t = {k: np.asarray(getattr(ledger_obj, k), dtype=np.float64) for k in LEDGER_TERMS}
    lhs = t["initial_mmol"] + t["source_cum_mmol"]
    rhs = (t["field_mmol"] + t["agents_mmol"] + t["detritus_mmol"]
           + t["sink_cum_mmol"] + t["respiration_loss_mmol"])
    allowed = ATOL + RTOL * np.maximum(np.abs(lhs), np.abs(rhs))
    return lhs - rhs, allowed


def ledger_tight_excess(ledger_obj):
    """Scale-aware check, stricter than D1: |residual| must be <= 1e-12 * (initial + source) per species.
    Returns the max amount (mmol) by which any species exceeds that; <= 0 means closed."""
    r, _ = ledger_residuals(ledger_obj)
    scale = (np.abs(np.asarray(ledger_obj.initial_mmol, dtype=np.float64))
             + np.abs(np.asarray(ledger_obj.source_cum_mmol, dtype=np.float64)))
    return float(np.max(np.abs(r) - 1e-12 * scale))


def ledger_ok(ledger_obj):
    r, allowed = ledger_residuals(ledger_obj)
    return bool((np.abs(r) <= allowed).all())


# ------------------------------------------------------------------ construction
def _contract_modules():
    mods = []
    for name in ("microbiome.core.contracts", "microbiome.sim.engine",
                 "microbiome.sim", "microbiome.core"):
        try:
            mods.append(importlib.import_module(name))
        except ImportError:
            pass
    return mods


def contract_attr(name):
    for m in _contract_modules():
        if hasattr(m, name):
            return getattr(m, name)
    raise ImportError(f"D1 requires '{name}' in microbiome.core.contracts or microbiome.sim.engine")


def build_instance(cls, pool, module_hint=None):
    """Instantiate a D1 dataclass from a flat {d1_name: value} pool, recursing into
    dataclass-typed fields. Fields not in the pool must have defaults."""
    hints = typing.get_type_hints(cls, localns={
        n: getattr(m, n) for m in _contract_modules() for n in dir(m)
    })
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in pool:
            kwargs[f.name] = pool[f.name]
            continue
        t = hints.get(f.name)
        if isinstance(t, type) and dataclasses.is_dataclass(t):
            kwargs[f.name] = build_instance(t, pool)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise TypeError(f"{cls.__name__}.{f.name} is not a D1 name and has no default; "
                            f"tests cannot construct it (add it to the contract or give it a default)")
    return cls(**kwargs)


def build_world(seed, n_agents=150, nx=12, ny=12, K=2, scenario="mixed",
                dx=7.3, dy=5.9, depth=3.1, D=10.0, conc=(2.0, 50.0), permute_rows=False,
                engine_seed=None, antagonism=False):
    """Deterministic initial (state, env, phenotypes, rngs) from `seed`.

    scenario 'mixed'      : divisions and deaths both occur, nutrient not limiting.
    scenario 'steady'     : like 'mixed' but division and age-death disabled, so the population
                            stays constant (used by the benchmark to measure cost at exactly N).
    scenario 'full_cell'  : 20 hungry agents in every cell, D=0, so every cell is
                            fully consumed in one tick (the strict-concentration case).
    """
    rng = np.random.default_rng(seed)  # the TEST's placement stream, not the engine's
    W, H = nx * dx, ny * dy
    molar = np.array([180.16, 60.05][:K] + [100.0] * max(0, K - 2))
    if scenario == "full_cell":
        n_agents = nx * ny * 20
        D = 0.0
        cell = np.repeat(np.arange(nx * ny), 20)
        px = (cell % nx + rng.random(n_agents)) * dx
        py = (cell // nx + rng.random(n_agents)) * dy
        pos = np.stack([px, py], axis=1)
        c0 = 0.05 + rng.random((K, ny, nx)) * 9.0
        genome_idx = np.zeros(n_agents, dtype=np.int64)
    else:
        pos = rng.random((n_agents, 2)) * np.array([W, H])
        c0 = conc[0] + rng.random((K, ny, nx)) * (conc[1] - conc[0])
        genome_idx = rng.integers(0, 3, n_agents)
    pos = np.minimum(pos, np.nextafter(np.array([W, H]), 0))

    G = 3
    genome_id = np.array([101, 202, 303], dtype=np.uint64)
    start_mass = 2.0  # pg
    share = rng.random((n_agents, K)) + 0.5
    biomass0 = (share / share.sum(axis=1, keepdims=True)) * start_mass / (molar[None, :] * 1e9)
    mass0 = (biomass0 * molar[None, :] * 1e9).sum(axis=1)
    store0 = rng.random((n_agents, K)) * 1e-12
    hungry = scenario == "full_cell"
    pheno = {
        "genome_id": genome_id,
        "max_uptake_mmol_s": np.tile(np.array([1e-6 if hungry else 5e-12, 2e-12, 8e-12])[:, None], (1, K)),
        "storage_fraction": np.tile(np.array([0.2, 0.3, 0.1])[:, None], (1, K)),
        "biomass_fraction": np.tile(np.array([0.5, 0.4, 0.6])[:, None], (1, K)),
        "energy_yield_pj_per_mmol": np.tile(np.array([3e9, 2e9, 4e9])[:, None], (1, K)),
        "motility_um_s": np.array([0.0, 3.0, 6.0]) if not hungry else np.zeros(3),
        "division_mass_pg": (np.full(3, 1e9) if scenario == "steady"
                             else np.array([1e9 if hungry else 2.02, 2.02, 1e9])),
        "max_age_s": np.full(3, 1e9) if (hungry or scenario == "steady") else np.array([1e9, 1e9, 1.5]),
    }
    if antagonism:
        # The engine skips the SpatialHash/contact-antagonism phase entirely when no genome has
        # antagonism_rate_s > 0, so a benchmark without this would not measure that phase.
        pheno["antagonism_radius_um"] = np.full(G, 3.0)
        pheno["antagonism_rate_s"] = np.full(G, 1e-3)
    agents = {
        "agent_id": np.arange(1, n_agents + 1, dtype=np.uint64),
        "position_um": pos,
        "velocity_um_s": np.zeros((n_agents, 2)),
        "mass_pg": mass0,
        "radius_um": np.full(n_agents, 0.5),
        "energy_pj": np.full(n_agents, 1e3),
        "species_id": genome_idx.astype(np.uint32),
        "genome_id": genome_id[genome_idx],
        "age_ticks": rng.integers(0, 60, n_agents).astype(np.uint64),
        "nutrient_store_mmol": store0,
        "biomass_mmol": biomass0,
    }
    vcell = dx * dy * depth * 1e-15
    zeros = np.zeros(K)
    pool = {  # NOT merged with pheno: both tables have a 'genome_id' column of different length
        **agents,
        "tick_index": np.uint64(0), "next_agent_id": np.uint64(n_agents + 1),
        "width_um": W, "height_um": H, "depth_um": depth, "dx_um": dx, "dy_um": dy, "nx": nx, "ny": ny,
        "nutrient_id": np.arange(1, K + 1, dtype=np.uint32),
        "concentration_mmol_l": c0,
        "diffusion_um2_s": np.full(K, D),
        "molar_mass_g_mol": molar,
        "detritus_mmol": np.zeros((K, ny, nx)),
        "initial_mmol": c0.sum(axis=(1, 2)) * vcell + store0.sum(axis=0) + biomass0.sum(axis=0),
        "source_cum_mmol": zeros.copy(), "sink_cum_mmol": zeros.copy(),
        "respiration_loss_mmol": zeros.copy(),
    }
    state = build_instance(contract_attr("SimulationState"), pool)
    env = build_instance(contract_attr("Environment"), pool)
    phenotypes = build_instance(contract_attr("PhenotypeTable"), pheno)
    rngs = contract_attr("initialize_rngs")(seed if engine_seed is None else engine_seed)
    if permute_rows:
        permute_agent_rows(state, np.random.default_rng(seed + 9999))
    return state, env, phenotypes, rngs


def permute_agent_rows(state, rng):
    """Shuffle every per-agent array in the same way (D1: rows may be permuted at step entry)."""
    flat = collect(state, "state")
    n = len(find(flat, "agent_id"))
    perm = rng.permutation(n)
    for path, arr in flat.items():
        leaf = path.split(".")[-1]
        if leaf in AGENT_ROWS:
            arr[:] = arr[perm]  # collect() returns views of the live arrays
    return perm


def run_trace(seed, ticks, include_rngs=True, between=None, **world_kwargs):
    """Digest after every tick (and the initial state). The engine API is D1's: step() mutates in place.

    between: optional callable run before every step, used to perturb process-global
    state (global RNGs, wall clock) that a deterministic step must be immune to.
    """
    step = contract_attr("step")
    ledger = contract_attr("ledger")
    state, env, pheno, rngs = build_world(seed, **world_kwargs)
    r = rngs if include_rngs else None
    out = [snapshot_digest(state, env, r, ledger(state, env))]
    for _ in range(ticks):
        if between is not None:
            between()
        step(state, env, pheno, rngs)
        out.append(snapshot_digest(state, env, r, ledger(state, env)))
    return out


def run_checked(seed, ticks, **world_kwargs):
    """Step while recording, per tick: violations, ledger residual, id births/deaths, tick delta."""
    step = contract_attr("step")
    ledger = contract_attr("ledger")
    state, env, pheno, rngs = build_world(seed, **world_kwargs)
    ids = lambda: find(collect(state, "state"), "agent_id").copy()  # noqa: E731
    tick = lambda: int(find(collect(state, "state"), "tick_index"))  # noqa: E731
    prev_ids, prev_tick = ids(), tick()
    log = []
    for _ in range(ticks):
        step(state, env, pheno, rngs)
        cur = ids()
        lg = ledger(state, env)
        resid, allowed = ledger_residuals(lg)
        log.append({
            "ledger_tight_excess": ledger_tight_excess(lg),
            "tick": tick(), "tick_delta": tick() - prev_tick,
            "violations": violations(state, env),
            "ledger_excess": float(np.max(np.abs(resid) - allowed)),
            "ledger_resid": float(np.max(np.abs(resid))),
            "births": int(np.setdiff1d(cur, prev_ids).size),
            "deaths": int(np.setdiff1d(prev_ids, cur).size),
            "n": int(cur.size),
        })
        prev_ids, prev_tick = cur, tick()
    return log
