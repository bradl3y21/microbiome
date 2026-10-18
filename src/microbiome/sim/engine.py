"""Fixed-timestep, vectorized deterministic agent-based simulation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np

from .model import (
    AGENT_ARRAY_FIELDS,
    DT_S,
    RNG_ANTAGONISM,
    RNG_DEATH,
    RNG_DIVISION,
    RNG_MOTILITY,
    RNG_MUTATION,
    AgentState,
    NutrientEnvironment,
    NutrientLedger,
    Phenotype,
    PhenotypeArrays,
    make_rngs,
    phenotype_arrays,
    validate_rngs,
)
from .spatial import SpatialHash
from .contracts import (
    AgentArrays,
    Environment,
    NutrientField,
    PhenotypeTable,
    RNGStreams,
    SimulationState,
)


# The supported real-time workload is 10,000 simultaneous agents. Births are
# admitted only into vacant slots so a growing population cannot outrun the
# frame budget; deaths in the same tick free slots before division is applied.
MAX_AGENT_POPULATION = 10_000


def _group_sum(values: np.ndarray, keys: np.ndarray, size: int) -> np.ndarray:
    """Float64 grouped sum with a defined, stable input traversal order."""
    result = np.zeros(size, dtype=np.float64)
    np.add.at(result, keys, np.asarray(values, dtype=np.float64))
    return result


def _proportional_grants(requests: np.ndarray, available: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Allocate each cell/species pool proportionally, with deterministic residuals.

    ``keys`` are already ordered by canonical agent ID. If requests exceed a
    pool, all requests receive the same share apart from the final one-ulp
    residual, which is assigned to the highest agent ID in that group. This
    guarantees the accumulated grants do not exceed the available mmol.
    """
    requests = np.asarray(requests, dtype=np.float64)
    available = np.asarray(available, dtype=np.float64)
    n_groups = available.size
    total_requested = _group_sum(requests, keys, n_groups)
    scale = np.ones(n_groups, dtype=np.float64)
    limited = (total_requested > available) & (total_requested > 0.0)
    scale[limited] = np.nextafter(available[limited] / total_requested[limited], 0.0)
    grants = requests * scale[keys]

    # The last requester in each saturated group receives the exact residual
    # after summing every preceding grant. In canonical ID order this makes the
    # subsequent grouped accumulation equal the field's available mmol.
    last = np.full(n_groups, -1, dtype=np.int64)
    rows = np.arange(requests.size, dtype=np.int64)
    positive = requests > 0.0
    np.maximum.at(last, keys[positive], rows[positive])
    saturated = (total_requested >= available) & (total_requested > 0.0)
    groups = np.flatnonzero(saturated & (last >= 0))
    if groups.size:
        residual_base = grants.copy()
        residual_base[last[groups]] = 0.0
        preceding = _group_sum(residual_base, keys, n_groups)
        residual = available[groups] - preceding[groups]
        if np.any(residual < 0.0):
            raise FloatingPointError("proportional uptake rounding exceeded the available nutrient pool")
        grants[last[groups]] = residual

    granted = _group_sum(grants, keys, n_groups)
    # When the last request dominates its group, subtraction followed by
    # addition can round one ulp above the available amount. Move only that
    # deterministic residual recipient downward until the exact accumulator is
    # capped. The bounded loop corrects reductions, not individual agents.
    for _ in range(4):
        over = (granted > available) & (last >= 0)
        if not np.any(over):
            break
        grants[last[over]] = np.nextafter(grants[last[over]], 0.0)
        granted = _group_sum(grants, keys, n_groups)
    if np.any(granted > available):
        raise FloatingPointError("uptake grants exceed available nutrient mmol")
    return grants, granted


def diffuse(env: NutrientEnvironment, dt_s: float = DT_S) -> None:
    """Synchronous periodic five-point finite-volume diffusion update."""
    env.validate(dt_s)
    field = env.concentration_mmol_l
    rx = env.diffusion_um2_s * dt_s / (env.dx_um * env.dx_um)
    ry = env.diffusion_um2_s * dt_s / (env.dy_um * env.dy_um)
    center_weight = 1.0 - 2.0 * rx - 2.0 * ry
    # Flux-form convex weights avoid the cancellation in c + dt*laplacian(c),
    # so a nonnegative input and the frozen stability bound remain nonnegative
    # without clipping a negative concentration.
    env.concentration_mmol_l = (
        field * center_weight[:, None, None]
        + rx[:, None, None] * (np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2))
        + ry[:, None, None] * (np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1))
    )


def _field_amounts(env: NutrientEnvironment) -> np.ndarray:
    """Return flattened per-cell/species mmol in cell-major, nutrient-minor order."""
    return (env.concentration_mmol_l.transpose(1, 2, 0).reshape(-1) * env.cell_volume_l)


def _wrapped_cell_indices(env: NutrientEnvironment, positions_um: np.ndarray):
    """Map positions already guaranteed wrapped by the step invariant to cells."""
    ix = np.floor(positions_um[:, 0] / env.dx_um).astype(np.int64)
    iy = np.floor(positions_um[:, 1] / env.dy_um).astype(np.int64)
    return iy * env.nx + ix, ix, iy


def _update_uptake(state: AgentState, env: NutrientEnvironment, traits: PhenotypeArrays) -> None:
    n = state.n_agents
    k = env.n_nutrients
    if n == 0:
        return
    cell_id, _, _ = _wrapped_cell_indices(env, state.positions_um)
    group_keys = (cell_id[:, None] * k + np.arange(k, dtype=np.int64)[None, :]).reshape(-1)
    available = _field_amounts(env)
    requests = (traits.max_uptake_mmol_s * DT_S).reshape(-1)
    grants, grant_sum = _proportional_grants(requests, available, group_keys)
    grant_by_agent = grants.reshape(n, k)

    stored = grant_by_agent * traits.storage_fraction
    biomass = np.minimum(grant_by_agent * traits.biomass_fraction, grant_by_agent - stored)
    respired = grant_by_agent - stored - biomass
    state.nutrient_store_mmol += stored
    state.biomass_mmol += biomass
    state.energy_pj += np.sum(grant_by_agent * traits.energy_yield_pj_per_mmol, axis=1)
    state.respiration_loss_mmol += np.sum(respired, axis=0)

    remaining_mmol = available - grant_sum
    if np.any(remaining_mmol < 0.0):
        raise FloatingPointError("field update would produce a negative nutrient amount")
    env.concentration_mmol_l = (
        (remaining_mmol.reshape(env.ny * env.nx, k) / env.cell_volume_l)
        .reshape(env.ny, env.nx, k)
        .transpose(2, 0, 1)
        .copy()
    )


def _chemotaxis_and_movement(
    state: AgentState,
    env: NutrientEnvironment,
    traits: PhenotypeArrays,
    motility_rng: np.random.Generator,
) -> None:
    n = state.n_agents
    if n == 0:
        return
    # A total nutrient gradient is the signal; all calculations are grid-wide
    # or gathered vector operations, with no per-agent Python loop.
    total = np.sum(env.concentration_mmol_l, axis=0)
    grad_x = (np.roll(total, -1, axis=1) - np.roll(total, 1, axis=1)) / (2.0 * env.dx_um)
    grad_y = (np.roll(total, -1, axis=0) - np.roll(total, 1, axis=0)) / (2.0 * env.dy_um)
    _, ix, iy = _wrapped_cell_indices(env, state.positions_um)
    gradient = np.column_stack((grad_x[iy, ix], grad_y[iy, ix]))
    magnitude = np.linalg.norm(gradient, axis=1)
    chem = np.zeros_like(gradient)
    nonzero = magnitude > 0.0
    chem[nonzero] = gradient[nonzero] / magnitude[nonzero, None]
    angles = motility_rng.random(n) * (2.0 * np.pi)
    random_direction = np.column_stack((np.cos(angles), np.sin(angles)))
    direction = (
        chem * traits.chemotaxis_sensitivity[:, None]
        + random_direction * traits.random_motility_fraction[:, None]
    )
    length = np.linalg.norm(direction, axis=1)
    moving = (length > 0.0) & (traits.motility_um_s > 0.0)
    velocity = np.zeros_like(direction)
    velocity[moving] = direction[moving] / length[moving, None] * traits.motility_um_s[moving, None]
    state.velocities_um_s = velocity
    state.positions_um = env.wrap_positions(state.positions_um + velocity * DT_S)


def _antagonism_pressure(state: AgentState, env: NutrientEnvironment, traits: PhenotypeArrays) -> np.ndarray:
    n = state.n_agents
    pressure = np.zeros(n, dtype=np.float64)
    if n < 2 or not np.any(traits.antagonism_rate_s > 0.0):
        return pressure
    radius = float(np.max(traits.antagonism_radius_um))
    if radius <= 0.0:
        return pressure
    spatial = SpatialHash(env.width_um, env.height_um, radius)
    pairs, distance2 = spatial.pairs(
        state.positions_um, state.agent_id, radius, ordered=False, return_distance2=True,
        pair_groups=state.species_id,
    )
    if pairs.size == 0:
        return pressure
    left, right = pairs[:, 0], pairs[:, 1]
    attack_left = (distance2 <= traits.antagonism_radius_um[left] ** 2) & (traits.antagonism_rate_s[left] > 0.0)
    attack_right = (distance2 <= traits.antagonism_radius_um[right] ** 2) & (traits.antagonism_rate_s[right] > 0.0)
    # This is a bacteriocin-like contact hazard: antagonism adds a mortality
    # hazard to unlike species in the attacker's declared periodic radius.
    np.add.at(pressure, right[attack_left], traits.antagonism_rate_s[left[attack_left]])
    np.add.at(pressure, left[attack_right], traits.antagonism_rate_s[right[attack_right]])
    return pressure


def _add_dead_to_detritus(state: AgentState, env: NutrientEnvironment, dead: np.ndarray) -> None:
    if not np.any(dead):
        return
    cells, _, _ = _wrapped_cell_indices(env, state.positions_um[dead])
    k = env.n_nutrients
    values = state.nutrient_store_mmol[dead] + state.biomass_mmol[dead]
    keys = (cells[:, None] * k + np.arange(k, dtype=np.int64)[None, :]).reshape(-1)
    additions = _group_sum(values.reshape(-1), keys, env.nx * env.ny * k)
    detritus = env.detritus_mmol.transpose(1, 2, 0).reshape(-1)
    detritus += additions
    env.detritus_mmol = detritus.reshape(env.ny, env.nx, k).transpose(2, 0, 1).copy()


_PHENOTYPE_TABLE_FIELDS = (
    "max_uptake_mmol_s",
    "storage_fraction",
    "biomass_fraction",
    "energy_yield_pj_per_mmol",
    "motility_um_s",
    "chemotaxis_sensitivity",
    "random_motility_fraction",
    "division_rate_s",
    "division_mass_pg",
    "death_rate_s",
    "max_age_s",
    "antagonism_radius_um",
    "antagonism_rate_s",
)


def _register_variant_phenotype(phenotypes, phenotype, genome_id: int, variant_id: int) -> None:
    """Append a derived phenotype row to the mutable pair-keyed D1 catalogue."""
    if not hasattr(phenotypes, "genome_id"):
        raise TypeError("a mutation provider requires a pair-keyed PhenotypeTable")
    if int(getattr(phenotype, "genome_id")) != genome_id:
        raise ValueError("derived phenotype genome_id does not match the parent genome")
    if int(getattr(phenotype, "variant_id")) != variant_id:
        raise ValueError("derived phenotype variant_id does not match the derived record")

    genome_rows = np.asarray(phenotypes.genome_id, dtype=np.uint64)
    variant_rows = getattr(phenotypes, "variant_id", None)
    if variant_rows is None:
        variant_rows = np.zeros(genome_rows.shape, dtype=np.uint64)
    else:
        variant_rows = np.asarray(variant_rows, dtype=np.uint64)
        if variant_rows.shape != genome_rows.shape:
            raise ValueError("phenotype table variant_id must match genome_id rows")
    if np.any((genome_rows == genome_id) & (variant_rows == variant_id)):
        raise ValueError(f"derived phenotype key ({genome_id}, {variant_id}) is already registered")

    genome_rows = np.concatenate((genome_rows, np.asarray([genome_id], dtype=np.uint64)))
    variant_rows = np.concatenate((variant_rows, np.asarray([variant_id], dtype=np.uint64)))
    updated_fields = {}
    for name in _PHENOTYPE_TABLE_FIELDS:
        current = getattr(phenotypes, name, None)
        new_value = np.asarray(getattr(phenotype, name), dtype=np.float64)
        if current is None:
            # Legacy optional D1 columns are materialized from their D1 defaults
            # before adding the first derived row.
            defaults = {
                "max_uptake_mmol_s": 0.0,
                "storage_fraction": 0.0,
                "biomass_fraction": 1.0,
                "energy_yield_pj_per_mmol": 0.0,
                "motility_um_s": 0.0,
                "chemotaxis_sensitivity": 1.0,
                "random_motility_fraction": 0.15,
                "division_rate_s": 1e6,
                "division_mass_pg": np.inf,
                "death_rate_s": 0.0,
                "max_age_s": np.inf,
                "antagonism_radius_um": 0.0,
                "antagonism_rate_s": 0.0,
            }
            shape = (genome_rows.size - 1, new_value.size) if new_value.ndim == 1 and name in {
                "max_uptake_mmol_s", "storage_fraction", "biomass_fraction", "energy_yield_pj_per_mmol"
            } else (genome_rows.size - 1,)
            current_array = np.full(shape, defaults[name], dtype=np.float64)
        else:
            current_array = np.asarray(current, dtype=np.float64)
        expected_shape = current_array.shape[1:]
        if new_value.shape != expected_shape:
            raise ValueError(f"derived phenotype {name} shape {new_value.shape} != {expected_shape}")
        updated_fields[name] = np.ascontiguousarray(
            np.concatenate((current_array, new_value.reshape((1,) + expected_shape)), axis=0)
        )

    order = np.lexsort((variant_rows, genome_rows))
    phenotypes.genome_id = np.ascontiguousarray(genome_rows[order])
    phenotypes.variant_id = np.ascontiguousarray(variant_rows[order])
    for name in _PHENOTYPE_TABLE_FIELDS:
        setattr(phenotypes, name, np.ascontiguousarray(updated_fields[name][order]))


def _divide(
    state: AgentState,
    env: NutrientEnvironment,
    traits: PhenotypeArrays,
    division_draw: np.ndarray,
    division_angle: np.ndarray,
    *,
    phenotypes=None,
    mutation_provider=None,
    mutation_rng: np.random.Generator | None = None,
    origin_tick: int = 0,
) -> np.ndarray:
    """Binary division; returns the parent rows (ascending), whose daughters are appended in the same order."""
    probability = -np.expm1(-traits.division_rate_s * DT_S)
    divide = (state.mass_pg >= traits.division_mass_pg) & (division_draw < probability)
    parents = np.flatnonzero(divide)
    capacity = MAX_AGENT_POPULATION - state.n_agents
    if capacity <= 0:
        return np.empty(0, dtype=np.intp)
    if parents.size > capacity:
        # A full birth queue must not favor low agent IDs. The independent
        # division draws provide deterministic uniform admission priority;
        # restore ID order afterward so allocated daughter IDs remain stable.
        priority = np.argsort(division_draw[parents], kind="stable")[:capacity]
        parents = np.sort(parents[priority])
    count = parents.size
    if count == 0:
        return np.empty(0, dtype=np.intp)
    max_id = np.iinfo(np.uint64).max
    if state.next_agent_id > max_id - count:
        raise OverflowError("agent_id uint64 space exhausted")

    daughter_store = state.nutrient_store_mmol[parents].copy()
    daughter_biomass = state.biomass_mmol[parents].copy()
    daughter_energy = state.energy_pj[parents].copy()
    state.nutrient_store_mmol[parents] *= 0.5
    state.biomass_mmol[parents] *= 0.5
    state.energy_pj[parents] *= 0.5
    daughter_store -= state.nutrient_store_mmol[parents]
    daughter_biomass -= state.biomass_mmol[parents]
    daughter_energy -= state.energy_pj[parents]

    old_positions = state.positions_um[parents].copy()
    new_radius = state.radius_um[parents] * np.cbrt(0.5)
    offset = 0.5 * new_radius
    unit = np.column_stack((np.cos(division_angle[parents]), np.sin(division_angle[parents])))
    state.positions_um[parents] = env.wrap_positions(old_positions - unit * offset[:, None])
    daughters = env.wrap_positions(old_positions + unit * offset[:, None])
    state.radius_um[parents] = new_radius

    # Every population-sized operation below is vectorized; new IDs are
    # allocated in parent-ID order because the state was canonicalized first.
    child_id = np.arange(state.next_agent_id, state.next_agent_id + count, dtype=np.uint64)
    child_values = {
        "agent_id": child_id,
        "positions_um": daughters,
        "velocities_um_s": state.velocities_um_s[parents].copy(),
        "mass_pg": np.zeros(count, dtype=np.float64),
        "radius_um": new_radius.copy(),
        "energy_pj": daughter_energy,
        "species_id": state.species_id[parents].copy(),
        "genome_id": state.genome_id[parents].copy(),
        "variant_id": state.variant_id[parents].copy(),
        "age_s": np.zeros(count, dtype=np.float64),
        "nutrient_store_mmol": daughter_store,
        "biomass_mmol": daughter_biomass,
    }
    if mutation_provider is not None:
        if mutation_rng is None or phenotypes is None:
            raise ValueError("mutation requires both the mutation RNG and pair-keyed phenotype table")
        daughter_variants = child_values["variant_id"].copy()
        for child_row, parent_row in enumerate(parents):
            result = mutation_provider.create_offspring_variant(
                int(state.genome_id[parent_row]),
                int(state.variant_id[parent_row]),
                mutation_rng=mutation_rng,
                origin_tick=np.uint64(origin_tick),
                parent_agent_id=np.uint64(state.agent_id[parent_row]),
                daughter_agent_id=np.uint64(child_id[child_row]),
            )
            if result is None:
                continue
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("mutation provider must return (DerivedVariant, GenomePhenotype) or None")
            variant, phenotype = result
            variant_id = int(variant.variant_id)
            if not 0 < variant_id <= np.iinfo(np.uint64).max:
                raise ValueError("derived variant_id must be a nonzero uint64")
            root_genome_id = int(state.genome_id[parent_row])
            if int(variant.root_genome_id) != root_genome_id:
                raise ValueError("derived variant root_genome_id does not match the parent genome")
            if int(variant.parent_variant_id) != int(state.variant_id[parent_row]):
                raise ValueError("derived variant parent_variant_id does not match the parent agent")
            if int(variant.origin_tick) != int(origin_tick):
                raise ValueError("derived variant origin_tick does not match the division tick")
            if int(variant.parent_agent_id) != int(state.agent_id[parent_row]):
                raise ValueError("derived variant parent_agent_id does not match the dividing agent")
            if int(variant.daughter_agent_id) != int(child_id[child_row]):
                raise ValueError("derived variant daughter_agent_id does not match the allocated child")
            _register_variant_phenotype(phenotypes, phenotype, root_genome_id, variant_id)
            daughter_variants[child_row] = np.uint64(variant_id)
        child_values["variant_id"] = daughter_variants
    for name in AGENT_ARRAY_FIELDS:
        setattr(state, name, np.ascontiguousarray(np.concatenate((getattr(state, name), child_values[name]), axis=0)))
    state.next_agent_id += count
    return parents


def _derive_mass(state: AgentState, env: NutrientEnvironment) -> None:
    state.mass_pg = np.ascontiguousarray(
        np.sum(state.biomass_mmol * (env.molar_mass_g_mol * 1e9)[None, :], axis=1)
    )


def _internal_ledger(state: AgentState, env: NutrientEnvironment) -> NutrientLedger:
    """Return the per-nutrient, globally mass-balanced mmol ledger."""
    field_mmol = np.sum(env.concentration_mmol_l, axis=(1, 2), dtype=np.float64) * env.cell_volume_l
    agents_mmol = np.sum(state.nutrient_store_mmol + state.biomass_mmol, axis=0, dtype=np.float64)
    detritus_mmol = np.sum(env.detritus_mmol, axis=(1, 2), dtype=np.float64)
    return NutrientLedger(
        initial_mmol=state.initial_mmol.copy(),
        field_mmol=field_mmol,
        agents_mmol=agents_mmol,
        detritus_mmol=detritus_mmol,
        source_cum_mmol=state.source_cum_mmol.copy(),
        sink_cum_mmol=state.sink_cum_mmol.copy(),
        respiration_loss_mmol=state.respiration_loss_mmol.copy(),
    )


def _initialize_ledger(state: AgentState, env: NutrientEnvironment) -> None:
    if state.ledger_initialized:
        return
    initial_field = np.sum(env.concentration_mmol_l, axis=(1, 2), dtype=np.float64) * env.cell_volume_l
    initial_agents = np.sum(state.nutrient_store_mmol + state.biomass_mmol, axis=0, dtype=np.float64)
    initial_detritus = np.sum(env.detritus_mmol, axis=(1, 2), dtype=np.float64)
    state.initial_mmol[:] = initial_field + initial_agents + initial_detritus
    state.ledger_initialized = True


def snapshot(state: AgentState, env: NutrientEnvironment, rngs: Mapping[int, np.random.Generator]) -> dict:
    """Return named canonical arrays and JSON-ready deterministic metadata."""
    validate_rngs(rngs)
    arrays = {name: np.ascontiguousarray(getattr(state, name)) for name in AGENT_ARRAY_FIELDS}
    arrays.update({
        "env.nutrient_id": np.ascontiguousarray(env.nutrient_id),
        "env.concentration_mmol_l": np.ascontiguousarray(env.concentration_mmol_l),
        "env.detritus_mmol": np.ascontiguousarray(env.detritus_mmol),
        "env.diffusion_um2_s": np.ascontiguousarray(env.diffusion_um2_s),
        "env.molar_mass_g_mol": np.ascontiguousarray(env.molar_mass_g_mol),
        "ledger.initial_mmol": np.ascontiguousarray(state.initial_mmol),
        "ledger.source_cum_mmol": np.ascontiguousarray(state.source_cum_mmol),
        "ledger.sink_cum_mmol": np.ascontiguousarray(state.sink_cum_mmol),
        "ledger.respiration_loss_mmol": np.ascontiguousarray(state.respiration_loss_mmol),
    })
    metadata = {
        "tick": state.tick,
        "next_agent_id": state.next_agent_id,
        "ledger_initialized": bool(state.ledger_initialized),
        "environment": {
            "dx_um": env.dx_um,
            "dy_um": env.dy_um,
            "depth_um": env.depth_um,
            "width_um": env.width_um,
            "height_um": env.height_um,
        },
        "rng_states": {
            str(tag): rngs[tag].bit_generator.state for tag in sorted(rngs)
        },
    }
    return {"arrays": arrays, "metadata": metadata}


def state_digest(state: AgentState, env: NutrientEnvironment, rngs: Mapping[int, np.random.Generator]) -> str:
    """SHA-256 digest over every canonical array, scalar counter and RNG state."""
    snap = snapshot(state, env, rngs)
    hasher = hashlib.sha256()
    for name in sorted(snap["arrays"]):
        array = snap["arrays"][name]
        hasher.update(name.encode("utf-8"))
        hasher.update(array.dtype.str.encode("ascii"))
        hasher.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        hasher.update(array.tobytes(order="C"))
    hasher.update(json.dumps(snap["metadata"], sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return hasher.hexdigest()


class Simulation:
    """One synchronous 1/60-second biological update.

    The phase order is diffusion, competitive uptake/growth, chemotaxis and
    motion, contact antagonism/mortality, then ordered binary division.
    """

    dt_s = DT_S

    def step(
        self,
        state: AgentState | SimulationState,
        env: NutrientEnvironment | Environment,
        phenotypes,
        rngs,
        mutation_provider=None,
    ) -> None:
        if isinstance(state, SimulationState):
            if not isinstance(env, Environment):
                raise TypeError("nested SimulationState requires nested D1 Environment")
            _contract_step(state, env, phenotypes, rngs, mutation_provider)
            return None
        validate_rngs(rngs)
        state.canonicalize()
        env.validate(DT_S)
        state.validate(env)
        n_initial = state.n_agents
        traits = phenotype_arrays(phenotypes, state.genome_id, env.n_nutrients, state.variant_id)
        _initialize_ledger(state, env)

        # Draw fixed-sized batches before population changes. Streams are
        # purpose-separated, and row order is canonical at this point.
        division_draw = rngs[RNG_DIVISION].random(n_initial)
        division_angle = rngs[RNG_DIVISION].random(n_initial) * (2.0 * np.pi)
        death_draw = rngs[RNG_DEATH].random(n_initial)

        diffuse(env, DT_S)
        _update_uptake(state, env, traits)
        _derive_mass(state, env)
        _chemotaxis_and_movement(state, env, traits, rngs[RNG_MOTILITY])
        state.age_s += DT_S

        attack_pressure = _antagonism_pressure(state, env, traits)
        total_hazard = traits.death_rate_s + attack_pressure
        death_probability = -np.expm1(-total_hazard * DT_S)
        reaches_max_age = state.age_s >= traits.max_age_s
        dead = (death_draw < death_probability) | reaches_max_age
        _add_dead_to_detritus(state, env, dead)

        survivors = ~dead
        if np.any(dead):
            for name in AGENT_ARRAY_FIELDS:
                setattr(state, name, np.ascontiguousarray(getattr(state, name)[survivors]))
            division_draw = division_draw[survivors]
            division_angle = division_angle[survivors]
            traits = PhenotypeArrays(**{
                name: getattr(traits, name)[survivors]
                for name in PhenotypeArrays.__dataclass_fields__
            })

        _divide(
            state,
            env,
            traits,
            division_draw,
            division_angle,
            phenotypes=phenotypes,
            mutation_provider=mutation_provider,
            mutation_rng=rngs[RNG_MUTATION] if mutation_provider is not None else None,
            origin_tick=state.tick,
        )
        _derive_mass(state, env)
        state.tick += 1
        state.canonicalize()
        state.validate(env)
        _internal_ledger(state, env).assert_conserved()


def add_nutrient_source(env: NutrientEnvironment, state: AgentState, nutrient_index: int, cell_x: int, cell_y: int, amount_mmol: float) -> None:
    """Add a named field source while updating the conservation ledger."""
    amount = float(amount_mmol)
    if amount < 0.0 or not np.isfinite(amount):
        raise ValueError("amount_mmol must be finite and nonnegative")
    if not 0 <= nutrient_index < env.n_nutrients:
        raise IndexError("nutrient_index out of range")
    _initialize_ledger(state, env)
    env.concentration_mmol_l[nutrient_index, cell_y % env.ny, cell_x % env.nx] += amount / env.cell_volume_l
    state.source_cum_mmol[nutrient_index] += amount


def remove_nutrient_sink(env: NutrientEnvironment, state: AgentState, nutrient_index: int, cell_x: int, cell_y: int, amount_mmol: float) -> None:
    """Remove available field nutrient and record an external sink."""
    amount = float(amount_mmol)
    if amount < 0.0 or not np.isfinite(amount):
        raise ValueError("amount_mmol must be finite and nonnegative")
    if not 0 <= nutrient_index < env.n_nutrients:
        raise IndexError("nutrient_index out of range")
    _initialize_ledger(state, env)
    iy, ix = cell_y % env.ny, cell_x % env.nx
    available = env.concentration_mmol_l[nutrient_index, iy, ix] * env.cell_volume_l
    if amount > available:
        raise ValueError("sink removal exceeds nutrient available in the cell")
    remaining = available - amount
    if remaining < 0.0:
        raise FloatingPointError("sink removal would make nutrient concentration negative")
    env.concentration_mmol_l[nutrient_index, iy, ix] = remaining / env.cell_volume_l
    state.sink_cum_mmol[nutrient_index] += amount


def initialize_rngs(root_seed: int) -> RNGStreams:
    """Construct the frozen nested D1 RNG bundle from purpose-tagged PCG64 streams."""
    streams = make_rngs(root_seed)
    return RNGStreams(
        motility=streams[RNG_MOTILITY],
        division=streams[RNG_DIVISION],
        mutation=streams[RNG_MUTATION],
        death=streams[RNG_DEATH],
        antagonism=streams[RNG_ANTAGONISM],
    )


def _contract_rng_map(rngs) -> Mapping[int, np.random.Generator]:
    if isinstance(rngs, Mapping):
        return rngs
    return {
        RNG_MOTILITY: rngs.motility,
        RNG_DIVISION: rngs.division,
        RNG_MUTATION: rngs.mutation,
        RNG_DEATH: rngs.death,
        RNG_ANTAGONISM: rngs.antagonism,
    }


def _contract_step(
    state: SimulationState, env: Environment, phenotypes: PhenotypeTable, rngs, mutation_provider=None,
    simulation=None,
) -> None:
    agents = state.agents
    field = env.field
    if np.asarray(field.concentration_mmol_l).shape != (len(field.nutrient_id), env.ny, env.nx):
        raise ValueError("contract environment grid metadata does not match concentration shape")
    nutrient_env = NutrientEnvironment(
        nutrient_id=field.nutrient_id,
        concentration_mmol_l=field.concentration_mmol_l,
        diffusion_um2_s=field.diffusion_um2_s,
        molar_mass_g_mol=field.molar_mass_g_mol,
        detritus_mmol=field.detritus_mmol,
        dx_um=env.dx_um,
        dy_um=env.dy_um,
        depth_um=env.depth_um,
        width_um=env.width_um,
        height_um=env.height_um,
    )
    if nutrient_env.nx != env.nx or nutrient_env.ny != env.ny:
        raise ValueError("environment nx/ny do not match nutrient field")
    internal = AgentState(
        agent_id=agents.agent_id,
        positions_um=agents.position_um,
        velocities_um_s=agents.velocity_um_s,
        mass_pg=agents.mass_pg,
        radius_um=agents.radius_um,
        energy_pj=agents.energy_pj,
        species_id=agents.species_id,
        genome_id=agents.genome_id,
        variant_id=agents.variant_id,
        age_s=np.asarray(agents.age_ticks, dtype=np.float64) * DT_S,
        nutrient_store_mmol=agents.nutrient_store_mmol,
        biomass_mmol=agents.biomass_mmol,
        tick=int(state.tick_index),
        next_agent_id=int(state.next_agent_id),
        initial_mmol=field.initial_mmol,
        source_cum_mmol=field.source_cum_mmol,
        sink_cum_mmol=field.sink_cum_mmol,
        respiration_loss_mmol=field.respiration_loss_mmol,
        ledger_initialized=True,
    )
    (simulation or Simulation()).step(
        internal, nutrient_env, phenotypes, _contract_rng_map(rngs), mutation_provider=mutation_provider
    )

    # Commit every agent row and counter back into the frozen nested state.
    agents.agent_id = internal.agent_id
    agents.position_um = internal.positions_um
    agents.velocity_um_s = internal.velocities_um_s
    agents.mass_pg = internal.mass_pg
    agents.radius_um = internal.radius_um
    agents.energy_pj = internal.energy_pj
    agents.species_id = internal.species_id
    agents.genome_id = internal.genome_id
    agents.variant_id = internal.variant_id
    agents.age_ticks = np.rint(internal.age_s / DT_S).astype(np.uint64)
    agents.nutrient_store_mmol = internal.nutrient_store_mmol
    agents.biomass_mmol = internal.biomass_mmol
    state.tick_index = np.uint64(internal.tick)
    state.next_agent_id = np.uint64(internal.next_agent_id)
    field.concentration_mmol_l = nutrient_env.concentration_mmol_l
    field.detritus_mmol = nutrient_env.detritus_mmol
    field.initial_mmol = internal.initial_mmol
    field.source_cum_mmol = internal.source_cum_mmol
    field.sink_cum_mmol = internal.sink_cum_mmol
    field.respiration_loss_mmol = internal.respiration_loss_mmol


def step(
    state: SimulationState | AgentState,
    env: Environment | NutrientEnvironment,
    phenotypes,
    rngs,
    mutation_provider=None,
    simulation=None,
) -> None:
    """Advance one fixed D1 tick, mutating the supplied state and environment.

    ``simulation`` optionally supplies the per-tick implementation (see ``sim.realism``); ``None`` is the frozen D1 step.
    """
    if isinstance(state, SimulationState):
        if not isinstance(env, Environment):
            raise TypeError("nested SimulationState requires nested D1 Environment")
        if not isinstance(phenotypes, PhenotypeTable):
            raise TypeError("nested SimulationState requires a PhenotypeTable")
        _contract_step(state, env, phenotypes, rngs, mutation_provider, simulation)
        return None
    (simulation or Simulation()).step(state, env, phenotypes, _contract_rng_map(rngs), mutation_provider=mutation_provider)
    return None


def ledger(state: SimulationState | AgentState, env: Environment | NutrientEnvironment) -> NutrientLedger:
    """Return a canonical nutrient ledger for either the D1 schema or native types."""
    if isinstance(state, SimulationState):
        field = env.field
        vcell = env.dx_um * env.dy_um * env.depth_um * 1e-15
        return NutrientLedger(
            initial_mmol=np.asarray(field.initial_mmol, dtype=np.float64).copy(),
            field_mmol=np.sum(field.concentration_mmol_l, axis=(1, 2), dtype=np.float64) * vcell,
            agents_mmol=np.sum(state.agents.nutrient_store_mmol + state.agents.biomass_mmol, axis=0, dtype=np.float64),
            detritus_mmol=np.sum(field.detritus_mmol, axis=(1, 2), dtype=np.float64),
            source_cum_mmol=np.asarray(field.source_cum_mmol, dtype=np.float64).copy(),
            sink_cum_mmol=np.asarray(field.sink_cum_mmol, dtype=np.float64).copy(),
            respiration_loss_mmol=np.asarray(field.respiration_loss_mmol, dtype=np.float64).copy(),
        )
    return _internal_ledger(state, env)
