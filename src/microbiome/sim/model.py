"""Typed state and fixed-unit data structures for the simulation engine.

All agent collections are structure-of-arrays. Agent rows are canonicalized by
``agent_id`` before any random draws or reductions, so caller row order cannot
change a simulation step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import weakref

import numpy as np


DT_S = 1.0 / 60.0

# These integer tags are the frozen D1 RNG contract. Never derive a purpose
# tag from Python's process-randomized hash().
RNG_MOTILITY = 1
RNG_DIVISION = 2
RNG_MUTATION = 3
RNG_DEATH = 4
RNG_ANTAGONISM = 5
RNG_PURPOSES = (
    RNG_MOTILITY,
    RNG_DIVISION,
    RNG_MUTATION,
    RNG_DEATH,
    RNG_ANTAGONISM,
)

AGENT_ARRAY_FIELDS = (
    "agent_id",
    "positions_um",
    "velocities_um_s",
    "mass_pg",
    "radius_um",
    "energy_pj",
    "species_id",
    "genome_id",
    "variant_id",
    "age_s",
    "nutrient_store_mmol",
    "biomass_mmol",
)


def _array(value, dtype, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.ndim}")
    return np.ascontiguousarray(result)


@dataclass
class AgentState:
    """Vectorized agent state using the D1 units and canonical dtypes."""

    agent_id: np.ndarray
    positions_um: np.ndarray
    velocities_um_s: np.ndarray
    mass_pg: np.ndarray
    radius_um: np.ndarray
    energy_pj: np.ndarray
    species_id: np.ndarray
    genome_id: np.ndarray
    age_s: np.ndarray
    nutrient_store_mmol: np.ndarray
    biomass_mmol: np.ndarray
    tick: int = 0
    next_agent_id: int | None = None
    initial_mmol: np.ndarray | None = None
    source_cum_mmol: np.ndarray | None = None
    sink_cum_mmol: np.ndarray | None = None
    respiration_loss_mmol: np.ndarray | None = None
    ledger_initialized: bool = False
    variant_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.variant_id is None:
            self.variant_id = np.zeros(np.asarray(self.agent_id).size, dtype=np.uint64)
        dtypes = {
            "agent_id": np.uint64,
            "positions_um": np.float64,
            "velocities_um_s": np.float64,
            "mass_pg": np.float64,
            "radius_um": np.float64,
            "energy_pj": np.float64,
            "species_id": np.uint32,
            "genome_id": np.uint64,
            "variant_id": np.uint64,
            "age_s": np.float64,
            "nutrient_store_mmol": np.float64,
            "biomass_mmol": np.float64,
        }
        dimensions = {
            "positions_um": 2,
            "velocities_um_s": 2,
            "nutrient_store_mmol": 2,
            "biomass_mmol": 2,
        }
        for name, dtype in dtypes.items():
            setattr(self, name, _array(getattr(self, name), dtype, name, dimensions.get(name, 1)))

        n = self.agent_id.size
        for name in ("mass_pg", "radius_um", "energy_pj", "species_id", "genome_id", "variant_id", "age_s"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        if self.positions_um.shape != (n, 2) or self.velocities_um_s.shape != (n, 2):
            raise ValueError("positions_um and velocities_um_s must have shape (N, 2)")
        if self.nutrient_store_mmol.shape != self.biomass_mmol.shape:
            raise ValueError("nutrient_store_mmol and biomass_mmol must have the same shape")
        if self.nutrient_store_mmol.shape[0] != n:
            raise ValueError("nutrient arrays must have N rows")

        k = self.nutrient_store_mmol.shape[1]
        for name in ("initial_mmol", "source_cum_mmol", "sink_cum_mmol", "respiration_loss_mmol"):
            value = getattr(self, name)
            if value is None:
                value = np.zeros(k, dtype=np.float64)
            value = _array(value, np.float64, name, 1)
            if value.shape != (k,):
                raise ValueError(f"{name} must have shape ({k},)")
            setattr(self, name, value)
        if int(self.tick) < 0:
            raise ValueError("tick must be nonnegative")
        self.tick = int(self.tick)
        if self.next_agent_id is None:
            self.next_agent_id = 0 if n == 0 else int(self.agent_id.max()) + 1
        self.next_agent_id = int(self.next_agent_id)
        if not 0 <= self.next_agent_id <= np.iinfo(np.uint64).max:
            raise ValueError("next_agent_id is outside uint64 range")
        self.canonicalize()

    @property
    def n_agents(self) -> int:
        return int(self.agent_id.size)

    @property
    def n_nutrients(self) -> int:
        return int(self.nutrient_store_mmol.shape[1])

    def canonicalize(self) -> None:
        """Sort every agent array in stable ID order (safe for shuffled input)."""
        if self.agent_id.size < 2:
            return
        order = np.argsort(self.agent_id, kind="stable")
        if np.array_equal(order, np.arange(order.size)):
            return
        for name in AGENT_ARRAY_FIELDS:
            setattr(self, name, np.ascontiguousarray(getattr(self, name)[order]))

    def validate(self, env: "Environment") -> None:
        if self.n_nutrients != env.n_nutrients:
            raise ValueError("agent nutrient arrays do not match environment species count")
        if self.agent_id.size > 1 and np.any(self.agent_id[1:] <= self.agent_id[:-1]):
            raise ValueError("agent_id must be strictly increasing")
        if self.agent_id.size and self.next_agent_id <= int(self.agent_id[-1]):
            raise ValueError("next_agent_id must be greater than every allocated agent_id")
        for name in ("positions_um", "velocities_um_s", "mass_pg", "radius_um", "energy_pj", "age_s", "nutrient_store_mmol", "biomass_mmol"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} must contain only finite values")
        if np.any(self.mass_pg < 0) or np.any(self.radius_um < 0) or np.any(self.energy_pj < 0) or np.any(self.age_s < 0):
            raise ValueError("mass, radius, energy, and age must be nonnegative")
        if np.any(self.nutrient_store_mmol < 0) or np.any(self.biomass_mmol < 0):
            raise ValueError("agent nutrient quantities must be nonnegative")
        if np.any(self.positions_um[:, 0] < 0) or np.any(self.positions_um[:, 0] >= env.width_um):
            raise ValueError("x positions must be wrapped into [0, width_um)")
        if np.any(self.positions_um[:, 1] < 0) or np.any(self.positions_um[:, 1] >= env.height_um):
            raise ValueError("y positions must be wrapped into [0, height_um)")


@dataclass
class NutrientEnvironment:
    """Periodic 2D finite-volume nutrient slab with fixed depth."""

    nutrient_id: np.ndarray
    concentration_mmol_l: np.ndarray
    diffusion_um2_s: np.ndarray
    molar_mass_g_mol: np.ndarray
    dx_um: float
    dy_um: float
    depth_um: float
    width_um: float
    height_um: float
    detritus_mmol: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.nutrient_id = _array(self.nutrient_id, np.uint32, "nutrient_id", 1)
        self.concentration_mmol_l = _array(self.concentration_mmol_l, np.float64, "concentration_mmol_l", 3)
        self.diffusion_um2_s = _array(self.diffusion_um2_s, np.float64, "diffusion_um2_s", 1)
        self.molar_mass_g_mol = _array(self.molar_mass_g_mol, np.float64, "molar_mass_g_mol", 1)
        k = self.nutrient_id.size
        if self.concentration_mmol_l.shape[0] != k or self.diffusion_um2_s.shape != (k,) or self.molar_mass_g_mol.shape != (k,):
            raise ValueError("nutrient metadata and field must agree on K")
        if k == 0:
            raise ValueError("environment must define at least one nutrient")
        if self.detritus_mmol is None:
            self.detritus_mmol = np.zeros_like(self.concentration_mmol_l)
        else:
            self.detritus_mmol = _array(self.detritus_mmol, np.float64, "detritus_mmol", 3)
        if self.detritus_mmol.shape != self.concentration_mmol_l.shape:
            raise ValueError("detritus_mmol must match concentration_mmol_l shape")
        self.dx_um = float(self.dx_um)
        self.dy_um = float(self.dy_um)
        self.depth_um = float(self.depth_um)
        self.width_um = float(self.width_um)
        self.height_um = float(self.height_um)
        self.validate()

    @property
    def n_nutrients(self) -> int:
        return int(self.nutrient_id.size)

    @property
    def ny(self) -> int:
        return int(self.concentration_mmol_l.shape[1])

    @property
    def nx(self) -> int:
        return int(self.concentration_mmol_l.shape[2])

    @property
    def cell_volume_l(self) -> float:
        return self.dx_um * self.dy_um * self.depth_um * 1e-15

    def validate(self, dt_s: float = DT_S) -> None:
        if self.nutrient_id.size > 1 and np.any(self.nutrient_id[1:] <= self.nutrient_id[:-1]):
            raise ValueError("nutrient_id must be strictly increasing")
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nutrient field must have nonempty x and y dimensions")
        if min(self.dx_um, self.dy_um, self.depth_um, self.width_um, self.height_um) <= 0:
            raise ValueError("environment dimensions must be positive")
        if not np.isclose(self.width_um, self.nx * self.dx_um, rtol=1e-12, atol=1e-12):
            raise ValueError("width_um must equal nx*dx_um")
        if not np.isclose(self.height_um, self.ny * self.dy_um, rtol=1e-12, atol=1e-12):
            raise ValueError("height_um must equal ny*dy_um")
        for name, values in (("concentration_mmol_l", self.concentration_mmol_l), ("detritus_mmol", self.detritus_mmol), ("diffusion_um2_s", self.diffusion_um2_s), ("molar_mass_g_mol", self.molar_mass_g_mol)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
        if np.any(self.concentration_mmol_l < 0) or np.any(self.detritus_mmol < 0):
            raise ValueError("nutrient concentration and detritus must be nonnegative")
        if np.any(self.diffusion_um2_s < 0) or np.any(self.molar_mass_g_mol <= 0):
            raise ValueError("diffusion must be nonnegative and molar mass positive")
        stability = self.diffusion_um2_s * dt_s * (1.0 / self.dx_um**2 + 1.0 / self.dy_um**2)
        if np.any(stability > 0.5):
            raise ValueError("explicit diffusion is unstable: require D*dt*(1/dx^2+1/dy^2) <= 1/2")

    def wrap_positions(self, positions_um: np.ndarray) -> np.ndarray:
        wrapped = np.remainder(positions_um, np.array([self.width_um, self.height_um]))
        # Floating remainder is specified to be < period, but guard an exact
        # upper edge explicitly to preserve the half-open periodic domain.
        wrapped[:, 0] = np.where(wrapped[:, 0] >= self.width_um, 0.0, wrapped[:, 0])
        wrapped[:, 1] = np.where(wrapped[:, 1] >= self.height_um, 0.0, wrapped[:, 1])
        return wrapped

    def cell_indices(self, positions_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wrapped = self.wrap_positions(positions_um)
        ix = np.floor(wrapped[:, 0] / self.dx_um).astype(np.int64) % self.nx
        iy = np.floor(wrapped[:, 1] / self.dy_um).astype(np.int64) % self.ny
        return iy * self.nx + ix, ix, iy


@dataclass(frozen=True)
class Phenotype:
    """Genome-derived coefficients consumed by the engine.

    Nutrient-specific fields accept a scalar or a length-K vector. Division,
    mortality, motility and antagonism values are per-second or physical rates.
    """

    max_uptake_mmol_s: float | np.ndarray = 0.0
    storage_fraction: float | np.ndarray = 0.0
    biomass_fraction: float | np.ndarray = 1.0
    energy_yield_pj_per_mmol: float | np.ndarray = 0.0
    motility_um_s: float = 0.0
    chemotaxis_sensitivity: float = 1.0
    random_motility_fraction: float = 0.15
    division_rate_s: float = 0.0
    division_mass_pg: float = np.inf
    death_rate_s: float = 0.0
    max_age_s: float = np.inf
    antagonism_radius_um: float = 0.0
    antagonism_rate_s: float = 0.0


@dataclass
class PhenotypeArrays:
    max_uptake_mmol_s: np.ndarray
    storage_fraction: np.ndarray
    biomass_fraction: np.ndarray
    energy_yield_pj_per_mmol: np.ndarray
    motility_um_s: np.ndarray
    chemotaxis_sensitivity: np.ndarray
    random_motility_fraction: np.ndarray
    division_rate_s: np.ndarray
    division_mass_pg: np.ndarray
    death_rate_s: np.ndarray
    max_age_s: np.ndarray
    antagonism_radius_um: np.ndarray
    antagonism_rate_s: np.ndarray


def _nutrient_vector(value, k: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(k, float(array), dtype=np.float64)
    if array.shape != (k,) or not np.all(np.isfinite(array)):
        raise ValueError(f"phenotype {name} must be finite scalar or length-K vector")
    return array


def _validate_phenotype_tables(tables: dict[str, np.ndarray]) -> PhenotypeArrays:
    if np.any(tables["max_uptake_mmol_s"] < 0):
        raise ValueError("max_uptake_mmol_s must be nonnegative")
    if np.any(tables["storage_fraction"] < 0) or np.any(tables["biomass_fraction"] < 0):
        raise ValueError("storage and biomass fractions must be nonnegative")
    if np.any(tables["storage_fraction"] + tables["biomass_fraction"] > 1.0):
        raise ValueError("storage_fraction + biomass_fraction must be <= 1")
    for name in ("motility_um_s", "division_rate_s", "death_rate_s", "antagonism_radius_um", "antagonism_rate_s", "chemotaxis_sensitivity", "random_motility_fraction"):
        if np.any(tables[name] < 0):
            raise ValueError(f"phenotype {name} must be nonnegative")
    if np.any(tables["energy_yield_pj_per_mmol"] < 0):
        raise ValueError("energy_yield_pj_per_mmol must be nonnegative")
    if np.any(tables["division_mass_pg"] <= 0):
        raise ValueError("division_mass_pg must be positive")
    if np.any(tables["max_age_s"] <= 0):
        raise ValueError("max_age_s must be positive")
    return PhenotypeArrays(**tables)


_PHENOTYPE_GATHER_CACHE: dict[
    int, tuple[weakref.ReferenceType, tuple, np.ndarray, np.ndarray, PhenotypeArrays]
] = {}


def _phenotype_table_signature(table) -> tuple:
    """Small-content signature so in-place trait edits invalidate cached gathers."""
    names = (
        "genome_id", "variant_id", "max_uptake_mmol_s", "storage_fraction", "biomass_fraction",
        "energy_yield_pj_per_mmol", "motility_um_s", "chemotaxis_sensitivity",
        "random_motility_fraction", "division_rate_s", "division_mass_pg",
        "death_rate_s", "max_age_s", "antagonism_radius_um", "antagonism_rate_s",
    )
    signature = []
    for name in names:
        value = getattr(table, name, None)
        if value is None:
            signature.append(None)
        else:
            array = np.asarray(value)
            signature.append((array.shape, array.dtype.str, array.tobytes()))
    return tuple(signature)


class _PreparedPhenotypeTable:
    """Per-table work done once: key order, uniqueness, defaults, and a per-genome validity mask.

    Gathering per-agent traits is then one row lookup plus array takes. Values are the same bits the
    one-shot path produced (sorted source rows, then ``[rows]``); a row that fails validation raises
    only if an agent actually uses it, with the original messages.
    """

    NUTRIENT_DEFAULTS = {
        "max_uptake_mmol_s": 0.0,
        "storage_fraction": 0.0,
        "biomass_fraction": 1.0,
        "energy_yield_pj_per_mmol": 0.0,
    }
    SCALAR_DEFAULTS = {
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

    def __init__(self, phenotypes, k: int) -> None:
        ids = np.asarray(phenotypes.genome_id, dtype=np.uint64)
        table_variant_ids = getattr(phenotypes, "variant_id", None)
        if table_variant_ids is None:
            table_variant_ids = np.zeros(ids.shape, dtype=np.uint64)
        else:
            table_variant_ids = np.asarray(table_variant_ids, dtype=np.uint64)
            if table_variant_ids.shape != ids.shape:
                raise ValueError("phenotype variant_id must have the same shape as genome_id")
        order = np.lexsort((table_variant_ids, ids))
        self.sorted_ids = ids[order]
        sorted_variant_ids = table_variant_ids[order]
        if self.sorted_ids.size > 1 and np.any(
            (self.sorted_ids[1:] == self.sorted_ids[:-1])
            & (sorted_variant_ids[1:] == sorted_variant_ids[:-1])
        ):
            raise ValueError("phenotype (genome_id, variant_id) keys must be unique")
        self.size = int(ids.size)
        self.has_variants = bool(np.any(sorted_variant_ids != 0))
        self.key_dtype = np.dtype([("genome_id", np.uint64), ("variant_id", np.uint64)])
        self.sorted_keys = np.empty(self.size, dtype=self.key_dtype)
        self.sorted_keys["genome_id"] = self.sorted_ids
        self.sorted_keys["variant_id"] = sorted_variant_ids

        columns: dict[str, np.ndarray] = {}
        for name, default in self.NUTRIENT_DEFAULTS.items():
            source = getattr(phenotypes, name, None)
            if source is None:
                columns[name] = np.full((self.size, k), default, dtype=np.float64)
                continue
            source = np.asarray(source, dtype=np.float64)[order]
            if source.shape == (self.size,):
                source = np.broadcast_to(source[:, None], (self.size, k))
            if source.shape != (self.size, k):
                raise ValueError(f"phenotype {name} must have shape (G, K)")
            columns[name] = np.ascontiguousarray(source)
        for name, default in self.SCALAR_DEFAULTS.items():
            source = getattr(phenotypes, name, None)
            if source is None:
                columns[name] = np.full(self.size, default, dtype=np.float64)
                continue
            source = np.asarray(source, dtype=np.float64)
            if source.ndim == 0:
                source = np.full(self.size, float(source), dtype=np.float64)
            if source.shape != (self.size,):
                raise ValueError(f"phenotype {name} must have shape (G,)")
            columns[name] = np.ascontiguousarray(source[order])
        self.columns = columns

        def per_row(values: np.ndarray) -> np.ndarray:
            return values if values.ndim == 1 else np.all(values, axis=1)

        ok = np.ones(self.size, dtype=bool)
        for name, values in columns.items():
            infinity_allowed = name in ("division_mass_pg", "max_age_s")
            ok &= per_row(~np.isnan(values) if infinity_allowed else np.isfinite(values))
        ok &= per_row(columns["max_uptake_mmol_s"] >= 0)
        ok &= per_row(columns["storage_fraction"] >= 0) & per_row(columns["biomass_fraction"] >= 0)
        ok &= per_row(columns["storage_fraction"] + columns["biomass_fraction"] <= 1.0)
        for name in ("motility_um_s", "division_rate_s", "death_rate_s", "antagonism_radius_um",
                     "antagonism_rate_s", "chemotaxis_sensitivity", "random_motility_fraction"):
            ok &= columns[name] >= 0
        ok &= per_row(columns["energy_yield_pj_per_mmol"] >= 0)
        ok &= columns["division_mass_pg"] > 0
        ok &= columns["max_age_s"] > 0
        self.row_ok = ok

    def lookup(self, genome_ids: np.ndarray, variant_ids: np.ndarray) -> np.ndarray:
        if not self.has_variants and not np.any(variant_ids):
            rows = np.searchsorted(self.sorted_ids, genome_ids)
            if rows.size and (
                np.any(rows >= self.size)
                or np.any(self.sorted_ids[np.minimum(rows, self.size - 1)] != genome_ids)
            ):
                raise ValueError("phenotype table is missing a (genome_id, variant_id) used by an agent")
            return rows
        target_keys = np.empty(genome_ids.size, dtype=self.key_dtype)
        target_keys["genome_id"] = genome_ids
        target_keys["variant_id"] = variant_ids
        rows = np.searchsorted(self.sorted_keys, target_keys)
        if rows.size and (
            np.any(rows >= self.sorted_keys.size) or np.any(self.sorted_keys[rows] != target_keys)
        ):
            raise ValueError("phenotype table is missing a (genome_id, variant_id) used by an agent")
        return rows

    def gather(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        tables = {name: column[rows] for name, column in self.columns.items()}
        if rows.size and not self.row_ok[rows].all():
            for name, values in tables.items():
                allow_infinity = name in ("division_mass_pg", "max_age_s")
                if np.any(np.isnan(values)) or (not allow_infinity and not np.all(np.isfinite(values))):
                    raise ValueError(f"phenotype {name} contains invalid values")
            _validate_phenotype_tables(tables)
        return tables


_PREPARED_TABLE_CACHE: dict[int, tuple[weakref.ReferenceType, tuple, int, _PreparedPhenotypeTable]] = {}


def _prepared_table(phenotypes, k: int, signature: tuple) -> _PreparedPhenotypeTable:
    key = id(phenotypes)
    entry = _PREPARED_TABLE_CACHE.get(key)
    if entry is not None and entry[0]() is phenotypes and entry[1] == signature and entry[2] == k:
        return entry[3]
    prepared = _PreparedPhenotypeTable(phenotypes, k)

    def discard(dead_ref, key=key):
        stale = _PREPARED_TABLE_CACHE.get(key)
        if stale is not None and stale[0] is dead_ref:
            _PREPARED_TABLE_CACHE.pop(key, None)

    _PREPARED_TABLE_CACHE[key] = (weakref.ref(phenotypes, discard), signature, k, prepared)
    return prepared


def phenotype_arrays(
    phenotypes, genome_ids: np.ndarray, k: int, variant_ids: np.ndarray | None = None
) -> PhenotypeArrays:
    """Gather genome-keyed traits with vectorized lookups, never per-agent loops.

    The frozen table representation takes a fully vectorized path. A Mapping is
    also accepted for callers already holding one Phenotype per genome; only
    that (genome catalog) conversion iterates records, not agent rows.
    """
    genome_ids = np.asarray(genome_ids, dtype=np.uint64)
    if variant_ids is None:
        variant_ids = np.zeros(genome_ids.shape, dtype=np.uint64)
    else:
        variant_ids = np.asarray(variant_ids, dtype=np.uint64)
        if variant_ids.shape != genome_ids.shape:
            raise ValueError("variant_id must have the same shape as genome_id")

    table_ids = getattr(phenotypes, "genome_id", None)
    if table_ids is not None:
        cache_key = id(phenotypes)
        signature = _phenotype_table_signature(phenotypes)
        cached = _PHENOTYPE_GATHER_CACHE.get(cache_key)
        if (
            cached is not None
            and cached[0]() is phenotypes
            and cached[1] == signature
            and cached[2].shape == genome_ids.shape
            and np.array_equal(cached[2], genome_ids)
            and np.array_equal(cached[3], variant_ids)
        ):
            return cached[4]
        prepared = _prepared_table(phenotypes, k, signature)
        rows = prepared.lookup(genome_ids, variant_ids)
        gathered = PhenotypeArrays(**prepared.gather(rows))
        ids_for_cache = np.ascontiguousarray(genome_ids.copy())
        variant_ids_for_cache = np.ascontiguousarray(variant_ids.copy())

        def discard(dead_ref, key=cache_key):
            entry = _PHENOTYPE_GATHER_CACHE.get(key)
            if entry is not None and entry[0] is dead_ref:
                _PHENOTYPE_GATHER_CACHE.pop(key, None)

        _PHENOTYPE_GATHER_CACHE[cache_key] = (
            weakref.ref(phenotypes, discard), signature, ids_for_cache, variant_ids_for_cache, gathered
        )
        return gathered

    if np.any(variant_ids != 0):
        raise ValueError("non-root variant IDs require a pair-keyed PhenotypeTable")
    if not isinstance(phenotypes, Mapping) or not phenotypes:
        raise ValueError("at least one genome phenotype is required")
    genome_keys = np.asarray(sorted(int(key) for key in phenotypes), dtype=np.uint64)
    records: list[Phenotype] = []
    for key in genome_keys:
        value = phenotypes[int(key)]
        records.append(value if isinstance(value, Phenotype) else Phenotype(**value))
    rows = np.searchsorted(genome_keys, genome_ids)
    if rows.size and (np.any(rows >= genome_keys.size) or np.any(genome_keys[rows] != genome_ids)):
        raise ValueError("phenotype mapping is missing a genome_id used by an agent")

    nutrient_fields = ("max_uptake_mmol_s", "storage_fraction", "biomass_fraction", "energy_yield_pj_per_mmol")
    tables = {}
    for name in nutrient_fields:
        table = np.stack([_nutrient_vector(getattr(record, name), k, name) for record in records])
        tables[name] = table[rows]

    scalar_fields = (
        "motility_um_s", "chemotaxis_sensitivity", "random_motility_fraction",
        "division_rate_s", "division_mass_pg", "death_rate_s", "max_age_s",
        "antagonism_radius_um", "antagonism_rate_s",
    )
    for name in scalar_fields:
        values = np.asarray([getattr(record, name) for record in records], dtype=np.float64)
        allow_infinity = name in ("division_mass_pg", "max_age_s")
        if np.any(np.isnan(values)) or (not allow_infinity and not np.all(np.isfinite(values))):
            raise ValueError(f"phenotype {name} contains invalid values")
        tables[name] = values[rows]
    return _validate_phenotype_tables(tables)


def make_rngs(root_seed: int) -> dict[int, np.random.Generator]:
    """Create the frozen, independently seeded PCG64 streams."""
    if int(root_seed) < 0:
        raise ValueError("root_seed must be nonnegative")
    return {
        purpose_tag: np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(int(root_seed), spawn_key=(purpose_tag,)))
        )
        for purpose_tag in RNG_PURPOSES
    }


def validate_rngs(rngs: Mapping[int, np.random.Generator]) -> None:
    missing = set(RNG_PURPOSES) - set(rngs)
    if missing:
        raise ValueError(f"rngs is missing fixed purpose tags: {sorted(missing)}")
    for tag in RNG_PURPOSES:
        rng = rngs[tag]
        if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
            raise TypeError(f"rng purpose {tag} must be a NumPy Generator backed by PCG64")


@dataclass(frozen=True)
class NutrientLedger:
    initial_mmol: np.ndarray
    field_mmol: np.ndarray
    agents_mmol: np.ndarray
    detritus_mmol: np.ndarray
    source_cum_mmol: np.ndarray
    sink_cum_mmol: np.ndarray
    respiration_loss_mmol: np.ndarray

    @property
    def residual_mmol(self) -> np.ndarray:
        return (
            self.initial_mmol + self.source_cum_mmol
            - self.field_mmol - self.agents_mmol - self.detritus_mmol
            - self.sink_cum_mmol - self.respiration_loss_mmol
        )

    def assert_conserved(self, rtol: float = 1e-12, atol: float = 1e-15) -> None:
        if not np.allclose(self.residual_mmol, 0.0, rtol=rtol, atol=atol):
            raise RuntimeError(f"nutrient ledger failed conservation: residual={self.residual_mmol!r} mmol")
