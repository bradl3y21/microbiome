"""Frozen D1 public state schema.

These nested dataclasses are the interface shared with render/UI/QA. Simulation
internals adapt them to the same contiguous NumPy arrays and write the results
back in place after every tick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
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
    # D4 extension. None is accepted as the legacy, all-root (variant 0) state.
    variant_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.variant_id is None:
            self.variant_id = np.zeros(np.asarray(self.agent_id).size, dtype=np.uint64)
        else:
            self.variant_id = np.ascontiguousarray(self.variant_id, dtype=np.uint64)
            if self.variant_id.shape != np.asarray(self.agent_id).shape:
                raise ValueError("variant_id must have one row per agent")


@dataclass
class SimulationState:
    agents: AgentArrays
    tick_index: np.uint64
    next_agent_id: np.uint64


@dataclass
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


@dataclass
class Environment:
    width_um: float
    height_um: float
    depth_um: float
    dx_um: float
    dy_um: float
    nx: int
    ny: int
    field: NutrientField


@dataclass
class PhenotypeTable:
    genome_id: np.ndarray
    max_uptake_mmol_s: np.ndarray
    storage_fraction: np.ndarray
    biomass_fraction: np.ndarray
    energy_yield_pj_per_mmol: np.ndarray
    motility_um_s: np.ndarray
    division_mass_pg: np.ndarray
    max_age_s: np.ndarray
    division_rate_s: np.ndarray | None = None
    death_rate_s: np.ndarray | None = None
    antagonism_radius_um: np.ndarray | None = None
    antagonism_rate_s: np.ndarray | None = None
    chemotaxis_sensitivity: np.ndarray | None = None
    random_motility_fraction: np.ndarray | None = None
    # D4 extension; legacy D2 tables contain only unmutated root rows.
    variant_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.variant_id is None:
            self.variant_id = np.zeros(np.asarray(self.genome_id).size, dtype=np.uint64)
        else:
            self.variant_id = np.ascontiguousarray(self.variant_id, dtype=np.uint64)
            if self.variant_id.shape != np.asarray(self.genome_id).shape:
                raise ValueError("phenotype variant_id must have one row per genome/variant")


@dataclass
class RNGStreams:
    motility: np.random.Generator
    division: np.random.Generator
    mutation: np.random.Generator
    death: np.random.Generator
    antagonism: np.random.Generator
