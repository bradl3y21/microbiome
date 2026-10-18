"""Controlled worlds for virtual experiments, built from the real genomes and the same tables the application uses.

A ``Rig`` is a small periodic slab holding cells of one organism, with named helpers to clamp a chemical to a value (booked
on the mass ledger as a source or sink, so the engine's conservation assertion stays on) and to advance the realistic tick.
Nothing here re-implements the physiology: the cells run in ``sim.realism`` exactly as in the application.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..genome.calibration import Calibration
from ..genome.models import GenomeRecord, NutrientChannel
from ..genome.organisms import MX, PA, SA
from ..genome.parser import _PROJECT_ROOT, load_genome
from ..genome.phenotype import build_phenotype_table
from ..genome.physiology import (
    BS,
    CHANNEL_NAMES,
    DIFFUSIVITY_UM2_S,
    EC,
    MOLAR_MASS_G_MOL,
    REALISM_CHANNELS,
    PhysiologyTables,
    derive_physiology,
    extend_phenotype_table,
    load_physiology_tables,
)
from ..sim.model import DT_S, AgentState, NutrientEnvironment, make_rngs
from ..sim.realism import RealismConfig, RealisticSimulation

GLC, LAC, O2, ACE, TOX, SIG, TOX2 = range(7)
K = len(CHANNEL_NAMES)
ACCESSIONS = (EC, BS, PA, SA, MX)
MOLAR = np.asarray(MOLAR_MASS_G_MOL, dtype=np.float64)
GLUCOSE_G_PER_MMOL = MOLAR[GLC] / 1000.0


@dataclass
class _Context:
    genomes: dict[str, GenomeRecord]
    tables: PhysiologyTables
    base_table: object


_CONTEXTS: dict[Path, _Context] = {}


def context(project_root: str | Path | None = None) -> _Context:
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    found = _CONTEXTS.get(root)
    if found is None:
        genomes = {accession: load_genome(accession, root) for accession in ACCESSIONS}
        sugars = (NutrientChannel(0, "glucose"), NutrientChannel(1, "lactose"))
        base = build_phenotype_table(list(genomes.values()), sugars, root)
        found = _CONTEXTS[root] = _Context(genomes, load_physiology_tables(root), base)
    return found


class Rig:
    """One organism in a small periodic film. ``mass_fraction`` is the cells' dry mass as a fraction of the division mass."""

    def __init__(self, accession: str, positions, *, nx: int = 8, ny: int = 8, dx: float = 10.0, depth: float = 10.0,
                 mass_fraction: float = 0.7, seed: int = 1, config: RealismConfig | None = None,
                 calibration: Calibration | None = None, project_root: str | Path | None = None) -> None:
        ctx = context(project_root)
        self._root = project_root
        self.accession = accession
        self.genome = ctx.genomes[accession]
        tables = ctx.tables if calibration is None else dataclasses.replace(ctx.tables, calibration=calibration)
        self.tables = tables
        self.table = extend_phenotype_table(ctx.base_table, list(ctx.genomes.values()), tables)
        physiology = {int(g.genome_id): derive_physiology(g, tables) for g in ctx.genomes.values()}
        self.physiology = physiology[int(self.genome.genome_id)]
        self.config = config or RealismConfig()
        self.sim = RealisticSimulation(physiology, self.config, REALISM_CHANNELS)
        self.rngs = make_rngs(seed)
        row = int(np.flatnonzero(self.table.genome_id == np.uint64(self.genome.genome_id))[0])
        self.division_mass_pg = float(self.table.division_mass_pg[row])
        self.env = NutrientEnvironment(
            nutrient_id=np.arange(K, dtype=np.uint32), concentration_mmol_l=np.zeros((K, ny, nx)),
            diffusion_um2_s=np.asarray(DIFFUSIVITY_UM2_S, dtype=np.float64), molar_mass_g_mol=MOLAR.copy(),
            dx_um=dx, dy_um=dx, depth_um=depth, width_um=nx * dx, height_um=ny * dx)
        self.env.concentration_mmol_l[O2] = self.config.oxygen_saturation_mmol_l
        positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        n = len(positions)
        biomass = np.zeros((n, K))
        biomass[:, GLC] = mass_fraction * self.division_mass_pg / (MOLAR[GLC] * 1e9)
        mass_pg = biomass @ (MOLAR * 1e9)
        self.state = AgentState(
            agent_id=np.arange(1, n + 1, dtype=np.uint64), positions_um=positions.copy(), velocities_um_s=np.zeros((n, 2)),
            mass_pg=mass_pg, radius_um=np.cbrt(3 * mass_pg / (4 * np.pi)), energy_pj=np.zeros(n),
            species_id=np.zeros(n, dtype=np.uint32), genome_id=np.full(n, self.genome.genome_id, dtype=np.uint64),
            age_s=np.zeros(n), nutrient_store_mmol=np.zeros((n, K)), biomass_mmol=biomass)

    def add_cells(self, accession: str, positions, *, mass_pg: float = 0.3) -> np.ndarray:
        """Add cells of another organism (e.g. prey) to the slab; returns their agent ids."""
        from ..sim.model import AGENT_ARRAY_FIELDS

        genome = context(self._root).genomes[accession]
        positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        n = len(positions)
        biomass = np.zeros((n, K))
        biomass[:, GLC] = mass_pg / (MOLAR[GLC] * 1e9)
        mass = biomass @ (MOLAR * 1e9)
        first = int(self.state.next_agent_id)
        ids = np.arange(first, first + n, dtype=np.uint64)
        new = {"agent_id": ids, "positions_um": positions, "velocities_um_s": np.zeros((n, 2)), "mass_pg": mass,
               "radius_um": np.cbrt(3 * mass / (4 * np.pi)), "energy_pj": np.zeros(n), "species_id": np.zeros(n, dtype=np.uint32),
               "genome_id": np.full(n, genome.genome_id, dtype=np.uint64), "age_s": np.zeros(n),
               "nutrient_store_mmol": np.zeros((n, K)), "biomass_mmol": biomass}
        for name in AGENT_ARRAY_FIELDS:
            old = getattr(self.state, name)
            addition = new.get(name)
            if addition is None:
                addition = np.zeros((n,) + old.shape[1:], dtype=old.dtype)
            setattr(self.state, name, np.ascontiguousarray(np.concatenate((old, addition.astype(old.dtype, copy=False)))))
        self.state.next_agent_id = first + n
        return ids

    # ------------------------------------------------------------------------------------------ driving
    def step(self, ticks: int = 1) -> None:
        for _ in range(ticks):
            self.sim.step(self.state, self.env, self.table, self.rngs)

    @property
    def volume_l(self) -> float:
        return self.env.cell_volume_l

    def total_mmol(self, channel: int) -> float:
        return float(self.env.concentration_mmol_l[channel].sum()) * self.volume_l

    def clamp(self, channel: int, values) -> float:
        """Set a channel to ``values`` (scalar or [ny, nx]); returns mmol added (negative: removed). Booked on the ledger."""
        old = self.total_mmol(channel)
        self.env.concentration_mmol_l[channel] = values
        delta = self.total_mmol(channel) - old
        if self.state.ledger_initialized:
            if delta >= 0:
                self.state.source_cum_mmol[channel] += delta
            else:
                self.state.sink_cum_mmol[channel] += -delta
        return delta

    @property
    def mass_g(self) -> float:
        return float(self.state.mass_pg.sum()) * 1e-12


def default_config(**changes) -> RealismConfig:
    """Fully adapted cells (no lag), no contacts: the clean setting for rate measurements."""
    base = dict(lag_initial=1.0, collisions=False, gas_exchange_per_s=0.0)
    base.update(changes)
    return dataclasses.replace(RealismConfig(), **base)


__all__ = ["ACCESSIONS", "ACE", "BS", "EC", "GLC", "GLUCOSE_G_PER_MMOL", "K", "LAC", "MX", "O2", "PA", "Rig", "SA", "SIG", "TOX", "TOX2", "context", "default_config"]
