"""Read-only render projection built from the frozen D3 simulation snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .atlas import SpriteAtlas

if TYPE_CHECKING:
    from microbiome.core.ui_api import RenderSnapshot


OVERLAY_CHANNEL = {"oxygen": "oxygen", "acidity": "acetate", "toxin": "bacilysin", "signal": "ai2"}
NUTRIENT_CHANNELS = ("glucose", "lactose", "acetate")


def _readonly_array(value: Any, dtype: Any, ndim: int, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if source.dtype != np.dtype(dtype) or not source.flags.c_contiguous or source.flags.writeable:
        source = np.array(source, dtype=dtype, order="C", copy=True)
    source.setflags(write=False)
    return source


@dataclass(frozen=True, slots=True)
class WorldView:
    """The only per-frame data projection the renderer consumes.

    Arrays are C-contiguous and read-only. If a caller supplies writable or
    non-contiguous input, construction detaches it with a copy. The normal D3
    path reuses Core's already detached read-only RenderSnapshot arrays.
    """

    tick_index: int
    width_um: float
    height_um: float
    position_um: np.ndarray
    radius_um: np.ndarray
    colour_index: np.ndarray
    motile: np.ndarray
    field_density: np.ndarray
    agent_id: np.ndarray | None = None
    cell_state: np.ndarray | None = None   # 0 vegetative, 1 sporulating, 2 spore, 3 matrix-producing (realism snapshots only)
    eps: np.ndarray | None = None
    length_um: np.ndarray | None = None    # true rod length (realism snapshots only)
    heading_rad: np.ndarray | None = None  # rod axis direction in world axes
    cell_width_um: float = 0.8

    def __post_init__(self) -> None:
        tick_index = int(self.tick_index)
        width_um, height_um = float(self.width_um), float(self.height_um)
        if tick_index < 0:
            raise ValueError("tick_index must be nonnegative")
        if not np.isfinite(width_um) or not np.isfinite(height_um) or width_um <= 0 or height_um <= 0:
            raise ValueError("world dimensions must be finite and positive")

        positions = _readonly_array(self.position_um, np.float64, 2, "position_um")
        radii = _readonly_array(self.radius_um, np.float64, 1, "radius_um")
        colours = _readonly_array(self.colour_index, np.uint8, 1, "colour_index")
        motile = _readonly_array(self.motile, np.bool_, 1, "motile")
        density = _readonly_array(self.field_density, np.float64, 2, "field_density")
        n_agents = positions.shape[0]
        if positions.shape[1:] != (2,) or radii.shape != (n_agents,) or colours.shape != (n_agents,) or motile.shape != (n_agents,):
            raise ValueError("agent arrays in WorldView must be row-aligned with position_um[N,2]")
        if density.shape[0] < 1 or density.shape[1] < 1:
            raise ValueError("field_density must have a nonempty [ny,nx] grid")

        object.__setattr__(self, "tick_index", tick_index)
        object.__setattr__(self, "width_um", width_um)
        object.__setattr__(self, "height_um", height_um)
        object.__setattr__(self, "position_um", positions)
        object.__setattr__(self, "radius_um", radii)
        object.__setattr__(self, "colour_index", colours)
        object.__setattr__(self, "motile", motile)
        object.__setattr__(self, "field_density", density)
        for name, dtype in ("cell_state", np.uint8), ("eps", np.float64), ("length_um", np.float64), ("heading_rad", np.float64):
            value = getattr(self, name)
            if value is not None:
                array = _readonly_array(value, dtype, 1, name)
                if array.shape != (n_agents,):
                    raise ValueError(f"{name} in WorldView must be row-aligned with position_um[N,2]")
                object.__setattr__(self, name, array)
        if self.agent_id is not None:
            ids = _readonly_array(self.agent_id, np.uint64, 1, "agent_id")
            if ids.shape != (n_agents,):
                raise ValueError("agent_id in WorldView must be row-aligned with position_um[N,2]")
            object.__setattr__(self, "agent_id", ids)

    @classmethod
    def from_snapshot(cls, snapshot: RenderSnapshot, overlay: str = "nutrients") -> WorldView:
        """Project the detached D3 snapshot once for a render frame; ``overlay`` picks which chemistry the field layer shows."""
        positions = np.asarray(snapshot.position_um, dtype=np.float64)
        radii = np.asarray(snapshot.radius_um, dtype=np.float64)
        genome_ids = np.asarray(snapshot.genome_id, dtype=np.uint64)
        table_genome_ids = np.asarray(snapshot.phenotype_genome_id, dtype=np.uint64)
        motility = np.asarray(snapshot.phenotype_motility_um_s, dtype=np.float64)
        max_uptake = np.asarray(snapshot.phenotype_max_uptake_mmol_s, dtype=np.float64)
        concentration = np.asarray(snapshot.concentration_mmol_l, dtype=np.float64)

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("RenderSnapshot.position_um must have shape [N,2]")
        if radii.shape != (len(positions),) or genome_ids.shape != (len(positions),):
            raise ValueError("RenderSnapshot agent arrays must be aligned")
        if table_genome_ids.ndim != 1 or motility.shape != table_genome_ids.shape:
            raise ValueError("RenderSnapshot phenotype genome IDs and motility must be aligned")
        if max_uptake.ndim != 2 or max_uptake.shape[0] != len(table_genome_ids):
            raise ValueError("RenderSnapshot.phenotype_max_uptake_mmol_s must have shape [G,K]")
        if concentration.ndim != 3:
            raise ValueError("RenderSnapshot.concentration_mmol_l must have shape [K,ny,nx]")
        if np.any(table_genome_ids[1:] <= table_genome_ids[:-1]):
            raise ValueError("RenderSnapshot phenotype_genome_id must be strictly increasing")

        rows = np.searchsorted(table_genome_ids, genome_ids)
        if np.any(rows >= len(table_genome_ids)) or not np.array_equal(table_genome_ids[rows], genome_ids):
            raise ValueError("RenderSnapshot phenotype table must cover all agent genomes")

        styles = getattr(snapshot, "genome_style", None)
        agent_colours = (rows % SpriteAtlas.PALETTE_COUNT if styles is None else np.asarray(styles)[rows]).astype(np.uint8)
        agent_motile = motility[rows] > 0.0
        names = tuple(getattr(snapshot, "channel_names", ()))
        if names and len(names) == concentration.shape[0]:
            index = {name: i for i, name in enumerate(names)}
            wanted = (OVERLAY_CHANNEL.get(overlay),) if overlay in OVERLAY_CHANNEL else NUTRIENT_CHANNELS
            chosen = [index[name] for name in wanted if name in index] or list(range(len(names)))
            field_density = np.sum(concentration[chosen], axis=0, dtype=np.float64)
        else:
            field_density = np.sum(concentration, axis=0, dtype=np.float64)
        state = getattr(snapshot, "cell_state", None)
        matrix = getattr(snapshot, "eps", None)
        if state is not None:
            state = np.asarray(state, dtype=np.uint8)
            agent_colours = np.where(state == 2, SpriteAtlas.SPORE_STYLE, agent_colours).astype(np.uint8)
            agent_motile = agent_motile & (state != 2)
            agent_colours.setflags(write=False)
            agent_motile.setflags(write=False)
        agent_colours.setflags(write=False)
        agent_motile.setflags(write=False)
        field_density.setflags(write=False)

        return cls(
            tick_index=int(snapshot.tick_index),
            width_um=float(snapshot.width_um),
            height_um=float(snapshot.height_um),
            position_um=positions,
            radius_um=radii,
            colour_index=agent_colours,
            motile=agent_motile,
            field_density=field_density,
            agent_id=np.asarray(snapshot.agent_id, dtype=np.uint64),
            cell_state=state,
            eps=None if matrix is None else np.asarray(matrix, dtype=np.float64),
            length_um=None if getattr(snapshot, "length_um", None) is None else np.asarray(snapshot.length_um, dtype=np.float64),
            heading_rad=None if getattr(snapshot, "heading_rad", None) is None else np.asarray(snapshot.heading_rad, dtype=np.float64),
            cell_width_um=float(getattr(snapshot, "cell_width_um", 0.8)),
        )
