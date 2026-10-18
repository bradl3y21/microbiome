"""Deterministic periodic spatial hashing for short-range interactions."""

from __future__ import annotations

import numpy as np


class SpatialHash:
    """Build a sparse uniform grid and return candidate pairs within a radius.

    Work is proportional to occupancy of the queried neighboring bins rather
    than N squared. The nine-bin stencil is a fixed loop (not an agent loop).
    Returned row-index pairs are unique and lexicographically ordered.
    """

    def __init__(self, width_um: float, height_um: float, cell_size_um: float):
        self.width_um = float(width_um)
        self.height_um = float(height_um)
        self.cell_size_um = float(cell_size_um)
        if min(self.width_um, self.height_um, self.cell_size_um) <= 0:
            raise ValueError("spatial hash dimensions and cell size must be positive")
        self.nx = max(1, int(np.floor(self.width_um / self.cell_size_um)))
        self.ny = max(1, int(np.floor(self.height_um / self.cell_size_um)))
        self.bin_width_um = self.width_um / self.nx
        self.bin_height_um = self.height_um / self.ny

    def pairs(self, positions_um: np.ndarray, agent_id: np.ndarray, radius_um: float,
              *, ordered: bool = True, return_distance2: bool = False,
              pair_groups: np.ndarray | None = None):
        """Return unique pairs at periodic distance <= r.

        The public default remains lexicographically ordered by agent ID.
        Internal consumers may request deterministic generation order and the
        already-computed squared distances to avoid redundant work.
        """
        positions = np.asarray(positions_um, dtype=np.float64)
        ids = np.asarray(agent_id, dtype=np.uint64)
        n = ids.size
        groups = None if pair_groups is None else np.asarray(pair_groups)
        if groups is not None and groups.shape != (n,):
            raise ValueError("pair_groups must have shape (N,)")
        radius = float(radius_um)
        if positions.shape != (n, 2):
            raise ValueError("positions_um must have shape (N, 2)")
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius_um must be finite and nonnegative")
        if n < 2 or radius == 0:
            empty = np.empty((0, 2), dtype=np.int64)
            return (empty, np.empty(0, dtype=np.float64)) if return_distance2 else empty
        if np.any(positions < 0) or np.any(positions[:, 0] >= self.width_um) or np.any(positions[:, 1] >= self.height_um):
            raise ValueError("positions must be wrapped into the periodic domain")

        cx = np.floor(positions[:, 0] / self.bin_width_um).astype(np.int64) % self.nx
        cy = np.floor(positions[:, 1] / self.bin_height_um).astype(np.int64) % self.ny
        cell_key = cy * self.nx + cx
        # The engine supplies canonical (increasing-ID) rows. Stable sorting by
        # cell then preserves that order without paying for a second sort key;
        # shuffled callers retain the explicit ID tiebreak path.
        ids_in_order = ids.size < 2 or bool(np.all(ids[1:] >= ids[:-1]))
        n_cells = self.nx * self.ny
        reach_x = max(1, int(np.ceil(radius / self.bin_width_um)))
        reach_y = max(1, int(np.ceil(radius / self.bin_height_um)))
        order = (np.argsort(cell_key, kind="stable") if ids_in_order
                 else np.lexsort((ids, cell_key)))
        occupancy = np.bincount(cell_key, minlength=n_cells).astype(np.int64, copy=False)
        starts = np.cumsum(occupancy, dtype=np.int64) - occupancy

        # With >=3 bins on each axis, enumerate each undirected pair of adjacent
        # cells exactly once (same cell plus a half-stencil). This avoids 18
        # per-agent searchsorted calls and a global duplicate-removal sort. Tiny
        # periodic grids use the alias-safe query path below.
        if (self.nx > 2 * reach_x and self.ny > 2 * reach_y
                and n_cells <= max(1_000_000, 64 * n)):
            pairs = self._half_stencil_pairs(
                positions, ids, order, cell_key, occupancy, starts, radius,
                ids_in_order, ordered, return_distance2, groups, reach_x, reach_y
            )
        else:
            pairs = self._query_pairs(
                positions, ids, order, cell_key, radius, ordered, return_distance2,
                groups, ids_in_order,
                reach_x, reach_y
            )
        return pairs

    def _half_stencil_pairs(self, positions, ids, order, cell_key, occupancy, starts, radius,
                            ids_in_order, ordered, return_distance2, groups, reach_x, reach_y):
        n = cell_key.size
        row = np.arange(n, dtype=np.int64)
        cx = cell_key % self.nx
        cy = cell_key // self.nx
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        # Query occupied bins directly for each agent. A bin's physical width
        # and height are both >= radius_um, so this five-offset half-stencil is
        # complete while avoiding work proportional to the often mostly-empty
        # dense grid extent.
        offsets = [(0, 0)]
        offsets.extend(
            (ox, oy)
            for oy in range(-reach_y, reach_y + 1)
            for ox in range(-reach_x, reach_x + 1)
            if oy > 0 or (oy == 0 and ox > 0)
        )
        for ox, oy in offsets:
            neighbor_key = ((cy + oy) % self.ny) * self.nx + ((cx + ox) % self.nx)
            counts = occupancy[neighbor_key]
            total = int(counts.sum())
            if total == 0:
                continue
            owner = np.repeat(row, counts)
            prefix = np.cumsum(counts, dtype=np.int64) - counts
            # For each owner, flatten its contiguous neighbor range directly.
            # Folding the start and prefix together before repeat avoids an
            # intermediate per-candidate ``within`` array and owner->cell gather.
            candidate_offset = starts[neighbor_key] - prefix
            candidate = order[np.repeat(candidate_offset, counts) + np.arange(total, dtype=np.int64)]
            keep = owner < candidate if (ox == 0 and oy == 0) else np.ones(total, dtype=np.bool_)
            if groups is not None:
                keep &= groups[owner] != groups[candidate]
            if np.any(keep):
                a = owner[keep]
                b = candidate[keep]
                left_parts.append(np.minimum(a, b))
                right_parts.append(np.maximum(a, b))

        if not left_parts:
            empty = np.empty((0, 2), dtype=np.int64)
            return (empty, np.empty(0, dtype=np.float64)) if return_distance2 else empty
        pairs = np.column_stack((np.concatenate(left_parts), np.concatenate(right_parts)))
        return self._within_radius(positions, ids, pairs, radius, unique=False,
                                   ids_in_order=ids_in_order, ordered=ordered,
                                   return_distance2=return_distance2)

    def _query_pairs(self, positions, ids, order, cell_key, radius, ordered,
                     return_distance2, groups, ids_in_order, reach_x, reach_y):
        """Alias-safe sparse query fallback for periodic grids with 1–2 bins."""
        sorted_key = cell_key[order]
        cy = cell_key // self.nx
        cx = cell_key % self.nx
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        row = np.arange(cell_key.size, dtype=np.int64)
        for oy in range(-reach_y, reach_y + 1):
            ny = (cy + oy) % self.ny
            for ox in range(-reach_x, reach_x + 1):
                nx = (cx + ox) % self.nx
                neighbor_key = ny * self.nx + nx
                lo = np.searchsorted(sorted_key, neighbor_key, side="left")
                hi = np.searchsorted(sorted_key, neighbor_key, side="right")
                counts = hi - lo
                total = int(counts.sum())
                if total == 0:
                    continue
                owner = np.repeat(row, counts)
                prefix = np.cumsum(counts, dtype=np.int64) - counts
                within = np.arange(total, dtype=np.int64) - np.repeat(prefix, counts)
                candidate = order[lo[owner] + within]
                keep = owner < candidate
                if np.any(keep):
                    left_parts.append(owner[keep])
                    right_parts.append(candidate[keep])
        if not left_parts:
            empty = np.empty((0, 2), dtype=np.int64)
            return (empty, np.empty(0, dtype=np.float64)) if return_distance2 else empty
        pairs = np.column_stack((np.concatenate(left_parts), np.concatenate(right_parts)))
        pairs = np.unique(pairs, axis=0)
        if groups is not None:
            pairs = pairs[groups[pairs[:, 0]] != groups[pairs[:, 1]]]
        return self._within_radius(positions, ids, pairs, radius, unique=True,
                                   ids_in_order=ids_in_order, ordered=ordered,
                                   return_distance2=return_distance2)

    def _within_radius(self, positions, ids, pairs, radius, unique, ids_in_order=False,
                       ordered=True, return_distance2=False):
        if pairs.size == 0:
            empty = np.empty((0, 2), dtype=np.int64)
            return (empty, np.empty(0, dtype=np.float64)) if return_distance2 else empty
        a, b = pairs[:, 0], pairs[:, 1]
        dx = positions[b, 0] - positions[a, 0]
        dy = positions[b, 1] - positions[a, 1]
        dx -= np.rint(dx / self.width_um) * self.width_um
        dy -= np.rint(dy / self.height_um) * self.height_um
        distance2 = dx * dx + dy * dy
        within = distance2 <= radius * radius
        pairs = pairs[within]
        distance2 = distance2[within]
        if pairs.size == 0:
            empty = np.empty((0, 2), dtype=np.int64)
            return (empty, np.empty(0, dtype=np.float64)) if return_distance2 else empty
        if ids is not None:
            if not ordered:
                pass
            elif ids_in_order:
                # Rows already follow ID order, so one packed integer key sorts
                # the pair tuples exactly as the two-key lexsort would.
                key = pairs[:, 0] * ids.size + pairs[:, 1]
                order = np.argsort(key, kind="stable")
                pairs = pairs[order]
                distance2 = distance2[order]
            else:
                # Return pairs in stable ID order even for a shuffled caller array.
                swap = ids[pairs[:, 0]] > ids[pairs[:, 1]]
                pairs[swap] = pairs[swap, ::-1]
                order = np.lexsort((ids[pairs[:, 1]], ids[pairs[:, 0]]))
                pairs = pairs[order]
                distance2 = distance2[order]
        elif unique:
            pairs, unique_rows = np.unique(pairs, axis=0, return_index=True)
            distance2 = distance2[unique_rows]
        return (pairs, distance2) if return_distance2 else pairs
