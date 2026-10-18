"""Batched pygame renderer: deep-water backdrop, glowing nutrient clouds, shaded flagellated microbes."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pygame

from .atlas import SpriteAtlas
from .camera import Camera2D
from .scene import Backdrop
from .view import WorldView

if TYPE_CHECKING:
    from microbiome.core.ui_api import RenderSnapshot


@dataclass(frozen=True, slots=True)
class RenderStats:
    """Small per-frame counters useful for instrumentation and tests."""

    visible_cells: int
    density_cells: int
    blit_calls: int


# (low colour, high colour, alpha at low, alpha at high, logarithmic normalisation). Darker/fainter = less of the chemical.
OVERLAY_STYLE = {
    "nutrients": ((40, 46, 52), (158, 120, 58), 5.0, 83.0, True),
    "oxygen": ((20, 36, 58), (70, 150, 210), 4.0, 70.0, False),
    "acidity": ((50, 40, 34), (214, 118, 58), 4.0, 90.0, True),
    "toxin": ((46, 34, 54), (196, 62, 150), 4.0, 100.0, True),
    "signal": ((38, 50, 34), (150, 214, 92), 4.0, 90.0, True),
}
OVERLAYS = tuple(OVERLAY_STYLE)


class PygameRenderer:
    """Render a detached D3 snapshot through the render-only WorldView.

    Cells are drawn larger than their true size (``VISUAL_SCALE``): a bacterium is ~0.5 um across and would
    be sub-pixel at any zoom that shows the whole field. This is presentation only; picking, the simulation
    and the snapshot all keep true micrometre radii.
    """

    BACKGROUND = (5, 10, 22)
    FIELD_EDGE = (66, 112, 148)
    DENSITY_REFRESH_TICKS = 2
    VISUAL_SCALE = 14.0
    SPRITE_PIXEL_BUDGET = 900_000
    MIN_HALF_WIDTH_PX = 2.0
    WAVE_TICKS = 7
    MAX_SPECKS = 450
    HEADING_BLEND = 0.35

    def __init__(
        self,
        camera: Camera2D | None = None,
        atlas: SpriteAtlas | None = None,
    ) -> None:
        self.camera = camera or Camera2D()
        self.atlas = atlas or SpriteAtlas()
        self.backdrop = Backdrop()
        self.overlay = "nutrients"      # which chemistry the field layer shows (see OVERLAY_STYLE)
        self._density_surface: pygame.Surface | None = None
        self._density_cache_key: tuple[Any, ...] | None = None
        self._scaled_density_surface: pygame.Surface | None = None
        self._scaled_density_key: tuple[Any, ...] | None = None
        self._scaled_density_destination: tuple[int, int] | None = None
        self._level: np.ndarray | None = None
        self._speck_cache_key: tuple[Any, ...] | None = None
        self._speck_table: tuple[np.ndarray, ...] | None = None
        self._prev_ids: np.ndarray | None = None
        self._prev_positions: np.ndarray | None = None
        self._prev_angle: np.ndarray | None = None

    def handle_event(
        self,
        event: pygame.event.Event,
        pointer_position: tuple[int, int] | None = None,
    ) -> bool:
        return self.camera.handle_event(event, pointer_position)

    def draw(
        self,
        target: pygame.Surface,
        snapshot: RenderSnapshot,
    ) -> RenderStats:
        """Build one WorldView and draw it; the cells go out in a single batched blits call."""
        view = WorldView.from_snapshot(snapshot, self.overlay)
        target_size = target.get_size()
        self.camera.resize(target_size)
        self.camera.set_field_size((view.width_um, view.height_um))

        if len(view.colour_index):
            for palette in np.flatnonzero(np.bincount(view.colour_index, minlength=self.atlas.STYLE_COUNT)):
                self.atlas.ensure_palette(int(palette))

        field_centre = np.array((view.width_um * 0.5, view.height_um * 0.5))
        shift = -(self.camera.center_um - field_centre) * self.camera.zoom
        self.backdrop.draw(target, (float(shift[0]), float(shift[1])), view.tick_index)
        self._draw_field_lit(target, view.width_um, view.height_um)
        density_cells = self._draw_density(
            target,
            view.field_density,
            view.width_um,
            view.height_um,
            view.tick_index,
        )
        self._draw_specks(target, view)

        positions = view.position_um
        radii = view.radius_um
        count = len(positions)
        angle = self._track_headings(view)
        if count:
            visible = self._visible_indices(positions, radii)
            if len(visible):
                screen = np.asarray(self.camera.world_to_screen(positions[visible]), dtype=np.float64)
                screen_radii = radii[visible] * self.camera.zoom
                if view.agent_id is not None:
                    ids = view.agent_id[visible]
                else:
                    ids = np.zeros(len(visible), dtype=np.uint64)
                phase = ((view.tick_index // self.WAVE_TICKS) + ids.astype(np.int64)) & 1
                if view.cell_state is not None:
                    self._draw_matrix(target, screen, screen_radii, view.cell_state[visible], len(visible))
                # The command list holds ~10^4 short-lived containers; a cyclic-GC pass in the middle of
                # building it shows up as a frame-time spike. They are freed by refcount, so pause GC here.
                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    commands = self._cell_blits(
                        screen,
                        screen_radii,
                        view.colour_index[visible],
                        view.motile[visible],
                        angle[visible],
                        phase,
                    )
                    # Keep all cells in one native batch; no per-cell Surface is made.
                    target.blits(commands, doreturn=False)
                    visible_count = len(commands)
                    del commands
                finally:
                    if gc_was_enabled:
                        gc.enable()
                blit_calls = 1
            else:
                visible_count = 0
                blit_calls = 0
        else:
            visible_count = 0
            blit_calls = 0

        self._draw_field_edge(target, view.width_um, view.height_um)
        return RenderStats(visible_count, density_cells, blit_calls)

    def _visible_indices(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        left, top, right, bottom = self.camera.visible_world_rect()
        margin = radii + 40.0 / self.camera.zoom
        return np.flatnonzero(
            (positions[:, 0] + margin >= left)
            & (positions[:, 0] - margin <= right)
            & (positions[:, 1] + margin >= top)
            & (positions[:, 1] - margin <= bottom)
        )

    def _track_headings(self, view: WorldView) -> np.ndarray:
        """Heading (radians, screen axes) per agent from frame-to-frame motion; a render-side cache only."""
        n = len(view.position_um)
        if n == 0 or view.agent_id is None:
            return np.zeros(n, dtype=np.float64)
        ids = view.agent_id
        seed_angle = ((ids * np.uint64(2654435761)) % np.uint64(3600)).astype(np.float64) * (2.0 * np.pi / 3600.0)
        angle = seed_angle
        ordered = n < 2 or bool(np.all(ids[1:] > ids[:-1]))
        if ordered and self._prev_ids is not None and len(self._prev_ids):
            index = np.minimum(np.searchsorted(self._prev_ids, ids), len(self._prev_ids) - 1)
            known = self._prev_ids[index] == ids
            angle = np.where(known, self._prev_angle[index], seed_angle)
            delta = view.position_um - self._prev_positions[index]
            for axis, extent in ((0, view.width_um), (1, view.height_um)):
                delta[:, axis] -= extent * np.rint(delta[:, axis] / extent)
            moved = known & (np.einsum("ij,ij->i", delta, delta) > 1e-12)
            if np.any(moved):
                heading = np.arctan2(delta[:, 1], delta[:, 0])
                blend = self.HEADING_BLEND
                mixed = np.arctan2(
                    blend * np.sin(heading) + (1.0 - blend) * np.sin(angle),
                    blend * np.cos(heading) + (1.0 - blend) * np.cos(angle),
                )
                angle = np.where(moved, mixed, angle)
        if ordered:
            self._prev_ids = np.array(ids, copy=True)
            self._prev_positions = np.array(view.position_um, copy=True)
            self._prev_angle = np.array(angle, copy=True)
        return angle

    def _half_width(self, radii_px: np.ndarray, count: int) -> np.ndarray:
        """Drawn half-width in pixels: the true radius times VISUAL_SCALE, capped so the frame's sprite area stays bounded.

        A flagellated rod's crop box covers about 40 * half_width^2 px. The count is rounded up to a 1.25x geometric
        step so the size bin does not flicker near a boundary.
        """
        steps = np.ceil(np.log(max(count, 1)) / np.log(1.25))
        budget_cap = (self.SPRITE_PIXEL_BUDGET / (40.0 * 1.25 ** steps)) ** 0.5
        return np.minimum(radii_px * self.VISUAL_SCALE, budget_cap)

    def _draw_matrix(self, target, screen, radii_px, state, count) -> None:
        """A pale halo under every matrix-producing cell, so a biofilm reads as one connected mass."""
        mask = state == 3
        if not mask.any():
            return
        sizes = self.atlas.size_index(np.maximum(self._half_width(radii_px[mask], count), self.MIN_HALF_WIDTH_PX))
        commands = []
        for (x, y), size in zip(screen[mask].tolist(), sizes.tolist()):
            sprite = self.atlas.matrix_sprite(size)
            commands.append((sprite, (int(x) - sprite.get_width() // 2, int(y) - sprite.get_height() // 2)))
        target.blits(commands, doreturn=False)

    def _cell_blits(
        self,
        screen: np.ndarray,
        radii_px: np.ndarray,
        palette: np.ndarray,
        motile: np.ndarray,
        angle: np.ndarray | None = None,
        phase: np.ndarray | None = None,
    ) -> list[tuple[Any, ...]]:
        atlas = self.atlas
        count = len(screen)
        angle = np.zeros(count) if angle is None else angle
        phase = np.zeros(count, dtype=np.int64) if phase is None else phase
        palette_index = np.asarray(palette, dtype=np.intp) % atlas.STYLE_COUNT
        half_width = self._half_width(radii_px, count) * np.where(palette_index == atlas.SPORE_STYLE, 0.6, 1.0)
        size_index = atlas.size_index(np.maximum(half_width, self.MIN_HALF_WIDTH_PX))
        motile_flag = np.asarray(motile, dtype=bool)
        heading = np.rint(angle * (atlas.HEADINGS / (2.0 * np.pi))).astype(np.intp) % atlas.HEADINGS
        variant = np.where(motile_flag, atlas.HEADINGS * (1 + (phase & 1)) + heading, heading).astype(np.intp)

        for pair in np.unique(palette_index * atlas.n_sizes + size_index):
            atlas.ensure(int(pair) // atlas.n_sizes, int(pair) % atlas.n_sizes)
        tiles = atlas.tile_size[palette_index, size_index].astype(np.int64)
        box = atlas.crop[palette_index, size_index, variant].astype(np.int64)
        destinations = np.rint(screen - tiles[:, None] * 0.5).astype(np.int64) + box[:, :2]
        sources = np.column_stack((
            (variant % atlas.COLUMNS) * tiles + box[:, 0],
            (variant // atlas.COLUMNS) * tiles + box[:, 1],
            box[:, 2],
            box[:, 3],
        ))
        # Gather the per-cell surface with one object-array take, then zip the three lists in C: building
        # ~10^4 tuples in a Python comprehension was the single largest cost of this method.
        surface_table = np.empty(len(atlas.surfaces), dtype=object)
        surface_table[:] = atlas.surfaces
        surfaces = surface_table[palette_index * atlas.n_sizes + size_index].tolist()
        return list(zip(surfaces, destinations.tolist(), sources.tolist()))

    def _density_texture(
        self,
        density: np.ndarray,
        tick_index: Any,
    ) -> pygame.Surface:
        # The simulation advances at 60 Hz. Rebuilding the small density layer
        # every second tick halves conversion work while keeping the field
        # current within 1/30 second. Key by the public tick and shape, not the
        # concentration array identity: the engine replaces that array in-place
        # at each fixed step.
        key = None
        if tick_index is not None:
            key = (int(tick_index) // self.DENSITY_REFRESH_TICKS, density.shape, self.overlay)
            if key == self._density_cache_key and self._density_surface is not None:
                return self._density_surface

        density = np.asarray(density, dtype=np.float64)
        if density.ndim != 2:
            raise ValueError("WorldView.field_density must have shape [ny,nx]")
        positive = density[density > 0.0]
        scale = float(np.percentile(positive, 80)) if positive.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        low, high, alpha_low, alpha_high, logarithmic = OVERLAY_STYLE.get(self.overlay, OVERLAY_STYLE["nutrients"])
        if logarithmic:
            level = np.log1p(np.maximum(density, 0.0) / scale)
        else:
            level = np.maximum(density, 0.0)
        maximum = float(np.max(level)) if level.size else 0.0
        if maximum > 0.0:
            level = np.clip(level / maximum, 0.0, 1.0)
        self._level = level
        low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
        rgb = np.clip(low + (high - low) * level[..., None], 0, 255).astype(np.uint8)
        alpha = np.clip(alpha_low + (alpha_high - alpha_low) * level ** 1.1, 0, 255).astype(np.uint8)

        surface = pygame.Surface((density.shape[1], density.shape[0]), pygame.SRCALPHA, 32)
        rgb_view = pygame.surfarray.pixels3d(surface)
        rgb_view[:] = np.transpose(rgb, (1, 0, 2))
        del rgb_view
        alpha_view = pygame.surfarray.pixels_alpha(surface)
        alpha_view[:] = alpha.T
        del alpha_view
        self._density_surface = surface
        self._density_cache_key = key
        return surface

    def _draw_density(
        self,
        target: pygame.Surface,
        density: np.ndarray,
        width_um: float,
        height_um: float,
        tick_index: Any,
    ) -> int:
        texture = self._density_texture(density, tick_index)
        nx, ny = texture.get_size()
        dx, dy = width_um / nx, height_um / ny
        left, top, right, bottom = self.camera.visible_world_rect()
        x_min, y_min = max(0.0, left), max(0.0, top)
        x_max, y_max = min(width_um, right), min(height_um, bottom)
        if x_min >= x_max or y_min >= y_max:
            return 0

        x0 = max(0, int(np.floor(x_min / dx)))
        y0 = max(0, int(np.floor(y_min / dy)))
        x1 = min(nx, int(np.ceil(x_max / dx)))
        y1 = min(ny, int(np.ceil(y_max / dy)))
        if x0 >= x1 or y0 >= y1:
            return 0

        source_area = (x0, y0, x1 - x0, y1 - y0)
        destination = self.camera.world_to_screen((x0 * dx, y0 * dy))
        destination = (int(round(destination[0])), int(round(destination[1])))
        output_size = (
            max(1, int(round((x1 - x0) * dx * self.camera.zoom))),
            max(1, int(round((y1 - y0) * dy * self.camera.zoom))),
        )
        scale_key = None
        if self._density_cache_key is not None:
            scale_key = (self._density_cache_key, source_area, output_size, destination)
        if scale_key is None or scale_key != self._scaled_density_key:
            cropped = texture.subsurface(source_area)
            self._scaled_density_surface = pygame.transform.smoothscale(cropped, output_size)
            self._scaled_density_key = scale_key
            self._scaled_density_destination = destination
        target.blit(self._scaled_density_surface, self._scaled_density_destination)
        return (x1 - x0) * (y1 - y0)

    def _speck_candidates(self, width_um: float, height_um: float) -> tuple[np.ndarray, ...] | None:
        """Per-texture-refresh table of speck anchors: world x, y, phase, size class (render-local hash)."""
        level = self._level
        if level is None:
            return None
        tick_key = self._density_cache_key[0] // 4 if self._density_cache_key else None
        key = (tick_key, level.shape, width_um, height_um)
        if key == self._speck_cache_key and self._speck_table is not None:
            return self._speck_table
        ny, nx = level.shape
        iy, ix = np.nonzero(level > 0.22)
        if len(ix) == 0:
            self._speck_cache_key, self._speck_table = key, (np.empty(0),) * 4
            return self._speck_table
        per_cell = np.minimum(1 + (level[iy, ix] * 4.0).astype(np.intp), 5)
        cell = np.repeat(np.arange(len(ix)), per_cell)
        slot = np.arange(len(cell)) - np.repeat(np.cumsum(per_cell) - per_cell, per_cell)
        h = ix[cell].astype(np.uint64) * np.uint64(73856093) ^ iy[cell].astype(np.uint64) * np.uint64(19349663) \
            ^ slot.astype(np.uint64) * np.uint64(83492791)
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        h = h ^ (h >> np.uint64(31))
        u = (h & np.uint64(0xFFFF)).astype(np.float64) / 65536.0
        v = ((h >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.float64) / 65536.0
        w = ((h >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.float64) / 65536.0
        x = (ix[cell] + u) * (width_um / nx)
        y = (iy[cell] + v) * (height_um / ny)
        size_class = np.minimum((w * 3.0).astype(np.intp), 2)
        if len(x) > self.MAX_SPECKS:
            keep = np.linspace(0, len(x) - 1, self.MAX_SPECKS).astype(np.intp)
            x, y, w, size_class = x[keep], y[keep], w[keep], size_class[keep]
        self._speck_cache_key, self._speck_table = key, (x, y, w * 6.283, size_class)
        return self._speck_table

    def _draw_specks(self, target: pygame.Surface, view: WorldView) -> None:
        if self.overlay != "nutrients":
            return
        table = self._speck_candidates(view.width_um, view.height_um)
        if table is None or len(table[0]) == 0:
            return
        x, y, phase, size_class = table
        t = float(view.tick_index)
        wobble = 3.0 / max(self.camera.zoom, 0.05)
        wx = x + wobble * np.sin(t * 0.021 + phase)
        wy = y + wobble * np.cos(t * 0.017 + phase * 1.3)
        screen = np.asarray(self.camera.world_to_screen(np.column_stack((wx, wy))))
        width, height = target.get_size()
        on = (screen[:, 0] > -8) & (screen[:, 0] < width + 8) & (screen[:, 1] > -8) & (screen[:, 1] < height + 8)
        sprites = self.atlas.speck_sprites
        half = [s.get_width() // 2 for s in sprites]
        commands = [
            (sprites[c], (int(px) - half[c], int(py) - half[c]))
            for (px, py), c in zip(screen[on].tolist(), size_class[on].tolist())
        ]
        target.blits(commands, doreturn=False)

    def _draw_field_lit(self, target: pygame.Surface, width_um: float, height_um: float) -> None:
        top_left = self.camera.world_to_screen((0.0, 0.0))
        bottom_right = self.camera.world_to_screen((width_um, height_um))
        rect = pygame.Rect(
            int(round(top_left[0])),
            int(round(top_left[1])),
            max(1, int(round(bottom_right[0] - top_left[0]))),
            max(1, int(round(bottom_right[1] - top_left[1]))),
        )
        target.fill((4, 6, 7), rect.clip(target.get_rect()), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_field_edge(self, target: pygame.Surface, width_um: float, height_um: float) -> None:
        top_left = self.camera.world_to_screen((0.0, 0.0))
        bottom_right = self.camera.world_to_screen((width_um, height_um))
        rect = pygame.Rect(
            int(round(top_left[0])),
            int(round(top_left[1])),
            max(1, int(round(bottom_right[0] - top_left[0]))),
            max(1, int(round(bottom_right[1] - top_left[1]))),
        )
        for grow, colour in ((2, (22, 30, 34)), (1, (44, 58, 64)), (0, (92, 112, 118))):
            pygame.draw.rect(target, colour, rect.inflate(2 * grow, 2 * grow), width=1, border_radius=3 + grow)


Renderer = PygameRenderer
