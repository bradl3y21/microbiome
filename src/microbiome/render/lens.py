"""A renderer that shows the world the way a microscope does: true scale, optics, no game graphics.

``MicroscopeRenderer`` is a drop-in for ``PygameRenderer`` (same camera, ``draw(target, snapshot)`` and ``overlay``)
but every cell is an image computed by ``render.microscope`` from its real rod length, width and orientation, at the
magnification of the chosen objective. The mouse wheel changes objective, not an arbitrary zoom: 10x, 20x, 40x, 60x,
100x. The sample is periodic (as the simulation is), so panning never reaches an edge.

Controls it owns: objective (wheel), imaging mode (``cycle_mode``), focus (``nudge_focus``), eyepiece mask
(``toggle_eyepiece``). Nutrients are invisible in a real microscope, so the default overlay ("nutrients") draws
nothing; the other overlays are an optional analysis layer and are labelled as such.
"""

from __future__ import annotations

import gc
import math
from typing import Any

import numpy as np
import pygame

from . import microscope as M
from .camera import Camera2D
from .renderer import PygameRenderer, RenderStats
from .view import WorldView

TEXT_ON_LIGHT = (40, 40, 44)
TEXT_ON_DARK = (205, 208, 214)
DEBRIS_COUNT = 220
DEBRIS_SIZES_UM = (0.3, 0.55, 0.95, 1.6)


class MicroscopeRenderer(PygameRenderer):
    def __init__(self, mode: str = "phase", objective: int = 1, camera: Camera2D | None = None) -> None:
        if mode not in M.MODES:
            raise ValueError(f"unknown imaging mode {mode!r}")
        if not 0 <= objective < len(M.OBJECTIVES):
            raise ValueError("objective index out of range")
        zoom = M.OBJECTIVES[objective][1]
        super().__init__(camera=camera or Camera2D(zoom=zoom, min_zoom=M.OBJECTIVES[0][1], max_zoom=M.OBJECTIVES[-1][1]))
        self.camera.zoom = zoom
        self.mode = mode
        self.objective = objective
        self.focus_um = 0.5 * M.FILM_DEPTH_UM
        self.eyepiece = False
        self.style_remap: np.ndarray | None = None      # colour_index -> imaging style (e.g. Gram-negative = 0, Gram-positive = 1)
        self.show_label = True
        self._atlas: M.MicroAtlas | None = None
        self._field: pygame.Surface | None = None
        self._field_key: Any = None
        self._noise: list[pygame.Surface] = []
        self._eyepiece_surface: pygame.Surface | None = None
        self._eyepiece_size: tuple[int, int] | None = None
        self._debris_table: tuple[np.ndarray, ...] | None = None
        self._debris_key: Any = None
        self._font: pygame.font.Font | None = None

    # ------------------------------------------------------------------------------------------ controls
    @property
    def px_per_um(self) -> float:
        return M.OBJECTIVES[self.objective][1]

    @property
    def magnification(self) -> int:
        return M.OBJECTIVES[self.objective][0]

    def set_objective(self, index: int, anchor_px: tuple[float, float] | None = None) -> bool:
        index = int(np.clip(index, 0, len(M.OBJECTIVES) - 1))
        if index == self.objective:
            return False
        old = self.camera.zoom
        self.objective = index
        self.camera.zoom_at(anchor_px or (self.camera.viewport_size[0] / 2.0, self.camera.viewport_size[1] / 2.0),
                            M.OBJECTIVES[index][1] / old)
        self.camera.zoom = M.OBJECTIVES[index][1]
        return True

    def cycle_mode(self, step: int = 1) -> str:
        self.mode = M.MODES[(M.MODES.index(self.mode) + step) % len(M.MODES)]
        return self.mode

    def nudge_focus(self, delta_um: float) -> float:
        self.focus_um = float(np.clip(self.focus_um + delta_um, 0.0, M.FILM_DEPTH_UM))
        return self.focus_um

    def toggle_eyepiece(self) -> bool:
        self.eyepiece = not self.eyepiece
        return self.eyepiece

    def handle_event(self, event: pygame.event.Event, pointer_position: tuple[int, int] | None = None) -> bool:
        wheel = getattr(pygame, "MOUSEWHEEL", -1)
        step = 0
        if event.type == wheel:
            step = int(np.sign(getattr(event, "y", 0)))
        elif event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) in (4, 5):
            step = 1 if event.button == 4 else -1
        if event.type == wheel or step:
            pos = pointer_position or pygame.mouse.get_pos()
            if step:
                self.set_objective(self.objective + step, pos)
            return True
        return self.camera.handle_event(event, pointer_position)

    # ------------------------------------------------------------------------------------------ image pieces
    def _ensure_optics(self, size: tuple[int, int]) -> None:
        if self._atlas is None or self._atlas.mode != self.mode or self._atlas.objective != self.objective:
            self._atlas = M.MicroAtlas(self.mode, self.objective)
        key = (self.mode, size)
        if self._field_key != key:
            self._field = M.build_field(self.mode, size)
            self._noise = M.build_noise(self.mode, size)
            self._field_key = key

    def _debris(self, width_um: float, height_um: float) -> tuple[np.ndarray, ...]:
        key = (round(width_um, 3), round(height_um, 3))
        if self._debris_key != key:
            rng = np.random.default_rng(77)
            pos = rng.random((DEBRIS_COUNT, 2)) * (width_um, height_um)
            size = rng.choice(4, DEBRIS_COUNT, p=(0.60, 0.28, 0.10, 0.02))
            self._debris_table = (pos, size)
            self._debris_key = key
        return self._debris_table

    def _wrapped(self, positions: np.ndarray, width_um: float, height_um: float, margin_um: float):
        """Indices and screen positions of every periodic image of ``positions`` that falls in (or near) the window."""
        left, top, right, bottom = self.camera.visible_world_rect()
        rows, screens = [], []
        for kx in (-1, 0, 1):
            for ky in (-1, 0, 1):
                shifted = positions + (kx * width_um, ky * height_um)
                keep = np.flatnonzero((shifted[:, 0] >= left - margin_um) & (shifted[:, 0] <= right + margin_um)
                                      & (shifted[:, 1] >= top - margin_um) & (shifted[:, 1] <= bottom + margin_um))
                if keep.size:
                    rows.append(keep)
                    screens.append(np.asarray(self.camera.world_to_screen(shifted[keep]), dtype=np.float64))
        if not rows:
            return np.empty(0, dtype=np.intp), np.empty((0, 2))
        return np.concatenate(rows), np.concatenate(screens)

    @property
    def _additive(self) -> bool:
        return self.mode in M.ADDITIVE

    def _flag(self) -> tuple[int, ...]:
        return (pygame.BLEND_ADD,) if self._additive else ()

    def _draw_debris(self, target: pygame.Surface, view: WorldView) -> None:
        pos, size = self._debris(view.width_um, view.height_um)
        rows, screen = self._wrapped(pos, view.width_um, view.height_um, 3.0)
        if not rows.size:
            return
        commands = []
        flag = self._flag()
        for bin_ in np.unique(size[rows]):
            sprite = self._atlas.debris(int(bin_))
            half = (sprite.get_width() // 2, sprite.get_height() // 2)
            pick = size[rows] == bin_
            commands.extend((sprite, (int(x) - half[0], int(y) - half[1]), None, *flag) for x, y in screen[pick].tolist())
        target.blits(commands, doreturn=False)

    def _cells(self, target: pygame.Surface, view: WorldView, angle: np.ndarray) -> int:
        count = len(view.position_um)
        if not count:
            return 0
        atlas = self._atlas
        length = view.length_um if view.length_um is not None else M.rod_length_from_radius(view.radius_um)
        rows, screen = self._wrapped(view.position_um, view.width_um, view.height_um, 0.5 * float(length.max()) + 1.5)
        if not rows.size:
            return 0
        if view.cell_state is not None:
            self._draw_halos(target, screen, rows, view)
        style = view.colour_index[rows].astype(np.int64)
        spore = style == M.SPORE_STYLE
        if self.style_remap is not None:
            remap = np.asarray(self.style_remap, dtype=np.int64)
            style = np.where(spore, style, remap[np.minimum(style, len(remap) - 1)])
        ids = view.agent_id[rows] if view.agent_id is not None else np.arange(rows.size, dtype=np.uint64)
        depth = M.pseudo_depth_um(ids)
        dof = M.DEPTH_OF_FIELD_UM[self.objective]
        away = np.abs(depth - self.focus_um)
        blur = np.where(away <= dof, 0, np.where(away <= 2.2 * dof, 1, 2)).astype(np.int64)
        bins = np.clip(np.rint(np.log(np.maximum(length[rows], 0.8) / 0.8) / math.log(6.0 / 0.8) * (len(M.LENGTH_BINS) - 1)),
                       0, len(M.LENGTH_BINS) - 1).astype(np.int64)
        bins = np.where(spore, 0, bins)
        heading = np.rint((angle[rows] % math.pi) / (math.pi / M.HEADINGS)).astype(np.int64) % M.HEADINGS
        keys = M.MicroAtlas.key(style, bins, blur)
        order = np.argsort(-blur, kind="stable")           # defocused cells first, so sharp ones lie on top
        flag = self._flag()
        commands: list[tuple[Any, ...]] = []
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for key in np.unique(keys[order]):
                strip, tile = atlas.strip(int(key))
                pick = order[keys[order] == key]
                half = tile // 2
                xs = (screen[pick, 0] - half).astype(np.int64).tolist()
                ys = (screen[pick, 1] - half).astype(np.int64).tolist()
                hs = (heading[pick] * tile).tolist()
                commands.extend((strip, (x, y), (h, 0, tile, tile), *flag) for x, y, h in zip(xs, ys, hs))
            target.blits(commands, doreturn=False)
        finally:
            if gc_was_enabled:
                gc.enable()
        return len(commands)

    def _draw_halos(self, target: pygame.Surface, screen: np.ndarray, rows: np.ndarray, view: WorldView) -> None:
        mask = view.cell_state[rows] == 3
        if not mask.any():
            return
        sprite = self._atlas.matrix()
        if self.mode in ("brightfield", "fluorescence"):
            return
        half = (sprite.get_width() // 2, sprite.get_height() // 2)
        flag = self._flag()
        target.blits([(sprite, (int(x) - half[0], int(y) - half[1]), None, *flag) for x, y in screen[mask].tolist()], doreturn=False)

    # ------------------------------------------------------------------------------------------ the picture
    def scale_bar(self) -> tuple[float, int]:
        """(length in micrometres, length in pixels) of the scale bar for the current objective."""
        um = M.nice_scale_bar_um(self.px_per_um, 0.22 * self.camera.viewport_size[0])
        return um, int(round(um * self.px_per_um))

    def label(self) -> str:
        return f"{M.MODE_LABEL[self.mode]}   {self.magnification}x   focus {self.focus_um - 0.5 * M.FILM_DEPTH_UM:+.1f} um"

    def draw(self, target: pygame.Surface, snapshot) -> RenderStats:
        view = WorldView.from_snapshot(snapshot, self.overlay)
        size = target.get_size()
        self.camera.resize(size)
        self.camera.set_field_size((view.width_um, view.height_um))
        self.camera.center_um[0] %= view.width_um
        self.camera.center_um[1] %= view.height_um
        self.camera.zoom = self.px_per_um
        self._ensure_optics(size)

        target.blit(self._field, (0, 0))
        density_cells = 0
        if self.overlay != "nutrients":
            density_cells = self._draw_density(target, view.field_density, view.width_um, view.height_um, view.tick_index)
        self._draw_debris(target, view)
        angle = view.heading_rad if view.heading_rad is not None else self._track_headings(view)
        drawn = self._cells(target, view, angle)
        target.blit(self._noise[(view.tick_index // 2) % len(self._noise)], (0, 0), special_flags=pygame.BLEND_ADD)
        if self.eyepiece:
            if self._eyepiece_size != size:
                self._eyepiece_surface = M.build_eyepiece(size)
                self._eyepiece_size = size
            target.blit(self._eyepiece_surface, (0, 0))
        self._draw_scale(target)
        return RenderStats(drawn, density_cells, 1 if drawn else 0)

    def _draw_scale(self, target: pygame.Surface) -> None:
        if not self.show_label:
            return
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.Font(None, 20)
        colour = TEXT_ON_LIGHT if self.mode in ("phase", "brightfield") else TEXT_ON_DARK
        width, height = target.get_size()
        um, px = self.scale_bar()
        x1, y = width - 28, height - 30
        x0 = x1 - px
        pygame.draw.line(target, colour, (x0, y), (x1, y), 3)
        for x in (x0, x1):
            pygame.draw.line(target, colour, (x, y - 5), (x, y + 5), 2)
        text = f"{um:g} um" if um < 1000 else f"{um / 1000:g} mm"
        rendered = self._font.render(text, True, colour)
        target.blit(rendered, rendered.get_rect(midbottom=((x0 + x1) // 2, y - 8)))
        note = self._font.render(self.label() + ("" if self.overlay == "nutrients" else f"   [analysis overlay: {self.overlay}]"), True, colour)
        target.blit(note, (24, height - 30 - note.get_height() // 2))
