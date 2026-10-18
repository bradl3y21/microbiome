"""Pre-rendered microbe sprites: translucent rods in the look of a phase-contrast microscope image.

Everything here is baked once and blitted in one batch per frame, so the frame cost does not depend on how
detailed a sprite is. A sprite block holds every heading (16) of the resting body, and every heading of two
flagellar-wave phases for motile bodies, so a swimming cell faces where it is going and its flagella beat.
Blocks are built lazily the first time a (palette, size) pair is drawn.

The look is deliberately muted: desaturated body colours, a faint bright halo and a thin bright wall edge (as
phase contrast shows them), diffuse nucleoid and granules, thin flagella, and no specular highlights.
"""

from __future__ import annotations

import math

import numpy as np
import pygame


def _lerp(a: tuple[float, ...], b: tuple[float, ...], t: float) -> tuple[int, ...]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _scale(colour: tuple[float, ...], factor: float) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, round(c * factor)))) for c in colour)  # type: ignore[return-value]


class SpriteAtlas:
    """Lazy per-(palette, size) atlases of rod-shaped, flagellated microbes."""

    PALETTE_COUNT = 8           # species colours
    SPORE_STYLE = 8             # a ninth style: the phase-bright oval of a dormant endospore
    STYLE_COUNT = 9
    HALF_WIDTHS_PX = (2, 3, 4, 5, 6, 8, 10, 13, 16)
    HEADINGS = 16
    PHASES = 2
    VARIANTS = HEADINGS * (1 + PHASES)
    COLUMNS = 8
    TAIL = 1.3  # flagellum length as a multiple of the body half-length

    # (muted body colour, rod length:width, flagella). Real rods differ in proportion; colour is only identity.
    STYLES = (
        ((214, 190, 172), 2.3, 5),   # warm ivory-rose: short peritrichous rod (E. coli-like)
        ((166, 194, 182), 3.4, 4),   # sage grey: long rod (Bacillus-like)
        ((172, 184, 206), 2.6, 3),
        ((206, 174, 194), 3.0, 4),
        ((214, 200, 156), 1.9, 5),
        ((190, 158, 148), 2.8, 3),
        ((158, 196, 158), 2.4, 4),
        ((176, 178, 172), 3.2, 5),
        ((236, 240, 228), 1.45, 0),  # spore
    )

    def __init__(self) -> None:
        sizes = len(self.HALF_WIDTHS_PX)
        self.surfaces: list[pygame.Surface | None] = [None] * (self.STYLE_COUNT * sizes)
        self.tile_size = np.zeros((self.STYLE_COUNT, sizes), dtype=np.int32)
        self.crop = np.zeros((self.STYLE_COUNT, sizes, self.VARIANTS, 4), dtype=np.int32)
        self.matrix_sprites: dict[int, pygame.Surface] = {}
        self.size_boundaries = np.sqrt(np.asarray(self.HALF_WIDTHS_PX[:-1]) * np.asarray(self.HALF_WIDTHS_PX[1:]))
        self.speck_sprites = [self._soft_dot(r, (176, 150, 104), 0.5) for r in (1, 1, 2)]

    # ---------------------------------------------------------------- public lookups
    @property
    def n_sizes(self) -> int:
        return len(self.HALF_WIDTHS_PX)

    def size_index(self, half_width_px: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.size_boundaries, half_width_px).astype(np.intp)

    def matrix_sprite(self, size: int) -> pygame.Surface:
        """A soft, pale halo of extracellular matrix, a few cell-widths across, for one sprite size."""
        found = self.matrix_sprites.get(size)
        if found is None:
            radius = max(4, int(round(self.HALF_WIDTHS_PX[size] * 2.6)))
            side = 2 * radius + 2
            found = pygame.Surface((side, side), pygame.SRCALPHA, 32)
            for step in range(radius, 0, -1):
                t = 1.0 - step / radius
                pygame.draw.circle(found, (150, 176, 168, int(8 + 38 * t * t)), (side // 2, side // 2), step)
            self.matrix_sprites[size] = found
        return found

    def ensure_palette(self, palette: int) -> None:
        """Bake every size for one species at once, so zooming never stalls on a first-use bake."""
        for size in range(self.n_sizes):
            self.ensure(palette, size)

    def ensure(self, palette: int, size: int) -> None:
        palette, size = int(palette) % self.STYLE_COUNT, int(size)
        if self.tile_size[palette, size] == 0:
            self._build_block(palette, size)

    # ---------------------------------------------------------------- colours
    @classmethod
    def palette_colors(cls, palette: int) -> dict[str, tuple[int, int, int]]:
        base = cls.STYLES[palette % cls.STYLE_COUNT][0]
        return {
            "core": _scale(base, 1.04),
            "mid": _scale(base, 0.86),
            "rim": _scale(base, 0.5),
            "glow": _scale(base, 0.7),
            "dark": _scale(base, 0.26),
            "light": tuple(min(255, int(c * 1.12 + 22)) for c in base),  # type: ignore[misc]
        }

    @staticmethod
    def _soft_dot(radius: int, colour: tuple[int, int, int], strength: float = 1.0) -> pygame.Surface:
        size = radius * 4 + 2
        surface = pygame.Surface((size, size), pygame.SRCALPHA, 32)
        centre = size // 2
        for step in range(radius * 2, 0, -1):
            t = 1.0 - step / (radius * 2)
            alpha = int((18 + 200 * t * t) * strength)
            pygame.draw.circle(surface, (*colour, min(255, alpha)), (centre, centre), max(1, step))
        return surface

    # ---------------------------------------------------------------- drawing helpers
    @staticmethod
    def _capsule(surface: pygame.Surface, colour: tuple[int, ...], cx: float, cy: float,
                 half_length: float, half_width: float) -> None:
        straight = max(0.0, half_length - half_width)
        width = max(1, int(round(half_width)))
        pygame.draw.rect(surface, colour, (int(cx - straight), int(cy - half_width), int(2 * straight) + 1,
                                           int(2 * half_width) + 1))
        pygame.draw.circle(surface, colour, (int(cx - straight), int(cy)), width)
        pygame.draw.circle(surface, colour, (int(cx + straight), int(cy)), width)

    def _draw_master(self, palette: int, size: int, motile: bool, phase: int, scale: int, extent: int) -> pygame.Surface:
        """One unrotated cell facing +x on a (2*extent*scale)^2 canvas; the flagella trail toward -x."""
        _, aspect, flagella = self.STYLES[palette]
        colours = self.palette_colors(palette)
        h = self.HALF_WIDTHS_PX[size] * scale
        half_length = h * aspect
        straight = half_length - h
        side = 2 * extent * scale
        surface = pygame.Surface((side, side), pygame.SRCALPHA, 32)
        cx = cy = side / 2.0

        # Phase-contrast halo: a faint bright fringe just outside the wall.
        for grow, alpha in ((0.55, 6), (0.3, 12), (0.14, 20)):
            self._capsule(surface, (*colours["light"], alpha), cx, cy, half_length + h * grow, h * (1 + grow))

        if motile and self.HALF_WIDTHS_PX[size] >= 4:
            wave_phase = phase * math.pi
            length = half_length * self.TAIL
            width = max(1, int(round(scale * 0.8)))
            for index in range(flagella):
                spread = 0.0 if flagella == 1 else -1.15 + 2.3 * index / (flagella - 1)
                bx = cx - straight - math.cos(spread) * h * 0.95
                by = cy + math.sin(spread) * h * 0.95
                fan = spread * 0.7
                direction = (-math.cos(fan), math.sin(fan))
                normal = (-direction[1], direction[0])
                points = []
                for k in range(17):
                    s = k / 16.0
                    wiggle = h * 0.42 * s * math.sin(2.0 * math.pi * 1.5 * s + wave_phase + index * 1.7)
                    points.append((bx + direction[0] * s * length + normal[0] * wiggle,
                                   by + direction[1] * s * length + normal[1] * wiggle))
                for k in range(16):
                    s = k / 16.0
                    pygame.draw.line(surface, (*colours["light"], int(120 - 70 * s)), points[k], points[k + 1], width)

        # Body: concentric capsules, a little brighter along the axis than at the wall (cylinder shading).
        steps = max(4, min(10, int(h)))
        for i in range(steps):
            t = i / (steps - 1)
            hw = h * (1.0 - 0.8 * t)
            colour = _lerp(colours["rim"], colours["core"], t ** 0.8)
            self._capsule(surface, (*colour, 196), cx, cy, half_length - (h - hw), hw)

        # Everything below is drawn on an overlay so it cannot punch holes in the body's alpha.
        overlay = pygame.Surface((side, side), pygame.SRCALPHA, 32)
        rng = np.random.default_rng(1000 + palette * 31 + size)
        if self.HALF_WIDTHS_PX[size] >= 3 and palette != self.SPORE_STYLE:
            span = straight * 1.0 + h * 0.5
            for _ in range(3):  # diffuse nucleoid: a few overlapping soft dark blobs
                bx = cx + (rng.random() * 2.0 - 1.0) * span * 0.55
                by = cy + (rng.random() * 2.0 - 1.0) * h * 0.18
                pygame.draw.ellipse(overlay, (*colours["dark"], 74),
                                    (bx - h * 0.95, by - h * 0.46, h * 1.9, h * 0.92))
            for k in range(int(5 + h * 0.5)):  # granules and storage bodies, low contrast
                gx = cx + (rng.random() * 2.0 - 1.0) * (straight + h * 0.55)
                gy = cy + (rng.random() * 2.0 - 1.0) * h * 0.52
                dark = k % 3 != 0
                colour = (*colours["dark"], 118) if dark else (*colours["light"], 112)
                pygame.draw.circle(overlay, colour, (int(gx), int(gy)), max(1, int(h * (0.07 + 0.06 * rng.random()))))
        # Thin bright wall edge, as phase contrast renders it.
        edge = max(1, int(round(h * 0.09)))
        wall = (*colours["light"], 168)
        pygame.draw.line(overlay, wall, (cx - straight, cy - h + edge * 0.5), (cx + straight, cy - h + edge * 0.5), edge)
        pygame.draw.line(overlay, wall, (cx - straight, cy + h - edge * 0.5), (cx + straight, cy + h - edge * 0.5), edge)
        pygame.draw.circle(overlay, wall, (int(cx - straight), int(cy)), int(round(h)), edge,
                           draw_top_left=True, draw_bottom_left=True)
        pygame.draw.circle(overlay, wall, (int(cx + straight), int(cy)), int(round(h)), edge,
                           draw_top_right=True, draw_bottom_right=True)
        surface.blit(overlay, (0, 0))
        return surface

    def _build_block(self, palette: int, size: int) -> None:
        h = self.HALF_WIDTHS_PX[size]
        aspect = self.STYLES[palette][1]
        half_length = h * aspect
        extent = int(math.ceil(half_length * (1.0 + self.TAIL) + h * 0.7 + 3))
        tile = 2 * extent
        scale = 2 if h <= 8 else 1
        rows = math.ceil(self.VARIANTS / self.COLUMNS)
        atlas = pygame.Surface((self.COLUMNS * tile, rows * tile), pygame.SRCALPHA, 32)
        atlas.fill((0, 0, 0, 0))

        for motile, phases in ((False, (0,)), (True, tuple(range(self.PHASES)))):
            for phase in phases:
                master = self._draw_master(palette, size, motile, phase, scale, extent)
                for heading in range(self.HEADINGS):
                    degrees = heading * 360.0 / self.HEADINGS
                    rotated = pygame.transform.rotozoom(master, -degrees, 1.0 / scale)
                    cell = pygame.Surface((tile, tile), pygame.SRCALPHA, 32)
                    cell.blit(rotated, rotated.get_rect(center=(extent, extent)))
                    index = self.HEADINGS * (1 + phase) + heading if motile else heading
                    box = cell.get_bounding_rect(min_alpha=6)
                    self.crop[palette, size, index] = (box.x, box.y, max(1, box.w), max(1, box.h))
                    atlas.blit(cell, ((index % self.COLUMNS) * tile, (index // self.COLUMNS) * tile),
                               special_flags=pygame.BLEND_RGBA_ADD)
        self.surfaces[palette * self.n_sizes + size] = atlas
        self.tile_size[palette, size] = tile
