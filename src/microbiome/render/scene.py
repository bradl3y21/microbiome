"""Microscope backdrop: uneven illumination, vignette and grain baked once; out-of-focus debris drifts with the tick.

Every position is a pure function of (camera, tick_index), never of wall-clock time or call history, so a frame is
a function of its snapshot. All randomness is a fixed-seed local generator used only for looks; nothing here
touches simulation state or any simulation RNG stream.
"""

from __future__ import annotations

import math

import numpy as np
import pygame


class Backdrop:
    DEFOCUS = 12   # large, faint, out-of-focus discs on a far plane
    DEBRIS = 120   # small particles drifting in the medium

    CENTRE = np.array([46.0, 62.0, 68.0])
    EDGE = np.array([9.0, 13.0, 17.0])

    def __init__(self) -> None:
        self._size: tuple[int, int] | None = None
        self._base: pygame.Surface | None = None
        rng = np.random.default_rng(20261017)
        self._defocus_u = rng.random(self.DEFOCUS)
        self._defocus_v = rng.random(self.DEFOCUS)
        self._defocus_radius = rng.uniform(18.0, 80.0, self.DEFOCUS)
        self._defocus_parallax = np.sort(rng.uniform(0.04, 0.22, self.DEFOCUS))
        self._defocus_speed = rng.uniform(0.004, 0.012, self.DEFOCUS)
        self._defocus = [self._make_defocus(r) for r in self._defocus_radius]
        self._debris_u = rng.random(self.DEBRIS)
        self._debris_v = rng.random(self.DEBRIS)
        self._debris_parallax = rng.uniform(0.3, 0.95, self.DEBRIS)
        self._debris_phase = rng.uniform(0.0, 6.28, self.DEBRIS)
        self._debris_kind = rng.integers(0, 3, self.DEBRIS)
        self._debris_sprites = [self._make_debris(r) for r in (1, 2, 3)]

    @staticmethod
    def _make_defocus(radius: float) -> pygame.Surface:
        """A soft flat-topped disc with a slightly brighter edge, the look of a particle far out of focus."""
        r = int(radius)
        side = 2 * r + 4
        y, x = np.mgrid[0:side, 0:side].astype(np.float64)
        d = np.hypot(x - side / 2.0, y - side / 2.0) / r
        body = np.clip((1.0 - d) / 0.45, 0.0, 1.0) ** 1.4
        ring = np.exp(-(((d - 0.82) / 0.1) ** 2)) * (d < 1.0)
        alpha = np.clip(12.0 * body + 14.0 * ring, 0, 255)
        surface = pygame.Surface((side, side), pygame.SRCALPHA, 32)
        rgb = pygame.surfarray.pixels3d(surface)
        rgb[:] = np.array([128, 156, 156], dtype=np.uint8)
        del rgb
        view = pygame.surfarray.pixels_alpha(surface)
        view[:] = alpha.T.astype(np.uint8)
        del view
        return surface

    @staticmethod
    def _make_debris(radius: int) -> pygame.Surface:
        side = radius * 4 + 2
        surface = pygame.Surface((side, side), pygame.SRCALPHA, 32)
        for step in range(radius * 2, 0, -1):
            t = 1.0 - step / (radius * 2)
            pygame.draw.circle(surface, (150, 164, 158, int(10 + 60 * t * t)), (side // 2, side // 2), max(1, step))
        return surface

    def _build_base(self, size: tuple[int, int], like: pygame.Surface) -> None:
        width, height = size
        xs = np.linspace(-1.0, 1.0, width)[:, None]
        ys = np.linspace(-1.0, 1.0, height)[None, :]
        radial = np.sqrt(0.85 * xs * xs + 0.95 * ys * ys)
        light = 1.0 - 0.8 * np.clip(radial / 1.3, 0.0, 1.0) ** 1.6
        light *= 0.93 + 0.07 * (1.0 - (ys + 1.0) * 0.5)          # a touch darker toward the top

        # Uneven illumination: a very low-frequency luminance field, scaled up in steps so it has no creases.
        rng = np.random.default_rng(77)
        cloud = pygame.Surface((9, 6))
        for ix in range(9):
            for iy in range(6):
                value = int(rng.integers(0, 256))
                cloud.set_at((ix, iy), (value, value, value))
        for factor in (4, 2, 1):
            cloud = pygame.transform.smoothscale(cloud, (max(2, width // factor), max(2, height // factor)))
        unevenness = pygame.surfarray.array3d(cloud)[..., 0].astype(np.float64) / 255.0 - 0.5

        base = self.EDGE + (self.CENTRE - self.EDGE) * light[..., None]
        base += (unevenness * 9.0 * light)[..., None]
        noise = np.random.default_rng(5).random((2, width, height, 1))
        base += (noise[0] + noise[1] - 1.0) * 3.2              # sensor grain, triangular so it has no flat steps
        self._base = pygame.surfarray.make_surface(np.clip(np.rint(base), 0, 255).astype(np.uint8)).convert(like)
        self._size = size

    def draw(self, target: pygame.Surface, centre_shift_px: tuple[float, float], tick_index: int) -> None:
        size = target.get_size()
        if size != self._size or self._base is None:
            self._build_base(size, target)
        target.blit(self._base, (0, 0))
        width, height = size
        shift_x, shift_y = centre_shift_px
        tick = float(tick_index)

        discs = []
        for i in range(self.DEFOCUS):
            r = self._defocus_radius[i]
            span_x, span_y = width + 2.0 * r, height + 2.0 * r
            par = self._defocus_parallax[i]
            px = (self._defocus_u[i] * span_x + shift_x * par + 6.0 * math.sin(tick * self._defocus_speed[i] + i)) % span_x - r
            py = (self._defocus_v[i] * span_y + shift_y * par + 5.0 * math.cos(tick * self._defocus_speed[i] * 0.8 + i * 2.0)) % span_y - r
            discs.append((self._defocus[i], (int(px - r), int(py - r))))
        target.blits(discs, doreturn=False)

        debris = []
        for i in range(self.DEBRIS):
            par = self._debris_parallax[i]
            phase = self._debris_phase[i]
            px = (self._debris_u[i] * width + shift_x * par + 5.0 * math.sin(tick * 0.013 + phase)
                  + 2.0 * math.sin(tick * 0.047 + phase * 2.3)) % width
            py = (self._debris_v[i] * height + shift_y * par + 4.0 * math.cos(tick * 0.011 + phase * 1.7)
                  + 2.0 * math.cos(tick * 0.051 + phase)) % height
            sprite = self._debris_sprites[int(self._debris_kind[i])]
            debris.append((sprite, (int(px) - sprite.get_width() // 2, int(py) - sprite.get_height() // 2)))
        target.blits(debris, doreturn=False)
