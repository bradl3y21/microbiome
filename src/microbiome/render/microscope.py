"""Image formation for a microscope view: true-scale rods rendered by the physics of four imaging techniques.

Nothing here is a hand-painted sprite. Each cell image is computed from its real shape (a capsule: a cylinder with two
hemispherical caps, with a circular cross-section giving a thickness profile) and turned into an image by a model of
the technique, then blurred by the objective's resolution:

* ``phase``: positive phase contrast. Thick, dense cytoplasm retards light, so the cell is dark against a grey field,
  ringed by the bright halo this technique produces. Endospores are phase-bright with a dark rim.
* ``brightfield``: Gram stain. Gram-negative cells (E. coli) take the pink counterstain, Gram-positive (B. subtilis)
  the purple crystal violet; edges show a faint refractive rim; spores are unstained and show only a pale outline.
* ``darkfield``: only scattered light reaches the objective, so cells are bright, brightest at the edge, on black.
* ``fluorescence``: a fluorescent protein in the cytoplasm (green for the first organism, red for the second): bright
  where the cell is thick, dark elsewhere, spores nearly dark.

Scale is real: the camera runs in pixels per micrometre, set by the objective. Cells away from the focal plane (each has
a fixed depth inside the film) are blurred more and lose contrast, so focus matters. Flagella are 20 nm thick and
invisible in these techniques; so are nutrients.

This module imports only numpy and pygame: the renderer may not depend on the simulation.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

# (magnification, pixels per micrometre): 1280 px across is a 800 um field at 10x and an 80 um field at 100x
OBJECTIVES = ((10, 1.6), (20, 3.2), (40, 6.4), (60, 9.6), (100, 16.0))
DEPTH_OF_FIELD_UM = (9.0, 4.5, 2.2, 1.4, 0.9)          # how far from the focal plane a cell stays sharp
RESOLUTION_SIGMA_UM = (0.47, 0.30, 0.18, 0.14, 0.085)    # Gaussian width of the point-spread function (FWHM / 2.35)
MODES = ("phase", "brightfield", "darkfield", "fluorescence")
MODE_LABEL = {"phase": "Phase contrast", "brightfield": "Brightfield, Gram stain", "darkfield": "Darkfield",
              "fluorescence": "Fluorescence (GFP / mCherry)"}
ADDITIVE = {"darkfield", "fluorescence"}                 # light on black: sprites add to the field instead of covering it

FILM_DEPTH_UM = 10.0
WIDTH_UM = 0.8                                           # mirrors sim.realism's default; RealismSnapshot carries the real lengths
VOLUME_UM3_PER_PG = 3.3
SPORE_WIDTH_UM, SPORE_LENGTH_UM = 0.65, 1.05
STYLE_WIDTH_UM = (0.8, 0.8, 0.75, 0.0, 0.5)               # per organism style; 0 = a coccus, as wide as it is long (see genome.organisms)
HEADINGS = 12                                            # a rod looks the same turned by 180 degrees: 12 headings span 0..pi
LENGTH_BINS = np.geomspace(0.8, 6.0, 14)
BLUR_LEVELS = 3
SPORE_STYLE = 8
STYLES = 9

STAIN = ((226, 108, 150), (98, 58, 154), (210, 120, 90), (120, 90, 190), (200, 140, 160), (150, 110, 170), (190, 100, 120), (110, 80, 160), (200, 196, 190))
FLUORESCENCE = ((70, 238, 96), (255, 74, 58), (72, 202, 255), (255, 218, 72), (232, 92, 232), (255, 152, 52), (150, 255, 204), (202, 202, 255), (60, 70, 60))
SCATTER = ((214, 228, 255), (232, 224, 255), (255, 232, 214), (214, 255, 232), (240, 240, 255), (255, 240, 224), (224, 255, 255), (255, 224, 240), (255, 255, 255))
BACKGROUND = {"phase": (176, 178, 172), "brightfield": (238, 233, 226), "darkfield": (5, 6, 9), "fluorescence": (3, 4, 5)}
NOISE_SIGMA = {"phase": 2.4, "brightfield": 2.0, "darkfield": 2.2, "fluorescence": 3.2}
VIGNETTE = {"phase": 0.26, "brightfield": 0.18, "darkfield": 0.0, "fluorescence": 0.0}


def rod_length_from_radius(radius_um: np.ndarray) -> np.ndarray:
    """Rod length from the simulation's sphere-equivalent radius, for snapshots that carry no true length."""
    mass_pg = 4.0 / 3.0 * np.pi * np.maximum(np.asarray(radius_um, dtype=np.float64), 0.0) ** 3
    volume = VOLUME_UM3_PER_PG * mass_pg
    cylinder = np.maximum(volume - math.pi * WIDTH_UM ** 3 / 6.0, 0.0) / (math.pi * WIDTH_UM ** 2 / 4.0)
    return cylinder + WIDTH_UM


def pseudo_depth_um(agent_id: np.ndarray) -> np.ndarray:
    """A cell's fixed depth in the film: a hash of its id, so it never changes and needs no simulation state."""
    h = (np.asarray(agent_id, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(97)) % np.uint64(10007)
    return h.astype(np.float64) / 10007.0 * FILM_DEPTH_UM


def nearest_objective(px_per_um: float) -> int:
    return int(np.argmin(np.abs(np.log(np.asarray([p for _, p in OBJECTIVES]) / max(px_per_um, 1e-9)))))


def nice_scale_bar_um(px_per_um: float, max_px: float) -> float:
    """The largest 1-2-5 length in micrometres whose bar fits in ``max_px`` pixels."""
    best = 1.0
    for decade in (0.1, 1.0, 10.0, 100.0, 1000.0):
        for step in (1.0, 2.0, 5.0):
            if decade * step * px_per_um <= max_px:
                best = decade * step
    return best


# ------------------------------------------------------------------------------------------------ optics helpers
def _blur(stack: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur of every (T, T) plane of ``stack`` (H, T, T) by FFT, zero-padded so nothing wraps around."""
    if sigma_px < 0.3:
        return stack
    pad = int(math.ceil(3.0 * sigma_px))
    size = stack.shape[-1] + 2 * pad
    size += size % 2
    freq = np.fft.rfftfreq(size)
    transfer = np.exp(-2.0 * (math.pi * sigma_px) ** 2 * (freq[None, :] ** 2 + np.fft.fftfreq(size)[:, None] ** 2))
    padded = np.zeros(stack.shape[:-2] + (size, size), dtype=np.float32)
    padded[..., :stack.shape[-2], :stack.shape[-1]] = stack
    out = np.fft.irfft2(np.fft.rfft2(padded) * transfer, s=(size, size))
    return out[..., :stack.shape[-2], :stack.shape[-1]].astype(np.float32)


def _rod_fields(length_um: float, width_um: float, tile: int, s: float, seed: int):
    """Per heading: distance to the capsule surface (um, negative inside), thickness profile and texture, on a (H,T,T) grid."""
    c = (tile - 1) / 2.0
    gy, gx = np.mgrid[0:tile, 0:tile].astype(np.float32)
    x, y = (gx - c) / s, (gy - c) / s
    angle = (np.arange(HEADINGS, dtype=np.float32) * math.pi / HEADINGS)[:, None, None]
    ca, sa = np.cos(angle), np.sin(angle)
    xr, yr = x[None] * ca + y[None] * sa, -x[None] * sa + y[None] * ca
    half_straight = max(length_um - width_um, 0.0) / 2.0
    dist = np.hypot(np.maximum(np.abs(xr) - half_straight, 0.0), yr)
    d = dist - width_um / 2.0
    thickness = np.sqrt(np.clip(1.0 - (dist / (width_um / 2.0)) ** 2, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    nucleoid = np.zeros_like(xr)
    granules = np.zeros_like(xr)
    for _ in range(3):                                           # a diffuse nucleoid: a few broad blobs along the axis
        px, py = rng.uniform(-half_straight, half_straight + 1e-6), rng.uniform(-0.12, 0.12)
        nucleoid += np.exp(-(((xr - px) / 0.40) ** 2 + ((yr - py) / 0.17) ** 2))
    for _ in range(5):                                           # storage granules: small dense spots
        px, py = rng.uniform(-half_straight - 0.2, half_straight + 0.2), rng.uniform(-0.2, 0.2)
        granules += np.exp(-(((xr - px) ** 2 + (yr - py) ** 2) / (2 * 0.06 ** 2)))
    return d, thickness, np.clip(nucleoid, 0, 1), np.clip(granules, 0, 1)


def _to_surface(rgb: np.ndarray, alpha: np.ndarray) -> pygame.Surface:
    """(H, T, T, 3) and (H, T, T) uint8 arrays to one horizontal strip surface, heading k at x = k * T."""
    h, t = rgb.shape[0], rgb.shape[1]
    strip = np.concatenate([np.dstack([rgb[k], alpha[k]]) for k in range(h)], axis=1)
    return pygame.image.frombuffer(np.ascontiguousarray(strip, dtype=np.uint8).tobytes(), (t * h, t), "RGBA").convert_alpha() \
        if pygame.display.get_surface() else pygame.image.frombuffer(np.ascontiguousarray(strip, dtype=np.uint8).tobytes(), (t * h, t), "RGBA")


def build_cell_strip(mode: str, px_per_um: float, objective: int, style: int, length_um: float, blur: int) -> tuple[pygame.Surface, int]:
    """One strip of ``HEADINGS`` images of a rod (or spore) of this length, plus the tile size in pixels."""
    spore = style == SPORE_STYLE
    own = STYLE_WIDTH_UM[style] if style < len(STYLE_WIDTH_UM) else WIDTH_UM
    width = SPORE_WIDTH_UM if spore else (length_um if own == 0.0 else own)
    length = SPORE_LENGTH_UM if spore else max(length_um, width)
    sigma_um = math.sqrt(RESOLUTION_SIGMA_UM[objective] ** 2 + (0.0, 0.25, 0.6)[blur] ** 2)
    sigma_px = sigma_um * px_per_um
    margin = 0.9 + 3.0 * sigma_um
    tile = int(math.ceil((length + 2.0 * margin) * px_per_um)) | 1
    d, t, nucleoid, granules = _rod_fields(length, width, tile, px_per_um, seed=style * 101 + int(length * 10))
    coverage = np.clip(0.5 - d * px_per_um, 0.0, 1.0)             # one pixel of anti-aliasing at the edge
    contrast = (1.0, 0.72, 0.5)[blur]
    outside = np.maximum(d, 0.0)
    if mode == "phase":
        if spore:
            body = 1.0 + 0.85 * t ** 0.4
            ring = -0.55 * np.exp(-((outside - 0.03) / 0.07) ** 2)
            intensity = 1.0 + coverage * (body - 1.0) + (1.0 - coverage) * ring
        else:
            body = 1.0 - 0.58 * t ** 0.75 - 0.20 * granules * t + 0.06 * nucleoid * t
            halo = 0.46 * np.exp(-(outside / 0.30) ** 2)
            intensity = 1.0 + coverage * (body - 1.0) + (1.0 - coverage) * halo
        delta = _blur((intensity - 1.0) * contrast, sigma_px)
        bright = delta > 0.0
        alpha = np.clip(np.where(bright, delta * 2.0, -delta), 0.0, 1.0)
        rgb = np.where(bright[..., None], 255, 0).astype(np.uint8).repeat(3, axis=-1)[..., :3] if False else \
            np.repeat(np.where(bright, 255, 0).astype(np.uint8)[..., None], 3, axis=-1)
    elif mode == "brightfield":
        stain = np.asarray(STAIN[style % STYLES], dtype=np.float32)
        if spore:
            a_stain = 0.07 * coverage * t
            a_rim = 0.42 * np.exp(-((outside - 0.02) / 0.08) ** 2)
        else:
            a_stain = 0.80 * coverage * t ** 0.55 * (1.0 - 0.15 * granules)
            a_rim = 0.30 * np.exp(-(outside / 0.09) ** 2) * (1.0 - coverage)
        a_stain = _blur(a_stain * contrast, sigma_px)
        a_rim = _blur(a_rim * contrast, sigma_px)
        alpha = np.clip(1.0 - (1.0 - a_stain) * (1.0 - a_rim), 0.0, 1.0)
        weight = np.where(alpha > 1e-6, a_stain * (1.0 - a_rim) / np.where(alpha > 1e-6, alpha, 1.0), 0.0)
        rgb = np.clip(stain[None, None, None, :] * weight[..., None], 0, 255).astype(np.uint8)
    else:
        colour = np.asarray((SCATTER if mode == "darkfield" else FLUORESCENCE)[style % STYLES], dtype=np.float32)
        if mode == "darkfield":
            light = coverage * (0.30 * t ** 0.6 + 0.70 * np.exp(-(np.maximum(-d, 0.0) / 0.10) ** 2)) + 0.12 * np.exp(-(outside / 0.35))
            light = light * (1.15 if spore else 1.0)
        else:
            light = coverage * t ** 0.9 * 0.95 + 0.05 * np.exp(-(outside / 0.4) ** 2)
            light = light * (0.18 if spore else 1.0)
        light = np.clip(_blur(light * contrast ** 0.5, sigma_px), 0.0, 1.0)
        rgb = np.clip(colour[None, None, None, :] * light[..., None], 0, 255).astype(np.uint8)
        alpha = np.full(light.shape, 255, dtype=np.uint8)
        return _to_surface(rgb, alpha), tile
    return _to_surface(rgb, (alpha * 255.0).astype(np.uint8)), tile


def build_soft_disc(mode: str, px_per_um: float, radius_um: float, strength: float, sigma_um: float) -> pygame.Surface:
    """A faint round halo (matrix, debris glow): white-ish on bright fields, additive on dark ones."""
    radius = max(2, int(math.ceil((radius_um + 3.0 * sigma_um) * px_per_um)))
    size = 2 * radius + 1
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(x - radius, y - radius) / px_per_um
    level = strength * np.clip((radius_um + sigma_um - r) / max(sigma_um, 1e-6), 0.0, 1.0) ** 1.5
    surface = pygame.Surface((size, size), pygame.SRCALPHA, 32)
    if mode in ADDITIVE:
        value = np.clip(level * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([value, value, value], axis=-1)
        alpha = np.full((size, size), 255, dtype=np.uint8)
    else:
        rgb = np.full((size, size, 3), 255 if mode == "phase" else 120, dtype=np.uint8)
        alpha = np.clip(level * 255.0, 0, 255).astype(np.uint8)
    pygame.surfarray.pixels3d(surface)[:] = rgb.transpose(1, 0, 2)
    pygame.surfarray.pixels_alpha(surface)[:] = alpha.T
    return surface


def build_debris(mode: str, px_per_um: float, objective: int, diameter_um: float) -> pygame.Surface:
    """A small round particle (dust, cell debris) lying on the slide, imaged by the same technique."""
    sigma_px = RESOLUTION_SIGMA_UM[objective] * px_per_um
    tile = int(math.ceil((diameter_um + 1.6 + 6.0 * RESOLUTION_SIGMA_UM[objective]) * px_per_um)) | 1
    c = (tile - 1) / 2.0
    y, x = np.mgrid[0:tile, 0:tile].astype(np.float32)
    r = np.hypot(x - c, y - c) / px_per_um
    d = r - diameter_um / 2.0
    cover = np.clip(0.5 - d * px_per_um, 0.0, 1.0)
    t = np.sqrt(np.clip(1.0 - (r / (diameter_um / 2.0)) ** 2, 0.0, 1.0))
    outside = np.maximum(d, 0.0)
    surface = pygame.Surface((tile, tile), pygame.SRCALPHA, 32)
    if mode == "phase":
        delta = _blur((cover * (-0.45 * t ** 0.6) + (1.0 - cover) * 0.35 * np.exp(-(outside / 0.3) ** 2))[None], sigma_px)[0]
        bright = delta > 0
        rgb = np.repeat(np.where(bright, 255, 0).astype(np.uint8)[..., None], 3, axis=-1)
        alpha = np.clip(np.where(bright, delta * 2.0, -delta), 0, 1)
    elif mode == "brightfield":
        a = _blur((0.55 * cover * t ** 0.5)[None], sigma_px)[0]
        rgb = np.full((tile, tile, 3), 70, dtype=np.uint8)
        alpha = np.clip(a, 0, 1)
    else:
        light = np.clip(_blur((cover * t ** 0.8 * (0.55 if mode == "darkfield" else 0.10))[None], sigma_px)[0], 0, 1)
        value = (light * (225 if mode == "darkfield" else 255)).astype(np.uint8)
        rgb = np.stack([value, value, value if mode == "darkfield" else (value * 0.4).astype(np.uint8)], axis=-1)
        alpha = np.full((tile, tile), 1.0, dtype=np.float32)
    pygame.surfarray.pixels3d(surface)[:] = rgb.transpose(1, 0, 2)
    pygame.surfarray.pixels_alpha(surface)[:] = (alpha * 255).astype(np.uint8).T
    return surface


# ------------------------------------------------------------------------------------------------ the whole field
class MicroAtlas:
    """Lazily built strips of cell images for one imaging mode at one objective."""

    def __init__(self, mode: str, objective: int) -> None:
        self.mode = mode
        self.objective = objective
        self.px_per_um = OBJECTIVES[objective][1]
        self._strips: dict[int, tuple[pygame.Surface, int]] = {}
        self._debris: dict[int, pygame.Surface] = {}
        self._matrix: pygame.Surface | None = None

    @staticmethod
    def key(style: np.ndarray, length_bin: np.ndarray, blur: np.ndarray) -> np.ndarray:
        return (np.asarray(style, dtype=np.int64) * len(LENGTH_BINS) + length_bin) * BLUR_LEVELS + blur

    def strip(self, key: int) -> tuple[pygame.Surface, int]:
        found = self._strips.get(key)
        if found is None:
            blur = key % BLUR_LEVELS
            length_bin = (key // BLUR_LEVELS) % len(LENGTH_BINS)
            style = key // (BLUR_LEVELS * len(LENGTH_BINS))
            found = self._strips[key] = build_cell_strip(self.mode, self.px_per_um, self.objective, style,
                                                         float(LENGTH_BINS[length_bin]), blur)
        return found

    def debris(self, size_bin: int) -> pygame.Surface:
        found = self._debris.get(size_bin)
        if found is None:
            found = self._debris[size_bin] = build_debris(self.mode, self.px_per_um, self.objective, (0.3, 0.55, 0.95, 1.6)[size_bin])
        return found

    def matrix(self) -> pygame.Surface:
        if self._matrix is None:
            strength = {"phase": 0.10, "brightfield": 0.0, "darkfield": 0.07, "fluorescence": 0.0}[self.mode]
            self._matrix = build_soft_disc(self.mode, self.px_per_um, 1.7, strength, 0.8)
        return self._matrix


def build_field(mode: str, size: tuple[int, int]) -> pygame.Surface:
    """The empty field: uneven illumination, a vignette, and the faint rings of dust on the optics (fixed to the screen)."""
    width, height = size
    rng = np.random.default_rng(31)
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[:, None]
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[None, :]
    radial = np.sqrt(0.5 * (xs * xs + ys * ys))
    light = 1.0 - VIGNETTE[mode] * np.clip(radial, 0.0, 1.0) ** 2.2
    light = light * (1.0 + 0.025 * (0.6 * xs - 0.4 * ys))              # one side of the lamp slightly brighter
    cloud = pygame.Surface((9, 6))
    for ix in range(9):
        for iy in range(6):
            v = int(rng.integers(0, 256))
            cloud.set_at((ix, iy), (v, v, v))
    for factor in (4, 2, 1):
        cloud = pygame.transform.smoothscale(cloud, (max(2, width // factor), max(2, height // factor)))
    unevenness = pygame.surfarray.array3d(cloud)[..., 0].astype(np.float32) / 255.0 - 0.5
    scale = 1.0 if mode in ("phase", "brightfield") else 0.0
    light = light * (1.0 + 0.045 * unevenness * scale)
    for _ in range(6):                                                  # out-of-focus dust on the sensor: faint donuts
        cx, cy, r = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.02, 0.07)
        dist = np.hypot((xs - cx) * width / height, ys - cy)
        ring = np.exp(-(((dist - r) / (0.18 * r)) ** 2)) * 0.5 + np.clip((r - dist) / r, 0, 1) * 0.12
        light = light * (1.0 - (0.05 if scale else 0.0) * ring)
    base = np.asarray(BACKGROUND[mode], dtype=np.float32)
    mean_noise = NOISE_SIGMA[mode] * 0.8 if mode in ("phase", "brightfield") else 0.0   # the animated noise is added on top
    rgb = np.clip(base[None, None, :] * light[..., None] - mean_noise, 0, 255).astype(np.uint8)
    return pygame.surfarray.make_surface(rgb)


def build_noise(mode: str, size: tuple[int, int], count: int = 4) -> list[pygame.Surface]:
    """A few frames of sensor noise to add to the image in turn: it never repeats in an obvious way and costs one blit."""
    width, height = size
    sigma = NOISE_SIGMA[mode]
    surfaces = []
    for index in range(count):
        rng = np.random.default_rng(1000 + index)
        n = np.clip(rng.normal(sigma * 0.8, sigma, (width, height, 1)), 0, 40)
        surfaces.append(pygame.surfarray.make_surface(np.repeat(n, 3, axis=2).astype(np.uint8)))
    return surfaces


def build_eyepiece(size: tuple[int, int]) -> pygame.Surface:
    """The black surround of an eyepiece: a circular field stop with a slightly soft edge."""
    width, height = size
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = 0.5 * min(width, height) * 0.985
    edge = np.clip((np.hypot(xs - width / 2.0, ys - height / 2.0) - radius) / 3.0, 0.0, 1.0)
    surface = pygame.Surface(size, pygame.SRCALPHA, 32)
    pygame.surfarray.pixels_alpha(surface)[:] = (edge * 255).astype(np.uint8).T
    pygame.surfarray.pixels3d(surface)[:] = 0
    return surface
