"""The SPORE-style look is presentation only, but it still has contracts worth pinning.

  * a frame is a pure function of (snapshot, camera): same input, same pixels, on a fresh renderer
  * a cell faces the way it swims (heading comes from frame-to-frame motion, with periodic wrap)
  * different genomes get different palettes, so species are tellable apart
  * the sprite pixel budget bounds the per-frame blit area however many cells are visible, which is what
    keeps the 10,000-agent / 60 fps target (README) from regressing when sprites get prettier
"""
import numpy as np
import pygame
import pytest

from microbiome.render.atlas import SpriteAtlas
from microbiome.render.renderer import PygameRenderer
from microbiome.render.view import WorldView

W, H = 200.0, 150.0


def view(positions, ids=None, tick=0, colours=None, motile=None, radius=0.4):
    n = len(positions)
    return WorldView(
        tick_index=tick, width_um=W, height_um=H,
        position_um=np.asarray(positions, dtype=np.float64),
        radius_um=np.full(n, radius),
        colour_index=np.zeros(n, np.uint8) if colours is None else np.asarray(colours, np.uint8),
        motile=np.ones(n, bool) if motile is None else np.asarray(motile, bool),
        field_density=np.full((6, 8), 1e-3),
        agent_id=np.arange(1, n + 1, dtype=np.uint64) if ids is None else np.asarray(ids, np.uint64),
    )


@pytest.fixture(scope="module", autouse=True)
def pygame_up():
    pygame.init()
    yield
    pygame.quit()


def test_heading_follows_motion_including_across_the_periodic_edge():
    r = PygameRenderer()
    r.camera.set_field_size((W, H))
    heading = None
    for step in range(30):                                   # moves +y at 0.2 um per frame
        heading = r._track_headings(view([[50.0, 20.0 + 0.2 * step]], tick=step))
    assert heading[0] == pytest.approx(np.pi / 2, abs=0.05)
    r2 = PygameRenderer()
    for step in range(30):                                   # +x across x = W: 199.7 -> wraps to 0.1
        x = (W - 3.0 + 0.2 * step) % W
        heading = r2._track_headings(view([[x, 70.0]], tick=step))
    assert abs(np.arctan2(np.sin(heading[0]), np.cos(heading[0]))) < 0.05, "a wrap must not read as a jump"


def test_a_resting_cell_keeps_its_heading():
    r = PygameRenderer()
    first = r._track_headings(view([[10.0, 10.0]], tick=0)).copy()
    assert np.array_equal(r._track_headings(view([[10.0, 10.0]], tick=1)), first)


def test_a_frame_is_a_pure_function_of_the_snapshot():
    rng = np.random.default_rng(3)
    pos = rng.random((300, 2)) * [W, H]
    colours = rng.integers(0, 2, 300)
    pixels = []
    for _ in range(2):
        screen = pygame.Surface((640, 480))
        r = PygameRenderer()
        r.draw(screen, _snapshot(view(pos, colours=colours, tick=40)))
        pixels.append(pygame.image.tobytes(screen, "RGB"))
    assert pixels[0] == pixels[1]


def test_different_genomes_get_visibly_different_sprites():
    r = PygameRenderer()
    sprites = []
    for palette in (0, 1):
        commands = r._cell_blits(np.array([[50.0, 50.0]]), np.array([0.4 * 4.0]), np.array([palette]), np.array([True]))
        surface, _, area = commands[0]
        sprites.append(pygame.image.tobytes(surface.subsurface(area), "RGBA"))
    assert sprites[0] != sprites[1]


def test_sprite_blit_area_is_bounded_for_any_visible_population():
    r = PygameRenderer()
    rng = np.random.default_rng(5)
    for n in (50, 600, 10_000):
        screen = rng.random((n, 2)) * [1280.0, 720.0]
        commands = r._cell_blits(screen, np.full(n, 0.415 * 3.0), rng.integers(0, 2, n), rng.random(n) > 0.5)
        area = sum(a[2] * a[3] for _, _, a in commands)
        assert area <= 1.6 * r.SPRITE_PIXEL_BUDGET, f"{n} cells blit {area} px"


def test_atlas_crops_are_inside_their_tiles_and_nonempty():
    atlas = SpriteAtlas()
    atlas.ensure(0, 4)
    tile = int(atlas.tile_size[0, 4])
    boxes = atlas.crop[0, 4]
    assert (boxes[:, 2] > 0).all() and (boxes[:, 3] > 0).all()
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()
    assert (boxes[:, 0] + boxes[:, 2] <= tile).all() and (boxes[:, 1] + boxes[:, 3] <= tile).all()


class _Snap:
    """Minimal D3 snapshot built from a WorldView-shaped fixture."""


def _snapshot(v):
    from microbiome.core.ui_api import RenderSnapshot

    n = len(v.position_um)
    ids = np.array([7, 9], dtype=np.uint64)
    fields = {f: None for f in RenderSnapshot.__dataclass_fields__}
    fields.update(
        tick_index=np.uint64(v.tick_index), width_um=v.width_um, height_um=v.height_um,
        agent_id=np.arange(1, n + 1, dtype=np.uint64), position_um=v.position_um, radius_um=v.radius_um,
        genome_id=ids[v.colour_index.astype(int) % 2],
        nutrient_id=np.array([0], dtype=np.uint32),
        concentration_mmol_l=v.field_density[None, :, :].copy(),
        phenotype_genome_id=ids, phenotype_motility_um_s=np.array([10.0, 10.0]),
        phenotype_max_uptake_mmol_s=np.ones((2, 1)),
    )
    for name in fields:
        if isinstance(fields[name], np.ndarray):
            fields[name] = np.ascontiguousarray(fields[name])
            fields[name].setflags(write=False)
    return RenderSnapshot(**fields)


def test_a_species_is_baked_at_every_size_the_first_time_it_is_seen():
    """Zooming must never stall on a first-use sprite bake: drawing one frame bakes all sizes of each species."""
    screen = pygame.Surface((320, 240))
    r = PygameRenderer()
    assert not r.atlas.tile_size.any()
    r.draw(screen, _snapshot(view([[50.0, 50.0], [60.0, 60.0]], colours=[0, 1], tick=0)))
    assert (r.atlas.tile_size[0] > 0).all() and (r.atlas.tile_size[1] > 0).all()
    assert not r.atlas.tile_size[2:].any(), "species that are not on screen are not baked"
