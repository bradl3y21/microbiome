"""The microscope view: true scale, true rod orientation, the physics of each imaging mode, focus and the controls."""
import dataclasses
import re
from pathlib import Path

import numpy as np
import pygame
import pytest

from microbiome.render import microscope as M
from microbiome.render.lens import MicroscopeRenderer
from microbiome.sim.controller import SimulationController
from microbiome.sim.realism import RealismConfig, rod_length_um
from microbiome.ui.app import MicrobiomeUI, dispatch_command

ROOT = Path(__file__).resolve().parent.parent
SIZE = (1280, 720)
EC, BS = "GCF_000005845.2", "GCF_000009045.1"


@pytest.fixture(scope="module", autouse=True)
def pygame_up():
    pygame.init()
    pygame.display.set_mode((16, 16))
    yield
    pygame.quit()


def world(n_ec=1, n_bs=0, seed=3):
    c = SimulationController.create_demo(seed=seed, realism=True)
    ids = {o.accession_version: int(o.genome_id) for o in c.list_genomes()}
    c.advance_ticks(1)
    for i in range(n_ec):
        c.introduce_organism(ids[EC], (400.0 + 7 * i, 300.0))
    for i in range(n_bs):
        c.introduce_organism(ids[BS], (400.0 + 7 * i, 320.0))
    c.advance_ticks(1)
    return c, ids


def with_heading(c, heading):
    c._realism.simulation.life.heading[:] = heading
    return c.render_snapshot()


def shoot(renderer, snapshot, centre=None, label=False, focus_on=0):
    """One frame centred on a cell; the focal plane follows cell ``focus_on`` unless it is None (depth is a hash of the id)."""
    renderer.show_label = label
    if focus_on is not None:
        renderer.focus_um = float(M.pseudo_depth_um(snapshot.agent_id)[focus_on])
    renderer.camera.set_field_size((snapshot.width_um, snapshot.height_um))
    renderer.camera.center_um[:] = snapshot.position_um[0] if centre is None else centre
    surface = pygame.Surface(SIZE)
    renderer.draw(surface, snapshot)
    return pygame.surfarray.array3d(surface).astype(np.int32)


def far_away(snapshot):
    return dataclasses.replace(snapshot, position_um=snapshot.position_um + np.array((300.0, 0.0)))


def difference(renderer, snapshot, focus_on=0):
    return (shoot(renderer, snapshot, focus_on=focus_on)
            - shoot(renderer, far_away(snapshot), centre=snapshot.position_um[0], focus_on=focus_on))


def extent(mask):
    xs, ys = np.nonzero(mask)
    return (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) if xs.size else (0, 0)


# ------------------------------------------------------------------------------------------ the optics, by number
def test_objectives_are_real_magnifications_and_the_resolution_improves_with_them():
    assert [m for m, _ in M.OBJECTIVES] == [10, 20, 40, 60, 100]
    scale = [p for _, p in M.OBJECTIVES]
    assert scale == sorted(scale) and scale[0] == pytest.approx(1.6) and scale[-1] == pytest.approx(16.0)
    assert list(M.RESOLUTION_SIGMA_UM) == sorted(M.RESOLUTION_SIGMA_UM, reverse=True)
    assert list(M.DEPTH_OF_FIELD_UM) == sorted(M.DEPTH_OF_FIELD_UM, reverse=True)
    assert 0.15 < 2.355 * M.RESOLUTION_SIGMA_UM[-1] < 0.3, "a 100x oil objective resolves about 0.2 um"


def test_the_scale_bar_is_a_one_two_five_length_that_fits_and_its_pixels_match_the_scale():
    for ppu in (1.6, 3.2, 6.4, 9.6, 16.0):
        um = M.nice_scale_bar_um(ppu, 280.0)
        assert um * ppu <= 280.0 and um * ppu > 280.0 / 2.5, (ppu, um)
        mantissa = um / 10 ** np.floor(np.log10(um))
        assert round(mantissa, 6) in (1.0, 2.0, 5.0)
    r = MicroscopeRenderer(objective=2)
    r.camera.resize(SIZE)
    um, px = r.scale_bar()
    assert px == round(um * 6.4)


def test_the_stain_colours_gram_negative_pink_gram_positive_purple():
    pink, purple = (np.array(M.STAIN[i], dtype=float) for i in (0, 1))
    assert pink[0] > pink[2] > pink[1], "pink/red: red strongest, green weakest"
    assert purple[2] > purple[0] > purple[1], "purple: blue strongest"


def test_pseudo_depth_is_a_fixed_function_of_the_id_and_fills_the_film():
    ids = np.arange(1, 5000, dtype=np.uint64)
    d = M.pseudo_depth_um(ids)
    assert np.array_equal(d, M.pseudo_depth_um(ids)) and d.min() >= 0 and d.max() <= M.FILM_DEPTH_UM
    assert np.histogram(d, bins=5, range=(0, M.FILM_DEPTH_UM))[0].min() > 800, "roughly uniform through the film"


def test_the_geometry_constants_agree_with_the_simulation_so_a_plain_snapshot_still_draws_true_rods():
    cfg = RealismConfig()
    assert M.WIDTH_UM == cfg.cell_width_um and M.VOLUME_UM3_PER_PG == cfg.cell_volume_um3_per_pg
    mass = np.array([0.1, 0.3, 0.45, 0.6])
    radius = np.cbrt(3 * mass / (4 * np.pi))                # the engine's sphere-equivalent radius convention
    assert M.rod_length_from_radius(radius) == pytest.approx(rod_length_um(mass, cfg))


def test_the_renderer_never_imports_the_simulation():
    for name in ("lens.py", "microscope.py"):
        text = (ROOT / "src" / "microbiome" / "render" / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+(microbiome\.sim|\.\.sim)", text, flags=re.M), name


# ------------------------------------------------------------------------------------------ cells at true scale
def test_a_rod_is_drawn_at_true_length_and_width_and_turns_with_its_heading():
    c, _ = world()
    r = MicroscopeRenderer(objective=2)                                 # 40x: 6.4 px/um
    dark = difference(r, with_heading(c, 0.0)).sum(axis=2) < -60
    length_px = float(rod_length_um(c._state.agents.mass_pg, c._realism.simulation.config)[0]) * 6.4
    w, h = extent(dark)
    assert length_px - 5 <= w <= length_px + 2, f"dark body {w} px vs {length_px:.1f} px of true length"
    assert 2 <= h <= 0.8 * 6.4 + 2, f"dark body {h} px thick vs {0.8 * 6.4:.1f} px of true width"
    w2, h2 = extent(difference(r, with_heading(c, np.pi / 2)).sum(axis=2) < -60)
    assert abs(w2 - h) <= 1 and abs(h2 - w) <= 1, "a quarter turn swaps the extents"
    wd, hd = extent(difference(r, with_heading(c, np.pi / 4)).sum(axis=2) < -60)
    assert abs(wd - hd) <= 2 and wd < w, "at 45 degrees the box is square and shorter than the rod"


def test_a_rod_looks_the_same_turned_by_half_a_turn():
    c, _ = world()
    r = MicroscopeRenderer(objective=2)
    assert np.array_equal(shoot(r, with_heading(c, 0.3)), shoot(r, with_heading(c, 0.3 + np.pi)))


def test_magnification_scales_the_image_linearly():
    c, _ = world()
    snap = with_heading(c, 0.0)
    widths = []
    for objective in (2, 3, 4):
        d = difference(MicroscopeRenderer(objective=objective), snap)
        widths.append(extent(d.sum(axis=2) < -60)[0] / M.OBJECTIVES[objective][1])
    assert max(widths) - min(widths) < 0.5, f"cell length in um is the same at every magnification: {widths}"


def test_a_growing_cell_gets_longer_on_screen():
    c, _ = world()
    r = MicroscopeRenderer(objective=3)
    before = extent(difference(r, with_heading(c, 0.0)).sum(axis=2) < -60)[0]
    c._state.agents.mass_pg[:] = 0.55
    after = extent(difference(r, with_heading(c, 0.0)).sum(axis=2) < -60)[0]
    expected_gain = (float(rod_length_um(np.array([0.55]), RealismConfig())[0]) - float(rod_length_um(np.array([0.3]), RealismConfig())[0])) * 9.6
    assert after - before == pytest.approx(expected_gain, abs=3.0), (before, after, expected_gain)


# ------------------------------------------------------------------------------------------ the four techniques
def test_phase_contrast_cells_are_dark_with_a_bright_halo_and_spores_are_bright():
    c, _ = world()
    d = difference(MicroscopeRenderer(objective=3, mode="phase"), with_heading(c, 0.0)).sum(axis=2)
    assert d.min() < -150, "the cytoplasm is darker than the field"
    assert d.max() > 40, "ringed by the bright halo"
    c._realism.simulation.life.spore[:] = 1.0
    s = difference(MicroscopeRenderer(objective=3, mode="phase"), c.render_snapshot()).sum(axis=2)
    assert s.max() > 150 and s.max() > -s.min(), "an endospore is phase-bright"


def test_brightfield_gram_stain_colours_the_two_organisms_differently():
    c, ids = world(n_ec=1, n_bs=1)
    snap = c.render_snapshot()
    r = MicroscopeRenderer(objective=3, mode="brightfield")
    colours = {}
    for acc in (EC, BS):
        i = int(np.flatnonzero(snap.genome_id == np.uint64(ids[acc]))[0])
        image = shoot(r, snap, centre=snap.position_um[i], focus_on=i)
        base = shoot(r, far_away(snap), centre=snap.position_um[i], focus_on=i)
        x, y = np.unravel_index(np.argmin((image - base).sum(axis=2)), image.shape[:2])
        colours[acc] = image[x, y]
    pink, purple = colours[EC], colours[BS]
    assert pink[0] > pink[2], f"E. coli stains pink/red {pink}"
    assert purple[2] > purple[0], f"B. subtilis stains purple/blue {purple}"
    assert pink.sum() < 640 and purple.sum() < 640, "stained cells are darker than the pale field"


def test_darkfield_and_fluorescence_are_light_on_black_with_the_organisms_in_different_colours():
    c, ids = world(n_ec=1, n_bs=1)
    snap = c.render_snapshot()
    for mode in ("darkfield", "fluorescence"):
        image = shoot(MicroscopeRenderer(objective=3, mode=mode), snap, centre=snap.position_um[0], focus_on=0)
        assert np.median(image.sum(axis=2)) < 60, f"{mode}: the field is dark"
        assert image.sum(axis=2).max() > 250, f"{mode}: the cells are bright"
    r = MicroscopeRenderer(objective=3, mode="fluorescence")
    seen = {}
    for acc in (EC, BS):
        i = int(np.flatnonzero(snap.genome_id == np.uint64(ids[acc]))[0])
        img = shoot(r, snap, centre=snap.position_um[i], focus_on=i)
        x, y = np.unravel_index(np.argmax(img.sum(axis=2)), img.shape[:2])
        seen[acc] = img[x, y]
    assert seen[EC][1] > seen[EC][0] and seen[BS][0] > seen[BS][1], f"green and red: {seen}"


def test_additive_modes_never_darken_and_the_phase_field_is_a_mid_grey():
    c, _ = world()
    snap = with_heading(c, 0.0)
    assert (difference(MicroscopeRenderer(objective=2, mode="darkfield"), snap) >= 0).all()
    empty = shoot(MicroscopeRenderer(objective=2, mode="phase"), far_away(snap), centre=snap.position_um[0]).astype(float)
    assert 140 < empty.mean() < 185


# ------------------------------------------------------------------------------------------ focus
def test_cells_lose_contrast_away_from_the_focal_plane_and_return_when_focus_follows():
    c, _ = world()
    snap = with_heading(c, 0.0)
    depth = float(M.pseudo_depth_um(snap.agent_id)[0])
    r = MicroscopeRenderer(objective=4)
    sharp = difference(r, snap).sum(axis=2)
    r.focus_um = depth + 5.0 if depth < 5.0 else depth - 5.0
    blurred = (shoot(r, snap, focus_on=None)
               - shoot(r, far_away(snap), centre=snap.position_um[0], focus_on=None)).sum(axis=2)
    assert blurred.min() > 0.7 * sharp.min(), f"out of focus is fainter: {blurred.min()} vs {sharp.min()}"
    assert (blurred != sharp).any()
    assert np.array_equal(sharp, difference(r, snap).sum(axis=2)), "refocusing restores the same image"


def test_the_depth_of_field_shrinks_with_magnification():
    assert M.DEPTH_OF_FIELD_UM[0] > 4 * M.DEPTH_OF_FIELD_UM[-1]


def test_focus_is_clamped_to_the_film():
    r = MicroscopeRenderer()
    for _ in range(40):
        r.nudge_focus(1.0)
    assert r.focus_um == M.FILM_DEPTH_UM
    for _ in range(40):
        r.nudge_focus(-1.0)
    assert r.focus_um == 0.0


# ------------------------------------------------------------------------------------------ the sample and the controls
def test_the_sample_is_periodic_so_a_cell_beyond_the_edge_appears_on_the_other_side():
    c, _ = world()
    snap = dataclasses.replace(with_heading(c, 0.0), position_um=np.array([[799.0, 300.0]]))
    r = MicroscopeRenderer(objective=2)
    image = shoot(r, snap, centre=(2.0, 300.0))
    baseline = shoot(r, far_away(snap), centre=(2.0, 300.0))
    x, y = np.unravel_index(np.argmin((image - baseline).sum(axis=2)), image.shape[:2])
    expected = 640 + (799.0 - 800.0 - 2.0) * 6.4
    assert abs(x - expected) < 10 and abs(y - 360) < 5, f"found the cell at {x},{y}; expected near {expected:.0f},360"


def test_panning_wraps_the_camera_into_the_field():
    c, _ = world()
    snap = c.render_snapshot()
    r = MicroscopeRenderer(objective=2)
    r.camera.set_field_size((snap.width_um, snap.height_um))
    r.camera.center_um[:] = (-30.0, 1000.0)
    r.draw(pygame.Surface(SIZE), snap)
    assert tuple(r.camera.center_um) == pytest.approx((770.0, 360.0))


def test_the_wheel_selects_objectives_not_arbitrary_zooms_and_keeps_the_point_under_the_pointer():
    r = MicroscopeRenderer(objective=1)
    r.camera.set_field_size((800.0, 640.0))
    r.camera.resize(SIZE)
    anchor = (900, 200)
    before = np.array(r.camera.screen_to_world(anchor))
    assert r.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=1, x=0), anchor)
    assert r.objective == 2 and r.camera.zoom == pytest.approx(6.4)
    assert np.allclose(r.camera.screen_to_world(anchor), before)
    for _ in range(10):
        r.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=1, x=0), anchor)
    assert r.objective == 4 and r.camera.zoom == pytest.approx(16.0), "stops at the highest power"
    for _ in range(10):
        r.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=-1, x=0), anchor)
    assert r.objective == 0 and r.camera.zoom == pytest.approx(1.6)


def test_modes_cycle_through_all_four_and_back():
    r = MicroscopeRenderer()
    seen = [r.mode] + [r.cycle_mode() for _ in range(4)]
    assert seen[0] == seen[-1] == "phase" and set(seen) == set(M.MODES)


def test_the_label_names_the_mode_the_magnification_and_the_focus_and_the_scale_bar_is_drawn():
    c, _ = world()
    snap = c.render_snapshot()
    r = MicroscopeRenderer(objective=3, mode="brightfield")
    assert r.label() == "Brightfield, Gram stain   60x   focus +0.0 um"
    with_text = shoot(r, snap, label=True)
    without = shoot(r, snap)
    assert (with_text != without).any(axis=2).sum() > 200


def test_the_picture_is_deterministic():
    c, _ = world(n_ec=20, n_bs=20)
    snap = c.render_snapshot()
    assert np.array_equal(shoot(MicroscopeRenderer(objective=2), snap), shoot(MicroscopeRenderer(objective=2), snap))


def test_an_unknown_mode_or_objective_is_rejected():
    with pytest.raises(ValueError):
        MicroscopeRenderer(mode="x-ray")
    with pytest.raises(ValueError):
        MicroscopeRenderer(objective=9)


def test_the_eyepiece_mask_blacks_the_corners_and_leaves_the_centre():
    c, _ = world()
    snap = c.render_snapshot()
    r = MicroscopeRenderer(objective=2)
    r.show_label = False
    r.toggle_eyepiece()
    image = shoot(r, snap)
    assert image[:6, :6].sum() == 0 and image[-6:, -6:].sum() == 0
    assert image[640, 360].sum() > 100


def test_a_snapshot_without_realism_data_still_draws_rods_in_the_microscope():
    c = SimulationController.create_demo(seed=3)
    g = int(c.list_genomes()[0].genome_id)
    c.advance_ticks(1)
    c.introduce_organism(g, (400.0, 300.0))
    c.advance_ticks(2)
    snap = c.render_snapshot()
    r = MicroscopeRenderer(objective=3)
    assert (difference(r, snap).sum(axis=2) < -60).any()


# ------------------------------------------------------------------------------------------ the application around it
def microscope_ui(n=6):
    ui = MicrobiomeUI.from_demo(seed=5, realism=True, microscope=True, window_size=SIZE, project_root=ROOT)
    g = [int(o.genome_id) for o in ui.genomes]
    dispatch_command(ui.controller, "advance_ticks", count=1)
    for i in range(n):
        dispatch_command(ui.controller, "introduce_organism", genome_id=g[i % 2], position_um=(400.0 + 8 * i, 300.0))
    dispatch_command(ui.controller, "advance_ticks", count=3)
    ui.renderer.camera.set_field_size((800.0, 640.0))
    ui.renderer.camera.center_um[:] = (420.0, 300.0)
    ui.draw(pygame.Surface(SIZE))
    return ui


def key(ui, k):
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k))


def test_the_default_window_is_only_the_sample_with_no_game_interface():
    ui = microscope_ui()
    assert isinstance(ui.renderer, MicroscopeRenderer) and ui.minimal_hud
    ui.draw(pygame.Surface(SIZE))
    assert ui._buttons == [] and ui._topbar_rect.size == (0, 0) and ui._panel_rect.size == (0, 0)


def test_tab_brings_the_instrument_panel_back_and_hides_it_again():
    ui = microscope_ui()
    key(ui, pygame.K_TAB)
    ui.draw(pygame.Surface(SIZE))
    assert not ui.minimal_hud and ui._buttons and ui._topbar_rect.width > 0
    key(ui, pygame.K_TAB)
    ui.draw(pygame.Surface(SIZE))
    assert ui.minimal_hud and ui._buttons == []


def test_microscope_keys_change_mode_focus_and_eyepiece():
    ui = microscope_ui()
    modes = []
    for _ in range(4):
        key(ui, pygame.K_b)
        modes.append(ui.renderer.mode)
    assert modes[-1] == "phase" and len(set(modes)) == 4
    start = ui.renderer.focus_um
    key(ui, pygame.K_x)
    key(ui, pygame.K_x)
    key(ui, pygame.K_z)
    assert ui.renderer.focus_um == pytest.approx(start + 0.5)
    key(ui, pygame.K_e)
    assert ui.renderer.eyepiece
    ui.draw(pygame.Surface(SIZE))


def test_the_help_panel_lists_the_microscope_controls_and_opens_in_minimal_mode():
    ui = microscope_ui()
    key(ui, pygame.K_h)
    ui.draw(pygame.Surface(SIZE))
    text = "\n".join(ui._panel_lines())
    for fragment in ("objective", "imaging mode", "Z / X", "eyepiece", "Tab"):
        assert fragment in text, fragment
    assert ui._panel_rect.width > 0
    key(ui, pygame.K_ESCAPE)
    ui.draw(pygame.Surface(SIZE))
    assert ui._panel_rect.size == (0, 0)


def test_a_notice_is_shown_briefly_and_then_leaves_the_picture_clean(monkeypatch):
    import microbiome.ui.app as app

    ui = microscope_ui()
    clock = [100.0]
    monkeypatch.setattr(app.time, "monotonic", lambda: clock[0])
    ui.notice = "Hello"
    shown = pygame.Surface(SIZE)
    ui.draw(shown)
    clock[0] += ui.NOTICE_SECONDS + 1
    gone = pygame.Surface(SIZE)
    ui.draw(gone)
    assert pygame.image.tobytes(shown, "RGB") != pygame.image.tobytes(gone, "RGB")


def test_clicking_a_cell_across_the_seam_inspects_it_and_a_click_on_the_slide_wraps():
    ui = microscope_ui(2)
    snap = ui.controller.render_snapshot()
    ui.renderer.camera.center_um[:] = (2.0, float(snap.position_um[0, 1]))
    ui.draw(pygame.Surface(SIZE))
    world_point = ui._world_position((640, 360))
    assert world_point is not None and 0 <= world_point[0] < 800 and abs(world_point[0] - 2.0) < 1e-6
    seam = ui._world_position((0, 360))             # half a window left of the centre: beyond the edge, so the far side
    assert seam is not None and seam[0] == pytest.approx((2.0 - 640.0 / ui.renderer.px_per_um) % 800.0), seam
    ui.renderer.camera.center_um[:] = (snap.position_um[0, 0] + 800.0, snap.position_um[0, 1])
    ui._inspect_at((640, 360))
    assert ui.selected_inspection is not None, "the camera sits over a neighbouring copy of the sample"
