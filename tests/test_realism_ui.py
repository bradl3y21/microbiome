"""The realism layer as the user sees it: overlays, spores, biofilm halos, the V key and the cell physiology readout."""
import dataclasses
from pathlib import Path

import numpy as np
import pygame
import pytest

from microbiome.core.ui_api import RealismSnapshot
from microbiome.render.renderer import OVERLAYS, PygameRenderer
from microbiome.render.view import WorldView
from microbiome.sim.controller import SimulationController
from microbiome.ui import app as app_module
from microbiome.ui.app import MicrobiomeUI

ROOT = Path(__file__).resolve().parent.parent
SIZE = (1280, 720)


@pytest.fixture(scope="module", autouse=True)
def pygame_up():
    pygame.init()
    yield
    pygame.quit()


def realistic(n=6, realism=True):
    c = SimulationController.create_demo(seed=8, realism=realism)
    g = [int(o.genome_id) for o in c.list_genomes()]
    c.advance_ticks(1)
    for i in range(n):
        c.introduce_organism(g[i % 2], (300.0 + 12 * i, 300.0))
    c.place_nutrient(0, (320.0, 300.0), 4e-12)
    c.advance_ticks(3)
    return c


def with_states(snapshot, states, eps=None):
    return dataclasses.replace(snapshot, cell_state=np.asarray(states, dtype=np.uint8),
                               eps=np.zeros(len(states)) if eps is None else np.asarray(eps, dtype=float))


def test_each_overlay_selects_its_own_chemistry_and_nutrients_exclude_oxygen():
    snap = realistic().render_snapshot()
    conc = snap.concentration_mmol_l
    names = list(snap.channel_names)
    pick = lambda *ns: sum(conc[names.index(n)] for n in ns)
    assert np.allclose(WorldView.from_snapshot(snap, "nutrients").field_density, pick("glucose", "lactose", "acetate"))
    assert np.allclose(WorldView.from_snapshot(snap, "oxygen").field_density, pick("oxygen"))
    assert np.allclose(WorldView.from_snapshot(snap, "acidity").field_density, pick("acetate"))
    assert np.allclose(WorldView.from_snapshot(snap, "toxin").field_density, pick("bacilysin"))
    assert np.allclose(WorldView.from_snapshot(snap, "signal").field_density, pick("ai2"))
    assert conc[names.index("oxygen")].min() > 0.2, "oxygen is present everywhere"
    assert WorldView.from_snapshot(snap, "nutrients").field_density.min() == 0.0, \
        "so the nutrient view must not include it: far from the paint it is exactly zero"


def test_the_plain_two_channel_snapshot_still_sums_every_channel_as_before():
    snap = realistic(realism=False).render_snapshot()
    assert not isinstance(snap, RealismSnapshot)
    for overlay in OVERLAYS:
        assert np.allclose(WorldView.from_snapshot(snap, overlay).field_density, snap.concentration_mmol_l.sum(axis=0))
    assert WorldView.from_snapshot(snap).cell_state is None


def test_a_dormant_spore_gets_the_spore_sprite_and_stops_swimming():
    snap = realistic(4).render_snapshot()
    plain = WorldView.from_snapshot(snap)
    view = WorldView.from_snapshot(with_states(snap, [0, 2, 1, 3]))
    spore = PygameRenderer().atlas.SPORE_STYLE
    assert view.colour_index[1] == spore and not view.motile[1]
    assert (view.colour_index[[0, 2, 3]] == plain.colour_index[[0, 2, 3]]).all() and (view.motile[[0, 2, 3]] == plain.motile[[0, 2, 3]]).all()


def draw(snapshot, overlay="nutrients", renderer=None):
    renderer = renderer or PygameRenderer()
    renderer.overlay = overlay
    surface = pygame.Surface(SIZE)
    renderer.draw(surface, snapshot)
    return renderer, pygame.image.tobytes(surface, "RGB")


def test_spores_and_matrix_halos_change_the_picture_and_only_then():
    snap = realistic(6).render_snapshot()
    _, base = draw(with_states(snap, [0] * 6))
    _, spores = draw(with_states(snap, [2, 2, 0, 0, 0, 0]))
    _, halos = draw(with_states(snap, [3, 3, 3, 0, 0, 0], eps=[1, 1, 1, 0, 0, 0]))
    assert len({base, spores, halos}) == 3
    renderer, _ = draw(with_states(snap, [0] * 6))
    assert not renderer.atlas.tile_size[renderer.atlas.SPORE_STYLE].any(), "the spore sprite is only baked when a spore is on screen"
    renderer, _ = draw(with_states(snap, [2] * 6))
    assert renderer.atlas.tile_size[renderer.atlas.SPORE_STYLE].all()


def test_every_overlay_draws_and_they_look_different_from_each_other():
    c = realistic(6)
    c.place_nutrient(4, (300.0, 300.0), 1e-14)
    c.place_nutrient(5, (310.0, 300.0), 1e-14)
    c.advance_ticks(5)
    snap = c.render_snapshot()
    renderer = PygameRenderer()
    images = {name: draw(snap, name, renderer)[1] for name in OVERLAYS}
    assert len(set(images.values())) == len(OVERLAYS)


def realistic_ui():
    ui = MicrobiomeUI(controller=realistic(6), window_size=SIZE, project_root=ROOT)
    ui.draw(pygame.Surface(SIZE))
    return ui


def key(ui, k):
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k))


def test_the_v_key_cycles_the_overlay_in_a_realism_world_and_explains_itself_otherwise():
    ui = realistic_ui()
    assert ui.realism and ui.renderer.overlay == "nutrients"
    seen = []
    for _ in range(len(OVERLAYS)):
        key(ui, pygame.K_v)
        seen.append(ui.renderer.overlay)
    assert seen == list(OVERLAYS[1:]) + [OVERLAYS[0]]
    assert "Field overlay" in ui.notice
    plain = MicrobiomeUI(controller=SimulationController.create_demo(seed=2), window_size=SIZE, project_root=ROOT)
    key(plain, pygame.K_v)
    assert plain.renderer.overlay == "nutrients" and "realism" in plain.notice


def test_the_toolbar_offers_a_view_button_only_in_a_realism_world():
    ui = realistic_ui()
    ui.draw(pygame.Surface(SIZE))
    assert any(k == pygame.K_v for k, _ in ui._buttons)
    plain = MicrobiomeUI(controller=SimulationController.create_demo(seed=2), window_size=SIZE, project_root=ROOT)
    plain.draw(pygame.Surface(SIZE))
    assert not any(k == pygame.K_v for k, _ in plain._buttons)


def test_painting_offers_all_seven_chemicals_in_a_realism_world():
    ui = realistic_ui()
    assert [n for _, n in ui.nutrients] == ["glucose", "lactose", "oxygen", "acetate", "bacilysin", "ai2", "pyocyanin"]
    key(ui, pygame.K_3)
    ui.draw(pygame.Surface(SIZE))
    assert ui.mode == "paint" and ui.selected_nutrient == 2


def test_inspecting_a_cell_shows_its_physiology_and_the_chemistry_around_it():
    ui = realistic_ui()
    snap = ui.controller.render_snapshot()
    x, y = ui.renderer.camera.world_to_screen(snap.position_um[0])
    ui._inspect_at((int(x), int(y)))
    assert ui.selected_inspection is not None
    text = "\n".join(ui._inspection_lines())
    for fragment in ("Physiology (realism layer):", "state: vegetative", "growth readiness (lag)", "starvation stress",
                     "matrix (biofilm)", "energy reserve", "Local chemistry:", "oxygen:", "pH:"):
        assert fragment in text, fragment
    ui.draw(pygame.Surface(SIZE))


def test_a_plain_world_inspection_has_no_realism_block():
    c = realistic(3, realism=False)
    ui = MicrobiomeUI(controller=c, window_size=SIZE, project_root=ROOT)
    ui.draw(pygame.Surface(SIZE))
    snap = c.render_snapshot()
    x, y = ui.renderer.camera.world_to_screen(snap.position_um[0])
    ui._inspect_at((int(x), int(y)))
    assert ui.selected_inspection is not None and "Physiology" not in "\n".join(ui._inspection_lines())


def test_the_top_bar_reports_spores_and_biofilm_when_there_are_any():
    ui = realistic_ui()
    sim = ui.controller._realism.simulation
    sim.life.spore[:2] = 1.0
    sim.life.eps[2:4] = 1.0
    ui.controller.advance_ticks(2)
    ui.draw(pygame.Surface(SIZE))                                   # must not raise with every state present
    assert ui.controller.realism_stats()["spores"] == 2


def test_the_simulation_can_now_be_run_much_faster():
    assert max(app_module._SPEED_STEPS) == 64 and list(app_module._SPEED_STEPS) == sorted(app_module._SPEED_STEPS)
