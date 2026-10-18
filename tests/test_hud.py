"""The game-style HUD is presentation over the D3 facade; these pin the parts that could silently rot.

  * the population meter's display cap equals the engine's real cap (the UI may not import the engine)
  * toolbar buttons are clickable and drive exactly the keyboard actions, and a click on the bar never
    falls through to the field (no accidental paint / inspect / introduce)
  * the legend's species colours are the renderer's species colours
  * the layout survives small and odd window sizes
"""
from pathlib import Path

import numpy as np
import pygame
import pytest

from microbiome.render.view import WorldView
from microbiome.sim.controller import SimulationController
from microbiome.ui import app as app_module
from microbiome.ui.app import MicrobiomeUI, dispatch_command

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def pygame_up():
    pygame.init()
    yield
    pygame.quit()


def make_ui(size=(960, 640)):
    return MicrobiomeUI(controller=SimulationController.create_demo(seed=2026), window_size=size, project_root=ROOT)


def click(ui, position):
    return ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=position, button=1))


def button_rect(ui, key):
    return next(rect for k, rect in ui._buttons if k == key)


def test_population_meter_cap_matches_the_engine_cap():
    from microbiome.sim.engine import MAX_AGENT_POPULATION

    assert app_module.POPULATION_METER_CAP == MAX_AGENT_POPULATION


def test_toolbar_buttons_drive_the_same_actions_as_the_keys():
    ui = make_ui()
    surface = pygame.Surface((960, 640))
    ui.draw(surface)
    assert not ui.paused and ui.mode == "inspect"
    assert click(ui, button_rect(ui, pygame.K_SPACE).center)
    assert ui.paused
    ui.draw(surface)
    assert click(ui, button_rect(ui, pygame.K_p).center)
    assert ui.mode == "paint"
    ui.draw(surface)
    assert {k for k, _ in ui._buttons} >= {pygame.K_1, pygame.K_2}, "paint mode must offer a chip per nutrient"
    assert click(ui, button_rect(ui, pygame.K_i).center)
    assert ui.mode == "inspect"


def test_a_click_on_the_hud_never_reaches_the_field():
    ui = make_ui()
    surface = pygame.Surface((960, 640))
    ui.draw(surface)
    ui.mode = "paint"
    ui.draw(surface)
    bar = ui._toolbar_rect
    gap = (bar.x + 3, bar.y + 3)                    # on the bar, not on any button
    assert not any(rect.collidepoint(gap) for _, rect in ui._buttons)
    calls = []
    original = ui.controller.place_nutrient
    ui.controller.place_nutrient = lambda *a, **k: calls.append(1) or original(*a, **k)
    assert click(ui, gap)
    ui.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=gap, buttons=(1, 0, 0), rel=(0, 0)))
    assert not ui._painting and not calls
    assert click(ui, ui._topbar_rect.center) and not calls


def test_legend_colours_are_the_renderers_species_colours():
    ui = make_ui()
    dispatch_command(ui.controller, "advance_ticks", count=1)
    for option in ui.genomes:
        dispatch_command(ui.controller, "introduce_organism", genome_id=option.genome_id, position_um=(300.0, 300.0))
    dispatch_command(ui.controller, "advance_ticks", count=1)
    snapshot = ui.controller.render_snapshot()
    view = WorldView.from_snapshot(snapshot)
    seen = {}
    for genome_id, colour in zip(snapshot.genome_id, view.colour_index):
        seen[int(genome_id)] = int(colour)
    assert len(seen) == len(ui.genomes) >= 2
    for option in ui.genomes:
        assert ui._palette_of(snapshot, int(option.genome_id)) == seen[int(option.genome_id)]
    assert len(set(seen.values())) == len(seen), "each genome needs its own colour"


@pytest.mark.parametrize("size", [(320, 240), (640, 360), (1920, 1080)])
def test_layout_survives_odd_window_sizes(size):
    ui = make_ui(size)
    surface = pygame.Surface(size)
    ui.draw(surface)
    ui.paused = True
    ui.mode = "paint"
    ui.draw(surface)
    ui.mode = "organism"
    ui.draw(surface)
