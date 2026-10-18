"""UI sound: synthesis properties, playback rules, event mapping and its isolation from the simulation.

Nobody can listen in CI, so these pin what is checkable: each sound is finite, quiet enough, click-free at both
ends and moves in the direction its design says (a drop rises, a pop and a thud fall); the ambience loops without a
seam; the board is silent and harmless with no audio device; each UI action and each population change selects the
intended sound; and none of it can change a simulation (the same script yields the same bytes with sound on, off
or recorded).
"""
import wave
from pathlib import Path

import numpy as np
import pygame
import pytest

from microbiome.sim.controller import SimulationController
from microbiome.ui import audio as A
from microbiome.ui.app import MicrobiomeUI
from microbiome.ui.audio import EFFECT_NAMES, SAMPLE_RATE, SoundBoard

ROOT = Path(__file__).resolve().parent.parent
ALL = [(name, v) for name in EFFECT_NAMES for v in range(SoundBoard.VARIANTS.get(name, 1))]


@pytest.fixture(scope="module", autouse=True)
def pygame_up():
    pygame.init()
    yield
    pygame.quit()


def dominant(x, start_s, length=1024):
    start = min(int(start_s * SAMPLE_RATE), max(0, len(x) - length))
    segment = x[start:start + length] * np.hanning(min(length, len(x)))
    spectrum = np.abs(np.fft.rfft(segment, 16384))
    return float(np.fft.rfftfreq(16384, 1 / SAMPLE_RATE)[int(np.argmax(spectrum))])


# ------------------------------------------------------------------------------ synthesis
@pytest.mark.parametrize("name, variant", ALL)
def test_every_effect_is_finite_quiet_click_free_and_audible(name, variant):
    x = A.synth_effect(name, variant)
    assert x.dtype == np.float32 and x.ndim == 1 and np.isfinite(x).all()
    assert 0.2 <= np.abs(x).max() <= 0.6, "loud enough to hear, quiet enough not to startle"
    assert np.sqrt(np.mean(x.astype(np.float64) ** 2)) > 0.02
    assert abs(float(x[0])) < 1e-3 and abs(float(x[-1])) < 1e-3, "a nonzero edge is a speaker click"
    assert 0.03 <= len(x) / SAMPLE_RATE <= 0.6


def test_pitch_moves_the_way_each_sound_is_designed_to():
    for variant in range(3):
        drop = A.synth_effect("drop", variant)
        assert dominant(drop, 0.06) > 1.3 * dominant(drop, 0.004), "a water drop rises in pitch"
        pop = A.synth_effect("divide", variant)
        assert dominant(pop, 0.04) < 0.75 * dominant(pop, 0.002), "a membrane pop falls"
    for variant in range(2):
        glub = A.synth_effect("die", variant)
        assert dominant(glub, 0.1) < 0.8 * dominant(glub, 0.004) and dominant(glub, 0.004) < 250, "a low, falling thud"
    assert dominant(A.synth_effect("pause"), 0.02) > dominant(A.synth_effect("pause"), 0.16)
    assert dominant(A.synth_effect("resume"), 0.02) < dominant(A.synth_effect("resume"), 0.16)


def test_variants_differ_and_synthesis_is_deterministic():
    for name in SoundBoard.VARIANTS:
        bytes_ = [A.synth_effect(name, v).tobytes() for v in range(SoundBoard.VARIANTS[name])]
        assert len(set(bytes_)) == len(bytes_), f"{name} variants must differ so repeats do not sound mechanical"
    for name, variant in ALL:
        assert A.synth_effect(name, variant).tobytes() == A.synth_effect(name, variant).tobytes()
    assert A.synth_ambient().tobytes() == A.synth_ambient().tobytes()


def test_the_ambience_is_a_seamless_low_stereo_loop():
    a = A.synth_ambient()
    assert a.shape == (int(SAMPLE_RATE * A.AMBIENT_SECONDS), 2) and a.dtype == np.float32 and np.isfinite(a).all()
    assert np.abs(a).max() <= 0.5 + 1e-6 and np.sqrt(np.mean(a.astype(np.float64) ** 2)) > 0.03
    for channel in (0, 1):
        seam = abs(float(a[0, channel]) - float(a[-1, channel]))
        assert seam <= float(np.abs(np.diff(a[:, channel])).max()), "the loop point must be no bigger than any normal step"
    spectrum = np.abs(np.fft.rfft(a[:, 0].astype(np.float64))) ** 2
    frequency = np.fft.rfftfreq(len(a), 1 / SAMPLE_RATE)
    assert spectrum[frequency < 1000].sum() / spectrum.sum() > 0.9, "underwater ambience is dark"
    assert abs(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])) < 0.5, "decorrelated channels give a sense of space"


def test_export_writes_a_playable_wav_for_every_sound(tmp_path):
    paths = A.export_wavs(tmp_path)
    assert len(paths) == len(ALL) + 1 and {p.suffix for p in paths} == {".wav"}
    for path in paths:
        with wave.open(str(path)) as handle:
            assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (2, 2, SAMPLE_RATE)
            assert handle.getnframes() > 1000
    with wave.open(str(tmp_path / "ambient.wav")) as handle:
        assert handle.getnframes() == int(SAMPLE_RATE * A.AMBIENT_SECONDS)


# ------------------------------------------------------------------------------ the board
class Clock:
    def __init__(self):
        self.now = 10_000

    def __call__(self):
        return self.now


@pytest.fixture()
def board():
    clock = Clock()
    b = SoundBoard(clock=clock)
    b.clock = clock
    assert b.start(), "the dummy audio driver must be able to open"
    yield b
    b.stop()


def test_board_plays_through_the_mixer_and_stops_cleanly(board):
    assert board.active and pygame.mixer.get_init()
    assert board.play("drop", 1.0, -1.0) and pygame.mixer.get_busy()
    board.stop()
    assert not board.active and not board.play("drop")
    assert board.start() and board.active, "can be restarted"


def test_rate_limits_mute_and_unknown_names(board):
    assert board.play("divide")
    assert not board.play("divide"), "inside the minimum gap"
    board.clock.now += SoundBoard.MIN_GAP_MS["divide"]
    assert board.play("divide")
    assert board.play("die"), "limits are per sound, not global"
    board.toggle_mute()
    board.clock.now += 5000
    assert board.muted and not board.play("select")
    board.toggle_mute()
    assert board.play("select")
    with pytest.raises(KeyError):
        board.play("explosion")


def test_loudness_is_applied_to_the_sound_and_survives_playing(board):
    board.play("divide", 0.5)
    assert board._sounds["divide"][0].get_volume() == pytest.approx(SoundBoard.MASTER * 0.5, abs=0.01)
    board.clock.now += 1000
    board.play("divide", 5.0)                   # out-of-range volumes are clamped, never amplified
    assert board._sounds["divide"][1].get_volume() == pytest.approx(SoundBoard.MASTER, abs=0.01)


def test_the_ambience_actually_starts_at_its_level_not_at_zero_or_full(board):
    assert board._ambient_channel.get_busy()
    assert board._ambient_channel.get_volume() == pytest.approx(SoundBoard.AMBIENT_VOLUME, abs=0.02)


def test_variants_cycle_round_robin(board):
    chosen = []
    original = board._sounds["drop"]
    for _ in range(6):
        board.clock.now += 1000
        before = board._cursor.get("drop", 0)
        board.play("drop")
        chosen.append(board._cursor["drop"] - before)
    assert chosen == [1] * 6 and len(original) == 3


def test_no_audio_device_means_silence_not_a_crash(monkeypatch):
    def broken(*args, **kwargs):
        raise pygame.error("no audio device")

    pygame.mixer.quit()
    monkeypatch.setattr(pygame.mixer, "init", broken)
    silent = SoundBoard()
    assert silent.start() is False and not silent.active
    assert silent.play("click") is False
    silent.set_muffled(True)
    assert silent.toggle_mute() is True
    silent.stop()


# ------------------------------------------------------------------------------ UI mapping
class Recorder(SoundBoard):
    """A board that records what would have played instead of opening a mixer."""

    def __init__(self):
        self.t = 0
        super().__init__(clock=self._tick)
        self.log = []
        self.active = True

    def _tick(self):
        self.t += 1000
        return self.t

    def _emit(self, name, volume, pan):
        self.log.append((name, volume, pan))
        return True

    def names(self):
        return [entry[0] for entry in self.log]


def make_ui(size=(1280, 720), controller=None):
    ui = MicrobiomeUI(controller=controller or SimulationController.create_demo(seed=2026), window_size=size,
                      project_root=ROOT)
    ui.audio = Recorder()
    ui.draw(pygame.Surface(size))
    ui.audio.log.clear()
    return ui


def key(ui, k):
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k))


def click(ui, position):
    return ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=position, button=1))


def test_keys_select_their_sounds():
    ui = make_ui()
    key(ui, pygame.K_SPACE)
    key(ui, pygame.K_SPACE)
    key(ui, pygame.K_PERIOD)
    key(ui, pygame.K_p)
    key(ui, pygame.K_EQUALS)
    assert ui.audio.names() == ["pause", "resume", "step", "click", "click"]


def test_m_mutes_everything_and_unmutes_with_a_click():
    ui = make_ui()
    key(ui, pygame.K_m)
    assert ui.audio.muted and ui.audio.log == []
    key(ui, pygame.K_p)
    key(ui, pygame.K_SPACE)
    assert ui.audio.log == [], "muted means nothing plays"
    key(ui, pygame.K_m)
    assert not ui.audio.muted and ui.audio.names() == ["click"]


def test_the_toolbar_has_a_working_mute_button():
    ui = make_ui()
    ui.draw(pygame.Surface((1280, 720)))
    rect = next(r for k, r in ui._buttons if k == pygame.K_m)
    assert click(ui, rect.center) and ui.audio.muted
    ui.draw(pygame.Surface((1280, 720)))
    assert any(k == pygame.K_m for k, _ in ui._buttons), "the button stays, now offering unmute"


def test_painting_drops_pan_with_where_you_paint():
    ui = make_ui()
    key(ui, pygame.K_1)
    ui.draw(pygame.Surface((1280, 720)))
    ui.audio.log.clear()
    click(ui, (300, 300))
    ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(300, 300), button=1))
    click(ui, (800, 300))
    drops = [entry for entry in ui.audio.log if entry[0] == "drop"]
    assert len(drops) == 2 and drops[0][2] < -0.1 and drops[1][2] > 0.1, drops


def test_introducing_a_cell_chimes_once_and_is_not_mistaken_for_a_division():
    ui = make_ui()
    key(ui, pygame.K_o)
    key(ui, pygame.K_RETURN)
    assert ui.audio.names().count("introduce") == 1
    ui.audio.log.clear()
    for _ in range(3):
        ui.advance_frame()
        ui.draw(pygame.Surface((1280, 720)))
    assert len(ui.controller.render_snapshot().agent_id) == 1
    assert "divide" not in ui.audio.names(), "a user-introduced cell is not a birth"
    assert ui._pending_introductions == 0


def test_inspecting_a_cell_ticks_and_missing_one_is_refused():
    ui = make_ui()
    click(ui, (400, 300))
    assert ui.audio.names() == ["nope"], "no cells yet"
    key(ui, pygame.K_o)
    key(ui, pygame.K_RETURN)
    ui.advance_frame()
    ui.draw(pygame.Surface((1280, 720)))
    ui.audio.log.clear()
    centre = ui.renderer.camera.world_to_screen(ui.controller.render_snapshot().position_um[0])
    assert click(ui, (int(centre[0]), int(centre[1])))
    assert ui.audio.names() == ["select"]


def forced_ui():
    controller = SimulationController.create_demo(seed=2026)
    table = controller._phenotypes
    table.division_mass_pg[:] = 1e-6
    table.division_rate_s[:] = 1e6
    table.death_rate_s[:] = 0.0
    controller.advance_ticks(1)
    controller.introduce_organism(int(controller.list_genomes()[0].genome_id), (200.0, 300.0))
    controller.advance_ticks(1)
    return controller, make_ui(controller=controller)


def test_divisions_pop_with_a_bounded_volume_and_a_valid_pan():
    controller, ui = forced_ui()
    controller.advance_ticks(3)
    ui.draw(pygame.Surface((1280, 720)))
    pops = [entry for entry in ui.audio.log if entry[0] == "divide"]
    assert pops, "dividing cells must be heard"
    for _, volume, pan in pops:
        assert 0.0 < volume <= 1.0 and -0.8 <= pan <= 0.8


def test_deaths_thud_and_only_when_cells_actually_vanish():
    controller, ui = forced_ui()
    ui.draw(pygame.Surface((1280, 720)))
    ui.audio.log.clear()
    ui.draw(pygame.Surface((1280, 720)))
    assert ui.audio.log == [], "an unchanged population is silent"
    controller._phenotypes.division_mass_pg[:] = 1e9
    controller._phenotypes.death_rate_s[:] = 1e6
    controller.advance_ticks(1)
    ui.draw(pygame.Surface((1280, 720)))
    assert ui.audio.names() == ["die"] and len(ui.controller.render_snapshot().agent_id) == 0


def test_the_ambience_quiets_while_paused_and_is_silent_when_muted(board):
    channel = board._ambient_channel
    assert channel.get_volume() == pytest.approx(SoundBoard.AMBIENT_VOLUME, abs=0.02)
    board.set_muffled(True)
    assert channel.get_volume() < SoundBoard.AMBIENT_VOLUME
    board.toggle_mute()
    assert channel.get_volume() == 0.0
    board.toggle_mute()
    board.set_muffled(False)
    assert channel.get_volume() == pytest.approx(SoundBoard.AMBIENT_VOLUME, abs=0.02)


def run_script(audio):
    controller = SimulationController.create_demo(seed=77)
    ui = MicrobiomeUI(controller=controller, window_size=(1280, 720), project_root=ROOT)
    ui.audio = audio
    surface = pygame.Surface((1280, 720))
    for step in range(40):
        if step == 2:
            key(ui, pygame.K_o)
            key(ui, pygame.K_RETURN)
        if step == 5:
            key(ui, pygame.K_1)
            click(ui, (500, 300))
        if step == 20:
            key(ui, pygame.K_SPACE)
        if step == 24:
            key(ui, pygame.K_SPACE)
        ui.advance_frame()
        ui.draw(surface)
    a, f = controller._state.agents, controller._environment.field
    return (a.agent_id.tobytes(), a.position_um.tobytes(), a.biomass_mmol.tobytes(), f.concentration_mmol_l.tobytes(),
            int(controller._state.tick_index))


def test_sound_cannot_change_the_simulation():
    off = run_script(SoundBoard())            # inactive: the default in every test
    recorded = run_script(Recorder())
    assert off == recorded
    muted = Recorder()
    muted.muted = True
    assert run_script(muted) == off
