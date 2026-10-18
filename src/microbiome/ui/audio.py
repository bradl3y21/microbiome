"""Procedural sound for the UI layer: an underwater ambience and short effects, all synthesised with numpy.

No sample files are shipped. Every sound is a deterministic function of its name and variant (fixed seeds), so a
sound is the same bytes on every run and the whole module is testable without a speaker. Nothing here reads or
writes simulation state: the UI feeds it events (key presses, clicks) and population deltas it derived from the
render snapshot, and it only ever talks to ``pygame.mixer``.

Usage: ``SoundBoard.start()`` opens the mixer (returns False and stays silent when there is no audio device),
``play(name, volume, pan)`` plays one effect, and ``python -m microbiome.ui.audio --export DIR`` writes every
sound as a WAV so it can be auditioned or inspected.
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pygame

SAMPLE_RATE = 44100
AMBIENT_SECONDS = 12.0
EFFECT_NAMES = ("click", "select", "nope", "drop", "introduce", "divide", "die", "pause", "resume", "step")


# ------------------------------------------------------------------------------------------ synthesis
def _time(seconds: float, sr: int) -> np.ndarray:
    return np.arange(int(round(seconds * sr)), dtype=np.float64) / sr


def _envelope(t: np.ndarray, attack: float, decay: float) -> np.ndarray:
    return (1.0 - np.exp(-t / attack)) * np.exp(-t / decay)


def _sweep(t: np.ndarray, f_start: float, f_end: float, tau: float, sr: int) -> np.ndarray:
    """A sine whose frequency moves from ``f_start`` to ``f_end`` exponentially with time constant ``tau``."""
    frequency = f_end + (f_start - f_end) * np.exp(-t / tau)
    return np.sin(2.0 * np.pi * np.cumsum(frequency) / sr)


def _lowpass(x: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    frequency = np.fft.rfftfreq(len(x), 1.0 / sr)
    return np.fft.irfft(spectrum / (1.0 + (frequency / cutoff) ** 4), n=len(x))


def _finish(x: np.ndarray, peak: float, sr: int, fade_ms: float = 6.0) -> np.ndarray:
    """Fade the tail to zero (no click), normalise to ``peak`` and return float32 mono."""
    n = min(len(x), max(1, int(sr * fade_ms / 1000.0)))
    x = x.copy()
    x[-n:] *= np.linspace(1.0, 0.0, n)
    m = min(len(x), max(1, int(sr * 0.0004)))   # 0.4 ms in: sharp enough for a pop, but never a speaker thump
    x[:m] *= np.linspace(0.0, 1.0, m)
    top = float(np.max(np.abs(x))) or 1.0
    return (x * (peak / top)).astype(np.float32)


def _noise(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def synth_effect(name: str, variant: int = 0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """One effect as float32 mono in [-1, 1]. Variants differ in pitch so repeats do not sound mechanical."""
    if name == "drop":                      # a water-drop "bloop": pitch rises as the bubble forms
        f0, f1 = ((300, 760), (260, 640), (340, 900))[variant % 3]
        t = _time(0.22, sr)
        tone = _sweep(t, f0, f1, 0.045, sr) + 0.2 * _sweep(t, 2 * f0, 2 * f1, 0.045, sr)
        return _finish(tone * _envelope(t, 0.003, 0.05), 0.55, sr)
    if name == "introduce":                 # a soft two-note rising chime (a fifth)
        t = _time(0.55, sr)
        out = np.zeros_like(t)
        for start, f in ((0.0, 392.0), (0.09, 587.0)):
            local = np.clip(t - start, 0.0, None)
            out += np.sin(2 * np.pi * f * local) * _envelope(local, 0.004, 0.16) * (t >= start)
        return _finish(_lowpass(out, 2600, sr), 0.45, sr)
    if name == "divide":                    # a small membrane "pop"
        f0, f1 = ((560, 190), (480, 170), (640, 220))[variant % 3]
        t = _time(0.09, sr)
        burst = _lowpass(_noise(40 + variant, len(t)), 4000, sr) * np.exp(-t / 0.004) * 0.35
        return _finish(_sweep(t, f0, f1, 0.02, sr) * _envelope(t, 0.0015, 0.022) + burst, 0.5, sr)
    if name == "die":                       # a low, dull "glub"
        f0, f1 = ((170, 55), (140, 50))[variant % 2]
        t = _time(0.28, sr)
        body = _sweep(t, f0, f1, 0.07, sr) * _envelope(t, 0.004, 0.085)
        rumble = _lowpass(_noise(60 + variant, len(t)), 600, sr) * np.exp(-t / 0.06) * 0.25
        return _finish(body + rumble, 0.45, sr)
    if name == "select":                    # a tiny bright tick
        t = _time(0.07, sr)
        tone = np.sin(2 * np.pi * 1400 * t) + 0.5 * np.sin(2 * np.pi * 2100 * t)
        return _finish(tone * _envelope(t, 0.001, 0.014), 0.3, sr)
    if name == "nope":                      # a dull, falling, low-volume refusal
        t = _time(0.15, sr)
        return _finish(_sweep(t, 240, 180, 0.05, sr) * _envelope(t, 0.004, 0.06), 0.3, sr)
    if name == "click":
        t = _time(0.045, sr)
        tone = np.sin(2 * np.pi * 820 * t) + 0.15 * _noise(7, len(t)) * np.exp(-t / 0.002)
        return _finish(tone * _envelope(t, 0.0008, 0.009), 0.28, sr)
    if name == "step":
        t = _time(0.055, sr)
        return _finish(np.sin(2 * np.pi * 620 * t) * _envelope(t, 0.001, 0.013), 0.3, sr)
    if name in ("pause", "resume"):
        first, second = (520.0, 390.0) if name == "pause" else (390.0, 520.0)
        t = _time(0.24, sr)
        out = np.zeros_like(t)
        for start, f in ((0.0, first), (0.1, second)):
            local = np.clip(t - start, 0.0, None)
            out += np.sin(2 * np.pi * f * local) * _envelope(local, 0.003, 0.05) * (t >= start)
        return _finish(out, 0.35, sr)
    raise KeyError(f"unknown sound {name!r}")


def synth_ambient(sr: int = SAMPLE_RATE, seconds: float = AMBIENT_SECONDS) -> np.ndarray:
    """A seamless stereo loop (n, 2) float32: filtered noise, slow swells, low drones and distant bubbles.

    Everything periodic uses a whole number of cycles per loop, so the last sample flows into the first with no
    click. The bubbles are placed away from the loop edges for the same reason.
    """
    n = int(round(sr * seconds))
    t = np.arange(n, dtype=np.float64) / sr
    frequency = np.fft.rfftfreq(n, 1.0 / sr)
    shape = (frequency / (frequency + 40.0)) / (1.0 + (frequency / 650.0) ** 3)
    channels = []
    for seed in (101, 102):
        rng = np.random.default_rng(seed)
        spectrum = shape * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, shape.size))
        spectrum[0] = 0.0
        hiss = np.fft.irfft(spectrum, n=n)
        hiss /= np.std(hiss) or 1.0
        swell = 1.0 + 0.3 * np.sin(2 * np.pi * 1 * t / seconds + seed) + 0.15 * np.sin(2 * np.pi * 3 * t / seconds + 2 * seed)
        channels.append(0.1 * hiss * swell)
    drone = np.zeros(n)
    for f, level, tremolo in ((55.0, 0.05, 2), (82.5, 0.03, 3), (110.0, 0.02, 5)):
        cycles = f * seconds
        assert abs(cycles - round(cycles)) < 1e-9, "drone frequency must fit a whole number of loop cycles"
        drone += level * np.sin(2 * np.pi * f * t) * (0.75 + 0.25 * np.sin(2 * np.pi * tremolo * t / seconds))
    rng = np.random.default_rng(303)
    bubbles = np.zeros((n, 2))
    for when in np.sort(rng.uniform(0.8, seconds - 0.8, 14)):
        f0, f1 = rng.uniform(220, 420), rng.uniform(520, 1100)
        length = int(0.22 * sr)
        local = np.arange(length) / sr
        blip = _sweep(local, f0, f1, 0.05, sr) * _envelope(local, 0.004, 0.06) * rng.uniform(0.03, 0.09)
        start = int(when * sr)
        pan = rng.uniform(0.2, 0.8)
        bubbles[start:start + length, 0] += blip * (1.0 - pan)
        bubbles[start:start + length, 1] += blip * pan
    out = np.stack(channels, axis=1) + drone[:, None] + bubbles
    return (out * (0.5 / float(np.max(np.abs(out))))).astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


# ------------------------------------------------------------------------------------------ playback
class SoundBoard:
    """Plays the effects through ``pygame.mixer`` with per-sound rate limits and stereo panning."""

    MASTER = 0.8
    AMBIENT_VOLUME = 0.34
    AMBIENT_PAUSED_VOLUME = 0.12
    VARIANTS = {"drop": 3, "divide": 3, "die": 2}
    MIN_GAP_MS = {"drop": 70, "divide": 90, "die": 140, "select": 40, "click": 30, "nope": 200, "step": 30}
    DEFAULT_GAP_MS = 60

    def __init__(self, clock: Callable[[], int] | None = None) -> None:
        self.active = False
        self.muted = False
        self._clock = clock or pygame.time.get_ticks
        self._sounds: dict[str, list[pygame.mixer.Sound]] = {}
        self._ambient: pygame.mixer.Sound | None = None
        self._ambient_channel: pygame.mixer.Channel | None = None
        self._last_play: dict[str, int] = {}
        self._cursor: dict[str, int] = {}
        self._muffled = False

    # -------------------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Open the mixer and build every sound; ``False`` (and silence) when no audio device works."""
        if self.active:
            return True
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(SAMPLE_RATE, -16, 2, 512)
            sr, fmt, channels = pygame.mixer.get_init()
            if fmt != -16 or channels not in (1, 2):
                pygame.mixer.quit()
                pygame.mixer.init(SAMPLE_RATE, -16, 2, 512, allowedchanges=0)
                sr, fmt, channels = pygame.mixer.get_init()
            pygame.mixer.set_num_channels(16)
            pygame.mixer.set_reserved(1)

            def make(stereo: np.ndarray) -> pygame.mixer.Sound:
                array = stereo if channels == 2 else stereo.mean(axis=1)
                return pygame.sndarray.make_sound(to_int16(array))

            for name in EFFECT_NAMES:
                built = []
                for variant in range(self.VARIANTS.get(name, 1)):
                    mono = synth_effect(name, variant, sr)
                    sound = make(np.stack([mono, mono], axis=1))
                    sound.set_volume(self.MASTER)
                    built.append(sound)
                self._sounds[name] = built
            self._ambient = make(synth_ambient(sr))
            self._ambient.set_volume(1.0)
            self._ambient_channel = pygame.mixer.Channel(0)
            self._ambient_channel.play(self._ambient, loops=-1, fade_ms=1500)
            self._apply_ambient_volume()   # after play(): a channel's volume does not survive being played on
        except (pygame.error, OSError, ValueError):
            self.active = False
            return False
        self.active = True
        return True

    def stop(self) -> None:
        if self._ambient_channel is not None:
            self._ambient_channel.stop()
        self.active = False
        self._sounds.clear()
        if pygame.mixer.get_init():
            pygame.mixer.quit()

    # -------------------------------------------------------------------------- controls
    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        self._apply_ambient_volume()
        return self.muted

    def set_muffled(self, muffled: bool) -> None:
        """Quiet the ambience while the simulation is paused."""
        self._muffled = bool(muffled)
        self._apply_ambient_volume()

    def _apply_ambient_volume(self) -> None:
        if self._ambient_channel is None:
            return
        level = 0.0 if self.muted else (self.AMBIENT_PAUSED_VOLUME if self._muffled else self.AMBIENT_VOLUME)
        self._ambient_channel.set_volume(level)

    # -------------------------------------------------------------------------- effects
    def play(self, name: str, volume: float = 1.0, pan: float = 0.0) -> bool:
        """Play one effect. ``pan`` is -1 (left) .. 1 (right). Returns whether it was actually emitted."""
        if name not in EFFECT_NAMES:
            raise KeyError(f"unknown sound {name!r}")
        if not self.active or self.muted:
            return False
        now = int(self._clock())
        gap = self.MIN_GAP_MS.get(name, self.DEFAULT_GAP_MS)
        last = self._last_play.get(name)
        if last is not None and now - last < gap:
            return False
        self._last_play[name] = now
        return self._emit(name, float(volume), float(pan))

    def _emit(self, name: str, volume: float, pan: float) -> bool:
        variants = self._sounds.get(name)
        if not variants:
            return False
        index = self._cursor.get(name, 0)
        self._cursor[name] = index + 1
        channel = pygame.mixer.find_channel()
        if channel is None:
            return False
        sound = variants[index % len(variants)]
        # Loudness lives on the Sound (it persists and is in force from the first sample); the channel only pans,
        # and must be set after play() because playing resets it. At most one mixer buffer is unpanned.
        sound.set_volume(self.MASTER * max(0.0, min(1.0, volume)))
        channel.play(sound)
        angle = (max(-1.0, min(1.0, pan)) + 1.0) * math.pi / 4.0
        channel.set_volume(min(1.0, math.cos(angle) * math.sqrt(2.0)), min(1.0, math.sin(angle) * math.sqrt(2.0)))
        return True


# ------------------------------------------------------------------------------------------ export
def export_wavs(directory: str | Path, sr: int = SAMPLE_RATE) -> list[Path]:
    """Write every effect variant and the ambient loop as 16-bit WAV files."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written = []

    def write(path: Path, data: np.ndarray) -> None:
        stereo = data if data.ndim == 2 else np.stack([data, data], axis=1)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(sr)
            handle.writeframes(to_int16(stereo).tobytes())
        written.append(path)

    for name in EFFECT_NAMES:
        for variant in range(SoundBoard.VARIANTS.get(name, 1)):
            suffix = f"_{variant}" if name in SoundBoard.VARIANTS else ""
            write(target / f"{name}{suffix}.wav", synth_effect(name, variant, sr))
    write(target / "ambient.wav", synth_ambient(sr))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m microbiome.ui.audio")
    parser.add_argument("--export", metavar="DIR", required=True, help="write every sound to DIR as WAV files")
    args = parser.parse_args(argv)
    for path in export_wavs(args.export):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
