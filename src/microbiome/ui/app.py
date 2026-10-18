"""Pygame controls over the frozen D3 simulation-controller boundary."""

from __future__ import annotations

import json
import math
import textwrap
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pygame

from microbiome.core.ui_api import CellInspection, GenomeInspection, GenomeOption, PhenotypePreview, RenderSnapshot
from microbiome.render import PygameRenderer, SpriteAtlas
from microbiome.render.lens import MicroscopeRenderer
from microbiome.render.renderer import OVERLAYS
from microbiome.sim.controller import SimulationController

from .audio import SoundBoard

from .commands import dispatch_command as _dispatch_command


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WINDOW_SIZE = (1120, 760)
FPS = 60
_SPEED_STEPS = (1, 2, 4, 8, 16, 32, 64)
_DEMO_DEPTH_UM = 10.0
# Display scale of the population meter (the README's 10,000-agent target). The UI may not import sim
# internals, so tests/test_hud.py pins this to the engine's real cap.
POPULATION_METER_CAP = 10_000


def dispatch_command(controller: SimulationController, action: str, **payload: Any) -> Any:
    """Expose the UI's one simulation-command path in this module as well."""
    return _dispatch_command(controller, action, **payload)


def _nutrient_names(project_root: str | Path) -> tuple[tuple[int, str], ...]:
    path = Path(project_root).resolve() / "data" / "gene_trait_table.json"
    table = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        (int(item["nutrient_id"]), str(item["name"]))
        for item in sorted(table["demo_nutrient_channels"], key=lambda item: int(item["nutrient_id"]))
    )


class MicrobiomeUI:
    """Interactive D3 client; no mutable world state is retained by the UI."""

    PANEL_COLOR = (8, 17, 32, 238)
    PANEL_EDGE = (66, 88, 100)
    TEXT_COLOR = (214, 224, 228)
    MUTED_COLOR = (136, 156, 166)
    ACCENT_COLOR = (150, 196, 190)
    WARNING_COLOR = (214, 182, 132)

    def __init__(
        self,
        controller: SimulationController,
        renderer: PygameRenderer | None = None,
        window_size: tuple[int, int] = WINDOW_SIZE,
        project_root: str | Path = PROJECT_ROOT,
    ) -> None:
        self.controller = controller
        self.project_root = Path(project_root).resolve()
        self.renderer = renderer or PygameRenderer()
        self.window_size = tuple(map(int, window_size))
        self.genomes: tuple[GenomeOption, ...] = tuple(controller.list_genomes())
        if not self.genomes:
            raise ValueError("MicrobiomeUI requires at least one verified real genome")
        self.nutrients = _nutrient_names(self.project_root)
        names = getattr(controller, "channel_names", None)
        if callable(names):
            self.nutrients = tuple(enumerate(names()))
        probe = getattr(controller, "realism_enabled", None)
        self.realism = bool(probe()) if callable(probe) else False
        self.overlay_index = 0
        self.microscope = hasattr(self.renderer, "cycle_mode")
        self.minimal_hud = self.microscope      # a microscope shows the sample, not a game interface; Tab brings the HUD back
        if self.realism:
            self.speed_index = 3      # x8: lag, growth and sporulation unfold over simulated minutes to hours
        self._nutrient_name_by_id = dict(self.nutrients)
        self._snapshot = controller.render_snapshot()
        self.renderer.camera.set_field_size((self._snapshot.width_um, self._snapshot.height_um))

        self.paused = False
        self.speed_index = 0
        self.mode = "inspect"
        self.selected_nutrient = 0
        self.brush_concentration_mmol_l = 1.0
        self.selection_index = 0
        self.selected_agent_id: int | None = None
        self.selected_inspection: CellInspection | None = None
        self.selected_genome_inspection: GenomeInspection | None = None
        self.panel_scroll = 0
        self.running = False
        self._painting = False
        self._painted_cells: set[tuple[int, int, int]] = set()
        self._notice_at = 0.0
        self.notice = "Click a cell to inspect. Press H for controls."
        self._panel_rect = pygame.Rect(0, 0, 0, 0)
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None
        self._title_font: pygame.font.Font | None = None
        self._big_font: pygame.font.Font | None = None
        self._text_cache: dict[tuple, pygame.Surface] = {}
        self._glass_cache: dict[tuple, pygame.Surface] = {}
        self._toolbar_rect = pygame.Rect(0, 0, 0, 0)
        self._topbar_rect = pygame.Rect(0, 0, 0, 0)
        self._buttons: list[tuple[int, pygame.Rect]] = []
        self._pointer: tuple[int, int] = (-1, -1)
        self.audio = SoundBoard()
        self._pending_introductions = 0
        self._audio_count: int | None = None
        self._audio_top = -1

    @classmethod
    def from_demo(
        cls,
        seed: int = 2026,
        project_root: str | Path = PROJECT_ROOT,
        window_size: tuple[int, int] = WINDOW_SIZE,
        mutation: bool = False,
        realism: bool = False,
        microscope: bool = False,
    ) -> "MicrobiomeUI":
        """Create the reproducible, manifest-backed D3 demo session.

        ``mutation`` enables D4 daughter SNVs; ``realism`` swaps in the realism layer (docs/REALISM.md).
        """
        root = Path(project_root).resolve()
        return cls(
            controller=SimulationController.create_demo(seed=seed, project_root=root, mutation=mutation, realism=realism),
            window_size=window_size,
            project_root=root,
            renderer=MicroscopeRenderer() if microscope else None,
        )

    @property
    def speed(self) -> int:
        return _SPEED_STEPS[self.speed_index]

    @property
    def current_genome(self) -> GenomeOption:
        return self.genomes[self.selection_index]

    @property
    def notice(self) -> str:
        return self._notice

    @notice.setter
    def notice(self, text: str) -> None:
        self._notice = text
        self._notice_at = time.monotonic()

    def _panel_width(self) -> int:
        return min(390, max(300, self.window_size[0] // 3))

    def _point_in_panel(self, position: tuple[int, int]) -> bool:
        return self._panel_rect.collidepoint(position)

    def _current_snapshot(self) -> RenderSnapshot:
        self._snapshot = self.controller.render_snapshot()
        return self._snapshot

    def _world_position(self, position: tuple[int, int]) -> tuple[float, float] | None:
        snapshot = self._snapshot
        world = self.renderer.camera.screen_to_world(position)
        x, y = float(world[0]), float(world[1])
        if self.microscope:           # the sample is periodic: the view may sit over a neighbouring copy of it
            x, y = x % snapshot.width_um, y % snapshot.height_um
        if not (0.0 <= x < snapshot.width_um and 0.0 <= y < snapshot.height_um):
            return None
        return x, y

    def _paint_at(self, position: tuple[int, int]) -> None:
        world = self._world_position(position)
        if world is None:
            return
        snapshot = self._snapshot
        _, ny, nx = snapshot.concentration_mmol_l.shape
        ix = min(nx - 1, int(world[0] / (snapshot.width_um / nx)))
        iy = min(ny - 1, int(world[1] / (snapshot.height_um / ny)))
        nutrient_id = int(snapshot.nutrient_id[self.selected_nutrient])
        key = (self.selected_nutrient, ix, iy)
        if key in self._painted_cells:
            return
        self._painted_cells.add(key)
        cell_volume_l = (
            (snapshot.width_um / nx)
            * (snapshot.height_um / ny)
            * _DEMO_DEPTH_UM
            * 1e-15
        )
        amount_mmol = cell_volume_l * self.brush_concentration_mmol_l
        ticket = dispatch_command(
            self.controller,
            "place_nutrient",
            nutrient_id=nutrient_id,
            position_um=world,
            amount_mmol=amount_mmol,
        )
        self._sfx("drop", 0.8, self._pan(position[0]))
        name = self._nutrient_name_by_id.get(nutrient_id, f"nutrient {nutrient_id}")
        self.notice = f"Queued {name} addition · command {int(ticket.command_id)}"

    def _commit_organism(self, position: tuple[float, float] | None = None) -> bool:
        if position is None:
            position = (self._snapshot.width_um * 0.5, self._snapshot.height_um * 0.5)
        x, y = map(float, position)
        if not (0.0 <= x < self._snapshot.width_um and 0.0 <= y < self._snapshot.height_um):
            self.notice = "Choose a point inside the field to introduce the cell."
            return False
        ticket = dispatch_command(
            self.controller,
            "introduce_organism",
            genome_id=self.current_genome.genome_id,
            position_um=(x, y),
        )
        self.notice = f"Queued {self.current_genome.organism} · command {int(ticket.command_id)}"
        self._pending_introductions += 1
        self._sfx("introduce", 1.0, self._pan(float(self.renderer.camera.world_to_screen((x, y))[0])))
        self.mode = "inspect"
        return True

    def _inspect_at(self, position: tuple[int, int]) -> None:
        snapshot = self._snapshot
        if not snapshot.agent_id.size:
            self.selected_agent_id = None
            self.selected_inspection = None
            self.selected_genome_inspection = None
            self.notice = "No cells yet. Press O to preview and introduce a real genome."
            self._sfx("nope", 0.6)
            return
        screen = np.asarray(self.renderer.camera.world_to_screen(snapshot.position_um), dtype=np.float64)
        delta = screen - np.asarray(position, dtype=np.float64)
        if self.microscope:
            period = np.array((snapshot.width_um, snapshot.height_um)) * self.renderer.camera.zoom
            delta -= period * np.rint(delta / period)
        distances2 = np.einsum("ij,ij->i", delta, delta)
        closest = int(np.argmin(distances2))
        screen_radius = float(snapshot.radius_um[closest]) * self.renderer.camera.zoom
        tolerance = max(9.0, screen_radius + 5.0)
        if distances2[closest] > tolerance * tolerance:
            self.selected_agent_id = None
            self.selected_inspection = None
            self.selected_genome_inspection = None
            self.notice = "No cell under the pointer. Zoom in or click a cell."
            self._sfx("nope", 0.6)
            return
        agent_id = int(snapshot.agent_id[closest])
        self._sfx("select", 0.8, self._pan(float(screen[closest, 0])))
        inspection = dispatch_command(self.controller, "inspect_cell", agent_id=agent_id)
        self.selected_agent_id = agent_id if inspection is not None else None
        self.selected_inspection = inspection
        self.selected_genome_inspection = (
            dispatch_command(self.controller, "inspect_genome", genome_id=inspection.genome_id)
            if inspection is not None
            else None
        )
        if inspection is None:
            self.notice = "That cell is no longer available. Click a visible cell to inspect."
        else:
            self.notice = (
                f"Inspecting {inspection.organism} ({inspection.accession_version}) · "
                f"agent {agent_id} · press H for controls"
            )

    _CLICK_KEYS = frozenset({
        pygame.K_ESCAPE, pygame.K_h, pygame.K_F1, pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_MINUS,
        pygame.K_KP_MINUS, pygame.K_p, pygame.K_i, pygame.K_o, pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET,
        pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_v, *range(pygame.K_1, pygame.K_9 + 1),
    })

    def _sfx(self, name: str, volume: float = 1.0, pan: float = 0.0) -> None:
        """Presentation-only sound; never reads or writes simulation state."""
        self.audio.play(name, volume, pan)

    def _pan(self, x_px: float) -> float:
        width = max(1, self.window_size[0])
        return float(np.clip((x_px / width * 2.0 - 1.0) * 0.8, -0.8, 0.8))

    def _audio_tick(self, snapshot: RenderSnapshot) -> None:
        """Turn population changes seen in the snapshot into sound: births pop, deaths thud.

        Agent ids only ever increase and snapshots are sorted by id, so new cells are the ids above the highest
        one seen. Cells the user introduced arrive the same way and are subtracted so they do not also pop.
        """
        ids = snapshot.agent_id
        count = int(ids.size)
        top = int(ids[-1]) if count else -1
        if self._audio_count is None or (count > 1 and ids[0] > ids[-1]):
            self._audio_count, self._audio_top = count, max(top, -1)
            return
        first_new = int(np.searchsorted(ids, np.uint64(self._audio_top + 1)))
        arrived = count - first_new
        deaths = max(0, self._audio_count + arrived - count)
        self._audio_count, self._audio_top = count, max(self._audio_top, top)
        introduced = min(arrived, self._pending_introductions)
        self._pending_introductions -= introduced
        births = arrived - introduced
        if not self.audio.active or not (births or deaths):
            return
        if births:
            x = float(np.mean(self.renderer.camera.world_to_screen(snapshot.position_um[first_new:])[:, 0]))
            self._sfx("divide", min(1.0, 0.35 + 0.15 * math.sqrt(births)), self._pan(x))
        if deaths:
            self._sfx("die", min(0.8, 0.25 + 0.1 * math.sqrt(deaths)))

    def advance_frame(self) -> None:
        """Advance deterministic fixed ticks through the D3 dispatcher."""
        if not self.paused:
            dispatch_command(self.controller, "advance_ticks", count=self.speed)

    def _toggle_paint(self) -> None:
        self.mode = "inspect" if self.mode == "paint" else "paint"
        self.notice = (
            "Paint mode: drag left mouse to seed nutrient; P returns to inspect."
            if self.mode == "paint"
            else "Inspect mode: click a cell to read its genome and live state."
        )

    def _set_speed(self, direction: int) -> None:
        self.speed_index = (self.speed_index + direction) % len(_SPEED_STEPS)
        self.notice = f"Simulation speed: {self.speed} fixed tick(s) per rendered frame."

    def _handle_key(self, key: int) -> None:
        was_paused = self.paused
        self._dispatch_key(key)
        if key == pygame.K_m:
            return
        if self.paused != was_paused:
            self.audio.set_muffled(self.paused)
            self._sfx("pause" if self.paused else "resume")
        elif key in (pygame.K_PERIOD, pygame.K_KP_PERIOD):
            self._sfx("step")
        elif key in self._CLICK_KEYS:
            self._sfx("click")

    def _microscope_key(self, key: int) -> None:
        lens = self.renderer
        if key == pygame.K_b:
            lens.cycle_mode()
            self.notice = f"Imaging mode: {lens.label().split('   ')[0]}."
        elif key in (pygame.K_z, pygame.K_x):
            focus = lens.nudge_focus(-0.5 if key == pygame.K_z else 0.5)
            self.notice = f"Focus {focus - 5.0:+.1f} um from mid-depth: cells at other depths blur."
        elif key == pygame.K_e:
            self.notice = "Eyepiece mask on." if lens.toggle_eyepiece() else "Eyepiece mask off."
        else:
            self.minimal_hud = not self.minimal_hud
            self.notice = "Instrument panel hidden." if self.minimal_hud else "Instrument panel shown (Tab hides it)."

    def _dispatch_key(self, key: int) -> None:
        if key == pygame.K_m:
            muted = self.audio.toggle_mute()
            self.notice = "Sound muted." if muted else "Sound on."
            if not muted:
                self._sfx("click")
            return
        if key == pygame.K_ESCAPE:
            if self.mode in {"help", "organism"}:
                self.mode = "inspect"
                self.notice = "Closed panel. Click a cell to inspect."
            else:
                self.running = False
            return
        if key in (pygame.K_h, pygame.K_F1):
            self.mode = "inspect" if self.mode == "help" else "help"
            return
        if self.mode == "help":
            return
        if self.mode == "organism":
            if key in (pygame.K_LEFT, pygame.K_UP):
                self.selection_index = (self.selection_index - 1) % len(self.genomes)
                self.panel_scroll = 0
            elif key in (pygame.K_RIGHT, pygame.K_DOWN):
                self.selection_index = (self.selection_index + 1) % len(self.genomes)
                self.panel_scroll = 0
            elif key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                self.panel_scroll = max(0, self.panel_scroll + (-8 if key == pygame.K_PAGEUP else 8))
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._commit_organism()
            return
        if key == pygame.K_SPACE:
            self.paused = not self.paused
            dispatch_command(self.controller, "set_paused", paused=self.paused)
            self.notice = "Paused." if self.paused else "Running."
        elif key in (pygame.K_PERIOD, pygame.K_KP_PERIOD):
            tick = dispatch_command(self.controller, "single_step")
            self.notice = f"Advanced one fixed tick · tick {int(tick)}"
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._set_speed(1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self._set_speed(-1)
        elif self.microscope and key in (pygame.K_b, pygame.K_z, pygame.K_x, pygame.K_e, pygame.K_TAB):
            self._microscope_key(key)
        elif key == pygame.K_v:
            if not self.realism:
                self.notice = "The chemistry overlays belong to the realism layer (SimulationController.create_demo(realism=True))."
            else:
                self.overlay_index = (self.overlay_index + 1) % len(OVERLAYS)
                self.renderer.overlay = OVERLAYS[self.overlay_index]
                self.notice = f"Field overlay: {self.renderer.overlay} (V cycles; brighter = more of it)."
        elif key == pygame.K_i:
            self.mode = "inspect"
            self.notice = "Inspect mode: click a cell to read its genome and live state."
        elif key == pygame.K_p:
            self._toggle_paint()
        elif key == pygame.K_o:
            self.mode = "inspect" if self.mode == "organism" else "organism"
            self.panel_scroll = 0
            if self.mode == "organism":
                self.notice = "Preview the selected genome and traits, then click the field to commit."
        elif pygame.K_1 <= key <= pygame.K_9:
            nutrient = key - pygame.K_1
            if nutrient < len(self._snapshot.nutrient_id):
                self.selected_nutrient = nutrient
                self.mode = "paint"
                nutrient_id = int(self._snapshot.nutrient_id[nutrient])
                name = self._nutrient_name_by_id.get(nutrient_id, str(nutrient_id))
                self.notice = f"Selected nutrient: {name}. Drag to paint."
        elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            factor = 0.5 if key == pygame.K_LEFTBRACKET else 2.0
            self.brush_concentration_mmol_l = float(
                np.clip(self.brush_concentration_mmol_l * factor, 1e-9, 1e6)
            )
            self.notice = f"Brush adds {self.brush_concentration_mmol_l:g} mmol/L per field cell."
        elif key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
            self.panel_scroll = max(0, self.panel_scroll + (-8 if key == pygame.K_PAGEUP else 8))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle one pygame event. Return True when the UI consumed it."""
        if event.type == pygame.QUIT:
            self.running = False
            return True
        if event.type == pygame.KEYDOWN:
            self._handle_key(event.key)
            return True

        position = tuple(getattr(event, "pos", pygame.mouse.get_pos()))
        if event.type == pygame.MOUSEMOTION:
            self._pointer = position
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for key, rect in self._buttons:
                if rect.collidepoint(position):
                    self._handle_key(key)
                    return True
            if self._toolbar_rect.collidepoint(position) or self._topbar_rect.collidepoint(position):
                return True
        if event.type == pygame.MOUSEWHEEL:
            if self._point_in_panel(position) and self.mode in {"organism", "help", "inspect"}:
                self.panel_scroll = max(0, self.panel_scroll - int(event.y) * 4)
                return True
            return self.renderer.handle_event(event, position)
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            if self.renderer.handle_event(event, position):
                return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._point_in_panel(position):
                return True
            if self.mode == "help":
                return True
            if self.mode == "organism":
                world = self._world_position(position)
                if world is not None:
                    self._commit_organism(world)
                return True
            if self.mode == "paint":
                self._painting = True
                self._painted_cells.clear()
                self._paint_at(position)
                return True
            self._inspect_at(position)
            return True
        if event.type == pygame.MOUSEMOTION and self._painting:
            if getattr(event, "buttons", (1,))[0] and not self._point_in_hud(position):
                self._paint_at(position)
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._painting = False
            self._painted_cells.clear()
            return True
        return False

    def _phenotype_lines(self, phenotype: PhenotypePreview) -> list[str]:
        lines: list[str] = []
        for name, values in (
            ("Max uptake (mmol/s)", phenotype.max_uptake_mmol_s),
            ("Storage fraction", phenotype.storage_fraction),
            ("Biomass fraction", phenotype.biomass_fraction),
            ("Energy yield (pJ/mmol)", phenotype.energy_yield_pj_per_mmol),
        ):
            for nutrient_id, value in zip(phenotype.nutrient_id, values, strict=True):
                nutrient = self._nutrient_name_by_id.get(int(nutrient_id), str(int(nutrient_id)))
                lines.append(f"{name} · {nutrient}: {float(value):.4g}")
        for name in (
            "motility_um_s", "chemotaxis_sensitivity", "random_motility_fraction",
            "division_rate_s", "division_mass_pg", "death_rate_s", "max_age_s",
            "antagonism_radius_um", "antagonism_rate_s",
        ):
            lines.append(f"{name.replace('_', ' ')}: {float(getattr(phenotype, name)):.4g}")
        lines.append("Applied gene-map rules:")
        lines.extend(f"  {rule}" for rule in phenotype.applied_rule_ids) if phenotype.applied_rule_ids else lines.append("  baseline only")
        return lines

    def _genome_lines(
        self,
        genome: GenomeOption,
        phenotype: PhenotypePreview | None = None,
        genome_inspection: GenomeInspection | None = None,
        organism: str | None = None,
        accession: str | None = None,
    ) -> list[str]:
        lines = [
            organism or genome.organism,
            f"Accession.version: {accession or genome.accession_version}",
            f"Taxonomy ID: {genome.tax_id}",
            f"Assembly: {genome.assembly_level}",
            f"Sequence SHA-256: {genome.sequence_sha256}",
            f"Annotation SHA-256: {genome.annotation_sha256}",
            f"Retrieved: {genome.fetched_at_utc}",
            f"Source: {genome.source_url}",
            f"Terms: {genome.terms_url}",
            "",
            "Trait preview (genome potential; calibration approximations):",
        ]
        if phenotype is not None:
            lines.extend(self._phenotype_lines(phenotype))
        if genome_inspection is not None:
            lines.extend(("", f"Annotated genes: {len(genome_inspection.genes):,}"))
            lines.append(f"FASTA: {Path(genome_inspection.sequence_fasta_path).name}")
            lines.append(f"Annotation: {Path(genome_inspection.annotation_gff3_path).name}")
            lines.append("Gene-to-trait evidence:")
            if genome_inspection.phenotype_evidence:
                for evidence in genome_inspection.phenotype_evidence:
                    genes = ", ".join(evidence.gene_keys) or "baseline"
                    refs = ", ".join(evidence.evidence_refs) or "no external refs"
                    lines.append(
                        f"  {evidence.rule_id}: {evidence.trait_id} ← {genes}; "
                        f"{evidence.effect_value:g} {evidence.unit}; refs {refs}"
                    )
                    if evidence.approximation_note:
                        lines.append(f"    Approximation: {evidence.approximation_note}")
            else:
                lines.append("  Baseline only; no genome-specific evidence")
        return lines

    def _view_help(self) -> list[str]:
        if not self.microscope:
            return ["Wheel · pointer-anchored zoom", "Middle-drag · pan"]
        return ["Wheel · change objective (10x to 100x)", "Middle-drag · move the slide",
                "B · imaging mode: phase, brightfield, darkfield, fluorescence",
                "Z / X · focus down / up (cells at other depths blur)", "E · eyepiece mask",
                "Tab · show / hide the instrument panel"]

    def _realism_lines(self, inspection: CellInspection) -> list[str]:
        """Physiology of the inspected cell and the chemistry around it (only when the controller offers them)."""
        life_of = getattr(self.controller, "life_of", None)
        here = getattr(self.controller, "chemistry_at", None)
        life = life_of(int(inspection.agent_id)) if callable(life_of) else None
        if life is None:
            return []
        state = "dormant spore" if life["spore"] >= 1.0 else ("sporulating" if life["spore"] > 0 else
                                                           ("starving" if life["stress"] > 0.5 else "vegetative"))
        lines = ["", "Physiology (realism layer):", f"  state: {state}",
                 f"  growth readiness (lag): {life['lag']:.0%}", f"  starvation stress: {life['stress']:.0%}",
                 f"  matrix (biofilm): {life['matrix']:.0%}", f"  energy reserve: {life['energy_pj']:.4g} pJ",
                 f"  alternative-sugar genes on: {life['alt_sugar_expression']:.0%}"]
        if callable(here):
            around = here(tuple(float(v) for v in inspection.position_um))
            lines.append("Local chemistry:")
            lines.extend(f"  {name}: {value:.4g}" + (" mmol/L" if name != "pH" else "") for name, value in around.items())
        return lines

    def _inspection_lines(self) -> list[str]:
        inspection = self.selected_inspection
        if inspection is None:
            return [
                "No cell selected.",
                "Click a visible cell to inspect its genome provenance, derived traits, and live state.",
                "",
                "Population controls:",
                "Space · pause / resume", ". · advance one fixed 1/60-second tick", "+ / - · simulation speed",
                "", "Field tools:", "P · nutrient paint mode", "1–9 · select a nutrient and paint",
                "[ / ] · halve / double brush concentration", "O · choose a real genome and preview traits",
                "Left / Right · cycle genome preview", "Click field / Enter · commit one founder",
                "", "View controls:", *self._view_help(), "H or F1 · help",
                "V · cycle the chemistry overlay (realism)", "M · mute sound", "Esc · close panel / quit",
            ]
        phenotype = inspection.phenotype
        p, v = inspection.position_um, inspection.velocity_um_s
        lines = [
            f"Live cell · agent {int(inspection.agent_id)}",
            f"Position: ({p[0]:.3f}, {p[1]:.3f}) µm",
            f"Velocity: ({v[0]:.3f}, {v[1]:.3f}) µm/s",
            f"Species ID: {int(inspection.species_id)}",
            f"Mass / radius: {inspection.mass_pg:.4g} pg / {inspection.radius_um:.4g} µm",
            f"Energy: {inspection.energy_pj:.4g} pJ",
            f"Age: {inspection.age_s:.4g} s",
            "Nutrient store (mmol):",
        ]
        lines.extend(
            f"  {self._nutrient_name_by_id.get(int(nutrient_id), str(int(nutrient_id)))}: {float(value):.4g}"
            for nutrient_id, value in zip(inspection.nutrient_id, inspection.nutrient_store_mmol, strict=True)
        )
        lines.append("Biomass substrates (mmol):")
        lines.extend(
            f"  {self._nutrient_name_by_id.get(int(nutrient_id), str(int(nutrient_id)))}: {float(value):.4g}"
            for nutrient_id, value in zip(inspection.nutrient_id, inspection.biomass_mmol, strict=True)
        )
        lines.extend(self._realism_lines(inspection))
        lines.extend(("", "Genome provenance and derived traits:"))
        genome_detail = self.selected_genome_inspection
        genome = (
            genome_detail.genome
            if genome_detail is not None
            else next((item for item in self.genomes if int(item.genome_id) == int(inspection.genome_id)), None)
        )
        if genome is not None:
            lines.extend(
                self._genome_lines(
                    genome,
                    phenotype,
                    genome_detail,
                    organism=inspection.organism,
                    accession=inspection.accession_version,
                )
            )
        return lines

    def _panel_lines(self) -> list[str]:
        if self.mode == "help":
            return [
                "Space · pause / resume", ". · advance one fixed 1/60-second tick", "+ / - · change speed",
                "P · toggle nutrient painting", "1–9 · select nutrient and paint", "[ / ] · change concentration",
                "O · choose a real genome and preview traits", "Left / Right · cycle genome preview",
                "Click field / Enter · introduce previewed founder", "Esc · close panel",
                "", "Click a cell in inspect mode to see provenance, derived traits and live state.",
                *self._view_help(),
            ]
        if self.mode == "organism":
            preview = dispatch_command(
                self.controller, "preview_phenotype", genome_id=self.current_genome.genome_id
            )
            return [
                f"Genome {self.selection_index + 1} of {len(self.genomes)}",
                "← / → choose; preview below; click field or press Enter to commit.",
                "Esc cancels without changing the population.", "",
                *self._genome_lines(self.current_genome, preview),
            ]
        if self.mode == "paint":
            nutrient_id, name = self.nutrients[min(self.selected_nutrient, len(self.nutrients) - 1)]
            return [
                f"Brush: {name}",
                f"Adds {self.brush_concentration_mmol_l:g} mmol/L to each field cell you drag over.",
                "Every addition is a command applied at the next fixed tick and booked as a source.",
                "",
                "Painting:",
                "Drag \u00b7 hold the left mouse button over the field",
                "1-9 \u00b7 choose the nutrient",
                "[ / ] \u00b7 halve or double the brush concentration",
                "P or I \u00b7 back to inspect",
                "",
                "Tip:",
                "Cells climb nutrient gradients, so a dense patch draws a swimming population toward it.",
            ]
        return self._inspection_lines()

    def _wrap_lines(self, lines: Iterable[str], columns: int) -> list[str]:
        result: list[str] = []
        for line in lines:
            if not line:
                result.append("")
            else:
                result.extend(textwrap.wrap(line, width=columns, break_long_words=True, break_on_hyphens=False) or [""])
        return result

    # ------------------------------------------------------------------ HUD (presentation only)
    def _ensure_fonts(self) -> None:
        if self._font is not None:
            return

        def make(size: int, bold: bool = False) -> pygame.font.Font:
            try:
                return pygame.font.SysFont("segoeui,helveticaneue,arial,dejavusans", size, bold=bold)
            except Exception:  # noqa: BLE001 - any font-system failure falls back to pygame's bundled font
                return pygame.font.Font(None, size + 4)

        self._font = make(16)
        self._small_font = make(14)
        self._title_font = make(20, True)
        self._big_font = make(30, True)

    def _text(self, font: pygame.font.Font, text: str, color: tuple[int, ...]) -> pygame.Surface:
        key = (id(font), text, tuple(color))
        surface = self._text_cache.get(key)
        if surface is None:
            if len(self._text_cache) > 900:
                self._text_cache.clear()
            surface = font.render(text, True, color)
            self._text_cache[key] = surface
        return surface

    def _glass(self, size: tuple[int, int], radius: int, fill: tuple[int, ...], edge: tuple[int, ...]) -> pygame.Surface:
        key = (size, radius, tuple(fill), tuple(edge))
        surface = self._glass_cache.get(key)
        if surface is None:
            surface = pygame.Surface(size, pygame.SRCALPHA, 32)
            rect = surface.get_rect()
            pygame.draw.rect(surface, fill, rect, border_radius=radius)
            if size[1] <= 80:  # a highlight on a tall panel just reads as a seam
                sheen = pygame.Rect(3, 3, max(2, size[0] - 6), max(2, size[1] // 3))
                pygame.draw.rect(surface, (255, 255, 255, 5), sheen, border_radius=max(1, radius - 3))
            pygame.draw.rect(surface, edge, rect, width=1, border_radius=radius)
            self._glass_cache[key] = surface
        return surface

    def _point_in_hud(self, position: tuple[int, int]) -> bool:
        return (
            self._panel_rect.collidepoint(position)
            or self._toolbar_rect.collidepoint(position)
            or self._topbar_rect.collidepoint(position)
        )

    def _palette_of(self, snapshot: RenderSnapshot, genome_id: int) -> int:
        table = np.asarray(snapshot.phenotype_genome_id)
        row = int(np.searchsorted(table, np.uint64(genome_id)))
        styles = getattr(snapshot, "genome_style", None)
        return int(styles[row]) % 8 if styles is not None and row < len(styles) else row % 8

    def _draw_icon(self, target: pygame.Surface, name: str, centre: tuple[int, int], color: tuple[int, ...]) -> None:
        x, y = centre
        if name == "inspect":
            pygame.draw.circle(target, color, (x - 2, y - 2), 8, 2)
            pygame.draw.line(target, color, (x + 4, y + 4), (x + 10, y + 10), 3)
        elif name == "paint":
            pygame.draw.circle(target, color, (x, y + 3), 7)
            pygame.draw.polygon(target, color, [(x - 6, y + 1), (x + 6, y + 1), (x, y - 10)])
        elif name == "genome":
            pygame.draw.rect(target, color, (x - 11, y - 6, 22, 12), 2, border_radius=6)
            pygame.draw.line(target, color, (x - 3, y), (x + 3, y), 2)
            pygame.draw.line(target, color, (x, y - 3), (x, y + 3), 2)
        elif name == "pause":
            pygame.draw.rect(target, color, (x - 7, y - 8, 5, 16), border_radius=2)
            pygame.draw.rect(target, color, (x + 2, y - 8, 5, 16), border_radius=2)
        elif name == "play":
            pygame.draw.polygon(target, color, [(x - 5, y - 9), (x - 5, y + 9), (x + 9, y)])
        elif name == "step":
            pygame.draw.polygon(target, color, [(x - 8, y - 8), (x - 8, y + 8), (x + 4, y)])
            pygame.draw.rect(target, color, (x + 6, y - 8, 3, 16), border_radius=1)
        elif name == "view":
            pygame.draw.ellipse(target, color, (x - 11, y - 6, 22, 12), 2)
            pygame.draw.circle(target, color, (x, y), 3)
        elif name in ("sound", "unmute"):
            pygame.draw.polygon(target, color, [(x - 9, y - 3), (x - 4, y - 3), (x + 2, y - 9), (x + 2, y + 9), (x - 4, y + 3), (x - 9, y + 3)])
            if name == "sound":
                pygame.draw.arc(target, color, (x - 2, y - 6, 12, 12), -1.0, 1.0, 2)
                pygame.draw.arc(target, color, (x - 2, y - 10, 20, 20), -0.9, 0.9, 2)
            else:
                pygame.draw.line(target, color, (x + 5, y - 5), (x + 13, y + 5), 2)
                pygame.draw.line(target, color, (x + 13, y - 5), (x + 5, y + 5), 2)
        else:
            glyph = {"slower": "-", "faster": "+", "help": "?"}[name]
            surface = self._text(self._big_font, glyph, color)
            target.blit(surface, surface.get_rect(center=(x, y - 1)))

    def _draw_topbar(self, target: pygame.Surface, snapshot: RenderSnapshot) -> None:
        width = target.get_width()
        self._topbar_rect = pygame.Rect(8, 6, width - 16, 34)
        target.blit(self._glass(self._topbar_rect.size, 17, (6, 16, 34, 205), (70, 130, 170, 150)), self._topbar_rect.topleft)
        x = self._topbar_rect.x + 16
        title = self._text(self._title_font, "MICROBIOME", self.ACCENT_COLOR)
        target.blit(title, (x, self._topbar_rect.centery - title.get_height() // 2))
        x += title.get_width() + 22
        if self.paused:
            dot, label = self.WARNING_COLOR, "PAUSED"
        else:
            dot, label = (138, 200, 150), f"RUNNING x{self.speed}"
        pygame.draw.circle(target, dot, (x + 5, self._topbar_rect.centery), 5)
        status = self._text(self._small_font, label, self.TEXT_COLOR)
        target.blit(status, (x + 16, self._topbar_rect.centery - status.get_height() // 2))
        x += 16 + status.get_width() + 26
        chips = [f"tick {int(snapshot.tick_index):,}", f"{len(snapshot.agent_id):,} cells"]
        if self.realism:
            stats_of = getattr(self.controller, "realism_stats", None)
            stats = stats_of() if callable(stats_of) else {}
            for key, label in (("spores", "spores"), ("sporulating", "sporulating"), ("starving", "starving"), ("biofilm", "in biofilm")):
                if stats.get(key):
                    chips.append(f"{stats[key]:,} {label}")
            chips.append(f"view: {self.renderer.overlay}")
        if self.mode == "paint":
            nutrient_id = int(snapshot.nutrient_id[self.selected_nutrient])
            nutrient = self._nutrient_name_by_id.get(nutrient_id, str(nutrient_id))
            chips.append(f"brush: {nutrient} {self.brush_concentration_mmol_l:g} mmol/L")
        for text in chips:
            chip = self._text(self._small_font, text, self.MUTED_COLOR)
            target.blit(chip, (x, self._topbar_rect.centery - chip.get_height() // 2))
            x += chip.get_width() + 22

    def _draw_legend(self, target: pygame.Surface, snapshot: RenderSnapshot) -> None:
        y = self._topbar_rect.bottom + 8
        genome_ids = np.asarray(snapshot.genome_id)
        for option in self.genomes:
            palette = self._palette_of(snapshot, int(option.genome_id))
            count = int(np.count_nonzero(genome_ids == np.uint64(option.genome_id)))
            label = self._text(self._small_font, " ".join(option.organism.split()[:2]), self.TEXT_COLOR)
            number = self._text(self._small_font, f"{count:,}", self.ACCENT_COLOR)
            rect = pygame.Rect(12, y, 38 + label.get_width() + 14 + number.get_width() + 12, 26)
            target.blit(self._glass(rect.size, 13, (6, 16, 34, 185), (70, 130, 170, 120)), rect.topleft)
            colour = SpriteAtlas.palette_colors(palette)["mid"]
            pygame.draw.circle(target, colour, (rect.x + 16, rect.centery), 7)
            pygame.draw.circle(target, SpriteAtlas.palette_colors(palette)["light"], (rect.x + 14, rect.centery - 2), 2)
            target.blit(label, (rect.x + 32, rect.centery - label.get_height() // 2))
            target.blit(number, (rect.x + 32 + label.get_width() + 14, rect.centery - number.get_height() // 2))
            y += 30

    def _draw_toolbar(self, target: pygame.Surface, snapshot: RenderSnapshot) -> None:
        width, height = target.get_size()
        play = "play" if self.paused else "pause"
        specs = [
            ("inspect", pygame.K_i, "I", self.mode == "inspect"),
            ("paint", pygame.K_p, "P", self.mode == "paint"),
            ("genome", pygame.K_o, "O", self.mode == "organism"),
            None,
            (play, pygame.K_SPACE, "Space", self.paused),
            ("step", pygame.K_PERIOD, ".", False),
            ("slower", pygame.K_MINUS, "-", False),
            ("faster", pygame.K_EQUALS, "+", False),
            None,
            *((("view", pygame.K_v, "V", self.renderer.overlay != "nutrients"),) if self.realism else ()),
            ("unmute" if self.audio.muted else "sound", pygame.K_m, "M", self.audio.muted),
            ("help", pygame.K_h, "H", self.mode == "help"),
        ]
        button, gap, group_gap = 46, 6, 16
        total = sum(group_gap if spec is None else button + gap for spec in specs) - gap
        extra = []
        if self.mode == "paint":
            for index, (nutrient_id, name) in enumerate(self.nutrients[:9]):
                extra.append((pygame.K_1 + index, name.capitalize(), index == self.selected_nutrient))
            total += group_gap + sum(self._text(self._small_font, label, self.TEXT_COLOR).get_width() + 34 + gap
                                     for _, label, _ in extra) - gap
        bar = pygame.Rect(0, 0, total + 28, 58)
        bar.midbottom = (width // 2, height - 24)
        self._toolbar_rect = bar
        target.blit(self._glass(bar.size, 20, (6, 16, 34, 215), (70, 150, 190, 170)), bar.topleft)
        self._buttons = []
        x = bar.x + 14
        for spec in specs:
            if spec is None:
                pygame.draw.line(target, (70, 110, 140), (x + group_gap // 2 - gap // 2, bar.y + 14),
                                 (x + group_gap // 2 - gap // 2, bar.bottom - 14), 1)
                x += group_gap
                continue
            name, key, hint, active = spec
            rect = pygame.Rect(x, bar.y + 6, button, 46)
            hover = rect.collidepoint(self._pointer)
            fill = (44, 78, 88, 235) if active else ((30, 48, 60, 220) if hover else (16, 26, 36, 205))
            target.blit(self._glass(rect.size, 14, fill, (150, 196, 190, 210) if active else (62, 82, 94, 160)),
                        rect.topleft)
            colour = self.ACCENT_COLOR if active else (self.TEXT_COLOR if hover else (170, 205, 225))
            self._draw_icon(target, name, (rect.centerx, rect.y + 19), colour)
            tag = self._text(self._small_font, hint, self.MUTED_COLOR)
            target.blit(tag, tag.get_rect(center=(rect.centerx, rect.bottom - 8)))
            self._buttons.append((key, rect))
            x += button + gap
        if extra:
            x += group_gap - gap
            for key, label, active in extra:
                text = self._text(self._small_font, label, self.TEXT_COLOR)
                rect = pygame.Rect(x, bar.y + 10, text.get_width() + 34, 38)
                fill = (70, 86, 52, 235) if active else (16, 26, 36, 205)
                target.blit(self._glass(rect.size, 14, fill, (190, 206, 140, 210) if active else (62, 82, 94, 160)),
                            rect.topleft)
                digit = self._text(self._small_font, str(key - pygame.K_0), self.WARNING_COLOR)
                target.blit(digit, (rect.x + 10, rect.centery - digit.get_height() // 2))
                target.blit(text, (rect.x + 24, rect.centery - text.get_height() // 2))
                self._buttons.append((key, rect))
                x += rect.w + gap

    def _draw_population_strip(self, target: pygame.Surface, snapshot: RenderSnapshot) -> None:
        width, height = target.get_size()
        track = pygame.Rect(12, height - 13, width - 24, 4)
        pygame.draw.rect(target, (22, 30, 38), track, border_radius=2)
        fraction = min(1.0, len(snapshot.agent_id) / float(POPULATION_METER_CAP))
        fill = pygame.Rect(track.x, track.y, max(0, int(track.w * fraction)), track.h)
        if fill.w >= 2:
            pygame.draw.rect(target, (112, 158, 140), fill, border_radius=2)
        for step in (0.25, 0.5, 0.75):
            x = track.x + int(track.w * step)
            pygame.draw.line(target, (84, 102, 112), (x, track.y - 2), (x, track.bottom + 1), 1)
        pygame.draw.rect(target, (70, 88, 98), track, width=1, border_radius=2)
        label = self._text(self._small_font, f"population {len(snapshot.agent_id):,} / {POPULATION_METER_CAP:,}", self.MUTED_COLOR)
        target.blit(label, (track.right - label.get_width() - 4, track.y - label.get_height() - 1))

    def _draw_notice(self, target: pygame.Surface) -> None:
        if not self.notice:
            return
        warn = self.notice.startswith(("No ", "Choose "))
        text = self._text(self._small_font, self.notice[:130], self.WARNING_COLOR if warn else self.TEXT_COLOR)
        rect = pygame.Rect(0, 0, text.get_width() + 30, 28)
        rect.midbottom = (target.get_width() // 2, self._toolbar_rect.y - 8)
        edge = (230, 180, 100, 200) if warn else (70, 140, 180, 150)
        target.blit(self._glass(rect.size, 14, (6, 16, 34, 205), edge), rect.topleft)
        target.blit(text, text.get_rect(center=rect.center))

    def _draw_panel(self, target: pygame.Surface) -> None:
        rect = self._panel_rect
        target.blit(self._glass(rect.size, 16, (4, 11, 26, 244), (80, 150, 190, 190)), rect.topleft)
        title = {"help": "Quick controls", "organism": "Introduce a real genome", "inspect": "Cell inspection",
                 "paint": "Nutrient painting"}.get(self.mode, "Cell inspection")
        target.blit(self._text(self._title_font, title, self.ACCENT_COLOR), (rect.x + 18, rect.y + 14))
        pygame.draw.line(target, (66, 98, 104), (rect.x + 16, rect.y + 44), (rect.right - 16, rect.y + 44), 1)

        columns = max(24, (rect.w - 32) // 8)
        key_width = 124
        entries: list[tuple[str, str, str]] = []
        for raw in self._panel_lines():
            head, separator, rest = raw.partition(" \u00b7 ")
            if separator and len(head) <= 20:
                label = {".": "Period", "+ / -": "+  /  -"}.get(head, head)
                wrapped = textwrap.wrap(rest, width=max(12, columns - key_width // 8), break_long_words=True) or [""]
                entries.append(("key", label, wrapped[0]))
                entries.extend(("cont", "", more) for more in wrapped[1:])
            else:
                entries.extend(("text", "", line) for line in self._wrap_lines([raw], columns))
        clip = pygame.Rect(rect.x + 14, rect.y + 52, rect.w - 28, rect.h - 64)
        target.set_clip(clip)
        line_height = 18
        y = clip.y - self.panel_scroll * line_height
        for kind, label, line in entries:
            if y + line_height >= clip.y and y < clip.bottom:
                if kind == "key":
                    target.blit(self._text(self._small_font, label, self.WARNING_COLOR), (clip.x, y))
                    target.blit(self._text(self._small_font, line, self.TEXT_COLOR), (clip.x + key_width, y))
                elif kind == "cont":
                    target.blit(self._text(self._small_font, line, self.TEXT_COLOR), (clip.x + key_width, y))
                else:
                    if line.endswith(":") and not line.startswith(" "):
                        colour = self.ACCENT_COLOR
                    elif line.startswith("  ") or line.startswith(("Source:", "Terms:")):
                        colour = self.MUTED_COLOR
                    else:
                        colour = self.TEXT_COLOR
                    target.blit(self._text(self._small_font, line, colour), (clip.x, y))
            y += line_height
        target.set_clip(None)

    NOTICE_SECONDS = 4.0

    def _draw_minimal(self, target: pygame.Surface) -> None:
        """Only the sample, a scale bar (drawn by the renderer), a transient note and any panel the user opened."""
        width, height = target.get_size()
        self._buttons = []
        self._topbar_rect = pygame.Rect(0, 0, 0, 0)
        self._toolbar_rect = pygame.Rect(0, height, 0, 0)
        light = getattr(self.renderer, "mode", "phase") in ("phase", "brightfield")
        colour = (40, 40, 44) if light else (205, 208, 214)
        y = 14
        if self.paused:
            banner = self._text(self._small_font, "PAUSED", colour)
            target.blit(banner, (16, y))
            y += 20
        if self.notice and time.monotonic() - self._notice_at < self.NOTICE_SECONDS:
            text = self._text(self._small_font, self.notice[:140], colour)
            target.blit(text, (16, y))
        wants_panel = self.mode in {"help", "organism", "paint"} or self.selected_inspection is not None
        if wants_panel:
            self._panel_rect = pygame.Rect(width - self._panel_width() - 10, 10, self._panel_width(), height - 20)
            self._draw_panel(target)
        else:
            self._panel_rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, target: pygame.Surface) -> None:
        self.window_size = target.get_size()
        snapshot = self._current_snapshot()
        self._audio_tick(snapshot)
        self.renderer.draw(target, snapshot)
        self._ensure_fonts()
        width, height = target.get_size()

        if self.minimal_hud:
            self._draw_minimal(target)
            return
        self._draw_topbar(target, snapshot)
        self._draw_legend(target, snapshot)
        self._draw_toolbar(target, snapshot)
        top = self._topbar_rect.bottom + 10
        self._panel_rect = pygame.Rect(
            width - self._panel_width() - 10, top, self._panel_width(), max(80, self._toolbar_rect.y - 10 - top)
        )
        self._draw_panel(target)
        self._draw_population_strip(target, snapshot)
        self._draw_notice(target)
        if self.paused:
            banner = self._text(self._big_font, "PAUSED", self.WARNING_COLOR)
            target.blit(banner, banner.get_rect(midtop=(max(0, self._panel_rect.x) // 2, self._topbar_rect.bottom + 14)))

    def run(self) -> None:
        """Run the interactive window at a 60 Hz render cap."""
        pygame.init()
        pygame.display.set_caption("Microbiome · real-genome sandbox")
        screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        if not self.audio.start():
            self.notice = "No audio device available; running silent."
        self.running = True
        clock = pygame.time.Clock()
        try:
            while self.running:
                for event in pygame.event.get():
                    self.handle_event(event)
                if not self.running:
                    break
                self.advance_frame()
                self.draw(screen)
                pygame.display.flip()
                clock.tick(FPS)
        finally:
            self.audio.stop()
            pygame.quit()


def run_demo(seed: int = 2026, mutation: bool = True, realism: bool = True, microscope: bool = True) -> None:
    """Launch an empty, reproducible field ready for human intervention.

    Mutation and the realism layer are on in the interactive app (rate model ``snv-poisson-2e-10-v1``; mutation is rare
    by design, so a phenotype change in a short session is unlikely). Tests and tools construct the plain D1 world.
    """
    MicrobiomeUI.from_demo(seed=seed, mutation=mutation, realism=realism, microscope=microscope).run()
