"""World/screen camera with pointer-anchored zoom and middle-button panning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pygame


class Camera2D:
    """A small camera for a micrometre-space field.

    Coordinates are continuous world micrometres. ``zoom`` is pixels per
    micrometre. Wheel zoom keeps the world point under the pointer stationary;
    dragging with the middle mouse button moves the view.
    """

    def __init__(
        self,
        field_size_um: tuple[float, float] | None = None,
        viewport_size: tuple[int, int] = (800, 600),
        zoom: float | None = None,
        min_zoom: float = 0.01,
        max_zoom: float = 100.0,
    ) -> None:
        if min_zoom <= 0 or max_zoom < min_zoom:
            raise ValueError("zoom bounds must be positive and ordered")
        self.min_zoom = float(min_zoom)
        self.max_zoom = float(max_zoom)
        self._zoom_explicit = zoom is not None
        self.zoom = float(zoom) if zoom is not None else 1.0
        self._validate_zoom(self.zoom)
        self.viewport_size = (max(1, int(viewport_size[0])), max(1, int(viewport_size[1])))
        self.field_size_um: tuple[float, float] | None = None
        self.center_um = np.zeros(2, dtype=np.float64)
        self._dragging = False
        self._last_pointer: tuple[float, float] | None = None
        if field_size_um is not None:
            self.set_field_size(field_size_um)

    def _validate_zoom(self, zoom: float) -> None:
        if not np.isfinite(zoom) or zoom <= 0:
            raise ValueError("zoom must be a finite positive number")

    def set_field_size(self, size_um: tuple[float, float]) -> None:
        width, height = float(size_um[0]), float(size_um[1])
        if not np.isfinite(width + height) or width <= 0 or height <= 0:
            raise ValueError("field dimensions must be finite and positive")
        first_size = self.field_size_um is None
        self.field_size_um = (width, height)
        if first_size:
            self.center_um[:] = (width * 0.5, height * 0.5)
            if not self._zoom_explicit:
                self._fit_zoom()

    def resize(self, viewport_size: tuple[int, int]) -> None:
        self.viewport_size = (max(1, int(viewport_size[0])), max(1, int(viewport_size[1])))

    def _fit_zoom(self) -> None:
        if self.field_size_um is None:
            return
        width, height = self.field_size_um
        view_width, view_height = self.viewport_size
        self.zoom = float(np.clip(0.92 * min(view_width / width, view_height / height), self.min_zoom, self.max_zoom))

    def world_to_screen(self, point_um: Any) -> tuple[float, float] | np.ndarray:
        points = np.asarray(point_um, dtype=np.float64)
        view_center = np.asarray(self.viewport_size, dtype=np.float64) * 0.5
        result = view_center + (points - self.center_um) * self.zoom
        if result.shape == (2,):
            return float(result[0]), float(result[1])
        return result

    def screen_to_world(self, point_px: Any) -> tuple[float, float] | np.ndarray:
        points = np.asarray(point_px, dtype=np.float64)
        view_center = np.asarray(self.viewport_size, dtype=np.float64) * 0.5
        result = self.center_um + (points - view_center) / self.zoom
        if result.shape == (2,):
            return float(result[0]), float(result[1])
        return result

    def visible_world_rect(self) -> tuple[float, float, float, float]:
        half_width = self.viewport_size[0] / (2.0 * self.zoom)
        half_height = self.viewport_size[1] / (2.0 * self.zoom)
        return (
            float(self.center_um[0] - half_width),
            float(self.center_um[1] - half_height),
            float(self.center_um[0] + half_width),
            float(self.center_um[1] + half_height),
        )

    def pan_by_screen(self, dx_px: float, dy_px: float) -> None:
        self.center_um[0] -= float(dx_px) / self.zoom
        self.center_um[1] -= float(dy_px) / self.zoom

    def zoom_at(self, point_px: tuple[float, float], factor: float) -> None:
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("zoom factor must be finite and positive")
        anchor_um = np.asarray(self.screen_to_world(point_px), dtype=np.float64)
        self.zoom = float(np.clip(self.zoom * factor, self.min_zoom, self.max_zoom))
        pointer = np.asarray(point_px, dtype=np.float64)
        view_center = np.asarray(self.viewport_size, dtype=np.float64) * 0.5
        self.center_um[:] = anchor_um - (pointer - view_center) / self.zoom

    def handle_event(
        self,
        event: pygame.event.Event,
        pointer_position: tuple[int, int] | None = None,
    ) -> bool:
        """Handle pan/zoom input. Returns whether the event was consumed."""
        event_type = event.type
        if event_type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) == 2:
            self._dragging = True
            self._last_pointer = tuple(getattr(event, "pos", pointer_position or (0, 0)))
            return True
        if event_type == pygame.MOUSEBUTTONUP and getattr(event, "button", None) == 2:
            self._dragging = False
            self._last_pointer = None
            return True
        if event_type == pygame.MOUSEMOTION and self._dragging:
            pos = tuple(getattr(event, "pos", pointer_position or (0, 0)))
            if self._last_pointer is not None:
                self.pan_by_screen(pos[0] - self._last_pointer[0], pos[1] - self._last_pointer[1])
            self._last_pointer = pos
            return True
        if event_type == getattr(pygame, "MOUSEWHEEL", -1):
            pos = pointer_position
            if pos is None and hasattr(event, "mouse_x") and hasattr(event, "mouse_y"):
                pos = (int(event.mouse_x), int(event.mouse_y))
            if pos is None:
                pos = pygame.mouse.get_pos()
            self.zoom_at(pos, 1.2 ** int(getattr(event, "y", 0)))
            return True
        if event_type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) in (4, 5):
            pos = tuple(getattr(event, "pos", pointer_position or pygame.mouse.get_pos()))
            factor = 1.2 if event.button == 4 else 1.0 / 1.2
            self.zoom_at(pos, factor)
            return True
        return False
