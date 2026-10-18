"""Pygame rendering for the microbiome field.

Rendering is deliberately a consumer of simulation state: it never advances
the fixed-step model or mutates the state/environment arrays.
"""

from .atlas import SpriteAtlas
from .camera import Camera2D
from .lens import MicroscopeRenderer
from .renderer import PygameRenderer, RenderStats, Renderer
from .view import WorldView

__all__ = ["Camera2D", "MicroscopeRenderer", "PygameRenderer", "RenderStats", "Renderer", "SpriteAtlas", "WorldView"]
