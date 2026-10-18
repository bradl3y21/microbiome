"""The single UI-to-simulation command dispatcher for the frozen D3 API."""

from __future__ import annotations

from typing import Any

from microbiome.core.ui_api import SimulationController


def dispatch_command(controller: SimulationController, action: str, **payload: Any) -> Any:
    """Dispatch one human action through the D3 simulation facade.

    The UI never edits simulation state itself. Mutating actions enqueue commands
    or advance the controller; inspection actions return detached D3 DTOs.
    """
    if action == "place_nutrient":
        return controller.place_nutrient(**payload)
    if action == "introduce_organism":
        return controller.introduce_organism(**payload)
    if action == "inspect_cell":
        return controller.inspect_cell(**payload)
    if action == "inspect_genome":
        return controller.inspect_genome(**payload)
    if action == "preview_phenotype":
        return controller.preview_phenotype(**payload)
    if action == "command_result":
        return controller.command_result(**payload)
    if action == "set_paused":
        return controller.set_paused(**payload)
    if action == "advance_ticks":
        return controller.advance_ticks(**payload)
    if action == "single_step":
        return controller.single_step(**payload)
    raise ValueError(f"unknown simulation action: {action!r}")


__all__ = ["dispatch_command"]
