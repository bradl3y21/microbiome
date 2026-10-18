"""Nutrient-ledger accounting on the first tick of a pristine world.

FIXED: the controller now tracks "ledger initialised" explicitly instead of inferring it from
``initial_mmol`` being nonzero (an empty world legitimately has initial_mmol == 0). The defect
description below is kept as the record of what was measured.

DEFECT (measured by Assay, 2026-10-17). Two or more ``place_nutrient`` commands
applied in the *same* tick while the demo world is still at ``tick_index == 0``
raise ``RuntimeError: nutrient ledger failed conservation: residual=array([1.e-12, 0.e+00]) mmol``
from ``microbiome.sim.engine._contract_step`` -> ``NutrientLedger.assert_conserved``.

Exact API reproduction (no pygame, no UI)::

    from microbiome.sim.controller import SimulationController
    c = SimulationController.create_demo(seed=2026)
    c.place_nutrient(0, (5.0, 5.0), 1e-12)     # one field cell of glucose
    c.place_nutrient(1, (5.0, 5.0), 1e-12)     # a second placement in the same tick
    c.advance_ticks(1)                          # -> RuntimeError

Measured boundary (``tests``-style probe, seed-independent):

| placements queued for one tick | world at tick 0 | world after >=1 tick |
| ------------------------------ | --------------- | -------------------- |
| 0 or 1                         | ok              | ok                   |
| 2 or more                      | RuntimeError    | ok                   |

The residual equals exactly the *first* placement's amount, which is a double
count, not float noise: the first placement both credits ``source_cum_mmol`` and
raises ``concentration_mmol_l``, and only then does ``_initialize_ledger`` run and
set ``initial_mmol`` to the (already credited) field content. The mismatch is
possible because ``SimulationController._apply_command`` derives
``ledger_initialized`` from ``bool(np.any(field.initial_mmol) or tick_index > 0)``
*per command*, so within tick 0 the first command sees ``False`` and the second
sees ``True``.

Ownership: the fault is in ``src/microbiome/sim/**`` (Cilia · Sim). Assay owns only
``tests/**`` and is recording the defect, not fixing it.

Why the strict xfail: it goes XPASS-red the moment the accounting is fixed, which
forces the mark to be deleted. Loosening ``assert_conserved``'s tolerance is *not*
a fix — the residual is a whole placement, i.e. missing bookkeeping; the
conservation identity must hold for the full amount, not within a wider epsilon.
"""
import numpy as np
import pytest

from microbiome.sim.controller import SimulationController

AMOUNT_MMOL = 1e-12  # one 10x10x10 um field cell of glucose at 1 mmol/L


def _two_placements_at_tick_zero():
    controller = SimulationController.create_demo(seed=2026)
    assert int(controller.render_snapshot().tick_index) == 0
    controller.place_nutrient(0, (5.0, 5.0), AMOUNT_MMOL)
    controller.place_nutrient(1, (5.0, 5.0), AMOUNT_MMOL)
    return controller


def test_two_placements_in_the_first_tick_are_conserved_and_both_land():
    controller = _two_placements_at_tick_zero()
    controller.advance_ticks(1)
    snap = controller.render_snapshot()
    field_total_mmol = float(np.sum(snap.concentration_mmol_l)) * 1e-12  # cell volume, litres
    assert field_total_mmol == pytest.approx(2.0 * AMOUNT_MMOL, rel=1e-9, abs=0.0)


def test_one_placement_in_the_first_tick_lands_and_is_conserved():
    """The working half of the boundary: a single first-tick placement must keep working."""
    controller = SimulationController.create_demo(seed=2026)
    controller.place_nutrient(0, (5.0, 5.0), AMOUNT_MMOL)
    controller.advance_ticks(1)
    snap = controller.render_snapshot()
    assert int(snap.tick_index) == 1
    # Diffusion redistributes within the tick, so check the field total, not one cell.
    assert float(np.sum(snap.concentration_mmol_l)) == pytest.approx(1.0, rel=1e-9)


def test_two_placements_after_the_world_has_stepped_are_conserved():
    """The same batch is fine once tick_index > 0; the fix must not break this path."""
    controller = SimulationController.create_demo(seed=2026)
    controller.advance_ticks(1)
    controller.place_nutrient(0, (5.0, 5.0), AMOUNT_MMOL)
    controller.place_nutrient(1, (5.0, 5.0), AMOUNT_MMOL)
    controller.advance_ticks(1)
    snap = controller.render_snapshot()
    assert int(snap.tick_index) == 2
    assert float(np.sum(snap.concentration_mmol_l)) == pytest.approx(2.0, rel=1e-9)


def _total_mmol(snap):
    return float(np.sum(snap.concentration_mmol_l))


@pytest.mark.parametrize("count", [2, 3, 7])
def test_any_number_of_first_tick_placements_are_conserved(count):
    controller = SimulationController.create_demo(seed=2026)
    for index in range(count):
        controller.place_nutrient(index % 2, (5.0 + 10.0 * index, 5.0), AMOUNT_MMOL)
    controller.advance_ticks(3)
    assert _total_mmol(controller.render_snapshot()) == pytest.approx(float(count), rel=1e-9)


@pytest.mark.parametrize("order", ["place_then_introduce", "introduce_then_place", "introduce_twice_then_place"])
def test_first_tick_mixed_commands_are_conserved(order):
    """Introductions credit source_cum_mmol too, so every ordering must step without a ledger error."""
    controller = SimulationController.create_demo(seed=2026)
    genome_id = int(controller.list_genomes()[0].genome_id)
    steps = {
        "place_then_introduce": ["place", "introduce"],
        "introduce_then_place": ["introduce", "place"],
        "introduce_twice_then_place": ["introduce", "introduce", "place", "place"],
    }[order]
    for step in steps:
        if step == "place":
            controller.place_nutrient(0, (5.0, 5.0), AMOUNT_MMOL)
        else:
            controller.introduce_organism(genome_id, (50.0, 50.0))
    controller.advance_ticks(3)                 # must not raise RuntimeError from assert_conserved
    snap = controller.render_snapshot()
    assert len(snap.agent_id) == steps.count("introduce")


def test_the_fix_is_deterministic():
    totals = []
    for _ in range(2):
        controller = SimulationController.create_demo(seed=2026)
        controller.place_nutrient(0, (5.0, 5.0), AMOUNT_MMOL)
        controller.introduce_organism(int(controller.list_genomes()[0].genome_id), (50.0, 50.0))
        controller.place_nutrient(1, (95.0, 5.0), AMOUNT_MMOL)
        controller.advance_ticks(20)
        snap = controller.render_snapshot()
        totals.append((snap.concentration_mmol_l.tobytes(), snap.position_um.tobytes()))
    assert totals[0] == totals[1]