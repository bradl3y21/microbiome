"""The checkers in sim_support must be able to fail before they judge the engine.

Independent of the engine: runs on synthetic dataclasses.
"""
import dataclasses

import numpy as np
import pytest

from sim_support import (
    collect, find, ledger_ok, ledger_residuals, snapshot_digest, violations,
)


@dataclasses.dataclass
class Agents:
    agent_id: np.ndarray
    position_um: np.ndarray
    velocity_um_s: np.ndarray
    mass_pg: np.ndarray
    radius_um: np.ndarray
    energy_pj: np.ndarray
    species_id: np.ndarray
    genome_id: np.ndarray
    age_ticks: np.ndarray
    nutrient_store_mmol: np.ndarray
    biomass_mmol: np.ndarray


@dataclasses.dataclass
class State:
    agents: Agents
    tick_index: np.uint64
    next_agent_id: np.uint64


@dataclasses.dataclass
class Env:
    width_um: float
    height_um: float
    concentration_mmol_l: np.ndarray
    detritus_mmol: np.ndarray
    molar_mass_g_mol: np.ndarray


@dataclasses.dataclass
class Ledger:
    initial_mmol: np.ndarray
    field_mmol: np.ndarray
    agents_mmol: np.ndarray
    detritus_mmol: np.ndarray
    source_cum_mmol: np.ndarray
    sink_cum_mmol: np.ndarray
    respiration_loss_mmol: np.ndarray


def good(n=4, K=2):
    molar = np.array([180.0, 60.0])
    bio = np.full((n, K), 1e-12)
    a = Agents(
        agent_id=np.arange(1, n + 1, dtype=np.uint64),
        position_um=np.full((n, 2), 5.0), velocity_um_s=np.zeros((n, 2)),
        mass_pg=(bio * molar * 1e9).sum(axis=1), radius_um=np.full(n, 0.5),
        energy_pj=np.ones(n), species_id=np.zeros(n, np.uint32),
        genome_id=np.ones(n, np.uint64), age_ticks=np.zeros(n, np.uint64),
        nutrient_store_mmol=np.zeros((n, K)), biomass_mmol=bio,
    )
    s = State(a, np.uint64(0), np.uint64(n + 1))
    e = Env(10.0, 10.0, np.ones((K, 3, 3)), np.zeros((K, 3, 3)), molar)
    return s, e


def test_a_clean_state_has_no_violations():
    assert violations(*good()) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda s, e: s.agents.mass_pg.__setitem__(0, np.nan), "non-finite"),
    (lambda s, e: s.agents.agent_id.__setitem__(slice(0, 2), np.array([2, 1], np.uint64)), "strictly increasing"),
    (lambda s, e: s.agents.position_um.__setitem__((0, 0), 10.0), "position outside"),
    (lambda s, e: s.agents.position_um.__setitem__((0, 1), -1e-9), "position outside"),
    (lambda s, e: e.concentration_mmol_l.__setitem__((0, 0, 0), -1e-30), "negative concentration"),
    (lambda s, e: e.detritus_mmol.__setitem__((0, 0, 0), -1e-30), "negative detritus"),
    (lambda s, e: s.agents.energy_pj.__setitem__(0, -1.0), "negative energy"),
    (lambda s, e: s.agents.radius_um.__setitem__(0, 0.0), "radius"),
    (lambda s, e: s.agents.mass_pg.__setitem__(1, s.agents.mass_pg[1] * (1 + 1e-9)), "mass_pg !="),
    (lambda s, e: setattr(s, "next_agent_id", np.uint64(3)), "id would be reused"),
])
def test_each_impossible_state_is_caught(mutate, needle):
    s, e = good()
    mutate(s, e)
    assert any(needle in v for v in violations(s, e)), violations(s, e)


def test_row_count_mismatch_is_caught():
    s, e = good()
    s.agents.age_ticks = s.agents.age_ticks[:-1]
    assert any("rows" in v for v in violations(s, e))


def balanced():
    z = np.zeros(2)
    return Ledger(initial_mmol=np.array([1e-9, 2e-9]), field_mmol=np.array([6e-10, 1.5e-9]),
                  agents_mmol=np.array([3e-10, 4e-10]), detritus_mmol=np.array([1e-10, 1e-10]),
                  source_cum_mmol=z.copy(), sink_cum_mmol=z.copy(), respiration_loss_mmol=z.copy())


def test_balanced_ledger_passes_and_a_leak_is_caught():
    assert ledger_ok(balanced())
    leaky = balanced()
    leaky.agents_mmol = leaky.agents_mmol - np.array([0.0, 1e-12])  # 0.05% of species 1
    assert not ledger_ok(leaky)
    assert ledger_residuals(leaky)[0][1] == pytest.approx(1e-12, rel=1e-3)


def test_ledger_source_and_sink_are_booked():
    lg = balanced()
    lg.source_cum_mmol = np.array([5e-10, 0.0])
    assert not ledger_ok(lg)  # extra source with nowhere to go
    lg.field_mmol = lg.field_mmol + np.array([5e-10, 0.0])
    assert ledger_ok(lg)


def test_digest_sees_nested_changes_rng_state_and_scalars():
    s, e = good()
    rng = np.random.default_rng(1)
    base = snapshot_digest(s, e, rng)
    assert base == snapshot_digest(*good(), np.random.default_rng(1))
    s.agents.position_um[2, 1] = np.nextafter(5.0, 6.0)
    assert snapshot_digest(s, e, rng) != base
    s2, e2 = good()
    rng.random()
    assert snapshot_digest(s2, e2, rng) != base  # RNG state is covered
    s3, e3 = good()
    s3.tick_index = np.uint64(1)
    assert snapshot_digest(s3, e3, np.random.default_rng(1)) != base
    e4 = good()[1]
    assert snapshot_digest(s2, e4, None) == snapshot_digest(*good(), None)  # rngs=None excludes them


def test_digest_refuses_state_it_cannot_see():
    with pytest.raises(TypeError):
        collect({"x": object()}, "state")
    with pytest.raises(TypeError):
        collect({"x": np.array([object()], dtype=object)}, "state")


def test_find_requires_exactly_one_match():
    c = {"state.a.agent_id": np.zeros(1), "env.agent_id": np.zeros(1)}
    with pytest.raises(LookupError):
        find(c, "agent_id")


def test_naive_uptake_update_goes_negative_but_mmol_space_form_never_does():
    rng = np.random.default_rng(1)
    n = 400_000
    c = rng.random(n) * 10.0
    V = (rng.random(n) * 50 + 1) * (rng.random(n) * 50 + 1) * (rng.random(n) * 20 + 1) * 1e-15
    grant = c * V  # D1 cap: take everything available, exact in mmol
    naive = c - grant / V
    exact = (c * V - grant) / V
    assert (naive < 0).sum() > 0, "the defect did not reproduce"
    assert (exact < 0).sum() == 0 and (exact == 0).all()
