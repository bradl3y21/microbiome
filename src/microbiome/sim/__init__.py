"""Deterministic, vectorized microbiome simulation engine."""

from .engine import (
    AgentArrays,
    Environment,
    NutrientField,
    PhenotypeTable,
    RNGStreams,
    Simulation,
    SimulationState,
    add_nutrient_source,
    diffuse,
    initialize_rngs,
    ledger,
    remove_nutrient_sink,
    snapshot,
    step,
    state_digest,
)
from .model import (
    DT_S,
    RNG_ANTAGONISM,
    RNG_DEATH,
    RNG_DIVISION,
    RNG_MOTILITY,
    RNG_MUTATION,
    AgentState,
    NutrientEnvironment,
    NutrientLedger,
    Phenotype,
    make_rngs,
)
from .spatial import SpatialHash


def __getattr__(name: str):
    """Load the facade lazily to avoid a genome/phenotype package cycle."""
    if name == "SimulationController":
        from .controller import SimulationController

        return SimulationController
    raise AttributeError(name)

__all__ = [
    "DT_S",
    "RNG_ANTAGONISM",
    "RNG_DEATH",
    "RNG_DIVISION",
    "RNG_MOTILITY",
    "RNG_MUTATION",
    "AgentState",
    "AgentArrays",
    "Environment",
    "NutrientEnvironment",
    "NutrientField",
    "NutrientLedger",
    "Phenotype",
    "PhenotypeTable",
    "RNGStreams",
    "Simulation",
    "SimulationController",
    "SimulationState",
    "SpatialHash",
    "add_nutrient_source",
    "diffuse",
    "initialize_rngs",
    "ledger",
    "make_rngs",
    "remove_nutrient_sink",
    "snapshot",
    "step",
    "state_digest",
]
