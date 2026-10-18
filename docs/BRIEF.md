# Microbiome — brief and decisions

A 2D microbiome simulator. Real bacterial genomes define the organisms; an
agent-based simulation runs thousands of them interacting in a shared
environment; you watch it in the visual language of SPORE's cell stage.

Decided by the human on 2026-10-17. **These are constraints, not suggestions.**

| # | Decision |
| - | -------- |
| P1 | **Python + numpy**, rendered with **pygame**. No web app, no C++. |
| P2 | **Fully real genomes.** Organisms are derived from actual sequenced genomes and their annotations — not a synthetic DNA metaphor. |
| P3 | **Simulation-first, deterministic.** Target ~10,000 simultaneous agents. Seedable: the same seed and the same code must produce the same run, bit for bit where floats allow. It has to be testable, not just pretty. |
| P4 | **2D rendering, SPORE cell-stage look.** Organisms are visible cells; the environment is a 2D field. |

## What the application must do

1. **Create organisms from real DNA.** Load a real genome, read its annotated
   genes, and derive a phenotype from them through a documented gene → trait
   table. Two organisms with different genomes must behave differently for
   reasons you can point at.
2. **Run a microbiome.** Thousands of agents in one environment: nutrient
   uptake, growth, division, death, chemotaxis, competition and at least one
   form of antagonism. Population-level behaviour must emerge from per-agent
   rules, not be scripted.
3. **Let a human intervene.** Place nutrients, introduce an organism of your
   own making, inspect any cell's genome and traits while it runs.
4. **Render it in 2D.** A field you can pan and zoom, cells drawn as visible
   organisms, density readable at a glance.

## Performance target (measured, not assumed)

10,000 agents at 60 fps on a normal desktop. If pure-Python per-agent loops
cannot reach that, the answer is vectorised numpy arrays and batched pygame
blits — measured with a benchmark in `tests/` or `bench/`, with the number in
the finding. "It runs" is not a performance claim.

## Non-negotiables

- **Determinism.** One seeded RNG stream per purpose, fixed timestep, no
  dependence on dict ordering or wall-clock time inside the step. A run must
  reproduce exactly from its seed, and a test must prove it.
- **Provenance for the genome data.** Every sequence file records where it came
  from (accession), when it was fetched, and its licence/terms. Real data means
  answering "which genome is this" without guessing.
- **No invented biology.** If a trait mapping is a simplification, say so in the
  table and in `docs/`. A documented approximation is fine; a silent one is not.
