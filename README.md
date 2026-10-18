# Microbiome

A 2D microbiome simulator. Real bacterial genomes define the organisms, a deterministic agent-based simulation runs them in a shared chemical environment, and you watch it in a microscope-style view (phase contrast, Gram-stain brightfield, darkfield, fluorescence).

> **Status: an exploratory simulator and teaching tool, not a research instrument.** Parts of it are calibrated against published single-cell numbers, but nothing has been validated against population-level data. It was built by AI coding agents working for one person, and the checks were run by those same agents; nobody outside that workflow has reproduced any of it. Read [`docs/CALIBRATION.md`](docs/CALIBRATION.md) and [`docs/ORGANISMS.md`](docs/ORGANISMS.md), which say what is matched by construction, what is held out, what is assumed and what is not reproduced, before relying on any output.

## What it does

- **Five real organisms**, each a complete NCBI RefSeq assembly listed in `data/manifest.json` with hashes and provenance: *Escherichia coli* K-12 MG1655, *Bacillus subtilis* 168, *Pseudomonas aeruginosa* PAO1, *Staphylococcus aureus* Newman and *Myxococcus xanthus* DK 1622.
- **Genes decide traits.** Annotated genes map to physiology through versioned tables (`data/*.tsv`, rules in `src/microbiome/genome/`). Removing a required gene removes the trait.
- **A deterministic engine.** Same seed, same run, bit for bit. Every nutrient channel is conserved to rounding under an assertion that runs every tick. About 10,000 agents at 60 fps in the plain world on the author's machine (that benchmark predates the added organisms and the extra chemical channel).
- **A realism layer** (`src/microbiome/sim/realism.py`, [`docs/REALISM.md`](docs/REALISM.md)): run-and-tumble chemotaxis, rod-shaped cells with contact mechanics, oxygen, acetate overflow, catabolite repression, toxins, quorum signalling, biofilm matrix, sporulation, natural transformation, gliding, contact predation, staphylococcal clusters.
- **Mutation** (opt-in): daughter SNVs from the real sequence, and natural transformation moving them between cells.
- **A calibration module** (`python -m microbiome.calibration`) with 52 targets (28 fitted, 5 held out, 7 range checks, 6 references, 6 assumptions), each recorded with its role and, except the assumptions, its published source, plus a fit and a report that tests keep in sync.

## What it does not do

- It does not reproduce *M. xanthus* fruiting-body formation (many variants were tried and failed), its growth on casitone, or the published development timeline.
- Population dynamics, competition and ecology are **not** validated against data. Many parameters remain uncalibrated guesses (listed in the calibration document).
- Maintenance energy for three organisms is a placeholder, and several literature values mix strains and conditions.
- The world is a 2D film on a 10 um grid with fixed 1/60 s steps.

## Run it

Python 3.11 or newer. It is developed on Python 3.15 with `pygame-ce` (plain `pygame` has no wheel there).

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m microbiome            # the interactive microscope view
```

Keys in the viewer: mouse wheel for the objective, `B` imaging mode, `Z`/`X` focus, `Tab` instrument panel, `H` help. Speed up with the toolbar to see growth and predation; calibrated *E. coli* and *B. subtilis* divide about once an hour.

## Tests

```
pip install pytest
SDL_VIDEODRIVER=dummy python -m pytest -q          # about 9 minutes
```

The last complete run before publication was 616 passed, 1 xfailed in about 10 minutes. The one expected failure (`xfail`) is a measured test-power gap documented in `PROJECT_STATUS.md`. Windows paths and PowerShell were used throughout development; other platforms are untested.

## Where to read next

| Document | What it is |
| --- | --- |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | what is done, how it was verified, what is not |
| [`docs/ORGANISMS.md`](docs/ORGANISMS.md) | the five organisms and what is not reproduced |
| [`docs/CALIBRATION.md`](docs/CALIBRATION.md) | every published target and how well the model meets it (generated) |
| [`docs/REALISM.md`](docs/REALISM.md) | the physics, every parameter, the microscope view |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | module layout and the frozen interfaces |
| [`docs/BRIEF.md`](docs/BRIEF.md) | the original project brief |

## Data and licence

Genome sequences and annotations in `data/genomes/` come from NCBI RefSeq via NCBI Datasets and are redistributed with the terms notice recorded in `data/manifest.json`. **No licence has been chosen for the code yet**, so by default all rights are reserved until one is added.
