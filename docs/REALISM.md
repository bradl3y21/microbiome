# The realism layer (D5)

The frozen D1 step models uptake, growth, division, death, a nutrient field and a contact antagonism hazard. It
leaves out most of what makes a microbial community behave like one. The realism layer adds that, **opt-in**:

```python
SimulationController.create_demo(realism=True)          # seven chemical channels, the physics below
SimulationController.create_demo(realism=True, mutation=True)   # also natural transformation
```

The interactive app (`python -m microbiome`) turns both on. Tests and tools build the plain D1 world, which is
unchanged bit for bit: the realism layer is a separate per-tick implementation (`sim/realism.py`,
`RealisticSimulation`) that reuses the engine's allocator, ledger, division and detritus code, so **every mechanism
below runs under the same per-tick, per-channel mass-conservation assertion** as the original. Growth, uptake,
swimming and chemotaxis are calibrated against published single-organism measurements (`docs/CALIBRATION.md` lists
every target, its source, whether it was fitted or held out, and the mismatches); every other parameter is still an
explicit, named calibration with no measurement behind it, and the gene-to-trait rules say so in every row.

## What it adds, and the one-line reason it matters

| Not modelled before | Now |
| --- | --- |
| Straight-line chemotaxis, no noise | Run-and-tumble with temporal sensing and adaptation, plus translational and rotational Brownian motion |
| Cells are points that overlap freely | True rods (a capsule of fixed width whose length follows mass); capsule contact mechanics; daughters lie end to end; matrix-producing cells attract |
| Uptake at a fixed rate per cell until the sugar is gone | Michaelis-Menten saturation, proportional to dry mass (exponential growth), with Pirt maintenance taken first |
| One sugar at a time, no regulation | Catabolite repression with a diauxic lag; acetate overflow and re-uptake (the acetate switch) |
| No oxygen, no pH | Dissolved oxygen with gas exchange; aerobic vs anaerobic yield and fermentation; pH from acetate with acid damage |
| Contact-only antagonism | A toxin that is produced, diffuses, decays and kills by dose |
| No cell-cell communication, no biofilm | An AI-2-like signal; quorum-triggered matrix with cohesion and protection |
| No lag, no starvation, immortal storage | Lag phase, maintenance energy, starvation stress and death, storage burning |
| No dormancy | Sporulation (8 h) and germination; spores are small, immobile, resistant |
| No nutrient recycling | Dead cells return their contents to the field |
| Still water, nutrient diffusion of 10 um^2/s | Realistic diffusivities; optional uniform flow that carries cells and solutes |
| Mutation only | Natural transformation: competent cells recombine DNA released by dead neighbours of their own species |
| 10 genes with effects per organism | 72 gene-to-physiology rules over real operons, same D2 rules (exact locus tags, all-of operons, evidence) |

## Channels

Seven field channels, ids 0-6: **glucose, lactose, oxygen, acetate, bacilysin, ai2, pyocyanin**. Diffusivities (um^2/s) are
600, 450, 1400, 1100, 400, 500, 400 (pyocyanin's is an assumption). The oxygen value is the *effective* value in dense cell layers (free water is
about 2000); it is also the largest value the explicit scheme can keep stable on the 10 um grid. The gene-to-trait
table that D2 defines knows the first two; oxygen, acetate, toxin and signal are appended to the phenotype table
(`genome.physiology.extend_phenotype_table`), and a variant's phenotype row is widened the same way.

## Mechanisms

**Movement.** Each cell has a heading. It senses the sugars it can use (bilinear interpolation of the field at its
position), keeps an adapted memory `M` (`dM/dt = (C - M) / tau_a`), and tumbles at rate
`lambda = lambda0 * exp(-g * s * ln((C+K)/(M+K)))` (clipped to 0.05-5 lambda0), where `s` is the genome's D2
chemotaxis sensitivity and `g` the gain. A tumble turns by a clipped normal angle (mean 68 degrees). Between tumbles
the heading diffuses (`D_r`), and the position gets a Brownian step (`sqrt(2 D_t dt)`). Speed is the D2 motility
coefficient times `swim_speed_scale`, reduced by the matrix (`x (1 - 0.85 eps)`), zero for spores and for cells with
no energy. Overlapping cells are pushed apart (position-based relaxation along the contact normal, deterministic pair
order); matrix-producing cells within `cohesion_range` of contact are drawn together. A cell is a capsule of fixed width
(`cell_width_um`, 0.8 um) whose length follows its mass (a volume of `cell_volume_um3_per_pg` per pg): a 0.3 pg founder is
2.2 um long, a 0.6 pg cell about to divide 4.2 um. Contact is between the closest points of the two rods' axes (exact
segment-segment distance, tested against brute force); a dividing rod splits across its short axis, so the daughters lie
end to end along the mother's axis. Only flagellated cells tumble.

**Chemistry.** Uptake of sugar `k` per tick is `Vmax_k dt * S/(Km+S) * gate * (mass / reference_mass)`, with `gate = lag * pH factor * (1 - spore)`;
lactose and acetate are further multiplied by the cell's *derepression level*, which relaxes toward
`1 - ccr * glucose/(glucose + K)` (fast down, slow up: the diauxic lag). Oxygen demand is
`6 x respired glucose + 12 x lactose + 2 x acetate` (complete oxidation). The oxygen the cell actually receives sets
`a` (aerobic fraction): sugar requests are scaled by `a + (1-a) x fermentation_capacity` (a strict aerobe stops without
oxygen), biomass yield by `a + (1-a) x 0.3`, and the un-oxidised carbon is excreted as acetate. Even aerobically a
fraction of respired carbon overflows as acetate in proportion to the sugar level (the Pta-AckA pathway). Acetate
carbon never exceeds the carbon respired (`2 x acetate <= 6 x respired glucose + 12 x respired lactose`; tested).
Acetate sets the pH, `pH = 7 - log10(1 + acetate/20 mM)`; growth follows a saturating curve of pH between the
organism's minimum and 6.8 (about 90% of maximum at pH 6, 55% at 5, 10% at 4.5 for an acid-resistant *E. coli*), and
below the minimum cells die at a rate proportional to the shortfall. A producer secretes toxin and signal into its
grid cell; both decay first-order; toxin kills at `h_max * C^2/(K^2 + C^2) * susceptibility * (1 - 0.7 eps)`.

**Physiology.** A new cell starts at 5% growth readiness (`lag`), recovering with a 20 min time constant when food
is present. Maintenance follows Pirt: each organism has a coefficient in glucose equivalents per gram of dry mass per hour
(0.39 measured for B. subtilis, 0.37 derived for E. coli); that share of what a cell eats is respired before anything is stored
or built, so the uptake needed for zero growth equals the coefficient and `q = mu / Y + m` holds (tested). Any shortfall is paid from a 2 h
energy reserve; when that runs out the cell burns
stored substrate (booked as respiration), then accumulates starvation stress (`1 - exp(-t/tau)`), which raises its
death rate (reduced by stress-resistance genes) and stops division. A sporulation-capable cell whose stress passes
0.6 (and whose quorum is active, or stress passes 0.9) takes 8 h to become a spore; a spore does not move, eat or
divide, dies 1000 times more slowly, and germinates in sugar. Quorum activation (Hill, n = 2, on the signal) raises
the matrix level `eps` toward the genome's biofilm capacity. Division needs the D1 mass threshold, lag >= 0.5, stress
< 0.5 and a tolerable pH, then follows after a mean `division_commitment_s`: the doubling time is set by growth,
not by a fixed hazard. Radius and length now follow mass (the D1 step never grew either).

**Transformation.** A cell that dies leaves its genome and variant in its grid cell for 30 min. A cell with the
competence genes, starving or at high quorum, may take up one such fragment of its own species per attempt and
recombine a 50 kb tract (a deterministic window from ids and tick, never a random stream): wherever the donor's base
differs from its own it takes the donor's, so it can gain a donor mutation or repair its own. The result is an
ordinary derived-variant record (`transformation-snv-v1`) whose phenotype is re-derived by the unchanged D2 rules, so
a transferred stop codon can remove a trait. Cross-species transfer and plasmids are not representable: neither
assembly carries a plasmid, and D4 forbids inserted sequence.

## Genes to physiology

`data/physiology_schema.tsv` and `data/physiology_map.tsv` are the D2-style versioned tables for these traits,
regenerated with `python -m microbiome.genome.physiology export` from the rules in `genome/physiology.py`. Every
rule names genes by symbol; the exporter resolves them against the real annotation of *that* assembly and refuses a
missing or ambiguous match. That is not a formality: `bacA` is bacilysin synthesis in *B. subtilis* but undecaprenyl
pyrophosphatase in *E. coli*, and *B. subtilis* `csgA` and `cadA` are not curli or an acid-resistance gene. Operons are
all-of rules: lose one `cyo` subunit and the whole bo3 oxidase goes (tested).

| Trait | Unit | Meaning |
| --- | --- | --- |
| `aerobic_capacity` | fraction | Share of full oxygen use the genome supports; scales the oxygen uptake capacity. |
| `fermentation_capacity` | fraction | Share of growth that can continue without oxygen by fermenting. |
| `acetate_overflow` | fraction | Fraction of respired carbon excreted as acetate at saturating sugar (overflow metabolism). |
| `acetate_uptake_capacity` | fraction | Share of the acetate uptake capacity the genome supports. |
| `ccr_strength` | fraction | How completely glucose represses alternative sugars (diauxie). |
| `stress_resistance` | fraction | Reduces starvation-induced death (stationary-phase resistance). |
| `sporulation_capacity` | fraction | Whether starving cells can form dormant spores. |
| `competence_capacity` | fraction | Whether cells can take up DNA released by dead cells (natural transformation). |
| `biofilm_capacity` | fraction | How much extracellular matrix cells can build once quorum is reached. |
| `qs_capacity` | fraction | Production of, and response to, the AI-2-like signal. |
| `ph_min` | pH | The pH below which growth stops; acid-resistance systems lower it. |
| `toxin_production` | fraction | Fraction of the maximum toxin secretion rate. |
| `toxin_susceptibility` | fraction | Scales the killing hazard; producers are assumed immune (the immunity gene is not curated). |
| `toxin2_production` | fraction | Fraction of the maximum secretion rate of the second toxin (pyocyanin, from a phenazine cluster). |
| `toxin2_susceptibility` | fraction | Scales the second toxin's killing hazard; producers are assumed immune. |
| `predation_capacity` | fraction | Rate at which a cell in contact with prey of another genome kills it and takes up part of its biomass. |
| `gliding` | fraction | 1 makes the cell a surface glider that reverses instead of swimming and tumbling. |
| `starvation_signalling` | fraction | 1 makes signalling rise with starvation and act by contact (a developmental signal, as in myxobacteria). |

References were resolved against Crossref when the table was written (Wolfe 2005; Gorke & Stulke 2008;
Hengge-Aronis 2002; Hecker et al. 2007; Higgins & Dworkin 2012; Setlow 2007; Dubnau 1991; Branda et al. 2001;
Danese et al. 2000; Surette et al. 1999; Foster 2004; Kenig & Abraham 1976; and for the added organisms Friedman & Kolter 2003; Ma et al. 2006; Gambello & Iglewski 1991; Mavrodi et al. 2001; Sonnleitner et al. 2014; Schuster et al. 2003; Novick et al. 1993; Cramton et al. 1999; Hartzell & Kaiser 1991; Lu et al. 2005; Harris et al. 1998; Ellehauge et al. 1998; Zhang et al. 2020). Only their existence and bibliographic details were checked against Crossref, not their text. They support the *qualitative*
mechanism a gene underlies; none validates a numeric coefficient.

## Parameters

Every parameter of `RealismConfig`. All are calibrations, none is fitted to data.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `oxygen_saturation_mmol_l` | 0.21 | Air-saturated water near 37 C (about 0.21 mM). |
| `gas_exchange_per_s` | 0.05 | Relaxation rate of dissolved oxygen toward saturation: a thin film open to the air. |
| `toxin_decay_per_s` | 0.0003851 | First-order loss of the toxin (half-life 30 min). |
| `signal_decay_per_s` | 0.0001925 | First-order loss of the AI-2-like signal (half-life 1 h). |
| `detritus_mineralization_per_s` | 2e-05 | Dead cells' contents return to the field as food with a time constant of about 14 h. |
| `flow_um_s` | (0.0, 0.0) | Uniform flow velocity (x, y). Zero by default; set it on the config or on `simulation.flow_um_s`. |
| `ph_neutral` | 7 | pH of the medium with no acid. |
| `ph_buffer_mmol_l` | 20 | Buffer strength in `pH = neutral - log10(1 + acetate / buffer)`. |
| `ph_optimum` | 6.8 | pH at and above which growth is unimpaired. |
| `swim_speed_scale` | 3 | Multiplies the D2 motility coefficient (10 um/s) to 30 um/s, the modern tracking value for E. coli (B. subtilis has its own factor, to 18.6 um/s). Fitted; see docs/CALIBRATION.md. |
| `tumble_rate_per_s` | 1.562 | Baseline tumble rate of a cell that senses no change: 1 / the 0.64 s mean run of Taute et al. 2015. |
| `tumble_angle_mean_deg` | 60.67 | Underlying mean of the reorientation at a tumble; after clipping to 10-170 degrees the realised mean is 62.5 degrees (Berg and Brown 68, Taute 57). |
| `tumble_angle_sd_deg` | 39.41 | Underlying spread; the realised spread after clipping is 36 degrees (Berg and Brown 1972). |
| `chemotaxis_gain` | 9.179 | How strongly the sensed change modulates the tumble rate; fitted so the largest drift over a scan of gradient steepness is 30 % of swimming speed (published best case 35 %, typical 15 %). |
| `adaptation_time_s` | 2 | Time constant of the receptor adaptation memory. |
| `sensing_half_saturation_mmol_l` | 0.005 | Concentration scale of the logarithmic sensing, `ln((C + K) / (M + K))`. |
| `translational_diffusion_um2_s` | 0.4 | Brownian diffusion coefficient of a ~1 um cell. |
| `rotational_diffusion_rad2_s` | 0.15 | Rotational diffusion: swimming direction drifts even between tumbles. |
| `cell_width_um` | 0.8 | Rod width; E. coli and B. subtilis are both about 0.7-1 um wide. |
| `cell_volume_um3_per_pg` | 3.3 | Wet volume per pg of dry mass, so a 0.3 pg cell is about 1 um^3. |
| `collisions` | True | Turn contact mechanics on or off. |
| `collision_stiffness` | 0.5 | Fraction of an overlap resolved per resolved tick (split between the two cells). |
| `collision_interval_ticks` | 2 | Contacts are resolved every n-th tick, n times as firmly: same stability, 1/n of the cost. |
| `cohesion_range_um` | 0.3 | How far beyond contact two matrix-producing cells still attract. |
| `cohesion_strength` | 0.1 | Strength of that attraction. |
| `anaerobic_yield_ratio` | 0.3 | Biomass yield per sugar without oxygen, relative to respiration. |
| `acetate_carbon_recovery` | 0.7 | Fraction of overflow/fermented carbon that ends up as acetate (the rest leaves as CO2). |
| `overflow_half_saturation_mmol_l` | 0.5 | Sugar level at which aerobic acetate overflow reaches half its maximum. |
| `ccr_half_saturation_mmol_l` | 0.05 | Glucose level at which catabolite repression is half-strength. |
| `derepression_rise_tau_s` | 600 | Time to synthesise the enzymes of an alternative sugar once glucose is gone: the diauxic lag. |
| `derepression_fall_tau_s` | 30 | Time to shut them off when glucose returns (inducer exclusion is fast). |
| `reference_mass_pg` | 0.3 | The D2 per-cell uptake rates are for a cell of this dry mass; uptake is proportional to mass, so growth is exponential. |
| `initial_energy_reserve_s` | 7200 | An energy reserve of this many seconds of maintenance (also its cap). |
| `lag_initial` | 0.05 | Growth readiness of a freshly introduced cell (0 = none, 1 = fully adapted). |
| `lag_recovery_tau_s` | 1200 | Time constant of recovery from lag when food is present. |
| `lag_decay_tau_s` | 1.08e+04 | Time constant of losing readiness without food. |
| `lag_carbon_threshold_mmol_l` | 0.001 | Total sugar/acetate above which a cell counts as having food. |
| `starvation_tau_s` | 1800 | Time constant of building starvation stress. |
| `starvation_hazard_per_s` | 5e-05 | Maximum death rate of a fully stressed, non-resistant cell. |
| `division_stress_limit` | 0.5 | Cells stressed beyond this do not divide. |
| `division_commitment_s` | 30 | Mean wait between reaching the division mass and dividing (division speed is otherwise growth-limited). |
| `sporulation_stress` | 0.6 | Stress above which a sporulation-capable cell may commit. |
| `sporulation_time_s` | 2.88e+04 | Time from commitment to dormant spore (8 h). |
| `germination_rate_per_s` | 0.001667 | Chance per second that a spore germinates while sugar is present. |
| `germination_threshold_mmol_l` | 0.01 | Sugar needed to trigger germination. |
| `spore_hazard_factor` | 0.001 | Spores die this fraction as fast as vegetative cells. |
| `signal_production_mmol_s` | 1e-18 | Signal made per quorum-capable cell. |
| `quorum_half_saturation_mmol_l` | 0.0002 | Signal level at which quorum activation is half-maximal (Hill n = 2). |
| `eps_rise_tau_s` | 900 | Time constant of building the biofilm matrix. |
| `eps_fall_tau_s` | 7200 | Time constant of losing it. |
| `toxin_production_mmol_s` | 5e-17 | Toxin made per producer at full capacity (boosted by quorum). |
| `toxin_half_saturation_mmol_l` | 0.0001 | Toxin level at which the killing hazard is half-maximal (Hill n = 2). |
| `toxin_max_hazard_per_s` | 0.01 | Maximum toxin-induced death rate of a susceptible cell. |
| `acid_hazard_per_s_per_ph` | 0.001 | Death rate per pH unit below the organism's minimum. |
| `transformation_rate_per_s` | 0.002 | Chance per second that a competent cell takes up DNA lying in its grid cell. |
| `competence_stress` | 0.2 | Starvation stress that makes a competence-capable cell competent ... |
| `competence_signal` | 0.1 | ... or a quorum activation this high. |
| `dna_lifetime_s` | 1800 | Free DNA is degraded by nucleases after this long. |
| `transformation_window_bp` | 50000 | Length of the recombination tract. |
| `dna_pool_limit` | 20000 | Maximum number of free DNA fragments kept. |
| `toxin2_decay_per_s` | 0.0001925 | Decay rate of the second toxin (a redox-active phenazine, more persistent than the first). |
| `toxin2_production_mmol_s` | 5e-17 | Second toxin made per producer at full capacity. |
| `toxin2_half_saturation_mmol_l` | 0.0001 | Second-toxin level at which the killing hazard is half-maximal (Hill n = 2). |
| `toxin2_max_hazard_per_s` | 0.01 | Maximum death rate from the second toxin. |
| `reversal_rate_per_s` | 0.002222 | Reversal rate of a gliding cell (fitted to 7.5 min between reversals, data/calibration_fit.json). |
| `reversal_suppression` | 0 | How much activation suppresses reversals. 0: the published reversal periods inside and outside aggregates are equal (7.5 min). |
| `predation_reach_um` | 0.5 | A predator reaches this far beyond touching. |
| `predation_rate_per_s` | 0.002222 | Chance per second that a predator in contact kills a prey cell (a kill takes 5-10 min of contact). |
| `predation_assimilation` | 0.1353 | Share of a killed cell's biomass the predator takes up; the rest goes to detritus. Fitted so M. xanthus among excess E. coli grows at 0.183/h (Mendes-Soares and Velicer 2013). |
| `predation_max_intake_per_s` | 0.0018 | A predator takes in at most this fraction of its own mass per second of feeding (assumed). |
| `predation_feeding_window_s` | 281 | Seconds a predator stays on a prey cell: the measured residence time (Zhang 2020). |
| `spore_predation_factor` | 0.05 | Spores are killed this much as readily as vegetative cells (assumed). |
| `development_speed_factor` | 3.13 | How much faster a fully starved developing glider moves than a growing one (0.12 against 0.0383 um/s; fitted). |
| `aggregate_speed_ratio` | 0.1917 | Fraction of that speed a cell keeps when fully activated by contact, inside an aggregate (0.023 / 0.12; fitted). |
| `cluster_separation_rate_per_s` | 2e-05 | Chance per second that a bond between sister cocci is cut (autolysin separation; assumed, 14 h on average). |
| `cluster_stiffness` | 0.5 | How fast a stretched bond pulls its two cells back into contact. |
| `contact_signal_hill` | 4 | Steepness (Hill n) of the contact response of a developing cell (assumed). |
| `contact_signal_half_neighbours` | 4 | Touching neighbours at which contact signalling is half-maximal for a starving, signalling cell (assumed). |

## The microscope view

`python -m microbiome` shows the sample as a microscope does (`render/lens.py`, `render/microscope.py`). There are no
sprites: every cell image is computed from its true rod length, width and orientation (a cylinder with hemispherical caps,
a circular cross-section giving a thickness profile) by a model of the imaging technique, then blurred by the objective's
point-spread function. The scale is real: the mouse wheel selects an objective (10x, 20x, 40x, 60x, 100x = 1.6 to 16 px/um,
with resolution sigma 0.47 to 0.085 um) and a scale bar gives the true length.

| Key | Mode | What is imaged |
| --- | --- | --- |
| `B` | Phase contrast | Dark, dense cells with the bright halo of positive phase contrast; endospores phase-bright |
| `B` | Brightfield, Gram stain | Gram-negative cells pink, Gram-positive purple; spores unstained |
| `B` | Darkfield | Scattered light only: bright rims on black |
| `B` | Fluorescence | A cytoplasmic fluorescent protein: green and red; spores nearly dark |

`Z` / `X` move the focal plane: each cell has a fixed pseudo-depth in the 10 um film (a hash of its id), and cells away from
the plane blur and lose contrast, with a depth of field that shrinks as magnification grows. `E` adds an eyepiece mask, `Tab`
shows the old instrument panel (hidden by default), `V` still cycles the chemistry overlays but these are labelled *analysis
overlay* because nutrients are invisible in a microscope. The sample is periodic, as the simulation is, so panning never
reaches an edge. Image defects are modelled: uneven illumination, a vignette, faint out-of-focus dust rings on the optics,
sensor noise, and debris on the slide. This is an approximation of the optics (a thickness-profile contrast model plus a
Gaussian PSF), not a wave-optics simulation, and the colours of the stain and fluorophore are chosen, not measured.

## What changed in the frozen interfaces (all additive)

D1: `engine._divide` now returns the dividing rows (callers that ignored the old `None` are unaffected) and
`engine.step` accepts an optional `simulation=` object; with `None` it is the original step. D3: `RealismSnapshot` is a
*subclass* of `RenderSnapshot` (extra fields `channel_names`, `cell_state`, `eps`), returned only in realism worlds;
the frozen field list of the base class is untouched. The controller gains optional methods outside the D3 protocol
(`realism_enabled`, `channel_names`, `realism_stats`, `chemistry_at`, `life_of`) that the UI uses only if present.
`RealismSnapshot` also carries `length_um`, `cell_width_um` and `heading_rad` (the true rod geometry), and `WorldView` projects them.
D4: `DerivedVariant.mutation_model_id` may also be `transformation-snv-v1`. The renderer still imports nothing from
`sim`. The controller initialises its conservation ledger at construction in realism worlds, because the oxygen
field starts non-zero.

## Cost

Measured through the real controller on the development machine (simulation tick, snapshot and drawing together,
1280x720): 1,000 cells 163 fps; 5,000 cells 83 fps (73 at p95); 9,000 cells 58 fps (53 at p95). The D1 step at 10,000
cells is about 7 ms; the realistic tick is about 12 ms (10,000), mainly contacts (every other tick), sensing and
the metabolism arrays. **60 fps holds to roughly 8,000 cells in realism mode.**

## Verification

`tests/test_realism_kinetics.py`, `_movement.py`, `_chemistry.py`, `_life.py`, `_world.py`, `_controller.py`,
`_ui.py`, `tests/test_physiology.py`, `tests/test_transformation.py`. Each mechanism is compared with a closed form
(Michaelis-Menten half rate at Km, the Hill half-point, the Einstein MSD `4Dt`, the rotational variance `2 D_r t`, the
tumble rate and angle distribution, the lag recursion, the starvation survival integral, the oxygen/toxin/signal
time constants, exact acetate stoichiometry, carbon never created) and the per-channel ledger must close every tick.
Fourteen deliberate breakages of the physics (saturation off, no repression, no adaptation, wrong-way chemotaxis,
doubled Brownian step, no contacts, no lag, spores that eat, immunity ignored, self-DNA uptake, reversed flow, ...)
are each caught by a named test.

Findings from building it, all fixed: a linear pH ramp gave *E. coli* only 67% growth at pH 6 (replaced by a
saturating curve); regulation was instantaneous so the diauxic switch was smooth (a derepression lag was added);
several of my own tests were vacuous (`np.allclose` at 1e-12 mmol) and were made exact; and in this thin 2D film a
microcolony does **not** run out of oxygen (it diffuses too fast), so hypoxia is tested in a sealed chamber.

## Still not modelled

Rods bending, flagella and pili (flagella are invisible in the imaging modes anyway); the third dimension (cells have a
fixed pseudo-depth only for focus blur), so
biofilms are monolayers and oxygen limitation inside thick colonies cannot arise; chemorepellents and aerotaxis; gene
expression (a gene is present or absent); regulation other than catabolite repression and quorum sensing; metabolism
beyond two sugars, acetate and oxygen; antibiotics other than the one toxin; cross-species gene transfer and
plasmids; variants changing anything but the 13 D1 traits (their physiology is copied from the root genome); a way
to toggle flow from the UI (it is a config value); comparison of any population-level behaviour (a growth curve, a colony, a
competition) with laboratory data; and calibration of every parameter not listed in `docs/CALIBRATION.md`.
