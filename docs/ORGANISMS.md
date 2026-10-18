# The organisms

Five real bacteria, each a complete NCBI RefSeq assembly listed in `data/manifest.json` with its hashes and provenance.
The first two came with the project; the other three were added with `python -m microbiome.genome.fetch <accession>`
(it downloads from NCBI Datasets, hashes every file and writes the manifest entry; only "Complete Genome" assemblies are
accepted).

| Organism | Accession | Genome | Shape | Gram | What its genes switch on in the model |
| --- | --- | --- | --- | --- | --- |
| *Escherichia coli* K-12 MG1655 | GCF_000005845.2 | 4.64 Mb | rod 0.8 um wide | negative | chemotaxis, glucose PTS, overflow, acid resistance, biofilm, AI-2 |
| *Bacillus subtilis* 168 | GCF_000009045.1 | 4.22 Mb | rod 0.8 um wide | positive | sporulation, competence, bacilysin, biofilm |
| *Pseudomonas aeruginosa* PAO1 | GCF_000006765.1 | 6.26 Mb | rod 0.75 um wide | negative | one polar flagellum, OprB glucose porin, three terminal oxidases, Pel and Psl matrix, las signal, **pyocyanin** (a second toxin) |
| *Staphylococcus aureus* Newman | GCF_000010465.1 | 2.88 Mb | **coccus**, about 1 um | positive | two glucose PTS transporters, fermentation, agr signal, ica matrix, SigB |
| *Myxococcus xanthus* DK 1622 | GCF_000012685.1 | 9.14 Mb | rod 0.5 um wide, 5-7 um long | negative | **gliding** (MglA/MglB), **predation on other organisms**, starvation signalling, development genes (FruA, MrpC, RelA), EPS matrix |

*S. aureus* NCTC 8325 was tried first and replaced by Newman: its assembly contains one ambiguous base (`N`), which the
D4 mutation code (it needs unambiguous A/C/G/T) rejects.

Shape, width and imaging style (the Gram colour in the brightfield view) come from `genome/organisms.py` (`PROFILES`);
they are standard microbiology, not derived from genes. Everything else comes from the genes by the rules in
`genome/physiology.py` (72 rules, each naming exact locus tags, all-of operons, with a DOI where one was found) and the
D2 rules written by `python -m microbiome.genome.organisms export`.

## What is new in the simulation

* **A second diffusing toxin** (pyocyanin, channel 6). A producer is mostly immune; others die in proportion to the level.
* **Gliding.** A glider moves at a constant speed along its axis, reverses (a 180 degree turn) at random instead of
  tumbling, and has no Brownian jitter. A starved, developing glider moves 3.1 times faster (0.12 um/s, outside an
  aggregate) and, fully activated by contact, at 0.19 of that (0.023 um/s, inside), both from Sliusarenko et al. 2007.
  Reversals are not suppressed, because the same table gives the same reversal period (7.5 min) inside and outside.
* **Swimmers with their own statistics.** *P. aeruginosa* runs 2.18 s and then reverses by 165 +/- 9 degrees (Tian et
  al. 2022: pull and push runs of 2.21 and 2.14 s; 90 % of the turns between 150 and 180 degrees) instead of tumbling
  like *E. coli*. The wrap mode of that paper (short, 40 % of reversals) is not modelled.
* **Predation.** A predator in contact with a cell of *another genome* kills it at a rate per second (a kill takes
  5-10 min of contact). It takes up a share of the prey's biomass, capped per kill by an intake limit; the rest of the
  prey goes to detritus like any death. The share (0.135) is fitted so that *M. xanthus* among an excess of *E. coli* grows
  at 0.183/h, the one number found that links predator growth to prey (Mendes-Soares and Velicer 2013). Mass is conserved
  to rounding (tested).
* **Contact signalling.** A starving cell with `starvation_signalling` is activated by how many same-genome cells touch
  it, not by a diffusing level (in a world 800 um wide, a diffusing signal from a few hundred cells is far too dilute).
  Such a cell commits to sporulation only when activated, i.e. only among neighbours.
* **Clusters.** Sister cocci are bonded after division (a stretched bond pulls them back into contact) and the bond is
  cut at random (autolysin separation); the division axis turns 90 degrees each generation (the three alternating
  planes, flattened to two). Tested: bonded cells stay in contact, and a control with bonds cut at once does not.
  In a 2.7 h run from 4 cells the bonded culture was 25 cells in 9 clusters (largest 8); the control was 25 single cells.
  The cut rate (one per 14 h on average) is an assumption.
* **Shapes.** A coccus is a sphere of the cell's volume; rods keep their own width.

## Calibration of the new organisms

See `docs/CALIBRATION.md` (regenerated from the fit). Matched by construction:

* *P. aeruginosa*: growth rate 0.91/h and glucose uptake 9.52 mmol/gDW/h (Berger et al. 2014); swimming speed 37.7 um/s,
  run time 2.18 s and reversal angle 165 degrees (Tian et al. 2022). Held out: growth rate 0.88/h from a second
  laboratory (Dolan et al. 2020), which the model matches (0.91, inside the +/-0.05 band).
* *S. aureus* (strain COL-S in the source, not Newman): growth rate 1.05/h and maximal glucose uptake 12.4
  mmol/gDW/h (Ferreira et al. 2013); cell volume at division 1.04 um^3 from 0.52 um^3 at birth (Monteiro et al. 2015).
* *M. xanthus*: gliding speed 2.3 um/min (Zhang et al. 2020), 7.5 min between reversals and the speeds of starved cells
  outside and inside aggregates (Sliusarenko et al. 2007), mean length 6 um (Kaiser and Welch 2004), growth rate among
  prey 0.183/h (Mendes-Soares and Velicer 2013; the table prints no unit and the slope of ln population size per hour
  is assumed; the model gives 0.207/h). A kill takes 5-10 min of contact (a review sentence).

Maintenance for all three is *E. coli*'s assumed value, a placeholder: no coefficient was found. No per-cell dry mass was
found for *P. aeruginosa* or *S. aureus* (the *S. aureus* size is fitted by volume instead).

## What is not reproduced, said plainly

* **Fruiting bodies.** Starved *M. xanthus* cells do not aggregate into mounds in this model, after repeated attempts.
  The runs were 3 to 4 simulated hours with the 2 h energy reserve shortened to 0 so that starvation starts at once, in
  periodic boxes of 100 um, 800 to 1200 cells (0.08-0.12 cells/um^2), with the published inside and outside speeds, no
  reversal suppression, and contact thresholds of 1 to 4 touching neighbours with Hill steepness 2 to 8. The clumping
  index (variance over mean of counts in 10 um boxes; 1 is random) ended between 0.5 and 1.1, i.e. no clumping or a more
  even spread, in every run. A rough analysis says why with a shallow response: cells that slow where it is crowded gather
  only if speed falls faster than 1/density, which the 0.19 speed ratio allows only with a steep switch, and the steep
  settings did not produce it either. Alignment of neighbouring rods (streaming), which real aggregation relies on, is not
  modelled and was not added. What does work: sporulation is confined to cells that have neighbours. The published timeline
  (aggregates at 13-14 h, fruiting bodies at about 22 h, about 10 % of cells becoming spores) was never reached and is
  recorded as a reference only; no run was carried to those times.
* **Predator yield per prey cell** is still not measured: the fitted share (0.135) reproduces the one published growth
  rate, but the source gives a maximum yield per plate, not per prey cell, and the model's kill and intake details
  beyond that are assumptions. A 20 min test of 40 predators among 300 *E. coli* in an 80 um patch lost 7 *E. coli*,
  about 4 of them to background death, and did not count predation deaths separately.
* *S. aureus* acetate and lactate: the model makes acetate only (no lactate).
* *M. xanthus* does not grow on sugar (it cannot in reality), and the model grows it on prey only; **its growth on
  casitone (5.1 h generation time) was not added**: it needs a new nutrient channel and a change from three carbon
  sources to four in the way founders, uptake and predation are written, which touches most of the test suite.
* The `P. aeruginosa` wrap mode, and everything else listed under "Known mismatches" in `docs/CALIBRATION.md`
  (the acetate threshold in batch culture, tumble duration, oxygen diffusivity, the size-growth law), was left as it was.

Reproduced since the first version of this document: *P. aeruginosa* run time and reversals; staphylococcal clusters as
a mechanism; the *S. aureus* delay (it was an artifact of the check, in which a cell that died let a daughter be counted:
the whole-cycle time is now 40.1 min against 40.1 expected).

## Using them

`python -m microbiome` starts the microscope view with all five available. `Tab` shows the instrument panel; the genome
list has one entry per organism. Place a few *M. xanthus* cells next to *E. coli* and let the speed rise to x32 or more:
a 7.5 min kill time is real time.
