# Genome-to-trait mapping: biology and limits

This document explains the versioned mapping in `data/gene_trait_table.json`.
That JSON is the executable source of rule IDs, exact values, bounds, and
per-rule citations; this document records the biological rationale and the
limits a reader must not infer away. Current table revision: **2**. The D2
contract is decision `kb_mvc38cjpkCBIp7jFgo55`.

## Source genomes and annotation

The inputs are complete, accessioned RefSeq assemblies listed in
`data/manifest.json`, not synthetic sequences:

| Assembly | Organism | Role in the current table |
| --- | --- | --- |
| [GCF_000005845.2](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/) | *Escherichia coli* str. K-12 substr. MG1655 | Chemotaxis, flagellar motility, glucose/lactose uptake, growth proxy, division threshold |
| [GCF_000009045.1](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/) | *Bacillus subtilis* subsp. *subtilis* str. 168 | Chemotaxis, flagellar motility, glucose uptake/growth proxy, division threshold, bacilysin-cluster antagonism |

The sequence and GFF3 annotation, source URLs, fetch timestamps, sizes, SHA-256
checksums, accession versions, and NCBI terms are recorded in
`data/manifest.json`. The annotation source is the NCBI RefSeq GFF3 for the
matching accession.version. Gene keys are assembly-scoped locus tags (or the
documented coordinate fallback); they are not generic gene names. `gene` and
`CDS` rows are grouped so a multipart feature is not counted as multiple genes.
The parser does not scan raw DNA motifs, fuzzy-match product text, infer RNA
expression, or mutate a source sequence.

## What a mapped trait means

This is a **binary genome-potential model**, not an expression or laboratory
phenotype model. A complete, non-pseudogene annotated feature that satisfies an
exact rule has presence weight 1.0. A missing, partial, pseudogene, or unmatched
feature contributes zero. The rule starts with its explicit schema baseline,
adds its numeric effect in the declared unit, then the phenotype builder clamps
to the trait bound. Rules run in stable `rule_id` order. Multi-gene rules run
once only when every required CDS gene key is present. The resulting trait
values are calibration coefficients for the simulator; they do not assert that
the gene is transcribed, that the encoded protein is abundant, or that the
measured rate occurs under arbitrary environmental conditions.

Every rule below has `approximation_status: calibrated_simplification` in the
JSON table. The references support the annotated function/gene basis; they do
not validate the numeric simulator coefficient unless stated explicitly.

## Trait baselines, bounds, and units

Baselines and bounds are part of table revision 2. Numeric baselines/effects are
calibrations, not measured values. Per-nutrient fields have one value per
explicit channel (currently glucose and lactose); scalar fields have one value
per genome.

| Trait | Baseline | Bounds | Unit | Meaning/limit |
| --- | ---: | ---: | --- | --- |
| `max_uptake_mmol_s` | 0 per channel | 0–0.001 | mmol/s per cell per nutrient | Single-channel substrate capacity calibration; not measured maximum flux |
| `storage_fraction` | 0.1 | 0–0.5 | dimensionless | Fraction stored under the simplified uptake accounting |
| `biomass_fraction` | 0.75 | 0–0.9 | dimensionless | Fraction assigned to biomass-equivalent material; sum with storage must be ≤1 |
| `energy_yield_pj_per_mmol` | 1e15 per channel | 0–1e16 | pJ/mmol | Explicit energy coefficient; energy is not conserved by D1 |
| `motility_um_s` | 0 | 0–50 | µm/s | Potential speed coefficient, not an observed swim speed |
| `chemotaxis_sensitivity` | 0.15 | 0–2 | dimensionless | Directional response coefficient, not a measured signaling response |
| `random_motility_fraction` | 0.15 | 0–1 | dimensionless | Model random-direction component |
| `division_rate_s` | 0.00005 | 0–0.01 | s⁻¹ | Simulation division hazard, not a measured doubling rate |
| `division_mass_pg` | 0.5 | 0.1–10 | pg | Simulation mass threshold, not a measured cell-size prediction |
| `death_rate_s` | 0.00001 | 0–0.01 | s⁻¹ | Simulation death hazard; no rule-specific death effect is currently mapped |
| `max_age_s` | 86400 | 1–1e9 | seconds | Model age limit; not a measured lifespan |
| `antagonism_radius_um` | 0 | 0–100 | µm | Contact-model interaction radius, not antibiotic diffusion length |
| `antagonism_rate_s` | 0 | 0–0.01 | s⁻¹ | Contact-mortality hazard, not a measured toxin dose/killing rate |

Baseline-only traits in this revision are storage fraction, biomass fraction,
energy yield, random motility, death rate, and maximum age. There is no gene
effect for them in the current table; their baseline is not evidence that the
annotated genome uniquely has or lacks that capability.

## Rule-by-rule basis and limitations

### *E. coli* MG1655 — GCF_000005845.2

| Rule ID | Annotated gene basis | Effect and unit | Evidence / explicit limit |
| --- | --- | --- | --- |
| `ecoli-chea-chemotaxis` | `b1888`, `cheA`, CheA histidine kinase | +0.35 chemotaxis sensitivity | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [Jahreis et al. 2004](https://pmc.ncbi.nlm.nih.gov/articles/PMC387806/). Potential signaling coefficient, not measured activity. |
| `ecoli-cheb-chemotaxis-adaptation` | `b1883`, `cheB`, response regulator/methylesterase | +0.05 chemotaxis sensitivity | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [Jahreis et al. 2004](https://pmc.ncbi.nlm.nih.gov/articles/PMC387806/). Adaptation capacity proxy; methylation dynamics are not modeled. |
| `ecoli-cher-chemotaxis-adaptation` | `b1884`, `cheR`, chemotaxis methyltransferase | +0.05 chemotaxis sensitivity | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [Jahreis et al. 2004](https://pmc.ncbi.nlm.nih.gov/articles/PMC387806/). Binary adaptation-potential coefficient, not expression. |
| `ecoli-chey-chemotaxis-output` | `b1882`, `cheY`, chemotaxis output regulator | +0.25 chemotaxis sensitivity | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [UniProt P0AE67](https://www.uniprot.org/uniprotkb/P0AE67/entry). Not measured motor bias. |
| `ecoli-flic-flagellar-filament` | `b1923`, `fliC`, flagellar filament protein | +8 µm/s motility | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [Wood et al. 2006](https://pubmed.ncbi.nlm.nih.gov/16397770/). Calibrated speed contribution, not measured MG1655 swimming speed. |
| `ecoli-mota-flagellar-motor` | `b1890`, `motA`, flagellar motor component | +1 µm/s motility | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [Wood et al. 2006](https://pubmed.ncbi.nlm.nih.gov/16397770/). Motor potential, not protein abundance or actual velocity. |
| `ecoli-motb-flagellar-motor` | `b1889`, `motB`, stator partner | +1 µm/s motility | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [UniProt P0AF06](https://www.uniprot.org/uniprotkb/P0AF06/entry). Potential component, not measured speed. |
| `ecoli-ptsg-glucose-uptake` | `b1101`, `ptsG`, glucose-specific PTS IIBC | +8.8809946714e-16 mmol/s glucose uptake | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [Carreón-Rodríguez et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10305011/). Value comes from the internal mass-balance calibration below, not an MG1655 flux measurement. |
| `ecoli-ptsg-glucose-growth-proxy` | `b1101`, `ptsG`, glucose import | +0.00015 s⁻¹ division hazard | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [Carreón-Rodríguez et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10305011/). Indirect, condition-free growth proxy; not a constitutive growth rate. |
| `ecoli-lacy-lactose-uptake` | `b0343`, `lacY`, lactose permease | +4.67426234297e-16 mmol/s lactose uptake | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [UniProt P02920](https://www.uniprot.org/uniprotkb/P02920/entry). Calibrated potential, not a measured lactose flux or lactose-growth result. |
| `ecoli-ftsz-division-threshold` | `b0095`, `ftsZ`, cell-division protein | +0.1 pg division-mass threshold | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/); [Romberg & Levin 2003](https://pmc.ncbi.nlm.nih.gov/articles/PMC5517307/). Enables a finite model threshold; not a causal cell-size prediction. |

No `lacY` division-rate rule remains in revision 2; it was removed as an
unsupported proxy. Lactose effects are specific to the E. coli `lacY` mapping.

### *B. subtilis* 168 — GCF_000009045.1

| Rule ID | Annotated gene basis | Effect and unit | Evidence / explicit limit |
| --- | --- | --- | --- |
| `bacillus-chea-chemotaxis` | `BSU_16430`, `cheA`, chemotactic sensor kinase | +0.35 chemotaxis sensitivity | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/); [Cannistraro et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3133262/). Potential signaling coefficient, not activity measurement. |
| `bacillus-chec-chemotaxis-adaptation` | `BSU_16450`, `cheC`, adaptation/signal-terminating phosphatase | +0.05 chemotaxis sensitivity | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/); [Mukherjee & Kearns 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4869327/). Adaptation proxy; detailed signaling is omitted. |
| `bacillus-ched-chemotaxis-adaptation` | `BSU_16460`, `cheD`, receptor-specificity/adaptation protein | +0.05 chemotaxis sensitivity | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/); [Mukherjee & Kearns 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4869327/). Does not model methylation/deamidation dynamics. |
| `bacillus-chey-chemotaxis-output` | `BSU_16330`, `cheY`, chemotaxis/motility regulator | +0.25 chemotaxis sensitivity | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/); [Cannistraro et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3133262/). Genome potential, not transcriptional activity. |
| `bacillus-hag-flagellar-filament` | `BSU_35360`, `hag`, flagellin | +8 µm/s motility | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/); [Mukherjee & Kearns 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4869327/). Calibration, not observed swimming speed. |
| `bacillus-mota-flagellar-motor` | `BSU_13690`, `motA`, H⁺-coupled stator/motor component | +1 µm/s motility | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/); [Mukherjee & Kearns 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4869327/). Potential, not measured motor output. |
| `bacillus-motb-flagellar-motor` | `BSU_13680`, `motB`, stator partner | +1 µm/s motility | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/); [Mukherjee & Kearns 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4869327/). Potential, not measured speed. |
| `bacillus-ptsg-glucose-uptake` | `BSU_13890`, `ptsG`, glucose-specific PTS IICBA | +8.8809946714e-16 mmol/s glucose uptake | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/) and its cited annotation evidence. Internal mass-balance calibration, not measured B. subtilis flux. |
| `bacillus-ptsg-glucose-growth-proxy` | `BSU_13890`, `ptsG`, glucose import | +0.00015 s⁻¹ division hazard | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/) and its cited annotation evidence. Condition-free indirect proxy, not a measured growth rate. |
| `bacillus-ftsz-division-threshold` | `BSU_15290`, `ftsZ`, cell-division initiation protein | +0.1 pg division-mass threshold | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/); [Romberg & Levin 2003](https://pmc.ncbi.nlm.nih.gov/articles/PMC5517307/). Model coefficient, not a measured FtsZ-dependent cell size. |
| `bacillus-bacilysin-antagonism-radius` | Complete CDS set `BSU_37680`–`BSU_37740` (`bacG/F/E/D/C/B/A`) | +12 µm antagonism radius | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/); [Steinborn et al. 2005](https://pubmed.ncbi.nlm.nih.gov/15609023/). Fires once only when all seven keys are present; radius is an engine calibration, not a measured antibiotic diffusion length. |
| `bacillus-bacilysin-antagonism-rate` | Same complete annotated `bacA-G` cluster | +0.001 s⁻¹ antagonism hazard | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/); [Steinborn et al. 2005](https://pubmed.ncbi.nlm.nih.gov/15609023/). Contact-mortality calibration, not bacilysin dose or measured killing rate. |

## Uptake calibration and explicit limitations

Revision 2 replaced the earlier uptake values that implied sub-microsecond
biomass doubling. For each nutrient channel treated as the sole available
substrate, the table derives a per-cell uptake coefficient from its own
division calibration:

`max_uptake_mmol_s = division_mass_pg * division_rate_s / (biomass_fraction * molar_mass_g_mol * 1e9)`

The current calibration uses 0.6 pg division mass after the FtsZ rule,
0.0002 s⁻¹ division hazard after the PtsG proxy, biomass fraction 0.75, and
molar masses 180.16 g/mol for glucose and 342.3 g/mol for lactose. This yields
8.8809946714e-16 mmol/s for glucose and 4.67426234297e-16 mmol/s for lactose.
These are internally consistent simulator coefficients, not measured maximum
uptake capacities. Simultaneously saturated nutrient channels can sum above
the single-channel calibration because there is no shared total-carbon cap.

Assay finding `kb_mvc59d7hEc6dBMViUn24` documents a separate conditional E. coli
B/r carbon-balance comparison (~8e-16 to 3e-15 mmol/s per cell under stated
mass, medium, temperature, doubling-time and assumed carbon-fraction inputs).
It is not transferred to MG1655, B. subtilis 168, lactose, or a universal
maximum rate. The earlier HGT-strain maximum and flux-balance constraint
citations were withdrawn as support for wild-type per-cell uptake.

The zero B. subtilis lactose baseline means only that this revision has no
curated lactose-specific mapping for that genome; it does not prove the strain
lacks all lactose-uptake routes. Likewise, gene presence, calibrated division
hazards, motility coefficients, and the bacilysin cluster indicate modeled
potential only. They are not measurements of expression, flux, growth,
swimming speed, antibiotic diffusion, or killing. Population growth divergence
between the two genomes has not yet been shown; the current division-rate
calibration is the same for both.

## Evidence links

Per-rule evidence and source GFF3 feature descriptions are stored in
`data/gene_trait_table.json` alongside every rule. Key references include:

- [E. coli chemotaxis signaling (Jahreis et al. 2004)](https://pmc.ncbi.nlm.nih.gov/articles/PMC387806/)
- [MG1655 motility loci (Wood et al. 2006)](https://pubmed.ncbi.nlm.nih.gov/16397770/)
- [PTS sugar transport (Carreón-Rodríguez et al. 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10305011/)
- [FtsZ and bacterial division (Romberg & Levin 2003)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5517307/)
- [B. subtilis chemotaxis (Cannistraro et al. 2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3133262/)
- [B. subtilis motility systems (Mukherjee & Kearns 2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4869327/)
- [Bacilysin biosynthesis genes (Steinborn et al. 2005)](https://pubmed.ncbi.nlm.nih.gov/15609023/)
- [PubChem glucose molecular weight](https://pubchem.ncbi.nlm.nih.gov/compound/107526) and [anhydrous lactose molecular weight](https://pubchem.ncbi.nlm.nih.gov/compound/Lactose_-anhydrous)
