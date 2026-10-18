# Architecture

## Frozen decisions

This file is the canonical interface reference for fresh sessions. It records
the D1 simulation contract, D2 genome-to-phenotype contract, and D3 boundary
decision. AgentDock findings: D1 `kb_mvc1pzs7_F3CTKUGMEw4`; D2
`kb_mvc38cjpkCBIp7jFgo55`; D3 `kb_mvc746dnDCSQvHtSJL-6`.

Project constraints remain Python + numpy + pygame, real accessioned genomes,
seeded deterministic simulation, and 2D rendering. This document specifies
interfaces; it is not a claim that every behavior is implemented or verified.

## D1 — deterministic simulation contract

### Tick and public calls

- `DT_S = 1.0 / 60.0` seconds, fixed and not configurable. Tick n is the interval
  `[n*DT_S,(n+1)*DT_S)`. Simulation time is `tick_index*DT_S`, not wall-clock
  accumulated time; rendering cadence is independent.
- `step(state: SimulationState, environment: Environment,
  phenotypes: PhenotypeTable, rngs: RNGStreams) -> None` mutates the supplied
  world in place and advances exactly one tick. `ledger(state, environment) ->
  NutrientLedger` reports per-nutrient accounting.
- Tick order: apply tick-boundary commands by `(target_tick, sequence)`; diffuse
  from the old field; calculate uptake at tick-start positions; proportional
  allocation; synchronous agent update; death/division; canonicalize rows and
  increment tick once. Newborns first act on the next tick.

### State arrays and types

AgentArrays has aligned rows for N live agents. Float arrays are float64 and
IDs are stable IDs, never row indices:

| Array | dtype / shape | Unit |
| --- | --- | --- |
| agent_id | uint64[N] | unique, increasing, never reused |
| position_um, velocity_um_s | float64[N,2] | µm; µm/s |
| mass_pg, radius_um, energy_pj | float64[N] | pg dry-mass proxy; µm; pJ |
| species_id, genome_id | uint32[N], uint64[N] | stable catalogue IDs |
| age_ticks | uint64[N] | fixed ticks |
| nutrient_store_mmol, biomass_mmol | float64[N,K] | mmol per nutrient |

SimulationState also has tick_index:uint64 and next_agent_id:uint64. Entry rows
may be permuted; step canonicalizes by agent_id and emits increasing IDs.
Positions obey `0 <= x < width_um`, `0 <= y < height_um`; an exact upper edge
from floating wrap maps to zero.

### PhenotypeTable

Rows are sorted by genome_id:uint64[G]. Nutrient columns are float64[G,K] in
ascending nutrient_id order; scalar columns are float64[G]. The 13 fields are:

| Scope | Field | Unit |
| --- | --- | --- |
| per nutrient | max_uptake_mmol_s | mmol/s |
| per nutrient | storage_fraction, biomass_fraction | dimensionless |
| per nutrient | energy_yield_pj_per_mmol | pJ/mmol |
| scalar | motility_um_s | µm/s |
| scalar | chemotaxis_sensitivity, random_motility_fraction | dimensionless |
| scalar | division_rate_s, death_rate_s, antagonism_rate_s | s⁻¹ |
| scalar | division_mass_pg | pg |
| scalar | max_age_s | s |
| scalar | antagonism_radius_um | µm |

All fields are explicit in the genome-derived table; simulation compatibility
defaults are not biological evidence. Values obey evidence-backed bounds;
storage_fraction + biomass_fraction <= 1 per nutrient.

### Environment, mass ledger, RNG

Environment is a periodic 2D fixed-depth slab. Positive float64 dimensions are
width_um, height_um, depth_um, dx_um, dy_um; nx,ny >= 3; dx=width/nx,
dy=height/ny. Cell volume is `dx_um*dy_um*depth_um*1e-15` litres. NutrientField:
nutrient_id:uint32[K], concentration_mmol_l:float64[K,ny,nx],
diffusion_um2_s:float64[K], molar_mass_g_mol:float64[K],
detritus_mmol:float64[K,ny,nx], and float64[K] initial/source/sink/respiration
counters. Diffusion is synchronous periodic flux-form five-point explicit;
reject unless `D*DT_S*(1/dx_um²+1/dy_um²) <= 1/2` per nutrient. Never clamp a
negative field. Uptake is proportional-shared with a deterministic residual so
grants do not exceed availability. U mmol partitions as U=store+biomass+
respired; biomass_pg is `sum_k(biomass_mmol[i,k]*molar_mass_g_mol[k]*1e9)`.
Death moves store+biomass to detritus; division conserves each nutrient with the
second daughter receiving the exact remainder. Energy is not conserved.
Per nutrient, every tick: initial+source=field+agents+detritus+sink+respiration,
checked at rtol 1e-12 / atol 1e-15 mmol.

`initialize_rngs(root_seed:uint64) -> RNGStreams` creates one NumPy Generator
backed by PCG64 per fixed purpose tag using
`SeedSequence(root_seed, spawn_key=(purpose_tag,))`: motility=1, division=2,
mutation=3, death=4, antagonism=5. Never use Python hash(), global RNG state,
wall-clock time, unordered iteration or nondeterministic reductions in a tick.
For identical code/dependency build, Python/NumPy version and platform, seed,
config, genome/phenotype catalogue, initial state and ordered command log, each
tick's canonical snapshot must match exactly, including float64 bytes and RNG
states. Digest sorted array names/dtypes/shapes/C-order bytes plus canonical
sorted-key JSON RNG state. Cross-platform/version identity is not promised.

## D2 — real genome to phenotype contract

### Data and provenance

Keep one immutable NCBI Datasets Genome Package per assembly at
`data/genomes/<assembly_accession.version>/`; canonical raw inputs are genomic
FASTA and matching GFF3, plus assembly/sequence JSONL reports, dataset catalog,
README/checksums and original archive. `data/manifest.json` pins accession.version,
organism/strain/taxid, source URL, retrieval time/tool version, annotation
source/date, terms URL/text and every retained artifact's role/path/size/SHA-256.
Raw files are never rewritten; normalized rows record their source GFF3 hash
and parser version. No synthetic DNA is accepted.

genome_id:uint64 is the unsigned big-endian integer from the first 8 bytes of
SHA-256 over the exact ASCII assembly_accession.version; reject collisions and
preserve that full string as canonical identity. D2 references NCBI Datasets,
NCBI GFF3 and GenBank accession.version documentation in finding
kb_mvc38cjpkCBIp7jFgo55.

### Gene table and mapping

gene_features.tsv has one row per GFF3 feature part; multipart rows share a
gene_key. Columns: genome_id, assembly_accession_version, gene_key, seqid,
feature_type, source_id, parent_id, start_1based, end_1based, strand, phase,
locus_tag, ncbi_gene_id, gene_symbol, product, protein_id_version, dbxrefs,
pseudo, partial, attributes_json, source_annotation_sha256. Coordinates stay
GFF3 1-based inclusive. Use accession.version|locus_tag when unique; otherwise
accession-version-scoped seqid/type/strand/sorted coordinates. Preserve GFF3
ID/Parent for audit, not as cross-version identity. Sort by
seqid/start/end/strand/type/source ID.

trait_schema.tsv is versioned and records trait_id, scope, nutrient_id where
applicable, float64 dtype, unit, baseline, min/max, sum_clip aggregation,
baseline evidence and approximation note. gene_trait_map.tsv records mapping_id,
optional scoped genome_id, exact match namespace/value, feature_type, trait_id,
nutrient_id, effect/unit, priority, evidence class/ref/URL, rationale and
approximation_note. Exact namespaces only: gene_key, locus_tag, NCBI GeneID, EC,
KO or GO; no fuzzy product matching or motif scans. Lowest integer priority wins;
ties fail. A gene contributes at most once per trait/channel; distinct paralogs
count separately.

Only complete, non-pseudogene annotated features explicitly matched by a rule
are eligible. Presence weight is 1.0; absent, partial, pseudo or unmapped genes
contribute zero. This is genome-potential, not measured transcription or proof
of current expression. Value = clamp(baseline + sum(selected gene effects),
min,max). Every baseline/effect is evidence-backed or explicitly labeled an
approximation; evidence exposes gene keys, rule IDs and references. Validate
cross-trait constraints; never silently normalize. Mutation remains disabled
until the D4 contract below is implemented; division copies both genome_id and
variant_id.

### Producer types/signatures

Frozen D2 types are GenomeRecord, AnnotatedGene, TraitSchema, GeneTraitMap,
TraitEvidence and the D1 PhenotypeTable. Producer signatures:
`load_genome_bundle(Path) -> GenomeRecord`,
`normalize_gene_features(GenomeRecord) -> tuple[AnnotatedGene,...]`,
`load_trait_schema(Path) -> TraitSchema`,
`load_gene_trait_map(Path) -> GeneTraitMap`,
`build_phenotype_table(Sequence[GenomeRecord], TraitSchema, GeneTraitMap) ->
PhenotypeTable`, and `explain_phenotype(...) -> tuple[TraitEvidence,...]`.

## D4 — mutation and accession-preserving derived variants

Decision findings: `kb_mvcua1q2AEgePLhS8NZU` and its authoritative API addendum
`kb_mvcubh7428pLWzFFpQe3` (item 218). Implemented in
`src/microbiome/genome/variants.py` and opt-in (see Implementation conformance
below); the contract text here is unchanged. Raw genome
packages and annotations remain immutable, and a variant is never presented as
a newly sequenced/assembled accession.

### Mutation scope and rate

- Mutations are single-nucleotide substitutions (SNVs) in the inherited real
  assembly sequence. Insertions, deletions, recombination, expression changes,
  direct phenotype noise and edits to the source annotation/trait map are out of
  scope. Each daughter independently inherits its parent's variant and may
  acquire new SNVs; the parent is unchanged.
- Mutation may affect only D2-mapped phenotype contributions from a complete
  CDS that becomes a pseudogene because a substitution creates an in-frame
  premature stop before its terminal stop. The affected gene's D2 contribution
  is removed and the phenotype is re-derived using the unchanged D2 aggregation,
  bounds and cross-trait validation. Therefore only the following D1 fields can
  change, and only when the knocked-out gene has an applicable D2 mapping:
  `max_uptake_mmol_s`, `storage_fraction`, `biomass_fraction`,
  `energy_yield_pj_per_mmol`, `motility_um_s`, `chemotaxis_sensitivity`,
  `random_motility_fraction`, `division_rate_s`, `division_mass_pg`,
  `death_rate_s`, `max_age_s`, `antagonism_radius_um`, and
  `antagonism_rate_s`. Synonymous, missense, intergenic and regulatory variants
  have no modeled phenotype effect; there is no unsupported inference of altered
  enzyme kinetics or expression. No directly edited trait values are allowed.
- The rate model is `snv-poisson-2e-10-v1`: for a genome of `L` bases, each
  daughter division draws `K ~ Poisson(L * 2.0e-10)` SNVs from the mutation RNG;
  choose `K` distinct positions uniformly without replacement, sort positions,
  and at each position choose uniformly among the other three uppercase bases
  in `A,C,G,T` order. `ref` is the base in the parent's inherited sequence.
  The mutation stream is D1 `RNG_MUTATION=3` (PCG64/SeedSequence stream), used
  only for daughter mutation, with divisions processed by ascending parent
  agent_id. `K=0` means inherit the parent's variant_id unchanged. One division
  is one replication. The `2.0e-10` substitutions/base/replication rate is
  anchored to the wild-type *E. coli* K-12 base-substitution estimate (Lee et
  al. 2012; Foster et al. 2015); using it for *B. subtilis* is an explicit
  interim cross-species approximation, not a measured *B. subtilis* rate. A
  different accession-specific rate requires a new decision. The source reports
  roughly `2 x 10^-10` base-pair substitutions per nucleotide per generation:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4640725/ and
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3478608/.

### Derived-variant record and APIs

The immutable `NucleotideSubstitution` has fields
`position_0based:uint64`, `ref:Literal["A","C","G","T"]`, and
`alt:Literal["A","C","G","T"]` (`ref != alt`). The immutable
`DerivedVariant` JSON record has exactly these fields:

| Field | Type / meaning |
| --- | --- |
| schema_version | literal `1` |
| variant_id | uint64; `0` is the unmutated accession genome |
| record_sha256 | full lowercase SHA-256 hex digest of the canonical payload below |
| root_accession_version | exact source assembly accession.version |
| root_genome_id | D2 uint64 derived from that accession; serialized as decimal string |
| root_sequence_sha256, root_annotation_sha256 | immutable source-manifest hashes |
| parent_variant_id | uint64; `0` for a direct child of the accession genome |
| parent_record_sha256 | full parent record digest; null when parent_variant_id is `0` |
| parent_sequence_sha256, derived_sequence_sha256 | sequence digests before and after this record's substitutions |
| sequence_length_bp | uint64; unchanged by SNVs |
| origin_tick, parent_agent_id, daughter_agent_id | uint64 simulation lineage values; serialized as decimal strings |
| mutation_model_id | literal `"snv-poisson-2e-10-v1"` |
| substitutions | tuple of direct-parent SNVs, sorted by position, no duplicate positions |
| trait_schema_sha256, gene_trait_map_sha256 | hashes of the exact D2 inputs used for phenotype derivation |
| phenotype_sha256 | SHA-256 of the resulting D2 phenotype row, with float64 values encoded by `float.hex()` |

Canonical payload is UTF-8 JSON with sorted keys, separators `(',', ':')`,
`ensure_ascii=False`, and `allow_nan=False`, excluding `variant_id` and
`record_sha256`. `record_sha256` is SHA-256(payload); `variant_id` is the
unsigned big-endian integer represented by its first eight digest bytes. Reject
zero and any collision with another variant or accession ID; never resolve a
collision by overwriting. In serialized JSON, every uint64 identifier/counter
is a base-10 string; `position_0based` is a JSON integer. The record stores only
the direct-parent substitutions, while parent_variant_id/parent_record_sha256
and the root accession/hashes preserve the full auditable chain. Reconstruct
the derived sequence from the immutable root sequence plus the chain; verify
each parent/ref allele and both sequence digests before accepting it.

Genome/Codon owns these exact producer signatures:

```text
create_derived_variant(
    genome: GenomeRecord,
    parent_variant: DerivedVariant | None,
    *, mutation_rng: numpy.random.Generator,
    origin_tick: uint64, parent_agent_id: uint64, daughter_agent_id: uint64,
    schema: TraitSchema, gene_trait_map: GeneTraitMap,
    trait_schema_sha256: str, gene_trait_map_sha256: str,
) -> (DerivedVariant, GenomePhenotype) | None
derive_variant_phenotype(
    genome: GenomeRecord, variant: DerivedVariant,
    schema: TraitSchema, gene_trait_map: GeneTraitMap,
) -> GenomePhenotype
build_variant_phenotype_table(
    genomes: Sequence[GenomeRecord], variants: Sequence[DerivedVariant],
    schema: TraitSchema, gene_trait_map: GeneTraitMap,
) -> PhenotypeTable
```

`create_derived_variant` returns `None` iff `K=0`; otherwise it validates and
reconstructs against the immutable source package, derives the D2 phenotype
with the supplied schema/map, then hashes the final variant record and
phenotype row. The `trait_schema_sha256` and `gene_trait_map_sha256` arguments
are the exact source-file hashes used for that derivation. Sim/Cilia
owns division order and calls it once per daughter using only the D1 mutation
stream. The D1 extension is `AgentArrays.variant_id:uint64[N]` (aligned with
agent_id; `0` means unmutated accession genome) and
`PhenotypeTable.variant_id:uint64[G]`; each phenotype row retains its root
`genome_id:uint64`, is keyed uniquely by `(genome_id, variant_id)`, and rows sort
by that pair. `GenomePhenotype` likewise gains `variant_id:uint64`. The D1
`step(state, environment, phenotypes, rngs, mutation_provider=None) -> None`
extension preserves the existing no-provider behavior; when provided, the
provider signature is
`create_offspring_variant(genome_id:uint64, parent_variant_id:uint64, *, mutation_rng:Generator, origin_tick:uint64, parent_agent_id:uint64, daughter_agent_id:uint64) -> (DerivedVariant, GenomePhenotype) | None`; it delegates to the genome producer with the catalogue's `GenomeRecord`, trait schema/map and their file hashes.
On `None`, the daughter inherits its parent's variant_id; on a result, the
simulation registers the pair-keyed phenotype before the next tick and assigns
the new variant_id to the daughter only. The variant record itself is the
read-only lineage/provenance artifact; no raw genome, annotation or phenotype
catalogue row is exposed for mutation.

## D3 — sim/render/UI boundary

### Ownership and command facade

Sim owns mutable D1 world state, environment, RNGs, phenotype catalogue, tick,
next_agent_id and external command log. Core owns shared protocol/DTOs in
src/microbiome/core/ui_api.py; Cilia implements SimulationController in
src/microbiome/sim/controller.py. Render owns PygameRenderer, Camera2D, atlas and
caches. UI owns only widgets/modes/selection/scheduler state.

SimulationController signatures:
`list_genomes() -> tuple[GenomeOption,...]`,
`preview_phenotype(genome_id) -> PhenotypePreview`,
`place_nutrient(nutrient_id, position_um, amount_mmol) -> CommandTicket`,
`introduce_organism(genome_id, position_um) -> CommandTicket`,
`command_result(command_id) -> CommandResult | None`,
`inspect_cell(agent_id) -> CellInspection | None`,
`inspect_genome(genome_id) -> GenomeInspection | None`,
`set_paused(bool) -> None`, `advance_ticks(count) -> uint64`,
`single_step() -> uint64`, `render_snapshot() -> RenderSnapshot`.
Commands validate before enqueue, target the next tick, receive monotonic IDs in
submission order, and apply in that order at the tick boundary before the D1
step. Nutrient amount is mmol, converted using D1 cell volume and booked to
source_cum_mmol. Introduction uses a manifest-listed D2 genome_id; sim resolves
phenotype/species, allocates next_agent_id, and returns persistent agent_id in
the applied result. Selection uses IDs, never row indices. Pause affects only
the scheduler; advance_ticks and single_step execute fixed ticks, no variable
dt/wall-clock input.

### Render projection

RenderSnapshot is a detached C-contiguous read-only copy produced between ticks:
tick_index:uint64; width_um/height_um:float64; agent_id:uint64[N];
position_um:float64[N,2]; radius_um:float64[N]; genome_id:uint64[N];
nutrient_id:uint32[K]; concentration_mmol_l:float64[K,ny,nx];
phenotype_genome_id:uint64[G]; phenotype_motility_um_s:float64[G];
phenotype_max_uptake_mmol_s:float64[G,K]. No live references, RNG, ledger or
command queue are exposed.

Lumen implements `PygameRenderer.draw(target: pygame.Surface,
snapshot: RenderSnapshot) -> RenderStats` (visible_cells, density_cells,
blit_calls). It may read only RenderSnapshot and may write only the target
Surface/renderer caches. It owns Camera2D/atlas, routes pan/zoom via
`handle_event(event,pointer_position=None)->bool`, and offers the read-only
`screen_to_world(point_px)->Vec2Um` query. It never steps the world or queues
commands.

World coordinates are µm with origin upper-left, +x right, +y down, matching
Camera2D. Valid bounds are 0<=x<width_um and 0<=y<height_um; outside placement is
rejected, not periodically wrapped. UI converts screen pixels through the
renderer before submitting world positions; grid cell is floor(x/dx),floor(y/dy).
CellInspection is a detached D1 state snapshot plus accession/organism/taxid,
D2 traits, nutrient stores/biomass and gene/rule evidence. GenomeInspection
returns provenance and annotated genes, not mutable sequence storage. UI/render
may not write state, field, ledger, RNG, command log, genome or phenotype data;
UI may not call Simulation.step.

The final D3 finding is kb_mvc746dnDCSQvHtSJL-6. It supersedes the conflicting
early D3 drafts and is the source for D3 interfaces. D3 requires a deliberate
renderer/UI API migration from the current live-state draw/constructor signatures.

## Implementation conformance (not a contract amendment)

The frozen contracts above are authoritative even where the current tree has
not caught up. Do not infer conformance from a module or backlog status:

- **Decision (backlog item 209, AgentDock `kb_mvcfxfehCE1GD3p4Lhx9`): retain
  frozen D1 unchanged and align the code.**
  The D1 purpose tags remain motility=1, division=2, mutation=3, death=4,
  antagonism=5; `initialize_rngs` must derive a PCG64 stream for every tag using
  `SeedSequence(root_seed, spawn_key=(purpose_tag,))`. Do not re-freeze D1 to
  implementation-specific tags. `src/microbiome/sim/model.py` currently uses
  101/202/303 and exposes only three streams, so it is nonconformant until Cilia
  updates its owned simulation code. The mutation stream remains part of the
  contract although mutations are currently disabled; antagonism also requires
  its own stream. The frozen D1 contract remains authoritative during that fix.
- **Decision (backlog item 211, AgentDock `kb_mvcg10wx-73fU_Uf9ZCy`): retain
  frozen D2 and align the implementation.** The current
  `load_genome(accession, project_root)`/JSON API is a working implementation,
  but it is not a reason to silently amend the frozen contract: immutable
  source-bundle provenance, versioned TSV schemas, producer interfaces, and
  canonical identity are stronger reproducibility/audit guarantees than
  implementation convenience. D2's exact genome ID remains the unsigned
  big-endian integer from the first 8 bytes of SHA-256 over the exact ASCII
  `assembly_accession.version`; reject collisions and preserve the full
  accession.version as canonical identity. The namespace-prefixed hash was
  nonconformant and has been replaced; the code now matches. The migration to the D2
  `load_genome_bundle(Path)`, producer signatures and versioned TSV schemas is done (below).
  Duplicate `locus_tag` values are not identity: use
  `accession.version|locus_tag` only when unique; otherwise key distinct genes
  by accession-version-scoped seqid/type/strand/sorted coordinates. Never fuse
  distinct genes solely because their locus tags match. Multipart feature rows
  may share a gene_key only when their source GFF3 identity establishes that
  they are parts of the same gene; distinct paralogs remain distinct. D2 stays
  frozen and authoritative until implementation catches up. The ID formula
  and the duplicate-locus keying conform and their two xfails are closed.
  **The D2 producer API now exists** (`src/microbiome/genome/bundle.py`,
  `tables.py`): `load_genome_bundle`, `load_trait_schema`, `load_gene_trait_map`,
  `normalize_gene_features`, `build_phenotype_table(genomes, schema, gene_map)`
  and `explain_phenotype`, over `data/trait_schema.tsv` and
  `data/gene_trait_map.tsv` (generated from `data/gene_trait_table.json` by
  `python -m microbiome.genome.tables export`; `tests/test_d2_tables.py` pins
  that the committed TSVs equal the exporter's output and that the TSV path
  reproduces the JSON path bit for bit on both real genomes). The original
  `build_phenotype_table(genomes, nutrient_channels)` call shape still works.
  Two interpretations the D2 text does not settle, recorded in `tables.py`:
  rows sharing one `mapping_id` form an all-of rule (the bacilysin operon needs
  all seven genes); and `gene_features.tsv` rows are produced by
  `gene_feature_rows`, one per GFF3 feature part, while `normalize_gene_features`
  returns one `AnnotatedGene` per gene.
- **D4 is implemented and opt-in** (`src/microbiome/genome/variants.py`):
  `create_derived_variant`, `derive_variant_phenotype`,
  `build_variant_phenotype_table` and the `DerivedVariant` record with the
  hashing of this document, enabled by `SimulationController.create_demo(
  mutation=True)` (on in the interactive app, off by default elsewhere). RNG
  draws per daughter, the sequence-digest definition and the isoform rule are
  fixed in that module's docstring because D4 leaves them open. At the D4 rate a
  phenotype change is very rare (about one mutation per thousand divisions, and
  a change needs a premature stop in a mapped gene).
- **D5 realism extension (additive, opt-in; full description in `docs/REALISM.md`).**
  `SimulationController.create_demo(realism=True)` runs `sim/realism.py`'s
  `RealisticSimulation` through `engine.step(..., simulation=...)` over six field
  channels (glucose, lactose, oxygen, acetate, bacilysin, ai2) with run-and-tumble
  movement, Brownian motion, contact mechanics, saturating uptake, catabolite
  repression, oxygen and fermentation, acetate overflow, pH, a diffusing toxin,
  quorum sensing and biofilm matrix, lag, starvation, sporulation, necromass
  recycling, optional flow and natural transformation. It is a D1/D3/D4 *extension*,
  not an amendment: the D1 step is unchanged bit for bit (`_divide` now returns the
  dividing rows and `step` takes an optional `simulation=`), `RealismSnapshot`
  subclasses `RenderSnapshot` without touching its frozen fields, and
  `DerivedVariant.mutation_model_id` may also be `transformation-snv-v1`. Every
  mechanism runs under the engine's per-tick per-channel ledger assertion. Gene
  effects come from `data/physiology_schema.tsv` / `physiology_map.tsv` through the
  same D2 rules (exact locus tags scoped to one assembly, all-of operons, evidence).
  `tests/test_realism_docs.py` fails if the documentation drifts from the code.
- Core D3 DTOs, the `SimulationController` protocol and its implementation
  (`src/microbiome/sim/controller.py`) exist, and the UI routes its actions
  through the controller facade (`tests/test_ui_smoke.py` enforces the layering
  and that drawing only calls `render_snapshot`). The statements that said the
  controller was absent and that the UI still stepped D1 state directly were
  stale and have been removed.
- `PygameRenderer.draw(target, snapshot)` and `WorldView.from_snapshot` implement
  the render projection; item 14 is satisfied.
