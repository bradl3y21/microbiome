"""Natural transformation: the recombination tract, who may take up whose DNA, and the end-to-end pipeline."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

import realism_support as R
from microbiome.genome import (NucleotideSubstitution, VariantStore, load_genome_bundle, load_gene_trait_map,
                               load_trait_schema, variant_from_substitutions)
from microbiome.genome import variants as V
from microbiome.sim.controller import SimulationController
from test_variants import coding_positions, force_codon, isoforms

ROOT = Path(__file__).resolve().parent.parent
EC, BS = "GCF_000005845.2", "GCF_000009045.1"


@pytest.fixture(scope="module")
def genome():
    return load_genome_bundle(ROOT / "data" / "genomes" / EC)


@pytest.fixture(scope="module")
def tables():
    return load_trait_schema(ROOT / "data" / "trait_schema.tsv"), load_gene_trait_map(ROOT / "data" / "gene_trait_map.tsv")


@pytest.fixture()
def store():
    return VariantStore(ROOT)


def snv(store, genome, position):
    ref = int(store.root(genome).codes[position])
    return NucleotideSubstitution(position, V.BASES[ref], V.BASES[(ref + 1) % 4])


def craft(genome, tables, store, subs, parent=None, daughter=9):
    schema, gmap = tables
    return variant_from_substitutions(genome, parent, subs, origin_tick=5, parent_agent_id=1, daughter_agent_id=daughter,
                                      schema=schema, gene_trait_map=gmap, trait_schema_sha256=schema.sha256,
                                      gene_trait_map_sha256=gmap.sha256, store=store)[0]


def transform(genome, tables, store, recipient, donor, start, length, recipient_id=20, donor_id=10):
    schema, gmap = tables
    return V.create_transformant(genome, recipient, donor, window_start=start, window_length=length, origin_tick=50,
                                 recipient_agent_id=recipient_id, donor_agent_id=donor_id, schema=schema, gene_trait_map=gmap,
                                 trait_schema_sha256=schema.sha256, gene_trait_map_sha256=gmap.sha256, store=store)


# ------------------------------------------------------------------------------ the tract
def test_the_recipient_gains_donor_alleles_inside_the_tract_and_only_there(genome, tables, store):
    inside, outside = 500_000, 900_000
    donor = craft(genome, tables, store, [snv(store, genome, inside), snv(store, genome, outside)])
    variant, phenotype = transform(genome, tables, store, None, donor, inside - 10, 50)
    assert [s.position_0based for s in variant.substitutions] == [inside]
    assert variant.substitutions[0] == snv(store, genome, inside)
    got = store.sequence(genome, variant)
    root = store.root(genome).codes
    assert got[inside] == (root[inside] + 1) % 4 and got[outside] == root[outside]
    assert (got != root).sum() == 1
    variant.verify()
    assert variant.mutation_model_id == V.TRANSFORMATION_MODEL_ID
    assert (variant.parent_agent_id, variant.daughter_agent_id) == (10, 20), "the donor is the source, the recipient the daughter"
    assert variant.parent_variant_id == 0 and phenotype.variant_id == variant.variant_id


def test_a_donor_wild_type_base_repairs_a_recipient_mutation(genome, tables, store):
    p = 700_000
    mutant = craft(genome, tables, store, [snv(store, genome, p)])
    repaired, _ = transform(genome, tables, store, mutant, None, p - 100, 200)
    root = store.root(genome)
    assert repaired.substitutions == (NucleotideSubstitution(p, V.BASES[(int(root.codes[p]) + 1) % 4], V.BASES[int(root.codes[p])]),)
    assert repaired.derived_sequence_sha256 == root.sha256, "the repaired chromosome is the wild type again"
    assert repaired.parent_variant_id == mutant.variant_id and repaired.parent_record_sha256 == mutant.record_sha256
    assert np.array_equal(store.sequence(genome, repaired), root.codes)


def test_no_difference_in_the_tract_means_no_new_variant(genome, tables, store):
    p = 600_000
    donor = craft(genome, tables, store, [snv(store, genome, p)])
    assert transform(genome, tables, store, None, None, 0, 4_000_000) is None, "two wild types are identical"
    assert transform(genome, tables, store, donor, donor, 0, 4_000_000) is None, "the same variant everywhere"
    assert transform(genome, tables, store, None, donor, p + 1000, 500) is None, "a difference outside the tract is not transferred"


def test_the_tract_wraps_around_the_circular_chromosome(genome, tables, store):
    length = genome.sequence_length_bp
    donor = craft(genome, tables, store, [snv(store, genome, 10)])
    variant, _ = transform(genome, tables, store, None, donor, length - 20, 60)
    assert [s.position_0based for s in variant.substitutions] == [10]
    assert transform(genome, tables, store, None, donor, length - 20, 25) is None, "a tract that stops before the origin misses it"


def test_a_tract_longer_than_the_chromosome_is_just_the_whole_chromosome(genome, tables, store):
    donor = craft(genome, tables, store, [snv(store, genome, 1_000_000), snv(store, genome, 3_000_000)])
    variant, _ = transform(genome, tables, store, None, donor, 123, 10**9)
    assert len(variant.substitutions) == 2


def test_a_transferred_knockout_removes_the_recipients_gene_function(genome, tables, store):
    schema, gmap = tables
    che_a = next(g for g in genome.genes if g.locus_tag == "b1888")
    short, _ = isoforms(che_a)
    knock = force_codon(store.root(genome).codes, che_a, short, 10, "TAA")
    donor = craft(genome, tables, store, knock)
    variant, phenotype = transform(genome, tables, store, None, donor, che_a.start - 500, 3000)
    from microbiome.genome.tables import derive_genome_phenotype

    wild = derive_genome_phenotype(genome, schema, gmap)
    assert phenotype.chemotaxis_sensitivity == pytest.approx(wild.chemotaxis_sensitivity - 0.35)
    assert "ecoli-chea-chemotaxis" not in phenotype.applied_rule_ids
    assert V.derive_variant_phenotype(genome, variant, schema, gmap, store=store) == phenotype


def test_transformation_across_species_is_refused(genome, tables, store):
    other = dataclasses.replace(genome, genome_id=genome.genome_id + 1)
    donor = craft(genome, tables, store, [snv(store, genome, 5000)])
    with pytest.raises(ValueError, match="same accession genome"):
        transform(other, tables, store, None, donor, 0, 10_000)


def test_the_window_is_a_fixed_function_of_ids_and_tick_and_always_in_range():
    a = V.transformation_window_start(5, 9, 100, 4_641_652)
    assert a == V.transformation_window_start(5, 9, 100, 4_641_652)
    starts = {V.transformation_window_start(r, d, t, 4_641_652) for r in range(1, 20) for d in range(20, 30) for t in (1, 2)}
    assert len(starts) > 300 and all(0 <= s < 4_641_652 for s in starts), "spread over the chromosome, never out of range"


# ------------------------------------------------------------------------------ who may take up whose DNA
class Recorder:
    def __init__(self):
        self.calls = []

    def create_transformant(self, genome_id, recipient_variant, donor_variant, **kw):
        self.calls.append((genome_id, recipient_variant, donor_variant, kw["recipient_agent_id"], kw["donor_agent_id"]))
        return None


def fragments(*rows):
    """rows: (grid cell index, genome id, variant id, donor agent id, tick)."""
    cols = list(zip(*rows)) if rows else [[]] * 5
    return {"cell": np.array(cols[0], dtype=np.int64), "genome": np.array(cols[1], dtype=np.uint64), "variant": np.array(cols[2], dtype=np.uint64),
            "donor": np.array(cols[3], dtype=np.uint64), "tick": np.array(cols[4], dtype=np.int64)}


def competent_world(n=1, *, capacity=1.0, **extra):
    cfg = R.quiet(transformation_rate_per_s=1e6, lag_initial=1.0, **extra)
    positions = [(105.0 + 2 * i, 105.0) for i in range(n)]
    w = R.world(positions, config=cfg, energy=1e9, signal=2e-3, phys=R.physiology(competence_capacity=capacity),
                trait=R.traits(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6)))
    w.sim.transformer = Recorder()
    return w


CELL = 10 * 20 + 10          # the grid cell containing (105, 105) on a 20-wide grid


def test_a_competent_cell_takes_up_dna_from_its_own_species_in_its_own_grid_cell():
    w = competent_world()
    w.sim.dna = fragments((CELL, R.GENOME, 0, 77, 0))
    w.step()
    assert w.sim.transformer.calls == [(R.GENOME, 0, 0, 1, 77)]
    assert w.sim.dna["cell"].size == 0, "a fragment is used up"


@pytest.mark.parametrize("fragment, why", [
    ((CELL + 1, R.GENOME, 0, 77, 0), "DNA in a neighbouring grid cell is out of reach"),
    ((CELL, R.GENOME + 1, 0, 77, 0), "another species' DNA is not recombined"),
    ((CELL, R.GENOME, 0, 1, 0), "a cell cannot take up its own DNA"),
])
def test_dna_the_cell_cannot_use_is_ignored_and_left_in_the_pool(fragment, why):
    w = competent_world()
    w.sim.dna = fragments(fragment)
    w.step()
    assert w.sim.transformer.calls == [], why
    assert w.sim.dna["cell"].size == 1


def test_a_cell_without_the_competence_genes_never_transforms():
    w = competent_world(capacity=0.0)
    w.sim.dna = fragments((CELL, R.GENOME, 0, 77, 0))
    w.step(30)
    assert w.sim.transformer.calls == []


def test_a_cell_is_competent_only_when_starving_or_the_quorum_is_high():
    calm = R.world(((105.0, 105.0),), config=R.quiet(transformation_rate_per_s=1e6, lag_initial=1.0), energy=1e9,
                   phys=R.physiology(competence_capacity=1.0), trait=R.traits(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6)))
    calm.sim.transformer = Recorder()
    calm.sim.dna = fragments((CELL, R.GENOME, 0, 77, 0))
    calm.step(5)
    assert calm.sim.transformer.calls == [], "fed, sparse cells are not competent"
    starving = R.world(((105.0, 105.0),), config=R.quiet(transformation_rate_per_s=1e6, initial_energy_reserve_s=0.0, starvation_tau_s=2.0),
                       energy=0.0, phys=R.maintaining(competence_capacity=1.0, sporulation_capacity=0.0),
                       trait=R.traits(division_rate_s=0.0, death_rate_s=0.0, max_uptake_mmol_s=np.zeros(6)))
    starving.sim.transformer = Recorder()
    starving.sim.dna = fragments((CELL, R.GENOME, 0, 77, 99999))
    starving.step(300)
    assert len(starving.sim.transformer.calls) == 1, "starvation stress makes it competent"


def test_one_fragment_serves_one_recipient_and_old_dna_is_degraded():
    w = competent_world(2)
    w.sim.dna = fragments((CELL, R.GENOME, 0, 77, 0))
    w.step()
    assert len(w.sim.transformer.calls) == 1, "two competent cells, one fragment"
    old = competent_world()
    old.sim.dna = fragments((CELL, R.GENOME, 0, 77, 0))
    old.state.tick = int(old.sim.config.dna_lifetime_s * 60) + 100
    old.step()
    assert old.sim.transformer.calls == [] and old.sim.dna["cell"].size == 0, "DNA older than its lifetime is gone"


def test_cells_that_die_leave_their_dna_and_without_a_transformer_nothing_is_kept():
    cfg = R.quiet(lag_initial=1.0)
    lethal = R.traits(division_rate_s=0.0, death_rate_s=1e6, max_uptake_mmol_s=np.zeros(6))
    w = R.world([(101.0 + 2 * i, 105.0) for i in range(5)], config=cfg, energy=1e9, trait=lethal)
    w.sim.transformer = Recorder()
    w.step()
    assert w.n == 0 and w.sim.dna["cell"].size == 5
    assert (w.sim.dna["genome"] == R.GENOME).all() and (w.sim.dna["cell"] == CELL).all()
    assert sorted(w.sim.dna["donor"].tolist()) == [1, 2, 3, 4, 5]
    bare = R.world([(105.0, 105.0)], config=cfg, energy=1e9, trait=lethal)
    bare.step()
    assert bare.n == 0 and bare.sim.dna["cell"].size == 0


def test_the_pool_never_grows_past_its_limit():
    cfg = R.quiet(lag_initial=1.0, dna_pool_limit=3)
    lethal = R.traits(division_rate_s=0.0, death_rate_s=1e6, max_uptake_mmol_s=np.zeros(6))
    w = R.world([(101.0 + i, 105.0) for i in range(8)], config=cfg, energy=1e9, trait=lethal)
    w.sim.transformer = Recorder()
    w.step()
    assert w.sim.dna["cell"].size == 3


# ------------------------------------------------------------------------------ the whole pipeline
class ForceOne:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def poisson(self, lam):
        return 1

    def __getattr__(self, name):
        return getattr(self.rng, name)


def test_end_to_end_a_competent_bacillus_acquires_a_dead_neighbours_mutation():
    from microbiome.sim.engine import _register_variant_phenotype

    length = 4_215_606
    cfg = dataclasses.replace(__import__("microbiome.sim.realism", fromlist=["RealismConfig"]).RealismConfig(),
                              transformation_rate_per_s=1e6, transformation_window_bp=length)
    c = SimulationController.create_demo(seed=3, realism=cfg, mutation=True)
    bs = next(int(o.genome_id) for o in c.list_genomes() if o.accession_version == BS)
    c.advance_ticks(1)
    c.introduce_organism(bs, (205.0, 205.0))                                  # the recipient: agent 0 in the controller's numbering
    c.advance_ticks(1)
    recipient = int(c._state.agents.agent_id[0])
    variant, phenotype = c._mutation.create_offspring_variant(bs, 0, mutation_rng=ForceOne(1), origin_tick=2,
                                                              parent_agent_id=900, daughter_agent_id=901)
    _register_variant_phenotype(c._phenotypes, phenotype, bs, variant.variant_id)
    sim = c._realism.simulation
    cell = int(205 // 10) * 80 + int(205 // 10)
    sim.dna = fragments((cell, bs, int(variant.variant_id), 901, int(c._state.tick_index)))
    c.place_nutrient(5, (205.0, 205.0), 1e-14)                                 # a burst of the quorum signal: the cell becomes competent
    c.advance_ticks(5)
    now = int(c._state.agents.variant_id[0])
    assert now != 0, "the recipient's chromosome now differs from the wild type"
    record = c._mutation.store.get(now)
    assert record.mutation_model_id == V.TRANSFORMATION_MODEL_ID and record.daughter_agent_id == recipient
    assert record.substitutions == variant.substitutions, "it took exactly the donor's mutation"
    keys = set(zip(c._phenotypes.genome_id.tolist(), c._phenotypes.variant_id.tolist()))
    assert (bs, now) in keys, "the new variant has a row in the phenotype table, in all seven channels"
    assert c._phenotypes.max_uptake_mmol_s.shape[1] == 7
    c.advance_ticks(60)                                                         # and the world keeps conserving mass afterwards
    assert sim.dna["cell"].size == 0
