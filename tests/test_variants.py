"""D4: SNV-derived variants of real accession genomes (rate model, records, knockouts, engine integration).

Mutation is rare by construction (about 1e-3 per division, and a phenotype change needs a premature stop in a
mapped gene), so these tests force the rare events: ``ForceK`` pins the Poisson count and everything else is the
real generator; specific mutations are built with ``variant_from_substitutions``.
"""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from microbiome.genome import (
    DerivedVariant,
    NucleotideSubstitution,
    VariantMutationProvider,
    VariantStore,
    build_variant_phenotype_table,
    create_derived_variant,
    derive_variant_phenotype,
    load_genome_bundle,
    load_gene_trait_map,
    load_trait_schema,
    variant_from_substitutions,
)
from microbiome.genome import variants as V
from microbiome.genome.tables import derive_genome_phenotype, mapped_genes

ROOT = Path(__file__).resolve().parent.parent
EC = "GCF_000005845.2"
TICK, PARENT, DAUGHTER = 41, 7, 8


@pytest.fixture(scope="module")
def genome():
    return load_genome_bundle(ROOT / "data" / "genomes" / EC)


@pytest.fixture(scope="module")
def tables():
    return load_trait_schema(ROOT / "data" / "trait_schema.tsv"), load_gene_trait_map(ROOT / "data" / "gene_trait_map.tsv")


@pytest.fixture()
def store():
    return VariantStore(ROOT)


class ForceK:
    """A real Generator except that ``poisson`` returns a pinned count (and records its lambda)."""

    def __init__(self, seed, k):
        self.rng, self.k, self.lams = np.random.default_rng(seed), k, []

    def poisson(self, lam):
        self.lams.append(lam)
        return self.k

    def __getattr__(self, name):
        return getattr(self.rng, name)


def make(genome, tables, store, rng, parent=None, tick=TICK, parent_agent=PARENT, daughter=DAUGHTER):
    schema, gmap = tables
    return create_derived_variant(
        genome, parent, mutation_rng=rng, origin_tick=tick, parent_agent_id=parent_agent, daughter_agent_id=daughter,
        schema=schema, gene_trait_map=gmap, trait_schema_sha256=schema.sha256, gene_trait_map_sha256=gmap.sha256,
        store=store)


def craft(genome, tables, store, substitutions, parent=None, tick=TICK, parent_agent=PARENT, daughter=DAUGHTER):
    schema, gmap = tables
    return variant_from_substitutions(
        genome, parent, substitutions, origin_tick=tick, parent_agent_id=parent_agent, daughter_agent_id=daughter,
        schema=schema, gene_trait_map=gmap, trait_schema_sha256=schema.sha256, gene_trait_map_sha256=gmap.sha256,
        store=store)


# ------------------------------------------------------------------------------ the rate model
def test_the_poisson_rate_is_two_e_minus_ten_per_base_over_the_whole_genome(genome, tables, store):
    rng = ForceK(1, 0)
    assert make(genome, tables, store, rng) is None
    assert rng.lams == [pytest.approx(genome.sequence_length_bp * 2.0e-10, rel=1e-12)]
    assert V.MUTATION_MODEL_ID == "snv-poisson-2e-10-v1"


def test_k_zero_returns_none_and_draws_nothing_else(genome, tables, store):
    a, b = np.random.default_rng(6), np.random.default_rng(6)   # seed 6's first K>0 is at draw 4343
    lam = genome.sequence_length_bp * 2.0e-10
    zeros = 0
    for _ in range(200):
        k = int(b.poisson(lam))                                  # the twin says what this division should draw
        result = make(genome, tables, store, a)
        assert (result is None) == (k == 0)
        if k:
            break                                                # a real mutation consumes more draws; stop comparing
        zeros += 1
        assert a.bit_generator.state == b.bit_generator.state, "a K=0 division must consume exactly one draw"
    assert zeros >= 150 and len(store) <= 1


def test_a_forced_mutation_picks_distinct_sorted_positions_and_the_other_three_bases(genome, tables, store):
    variant, _ = make(genome, tables, store, ForceK(3, 6))
    positions = [s.position_0based for s in variant.substitutions]
    assert len(positions) == 6 == len(set(positions)) and positions == sorted(positions)
    root = store.root(genome).codes
    for s in variant.substitutions:
        assert V.BASES[int(root[s.position_0based])] == s.ref, "ref is the inherited base"
        assert s.alt in V.BASES and s.alt != s.ref
    alts = {s.alt for s in make(genome, tables, VariantStore(ROOT), ForceK(9, 300))[0].substitutions}
    assert alts == set("ACGT"), "all four bases occur among 300 random alts"


def test_the_same_seed_gives_the_same_record_bit_for_bit(genome, tables):
    first = make(genome, tables, VariantStore(ROOT), ForceK(11, 4))[0]
    again = make(genome, tables, VariantStore(ROOT), ForceK(11, 4))[0]
    other = make(genome, tables, VariantStore(ROOT), ForceK(12, 4))[0]
    assert first == again and first.to_json() == again.to_json()
    assert first.variant_id != other.variant_id
    later = make(genome, tables, VariantStore(ROOT), ForceK(11, 4), tick=TICK + 1)[0]
    assert later.variant_id != first.variant_id, "lineage fields are part of the hashed payload"


# ------------------------------------------------------------------------------ the record
def test_record_fields_ids_and_hashes_follow_the_spec(genome, tables, store):
    variant, phenotype = make(genome, tables, store, ForceK(2, 2))
    schema, gmap = tables
    d = variant.to_json_dict()
    assert set(d) == {"schema_version", "variant_id", "record_sha256", "root_accession_version", "root_genome_id",
                      "root_sequence_sha256", "root_annotation_sha256", "parent_variant_id", "parent_record_sha256",
                      "parent_sequence_sha256", "derived_sequence_sha256", "sequence_length_bp", "origin_tick",
                      "parent_agent_id", "daughter_agent_id", "mutation_model_id", "substitutions",
                      "trait_schema_sha256", "gene_trait_map_sha256", "phenotype_sha256"}
    payload = {k: v for k, v in d.items() if k not in {"variant_id", "record_sha256"}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    assert d["record_sha256"] == digest
    assert d["variant_id"] == str(int.from_bytes(bytes.fromhex(digest)[:8], "big")) and variant.variant_id != 0
    for key in ("root_genome_id", "parent_variant_id", "sequence_length_bp", "origin_tick", "parent_agent_id",
                "daughter_agent_id", "variant_id"):
        assert isinstance(d[key], str), f"{key} is a decimal string in JSON"
    assert all(isinstance(s["position_0based"], int) for s in d["substitutions"])
    assert d["root_genome_id"] == str(genome.genome_id) and d["parent_record_sha256"] is None
    assert d["parent_sequence_sha256"] == store.root(genome).sha256 and d["derived_sequence_sha256"] != d["parent_sequence_sha256"]
    assert d["trait_schema_sha256"] == schema.sha256 and d["gene_trait_map_sha256"] == gmap.sha256
    assert d["root_sequence_sha256"] == genome.sequence_sha256 and d["root_annotation_sha256"] == genome.annotation_sha256
    assert phenotype.variant_id == variant.variant_id and phenotype.genome_id == genome.genome_id
    assert DerivedVariant.from_json_dict(json.loads(variant.to_json())) == variant


def test_any_edit_to_a_record_is_detected(genome, tables, store):
    variant, _ = make(genome, tables, store, ForceK(4, 2))
    d = variant.to_json_dict()
    for key, bad in (("origin_tick", "1"), ("daughter_agent_id", "999"), ("phenotype_sha256", "0" * 64),
                     ("trait_schema_sha256", "1" * 64), ("mutation_model_id", "other")):
        edited = {**d, key: bad}
        with pytest.raises(ValueError):
            DerivedVariant.from_json_dict(edited)
    subs = [dict(s) for s in d["substitutions"]]
    subs[0]["alt"] = next(b for b in "ACGT" if b not in (subs[0]["alt"], subs[0]["ref"]))
    with pytest.raises(ValueError, match="record_sha256"):
        DerivedVariant.from_json_dict({**d, "substitutions": subs})
    with pytest.raises(ValueError, match="variant_id"):
        DerivedVariant.from_json_dict({**d, "variant_id": "5"})


def test_substitutions_must_change_the_base():
    with pytest.raises(ValueError, match="ref != alt"):
        NucleotideSubstitution(5, "A", "A")
    with pytest.raises(ValueError, match="A, C, G, T"):
        NucleotideSubstitution(5, "A", "N")


# ------------------------------------------------------------------------------ chains
def test_a_child_chains_to_its_parent_and_reconstructs_both_edits(genome, tables, store):
    parent, _ = make(genome, tables, store, ForceK(21, 2))
    child, _ = make(genome, tables, store, ForceK(22, 3), parent=parent, tick=TICK + 5, parent_agent=DAUGHTER, daughter=9)
    assert child.parent_variant_id == parent.variant_id and child.parent_record_sha256 == parent.record_sha256
    assert child.parent_sequence_sha256 == parent.derived_sequence_sha256
    root = store.root(genome).codes
    by_hand = np.array(root)
    for s in parent.substitutions + child.substitutions:
        by_hand[s.position_0based] = V.BASES.index(s.alt)
    assert np.array_equal(store.sequence(genome, child), by_hand)
    assert [v.variant_id for v in store.chain(child)] == [parent.variant_id, child.variant_id]


def test_reconstruction_rejects_a_wrong_ref_allele_and_a_wrong_digest(genome, tables, store):
    parent, _ = make(genome, tables, store, ForceK(31, 1))
    schema, gmap = tables
    s = parent.substitutions[0]
    wrong_ref = next(b for b in "ACGT" if b not in (s.ref, s.alt))
    forged = V._seal(**{**{f.name: getattr(parent, f.name) for f in dataclasses.fields(parent)
                           if f.name not in {"variant_id", "record_sha256"}},
                        "substitutions": (NucleotideSubstitution(s.position_0based, wrong_ref, s.alt),)})
    fresh = VariantStore(ROOT)
    fresh.register(forged)
    with pytest.raises(ValueError, match="ref allele"):
        fresh.sequence(genome, forged)
    bad_digest = V._seal(**{**{f.name: getattr(parent, f.name) for f in dataclasses.fields(parent)
                               if f.name not in {"variant_id", "record_sha256"}},
                            "derived_sequence_sha256": "0" * 64})
    fresh.register(bad_digest)
    with pytest.raises(ValueError, match="derived sequence digest"):
        fresh.sequence(genome, bad_digest)


def test_a_variant_id_may_not_collide_with_an_accession_genome_id(genome, tables, store):
    variant, _ = make(genome, tables, store, ForceK(41, 1))
    other = VariantStore(ROOT)
    other.reserve_genome_ids([variant.variant_id])
    with pytest.raises(ValueError, match="collides with an accession genome id"):
        other.register(variant)


def test_a_parent_from_another_genome_is_rejected(genome, tables, store):
    schema, gmap = tables
    parent, _ = make(genome, tables, store, ForceK(51, 1))
    alien = dataclasses.replace(genome, genome_id=genome.genome_id + 1)
    with pytest.raises(ValueError, match="different accession genome"):
        make(alien, tables, store, ForceK(52, 1), parent=parent)


# ------------------------------------------------------------------------------ knockouts
def coding_positions(gene, cds_id):
    parts = sorted((p for p in gene.cds_parts if p[3] == cds_id), key=lambda p: p[0], reverse=gene.strand == "-")
    positions = []
    for start, end, _phase, _ in parts:
        positions.extend(range(start - 1, end) if gene.strand == "+" else range(end - 1, start - 2, -1))
    return positions[parts[0][2]:]


def force_codon(codes, gene, cds_id, codon_index, codon):
    positions = coding_positions(gene, cds_id)
    out = []
    for j, base in enumerate(codon):
        p = positions[3 * codon_index + j]
        want = V.BASES.index(base) if gene.strand == "+" else 3 - V.BASES.index(base)
        if int(codes[p]) != want:
            out.append(NucleotideSubstitution(p, V.BASES[int(codes[p])], V.BASES[want]))
    return out


@pytest.fixture(scope="module")
def che_a(genome):
    return next(g for g in genome.genes if g.locus_tag == "b1888")


def isoforms(gene):
    ids = sorted({p[3] for p in gene.cds_parts}, key=lambda i: len(coding_positions(gene, i)))
    return ids  # shortest first


def test_a_premature_stop_in_a_mapped_gene_removes_exactly_its_contribution(genome, tables, store, che_a):
    schema, gmap = tables
    root = store.root(genome).codes
    short, long_ = isoforms(che_a)
    # codon 10 of the SHORT isoform lies in the region both isoforms share, in the same frame
    subs = force_codon(root, che_a, short, 10, "TAA")
    assert subs, "the chosen codon is not already a stop"
    variant, phenotype = craft(genome, tables, store, subs)
    wild = derive_genome_phenotype(genome, schema, gmap)
    assert "ecoli-chea-chemotaxis" in wild.applied_rule_ids and "ecoli-chea-chemotaxis" not in phenotype.applied_rule_ids
    assert phenotype.chemotaxis_sensitivity == pytest.approx(wild.chemotaxis_sensitivity - 0.35, abs=1e-12)
    for name in ("motility_um_s", "division_rate_s", "max_uptake_mmol_s", "antagonism_rate_s", "death_rate_s"):
        assert getattr(phenotype, name) == getattr(wild, name), f"only cheA's trait may move, not {name}"
    expected = dataclasses.replace(derive_genome_phenotype(genome, schema, gmap, knockouts=frozenset({che_a.gene_key})),
                                   variant_id=variant.variant_id)
    assert phenotype == expected
    assert derive_variant_phenotype(genome, variant, schema, gmap, store=store) == phenotype


def test_a_stop_only_in_one_isoforms_extension_does_not_knock_the_gene_out(genome, tables, store, che_a):
    schema, gmap = tables
    root = store.root(genome).codes
    short, long_ = isoforms(che_a)
    # codon 5 of the LONG isoform is inside its extra N-terminal region, which the short isoform lacks
    subs = force_codon(root, che_a, long_, 5, "TAG")
    variant, phenotype = craft(genome, tables, store, subs)
    wild = derive_genome_phenotype(genome, schema, gmap)
    assert variant.substitutions and phenotype.chemotaxis_sensitivity == wild.chemotaxis_sensitivity
    assert phenotype.applied_rule_ids == wild.applied_rule_ids


def test_non_events_leave_the_phenotype_unchanged(genome, tables, store, che_a):
    schema, gmap = tables
    root = store.root(genome).codes
    wild = derive_genome_phenotype(genome, schema, gmap)
    short, _ = isoforms(che_a)
    positions = coding_positions(che_a, short)
    cases = {}
    third = positions[3 * 20 + 2]                                  # wobble base of an interior codon
    for alt in "ACGT":
        if V.BASES[int(root[third])] != alt:
            codon = [int(root[p]) for p in positions[3 * 20: 3 * 20 + 3]]
            if che_a.strand == "-":
                codon = [3 - c for c in codon]
            trial = codon[:2] + [V.BASES.index(alt) if che_a.strand == "+" else 3 - V.BASES.index(alt)]
            if trial[0] * 16 + trial[1] * 4 + trial[2] not in (48, 50, 56):
                cases["synonymous-or-missense"] = [NucleotideSubstitution(third, V.BASES[int(root[third])], alt)]
                break
    unmapped = next(g for g in genome.genes if g.cds_parts and not g.pseudogene and g.strand == "+"
                    and g.gene_key not in {m.gene_key for m in mapped_genes(genome, gmap)}
                    and len({p[3] for p in g.cds_parts}) == 1 and len(coding_positions(g, g.cds_parts[0][3])) > 90)
    cases["stop in an unmapped gene"] = force_codon(root, unmapped, unmapped.cds_parts[0][3], 10, "TAA")
    intergenic = next(i for i in range(10, 400) if all(not (g.start <= i + 1 <= g.end) for g in genome.genes))
    cases["intergenic"] = [NucleotideSubstitution(intergenic, V.BASES[int(root[intergenic])],
                                                  "A" if int(root[intergenic]) != 0 else "C")]
    for label, subs in cases.items():
        variant, phenotype = craft(genome, tables, VariantStore(ROOT), subs, daughter=100)
        assert phenotype == dataclasses.replace(wild, variant_id=variant.variant_id), label
        assert variant.phenotype_sha256 == V.phenotype_sha256(wild), label


def test_changing_the_terminal_stop_to_another_stop_is_not_premature(genome, tables, store, che_a):
    schema, gmap = tables
    root = store.root(genome).codes
    for cds_id in isoforms(che_a):
        n = len(coding_positions(che_a, cds_id)) // 3
        current = "".join(V.BASES[int(root[p])] if che_a.strand == "+" else V.BASES[3 - int(root[p])]
                          for p in coding_positions(che_a, cds_id)[-3:])
        assert current in {"TAA", "TAG", "TGA"}
    short, _ = isoforms(che_a)
    n = len(coding_positions(che_a, short)) // 3
    swap = "TAG" if "".join(V.BASES[int(root[p])] if che_a.strand == "+" else V.BASES[3 - int(root[p])]
                            for p in coding_positions(che_a, short)[-3:]) != "TAG" else "TGA"
    _, phenotype = craft(genome, tables, store, force_codon(root, che_a, short, n - 1, swap))
    assert "ecoli-chea-chemotaxis" in phenotype.applied_rule_ids


def test_an_isoform_that_no_longer_ends_in_a_stop_is_undecided_not_knocked_out(genome, tables, store, che_a):
    """Documented policy: only a CDS that still ends in a stop codon is judged; otherwise the gene is left alone."""
    schema, gmap = tables
    root = store.root(genome).codes
    short, long_ = isoforms(che_a)
    n_short = len(coding_positions(che_a, short)) // 3
    n_long = len(coding_positions(che_a, long_)) // 3
    subs = force_codon(root, che_a, short, 10, "TAA")                      # an early stop in the shared region...
    by_position = {s.position_0based: s for s in subs}
    for cds_id, n in ((short, n_short), (long_, n_long)):                  # ...and neither isoform ends in a stop
        for s in force_codon(root, che_a, cds_id, n - 1, "TTT"):
            by_position[s.position_0based] = s
    _, phenotype = craft(genome, tables, store, list(by_position.values()))
    wild = derive_genome_phenotype(genome, schema, gmap)
    assert phenotype.chemotaxis_sensitivity == wild.chemotaxis_sensitivity
    assert "ecoli-chea-chemotaxis" in phenotype.applied_rule_ids


def test_an_inherited_knockout_survives_later_harmless_mutations(genome, tables, store, che_a):
    schema, gmap = tables
    root = store.root(genome).codes
    short, _ = isoforms(che_a)
    parent, knocked = craft(genome, tables, store, force_codon(root, che_a, short, 10, "TAA"))
    wild = derive_genome_phenotype(genome, schema, gmap)
    child, child_pheno = make(genome, tables, store, ForceK(61, 2), parent=parent, tick=TICK + 3, daughter=DAUGHTER + 1)
    assert child_pheno.chemotaxis_sensitivity == knocked.chemotaxis_sensitivity < wild.chemotaxis_sensitivity
    assert "ecoli-chea-chemotaxis" not in child_pheno.applied_rule_ids


def test_a_recorded_phenotype_hash_is_checked_on_rederivation(genome, tables, store):
    schema, gmap = tables
    variant, _ = make(genome, tables, store, ForceK(71, 1))
    edited = dataclasses.replace(gmap, sha256="2" * 64)
    with pytest.raises(ValueError, match="hashes"):
        derive_variant_phenotype(genome, variant, schema, edited, store=store)


# ------------------------------------------------------------------------------ tables
def test_the_pair_keyed_table_has_every_genome_and_variant_sorted_by_the_pair(genome, tables, store, che_a):
    schema, gmap = tables
    root = store.root(genome).codes
    short, _ = isoforms(che_a)
    v1, _ = craft(genome, tables, store, force_codon(root, che_a, short, 10, "TAA"), daughter=20)
    v2, _ = make(genome, tables, store, ForceK(81, 1), daughter=21)
    table = build_variant_phenotype_table([genome], [v2, v1], schema, gmap, store=store)
    assert table.genome_id.tolist() == [genome.genome_id] * 3
    assert table.variant_id.tolist() == sorted([0, v1.variant_id, v2.variant_id])
    assert table.max_uptake_mmol_s.shape == (3, 2) and table.motility_um_s.shape == (3,)
    row = {int(v): i for i, v in enumerate(table.variant_id)}
    assert table.chemotaxis_sensitivity[row[v1.variant_id]] < table.chemotaxis_sensitivity[0]
    with pytest.raises(ValueError, match="not in the catalogue"):
        build_variant_phenotype_table([dataclasses.replace(genome, genome_id=5)], [v1], schema, gmap, store=store)


# ------------------------------------------------------------------------------ engine integration
class ForcedProvider(VariantMutationProvider):
    """The real provider, but every daughter draws K = 1 (so every division mutates)."""

    def create_offspring_variant(self, genome_id, parent_variant_id, *, mutation_rng, **lineage):
        return super().create_offspring_variant(genome_id, parent_variant_id,
                                                mutation_rng=ForceK(int(lineage["daughter_agent_id"]), 1), **lineage)


def dividing_controller(seed=2026):
    from microbiome.sim.controller import SimulationController

    controller = SimulationController.create_demo(seed=seed, mutation=False)
    provider = ForcedProvider(controller._genomes, load_trait_schema(ROOT / "data" / "trait_schema.tsv"),
                              load_gene_trait_map(ROOT / "data" / "gene_trait_map.tsv"), project_root=ROOT)
    controller._mutation = provider
    table = controller._phenotypes
    table.division_mass_pg[:] = 1e-6                    # every cell is above the division mass
    table.division_rate_s[:] = 1e6                      # and divides with probability ~1 per tick
    table.death_rate_s[:] = 0.0
    controller.advance_ticks(1)
    controller.introduce_organism(int(controller.list_genomes()[0].genome_id), (200.0, 300.0))
    return controller, provider


def test_divisions_drive_the_provider_and_the_engine_registers_pair_keyed_rows():
    controller, provider = dividing_controller()
    controller.advance_ticks(4)
    state, table = controller._state, controller._phenotypes
    n = len(state.agents.agent_id)
    # Only the founder's row was edited to divide every tick. Each daughter carries a variant, and the engine
    # correctly uses that variant's own phenotype row (the real division traits), so growth is one cell per tick.
    assert n >= 4
    daughters = int(np.count_nonzero(state.agents.variant_id))
    assert daughters == len(provider.store) == n - 1, "the founder keeps variant 0; every daughter got a variant"
    keys = list(zip(table.genome_id.tolist(), table.variant_id.tolist()))
    assert keys == sorted(keys) and len(keys) == len(set(keys)) == len(controller.list_genomes()) + len(provider.store)
    assert set(state.agents.variant_id.tolist()) - {0} == set(v for _, v in keys) - {0}
    snapshot = controller.render_snapshot()          # the renderer still sees unique root genome rows
    assert len(set(snapshot.phenotype_genome_id.tolist())) == len(snapshot.phenotype_genome_id) == len(controller.list_genomes())
    for variant in provider.store._variants.values():
        variant.verify()
        assert len(variant.substitutions) == 1 and int(variant.origin_tick) >= 1


def test_the_mutating_simulation_is_deterministic_and_conserves_nutrient():
    def run():
        controller, provider = dividing_controller(seed=33)
        controller.place_nutrient(0, (200.0, 300.0), 1e-12)
        controller.advance_ticks(6)                       # the engine asserts the ledger every step
        a = controller._state.agents
        return (a.agent_id.tobytes(), a.variant_id.tobytes(), a.position_um.tobytes(),
                controller._phenotypes.variant_id.tobytes(), controller._environment.field.concentration_mmol_l.tobytes(),
                tuple(sorted(provider.store._variants)))

    assert run() == run()


def test_mutation_off_is_the_unchanged_no_provider_behaviour():
    from microbiome.sim.controller import SimulationController

    def run(flag):
        controller = SimulationController.create_demo(seed=5, mutation=flag)
        controller.advance_ticks(1)
        controller.introduce_organism(int(controller.list_genomes()[0].genome_id), (100.0, 100.0))
        controller.advance_ticks(30)
        a = controller._state.agents
        return a.position_um.tobytes(), a.variant_id.tobytes(), controller._phenotypes.genome_id.tobytes()

    off, on = run(False), run(True)
    assert off[1] == bytes(8 * 1), "no variants without a provider"
    assert off[0] == on[0], "at the real rate nothing mutates in 30 ticks, so enabling it changes no position"
