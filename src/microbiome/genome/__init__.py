"""Load annotated RefSeq genomes and derive deterministic genome potentials.

The public entry points are:

* :func:`load_genome` — verify the manifest checksums and parse FASTA/GFF3.
* :func:`derive_phenotype` — apply the versioned, exact-key gene trait table.
* :func:`build_phenotype_table` — produce the simulation's genome-sorted,
  float64 ``PhenotypeTable`` for explicitly named nutrient channels.

Trait values represent documented genome-potential/calibration approximations,
not expression measurements or organism-specific laboratory kinetics.
"""

from .bundle import load_genome_bundle
from .models import AnnotatedGene, GenomePhenotype, GenomeRecord, NutrientChannel
from .parser import load_genome, parse_gff3, parse_fasta
from .phenotype import build_phenotype_table, derive_phenotype, phenotype_demo
from .variants import (
    DerivedVariant,
    NucleotideSubstitution,
    VariantMutationProvider,
    VariantStore,
    build_variant_phenotype_table,
    create_derived_variant,
    derive_variant_phenotype,
    variant_from_substitutions,
)
from .tables import (
    GeneTraitMap,
    TraitSchema,
    explain_phenotype,
    gene_feature_rows,
    load_gene_trait_map,
    load_trait_schema,
    normalize_gene_features,
    write_gene_features_tsv,
)

__all__ = [
    "AnnotatedGene",
    "DerivedVariant",
    "NucleotideSubstitution",
    "VariantMutationProvider",
    "VariantStore",
    "build_variant_phenotype_table",
    "create_derived_variant",
    "derive_variant_phenotype",
    "variant_from_substitutions",
    "GeneTraitMap",
    "GenomePhenotype",
    "GenomeRecord",
    "NutrientChannel",
    "TraitSchema",
    "build_phenotype_table",
    "derive_phenotype",
    "explain_phenotype",
    "gene_feature_rows",
    "load_gene_trait_map",
    "load_genome",
    "load_genome_bundle",
    "load_trait_schema",
    "normalize_gene_features",
    "parse_fasta",
    "parse_gff3",
    "phenotype_demo",
    "write_gene_features_tsv",
]
