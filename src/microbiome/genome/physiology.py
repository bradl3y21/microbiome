"""Genes -> physiology traits for the realism layer, through the same D2 machinery as the phenotype table.

``data/physiology_schema.tsv`` and ``data/physiology_map.tsv`` are the versioned, hashable rendering of the rules in
``RULES`` below (regenerate with ``python -m microbiome.genome.physiology export``). Each rule names genes by symbol;
the exporter resolves them against the genome's real annotation, scoped to one assembly, and refuses a missing or
ambiguous symbol (``bacA`` is bacilysin synthesis in B. subtilis but undecaprenyl pyrophosphatase in E. coli, which is
exactly why rules are genome-scoped locus tags and never bare names). Rows that share a ``mapping_id`` are an all-of
operon rule: it fires once, and only if every gene is present, complete and not a pseudogene.

Every effect is a calibrated simplification of a qualitative mechanism; the table says so in each row.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..sim.contracts import PhenotypeTable
from ..sim.realism import ChannelMap, Physiology
from .organisms import MX, PA, PROFILES, SA
from .calibration import Calibration, load_calibration
from .models import AnnotatedGene, GenomePhenotype, GenomeRecord
from .parser import _PROJECT_ROOT, derive_genome_id
from .tables import (
    MAP_COLUMNS,
    SCHEMA_COLUMNS,
    GeneTraitMap,
    TraitSchema,
    TraitSet,
    _write_tsv,
    derive_trait_values,
    load_gene_trait_map,
    load_trait_schema,
)

CHANNEL_NAMES = ("glucose", "lactose", "oxygen", "acetate", "bacilysin", "ai2", "pyocyanin")
MOLAR_MASS_G_MOL = (180.156, 342.296, 31.998, 60.052, 270.28, 134.13, 210.23)   # the seven channels, in order
DIFFUSIVITY_UM2_S = (600.0, 450.0, 1400.0, 1100.0, 400.0, 500.0, 400.0)         # see data/calibration_targets.json; pyocyanin's is an assumption
REALISM_CHANNELS = ChannelMap(toxin2=6)                                          # which channel plays which chemical role
EC, BS = "GCF_000005845.2", "GCF_000009045.1"
SCALAR_TRAITS = (
    "aerobic_capacity", "fermentation_capacity", "acetate_overflow", "acetate_uptake_capacity", "ccr_strength",
    "stress_resistance", "sporulation_capacity", "competence_capacity", "biofilm_capacity", "qs_capacity", "ph_min",
    "toxin_production", "toxin_susceptibility", "toxin2_production", "toxin2_susceptibility", "predation_capacity", "gliding",
    "starvation_signalling",
)
PHYSIOLOGY_TRAITS = TraitSet(nutrient=("half_saturation_mmol_l",), scalar=SCALAR_TRAITS)

# Extra D1 phenotype columns for the realism channels (oxygen, acetate, toxin, signal): calibrations, see docs/REALISM.md
O2_VMAX_MMOL_S = 3.0e-15
ACETATE_VMAX_MMOL_S = 6.0e-16
ACETATE_STORAGE_FRACTION = 0.1
ACETATE_BIOMASS_FRACTION = 0.5
ACETATE_ENERGY_YIELD_PJ_PER_MMOL = 3.3e14   # one third of glucose's 1e15: 2 carbons against 6

KS_BASELINES = (0.01, 0.2, 0.002, 0.3, 1.0, 1.0, 1.0)   # mmol/L per channel; the last three are not taken up
KS_NOTE = ("Half-saturation constants are order-of-magnitude values typical of bacterial transporters "
           "(glucose PTS ~10 uM, lactose permease ~0.2 mM, terminal oxidase ~2 uM, acetate ~0.3 mM), not measured for these strains.")

TRAIT_META = {   # trait -> (baseline, min, max, unit, baseline evidence, approximation note)
    "aerobic_capacity": (0.0, 0.0, 1.0, "fraction", "none: zero unless terminal-oxidase genes are annotated",
                         "Share of full oxygen use the genome supports; scales the oxygen uptake capacity."),
    "fermentation_capacity": (0.0, 0.0, 1.0, "fraction", "none: zero unless fermentation genes are annotated",
                              "Share of growth that can continue without oxygen by fermenting."),
    "acetate_overflow": (0.0, 0.0, 0.6, "fraction", "none: zero unless acetate-pathway genes are annotated",
                         "Fraction of respired carbon excreted as acetate at saturating sugar (overflow metabolism)."),
    "acetate_uptake_capacity": (0.0, 0.0, 1.0, "fraction", "none: zero unless acetate-activation genes are annotated",
                                "Share of the acetate uptake capacity the genome supports."),
    "ccr_strength": (0.0, 0.0, 1.0, "fraction", "none: zero unless catabolite-repression regulators are annotated",
                     "How completely glucose represses alternative sugars (diauxie)."),
    "stress_resistance": (0.0, 0.0, 1.0, "fraction", "none: zero unless a general stress sigma factor is annotated",
                          "Reduces starvation-induced death (stationary-phase resistance)."),
    "sporulation_capacity": (0.0, 0.0, 1.0, "fraction", "none: zero unless the sporulation cascade is annotated",
                             "Whether starving cells can form dormant spores."),
    "competence_capacity": (0.0, 0.0, 1.0, "fraction", "none: zero unless the competence machinery is annotated",
                            "Whether cells can take up DNA released by dead cells (natural transformation)."),
    "biofilm_capacity": (0.0, 0.0, 1.0, "fraction", "none: zero unless matrix-synthesis genes are annotated",
                         "How much extracellular matrix cells can build once quorum is reached."),
    "qs_capacity": (0.0, 0.0, 1.0, "fraction", "none: zero unless luxS is annotated",
                    "Production of, and response to, the AI-2-like signal."),
    "ph_min": (5.8, 3.0, 6.5, "pH", "typical neutrophile minimum growth pH; the genes below only lower it",
               "The pH below which growth stops; acid-resistance systems lower it."),
    "toxin_production": (0.0, 0.0, 1.0, "fraction", "none: zero unless a toxin cluster is annotated",
                         "Fraction of the maximum toxin secretion rate."),
    "toxin_susceptibility": (1.0, 0.0, 1.0, "fraction", "default: every cell is susceptible unless immunity is annotated",
                             "Scales the killing hazard; producers are assumed immune (the immunity gene is not curated)."),
    "toxin2_production": (0.0, 0.0, 1.0, "fraction", "none: zero unless a phenazine cluster is annotated",
                          "Fraction of the maximum secretion rate of the second toxin (pyocyanin)."),
    "toxin2_susceptibility": (1.0, 0.0, 1.0, "fraction", "default: every cell is susceptible unless immunity is annotated",
                              "Scales the second toxin's killing hazard; producers are assumed immune."),
    "predation_capacity": (0.0, 0.0, 1.0, "fraction", "none: zero unless predatory machinery is annotated",
                           "Rate at which a cell in contact with prey of another genome kills it and takes up part of its biomass."),
    "gliding": (0.0, 0.0, 1.0, "fraction", "none: zero unless gliding regulators are annotated",
                "1 makes the cell a surface glider that reverses instead of swimming and tumbling."),
    "starvation_signalling": (0.0, 0.0, 1.0, "fraction", "none: zero unless a starvation-triggered developmental programme is annotated",
                              "1 makes signal production rise with starvation (a developmental signal), as in myxobacterial development."),
}

# ---- evidence: (class, references, urls). Every DOI below was resolved against Crossref when this table was written.
GENOME_PAGE = {acc: f"https://www.ncbi.nlm.nih.gov/datasets/genome/{acc}/" for acc in (EC, BS, PA, SA, MX)}
GENE_DB = {EC: "https://ecocyc.org/", BS: "https://subtiwiki.uni-goettingen.de/", PA: "https://www.pseudomonas.com/",
           SA: "https://aureowiki.med.uni-greifswald.de/", MX: "https://www.ncbi.nlm.nih.gov/gene/"}
LITERATURE = {
    "acetate": ("Wolfe 2005, The acetate switch, Microbiol Mol Biol Rev 69:12-50", "https://doi.org/10.1128/MMBR.69.1.12-50.2005"),
    "ccr": ("Gorke & Stulke 2008, Carbon catabolite repression in bacteria, Nat Rev Microbiol 6:613-624", "https://doi.org/10.1038/nrmicro1932"),
    "rpoS": ("Hengge-Aronis 2002, Signal transduction and regulatory mechanisms involved in control of the sigma-S subunit, Microbiol Mol Biol Rev 66:373-395", "https://doi.org/10.1128/MMBR.66.3.373-395.2002"),
    "sigB": ("Hecker, Pane-Farre & Volker 2007, SigB-dependent general stress response in Bacillus subtilis, Annu Rev Microbiol 61:215-236", "https://doi.org/10.1146/annurev.micro.61.080706.093445"),
    "spore": ("Higgins & Dworkin 2012, Recent progress in Bacillus subtilis sporulation, FEMS Microbiol Rev 36:131-148", "https://doi.org/10.1111/j.1574-6976.2011.00310.x"),
    "spore-resistance": ("Setlow 2007, I will survive: DNA protection in bacterial spores, Trends Microbiol 15:172-180", "https://doi.org/10.1016/j.tim.2007.02.004"),
    "competence": ("Dubnau 1991, Genetic competence in Bacillus subtilis, Microbiol Rev 55:395-424", "https://doi.org/10.1128/mr.55.3.395-424.1991"),
    "bsub-biofilm": ("Branda et al. 2001, Fruiting body formation by Bacillus subtilis, PNAS 98:11621-11626", "https://doi.org/10.1073/pnas.191384198"),
    "ecoli-biofilm": ("Danese, Pratt & Kolter 2000, Exopolysaccharide production is required for development of E. coli K-12 biofilm architecture, J Bacteriol 182:3593-3596", "https://doi.org/10.1128/JB.182.12.3593-3596.2000"),
    "luxS": ("Surette, Miller & Bassler 1999, Quorum sensing in E. coli, Salmonella typhimurium and Vibrio harveyi, PNAS 96:1639-1644", "https://doi.org/10.1073/pnas.96.4.1639"),
    "acid": ("Foster 2004, Escherichia coli acid resistance: tales of an amateur acidophile, Nat Rev Microbiol 2:898-907", "https://doi.org/10.1038/nrmicro1021"),
    "bacilysin": ("Kenig & Abraham 1976, Antimicrobial activities and antagonists of bacilysin and anticapsin, J Gen Microbiol 94:37-45", "https://doi.org/10.1099/00221287-94-1-37"),
    "pa-pel": ("Friedman & Kolter 2003, Genes involved in matrix formation in Pseudomonas aeruginosa PA14 biofilms, Mol Microbiol", "https://doi.org/10.1046/j.1365-2958.2003.03877.x"),
    "pa-psl": ("Ma et al. 2006, Analysis of Pseudomonas aeruginosa conditional Psl variants reveals roles for the Psl polysaccharide in adhesion and maintaining biofilm structure, J Bacteriol", "https://doi.org/10.1128/jb.01202-06"),
    "pa-las": ("Gambello & Iglewski 1991, Cloning and characterization of the Pseudomonas aeruginosa lasR gene, a transcriptional activator of elastase expression, J Bacteriol 173:3000-3009", "https://doi.org/10.1128/jb.173.9.3000-3009.1991"),
    "pa-phz": ("Mavrodi et al. 2001, Functional analysis of genes for biosynthesis of pyocyanin and phenazine-1-carboxamide from Pseudomonas aeruginosa PAO1, J Bacteriol 183:6454-6465", "https://doi.org/10.1128/jb.183.21.6454-6465.2001"),
    "pa-hfq": ("Sonnleitner et al. 2014, Regulation of Hfq by the RNA CrcZ in Pseudomonas aeruginosa carbon catabolite repression, PLoS Genet", "https://doi.org/10.1371/journal.pgen.1004440"),
    "pa-rpoS": ("Schuster et al. 2003, The Pseudomonas aeruginosa RpoS regulon and its relationship to quorum sensing, Mol Microbiol", "https://doi.org/10.1046/j.1365-2958.2003.03886.x"),
    "sa-agr": ("Novick et al. 1993, Synthesis of staphylococcal virulence factors is controlled by a regulatory RNA molecule, EMBO J", "https://doi.org/10.1002/j.1460-2075.1993.tb06074.x"),
    "sa-ica": ("Cramton et al. 1999, The intercellular adhesion (ica) locus is present in Staphylococcus aureus and is required for biofilm formation, Infect Immun 67:5427-5433", "https://doi.org/10.1128/iai.67.10.5427-5433.1999"),
    "mx-gliding": ("Hartzell & Kaiser 1991, Function of MglA, a 22-kilodalton protein essential for gliding in Myxococcus xanthus, J Bacteriol 173:7615-7624", "https://doi.org/10.1128/jb.173.23.7615-7624.1991"),
    "mx-eps": ("Lu et al. 2005, Exopolysaccharide biosynthesis genes required for social motility in Myxococcus xanthus, Mol Microbiol", "https://doi.org/10.1111/j.1365-2958.2004.04369.x"),
    "mx-relA": ("Harris, Kaiser & Singer 1998, The guanosine nucleotide (p)ppGpp initiates development and A-factor production in Myxococcus xanthus, Genes Dev 12:1022-1035", "https://doi.org/10.1101/gad.12.7.1022"),
    "mx-fruA": ("Ellehauge, Norregaard-Madsen & Sogaard-Andersen 1998, The FruA signal transduction protein provides a checkpoint for the temporal co-ordination of intercellular signals in Myxococcus xanthus development, Mol Microbiol", "https://doi.org/10.1046/j.1365-2958.1998.01113.x"),
    "mx-predation": ("Zhang et al. 2020, Dynamics of solitary predation by Myxococcus xanthus on Escherichia coli observed at the single-cell level, Appl Environ Microbiol 86:e02286-19", "https://doi.org/10.1128/AEM.02286-19"),
}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    accession: str
    symbols: tuple[str, ...]
    trait: str
    effect: float
    literature: tuple[str, ...]          # keys of LITERATURE; may be empty (annotation + gene database only)
    rationale: str
    loci: Mapping[str, str] | None = None   # symbol -> locus tag, where a symbol is not unique in the annotation


def _r(rule_id, accession, symbols, trait, effect, literature, rationale, loci=None) -> Rule:
    return Rule(rule_id, accession, tuple(symbols.split()), trait, float(effect), tuple(literature), rationale, loci)


RULES: tuple[Rule, ...] = (
    # ------------------------------------------------------------------ E. coli K-12 MG1655
    _r("ecoli-cyo-oxidase", EC, "cyoA cyoB cyoC cyoD", "aerobic_capacity", 0.7, (), "cytochrome bo3 ubiquinol oxidase, the main aerobic terminal oxidase (all four subunits required)"),
    _r("ecoli-cyd-oxidase", EC, "cydA cydB", "aerobic_capacity", 0.3, (), "cytochrome bd oxidase, the high-affinity oxidase used at low oxygen"),
    _r("ecoli-pfl-fermentation", EC, "pflA pflB", "fermentation_capacity", 0.4, (), "pyruvate formate-lyase and its activating enzyme: the entry to mixed-acid fermentation"),
    _r("ecoli-ldhA-fermentation", EC, "ldhA", "fermentation_capacity", 0.2, (), "fermentative lactate dehydrogenase"),
    _r("ecoli-adhE-fermentation", EC, "adhE", "fermentation_capacity", 0.2, (), "alcohol/aldehyde dehydrogenase: ethanol fermentation"),
    _r("ecoli-frd-fermentation", EC, "frdA frdB frdC frdD", "fermentation_capacity", 0.2, (), "fumarate reductase: anaerobic respiration on fumarate"),
    _r("ecoli-pta-ackA-overflow", EC, "pta ackA", "acetate_overflow", 0.4, ("acetate",), "phosphotransacetylase + acetate kinase: the Pta-AckA acetate-excretion pathway"),
    _r("ecoli-acs-acetate-uptake", EC, "acs", "acetate_uptake_capacity", 0.7, ("acetate",), "acetyl-CoA synthetase: high-affinity acetate scavenging"),
    _r("ecoli-actP-acetate-uptake", EC, "actP", "acetate_uptake_capacity", 0.3, ("acetate",), "acetate permease"),
    _r("ecoli-crp-ccr", EC, "crp", "ccr_strength", 0.6, ("ccr",), "cAMP receptor protein: the global activator relieved by glucose"),
    _r("ecoli-cya-ccr", EC, "cyaA", "ccr_strength", 0.2, ("ccr",), "adenylate cyclase: makes the cAMP signal"),
    _r("ecoli-pts-ccr", EC, "ptsH ptsI", "ccr_strength", 0.2, ("ccr",), "PTS phosphocarrier proteins: sense glucose flux and exclude inducers"),
    _r("ecoli-rpoS-stress", EC, "rpoS", "stress_resistance", 1.0, ("rpoS",), "RpoS, the stationary-phase/general-stress sigma factor"),
    _r("ecoli-curli-biofilm", EC, "csgA csgB", "biofilm_capacity", 0.4, ("ecoli-biofilm",), "curli fibre subunits (CsgA, CsgB): the amyloid matrix component"),
    _r("ecoli-cellulose-biofilm", EC, "bcsA bcsB", "biofilm_capacity", 0.3, ("ecoli-biofilm",), "cellulose synthase catalytic and regulatory subunits"),
    _r("ecoli-pga-biofilm", EC, "pgaA pgaB", "biofilm_capacity", 0.3, ("ecoli-biofilm",), "poly-beta-1,6-GlcNAc export and deacetylase: the PGA matrix polysaccharide"),
    _r("ecoli-luxS-signal", EC, "luxS", "qs_capacity", 1.0, ("luxS",), "S-ribosylhomocysteine lyase: synthesises the AI-2 signal precursor"),
    _r("ecoli-gad-acid", EC, "gadA gadB gadC", "ph_min", -0.7, ("acid",), "glutamate decarboxylase system (GadA, GadB, antiporter GadC): the strongest acid-resistance system"),
    _r("ecoli-adi-acid", EC, "adiA adiC", "ph_min", -0.4, ("acid",), "arginine decarboxylase system (AdiA, antiporter AdiC)"),
    _r("ecoli-cad-acid", EC, "cadA cadB", "ph_min", -0.3, ("acid",), "lysine decarboxylase system (CadA, antiporter CadB)"),
    # ------------------------------------------------------------------ B. subtilis 168
    _r("bsub-qox-oxidase", BS, "qoxA qoxB qoxC qoxD", "aerobic_capacity", 0.4, (), "quinol oxidase aa3 (QoxABCD)"),
    _r("bsub-cta-oxidase", BS, "ctaC ctaD ctaE ctaF", "aerobic_capacity", 0.4, (), "cytochrome caa3 oxidase (CtaCDEF), the main terminal oxidase"),
    _r("bsub-cyd-oxidase", BS, "cydA cydB", "aerobic_capacity", 0.2, (), "cytochrome bd oxidase"),
    _r("bsub-ldh-fermentation", BS, "lctE", "fermentation_capacity", 0.2, (), "L-lactate dehydrogenase (Ldh): fermentative lactate production"),
    _r("bsub-als-fermentation", BS, "alsS alsD", "fermentation_capacity", 0.15, (), "acetolactate synthase and decarboxylase: acetoin fermentation"),
    _r("bsub-pta-ackA-overflow", BS, "pta ackA", "acetate_overflow", 0.4, ("acetate",), "phosphotransacetylase + acetate kinase: acetate excretion"),
    _r("bsub-acsA-acetate-uptake", BS, "acsA", "acetate_uptake_capacity", 1.0, ("acetate",), "acetyl-CoA synthetase AcsA: acetate assimilation"),
    _r("bsub-ccpA-ccr", BS, "ccpA", "ccr_strength", 0.7, ("ccr",), "CcpA, the master carbon-catabolite-repression regulator of Firmicutes"),
    _r("bsub-pts-ccr", BS, "ptsH ptsI", "ccr_strength", 0.3, ("ccr",), "HPr and enzyme I: HPr-Ser46-P is CcpA's co-repressor"),
    _r("bsub-sigB-stress", BS, "sigB", "stress_resistance", 0.6, ("sigB",), "SigB, the general stress sigma factor (weaker than RpoS for starvation survival)"),
    _r("bsub-sporulation", BS, "spo0A sigF sigE sigG spoIIE", "sporulation_capacity", 1.0, ("spore",), "Spo0A master regulator and the sigma-F, sigma-E, sigma-G cascade with SpoIIE: the sporulation programme (all required)"),
    _r("bsub-competence", BS, "comK comEA comEC", "competence_capacity", 1.0, ("competence",), "ComK master regulator with the DNA-binding (ComEA) and channel (ComEC) proteins of the uptake machinery (all required)"),
    _r("bsub-eps-biofilm", BS, "epsA epsB epsC epsD epsE epsF epsG epsH epsI epsJ epsK epsL epsM epsN epsO", "biofilm_capacity", 0.5, ("bsub-biofilm",),
       "the 15-gene eps operon for the exopolysaccharide matrix (all required)", {"epsJ": "BSU_34280", "epsK": "BSU_34265"}),
    _r("bsub-tasA-biofilm", BS, "tasA", "biofilm_capacity", 0.3, ("bsub-biofilm",), "TasA amyloid-like matrix fibre"),
    _r("bsub-bslA-biofilm", BS, "bslA", "biofilm_capacity", 0.2, ("bsub-biofilm",), "BslA hydrophobin-like biofilm surface layer"),
    _r("bsub-luxS-signal", BS, "luxS", "qs_capacity", 1.0, ("luxS",), "S-ribosylhomocysteine lyase: AI-2 precursor synthesis"),
    _r("bsub-bacilysin-production", BS, "bacA bacB bacC bacD bacE bacF bacG", "toxin_production", 1.0, ("bacilysin",), "the bacABCDEFG bacilysin biosynthesis operon (all required)"),
    _r("bsub-bacilysin-immunity", BS, "bacA bacB bacC bacD bacE bacF bacG", "toxin_susceptibility", -0.9, ("bacilysin",),
       "approximation: a strain that makes bacilysin is assumed to survive it; the specific self-resistance gene is not curated here"),
    # ------------------------------------------------------------------ P. aeruginosa PAO1
    _r("pseudomonas-cco1-oxidase", PA, "ccoN1 ccoO1 ccoP1", "aerobic_capacity", 0.5, (), "cbb3-type cytochrome c oxidase cco1 (CcoN1, CcoO1, CcoP1): the main high-affinity terminal oxidase (all required)"),
    _r("pseudomonas-cyo-oxidase", PA, "cyoA cyoB cyoC cyoD", "aerobic_capacity", 0.3, (), "cytochrome bo3 ubiquinol oxidase (all four subunits required)"),
    _r("pseudomonas-cio-oxidase", PA, "cioA cioB", "aerobic_capacity", 0.2, (), "cyanide-insensitive bd-type oxidase"),
    _r("pseudomonas-arc-fermentation", PA, "arcD arcA arcB arcC", "fermentation_capacity", 0.1, (), "arginine deiminase pathway: the only substrate-level ATP source without oxygen or nitrate (a weak one)"),
    _r("pseudomonas-pta-ackA-overflow", PA, "pta ackA", "acetate_overflow", 0.4, ("acetate",), "phosphotransacetylase + acetate kinase"),
    _r("pseudomonas-acsA-acetate-uptake", PA, "acsA", "acetate_uptake_capacity", 1.0, ("acetate",), "acetyl-CoA synthetase: acetate assimilation"),
    _r("pseudomonas-crc-ccr", PA, "crc", "ccr_strength", 0.6, ("ccr",), "Crc, the catabolite repression control protein of Pseudomonas"),
    _r("pseudomonas-hfq-ccr", PA, "hfq", "ccr_strength", 0.2, ("pa-hfq",), "Hfq, the RNA chaperone that partners Crc in catabolite repression"),
    _r("pseudomonas-rpoS-stress", PA, "rpoS", "stress_resistance", 1.0, ("pa-rpoS",), "RpoS, the stationary-phase sigma factor"),
    _r("pseudomonas-pel-biofilm", PA, "pelA pelB pelC pelD pelE pelF pelG", "biofilm_capacity", 0.5, ("pa-pel",), "the pelABCDEFG operon for the Pel matrix polysaccharide (all required)"),
    _r("pseudomonas-psl-biofilm", PA, "pslA pslB pslC pslD pslE pslF pslG pslH pslI pslJ pslK pslL", "biofilm_capacity", 0.5, ("pa-psl",), "the pslA-L operon for the Psl matrix polysaccharide (all required)"),
    _r("pseudomonas-las-signal", PA, "lasI", "qs_capacity", 1.0, ("pa-las",), "LasI, the acyl-homoserine lactone synthase of the las quorum-sensing system (the model's single signal is generic, not AI-2)"),
    _r("pseudomonas-phenazine-production", PA, "phzA1 phzB1 phzC1 phzD1 phzE1 phzF1 phzG1 phzM phzS", "toxin2_production", 1.0, ("pa-phz",),
       "the phzA1-G1 operon for phenazine-1-carboxylic acid plus phzM and phzS, which convert it to pyocyanin (all required)"),
    _r("pseudomonas-phenazine-immunity", PA, "phzA1 phzB1 phzC1 phzD1 phzE1 phzF1 phzG1 phzM phzS", "toxin2_susceptibility", -0.9, ("pa-phz",),
       "approximation: a strain that makes pyocyanin is assumed to survive it; the specific self-resistance genes are not curated here"),
    # ------------------------------------------------------------------ S. aureus Newman
    _r("staphylococcus-qox-oxidase", SA, "qoxA qoxB qoxC qoxD", "aerobic_capacity", 0.6, (), "quinol oxidase aa3 (QoxABCD), the main terminal oxidase (all required)"),
    _r("staphylococcus-cyd-oxidase", SA, "cydA cydB", "aerobic_capacity", 0.4, (), "cytochrome bd oxidase (annotated by product as cytochrome d ubiquinol oxidase subunits I and II)",
       {"cydA": "NWMN_RS05335", "cydB": "NWMN_RS05340"}),
    _r("staphylococcus-ldh-fermentation", SA, "ldh", "fermentation_capacity", 0.3, (), "L-lactate dehydrogenase: the main fermentative route", {"ldh": "NWMN_RS00980"}),
    _r("staphylococcus-pfl-fermentation", SA, "pflA", "fermentation_capacity", 0.2, (), "pyruvate formate-lyase activating enzyme (its partner pflB is not annotated under a name this table can resolve)"),
    _r("staphylococcus-adhE-fermentation", SA, "adhE", "fermentation_capacity", 0.2, (), "bifunctional acetaldehyde-CoA/alcohol dehydrogenase: ethanol fermentation"),
    _r("staphylococcus-pta-ackA-overflow", SA, "pta ackA", "acetate_overflow", 0.4, ("acetate",), "phosphate acetyltransferase + acetate kinase: aerobic S. aureus excretes acetate", {"ackA": "NWMN_RS09000"}),
    _r("staphylococcus-ccpA-ccr", SA, "ccpA", "ccr_strength", 0.8, ("ccr",), "CcpA, the master catabolite-repression regulator of Firmicutes"),
    _r("staphylococcus-hpr-ccr", SA, "ptsH", "ccr_strength", 0.2, ("ccr",), "HPr, whose Ser46-phosphorylated form is CcpA's co-repressor", {"ptsH": "NWMN_RS05320"}),
    _r("staphylococcus-sigB-stress", SA, "sigB", "stress_resistance", 0.6, ("sigB",), "SigB, the general stress sigma factor"),
    _r("staphylococcus-ica-biofilm", SA, "icaA icaD icaC", "biofilm_capacity", 1.0, ("sa-ica",), "icaADC: synthesis and export of the polysaccharide intercellular adhesin (icaB is not matched by name; all three required)"),
    _r("staphylococcus-agr-signal", SA, "agrA agrB agrC agrD", "qs_capacity", 1.0, ("sa-agr",), "the agr quorum-sensing locus: autoinducing peptide AgrD, processing AgrB, sensor AgrC, regulator AgrA (all required)"),
    # ------------------------------------------------------------------ M. xanthus DK 1622
    _r("myxococcus-aa3-oxidase", MX, "ctaD cox3 coxB", "aerobic_capacity", 0.5, (), "cytochrome c oxidase aa3 (subunits I, III, II; all required)",
       {"ctaD": "MXAN_RS18790", "cox3": "MXAN_RS18785", "coxB": "MXAN_RS18795"}),
    _r("myxococcus-aa3b-oxidase", MX, "ctaD2 cox3b coxB2", "aerobic_capacity", 0.3, (), "a second cytochrome c oxidase aa3 operon (all required)",
       {"ctaD2": "MXAN_RS29530", "coxB2": "MXAN_RS29525", "cox3b": "MXAN_RS29535"}),
    _r("myxococcus-cbb3-oxidase", MX, "ccoN cco3", "aerobic_capacity", 0.2, (), "cbb3-type cytochrome oxidase (subunits I and 3; all required)",
       {"ccoN": "MXAN_RS26850", "cco3": "MXAN_RS26845"}),
    _r("myxococcus-gliding", MX, "mglA mglB", "gliding", 1.0, ("mx-gliding",), "MglA and MglB, the polarity regulators of gliding motility (both required)"),
    _r("myxococcus-predation-capacity", MX, "mglA mglB pilA pilB pilQ", "predation_capacity", 1.0, ("mx-predation",),
       "approximation: predation needs the gliding and type IV pilus machinery to reach prey (both motility systems are annotated). The contact-dependent killing genes were not found by name in this annotation, so capacity is inferred from motility, not from a killing locus"),
    _r("myxococcus-development-sporulation", MX, "fruA mrpC relA", "sporulation_capacity", 1.0, ("mx-fruA",),
       "FruA and MrpC, the developmental regulators of fruiting-body formation and sporulation, with the stringent-response enzyme RelA that starts development (all required)",
       {"fruA": "MXAN_RS15110", "relA": "MXAN_RS15515"}),
    _r("myxococcus-stringent-signalling", MX, "relA", "starvation_signalling", 1.0, ("mx-relA",), "RelA/SpoT family protein: the stringent response that starvation uses to trigger development", {"relA": "MXAN_RS15515"}),
    _r("myxococcus-development-signal", MX, "fruA mrpC", "qs_capacity", 1.0, ("mx-fruA",),
       "approximation: C-signalling (CsgA) was not found by name; the response regulators FruA and MrpC stand in for the contact signalling that organises development", {"fruA": "MXAN_RS15110"}),
    _r("myxococcus-eps-matrix", MX, "epsZ wzx epsY epsX epsV epsU", "biofilm_capacity", 0.6, ("mx-eps",), "the eps locus for the exopolysaccharide matrix that holds cells together (all required)", {"wzx": "MXAN_RS35875"}),
)


# ------------------------------------------------------------------------------------------ exporter
def _resolve(genome: GenomeRecord, symbol: str, loci: Mapping[str, str] | None) -> AnnotatedGene:
    complete = [g for g in genome.genes if "CDS" in g.source_feature_types and not g.pseudogene]
    if loci and symbol in loci:
        candidates = [g for g in complete if g.locus_tag == loci[symbol]]       # named by locus tag: the label need not be its symbol
    else:
        candidates = [g for g in complete if g.gene_symbol == symbol]
    if len(candidates) != 1:
        raise ValueError(f"{genome.accession}: symbol {symbol!r} matches {len(candidates)} complete CDS genes; "
                         f"give its locus tag in the rule instead")
    return candidates[0]


def _schema_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(CHANNEL_NAMES):
        rows.append({"trait_id": "half_saturation_mmol_l", "scope": "nutrient", "nutrient_id": index, "dtype": "float64",
                     "unit": "mmol/L", "baseline": repr(KS_BASELINES[index]), "min": repr(1e-6), "max": repr(10.0),
                     "aggregation": "sum_clip", "baseline_evidence": f"order of magnitude for {name}: see approximation note",
                     "approximation_note": KS_NOTE})
    for trait in SCALAR_TRAITS:
        baseline, low, high, unit, evidence, note = TRAIT_META[trait]
        rows.append({"trait_id": trait, "scope": "scalar", "nutrient_id": "", "dtype": "float64", "unit": unit,
                     "baseline": repr(float(baseline)), "min": repr(float(low)), "max": repr(float(high)),
                     "aggregation": "sum_clip", "baseline_evidence": evidence, "approximation_note": note})
    return rows


def _map_rows(genomes: Mapping[str, GenomeRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rule in sorted(RULES, key=lambda r: r.rule_id):
        genome = genomes[rule.accession]
        refs = [f"NCBI RefSeq annotation of {', '.join(rule.symbols)} in {rule.accession}", f"gene summaries: {GENE_DB[rule.accession]}"]
        urls = [GENOME_PAGE[rule.accession], GENE_DB[rule.accession]]
        for key in rule.literature:
            refs.append(LITERATURE[key][0])
            urls.append(LITERATURE[key][1])
        unit = TRAIT_META[rule.trait][3]
        genes = [(symbol, _resolve(genome, symbol, rule.loci)) for symbol in rule.symbols]
        detail = ", ".join(f"{symbol}={gene.locus_tag}" for symbol, gene in genes)   # identical on every row of an all-of rule
        for symbol, gene in genes:
            rows.append({
                "mapping_id": rule.rule_id, "genome_id": derive_genome_id(rule.accession), "match_namespace": "locus_tag",
                "match_value": gene.locus_tag, "feature_type": "CDS", "trait_id": rule.trait, "nutrient_id": "",
                "effect": repr(rule.effect), "unit": unit, "priority": 100,
                "evidence_class": "annotation_plus_qualitative_literature", "evidence_ref": " | ".join(refs),
                "evidence_url": " | ".join(urls), "rationale": f"{rule.rationale} [{detail}]",
                "approximation_note": "calibrated_simplification: genome potential for a qualitative mechanism, not a measured rate",
            })
    return rows


def export_physiology_tables(project_root: str | Path | None = None) -> tuple[Path, Path]:
    from .bundle import load_genome_bundle

    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    genomes = {acc: load_genome_bundle(root / "data" / "genomes" / acc) for acc in (EC, BS, PA, SA, MX)}
    meta = {"schema_version": 1, "table_revision": 1, "source": "src/microbiome/genome/physiology.py"}
    schema_path, map_path = root / "data" / "physiology_schema.tsv", root / "data" / "physiology_map.tsv"
    _write_tsv(schema_path, meta, SCHEMA_COLUMNS, _schema_rows())
    _write_tsv(map_path, meta, MAP_COLUMNS, _map_rows(genomes))
    return schema_path, map_path


# ------------------------------------------------------------------------------------------ loading and use
@dataclass(frozen=True)
class PhysiologyTables:
    schema: TraitSchema
    gene_map: GeneTraitMap
    calibration: Calibration = Calibration()


def load_physiology_tables(project_root: str | Path | None = None) -> PhysiologyTables:
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    return PhysiologyTables(load_trait_schema(root / "data" / "physiology_schema.tsv", PHYSIOLOGY_TRAITS),
                            load_gene_trait_map(root / "data" / "physiology_map.tsv", PHYSIOLOGY_TRAITS),
                            load_calibration(root))


def derive_physiology(genome: GenomeRecord, tables: PhysiologyTables) -> Physiology:
    values, _ = derive_trait_values(genome, tables.schema, tables.gene_map)
    scalars = {name: float(values[(name, None)]) for name in SCALAR_TRAITS}
    half = tuple(float(values[("half_saturation_mmol_l", i)]) for i in range(len(CHANNEL_NAMES)))
    fields = {f.name for f in dataclasses.fields(Physiology)}
    cal = tables.calibration.for_accession(genome.accession)
    scalars["acetate_overflow"] = min(0.6, scalars["acetate_overflow"] * cal.overflow_scale)
    profile = PROFILES[genome.accession]
    return Physiology(**{k: v for k, v in scalars.items() if k in fields}, half_saturation_mmol_l=half,
                      maintenance_mmol_glc_gdw_h=cal.maintenance_mmol_glc_gdw_h, width_um=profile.width_um,
                      run_time_s=cal.run_time_s, turn_mean_deg=cal.turn_mean_deg, turn_sd_deg=cal.turn_sd_deg,
                      cluster=1.0 if profile.clusters else 0.0,
                      round_cell=1.0 if profile.shape == "coccus" else 0.0, imaging_style=float(profile.style))


def derive_acetate_uptake_capacity(genome: GenomeRecord, tables: PhysiologyTables) -> float:
    values, _ = derive_trait_values(genome, tables.schema, tables.gene_map)
    return float(values[("acetate_uptake_capacity", None)])


def extend_phenotype_table(base: PhenotypeTable, genomes: Sequence[GenomeRecord], tables: PhysiologyTables) -> PhenotypeTable:
    """The D2 two-nutrient table widened to the six realism channels (oxygen, acetate, toxin, signal appended)."""
    by_id = {int(g.genome_id): g for g in genomes}
    rows = [by_id[int(i)] for i in base.genome_id]
    aerobic = np.array([derive_physiology(g, tables).aerobic_capacity for g in rows])
    acetate = np.array([derive_acetate_uptake_capacity(g, tables) for g in rows])
    g = len(rows)

    def widen(old: np.ndarray, extra: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.concatenate([old, extra], axis=1))

    zeros = np.zeros(g)
    cals = [tables.calibration.for_accession(row.accession) for row in rows]
    factor = lambda name: np.array([getattr(c, name) for c in cals])
    scale = lambda matrix, name: np.ascontiguousarray(matrix * factor(name)[:, None])   # the two D2 sugar columns take the factors
    return dataclasses.replace(
        base,
        max_uptake_mmol_s=widen(scale(base.max_uptake_mmol_s, "uptake_scale"),
                                np.column_stack([aerobic * O2_VMAX_MMOL_S, acetate * ACETATE_VMAX_MMOL_S, zeros, zeros, zeros])),
        storage_fraction=widen(scale(base.storage_fraction, "storage_scale"),
                               np.column_stack([zeros, np.full(g, ACETATE_STORAGE_FRACTION), zeros, zeros, zeros])),
        biomass_fraction=widen(scale(base.biomass_fraction, "biomass_scale"),
                               np.column_stack([zeros, np.full(g, ACETATE_BIOMASS_FRACTION), zeros, zeros, zeros])),
        energy_yield_pj_per_mmol=widen(base.energy_yield_pj_per_mmol,
                                       np.column_stack([zeros, np.full(g, ACETATE_ENERGY_YIELD_PJ_PER_MMOL), zeros, zeros, zeros])),
        motility_um_s=np.ascontiguousarray(base.motility_um_s * factor("motility_scale")),
        division_mass_pg=np.ascontiguousarray(base.division_mass_pg * factor("division_mass_scale")),
    )


def widen_genome_phenotype(phenotype: GenomePhenotype, genome: GenomeRecord, tables: PhysiologyTables) -> GenomePhenotype:
    """The same widening for a single genome's D4-style record (so variants register in the six-channel table)."""
    aerobic = derive_physiology(genome, tables).aerobic_capacity
    acetate = derive_acetate_uptake_capacity(genome, tables)
    cal = tables.calibration.for_accession(genome.accession)
    times = lambda values, factor: tuple(float(v) * factor for v in values)
    return dataclasses.replace(
        phenotype,
        max_uptake_mmol_s=times(phenotype.max_uptake_mmol_s, cal.uptake_scale) + (aerobic * O2_VMAX_MMOL_S, acetate * ACETATE_VMAX_MMOL_S, 0.0, 0.0, 0.0),
        storage_fraction=times(phenotype.storage_fraction, cal.storage_scale) + (0.0, ACETATE_STORAGE_FRACTION, 0.0, 0.0, 0.0),
        biomass_fraction=times(phenotype.biomass_fraction, cal.biomass_scale) + (0.0, ACETATE_BIOMASS_FRACTION, 0.0, 0.0, 0.0),
        energy_yield_pj_per_mmol=tuple(phenotype.energy_yield_pj_per_mmol) + (0.0, ACETATE_ENERGY_YIELD_PJ_PER_MMOL, 0.0, 0.0, 0.0),
        motility_um_s=float(phenotype.motility_um_s) * cal.motility_scale,
        division_mass_pg=float(phenotype.division_mass_pg) * cal.division_mass_scale,
    )


class RealismMutationProvider:
    """Wraps the D4 provider so a variant's phenotype row has the same six channels as the realism table.

    Only the 13 D1 traits can change in a variant (D4); the realism columns copy the root genome's physiology.
    """

    def __init__(self, inner, genomes: Sequence[GenomeRecord], tables: PhysiologyTables) -> None:
        self._inner = inner
        self._genomes = {int(g.genome_id): g for g in genomes}
        self._tables = tables

    def create_offspring_variant(self, genome_id, parent_variant_id, **lineage):
        result = self._inner.create_offspring_variant(genome_id, parent_variant_id, **lineage)
        if result is None:
            return None
        variant, phenotype = result
        return variant, widen_genome_phenotype(phenotype, self._genomes[int(genome_id)], self._tables)

    def create_transformant(self, genome_id, *args, **kwargs):
        result = self._inner.create_transformant(genome_id, *args, **kwargs)
        if result is None:
            return None
        variant, phenotype = result
        return variant, widen_genome_phenotype(phenotype, self._genomes[int(genome_id)], self._tables)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m microbiome.genome.physiology")
    parser.add_argument("command", choices=["export"])
    parser.parse_args(argv)
    for path in export_physiology_tables():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
