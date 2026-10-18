"""The organisms of the world: non-gene attributes (shape, imaging style) and the gene-to-trait rules for the added genomes.

Each organism is a real, complete RefSeq assembly in ``data/manifest.json`` (added with ``python -m microbiome.genome.fetch``).
What a genome says about a trait is written as rules, here for the three organisms added after the original pair
(*Pseudomonas aeruginosa* PAO1, *Staphylococcus aureus* Newman, *Myxococcus xanthus* DK 1622). Rules name genes by
locus tag, the evidence text is read from the real annotation when the table is generated, and nothing is inferred by
fuzzy matching. ``python -m microbiome.genome.organisms export`` rewrites the D2 rules inside ``data/gene_trait_table.json``
(the rest of that file is untouched), after which ``python -m microbiome.genome.tables export`` regenerates the TSVs.

A *profile* holds what is not in a genome sheet: the shape of the cell and how it looks under the microscope. These come
from standard microbiology (Gram reaction, coccus or rod) and are not claimed to be derived from genes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .parser import _PROJECT_ROOT, load_genome

EC, BS, PA, SA, MX = "GCF_000005845.2", "GCF_000009045.1", "GCF_000006765.1", "GCF_000010465.1", "GCF_000012685.1"
NEW_ORGANISMS = (PA, SA, MX)


@dataclass(frozen=True)
class Profile:
    accession: str
    short: str
    shape: str                      # "rod" or "coccus"
    width_um: float                 # rod width (a coccus takes its width from its volume)
    gram: str                       # "negative" or "positive": the Gram stain colour under the brightfield view
    style: int                      # index into the imaging colour tables (render/microscope.py)
    clusters: bool                  # daughters stay attached (staphylococci)
    note: str


PROFILES = {
    EC: Profile(EC, "E. coli", "rod", 0.8, "negative", 0, False, "rod"),
    BS: Profile(BS, "B. subtilis", "rod", 0.8, "positive", 1, False, "rod, 2-6 um long and just under 1 um wide (Microbe Profile, PMC7376258)"),
    PA: Profile(PA, "P. aeruginosa", "rod", 0.75, "negative", 2, False,
                "Gram-negative rod 1-5 um long and 0.5-1.0 um wide (Diggle and Whiteley 2020, doi 10.1099/mic.0.000860); width is the midpoint"),
    SA: Profile(SA, "S. aureus", "coccus", 1.0, "positive", 3, True,
                "Gram-positive coccus about 1 um across (0.52 um^3); daughters stay attached and form clusters (Monteiro 2015, Missiakas and Schneewind 2013)"),
    MX: Profile(MX, "M. xanthus", "rod", 0.5, "negative", 4, False,
                "Gram-negative rod 5-7 um long and 0.5 um wide (Kaiser and Welch 2004, doi 10.1128/JB.186.4.919-927.2004)"),
}


# ------------------------------------------------------------------------------------------ D2 rules
GENOME_PAGE = "https://www.ncbi.nlm.nih.gov/datasets/genome/{accession}/"


@dataclass(frozen=True)
class D2Rule:
    rule_id: str
    accession: str
    loci: tuple[str, ...]            # all required (an operon rule fires once, and only if every gene is present)
    trait: str
    value: float
    unit: str
    why: str
    nutrient: str | None = None


def _d2(rule_id, accession, loci, trait, value, unit, why, nutrient=None) -> D2Rule:
    return D2Rule(rule_id, accession, tuple(loci.split()), trait, float(value), unit, why, nutrient)


UPTAKE = 8.8809946714e-16        # the D2 calibration for one sole carbon channel (see uptake_division_calibration)
D2_RULES: tuple[D2Rule, ...] = (
    # ---------------------------------------------------------------------------------- Pseudomonas aeruginosa PAO1
    _d2("pseudomonas-chey-chemotaxis-output", PA, "PA1456", "chemotaxis_sensitivity", 0.25, "dimensionless", "CheY, the chemotaxis response regulator that controls the flagellar motor."),
    _d2("pseudomonas-chez-chemotaxis-phosphatase", PA, "PA1457", "chemotaxis_sensitivity", 0.05, "dimensionless", "CheZ, the annotated CheY phosphatase: terminates the signal."),
    _d2("pseudomonas-cheb-chemotaxis-adaptation", PA, "PA1459", "chemotaxis_sensitivity", 0.05, "dimensionless", "Annotated chemotaxis-specific methylesterase (CheB): adaptation capacity."),
    _d2("pseudomonas-chew-chemotaxis-coupling", PA, "PA1464", "chemotaxis_sensitivity", 0.10, "dimensionless", "Annotated purine-binding chemotaxis protein (CheW-type coupling protein) of the che cluster."),
    _d2("pseudomonas-mcp-chemotaxis-receptor", PA, "PA1930", "chemotaxis_sensitivity", 0.20, "dimensionless", "An annotated chemotaxis transducer (methyl-accepting receptor)."),
    _d2("pseudomonas-flic-flagellar-filament", PA, "PA1092", "motility_um_s", 8.0, "um/s", "FliC, the flagellin of the single polar flagellum."),
    _d2("pseudomonas-mota-flagellar-motor", PA, "PA4954", "motility_um_s", 1.0, "um/s", "MotA, a stator subunit of the flagellar motor."),
    _d2("pseudomonas-motb-flagellar-motor", PA, "PA4953", "motility_um_s", 1.0, "um/s", "MotB, a stator subunit of the flagellar motor."),
    _d2("pseudomonas-oprb-glucose-uptake", PA, "PA3186", "max_uptake_mmol_s", UPTAKE, "mmol/s",
        "OprB is the annotated glucose-selective porin (P. aeruginosa has no glucose PTS). Its rate is the same D2 single-channel calibration as for the other organisms; the organism-specific rate is set by data/calibration_fit.json.", "glucose"),
    _d2("pseudomonas-oprb-glucose-growth-proxy", PA, "PA3186", "division_rate_s", 0.00015, "s^-1", "Glucose import can support growth; the same deliberately indirect division-hazard proxy as the other organisms."),
    _d2("pseudomonas-ftsz-division-threshold", PA, "PA4407", "division_mass_pg", 0.1, "pg", "FtsZ is conserved division machinery that assembles at the division site; same calibration as the other organisms."),
    # ---------------------------------------------------------------------------------- Staphylococcus aureus Newman
    _d2("staphylococcus-ptsg-glucose-uptake-1", SA, "NWMN_RS00735", "max_uptake_mmol_s", UPTAKE / 2, "mmol/s", "Annotated glucose-specific PTS transporter IIBC (ptsG); the genome has two paralogs, each carries half of the single-channel calibration.", "glucose"),
    _d2("staphylococcus-ptsg-glucose-uptake-2", SA, "NWMN_RS14005", "max_uptake_mmol_s", UPTAKE / 2, "mmol/s", "The second annotated glucose-specific PTS transporter IIBC (ptsG paralog).", "glucose"),
    _d2("staphylococcus-ptsg-glucose-growth-proxy", SA, "NWMN_RS00735", "division_rate_s", 0.00015, "s^-1", "Glucose import can support growth; the same indirect division-hazard proxy as the other organisms."),
    _d2("staphylococcus-ftsz-division-threshold", SA, "NWMN_RS06210", "division_mass_pg", 0.1, "pg", "FtsZ is conserved division machinery; same calibration as the other organisms."),
    # ---------------------------------------------------------------------------------- Myxococcus xanthus DK 1622
    _d2("myxococcus-mgla-gliding-regulator", MX, "MXAN_RS09335", "motility_um_s", 0.02, "um/s", "MglA, the Ras-like GTPase that polarises gliding motility."),
    _d2("myxococcus-mglb-gliding-regulator", MX, "MXAN_RS09340", "motility_um_s", 0.01, "um/s", "MglB, the GTPase-activating protein that works with MglA."),
    _d2("myxococcus-gltj-gliding-machinery", MX, "MXAN_RS23610", "motility_um_s", 0.01, "um/s", "Annotated adventurous gliding motility protein GltJ."),
    _d2("myxococcus-ftsz-division-threshold", MX, "MXAN_RS27120", "division_mass_pg", 0.1, "pg", "FtsZ is conserved division machinery; same calibration as the other organisms."),
)


def d2_rule_json(rule: D2Rule, genome) -> dict:
    by_locus = {gene.locus_tag: gene for gene in genome.genes}
    keys, symbols, features = [], {}, []
    for locus in rule.loci:
        gene = by_locus[locus]
        if "CDS" not in gene.source_feature_types or gene.pseudogene:
            raise ValueError(f"{rule.rule_id}: {locus} is not a complete CDS")
        key = f"{rule.accession}|locus_tag:{locus}"
        keys.append(key)
        symbol = gene.gene_symbol or locus
        symbols[key] = symbol
        features.append({"source": "NCBI RefSeq annotation", "url": GENOME_PAGE.format(accession=rule.accession),
                         "feature": f"GFF3 CDS gene={symbol}, locus_tag={locus}; product: {gene.product}"})
    out = {
        "rule_id": rule.rule_id, "genome_accession": rule.accession, "gene_key": keys[-1] if len(keys) > 1 else keys[0],
        "gene_symbols": symbols,
    }
    if len(keys) > 1:
        out["requires_gene_keys"] = keys
    out.update({"source_feature_type": "CDS", "trait": rule.trait})
    if rule.nutrient:
        out["nutrient_name"] = rule.nutrient
    out.update({"operation": "add", "value": rule.value, "unit": rule.unit, "approximation_status": "calibrated_simplification",
                "justification": rule.why + " The coefficient encodes genome potential only and is not a measurement.", "evidence": features})
    return out


def generated_rules(project_root: str | Path | None = None) -> list[dict]:
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    genomes = {accession: load_genome(accession, root) for accession in NEW_ORGANISMS}
    return [d2_rule_json(rule, genomes[rule.accession]) for rule in D2_RULES]


def export_d2_rules(project_root: str | Path | None = None) -> Path:
    """Replace the generated organisms' rules inside data/gene_trait_table.json (the hand-written rules are kept as they are)."""
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    path = root / "data" / "gene_trait_table.json"
    table = json.loads(path.read_text(encoding="utf-8"))
    kept = [r for r in table["rules"] if r["genome_accession"] not in NEW_ORGANISMS]
    table["rules"] = kept + generated_rules(root)
    table["table_revision"] = max(int(table["table_revision"]), 3)
    accessions = sorted({r["genome_accession"] for r in table["rules"]})
    table["demo_accessions"] = accessions
    path.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m microbiome.genome.organisms")
    parser.add_argument("command", choices=["export"])
    parser.parse_args(argv)
    print(export_d2_rules())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
