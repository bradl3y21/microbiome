# data

Real sequenced genomes and their annotations live under `genomes/`.
Every acquired source package and extracted source artifact is named in
`manifest.json` with accession, source URL, fetch date, byte size, SHA-256,
and licence/terms. `gene_trait_table.json` is a derived, versioned model mapping;
it records exact annotation keys, numeric baselines/effects/bounds, approximation
status, rationale, and evidence URLs. Its biology and limitations must also be
documented in `docs/genome_trait_mapping.md` by the docs owner. Do not invent
sequence or annotation data.
