"""D3 command and inspection facade for a deterministic simulation session.

The controller is the only public entry point needed by UI/render consumers.
Mutable D1 state, the phenotype catalogue, RNG streams, and the ordered external
command log remain private to this module; every value returned to a consumer is
a detached DTO declared by :mod:`microbiome.core.ui_api`.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from microbiome.core.ui_api import (
    CellInspection,
    CommandId,
    CommandResult,
    CommandTicket,
    GenomeId,
    GenomeInspection,
    GenomeOption,
    NutrientId,
    PhenotypePreview,
    RealismSnapshot,
    RenderSnapshot,
    SimulationController as SimulationControllerProtocol,
    TraitEvidence,
    Vec2Um,
)
from microbiome.genome.models import GenomePhenotype, GenomeRecord, NutrientChannel
from microbiome.genome.parser import _PROJECT_ROOT, load_genome
from microbiome.genome.phenotype import _table, build_phenotype_table, derive_phenotype

from .contracts import AgentArrays, Environment, NutrientField, RNGStreams, SimulationState
from .engine import DT_S, add_nutrient_source, initialize_rngs, step
from .model import AgentState, NutrientEnvironment
from .realism import RealismConfig, RealismSetup, RealisticSimulation, ph_from_acetate, rod_length_um

_UINT64_MAX = int(np.iinfo(np.uint64).max)
_UINT32_MAX = int(np.iinfo(np.uint32).max)
_NUTRIENT_TRAITS = frozenset(
    {
        "max_uptake_mmol_s",
        "storage_fraction",
        "biomass_fraction",
        "energy_yield_pj_per_mmol",
    }
)


@dataclass(frozen=True, slots=True)
class _QueuedCommand:
    command_id: int
    target_tick: int
    kind: str
    nutrient_id: int | None = None
    position_um: Vec2Um | None = None
    amount_mmol: float | None = None
    genome_id: int | None = None


class SimulationController:
    """Own a D1 session and implement the frozen D3 ``SimulationController``.

    ``state``, ``environment`` and ``rngs`` are the conventional mutable D1
    values, held privately after construction. Production callers may use
    :meth:`create_demo` for a checksum-verified catalogue of the shipped real
    genomes, or supply an existing D1 session and its manifest-loaded genomes.
    """

    def __init__(
        self,
        state: SimulationState,
        environment: Environment,
        rngs: RNGStreams,
        genomes: Sequence[GenomeRecord],
        nutrient_channels: Sequence[NutrientChannel],
        *,
        project_root: str | Path | None = None,
        provenance: Mapping[str, Mapping[str, str]] | None = None,
        mutation_provider=None,
        realism: RealismSetup | None = None,
    ) -> None:
        # D4: ``None`` keeps mutation off (the D1 no-provider behaviour); otherwise any object with
        # ``create_offspring_variant`` (see microbiome.genome.variants.VariantMutationProvider).
        self._mutation = mutation_provider
        self._state = state
        self._environment = environment
        self._rngs = rngs
        if self._state.agents.variant_id is None:
            self._state.agents.variant_id = np.zeros(self._state.agents.agent_id.size, dtype=np.uint64)
        self._project_root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
        self._channels = tuple(nutrient_channels)
        self._realism = realism
        # A realism world has more field channels than the two the D2 gene-to-trait table knows (oxygen, acetate, toxin,
        # signal are appended); the D2 derivations always run on the first two.
        self._d2_channels = self._channels[:2] if realism is not None else self._channels
        if not self._channels:
            raise ValueError("at least one nutrient channel is required")
        channel_ids = tuple(int(channel.nutrient_id) for channel in self._channels)
        if (
            any(not 0 <= value <= _UINT32_MAX for value in channel_ids)
            or channel_ids != tuple(sorted(channel_ids))
            or len(channel_ids) != len(set(channel_ids))
        ):
            raise ValueError("nutrient channels must have unique, increasing uint32 IDs")
        if not np.array_equal(environment.field.nutrient_id, np.asarray(channel_ids, dtype=np.uint32)):
            raise ValueError("environment nutrient IDs must match the ordered nutrient channels")
        if environment.field.concentration_mmol_l.shape != (len(channel_ids), environment.ny, environment.nx):
            raise ValueError("environment field shape does not match its D1 grid and nutrient channels")

        self._genomes = tuple(sorted(genomes, key=lambda record: (int(record.genome_id), record.accession)))
        if not self._genomes:
            raise ValueError("at least one manifest-listed genome is required")
        genome_ids = [int(genome.genome_id) for genome in self._genomes]
        accessions = [genome.accession for genome in self._genomes]
        if len(genome_ids) != len(set(genome_ids)) or len(accessions) != len(set(accessions)):
            raise ValueError("genome IDs and accession.version values must be unique")
        self._genome_by_id = {int(genome.genome_id): genome for genome in self._genomes}
        self._phenotype_by_id = {
            int(genome.genome_id): derive_phenotype(genome, self._d2_channels, self._project_root)
            for genome in self._genomes
        }
        self._phenotypes = build_phenotype_table(self._genomes, self._d2_channels, self._project_root)
        if realism is not None:
            self._phenotypes = realism.widen_table(self._phenotypes)
        if self._phenotypes.variant_id is None:
            self._phenotypes.variant_id = np.zeros(self._phenotypes.genome_id.size, dtype=np.uint64)
        self._provenance = {
            str(accession): {str(key): str(value) for key, value in fields.items()}
            for accession, fields in (provenance or {}).items()
        }
        self._manifest_entries = self._load_manifest_entries()
        for genome in self._genomes:
            entry = self._manifest_entries.get(genome.accession)
            if entry is None or int(entry.get("genome_id", genome.genome_id)) != int(genome.genome_id):
                # Older manifest schema does not pin genome_id, so accession is
                # the authoritative key; the loader itself verifies the data.
                if entry is None:
                    raise ValueError(f"genome {genome.accession!r} is not listed in data/manifest.json")
        present_ids = set(map(int, state.agents.genome_id))
        if present_ids - set(self._genome_by_id):
            raise ValueError("initial state contains genome IDs absent from the manifest catalogue")

        self._paused = False
        # Whether the conservation ledger's initial total has been captured. It cannot be inferred from the
        # data: an empty pristine world legitimately has initial_mmol == 0, and re-capturing after a command
        # has landed double-counts that command. Sessions that arrive already used count as initialised.
        field = environment.field
        self._ledger_initialized = bool(
            np.any(field.initial_mmol) or np.any(field.source_cum_mmol) or np.any(field.sink_cum_mmol)
            or int(state.tick_index) > 0
        )
        self._next_command_id = 0
        self._command_log: list[_QueuedCommand] = []
        self._results: dict[int, CommandResult] = {}
        self._species_by_genome: dict[int, int] = {}
        for genome_id, species_id in zip(state.agents.genome_id, state.agents.species_id, strict=True):
            key, value = int(genome_id), int(species_id)
            previous = self._species_by_genome.setdefault(key, value)
            if previous != value:
                raise ValueError(f"genome {key} occurs under multiple species IDs in the initial state")
        if realism is not None:
            # The oxygen field starts non-zero, so the ledger's initial total must exist before the first tick.
            self._initialize_ledger()

    @classmethod
    def create_demo(
        cls, seed: int = 2026, project_root: str | Path | None = None, mutation: bool = False,
        realism: bool | RealismConfig = False,
    ) -> "SimulationController":
        """Create an empty D1 world backed by the shipped real genome catalog.

        ``mutation=True`` enables D4 daughter SNVs (rate model ``snv-poisson-2e-10-v1``); it is off by default.
        ``realism=True`` (or a ``RealismConfig``) swaps in the realism layer: six chemical channels (oxygen, acetate, a
        toxin and a signal join the two sugars), run-and-tumble movement, physiology and more; see docs/REALISM.md.
        """
        root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
        table = _table(root)
        channels = tuple(
            NutrientChannel(int(row["nutrient_id"]), str(row["name"]))
            for row in sorted(table["demo_nutrient_channels"], key=lambda item: int(item["nutrient_id"]))
        )
        manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
        entries = sorted(manifest["genomes"], key=lambda item: str(item["accession"]))
        genomes = tuple(load_genome(str(entry["accession"]), root) for entry in entries)
        provenance = {
            str(entry["accession"]): {
                name: str(entry.get(name, ""))
                for name in ("source_url", "fetched_at_utc", "terms_url", "assembly_record_url", "license_or_terms")
            }
            for entry in entries
        }
        realism_config = (realism if isinstance(realism, RealismConfig) else RealismConfig()) if realism else None
        if realism_config is not None:
            from microbiome.genome.physiology import CHANNEL_NAMES

            channels = tuple(NutrientChannel(i, name) for i, name in enumerate(CHANNEL_NAMES))
        nx, ny, dx_um, dy_um, depth_um = 80, 64, 10.0, 10.0, 10.0
        molar_mass = {"glucose": 180.156, "lactose": 342.296, "oxygen": 31.998, "acetate": 60.052,
                      "bacilysin": 270.28, "ai2": 134.13, "pyocyanin": 210.23}
        diffusivity = {"glucose": 600.0, "lactose": 450.0, "oxygen": 1400.0, "acetate": 1100.0,
                       "bacilysin": 400.0, "ai2": 500.0, "pyocyanin": 400.0}   # um^2/s; oxygen is the effective value in dense cell layers
        try:
            molar_masses = np.asarray([molar_mass[channel.name] for channel in channels], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"no demo molar mass configured for nutrient {exc.args[0]!r}") from exc
        k = len(channels)
        conc0 = np.zeros((k, ny, nx), dtype=np.float64)
        if realism_config is not None:
            conc0[[c.name for c in channels].index("oxygen")] = realism_config.oxygen_saturation_mmol_l
        field = NutrientField(
            nutrient_id=np.asarray([channel.nutrient_id for channel in channels], dtype=np.uint32),
            concentration_mmol_l=conc0,
            diffusion_um2_s=(np.asarray([diffusivity[c.name] for c in channels]) if realism_config is not None
                             else np.full(k, 10.0, dtype=np.float64)),
            molar_mass_g_mol=molar_masses,
            detritus_mmol=np.zeros((k, ny, nx), dtype=np.float64),
            initial_mmol=np.zeros(k, dtype=np.float64),
            source_cum_mmol=np.zeros(k, dtype=np.float64),
            sink_cum_mmol=np.zeros(k, dtype=np.float64),
            respiration_loss_mmol=np.zeros(k, dtype=np.float64),
        )
        environment = Environment(
            width_um=nx * dx_um,
            height_um=ny * dy_um,
            depth_um=depth_um,
            dx_um=dx_um,
            dy_um=dy_um,
            nx=nx,
            ny=ny,
            field=field,
        )
        state = SimulationState(
            agents=AgentArrays(
                agent_id=np.empty(0, dtype=np.uint64),
                position_um=np.empty((0, 2), dtype=np.float64),
                velocity_um_s=np.empty((0, 2), dtype=np.float64),
                mass_pg=np.empty(0, dtype=np.float64),
                radius_um=np.empty(0, dtype=np.float64),
                energy_pj=np.empty(0, dtype=np.float64),
                species_id=np.empty(0, dtype=np.uint32),
                genome_id=np.empty(0, dtype=np.uint64),
                variant_id=np.empty(0, dtype=np.uint64),
                age_ticks=np.empty(0, dtype=np.uint64),
                nutrient_store_mmol=np.empty((0, k), dtype=np.float64),
                biomass_mmol=np.empty((0, k), dtype=np.float64),
            ),
            tick_index=np.uint64(0),
            next_agent_id=np.uint64(0),
        )
        provider = None
        if mutation:
            from microbiome.genome.tables import load_gene_trait_map, load_trait_schema
            from microbiome.genome.variants import VariantMutationProvider

            provider = VariantMutationProvider(
                genomes,
                load_trait_schema(root / "data" / "trait_schema.tsv"),
                load_gene_trait_map(root / "data" / "gene_trait_map.tsv"),
                project_root=root,
            )
            provider.warm()
        setup = None
        if realism_config is not None:
            from microbiome.genome.physiology import (
                CHANNEL_NAMES, REALISM_CHANNELS, RealismMutationProvider, derive_physiology, extend_phenotype_table,
                load_physiology_tables,
            )

            tables = load_physiology_tables(root)
            simulation = RealisticSimulation({int(g.genome_id): derive_physiology(g, tables) for g in genomes}, realism_config,
                                             REALISM_CHANNELS)
            setup = RealismSetup(simulation, tuple(CHANNEL_NAMES), lambda table: extend_phenotype_table(table, genomes, tables))
            if provider is not None:
                provider = RealismMutationProvider(provider, genomes, tables)
                simulation.transformer = provider       # natural transformation moves the variants mutation creates
        return cls(
            state,
            environment,
            initialize_rngs(seed),
            genomes,
            channels,
            project_root=root,
            provenance=provenance,
            mutation_provider=provider,
            realism=setup,
        )

    def _load_manifest_entries(self) -> dict[str, dict[str, object]]:
        path = self._project_root / "data" / "manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        entries = value.get("genomes")
        if not isinstance(entries, list):
            raise ValueError("data/manifest.json must contain a genome list")
        return {str(entry["accession"]): entry for entry in entries}

    def _position(self, position_um: Vec2Um) -> Vec2Um:
        try:
            position = tuple(float(value) for value in position_um)
        except (TypeError, ValueError) as exc:
            raise ValueError("position_um must contain two finite coordinates") from exc
        if len(position) != 2 or not all(np.isfinite(value) for value in position):
            raise ValueError("position_um must contain two finite coordinates")
        x, y = position
        if not (0.0 <= x < self._environment.width_um and 0.0 <= y < self._environment.height_um):
            raise ValueError("position_um must lie inside the half-open world bounds")
        return (x, y)

    def _queue(
        self,
        kind: str,
        *,
        nutrient_id: int | None = None,
        position_um: Vec2Um | None = None,
        amount_mmol: float | None = None,
        genome_id: int | None = None,
    ) -> CommandTicket:
        if self._next_command_id > _UINT64_MAX:
            raise OverflowError("command_id uint64 space exhausted")
        command_id = self._next_command_id
        target_tick = int(self._state.tick_index) + 1
        if target_tick > _UINT64_MAX:
            raise OverflowError("tick_index uint64 space exhausted")
        command = _QueuedCommand(
            command_id=command_id,
            target_tick=target_tick,
            kind=kind,
            nutrient_id=nutrient_id,
            position_um=position_um,
            amount_mmol=amount_mmol,
            genome_id=genome_id,
        )
        self._command_log.append(command)
        self._results[command_id] = CommandResult(
            np.uint64(command_id), "pending", np.uint64(target_tick)
        )
        self._next_command_id += 1
        return CommandTicket(np.uint64(command_id), np.uint64(target_tick))

    def list_genomes(self) -> tuple[GenomeOption, ...]:
        result = []
        for genome in self._genomes:
            entry = self._manifest_entries[genome.accession]
            provenance = self._provenance.get(genome.accession, {})
            sequence = entry["sequence_file"]
            annotation = entry["annotation_file"]
            result.append(
                GenomeOption(
                    genome_id=np.uint64(genome.genome_id),
                    accession_version=genome.accession,
                    organism=genome.organism,
                    tax_id=genome.tax_id,
                    assembly_level=genome.assembly_level,
                    sequence_sha256=str(sequence["sha256"]),
                    annotation_sha256=str(annotation["sha256"]),
                    source_url=provenance.get("source_url", str(entry.get("source_url", ""))),
                    fetched_at_utc=provenance.get("fetched_at_utc", str(entry.get("fetched_at_utc", ""))),
                    terms_url=provenance.get("terms_url", str(entry.get("terms_url", ""))),
                )
            )
        return tuple(result)

    def _evidence(self, genome_id: int) -> tuple[TraitEvidence, ...]:
        phenotype = self._phenotype_by_id[genome_id]
        table = _table(self._project_root)
        by_rule = {str(rule["rule_id"]): rule for rule in table["rules"]}
        channel_index = {channel.name: int(channel.nutrient_id) for channel in self._channels}
        result = []
        for rule_id in phenotype.applied_rule_ids:
            rule = by_rule[rule_id]
            trait_id = str(rule["trait"])
            nutrient_name = str(rule.get("nutrient_name", ""))
            gene_keys = tuple(str(key) for key in rule.get("requires_gene_keys", (rule["gene_key"],)))
            result.append(
                TraitEvidence(
                    rule_id=rule_id,
                    gene_keys=gene_keys,
                    trait_id=trait_id,
                    nutrient_id=(np.uint32(channel_index[nutrient_name]) if nutrient_name in channel_index else None),
                    effect_value=float(rule["value"]),
                    unit=str(rule["unit"]),
                    evidence_refs=tuple(str(item["url"]) for item in rule["evidence"]),
                    approximation_note=(
                        f"{rule['approximation_status']}: {rule['justification']}"
                    ),
                )
            )
        return tuple(result)

    def preview_phenotype(self, genome_id: GenomeId) -> PhenotypePreview:
        key = int(genome_id)
        phenotype = self._phenotype_by_id.get(key)
        if phenotype is None:
            raise KeyError(f"unknown manifest genome_id: {key}")
        return PhenotypePreview(
            genome_id=np.uint64(key),
            nutrient_id=np.asarray([channel.nutrient_id for channel in self._d2_channels], dtype=np.uint32),
            max_uptake_mmol_s=phenotype.max_uptake_mmol_s,
            storage_fraction=phenotype.storage_fraction,
            biomass_fraction=phenotype.biomass_fraction,
            energy_yield_pj_per_mmol=phenotype.energy_yield_pj_per_mmol,
            motility_um_s=phenotype.motility_um_s,
            chemotaxis_sensitivity=phenotype.chemotaxis_sensitivity,
            random_motility_fraction=phenotype.random_motility_fraction,
            division_rate_s=phenotype.division_rate_s,
            division_mass_pg=phenotype.division_mass_pg,
            death_rate_s=phenotype.death_rate_s,
            max_age_s=phenotype.max_age_s,
            antagonism_radius_um=phenotype.antagonism_radius_um,
            antagonism_rate_s=phenotype.antagonism_rate_s,
            applied_rule_ids=phenotype.applied_rule_ids,
            evidence=self._evidence(key),
        )

    def place_nutrient(self, nutrient_id: NutrientId, position_um: Vec2Um, amount_mmol: float) -> CommandTicket:
        nutrient_id = int(nutrient_id)
        nutrient_ids = tuple(int(value) for value in self._environment.field.nutrient_id)
        if nutrient_id not in nutrient_ids:
            raise ValueError(f"unknown nutrient_id: {nutrient_id}")
        position = self._position(position_um)
        amount = float(amount_mmol)
        if not np.isfinite(amount) or amount < 0.0:
            raise ValueError("amount_mmol must be finite and nonnegative")
        return self._queue("place_nutrient", nutrient_id=nutrient_id, position_um=position, amount_mmol=amount)

    def introduce_organism(self, genome_id: GenomeId, position_um: Vec2Um) -> CommandTicket:
        key = int(genome_id)
        if key not in self._genome_by_id:
            raise ValueError(f"unknown manifest genome_id: {key}")
        position = self._position(position_um)
        pending_introductions = sum(
            command.kind == "introduce_organism" and command.target_tick >= int(self._state.tick_index) + 1
            for command in self._command_log
        )
        if int(self._state.next_agent_id) + pending_introductions >= _UINT64_MAX:
            raise OverflowError("agent_id uint64 space exhausted")
        existing_species = list(map(int, self._state.agents.species_id))
        queued_genomes = {
            int(command.genome_id)
            for command in self._command_log
            if command.kind == "introduce_organism" and command.genome_id is not None
        }
        new_species_count = len(queued_genomes - set(self._species_by_genome))
        if key not in self._species_by_genome and key not in queued_genomes:
            new_species_count += 1
        next_species = max(existing_species, default=-1) + new_species_count
        if next_species > _UINT32_MAX:
            raise OverflowError("species_id uint32 space exhausted")
        return self._queue("introduce_organism", genome_id=key, position_um=position)

    def command_result(self, command_id: CommandId) -> CommandResult | None:
        try:
            key = int(command_id)
        except (TypeError, ValueError, OverflowError):
            return None
        return self._results.get(key)

    def inspect_cell(self, agent_id: int) -> CellInspection | None:
        agents = self._state.agents
        matches = np.flatnonzero(agents.agent_id == np.uint64(agent_id))
        if matches.size == 0:
            return None
        row = int(matches[0])
        genome_id = int(agents.genome_id[row])
        phenotype = self.preview_phenotype(np.uint64(genome_id))
        genome = self._genome_by_id[genome_id]
        return CellInspection(
            agent_id=agents.agent_id[row],
            species_id=agents.species_id[row],
            genome_id=agents.genome_id[row],
            accession_version=genome.accession,
            organism=genome.organism,
            tax_id=genome.tax_id,
            position_um=tuple(agents.position_um[row]),
            velocity_um_s=tuple(agents.velocity_um_s[row]),
            mass_pg=agents.mass_pg[row],
            radius_um=agents.radius_um[row],
            energy_pj=agents.energy_pj[row],
            age_s=float(agents.age_ticks[row]) * DT_S,
            nutrient_id=self._environment.field.nutrient_id,
            nutrient_store_mmol=agents.nutrient_store_mmol[row],
            biomass_mmol=agents.biomass_mmol[row],
            phenotype=phenotype,
        )

    def inspect_genome(self, genome_id: GenomeId) -> GenomeInspection | None:
        key = int(genome_id)
        genome = self._genome_by_id.get(key)
        if genome is None:
            return None
        entry = self._manifest_entries[genome.accession]
        return GenomeInspection(
            genome=next(option for option in self.list_genomes() if int(option.genome_id) == key),
            sequence_fasta_path=str((self._project_root / str(entry["sequence_file"]["path"])).resolve()),
            annotation_gff3_path=str((self._project_root / str(entry["annotation_file"]["path"])).resolve()),
            genes=genome.genes,
            phenotype_evidence=self._evidence(key),
        )

    def set_paused(self, paused: bool) -> None:
        if not isinstance(paused, (bool, np.bool_)):
            raise TypeError("paused must be bool")
        self._paused = bool(paused)

    def _initialize_ledger(self) -> None:
        field = self._environment.field
        if self._ledger_initialized:
            return
        cell_volume_l = self._environment.dx_um * self._environment.dy_um * self._environment.depth_um * 1e-15
        field.initial_mmol[:] = (
            np.sum(field.concentration_mmol_l, axis=(1, 2), dtype=np.float64) * cell_volume_l
            + np.sum(self._state.agents.nutrient_store_mmol + self._state.agents.biomass_mmol, axis=0, dtype=np.float64)
            + np.sum(field.detritus_mmol, axis=(1, 2), dtype=np.float64)
        )
        self._ledger_initialized = True

    def _introduce(self, genome_id: int, position_um: Vec2Um) -> int:
        state, environment = self._state, self._environment
        agents, field = state.agents, environment.field
        next_id = int(state.next_agent_id)
        if next_id >= _UINT64_MAX:
            raise OverflowError("agent_id uint64 space exhausted")
        if genome_id in self._species_by_genome:
            species_id = self._species_by_genome[genome_id]
        else:
            species_id = max(map(int, agents.species_id), default=-1) + 1
            if species_id > _UINT32_MAX:
                raise OverflowError("species_id uint32 space exhausted")

        phenotype = self._phenotype_by_id[genome_id]
        target_mass_pg = float(phenotype.division_mass_pg) * 0.5
        given = np.asarray(phenotype.max_uptake_mmol_s, dtype=np.float64)
        weights = np.zeros(field.molar_mass_g_mol.size, dtype=np.float64)
        weights[: given.size] = given
        founder = np.ones_like(weights)
        if self._realism is not None:
            founder = np.zeros_like(weights)
            founder[list(self._realism.founder_channels)] = 1.0
            weights *= founder
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("genome phenotype has invalid nutrient uptake values")
        if float(weights.sum()) <= 0.0:
            weights[:] = founder
        weights /= weights.sum()
        biomass = target_mass_pg * weights / (np.asarray(field.molar_mass_g_mol, dtype=np.float64) * 1e9)
        mass_pg = float(np.sum(biomass * field.molar_mass_g_mol * 1e9, dtype=np.float64))
        radius_um = float(np.cbrt(3.0 * mass_pg / (4.0 * np.pi)))
        row_values = {
            "agent_id": np.asarray([next_id], dtype=np.uint64),
            "position_um": np.asarray([position_um], dtype=np.float64),
            "velocity_um_s": np.zeros((1, 2), dtype=np.float64),
            "mass_pg": np.asarray([mass_pg], dtype=np.float64),
            "radius_um": np.asarray([radius_um], dtype=np.float64),
            "energy_pj": np.zeros(1, dtype=np.float64),
            "species_id": np.asarray([species_id], dtype=np.uint32),
            "genome_id": np.asarray([genome_id], dtype=np.uint64),
            "variant_id": np.zeros(1, dtype=np.uint64),
            "age_ticks": np.zeros(1, dtype=np.uint64),
            "nutrient_store_mmol": np.zeros((1, len(self._channels)), dtype=np.float64),
            "biomass_mmol": biomass.reshape(1, -1),
        }
        self._initialize_ledger()
        for name, row in row_values.items():
            setattr(agents, name, np.ascontiguousarray(np.concatenate((getattr(agents, name), row), axis=0)))
        state.next_agent_id = np.uint64(next_id + 1)
        field.source_cum_mmol[:] += biomass
        self._species_by_genome[genome_id] = species_id
        return next_id

    def _apply_command(self, command: _QueuedCommand, applied_tick: int) -> None:
        if command.kind == "place_nutrient":
            nutrient_ids = list(map(int, self._environment.field.nutrient_id))
            nutrient_index = nutrient_ids.index(int(command.nutrient_id))
            x, y = command.position_um
            ix = min(self._environment.nx - 1, int(x / self._environment.dx_um))
            iy = min(self._environment.ny - 1, int(y / self._environment.dy_um))
            # Route through the same conservation-aware engine utility used by
            # native D1 state; bridge the nested D1 records without exposing them.
            state = self._state
            env = self._environment
            native_state = AgentState(
                agent_id=state.agents.agent_id,
                positions_um=state.agents.position_um,
                velocities_um_s=state.agents.velocity_um_s,
                mass_pg=state.agents.mass_pg,
                radius_um=state.agents.radius_um,
                energy_pj=state.agents.energy_pj,
                species_id=state.agents.species_id,
                genome_id=state.agents.genome_id,
                variant_id=state.agents.variant_id,
                age_s=np.asarray(state.agents.age_ticks, dtype=np.float64) * DT_S,
                nutrient_store_mmol=state.agents.nutrient_store_mmol,
                biomass_mmol=state.agents.biomass_mmol,
                tick=int(state.tick_index),
                next_agent_id=int(state.next_agent_id),
                initial_mmol=env.field.initial_mmol,
                source_cum_mmol=env.field.source_cum_mmol,
                sink_cum_mmol=env.field.sink_cum_mmol,
                respiration_loss_mmol=env.field.respiration_loss_mmol,
                ledger_initialized=self._ledger_initialized or int(state.tick_index) > 0,
            )
            native_env = NutrientEnvironment(
                nutrient_id=env.field.nutrient_id,
                concentration_mmol_l=env.field.concentration_mmol_l,
                diffusion_um2_s=env.field.diffusion_um2_s,
                molar_mass_g_mol=env.field.molar_mass_g_mol,
                detritus_mmol=env.field.detritus_mmol,
                dx_um=env.dx_um,
                dy_um=env.dy_um,
                depth_um=env.depth_um,
                width_um=env.width_um,
                height_um=env.height_um,
            )
            add_nutrient_source(
                native_env,
                native_state,
                nutrient_index,
                ix,
                iy,
                float(command.amount_mmol),
            )
            env.field.concentration_mmol_l = native_env.concentration_mmol_l
            env.field.initial_mmol = native_state.initial_mmol
            env.field.source_cum_mmol = native_state.source_cum_mmol
            self._ledger_initialized = True
            self._results[command.command_id] = CommandResult(
                np.uint64(command.command_id), "applied", np.uint64(command.target_tick), np.uint64(applied_tick)
            )
            return
        if command.kind == "introduce_organism":
            agent_id = self._introduce(int(command.genome_id), command.position_um)
            self._results[command.command_id] = CommandResult(
                np.uint64(command.command_id),
                "applied",
                np.uint64(command.target_tick),
                np.uint64(applied_tick),
                np.uint64(agent_id),
            )
            return
        raise RuntimeError(f"unknown command kind: {command.kind}")

    def advance_ticks(self, count: int) -> np.uint64:
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise TypeError("count must be an integer number of ticks")
        count = int(count)
        if count < 0:
            raise ValueError("count must be nonnegative")
        if int(self._state.tick_index) + count > _UINT64_MAX:
            raise OverflowError("tick_index uint64 space exhausted")
        for _ in range(count):
            target_tick = int(self._state.tick_index) + 1
            for command in self._command_log:
                if command.target_tick != target_tick:
                    continue
                try:
                    self._apply_command(command, target_tick)
                except (ValueError, OverflowError, IndexError, FloatingPointError) as exc:
                    self._results[command.command_id] = CommandResult(
                        np.uint64(command.command_id),
                        "rejected",
                        np.uint64(command.target_tick),
                        np.uint64(target_tick),
                        error=str(exc),
                    )
            step(self._state, self._environment, self._phenotypes, self._rngs, mutation_provider=self._mutation,
                 simulation=self._realism.simulation if self._realism is not None else None)
        return np.uint64(self._state.tick_index)

    def single_step(self) -> np.uint64:
        return self.advance_ticks(1)

    def render_snapshot(self) -> RenderSnapshot:
        agents = self._state.agents
        phenotype_order = np.argsort(self._phenotypes.genome_id, kind="stable")
        variant_rows = getattr(self._phenotypes, "variant_id", None)
        if variant_rows is not None:
            # D3's frozen snapshot is keyed by root genome_id. Until that DTO
            # carries variant_id, render its unique unmutated root rows rather
            # than exposing duplicate genome IDs to the renderer.
            root_rows = np.asarray(variant_rows, dtype=np.uint64) == 0
            phenotype_order = phenotype_order[root_rows[phenotype_order]]
        snapshot = RenderSnapshot(
            tick_index=self._state.tick_index,
            width_um=self._environment.width_um,
            height_um=self._environment.height_um,
            agent_id=agents.agent_id,
            position_um=agents.position_um,
            radius_um=agents.radius_um,
            genome_id=agents.genome_id,
            nutrient_id=self._environment.field.nutrient_id,
            concentration_mmol_l=self._environment.field.concentration_mmol_l,
            phenotype_genome_id=self._phenotypes.genome_id[phenotype_order],
            phenotype_motility_um_s=self._phenotypes.motility_um_s[phenotype_order],
            phenotype_max_uptake_mmol_s=self._phenotypes.max_uptake_mmol_s[phenotype_order],
        )
        if self._realism is None:
            return snapshot
        sim = self._realism.simulation
        life, _ = sim.life.aligned(agents.agent_id, sim.config)
        code = np.zeros(len(agents.agent_id), dtype=np.uint8)
        code[(life.spore > 0.0) & (life.spore < 1.0)] = 1
        code[life.spore >= 1.0] = 2
        code[(life.eps > 0.5) & (life.spore <= 0.0)] = 3
        return RealismSnapshot(
            **{f.name: getattr(snapshot, f.name) for f in dataclasses.fields(RenderSnapshot)},
            channel_names=self._realism.channel_names, cell_state=code, eps=life.eps,
            length_um=sim.cell_shape(agents.genome_id, agents.mass_pg)[0], cell_width_um=sim.config.cell_width_um,
            heading_rad=life.heading, genome_style=sim.imaging_styles(snapshot.phenotype_genome_id),
        )

    # ---- optional extras beyond the frozen D3 facade; the UI uses them only when present --------------------------
    def realism_enabled(self) -> bool:
        return self._realism is not None

    def channel_names(self) -> tuple[str, ...]:
        return self._realism.channel_names if self._realism is not None else tuple(c.name for c in self._channels)

    def realism_stats(self) -> dict[str, int]:
        """Counts from the last tick: births, deaths by cause, spores, tumbles, and so on."""
        return dict(self._realism.simulation.stats) if self._realism is not None else {}

    def chemistry_at(self, position_um: Vec2Um) -> dict[str, float]:
        """Concentrations (mmol/L) by channel name at a point, plus pH when acetate is modelled."""
        env = self._environment
        _, ix, iy = NutrientEnvironment.cell_indices(
            _GridView(env), np.asarray([position_um], dtype=np.float64))
        values = env.field.concentration_mmol_l[:, int(iy[0]), int(ix[0])]
        out = {name: float(v) for name, v in zip(self.channel_names(), values)}
        if self._realism is not None:
            out["pH"] = float(ph_from_acetate(np.float64(out["acetate"]), self._realism.simulation.config))
        return out

    def life_of(self, agent_id: int) -> dict[str, float] | None:
        """Realism state of one cell (lag, stress, spore progress, matrix, energy), or None if it is gone."""
        if self._realism is None:
            return None
        agents = self._state.agents
        rows = np.flatnonzero(agents.agent_id == np.uint64(agent_id))
        if rows.size == 0:
            return None
        sim = self._realism.simulation
        life, _ = sim.life.aligned(agents.agent_id, sim.config)
        row = int(rows[0])
        stress = 1.0 - float(np.exp(-life.stress_s[row] / sim.config.starvation_tau_s))
        return {"lag": float(life.lag[row]), "stress": stress, "spore": float(life.spore[row]),
                "matrix": float(life.eps[row]), "energy_pj": float(agents.energy_pj[row]),
                "alt_sugar_expression": float(life.derepression[row])}


class _GridView:
    """The few geometry attributes ``NutrientEnvironment.cell_indices`` reads, taken from a D1 Environment."""

    def __init__(self, env: Environment) -> None:
        self.width_um, self.height_um = env.width_um, env.height_um
        self.dx_um, self.dy_um, self.nx, self.ny = env.dx_um, env.dy_um, env.nx, env.ny

    wrap_positions = NutrientEnvironment.wrap_positions


__all__ = ["SimulationController", "SimulationControllerProtocol"]
