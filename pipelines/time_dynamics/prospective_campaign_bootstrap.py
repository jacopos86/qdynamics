"""Frozen identities and dependency planning for prospective Paper-II lanes.

This module deliberately stops before execution.  It turns the authorized
parallel benchmark matrix into three things that a production adapter can use
without rediscovering scientific intent:

* one seed identity per physical Hamiltonian (shared across its drives),
* one full-horizon exact-reference identity per physical drive cell, and
* an idempotent bootstrap graph whose ready actions are explicit.

The graph keeps exact-reference construction and the reference-blind AVQDS
screen parallel once a seed is verified.  Offline scoring joins those branches.
Full APM is intentionally absent here: the comparator-first scientific planner
may open it only after the AVQDS scout has been scored.  SIAM already has that
planner and is represented as a delegated, verified-binding lane rather than a
second competing action graph.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.paper_ii_runs import DRIVES
from pipelines.time_dynamics.parallel_benchmark_matrix import (
    ParallelBenchmarkMatrix,
)


IDENTITY_REGISTRY_SCHEMA_V1 = "paper_ii_prospective_identity_registry_v1"
BOOTSTRAP_PLAN_SCHEMA_V1 = "paper_ii_prospective_bootstrap_plan_v1"
PROSPECTIVE_EXACT_REFERENCE_DATA_V1 = (
    "paper_ii_prospective_exact_reference_data_v1"
)

LANE_HH = "paper_ii_hh_product_v1"
LANE_SIAM = "siam_nb4_pilot_v1"
LANE_HUBBARD = "paper_ii_hubbard_l4_scale_v1"

EXEC_HH_SEED = "paper_ii.hh.generator_derived_fixed_structure_vqe_seed_v1"
EXEC_HUBBARD_SEED = "paper_ii.hubbard_l4.generator_derived_fixed_structure_vqe_seed_v1"
EXEC_HH_REFERENCE = "paper_ii.hh.full_horizon_exact_reference_v1"
EXEC_HUBBARD_REFERENCE = "paper_ii.hubbard_l4.full_horizon_exact_reference_v1"
EXEC_HH_AVQDS = "paper_ii.hh.state_motion_avqds_calibration_v1"
EXEC_HUBBARD_AVQDS = "paper_ii.hubbard_l4.state_motion_avqds_calibration_v1"
EXEC_HH_SCORE = "paper_ii.hh.response_normalized_offline_score_v1"
EXEC_HUBBARD_SCORE = "paper_ii.hubbard_l4.response_normalized_offline_score_v1"

DELEGATED_SIAM_PLANNER = (
    "pipelines.time_dynamics.siam_nb4_ladder.plan_siam_nb4_ladder"
)


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    token = str(value).strip().lower()
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return token


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activation_cut_ladder(contract: Mapping[str, Any]) -> tuple[float, ...]:
    """Expand the registered ``1e-k, 3e-(k+1)`` ladder through its floor."""

    start = float(contract["start"])
    ratio = float(contract["ratio"])
    floor = float(contract["floor"])
    if start <= 0.0 or floor <= 0.0 or floor >= start or ratio != 3.0:
        raise ValueError("activation ladder must be a positive descending 1/3 ladder")
    values = [start]
    while values[-1] > floor:
        mantissa_text, exponent_text = f"{values[-1]:.0e}".split("e")
        mantissa = int(mantissa_text)
        exponent = int(exponent_text)
        if mantissa == 1:
            candidate = 3.0 * 10.0 ** (exponent - 1)
        elif mantissa == 3:
            candidate = 1.0 * 10.0 ** exponent
        else:
            raise ValueError("activation ladder must use registered 1/3-decade rungs")
        if candidate <= floor:
            if bool(contract.get("include_floor", False)):
                values.append(floor)
            break
        values.append(float(f"{candidate:.12g}"))
    if values[-1] != floor:
        raise ValueError("activation ladder contract does not include its floor")
    return tuple(values)


def _uniform_grid_contract(
    *, start: float, stop: float, points: int, prefix_stop: float, prefix_points: int
) -> dict[str, Any]:
    start_f = float(start)
    stop_f = float(stop)
    count = int(points)
    prefix_count = int(prefix_points)
    if start_f != 0.0 or stop_f <= start_f or count < 2:
        raise ValueError("prospective grids must start at zero and contain two points")
    step = (stop_f - start_f) / float(count - 1)
    times = tuple(start_f + float(index) * step for index in range(count))
    prefix = times[:prefix_count]
    if not prefix or not math.isclose(prefix[-1], float(prefix_stop), abs_tol=1e-14):
        raise ValueError("prospective prefix is not an exact ordered-grid prefix")
    return {
        "start": start_f,
        "stop": stop_f,
        "points": count,
        "step": step,
        "ordered_times_sha256": _canonical_sha(times),
        "prefix_stop": float(prefix_stop),
        "prefix_points": prefix_count,
        "ordered_prefix_sha256": _canonical_sha(prefix),
        "construction": "ordered_uniform_index_times_step",
    }


@dataclass(frozen=True)
class SeedIdentity:
    """One immutable seed plan shared by all drives of a Hamiltonian."""

    seed_id: str
    lane_id: str
    model: str
    physical: Mapping[str, Any]
    policy: str
    energy_gate: float | None
    artifact_path: str
    external_artifact_sha256: str | None = None
    initializer: str | None = None
    restarts: int | None = None
    optimizer_seed: int | None = None

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha(
            {
                "seed_id": self.seed_id,
                "lane_id": self.lane_id,
                "model": self.model,
                "physical": dict(self.physical),
                "policy": self.policy,
                "energy_gate": self.energy_gate,
                "initializer": self.initializer,
                "restarts": self.restarts,
                "optimizer_seed": self.optimizer_seed,
            }
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "lane_id": self.lane_id,
            "model": self.model,
            "physical": dict(self.physical),
            "policy": self.policy,
            "energy_gate": self.energy_gate,
            "artifact_path": self.artifact_path,
            "external_artifact_sha256": self.external_artifact_sha256,
            "initializer": self.initializer,
            "restarts": self.restarts,
            "optimizer_seed": self.optimizer_seed,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True)
class ExactReferenceIdentity:
    """Physical reference plan; its content key also binds the seed artifact."""

    reference_id: str
    lane_id: str
    cell_id: str
    seed_id: str
    model: str
    physical: Mapping[str, Any]
    drive: Mapping[str, Any]
    grid: Mapping[str, Any]
    propagator: str
    observable_schema: str
    artifact_root: str
    external_manifest_path: str | None = None
    external_manifest_sha256: str | None = None
    external_data_path: str | None = None
    external_data_sha256: str | None = None

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha(
            {
                "reference_id": self.reference_id,
                "lane_id": self.lane_id,
                "cell_id": self.cell_id,
                "seed_id": self.seed_id,
                "model": self.model,
                "physical": dict(self.physical),
                "drive": dict(self.drive),
                "grid": dict(self.grid),
                "propagator": self.propagator,
                "observable_schema": self.observable_schema,
            }
        )

    def materialized_key(self, seed_artifact_sha256: str) -> str:
        """Return the final content key after a seed artifact is verified."""

        seed_sha = _require_sha256(
            seed_artifact_sha256, label="seed artifact digest"
        )
        return _canonical_sha(
            {
                "reference_identity_sha256": self.identity_sha256,
                "seed_artifact_sha256": seed_sha,
            }
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "lane_id": self.lane_id,
            "cell_id": self.cell_id,
            "seed_id": self.seed_id,
            "model": self.model,
            "physical": dict(self.physical),
            "drive": dict(self.drive),
            "grid": dict(self.grid),
            "propagator": self.propagator,
            "observable_schema": self.observable_schema,
            "artifact_root": self.artifact_root,
            "external_manifest_path": self.external_manifest_path,
            "external_manifest_sha256": self.external_manifest_sha256,
            "external_data_path": self.external_data_path,
            "external_data_sha256": self.external_data_sha256,
            "identity_sha256": self.identity_sha256,
            "materialized_key_requires": "verified_seed_artifact_sha256",
        }


@dataclass(frozen=True)
class ProspectiveIdentityRegistry:
    """Complete frozen seed/reference identities for all prospective cells."""

    matrix_id: str
    matrix_payload_sha256: str
    matrix_raw_sha256: str
    matrix_expanded_semantic_sha256: str
    accuracy_policy: Mapping[str, Any]
    activation_ladders: Mapping[str, Any]
    seeds: tuple[SeedIdentity, ...]
    references: tuple[ExactReferenceIdentity, ...]
    delegated_planners: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_sha256(self.matrix_payload_sha256, label="matrix payload digest")
        _require_sha256(self.matrix_raw_sha256, label="matrix raw digest")
        _require_sha256(
            self.matrix_expanded_semantic_sha256,
            label="matrix expanded semantic digest",
        )
        seed_ids = [row.seed_id for row in self.seeds]
        reference_ids = [row.reference_id for row in self.references]
        if len(seed_ids) != len(set(seed_ids)):
            raise ValueError("prospective identity registry repeats a seed_id")
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("prospective identity registry repeats a reference_id")
        known_seeds = set(seed_ids)
        for row in self.references:
            if row.seed_id not in known_seeds:
                raise ValueError("prospective reference names an unknown seed")

    @property
    def digest(self) -> str:
        return _canonical_sha(self.to_json_dict(include_digest=False))

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema": IDENTITY_REGISTRY_SCHEMA_V1,
            "matrix_id": self.matrix_id,
            "matrix_payload_sha256": self.matrix_payload_sha256,
            "matrix_raw_sha256": self.matrix_raw_sha256,
            "matrix_expanded_semantic_sha256": (
                self.matrix_expanded_semantic_sha256
            ),
            "accuracy_policy": dict(self.accuracy_policy),
            "accuracy_policy_sha256": _canonical_sha(self.accuracy_policy),
            "activation_ladders": dict(self.activation_ladders),
            "activation_ladders_sha256": _canonical_sha(self.activation_ladders),
            "seeds": [row.to_json_dict() for row in self.seeds],
            "references": [row.to_json_dict() for row in self.references],
            "delegated_planners": dict(self.delegated_planners),
            "seed_count": len(self.seeds),
            "reference_count": len(self.references),
        }
        if include_digest:
            payload["registry_sha256"] = self.digest
        return payload

    def seed_for_cell(self, cell_id: str) -> SeedIdentity:
        matching = [row for row in self.references if row.cell_id == cell_id]
        if len(matching) != 1:
            raise KeyError(f"unknown or repeated prospective cell {cell_id!r}")
        seed_id = matching[0].seed_id
        return next(row for row in self.seeds if row.seed_id == seed_id)


def _drive_contract(matrix: ParallelBenchmarkMatrix, drive_id: str) -> dict[str, Any]:
    if drive_id == "hybridization_quench":
        return {
            "drive_id": drive_id,
            "kind": "sudden_hybridization_quench",
            "V_pre_quench": float(matrix.payload["siam_nb4"]["V_pre_quench"]),
            "V_post_quench": float(matrix.payload["siam_nb4"]["V"]),
            "quench_time": 0.0,
        }
    if drive_id not in DRIVES:
        raise ValueError(f"prospective matrix names unknown drive {drive_id!r}")
    drive = DRIVES[drive_id]
    waveform = dict(matrix.payload["common"]["drive_waveform"])
    return {
        "drive_id": drive_id,
        "amplitude": float(drive.amplitude),
        "omega": float(drive.omega),
        **waveform,
    }


def _seed_key(cell: Any) -> tuple[str, str]:
    if cell.lane_id == LANE_HH:
        return cell.lane_id, (
            f"nph{int(cell.physics['n_ph_max'])}__{cell.physics['regime']}"
        )
    if cell.lane_id == LANE_SIAM:
        return cell.lane_id, "nb4__givens_prequench"
    if cell.lane_id == LANE_HUBBARD:
        return cell.lane_id, "l4__u4__half_filled"
    raise ValueError(f"unsupported prospective lane {cell.lane_id!r}")


def build_identity_registry(
    matrix: ParallelBenchmarkMatrix | None = None,
) -> ProspectiveIdentityRegistry:
    """Expand the matrix into immutable seed and exact-reference identities."""

    campaign = matrix or ParallelBenchmarkMatrix.read()
    if campaign.payload["common"].get("qiskit_compilation") is not False:
        raise ValueError("prospective identities prohibit Qiskit compilation")

    seed_rows: dict[tuple[str, str], SeedIdentity] = {}
    reference_rows: list[ExactReferenceIdentity] = []
    common = campaign.payload["common"]
    driven_grid = _uniform_grid_contract(
        start=common["driven_grid"]["start"],
        stop=common["driven_grid"]["stop"],
        points=common["driven_grid"]["points"],
        prefix_stop=common["driven_grid"]["calibration_prefix_stop"],
        prefix_points=common["driven_grid"]["calibration_prefix_points"],
    )
    siam_grid = _uniform_grid_contract(
        start=common["siam_grid"]["start"],
        stop=common["siam_grid"]["stop"],
        points=common["siam_grid"]["points"],
        prefix_stop=common["siam_grid"]["screen_prefix_stop"],
        prefix_points=common["siam_grid"]["screen_prefix_points"],
    )

    for cell in campaign.cells:
        seed_key = _seed_key(cell)
        if seed_key not in seed_rows:
            lane, token = seed_key
            if lane == LANE_HH:
                row = campaign.payload["hh_product"]
                external_sha = None
                artifact_path = f"output/{lane}/seeds/{token}/seed.json"
                energy_gate = float(row["seed_energy_gate"])
                policy = str(row["seed_policy"])
                initializer = "generator_derived_fixed_structure_vqe_v1"
                restarts = 4
            elif lane == LANE_HUBBARD:
                row = campaign.payload["hubbard_l4"]
                external_sha = None
                artifact_path = f"output/{lane}/seeds/{token}/seed.json"
                energy_gate = float(row["seed_energy_gate"])
                policy = str(row["seed_policy"])
                initializer = "generator_derived_fixed_structure_vqe_v1"
                restarts = 4
            else:
                row = campaign.payload["siam_nb4"]
                seed_artifact = row["seed_artifact"]
                artifact_path = str(seed_artifact["path"])
                external_sha = _require_sha256(
                    seed_artifact["sha256"], label="SIAM seed digest"
                )
                energy_gate = None
                policy = str(row["seed_policy"])
                initializer = None
                restarts = None
            seed_id = f"{lane}::seed::{token}"
            optimizer_seed = (
                None
                if initializer is None
                else int(_canonical_sha({"seed_id": seed_id})[:8], 16)
            )
            seed_rows[seed_key] = SeedIdentity(
                seed_id=seed_id,
                lane_id=lane,
                model=cell.model,
                physical=dict(cell.physics),
                policy=policy,
                energy_gate=energy_gate,
                artifact_path=artifact_path,
                external_artifact_sha256=external_sha,
                initializer=initializer,
                restarts=restarts,
                optimizer_seed=optimizer_seed,
            )

        seed = seed_rows[seed_key]
        if cell.lane_id == LANE_HH:
            lane_row = campaign.payload["hh_product"]
            propagator = "paper_ii_exact_driven_reference_v1"
            grid = driven_grid
            external = {}
        elif cell.lane_id == LANE_HUBBARD:
            lane_row = campaign.payload["hubbard_l4"]
            propagator = "paper_ii_exact_driven_reference_v1"
            grid = driven_grid
            external = {}
        else:
            lane_row = campaign.payload["siam_nb4"]
            propagator = "siam_exact_sector_eigh_v1"
            grid = siam_grid
            external = dict(lane_row["exact_reference_artifact"])
        reference_id = f"{cell.lane_id}::reference::{cell.cell_id}"
        reference_rows.append(
            ExactReferenceIdentity(
                reference_id=reference_id,
                lane_id=cell.lane_id,
                cell_id=cell.cell_id,
                seed_id=seed.seed_id,
                model=cell.model,
                physical=dict(cell.physics),
                drive=_drive_contract(campaign, cell.drive),
                grid=dict(grid),
                propagator=propagator,
                observable_schema=str(lane_row["observable_schema"]),
                artifact_root=(
                    f"output/{cell.lane_id}/exact_references"
                    if cell.lane_id != LANE_SIAM
                    else "output/siam_screening/nb4_pilot_v1/exact_references"
                ),
                external_manifest_path=external.get("manifest_path"),
                external_manifest_sha256=external.get("manifest_sha256"),
                external_data_path=external.get("data_path"),
                external_data_sha256=external.get("data_sha256"),
            )
        )

    seeds = tuple(sorted(seed_rows.values(), key=lambda row: row.seed_id))
    references = tuple(sorted(reference_rows, key=lambda row: row.reference_id))
    if len(seeds) != 8 or len(references) != 39:
        raise ValueError(
            "prospective matrix must bind exactly 8 seeds and 39 references"
        )
    return ProspectiveIdentityRegistry(
        matrix_id=campaign.matrix_id,
        matrix_payload_sha256=campaign.payload_sha256,
        matrix_raw_sha256=campaign.raw_sha256,
        matrix_expanded_semantic_sha256=campaign.expanded_semantic_sha256,
        accuracy_policy=dict(campaign.payload["accuracy_policy"]),
        activation_ladders=dict(campaign.payload["activation_ladders"]),
        seeds=seeds,
        references=references,
        delegated_planners={LANE_SIAM: DELEGATED_SIAM_PLANNER},
    )


def _verify_siam_seed_contract(
    *, seed: SeedIdentity, seed_path: Path, document: Mapping[str, Any]
) -> None:
    """Fail closed unless the portable SIAM seed binds the registered physics."""

    physical = document.get("physical")
    layout = document.get("layout")
    pool = document.get("pool")
    digests = document.get("digests")
    if not all(
        isinstance(value, Mapping) for value in (physical, layout, pool, digests)
    ):
        raise ValueError(
            "external SIAM seed lacks physical/layout/pool/digest contracts"
        )
    expected = seed.physical
    exact_pairs = {
        "n_bath": (physical.get("n_bath"), expected.get("n_bath")),
        "mode_order": (physical.get("mode_order"), expected.get("ordering")),
    }
    for label, (actual, registered) in exact_pairs.items():
        if actual != registered:
            raise ValueError(f"external SIAM seed {label} diverged")
    numeric_pairs = {
        "U": (physical.get("u"), expected.get("U")),
        "V_post_quench": (physical.get("v_post_quench"), expected.get("V")),
        "V_pre_quench": (
            physical.get("v_pre_quench"),
            expected.get("V_pre_quench"),
        ),
        "bath_hopping": (physical.get("hopping"), expected.get("t_b")),
    }
    for label, (actual, registered) in numeric_pairs.items():
        if actual is None or registered is None or not math.isclose(
            float(actual), float(registered), rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise ValueError(f"external SIAM seed {label} diverged")
    electron_counts = physical.get("electron_counts")
    if not isinstance(electron_counts, Mapping) or (
        int(electron_counts.get("up", -1)) != int(expected["n_up"])
        or int(electron_counts.get("down", -1)) != int(expected["n_down"])
    ):
        raise ValueError("external SIAM seed electron counts diverged")
    if physical.get("model") != "siam_finite_box_screening_v1":
        raise ValueError("external SIAM seed model identity diverged")
    if document.get("schema") != "siam_screening_seed_v2":
        raise ValueError("external SIAM seed schema diverged")
    expected_qubits = 2 * (int(expected["n_bath"]) + 1)
    expected_sector = math.comb(
        int(expected["n_bath"]) + 1, int(expected["n_up"])
    ) * math.comb(int(expected["n_bath"]) + 1, int(expected["n_down"]))
    if (
        layout.get("mode_order") != expected["ordering"]
        or int(layout.get("n_qubits", -1)) != expected_qubits
        or int(layout.get("sector_dimension", -1)) != expected_sector
        or dict(layout.get("electron_counts") or {})
        != {"up": int(expected["n_up"]), "down": int(expected["n_down"])}
    ):
        raise ValueError("external SIAM seed layout/sector identity diverged")

    labels = pool.get("ordered_labels")
    if not isinstance(labels, list) or not labels or not all(
        isinstance(value, str) and value for value in labels
    ):
        raise ValueError("external SIAM seed has no ordered logical pool")
    if int(pool.get("block_count", -1)) != len(labels):
        raise ValueError("external SIAM seed pool count diverged")
    observed_families = tuple(dict.fromkeys(value.split("_", 1)[0] for value in labels))
    if observed_families != tuple(expected.get("pool_families") or ()):
        raise ValueError("external SIAM seed pool families/order diverged")
    for name in (
        "initial_state_semantic_sha256",
        "ordered_logical_pool_sha256",
        "post_quench_hamiltonian_sha256",
        "pre_quench_hamiltonian_sha256",
    ):
        _require_sha256(str(digests.get(name) or ""), label=f"SIAM seed {name}")

    package_manifest_path = seed_path.with_name("manifest.json")
    if not package_manifest_path.is_file():
        raise FileNotFoundError(
            "external SIAM seed lacks its source-epoch package manifest"
        )
    package = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    source_epoch = package.get("source_epoch")
    files = package.get("files")
    if not isinstance(source_epoch, Mapping) or not isinstance(files, Mapping):
        raise ValueError("external SIAM package lacks source epoch/file bindings")
    seed_file = files.get("seed")
    if not isinstance(seed_file, Mapping):
        raise ValueError("external SIAM package does not bind its seed")
    expected_seed_sha = _file_sha256(seed_path)
    if seed_file.get("sha256") != expected_seed_sha or package.get(
        "seed_sha256"
    ) != expected_seed_sha:
        raise ValueError("external SIAM package/seed digest binding diverged")
    if source_epoch.get("source_epoch_id") != expected.get("source_epoch"):
        raise ValueError("external SIAM source epoch diverged")
    if source_epoch.get("source_revision") != expected.get("source_revision"):
        raise ValueError("external SIAM source revision diverged")
    if package.get("source_revision") != expected.get("source_revision"):
        raise ValueError("external SIAM package source revision diverged")
    _require_sha256(
        str(source_epoch.get("source_tree_files_sha256") or ""),
        label="SIAM source tree digest",
    )
    # The owning loader regenerates the state, both Hamiltonians, and every
    # ordered pool operator from the declared physics and checks all digests.
    from pipelines.time_dynamics.siam_seed import load_siam_seed_manifest

    load_siam_seed_manifest(seed_path)


def _verify_siam_reference_contract(
    *,
    reference: ExactReferenceIdentity,
    seed_document: Mapping[str, Any],
    document: Mapping[str, Any],
) -> None:
    """Bind the exact reference to registered SIAM physics/quench/propagator."""

    physical = document.get("physical")
    seed_physical = seed_document.get("physical")
    if not isinstance(physical, Mapping) or not isinstance(seed_physical, Mapping):
        raise ValueError("external SIAM reference lacks its physical contract")
    if dict(physical) != dict(seed_physical):
        raise ValueError("external SIAM reference/seed Hamiltonians diverged")
    if document.get("propagator_schema") != reference.propagator:
        raise ValueError("external SIAM reference propagator diverged")
    if reference.drive.get("kind") != "sudden_hybridization_quench" or not math.isclose(
        float(reference.drive.get("quench_time", math.nan)),
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("registered SIAM quench identity is unsupported")
    if not math.isclose(
        float(physical.get("v_pre_quench", math.nan)),
        float(reference.drive["V_pre_quench"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ) or not math.isclose(
        float(physical.get("v_post_quench", math.nan)),
        float(reference.drive["V_post_quench"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("external SIAM reference quench parameters diverged")


def verify_external_bindings(
    registry: ProspectiveIdentityRegistry,
    *,
    workspace_root: str | Path,
) -> BootstrapEvidence:
    """Verify externally frozen seed/reference artifacts, currently SIAM.

    This is a direct file/digest/identity check, not a discovery heuristic.  It
    returns only positive bindings and fails closed on any missing or divergent
    artifact.
    """

    root = Path(workspace_root)
    seed_bindings: dict[str, str] = {}
    seed_documents: dict[str, Mapping[str, Any]] = {}
    reference_bindings: dict[str, str] = {}
    source_epochs: dict[str, SourceEpochIdentity] = {}
    for seed in registry.seeds:
        if seed.external_artifact_sha256 is None:
            continue
        seed_path = root / seed.artifact_path
        if not seed_path.is_file():
            raise FileNotFoundError(f"external seed artifact is missing: {seed_path}")
        actual = _file_sha256(seed_path)
        if actual != seed.external_artifact_sha256:
            raise ValueError(f"external seed artifact digest mismatch: {seed.seed_id}")
        document = json.loads(seed_path.read_text(encoding="utf-8"))
        if seed.lane_id == LANE_SIAM:
            _verify_siam_seed_contract(
                seed=seed, seed_path=seed_path, document=document
            )
            source_epochs[seed.lane_id] = SourceEpochIdentity(
                lane_id=seed.lane_id,
                source_epoch_id=str(seed.physical["source_epoch"]),
                source_revision=str(seed.physical["source_revision"]),
            )
        seed_bindings[seed.seed_id] = actual
        seed_documents[seed.seed_id] = document

    for reference in registry.references:
        if reference.external_manifest_sha256 is None:
            continue
        if not all(
            value is not None
            for value in (
                reference.external_manifest_path,
                reference.external_data_path,
                reference.external_data_sha256,
            )
        ):
            raise ValueError("external reference binding is incomplete")
        manifest_path = root / str(reference.external_manifest_path)
        data_path = root / str(reference.external_data_path)
        if not manifest_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(
                f"external exact reference is incomplete: {reference.reference_id}"
            )
        manifest_sha = _file_sha256(manifest_path)
        data_sha = _file_sha256(data_path)
        if manifest_sha != reference.external_manifest_sha256:
            raise ValueError("external exact-reference manifest digest mismatch")
        if data_sha != reference.external_data_sha256:
            raise ValueError("external exact-reference data digest mismatch")
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if document.get("data_sha256") != data_sha:
            raise ValueError("external exact-reference manifest/data binding diverged")
        seed_sha = seed_bindings.get(reference.seed_id)
        if document.get("seed_sha256") != seed_sha:
            raise ValueError(
                "external exact reference does not bind the registered seed"
            )
        if document.get("observable_schema") != reference.observable_schema:
            raise ValueError("external exact-reference observable schema diverged")
        times = tuple(float(value) for value in document.get("times") or ())
        if len(times) != int(reference.grid["points"]):
            raise ValueError("external exact-reference grid length diverged")
        if _canonical_sha(times) != reference.grid["ordered_times_sha256"]:
            raise ValueError("external exact-reference ordered grid diverged")
        if reference.lane_id == LANE_SIAM:
            seed_document = seed_documents.get(reference.seed_id)
            if seed_document is None:
                raise ValueError(
                    "external SIAM reference has no verified seed contract"
                )
            _verify_siam_reference_contract(
                reference=reference,
                seed_document=seed_document,
                document=document,
            )
        reference_bindings[reference.reference_id] = manifest_sha
    return BootstrapEvidence(
        seed_artifact_sha256=seed_bindings,
        reference_artifact_sha256=reference_bindings,
        source_epochs=source_epochs,
    )


@dataclass(frozen=True)
class BootstrapAction:
    """One local or remote action contract; this class performs no execution."""

    action_id: str
    lane_id: str
    kind: str
    subject_id: str
    executor_binding: str
    prerequisites: tuple[str, ...]
    inputs: Mapping[str, Any]
    exact_reference_blind: bool = False
    qiskit_compilation_requested: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "lane_id": self.lane_id,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "executor_binding": self.executor_binding,
            "prerequisites": list(self.prerequisites),
            "inputs": dict(self.inputs),
            "exact_reference_blind": self.exact_reference_blind,
            "qiskit_compilation_requested": self.qiskit_compilation_requested,
        }


@dataclass(frozen=True)
class SourceEpochIdentity:
    """Immutable source identity for one prospective execution lane."""

    lane_id: str
    source_epoch_id: str
    source_revision: str | None = None
    source_archive_sha256: str | None = None

    def __post_init__(self) -> None:
        lane_id = str(self.lane_id).strip()
        epoch_id = str(self.source_epoch_id).strip()
        if not lane_id:
            raise ValueError("source epoch lane_id must be nonempty")
        if not epoch_id:
            raise ValueError("source epoch source_epoch_id must be nonempty")
        revision = str(self.source_revision or "").strip().lower()
        archive = str(self.source_archive_sha256 or "").strip().lower()
        if not revision and not archive:
            raise ValueError(
                "source epoch must bind a source revision or source archive digest"
            )
        if revision and (
            len(revision) not in {40, 64}
            or any(ch not in "0123456789abcdef" for ch in revision)
        ):
            raise ValueError("source epoch source_revision must be a full hex identity")
        if archive:
            _require_sha256(archive, label="source epoch archive digest")
        object.__setattr__(self, "lane_id", lane_id)
        object.__setattr__(self, "source_epoch_id", epoch_id)
        object.__setattr__(self, "source_revision", revision or None)
        object.__setattr__(self, "source_archive_sha256", archive or None)

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha(self.to_json_dict(include_digest=False))

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "lane_id": self.lane_id,
            "source_epoch_id": self.source_epoch_id,
            "source_revision": self.source_revision,
            "source_archive_sha256": self.source_archive_sha256,
        }
        if include_digest:
            payload["source_epoch_identity_sha256"] = self.identity_sha256
        return payload


@dataclass(frozen=True)
class VerifiedActionEvidence:
    """Completion evidence that binds a trajectory to its durable receipt."""

    action_id: str
    run_path: str
    trajectory_sha256: str
    completion_receipt_sha256: str

    def __post_init__(self) -> None:
        action_id = str(self.action_id).strip()
        run_path = str(self.run_path).strip()
        trajectory_sha = str(self.trajectory_sha256).strip().lower()
        receipt_sha = str(self.completion_receipt_sha256).strip().lower()
        if not action_id:
            raise ValueError("verified action evidence has no action_id")
        if not run_path:
            raise ValueError("verified action evidence has no run_path")
        _require_sha256(
            trajectory_sha,
            label=f"trajectory binding {self.action_id}",
        )
        _require_sha256(
            receipt_sha,
            label=f"completion receipt binding {self.action_id}",
        )
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "run_path", run_path)
        object.__setattr__(self, "trajectory_sha256", trajectory_sha)
        object.__setattr__(self, "completion_receipt_sha256", receipt_sha)

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha(self.to_json_dict(include_digest=False))

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "action_id": self.action_id,
            "run_path": self.run_path,
            "trajectory_sha256": self.trajectory_sha256,
            "completion_receipt_sha256": self.completion_receipt_sha256,
        }
        if include_digest:
            payload["verified_action_evidence_sha256"] = self.identity_sha256
        return payload


@dataclass(frozen=True)
class BootstrapEvidence:
    """Positive bindings already verified by their owning validators."""

    seed_artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    reference_artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    verified_action_evidence: Mapping[str, VerifiedActionEvidence] = field(
        default_factory=dict
    )
    source_epochs: Mapping[str, SourceEpochIdentity] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.seed_artifact_sha256.items():
            _require_sha256(value, label=f"seed binding {key}")
        for key, value in self.reference_artifact_sha256.items():
            _require_sha256(value, label=f"reference binding {key}")
        for key, value in self.verified_action_evidence.items():
            if not isinstance(value, VerifiedActionEvidence):
                raise TypeError(
                    "verified_action_evidence values must be VerifiedActionEvidence"
                )
            if key != value.action_id:
                raise ValueError("verified action evidence key/action_id diverged")
        for lane_id, value in self.source_epochs.items():
            if not isinstance(value, SourceEpochIdentity):
                raise TypeError("source_epochs values must be SourceEpochIdentity")
            if lane_id != value.lane_id:
                raise ValueError("source epoch mapping key/lane_id diverged")


@dataclass(frozen=True)
class BootstrapPlan:
    """Idempotent ready/waiting view over the prospective bootstrap graph."""

    registry_sha256: str
    ready_actions: tuple[BootstrapAction, ...]
    waiting_actions: tuple[Mapping[str, Any], ...]
    delegated_lanes: Mapping[str, str]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": BOOTSTRAP_PLAN_SCHEMA_V1,
            "registry_sha256": self.registry_sha256,
            "ready_actions": [row.to_json_dict() for row in self.ready_actions],
            "waiting_actions": [dict(row) for row in self.waiting_actions],
            "delegated_lanes": dict(self.delegated_lanes),
            "qiskit_compilation_requested": False,
        }


def _executor_bindings(lane_id: str) -> tuple[str, str, str, str]:
    if lane_id == LANE_HH:
        return EXEC_HH_SEED, EXEC_HH_REFERENCE, EXEC_HH_AVQDS, EXEC_HH_SCORE
    if lane_id == LANE_HUBBARD:
        return (
            EXEC_HUBBARD_SEED,
            EXEC_HUBBARD_REFERENCE,
            EXEC_HUBBARD_AVQDS,
            EXEC_HUBBARD_SCORE,
        )
    raise ValueError(f"bootstrap graph does not own lane {lane_id!r}")


def _action_id(kind: str, subject: str, inputs: Mapping[str, Any]) -> str:
    return f"{kind}::{subject}::{_canonical_sha(dict(inputs))[:16]}"


def plan_bootstrap_actions(
    registry: ProspectiveIdentityRegistry,
    *,
    evidence: BootstrapEvidence | None = None,
    available_executors: Sequence[str] = (),
) -> BootstrapPlan:
    """Return every independently ready HH/Hubbard bootstrap action.

    Executor names and source epochs are positive bindings.  Their absence is
    reported as a blocker rather than silently treating a frozen action as
    executable.  Verified SIAM artifacts are owned by its existing ladder
    planner and therefore never appear as duplicate work here.
    """

    facts = evidence or BootstrapEvidence()
    executors = frozenset(str(value) for value in available_executors)
    known_seed_ids = {row.seed_id for row in registry.seeds}
    known_reference_ids = {row.reference_id for row in registry.references}
    if not set(facts.seed_artifact_sha256).issubset(known_seed_ids):
        raise ValueError("bootstrap evidence contains an unknown seed binding")
    if not set(facts.reference_artifact_sha256).issubset(known_reference_ids):
        raise ValueError("bootstrap evidence contains an unknown reference binding")

    ready: list[BootstrapAction] = []
    waiting: list[dict[str, Any]] = []

    def consider(action: BootstrapAction, blockers: Sequence[str]) -> None:
        if action.action_id in facts.verified_action_evidence:
            return
        missing = tuple(sorted(set(str(value) for value in blockers if value)))
        if missing:
            waiting.append(
                {
                    "action_id": action.action_id,
                    "lane_id": action.lane_id,
                    "kind": action.kind,
                    "subject_id": action.subject_id,
                    "blockers": list(missing),
                }
            )
        else:
            ready.append(action)

    seed_by_id = {row.seed_id: row for row in registry.seeds}
    for seed in registry.seeds:
        if seed.lane_id == LANE_SIAM or seed.seed_id in facts.seed_artifact_sha256:
            continue
        seed_executor, _ref_executor, _run_executor, _score_executor = (
            _executor_bindings(seed.lane_id)
        )
        source_epoch = facts.source_epochs.get(seed.lane_id)
        inputs = {
            "seed_id": seed.seed_id,
            "lane_id": seed.lane_id,
            "model": seed.model,
            "physical": dict(seed.physical),
            "seed_policy": seed.policy,
            "seed_identity_sha256": seed.identity_sha256,
            "artifact_path": seed.artifact_path,
            "energy_gate": seed.energy_gate,
            "initializer": seed.initializer,
            "restarts": seed.restarts,
            "optimizer_seed": seed.optimizer_seed,
            "source_epoch": (
                source_epoch.to_json_dict() if source_epoch is not None else None
            ),
            "matrix_payload_sha256": registry.matrix_payload_sha256,
            "matrix_raw_sha256": registry.matrix_raw_sha256,
            "matrix_expanded_semantic_sha256": (
                registry.matrix_expanded_semantic_sha256
            ),
            "deterministic_scientific_execution": 1,
        }
        action = BootstrapAction(
            action_id=_action_id("build_seed", seed.seed_id, inputs),
            lane_id=seed.lane_id,
            kind="build_seed",
            subject_id=seed.seed_id,
            executor_binding=seed_executor,
            prerequisites=(),
            inputs=inputs,
        )
        consider(
            action,
            (
                "source_epoch_binding" if source_epoch is None else "",
                f"executor:{seed_executor}" if seed_executor not in executors else "",
            ),
        )

    for reference in registry.references:
        if reference.lane_id == LANE_SIAM:
            continue
        seed = seed_by_id[reference.seed_id]
        seed_sha = facts.seed_artifact_sha256.get(seed.seed_id)
        _seed_executor, ref_executor, run_executor, score_executor = (
            _executor_bindings(reference.lane_id)
        )
        source_epoch = facts.source_epochs.get(reference.lane_id)
        source_epoch_payload = (
            source_epoch.to_json_dict() if source_epoch is not None else None
        )

        if (
            seed_sha is not None
            and reference.reference_id not in facts.reference_artifact_sha256
        ):
            key = reference.materialized_key(seed_sha)
            ref_inputs = {
                "reference_id": reference.reference_id,
                "lane_id": reference.lane_id,
                "cell_id": reference.cell_id,
                "seed_id": reference.seed_id,
                "seed_identity_sha256": seed.identity_sha256,
                "reference_identity_sha256": reference.identity_sha256,
                "materialized_reference_key": key,
                "seed_artifact_sha256": seed_sha,
                "model": reference.model,
                "physical": dict(reference.physical),
                "drive": dict(reference.drive),
                "ordered_grid": dict(reference.grid),
                "propagator": reference.propagator,
                "observable_schema": reference.observable_schema,
                "exact_data_schema": PROSPECTIVE_EXACT_REFERENCE_DATA_V1,
                "artifact_path": f"{reference.artifact_root}/{key}/manifest.json",
                "source_epoch": source_epoch_payload,
                "matrix_payload_sha256": registry.matrix_payload_sha256,
                "matrix_raw_sha256": registry.matrix_raw_sha256,
                "matrix_expanded_semantic_sha256": (
                    registry.matrix_expanded_semantic_sha256
                ),
                "reporting_only": True,
            }
            ref_action = BootstrapAction(
                action_id=_action_id(
                    "build_exact_reference", reference.reference_id, ref_inputs
                ),
                lane_id=reference.lane_id,
                kind="build_exact_reference",
                subject_id=reference.reference_id,
                executor_binding=ref_executor,
                prerequisites=(seed.seed_id,),
                inputs=ref_inputs,
            )
            consider(
                ref_action,
                (
                    "source_epoch_binding" if source_epoch is None else "",
                    f"executor:{ref_executor}" if ref_executor not in executors else "",
                ),
            )

        if seed_sha is not None:
            ladder_contract = dict(
                registry.activation_ladders["hh_and_hubbard_l4"]
            )
            ladder_values = activation_cut_ladder(ladder_contract)
            ladder_identity = {
                "contract": ladder_contract,
                "ordered_values": list(ladder_values),
            }
            accuracy_policy = dict(registry.accuracy_policy)
            run_inputs = {
                "lane_id": reference.lane_id,
                "cell_id": reference.cell_id,
                "reference_id": reference.reference_id,
                "reference_identity_sha256": reference.identity_sha256,
                "seed_identity_sha256": seed.identity_sha256,
                "source_epoch_identity_sha256": (
                    None if source_epoch is None else source_epoch.identity_sha256
                ),
                "seed_artifact_sha256": seed_sha,
                "physical": dict(reference.physical),
                "drive": dict(reference.drive),
                "ordered_grid": dict(reference.grid),
                "controller": "state_motion_5e-5",
                "method": "avqds",
                "segment": "calibration_prefix_t0_t5_126",
                "activation_cut": ladder_values[0],
                "ladder_index": 0,
                "activation_ladder": ladder_identity,
                "activation_ladder_sha256": _canonical_sha(ladder_identity),
                "accuracy_policy": accuracy_policy,
                "accuracy_policy_sha256": _canonical_sha(accuracy_policy),
                "source_epoch": source_epoch_payload,
                "matrix_payload_sha256": registry.matrix_payload_sha256,
                "matrix_raw_sha256": registry.matrix_raw_sha256,
                "matrix_expanded_semantic_sha256": (
                    registry.matrix_expanded_semantic_sha256
                ),
                "record_statevector": True,
            }
            run_action = BootstrapAction(
                action_id=_action_id("screen_avqds", reference.cell_id, run_inputs),
                lane_id=reference.lane_id,
                kind="screen_avqds",
                subject_id=reference.cell_id,
                executor_binding=run_executor,
                prerequisites=(seed.seed_id,),
                inputs=run_inputs,
                exact_reference_blind=True,
            )
            consider(
                run_action,
                (
                    "source_epoch_binding" if source_epoch is None else "",
                    f"executor:{run_executor}" if run_executor not in executors else "",
                ),
            )

            reference_sha = facts.reference_artifact_sha256.get(reference.reference_id)
            if reference_sha is not None:
                screen_evidence = facts.verified_action_evidence.get(
                    run_action.action_id
                )
                score_inputs = {
                    "lane_id": reference.lane_id,
                    "cell_id": reference.cell_id,
                    "reference_id": reference.reference_id,
                    "reference_identity_sha256": reference.identity_sha256,
                    "seed_id": seed.seed_id,
                    "seed_identity_sha256": seed.identity_sha256,
                    "seed_artifact_sha256": seed_sha,
                    "source_epoch_identity_sha256": (
                        None if source_epoch is None else source_epoch.identity_sha256
                    ),
                    "reference_artifact_sha256": reference_sha,
                    "screen_action_id": run_action.action_id,
                    "screen_action_evidence": (
                        screen_evidence.to_json_dict()
                        if screen_evidence is not None
                        else None
                    ),
                    "accuracy_policy": accuracy_policy,
                    "accuracy_policy_id": str(accuracy_policy["policy_id"]),
                    "accuracy_policy_sha256": _canonical_sha(accuracy_policy),
                }
                score_action = BootstrapAction(
                    action_id=_action_id(
                        "score_avqds", reference.cell_id, score_inputs
                    ),
                    lane_id=reference.lane_id,
                    kind="score_avqds",
                    subject_id=reference.cell_id,
                    executor_binding=score_executor,
                    prerequisites=(run_action.action_id, reference.reference_id),
                    inputs=score_inputs,
                )
                consider(
                    score_action,
                    (
                        "screen_action_completion"
                        if screen_evidence is None
                        else "",
                        f"executor:{score_executor}"
                        if score_executor not in executors
                        else "",
                    ),
                )

    return BootstrapPlan(
        registry_sha256=registry.digest,
        ready_actions=tuple(sorted(ready, key=lambda row: row.action_id)),
        waiting_actions=tuple(sorted(waiting, key=lambda row: str(row["action_id"]))),
        delegated_lanes=dict(registry.delegated_planners),
    )


__all__ = [
    "BOOTSTRAP_PLAN_SCHEMA_V1",
    "BootstrapAction",
    "BootstrapEvidence",
    "BootstrapPlan",
    "DELEGATED_SIAM_PLANNER",
    "ExactReferenceIdentity",
    "IDENTITY_REGISTRY_SCHEMA_V1",
    "LANE_HH",
    "LANE_HUBBARD",
    "LANE_SIAM",
    "ProspectiveIdentityRegistry",
    "PROSPECTIVE_EXACT_REFERENCE_DATA_V1",
    "SeedIdentity",
    "SourceEpochIdentity",
    "VerifiedActionEvidence",
    "activation_cut_ladder",
    "build_identity_registry",
    "plan_bootstrap_actions",
    "verify_external_bindings",
]
