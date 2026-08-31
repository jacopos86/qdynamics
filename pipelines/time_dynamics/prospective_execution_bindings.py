"""Executable, fail-closed bridge for prospective Paper-II action contracts.

The prospective planners emit immutable scientific actions.  This module turns
their symbolic executor IDs into importable local handlers and supplies the
typed receipt boundary that connects bootstrap artifacts to the per-cell
scientific planner.  It deliberately does not submit Condor jobs: package,
transport, and workdir materialization remain explicit lane blockers until a
source epoch is frozen and audited.

Every resolver returns an execution request containing the complete immutable
input payload and declared outputs.  The request may be written to the durable
action registry, passed to a reviewed package builder, and later joined to
verified receipts.  Arbitrary digest mappings are not accepted as seed,
reference, or score evidence.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from pipelines.time_dynamics.acquisition_supervisor import (
    CampaignRegistry,
    LaneScientificAuthority,
)
from pipelines.time_dynamics.parallel_benchmark_matrix import (
    DEFAULT_MATRIX_PATH,
    ParallelBenchmarkMatrix,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    EXEC_HH_AVQDS,
    EXEC_HH_REFERENCE,
    EXEC_HH_SCORE,
    EXEC_HH_SEED,
    EXEC_HUBBARD_AVQDS,
    EXEC_HUBBARD_REFERENCE,
    EXEC_HUBBARD_SCORE,
    EXEC_HUBBARD_SEED,
    LANE_HH,
    LANE_HUBBARD,
    PROSPECTIVE_EXACT_REFERENCE_DATA_V1,
    BootstrapAction,
    ExactReferenceIdentity,
    SeedIdentity,
    SourceEpochIdentity,
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ARM_APM,
    ARM_AVQDS,
    KIND_RECORD_SAFEGUARD,
    KIND_RUN,
    KIND_SCORE,
    ProspectiveNextAction,
    ProspectiveEvidence,
    ProspectiveRunEvidence,
    ProspectiveSafeguardEvidence,
    ProspectiveScientificBindings,
    PROSPECTIVE_SCIENTIFIC_PLAN_V1,
    SAFEGUARD_LADDER_UNRESOLVED,
    SAFEGUARD_OPERATIONALLY_INFEASIBLE,
    HORIZON_FULL,
    HORIZON_PREFIX,
    FAILURE_MEMORY,
    FAILURE_OTHER,
    SCORE_COMPLETED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_IN_FLIGHT,
    STATE_SUBMITTED,
    prospective_run_action_id,
    prospective_safeguard_id,
)
from pipelines.time_dynamics.staged_campaign import AccuracyMetrics, AccuracyPolicy


EXECUTION_BINDING_REGISTRY_V1 = "paper_ii_prospective_execution_bindings_v1"
EXECUTION_REQUEST_V1 = "paper_ii_prospective_execution_request_v1"
SEED_QUALIFICATION_RECEIPT_V1 = "paper_ii_seed_qualification_receipt_v1"
EXACT_REFERENCE_RECEIPT_V1 = "paper_ii_exact_reference_receipt_v1"
RUN_LIFECYCLE_RECEIPT_V1 = "paper_ii_run_lifecycle_receipt_v1"
RUN_STATE_EVIDENCE_V1 = "paper_ii_run_state_evidence_v1"
RUN_FAILURE_EVIDENCE_V1 = "paper_ii_run_failure_evidence_v1"
SCORE_QUALIFICATION_RECEIPT_V1 = "paper_ii_score_qualification_receipt_v1"
SAFEGUARD_QUALIFICATION_RECEIPT_V1 = (
    "paper_ii_safeguard_qualification_receipt_v1"
)
SCIENTIFIC_PRE_SUBMIT_AUDIT_V1 = "paper_ii_scientific_pre_submit_audit_v1"
ACTION_REGISTRY_ENTRY_V1 = "paper_ii_prospective_action_registry_entry_v1"
SCIENTIFIC_STATE_V1 = "paper_ii_prospective_scientific_state_v1"
SOURCE_EPOCH_RECEIPT_V1 = "paper_ii_source_epoch_receipt_v1"
SOURCE_TREE_MANIFEST_V1 = "paper_ii_source_tree_manifest_v1"
SEED_ARTIFACT_V1 = "paper_ii_vqe_seed_artifact_v1"
EXACT_REFERENCE_MANIFEST_V1 = "paper_ii_exact_reference_manifest_v1"
EXACT_REFERENCE_DATA_V1 = PROSPECTIVE_EXACT_REFERENCE_DATA_V1
SCORE_METRICS_V1 = "paper_ii_exact_scored_metrics_v1"

SEED_ENERGY_GATE = 1.0e-6

EXEC_PROSPECTIVE_APM = "paper_ii.prospective.full_apm_trajectory_v1"
EXEC_PROSPECTIVE_AVQDS = "paper_ii.prospective.avqds_trajectory_v1"
EXEC_PROSPECTIVE_SCORE = "paper_ii.prospective.digest_join_score_v1"
EXEC_PROSPECTIVE_SAFEGUARD = "paper_ii.prospective.record_safeguard_v1"
EXEC_ACTION_REGISTRY = "paper_ii.prospective.action_registry_v1"
EXEC_SCIENTIFIC_STATE = "paper_ii.prospective.scientific_state_v1"
EXEC_PACKAGE_BUILDER = "paper_ii.prospective.chtc_package_builder_v1"
EXEC_REPORT_SLOT = "paper_ii.prospective.evidence_bound_report_slot_v1"

KIND_BUILD_PACKAGE = "build_package"
KIND_UPDATE_REPORT_SLOT = "update_report_slot"


class ProspectiveExecutionError(ValueError):
    """An action or receipt is not executable under the frozen contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validated_reference_response_scales(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping) or set(value) != {
        HORIZON_PREFIX,
        HORIZON_FULL,
    }:
        raise ProspectiveExecutionError(
            "exact reference must bind response scales for t5 and t20"
        )
    required = {
        "energy_rms_excursion",
        "doublon_rms_excursion",
        "max_site_occupation_rms_excursion",
    }
    result: dict[str, dict[str, float]] = {}
    for horizon in (HORIZON_PREFIX, HORIZON_FULL):
        row = value.get(horizon)
        if not isinstance(row, Mapping) or set(row) != required:
            raise ProspectiveExecutionError(
                f"exact reference {horizon} response-scale row is incomplete"
            )
        converted = {name: float(row[name]) for name in sorted(required)}
        if not all(math.isfinite(item) and item > 0.0 for item in converted.values()):
            raise ProspectiveExecutionError(
                f"exact reference {horizon} response scales must be positive"
            )
        result[horizon] = converted
    return result


def _frozen_reference_response_scales(
    value: Any,
) -> Mapping[str, Mapping[str, float]]:
    normalized = _validated_reference_response_scales(value)
    return MappingProxyType(
        {
            horizon: MappingProxyType(dict(row))
            for horizon, row in normalized.items()
        }
    )


def _prospective_exact_trajectory_scales(
    path: Path,
    *,
    expected_reference: ExactReferenceIdentity,
    materialized_reference_key: str,
) -> dict[str, dict[str, float]]:
    """Recompute t5/t20 response scales from one exact trajectory artifact."""

    payload = _read_object(path, label="prospective exact trajectory data")
    required_keys = {
        "schema",
        "reference_identity_sha256",
        "materialized_reference_key",
        "observable_schema",
        "ordered_times_sha256",
        "time",
        "energy",
        "doublon",
        "site_occupations",
        "statevector_real",
        "statevector_imag",
    }
    if set(payload) != required_keys or payload.get("schema") != EXACT_REFERENCE_DATA_V1:
        raise ProspectiveExecutionError(
            "unsupported prospective exact trajectory data contract"
        )
    if (
        payload.get("reference_identity_sha256")
        != expected_reference.identity_sha256
        or payload.get("materialized_reference_key")
        != materialized_reference_key
        or payload.get("observable_schema") != expected_reference.observable_schema
    ):
        raise ProspectiveExecutionError(
            "prospective exact trajectory data identity mismatch"
        )

    grid = expected_reference.grid
    try:
        points = int(grid["points"])
        prefix_points = int(grid["prefix_points"])
        start = float(grid["start"])
        stop = float(grid["stop"])
        prefix_stop = float(grid["prefix_stop"])
        times = tuple(float(value) for value in payload["time"])
        energy = tuple(float(value) for value in payload["energy"])
        doublon = tuple(float(value) for value in payload["doublon"])
        occupations = tuple(
            tuple(float(value) for value in row)
            for row in payload["site_occupations"]
        )
        statevector_real = tuple(
            tuple(float(value) for value in row)
            for row in payload["statevector_real"]
        )
        statevector_imag = tuple(
            tuple(float(value) for value in row)
            for row in payload["statevector_imag"]
        )
        expected_site_count = int(expected_reference.physical["lattice_sites"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "prospective exact trajectory data arrays are malformed"
        ) from exc
    if (
        points < 2
        or prefix_points < 2
        or prefix_points > points
        or len(times) != points
        or len(energy) != points
        or len(doublon) != points
        or len(occupations) != points
        or len(statevector_real) != points
        or len(statevector_imag) != points
        or not occupations
        or not occupations[0]
        or len(occupations[0]) != expected_site_count
        or any(len(row) != len(occupations[0]) for row in occupations)
        or not statevector_real
        or not statevector_real[0]
        or any(len(row) != len(statevector_real[0]) for row in statevector_real)
        or any(len(row) != len(statevector_real[0]) for row in statevector_imag)
        or not all(math.isfinite(value) for value in times)
        or not all(math.isfinite(value) for value in energy)
        or not all(math.isfinite(value) for value in doublon)
        or not all(math.isfinite(value) for row in occupations for value in row)
        or not all(math.isfinite(value) for row in statevector_real for value in row)
        or not all(math.isfinite(value) for row in statevector_imag for value in row)
    ):
        raise ProspectiveExecutionError(
            "prospective exact trajectory data shapes or values are invalid"
        )
    if any(
        not math.isclose(
            math.fsum(real * real + imag * imag for real, imag in zip(real_row, imag_row)),
            1.0,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
        for real_row, imag_row in zip(statevector_real, statevector_imag)
    ):
        raise ProspectiveExecutionError(
            "prospective exact trajectory statevectors are not normalized"
        )
    if (
        not math.isclose(times[0], start, rel_tol=0.0, abs_tol=1.0e-15)
        or not math.isclose(times[-1], stop, rel_tol=0.0, abs_tol=1.0e-14)
        or not math.isclose(
            times[prefix_points - 1],
            prefix_stop,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        or payload.get("ordered_times_sha256")
        != grid.get("ordered_times_sha256")
        or _canonical_sha(times) != grid.get("ordered_times_sha256")
        or _canonical_sha(times[:prefix_points])
        != grid.get("ordered_prefix_sha256")
    ):
        raise ProspectiveExecutionError(
            "prospective exact trajectory data disagrees with the expected grid"
        )

    def _rms(values: Sequence[float], count: int) -> float:
        baseline = float(values[0])
        return math.sqrt(
            math.fsum((float(value) - baseline) ** 2 for value in values[:count])
            / float(count)
        )

    def _site_rms(count: int) -> float:
        baseline = occupations[0]
        excursions = (
            max(abs(row[index] - baseline[index]) for index in range(len(baseline)))
            for row in occupations[:count]
        )
        return math.sqrt(
            math.fsum(value * value for value in excursions) / float(count)
        )

    result = {
        horizon: {
            "energy_rms_excursion": _rms(energy, count),
            "doublon_rms_excursion": _rms(doublon, count),
            "max_site_occupation_rms_excursion": _site_rms(count),
        }
        for horizon, count in (
            (HORIZON_PREFIX, prefix_points),
            (HORIZON_FULL, points),
        )
    }
    return _validated_reference_response_scales(result)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, *, label: str) -> str:
    token = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ProspectiveExecutionError(f"{label} must be a lowercase SHA-256")
    return token


def _token(value: Any, *, label: str) -> str:
    token = str(value or "").strip()
    if not token:
        raise ProspectiveExecutionError(f"{label} must be nonempty")
    return token


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ProspectiveExecutionError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ProspectiveExecutionError(f"{label} must be a JSON object")
    return payload


def _read_required_mapping(
    payload: Mapping[str, Any], key: str, *, label: str
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProspectiveExecutionError(f"{label} has no {key} mapping")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class VerifiedFile:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        path = Path(_token(self.path, label="artifact path"))
        if path.is_absolute() or ".." in path.parts:
            raise ProspectiveExecutionError(
                "artifact paths must be repository-relative and canonical"
            )
        object.__setattr__(self, "path", path.as_posix())
        object.__setattr__(self, "sha256", _sha(self.sha256, label="artifact digest"))

    def verify(self, root: Path) -> None:
        target = root / self.path
        if not target.is_file() or _file_sha(target) != self.sha256:
            raise ProspectiveExecutionError(f"artifact digest mismatch: {self.path}")

    def to_json_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


def _verify_self_digest(
    payload: Mapping[str, Any], *, key: str, label: str
) -> str:
    unsigned = dict(payload)
    declared = unsigned.pop(key, None)
    actual = _canonical_sha(unsigned)
    if declared != actual:
        raise ProspectiveExecutionError(f"{label} self digest mismatch")
    return actual


@dataclass(frozen=True)
class SourceEpochReceipt:
    lane_id: str
    source_epoch: SourceEpochIdentity
    source_tree_manifest: VerifiedFile
    source_tree_sha256: str
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        expected_epoch: SourceEpochIdentity | None = None,
    ) -> "SourceEpochReceipt":
        root = Path(repository_root).resolve()
        source = Path(path)
        if not source.is_absolute():
            source = root / source
        source = source.resolve(strict=True)
        payload = _read_object(source, label="source epoch receipt")
        if payload.get("schema") != SOURCE_EPOCH_RECEIPT_V1:
            raise ProspectiveExecutionError("unsupported source epoch receipt")
        self_sha = _verify_self_digest(
            payload, key="source_epoch_receipt_sha256", label="source epoch receipt"
        )
        epoch_row = payload.get("source_epoch")
        manifest_row = payload.get("source_tree_manifest")
        if not isinstance(epoch_row, Mapping) or not isinstance(manifest_row, Mapping):
            raise ProspectiveExecutionError("source epoch receipt is incomplete")
        epoch = SourceEpochIdentity(
            lane_id=_token(epoch_row.get("lane_id"), label="source epoch lane"),
            source_epoch_id=_token(
                epoch_row.get("source_epoch_id"), label="source epoch id"
            ),
            source_revision=epoch_row.get("source_revision"),
            source_archive_sha256=epoch_row.get("source_archive_sha256"),
        )
        if epoch_row.get("source_epoch_identity_sha256") != epoch.identity_sha256:
            raise ProspectiveExecutionError("source epoch identity digest mismatch")
        if expected_epoch is not None and epoch != expected_epoch:
            raise ProspectiveExecutionError("source epoch receipt identity diverged")
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProspectiveExecutionError(
                "source epoch receipt is outside repository"
            ) from exc
        result = cls(
            lane_id=epoch.lane_id,
            source_epoch=epoch,
            source_tree_manifest=VerifiedFile(
                str(manifest_row.get("path") or ""),
                str(manifest_row.get("sha256") or ""),
            ),
            source_tree_sha256=_sha(
                payload.get("source_tree_sha256"), label="source tree"
            ),
            receipt_path=relative,
            receipt_sha256=_file_sha(source),
            receipt_self_sha256=self_sha,
        )
        result.source_tree_manifest.verify(root)
        tree = _read_object(
            root / result.source_tree_manifest.path, label="source tree manifest"
        )
        if tree.get("schema") != SOURCE_TREE_MANIFEST_V1:
            raise ProspectiveExecutionError("unsupported source tree manifest")
        _verify_self_digest(
            tree,
            key="source_tree_manifest_sha256",
            label="source tree manifest",
        )
        if (
            tree.get("lane_id") != epoch.lane_id
            or tree.get("source_epoch_identity_sha256") != epoch.identity_sha256
            or tree.get("source_revision") != epoch.source_revision
            or tree.get("source_archive_sha256") != epoch.source_archive_sha256
            or tree.get("source_tree_sha256") != result.source_tree_sha256
            or tree.get("provenance_kind")
            not in {"git_archive", "verified_source_archive"}
        ):
            raise ProspectiveExecutionError("source tree provenance binding mismatch")
        return result


@dataclass(frozen=True)
class SeedQualificationReceipt:
    seed_id: str
    seed_identity_sha256: str
    artifact: VerifiedFile
    energy_error: float
    exact_state_infidelity: float
    initializer: str
    restarts: int
    optimizer_seed: int
    optimizer_execution_id: str
    variational_energy: float
    exact_energy: float
    source_epoch_identity_sha256: str
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        expected_seed: SeedIdentity,
        expected_source_epoch: SourceEpochIdentity,
    ) -> "SeedQualificationReceipt":
        root = Path(repository_root).resolve()
        source = Path(path)
        if not source.is_absolute():
            source = root / source
        source = source.resolve(strict=True)
        payload = _read_object(source, label="seed qualification receipt")
        if payload.get("schema") != SEED_QUALIFICATION_RECEIPT_V1:
            raise ProspectiveExecutionError("unsupported seed receipt schema")
        self_sha = _verify_self_digest(
            payload, key="seed_receipt_sha256", label="seed receipt"
        )
        artifact = payload.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ProspectiveExecutionError("seed receipt has no artifact binding")
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProspectiveExecutionError("seed receipt is outside repository") from exc
        result = cls(
            seed_id=_token(payload.get("seed_id"), label="seed_id"),
            seed_identity_sha256=_sha(
                payload.get("seed_identity_sha256"), label="seed identity"
            ),
            artifact=VerifiedFile(
                path=str(artifact.get("path") or ""),
                sha256=str(artifact.get("sha256") or ""),
            ),
            energy_error=float(payload.get("energy_error")),
            exact_state_infidelity=float(payload.get("exact_state_infidelity")),
            initializer=_token(payload.get("initializer"), label="initializer"),
            restarts=int(payload.get("restarts")),
            optimizer_seed=int(payload.get("optimizer_seed")),
            optimizer_execution_id=_token(
                payload.get("optimizer_execution_id"),
                label="optimizer execution id",
            ),
            variational_energy=float(payload.get("variational_energy")),
            exact_energy=float(payload.get("exact_energy")),
            source_epoch_identity_sha256=_sha(
                payload.get("source_epoch_identity_sha256"),
                label="source epoch identity",
            ),
            receipt_path=relative,
            receipt_sha256=_file_sha(source),
            receipt_self_sha256=self_sha,
        )
        result.validate(
            root,
            expected_seed=expected_seed,
            expected_source_epoch=expected_source_epoch,
        )
        return result

    def validate(
        self,
        root: Path,
        *,
        expected_seed: SeedIdentity,
        expected_source_epoch: SourceEpochIdentity,
    ) -> None:
        if (
            self.seed_id != expected_seed.seed_id
            or self.seed_identity_sha256 != expected_seed.identity_sha256
            or self.initializer != expected_seed.initializer
            or self.restarts != expected_seed.restarts
            or self.optimizer_seed != expected_seed.optimizer_seed
            or self.source_epoch_identity_sha256
            != expected_source_epoch.identity_sha256
            or expected_seed.lane_id != expected_source_epoch.lane_id
        ):
            raise ProspectiveExecutionError("seed qualification identity diverged")
        if not math.isfinite(self.energy_error) or self.energy_error < 0.0:
            raise ProspectiveExecutionError("seed energy error must be nonnegative")
        if self.energy_error > SEED_ENERGY_GATE:
            raise ProspectiveExecutionError("seed fails the hard Delta E <= 1e-6 gate")
        if not math.isfinite(self.exact_state_infidelity) or not (
            0.0 <= self.exact_state_infidelity <= 1.0
        ):
            raise ProspectiveExecutionError("seed exact-state infidelity is invalid")
        if self.restarts < 1:
            raise ProspectiveExecutionError("seed receipt must record VQE restarts")
        self.artifact.verify(root)
        artifact = _read_object(root / self.artifact.path, label="seed artifact")
        if artifact.get("schema") != SEED_ARTIFACT_V1:
            raise ProspectiveExecutionError("unsupported seed artifact schema")
        _verify_self_digest(
            artifact, key="seed_artifact_sha256", label="seed artifact"
        )
        if (
            artifact.get("seed_id") != self.seed_id
            or artifact.get("seed_identity_sha256") != self.seed_identity_sha256
            or artifact.get("lane_id") != expected_seed.lane_id
            or artifact.get("model") != expected_seed.model
            or artifact.get("physical") != dict(expected_seed.physical)
            or artifact.get("seed_policy") != expected_seed.policy
            or artifact.get("initializer") != self.initializer
            or int(artifact.get("restarts", -1)) != self.restarts
            or int(artifact.get("optimizer_seed", -1)) != self.optimizer_seed
            or artifact.get("optimizer_execution_id")
            != self.optimizer_execution_id
            or artifact.get("source_epoch_identity_sha256")
            != self.source_epoch_identity_sha256
            or not math.isclose(
                float(artifact.get("exact_state_infidelity")),
                self.exact_state_infidelity,
                rel_tol=0.0,
                abs_tol=1e-14,
            )
        ):
            raise ProspectiveExecutionError("seed artifact contract mismatch")
        variational = float(artifact.get("variational_energy"))
        exact = float(artifact.get("exact_energy"))
        recomputed = abs(variational - exact)
        if (
            not all(math.isfinite(value) for value in (variational, exact))
            or not math.isclose(
                variational, self.variational_energy, rel_tol=0.0, abs_tol=1e-14
            )
            or not math.isclose(exact, self.exact_energy, rel_tol=0.0, abs_tol=1e-14)
            or not math.isclose(
                recomputed, self.energy_error, rel_tol=1e-12, abs_tol=1e-14
            )
        ):
            raise ProspectiveExecutionError("seed energy claim is not reproducible")


def validate_seed_receipt_collection(
    receipts: Sequence[SeedQualificationReceipt],
    *,
    expected_seeds: Sequence[SeedIdentity],
) -> None:
    expected = {row.seed_id: row for row in expected_seeds}
    if len(expected) != len(expected_seeds) or len(receipts) != len(expected):
        raise ProspectiveExecutionError("seed receipt collection is incomplete")
    artifact_digests: set[str] = set()
    optimizer_executions: set[str] = set()
    for receipt in receipts:
        seed = expected.get(receipt.seed_id)
        if seed is None or receipt.seed_identity_sha256 != seed.identity_sha256:
            raise ProspectiveExecutionError("seed receipt collection identity mismatch")
        if receipt.initializer != seed.initializer:
            raise ProspectiveExecutionError("seed initializer identity mismatch")
        if receipt.artifact.sha256 in artifact_digests:
            raise ProspectiveExecutionError("independent seeds reuse one artifact")
        if receipt.optimizer_execution_id in optimizer_executions:
            raise ProspectiveExecutionError(
                "independent seeds reuse one optimizer execution"
            )
        artifact_digests.add(receipt.artifact.sha256)
        optimizer_executions.add(receipt.optimizer_execution_id)


@dataclass(frozen=True)
class ExactReferenceReceipt:
    reference_id: str
    reference_identity_sha256: str
    lane_id: str
    cell_id: str
    seed_id: str
    seed_artifact_sha256: str
    manifest: VerifiedFile
    data: VerifiedFile
    source_epoch_identity_sha256: str
    materialized_reference_key: str
    reference_contract: Mapping[str, Any]
    response_scales: Mapping[str, Mapping[str, float]]
    response_scales_sha256: str
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        expected_reference: ExactReferenceIdentity,
        expected_seed_identity: SeedIdentity,
        expected_seed_receipt: SeedQualificationReceipt,
        expected_source_epoch: SourceEpochIdentity,
    ) -> "ExactReferenceReceipt":
        root = Path(repository_root).resolve()
        source = Path(path)
        if not source.is_absolute():
            source = root / source
        source = source.resolve(strict=True)
        payload = _read_object(source, label="exact-reference receipt")
        if payload.get("schema") != EXACT_REFERENCE_RECEIPT_V1:
            raise ProspectiveExecutionError("unsupported exact-reference receipt")
        self_sha = _verify_self_digest(
            payload, key="exact_reference_receipt_sha256",
            label="exact-reference receipt",
        )
        manifest = payload.get("manifest")
        data = payload.get("data")
        if not isinstance(manifest, Mapping) or not isinstance(data, Mapping):
            raise ProspectiveExecutionError("exact receipt needs manifest and data")
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProspectiveExecutionError("exact receipt is outside repository") from exc
        response_scales = _validated_reference_response_scales(
            payload.get("response_scales")
        )
        response_scales_sha256 = _sha(
            payload.get("response_scales_sha256"),
            label="exact response scales",
        )
        if _canonical_sha(response_scales) != response_scales_sha256:
            raise ProspectiveExecutionError(
                "exact-reference receipt response-scale digest mismatch"
            )
        result = cls(
            reference_id=_token(payload.get("reference_id"), label="reference_id"),
            reference_identity_sha256=_sha(
                payload.get("reference_identity_sha256"),
                label="reference identity",
            ),
            lane_id=_token(payload.get("lane_id"), label="lane_id"),
            cell_id=_token(payload.get("cell_id"), label="cell_id"),
            seed_id=_token(payload.get("seed_id"), label="seed_id"),
            seed_artifact_sha256=_sha(
                payload.get("seed_artifact_sha256"), label="seed artifact"
            ),
            manifest=VerifiedFile(
                str(manifest.get("path") or ""), str(manifest.get("sha256") or "")
            ),
            data=VerifiedFile(
                str(data.get("path") or ""), str(data.get("sha256") or "")
            ),
            source_epoch_identity_sha256=_sha(
                payload.get("source_epoch_identity_sha256"),
                label="source epoch identity",
            ),
            materialized_reference_key=_sha(
                payload.get("materialized_reference_key"),
                label="materialized reference key",
            ),
            reference_contract=dict(payload.get("reference_contract") or {}),
            response_scales=_frozen_reference_response_scales(response_scales),
            response_scales_sha256=response_scales_sha256,
            receipt_path=relative,
            receipt_sha256=_file_sha(source),
            receipt_self_sha256=self_sha,
        )
        reloaded_seed = SeedQualificationReceipt.load(
            root / expected_seed_receipt.receipt_path,
            repository_root=root,
            expected_seed=expected_seed_identity,
            expected_source_epoch=expected_source_epoch,
        )
        if reloaded_seed.receipt_sha256 != expected_seed_receipt.receipt_sha256:
            raise ProspectiveExecutionError("seed receipt changed after validation")
        result.manifest.verify(root)
        result.data.verify(root)
        manifest_payload = _read_object(root / result.manifest.path, label="exact manifest")
        if manifest_payload.get("schema") != EXACT_REFERENCE_MANIFEST_V1:
            raise ProspectiveExecutionError("unsupported exact-reference manifest")
        _verify_self_digest(
            manifest_payload, key="exact_manifest_sha256",
            label="exact-reference manifest",
        )
        manifest_response_scales = _validated_reference_response_scales(
            manifest_payload.get("response_scales")
        )
        manifest_response_scales_sha256 = _sha(
            manifest_payload.get("response_scales_sha256"),
            label="exact manifest response scales",
        )
        expected_contract = {
            "reference_id": expected_reference.reference_id,
            "reference_identity_sha256": expected_reference.identity_sha256,
            "lane_id": expected_reference.lane_id,
            "cell_id": expected_reference.cell_id,
            "seed_id": expected_reference.seed_id,
            "seed_identity_sha256": expected_seed_identity.identity_sha256,
            "model": expected_reference.model,
            "physical": dict(expected_reference.physical),
            "drive": dict(expected_reference.drive),
            "grid": dict(expected_reference.grid),
            "propagator": expected_reference.propagator,
            "observable_schema": expected_reference.observable_schema,
        }
        materialized = expected_reference.materialized_key(
            expected_seed_receipt.artifact.sha256
        )
        if (
            result.reference_id != expected_reference.reference_id
            or result.reference_identity_sha256 != expected_reference.identity_sha256
            or result.lane_id != expected_reference.lane_id
            or result.cell_id != expected_reference.cell_id
            or result.seed_id != expected_seed_identity.seed_id
            or result.seed_artifact_sha256 != expected_seed_receipt.artifact.sha256
            or result.source_epoch_identity_sha256
            != expected_source_epoch.identity_sha256
            or result.materialized_reference_key != materialized
            or dict(result.reference_contract) != expected_contract
            or manifest_payload.get("reference_contract") != expected_contract
            or manifest_payload.get("materialized_reference_key") != materialized
            or manifest_payload.get("source_epoch_identity_sha256")
            != expected_source_epoch.identity_sha256
            or manifest_payload.get("seed_artifact_sha256")
            != expected_seed_receipt.artifact.sha256
            or manifest_payload.get("data_sha256") != result.data.sha256
            or manifest_response_scales != dict(result.response_scales)
            or manifest_response_scales_sha256 != result.response_scales_sha256
            or _canonical_sha(manifest_response_scales)
            != manifest_response_scales_sha256
        ):
            raise ProspectiveExecutionError(
                "exact-reference registered identity binding mismatch"
            )
        recomputed_response_scales = _prospective_exact_trajectory_scales(
            root / result.data.path,
            expected_reference=expected_reference,
            materialized_reference_key=materialized,
        )
        claimed_response_scales = _validated_reference_response_scales(
            result.response_scales
        )
        if any(
            not math.isclose(
                claimed_response_scales[horizon][name],
                recomputed_response_scales[horizon][name],
                rel_tol=1.0e-13,
                abs_tol=1.0e-15,
            )
            for horizon in (HORIZON_PREFIX, HORIZON_FULL)
            for name in claimed_response_scales[horizon]
        ):
            raise ProspectiveExecutionError(
                "response scales disagree with the exact trajectory data"
            )
        return result

    def policy_response_scales(self, drive_id: str) -> dict[str, Any]:
        """Return this reference's scales in ``AccuracyPolicy`` index form."""

        drive = _token(drive_id, label="drive_id")
        normalized = _validated_reference_response_scales(self.response_scales)
        if _canonical_sha(normalized) != self.response_scales_sha256:
            raise ProspectiveExecutionError(
                "exact-reference response scales changed after validation"
            )
        return {
            drive: {
                horizon: dict(row)
                for horizon, row in normalized.items()
            }
        }


def _verified_binding(
    payload: Mapping[str, Any], name: str, *, label: str
) -> VerifiedFile:
    row = payload.get(name)
    if not isinstance(row, Mapping):
        raise ProspectiveExecutionError(f"{label} has no {name} binding")
    return VerifiedFile(str(row.get("path") or ""), str(row.get("sha256") or ""))


def _load_action_registry_envelope(
    file: VerifiedFile, *, root: Path
) -> tuple["ExecutionRequest", Mapping[str, Any]]:
    """Authenticate the registry envelope without reconstructing matrix identities.

    The stronger :func:`load_action_registry_entry` join is still used at the
    campaign boundary.  This narrower loader lets run and score receipts prove
    that they name the exact immutable request and exact source-receipt files
    even when only the per-cell scientific binding is available.
    """

    file.verify(root)
    payload = _read_object(root / file.path, label="action registry entry")
    if payload.get("schema") != ACTION_REGISTRY_ENTRY_V1:
        raise ProspectiveExecutionError("unsupported action registry entry")
    _verify_self_digest(payload, key="entry_sha256", label="action registry entry")
    request_row = payload.get("execution_request")
    receipt_rows = payload.get("source_receipts")
    if not isinstance(request_row, Mapping) or not isinstance(receipt_rows, list):
        raise ProspectiveExecutionError("action registry entry is incomplete")
    request = ExecutionRequest.from_mapping(request_row)
    receipts = tuple(
        VerifiedFile(str(row.get("path") or ""), str(row.get("sha256") or ""))
        for row in receipt_rows
        if isinstance(row, Mapping)
    )
    if len(receipts) != len(receipt_rows):
        raise ProspectiveExecutionError("action registry receipt binding malformed")
    self_keys = {
        SOURCE_EPOCH_RECEIPT_V1: "source_epoch_receipt_sha256",
        SEED_QUALIFICATION_RECEIPT_V1: "seed_receipt_sha256",
        EXACT_REFERENCE_RECEIPT_V1: "exact_reference_receipt_sha256",
    }
    by_schema: dict[str, VerifiedFile] = {}
    for receipt in receipts:
        receipt.verify(root)
        receipt_payload = _read_object(
            root / receipt.path, label="action source receipt"
        )
        schema = str(receipt_payload.get("schema") or "")
        if schema not in self_keys or schema in by_schema:
            raise ProspectiveExecutionError(
                "action registry requires unique typed source receipts"
            )
        _verify_self_digest(
            receipt_payload,
            key=self_keys[schema],
            label="action source receipt",
        )
        by_schema[schema] = receipt
    if set(by_schema) != set(self_keys):
        raise ProspectiveExecutionError(
            "action registry requires exact source/seed/reference receipts"
        )
    if (
        payload.get("action_id") != request.action_id
        or payload.get("lane_id") != request.lane_id
        or payload.get("kind") != request.kind
        or payload.get("bindings_sha256") != request.inputs.get("bindings_sha256")
        or payload.get("execution_request_sha256") != request.digest
        or payload.get("source_receipt_sha256")
        != by_schema[SOURCE_EPOCH_RECEIPT_V1].sha256
        or payload.get("seed_receipt_sha256")
        != by_schema[SEED_QUALIFICATION_RECEIPT_V1].sha256
        or payload.get("exact_reference_receipt_sha256")
        != by_schema[EXACT_REFERENCE_RECEIPT_V1].sha256
    ):
        raise ProspectiveExecutionError(
            "action registry envelope does not bind its execution request"
        )
    return request, payload


@dataclass(frozen=True)
class RunLifecycleReceipt:
    """One immutable scheduler/run state transition for a scientific attempt."""

    scientific_action_id: str
    execution_id: str
    cell_id: str
    method: str
    horizon: str
    activation_cut: float
    attempt: int
    planner_sequence: int
    requested_memory_gib: int
    state: str
    bindings_sha256: str
    execution_request_sha256: str
    action_registry: VerifiedFile
    source_epoch_identity_sha256: str
    seed_receipt_sha256: str
    exact_reference_receipt_sha256: str
    accuracy_policy_sha256: str
    resume_from_action_id: str | None
    resume_run_receipt_sha256: str | None
    resume_run_sha256: str | None
    previous_receipt: VerifiedFile | None
    state_evidence: VerifiedFile
    run: VerifiedFile | None
    failure_kind: str | None
    failure_reason: str
    failure_evidence: VerifiedFile | None
    observed_peak_gib: float | None
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        _seen: frozenset[str] = frozenset(),
    ) -> "RunLifecycleReceipt":
        root = Path(repository_root).resolve()
        source = Path(path)
        if not source.is_absolute():
            source = root / source
        source = source.resolve(strict=True)
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProspectiveExecutionError(
                "run lifecycle receipt is outside repository"
            ) from exc
        if relative in _seen:
            raise ProspectiveExecutionError("run lifecycle receipt chain is cyclic")
        payload = _read_object(source, label="run lifecycle receipt")
        if payload.get("schema") != RUN_LIFECYCLE_RECEIPT_V1:
            raise ProspectiveExecutionError("unsupported run lifecycle receipt")
        self_sha = _verify_self_digest(
            payload,
            key="run_lifecycle_receipt_sha256",
            label="run lifecycle receipt",
        )

        def optional_binding(name: str) -> VerifiedFile | None:
            row = payload.get(name)
            if row is None:
                return None
            if not isinstance(row, Mapping):
                raise ProspectiveExecutionError(
                    f"run lifecycle receipt has malformed {name} binding"
                )
            return VerifiedFile(
                str(row.get("path") or ""), str(row.get("sha256") or "")
            )

        try:
            result = cls(
                scientific_action_id=_token(
                    payload.get("scientific_action_id"), label="scientific action"
                ),
                execution_id=_token(
                    payload.get("execution_id"), label="execution id"
                ),
                cell_id=_token(payload.get("cell_id"), label="cell_id"),
                method=_token(payload.get("method"), label="method"),
                horizon=_token(payload.get("horizon"), label="horizon"),
                activation_cut=float(payload.get("activation_cut")),
                attempt=int(payload.get("attempt")),
                planner_sequence=int(payload.get("planner_sequence")),
                requested_memory_gib=int(payload.get("requested_memory_gib")),
                state=_token(payload.get("state"), label="run state"),
                bindings_sha256=_sha(
                    payload.get("bindings_sha256"), label="scientific bindings"
                ),
                execution_request_sha256=_sha(
                    payload.get("execution_request_sha256"),
                    label="execution request",
                ),
                action_registry=_verified_binding(
                    payload, "action_registry", label="run lifecycle receipt"
                ),
                source_epoch_identity_sha256=_sha(
                    payload.get("source_epoch_identity_sha256"),
                    label="source epoch identity",
                ),
                seed_receipt_sha256=_sha(
                    payload.get("seed_receipt_sha256"), label="seed receipt"
                ),
                exact_reference_receipt_sha256=_sha(
                    payload.get("exact_reference_receipt_sha256"),
                    label="exact-reference receipt",
                ),
                accuracy_policy_sha256=_sha(
                    payload.get("accuracy_policy_sha256"),
                    label="accuracy policy",
                ),
                resume_from_action_id=(
                    None
                    if payload.get("resume_from_action_id") is None
                    else _token(
                        payload.get("resume_from_action_id"),
                        label="resume action",
                    )
                ),
                resume_run_receipt_sha256=(
                    None
                    if payload.get("resume_run_receipt_sha256") is None
                    else _sha(
                        payload.get("resume_run_receipt_sha256"),
                        label="resume run receipt",
                    )
                ),
                resume_run_sha256=(
                    None
                    if payload.get("resume_run_sha256") is None
                    else _sha(payload.get("resume_run_sha256"), label="resume run")
                ),
                previous_receipt=optional_binding("previous_receipt"),
                state_evidence=_verified_binding(
                    payload, "state_evidence", label="run lifecycle receipt"
                ),
                run=optional_binding("run"),
                failure_kind=(
                    None
                    if payload.get("failure_kind") is None
                    else _token(payload.get("failure_kind"), label="failure kind")
                ),
                failure_reason=str(payload.get("failure_reason") or "").strip(),
                failure_evidence=optional_binding("failure_evidence"),
                observed_peak_gib=(
                    None
                    if payload.get("observed_peak_gib") is None
                    else float(payload.get("observed_peak_gib"))
                ),
                receipt_path=relative,
                receipt_sha256=_file_sha(source),
                receipt_self_sha256=self_sha,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ProspectiveExecutionError):
                raise
            raise ProspectiveExecutionError(
                "run lifecycle receipt has invalid scalar fields"
            ) from exc
        result._verify(root, seen=_seen | {relative})
        return result

    def _verify(self, root: Path, *, seen: frozenset[str]) -> None:
        if self.method not in {ARM_APM, ARM_AVQDS}:
            raise ProspectiveExecutionError(
                "run lifecycle receipt names an unsupported method"
            )
        if self.horizon not in {HORIZON_PREFIX, HORIZON_FULL}:
            raise ProspectiveExecutionError(
                "run lifecycle receipt names an unsupported horizon"
            )
        if self.state not in {
            STATE_SUBMITTED,
            STATE_IN_FLIGHT,
            STATE_COMPLETED,
            STATE_FAILED,
        }:
            raise ProspectiveExecutionError("unsupported run lifecycle state")
        if self.attempt not in (0, 1) or self.planner_sequence < 0:
            raise ProspectiveExecutionError("invalid run lifecycle attempt identity")
        if (
            self.requested_memory_gib < 4
            or self.requested_memory_gib > 32
            or self.requested_memory_gib % 4
        ):
            raise ProspectiveExecutionError("invalid run lifecycle memory tier")

        request, registry = _load_action_registry_envelope(
            self.action_registry, root=root
        )
        inputs = request.inputs
        try:
            request_cut = float(inputs.get("activation_cut"))
        except (TypeError, ValueError) as exc:
            raise ProspectiveExecutionError(
                "registered run request has an invalid activation cut"
            ) from exc
        if (
            request.kind != KIND_RUN
            or request.action_id != self.scientific_action_id
            or request.execution_id != self.execution_id
            or request.arm != self.method
            or request.attempt != self.attempt
            or request.planner_sequence != self.planner_sequence
            or request.requested_memory_gib != self.requested_memory_gib
            or request.digest != self.execution_request_sha256
            or registry.get("bindings_sha256") != self.bindings_sha256
            or inputs.get("bindings_sha256") != self.bindings_sha256
            or inputs.get("cell_id") != self.cell_id
            or inputs.get("arm") != self.method
            or inputs.get("horizon") != self.horizon
            or not math.isclose(
                request_cut,
                self.activation_cut,
                rel_tol=5e-12,
                abs_tol=0.0,
            )
            or inputs.get("source_epoch_identity_sha256")
            != self.source_epoch_identity_sha256
            or inputs.get("seed_receipt_sha256") != self.seed_receipt_sha256
            or inputs.get("reference_receipt_sha256")
            != self.exact_reference_receipt_sha256
            or inputs.get("accuracy_policy_sha256")
            != self.accuracy_policy_sha256
            or registry.get("seed_receipt_sha256") != self.seed_receipt_sha256
            or registry.get("exact_reference_receipt_sha256")
            != self.exact_reference_receipt_sha256
        ):
            raise ProspectiveExecutionError(
                "run lifecycle receipt diverges from its registered request"
            )
        request_resume = (
            inputs.get("resume_from_action_id"),
            inputs.get("resume_run_receipt_sha256"),
            inputs.get("resume_run_sha256"),
        )
        receipt_resume = (
            self.resume_from_action_id,
            self.resume_run_receipt_sha256,
            self.resume_run_sha256,
        )
        if request_resume != receipt_resume:
            raise ProspectiveExecutionError(
                "run lifecycle continuation provenance diverges from request"
            )
        if self.horizon == HORIZON_PREFIX and any(
            value is not None for value in receipt_resume
        ):
            raise ProspectiveExecutionError(
                "prefix lifecycle receipt cannot carry resume provenance"
            )
        if self.horizon == HORIZON_FULL and any(
            value is None for value in receipt_resume
        ):
            raise ProspectiveExecutionError(
                "full lifecycle receipt requires resume provenance"
            )

        self.state_evidence.verify(root)
        state_payload = _read_object(
            root / self.state_evidence.path, label="run state evidence"
        )
        if state_payload.get("schema") != RUN_STATE_EVIDENCE_V1:
            raise ProspectiveExecutionError("unsupported run state evidence")
        _verify_self_digest(
            state_payload,
            key="run_state_evidence_sha256",
            label="run state evidence",
        )
        if (
            state_payload.get("scientific_action_id")
            != self.scientific_action_id
            or state_payload.get("execution_id") != self.execution_id
            or state_payload.get("state") != self.state
        ):
            raise ProspectiveExecutionError(
                "run state evidence diverges from lifecycle receipt"
            )

        if self.state == STATE_COMPLETED:
            if self.run is None or any(
                value is not None
                for value in (
                    self.failure_kind,
                    self.failure_evidence,
                    self.observed_peak_gib,
                )
            ) or self.failure_reason:
                raise ProspectiveExecutionError(
                    "completed lifecycle receipt has invalid terminal evidence"
                )
            self.run.verify(root)
        elif self.state == STATE_FAILED:
            if (
                self.run is not None
                or self.failure_kind not in {FAILURE_MEMORY, FAILURE_OTHER}
                or not self.failure_reason
                or self.failure_evidence is None
            ):
                raise ProspectiveExecutionError(
                    "failed lifecycle receipt has invalid terminal evidence"
                )
            if self.failure_kind == FAILURE_MEMORY and (
                self.observed_peak_gib is None
                or not math.isfinite(self.observed_peak_gib)
                or self.observed_peak_gib <= 0.0
            ):
                raise ProspectiveExecutionError(
                    "memory failure requires a positive observed peak"
                )
            if self.observed_peak_gib is not None and (
                not math.isfinite(self.observed_peak_gib)
                or self.observed_peak_gib <= 0.0
            ):
                raise ProspectiveExecutionError(
                    "failed lifecycle receipt has an invalid observed peak"
                )
            self.failure_evidence.verify(root)
            failure = _read_object(
                root / self.failure_evidence.path, label="run failure evidence"
            )
            if failure.get("schema") != RUN_FAILURE_EVIDENCE_V1:
                raise ProspectiveExecutionError("unsupported run failure evidence")
            _verify_self_digest(
                failure,
                key="run_failure_evidence_sha256",
                label="run failure evidence",
            )
            if (
                failure.get("scientific_action_id")
                != self.scientific_action_id
                or failure.get("execution_id") != self.execution_id
                or failure.get("failure_kind") != self.failure_kind
                or str(failure.get("failure_reason") or "").strip()
                != self.failure_reason
                or failure.get("observed_peak_gib") != self.observed_peak_gib
            ):
                raise ProspectiveExecutionError(
                    "run failure evidence diverges from lifecycle receipt"
                )
        elif any(
            value is not None
            for value in (
                self.run,
                self.failure_kind,
                self.failure_evidence,
                self.observed_peak_gib,
            )
        ) or self.failure_reason:
            raise ProspectiveExecutionError(
                "active lifecycle receipt cannot carry terminal evidence"
            )

        previous = (
            None
            if self.previous_receipt is None
            else RunLifecycleReceipt.load(
                root / self.previous_receipt.path,
                repository_root=root,
                _seen=seen,
            )
        )
        if previous is not None and (
            self.previous_receipt is None
            or previous.receipt_sha256 != self.previous_receipt.sha256
        ):
            raise ProspectiveExecutionError(
                "previous run lifecycle receipt digest mismatch"
            )
        self._verify_transition(previous)

    def _same_attempt_as(self, previous: "RunLifecycleReceipt") -> bool:
        return (
            self.scientific_action_id == previous.scientific_action_id
            and self.execution_id == previous.execution_id
            and self.cell_id == previous.cell_id
            and self.method == previous.method
            and self.horizon == previous.horizon
            and self.activation_cut == previous.activation_cut
            and self.attempt == previous.attempt
            and self.planner_sequence == previous.planner_sequence
            and self.requested_memory_gib == previous.requested_memory_gib
            and self.bindings_sha256 == previous.bindings_sha256
            and self.execution_request_sha256
            == previous.execution_request_sha256
            and self.action_registry == previous.action_registry
            and self.source_epoch_identity_sha256
            == previous.source_epoch_identity_sha256
            and self.seed_receipt_sha256 == previous.seed_receipt_sha256
            and self.exact_reference_receipt_sha256
            == previous.exact_reference_receipt_sha256
            and self.accuracy_policy_sha256
            == previous.accuracy_policy_sha256
            and self.resume_from_action_id == previous.resume_from_action_id
            and self.resume_run_receipt_sha256
            == previous.resume_run_receipt_sha256
            and self.resume_run_sha256 == previous.resume_run_sha256
        )

    def _verify_transition(
        self, previous: "RunLifecycleReceipt | None"
    ) -> None:
        if self.state == STATE_SUBMITTED:
            if self.attempt == 0:
                if previous is not None:
                    raise ProspectiveExecutionError(
                        "initial submitted lifecycle receipt cannot have a previous receipt"
                    )
                return
            if (
                previous is None
                or previous.state != STATE_FAILED
                or previous.failure_kind != FAILURE_MEMORY
                or previous.attempt != 0
                or previous.scientific_action_id != self.scientific_action_id
                or previous.cell_id != self.cell_id
                or previous.method != self.method
                or previous.horizon != self.horizon
                or previous.activation_cut != self.activation_cut
                or previous.bindings_sha256 != self.bindings_sha256
                or previous.source_epoch_identity_sha256
                != self.source_epoch_identity_sha256
                or previous.seed_receipt_sha256 != self.seed_receipt_sha256
                or previous.exact_reference_receipt_sha256
                != self.exact_reference_receipt_sha256
                or previous.accuracy_policy_sha256
                != self.accuracy_policy_sha256
                or previous.resume_from_action_id != self.resume_from_action_id
                or previous.resume_run_receipt_sha256
                != self.resume_run_receipt_sha256
                or previous.resume_run_sha256 != self.resume_run_sha256
                or previous.observed_peak_gib is None
            ):
                raise ProspectiveExecutionError(
                    "retry submitted lifecycle receipt lacks a verified memory failure"
                )
            scaled = int(math.ceil(1.5 * previous.observed_peak_gib / 4.0) * 4)
            expected_memory = min(
                32, max(scaled, previous.requested_memory_gib + 4)
            )
            if (
                previous.requested_memory_gib >= 32
                or self.requested_memory_gib != expected_memory
                or self.planner_sequence <= previous.planner_sequence
            ):
                raise ProspectiveExecutionError(
                    "retry submitted lifecycle receipt uses an unauthorized memory tier"
                )
            return
        if previous is None:
            raise ProspectiveExecutionError(
                f"{self.state} lifecycle receipt requires a previous receipt"
            )
        if not self._same_attempt_as(previous):
            raise ProspectiveExecutionError(
                "run lifecycle transition changes immutable attempt identity"
            )
        if self.state == STATE_IN_FLIGHT:
            legal = previous.state == STATE_SUBMITTED
        else:
            legal = previous.state in {STATE_SUBMITTED, STATE_IN_FLIGHT}
        if not legal:
            raise ProspectiveExecutionError(
                f"illegal run lifecycle transition {previous.state}->{self.state}"
            )

    def validate_bindings(
        self, root: Path, bindings: ProspectiveScientificBindings
    ) -> None:
        current = self.reload(root)
        if (
            current.bindings_sha256 != bindings.digest
            or current.cell_id != bindings.cell_id
            or current.source_epoch_identity_sha256
            != bindings.source_epoch_identity_sha256
            or current.seed_receipt_sha256 != bindings.seed_receipt_sha256
            or current.exact_reference_receipt_sha256
            != bindings.reference_receipt_sha256
            or current.accuracy_policy_sha256
            != bindings.accuracy_policy_sha256
        ):
            raise ProspectiveExecutionError(
                "run lifecycle receipt diverges from scientific bindings"
            )
        expected_action = prospective_run_action_id(
            bindings,
            arm=current.method,
            horizon=current.horizon,
            activation_cut=current.activation_cut,
        )
        if current.scientific_action_id != expected_action:
            raise ProspectiveExecutionError(
                "run lifecycle scientific action identity is not canonical"
            )

    def to_planner_evidence(
        self, root: Path, bindings: ProspectiveScientificBindings
    ) -> ProspectiveRunEvidence:
        self.validate_bindings(root, bindings)
        return ProspectiveRunEvidence(
            action_id=self.scientific_action_id,
            bindings_sha256=bindings.digest,
            arm=self.method,
            horizon=self.horizon,
            activation_cut=self.activation_cut,
            attempt=self.attempt,
            planner_sequence=self.planner_sequence,
            requested_memory_gib=self.requested_memory_gib,
            state=self.state,
            resume_from_action_id=self.resume_from_action_id,
            resume_run_receipt_sha256=self.resume_run_receipt_sha256,
            resume_run_sha256=self.resume_run_sha256,
            run_receipt_sha256=(
                self.receipt_sha256 if self.state == STATE_COMPLETED else None
            ),
            run_sha256=(
                self.run.sha256
                if self.state == STATE_COMPLETED and self.run is not None
                else None
            ),
            failure_kind=(
                self.failure_kind if self.state == STATE_FAILED else None
            ),
            failure_receipt_sha256=(
                self.receipt_sha256 if self.state == STATE_FAILED else None
            ),
            failure_reason=(
                self.failure_reason if self.state == STATE_FAILED else ""
            ),
            observed_peak_gib=(
                self.observed_peak_gib if self.state == STATE_FAILED else None
            ),
        )

    def reload(self, root: Path) -> "RunLifecycleReceipt":
        current = RunLifecycleReceipt.load(
            root / self.receipt_path, repository_root=root
        )
        if (
            current.receipt_sha256 != self.receipt_sha256
            or current != self
        ):
            raise ProspectiveExecutionError(
                "run lifecycle receipt changed after validation"
            )
        return current


def write_run_lifecycle_receipt(
    path: str | Path,
    *,
    request: "ExecutionRequest",
    state: str,
    action_registry: VerifiedFile,
    state_evidence: VerifiedFile,
    repository_root: str | Path,
    previous_receipt: VerifiedFile | None = None,
    run: VerifiedFile | None = None,
    failure_kind: str | None = None,
    failure_reason: str = "",
    failure_evidence: VerifiedFile | None = None,
    observed_peak_gib: float | None = None,
) -> RunLifecycleReceipt:
    """Write one immutable lifecycle transition and reload its full chain."""

    root = Path(repository_root).resolve()
    if request.kind != KIND_RUN or request.requested_memory_gib is None:
        raise ProspectiveExecutionError(
            "run lifecycle writer requires a scientific run request"
        )
    registered_request, registry = _load_action_registry_envelope(
        action_registry, root=root
    )
    if registered_request != request:
        raise ProspectiveExecutionError(
            "run lifecycle writer received another registered request"
        )
    inputs = request.inputs
    payload = {
        "schema": RUN_LIFECYCLE_RECEIPT_V1,
        "scientific_action_id": request.action_id,
        "execution_id": request.execution_id,
        "cell_id": inputs.get("cell_id"),
        "method": request.arm,
        "horizon": inputs.get("horizon"),
        "activation_cut": inputs.get("activation_cut"),
        "attempt": request.attempt,
        "planner_sequence": request.planner_sequence,
        "requested_memory_gib": request.requested_memory_gib,
        "state": state,
        "bindings_sha256": inputs.get("bindings_sha256"),
        "execution_request_sha256": request.digest,
        "action_registry": action_registry.to_json_dict(),
        "source_epoch_identity_sha256": inputs.get(
            "source_epoch_identity_sha256"
        ),
        "seed_receipt_sha256": registry.get("seed_receipt_sha256"),
        "exact_reference_receipt_sha256": registry.get(
            "exact_reference_receipt_sha256"
        ),
        "accuracy_policy_sha256": inputs.get("accuracy_policy_sha256"),
        "resume_from_action_id": inputs.get("resume_from_action_id"),
        "resume_run_receipt_sha256": inputs.get("resume_run_receipt_sha256"),
        "resume_run_sha256": inputs.get("resume_run_sha256"),
        "previous_receipt": (
            None if previous_receipt is None else previous_receipt.to_json_dict()
        ),
        "state_evidence": state_evidence.to_json_dict(),
        "run": None if run is None else run.to_json_dict(),
        "failure_kind": failure_kind,
        "failure_reason": str(failure_reason).strip(),
        "failure_evidence": (
            None if failure_evidence is None else failure_evidence.to_json_dict()
        ),
        "observed_peak_gib": observed_peak_gib,
    }
    payload["run_lifecycle_receipt_sha256"] = _canonical_sha(payload)
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "run lifecycle target is outside repository"
        ) from exc
    if target.exists():
        if _read_object(target, label="run lifecycle receipt") != payload:
            raise ProspectiveExecutionError("run lifecycle identity collision")
    else:
        _atomic_write_json(target, payload)
    return RunLifecycleReceipt.load(target, repository_root=root)


@dataclass(frozen=True)
class ScoreQualificationReceipt:
    score_action_id: str
    scientific_action_id: str
    execution_id: str
    cell_id: str
    method: str
    horizon: str
    activation_cut: float
    attempt: int
    planner_sequence: int
    requested_memory_gib: int
    run_receipt: VerifiedFile
    action_registry: VerifiedFile
    run: VerifiedFile
    resume_from_action_id: str | None
    resume_run_receipt_sha256: str | None
    resume_run_sha256: str | None
    exact_reference_id: str
    exact_reference_identity_sha256: str
    exact_reference_receipt: VerifiedFile
    exact_manifest_sha256: str
    exact_data_sha256: str
    bindings_sha256: str
    source_epoch_identity_sha256: str
    accuracy_policy_sha256: str
    metrics: VerifiedFile
    planner_receipt: VerifiedFile
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "ScoreQualificationReceipt":
        root = Path(repository_root).resolve()
        source = Path(path)
        if not source.is_absolute():
            source = root / source
        source = source.resolve(strict=True)
        payload = _read_object(source, label="score qualification receipt")
        if payload.get("schema") != SCORE_QUALIFICATION_RECEIPT_V1:
            raise ProspectiveExecutionError("unsupported score receipt schema")
        self_sha = _verify_self_digest(
            payload, key="score_receipt_sha256", label="score receipt"
        )

        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProspectiveExecutionError("score receipt is outside repository") from exc
        run_receipt = _verified_binding(
            payload, "run_receipt", label="score receipt"
        )
        action_registry = _verified_binding(
            payload, "action_registry", label="score receipt"
        )
        lifecycle = RunLifecycleReceipt.load(
            root / run_receipt.path, repository_root=root
        )
        if lifecycle.receipt_sha256 != run_receipt.sha256:
            raise ProspectiveExecutionError(
                "score receipt run-lifecycle digest mismatch"
            )
        if lifecycle.state != STATE_COMPLETED or lifecycle.run is None:
            raise ProspectiveExecutionError(
                "score receipt requires a typed completed run receipt"
            )
        if lifecycle.action_registry != action_registry:
            raise ProspectiveExecutionError(
                "score receipt action registry diverges from completed run"
            )
        result = cls(
            score_action_id=_token(payload.get("score_action_id"), label="score action"),
            scientific_action_id=_token(
                payload.get("scientific_action_id"), label="scientific action"
            ),
            execution_id=_token(payload.get("execution_id"), label="execution id"),
            cell_id=_token(payload.get("cell_id"), label="cell_id"),
            method=_token(payload.get("method"), label="method"),
            horizon=_token(payload.get("horizon"), label="horizon"),
            activation_cut=float(payload.get("activation_cut")),
            attempt=int(payload.get("attempt")),
            planner_sequence=int(payload.get("planner_sequence")),
            requested_memory_gib=int(payload.get("requested_memory_gib")),
            run_receipt=run_receipt,
            action_registry=action_registry,
            run=lifecycle.run,
            resume_from_action_id=lifecycle.resume_from_action_id,
            resume_run_receipt_sha256=lifecycle.resume_run_receipt_sha256,
            resume_run_sha256=lifecycle.resume_run_sha256,
            exact_reference_id=_token(
                payload.get("exact_reference_id"), label="exact reference"
            ),
            exact_reference_identity_sha256=_sha(
                payload.get("exact_reference_identity_sha256"),
                label="exact reference identity",
            ),
            exact_reference_receipt=_verified_binding(
                payload, "exact_reference_receipt", label="score receipt"
            ),
            exact_manifest_sha256=_sha(
                payload.get("exact_manifest_sha256"), label="exact manifest"
            ),
            exact_data_sha256=_sha(
                payload.get("exact_data_sha256"), label="exact data"
            ),
            bindings_sha256=_sha(
                payload.get("bindings_sha256"), label="scientific bindings"
            ),
            source_epoch_identity_sha256=_sha(
                payload.get("source_epoch_identity_sha256"),
                label="source epoch identity",
            ),
            accuracy_policy_sha256=_sha(
                payload.get("accuracy_policy_sha256"), label="accuracy policy"
            ),
            metrics=_verified_binding(payload, "metrics", label="score receipt"),
            planner_receipt=_verified_binding(
                payload, "planner_receipt", label="score receipt"
            ),
            receipt_path=relative,
            receipt_sha256=_file_sha(source),
            receipt_self_sha256=self_sha,
        )
        if result.method not in {ARM_APM, ARM_AVQDS}:
            raise ProspectiveExecutionError("score receipt names an unsupported method")
        if result.horizon not in {HORIZON_PREFIX, HORIZON_FULL}:
            raise ProspectiveExecutionError("score receipt has an unsupported horizon")
        if result.score_action_id != f"score::{result.scientific_action_id}":
            raise ProspectiveExecutionError("score action identity is not canonical")
        if (
            lifecycle.scientific_action_id != result.scientific_action_id
            or lifecycle.execution_id != result.execution_id
            or lifecycle.cell_id != result.cell_id
            or lifecycle.method != result.method
            or lifecycle.horizon != result.horizon
            or not math.isclose(
                lifecycle.activation_cut,
                result.activation_cut,
                rel_tol=5e-12,
                abs_tol=0.0,
            )
            or lifecycle.attempt != result.attempt
            or lifecycle.planner_sequence != result.planner_sequence
            or lifecycle.requested_memory_gib != result.requested_memory_gib
            or lifecycle.bindings_sha256 != result.bindings_sha256
            or lifecycle.source_epoch_identity_sha256
            != result.source_epoch_identity_sha256
            or lifecycle.exact_reference_receipt_sha256
            != result.exact_reference_receipt.sha256
            or lifecycle.accuracy_policy_sha256
            != result.accuracy_policy_sha256
        ):
            raise ProspectiveExecutionError(
                "score receipt diverges from its typed completed run"
            )
        for artifact in (
            result.run,
            result.run_receipt,
            result.action_registry,
            result.exact_reference_receipt,
            result.metrics,
            result.planner_receipt,
        ):
            artifact.verify(root)
        metrics = _read_object(root / result.metrics.path, label="score metrics")
        if metrics.get("schema") != SCORE_METRICS_V1:
            raise ProspectiveExecutionError("unsupported score metrics schema")
        _verify_self_digest(
            metrics, key="score_metrics_sha256", label="score metrics"
        )
        if (
            metrics.get("scientific_action_id") != result.scientific_action_id
            or metrics.get("cell_id") != result.cell_id
            or metrics.get("method") != result.method
            or metrics.get("score_action_id") != result.score_action_id
            or metrics.get("bindings_sha256") != result.bindings_sha256
            or metrics.get("source_epoch_identity_sha256")
            != result.source_epoch_identity_sha256
            or metrics.get("exact_reference_id") != result.exact_reference_id
            or metrics.get("exact_reference_identity_sha256")
            != result.exact_reference_identity_sha256
            or metrics.get("execution_id") != result.execution_id
            or metrics.get("horizon") != result.horizon
            or float(metrics.get("activation_cut")) != result.activation_cut
            or int(metrics.get("attempt")) != result.attempt
            or int(metrics.get("planner_sequence")) != result.planner_sequence
            or int(metrics.get("requested_memory_gib"))
            != result.requested_memory_gib
            or metrics.get("run_receipt_sha256") != result.run_receipt_sha256
            or metrics.get("resume_from_action_id")
            != result.resume_from_action_id
            or metrics.get("resume_run_receipt_sha256")
            != result.resume_run_receipt_sha256
            or metrics.get("resume_run_sha256") != result.resume_run_sha256
            or metrics.get("exact_reference_receipt_sha256")
            != result.exact_reference_receipt.sha256
            or metrics.get("exact_manifest_sha256")
            != result.exact_manifest_sha256
            or metrics.get("exact_data_sha256") != result.exact_data_sha256
            or metrics.get("run_sha256") != result.run.sha256
            or metrics.get("accuracy_policy_sha256")
            != result.accuracy_policy_sha256
        ):
            raise ProspectiveExecutionError("score metrics identity mismatch")
        policy_row = metrics.get("accuracy_policy")
        if not isinstance(policy_row, Mapping):
            raise ProspectiveExecutionError("score metrics omit the accuracy policy")
        policy = AccuracyPolicy.from_json_dict(policy_row)
        if _canonical_sha(policy.to_json_dict()) != result.accuracy_policy_sha256:
            raise ProspectiveExecutionError("score metrics accuracy-policy drift")
        try:
            AccuracyMetrics.from_json_dict(
                _read_required_mapping(metrics, "metrics", label="score metrics")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProspectiveExecutionError("score metrics values are invalid") from exc
        planner = _read_object(
            root / result.planner_receipt.path, label="scientific planner receipt"
        )
        if planner.get("schema") != PROSPECTIVE_SCIENTIFIC_PLAN_V1:
            raise ProspectiveExecutionError("unsupported scientific planner receipt")
        _verify_self_digest(
            planner, key="planner_receipt_sha256", label="scientific planner receipt"
        )
        if planner.get("bindings_sha256") != result.bindings_sha256:
            raise ProspectiveExecutionError("planner/score binding mismatch")
        return result

    @property
    def run_receipt_sha256(self) -> str:
        return self.run_receipt.sha256

    @property
    def exact_reference_receipt_sha256(self) -> str:
        return self.exact_reference_receipt.sha256

    def run_lifecycle(self, root: Path) -> RunLifecycleReceipt:
        current = RunLifecycleReceipt.load(
            root / self.run_receipt.path, repository_root=root
        )
        if current.receipt_sha256 != self.run_receipt.sha256:
            raise ProspectiveExecutionError(
                "score run-lifecycle receipt changed after validation"
            )
        return current

    def accuracy_metrics(self, root: Path) -> AccuracyMetrics:
        payload = _read_object(root / self.metrics.path, label="score metrics")
        self.metrics.verify(root)
        _verify_self_digest(
            payload, key="score_metrics_sha256", label="score metrics"
        )
        return AccuracyMetrics.from_json_dict(
            _read_required_mapping(payload, "metrics", label="score metrics")
        )

    def accuracy_policy(self, root: Path) -> AccuracyPolicy:
        payload = _read_object(root / self.metrics.path, label="score metrics")
        policy = AccuracyPolicy.from_json_dict(
            _read_required_mapping(payload, "accuracy_policy", label="score metrics")
        )
        if _canonical_sha(policy.to_json_dict()) != self.accuracy_policy_sha256:
            raise ProspectiveExecutionError("score accuracy-policy digest mismatch")
        return policy

    def validate_bindings(
        self, root: Path, bindings: ProspectiveScientificBindings
    ) -> None:
        current = self.reload(root)
        lifecycle = current.run_lifecycle(root)
        lifecycle.validate_bindings(root, bindings)
        if (
            current.cell_id != bindings.cell_id
            or current.bindings_sha256 != bindings.digest
            or current.exact_reference_id != bindings.reference_id
            or current.exact_reference_identity_sha256
            != bindings.reference_identity_sha256
            or current.exact_reference_receipt.sha256
            != bindings.reference_receipt_sha256
            or current.exact_manifest_sha256
            != bindings.reference_manifest_sha256
            or current.exact_data_sha256 != bindings.reference_data_sha256
            or current.source_epoch_identity_sha256
            != bindings.source_epoch_identity_sha256
            or current.accuracy_policy_sha256
            != bindings.accuracy_policy_sha256
        ):
            raise ProspectiveExecutionError(
                "score receipt diverges from scientific bindings"
            )
        expected_action = prospective_run_action_id(
            bindings,
            arm=current.method,
            horizon=current.horizon,
            activation_cut=current.activation_cut,
        )
        if current.scientific_action_id != expected_action:
            raise ProspectiveExecutionError(
                "score receipt scientific action identity is not canonical"
            )

    def to_planner_evidence(
        self, root: Path, bindings: ProspectiveScientificBindings
    ) -> ProspectiveRunEvidence:
        self.validate_bindings(root, bindings)
        return ProspectiveRunEvidence(
            action_id=self.scientific_action_id,
            bindings_sha256=bindings.digest,
            arm=self.method,
            horizon=self.horizon,
            activation_cut=self.activation_cut,
            attempt=self.attempt,
            planner_sequence=self.planner_sequence,
            requested_memory_gib=self.requested_memory_gib,
            state=STATE_COMPLETED,
            resume_from_action_id=self.resume_from_action_id,
            resume_run_receipt_sha256=self.resume_run_receipt_sha256,
            resume_run_sha256=self.resume_run_sha256,
            run_receipt_sha256=self.run_receipt_sha256,
            run_sha256=self.run.sha256,
            score_state=SCORE_COMPLETED,
            score_receipt_sha256=self.receipt_sha256,
            score_sha256=self.metrics.sha256,
            metrics=self.accuracy_metrics(root),
        )

    def reload(self, root: Path) -> "ScoreQualificationReceipt":
        reloaded = ScoreQualificationReceipt.load(
            root / self.receipt_path, repository_root=root
        )
        if (
            reloaded.receipt_sha256 != self.receipt_sha256
            or reloaded != self
        ):
            raise ProspectiveExecutionError("score receipt changed after validation")
        return reloaded


@dataclass(frozen=True)
class SafeguardQualificationReceipt:
    safeguard_action_id: str
    reason: str
    subject_action_id: str
    planner_sequence: int
    cell_id: str
    method: str
    bindings_sha256: str
    action_registry: VerifiedFile
    subject_run_receipts: tuple[VerifiedFile, ...]
    subject_score_receipt: VerifiedFile | None
    planner_receipt: VerifiedFile
    source_epoch_identity_sha256: str
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "SafeguardQualificationReceipt":
        root = Path(repository_root).resolve()
        source = Path(path)
        if not source.is_absolute():
            source = root / source
        source = source.resolve(strict=True)
        payload = _read_object(source, label="safeguard qualification receipt")
        if payload.get("schema") != SAFEGUARD_QUALIFICATION_RECEIPT_V1:
            raise ProspectiveExecutionError("unsupported safeguard receipt schema")
        self_sha = _verify_self_digest(
            payload, key="safeguard_receipt_sha256", label="safeguard receipt"
        )
        run_rows = payload.get("subject_run_receipts")
        if not isinstance(run_rows, list) or not run_rows:
            raise ProspectiveExecutionError(
                "safeguard receipt has no subject run receipts"
            )
        subject_runs = tuple(
            VerifiedFile(str(row.get("path") or ""), str(row.get("sha256") or ""))
            for row in run_rows
            if isinstance(row, Mapping)
        )
        if len(subject_runs) != len(run_rows):
            raise ProspectiveExecutionError(
                "safeguard subject run bindings are malformed"
            )
        score_row = payload.get("subject_score_receipt")
        if score_row is not None and not isinstance(score_row, Mapping):
            raise ProspectiveExecutionError(
                "safeguard subject score binding is malformed"
            )
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProspectiveExecutionError(
                "safeguard receipt is outside repository"
            ) from exc
        result = cls(
            safeguard_action_id=_token(
                payload.get("safeguard_action_id"), label="safeguard action"
            ),
            reason=_token(payload.get("reason"), label="safeguard reason"),
            subject_action_id=_token(
                payload.get("subject_action_id"), label="safeguard subject"
            ),
            planner_sequence=int(payload.get("planner_sequence")),
            cell_id=_token(payload.get("cell_id"), label="cell_id"),
            method=_token(payload.get("method"), label="method"),
            bindings_sha256=_sha(
                payload.get("bindings_sha256"), label="scientific bindings"
            ),
            action_registry=_verified_binding(
                payload, "action_registry", label="safeguard receipt"
            ),
            subject_run_receipts=subject_runs,
            subject_score_receipt=(
                None
                if score_row is None
                else VerifiedFile(
                    str(score_row.get("path") or ""),
                    str(score_row.get("sha256") or ""),
                )
            ),
            planner_receipt=_verified_binding(
                payload, "planner_receipt", label="safeguard receipt"
            ),
            source_epoch_identity_sha256=_sha(
                payload.get("source_epoch_identity_sha256"),
                label="source epoch identity",
            ),
            receipt_path=relative,
            receipt_sha256=_file_sha(source),
            receipt_self_sha256=self_sha,
        )
        if result.method not in {ARM_APM, ARM_AVQDS}:
            raise ProspectiveExecutionError(
                "safeguard receipt names an unsupported method"
            )
        if result.reason not in {
            SAFEGUARD_LADDER_UNRESOLVED,
            SAFEGUARD_OPERATIONALLY_INFEASIBLE,
        }:
            raise ProspectiveExecutionError("unsupported terminal safeguard reason")
        if result.planner_sequence < 0 or payload.get("qiskit_compilation") is not False:
            raise ProspectiveExecutionError(
                "safeguard receipt has invalid planner or compilation state"
            )

        request, registry = _load_action_registry_envelope(
            result.action_registry, root=root
        )
        inputs = request.inputs
        if (
            request.kind != KIND_RECORD_SAFEGUARD
            or request.action_id != result.safeguard_action_id
            or request.execution_id != result.safeguard_action_id
            or request.attempt != 0
            or request.planner_sequence != result.planner_sequence
            or request.requested_memory_gib is not None
            or request.arm is not None
            or request.lane_id != inputs.get("lane_id")
            or registry.get("bindings_sha256") != result.bindings_sha256
            or inputs.get("bindings_sha256") != result.bindings_sha256
            or inputs.get("cell_id") != result.cell_id
            or inputs.get("source_epoch_identity_sha256")
            != result.source_epoch_identity_sha256
            or inputs.get("safeguard_reason") != result.reason
            or inputs.get("subject_action_id") != result.subject_action_id
            or inputs.get("retain_all_numerical_evidence") is not True
        ):
            raise ProspectiveExecutionError(
                "safeguard receipt diverges from its registered action"
            )

        typed_runs: list[RunLifecycleReceipt] = []
        attempts: set[int] = set()
        for binding in result.subject_run_receipts:
            run = RunLifecycleReceipt.load(root / binding.path, repository_root=root)
            if run.receipt_sha256 != binding.sha256:
                raise ProspectiveExecutionError(
                    "safeguard subject run receipt digest mismatch"
                )
            if run.attempt in attempts:
                raise ProspectiveExecutionError(
                    "safeguard receipt repeats a subject run attempt"
                )
            attempts.add(run.attempt)
            if (
                run.scientific_action_id != result.subject_action_id
                or run.cell_id != result.cell_id
                or run.method != result.method
                or run.bindings_sha256 != result.bindings_sha256
                or run.source_epoch_identity_sha256
                != result.source_epoch_identity_sha256
                or run.planner_sequence >= result.planner_sequence
            ):
                raise ProspectiveExecutionError(
                    "safeguard subject run identity mismatch"
                )
            typed_runs.append(run)

        score: ScoreQualificationReceipt | None = None
        if result.subject_score_receipt is not None:
            score = ScoreQualificationReceipt.load(
                root / result.subject_score_receipt.path,
                repository_root=root,
            )
            if score.receipt_sha256 != result.subject_score_receipt.sha256:
                raise ProspectiveExecutionError(
                    "safeguard subject score receipt digest mismatch"
                )
        if result.reason == SAFEGUARD_LADDER_UNRESOLVED:
            if (
                score is None
                or score.scientific_action_id != result.subject_action_id
                or score.cell_id != result.cell_id
                or score.method != result.method
                or score.bindings_sha256 != result.bindings_sha256
                or score.run_receipt not in result.subject_run_receipts
            ):
                raise ProspectiveExecutionError(
                    "ladder safeguard lacks its exact scored subject run"
                )
        elif (
            score is not None
            or sorted(attempts) != [0, 1]
            or any(
                run.state != STATE_FAILED or run.failure_kind != FAILURE_MEMORY
                for run in typed_runs
            )
        ):
            raise ProspectiveExecutionError(
                "operational safeguard lacks both failed memory attempts"
            )

        result.planner_receipt.verify(root)
        planner_payload = _read_object(
            root / result.planner_receipt.path, label="scientific planner receipt"
        )
        if planner_payload.get("schema") != PROSPECTIVE_SCIENTIFIC_PLAN_V1:
            raise ProspectiveExecutionError("unsupported safeguard planner receipt")
        _verify_self_digest(
            planner_payload,
            key="planner_receipt_sha256",
            label="safeguard planner receipt",
        )
        if (
            planner_payload.get("bindings_sha256") != result.bindings_sha256
            or planner_payload.get("complete") is not True
            or planner_payload.get("unresolved") is not True
            or planner_payload.get("cost_comparison_admissible") is not False
            or planner_payload.get("qiskit_compilation_requested") is not False
        ):
            raise ProspectiveExecutionError(
                "terminal safeguard is not bound to an unresolved planner result"
            )
        return result

    @property
    def scientific_action_id(self) -> str:
        """Compatibility alias for the terminal subject action."""

        return self.subject_action_id

    @property
    def outcome(self) -> str:
        return (
            "threshold_ladder_exhausted"
            if self.reason == SAFEGUARD_LADDER_UNRESOLVED
            else "operationally_infeasible"
        )

    @property
    def score_receipt_sha256(self) -> str | None:
        return (
            None
            if self.subject_score_receipt is None
            else self.subject_score_receipt.sha256
        )

    def validate_bindings(
        self, root: Path, bindings: ProspectiveScientificBindings
    ) -> None:
        current = self.reload(root)
        if (
            current.bindings_sha256 != bindings.digest
            or current.cell_id != bindings.cell_id
            or current.source_epoch_identity_sha256
            != bindings.source_epoch_identity_sha256
            or current.safeguard_action_id
            != prospective_safeguard_id(
                bindings,
                reason=current.reason,
                subject_action_id=current.subject_action_id,
            )
        ):
            raise ProspectiveExecutionError(
                "safeguard receipt diverges from scientific bindings"
            )
        for row in current.subject_run_receipts:
            run = RunLifecycleReceipt.load(root / row.path, repository_root=root)
            run.validate_bindings(root, bindings)
        if current.subject_score_receipt is not None:
            score = ScoreQualificationReceipt.load(
                root / current.subject_score_receipt.path,
                repository_root=root,
            )
            score.validate_bindings(root, bindings)

    def to_planner_evidence(
        self, root: Path, bindings: ProspectiveScientificBindings
    ) -> ProspectiveSafeguardEvidence:
        self.validate_bindings(root, bindings)
        return ProspectiveSafeguardEvidence(
            safeguard_id=self.safeguard_action_id,
            bindings_sha256=bindings.digest,
            reason=self.reason,
            subject_action_id=self.subject_action_id,
            receipt_sha256=self.receipt_sha256,
            planner_sequence=self.planner_sequence,
        )

    def reload(self, root: Path) -> "SafeguardQualificationReceipt":
        current = SafeguardQualificationReceipt.load(
            root / self.receipt_path, repository_root=root
        )
        if current.receipt_sha256 != self.receipt_sha256 or current != self:
            raise ProspectiveExecutionError(
                "safeguard receipt changed after validation"
            )
        return current


def _embedded_lane_scientific_authority(
    value: Any,
) -> LaneScientificAuthority:
    try:
        return LaneScientificAuthority.from_json_dict(value)
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "invalid embedded lane scientific authority"
        ) from exc


@dataclass(frozen=True)
class ScientificPreSubmitAudit:
    audit_scope: str
    action_id: str
    execution_id: str
    execution_request_sha256: str
    execution_request: "ExecutionRequest"
    bindings_sha256: str
    lane_id: str
    lane_scientific_authority: LaneScientificAuthority
    matrix_path: str
    matrix_raw_sha256: str
    matrix_payload_sha256: str
    matrix_expanded_semantic_sha256: str
    matrix_accuracy_policy_sha256: str
    identity_registry_sha256: str
    executor_registry_sha256: str
    planned_report_contract: VerifiedFile
    planned_report_contract_sha256: str
    scientific_state: VerifiedFile
    action_registries: tuple[VerifiedFile, ...]
    run_lifecycle_tips: tuple[VerifiedFile, ...]
    score_receipts: tuple[VerifiedFile, ...]
    safeguard_receipts: tuple[VerifiedFile, ...]
    evidence_files: tuple[VerifiedFile, ...]
    source_epoch_receipts: tuple[VerifiedFile, ...]
    seed_receipts: tuple[VerifiedFile, ...]
    exact_reference_receipts: tuple[VerifiedFile, ...]
    source_epoch_identity_sha256: tuple[str, ...]
    seed_receipt_sha256: tuple[str, ...]
    exact_reference_receipt_sha256: tuple[str, ...]
    warnings: tuple[str, ...]
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "ScientificPreSubmitAudit":
        return _load_scientific_pre_submit_audit(
            path, repository_root=repository_root
        )

    def verify_contracts(
        self,
        *,
        lane_id: str,
        matrix_raw_sha256: str,
        matrix_payload_sha256: str,
        matrix_expanded_semantic_sha256: str,
        identity_registry_sha256: str,
        executor_registry_sha256: str,
        planned_report_contract_sha256: str,
    ) -> None:
        expected = (
            lane_id,
            matrix_raw_sha256,
            matrix_payload_sha256,
            matrix_expanded_semantic_sha256,
            identity_registry_sha256,
            executor_registry_sha256,
            planned_report_contract_sha256,
        )
        observed = (
            self.lane_id,
            self.matrix_raw_sha256,
            self.matrix_payload_sha256,
            self.matrix_expanded_semantic_sha256,
            self.identity_registry_sha256,
            self.executor_registry_sha256,
            self.planned_report_contract_sha256,
        )
        if observed != expected:
            raise ProspectiveExecutionError(
                "scientific pre-submit audit contract binding mismatch"
            )

    def verify_evidence(
        self,
        *,
        source_epoch_identity_sha256: str,
        seed_receipt_sha256: str,
        exact_reference_receipt_sha256: str,
    ) -> None:
        if (
            source_epoch_identity_sha256
            not in self.source_epoch_identity_sha256
            or seed_receipt_sha256 not in self.seed_receipt_sha256
            or exact_reference_receipt_sha256
            not in self.exact_reference_receipt_sha256
        ):
            raise ProspectiveExecutionError(
                "scientific pre-submit audit does not bind current evidence"
            )

    def verify_request(self, request: "ExecutionRequest") -> None:
        if (
            request != self.execution_request
            or request.kind != self.audit_scope
            or request.action_id != self.action_id
            or request.execution_id != self.execution_id
            or request.digest != self.execution_request_sha256
            or request.inputs.get("bindings_sha256") != self.bindings_sha256
            or request.lane_id != self.lane_id
            or _embedded_lane_scientific_authority(
                request.inputs.get("lane_scientific_authority")
            )
            != self.lane_scientific_authority
            or request.inputs.get("scientific_state")
            != self.scientific_state.to_json_dict()
            or request.inputs.get("planned_report_contract")
            != self.planned_report_contract.to_json_dict()
        ):
            raise ProspectiveExecutionError(
                "scientific pre-submit audit scope or request identity mismatch"
            )

    def verify_current_registry(
        self,
        campaign_registry: CampaignRegistry,
        *,
        repository_root: str | Path,
    ) -> None:
        root = Path(repository_root).resolve()
        if (
            not isinstance(campaign_registry, CampaignRegistry)
            or campaign_registry.repository_root.resolve() != root
        ):
            raise ProspectiveExecutionError(
                "scientific pre-submit audit has no current campaign registry"
            )
        try:
            current = campaign_registry.lane_scientific_authority(self.lane_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProspectiveExecutionError(
                "current campaign registry has no valid lane authority"
            ) from exc
        if current != self.lane_scientific_authority:
            raise ProspectiveExecutionError(
                "current campaign registry lane authority drifted"
            )

    def reload(self, root: Path) -> "ScientificPreSubmitAudit":
        current = ScientificPreSubmitAudit.load(
            root / self.receipt_path, repository_root=root
        )
        if current.receipt_sha256 != self.receipt_sha256 or current != self:
            raise ProspectiveExecutionError(
                "scientific pre-submit audit changed after validation"
            )
        return current


@dataclass(frozen=True)
class ExecutorBinding:
    binding_id: str
    lanes: tuple[str, ...]
    kinds: tuple[str, ...]
    handler: str
    output_role: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "lanes": list(self.lanes),
            "kinds": list(self.kinds),
            "handler": self.handler,
            "output_role": self.output_role,
        }

    def resolve(self) -> Callable[..., Any]:
        module_name, separator, name = self.handler.partition(":")
        if not separator:
            raise ProspectiveExecutionError("executor handler must be module:function")
        function = getattr(importlib.import_module(module_name), name, None)
        if not callable(function):
            raise ProspectiveExecutionError(
                f"executor handler is not callable: {self.handler}"
            )
        return function


@dataclass(frozen=True)
class ExecutionRequest:
    action_id: str
    lane_id: str
    kind: str
    binding_id: str
    handler: str
    inputs: Mapping[str, Any]
    declared_outputs: tuple[str, ...]
    exact_reference_blind: bool
    execution_id: str
    attempt: int
    planner_sequence: int
    requested_memory_gib: int | None
    arm: str | None

    def __post_init__(self) -> None:
        if self.attempt not in (0, 1) or self.planner_sequence < 0:
            raise ProspectiveExecutionError("invalid execution attempt identity")
        if self.kind in {KIND_RUN, KIND_SCORE} and self.arm not in {
            ARM_APM,
            ARM_AVQDS,
        }:
            raise ProspectiveExecutionError("scientific request has invalid arm")
        if self.kind == KIND_RUN and (
            self.requested_memory_gib is None or self.requested_memory_gib <= 0
        ):
            raise ProspectiveExecutionError("run request has no memory identity")
        if self.kind != KIND_RUN and self.requested_memory_gib is not None:
            raise ProspectiveExecutionError(
                "non-run request cannot carry requested memory"
            )
        if self.kind == KIND_RUN:
            expected_execution = (
                f"{self.action_id}::attempt{self.attempt}::"
                f"mem{self.requested_memory_gib}g"
            )
            if self.execution_id != expected_execution:
                raise ProspectiveExecutionError(
                    "run execution identity does not bind attempt and memory"
                )
        elif self.kind in {
            KIND_SCORE,
            KIND_RECORD_SAFEGUARD,
            KIND_BUILD_PACKAGE,
            KIND_UPDATE_REPORT_SLOT,
        }:
            if self.execution_id != self.action_id:
                raise ProspectiveExecutionError(
                    "non-run execution identity must equal its action identity"
                )
        if self.kind in {KIND_RUN, KIND_SCORE} and self.inputs.get("arm") != self.arm:
            raise ProspectiveExecutionError("request arm/input arm diverged")
        if self.kind in {KIND_BUILD_PACKAGE, KIND_UPDATE_REPORT_SLOT}:
            authority = _embedded_lane_scientific_authority(
                self.inputs.get("lane_scientific_authority")
            )
            if authority.lane_id != self.lane_id:
                raise ProspectiveExecutionError(
                    "terminal request lane authority crosses lanes"
                )
        if bool(
            self.inputs.get("qiskit_compilation")
            or self.inputs.get("qiskit_compilation_requested")
        ):
            raise ProspectiveExecutionError(
                "Qiskit compilation is outside this campaign"
            )

    @property
    def digest(self) -> str:
        return _canonical_sha(self.to_json_dict(include_digest=False))

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema": EXECUTION_REQUEST_V1,
            "action_id": self.action_id,
            "lane_id": self.lane_id,
            "kind": self.kind,
            "binding_id": self.binding_id,
            "handler": self.handler,
            "inputs": dict(self.inputs),
            "declared_outputs": list(self.declared_outputs),
            "exact_reference_blind": self.exact_reference_blind,
            "execution_id": self.execution_id,
            "attempt": self.attempt,
            "planner_sequence": self.planner_sequence,
            "requested_memory_gib": self.requested_memory_gib,
            "arm": self.arm,
            "qiskit_compilation_requested": False,
        }
        if include_digest:
            payload["execution_request_sha256"] = self.digest
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExecutionRequest":
        if payload.get("schema") != EXECUTION_REQUEST_V1:
            raise ProspectiveExecutionError("unsupported execution request schema")
        unsigned = dict(payload)
        declared = unsigned.pop("execution_request_sha256", None)
        if declared != _canonical_sha(unsigned):
            raise ProspectiveExecutionError("execution request self digest mismatch")
        outputs = payload.get("declared_outputs")
        inputs = payload.get("inputs")
        if not isinstance(outputs, list) or not isinstance(inputs, Mapping):
            raise ProspectiveExecutionError("execution request is incomplete")
        request = cls(
            action_id=_token(payload.get("action_id"), label="action id"),
            lane_id=_token(payload.get("lane_id"), label="lane id"),
            kind=_token(payload.get("kind"), label="action kind"),
            binding_id=_token(payload.get("binding_id"), label="binding id"),
            handler=_token(payload.get("handler"), label="handler"),
            inputs=dict(inputs),
            declared_outputs=tuple(str(value) for value in outputs),
            exact_reference_blind=bool(payload.get("exact_reference_blind")),
            execution_id=_token(payload.get("execution_id"), label="execution id"),
            attempt=int(payload.get("attempt")),
            planner_sequence=int(payload.get("planner_sequence")),
            requested_memory_gib=(
                None
                if payload.get("requested_memory_gib") is None
                else int(payload["requested_memory_gib"])
            ),
            arm=(None if payload.get("arm") is None else str(payload["arm"])),
        )
        binding = _binding(request.binding_id)
        if (
            request.handler != binding.handler
            or request.lane_id not in binding.lanes
            or request.kind not in binding.kinds
            or payload.get("qiskit_compilation_requested") is not False
        ):
            raise ProspectiveExecutionError("execution request resolver binding mismatch")
        return request


_HANDLER = "pipelines.time_dynamics.prospective_execution_bindings"


def executor_bindings() -> tuple[ExecutorBinding, ...]:
    lanes = (LANE_HH, LANE_HUBBARD)
    return (
        ExecutorBinding(EXEC_HH_SEED, (LANE_HH,), ("build_seed",), f"{_HANDLER}:seed_request", "seed_receipt"),
        ExecutorBinding(EXEC_HUBBARD_SEED, (LANE_HUBBARD,), ("build_seed",), f"{_HANDLER}:seed_request", "seed_receipt"),
        ExecutorBinding(EXEC_HH_REFERENCE, (LANE_HH,), ("build_exact_reference",), "pipelines.time_dynamics.prospective_exact_reference_worker:materialization_request", "exact_reference_receipt"),
        ExecutorBinding(EXEC_HUBBARD_REFERENCE, (LANE_HUBBARD,), ("build_exact_reference",), "pipelines.time_dynamics.prospective_exact_reference_worker:materialization_request", "exact_reference_receipt"),
        ExecutorBinding(EXEC_HH_AVQDS, (LANE_HH,), ("screen_avqds",), "pipelines.time_dynamics.prospective_avqds_scout_worker:materialization_request", "run_receipt"),
        ExecutorBinding(EXEC_HUBBARD_AVQDS, (LANE_HUBBARD,), ("screen_avqds",), "pipelines.time_dynamics.prospective_avqds_scout_worker:materialization_request", "run_receipt"),
        ExecutorBinding(EXEC_HH_SCORE, (LANE_HH,), ("score_avqds",), f"{_HANDLER}:score_request", "score_receipt"),
        ExecutorBinding(EXEC_HUBBARD_SCORE, (LANE_HUBBARD,), ("score_avqds",), f"{_HANDLER}:score_request", "score_receipt"),
        ExecutorBinding(EXEC_PROSPECTIVE_APM, lanes, (KIND_RUN,), "pipelines.time_dynamics.prospective_apm_exchange_worker:materialization_request", "run_receipt"),
        ExecutorBinding(EXEC_PROSPECTIVE_AVQDS, lanes, (KIND_RUN,), "pipelines.time_dynamics.prospective_avqds_trajectory_worker:materialization_request", "run_receipt"),
        ExecutorBinding(EXEC_PROSPECTIVE_SCORE, lanes, (KIND_SCORE,), "pipelines.time_dynamics.prospective_trajectory_scoring_worker:materialization_request", "score_receipt"),
        ExecutorBinding(EXEC_PROSPECTIVE_SAFEGUARD, lanes, (KIND_RECORD_SAFEGUARD,), f"{_HANDLER}:safeguard_request", "safeguard_receipt"),
        ExecutorBinding(EXEC_ACTION_REGISTRY, lanes, ("register_action",), f"{_HANDLER}:write_action_registry_entry", "action_registry"),
        ExecutorBinding(EXEC_SCIENTIFIC_STATE, lanes, ("write_scientific_state",), f"{_HANDLER}:write_scientific_state", "scientific_state"),
        ExecutorBinding(EXEC_PACKAGE_BUILDER, lanes, (KIND_BUILD_PACKAGE,), f"{_HANDLER}:package_request", "package_manifest"),
        ExecutorBinding(EXEC_REPORT_SLOT, lanes, (KIND_UPDATE_REPORT_SLOT,), f"{_HANDLER}:report_slot_request", "report_contract"),
    )


def binding_registry_payload() -> dict[str, Any]:
    rows = executor_bindings()
    for row in rows:
        row.resolve()
    payload = {
        "schema": EXECUTION_BINDING_REGISTRY_V1,
        "bindings": [row.to_json_dict() for row in rows],
        "qiskit_compilation": False,
        "production_submission_enabled": False,
    }
    payload["bindings_sha256"] = _canonical_sha(payload)
    return payload


def _binding(binding_id: str) -> ExecutorBinding:
    rows = [row for row in executor_bindings() if row.binding_id == binding_id]
    if len(rows) != 1:
        raise ProspectiveExecutionError(f"unknown or repeated executor {binding_id!r}")
    rows[0].resolve()
    return rows[0]


def _request(
    *,
    action_id: str,
    lane_id: str,
    kind: str,
    binding: ExecutorBinding,
    inputs: Mapping[str, Any],
    exact_reference_blind: bool,
    execution_id: str | None = None,
    attempt: int = 0,
    planner_sequence: int = 0,
    requested_memory_gib: int | None = None,
    arm: str | None = None,
) -> ExecutionRequest:
    if lane_id not in binding.lanes or kind not in binding.kinds:
        raise ProspectiveExecutionError("action kind/lane is incompatible with executor")
    if bool(inputs.get("qiskit_compilation") or inputs.get("qiskit_compilation_requested")):
        raise ProspectiveExecutionError("Qiskit compilation is outside this campaign")
    source_epoch = inputs.get("source_epoch")
    if kind in {"build_seed", "build_exact_reference", "screen_avqds"} and not isinstance(
        source_epoch, Mapping
    ):
        raise ProspectiveExecutionError("bootstrap action has no source epoch binding")
    if isinstance(source_epoch, Mapping):
        try:
            epoch = SourceEpochIdentity(
                lane_id=str(source_epoch.get("lane_id") or ""),
                source_epoch_id=str(source_epoch.get("source_epoch_id") or ""),
                source_revision=source_epoch.get("source_revision"),
                source_archive_sha256=source_epoch.get("source_archive_sha256"),
            )
        except ValueError as exc:
            raise ProspectiveExecutionError("source epoch input is invalid") from exc
        if source_epoch.get("source_epoch_identity_sha256") != epoch.identity_sha256:
            raise ProspectiveExecutionError("source epoch input digest mismatch")
        if epoch.lane_id != lane_id:
            raise ProspectiveExecutionError("source epoch input crosses lanes")
    if kind == "build_seed":
        required = (
            "seed_id", "seed_identity_sha256", "model", "physical",
            "initializer", "restarts", "optimizer_seed", "energy_gate",
        )
        if any(inputs.get(name) in (None, "") for name in required):
            raise ProspectiveExecutionError("seed request lacks immutable VQE inputs")
    if kind == "build_exact_reference":
        required = (
            "reference_id", "reference_identity_sha256", "cell_id", "seed_id",
            "seed_identity_sha256", "seed_artifact_sha256", "model", "physical",
            "drive", "ordered_grid", "propagator", "observable_schema",
            "materialized_reference_key", "exact_data_schema",
        )
        if any(inputs.get(name) in (None, "") for name in required):
            raise ProspectiveExecutionError(
                "exact-reference request lacks immutable physical inputs"
            )
    if kind == KIND_RUN and not (
        inputs.get("source_revision")
        and inputs.get("source_tree_sha256")
        and inputs.get("source_epoch_identity_sha256")
        and inputs.get("seed_identity_sha256")
        and inputs.get("reference_identity_sha256")
    ):
        raise ProspectiveExecutionError("scientific run has no source identity")
    if kind in {KIND_BUILD_PACKAGE, KIND_UPDATE_REPORT_SLOT}:
        required = (
            "bindings_sha256",
            "scientific_state",
            "planned_report_contract",
            "lane_scientific_authority",
            "audit_scope",
            "production_submission_enabled",
        )
        if any(name not in inputs for name in required):
            raise ProspectiveExecutionError(
                "terminal scientific request lacks its audit inputs"
            )
        if (
            inputs.get("audit_scope") != kind
            or inputs.get("production_submission_enabled") is not False
            or not isinstance(inputs.get("scientific_state"), Mapping)
            or not isinstance(inputs.get("planned_report_contract"), Mapping)
            or not isinstance(inputs.get("lane_scientific_authority"), Mapping)
        ):
            raise ProspectiveExecutionError(
                "terminal scientific request has invalid audit scope"
            )
        authority = _embedded_lane_scientific_authority(
            inputs.get("lane_scientific_authority")
        )
        if authority.lane_id != lane_id:
            raise ProspectiveExecutionError(
                "terminal request lane authority crosses lanes"
            )
    outputs = {
        "build_seed": ("seed_artifact", "seed_qualification_receipt"),
        "build_exact_reference": ("reference_manifest", "reference_data", "reference_receipt"),
        "screen_avqds": ("run_json", "trajectory_statevectors", "run_receipt"),
        KIND_RUN: ("run_json", "trajectory_statevectors", "run_receipt"),
        "score_avqds": ("metrics_json", "score_receipt"),
        KIND_SCORE: ("metrics_json", "score_receipt"),
        KIND_RECORD_SAFEGUARD: ("safeguard_receipt",),
        KIND_BUILD_PACKAGE: ("package_manifest",),
        KIND_UPDATE_REPORT_SLOT: ("report_contract",),
    }[kind]
    return ExecutionRequest(
        action_id=action_id,
        lane_id=lane_id,
        kind=kind,
        binding_id=binding.binding_id,
        handler=binding.handler,
        inputs=dict(inputs),
        declared_outputs=outputs,
        exact_reference_blind=exact_reference_blind,
        execution_id=str(execution_id or action_id),
        attempt=int(attempt),
        planner_sequence=int(planner_sequence),
        requested_memory_gib=(
            None if requested_memory_gib is None else int(requested_memory_gib)
        ),
        arm=arm,
    )


def resolve_bootstrap_action(action: BootstrapAction) -> ExecutionRequest:
    if action.qiskit_compilation_requested:
        raise ProspectiveExecutionError("bootstrap action requested Qiskit")
    binding = _binding(action.executor_binding)
    return _request(
        action_id=action.action_id,
        lane_id=action.lane_id,
        kind=action.kind,
        binding=binding,
        inputs=action.inputs,
        exact_reference_blind=action.exact_reference_blind,
    )


def resolve_scientific_action(
    bindings: ProspectiveScientificBindings,
    action: ProspectiveNextAction,
) -> ExecutionRequest:
    if action.qiskit_compilation_requested or action.inputs.get("qiskit_compilation"):
        raise ProspectiveExecutionError("scientific action requested Qiskit")
    if action.inputs.get("bindings_sha256") != bindings.digest:
        raise ProspectiveExecutionError("action/scientific binding digest mismatch")
    if action.kind == KIND_RUN:
        if action.arm == ARM_APM:
            binding_id = EXEC_PROSPECTIVE_APM
        elif action.arm == ARM_AVQDS:
            binding_id = EXEC_PROSPECTIVE_AVQDS
        else:
            raise ProspectiveExecutionError("run action has an unsupported arm")
    elif action.kind == KIND_SCORE:
        if action.arm not in {ARM_APM, ARM_AVQDS}:
            raise ProspectiveExecutionError("score action has an unsupported arm")
        binding_id = EXEC_PROSPECTIVE_SCORE
    elif action.kind == KIND_RECORD_SAFEGUARD:
        binding_id = EXEC_PROSPECTIVE_SAFEGUARD
    else:
        raise ProspectiveExecutionError(f"unsupported scientific action {action.kind!r}")
    return _request(
        action_id=action.action_id,
        lane_id=bindings.lane_id,
        kind=action.kind,
        binding=_binding(binding_id),
        inputs=action.inputs,
        exact_reference_blind=action.exact_reference_blind,
        execution_id=action.execution_id,
        attempt=action.attempt,
        planner_sequence=action.planner_sequence,
        requested_memory_gib=action.requested_memory_gib,
        arm=action.arm,
    )


def build_scientific_bindings(
    *,
    reference: ExactReferenceIdentity,
    seed: SeedIdentity,
    seed_receipt: SeedQualificationReceipt,
    reference_receipt: ExactReferenceReceipt,
    source_epoch: SourceEpochIdentity,
    source_epoch_receipt: SourceEpochReceipt,
    policy: AccuracyPolicy | None = None,
    matrix: ParallelBenchmarkMatrix | None = None,
) -> ProspectiveScientificBindings:
    """Join receipts under the matrix's canonical accuracy policy.

    A caller may pass ``policy`` as an assertion, but never as configuration:
    it must be byte-for-byte equal to the policy derived from the frozen
    matrix and this exact reference's response scales.
    """

    campaign = matrix or ParallelBenchmarkMatrix.read()
    campaign.validate_frozen()
    identity_registry = build_identity_registry(campaign)
    canonical_references = tuple(
        row for row in identity_registry.references if row.cell_id == reference.cell_id
    )
    if (
        len(canonical_references) != 1
        or reference.identity_sha256 != canonical_references[0].identity_sha256
    ):
        raise ProspectiveExecutionError(
            "exact reference is not the canonical matrix reference"
        )
    # Exact-blind AVQDS work orders deliberately omit artifact locations.  The
    # scientific identity digest excludes those transport fields, so restore
    # the matrix-owned locator only after the exact receipt is available.
    reference = canonical_references[0]
    canonical_seeds = tuple(
        row for row in identity_registry.seeds if row.seed_id == reference.seed_id
    )
    if (
        len(canonical_seeds) != 1
        or seed.identity_sha256 != canonical_seeds[0].identity_sha256
    ):
        raise ProspectiveExecutionError("seed is not the canonical matrix seed")
    seed = canonical_seeds[0]
    cells = [cell for cell in campaign.cells if cell.cell_id == reference.cell_id]
    if len(cells) != 1:
        raise ProspectiveExecutionError(
            "exact reference does not name one frozen matrix cell"
        )
    cell = cells[0]
    if (
        cell.lane_id != reference.lane_id
        or cell.model != reference.model
        or cell.drive != reference.drive.get("drive_id")
        or dict(cell.physics) != dict(reference.physical)
    ):
        raise ProspectiveExecutionError(
            "exact reference diverges from the frozen matrix cell"
        )
    try:
        canonical_policy = campaign.canonical_accuracy_policy(
            reference.lane_id,
            reference_receipt.policy_response_scales(cell.drive),
        )
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "cannot construct the matrix-bound accuracy policy"
        ) from exc
    if policy is not None and policy.to_json_dict() != canonical_policy.to_json_dict():
        raise ProspectiveExecutionError(
            "caller-supplied accuracy policy diverges from the matrix contract"
        )
    policy_payload = canonical_policy.to_json_dict()

    if reference.seed_id != seed.seed_id or seed_receipt.seed_id != seed.seed_id:
        raise ProspectiveExecutionError("cell methods do not share the registered seed")
    if seed_receipt.seed_identity_sha256 != seed.identity_sha256:
        raise ProspectiveExecutionError("seed receipt identity diverged")
    if (
        reference_receipt.reference_id != reference.reference_id
        or reference_receipt.reference_identity_sha256 != reference.identity_sha256
        or reference_receipt.lane_id != reference.lane_id
        or reference_receipt.cell_id != reference.cell_id
    ):
        raise ProspectiveExecutionError("exact-reference receipt identity diverged")
    if (
        reference_receipt.seed_id != seed.seed_id
        or reference_receipt.seed_artifact_sha256 != seed_receipt.artifact.sha256
    ):
        raise ProspectiveExecutionError("exact reference was built from another seed")
    if (
        seed_receipt.source_epoch_identity_sha256 != source_epoch.identity_sha256
        or reference_receipt.source_epoch_identity_sha256 != source_epoch.identity_sha256
    ):
        raise ProspectiveExecutionError("bootstrap receipts cross source epochs")
    if (
        source_epoch_receipt.source_epoch != source_epoch
        or source_epoch_receipt.lane_id != reference.lane_id
    ):
        raise ProspectiveExecutionError("source tree is not bound to the source epoch")
    revision = source_epoch.source_revision or source_epoch.source_archive_sha256
    if revision is None:
        raise ProspectiveExecutionError("source epoch has no immutable source identity")
    result = ProspectiveScientificBindings(
        lane_id=reference.lane_id,
        cell_id=reference.cell_id,
        policy_regime=str(reference.physical.get("regime") or "hubbard_l4_u4"),
        drive=str(reference.drive["drive_id"]),
        controller="state_motion_5e-5",
        cell_identity_sha256=reference.identity_sha256,
        seed_id=seed.seed_id,
        seed_identity_sha256=seed.identity_sha256,
        seed_artifact_sha256=seed_receipt.artifact.sha256,
        seed_receipt_sha256=seed_receipt.receipt_sha256,
        reference_id=reference.reference_id,
        reference_identity_sha256=reference.identity_sha256,
        reference_manifest_sha256=reference_receipt.manifest.sha256,
        reference_data_sha256=reference_receipt.data.sha256,
        reference_receipt_sha256=reference_receipt.receipt_sha256,
        source_epoch_identity_sha256=source_epoch.identity_sha256,
        source_revision=revision,
        source_tree_sha256=source_epoch_receipt.source_tree_sha256,
        matrix_id=campaign.matrix_id,
        matrix_raw_sha256=campaign.raw_sha256,
        matrix_payload_sha256=campaign.payload_sha256,
        matrix_expanded_semantic_sha256=campaign.expanded_semantic_sha256,
        matrix_accuracy_policy_id=str(
            campaign.payload["accuracy_policy"]["policy_id"]
        ),
        matrix_accuracy_policy_sha256=campaign.accuracy_policy_spec_sha256,
        accuracy_policy_mode=canonical_policy.mode,
        accuracy_policy=policy_payload,
        accuracy_policy_sha256=_canonical_sha(policy_payload),
        qiskit_compilation=False,
    )
    result.validate_against_matrix(campaign)
    return result


def seed_request(request: ExecutionRequest) -> Mapping[str, Any]:
    if request.kind != "build_seed":
        raise ProspectiveExecutionError("seed handler received another action kind")
    if float(request.inputs.get("energy_gate", math.inf)) != SEED_ENERGY_GATE:
        raise ProspectiveExecutionError("seed request changed the Delta E gate")
    return _prepared_request(request)


def exact_reference_request(request: ExecutionRequest) -> Mapping[str, Any]:
    if (
        request.kind != "build_exact_reference"
        or not request.inputs.get("reporting_only")
        or request.inputs.get("exact_data_schema") != EXACT_REFERENCE_DATA_V1
    ):
        raise ProspectiveExecutionError("invalid exact-reference request")
    return _prepared_request(request)


def trajectory_request(request: ExecutionRequest) -> Mapping[str, Any]:
    if request.kind not in {"screen_avqds", KIND_RUN}:
        raise ProspectiveExecutionError("trajectory handler received another action kind")
    if not request.exact_reference_blind or not request.inputs.get("record_statevector"):
        raise ProspectiveExecutionError("variational runs must be exact-blind statevectors")
    if request.inputs.get("controller") != "state_motion_5e-5":
        raise ProspectiveExecutionError("trajectory request changed the controller")
    return _prepared_request(request)


def score_request(request: ExecutionRequest) -> Mapping[str, Any]:
    if request.kind not in {"score_avqds", KIND_SCORE}:
        raise ProspectiveExecutionError("score handler received another action kind")
    return _prepared_request(request)


def safeguard_request(request: ExecutionRequest) -> Mapping[str, Any]:
    if request.kind != KIND_RECORD_SAFEGUARD:
        raise ProspectiveExecutionError("safeguard handler received another action kind")
    return _prepared_request(request)


def _prepared_request(request: ExecutionRequest) -> Mapping[str, Any]:
    """Return a non-successful executable request, never fake completion."""

    return {
        "status": "prepared_not_executed",
        "execution_request": request.to_json_dict(),
        "completion_receipt": None,
    }


def package_request(
    request: ExecutionRequest,
    *,
    scientific_pre_submit_audit: ScientificPreSubmitAudit,
    campaign_registry: CampaignRegistry,
    repository_root: str | Path,
) -> Mapping[str, Any]:
    """Validate a package recipe; source/workdir activation remains external."""

    if request.kind != KIND_BUILD_PACKAGE:
        raise ProspectiveExecutionError(
            "package handler received another action kind"
        )
    if not isinstance(scientific_pre_submit_audit, ScientificPreSubmitAudit):
        raise ProspectiveExecutionError(
            "package construction requires a passed scientific pre-submit audit"
        )
    root = Path(repository_root).resolve()
    current = scientific_pre_submit_audit.reload(root)
    current.verify_request(request)
    current.verify_current_registry(
        campaign_registry, repository_root=root
    )
    return {
        "status": "package_materialization_required",
        "execution_request": request.to_json_dict(),
        "scientific_pre_submit_audit": {
            "path": current.receipt_path,
            "sha256": current.receipt_sha256,
        },
        "lane_scientific_authority_sha256": (
            current.lane_scientific_authority.authority_sha256
        ),
        "completion_receipt": None,
    }


def report_slot_request(
    request: ExecutionRequest,
    *,
    scientific_pre_submit_audit: ScientificPreSubmitAudit,
    campaign_registry: CampaignRegistry,
    repository_root: str | Path,
) -> Mapping[str, Any]:
    if request.kind != KIND_UPDATE_REPORT_SLOT:
        raise ProspectiveExecutionError(
            "report-slot handler received another action kind"
        )
    if not isinstance(scientific_pre_submit_audit, ScientificPreSubmitAudit):
        raise ProspectiveExecutionError(
            "report promotion requires a passed scientific pre-submit audit"
        )
    root = Path(repository_root).resolve()
    current = scientific_pre_submit_audit.reload(root)
    current.verify_request(request)
    current.verify_current_registry(
        campaign_registry, repository_root=root
    )
    return {
        "status": "evidence_binding_required",
        "execution_request": request.to_json_dict(),
        "scientific_pre_submit_audit": {
            "path": current.receipt_path,
            "sha256": current.receipt_sha256,
        },
        "lane_scientific_authority_sha256": (
            current.lane_scientific_authority.authority_sha256
        ),
        "completion_receipt": None,
    }


@dataclass(frozen=True)
class ActionRegistryEntry:
    request: ExecutionRequest
    source_receipts: tuple[VerifiedFile, ...]
    entry_sha256: str
    file: VerifiedFile


def _typed_source_receipts(
    rows: Sequence[VerifiedFile],
    *,
    root: Path,
    expected_seed: SeedIdentity,
    expected_reference: ExactReferenceIdentity,
    expected_source_epoch: SourceEpochIdentity,
) -> tuple[SourceEpochReceipt, SeedQualificationReceipt, ExactReferenceReceipt]:
    by_schema: dict[str, VerifiedFile] = {}
    for row in rows:
        row.verify(root)
        payload = _read_object(root / row.path, label="action source receipt")
        schema = str(payload.get("schema") or "")
        if schema in by_schema:
            raise ProspectiveExecutionError("action source receipt schema repeats")
        by_schema[schema] = row
    required = {
        SOURCE_EPOCH_RECEIPT_V1,
        SEED_QUALIFICATION_RECEIPT_V1,
        EXACT_REFERENCE_RECEIPT_V1,
    }
    if set(by_schema) != required:
        raise ProspectiveExecutionError(
            "action registry requires exact source/seed/reference receipts"
        )
    source = SourceEpochReceipt.load(
        root / by_schema[SOURCE_EPOCH_RECEIPT_V1].path,
        repository_root=root,
        expected_epoch=expected_source_epoch,
    )
    seed = SeedQualificationReceipt.load(
        root / by_schema[SEED_QUALIFICATION_RECEIPT_V1].path,
        repository_root=root,
        expected_seed=expected_seed,
        expected_source_epoch=expected_source_epoch,
    )
    reference = ExactReferenceReceipt.load(
        root / by_schema[EXACT_REFERENCE_RECEIPT_V1].path,
        repository_root=root,
        expected_reference=expected_reference,
        expected_seed_identity=expected_seed,
        expected_seed_receipt=seed,
        expected_source_epoch=expected_source_epoch,
    )
    return source, seed, reference


def _validate_reference_bound_policy(
    bindings: ProspectiveScientificBindings,
    reference: ExactReferenceReceipt,
) -> None:
    """Rejoin persisted policy scales to the verified exact trajectory."""

    try:
        campaign = ParallelBenchmarkMatrix.read()
        canonical = campaign.canonical_accuracy_policy(
            bindings.lane_id,
            reference.policy_response_scales(bindings.drive),
        )
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "scientific bindings have no admissible reference response scales"
        ) from exc
    canonical_payload = canonical.to_json_dict()
    observed_payload = bindings.to_json_dict()["accuracy_policy"]
    if (
        observed_payload != canonical_payload
        or bindings.accuracy_policy_sha256 != _canonical_sha(canonical_payload)
    ):
        raise ProspectiveExecutionError(
            "scientific binding policy diverges from the exact-reference "
            "response scales"
        )


def load_action_registry_entry(
    file: VerifiedFile,
    *,
    repository_root: str | Path,
    bindings: ProspectiveScientificBindings,
    expected_seed: SeedIdentity,
    expected_reference: ExactReferenceIdentity,
    expected_source_epoch: SourceEpochIdentity,
) -> ActionRegistryEntry:
    root = Path(repository_root).resolve()
    bindings.validate_against_matrix()
    file.verify(root)
    payload = _read_object(root / file.path, label="action registry entry")
    if payload.get("schema") != ACTION_REGISTRY_ENTRY_V1:
        raise ProspectiveExecutionError("unsupported action registry entry")
    entry_sha = _verify_self_digest(
        payload, key="entry_sha256", label="action registry entry"
    )
    request_row = payload.get("execution_request")
    receipt_rows = payload.get("source_receipts")
    if not isinstance(request_row, Mapping) or not isinstance(receipt_rows, list):
        raise ProspectiveExecutionError("action registry entry is incomplete")
    request = ExecutionRequest.from_mapping(request_row)
    receipts = tuple(
        VerifiedFile(str(row.get("path") or ""), str(row.get("sha256") or ""))
        for row in receipt_rows
        if isinstance(row, Mapping)
    )
    if len(receipts) != len(receipt_rows):
        raise ProspectiveExecutionError("action registry receipt binding malformed")
    source, seed, reference = _typed_source_receipts(
        receipts,
        root=root,
        expected_seed=expected_seed,
        expected_reference=expected_reference,
        expected_source_epoch=expected_source_epoch,
    )
    _validate_reference_bound_policy(bindings, reference)
    inputs = request.inputs
    if (
        payload.get("action_id") != request.action_id
        or payload.get("lane_id") != request.lane_id
        or payload.get("kind") != request.kind
        or payload.get("bindings_sha256") != bindings.digest
        or payload.get("execution_request_sha256") != request.digest
        or payload.get("source_receipt_sha256") != source.receipt_sha256
        or payload.get("seed_receipt_sha256") != seed.receipt_sha256
        or payload.get("exact_reference_receipt_sha256")
        != reference.receipt_sha256
        or request.lane_id != bindings.lane_id
        or inputs.get("bindings_sha256") != bindings.digest
        or inputs.get("cell_id") != bindings.cell_id
        or inputs.get("seed_id") != bindings.seed_id
        or inputs.get("seed_identity_sha256") != bindings.seed_identity_sha256
        or inputs.get("seed_artifact_sha256") != bindings.seed_artifact_sha256
        or inputs.get("reference_id") != bindings.reference_id
        or inputs.get("reference_identity_sha256")
        != bindings.reference_identity_sha256
        or inputs.get("reference_manifest_sha256")
        != bindings.reference_manifest_sha256
        or inputs.get("reference_data_sha256") != bindings.reference_data_sha256
        or inputs.get("reference_receipt_sha256")
        != bindings.reference_receipt_sha256
        or inputs.get("source_epoch_identity_sha256")
        != bindings.source_epoch_identity_sha256
        or inputs.get("source_tree_sha256") != bindings.source_tree_sha256
        or source.source_epoch.identity_sha256
        != bindings.source_epoch_identity_sha256
        or source.source_tree_sha256 != bindings.source_tree_sha256
        or seed.receipt_sha256 != bindings.seed_receipt_sha256
        or reference.receipt_sha256 != bindings.reference_receipt_sha256
    ):
        raise ProspectiveExecutionError(
            "action registry entry diverges from scientific bindings"
        )
    return ActionRegistryEntry(request, receipts, entry_sha, file)


def write_action_registry_entry(
    path: str | Path,
    request: ExecutionRequest,
    *,
    source_receipts: Sequence[VerifiedFile],
    repository_root: str | Path,
    bindings: ProspectiveScientificBindings,
    expected_seed: SeedIdentity,
    expected_reference: ExactReferenceIdentity,
    expected_source_epoch: SourceEpochIdentity,
) -> Mapping[str, Any]:
    root = Path(repository_root).resolve()
    bindings.validate_against_matrix()
    source, seed_receipt, reference_receipt = _typed_source_receipts(
        tuple(source_receipts),
        root=root,
        expected_seed=expected_seed,
        expected_reference=expected_reference,
        expected_source_epoch=expected_source_epoch,
    )
    _validate_reference_bound_policy(bindings, reference_receipt)
    payload = {
        "schema": ACTION_REGISTRY_ENTRY_V1,
        "action_id": request.action_id,
        "lane_id": request.lane_id,
        "kind": request.kind,
        "bindings_sha256": bindings.digest,
        "execution_request_sha256": request.digest,
        "execution_request": request.to_json_dict(),
        "source_receipts": [row.to_json_dict() for row in source_receipts],
        "source_receipt_sha256": source.receipt_sha256,
        "seed_receipt_sha256": seed_receipt.receipt_sha256,
        "exact_reference_receipt_sha256": reference_receipt.receipt_sha256,
    }
    payload["entry_sha256"] = _canonical_sha(payload)
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    if target.exists():
        if _read_object(target, label="action registry entry") != payload:
            raise ProspectiveExecutionError("action registry identity collision")
    else:
        _atomic_write_json(target, payload)
    binding = VerifiedFile(target.relative_to(root).as_posix(), _file_sha(target))
    load_action_registry_entry(
        binding,
        repository_root=root,
        bindings=bindings,
        expected_seed=expected_seed,
        expected_reference=expected_reference,
        expected_source_epoch=expected_source_epoch,
    )
    return payload


@dataclass(frozen=True)
class ScientificState:
    """Reloaded, digest-verified durable history for one prospective cell."""

    bindings: ProspectiveScientificBindings
    action_entries: tuple[VerifiedFile, ...]
    run_lifecycle_tips: tuple[VerifiedFile, ...]
    score_receipts: tuple[VerifiedFile, ...]
    safeguard_receipts: tuple[VerifiedFile, ...]
    warnings: tuple[str, ...]
    evidence: ProspectiveEvidence
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        expected_seed: SeedIdentity,
        expected_reference: ExactReferenceIdentity,
        expected_source_epoch: SourceEpochIdentity,
    ) -> "ScientificState":
        return load_scientific_state(
            path,
            repository_root=repository_root,
            expected_seed=expected_seed,
            expected_reference=expected_reference,
            expected_source_epoch=expected_source_epoch,
        )


@dataclass(frozen=True)
class _ScientificStateComponents:
    action_entries: tuple[ActionRegistryEntry, ...]
    run_lifecycle_tips: tuple[RunLifecycleReceipt, ...]
    score_receipts: tuple[ScoreQualificationReceipt, ...]
    safeguard_receipts: tuple[SafeguardQualificationReceipt, ...]
    evidence: ProspectiveEvidence


def _planner_evidence_projection(evidence: ProspectiveEvidence) -> dict[str, Any]:
    def run_row(row: ProspectiveRunEvidence) -> dict[str, Any]:
        return {
            "action_id": row.action_id,
            "bindings_sha256": row.bindings_sha256,
            "arm": row.arm,
            "horizon": row.horizon,
            "activation_cut": row.activation_cut,
            "attempt": row.attempt,
            "planner_sequence": row.planner_sequence,
            "requested_memory_gib": row.requested_memory_gib,
            "state": row.state,
            "resume_from_action_id": row.resume_from_action_id,
            "resume_run_receipt_sha256": row.resume_run_receipt_sha256,
            "resume_run_sha256": row.resume_run_sha256,
            "run_receipt_sha256": row.run_receipt_sha256,
            "run_sha256": row.run_sha256,
            "score_state": row.score_state,
            "score_receipt_sha256": row.score_receipt_sha256,
            "score_sha256": row.score_sha256,
            "metrics": (
                None if row.metrics is None else row.metrics.to_json_dict()
            ),
            "failure_kind": row.failure_kind,
            "failure_receipt_sha256": row.failure_receipt_sha256,
            "failure_reason": row.failure_reason,
            "observed_peak_gib": row.observed_peak_gib,
        }

    return {
        "runs": [run_row(row) for row in evidence.runs],
        "safeguards": [
            {
                "safeguard_id": row.safeguard_id,
                "bindings_sha256": row.bindings_sha256,
                "reason": row.reason,
                "subject_action_id": row.subject_action_id,
                "receipt_sha256": row.receipt_sha256,
                "planner_sequence": row.planner_sequence,
            }
            for row in evidence.safeguards
        ],
    }


def _state_file_rows(payload: Mapping[str, Any], key: str) -> tuple[VerifiedFile, ...]:
    rows = payload.get(key)
    if not isinstance(rows, list):
        raise ProspectiveExecutionError(f"scientific state {key} is not a list")
    result: list[VerifiedFile] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise ProspectiveExecutionError(
                f"scientific state {key} has a malformed artifact binding"
            )
        result.append(
            VerifiedFile(str(row.get("path") or ""), str(row.get("sha256") or ""))
        )
    if len(set(result)) != len(result):
        raise ProspectiveExecutionError(f"scientific state repeats a {key} binding")
    return tuple(result)


def _state_components(
    *,
    bindings: ProspectiveScientificBindings,
    action_entries: Sequence[VerifiedFile],
    run_receipts: Sequence[RunLifecycleReceipt],
    score_receipts: Sequence[ScoreQualificationReceipt],
    safeguard_receipts: Sequence[SafeguardQualificationReceipt],
    root: Path,
    expected_seed: SeedIdentity,
    expected_reference: ExactReferenceIdentity,
    expected_source_epoch: SourceEpochIdentity,
) -> _ScientificStateComponents:
    try:
        bindings.validate_against_matrix()
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "scientific state bindings diverge from the current frozen matrix"
        ) from exc

    entries_by_execution: dict[str, ActionRegistryEntry] = {}
    seen_entry_files: set[VerifiedFile] = set()
    for row in action_entries:
        if row in seen_entry_files:
            raise ProspectiveExecutionError(
                "scientific state repeats an action entry"
            )
        seen_entry_files.add(row)
        entry = load_action_registry_entry(
            row,
            repository_root=root,
            bindings=bindings,
            expected_seed=expected_seed,
            expected_reference=expected_reference,
            expected_source_epoch=expected_source_epoch,
        )
        if entry.request.execution_id in entries_by_execution:
            raise ProspectiveExecutionError(
                "scientific state repeats an action entry execution"
            )
        entries_by_execution[entry.request.execution_id] = entry

    current_scores: list[ScoreQualificationReceipt] = []
    seen_score_files: set[VerifiedFile] = set()
    for receipt in score_receipts:
        score = receipt.reload(root)
        score.validate_bindings(root, bindings)
        file = VerifiedFile(score.receipt_path, score.receipt_sha256)
        if file in seen_score_files:
            raise ProspectiveExecutionError(
                "scientific state repeats a score receipt"
            )
        seen_score_files.add(file)
        current_scores.append(score)

    current_safeguards: list[SafeguardQualificationReceipt] = []
    seen_safeguard_files: set[VerifiedFile] = set()
    for receipt in safeguard_receipts:
        safeguard = receipt.reload(root)
        safeguard.validate_bindings(root, bindings)
        file = VerifiedFile(safeguard.receipt_path, safeguard.receipt_sha256)
        if file in seen_safeguard_files:
            raise ProspectiveExecutionError(
                "scientific state repeats a safeguard receipt"
            )
        seen_safeguard_files.add(file)
        current_safeguards.append(safeguard)

    supplied_runs = list(run_receipts)
    supplied_runs.extend(score.run_lifecycle(root) for score in current_scores)
    for safeguard in current_safeguards:
        supplied_runs.extend(
            RunLifecycleReceipt.load(root / row.path, repository_root=root)
            for row in safeguard.subject_run_receipts
        )

    by_digest: dict[str, RunLifecycleReceipt] = {}
    for receipt in supplied_runs:
        for row in _run_lifecycle_chain(receipt, root=root):
            row.validate_bindings(root, bindings)
            existing = by_digest.get(row.receipt_sha256)
            if existing is not None and existing != row:
                raise ProspectiveExecutionError(
                    "scientific state has contradictory lifecycle receipts"
                )
            by_digest[row.receipt_sha256] = row

    grouped: dict[tuple[str, int], list[RunLifecycleReceipt]] = {}
    for row in by_digest.values():
        grouped.setdefault((row.scientific_action_id, row.attempt), []).append(row)
    canonical_tips: list[RunLifecycleReceipt] = []
    for rows in grouped.values():
        extended = {
            row.previous_receipt.sha256
            for row in rows
            if row.previous_receipt is not None
            and any(
                candidate.receipt_sha256 == row.previous_receipt.sha256
                and candidate.attempt == row.attempt
                for candidate in rows
            )
        }
        tips = [row for row in rows if row.receipt_sha256 not in extended]
        if len(tips) != 1:
            raise ProspectiveExecutionError(
                "scientific state has divergent lifecycle tips"
            )
        canonical_tips.extend(tips)
    canonical_tips.sort(
        key=lambda row: (
            row.planner_sequence,
            row.scientific_action_id,
            row.attempt,
            row.receipt_path,
        )
    )

    required_action_files: set[VerifiedFile] = set()
    for run in by_digest.values():
        entry = entries_by_execution.get(run.execution_id)
        if entry is None or entry.file != run.action_registry:
            raise ProspectiveExecutionError(
                "run lifecycle receipt has no matching action-registry entry"
            )
        required_action_files.add(entry.file)
    for safeguard in current_safeguards:
        entry = entries_by_execution.get(safeguard.safeguard_action_id)
        if entry is None or entry.file != safeguard.action_registry:
            raise ProspectiveExecutionError(
                "safeguard receipt has no matching action-registry entry"
            )
        required_action_files.add(entry.file)
    orphan_entries = seen_entry_files - required_action_files
    if orphan_entries:
        raise ProspectiveExecutionError(
            "scientific state contains an orphan action entry"
        )
    if required_action_files != seen_entry_files:
        raise ProspectiveExecutionError(
            "scientific state omits a required action entry"
        )

    evidence = durable_receipts_to_evidence(
        tuple(canonical_tips),
        score_receipts=tuple(current_scores),
        safeguard_receipts=tuple(current_safeguards),
        repository_root=root,
        bindings=bindings,
    )
    entries = tuple(
        sorted(
            entries_by_execution.values(),
            key=lambda row: (
                row.request.planner_sequence,
                row.request.execution_id,
                row.file.path,
            ),
        )
    )
    current_scores.sort(
        key=lambda row: (
            row.planner_sequence,
            row.scientific_action_id,
            row.attempt,
            row.receipt_path,
        )
    )
    current_safeguards.sort(
        key=lambda row: (
            row.planner_sequence,
            row.safeguard_action_id,
            row.receipt_path,
        )
    )
    return _ScientificStateComponents(
        action_entries=entries,
        run_lifecycle_tips=tuple(canonical_tips),
        score_receipts=tuple(current_scores),
        safeguard_receipts=tuple(current_safeguards),
        evidence=evidence,
    )


def _canonical_warnings(warnings: Sequence[str], *, label: str) -> tuple[str, ...]:
    if any(not isinstance(row, str) or not row.strip() for row in warnings):
        raise ProspectiveExecutionError(f"{label} must be nonempty strings")
    result = tuple(warnings)
    if len(set(result)) != len(result):
        raise ProspectiveExecutionError(f"{label} must not repeat entries")
    return result


def _scientific_state_payload(
    bindings: ProspectiveScientificBindings,
    components: _ScientificStateComponents,
    *,
    warnings: Sequence[str],
) -> dict[str, Any]:
    payload = {
        "schema": SCIENTIFIC_STATE_V1,
        "bindings": bindings.to_json_dict(),
        "action_entries": [
            row.file.to_json_dict() for row in components.action_entries
        ],
        "run_lifecycle_tips": [
            VerifiedFile(row.receipt_path, row.receipt_sha256).to_json_dict()
            for row in components.run_lifecycle_tips
        ],
        "score_receipts": [
            VerifiedFile(row.receipt_path, row.receipt_sha256).to_json_dict()
            for row in components.score_receipts
        ],
        "safeguard_receipts": [
            VerifiedFile(row.receipt_path, row.receipt_sha256).to_json_dict()
            for row in components.safeguard_receipts
        ],
        "warnings": list(warnings),
        "planner_evidence": _planner_evidence_projection(components.evidence),
        "qiskit_compilation": False,
    }
    payload["scientific_state_sha256"] = _canonical_sha(payload)
    return payload


def write_scientific_state(
    path: str | Path,
    *,
    bindings: ProspectiveScientificBindings,
    action_entries: Sequence[VerifiedFile],
    score_receipts: Sequence[ScoreQualificationReceipt],
    repository_root: str | Path,
    expected_seed: SeedIdentity,
    expected_reference: ExactReferenceIdentity,
    expected_source_epoch: SourceEpochIdentity,
    run_receipts: Sequence[RunLifecycleReceipt] = (),
    safeguard_receipts: Sequence[SafeguardQualificationReceipt] = (),
    warnings: Sequence[str] = (),
) -> Mapping[str, Any]:
    root = Path(repository_root).resolve()
    canonical_warnings = _canonical_warnings(
        warnings, label="scientific state warnings"
    )
    components = _state_components(
        bindings=bindings,
        action_entries=action_entries,
        run_receipts=run_receipts,
        score_receipts=score_receipts,
        safeguard_receipts=safeguard_receipts,
        root=root,
        expected_seed=expected_seed,
        expected_reference=expected_reference,
        expected_source_epoch=expected_source_epoch,
    )
    payload = _scientific_state_payload(
        bindings, components, warnings=canonical_warnings
    )
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "scientific state target is outside repository"
        ) from exc
    _atomic_write_json(target, payload)
    return payload


def load_scientific_state(
    path: str | Path,
    *,
    repository_root: str | Path,
    expected_seed: SeedIdentity,
    expected_reference: ExactReferenceIdentity,
    expected_source_epoch: SourceEpochIdentity,
) -> ScientificState:
    """Reload every durable artifact and authenticate its planner projection."""

    root = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "scientific state is outside repository"
        ) from exc
    payload = _read_object(source, label="scientific state")
    required = {
        "schema",
        "bindings",
        "action_entries",
        "run_lifecycle_tips",
        "score_receipts",
        "safeguard_receipts",
        "warnings",
        "planner_evidence",
        "qiskit_compilation",
        "scientific_state_sha256",
    }
    if set(payload) != required or payload.get("schema") != SCIENTIFIC_STATE_V1:
        raise ProspectiveExecutionError("unsupported scientific state schema")
    self_sha = _verify_self_digest(
        payload, key="scientific_state_sha256", label="scientific state"
    )
    if payload.get("qiskit_compilation") is not False:
        raise ProspectiveExecutionError(
            "Qiskit compilation is excluded from scientific state"
        )
    bindings_row = payload.get("bindings")
    if not isinstance(bindings_row, Mapping):
        raise ProspectiveExecutionError("scientific state omits bindings")
    try:
        bindings = ProspectiveScientificBindings.from_json_dict(bindings_row)
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "scientific state has invalid matrix-bound bindings"
        ) from exc

    action_files = _state_file_rows(payload, "action_entries")
    run_files = _state_file_rows(payload, "run_lifecycle_tips")
    score_files = _state_file_rows(payload, "score_receipts")
    safeguard_files = _state_file_rows(payload, "safeguard_receipts")
    warnings_row = payload.get("warnings")
    if not isinstance(warnings_row, list):
        raise ProspectiveExecutionError("scientific state warnings is not a list")
    warnings = _canonical_warnings(
        warnings_row, label="scientific state warnings"
    )

    runs: list[RunLifecycleReceipt] = []
    for file in run_files:
        file.verify(root)
        receipt = RunLifecycleReceipt.load(root / file.path, repository_root=root)
        if receipt.receipt_sha256 != file.sha256:
            raise ProspectiveExecutionError(
                "scientific state run lifecycle digest mismatch"
            )
        runs.append(receipt)
    scores: list[ScoreQualificationReceipt] = []
    for file in score_files:
        file.verify(root)
        receipt = ScoreQualificationReceipt.load(
            root / file.path, repository_root=root
        )
        if receipt.receipt_sha256 != file.sha256:
            raise ProspectiveExecutionError(
                "scientific state score receipt digest mismatch"
            )
        scores.append(receipt)
    safeguards: list[SafeguardQualificationReceipt] = []
    for file in safeguard_files:
        file.verify(root)
        receipt = SafeguardQualificationReceipt.load(
            root / file.path, repository_root=root
        )
        if receipt.receipt_sha256 != file.sha256:
            raise ProspectiveExecutionError(
                "scientific state safeguard receipt digest mismatch"
            )
        safeguards.append(receipt)

    components = _state_components(
        bindings=bindings,
        action_entries=action_files,
        run_receipts=tuple(runs),
        score_receipts=tuple(scores),
        safeguard_receipts=tuple(safeguards),
        root=root,
        expected_seed=expected_seed,
        expected_reference=expected_reference,
        expected_source_epoch=expected_source_epoch,
    )
    canonical = _scientific_state_payload(
        bindings, components, warnings=warnings
    )
    if payload.get("planner_evidence") != canonical["planner_evidence"]:
        raise ProspectiveExecutionError(
            "scientific state planner evidence projection mismatch"
        )
    if payload != canonical:
        raise ProspectiveExecutionError(
            "scientific state canonical artifact projection mismatch"
        )
    return ScientificState(
        bindings=bindings,
        action_entries=tuple(row.file for row in components.action_entries),
        run_lifecycle_tips=tuple(
            VerifiedFile(row.receipt_path, row.receipt_sha256)
            for row in components.run_lifecycle_tips
        ),
        score_receipts=tuple(
            VerifiedFile(row.receipt_path, row.receipt_sha256)
            for row in components.score_receipts
        ),
        safeguard_receipts=tuple(
            VerifiedFile(row.receipt_path, row.receipt_sha256)
            for row in components.safeguard_receipts
        ),
        warnings=warnings,
        evidence=components.evidence,
        receipt_path=relative,
        receipt_sha256=_file_sha(source),
        receipt_self_sha256=self_sha,
    )


def score_receipts_to_evidence(
    receipts: Sequence[ScoreQualificationReceipt],
    *,
    repository_root: str | Path,
    bindings: ProspectiveScientificBindings,
    safeguard_receipts: Sequence[SafeguardQualificationReceipt] = (),
) -> ProspectiveEvidence:
    """Convert digest-verified score receipts into frozen planner evidence."""

    return durable_receipts_to_evidence(
        tuple(row.run_lifecycle(Path(repository_root).resolve()) for row in receipts),
        score_receipts=receipts,
        repository_root=repository_root,
        bindings=bindings,
        safeguard_receipts=safeguard_receipts,
    )


def _run_lifecycle_chain(
    tip: RunLifecycleReceipt, *, root: Path
) -> tuple[RunLifecycleReceipt, ...]:
    chain: list[RunLifecycleReceipt] = []
    seen: set[str] = set()
    current = tip.reload(root)
    while True:
        if current.receipt_sha256 in seen:
            raise ProspectiveExecutionError("run lifecycle receipt chain is cyclic")
        seen.add(current.receipt_sha256)
        chain.append(current)
        if current.previous_receipt is None:
            break
        current = RunLifecycleReceipt.load(
            root / current.previous_receipt.path, repository_root=root
        )
        if current.receipt_sha256 != chain[-1].previous_receipt.sha256:
            raise ProspectiveExecutionError(
                "previous run lifecycle receipt digest mismatch"
            )
    return tuple(reversed(chain))


def durable_receipts_to_evidence(
    run_receipts: Sequence[RunLifecycleReceipt],
    *,
    score_receipts: Sequence[ScoreQualificationReceipt] = (),
    repository_root: str | Path,
    bindings: ProspectiveScientificBindings,
    safeguard_receipts: Sequence[SafeguardQualificationReceipt] = (),
) -> ProspectiveEvidence:
    """Rebuild planner evidence from immutable lifecycle tips and scores.

    Every supplied tip pulls in its verified predecessor chain.  Exactly one
    unextended tip may exist for each scientific action attempt; divergent
    scheduler histories therefore fail closed instead of being ordered by
    whichever file happened to be read last.
    """

    root = Path(repository_root).resolve()
    by_digest: dict[str, RunLifecycleReceipt] = {}
    for receipt in run_receipts:
        for row in _run_lifecycle_chain(receipt, root=root):
            row.validate_bindings(root, bindings)
            existing = by_digest.get(row.receipt_sha256)
            if existing is not None and existing != row:
                raise ProspectiveExecutionError(
                    "run lifecycle digest identifies contradictory receipts"
                )
            by_digest[row.receipt_sha256] = row

    grouped: dict[tuple[str, int], list[RunLifecycleReceipt]] = {}
    for row in by_digest.values():
        grouped.setdefault((row.scientific_action_id, row.attempt), []).append(row)
    tips: dict[tuple[str, int], RunLifecycleReceipt] = {}
    for identity, rows in grouped.items():
        extended = {
            row.previous_receipt.sha256
            for row in rows
            if row.previous_receipt is not None
            and any(
                candidate.receipt_sha256 == row.previous_receipt.sha256
                and candidate.attempt == row.attempt
                for candidate in rows
            )
        }
        candidates = [row for row in rows if row.receipt_sha256 not in extended]
        if len(candidates) != 1:
            raise ProspectiveExecutionError(
                "run lifecycle evidence has divergent attempt tips"
            )
        tips[identity] = candidates[0]

    evidence_by_identity = {
        identity: tip.to_planner_evidence(root, bindings)
        for identity, tip in tips.items()
    }
    seen_scores: set[tuple[str, int]] = set()
    for score_receipt in score_receipts:
        score = score_receipt.reload(root)
        score.validate_bindings(root, bindings)
        identity = (score.scientific_action_id, score.attempt)
        if identity in seen_scores:
            raise ProspectiveExecutionError(
                "score receipts repeat one execution attempt"
            )
        seen_scores.add(identity)
        tip = tips.get(identity)
        if (
            tip is None
            or tip.state != STATE_COMPLETED
            or tip.receipt_sha256 != score.run_receipt.sha256
            or tip.action_registry != score.action_registry
        ):
            raise ProspectiveExecutionError(
                "score receipt has no matching completed lifecycle tip"
            )
        evidence_by_identity[identity] = score.to_planner_evidence(root, bindings)
    runs = tuple(
        sorted(
            evidence_by_identity.values(),
            key=lambda row: (row.planner_sequence, row.action_id, row.attempt),
        )
    )
    safeguards: list[ProspectiveSafeguardEvidence] = []
    seen_safeguards: set[str] = set()
    score_digests = {row.reload(root).receipt_sha256 for row in score_receipts}
    for safeguard_receipt in safeguard_receipts:
        safeguard = safeguard_receipt.reload(root)
        safeguard.validate_bindings(root, bindings)
        if safeguard.safeguard_action_id in seen_safeguards:
            raise ProspectiveExecutionError(
                "safeguard receipts repeat one canonical safeguard"
            )
        seen_safeguards.add(safeguard.safeguard_action_id)
        subject_rows = [
            (identity, tip)
            for identity, tip in tips.items()
            if identity[0] == safeguard.subject_action_id
        ]
        if not subject_rows:
            raise ProspectiveExecutionError(
                "safeguard receipt has no subject run evidence"
            )
        tip_bindings = {
            VerifiedFile(tip.receipt_path, tip.receipt_sha256)
            for _identity, tip in subject_rows
        }
        if set(safeguard.subject_run_receipts) != tip_bindings:
            raise ProspectiveExecutionError(
                "safeguard receipt does not bind every subject attempt tip"
            )
        if (
            safeguard.subject_score_receipt is not None
            and safeguard.subject_score_receipt.sha256 not in score_digests
        ):
            raise ProspectiveExecutionError(
                "safeguard receipt has an orphan subject score"
            )
        safeguards.append(safeguard.to_planner_evidence(root, bindings))
    safeguards.sort(key=lambda row: (row.planner_sequence, row.safeguard_id))
    return ProspectiveEvidence(runs=runs, safeguards=tuple(safeguards))


def _terminal_scientific_execution_request(
    kind: str,
    bindings: ProspectiveScientificBindings,
    *,
    scientific_state: ScientificState,
    planned_report_contract: VerifiedFile,
    lane_scientific_authority: LaneScientificAuthority,
) -> ExecutionRequest:
    if kind not in {KIND_BUILD_PACKAGE, KIND_UPDATE_REPORT_SLOT}:
        raise ProspectiveExecutionError("unsupported terminal scientific request")
    if (
        scientific_state.bindings.digest != bindings.digest
        or scientific_state.bindings.to_json_dict() != bindings.to_json_dict()
    ):
        raise ProspectiveExecutionError(
            "terminal request scientific state crosses bindings"
        )
    state_file = VerifiedFile(
        scientific_state.receipt_path, scientific_state.receipt_sha256
    )
    if (
        lane_scientific_authority.lane_id != bindings.lane_id
        or state_file.path
        != lane_scientific_authority.scientific_state_path
        or planned_report_contract.path
        != lane_scientific_authority.planned_report_contract_path
        or planned_report_contract.sha256
        != lane_scientific_authority.planned_report_contract_file_sha256
        or (
            lane_scientific_authority.matrix_raw_sha256,
            lane_scientific_authority.matrix_payload_sha256,
            lane_scientific_authority.matrix_expanded_semantic_sha256,
            lane_scientific_authority.matrix_accuracy_policy_sha256,
        )
        != (
            bindings.matrix_raw_sha256,
            bindings.matrix_payload_sha256,
            bindings.matrix_expanded_semantic_sha256,
            bindings.matrix_accuracy_policy_sha256,
        )
    ):
        raise ProspectiveExecutionError(
            "terminal request evidence diverges from lane scientific authority"
        )
    authority_payload = lane_scientific_authority.to_json_dict()
    sequence = 1 + max(
        (
            row.planner_sequence
            for row in (
                *scientific_state.evidence.runs,
                *scientific_state.evidence.safeguards,
            )
        ),
        default=-1,
    )
    identity_payload = {
        "kind": kind,
        "bindings_sha256": bindings.digest,
        "scientific_state": state_file.to_json_dict(),
        "planned_report_contract": planned_report_contract.to_json_dict(),
        "lane_scientific_authority": authority_payload,
    }
    action_id = (
        f"prospective::{bindings.lane_id}::{bindings.cell_id}::{kind}::"
        f"evidence{_canonical_sha(identity_payload)[:16]}"
    )
    binding_id = (
        EXEC_PACKAGE_BUILDER
        if kind == KIND_BUILD_PACKAGE
        else EXEC_REPORT_SLOT
    )
    return _request(
        action_id=action_id,
        lane_id=bindings.lane_id,
        kind=kind,
        binding=_binding(binding_id),
        inputs={
            **bindings.to_json_dict(),
            "audit_scope": kind,
            "scientific_state": state_file.to_json_dict(),
            "planned_report_contract": planned_report_contract.to_json_dict(),
            "lane_scientific_authority": authority_payload,
            "production_submission_enabled": False,
        },
        exact_reference_blind=False,
        execution_id=action_id,
        attempt=0,
        planner_sequence=sequence,
        requested_memory_gib=None,
        arm=None,
    )


def _current_terminal_lane_scientific_authority(
    campaign_registry: CampaignRegistry,
    *,
    bindings: ProspectiveScientificBindings,
    scientific_state: ScientificState,
    planned_report_contract: VerifiedFile,
) -> LaneScientificAuthority:
    """Resolve and verify the current registry authority for a terminal request."""

    if not isinstance(campaign_registry, CampaignRegistry):
        raise ProspectiveExecutionError(
            "terminal request requires a current campaign registry"
        )
    try:
        authority = campaign_registry.lane_scientific_authority(
            bindings.lane_id
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "terminal request has no valid lane scientific authority"
        ) from exc

    state_file = VerifiedFile(
        scientific_state.receipt_path, scientific_state.receipt_sha256
    )
    if (
        state_file.path != authority.scientific_state_path
        or planned_report_contract.path
        != authority.planned_report_contract_path
        or planned_report_contract.sha256
        != authority.planned_report_contract_file_sha256
    ):
        raise ProspectiveExecutionError(
            "terminal request evidence diverges from lane scientific authority"
        )

    root = campaign_registry.repository_root.resolve()
    state_file.verify(root)
    planned_report_contract.verify(root)
    report_payload = _read_object(
        root / planned_report_contract.path,
        label="planned report contract",
    )
    report_matrix = report_payload.get("matrix")
    if not isinstance(report_matrix, Mapping):
        raise ProspectiveExecutionError(
            "terminal request planned report has no matrix binding"
        )
    try:
        report_self_sha = _sha(
            report_payload.get("report_contract_sha256"),
            label="planned report contract self digest",
        )
        report_tuple_sha = _sha(
            report_payload.get("tuple_structure_sha256"),
            label="planned report contract tuple digest",
        )
    except ProspectiveExecutionError as exc:
        raise ProspectiveExecutionError(
            "terminal request planned report has invalid digests"
        ) from exc
    if (
        report_payload.get("lane_id") != bindings.lane_id
        or report_self_sha != authority.planned_report_contract_sha256
        or report_tuple_sha
        != authority.planned_report_tuple_structure_sha256
        or (
            report_matrix.get("raw_sha256"),
            report_matrix.get("payload_sha256"),
            report_matrix.get("expanded_semantic_sha256"),
        )
        != (
            authority.matrix_raw_sha256,
            authority.matrix_payload_sha256,
            authority.matrix_expanded_semantic_sha256,
        )
        or (
            bindings.matrix_raw_sha256,
            bindings.matrix_payload_sha256,
            bindings.matrix_expanded_semantic_sha256,
            bindings.matrix_accuracy_policy_sha256,
        )
        != (
            authority.matrix_raw_sha256,
            authority.matrix_payload_sha256,
            authority.matrix_expanded_semantic_sha256,
            authority.matrix_accuracy_policy_sha256,
        )
    ):
        raise ProspectiveExecutionError(
            "terminal request planned report diverges from lane scientific authority"
        )
    return authority


def build_package_execution_request(
    bindings: ProspectiveScientificBindings,
    *,
    scientific_state: ScientificState,
    planned_report_contract: VerifiedFile,
    campaign_registry: CampaignRegistry,
) -> ExecutionRequest:
    """Construct the sole canonical package request for one state snapshot."""

    authority = _current_terminal_lane_scientific_authority(
        campaign_registry,
        bindings=bindings,
        scientific_state=scientific_state,
        planned_report_contract=planned_report_contract,
    )
    return _terminal_scientific_execution_request(
        KIND_BUILD_PACKAGE,
        bindings,
        scientific_state=scientific_state,
        planned_report_contract=planned_report_contract,
        lane_scientific_authority=authority,
    )


def update_report_slot_execution_request(
    bindings: ProspectiveScientificBindings,
    *,
    scientific_state: ScientificState,
    planned_report_contract: VerifiedFile,
    campaign_registry: CampaignRegistry,
) -> ExecutionRequest:
    """Construct the sole canonical report-slot request for one state snapshot."""

    authority = _current_terminal_lane_scientific_authority(
        campaign_registry,
        bindings=bindings,
        scientific_state=scientific_state,
        planned_report_contract=planned_report_contract,
    )
    return _terminal_scientific_execution_request(
        KIND_UPDATE_REPORT_SLOT,
        bindings,
        scientific_state=scientific_state,
        planned_report_contract=planned_report_contract,
        lane_scientific_authority=authority,
    )


@dataclass(frozen=True)
class _AuditContext:
    request: ExecutionRequest
    bindings: ProspectiveScientificBindings
    lane_scientific_authority: LaneScientificAuthority
    matrix: ParallelBenchmarkMatrix
    matrix_file: VerifiedFile
    identity_registry_sha256: str
    executor_registry_sha256: str
    planned_report_contract: VerifiedFile
    planned_report_contract_sha256: str
    scientific_state_file: VerifiedFile
    scientific_state: ScientificState
    source_epoch_files: tuple[VerifiedFile, ...]
    source_epochs: tuple[SourceEpochReceipt, ...]
    seed_files: tuple[VerifiedFile, ...]
    seeds: tuple[SeedQualificationReceipt, ...]
    reference_files: tuple[VerifiedFile, ...]
    references: tuple[ExactReferenceReceipt, ...]
    evidence_files: tuple[VerifiedFile, ...]


def _audit_file(value: Any, *, label: str) -> VerifiedFile:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ProspectiveExecutionError(
            f"scientific pre-submit audit has invalid {label} binding"
        )
    return VerifiedFile(
        str(value.get("path") or ""), str(value.get("sha256") or "")
    )


def _audit_file_list(value: Any, *, label: str) -> tuple[VerifiedFile, ...]:
    if not isinstance(value, list):
        raise ProspectiveExecutionError(
            f"scientific pre-submit audit {label} is not a list"
        )
    rows = tuple(_audit_file(row, label=label) for row in value)
    if len(set(rows)) != len(rows):
        raise ProspectiveExecutionError(
            f"scientific pre-submit audit repeats {label}"
        )
    return rows


@dataclass(frozen=True)
class _ReportActionContext:
    bindings: ProspectiveScientificBindings
    action: ActionRegistryEntry
    source: SourceEpochReceipt
    seed: SeedQualificationReceipt
    reference: ExactReferenceReceipt


def _report_binding(value: Any, *, label: str) -> VerifiedFile:
    if not isinstance(value, Mapping):
        raise ProspectiveExecutionError(
            f"populated report slot has no {label} binding"
        )
    return VerifiedFile(
        str(value.get("path") or ""), str(value.get("sha256") or "")
    )


def _report_action_context(
    *,
    root: Path,
    matrix: ParallelBenchmarkMatrix,
    cell_id: str,
    file: VerifiedFile,
) -> _ReportActionContext:
    file.verify(root)
    payload = _read_object(root / file.path, label="report-slot action registry")
    request_row = payload.get("execution_request")
    if not isinstance(request_row, Mapping):
        raise ProspectiveExecutionError(
            "populated report slot action registry has no execution request"
        )
    inputs = request_row.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ProspectiveExecutionError(
            "populated report slot action registry has no scientific bindings"
        )
    try:
        bindings = ProspectiveScientificBindings.from_json_dict(inputs)
        bindings.validate_against_matrix(matrix)
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "populated report slot action registry is outside the current matrix"
        ) from exc
    if bindings.cell_id != cell_id:
        raise ProspectiveExecutionError(
            "populated report slot action registry crosses matrix cells"
        )
    identities = build_identity_registry(matrix)
    references = tuple(
        row for row in identities.references if row.cell_id == cell_id
    )
    if len(references) != 1:
        raise ProspectiveExecutionError(
            "populated report slot has no canonical exact reference"
        )
    reference_identity = references[0]
    seeds = tuple(
        row for row in identities.seeds if row.seed_id == reference_identity.seed_id
    )
    if len(seeds) != 1:
        raise ProspectiveExecutionError(
            "populated report slot has no canonical seed"
        )
    receipt_files = _audit_file_list(
        payload.get("source_receipts"), label="report-slot source receipts"
    )
    source_files = []
    for receipt_file in receipt_files:
        receipt_file.verify(root)
        receipt_payload = _read_object(
            root / receipt_file.path, label="report-slot source receipt"
        )
        if receipt_payload.get("schema") == SOURCE_EPOCH_RECEIPT_V1:
            source_files.append(receipt_file)
    if len(source_files) != 1:
        raise ProspectiveExecutionError(
            "populated report slot has no unique source epoch receipt"
        )
    source = SourceEpochReceipt.load(
        root / source_files[0].path, repository_root=root
    )
    action = load_action_registry_entry(
        file,
        repository_root=root,
        bindings=bindings,
        expected_seed=seeds[0],
        expected_reference=reference_identity,
        expected_source_epoch=source.source_epoch,
    )
    typed_source, seed, reference = _typed_source_receipts(
        action.source_receipts,
        root=root,
        expected_seed=seeds[0],
        expected_reference=reference_identity,
        expected_source_epoch=source.source_epoch,
    )
    return _ReportActionContext(
        bindings=bindings,
        action=action,
        source=typed_source,
        seed=seed,
        reference=reference,
    )


def _report_score_context(
    *,
    root: Path,
    matrix: ParallelBenchmarkMatrix,
    cell_id: str,
    method: str,
    provenance: Mapping[str, Any],
    prefix: str = "",
) -> tuple[ScoreQualificationReceipt, _ReportActionContext, tuple[VerifiedFile, ...]]:
    label = f"{prefix}score_receipt" if prefix else "score_receipt"
    score_file = _report_binding(provenance.get(label), label=label)
    action_label = f"{prefix}action_registry" if prefix else "action_registry"
    action_file = _report_binding(provenance.get(action_label), label=action_label)
    score_file.verify(root)
    score = ScoreQualificationReceipt.load(
        root / score_file.path, repository_root=root
    )
    if score.receipt_sha256 != score_file.sha256:
        raise ProspectiveExecutionError(
            "populated report slot score receipt digest changed after load"
        )
    action_context = _report_action_context(
        root=root,
        matrix=matrix,
        cell_id=cell_id,
        file=action_file,
    )
    score.validate_bindings(root, action_context.bindings)
    expected_arm = ARM_APM if method == "apm" else ARM_AVQDS
    run_label = f"{prefix}run_receipt" if prefix else "run_receipt"
    expected_run = _report_binding(provenance.get(run_label), label=run_label)
    if (
        score.action_registry != action_file
        or score.run_receipt != expected_run
        or score.cell_id != cell_id
        or score.method != expected_arm
        or score.horizon != HORIZON_FULL
        or provenance.get("accuracy_policy_sha256")
        != score.accuracy_policy_sha256
        or provenance.get("matrix_accuracy_policy_sha256")
        != action_context.bindings.matrix_accuracy_policy_sha256
        or (
            not prefix
            and (
                provenance.get("selected_action_id")
                != score.scientific_action_id
                or provenance.get("planner_sequence") != score.planner_sequence
            )
        )
    ):
        raise ProspectiveExecutionError(
            "populated report slot score/run/action bindings do not join"
        )
    if not prefix:
        direct_bindings = (
            ("run", score.run),
            ("metrics", score.metrics),
            ("planner_receipt", score.planner_receipt),
            ("exact_reference_receipt", score.exact_reference_receipt),
        )
        if any(
            _report_binding(provenance.get(name), label=name) != expected
            for name, expected in direct_bindings
        ) or (
            provenance.get("exact_reference_id") != score.exact_reference_id
            or provenance.get("exact_reference_identity_sha256")
            != score.exact_reference_identity_sha256
        ):
            raise ProspectiveExecutionError(
                "populated report slot score provenance is stale"
            )

    files: list[VerifiedFile] = [score_file, action_file]
    files.extend(action_context.action.source_receipts)
    files.extend(
        (
            VerifiedFile(
                action_context.source.receipt_path,
                action_context.source.receipt_sha256,
            ),
            action_context.source.source_tree_manifest,
            VerifiedFile(
                action_context.seed.receipt_path,
                action_context.seed.receipt_sha256,
            ),
            action_context.seed.artifact,
            VerifiedFile(
                action_context.reference.receipt_path,
                action_context.reference.receipt_sha256,
            ),
            action_context.reference.manifest,
            action_context.reference.data,
            score.run_receipt,
            score.run,
            score.metrics,
            score.planner_receipt,
            score.exact_reference_receipt,
        )
    )
    lifecycle = score.run_lifecycle(root)
    for run in _run_lifecycle_chain(lifecycle, root=root):
        files.extend(
            file
            for file in (
                VerifiedFile(run.receipt_path, run.receipt_sha256),
                run.action_registry,
                run.state_evidence,
                run.previous_receipt,
                run.run,
                run.failure_evidence,
            )
            if file is not None
        )
    return score, action_context, tuple(files)


def _validate_report_slot_audit(
    *,
    root: Path,
    matrix: ParallelBenchmarkMatrix,
    audit_file: VerifiedFile,
    action_context: _ReportActionContext,
    scores: Sequence[ScoreQualificationReceipt],
    safeguard: SafeguardQualificationReceipt | None,
) -> tuple[ScientificPreSubmitAudit, tuple[VerifiedFile, ...]]:
    audit_file.verify(root)
    audit = ScientificPreSubmitAudit.load(
        root / audit_file.path, repository_root=root
    )
    identities = build_identity_registry(matrix)
    if (
        audit.receipt_sha256 != audit_file.sha256
        or audit.audit_scope != KIND_UPDATE_REPORT_SLOT
        or audit.bindings_sha256 != action_context.bindings.digest
        or audit.lane_id != action_context.bindings.lane_id
        or audit.matrix_raw_sha256 != matrix.raw_sha256
        or audit.matrix_payload_sha256 != matrix.payload_sha256
        or audit.matrix_expanded_semantic_sha256
        != matrix.expanded_semantic_sha256
        or audit.matrix_accuracy_policy_sha256
        != matrix.accuracy_policy_spec_sha256
        or audit.identity_registry_sha256 != identities.digest
        or audit.executor_registry_sha256
        != binding_registry_payload()["bindings_sha256"]
    ):
        raise ProspectiveExecutionError(
            "populated report slot audit does not bind the current campaign"
        )
    audit.verify_evidence(
        source_epoch_identity_sha256=(
            action_context.source.source_epoch.identity_sha256
        ),
        seed_receipt_sha256=action_context.seed.receipt_sha256,
        exact_reference_receipt_sha256=action_context.reference.receipt_sha256,
    )
    required_actions = {score.action_registry for score in scores}
    required_runs = {score.run_receipt for score in scores}
    required_scores = {
        VerifiedFile(score.receipt_path, score.receipt_sha256) for score in scores
    }
    required_safeguards: set[VerifiedFile] = set()
    if safeguard is not None:
        required_actions.add(safeguard.action_registry)
        required_runs.update(safeguard.subject_run_receipts)
        if safeguard.subject_score_receipt is not None:
            required_scores.add(safeguard.subject_score_receipt)
        required_safeguards.add(
            VerifiedFile(safeguard.receipt_path, safeguard.receipt_sha256)
        )
    if (
        not required_actions.issubset(set(audit.action_registries))
        or not required_runs.issubset(set(audit.run_lifecycle_tips))
        or not required_scores.issubset(set(audit.score_receipts))
        or not required_safeguards.issubset(set(audit.safeguard_receipts))
        or not (
            required_actions | required_runs | required_scores | required_safeguards
        ).issubset(set(audit.evidence_files))
    ):
        raise ProspectiveExecutionError(
            "populated report slot evidence is absent from its authorizing audit"
        )
    return audit, (audit_file, *audit.evidence_files)


def _report_method_slot_inventory(
    *,
    root: Path,
    matrix: ParallelBenchmarkMatrix,
    cell_id: str,
    method: str,
    provenance: Mapping[str, Any],
) -> tuple[VerifiedFile, ...]:
    primary, action_context, primary_files = _report_score_context(
        root=root,
        matrix=matrix,
        cell_id=cell_id,
        method=method,
        provenance=provenance,
    )
    scores = [primary]
    files = list(primary_files)
    paired = provenance.get("paired_score_receipt")
    if paired is not None:
        peer_method = "apm" if method == "avqds" else "avqds"
        peer, peer_context, peer_files = _report_score_context(
            root=root,
            matrix=matrix,
            cell_id=cell_id,
            method=peer_method,
            provenance=provenance,
            prefix="paired_",
        )
        if (
            peer_context.bindings.to_json_dict()
            != action_context.bindings.to_json_dict()
        ):
            raise ProspectiveExecutionError(
                "qualified report slot peers cross scientific bindings"
            )
        scores.append(peer)
        files.extend(peer_files)

    safeguard: SafeguardQualificationReceipt | None = None
    safeguard_row = provenance.get("safeguard_receipt")
    if safeguard_row is not None:
        safeguard_file = _report_binding(
            safeguard_row, label="safeguard_receipt"
        )
        safeguard_file.verify(root)
        safeguard = SafeguardQualificationReceipt.load(
            root / safeguard_file.path, repository_root=root
        )
        if (
            safeguard.receipt_sha256 != safeguard_file.sha256
            or safeguard.subject_score_receipt
            != VerifiedFile(primary.receipt_path, primary.receipt_sha256)
            or safeguard_row.get("outcome") != safeguard.outcome
        ):
            raise ProspectiveExecutionError(
                "unresolved report slot safeguard binding is stale"
            )
        safeguard.validate_bindings(root, action_context.bindings)
        files.extend(
            (
                safeguard_file,
                safeguard.action_registry,
                *safeguard.subject_run_receipts,
                safeguard.planner_receipt,
            )
        )
        if safeguard.subject_score_receipt is not None:
            files.append(safeguard.subject_score_receipt)

    audit_file = _report_binding(
        provenance.get("scientific_pre_submit_audit"),
        label="scientific_pre_submit_audit",
    )
    _audit, audit_files = _validate_report_slot_audit(
        root=root,
        matrix=matrix,
        audit_file=audit_file,
        action_context=action_context,
        scores=tuple(scores),
        safeguard=safeguard,
    )
    if provenance.get("audit_warnings") != list(_audit.warnings):
        raise ProspectiveExecutionError(
            "populated report slot warning projection is stale"
        )
    files.extend(audit_files)
    return tuple(files)


def _report_exact_slot_inventory(
    *,
    root: Path,
    matrix: ParallelBenchmarkMatrix,
    cell_id: str,
    exact: Mapping[str, Any],
) -> tuple[VerifiedFile, ...]:
    artifact = exact.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ProspectiveExecutionError(
            "populated exact slot has no evidence artifact"
        )
    source_file = _report_binding(
        artifact.get("source_epoch_receipt"),
        label="exact-slot source_epoch_receipt",
    )
    seed_file = _report_binding(
        artifact.get("seed_receipt"), label="exact-slot seed_receipt"
    )
    reference_file = _report_binding(
        artifact.get("reference_receipt"),
        label="exact-slot reference_receipt",
    )
    manifest_file = _report_binding(
        artifact.get("manifest"), label="exact-slot manifest"
    )
    data_file = _report_binding(
        artifact.get("data"), label="exact-slot data"
    )
    seed_artifact_file = _report_binding(
        artifact.get("seed_artifact"), label="exact-slot seed_artifact"
    )
    audit_file = _report_binding(
        artifact.get("scientific_pre_submit_audit"),
        label="exact-slot scientific_pre_submit_audit",
    )
    for file in (
        source_file,
        seed_file,
        reference_file,
        manifest_file,
        data_file,
        seed_artifact_file,
        audit_file,
    ):
        file.verify(root)

    identities = build_identity_registry(matrix)
    references = tuple(
        row for row in identities.references if row.cell_id == cell_id
    )
    if len(references) != 1:
        raise ProspectiveExecutionError(
            "populated exact slot has no canonical exact reference"
        )
    reference_identity = references[0]
    seeds = tuple(
        row
        for row in identities.seeds
        if row.seed_id == reference_identity.seed_id
    )
    if len(seeds) != 1:
        raise ProspectiveExecutionError(
            "populated exact slot has no canonical seed"
        )
    source = SourceEpochReceipt.load(
        root / source_file.path, repository_root=root
    )
    seed = SeedQualificationReceipt.load(
        root / seed_file.path,
        repository_root=root,
        expected_seed=seeds[0],
        expected_source_epoch=source.source_epoch,
    )
    reference = ExactReferenceReceipt.load(
        root / reference_file.path,
        repository_root=root,
        expected_reference=reference_identity,
        expected_seed_identity=seeds[0],
        expected_seed_receipt=seed,
        expected_source_epoch=source.source_epoch,
    )
    if (
        source.receipt_sha256 != source_file.sha256
        or seed.receipt_sha256 != seed_file.sha256
        or reference.receipt_sha256 != reference_file.sha256
        or seed.artifact != seed_artifact_file
        or reference.manifest != manifest_file
        or reference.data != data_file
        or artifact.get("reference_id") != reference.reference_id
        or artifact.get("reference_identity_sha256")
        != reference.reference_identity_sha256
        or artifact.get("source_epoch_identity_sha256")
        != source.source_epoch.identity_sha256
        or exact.get("reference_id") != reference.reference_id
        or exact.get("reference_identity_sha256")
        != reference.reference_identity_sha256
    ):
        raise ProspectiveExecutionError(
            "populated exact slot receipt/artifact identities do not join"
        )

    audit = ScientificPreSubmitAudit.load(
        root / audit_file.path, repository_root=root
    )
    try:
        audit_bindings = ProspectiveScientificBindings.from_json_dict(
            audit.execution_request.inputs
        )
        audit_bindings.validate_against_matrix(matrix)
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "populated exact slot audit has invalid scientific bindings"
        ) from exc
    expected_source_files = {source_file}
    expected_seed_files = {seed_file}
    expected_reference_files = {reference_file}
    expected_evidence = {
        source_file,
        source.source_tree_manifest,
        seed_file,
        seed.artifact,
        reference_file,
        reference.manifest,
        reference.data,
    }
    if (
        audit.receipt_sha256 != audit_file.sha256
        or audit.audit_scope != KIND_UPDATE_REPORT_SLOT
        or audit_bindings.cell_id != cell_id
        or audit_bindings.seed_receipt_sha256 != seed.receipt_sha256
        or audit_bindings.reference_receipt_sha256
        != reference.receipt_sha256
        or audit_bindings.source_epoch_identity_sha256
        != source.source_epoch.identity_sha256
        or audit.lane_id != reference_identity.lane_id
        or audit.matrix_raw_sha256 != matrix.raw_sha256
        or audit.matrix_payload_sha256 != matrix.payload_sha256
        or audit.matrix_expanded_semantic_sha256
        != matrix.expanded_semantic_sha256
        or audit.matrix_accuracy_policy_sha256
        != matrix.accuracy_policy_spec_sha256
        or audit.identity_registry_sha256 != identities.digest
        or audit.executor_registry_sha256
        != binding_registry_payload()["bindings_sha256"]
        or not expected_source_files.issubset(set(audit.source_epoch_receipts))
        or not expected_seed_files.issubset(set(audit.seed_receipts))
        or not expected_reference_files.issubset(
            set(audit.exact_reference_receipts)
        )
        or not expected_evidence.issubset(set(audit.evidence_files))
    ):
        raise ProspectiveExecutionError(
            "populated exact slot audit does not authorize its evidence"
        )
    audit.verify_evidence(
        source_epoch_identity_sha256=source.source_epoch.identity_sha256,
        seed_receipt_sha256=seed.receipt_sha256,
        exact_reference_receipt_sha256=reference.receipt_sha256,
    )
    return (
        source_file,
        source.source_tree_manifest,
        seed_file,
        seed.artifact,
        reference_file,
        reference.manifest,
        reference.data,
        audit_file,
        *audit.evidence_files,
    )


def _report_slot_inventory(
    *,
    root: Path,
    matrix: ParallelBenchmarkMatrix,
    report_payload: Mapping[str, Any],
) -> tuple[VerifiedFile, ...]:
    files: list[VerifiedFile] = []
    cells = report_payload.get("cells")
    if not isinstance(cells, list):
        raise ProspectiveExecutionError("planned report contract has no cells")
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ProspectiveExecutionError("planned report contract cell is malformed")
        cell_id = _token(cell.get("cell_id"), label="report cell id")
        exact = cell.get("exact_reference")
        if not isinstance(exact, Mapping):
            raise ProspectiveExecutionError(
                "planned report exact slot is malformed"
            )
        if exact.get("state") == "verified":
            files.extend(
                _report_exact_slot_inventory(
                    root=root,
                    matrix=matrix,
                    cell_id=cell_id,
                    exact=exact,
                )
            )
        slots = cell.get("method_slots")
        if not isinstance(slots, list):
            raise ProspectiveExecutionError(
                "planned report contract has no method slots"
            )
        for slot in slots:
            if not isinstance(slot, Mapping):
                raise ProspectiveExecutionError(
                    "planned report method slot is malformed"
                )
            if slot.get("state") == "pending":
                continue
            method = _token(slot.get("method"), label="report method")
            if method not in {"apm", "avqds"}:
                raise ProspectiveExecutionError("report slot method is unsupported")
            provenance = slot.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ProspectiveExecutionError(
                    "populated report slot has no evidence provenance"
                )
            files.extend(
                _report_method_slot_inventory(
                    root=root,
                    matrix=matrix,
                    cell_id=cell_id,
                    method=method,
                    provenance=provenance,
                )
            )
            history = provenance.get("superseded_candidates", [])
            if not isinstance(history, list) or any(
                not isinstance(row, Mapping) for row in history
            ):
                raise ProspectiveExecutionError(
                    "report slot superseded evidence is malformed"
                )
            for row in history:
                files.extend(
                    _report_method_slot_inventory(
                        root=root,
                        matrix=matrix,
                        cell_id=cell_id,
                        method=method,
                        provenance=row,
                    )
                )
    return tuple(files)


def _audit_inventory(
    *,
    root: Path,
    matrix_file: VerifiedFile,
    report_file: VerifiedFile,
    state_file: VerifiedFile,
    state: ScientificState,
    source_epochs: Sequence[SourceEpochReceipt],
    seeds: Sequence[SeedQualificationReceipt],
    references: Sequence[ExactReferenceReceipt],
    matrix: ParallelBenchmarkMatrix,
    report_payload: Mapping[str, Any],
) -> tuple[VerifiedFile, ...]:
    by_path: dict[str, VerifiedFile] = {}

    def add(file: VerifiedFile | None) -> None:
        if file is None:
            return
        existing = by_path.get(file.path)
        if existing is not None and existing != file:
            raise ProspectiveExecutionError(
                "audit evidence path identifies contradictory digests"
            )
        by_path[file.path] = file

    add(matrix_file)
    add(report_file)
    add(state_file)
    for source in source_epochs:
        add(VerifiedFile(source.receipt_path, source.receipt_sha256))
        add(source.source_tree_manifest)
    for seed in seeds:
        add(VerifiedFile(seed.receipt_path, seed.receipt_sha256))
        add(seed.artifact)
    for reference in references:
        add(VerifiedFile(reference.receipt_path, reference.receipt_sha256))
        add(reference.manifest)
        add(reference.data)

    for file in state.action_entries:
        add(file)
    for tip_file in state.run_lifecycle_tips:
        tip = RunLifecycleReceipt.load(root / tip_file.path, repository_root=root)
        for run in _run_lifecycle_chain(tip, root=root):
            add(VerifiedFile(run.receipt_path, run.receipt_sha256))
            add(run.action_registry)
            add(run.state_evidence)
            add(run.previous_receipt)
            add(run.run)
            add(run.failure_evidence)
    for score_file in state.score_receipts:
        score = ScoreQualificationReceipt.load(
            root / score_file.path, repository_root=root
        )
        add(VerifiedFile(score.receipt_path, score.receipt_sha256))
        add(score.run_receipt)
        add(score.action_registry)
        add(score.run)
        add(score.exact_reference_receipt)
        add(score.metrics)
        add(score.planner_receipt)
    for safeguard_file in state.safeguard_receipts:
        safeguard = SafeguardQualificationReceipt.load(
            root / safeguard_file.path, repository_root=root
        )
        add(VerifiedFile(safeguard.receipt_path, safeguard.receipt_sha256))
        add(safeguard.action_registry)
        for file in safeguard.subject_run_receipts:
            add(file)
        add(safeguard.subject_score_receipt)
        add(safeguard.planner_receipt)

    try:
        report_slot_files = _report_slot_inventory(
            root=root,
            matrix=matrix,
            report_payload=report_payload,
        )
    except OSError as exc:
        raise ProspectiveExecutionError(
            "populated report slot evidence is missing or unreadable"
        ) from exc
    for file in report_slot_files:
        add(file)

    result = tuple(sorted(by_path.values(), key=lambda row: (row.path, row.sha256)))
    for file in result:
        file.verify(root)
    return result


def _validated_audit_context(
    *,
    request: ExecutionRequest,
    scientific_state: VerifiedFile,
    planned_report_contract: VerifiedFile,
    source_epoch_receipts: Sequence[VerifiedFile],
    seed_receipts: Sequence[VerifiedFile],
    exact_reference_receipts: Sequence[VerifiedFile],
    root: Path,
    matrix_path: str | Path,
    campaign_registry: CampaignRegistry | None,
) -> _AuditContext:
    try:
        reconstructed = ExecutionRequest.from_mapping(request.to_json_dict())
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit request is not canonical"
        ) from exc
    if reconstructed != request or request.kind not in {
        KIND_BUILD_PACKAGE,
        KIND_UPDATE_REPORT_SLOT,
    }:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit has unsupported request scope"
        )
    try:
        bindings = ProspectiveScientificBindings.from_json_dict(request.inputs)
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit request has invalid bindings"
        ) from exc
    authority = _embedded_lane_scientific_authority(
        request.inputs.get("lane_scientific_authority")
    )
    if authority.lane_id != bindings.lane_id:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit lane authority crosses bindings"
        )
    if campaign_registry is not None:
        if (
            not isinstance(campaign_registry, CampaignRegistry)
            or campaign_registry.repository_root.resolve() != root
        ):
            raise ProspectiveExecutionError(
                "scientific pre-submit audit requires the current campaign registry"
            )
        try:
            current_authority = campaign_registry.lane_scientific_authority(
                bindings.lane_id
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProspectiveExecutionError(
                "scientific pre-submit audit has no current lane authority"
            ) from exc
        if current_authority != authority:
            raise ProspectiveExecutionError(
                "scientific pre-submit audit lane authority drifted"
            )
    if (
        scientific_state.path != authority.scientific_state_path
        or planned_report_contract.path
        != authority.planned_report_contract_path
        or planned_report_contract.sha256
        != authority.planned_report_contract_file_sha256
    ):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit evidence is off-registry"
        )

    source = Path(matrix_path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    try:
        matrix_relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit matrix is outside repository"
        ) from exc
    try:
        matrix = ParallelBenchmarkMatrix.read(source)
        bindings.validate_against_matrix(matrix)
    except (OSError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit matrix binding is stale"
        ) from exc
    if (
        authority.matrix_raw_sha256,
        authority.matrix_payload_sha256,
        authority.matrix_expanded_semantic_sha256,
        authority.matrix_accuracy_policy_sha256,
    ) != (
        matrix.raw_sha256,
        matrix.payload_sha256,
        matrix.expanded_semantic_sha256,
        matrix.accuracy_policy_spec_sha256,
    ):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit matrix diverges from lane authority"
        )
    matrix_file = VerifiedFile(matrix_relative, matrix.raw_sha256)
    matrix_file.verify(root)
    identities = build_identity_registry(matrix)
    expected_references = tuple(
        row for row in identities.references if row.cell_id == bindings.cell_id
    )
    if len(expected_references) != 1:
        raise ProspectiveExecutionError(
            "audit bindings name no canonical exact reference"
        )
    expected_reference = expected_references[0]
    expected_seeds = tuple(
        row for row in identities.seeds if row.seed_id == bindings.seed_id
    )
    if len(expected_seeds) != 1:
        raise ProspectiveExecutionError("audit bindings name no canonical seed")
    expected_seed = expected_seeds[0]

    source_files = tuple(source_epoch_receipts)
    seed_files = tuple(seed_receipts)
    reference_files = tuple(exact_reference_receipts)
    if any(len(rows) != 1 or len(set(rows)) != 1 for rows in (
        source_files,
        seed_files,
        reference_files,
    )):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit requires one exact source/seed/reference"
        )
    for file in (*source_files, *seed_files, *reference_files):
        file.verify(root)
    source_receipt = SourceEpochReceipt.load(
        root / source_files[0].path, repository_root=root
    )
    if (
        source_receipt.receipt_sha256 != source_files[0].sha256
        or source_receipt.source_epoch.identity_sha256
        != bindings.source_epoch_identity_sha256
        or source_receipt.source_tree_sha256 != bindings.source_tree_sha256
    ):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit source receipt is stale"
        )
    seed_receipt = SeedQualificationReceipt.load(
        root / seed_files[0].path,
        repository_root=root,
        expected_seed=expected_seed,
        expected_source_epoch=source_receipt.source_epoch,
    )
    reference_receipt = ExactReferenceReceipt.load(
        root / reference_files[0].path,
        repository_root=root,
        expected_reference=expected_reference,
        expected_seed_identity=expected_seed,
        expected_seed_receipt=seed_receipt,
        expected_source_epoch=source_receipt.source_epoch,
    )
    if (
        seed_receipt.receipt_sha256 != seed_files[0].sha256
        or reference_receipt.receipt_sha256 != reference_files[0].sha256
        or seed_receipt.receipt_sha256 != bindings.seed_receipt_sha256
        or reference_receipt.receipt_sha256
        != bindings.reference_receipt_sha256
    ):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit bootstrap receipts are stale"
        )

    scientific_state.verify(root)
    state = load_scientific_state(
        root / scientific_state.path,
        repository_root=root,
        expected_seed=expected_seed,
        expected_reference=expected_reference,
        expected_source_epoch=source_receipt.source_epoch,
    )
    if (
        state.receipt_sha256 != scientific_state.sha256
        or state.bindings.to_json_dict() != bindings.to_json_dict()
    ):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit scientific state is stale"
        )

    planned_report_contract.verify(root)
    report_payload = _read_object(
        root / planned_report_contract.path, label="planned report contract"
    )
    try:
        from pipelines.time_dynamics.planned_report_contracts import (
            validate_planned_report_contract,
        )

        validate_planned_report_contract(
            report_payload,
            repository_root=root,
            matrix_path=source,
        )
    except (OSError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit planned report contract is stale"
        ) from exc
    report_self_sha = _sha(
        report_payload.get("report_contract_sha256"),
        label="planned report contract self digest",
    )
    report_tuple_sha = _sha(
        report_payload.get("tuple_structure_sha256"),
        label="planned report contract tuple digest",
    )
    report_cells = report_payload.get("cells")
    if (
        report_payload.get("lane_id") != bindings.lane_id
        or report_self_sha != authority.planned_report_contract_sha256
        or report_tuple_sha
        != authority.planned_report_tuple_structure_sha256
        or not isinstance(report_cells, list)
        or sum(
            1
            for row in report_cells
            if isinstance(row, Mapping) and row.get("cell_id") == bindings.cell_id
        )
        != 1
        or report_payload.get("qiskit_compilation")
        != {"requested": False, "used": False, "canonical": False}
    ):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit planned report scope mismatch"
        )

    expected_source_files = {
        source_files[0], seed_files[0], reference_files[0]
    }
    for action_file in state.action_entries:
        action = load_action_registry_entry(
            action_file,
            repository_root=root,
            bindings=bindings,
            expected_seed=expected_seed,
            expected_reference=expected_reference,
            expected_source_epoch=source_receipt.source_epoch,
        )
        if set(action.source_receipts) != expected_source_files:
            raise ProspectiveExecutionError(
                "audit action registry crosses bootstrap evidence"
            )

    canonical_request = _terminal_scientific_execution_request(
        request.kind,
        bindings,
        scientific_state=state,
        planned_report_contract=planned_report_contract,
        lane_scientific_authority=authority,
    )
    if (
        request != canonical_request
        or request.inputs.get("scientific_state")
        != scientific_state.to_json_dict()
        or request.inputs.get("planned_report_contract")
        != planned_report_contract.to_json_dict()
    ):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit request is not canonical for its evidence"
        )

    executor_registry_sha256 = _sha(
        binding_registry_payload()["bindings_sha256"],
        label="executor registry",
    )
    evidence_files = _audit_inventory(
        root=root,
        matrix_file=matrix_file,
        report_file=planned_report_contract,
        state_file=scientific_state,
        state=state,
        source_epochs=(source_receipt,),
        seeds=(seed_receipt,),
        references=(reference_receipt,),
        matrix=matrix,
        report_payload=report_payload,
    )
    return _AuditContext(
        request=request,
        bindings=bindings,
        lane_scientific_authority=authority,
        matrix=matrix,
        matrix_file=matrix_file,
        identity_registry_sha256=identities.digest,
        executor_registry_sha256=executor_registry_sha256,
        planned_report_contract=planned_report_contract,
        planned_report_contract_sha256=report_self_sha,
        scientific_state_file=scientific_state,
        scientific_state=state,
        source_epoch_files=source_files,
        source_epochs=(source_receipt,),
        seed_files=seed_files,
        seeds=(seed_receipt,),
        reference_files=reference_files,
        references=(reference_receipt,),
        evidence_files=evidence_files,
    )


def _scientific_pre_submit_audit_payload(
    context: _AuditContext,
) -> dict[str, Any]:
    matrix = context.matrix
    state = context.scientific_state
    payload = {
        "schema": SCIENTIFIC_PRE_SUBMIT_AUDIT_V1,
        "status": "passed",
        "audit_scope": context.request.kind,
        "action_id": context.request.action_id,
        "execution_id": context.request.execution_id,
        "execution_request_sha256": context.request.digest,
        "execution_request": context.request.to_json_dict(),
        "bindings_sha256": context.bindings.digest,
        "lane_id": context.bindings.lane_id,
        "lane_scientific_authority": (
            context.lane_scientific_authority.to_json_dict()
        ),
        "matrix": {
            "path": context.matrix_file.path,
            "matrix_id": matrix.matrix_id,
            "raw_sha256": matrix.raw_sha256,
            "payload_sha256": matrix.payload_sha256,
            "expanded_semantic_sha256": matrix.expanded_semantic_sha256,
            "accuracy_policy_sha256": matrix.accuracy_policy_spec_sha256,
        },
        "identity_registry_sha256": context.identity_registry_sha256,
        "executor_registry_sha256": context.executor_registry_sha256,
        "planned_report_contract": (
            context.planned_report_contract.to_json_dict()
        ),
        "planned_report_contract_sha256": (
            context.planned_report_contract_sha256
        ),
        "scientific_state": context.scientific_state_file.to_json_dict(),
        "action_registries": [
            row.to_json_dict() for row in state.action_entries
        ],
        "run_lifecycle_tips": [
            row.to_json_dict() for row in state.run_lifecycle_tips
        ],
        "score_receipts": [
            row.to_json_dict() for row in state.score_receipts
        ],
        "safeguard_receipts": [
            row.to_json_dict() for row in state.safeguard_receipts
        ],
        "source_epoch_receipts": [
            row.to_json_dict() for row in context.source_epoch_files
        ],
        "seed_receipts": [
            row.to_json_dict() for row in context.seed_files
        ],
        "exact_reference_receipts": [
            row.to_json_dict() for row in context.reference_files
        ],
        "source_epoch_identity_sha256": [
            row.source_epoch.identity_sha256 for row in context.source_epochs
        ],
        "seed_receipt_sha256": [row.receipt_sha256 for row in context.seeds],
        "exact_reference_receipt_sha256": [
            row.receipt_sha256 for row in context.references
        ],
        "planner_evidence": _planner_evidence_projection(state.evidence),
        "evidence_files": [
            row.to_json_dict() for row in context.evidence_files
        ],
        "errors": [],
        "warnings": list(state.warnings),
        "qiskit_compilation": False,
    }
    payload["audit_sha256"] = _canonical_sha(payload)
    return payload


def write_scientific_pre_submit_audit(
    path: str | Path,
    *,
    request: ExecutionRequest,
    scientific_state: VerifiedFile,
    planned_report_contract: VerifiedFile,
    source_epoch_receipts: Sequence[VerifiedFile],
    seed_receipts: Sequence[VerifiedFile],
    exact_reference_receipts: Sequence[VerifiedFile],
    campaign_registry: CampaignRegistry,
    repository_root: str | Path,
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
    warnings: Sequence[str] = (),
) -> ScientificPreSubmitAudit:
    """Write the canonical, fully replayable package/report authorization."""

    root = Path(repository_root).resolve()
    if not isinstance(campaign_registry, CampaignRegistry):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit requires the current campaign registry"
        )
    context = _validated_audit_context(
        request=request,
        scientific_state=scientific_state,
        planned_report_contract=planned_report_contract,
        source_epoch_receipts=source_epoch_receipts,
        seed_receipts=seed_receipts,
        exact_reference_receipts=exact_reference_receipts,
        root=root,
        matrix_path=matrix_path,
        campaign_registry=campaign_registry,
    )
    supplied_warnings = _canonical_warnings(
        warnings, label="pre-submit audit warnings"
    )
    if supplied_warnings != context.scientific_state.warnings:
        raise ProspectiveExecutionError(
            "pre-submit audit warnings differ from the verified scientific state"
        )
    payload = _scientific_pre_submit_audit_payload(context)
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit target is outside repository"
        ) from exc
    if target.exists():
        if _read_object(target, label="scientific pre-submit audit") != payload:
            raise ProspectiveExecutionError(
                "scientific pre-submit audit identity collision"
            )
    else:
        _atomic_write_json(target, payload)
    return ScientificPreSubmitAudit.load(target, repository_root=root)


def _load_scientific_pre_submit_audit(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> ScientificPreSubmitAudit:
    root = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit is outside repository"
        ) from exc
    payload = _read_object(source, label="scientific pre-submit audit")
    required = {
        "schema",
        "status",
        "audit_scope",
        "action_id",
        "execution_id",
        "execution_request_sha256",
        "execution_request",
        "bindings_sha256",
        "lane_id",
        "lane_scientific_authority",
        "matrix",
        "identity_registry_sha256",
        "executor_registry_sha256",
        "planned_report_contract",
        "planned_report_contract_sha256",
        "scientific_state",
        "action_registries",
        "run_lifecycle_tips",
        "score_receipts",
        "safeguard_receipts",
        "source_epoch_receipts",
        "seed_receipts",
        "exact_reference_receipts",
        "source_epoch_identity_sha256",
        "seed_receipt_sha256",
        "exact_reference_receipt_sha256",
        "planner_evidence",
        "evidence_files",
        "errors",
        "warnings",
        "qiskit_compilation",
        "audit_sha256",
    }
    if set(payload) != required or payload.get("schema") != SCIENTIFIC_PRE_SUBMIT_AUDIT_V1:
        raise ProspectiveExecutionError(
            "unsupported canonical audit contract"
        )
    self_sha = _verify_self_digest(
        payload, key="audit_sha256", label="scientific pre-submit audit"
    )
    if (
        payload.get("status") != "passed"
        or payload.get("errors") != []
        or payload.get("qiskit_compilation") is not False
    ):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit did not pass or requested Qiskit"
        )
    warnings = payload.get("warnings")
    request_row = payload.get("execution_request")
    matrix_row = payload.get("matrix")
    if (
        not isinstance(warnings, list)
        or any(not isinstance(row, str) for row in warnings)
        or not isinstance(request_row, Mapping)
        or not isinstance(matrix_row, Mapping)
    ):
        raise ProspectiveExecutionError(
            "scientific pre-submit audit has malformed canonical fields"
        )
    request = ExecutionRequest.from_mapping(request_row)
    embedded_authority = _embedded_lane_scientific_authority(
        payload.get("lane_scientific_authority")
    )
    scientific_state = _audit_file(
        payload.get("scientific_state"), label="scientific state"
    )
    report = _audit_file(
        payload.get("planned_report_contract"),
        label="planned report contract",
    )
    source_files = _audit_file_list(
        payload.get("source_epoch_receipts"), label="source epoch receipts"
    )
    seed_files = _audit_file_list(
        payload.get("seed_receipts"), label="seed receipts"
    )
    reference_files = _audit_file_list(
        payload.get("exact_reference_receipts"),
        label="exact reference receipts",
    )
    matrix_path = _token(matrix_row.get("path"), label="audit matrix path")
    context = _validated_audit_context(
        request=request,
        scientific_state=scientific_state,
        planned_report_contract=report,
        source_epoch_receipts=source_files,
        seed_receipts=seed_files,
        exact_reference_receipts=reference_files,
        root=root,
        matrix_path=matrix_path,
        campaign_registry=None,
    )
    if embedded_authority != context.lane_scientific_authority:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit lane authority projection mismatch"
        )
    canonical = _scientific_pre_submit_audit_payload(context)
    if payload.get("executor_registry_sha256") != context.executor_registry_sha256:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit executor registry is stale"
        )
    if payload.get("planner_evidence") != canonical["planner_evidence"]:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit planner evidence projection is stale"
        )
    if payload.get("evidence_files") != canonical["evidence_files"]:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit evidence inventory is stale"
        )
    if payload != canonical:
        raise ProspectiveExecutionError(
            "scientific pre-submit audit canonical evidence projection mismatch"
        )
    matrix = context.matrix
    return ScientificPreSubmitAudit(
        audit_scope=request.kind,
        action_id=request.action_id,
        execution_id=request.execution_id,
        execution_request_sha256=request.digest,
        execution_request=request,
        bindings_sha256=context.bindings.digest,
        lane_id=context.bindings.lane_id,
        lane_scientific_authority=context.lane_scientific_authority,
        matrix_path=context.matrix_file.path,
        matrix_raw_sha256=matrix.raw_sha256,
        matrix_payload_sha256=matrix.payload_sha256,
        matrix_expanded_semantic_sha256=matrix.expanded_semantic_sha256,
        matrix_accuracy_policy_sha256=matrix.accuracy_policy_spec_sha256,
        identity_registry_sha256=context.identity_registry_sha256,
        executor_registry_sha256=context.executor_registry_sha256,
        planned_report_contract=context.planned_report_contract,
        planned_report_contract_sha256=(
            context.planned_report_contract_sha256
        ),
        scientific_state=context.scientific_state_file,
        action_registries=context.scientific_state.action_entries,
        run_lifecycle_tips=context.scientific_state.run_lifecycle_tips,
        score_receipts=context.scientific_state.score_receipts,
        safeguard_receipts=context.scientific_state.safeguard_receipts,
        evidence_files=context.evidence_files,
        source_epoch_receipts=context.source_epoch_files,
        seed_receipts=context.seed_files,
        exact_reference_receipts=context.reference_files,
        source_epoch_identity_sha256=tuple(
            row.source_epoch.identity_sha256 for row in context.source_epochs
        ),
        seed_receipt_sha256=tuple(row.receipt_sha256 for row in context.seeds),
        exact_reference_receipt_sha256=tuple(
            row.receipt_sha256 for row in context.references
        ),
        warnings=context.scientific_state.warnings,
        receipt_path=relative,
        receipt_sha256=_file_sha(source),
        receipt_self_sha256=self_sha,
    )


__all__ = [
    "ACTION_REGISTRY_ENTRY_V1",
    "EXECUTION_BINDING_REGISTRY_V1",
    "EXECUTION_REQUEST_V1",
    "EXEC_ACTION_REGISTRY",
    "EXEC_PACKAGE_BUILDER",
    "EXEC_PROSPECTIVE_APM",
    "EXEC_PROSPECTIVE_AVQDS",
    "EXEC_PROSPECTIVE_SAFEGUARD",
    "EXEC_PROSPECTIVE_SCORE",
    "EXEC_REPORT_SLOT",
    "EXEC_SCIENTIFIC_STATE",
    "EXACT_REFERENCE_DATA_V1",
    "EXACT_REFERENCE_MANIFEST_V1",
    "EXACT_REFERENCE_RECEIPT_V1",
    "ExactReferenceReceipt",
    "ExecutionRequest",
    "ExecutorBinding",
    "KIND_BUILD_PACKAGE",
    "KIND_UPDATE_REPORT_SLOT",
    "ProspectiveExecutionError",
    "RUN_FAILURE_EVIDENCE_V1",
    "RUN_LIFECYCLE_RECEIPT_V1",
    "RUN_STATE_EVIDENCE_V1",
    "RunLifecycleReceipt",
    "SAFEGUARD_QUALIFICATION_RECEIPT_V1",
    "SCIENTIFIC_PRE_SUBMIT_AUDIT_V1",
    "SCIENTIFIC_STATE_V1",
    "SCORE_QUALIFICATION_RECEIPT_V1",
    "SEED_ENERGY_GATE",
    "SEED_QUALIFICATION_RECEIPT_V1",
    "ScoreQualificationReceipt",
    "SafeguardQualificationReceipt",
    "ScientificPreSubmitAudit",
    "ScientificState",
    "SeedQualificationReceipt",
    "VerifiedFile",
    "binding_registry_payload",
    "build_package_execution_request",
    "build_scientific_bindings",
    "durable_receipts_to_evidence",
    "executor_bindings",
    "load_scientific_state",
    "package_request",
    "report_slot_request",
    "resolve_bootstrap_action",
    "resolve_scientific_action",
    "write_action_registry_entry",
    "write_run_lifecycle_receipt",
    "write_scientific_pre_submit_audit",
    "write_scientific_state",
    "update_report_slot_execution_request",
]
