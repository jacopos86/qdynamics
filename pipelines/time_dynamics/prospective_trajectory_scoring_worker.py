"""Digest-joined exact scoring for prospective Paper-II trajectories.

The scientific planner keeps propagation exact-reference blind.  This worker
owns the later, reporting-only join between one typed ``completed`` lifecycle,
its independently preserved ``completed_unscored`` worker receipt, and the
canonical exact-reference artifact.  It validates every identity before
loading numerical arrays, stitches a full-horizon continuation to its exact
prefix, recomputes the canonical error components, and records Qiskit-free
logical/graph resources.

One invocation scores one arm only.  Pair matching remains the responsibility
of the pure prospective planner, which is the sole authority for the frozen
bidirectional ladder.  Consequently this module never infers monotonicity and
never qualifies evidence across cells, seeds, source epochs, horizons, or
arms.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.adapters.observables import hh_observable_snapshot
from pipelines.time_dynamics.diagnostics.terminal_logical_graph_resources import (
    CANONICAL_TUPLE_ORDER,
    TERMINAL_LOGICAL_GRAPH_RESOURCE_V1,
    build_terminal_logical_graph_resource_record,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    ExactReferenceIdentity,
    LANE_HH,
    LANE_HUBBARD,
    SeedIdentity,
    SourceEpochIdentity,
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_exact_reference_worker import (
    _drive_diagonal,
    _static_hamiltonian,
    _waveform,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ACTION_REGISTRY_ENTRY_V1,
    EXACT_REFERENCE_RECEIPT_V1,
    EXEC_PROSPECTIVE_SCORE,
    SCORE_METRICS_V1,
    SCORE_QUALIFICATION_RECEIPT_V1,
    SEED_QUALIFICATION_RECEIPT_V1,
    SOURCE_EPOCH_RECEIPT_V1,
    ExactReferenceReceipt,
    ExecutionRequest,
    ProspectiveExecutionError,
    RunLifecycleReceipt,
    ScoreQualificationReceipt,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    VerifiedFile,
    executor_bindings,
    load_action_registry_entry,
    resolve_scientific_action,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ARM_APM,
    ARM_AVQDS,
    HH_HUBBARD_ACTIVATION_CUT_LADDER,
    HORIZON_FULL,
    HORIZON_PREFIX,
    KIND_SCORE,
    PROSPECTIVE_SCIENTIFIC_PLAN_V1,
    ProspectiveNextAction,
    ProspectiveScientificBindings,
    prospective_run_action_id,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    RUNNER_SCHEMA_V1,
)
from pipelines.time_dynamics.staged_campaign import AccuracyMetrics, AccuracyPolicy


SCORING_WORKER_SCHEMA_V1 = "paper_ii_prospective_trajectory_scoring_worker_v1"
APM_COMPLETION_SCHEMA_V1 = "paper_ii_apm_exchange_completion_receipt_v1"
AVQDS_COMPLETION_SCHEMA_V1 = "paper_ii_avqds_prefix_completion_receipt_v1"
SCIENTIFIC_AVQDS_COMPLETION_SCHEMA_V1 = (
    "paper_ii_avqds_trajectory_completion_receipt_v1"
)
COMPLETED_UNSCORED = "completed_unscored"

_RAW_COMPLETION_METHOD = {
    APM_COMPLETION_SCHEMA_V1: ARM_APM,
    AVQDS_COMPLETION_SCHEMA_V1: ARM_AVQDS,
    SCIENTIFIC_AVQDS_COMPLETION_SCHEMA_V1: ARM_AVQDS,
}
_SELF_DIGEST_KEYS = {
    SOURCE_EPOCH_RECEIPT_V1: "source_epoch_receipt_sha256",
    SEED_QUALIFICATION_RECEIPT_V1: "seed_receipt_sha256",
    EXACT_REFERENCE_RECEIPT_V1: "exact_reference_receipt_sha256",
}
_STATE_NORM_RTOL = 1.0e-9
_STATE_NORM_ATOL = 1.0e-11
_RECORDED_OBSERVABLE_ATOL = 1.0e-9
_TIME_ATOL = 1.0e-13
_SCOUT_ADOPTION_EVIDENCE_KIND = "verified_bootstrap_scout_evidence_reuse"


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


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_immutable(path: Path, content: bytes) -> None:
    """Install immutable evidence atomically and accept byte-identical replay."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ProspectiveExecutionError(
                f"refusing to replace immutable trajectory score evidence: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ProspectiveExecutionError(
                    f"trajectory score identity collision: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError(f"{label} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ProspectiveExecutionError(f"{label} must be a JSON object")
    return dict(value)


def _verify_self_digest(
    payload: Mapping[str, Any], key: str, *, label: str
) -> str:
    unsigned = dict(payload)
    declared = unsigned.pop(key, None)
    actual = _canonical_sha(unsigned)
    if declared != actual:
        raise ProspectiveExecutionError(f"{label} self digest mismatch")
    return actual


def _repository_path(root: Path, path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(f"{label} is outside repository") from exc
    return candidate


def _output_path(root: Path, path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(f"{label} is outside repository") from exc
    return candidate


def _verified_source_receipts(
    lifecycle: RunLifecycleReceipt,
    *,
    root: Path,
    bindings: ProspectiveScientificBindings,
    supplied_reference: VerifiedFile,
) -> tuple[
    SourceEpochReceipt,
    SeedQualificationReceipt,
    ExactReferenceReceipt,
    SeedIdentity,
    ExactReferenceIdentity,
]:
    """Reload the canonical source/seed/reference join from the run registry."""

    lifecycle.action_registry.verify(root)
    registry_payload = _read_mapping(
        root / lifecycle.action_registry.path, label="run action registry"
    )
    if registry_payload.get("schema") != ACTION_REGISTRY_ENTRY_V1:
        raise ProspectiveExecutionError("unsupported run action registry schema")
    _verify_self_digest(
        registry_payload, "entry_sha256", label="run action registry"
    )
    rows = registry_payload.get("source_receipts")
    if not isinstance(rows, list):
        raise ProspectiveExecutionError("run action registry omits source receipts")
    by_schema: dict[str, VerifiedFile] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ProspectiveExecutionError("run source receipt binding is malformed")
        file = VerifiedFile(
            str(raw.get("path") or ""), str(raw.get("sha256") or "")
        )
        file.verify(root)
        payload = _read_mapping(root / file.path, label="run source receipt")
        schema = str(payload.get("schema") or "")
        key = _SELF_DIGEST_KEYS.get(schema)
        if key is None or schema in by_schema:
            raise ProspectiveExecutionError(
                "run registry needs unique source/seed/reference receipts"
            )
        _verify_self_digest(payload, key, label="run source receipt")
        by_schema[schema] = file
    if set(by_schema) != set(_SELF_DIGEST_KEYS):
        raise ProspectiveExecutionError(
            "run registry source/seed/reference receipt set is incomplete"
        )
    if by_schema[EXACT_REFERENCE_RECEIPT_V1] != supplied_reference:
        raise ProspectiveExecutionError(
            "scoring supplied a stale or different exact-reference receipt"
        )

    source = SourceEpochReceipt.load(
        root / by_schema[SOURCE_EPOCH_RECEIPT_V1].path,
        repository_root=root,
    )
    identities = build_identity_registry()
    references = tuple(
        row for row in identities.references if row.cell_id == bindings.cell_id
    )
    if len(references) != 1:
        raise ProspectiveExecutionError("score cell has no unique exact identity")
    reference_identity = references[0]
    seeds = tuple(
        row for row in identities.seeds if row.seed_id == reference_identity.seed_id
    )
    if len(seeds) != 1:
        raise ProspectiveExecutionError("score cell has no unique seed identity")
    seed_identity = seeds[0]
    seed = SeedQualificationReceipt.load(
        root / by_schema[SEED_QUALIFICATION_RECEIPT_V1].path,
        repository_root=root,
        expected_seed=seed_identity,
        expected_source_epoch=source.source_epoch,
    )
    reference = ExactReferenceReceipt.load(
        root / by_schema[EXACT_REFERENCE_RECEIPT_V1].path,
        repository_root=root,
        expected_reference=reference_identity,
        expected_seed_identity=seed_identity,
        expected_seed_receipt=seed,
        expected_source_epoch=source.source_epoch,
    )
    load_action_registry_entry(
        lifecycle.action_registry,
        repository_root=root,
        bindings=bindings,
        expected_seed=seed_identity,
        expected_reference=reference_identity,
        expected_source_epoch=source.source_epoch,
    )
    return source, seed, reference, seed_identity, reference_identity


@dataclass(frozen=True)
class _RawCompletion:
    file: VerifiedFile
    payload: Mapping[str, Any]
    method: str
    run: VerifiedFile
    resume_run: VerifiedFile | None
    resume_receipt: VerifiedFile | None


def _optional_verified(
    payload: Mapping[str, Any], name: str
) -> VerifiedFile | None:
    raw = payload.get(name)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ProspectiveExecutionError(
            f"raw completion receipt has malformed {name}"
        )
    return VerifiedFile(str(raw.get("path") or ""), str(raw.get("sha256") or ""))


def _load_raw_completion(
    file: VerifiedFile,
    *,
    root: Path,
    lifecycle: RunLifecycleReceipt,
    bindings: ProspectiveScientificBindings,
    seed: SeedQualificationReceipt,
    seed_identity: SeedIdentity,
    source_epoch: SourceEpochIdentity,
) -> _RawCompletion:
    file.verify(root)
    payload = _read_mapping(root / file.path, label="raw completion receipt")
    schema = str(payload.get("schema") or "")
    method = _RAW_COMPLETION_METHOD.get(schema)
    if method is None:
        raise ProspectiveExecutionError("unsupported raw completion receipt schema")
    _verify_self_digest(
        payload,
        "completion_receipt_sha256",
        label="raw completion receipt",
    )
    run = _optional_verified(payload, "run")
    if run is None:
        raise ProspectiveExecutionError("raw completion receipt has no run")
    resume_run = _optional_verified(payload, "resume_run")
    resume_receipt = _optional_verified(payload, "resume_receipt")
    run.verify(root)
    if resume_run is not None:
        resume_run.verify(root)
    if resume_receipt is not None:
        resume_receipt.verify(root)
    optional_method = payload.get("method")
    optional_horizon = payload.get("horizon")
    optional_bindings = payload.get("bindings_sha256")
    direct_scientific_identity = (
        payload.get("action_id") == lifecycle.scientific_action_id
        and payload.get("execution_request_sha256")
        == lifecycle.execution_request_sha256
    )
    adoption_evidence = _read_mapping(
        root / lifecycle.state_evidence.path,
        label="completed run state evidence",
    )
    adopted_bootstrap_identity = (
        method == ARM_AVQDS
        and lifecycle.horizon == HORIZON_PREFIX
        and adoption_evidence.get("evidence_kind")
        == _SCOUT_ADOPTION_EVIDENCE_KIND
        and adoption_evidence.get("scheduler_submission_performed") is False
        and adoption_evidence.get("bootstrap_action_id")
        == payload.get("action_id")
        and adoption_evidence.get("bootstrap_execution_request_sha256")
        == payload.get("execution_request_sha256")
        and adoption_evidence.get("bootstrap_completion_receipt")
        == file.to_json_dict()
        and adoption_evidence.get("reused_run") == run.to_json_dict()
        and adoption_evidence.get("remote_raw_evidence_policy") == "preserve"
        and adoption_evidence.get("scientific_qualification") is None
    )
    if (
        payload.get("status") != COMPLETED_UNSCORED
        or not (direct_scientific_identity or adopted_bootstrap_identity)
        or payload.get("lane_id") != bindings.lane_id
        or payload.get("cell_id") != bindings.cell_id
        or method != lifecycle.method
        or (optional_method is not None and optional_method != method)
        or (
            optional_horizon is not None
            and optional_horizon != lifecycle.horizon
        )
        or (
            optional_bindings is not None
            and optional_bindings != bindings.digest
        )
        or run != lifecycle.run
        or payload.get("seed_identity_sha256") != seed_identity.identity_sha256
        or payload.get("seed_artifact_sha256") != seed.artifact.sha256
        or payload.get("seed_receipt_sha256") != seed.receipt_sha256
        or payload.get("source_epoch_identity_sha256")
        != source_epoch.identity_sha256
        or not math.isclose(
            float(payload.get("activation_cut", math.nan)),
            lifecycle.activation_cut,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or payload.get("scientific_qualification") is not None
        or payload.get("score_receipt") is not None
        or payload.get("scoring_required_before_qualification") is not True
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or payload.get("remote_cleanup_authorized") is not False
        or payload.get("qiskit_compilation_used") is not False
    ):
        raise ProspectiveExecutionError(
            "raw completion receipt diverges from the scientific lifecycle"
        )
    if lifecycle.horizon == HORIZON_PREFIX:
        if resume_run is not None or resume_receipt is not None:
            raise ProspectiveExecutionError(
                "prefix raw completion cannot carry continuation inputs"
            )
    elif (
        resume_run is None
        or resume_receipt is None
        or resume_run.sha256 != lifecycle.resume_run_sha256
        or resume_receipt.sha256 != lifecycle.resume_run_receipt_sha256
    ):
        raise ProspectiveExecutionError(
            "full-horizon raw completion lacks its digest-bound prefix"
        )
    return _RawCompletion(file, payload, method, run, resume_run, resume_receipt)


def _score_action_from_plan(
    score_request: ExecutionRequest,
    planner_receipt: VerifiedFile,
    *,
    root: Path,
    bindings: ProspectiveScientificBindings,
) -> ProspectiveNextAction:
    planner_receipt.verify(root)
    payload = _read_mapping(
        root / planner_receipt.path, label="scientific planner receipt"
    )
    if payload.get("schema") != PROSPECTIVE_SCIENTIFIC_PLAN_V1:
        raise ProspectiveExecutionError("unsupported scientific planner receipt")
    _verify_self_digest(
        payload, "planner_receipt_sha256", label="scientific planner receipt"
    )
    if payload.get("bindings_sha256") != bindings.digest:
        raise ProspectiveExecutionError("scientific planner crosses score bindings")
    rows = payload.get("next_actions")
    if not isinstance(rows, list):
        raise ProspectiveExecutionError("scientific planner has no action list")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("action_id") == score_request.action_id
    ]
    if len(matches) != 1:
        raise ProspectiveExecutionError(
            "score request is not uniquely emitted by its scientific planner"
        )
    row = matches[0]
    inputs = row.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ProspectiveExecutionError("planner score action inputs are malformed")
    try:
        action = ProspectiveNextAction(
            kind=str(row.get("kind") or ""),
            action_id=str(row.get("action_id") or ""),
            execution_id=str(row.get("execution_id") or ""),
            arm=(None if row.get("arm") is None else str(row.get("arm"))),
            horizon=(
                None if row.get("horizon") is None else str(row.get("horizon"))
            ),
            activation_cut=(
                None
                if row.get("activation_cut") is None
                else float(row.get("activation_cut"))
            ),
            ladder_index=(
                None
                if row.get("ladder_index") is None
                else int(row.get("ladder_index"))
            ),
            attempt=int(row.get("attempt")),
            planner_sequence=int(row.get("planner_sequence")),
            requested_memory_gib=(
                None
                if row.get("requested_memory_gib") is None
                else int(row.get("requested_memory_gib"))
            ),
            reason=str(row.get("reason") or ""),
            inputs=dict(inputs),
            exact_reference_blind=bool(row.get("exact_reference_blind")),
            qiskit_compilation_requested=bool(
                row.get("qiskit_compilation_requested")
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError("planner score action is malformed") from exc
    canonical = resolve_scientific_action(bindings, action)
    if canonical.to_json_dict() != score_request.to_json_dict():
        raise ProspectiveExecutionError(
            "score execution request diverges from the exact planner action"
        )
    return action


def _validate_score_request(
    score_request: ExecutionRequest,
    *,
    lifecycle: RunLifecycleReceipt,
    bindings: ProspectiveScientificBindings,
) -> None:
    binding = next(
        (
            row
            for row in executor_bindings()
            if row.binding_id == EXEC_PROSPECTIVE_SCORE
        ),
        None,
    )
    expected_inputs = {
        **bindings.to_json_dict(),
        "arm": lifecycle.method,
        "scientific_action_id": lifecycle.scientific_action_id,
        "run_receipt_sha256": lifecycle.receipt_sha256,
        "run_sha256": lifecycle.run.sha256 if lifecycle.run is not None else None,
        "reference_manifest_sha256": bindings.reference_manifest_sha256,
        "reference_data_sha256": bindings.reference_data_sha256,
        "accuracy_policy_sha256": bindings.accuracy_policy_sha256,
        "reporting_only_exact_evaluation": True,
    }
    if (
        binding is None
        or score_request.kind != KIND_SCORE
        or score_request.binding_id != EXEC_PROSPECTIVE_SCORE
        or score_request.handler != binding.handler
        or score_request.declared_outputs != ("metrics_json", "score_receipt")
        or score_request.lane_id != bindings.lane_id
        or score_request.action_id != f"score::{lifecycle.scientific_action_id}"
        or score_request.execution_id != score_request.action_id
        or score_request.arm != lifecycle.method
        or score_request.attempt != lifecycle.attempt
        or score_request.planner_sequence <= lifecycle.planner_sequence
        or score_request.requested_memory_gib is not None
        or score_request.exact_reference_blind
        or dict(score_request.inputs) != expected_inputs
    ):
        raise ProspectiveExecutionError(
            "score execution request does not exactly bind the completed run"
        )


def _expected_dimension(reference: ExactReferenceIdentity) -> int:
    sites = int(reference.physical["lattice_sites"])
    if reference.lane_id == LANE_HUBBARD:
        return 1 << (2 * sites)
    if reference.lane_id == LANE_HH:
        levels = int(reference.physical["n_ph_max"]) + 1
        phonon_qubits = max(1, (levels - 1).bit_length())
        return 1 << (2 * sites + sites * phonon_qubits)
    raise ProspectiveExecutionError("score reference has an unsupported lane")


@dataclass(frozen=True)
class _TrajectoryArrays:
    times: tuple[float, ...]
    energy: tuple[float, ...]
    doublon: tuple[float, ...]
    occupations: tuple[tuple[float, ...], ...]
    states: tuple[np.ndarray, ...]
    accepted_steps: int


def _state_from_row(
    row: Mapping[str, Any], *, expected_dimension: int, label: str
) -> np.ndarray:
    real = row.get("psi_real")
    imag = row.get("psi_imag")
    if (
        not isinstance(real, list)
        or not isinstance(imag, list)
        or len(real) != expected_dimension
        or len(imag) != expected_dimension
    ):
        raise ProspectiveExecutionError(f"{label} state dimension is malformed")
    try:
        state = np.asarray(real, dtype=float) + 1.0j * np.asarray(imag, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(f"{label} statevector is malformed") from exc
    if not np.all(np.isfinite(state)) or not math.isclose(
        float(np.real(np.vdot(state, state))),
        1.0,
        rel_tol=_STATE_NORM_RTOL,
        abs_tol=_STATE_NORM_ATOL,
    ):
        raise ProspectiveExecutionError(f"{label} statevector is invalid")
    return np.asarray(state, dtype=complex)


def _segment_times(
    reference: ExactReferenceIdentity,
    *,
    horizon: str,
    continuation: bool,
) -> tuple[float, ...]:
    grid = reference.grid
    all_times = tuple(
        float(grid["start"]) + index * float(grid["step"])
        for index in range(int(grid["points"]))
    )
    prefix_count = int(grid["prefix_points"])
    if horizon == HORIZON_PREFIX or not continuation:
        return (
            all_times[:prefix_count]
            if horizon == HORIZON_PREFIX
            else all_times
        )
    return all_times[prefix_count - 1 :]


def _load_raw_segment(
    file: VerifiedFile,
    *,
    root: Path,
    reference: ExactReferenceIdentity,
    seed: SeedQualificationReceipt,
    horizon: str,
    continuation: bool,
) -> _TrajectoryArrays:
    file.verify(root)
    payload = _read_mapping(root / file.path, label="raw trajectory")
    if payload.get("schema") != RUNNER_SCHEMA_V1:
        raise ProspectiveExecutionError("raw trajectory has an unsupported schema")
    if {
        "scientific_qualification",
        "score_receipt",
        "qualification_status",
        "exact_reference_receipt",
    }.intersection(payload):
        raise ProspectiveExecutionError("raw trajectory contains premature scoring")
    flow = payload.get("decision_data_flow")
    lock = payload.get("run_lock")
    rows = payload.get("plot_rows")
    summary = payload.get("summary")
    if (
        not isinstance(flow, Mapping)
        or not isinstance(lock, Mapping)
        or not isinstance(rows, list)
        or not isinstance(summary, Mapping)
    ):
        raise ProspectiveExecutionError("raw trajectory contract is incomplete")
    if any(
        flow.get(name) is not False
        for name in (
            "uses_reference_for_decision",
            "uses_exact_reference_for_decision",
            "uses_future_exact_forecast_for_decision",
        )
    ):
        raise ProspectiveExecutionError("raw trajectory was not exact-reference blind")
    physics = lock.get("physics")
    if not isinstance(physics, Mapping):
        raise ProspectiveExecutionError("raw trajectory physics lock is missing")
    expected_times = _segment_times(
        reference, horizon=horizon, continuation=continuation
    )
    grid = physics.get("time_grid")
    if (
        physics.get("seed_sha256") != seed.artifact.sha256
        or physics.get("exact_reference_json") is not None
        or physics.get("exact_reference_sha256") is not None
        or not isinstance(grid, Mapping)
        or int(grid.get("point_count", -1)) != len(expected_times)
        or grid.get("ordered_values_sha256") != _canonical_sha(expected_times)
        or not math.isclose(
            float(grid.get("t_initial", math.nan)),
            expected_times[0],
            rel_tol=0.0,
            abs_tol=_TIME_ATOL,
        )
        or not math.isclose(
            float(grid.get("t_final", math.nan)),
            expected_times[-1],
            rel_tol=0.0,
            abs_tol=_TIME_ATOL,
        )
        or len(rows) != len(expected_times)
    ):
        raise ProspectiveExecutionError("raw trajectory grid/horizon binding is wrong")

    dimension = _expected_dimension(reference)
    static = _static_hamiltonian(reference)
    if static.shape != (dimension, dimension):
        raise ProspectiveExecutionError("score Hamiltonian dimension is inconsistent")
    drive_diagonal = _drive_diagonal(reference, dimension)
    sites = int(reference.physical["lattice_sites"])
    ordering = str(reference.physical["ordering"])
    times: list[float] = []
    energy: list[float] = []
    doublon: list[float] = []
    occupations: list[tuple[float, ...]] = []
    states: list[np.ndarray] = []
    for index, (raw_row, expected_time) in enumerate(zip(rows, expected_times)):
        if not isinstance(raw_row, Mapping):
            raise ProspectiveExecutionError("raw trajectory row is malformed")
        try:
            time_value = float(raw_row["time"])
            recorded_energy = float(raw_row["energy_expectation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProspectiveExecutionError(
                "raw trajectory observable schema is malformed"
            ) from exc
        has_recorded_doublon = "doublon" in raw_row
        has_recorded_occupations = "site_occupations" in raw_row
        if has_recorded_doublon != has_recorded_occupations:
            raise ProspectiveExecutionError(
                "raw trajectory recorded observables are only partially present"
            )
        if has_recorded_doublon:
            try:
                recorded_doublon = float(raw_row["doublon"])
                recorded_occupations = tuple(
                    float(value) for value in raw_row["site_occupations"]
                )
            except (TypeError, ValueError) as exc:
                raise ProspectiveExecutionError(
                    "raw trajectory observable schema is malformed"
                ) from exc
        else:
            recorded_doublon = None
            recorded_occupations = None
        if (
            not math.isclose(
                time_value,
                expected_time,
                rel_tol=0.0,
                abs_tol=_TIME_ATOL,
            )
            or not math.isfinite(recorded_energy)
            or (
                recorded_occupations is not None
                and (
                    len(recorded_occupations) != sites
                    or recorded_doublon is None
                    or not all(
                        math.isfinite(value)
                        for value in (recorded_doublon, *recorded_occupations)
                    )
                )
            )
        ):
            raise ProspectiveExecutionError(
                "raw trajectory observable/grid schema is invalid"
            )
        state = _state_from_row(
            raw_row,
            expected_dimension=dimension,
            label=f"raw trajectory row {index}",
        )
        instantaneous = np.array(static, copy=True)
        diagonal_indices = np.diag_indices_from(instantaneous)
        instantaneous[diagonal_indices] += (
            _waveform(reference, time_value) * drive_diagonal
        )
        state_energy = float(np.real(np.vdot(state, instantaneous @ state)))
        snapshot = hh_observable_snapshot(
            state, num_sites=sites, ordering=ordering
        )
        state_doublon = float(snapshot["doublon"])
        state_occupations = tuple(
            float(value) for value in snapshot["site_occupations"]
        )
        if (
            not math.isclose(
                recorded_energy,
                state_energy,
                rel_tol=1.0e-9,
                abs_tol=_RECORDED_OBSERVABLE_ATOL,
            )
            or (
                recorded_doublon is not None
                and not math.isclose(
                    recorded_doublon,
                    state_doublon,
                    rel_tol=1.0e-9,
                    abs_tol=_RECORDED_OBSERVABLE_ATOL,
                )
            )
            or (
                recorded_occupations is not None
                and not np.allclose(
                    recorded_occupations,
                    state_occupations,
                    rtol=1.0e-9,
                    atol=_RECORDED_OBSERVABLE_ATOL,
                )
            )
        ):
            raise ProspectiveExecutionError(
                "raw trajectory observables disagree with its statevector"
            )
        times.append(time_value)
        energy.append(recorded_energy)
        doublon.append(
            state_doublon if recorded_doublon is None else recorded_doublon
        )
        occupations.append(
            state_occupations
            if recorded_occupations is None
            else recorded_occupations
        )
        states.append(state)
    try:
        accepted_steps = int(summary["accepted_internal_substep_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "raw trajectory lacks accepted-step telemetry"
        ) from exc
    if accepted_steps <= 0:
        raise ProspectiveExecutionError(
            "raw trajectory accepted-step telemetry must be positive"
        )
    if (
        int(summary.get("point_count", -1)) != len(expected_times)
        or not math.isclose(
            float(summary.get("time_initial", math.nan)),
            expected_times[0],
            rel_tol=0.0,
            abs_tol=_TIME_ATOL,
        )
        or not math.isclose(
            float(summary.get("time_final", math.nan)),
            expected_times[-1],
            rel_tol=0.0,
            abs_tol=_TIME_ATOL,
        )
    ):
        raise ProspectiveExecutionError("raw trajectory summary grid is inconsistent")
    return _TrajectoryArrays(
        tuple(times),
        tuple(energy),
        tuple(doublon),
        tuple(occupations),
        tuple(states),
        accepted_steps,
    )


def _join_trajectory(
    raw: _RawCompletion,
    *,
    root: Path,
    lifecycle: RunLifecycleReceipt,
    bindings: ProspectiveScientificBindings,
    reference: ExactReferenceIdentity,
    seed: SeedQualificationReceipt,
) -> _TrajectoryArrays:
    if lifecycle.horizon == HORIZON_PREFIX:
        return _load_raw_segment(
            raw.run,
            root=root,
            reference=reference,
            seed=seed,
            horizon=HORIZON_PREFIX,
            continuation=False,
        )
    if raw.resume_run is None or raw.resume_receipt is None:
        raise ProspectiveExecutionError("full score has no prefix trajectory")
    prefix_lifecycle = RunLifecycleReceipt.load(
        root / raw.resume_receipt.path, repository_root=root
    )
    prefix_lifecycle.validate_bindings(root, bindings)
    if (
        prefix_lifecycle.receipt_sha256 != raw.resume_receipt.sha256
        or prefix_lifecycle.state != "completed"
        or prefix_lifecycle.run != raw.resume_run
        or prefix_lifecycle.method != lifecycle.method
        or prefix_lifecycle.horizon != HORIZON_PREFIX
        or not math.isclose(
            prefix_lifecycle.activation_cut,
            lifecycle.activation_cut,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or prefix_lifecycle.scientific_action_id
        != lifecycle.resume_from_action_id
    ):
        raise ProspectiveExecutionError(
            "full score prefix lifecycle is not the same-arm same-cut source"
        )
    prefix = _load_raw_segment(
        raw.resume_run,
        root=root,
        reference=reference,
        seed=seed,
        horizon=HORIZON_PREFIX,
        continuation=False,
    )
    continuation = _load_raw_segment(
        raw.run,
        root=root,
        reference=reference,
        seed=seed,
        horizon=HORIZON_FULL,
        continuation=True,
    )
    if (
        not math.isclose(
            prefix.times[-1],
            continuation.times[0],
            rel_tol=0.0,
            abs_tol=_TIME_ATOL,
        )
        or not math.isclose(
            prefix.energy[-1],
            continuation.energy[0],
            rel_tol=1.0e-9,
            abs_tol=_RECORDED_OBSERVABLE_ATOL,
        )
        or not math.isclose(
            prefix.doublon[-1],
            continuation.doublon[0],
            rel_tol=1.0e-9,
            abs_tol=_RECORDED_OBSERVABLE_ATOL,
        )
        or not np.allclose(
            prefix.occupations[-1],
            continuation.occupations[0],
            rtol=1.0e-9,
            atol=_RECORDED_OBSERVABLE_ATOL,
        )
        or not np.allclose(
            prefix.states[-1],
            continuation.states[0],
            rtol=1.0e-9,
            atol=_RECORDED_OBSERVABLE_ATOL,
        )
    ):
        raise ProspectiveExecutionError(
            "full score continuation does not exactly join its prefix"
        )
    return _TrajectoryArrays(
        prefix.times + continuation.times[1:],
        prefix.energy + continuation.energy[1:],
        prefix.doublon + continuation.doublon[1:],
        prefix.occupations + continuation.occupations[1:],
        prefix.states + continuation.states[1:],
        prefix.accepted_steps + continuation.accepted_steps,
    )


def _load_exact_arrays(
    reference: ExactReferenceReceipt,
    *,
    root: Path,
    identity: ExactReferenceIdentity,
    horizon: str,
) -> _TrajectoryArrays:
    reference.data.verify(root)
    payload = _read_mapping(root / reference.data.path, label="exact trajectory")
    required = {
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
    if (
        set(payload) != required
        or payload.get("reference_identity_sha256") != identity.identity_sha256
        or payload.get("materialized_reference_key")
        != reference.materialized_reference_key
        or payload.get("observable_schema") != identity.observable_schema
        or payload.get("ordered_times_sha256")
        != identity.grid["ordered_times_sha256"]
    ):
        raise ProspectiveExecutionError(
            "exact trajectory identity/observable schema is invalid"
        )
    try:
        times_all = tuple(float(value) for value in payload["time"])
        energy_all = tuple(float(value) for value in payload["energy"])
        doublon_all = tuple(float(value) for value in payload["doublon"])
        occupations_all = tuple(
            tuple(float(value) for value in row)
            for row in payload["site_occupations"]
        )
        real_all = payload["statevector_real"]
        imag_all = payload["statevector_imag"]
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError("exact trajectory arrays are malformed") from exc
    points = int(identity.grid["points"])
    sites = int(identity.physical["lattice_sites"])
    dimension = _expected_dimension(identity)
    if (
        any(
            len(values) != points
            for values in (
                times_all,
                energy_all,
                doublon_all,
                occupations_all,
                real_all,
                imag_all,
            )
        )
        or any(len(row) != sites for row in occupations_all)
        or not all(
            math.isfinite(value)
            for value in times_all + energy_all + doublon_all
        )
        or not all(
            math.isfinite(value) for row in occupations_all for value in row
        )
    ):
        raise ProspectiveExecutionError("exact trajectory array schema is invalid")
    states = tuple(
        _state_from_row(
            {"psi_real": real, "psi_imag": imag},
            expected_dimension=dimension,
            label=f"exact trajectory row {index}",
        )
        for index, (real, imag) in enumerate(zip(real_all, imag_all))
    )
    expected_times = _segment_times(
        identity, horizon=horizon, continuation=False
    )
    count = len(expected_times)
    if (
        times_all[:count] != expected_times
        or _canonical_sha(times_all) != identity.grid["ordered_times_sha256"]
        or (
            horizon == HORIZON_PREFIX
            and _canonical_sha(times_all[:count])
            != identity.grid["ordered_prefix_sha256"]
        )
    ):
        raise ProspectiveExecutionError("exact trajectory grid/horizon is invalid")
    return _TrajectoryArrays(
        times_all[:count],
        energy_all[:count],
        doublon_all[:count],
        occupations_all[:count],
        states[:count],
        0,
    )


def _infidelity(left: np.ndarray, right: np.ndarray) -> float:
    overlap = float(abs(np.vdot(left, right)) ** 2)
    if not math.isfinite(overlap):
        raise ProspectiveExecutionError("trajectory fidelity is nonfinite")
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


def _accuracy_metrics(
    observed: _TrajectoryArrays, exact: _TrajectoryArrays
) -> tuple[AccuracyMetrics, dict[str, float]]:
    if observed.times != exact.times or len(observed.states) != len(exact.states):
        raise ProspectiveExecutionError("raw and exact trajectories have different grids")
    count = len(observed.times)
    if count < 2:
        raise ProspectiveExecutionError("trajectory scoring requires multiple points")
    energy_errors = tuple(
        abs(left - right) for left, right in zip(observed.energy, exact.energy)
    )
    doublon_errors = tuple(
        abs(left - right) for left, right in zip(observed.doublon, exact.doublon)
    )
    site_errors = tuple(
        max(abs(left - right) for left, right in zip(observed_row, exact_row))
        for observed_row, exact_row in zip(
            observed.occupations, exact.occupations
        )
    )
    infidelities = tuple(
        _infidelity(left, right)
        for left, right in zip(observed.states, exact.states)
    )
    metrics = AccuracyMetrics(
        mean_abs_energy_error=math.fsum(energy_errors) / count,
        mean_abs_doublon_error=math.fsum(doublon_errors) / count,
        mean_max_abs_site_occupation_error=math.fsum(site_errors) / count,
        mean_infidelity=math.fsum(infidelities) / count,
    )
    maxima = {
        "max_abs_energy_error": max(energy_errors),
        "max_abs_doublon_error": max(doublon_errors),
        "max_abs_site_occupation_error": max(site_errors),
        "max_infidelity": max(infidelities),
    }
    return metrics, maxima


def _response_report(
    metrics: AccuracyMetrics,
    *,
    policy: AccuracyPolicy,
    bindings: ProspectiveScientificBindings,
    method: str,
    horizon: str,
) -> dict[str, Any]:
    key = bindings.key(method)
    components = policy.response_components(metrics, key=key, horizon=horizon)
    drive_scales = policy.response_scales[bindings.drive][horizon]
    assert policy.response_relative_target is not None
    assert policy.infidelity_ceiling is not None
    target = float(policy.response_relative_target)
    raw = metrics.to_json_dict()
    observable_rows: dict[str, Any] = {}
    specifications = {
        "mean_abs_energy_error": (
            "energy_rms_excursion",
            policy.absolute_energy_error_ceiling,
        ),
        "mean_abs_doublon_error": (
            "doublon_rms_excursion",
            policy.absolute_doublon_error_ceiling,
        ),
        "mean_max_abs_site_occupation_error": (
            "max_site_occupation_rms_excursion",
            policy.absolute_site_occupation_error_ceiling,
        ),
    }
    for name, (scale_name, ceiling) in specifications.items():
        value = float(raw[name])
        scale = float(drive_scales[scale_name])
        response_error = value / max(scale, float(policy.response_scale_floor))
        observable_rows[name] = {
            "mean_absolute_error": value,
            "exact_response_scale": scale,
            "response_normalized_mean_absolute_error": response_error,
            "response_relative_target": target,
            "target_normalized_q": float(components[name]),
            "target_pass": bool(float(components[name]) <= 1.0),
            "absolute_ceiling": float(ceiling),
            "absolute_cap_pass": bool(value <= float(ceiling)),
        }
    fidelity = float(metrics.mean_infidelity)
    return {
        "observable_components": observable_rows,
        "fidelity_component": {
            "mean_infidelity": fidelity,
            "infidelity_ceiling": float(policy.infidelity_ceiling),
            "target_normalized_q": float(components["mean_infidelity"]),
            "cap_pass": bool(fidelity <= float(policy.infidelity_ceiling)),
        },
        "target_normalized_q_components": {
            name: float(value) for name, value in components.items()
        },
        "aggregate_accuracy_burden": policy.accuracy_burden(
            metrics, key=key, horizon=horizon
        ),
        "individual_qualified": policy.qualifies(
            metrics, key=key, horizon=horizon
        ),
        "comparison_guard_eligible": policy.comparison_eligible(
            metrics, key=key, horizon=horizon
        ),
        "accuracy_band": policy.band(metrics, key=key, horizon=horizon),
        "pair_matching": {
            "state": "not_evaluated_single_arm",
            "bidirectional_ladder_authority": (
                "pipelines.time_dynamics.prospective_scientific_planner"
            ),
            "matched_accuracy_factor": float(policy.matched_accuracy_factor),
            "component_deadband": float(policy.matched_component_deadband),
            "requires_same_cell_seed_source_epoch_horizon": True,
            "cost_comparison_admissible": False,
            "monotonicity_inference_used": False,
        },
    }


def _canonical_resources(
    run: VerifiedFile,
    *,
    root: Path,
    accepted_steps: int,
) -> dict[str, Any]:
    try:
        record = build_terminal_logical_graph_resource_record(root / run.path)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "cannot reconstruct canonical terminal logical/graph resources"
        ) from exc
    terminal = record.get("terminal_resource_tuple")
    qiskit = record.get("qiskit_compilation")
    if (
        record.get("schema") != TERMINAL_LOGICAL_GRAPH_RESOURCE_V1
        or tuple(record.get("canonical_tuple_order") or ())
        != CANONICAL_TUPLE_ORDER
        or not isinstance(terminal, Mapping)
        or set(terminal) != set(CANONICAL_TUPLE_ORDER)
        or not isinstance(qiskit, Mapping)
        or qiskit.get("required") is not False
        or qiskit.get("performed") is not False
    ):
        raise ProspectiveExecutionError(
            "terminal resource builder returned a noncanonical contract"
        )
    graph = {
        name: int(terminal[name])
        for name in CANONICAL_TUPLE_ORDER
    }
    if any(value < 0 for value in graph.values()):
        raise ProspectiveExecutionError("terminal graph resources must be nonnegative")
    core = {
        "schema": "paper_ii_prospective_logical_graph_cost_v1",
        "trajectory_sha256": run.sha256,
        "logical_coordinates": {
            "accepted_integration_steps": int(accepted_steps),
            "terminal_independent_parameter_count": graph["N_theta"],
        },
        "routing_independent_graph_tuple": {
            "order": list(CANONICAL_TUPLE_ORDER),
            "values": [graph[name] for name in CANONICAL_TUPLE_ORDER],
            "named": graph,
        },
        "graph_snapshot": record["terminal_graph_span_ledger"].get(
            "graph_snapshot"
        ),
        "resource_source_schema": record["schema"],
        "resource_source_sha256": record["resource_sha256"],
        "qiskit_compilation": {
            "required": False,
            "performed": False,
            "role": "excluded_from_canonical_campaign_cost",
        },
    }
    return {**core, "logical_graph_cost_sha256": _canonical_sha(core)}


def materialization_request(request: ExecutionRequest) -> Mapping[str, Any]:
    """Return the inert local recipe for one canonical score action.

    The action registry may resolve this adapter during preflight.  It performs
    no scoring, scheduler contact, or qualification; execution requires an
    explicit later call to :func:`score_completed_trajectory` with all four
    durable receipt bindings.
    """

    binding = next(
        (
            row
            for row in executor_bindings()
            if row.binding_id == EXEC_PROSPECTIVE_SCORE
        ),
        None,
    )
    try:
        bindings = ProspectiveScientificBindings.from_json_dict(request.inputs)
        arm = str(request.arm or "")
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "score materialization request has invalid scientific bindings"
        ) from exc
    if arm not in {ARM_APM, ARM_AVQDS}:
        raise ProspectiveExecutionError(
            "score materialization request has invalid arm"
        )
    scientific_action = request.inputs.get("scientific_action_id")
    candidate_coordinates = tuple(
        (candidate_horizon, candidate_cut, candidate_action)
        for candidate_horizon in (HORIZON_PREFIX, HORIZON_FULL)
        for candidate_cut in HH_HUBBARD_ACTIVATION_CUT_LADDER
        if (
            candidate_action := prospective_run_action_id(
                bindings,
                arm=arm,
                horizon=candidate_horizon,
                activation_cut=candidate_cut,
            )
        )
        == scientific_action
    )
    if len(candidate_coordinates) != 1:
        raise ProspectiveExecutionError(
            "score materialization request has no unique canonical scientific action"
        )
    horizon, cut, expected_scientific_action = candidate_coordinates[0]
    digest_fields = (
        "run_receipt_sha256",
        "run_sha256",
        "reference_manifest_sha256",
        "reference_data_sha256",
        "reference_receipt_sha256",
        "accuracy_policy_sha256",
        "source_epoch_identity_sha256",
        "seed_receipt_sha256",
    )
    if (
        binding is None
        or request.kind != KIND_SCORE
        or request.binding_id != EXEC_PROSPECTIVE_SCORE
        or request.handler != binding.handler
        or request.lane_id != bindings.lane_id
        or request.arm != arm
        or request.inputs.get("arm") != arm
        or horizon not in {HORIZON_PREFIX, HORIZON_FULL}
        or request.inputs.get("scientific_action_id")
        != expected_scientific_action
        or request.action_id != f"score::{expected_scientific_action}"
        or request.execution_id != request.action_id
        or request.declared_outputs != ("metrics_json", "score_receipt")
        or request.requested_memory_gib is not None
        or request.exact_reference_blind
        or request.inputs.get("reporting_only_exact_evaluation") is not True
        or request.inputs.get("reference_manifest_sha256")
        != bindings.reference_manifest_sha256
        or request.inputs.get("reference_data_sha256")
        != bindings.reference_data_sha256
        or request.inputs.get("reference_receipt_sha256")
        != bindings.reference_receipt_sha256
        or request.inputs.get("accuracy_policy_sha256")
        != bindings.accuracy_policy_sha256
        or request.inputs.get("source_epoch_identity_sha256")
        != bindings.source_epoch_identity_sha256
        or request.inputs.get("seed_receipt_sha256")
        != bindings.seed_receipt_sha256
        or any(
            not isinstance(request.inputs.get(name), str)
            or len(str(request.inputs[name])) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(request.inputs[name])
            )
            for name in digest_fields
        )
        or bool(
            request.inputs.get("qiskit_compilation")
            or request.inputs.get("qiskit_compilation_requested")
        )
    ):
        raise ProspectiveExecutionError(
            "request is not a canonical prospective score action"
        )
    return {
        "status": "operational_materialization_ready",
        "execution_request": request.to_json_dict(),
        "worker": (
            "pipelines.time_dynamics.prospective_trajectory_scoring_worker:"
            "score_completed_trajectory"
        ),
        "runner_module": (
            "pipelines.time_dynamics.runners.score_prospective_trajectory"
        ),
        "required_receipts": [
            "completed_run_lifecycle",
            "completed_unscored_raw_receipt",
            "exact_reference_receipt",
            "scientific_planner_receipt",
        ],
        "declared_outputs": ["metrics_json", "score_receipt"],
        "scheduler_contacted": False,
        "scheduler_submission_authorized": False,
        "scientific_pair_qualification": None,
        "qiskit_compilation_requested": False,
        "production_submission_enabled": False,
    }


def score_completed_trajectory(
    score_request: ExecutionRequest,
    *,
    bindings: ProspectiveScientificBindings,
    completed_run_receipt: VerifiedFile,
    raw_completion_receipt: VerifiedFile,
    exact_reference_receipt: VerifiedFile,
    planner_receipt: VerifiedFile,
    metrics_path: str | Path,
    score_receipt_path: str | Path,
    repository_root: str | Path,
) -> ScoreQualificationReceipt:
    """Score and immutably seal one exact-blind prospective trajectory."""

    root = Path(repository_root).resolve(strict=True)
    bindings.validate_against_matrix()
    completed_run_receipt.verify(root)
    lifecycle = RunLifecycleReceipt.load(
        root / completed_run_receipt.path, repository_root=root
    )
    if (
        lifecycle.receipt_sha256 != completed_run_receipt.sha256
        or lifecycle.state != "completed"
        or lifecycle.run is None
    ):
        raise ProspectiveExecutionError(
            "trajectory scoring requires one typed completed lifecycle"
        )
    lifecycle.validate_bindings(root, bindings)
    _validate_score_request(
        score_request, lifecycle=lifecycle, bindings=bindings
    )
    _score_action_from_plan(
        score_request,
        planner_receipt,
        root=root,
        bindings=bindings,
    )
    source, seed, reference, seed_identity, reference_identity = (
        _verified_source_receipts(
            lifecycle,
            root=root,
            bindings=bindings,
            supplied_reference=exact_reference_receipt,
        )
    )
    if (
        source.source_epoch.identity_sha256
        != bindings.source_epoch_identity_sha256
        or seed.receipt_sha256 != bindings.seed_receipt_sha256
        or reference.receipt_sha256 != bindings.reference_receipt_sha256
        or reference.manifest.sha256 != bindings.reference_manifest_sha256
        or reference.data.sha256 != bindings.reference_data_sha256
    ):
        raise ProspectiveExecutionError(
            "score source/seed/exact receipt join changed after validation"
        )
    raw = _load_raw_completion(
        raw_completion_receipt,
        root=root,
        lifecycle=lifecycle,
        bindings=bindings,
        seed=seed,
        seed_identity=seed_identity,
        source_epoch=source.source_epoch,
    )
    observed = _join_trajectory(
        raw,
        root=root,
        lifecycle=lifecycle,
        bindings=bindings,
        reference=reference_identity,
        seed=seed,
    )
    exact = _load_exact_arrays(
        reference,
        root=root,
        identity=reference_identity,
        horizon=lifecycle.horizon,
    )
    metrics, maxima = _accuracy_metrics(observed, exact)
    policy = AccuracyPolicy.from_json_dict(
        bindings.to_json_dict()["accuracy_policy"]
    )
    response = _response_report(
        metrics,
        policy=policy,
        bindings=bindings,
        method=lifecycle.method,
        horizon=lifecycle.horizon,
    )
    resources = _canonical_resources(
        raw.run,
        root=root,
        accepted_steps=observed.accepted_steps,
    )

    metrics_target = _output_path(root, metrics_path, label="score metrics target")
    receipt_target = _output_path(
        root, score_receipt_path, label="score receipt target"
    )
    if metrics_target == receipt_target:
        raise ProspectiveExecutionError("score metrics and receipt targets collide")
    metrics_relative = metrics_target.relative_to(root).as_posix()
    receipt_relative = receipt_target.relative_to(root).as_posix()
    raw_segments = [raw.run.to_json_dict()]
    if raw.resume_run is not None:
        raw_segments.insert(0, raw.resume_run.to_json_dict())
    metrics_core = {
        "schema": SCORE_METRICS_V1,
        "scoring_worker_schema": SCORING_WORKER_SCHEMA_V1,
        "score_action_id": score_request.action_id,
        "score_execution_request_sha256": score_request.digest,
        "scientific_action_id": lifecycle.scientific_action_id,
        "execution_id": lifecycle.execution_id,
        "cell_id": bindings.cell_id,
        "method": lifecycle.method,
        "horizon": lifecycle.horizon,
        "activation_cut": lifecycle.activation_cut,
        "attempt": lifecycle.attempt,
        "planner_sequence": lifecycle.planner_sequence,
        "score_planner_sequence": score_request.planner_sequence,
        "requested_memory_gib": lifecycle.requested_memory_gib,
        "run_receipt_sha256": lifecycle.receipt_sha256,
        "raw_completion_receipt": raw.file.to_json_dict(),
        "raw_completion_receipt_sha256": raw.file.sha256,
        "raw_trajectory_segments": raw_segments,
        "resume_from_action_id": lifecycle.resume_from_action_id,
        "resume_run_receipt_sha256": lifecycle.resume_run_receipt_sha256,
        "resume_run_sha256": lifecycle.resume_run_sha256,
        "bindings_sha256": bindings.digest,
        "source_epoch_identity_sha256": bindings.source_epoch_identity_sha256,
        "seed_id": bindings.seed_id,
        "seed_identity_sha256": bindings.seed_identity_sha256,
        "seed_receipt_sha256": bindings.seed_receipt_sha256,
        "exact_reference_id": bindings.reference_id,
        "exact_reference_identity_sha256": bindings.reference_identity_sha256,
        "exact_reference_receipt_sha256": reference.receipt_sha256,
        "exact_manifest_sha256": reference.manifest.sha256,
        "exact_data_sha256": reference.data.sha256,
        "run_sha256": raw.run.sha256,
        "accuracy_policy_sha256": bindings.accuracy_policy_sha256,
        "accuracy_policy": policy.to_json_dict(),
        "metrics": metrics.to_json_dict(),
        "max_pointwise_errors": maxima,
        "response_normalized_evaluation": response,
        "canonical_costs": resources,
        "report_values": {
            **metrics.to_json_dict(),
            **maxima,
            "accepted_steps": observed.accepted_steps,
            "terminal_n_theta": resources["logical_coordinates"][
                "terminal_independent_parameter_count"
            ],
            "routing_independent_graph_tuple": resources[
                "routing_independent_graph_tuple"
            ]["values"],
            "individual_qualified": response["individual_qualified"],
            "comparison_guard_eligible": response[
                "comparison_guard_eligible"
            ],
            "cost_comparison_admissible": False,
        },
        "qiskit_compilation_used": False,
        "monotonicity_inference_used": False,
        "scientific_pair_qualification": None,
    }
    metrics_payload = dict(metrics_core)
    metrics_payload["score_metrics_sha256"] = _canonical_sha(metrics_core)
    _atomic_write_immutable(metrics_target, _pretty_bytes(metrics_payload))
    metrics_file = VerifiedFile(metrics_relative, _file_sha(metrics_target))
    planner_receipt.verify(root)
    receipt_core = {
        "schema": SCORE_QUALIFICATION_RECEIPT_V1,
        "scoring_worker_schema": SCORING_WORKER_SCHEMA_V1,
        "score_action_id": score_request.action_id,
        "score_execution_request_sha256": score_request.digest,
        "scientific_action_id": lifecycle.scientific_action_id,
        "execution_id": lifecycle.execution_id,
        "cell_id": bindings.cell_id,
        "method": lifecycle.method,
        "horizon": lifecycle.horizon,
        "activation_cut": lifecycle.activation_cut,
        "attempt": lifecycle.attempt,
        "planner_sequence": lifecycle.planner_sequence,
        "score_planner_sequence": score_request.planner_sequence,
        "requested_memory_gib": lifecycle.requested_memory_gib,
        "run_receipt": completed_run_receipt.to_json_dict(),
        "action_registry": lifecycle.action_registry.to_json_dict(),
        "raw_completion_receipt": raw.file.to_json_dict(),
        "exact_reference_id": bindings.reference_id,
        "exact_reference_identity_sha256": bindings.reference_identity_sha256,
        "exact_reference_receipt": exact_reference_receipt.to_json_dict(),
        "exact_manifest_sha256": reference.manifest.sha256,
        "exact_data_sha256": reference.data.sha256,
        "bindings_sha256": bindings.digest,
        "source_epoch_identity_sha256": bindings.source_epoch_identity_sha256,
        "accuracy_policy_sha256": bindings.accuracy_policy_sha256,
        "metrics": metrics_file.to_json_dict(),
        "planner_receipt": planner_receipt.to_json_dict(),
        "qiskit_compilation_used": False,
        "scientific_pair_qualification": None,
    }
    receipt_payload = dict(receipt_core)
    receipt_payload["score_receipt_sha256"] = _canonical_sha(receipt_core)
    _atomic_write_immutable(receipt_target, _pretty_bytes(receipt_payload))
    loaded = ScoreQualificationReceipt.load(
        receipt_target, repository_root=root
    )
    if loaded.receipt_path != receipt_relative:
        raise ProspectiveExecutionError("score receipt materialized at another path")
    return loaded


__all__ = [
    "APM_COMPLETION_SCHEMA_V1",
    "AVQDS_COMPLETION_SCHEMA_V1",
    "COMPLETED_UNSCORED",
    "SCIENTIFIC_AVQDS_COMPLETION_SCHEMA_V1",
    "SCORING_WORKER_SCHEMA_V1",
    "materialization_request",
    "score_completed_trajectory",
]
