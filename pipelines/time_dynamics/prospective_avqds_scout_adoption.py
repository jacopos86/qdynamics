"""Adopt a verified bootstrap AVQDS scout into the scientific lifecycle.

The bootstrap planner intentionally launches the first exact-blind AVQDS prefix
beside the exact reference.  Once that reference exists, the scientific planner
uses a different action envelope.  This module joins those envelopes without
rerunning or relabelling the bootstrap calculation: it proves that the frozen
scientific first rung is physically identical, records an explicit evidence-
reuse disposition, and emits the normal scientific lifecycle chain consumed by
the scorer and planner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from pipelines.time_dynamics.prospective_avqds_scout_worker import (
    AVQDSPrefixCompletionReceipt,
    AVQDSPrefixWorkOrder,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    ExactReferenceIdentity,
    SeedIdentity,
    SourceEpochIdentity,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    RUN_STATE_EVIDENCE_V1,
    ExactReferenceReceipt,
    ProspectiveExecutionError,
    RunLifecycleReceipt,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    VerifiedFile,
    build_scientific_bindings,
    load_action_registry_entry,
    resolve_scientific_action,
    write_action_registry_entry,
    write_run_lifecycle_receipt,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ARM_AVQDS,
    HORIZON_PREFIX,
    KIND_RUN,
    ProspectiveEvidence,
    ProspectiveScientificBindings,
    plan_prospective_cell,
)
from pipelines.time_dynamics.staged_campaign import AccuracyPolicy


SCOUT_ADOPTION_RECEIPT_V1 = "paper_ii_avqds_scout_adoption_receipt_v1"
SCOUT_ADOPTION_STATUS = "adopted_completed_unscored"
SCOUT_ADOPTION_EVIDENCE_KIND = "verified_bootstrap_scout_evidence_reuse"


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


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ProspectiveExecutionError(f"{label} must be a JSON object")
    return value


def _atomic_write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(payload), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ProspectiveExecutionError(f"adoption identity collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
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


def _inside(root: Path, value: str | Path, *, label: str) -> Path:
    target = Path(value)
    if not target.is_absolute():
        target = root / target
    target = target.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(f"{label} is outside repository") from exc
    return target


def _verified(root: Path, path: Path) -> VerifiedFile:
    return VerifiedFile(path.relative_to(root).as_posix(), _file_sha(path))


def _source_files(
    source: SourceEpochReceipt,
    seed: SeedQualificationReceipt,
    reference: ExactReferenceReceipt,
) -> tuple[VerifiedFile, ...]:
    return (
        VerifiedFile(source.receipt_path, source.receipt_sha256),
        VerifiedFile(seed.receipt_path, seed.receipt_sha256),
        VerifiedFile(reference.receipt_path, reference.receipt_sha256),
    )


def _canonical_scientific_request(
    bindings: ProspectiveScientificBindings,
):
    policy = AccuracyPolicy.from_json_dict(
        bindings.to_json_dict()["accuracy_policy"]
    )
    plan = plan_prospective_cell(
        bindings,
        policy,
        evidence=ProspectiveEvidence(),
    )
    actions = tuple(plan.next_actions)
    if (
        plan.phase != "avqds_prefix_scout"
        or len(actions) != 1
        or actions[0].kind != KIND_RUN
        or actions[0].arm != ARM_AVQDS
        or actions[0].horizon != HORIZON_PREFIX
        or actions[0].ladder_index != 0
        or actions[0].attempt != 0
    ):
        raise ProspectiveExecutionError(
            "scientific planner did not emit one canonical first AVQDS rung"
        )
    return actions[0], resolve_scientific_action(bindings, actions[0])


def _validate_context(
    *,
    order: AVQDSPrefixWorkOrder,
    source_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
    reference_receipt: ExactReferenceReceipt,
    bindings: ProspectiveScientificBindings,
) -> tuple[Any, Any]:
    canonical = build_scientific_bindings(
        reference=order.reference,
        seed=order.seed,
        seed_receipt=seed_receipt,
        reference_receipt=reference_receipt,
        source_epoch=source_receipt.source_epoch,
        source_epoch_receipt=source_receipt,
    )
    if canonical != bindings:
        raise ProspectiveExecutionError(
            "scout adoption scientific bindings are not canonical"
        )
    action, request = _canonical_scientific_request(bindings)
    bootstrap = order.request
    bootstrap_inputs = bootstrap.inputs
    scientific_inputs = request.inputs
    if (
        bootstrap.lane_id != bindings.lane_id
        or bootstrap_inputs.get("cell_id") != bindings.cell_id
        or bootstrap_inputs.get("reference_id") != bindings.reference_id
        or bootstrap_inputs.get("reference_identity_sha256")
        != bindings.reference_identity_sha256
        or bootstrap_inputs.get("seed_identity_sha256")
        != bindings.seed_identity_sha256
        or bootstrap_inputs.get("seed_artifact_sha256")
        != bindings.seed_artifact_sha256
        or bootstrap_inputs.get("source_epoch_identity_sha256")
        != bindings.source_epoch_identity_sha256
        or bootstrap_inputs.get("controller") != bindings.controller
        or bootstrap_inputs.get("method") != ARM_AVQDS
        or bootstrap_inputs.get("record_statevector") is not True
        or str(bootstrap_inputs.get("drive", {}).get("drive_id"))
        != bindings.drive
        or dict(bootstrap_inputs.get("physical") or {})
        != dict(order.reference.physical)
        or dict(bootstrap_inputs.get("ordered_grid") or {})
        != dict(order.reference.grid)
        or scientific_inputs.get("cell_id") != bootstrap_inputs.get("cell_id")
        or scientific_inputs.get("seed_identity_sha256")
        != bootstrap_inputs.get("seed_identity_sha256")
        or scientific_inputs.get("seed_artifact_sha256")
        != bootstrap_inputs.get("seed_artifact_sha256")
        or scientific_inputs.get("reference_identity_sha256")
        != bootstrap_inputs.get("reference_identity_sha256")
        or scientific_inputs.get("source_epoch_identity_sha256")
        != bootstrap_inputs.get("source_epoch_identity_sha256")
        or scientific_inputs.get("controller") != bootstrap_inputs.get("controller")
        or scientific_inputs.get("record_statevector") is not True
        or float(scientific_inputs.get("activation_cut"))
        != float(bootstrap_inputs.get("activation_cut"))
        or int(scientific_inputs.get("ladder_index"))
        != int(bootstrap_inputs.get("ladder_index"))
    ):
        raise ProspectiveExecutionError(
            "bootstrap scout is not physically identical to the scientific first rung"
        )
    return action, request


def _state_evidence(
    *,
    request: Any,
    state: str,
    bootstrap_order: AVQDSPrefixWorkOrder,
    bootstrap_receipt: AVQDSPrefixCompletionReceipt,
    bootstrap_receipt_file: VerifiedFile,
    submitted_lifecycle: VerifiedFile | None = None,
) -> dict[str, Any]:
    core = {
        "schema": RUN_STATE_EVIDENCE_V1,
        "scientific_action_id": request.action_id,
        "execution_id": request.execution_id,
        "state": state,
        "evidence_kind": SCOUT_ADOPTION_EVIDENCE_KIND,
        "scheduler_submission_performed": False,
        "bootstrap_action_id": bootstrap_order.request.action_id,
        "bootstrap_execution_request_sha256": bootstrap_order.request.digest,
        "bootstrap_work_order_sha256": bootstrap_order.work_order_sha256,
        "bootstrap_completion_receipt": bootstrap_receipt_file.to_json_dict(),
        "bootstrap_completion_receipt_self_sha256": (
            bootstrap_receipt.receipt_self_sha256
        ),
        "reused_run": bootstrap_receipt.run.to_json_dict(),
        "submitted_lifecycle": (
            None
            if submitted_lifecycle is None
            else submitted_lifecycle.to_json_dict()
        ),
        "remote_raw_evidence_policy": "preserve",
        "scientific_qualification": None,
    }
    result = dict(core)
    result["run_state_evidence_sha256"] = _canonical_sha(core)
    return result


@dataclass(frozen=True)
class ScoutAdoptionReceipt:
    scientific_action_id: str
    execution_request_sha256: str
    bootstrap_action_id: str
    bootstrap_completion_receipt: VerifiedFile
    run: VerifiedFile
    action_registry: VerifiedFile
    submitted_lifecycle: VerifiedFile
    completed_lifecycle: VerifiedFile
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        bootstrap_order: AVQDSPrefixWorkOrder,
        source_receipt: SourceEpochReceipt,
        seed_receipt: SeedQualificationReceipt,
        reference_receipt: ExactReferenceReceipt,
        bindings: ProspectiveScientificBindings,
    ) -> "ScoutAdoptionReceipt":
        root = Path(repository_root).resolve()
        source = _inside(root, path, label="scout adoption receipt")
        payload = _read_object(source, label="scout adoption receipt")
        unsigned = dict(payload)
        declared = unsigned.pop("scout_adoption_receipt_sha256", None)
        actual = _canonical_sha(unsigned)
        if declared != actual:
            raise ProspectiveExecutionError("scout adoption receipt digest mismatch")
        if (
            payload.get("schema") != SCOUT_ADOPTION_RECEIPT_V1
            or payload.get("status") != SCOUT_ADOPTION_STATUS
            or payload.get("scheduler_submission_performed") is not False
            or payload.get("remote_raw_evidence_policy") != "preserve"
            or payload.get("scientific_qualification") is not None
            or payload.get("qiskit_compilation_used") is not False
        ):
            raise ProspectiveExecutionError("invalid scout adoption disposition")

        def binding(name: str) -> VerifiedFile:
            row = payload.get(name)
            if not isinstance(row, Mapping):
                raise ProspectiveExecutionError(
                    f"scout adoption receipt lacks {name}"
                )
            result = VerifiedFile(
                str(row.get("path") or ""), str(row.get("sha256") or "")
            )
            result.verify(root)
            return result

        bootstrap_file = binding("bootstrap_completion_receipt")
        run = binding("run")
        registry_file = binding("action_registry")
        submitted_file = binding("submitted_lifecycle")
        completed_file = binding("completed_lifecycle")
        action, request = _validate_context(
            order=bootstrap_order,
            source_receipt=source_receipt,
            seed_receipt=seed_receipt,
            reference_receipt=reference_receipt,
            bindings=bindings,
        )
        bootstrap = AVQDSPrefixCompletionReceipt.load(
            bootstrap_file.path,
            repository_root=root,
            expected_order=bootstrap_order,
            expected_seed_receipt=seed_receipt,
            expected_source_epoch=source_receipt.source_epoch,
        )
        entry = load_action_registry_entry(
            registry_file,
            repository_root=root,
            bindings=bindings,
            expected_seed=bootstrap_order.seed,
            expected_reference=bootstrap_order.reference,
            expected_source_epoch=source_receipt.source_epoch,
        )
        submitted = RunLifecycleReceipt.load(
            submitted_file.path, repository_root=root
        )
        completed = RunLifecycleReceipt.load(
            completed_file.path, repository_root=root
        )
        completed.validate_bindings(root, bindings)
        submitted_state = _read_object(
            root / submitted.state_evidence.path,
            label="submitted scout adoption evidence",
        )
        completed_state = _read_object(
            root / completed.state_evidence.path,
            label="completed scout adoption evidence",
        )
        expected_bootstrap = bootstrap_file.to_json_dict()
        if (
            payload.get("bindings_sha256") != bindings.digest
            or payload.get("scientific_action_id") != action.action_id
            or payload.get("execution_request_sha256") != request.digest
            or payload.get("bootstrap_action_id") != bootstrap_order.request.action_id
            or payload.get("bootstrap_execution_request_sha256")
            != bootstrap_order.request.digest
            or payload.get("bootstrap_work_order_sha256")
            != bootstrap_order.work_order_sha256
            or entry.request != request
            or submitted.scientific_action_id != request.action_id
            or submitted.state != "submitted"
            or completed.scientific_action_id != request.action_id
            or completed.state != "completed"
            or completed.previous_receipt != submitted_file
            or completed.run != bootstrap.run
            or run != bootstrap.run
            or bootstrap_file.sha256 != bootstrap.receipt_sha256
            or any(
                row.get("evidence_kind") != SCOUT_ADOPTION_EVIDENCE_KIND
                or row.get("scheduler_submission_performed") is not False
                or row.get("bootstrap_action_id")
                != bootstrap_order.request.action_id
                or row.get("bootstrap_completion_receipt") != expected_bootstrap
                or row.get("reused_run") != bootstrap.run.to_json_dict()
                or row.get("remote_raw_evidence_policy") != "preserve"
                for row in (submitted_state, completed_state)
            )
            or completed_state.get("submitted_lifecycle")
            != submitted_file.to_json_dict()
        ):
            raise ProspectiveExecutionError("scout adoption evidence diverged")
        relative = source.relative_to(root).as_posix()
        return cls(
            scientific_action_id=request.action_id,
            execution_request_sha256=request.digest,
            bootstrap_action_id=bootstrap.action_id,
            bootstrap_completion_receipt=bootstrap_file,
            run=run,
            action_registry=registry_file,
            submitted_lifecycle=submitted_file,
            completed_lifecycle=completed_file,
            receipt_path=relative,
            receipt_sha256=_file_sha(source),
            receipt_self_sha256=actual,
        )


def adopt_completed_scout(
    *,
    bootstrap_order: AVQDSPrefixWorkOrder,
    source_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
    reference_receipt: ExactReferenceReceipt,
    bindings: ProspectiveScientificBindings,
    repository_root: str | Path,
    output_directory: str | Path | None = None,
) -> ScoutAdoptionReceipt:
    """Adopt one validated bootstrap first rung without a duplicate run."""

    root = Path(repository_root).resolve()
    action, request = _validate_context(
        order=bootstrap_order,
        source_receipt=source_receipt,
        seed_receipt=seed_receipt,
        reference_receipt=reference_receipt,
        bindings=bindings,
    )
    completion = AVQDSPrefixCompletionReceipt.load(
        bootstrap_order.completion_receipt_path,
        repository_root=root,
        expected_order=bootstrap_order,
        expected_seed_receipt=seed_receipt,
        expected_source_epoch=source_receipt.source_epoch,
    )
    completion_file = VerifiedFile(
        completion.receipt_path, completion.receipt_sha256
    )
    directory = _inside(
        root,
        output_directory
        or (
            Path("output/prospective_scientific")
            / bindings.lane_id
            / bindings.cell_id
            / f"adopt_scout_{completion.receipt_sha256[:16]}"
        ),
        label="scout adoption output directory",
    )
    registry_path = directory / "action_registry.json"
    write_action_registry_entry(
        registry_path,
        request,
        source_receipts=_source_files(
            source_receipt, seed_receipt, reference_receipt
        ),
        repository_root=root,
        bindings=bindings,
        expected_seed=bootstrap_order.seed,
        expected_reference=bootstrap_order.reference,
        expected_source_epoch=source_receipt.source_epoch,
    )
    registry_file = _verified(root, registry_path)

    submitted_evidence_path = directory / "submitted_evidence.json"
    _atomic_write_immutable(
        submitted_evidence_path,
        _state_evidence(
            request=request,
            state="submitted",
            bootstrap_order=bootstrap_order,
            bootstrap_receipt=completion,
            bootstrap_receipt_file=completion_file,
        ),
    )
    submitted_evidence = _verified(root, submitted_evidence_path)
    submitted_path = directory / "submitted_lifecycle.json"
    submitted = write_run_lifecycle_receipt(
        submitted_path,
        request=request,
        state="submitted",
        action_registry=registry_file,
        state_evidence=submitted_evidence,
        repository_root=root,
    )
    submitted_file = VerifiedFile(
        submitted.receipt_path, submitted.receipt_sha256
    )

    completed_evidence_path = directory / "completed_evidence.json"
    _atomic_write_immutable(
        completed_evidence_path,
        _state_evidence(
            request=request,
            state="completed",
            bootstrap_order=bootstrap_order,
            bootstrap_receipt=completion,
            bootstrap_receipt_file=completion_file,
            submitted_lifecycle=submitted_file,
        ),
    )
    completed_evidence = _verified(root, completed_evidence_path)
    completed_path = directory / "completed_lifecycle.json"
    completed = write_run_lifecycle_receipt(
        completed_path,
        request=request,
        state="completed",
        action_registry=registry_file,
        state_evidence=completed_evidence,
        repository_root=root,
        previous_receipt=submitted_file,
        run=completion.run,
    )
    completed_file = VerifiedFile(completed.receipt_path, completed.receipt_sha256)
    receipt_path = directory / "adoption_receipt.json"
    core = {
        "schema": SCOUT_ADOPTION_RECEIPT_V1,
        "status": SCOUT_ADOPTION_STATUS,
        "bindings_sha256": bindings.digest,
        "scientific_action_id": action.action_id,
        "execution_request_sha256": request.digest,
        "bootstrap_action_id": bootstrap_order.request.action_id,
        "bootstrap_execution_request_sha256": bootstrap_order.request.digest,
        "bootstrap_work_order_sha256": bootstrap_order.work_order_sha256,
        "bootstrap_completion_receipt": completion_file.to_json_dict(),
        "run": completion.run.to_json_dict(),
        "action_registry": registry_file.to_json_dict(),
        "submitted_lifecycle": submitted_file.to_json_dict(),
        "completed_lifecycle": completed_file.to_json_dict(),
        "scheduler_submission_performed": False,
        "remote_raw_evidence_policy": "preserve",
        "scientific_qualification": None,
        "qiskit_compilation_used": False,
    }
    payload = dict(core)
    payload["scout_adoption_receipt_sha256"] = _canonical_sha(core)
    _atomic_write_immutable(receipt_path, payload)
    return ScoutAdoptionReceipt.load(
        receipt_path,
        repository_root=root,
        bootstrap_order=bootstrap_order,
        source_receipt=source_receipt,
        seed_receipt=seed_receipt,
        reference_receipt=reference_receipt,
        bindings=bindings,
    )


__all__ = [
    "SCOUT_ADOPTION_EVIDENCE_KIND",
    "SCOUT_ADOPTION_RECEIPT_V1",
    "SCOUT_ADOPTION_STATUS",
    "ScoutAdoptionReceipt",
    "adopt_completed_scout",
]
