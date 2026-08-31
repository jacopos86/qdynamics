"""Recover the prospective Hubbard-L4 seed submission from durable evidence.

This is the Hubbard counterpart of :mod:`prospective_submission_recovery`.
It deliberately accepts only the sole frozen Hubbard-L4 seed action.  A
successful ``condor_submit`` result remains provisional until a *later*, single
combined queue/history snapshot contains exactly one matching scheduler row.

The recovery path is read-only apart from publishing the canonical supervisor
receipt.  It never contacts Condor, authorizes a submission, or mutates the
campaign registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from pipelines.time_dynamics.acquisition_supervisor import (
    SUBMISSION_INTENT_SCHEMA_V1,
    SUBMISSION_RECEIPT_SCHEMA_V1,
    SubmissionUnit,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HUBBARD,
    SourceEpochIdentity,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    ProspectiveExecutionError,
    SourceEpochReceipt,
)
from pipelines.time_dynamics.prospective_hubbard_seed_worker import (
    validate_canonical_seed_request,
)
from pipelines.time_dynamics.prospective_submission_recovery import (
    COMBINED_SCOPED_SNAPSHOT_V1,
    DurableSubmissionIntent,
    ProvisionalSubmissionResult,
    RecoveredSubmissionReceipt,
    SubmissionRecoveryError,
    _atomic_write_immutable,
    _binding,
    _digest,
    _file_sha256,
    _parse_timestamp,
    _read_object,
    _relative,
    _repository_path,
    _sha,
    _verify_local_binding,
    load_combined_scheduler_rows,
    load_provisional_submission_result,
)


HUBBARD_LANE_ID = LANE_HUBBARD
HUBBARD_SEED_TOKEN = "l4__u4__half_filled"
BOOTSTRAP_PACKAGE_MANIFEST_V1 = "paper_ii_bootstrap_package_manifest_v1"
_ACTION = re.compile(
    rf"build_seed::{re.escape(HUBBARD_LANE_ID)}::seed::"
    rf"{re.escape(HUBBARD_SEED_TOKEN)}::[0-9a-f]{{16}}"
)

# One historical submission crossed the scheduler boundary immediately before
# the intent-first two-snapshot absence proof became mandatory.  This binding
# is deliberately byte-exact: it can recover cluster 9714040 and nothing else.
# New submissions continue through ``load_provisional_submission_result`` and
# therefore retain the strict two-snapshot gate without an escape hatch.
_LEGACY_9714040 = {
    "cluster_id": 9714040,
    "unit_id": "hubbard-seed-ee58df1fb13e9669",
    "job_batch_name": (
        "paper-ii-seed-paper-ii-hubbard-l4-scale-v1-ee58df1fb13e9669"
    ),
    "action_id": (
        "build_seed::paper_ii_hubbard_l4_scale_v1::seed::"
        "l4__u4__half_filled::3f5f01b25de75f9f"
    ),
    "execution_request_sha256": (
        "ee58df1fb13e9669f90c7751a9543393ac23265fb2304c89ac6ff5f945124abd"
    ),
    "source_epoch_id": (
        "paper-ii-hubbard-l4-scale-v1-source-5eedf0dc546c-4ad1a8fc2205"
    ),
    "source_epoch_identity_sha256": (
        "1324fc47d82e7bd488d4479b3f8a1905eac909454a5382decd595a5689fb5235"
    ),
    "intent_path": (
        "output/paper_ii_hubbard_l4_scale_v1/acquisition_supervisor/"
        "submission_intents/hubbard-seed-ee58df1fb13e9669.json"
    ),
    "intent_file_sha256": (
        "7906f94caa4756522f89899445b5239baa446c888a9dbc5150f4a3062b2d9080"
    ),
    "intent_sha256": (
        "a5cdc5d7f51f375adb1199e23d8fb11a38d4348df74fb0967d968e4fda7d5e9a"
    ),
    "provisional_path": (
        "output/paper_ii_hubbard_l4_scale_v1/acquisition_supervisor/"
        "submission_results/activation_5eedf0dc/"
        "hubbard-seed-ee58df1fb13e9669.json"
    ),
    "provisional_file_sha256": (
        "327531df5d5ba7d531323a8d76c84b80833dc9a5424dfd1f366265cc2108b414"
    ),
    "provisional_sha256": (
        "4658a28739b7457633b29534c14b75e6a1e3af6b5cec45bbb09621cc0db8c32e"
    ),
    "pre_submit_snapshot_path": (
        "output/time_dynamics_acquisition/prospective_combined_snapshots/"
        "cycle_20260829_0104/combined_snapshot.json"
    ),
    "pre_submit_snapshot_file_sha256": (
        "345bf09bd53cc79dec3a803b9a465dd70f12ebea3f5202102dd28da0ed3049e8"
    ),
    "pre_submit_snapshot_sha256": (
        "5f9d6a9a24e8ad5829dcde697163ff9db71b7f0267853b25140ac168887c9a17"
    ),
    "initial_indeterminate_path": (
        "output/paper_ii_hubbard_l4_scale_v1/acquisition_supervisor/"
        "submission_reconciliations/hubbard-seed-ee58df1fb13e9669/"
        "000_initial_indeterminate.json"
    ),
    "initial_indeterminate_file_sha256": (
        "e7c9f4cbdaadee0273521c2bf86373442a6a9bba90ecce4fb6920d411af530f8"
    ),
    "submitted_at": "2026-08-29T01:07:52.573155-05:00",
}
_LEGACY_PREFLIGHT_WARNING_SCHEMA_V1 = (
    "paper_ii_hubbard_9714040_legacy_preflight_warning_v1"
)
_LEGACY_PREFLIGHT_WARNING_CODE = (
    "first_scheduler_snapshot_marked_verified_absent_without_"
    "digest_bound_prior_scheduler_snapshot"
)


@dataclass(frozen=True)
class HubbardDurableSubmissionIntent:
    """Authenticated intent plus its frozen request/source-epoch binding."""

    base: DurableSubmissionIntent
    request: ExecutionRequest
    source_epoch: SourceEpochIdentity
    package_manifest_path: str
    package_manifest_sha256: str


def _load_package_manifest(
    root: Path,
    *,
    unit: SubmissionUnit,
) -> tuple[ExecutionRequest, SourceEpochIdentity, str, str]:
    payload = unit.payload
    manifest_binding = _binding(
        payload.get("package_manifest"),
        label="Hubbard package manifest",
        require_bytes=False,
    )
    manifest_path = _repository_path(
        root,
        manifest_binding["path"],
        label="Hubbard package manifest",
        must_exist=True,
    )
    if _file_sha256(manifest_path) != manifest_binding["sha256"]:
        raise SubmissionRecoveryError("Hubbard package manifest digest changed")
    manifest = _read_object(manifest_path, label="Hubbard package manifest")
    unsigned = dict(manifest)
    declared = _sha(
        unsigned.pop("bootstrap_package_manifest_sha256", None),
        label="Hubbard package manifest self digest",
    )
    if declared != _digest(unsigned):
        raise SubmissionRecoveryError("Hubbard package manifest self digest mismatch")
    request_row = manifest.get("execution_request")
    source_row = manifest.get("source_epoch_receipt")
    if not isinstance(request_row, Mapping) or not isinstance(source_row, Mapping):
        raise SubmissionRecoveryError("Hubbard package manifest is incomplete")
    try:
        request = ExecutionRequest.from_mapping(request_row)
        seed = validate_canonical_seed_request(request)
        epoch_raw = request.inputs.get("source_epoch")
        if not isinstance(epoch_raw, Mapping):
            raise ValueError("missing request source epoch")
        epoch = SourceEpochIdentity(
            lane_id=str(epoch_raw.get("lane_id") or ""),
            source_epoch_id=str(epoch_raw.get("source_epoch_id") or ""),
            source_revision=epoch_raw.get("source_revision"),
            source_archive_sha256=epoch_raw.get("source_archive_sha256"),
        )
    except (ProspectiveExecutionError, TypeError, ValueError) as exc:
        raise SubmissionRecoveryError(
            "Hubbard package does not contain the canonical seed request"
        ) from exc
    request_sha = _sha(
        payload.get("execution_request_sha256"),
        label="Hubbard execution request digest",
    )
    package = _binding(
        payload.get("package"), label="Hubbard seed package", require_bytes=True
    )
    manifest_package = manifest.get("package")
    try:
        package_bytes = int(manifest.get("package_bytes"))
    except (TypeError, ValueError) as exc:
        raise SubmissionRecoveryError("Hubbard package byte count is invalid") from exc
    expected_batch = f"paper-ii-seed-{HUBBARD_LANE_ID.replace('_', '-')}-{request.digest[:16]}"
    if (
        seed.seed_id != f"{HUBBARD_LANE_ID}::seed::{HUBBARD_SEED_TOKEN}"
        or epoch_raw.get("source_epoch_identity_sha256") != epoch.identity_sha256
        or request.digest != request_sha
        or request.action_id != unit.action_ids[0]
        or payload.get("source_epoch_identity_sha256") != epoch.identity_sha256
        or unit.unit_id != f"hubbard-seed-{request.digest[:16]}"
        or payload.get("job_batch_name") != expected_batch
        or manifest.get("schema") != BOOTSTRAP_PACKAGE_MANIFEST_V1
        or manifest.get("status") != "prepared_local_only"
        or manifest.get("lane_id") != HUBBARD_LANE_ID
        or manifest.get("action_id") != request.action_id
        or manifest.get("execution_request_sha256") != request.digest
        or manifest.get("source_epoch_identity_sha256") != epoch.identity_sha256
        or manifest.get("job_batch_name") != expected_batch
        or manifest.get("qiskit_compilation") is not False
        or manifest.get("production_submission_enabled") is not False
        or not isinstance(manifest_package, Mapping)
        or manifest_package.get("path") != package["path"]
        or manifest_package.get("sha256") != package["sha256"]
        or package_bytes != package["bytes"]
    ):
        raise SubmissionRecoveryError(
            "Hubbard intent/package/request/source identity diverged"
        )
    source_path = _repository_path(
        root,
        str(source_row.get("path") or ""),
        label="Hubbard source epoch receipt",
        must_exist=True,
    )
    if _file_sha256(source_path) != source_row.get("sha256"):
        raise SubmissionRecoveryError("Hubbard source epoch receipt digest changed")
    try:
        source = SourceEpochReceipt.load(
            source_path, repository_root=root, expected_epoch=epoch
        )
    except (OSError, ProspectiveExecutionError, ValueError) as exc:
        raise SubmissionRecoveryError(
            "Hubbard source epoch receipt does not bind the request"
        ) from exc
    if source.lane_id != HUBBARD_LANE_ID:
        raise SubmissionRecoveryError("Hubbard source epoch crosses lanes")
    return (
        request,
        epoch,
        _relative(root, manifest_path, label="Hubbard package manifest"),
        manifest_binding["sha256"],
    )


def load_hubbard_durable_submission_intent(
    path: str | Path, *, repository_root: str | Path
) -> HubbardDurableSubmissionIntent:
    """Load the sole canonical Hubbard-L4 seed submission intent."""

    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="Hubbard submission intent", must_exist=True
    )
    payload = _read_object(source, label="Hubbard submission intent")
    unit_raw = payload.get("unit")
    if (
        payload.get("schema") != SUBMISSION_INTENT_SCHEMA_V1
        or payload.get("lane_id") != HUBBARD_LANE_ID
        or not isinstance(unit_raw, Mapping)
    ):
        raise SubmissionRecoveryError("submission intent is not Hubbard-L4")
    try:
        unit = SubmissionUnit(
            unit_id=str(unit_raw.get("unit_id") or ""),
            action_ids=tuple(str(value) for value in unit_raw.get("action_ids") or ()),
            payload=dict(unit_raw.get("payload") or {}),
        )
    except (TypeError, ValueError) as exc:
        raise SubmissionRecoveryError("Hubbard submission unit is invalid") from exc
    if (
        len(unit.action_ids) != 1
        or _ACTION.fullmatch(unit.action_ids[0]) is None
        or not re.fullmatch(r"hubbard-seed-[0-9a-f]{16}", unit.unit_id)
        or unit.payload.get("expected_process_count") != 1
        or unit.payload.get("submission_scope")
        != {"action_ids": [unit.action_ids[0]], "process_ids": [0]}
        or unit.payload.get("remote_raw_evidence_policy") != "preserve"
    ):
        raise SubmissionRecoveryError("Hubbard submission unit identity diverged")
    core = {
        "schema": SUBMISSION_INTENT_SCHEMA_V1,
        "lane_id": HUBBARD_LANE_ID,
        "unit": unit.to_json_dict(),
    }
    intent_sha = _sha(payload.get("intent_sha256"), label="Hubbard intent digest")
    if intent_sha != _digest(core):
        raise SubmissionRecoveryError("Hubbard submission intent self digest mismatch")
    recorded_at = str(payload.get("recorded_at") or "")
    _parse_timestamp(recorded_at, label="Hubbard submission intent recorded_at")
    expected_path = (
        Path("output")
        / HUBBARD_LANE_ID
        / "acquisition_supervisor/submission_intents"
        / f"{unit.unit_id}.json"
    ).as_posix()
    relative = _relative(root, source, label="Hubbard submission intent")
    if relative != expected_path:
        raise SubmissionRecoveryError("Hubbard intent is not at its canonical path")
    request, epoch, manifest_path, manifest_sha = _load_package_manifest(
        root, unit=unit
    )
    base = DurableSubmissionIntent(
        path=relative,
        file_sha256=_file_sha256(source),
        intent_sha256=intent_sha,
        lane_id=HUBBARD_LANE_ID,
        unit=unit,
        recorded_at=recorded_at,
    )
    return HubbardDurableSubmissionIntent(
        base=base,
        request=request,
        source_epoch=epoch,
        package_manifest_path=manifest_path,
        package_manifest_sha256=manifest_sha,
    )


def _is_exact_legacy_9714040_intent(
    intent: HubbardDurableSubmissionIntent,
) -> bool:
    """Return whether ``intent`` is the byte-bound historical 9714040 intent."""

    return (
        intent.base.path == _LEGACY_9714040["intent_path"]
        and intent.base.file_sha256 == _LEGACY_9714040["intent_file_sha256"]
        and intent.base.intent_sha256 == _LEGACY_9714040["intent_sha256"]
        and intent.base.unit.unit_id == _LEGACY_9714040["unit_id"]
        and intent.base.unit.action_ids == (_LEGACY_9714040["action_id"],)
        and intent.request.digest
        == _LEGACY_9714040["execution_request_sha256"]
        and intent.request.action_id == _LEGACY_9714040["action_id"]
        and intent.source_epoch.source_epoch_id
        == _LEGACY_9714040["source_epoch_id"]
        and intent.source_epoch.identity_sha256
        == _LEGACY_9714040["source_epoch_identity_sha256"]
        and intent.base.unit.payload.get("job_batch_name")
        == _LEGACY_9714040["job_batch_name"]
    )


def _load_exact_legacy_9714040_result(
    path: str | Path,
    *,
    root: Path,
    intent: HubbardDurableSubmissionIntent,
) -> tuple[ProvisionalSubmissionResult, dict[str, Any]]:
    """Authenticate the one pre-two-snapshot Hubbard submission byte-for-byte."""

    if not _is_exact_legacy_9714040_intent(intent):
        raise SubmissionRecoveryError(
            "legacy Hubbard recovery is restricted to the exact 9714040 intent"
        )
    source = _repository_path(
        root, path, label="legacy Hubbard provisional result", must_exist=True
    )
    relative = _relative(root, source, label="legacy Hubbard provisional result")
    payload = _read_object(source, label="legacy Hubbard provisional result")
    file_sha = _file_sha256(source)
    declared = _sha(
        payload.get("submission_result_sha256"),
        label="legacy Hubbard provisional result self digest",
    )
    core = dict(payload)
    core.pop("submission_result_sha256", None)
    if (
        relative != _LEGACY_9714040["provisional_path"]
        or file_sha != _LEGACY_9714040["provisional_file_sha256"]
        or declared != _LEGACY_9714040["provisional_sha256"]
        or declared != _digest(core)
    ):
        raise SubmissionRecoveryError(
            "legacy Hubbard 9714040 provisional bytes changed"
        )
    if (
        payload.get("schema") != "paper_ii_submission_result_v1"
        or payload.get("status") != "submitted"
        or payload.get("lane_id") != HUBBARD_LANE_ID
        or payload.get("unit_id") != _LEGACY_9714040["unit_id"]
        or payload.get("action_ids") != [_LEGACY_9714040["action_id"]]
        or payload.get("cluster_id") != _LEGACY_9714040["cluster_id"]
        or payload.get("expected_process_ids") != [0]
        or payload.get("job_batch_name") != _LEGACY_9714040["job_batch_name"]
        or payload.get("intent_path") != _LEGACY_9714040["intent_path"]
        or payload.get("intent_sha256") != _LEGACY_9714040["intent_sha256"]
        or payload.get("submitted_at") != _LEGACY_9714040["submitted_at"]
        or payload.get("lifecycle_validator_status") != "pass"
        or payload.get("dry_run_status") != "pass"
        or payload.get("condor_submit_returncode") != 0
        or payload.get("scheduler_acceptance_query_performed") is not False
        or payload.get("acceptance_evidence")
        != "condor_submit_zero_exit_and_cluster_id_stdout"
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or payload.get("pre_submit_snapshot_path")
        != _LEGACY_9714040["pre_submit_snapshot_path"]
        or payload.get("pre_submit_snapshot_sha256")
        != _LEGACY_9714040["pre_submit_snapshot_sha256"]
    ):
        raise SubmissionRecoveryError(
            "legacy Hubbard 9714040 provisional identity changed"
        )
    if (
        payload.get("package") != intent.base.unit.payload.get("package")
        or payload.get("runtime_source_archive")
        != intent.base.unit.payload.get("runtime_source_archive")
    ):
        raise SubmissionRecoveryError(
            "legacy Hubbard 9714040 package binding changed"
        )
    descriptor = _binding(
        payload.get("submit_descriptor"),
        label="legacy Hubbard submit descriptor",
        require_bytes=False,
    )
    if descriptor["path"] != intent.base.unit.payload.get("submit_descriptor"):
        raise SubmissionRecoveryError(
            "legacy Hubbard 9714040 submit descriptor changed"
        )
    _verify_local_binding(
        root,
        payload["package"],
        label="legacy Hubbard seed action package",
        require_bytes=True,
    )
    _verify_local_binding(
        root,
        payload["runtime_source_archive"],
        label="legacy Hubbard runtime source archive",
        require_bytes=True,
    )
    _verify_local_binding(
        root,
        descriptor,
        label="legacy Hubbard submit descriptor",
        require_bytes=False,
    )
    image = _binding(
        payload.get("runtime_image"),
        label="legacy Hubbard runtime image",
        require_bytes=True,
    )
    if image["path"] != "chtc/time_dynamics_optuna/image.sif":
        raise SubmissionRecoveryError(
            "legacy Hubbard 9714040 runtime image path changed"
        )

    pre_path = _repository_path(
        root,
        _LEGACY_9714040["pre_submit_snapshot_path"],
        label="legacy Hubbard pre-submit snapshot",
        must_exist=True,
    )
    pre = _read_object(pre_path, label="legacy Hubbard pre-submit snapshot")
    pre_core = dict(pre)
    pre_declared = _sha(
        pre_core.pop("snapshot_sha256", None),
        label="legacy Hubbard pre-submit snapshot self digest",
    )
    queries = pre.get("queries")
    scope = pre.get("scope")
    reconciliation = pre.get("reconciliation")
    outcome = (
        reconciliation.get(_LEGACY_9714040["job_batch_name"])
        if isinstance(reconciliation, Mapping)
        else None
    )
    if (
        _file_sha256(pre_path)
        != _LEGACY_9714040["pre_submit_snapshot_file_sha256"]
        or pre_declared != _LEGACY_9714040["pre_submit_snapshot_sha256"]
        or pre_declared != _digest(pre_core)
        or pre.get("schema") != COMBINED_SCOPED_SNAPSHOT_V1
        or pre.get("single_combined_snapshot") is not True
        or pre.get("polling_performed") is not False
        or pre.get("reconciliation_context") is not None
        or not isinstance(queries, Mapping)
        or not isinstance(scope, Mapping)
        or scope.get("prospective_job_batch_names")
        != [_LEGACY_9714040["job_batch_name"]]
        or outcome
        != {"status": "verified_absent", "match_count": 0, "matches": []}
    ):
        raise SubmissionRecoveryError(
            "legacy Hubbard 9714040 pre-submit snapshot changed"
        )
    for query_name in ("queue", "history"):
        query = queries.get(query_name)
        if (
            not isinstance(query, Mapping)
            or query.get("returncode") != 0
            or not isinstance(query.get("rows"), list)
        ):
            raise SubmissionRecoveryError(
                "legacy Hubbard 9714040 scheduler capture is invalid"
            )

    initial_path = _repository_path(
        root,
        _LEGACY_9714040["initial_indeterminate_path"],
        label="legacy Hubbard initial indeterminate receipt",
        must_exist=True,
    )
    initial = _read_object(
        initial_path, label="legacy Hubbard initial indeterminate receipt"
    )
    if (
        _file_sha256(initial_path)
        != _LEGACY_9714040["initial_indeterminate_file_sha256"]
        or initial.get("schema")
        != "paper_ii_bootstrap_submission_reconciliation_v1"
        or initial.get("lane_id") != HUBBARD_LANE_ID
        or initial.get("unit_id") != _LEGACY_9714040["unit_id"]
        or initial.get("action_ids") != [_LEGACY_9714040["action_id"]]
        or initial.get("status") != "indeterminate"
        or initial.get("intent")
        != {
            "path": _LEGACY_9714040["intent_path"],
            "sha256": _LEGACY_9714040["intent_sha256"],
        }
    ):
        raise SubmissionRecoveryError(
            "legacy Hubbard 9714040 initial indeterminate receipt changed"
        )
    intent_time = _parse_timestamp(
        intent.base.recorded_at, label="legacy Hubbard intent recorded_at"
    )
    initial_time = _parse_timestamp(
        initial.get("recorded_at"),
        label="legacy Hubbard initial indeterminate recorded_at",
    )
    pre_time = _parse_timestamp(
        pre.get("captured_at"), label="legacy Hubbard pre-submit captured_at"
    )
    submitted_time = _parse_timestamp(
        payload.get("submitted_at"), label="legacy Hubbard submitted_at"
    )
    if not (intent_time <= initial_time < pre_time < submitted_time):
        raise SubmissionRecoveryError(
            "legacy Hubbard 9714040 evidence ordering changed"
        )

    result = ProvisionalSubmissionResult(
        path=relative,
        file_sha256=file_sha,
        self_sha256=declared,
        lane_id=HUBBARD_LANE_ID,
        unit_id=_LEGACY_9714040["unit_id"],
        action_ids=(_LEGACY_9714040["action_id"],),
        cluster_id=_LEGACY_9714040["cluster_id"],
        process_ids=(0,),
        job_batch_name=_LEGACY_9714040["job_batch_name"],
        submitted_at=_LEGACY_9714040["submitted_at"],
        pre_submit_snapshot_path=_LEGACY_9714040[
            "pre_submit_snapshot_path"
        ],
        pre_submit_snapshot_sha256=_LEGACY_9714040[
            "pre_submit_snapshot_sha256"
        ],
    )
    warning = {
        "schema": _LEGACY_PREFLIGHT_WARNING_SCHEMA_V1,
        "status": "historical_submission_recovery_only",
        "warning_code": _LEGACY_PREFLIGHT_WARNING_CODE,
        "warning": (
            "Cluster 9714040 was submitted from the exact preserved intent and "
            "verified-absent capture before the digest-bound prior-snapshot "
            "requirement was implemented. This warning does not authorize a "
            "resubmission or relax the gate for any new submission."
        ),
        "strict_intent_first_gate_preserved_for_new_submissions": True,
        "resubmission_authorized": False,
        "exact_identity": {
            "cluster_id": _LEGACY_9714040["cluster_id"],
            "unit_id": _LEGACY_9714040["unit_id"],
            "action_id": _LEGACY_9714040["action_id"],
            "execution_request_sha256": _LEGACY_9714040[
                "execution_request_sha256"
            ],
            "source_epoch_id": _LEGACY_9714040["source_epoch_id"],
            "source_epoch_identity_sha256": _LEGACY_9714040[
                "source_epoch_identity_sha256"
            ],
        },
        "initial_indeterminate_receipt": {
            "path": _LEGACY_9714040["initial_indeterminate_path"],
            "file_sha256": _LEGACY_9714040[
                "initial_indeterminate_file_sha256"
            ],
        },
        "pre_submit_snapshot": {
            "path": _LEGACY_9714040["pre_submit_snapshot_path"],
            "sha256": _LEGACY_9714040["pre_submit_snapshot_sha256"],
            "file_sha256": _LEGACY_9714040[
                "pre_submit_snapshot_file_sha256"
            ],
        },
    }
    return result, warning


def _load_hubbard_provisional_submission_result_with_audit(
    path: str | Path,
    *,
    repository_root: str | Path,
    intent: HubbardDurableSubmissionIntent,
) -> tuple[ProvisionalSubmissionResult, dict[str, Any] | None]:
    """Load a strict result or the one byte-bound historical exception."""

    root = Path(repository_root).resolve(strict=True)
    try:
        result = load_provisional_submission_result(
            path, repository_root=root, intent=intent.base
        )
        warning = None
    except SubmissionRecoveryError:
        if not _is_exact_legacy_9714040_intent(intent):
            raise
        result, warning = _load_exact_legacy_9714040_result(
            path, root=root, intent=intent
        )
    source = _repository_path(
        root, path, label="Hubbard provisional result", must_exist=True
    )
    payload = _read_object(source, label="Hubbard provisional result")
    if payload.get("remote_raw_evidence_policy") != "preserve":
        raise SubmissionRecoveryError(
            "Hubbard provisional result does not preserve remote raw evidence"
        )
    return result, warning


def load_hubbard_provisional_submission_result(
    path: str | Path,
    *,
    repository_root: str | Path,
    intent: HubbardDurableSubmissionIntent,
) -> ProvisionalSubmissionResult:
    """Authenticate the provisional result and its raw-preservation policy."""

    result, _ = _load_hubbard_provisional_submission_result_with_audit(
        path, repository_root=repository_root, intent=intent
    )
    return result


def _matching_scheduler_row(
    *,
    intent: HubbardDurableSubmissionIntent,
    result: ProvisionalSubmissionResult,
    snapshot: Any,
) -> tuple[str, Mapping[str, Any]]:
    if not re.search(
        rf"\bClusterId\s*==\s*{result.cluster_id}\b", snapshot.scope_condition
    ):
        raise SubmissionRecoveryError(
            "later combined snapshot did not scope the Hubbard cluster"
        )
    matches = [
        (query, row)
        for query, row in snapshot.rows
        if int(row["ClusterId"]) == result.cluster_id
        and int(row["ProcId"]) == 0
    ]
    if len(matches) != 1:
        raise SubmissionRecoveryError(
            "later combined snapshot must contain exactly one Hubbard cluster/proc"
        )
    query, row = matches[0]
    lane = str(row.get("AcquisitionLaneId") or row.get("PaperIILaneId") or "")
    action = str(row.get("PaperIIActionId") or row.get("AcquisitionActionId") or "")
    try:
        status = int(row["JobStatus"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SubmissionRecoveryError("Hubbard scheduler status is invalid") from exc
    if (
        str(row.get("JobBatchName") or "") != result.job_batch_name
        or str(row.get("AcquisitionUnitId") or "") != intent.base.unit.unit_id
        or lane != HUBBARD_LANE_ID
        or action != intent.request.action_id
        or str(row.get("PaperIIExecutionRequestSha256") or "")
        != intent.request.digest
        or str(row.get("PaperIISourceEpochId") or "")
        != intent.source_epoch.source_epoch_id
        or status not in {1, 2, 3, 4, 5, 6, 7}
    ):
        raise SubmissionRecoveryError(
            "Hubbard scheduler row does not bind batch/unit/lane/action/request/source epoch"
        )
    return query, row


def recover_hubbard_submission_receipt(
    *,
    repository_root: str | Path,
    intent_path: str | Path,
    provisional_result_path: str | Path,
    later_snapshot_path: str | Path,
) -> RecoveredSubmissionReceipt:
    """Publish a canonical Hubbard receipt after a later exact snapshot."""

    root = Path(repository_root).resolve(strict=True)
    intent = load_hubbard_durable_submission_intent(
        intent_path, repository_root=root
    )
    result, preflight_warning = _load_hubbard_provisional_submission_result_with_audit(
        provisional_result_path,
        repository_root=root,
        intent=intent,
    )
    snapshot = load_combined_scheduler_rows(
        later_snapshot_path, repository_root=root
    )
    if snapshot.captured_time <= _parse_timestamp(
        result.submitted_at, label="Hubbard provisional submission submitted_at"
    ):
        raise SubmissionRecoveryError(
            "Hubbard recovery snapshot must be later than submission"
        )
    query, scheduler_row = _matching_scheduler_row(
        intent=intent, result=result, snapshot=snapshot
    )
    output = (
        root
        / "output"
        / HUBBARD_LANE_ID
        / "acquisition_supervisor/submission_receipts"
        / f"{intent.base.unit.unit_id}.json"
    )
    remote = {
        "cluster_id": result.cluster_id,
        "process_ids": [0],
        "unit_id": intent.base.unit.unit_id,
        "job_batch_name": result.job_batch_name,
        "action_ids": list(intent.base.unit.action_ids),
        "scheduler_identity": f"{result.cluster_id}.0",
        "source_epoch_identity_sha256": intent.source_epoch.identity_sha256,
        "source_epoch_id": intent.source_epoch.source_epoch_id,
        "recovery_evidence": {
            "snapshot": {
                "path": snapshot.path,
                "sha256": snapshot.self_sha256,
                "file_sha256": snapshot.file_sha256,
            },
            "query": query,
            "scheduler_row_sha256": _digest(dict(scheduler_row)),
            "provisional_submission_result": {
                "path": result.path,
                "sha256": result.self_sha256,
                "file_sha256": result.file_sha256,
            },
            "package_manifest": {
                "path": intent.package_manifest_path,
                "sha256": intent.package_manifest_sha256,
            },
            "scheduler_confirmation": True,
            **(
                {"legacy_submission_preflight_warning": preflight_warning}
                if preflight_warning is not None
                else {}
            ),
        },
    }
    receipt = {
        "schema": SUBMISSION_RECEIPT_SCHEMA_V1,
        "lane_id": HUBBARD_LANE_ID,
        "unit_id": intent.base.unit.unit_id,
        "action_ids": list(intent.base.unit.action_ids),
        "intent_sha256": intent.base.intent_sha256,
        "recovered_from_scheduler": True,
        "remote": remote,
        "recorded_at": snapshot.captured_at,
    }
    _atomic_write_immutable(output, receipt)
    return RecoveredSubmissionReceipt(
        path=_relative(root, output, label="Hubbard submission receipt"),
        file_sha256=_file_sha256(output),
        lane_id=HUBBARD_LANE_ID,
        unit_id=intent.base.unit.unit_id,
        cluster_id=result.cluster_id,
        process_ids=(0,),
        snapshot_sha256=snapshot.self_sha256,
    )


__all__ = [
    "BOOTSTRAP_PACKAGE_MANIFEST_V1",
    "COMBINED_SCOPED_SNAPSHOT_V1",
    "HUBBARD_LANE_ID",
    "HUBBARD_SEED_TOKEN",
    "HubbardDurableSubmissionIntent",
    "load_hubbard_durable_submission_intent",
    "load_hubbard_provisional_submission_result",
    "recover_hubbard_submission_receipt",
]
