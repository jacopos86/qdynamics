"""Bind a durable SIAM intent to the generic combined-snapshot protocol.

The SIAM pilot predates the Paper-II acquisition-supervisor intent schema.  Its
submission intent is nevertheless durable and digest-binds the exact action,
package manifest, submit descriptor, and verified remote upload receipt.  This
module authenticates that complete local control plane and derives the six
generic reconciliation fields without relabeling the action JSON digest as an
execution-request digest.

The bridge remains offline: it does not contact Condor, SSH, or the remote
filesystem and it never submits a job.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics import siam_chtc_package
from pipelines.time_dynamics.combined_scheduler_snapshot import (
    CAPTURE_SPEC_V1,
    COMBINED_SCOPED_SNAPSHOT_V1,
    build_combined_snapshot,
    validate_combined_snapshot,
    write_immutable_snapshot,
)


SIAM_SUBMISSION_INTENT_V1 = "siam_nb4_submission_intent_v1"
SIAM_REMOTE_UPLOAD_RECEIPT_V1 = "siam_nb4_remote_upload_receipt_v1"
SIAM_GENERIC_EXECUTION_REQUEST_V1 = (
    "siam_nb4_generic_reconciliation_execution_request_v1"
)
SIAM_GENERIC_INTENT_BRIDGE_V1 = "siam_nb4_generic_intent_bridge_v1"
SIAM_LATER_CAPTURE_PLAN_V1 = "siam_nb4_later_capture_plan_v1"
SIAM_SUBMISSION_RECONCILIATION_V2 = "siam_nb4_submission_reconciliation_v2"
SIAM_SUBMISSION_RECEIPT_V2 = "siam_nb4_submission_receipt_v2"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_VALUE = re.compile(r"[A-Za-z0-9_.:-]+")
_INTENT_FIELDS = (
    "job_batch_name",
    "acquisition_unit_id",
    "paper_ii_action_id",
    "paper_ii_execution_request_sha256",
    "intent_sha256",
    "recorded_at",
)
_SOURCE_INTENT_FIELDS = {
    "bindings",
    "created_at_utc",
    "qiskit_compilation_requested",
    "remote_output_path",
    "remote_workspace",
    "required_next_reconciliation",
    "schema",
    "scientific_identity",
    "source_revision",
    "state",
    "submission_permitted",
}
_REQUIRED_UPLOAD_CHECKS = {
    "condor_dry_run_passed",
    "local_package_manifest_verified",
    "local_source_epoch_control_plane_verified",
    "local_source_tree_contract_verified",
    "local_submit_lifecycle_verified",
    "quota_checked",
    "remote_archive_digest_verified",
    "remote_image_digest_verified",
    "remote_output_path_absent",
    "remote_package_file_digests_verified",
    "remote_source_archive_digest_verified",
    "remote_source_tree_contract_verified",
    "submission_autopilot_absent",
}
_SUBMIT_SUMMARY = re.compile(
    r"1 job\(s\) submitted to cluster (?P<cluster>[1-9][0-9]*)\."
)


class BridgeError(ValueError):
    """The SIAM intent bridge input is stale, ambiguous, or unauthenticated."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
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


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise BridgeError(f"{label} must be a regular nonsymlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BridgeError(f"{label} must be a JSON object")
    return value


def _repository_path(
    root: Path,
    value: str | Path,
    *,
    label: str,
    must_exist: bool = True,
) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise BridgeError(f"{label} must remain inside the repository") from exc
    if must_exist and (not resolved.is_file() or candidate.is_symlink()):
        raise BridgeError(f"{label} must be a regular nonsymlink file")
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BridgeError(f"{label} must be a lowercase SHA-256")
    return value


def _aware_timestamp(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BridgeError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BridgeError(f"{label} must include an explicit offset")
    return text


def _binding(
    root: Path,
    raw: Any,
    *,
    label: str,
    require_bytes: bool,
    expected_path: str,
) -> tuple[Path, dict[str, Any]]:
    expected_fields = {"path", "sha256", "bytes"} if require_bytes else {
        "path",
        "sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise BridgeError(f"{label} binding fields are not canonical")
    path_text = str(raw.get("path") or "")
    if path_text != expected_path:
        raise BridgeError(f"{label} binding path is not canonical")
    path = _repository_path(root, path_text, label=label)
    digest = _sha256(raw.get("sha256"), label=f"{label} digest")
    if _file_sha256(path) != digest:
        raise BridgeError(f"{label} digest changed")
    result: dict[str, Any] = {"path": path_text, "sha256": digest}
    if require_bytes:
        byte_count = raw.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or path.stat().st_size != byte_count
        ):
            raise BridgeError(f"{label} byte count changed")
        result["bytes"] = byte_count
    return path, result


def _string_assignment(text: str, name: str, *, label: str) -> str:
    pattern = re.compile(rf'^\s*{re.escape(name)}\s*=\s*"([^"\\]+)"\s*$')
    matches = [
        match.group(1)
        for line in text.splitlines()
        if (match := pattern.match(line))
    ]
    if len(matches) != 1:
        raise BridgeError(f"submit descriptor must define exactly one {label}")
    return matches[0]


def _plain_assignment(text: str, name: str, *, label: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*(\S+)\s*$")
    matches = [
        match.group(1)
        for line in text.splitlines()
        if (match := pattern.match(line))
    ]
    if len(matches) != 1:
        raise BridgeError(f"submit descriptor must define exactly one {label}")
    return matches[0]


# The compiled-Q route's canonical worker arm id.  Its batch name carries the
# arm string verbatim so the route is legible in the pool; every historical
# batch name keeps the fully hyphenated pre-route form, byte for byte.
_COMPILED_Q_ARM = "exchange_q_compiled"


def _accepted_batch_names(
    *, segment: str, arm: str, ladder_index: int
) -> frozenset[str]:
    historical = f"siam-nb4-{segment}-{arm}-rung{ladder_index}".replace("_", "-")
    if arm != _COMPILED_Q_ARM:
        return frozenset({historical})
    segment_token = str(segment).replace("_", "-")
    return frozenset(
        {
            historical,
            f"siam-nb4-{segment_token}-{arm}-rung{ladder_index}",
        }
    )


def _parse_submit_descriptor(
    path: Path,
    *,
    package_rel: str,
    source_epoch_id: str,
    action_id: str,
    segment: str,
    arm: str,
    ladder_index: int,
) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError("submit descriptor is not readable UTF-8") from exc
    accepted_batches = _accepted_batch_names(
        segment=segment, arm=arm, ladder_index=ladder_index
    )
    observed_epoch = _string_assignment(
        text, "+SIAMSourceEpochId", label="SIAM source epoch"
    )
    observed_action = _string_assignment(
        text, "+SIAMScientificActionId", label="SIAM scientific action"
    )
    observed_batch = _string_assignment(
        text, "+JobBatchName", label="job batch name"
    )
    executable = _plain_assignment(text, "executable", label="executable")
    arguments = _plain_assignment(text, "arguments", label="arguments")
    rebaseline = _plain_assignment(
        text,
        "+SIAMIntentionalSourceRebaseline",
        label="source-rebaseline flag",
    )
    # The transfer input value contains spaces after commas in canonical files;
    # recover it separately while retaining the exact single-assignment rule.
    transfer_lines = [
        line.split("=", 1)[1].strip()
        for line in text.splitlines()
        if line.strip().startswith("transfer_input_files") and "=" in line
    ]
    if len(transfer_lines) != 1:
        raise BridgeError("submit descriptor transfer input set is ambiguous")
    transfer_inputs = transfer_lines[0]
    queue_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("queue")
    ]
    if queue_lines != ["queue 1"]:
        raise BridgeError("submit descriptor must contain exactly queue 1")
    if observed_epoch != source_epoch_id or observed_action != action_id:
        raise BridgeError("submit descriptor scientific identity changed")
    if observed_batch not in accepted_batches:
        raise BridgeError("submit descriptor job batch is not the canonical batch")
    if executable != f"{package_rel}/run_action_apptainer.sh":
        raise BridgeError("submit descriptor executable changed")
    if arguments != action_id or rebaseline != "False":
        raise BridgeError("submit descriptor action/rebaseline binding changed")
    inputs = {value.strip() for value in transfer_inputs.split(",")}
    if package_rel not in inputs:
        raise BridgeError("submit descriptor does not transfer its package")
    return observed_batch


def _validate_bridge_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "source_submission_intent",
        "evidence_bindings",
        "scientific_identity",
        "execution_request",
        "execution_request_sha256",
        "generic_reconciliation_intent",
        "generic_reconciliation_intent_identity_sha256",
        "derivations",
        "remote_submit",
        "bridge_sha256",
    }
    if (
        receipt.get("schema") != SIAM_GENERIC_INTENT_BRIDGE_V1
        or set(receipt) != required
    ):
        raise BridgeError("SIAM intent bridge receipt fields are not canonical")
    execution_request = receipt.get("execution_request")
    generic = receipt.get("generic_reconciliation_intent")
    if not isinstance(execution_request, Mapping) or not isinstance(generic, Mapping):
        raise BridgeError("SIAM intent bridge payload is incomplete")
    execution_sha = _sha256(
        receipt.get("execution_request_sha256"), label="execution request digest"
    )
    if execution_request.get("schema") != SIAM_GENERIC_EXECUTION_REQUEST_V1 or _digest(
        execution_request
    ) != execution_sha:
        raise BridgeError("execution request digest mismatch")
    if set(generic) != set(_INTENT_FIELDS):
        raise BridgeError("generic reconciliation intent fields changed")
    for field in ("job_batch_name", "acquisition_unit_id", "paper_ii_action_id"):
        if not isinstance(generic.get(field), str) or not generic[field]:
            raise BridgeError(f"generic reconciliation {field} is invalid")
    if generic.get("paper_ii_execution_request_sha256") != execution_sha:
        raise BridgeError("generic reconciliation execution digest changed")
    _sha256(generic.get("intent_sha256"), label="source intent digest")
    _aware_timestamp(generic.get("recorded_at"), label="intent recorded_at")
    identity_sha = _sha256(
        receipt.get("generic_reconciliation_intent_identity_sha256"),
        label="generic reconciliation identity digest",
    )
    if identity_sha != _digest(generic):
        raise BridgeError("generic reconciliation identity digest mismatch")
    declared = _sha256(receipt.get("bridge_sha256"), label="bridge self digest")
    core = dict(receipt)
    core.pop("bridge_sha256")
    if declared != _digest(core):
        raise BridgeError("bridge self digest mismatch")
    return dict(receipt)


def build_bridge_receipt(
    submission_intent: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Authenticate one SIAM intent and derive its generic identity."""

    root = Path(repository_root).resolve(strict=True)
    intent_path = _repository_path(
        root, submission_intent, label="SIAM submission intent"
    )
    if intent_path.name != "submission_intent.json":
        raise BridgeError("SIAM submission intent path is not canonical")
    package = intent_path.parent
    package_rel = _relative(root, package)
    package_id = package.name
    intent = _read_object(intent_path, label="SIAM submission intent")
    if (
        intent.get("schema") != SIAM_SUBMISSION_INTENT_V1
        or set(intent) != _SOURCE_INTENT_FIELDS
    ):
        raise BridgeError("SIAM submission intent fields are not canonical")
    if (
        intent.get("state") != "indeterminate"
        or intent.get("submission_permitted") is not False
        or intent.get("required_next_reconciliation") != "verified_absent"
        or intent.get("qiskit_compilation_requested") is not False
    ):
        raise BridgeError("SIAM intent is not the pre-submit indeterminate state")
    recorded_at = _aware_timestamp(
        intent.get("created_at_utc"), label="SIAM intent created_at_utc"
    )
    identity = intent.get("scientific_identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "source_epoch_id",
        "action_id",
    }:
        raise BridgeError("SIAM scientific identity fields are invalid")
    source_epoch_id = str(identity.get("source_epoch_id") or "")
    action_id = str(identity.get("action_id") or "")
    if (
        _SAFE_VALUE.fullmatch(source_epoch_id) is None
        or _SAFE_VALUE.fullmatch(action_id) is None
    ):
        raise BridgeError("SIAM scientific identity values are unsafe")
    revision = str(intent.get("source_revision") or "")
    remote_workspace = str(intent.get("remote_workspace") or "")
    remote_output = str(intent.get("remote_output_path") or "")
    if (
        not remote_workspace.startswith("/")
        or not remote_output
        or Path(remote_output).is_absolute()
    ):
        raise BridgeError("SIAM remote workspace/output binding is invalid")

    bindings = intent.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "action",
        "manifest",
        "remote_upload_receipt",
        "submit",
    }:
        raise BridgeError("SIAM submission intent binding set is invalid")
    expected_paths = {
        "action": f"{package_rel}/input/action.json",
        "manifest": f"{package_rel}/input/manifest.json",
        "submit": f"{package_rel}/submit.sub",
        "remote_upload_receipt": f"{package_rel}/remote_upload_receipt.json",
    }
    verified: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for label in ("action", "manifest", "submit"):
        paths[label], verified[label] = _binding(
            root,
            bindings[label],
            label=label,
            require_bytes=True,
            expected_path=expected_paths[label],
        )
    upload_path, upload_source_binding = _binding(
        root,
        bindings["remote_upload_receipt"],
        label="remote upload receipt",
        require_bytes=False,
        expected_path=expected_paths["remote_upload_receipt"],
    )
    paths["remote_upload_receipt"] = upload_path
    verified["remote_upload_receipt"] = {
        **upload_source_binding,
        "bytes": upload_path.stat().st_size,
    }

    action = _read_object(paths["action"], label="SIAM action")
    manifest = _read_object(paths["manifest"], label="SIAM package manifest")
    upload = _read_object(upload_path, label="SIAM remote upload receipt")
    manifest_identity = {
        "scope": "source_epoch_id/action_id",
        "source_epoch_id": source_epoch_id,
        "action_id": action_id,
    }
    if (
        action.get("schema") != siam_chtc_package.SIAM_NB4_CHTC_ACTION_SCHEMA_V4
        or manifest.get("schema")
        != siam_chtc_package.SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4
        or action.get("package_id") != package_id
        or manifest.get("package_id") != package_id
        or action.get("action_id") != action_id
        or manifest.get("action_id") != action_id
        or action.get("scientific_identity") != manifest_identity
        or manifest.get("scientific_identity") != manifest_identity
        or action.get("source_revision") != revision
        or manifest.get("source_revision") != revision
        or action.get("trajectory_json") != remote_output
        or manifest.get("expected_output") != remote_output
        or action.get("compile_terminal_cost") is not False
        or action.get("qiskit_compilation_requested") is not False
        or dict(manifest.get("checks") or {}).get(
            "qiskit_compilation_requested"
        )
        is not False
        or manifest.get("status") != "PREPARED_NOT_SUBMITTED"
    ):
        raise BridgeError("SIAM action/manifest identity contract changed")
    if (
        manifest.get("action_json_sha256") != verified["action"]["sha256"]
        or manifest.get("submit_descriptor_sha256")
        != verified["submit"]["sha256"]
    ):
        raise BridgeError("SIAM manifest binding digest changed")
    try:
        siam_chtc_package._verify_manifest_files(manifest, repo_root=root)
        siam_chtc_package._verify_source_epoch_control_plane(
            action=action,
            manifest=manifest,
            repo_root=root,
        )
    except ValueError as exc:
        raise BridgeError(f"SIAM package control plane is invalid: {exc}") from exc

    segment = str(action.get("segment") or "")
    arm = str(action.get("arm") or "")
    ladder_index = action.get("activation_cut_ladder_index")
    if (
        not segment
        or not arm
        or isinstance(ladder_index, bool)
        or not isinstance(ladder_index, int)
        or ladder_index < 0
    ):
        raise BridgeError("SIAM action ladder identity is invalid")
    batch = _parse_submit_descriptor(
        paths["submit"],
        package_rel=package_rel,
        source_epoch_id=source_epoch_id,
        action_id=action_id,
        segment=segment,
        arm=arm,
        ladder_index=ladder_index,
    )

    upload_identity = {
        "source_epoch_id": source_epoch_id,
        "action_id": action_id,
    }
    upload_checks = upload.get("checks")
    upload_scheduler = upload.get("scheduler_reconciliation")
    if (
        upload.get("schema") != SIAM_REMOTE_UPLOAD_RECEIPT_V1
        or upload.get("status")
        != "REMOTE_PACKAGE_VERIFIED_AWAITING_RECONCILIATION_NOT_SUBMITTED"
        or upload.get("recorded_at_utc") != recorded_at
        or upload.get("remote_workspace") != remote_workspace
        or upload.get("remote_raw_evidence_preserved") is not True
        or upload.get("source_revision") != revision
        or upload.get("scientific_identity") != upload_identity
        or not isinstance(upload_checks, Mapping)
        or upload_checks.get("qiskit_compilation_requested") is not False
        or any(
            upload_checks.get(name) is not True
            for name in _REQUIRED_UPLOAD_CHECKS
        )
        or not isinstance(upload_scheduler, Mapping)
        or dict(upload_scheduler)
        != {
            "required_next_state": "verified_absent",
            "state": "indeterminate",
            "submission_permitted": False,
        }
    ):
        raise BridgeError("SIAM remote upload receipt is not submission-safe evidence")
    upload_bindings = upload.get("bindings")
    if not isinstance(upload_bindings, Mapping) or dict(upload_bindings) != {
        name: dict(bindings[name]) for name in ("action", "manifest", "submit")
    }:
        raise BridgeError("SIAM remote upload receipt binding set changed")

    intent_file_binding = {
        "path": _relative(root, intent_path),
        "bytes": intent_path.stat().st_size,
        "sha256": _file_sha256(intent_path),
    }
    execution_request = {
        "schema": SIAM_GENERIC_EXECUTION_REQUEST_V1,
        "package_id": package_id,
        "source_revision": revision,
        "scientific_identity": upload_identity,
        "job_batch_name": batch,
        "remote_workspace": remote_workspace,
        "remote_output_path": remote_output,
        "qiskit_compilation_requested": False,
        "evidence_bindings": verified,
    }
    execution_sha = _digest(execution_request)
    generic = {
        "job_batch_name": batch,
        "acquisition_unit_id": package_id,
        "paper_ii_action_id": action_id,
        "paper_ii_execution_request_sha256": execution_sha,
        "intent_sha256": intent_file_binding["sha256"],
        "recorded_at": recorded_at,
    }
    submit_rel = verified["submit"]["path"]
    core = {
        "schema": SIAM_GENERIC_INTENT_BRIDGE_V1,
        "source_submission_intent": intent_file_binding,
        "evidence_bindings": verified,
        "scientific_identity": upload_identity,
        "execution_request": execution_request,
        "execution_request_sha256": execution_sha,
        "generic_reconciliation_intent": generic,
        "generic_reconciliation_intent_identity_sha256": _digest(generic),
        "derivations": {
            "job_batch_name": "submit_descriptor.+JobBatchName",
            "acquisition_unit_id": "package_manifest.package_id",
            "paper_ii_action_id": "submission_intent.scientific_identity.action_id",
            "paper_ii_execution_request_sha256": (
                "sha256(canonical execution_request)"
            ),
            "intent_sha256": "sha256(submission_intent bytes)",
            "recorded_at": "submission_intent.created_at_utc",
        },
        "remote_submit": {
            "working_directory": remote_workspace,
            "argv": ["condor_submit", submit_rel],
        },
    }
    return _validate_bridge_receipt({**core, "bridge_sha256": _digest(core)})


def intent_bound_capture_spec(
    base_capture_spec: Mapping[str, Any],
    bridge_receipt: Mapping[str, Any],
    *,
    prior_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a new capture spec bound to one exact SIAM intent identity."""

    bridge = _validate_bridge_receipt(bridge_receipt)
    if base_capture_spec.get("schema") != CAPTURE_SPEC_V1:
        raise BridgeError("base capture specification schema is unsupported")
    if base_capture_spec.get("reconciliation_intents") not in ({}, None):
        raise BridgeError("base capture already contains reconciliation intents")
    if base_capture_spec.get("prior_snapshot") is not None:
        raise BridgeError("base capture already contains prior-snapshot evidence")
    generic = bridge["generic_reconciliation_intent"]
    batch = generic["job_batch_name"]
    identity = bridge["scientific_identity"]
    scope = base_capture_spec.get("scope")
    lanes = scope.get("lanes") if isinstance(scope, Mapping) else None
    siam_scope = lanes.get("siam") if isinstance(lanes, Mapping) else None
    condition = scope.get("condition") if isinstance(scope, Mapping) else None
    expected_clause = (
        f'(SIAMSourceEpochId == "{identity["source_epoch_id"]}" && '
        f'SIAMScientificActionId == "{identity["action_id"]}")'
    )
    if (
        not isinstance(siam_scope, Mapping)
        or batch not in siam_scope.get("job_batch_names", [])
        or not isinstance(condition, str)
        or expected_clause not in condition
    ):
        raise BridgeError("base capture does not contain the exact SIAM scope")
    queries = base_capture_spec.get("queries")
    if not isinstance(queries, Mapping):
        raise BridgeError("base capture query metadata is missing")
    for name in ("queue", "history"):
        query = queries.get(name)
        argv = query.get("argv") if isinstance(query, Mapping) else None
        if not isinstance(argv, list) or "-attributes" not in argv:
            raise BridgeError(f"base {name} query attributes are missing")
        index = argv.index("-attributes")
        if index + 1 >= len(argv):
            raise BridgeError(f"base {name} query attributes are missing")
        attributes = {value.strip() for value in str(argv[index + 1]).split(",")}
        if not {"SIAMSourceEpochId", "SIAMScientificActionId"}.issubset(attributes):
            raise BridgeError(f"base {name} query omits SIAM identity attributes")
    if prior_snapshot is not None:
        validate_combined_snapshot(prior_snapshot)
        prior_outcome = dict(prior_snapshot.get("reconciliation") or {}).get(batch)
        if (
            not isinstance(prior_outcome, Mapping)
            or prior_outcome.get("status") != "indeterminate"
            or prior_outcome.get("intent_identity_sha256")
            != bridge["generic_reconciliation_intent_identity_sha256"]
        ):
            raise BridgeError("prior snapshot is not the first exact-intent capture")
    result = copy.deepcopy(dict(base_capture_spec))
    result["reconciliation_intents"] = {batch: dict(generic)}
    result["prior_snapshot"] = None if prior_snapshot is None else dict(prior_snapshot)
    return result


def later_capture_plan(
    *,
    first_snapshot: Mapping[str, Any],
    bridge_receipt: Mapping[str, Any],
    first_snapshot_path: str,
) -> dict[str, Any]:
    """Describe the required second capture and gated exact submit argv."""

    bridge = _validate_bridge_receipt(bridge_receipt)
    validate_combined_snapshot(first_snapshot)
    batch = bridge["generic_reconciliation_intent"]["job_batch_name"]
    outcome = dict(first_snapshot.get("reconciliation") or {}).get(batch)
    if (
        first_snapshot.get("schema") != COMBINED_SCOPED_SNAPSHOT_V1
        or not isinstance(outcome, Mapping)
        or outcome.get("status") != "indeterminate"
        or outcome.get("indeterminate_reason")
        != "first_snapshot_for_exact_intent_identity"
        or outcome.get("intent_identity_sha256")
        != bridge["generic_reconciliation_intent_identity_sha256"]
    ):
        raise BridgeError("snapshot is not the first exact-intent SIAM capture")
    queries = first_snapshot["queries"]
    core = {
        "schema": SIAM_LATER_CAPTURE_PLAN_V1,
        "status": "AWAITING_FRESH_COMBINED_CAPTURE",
        "first_snapshot": {
            "path": first_snapshot_path,
            "snapshot_sha256": first_snapshot["snapshot_sha256"],
            "captured_at": first_snapshot["captured_at"],
            "captured_at_unix_ns": first_snapshot["captured_at_unix_ns"],
            "intent_identity_sha256": bridge[
                "generic_reconciliation_intent_identity_sha256"
            ],
        },
        "next_capture": {
            "single_combined_snapshot": True,
            "polling_performed": False,
            "capture_counts": {"queue": 1, "history": 1},
            "must_follow_captured_at_unix_ns": first_snapshot[
                "captured_at_unix_ns"
            ],
            "condition": first_snapshot["scope"]["condition"],
            "queue_argv": queries["queue"]["argv"],
            "history_argv": queries["history"]["argv"],
            "reconciliation_intent": bridge["generic_reconciliation_intent"],
            "prior_snapshot_must_embed": first_snapshot_path,
        },
        "verified_absent_acceptance": {
            "job_batch_name": batch,
            "required_status": "verified_absent",
            "required_match_count": 0,
            "required_intent_identity_sha256": bridge[
                "generic_reconciliation_intent_identity_sha256"
            ],
            "remote_output_path_must_be_reverified_absent": bridge[
                "execution_request"
            ]["remote_output_path"],
        },
        "remote_submit": {
            **bridge["remote_submit"],
            "submission_permitted_now": False,
            "permit_only_after_verified_absent_acceptance": True,
        },
    }
    return {**core, "plan_sha256": _digest(core)}


def verified_absent_reconciliation_receipt(
    *,
    repository_root: str | Path,
    bridge_path: str | Path,
    first_snapshot_path: str | Path,
    verified_snapshot_path: str | Path,
    queue_stdout_path: str | Path,
    history_stdout_path: str | Path,
) -> dict[str, Any]:
    """Authenticate and summarize the two-capture SIAM absence proof."""

    root = Path(repository_root).resolve(strict=True)
    paths = {
        "bridge": _repository_path(root, bridge_path, label="intent bridge"),
        "first": _repository_path(
            root, first_snapshot_path, label="first exact-intent snapshot"
        ),
        "verified": _repository_path(
            root, verified_snapshot_path, label="verified-absent snapshot"
        ),
        "queue": _repository_path(
            root, queue_stdout_path, label="verified-absent queue stdout"
        ),
        "history": _repository_path(
            root, history_stdout_path, label="verified-absent history stdout"
        ),
    }
    bridge = _validate_bridge_receipt(
        _read_object(paths["bridge"], label="intent bridge")
    )
    first = _read_object(paths["first"], label="first exact-intent snapshot")
    verified = _read_object(
        paths["verified"], label="verified-absent snapshot"
    )
    validate_combined_snapshot(first)
    validate_combined_snapshot(
        verified,
        queue_stdout=paths["queue"],
        history_stdout=paths["history"],
    )
    batch = bridge["generic_reconciliation_intent"]["job_batch_name"]
    intent_identity = bridge["generic_reconciliation_intent_identity_sha256"]
    first_outcome = dict(first.get("reconciliation") or {}).get(batch)
    outcome = dict(verified.get("reconciliation") or {}).get(batch)
    proof = outcome.get("absence_proof") if isinstance(outcome, Mapping) else None
    if (
        not isinstance(first_outcome, Mapping)
        or first_outcome.get("status") != "indeterminate"
        or first_outcome.get("intent_identity_sha256") != intent_identity
        or not isinstance(outcome, Mapping)
        or outcome.get("status") != "verified_absent"
        or outcome.get("match_count") != 0
        or outcome.get("matches") != []
        or outcome.get("intent_identity_sha256") != intent_identity
        or not isinstance(proof, Mapping)
        or proof.get("prior_snapshot_sha256") != first["snapshot_sha256"]
        or proof.get("intent_identity_sha256") != intent_identity
    ):
        raise BridgeError("combined snapshots do not prove exact-intent absence")
    if verified["captured_at_unix_ns"] <= first["captured_at_unix_ns"]:
        raise BridgeError("verified-absent snapshot does not follow the first view")

    def file_row(name: str) -> dict[str, Any]:
        path = paths[name]
        return {
            "path": _relative(root, path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }

    query_evidence = {
        name: {
            **file_row(name),
            "stdout_sha256": verified["queries"][name]["stdout_sha256"],
            "row_count": len(verified["queries"][name]["rows"]),
            "returncode": verified["queries"][name]["returncode"],
        }
        for name in ("queue", "history")
    }
    execution = bridge["execution_request"]
    core = {
        "schema": SIAM_SUBMISSION_RECONCILIATION_V2,
        "state": "verified_absent",
        "submission_permitted": True,
        "scientific_identity": bridge["scientific_identity"],
        "job_batch_name": batch,
        "intent_identity_sha256": intent_identity,
        "submission_intent": bridge["source_submission_intent"],
        "generic_intent_bridge": {
            **file_row("bridge"),
            "bridge_sha256": bridge["bridge_sha256"],
        },
        "first_exact_intent_snapshot": {
            **file_row("first"),
            "snapshot_sha256": first["snapshot_sha256"],
            "captured_at": first["captured_at"],
            "captured_at_unix_ns": first["captured_at_unix_ns"],
            "status": first_outcome["status"],
        },
        "verified_absent_snapshot": {
            **file_row("verified"),
            "snapshot_sha256": verified["snapshot_sha256"],
            "captured_at": verified["captured_at"],
            "captured_at_unix_ns": verified["captured_at_unix_ns"],
            "query_evidence": query_evidence,
            "absence_proof": dict(proof),
        },
        "remote_output_path": {
            "path": execution["remote_output_path"],
            "bound_upload_preflight_state": "verified_absent",
            "immediate_recheck_required_before_submit": True,
            "recheck_argv": ["test", "!", "-e", execution["remote_output_path"]],
        },
        "remote_submit": bridge["remote_submit"],
    }
    return {**core, "receipt_sha256": _digest(core)}


def _validated_reconciliation_receipt(
    path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    receipt = _read_object(path, label="verified-absence reconciliation receipt")
    declared = _sha256(
        receipt.get("receipt_sha256"), label="reconciliation receipt digest"
    )
    core = dict(receipt)
    core.pop("receipt_sha256", None)
    if (
        receipt.get("schema") != SIAM_SUBMISSION_RECONCILIATION_V2
        or receipt.get("state") != "verified_absent"
        or receipt.get("submission_permitted") is not True
        or _digest(core) != declared
    ):
        raise BridgeError("verified-absence reconciliation receipt is invalid")
    bridge = receipt.get("generic_intent_bridge")
    first = receipt.get("first_exact_intent_snapshot")
    verified = receipt.get("verified_absent_snapshot")
    queries = verified.get("query_evidence") if isinstance(verified, Mapping) else None
    if (
        not isinstance(bridge, Mapping)
        or not isinstance(first, Mapping)
        or not isinstance(verified, Mapping)
        or not isinstance(queries, Mapping)
    ):
        raise BridgeError("reconciliation receipt evidence bindings are incomplete")
    regenerated = verified_absent_reconciliation_receipt(
        repository_root=root,
        bridge_path=str(bridge.get("path") or ""),
        first_snapshot_path=str(first.get("path") or ""),
        verified_snapshot_path=str(verified.get("path") or ""),
        queue_stdout_path=str(dict(queries.get("queue") or {}).get("path") or ""),
        history_stdout_path=str(
            dict(queries.get("history") or {}).get("path") or ""
        ),
    )
    if receipt != regenerated:
        raise BridgeError("reconciliation receipt diverges from bound evidence")
    return receipt


def build_post_submit_receipt(
    *,
    repository_root: str | Path,
    reconciliation_receipt_path: str | Path,
    cluster_id: int,
    condor_submit_summary: str,
    recorded_at: str,
    remote_output_absence_recheck_passed: bool,
) -> dict[str, Any]:
    """Build the canonical local receipt for one accepted SIAM submit."""

    root = Path(repository_root).resolve(strict=True)
    reconciliation_path = _repository_path(
        root,
        reconciliation_receipt_path,
        label="verified-absence reconciliation receipt",
    )
    reconciliation = _validated_reconciliation_receipt(
        reconciliation_path,
        root=root,
    )
    if (
        isinstance(cluster_id, bool)
        or not isinstance(cluster_id, int)
        or cluster_id <= 0
    ):
        raise BridgeError("submitted cluster ID must be a positive integer")
    match = _SUBMIT_SUMMARY.fullmatch(condor_submit_summary)
    if match is None or int(match.group("cluster")) != cluster_id:
        raise BridgeError("Condor submit summary does not bind the cluster ID")
    if remote_output_absence_recheck_passed is not True:
        raise BridgeError("remote output absence must pass immediately before submit")
    timestamp = _aware_timestamp(recorded_at, label="submission recorded_at")
    bridge_row = reconciliation["generic_intent_bridge"]
    bridge = _validate_bridge_receipt(
        _read_object(root / bridge_row["path"], label="intent bridge")
    )
    action = bridge["evidence_bindings"]["action"]
    manifest = bridge["evidence_bindings"]["manifest"]
    submit = bridge["evidence_bindings"]["submit"]
    upload = bridge["evidence_bindings"]["remote_upload_receipt"]
    core = {
        "schema": SIAM_SUBMISSION_RECEIPT_V2,
        "status": "SUBMITTED",
        "recorded_at_utc": timestamp,
        "cluster_id": cluster_id,
        "process_ids": [0],
        "submitted_jobs": 1,
        "action_count": 1,
        "package_id": bridge["execution_request"]["package_id"],
        "action_id": bridge["scientific_identity"]["action_id"],
        "scientific_identity": bridge["scientific_identity"],
        "source_revision": bridge["execution_request"]["source_revision"],
        "job_batch_name": reconciliation["job_batch_name"],
        "remote_workspace": bridge["remote_submit"]["working_directory"],
        "condor_submit": {
            "argv": bridge["remote_submit"]["argv"],
            "returncode": 0,
            "stdout_summary": condor_submit_summary,
        },
        "scheduler_observation_after_submit": {
            "query_performed": False,
            "state": "not_queried_by_contract",
        },
        "checks": {
            "intent_first_verified_absence": True,
            "remote_output_absence_rechecked_immediately_before_submit": True,
            "exactly_one_submit_issued": True,
            "qiskit_compilation_requested": False,
        },
        "package": {
            "action": action,
            "manifest": manifest,
            "submit_descriptor": submit,
            "remote_upload_receipt": upload,
        },
        "submission_intent": bridge["source_submission_intent"],
        "verified_absent_reconciliation": {
            "path": _relative(root, reconciliation_path),
            "bytes": reconciliation_path.stat().st_size,
            "sha256": _file_sha256(reconciliation_path),
            "receipt_sha256": reconciliation["receipt_sha256"],
        },
    }
    return {**core, "submission_receipt_sha256": _digest(core)}


def validate_post_submit_receipt(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> str:
    """Validate one canonical SIAM post-submit receipt and its evidence."""

    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(root, path, label="SIAM post-submit receipt")
    receipt = _read_object(source, label="SIAM post-submit receipt")
    declared = _sha256(
        receipt.get("submission_receipt_sha256"),
        label="submission receipt digest",
    )
    core = dict(receipt)
    core.pop("submission_receipt_sha256", None)
    condor = receipt.get("condor_submit")
    reconciliation = receipt.get("verified_absent_reconciliation")
    if (
        receipt.get("schema") != SIAM_SUBMISSION_RECEIPT_V2
        or receipt.get("status") != "SUBMITTED"
        or _digest(core) != declared
        or not isinstance(condor, Mapping)
        or condor.get("returncode") != 0
        or not isinstance(reconciliation, Mapping)
    ):
        raise BridgeError("SIAM post-submit receipt is invalid")
    expected = build_post_submit_receipt(
        repository_root=root,
        reconciliation_receipt_path=str(reconciliation.get("path") or ""),
        cluster_id=int(receipt.get("cluster_id")),
        condor_submit_summary=str(condor.get("stdout_summary") or ""),
        recorded_at=str(receipt.get("recorded_at_utc") or ""),
        remote_output_absence_recheck_passed=dict(receipt.get("checks") or {}).get(
            "remote_output_absence_rechecked_immediately_before_submit"
        )
        is True,
    )
    if receipt != expected:
        raise BridgeError("SIAM post-submit receipt diverges from bound evidence")
    return declared


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    content = _pretty_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise BridgeError(f"refusing to overwrite immutable artifact: {path}")
        return "unchanged"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return "created"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge a durable SIAM intent into combined reconciliation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    view = subparsers.add_parser("build-view")
    view.add_argument("--repository-root", type=Path, required=True)
    view.add_argument("--submission-intent", type=Path, required=True)
    view.add_argument("--base-capture-spec", type=Path, required=True)
    view.add_argument("--queue-json", type=Path, required=True)
    view.add_argument("--history-json", type=Path, required=True)
    view.add_argument("--bridge-output", type=Path, required=True)
    view.add_argument("--capture-spec-output", type=Path, required=True)
    view.add_argument("--snapshot-output", type=Path, required=True)
    view.add_argument("--prior-snapshot", type=Path)
    plan = subparsers.add_parser("plan-later")
    plan.add_argument("--repository-root", type=Path, required=True)
    plan.add_argument("--submission-intent", type=Path, required=True)
    plan.add_argument("--first-snapshot", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    receipt = subparsers.add_parser("write-reconciliation")
    receipt.add_argument("--repository-root", type=Path, required=True)
    receipt.add_argument("--bridge", type=Path, required=True)
    receipt.add_argument("--first-snapshot", type=Path, required=True)
    receipt.add_argument("--verified-snapshot", type=Path, required=True)
    receipt.add_argument("--queue-json", type=Path, required=True)
    receipt.add_argument("--history-json", type=Path, required=True)
    receipt.add_argument("--output", type=Path, required=True)
    post = subparsers.add_parser("write-submission-receipt")
    post.add_argument("--repository-root", type=Path, required=True)
    post.add_argument("--reconciliation", type=Path, required=True)
    post.add_argument("--cluster-id", type=int, required=True)
    post.add_argument("--condor-submit-summary", required=True)
    post.add_argument("--recorded-at", required=True)
    post.add_argument("--remote-output-absence-recheck-passed", action="store_true")
    post.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("check-submission-receipt")
    check.add_argument("--repository-root", type=Path, required=True)
    check.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check-submission-receipt":
            digest = validate_post_submit_receipt(
                args.receipt,
                repository_root=args.repository_root,
            )
            result = {"status": "valid", "submission_receipt_sha256": digest}
        elif args.command == "write-submission-receipt":
            receipt = build_post_submit_receipt(
                repository_root=args.repository_root,
                reconciliation_receipt_path=args.reconciliation,
                cluster_id=args.cluster_id,
                condor_submit_summary=args.condor_submit_summary,
                recorded_at=args.recorded_at,
                remote_output_absence_recheck_passed=(
                    args.remote_output_absence_recheck_passed
                ),
            )
            result = {
                "status": _write_immutable_json(args.output, receipt),
                "submission_receipt_sha256": receipt[
                    "submission_receipt_sha256"
                ],
            }
        elif args.command == "write-reconciliation":
            receipt = verified_absent_reconciliation_receipt(
                repository_root=args.repository_root,
                bridge_path=args.bridge,
                first_snapshot_path=args.first_snapshot,
                verified_snapshot_path=args.verified_snapshot,
                queue_stdout_path=args.queue_json,
                history_stdout_path=args.history_json,
            )
            result = {
                "status": _write_immutable_json(args.output, receipt),
                "receipt_sha256": receipt["receipt_sha256"],
                "remote_submit": receipt["remote_submit"],
            }
        elif args.command in {"build-view", "plan-later"}:
            bridge = build_bridge_receipt(
                args.submission_intent,
                repository_root=args.repository_root,
            )
        if args.command == "build-view":
            base = _read_object(args.base_capture_spec, label="base capture spec")
            prior = (
                None
                if args.prior_snapshot is None
                else _read_object(args.prior_snapshot, label="prior snapshot")
            )
            spec = intent_bound_capture_spec(
                base,
                bridge,
                prior_snapshot=prior,
            )
            snapshot = build_combined_snapshot(
                queue_stdout=args.queue_json,
                history_stdout=args.history_json,
                capture_spec=spec,
            )
            statuses = {
                "bridge": _write_immutable_json(args.bridge_output, bridge),
                "capture_spec": _write_immutable_json(
                    args.capture_spec_output, spec
                ),
                "snapshot": write_immutable_snapshot(
                    args.snapshot_output, snapshot
                ),
            }
            result = {
                "statuses": statuses,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "reconciliation": snapshot["reconciliation"][
                    bridge["generic_reconciliation_intent"]["job_batch_name"]
                ],
            }
        elif args.command == "plan-later":
            first = _read_object(args.first_snapshot, label="first snapshot")
            plan = later_capture_plan(
                first_snapshot=first,
                bridge_receipt=bridge,
                first_snapshot_path=str(args.first_snapshot),
            )
            result = {
                "status": _write_immutable_json(args.output, plan),
                "plan_sha256": plan["plan_sha256"],
                "remote_submit": plan["remote_submit"],
            }
    except (BridgeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
