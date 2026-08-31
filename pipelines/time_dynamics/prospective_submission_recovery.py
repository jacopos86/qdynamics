"""One-way recovery of prospective HH submissions into supervisor receipts.

The initial prospective seed activation intentionally wrote durable supervisor
intents before contacting CHTC, then retained a provisional
``paper_ii_submission_result_v1`` for each successful ``condor_submit``.  Those
results are evidence, but they are not the canonical receipts consumed by
``acquisition_supervisor``.  This module supplies the narrow recovery bridge:

* provisional results may define *unconfirmed* bootstrap monitoring scopes;
* only a later, complete combined queue/history snapshot containing exactly one
  matching cluster/process row may produce a canonical
  ``time_dynamics_submission_receipt_v1``; and
* no function contacts a scheduler, changes the acquisition registry, or
  authorizes a submission.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.acquisition_supervisor import (
    SCOPE_AUTHORITY_SCHEMA_V1,
    SUBMISSION_INTENT_SCHEMA_V1,
    SUBMISSION_RECEIPT_SCHEMA_V1,
    SchedulerScope,
    SubmissionUnit,
)


PROVISIONAL_SUBMISSION_RESULT_V1 = "paper_ii_submission_result_v1"
COMBINED_SCOPED_SNAPSHOT_V1 = "paper_ii_combined_scoped_scheduler_snapshot_v1"
RECONCILIATION_CONTEXT_V1 = "paper_ii_intent_first_reconciliation_context_v1"
ABSENCE_PROOF_V1 = "paper_ii_intent_first_absence_proof_v1"
BOOTSTRAP_SCOPE_MANIFEST_DIGEST_KEY = "bootstrap_scope_manifest_sha256"
HH_LANE_ID = "paper_ii_hh_product_v1"
EXPECTED_HH_UNIT_COUNT = 6
EXPECTED_HH_SEED_SLUGS = frozenset(
    {
        "nph1__crossover",
        "nph1__mott_dominated",
        "nph1__phonon_dominated",
        "nph3__crossover",
        "nph3__mott_dominated",
        "nph3__phonon_dominated",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_SUBMIT_STDOUT = re.compile(
    r"\ASubmitting job\(s\)\.\s*1 job\(s\) submitted to cluster (\d+)\.\s*\Z"
)


class SubmissionRecoveryError(ValueError):
    """Recovery input is stale, ambiguous, noncanonical, or insufficient."""


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
    if not path.is_file():
        raise SubmissionRecoveryError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionRecoveryError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise SubmissionRecoveryError(f"{label} must be a JSON object")
    return value


def _repository_path(
    root: Path,
    value: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    candidate = Path(value)
    target = candidate if candidate.is_absolute() else root / candidate
    target = target.resolve(strict=must_exist)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SubmissionRecoveryError(
            f"{label} must remain inside repository root: {value}"
        ) from exc
    return target


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        value = path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:
        raise SubmissionRecoveryError(
            f"{label} must remain inside repository root: {path}"
        ) from exc
    if not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise SubmissionRecoveryError(f"{label} is not repository-relative")
    return value


def _atomic_write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    content = _pretty_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise SubmissionRecoveryError(
                f"refusing to overwrite immutable recovery artifact: {path}"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _parse_timestamp(value: Any, *, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SubmissionRecoveryError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise SubmissionRecoveryError(f"{label} must include a timezone")
    return parsed


def _sha(value: Any, *, label: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise SubmissionRecoveryError(f"{label} must be a lowercase SHA-256")
    return text


def _binding(value: Any, *, label: str, require_bytes: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SubmissionRecoveryError(f"{label} binding is missing")
    path = str(value.get("path") or "")
    digest = _sha(value.get("sha256"), label=f"{label} digest")
    result: dict[str, Any] = {"path": path, "sha256": digest}
    if not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise SubmissionRecoveryError(
            f"{label} path must be repository-relative"
        )
    if require_bytes:
        try:
            byte_count = int(value.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise SubmissionRecoveryError(
                f"{label} byte count is invalid"
            ) from exc
        if byte_count <= 0:
            raise SubmissionRecoveryError(
                f"{label} byte count must be positive"
            )
        result["bytes"] = byte_count
    return result


def _verify_local_binding(
    root: Path,
    row: Mapping[str, Any],
    *,
    label: str,
    require_bytes: bool,
) -> Path:
    binding = _binding(row, label=label, require_bytes=require_bytes)
    path = _repository_path(
        root, binding["path"], label=label, must_exist=True
    )
    if not path.is_file() or _file_sha256(path) != binding["sha256"]:
        raise SubmissionRecoveryError(f"{label} local digest changed")
    if require_bytes and path.stat().st_size != binding["bytes"]:
        raise SubmissionRecoveryError(f"{label} local byte count changed")
    return path


@dataclass(frozen=True)
class DurableSubmissionIntent:
    path: str
    file_sha256: str
    intent_sha256: str
    lane_id: str
    unit: SubmissionUnit
    recorded_at: str


def load_durable_submission_intent(
    path: str | Path, *, repository_root: str | Path
) -> DurableSubmissionIntent:
    """Load and authenticate one canonical prospective HH submission intent."""

    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="submission intent", must_exist=True
    )
    payload = _read_object(source, label="submission intent")
    unit_raw = payload.get("unit")
    if (
        payload.get("schema") != SUBMISSION_INTENT_SCHEMA_V1
        or payload.get("lane_id") != HH_LANE_ID
        or not isinstance(unit_raw, Mapping)
    ):
        raise SubmissionRecoveryError(
            "submission intent is not a prospective HH supervisor intent"
        )
    try:
        unit = SubmissionUnit(
            unit_id=str(unit_raw.get("unit_id") or ""),
            action_ids=tuple(
                str(value) for value in unit_raw.get("action_ids") or ()
            ),
            payload=dict(unit_raw.get("payload") or {}),
        )
    except (TypeError, ValueError) as exc:
        raise SubmissionRecoveryError("submission intent unit is invalid") from exc
    if (
        not _SAFE_ID.fullmatch(unit.unit_id)
        or not unit.unit_id.startswith("hh-seed-")
        or len(unit.action_ids) != 1
        or not unit.action_ids[0].startswith(
            "build_seed::paper_ii_hh_product_v1::seed::"
        )
    ):
        raise SubmissionRecoveryError("submission intent unit is not singular HH")
    core = {
        "schema": SUBMISSION_INTENT_SCHEMA_V1,
        "lane_id": HH_LANE_ID,
        "unit": unit.to_json_dict(),
    }
    intent_sha = _sha(
        payload.get("intent_sha256"), label="submission intent self digest"
    )
    if intent_sha != _digest(core):
        raise SubmissionRecoveryError("submission intent self digest mismatch")
    recorded_at = str(payload.get("recorded_at") or "")
    _parse_timestamp(recorded_at, label="submission intent recorded_at")
    expected = (
        Path("output")
        / HH_LANE_ID
        / "acquisition_supervisor"
        / "submission_intents"
        / f"{unit.unit_id}.json"
    ).as_posix()
    relative = _relative(root, source, label="submission intent")
    if relative != expected:
        raise SubmissionRecoveryError(
            f"submission intent is not at its canonical path: {relative}"
        )
    return DurableSubmissionIntent(
        path=relative,
        file_sha256=_file_sha256(source),
        intent_sha256=intent_sha,
        lane_id=HH_LANE_ID,
        unit=unit,
        recorded_at=recorded_at,
    )


@dataclass(frozen=True)
class ProvisionalSubmissionResult:
    path: str
    file_sha256: str
    self_sha256: str
    lane_id: str
    unit_id: str
    action_ids: tuple[str, ...]
    cluster_id: int
    process_ids: tuple[int, ...]
    job_batch_name: str
    submitted_at: str
    pre_submit_snapshot_path: str
    pre_submit_snapshot_sha256: str


def _verify_pre_submit_snapshot(
    root: Path,
    *,
    path: str,
    expected_sha256: str,
    intent: DurableSubmissionIntent,
) -> dt.datetime:
    source = _repository_path(
        root, path, label="pre-submit combined snapshot", must_exist=True
    )
    payload = _read_object(source, label="pre-submit combined snapshot")
    declared = _sha(
        payload.get("snapshot_sha256"), label="pre-submit snapshot self digest"
    )
    core = dict(payload)
    core.pop("snapshot_sha256", None)
    if declared != expected_sha256 or declared != _digest(core):
        raise SubmissionRecoveryError("pre-submit snapshot digest mismatch")
    queries = payload.get("queries")
    scope = payload.get("scope")
    reconciliation = payload.get("reconciliation")
    reconciliation_context = payload.get("reconciliation_context")
    if (
        payload.get("schema") != COMBINED_SCOPED_SNAPSHOT_V1
        or payload.get("single_combined_snapshot") is not True
        or payload.get("polling_performed") is not False
        or not isinstance(queries, Mapping)
        or not isinstance(scope, Mapping)
        or not isinstance(reconciliation, Mapping)
        or not isinstance(reconciliation_context, Mapping)
    ):
        raise SubmissionRecoveryError("pre-submit snapshot is not canonical")
    for name in ("queue", "history"):
        query = queries.get(name)
        if (
            not isinstance(query, Mapping)
            or query.get("returncode") != 0
            or not isinstance(query.get("rows"), list)
        ):
            raise SubmissionRecoveryError(
                f"pre-submit {name} query did not succeed"
            )
    batches = scope.get("prospective_job_batch_names")
    job_batch_name = str(intent.unit.payload.get("job_batch_name") or "")
    outcome = reconciliation.get(job_batch_name)
    intents = reconciliation_context.get("intents")
    prior = reconciliation_context.get("prior_snapshot")
    action_id = intent.unit.action_ids[0]
    intent_core = {
        "job_batch_name": job_batch_name,
        "acquisition_unit_id": intent.unit.unit_id,
        "paper_ii_action_id": action_id,
        "paper_ii_execution_request_sha256": _sha(
            intent.unit.payload.get("execution_request_sha256"),
            label="submission intent execution request digest",
        ),
        "intent_sha256": intent.intent_sha256,
        "recorded_at": intent.recorded_at,
    }
    intent_identity = _digest(intent_core)
    bound_intent = intents.get(job_batch_name) if isinstance(intents, Mapping) else None
    proof = outcome.get("absence_proof") if isinstance(outcome, Mapping) else None
    prior_outcomes = prior.get("intent_outcomes") if isinstance(prior, Mapping) else None
    prior_outcome = (
        prior_outcomes.get(job_batch_name)
        if isinstance(prior_outcomes, Mapping)
        else None
    )
    if (
        not isinstance(batches, list)
        or job_batch_name not in batches
        or reconciliation_context.get("schema") != RECONCILIATION_CONTEXT_V1
        or not isinstance(bound_intent, Mapping)
        or dict(bound_intent) != {
            **intent_core,
            "intent_identity_sha256": intent_identity,
        }
        or not isinstance(outcome, Mapping)
        or outcome.get("status") != "verified_absent"
        or outcome.get("match_count") != 0
        or outcome.get("matches") != []
        or outcome.get("intent_identity_sha256") != intent_identity
        or not isinstance(proof, Mapping)
        or proof.get("schema") != ABSENCE_PROOF_V1
        or proof.get("intent_identity_sha256") != intent_identity
        or not isinstance(prior, Mapping)
        or proof.get("prior_snapshot_sha256") != prior.get("snapshot_sha256")
        or proof.get("prior_captured_at") != prior.get("captured_at")
        or proof.get("prior_captured_at_unix_ns")
        != prior.get("captured_at_unix_ns")
        or not isinstance(prior_outcome, Mapping)
        or prior_outcome.get("intent_identity_sha256") != intent_identity
        or proof.get("prior_snapshot_status") != prior_outcome.get("status")
        or prior_outcome.get("status") not in {"indeterminate", "verified_absent"}
    ):
        raise SubmissionRecoveryError(
            "pre-submit snapshot does not prove verified-absent identity"
        )
    captured_time = _parse_timestamp(
        payload.get("captured_at"), label="pre-submit snapshot captured_at"
    )
    prior_time = _parse_timestamp(
        prior.get("captured_at"), label="prior snapshot captured_at"
    )
    if prior_time >= captured_time:
        raise SubmissionRecoveryError(
            "verified-absent snapshot does not follow its prior capture"
        )
    return captured_time


def load_provisional_submission_result(
    path: str | Path,
    *,
    repository_root: str | Path,
    intent: DurableSubmissionIntent,
) -> ProvisionalSubmissionResult:
    """Load and authenticate one provisional ``condor_submit`` result."""

    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="provisional submission result", must_exist=True
    )
    payload = _read_object(source, label="provisional submission result")
    declared = _sha(
        payload.get("submission_result_sha256"),
        label="provisional submission result self digest",
    )
    core = dict(payload)
    core.pop("submission_result_sha256", None)
    if declared != _digest(core):
        raise SubmissionRecoveryError(
            "provisional submission result self digest mismatch"
        )
    action_ids = tuple(str(value) for value in payload.get("action_ids") or ())
    expected_process_ids = tuple(
        int(value) for value in payload.get("expected_process_ids") or ()
    )
    try:
        cluster_id = int(payload.get("cluster_id"))
    except (TypeError, ValueError) as exc:
        raise SubmissionRecoveryError(
            "provisional submission result cluster_id is invalid"
        ) from exc
    job_batch_name = str(payload.get("job_batch_name") or "")
    stdout = str(payload.get("condor_submit_stdout") or "")
    match = _SUBMIT_STDOUT.fullmatch(stdout)
    if (
        payload.get("schema") != PROVISIONAL_SUBMISSION_RESULT_V1
        or payload.get("status") != "submitted"
        or payload.get("lane_id") != intent.lane_id
        or payload.get("unit_id") != intent.unit.unit_id
        or action_ids != intent.unit.action_ids
        or cluster_id < 0
        or expected_process_ids != (0,)
        or payload.get("intent_sha256") != intent.intent_sha256
        or payload.get("intent_path") != intent.path
        or payload.get("lifecycle_validator_status") != "pass"
        or payload.get("dry_run_status") != "pass"
        or payload.get("condor_submit_returncode") != 0
        or payload.get("scheduler_acceptance_query_performed") is not False
        or payload.get("acceptance_evidence")
        != "condor_submit_zero_exit_and_cluster_id_stdout"
        or match is None
        or int(match.group(1)) != cluster_id
        or not job_batch_name
        or job_batch_name != intent.unit.payload.get("job_batch_name")
    ):
        raise SubmissionRecoveryError(
            "provisional submission result identity or preflight contract diverged"
        )
    scope = intent.unit.payload.get("submission_scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("action_ids") != list(action_ids)
        or scope.get("process_ids") != [0]
        or intent.unit.payload.get("expected_process_count") != 1
    ):
        raise SubmissionRecoveryError("submission intent scope is not one process")
    if payload.get("package") != intent.unit.payload.get("package"):
        raise SubmissionRecoveryError("provisional package binding changed")
    if payload.get("runtime_source_archive") != intent.unit.payload.get(
        "runtime_source_archive"
    ):
        raise SubmissionRecoveryError("provisional runtime source binding changed")
    descriptor = _binding(
        payload.get("submit_descriptor"),
        label="submit descriptor",
        require_bytes=False,
    )
    if descriptor["path"] != intent.unit.payload.get("submit_descriptor"):
        raise SubmissionRecoveryError("provisional submit descriptor path changed")
    _verify_local_binding(
        root, payload["package"], label="seed action package", require_bytes=True
    )
    _verify_local_binding(
        root,
        payload["runtime_source_archive"],
        label="runtime source archive",
        require_bytes=True,
    )
    _verify_local_binding(
        root, descriptor, label="submit descriptor", require_bytes=False
    )
    image = _binding(
        payload.get("runtime_image"),
        label="runtime image",
        require_bytes=True,
    )
    if image["path"] != "chtc/time_dynamics_optuna/image.sif":
        raise SubmissionRecoveryError("runtime image path is not canonical")
    execution_sha = _sha(
        intent.unit.payload.get("execution_request_sha256"),
        label="execution request digest",
    )
    if not execution_sha:
        raise AssertionError("unreachable")
    submitted_at = str(payload.get("submitted_at") or "")
    submitted_time = _parse_timestamp(
        submitted_at, label="provisional submission submitted_at"
    )
    snapshot_path = str(payload.get("pre_submit_snapshot_path") or "")
    snapshot_sha = _sha(
        payload.get("pre_submit_snapshot_sha256"),
        label="pre-submit snapshot digest",
    )
    pre_submit_time = _verify_pre_submit_snapshot(
        root,
        path=snapshot_path,
        expected_sha256=snapshot_sha,
        intent=intent,
    )
    if submitted_time <= pre_submit_time:
        raise SubmissionRecoveryError(
            "provisional submission did not follow the verified-absent snapshot"
        )
    return ProvisionalSubmissionResult(
        path=_relative(root, source, label="provisional submission result"),
        file_sha256=_file_sha256(source),
        self_sha256=declared,
        lane_id=intent.lane_id,
        unit_id=intent.unit.unit_id,
        action_ids=action_ids,
        cluster_id=cluster_id,
        process_ids=expected_process_ids,
        job_batch_name=job_batch_name,
        submitted_at=submitted_at,
        pre_submit_snapshot_path=snapshot_path,
        pre_submit_snapshot_sha256=snapshot_sha,
    )


@dataclass(frozen=True)
class CombinedSchedulerRows:
    path: str
    file_sha256: str
    self_sha256: str
    captured_at: str
    captured_time: dt.datetime
    scope_condition: str
    rows: tuple[tuple[str, Mapping[str, Any]], ...]


def load_combined_scheduler_rows(
    path: str | Path, *, repository_root: str | Path
) -> CombinedSchedulerRows:
    """Load one later, single combined queue/history row set."""

    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="later combined scheduler snapshot", must_exist=True
    )
    payload = _read_object(source, label="later combined scheduler snapshot")
    declared = _sha(
        payload.get("snapshot_sha256"), label="later snapshot self digest"
    )
    core = dict(payload)
    core.pop("snapshot_sha256", None)
    if declared != _digest(core):
        raise SubmissionRecoveryError("later snapshot self digest mismatch")
    queries = payload.get("queries")
    scope = payload.get("scope")
    if (
        payload.get("schema") != COMBINED_SCOPED_SNAPSHOT_V1
        or payload.get("single_combined_snapshot") is not True
        or payload.get("polling_performed") is not False
        or not isinstance(queries, Mapping)
        or not isinstance(scope, Mapping)
    ):
        raise SubmissionRecoveryError("later snapshot is not a combined row set")
    condition = str(scope.get("condition") or "")
    if not condition:
        raise SubmissionRecoveryError("later snapshot has no scope condition")
    rows: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[tuple[int, int]] = set()
    for query_name in ("queue", "history"):
        query = queries.get(query_name)
        if (
            not isinstance(query, Mapping)
            or query.get("returncode") != 0
            or not isinstance(query.get("rows"), list)
        ):
            raise SubmissionRecoveryError(
                f"later {query_name} query did not succeed"
            )
        argv = query.get("argv")
        if not isinstance(argv, list) or "-json" not in argv:
            raise SubmissionRecoveryError(
                f"later {query_name} query was not a JSON query"
            )
        for raw in query["rows"]:
            if not isinstance(raw, Mapping):
                raise SubmissionRecoveryError(
                    f"later {query_name} row is not an object"
                )
            try:
                key = (int(raw["ClusterId"]), int(raw["ProcId"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise SubmissionRecoveryError(
                    f"later {query_name} row has no scheduler identity"
                ) from exc
            if key in seen:
                raise SubmissionRecoveryError(
                    f"later combined row set repeats scheduler identity {key}"
                )
            seen.add(key)
            rows.append((query_name, dict(raw)))
    captured_at = str(payload.get("captured_at") or "")
    return CombinedSchedulerRows(
        path=_relative(root, source, label="later combined scheduler snapshot"),
        file_sha256=_file_sha256(source),
        self_sha256=declared,
        captured_at=captured_at,
        captured_time=_parse_timestamp(
            captured_at, label="later snapshot captured_at"
        ),
        scope_condition=condition,
        rows=tuple(rows),
    )


def _matching_scheduler_row(
    *,
    intent: DurableSubmissionIntent,
    result: ProvisionalSubmissionResult,
    snapshot: CombinedSchedulerRows,
) -> tuple[str, Mapping[str, Any]]:
    if not re.search(
        rf"\bClusterId\s*==\s*{result.cluster_id}\b",
        snapshot.scope_condition,
    ):
        raise SubmissionRecoveryError(
            "later combined snapshot did not scope the submitted cluster"
        )
    matches = [
        (query, row)
        for query, row in snapshot.rows
        if int(row["ClusterId"]) == result.cluster_id
        and int(row["ProcId"]) == 0
    ]
    if len(matches) != 1:
        raise SubmissionRecoveryError(
            "later combined snapshot must contain exactly one matching cluster/proc"
        )
    query, row = matches[0]
    expected_action = intent.unit.action_ids[0]
    remote_action = str(
        row.get("PaperIIActionId") or row.get("AcquisitionActionId") or ""
    )
    remote_request = str(row.get("PaperIIExecutionRequestSha256") or "")
    remote_unit = str(row.get("AcquisitionUnitId") or "")
    remote_lane = str(row.get("AcquisitionLaneId") or "")
    try:
        status = int(row["JobStatus"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SubmissionRecoveryError("matching scheduler row has invalid status") from exc
    if (
        str(row.get("JobBatchName") or "") != result.job_batch_name
        or remote_action != expected_action
        or remote_request
        != intent.unit.payload.get("execution_request_sha256")
        or (remote_unit and remote_unit != intent.unit.unit_id)
        or (remote_lane and remote_lane != intent.lane_id)
        or status not in {1, 2, 3, 4, 5, 6, 7}
    ):
        raise SubmissionRecoveryError(
            "matching scheduler row does not bind the exact unit/action/request"
        )
    return query, row


@dataclass(frozen=True)
class RecoveredSubmissionReceipt:
    path: str
    file_sha256: str
    lane_id: str
    unit_id: str
    cluster_id: int
    process_ids: tuple[int, ...]
    snapshot_sha256: str


def recover_submission_receipt(
    *,
    repository_root: str | Path,
    intent_path: str | Path,
    provisional_result_path: str | Path,
    later_snapshot_path: str | Path,
) -> RecoveredSubmissionReceipt:
    """Emit one canonical receipt only after an exact later scheduler match."""

    root = Path(repository_root).resolve(strict=True)
    intent = load_durable_submission_intent(
        intent_path, repository_root=root
    )
    result = load_provisional_submission_result(
        provisional_result_path,
        repository_root=root,
        intent=intent,
    )
    snapshot = load_combined_scheduler_rows(
        later_snapshot_path, repository_root=root
    )
    submitted_time = _parse_timestamp(
        result.submitted_at, label="provisional submission submitted_at"
    )
    if snapshot.captured_time <= submitted_time:
        raise SubmissionRecoveryError(
            "recovery snapshot must be later than the provisional submission"
        )
    query, scheduler_row = _matching_scheduler_row(
        intent=intent, result=result, snapshot=snapshot
    )
    output = (
        root
        / "output"
        / intent.lane_id
        / "acquisition_supervisor"
        / "submission_receipts"
        / f"{intent.unit.unit_id}.json"
    )
    remote = {
        "cluster_id": result.cluster_id,
        "process_ids": [0],
        "unit_id": intent.unit.unit_id,
        "job_batch_name": result.job_batch_name,
        "action_ids": list(intent.unit.action_ids),
        "scheduler_identity": f"{result.cluster_id}.0",
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
            "scheduler_confirmation": True,
        },
    }
    receipt = {
        "schema": SUBMISSION_RECEIPT_SCHEMA_V1,
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_ids": list(intent.unit.action_ids),
        "intent_sha256": intent.intent_sha256,
        "recovered_from_scheduler": True,
        "remote": remote,
        "recorded_at": snapshot.captured_at,
    }
    _atomic_write_immutable(output, receipt)
    return RecoveredSubmissionReceipt(
        path=_relative(root, output, label="canonical submission receipt"),
        file_sha256=_file_sha256(output),
        lane_id=intent.lane_id,
        unit_id=intent.unit.unit_id,
        cluster_id=result.cluster_id,
        process_ids=(0,),
        snapshot_sha256=snapshot.self_sha256,
    )


@dataclass(frozen=True)
class BootstrapScopeManifest:
    path: str
    file_sha256: str
    self_sha256: str
    registry_id: str
    unit_count: int
    scopes: tuple[SchedulerScope, ...]


def build_bootstrap_scope_manifest(
    *,
    repository_root: str | Path,
    registry_id: str,
    provisional_result_paths: Sequence[str | Path],
    output_path: str | Path,
) -> BootstrapScopeManifest:
    """Build six provisional monitoring scopes without confirmation claims."""

    root = Path(repository_root).resolve(strict=True)
    if not registry_id or registry_id != registry_id.strip():
        raise SubmissionRecoveryError("registry_id is invalid")
    if len(provisional_result_paths) != EXPECTED_HH_UNIT_COUNT:
        raise SubmissionRecoveryError(
            f"bootstrap scope manifest requires exactly {EXPECTED_HH_UNIT_COUNT} units"
        )
    rows: list[dict[str, Any]] = []
    scopes: list[SchedulerScope] = []
    unit_ids: set[str] = set()
    action_ids: set[str] = set()
    cluster_ids: set[int] = set()
    seed_slugs: set[str] = set()
    for raw_result_path in provisional_result_paths:
        result_path = _repository_path(
            root,
            raw_result_path,
            label="provisional submission result",
            must_exist=True,
        )
        raw = _read_object(result_path, label="provisional submission result")
        intent_path = str(raw.get("intent_path") or "")
        intent = load_durable_submission_intent(
            intent_path, repository_root=root
        )
        result = load_provisional_submission_result(
            result_path, repository_root=root, intent=intent
        )
        if (
            result.unit_id in unit_ids
            or result.action_ids[0] in action_ids
            or result.cluster_id in cluster_ids
        ):
            raise SubmissionRecoveryError(
                "bootstrap scope manifest repeats a unit/action/cluster identity"
            )
        unit_ids.add(result.unit_id)
        action_ids.add(result.action_ids[0])
        cluster_ids.add(result.cluster_id)
        seed_slugs.add(result.action_ids[0].split("::")[-2])
        scope = SchedulerScope(
            lane_id=HH_LANE_ID,
            cluster_id=result.cluster_id,
            process_ids=(0,),
        )
        scopes.append(scope)
        rows.append(
            {
                "unit_id": result.unit_id,
                "action_ids": list(result.action_ids),
                "job_batch_name": result.job_batch_name,
                "cluster_id": result.cluster_id,
                "process_ids": [0],
                "intent": {
                    "path": intent.path,
                    "sha256": intent.intent_sha256,
                    "file_sha256": intent.file_sha256,
                },
                "provisional_submission_result": {
                    "path": result.path,
                    "sha256": result.self_sha256,
                    "file_sha256": result.file_sha256,
                },
                "scheduler_confirmation": False,
            }
        )
    if seed_slugs != EXPECTED_HH_SEED_SLUGS:
        raise SubmissionRecoveryError(
            "bootstrap scope manifest does not cover the six canonical HH seeds"
        )
    ordered_scopes = tuple(sorted(scopes))
    ordered_rows = sorted(rows, key=lambda row: row["unit_id"])
    core = {
        "schema": SCOPE_AUTHORITY_SCHEMA_V1,
        "registry_id": registry_id,
        "status": "provisional_submission_scope_only",
        "lane_id": HH_LANE_ID,
        "unit_count": EXPECTED_HH_UNIT_COUNT,
        "scopes": [scope.to_json_dict() for scope in ordered_scopes],
        "units": ordered_rows,
        "scope_origin": "condor_submit_zero_exit_and_cluster_id_stdout",
        "scheduler_confirmation": False,
        "scheduler_confirmation_snapshot": None,
        "scope_purpose": "monitoring_and_recovery_only",
        "production_submission_enabled": False,
        "submission_authorized": False,
    }
    manifest = {
        **core,
        BOOTSTRAP_SCOPE_MANIFEST_DIGEST_KEY: _digest(core),
    }
    output = _repository_path(
        root, output_path, label="bootstrap scope manifest", must_exist=False
    )
    _atomic_write_immutable(output, manifest)
    return BootstrapScopeManifest(
        path=_relative(root, output, label="bootstrap scope manifest"),
        file_sha256=_file_sha256(output),
        self_sha256=manifest[BOOTSTRAP_SCOPE_MANIFEST_DIGEST_KEY],
        registry_id=registry_id,
        unit_count=EXPECTED_HH_UNIT_COUNT,
        scopes=ordered_scopes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    recover = commands.add_parser("recover-one")
    recover.add_argument("--repo-root", default=".")
    recover.add_argument("--intent", required=True)
    recover.add_argument("--submission-result", required=True)
    recover.add_argument("--later-snapshot", required=True)
    scopes = commands.add_parser("build-bootstrap-scopes")
    scopes.add_argument("--repo-root", default=".")
    scopes.add_argument("--registry-id", required=True)
    scopes.add_argument("--submission-result", action="append", required=True)
    scopes.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "recover-one":
        result = recover_submission_receipt(
            repository_root=args.repo_root,
            intent_path=args.intent,
            provisional_result_path=args.submission_result,
            later_snapshot_path=args.later_snapshot,
        )
        summary = {
            "status": "canonical_submission_receipt_recovered",
            "receipt": result.path,
            "receipt_file_sha256": result.file_sha256,
            "unit_id": result.unit_id,
            "cluster_id": result.cluster_id,
            "process_ids": list(result.process_ids),
            "snapshot_sha256": result.snapshot_sha256,
            "submission_authorized": False,
        }
    elif args.command == "build-bootstrap-scopes":
        result = build_bootstrap_scope_manifest(
            repository_root=args.repo_root,
            registry_id=args.registry_id,
            provisional_result_paths=args.submission_result,
            output_path=args.output,
        )
        summary = {
            "status": "provisional_bootstrap_scopes_materialized",
            "manifest": result.path,
            "manifest_file_sha256": result.file_sha256,
            "manifest_sha256": result.self_sha256,
            "unit_count": result.unit_count,
            "scheduler_confirmation": False,
            "submission_authorized": False,
        }
    else:  # pragma: no cover
        raise AssertionError(f"unknown command: {args.command}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BOOTSTRAP_SCOPE_MANIFEST_DIGEST_KEY",
    "BootstrapScopeManifest",
    "COMBINED_SCOPED_SNAPSHOT_V1",
    "CombinedSchedulerRows",
    "DurableSubmissionIntent",
    "EXPECTED_HH_UNIT_COUNT",
    "EXPECTED_HH_SEED_SLUGS",
    "HH_LANE_ID",
    "PROVISIONAL_SUBMISSION_RESULT_V1",
    "ProvisionalSubmissionResult",
    "RecoveredSubmissionReceipt",
    "SubmissionRecoveryError",
    "build_bootstrap_scope_manifest",
    "load_combined_scheduler_rows",
    "load_durable_submission_intent",
    "load_provisional_submission_result",
    "main",
    "recover_submission_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
