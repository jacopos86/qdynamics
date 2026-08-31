#!/usr/bin/env python3
"""Build stable SIAM acquisition bindings from the current evidence chain.

The SIAM campaign historically wrote useful evidence into cluster-specific
directories.  The outer acquisition supervisor, however, needs two regular
files whose paths do not change when the active cluster changes.  This module
validates the current source epoch, comparator-first ladder, ingested result,
and reader-facing PDF chain before writing those two stable manifests.

No scientific result or PDF is regenerated.  Direct file digests are the
authority; stale backlinks in historical supersession receipts are recorded
as exclusions instead of being silently accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.diagnostics.siam_offline_scoring import (
    _verified_score,
)
from pipelines.time_dynamics.diagnostics.siam_segment_stitching import (
    SIAM_NB4_STITCH_COMPLETION_V1,
    validate_siam_nb4_stitch_completion_receipt,
)
from pipelines.time_dynamics.diagnostics.siam_results_contract import (
    SIAM_RESULTS_PDF_RECEIPT_V2,
    SIAM_RESULTS_PDF_V2,
    SIAM_RESULTS_PROMOTION_V2,
)
from pipelines.time_dynamics.siam_chtc_package import (
    SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4,
    SIAM_SOURCE_EPOCH_CONTRACT_V1,
    validate_siam_nb4_completion_receipt,
)
from pipelines.time_dynamics.siam_nb4_ladder import (
    SIAM_NB4_LADDER_PLAN_V1,
)


SIAM_STABLE_SCIENTIFIC_STATE_V1 = "siam_nb4_stable_scientific_state_v1"
SIAM_STABLE_REPORT_CONTRACT_V1 = "siam_nb4_stable_report_contract_v1"
SIAM_SOURCE_EPOCH_PLANNER_RECEIPT_V1 = (
    "siam_nb4_source_epoch_planner_receipt_v1"
)
SIAM_PREPARED_FRONTIER_V1 = "siam_nb4_prepared_frontier_v1"
SIAM_INCREMENT_RECEIPT_V2 = "siam_nb4_increment_receipt_v2"

DEFAULT_OUTPUT_DIRECTORY = Path(
    "output/siam_screening/nb4_pilot_v1/acquisition_supervisor"
)
DEFAULT_SCIENTIFIC_STATE_NAME = "scientific_state_current.json"
DEFAULT_REPORT_CONTRACT_NAME = "report_contract_current.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


class SIAMStableBindingError(ValueError):
    """The proposed SIAM stable manifest chain is not self-consistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SIAMStableBindingError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise SIAMStableBindingError(f"{label} must be a JSON array")
    return value


def _resolve_regular_file(repository_root: Path, value: str | Path) -> Path:
    root = repository_root.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise SIAMStableBindingError(
            f"artifact is absent or outside the repository: {value}"
        ) from exc
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise SIAMStableBindingError(f"artifact is not a regular file: {value}")
    return resolved


def _relative_path(repository_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _artifact_binding(
    repository_root: Path,
    path: Path,
    content: bytes,
    *,
    schema: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _relative_path(repository_root, path),
        "bytes": len(content),
        "sha256": _file_sha256(content),
    }
    if schema is not None:
        result["schema"] = schema
    return result


def _read_json(
    repository_root: Path,
    value: str | Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = _resolve_regular_file(repository_root, value)
    content = path.read_bytes()
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SIAMStableBindingError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SIAMStableBindingError(f"{label} must contain a JSON object")
    schema = payload.get("schema")
    return (
        payload,
        _artifact_binding(
            repository_root,
            path,
            content,
            schema=str(schema) if isinstance(schema, str) else None,
        ),
        path,
    )


def _read_bound_json(
    repository_root: Path,
    record: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    binding = _mapping(record, label=f"{label} binding")
    payload, actual, path = _read_json(
        repository_root,
        str(binding.get("path") or ""),
        label=label,
    )
    _require_declared_binding(binding, actual, label=label)
    return payload, actual, path


def _read_bound_bytes(
    repository_root: Path,
    record: Any,
    *,
    label: str,
) -> tuple[bytes, dict[str, Any], Path]:
    binding = _mapping(record, label=f"{label} binding")
    path = _resolve_regular_file(repository_root, str(binding.get("path") or ""))
    content = path.read_bytes()
    actual = _artifact_binding(repository_root, path, content)
    _require_declared_binding(binding, actual, label=label)
    return content, actual, path


def _require_declared_binding(
    declared: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    label: str,
) -> None:
    digest = str(declared.get("sha256") or "")
    if not _SHA256.fullmatch(digest) or digest != actual["sha256"]:
        raise SIAMStableBindingError(f"{label} SHA-256 binding mismatch")
    if "bytes" in declared and declared.get("bytes") != actual["bytes"]:
        raise SIAMStableBindingError(f"{label} byte-count binding mismatch")


def _require_same_artifact(
    repository_root: Path,
    declared: Any,
    actual: Mapping[str, Any],
    *,
    label: str,
) -> None:
    binding = _mapping(declared, label=f"{label} binding")
    path = _resolve_regular_file(repository_root, str(binding.get("path") or ""))
    if _relative_path(repository_root, path) != actual["path"]:
        raise SIAMStableBindingError(f"{label} path binding mismatch")
    _require_declared_binding(binding, actual, label=label)


def validate_siam_results_pdf_binding(
    promotion: Mapping[str, Any],
    report_receipt: Mapping[str, Any],
    pdf_bytes: bytes,
) -> None:
    """Validate promotion -> PDF receipt -> PDF without rendering the PDF."""

    promotion_document = dict(promotion)
    promotion_sha256 = promotion_document.pop("promotion_sha256", None)
    if (
        promotion_document.get("schema") != SIAM_RESULTS_PROMOTION_V2
        or promotion_sha256 != _canonical_sha256(promotion_document)
    ):
        raise SIAMStableBindingError("SIAM promotion self-digest mismatch")

    receipt_document = dict(report_receipt)
    receipt_sha256 = receipt_document.pop("receipt_sha256", None)
    if (
        receipt_document.get("schema") != SIAM_RESULTS_PDF_RECEIPT_V2
        or receipt_document.get("render_schema") != SIAM_RESULTS_PDF_V2
        or receipt_sha256 != _canonical_sha256(receipt_document)
    ):
        raise SIAMStableBindingError("SIAM report receipt self-digest mismatch")
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes.startswith(b"%PDF-"):
        raise SIAMStableBindingError("SIAM report is not a PDF byte stream")

    pdf_record = _mapping(receipt_document.get("pdf"), label="report PDF record")
    if (
        pdf_record.get("sha256") != _file_sha256(pdf_bytes)
        or pdf_record.get("bytes") != len(pdf_bytes)
        or receipt_document.get("promotion_sha256") != promotion_sha256
        or receipt_document.get("cell_id") != promotion_document.get("cell_id")
        or receipt_document.get("promotion_role")
        != promotion_document.get("promotion_role")
        or receipt_document.get("qiskit_compilation_used") is not False
    ):
        raise SIAMStableBindingError("SIAM promotion/report/PDF chain mismatch")


def _source_epoch_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != SIAM_SOURCE_EPOCH_PLANNER_RECEIPT_V1:
        raise SIAMStableBindingError("unexpected SIAM source-epoch receipt schema")
    epoch = dict(
        _mapping(receipt.get("source_epoch"), label="SIAM source epoch")
    )
    if (
        epoch.get("schema") != SIAM_SOURCE_EPOCH_CONTRACT_V1
        or not str(epoch.get("source_epoch_id") or "").startswith(
            "siam-nb4-source-"
        )
        or not _REVISION.fullmatch(str(epoch.get("source_revision") or ""))
        or not _SHA256.fullmatch(
            str(epoch.get("source_tree_files_sha256") or "")
        )
    ):
        raise SIAMStableBindingError("invalid SIAM source-epoch identity")
    return epoch


def _promotion_slot_from_completion_arm(value: Any) -> str:
    """Map the source-locked worker arm onto the reader-facing result slot."""

    arm = str(value or "")
    if arm == "avqds":
        return "avqds"
    if arm == "exchange":
        return "full_apm"
    raise SIAMStableBindingError(
        f"unsupported SIAM completion arm for promotion: {arm!r}"
    )


def _validated_result_evidence(
    *,
    repository_root: Path,
    result: Mapping[str, Any],
    epoch: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve either a raw screen or a stitched continuation evidence chain.

    A continuation has two deliberately different identities: the worker's
    raw 401-point segment is certified by its v4 completion receipt, while the
    offline score and promotion bind the independently verified 501-point
    stitched trajectory and stitch-completion receipt.  Keeping those records
    separate prevents a raw continuation digest from being mistaken for the
    full-horizon scientific result.
    """

    stitch_value = result.get("stitch")
    continuation = isinstance(stitch_value, Mapping)
    raw_completion_record = (
        result.get("source_completion_receipt")
        if continuation
        else result.get("completion_receipt")
    )
    raw_trajectory_record = (
        result.get("raw_continuation") if continuation else result.get("run")
    )
    raw_completion, raw_completion_binding, _raw_completion_path = (
        _read_bound_json(
            repository_root,
            raw_completion_record,
            label="raw completion receipt",
        )
    )
    _raw_bytes, raw_trajectory_binding, _raw_trajectory_path = _read_bound_bytes(
        repository_root,
        raw_trajectory_record,
        label="raw trajectory",
    )
    if raw_completion.get("schema") != SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4:
        raise SIAMStableBindingError("current source epoch requires a v4 completion")
    if raw_completion.get("source_epoch") != epoch:
        raise SIAMStableBindingError("completion receipt source epoch mismatch")
    try:
        verified_raw_completion = validate_siam_nb4_completion_receipt(
            raw_completion,
            expected_source_revision=str(epoch["source_revision"]),
            expected_trajectory_sha256=raw_trajectory_binding["sha256"],
            expected_segment=(
                "continue_t1_t5_401" if continuation else "screen_t0_t1_101"
            ),
            expected_source_tree_files_sha256=str(
                epoch["source_tree_files_sha256"]
            ),
        )
    except ValueError as exc:
        raise SIAMStableBindingError(str(exc)) from exc
    if verified_raw_completion.get("action_id") != result.get("action_id"):
        raise SIAMStableBindingError("result and raw completion action IDs differ")

    promotion_completion = raw_completion
    promotion_completion_binding = raw_completion_binding
    scored_trajectory_binding = raw_trajectory_binding
    extra_bindings: dict[str, dict[str, Any]] = {}
    if continuation:
        stitch = _mapping(stitch_value, label="result stitch artifacts")
        stitched, stitched_binding, _stitched_path = _read_bound_bytes(
            repository_root,
            stitch.get("trajectory"),
            label="stitched trajectory",
        )
        del stitched  # Only its immutable byte binding is needed here.
        stitch_completion, stitch_completion_binding, _stitch_receipt_path = (
            _read_bound_json(
                repository_root,
                stitch.get("completion_receipt"),
                label="stitch completion receipt",
            )
        )
        if stitch_completion.get("schema") != SIAM_NB4_STITCH_COMPLETION_V1:
            raise SIAMStableBindingError(
                "continuation result requires a stitched-completion receipt"
            )
        try:
            verified_stitch = validate_siam_nb4_stitch_completion_receipt(
                stitch_completion,
                expected_trajectory_sha256=stitched_binding["sha256"],
                expected_seed_sha256=str(verified_raw_completion["seed_sha256"]),
                expected_arm=str(verified_raw_completion["arm"]),
            )
        except ValueError as exc:
            raise SIAMStableBindingError(str(exc)) from exc
        source_rows = _sequence(
            verified_stitch.get("source_segments"),
            label="stitch completion sources",
        )
        continuation_rows = [
            _mapping(row, label="stitch completion source")
            for row in source_rows
            if isinstance(row, Mapping)
            and row.get("segment") == "continue_t1_t5_401"
        ]
        if len(continuation_rows) != 1 or dict(
            _mapping(
                continuation_rows[0].get("completion_receipt"),
                label="embedded continuation completion receipt",
            )
        ) != dict(raw_completion):
            raise SIAMStableBindingError(
                "stitch proof does not bind the current raw continuation receipt"
            )
        for row_value in source_rows:
            row = _mapping(row_value, label="stitch completion source")
            embedded = _mapping(
                row.get("completion_receipt"),
                label="embedded source completion receipt",
            )
            if (
                row.get("source_revision") != epoch["source_revision"]
                or embedded.get("source_epoch") != epoch
                or embedded.get("source_tree_files_sha256")
                != epoch["source_tree_files_sha256"]
            ):
                raise SIAMStableBindingError(
                    "stitched source segment is outside the current source epoch"
                )
        promotion_completion = stitch_completion
        promotion_completion_binding = stitch_completion_binding
        scored_trajectory_binding = stitched_binding
        extra_bindings = {
            "stitched_completion_receipt": stitch_completion_binding,
            "stitched_trajectory": stitched_binding,
        }

    score, score_binding, _score_path = _read_bound_json(
        repository_root, result.get("score"), label="offline score"
    )
    try:
        verified_score = _verified_score(score)
    except (KeyError, TypeError, ValueError) as exc:
        raise SIAMStableBindingError(str(exc)) from exc
    score_method = _mapping(verified_score.get("method"), label="score method")
    score_source = _mapping(verified_score.get("source"), label="score source")
    if (
        score_method.get("code_revision") != epoch["source_revision"]
        or score_method.get("structural_policy")
        != verified_raw_completion["arm"]
        or score_source.get("seed_sha256")
        != verified_raw_completion["seed_sha256"]
        or score_source.get("run_sha256")
        != scored_trajectory_binding["sha256"]
    ):
        raise SIAMStableBindingError("offline score source identity mismatch")

    promotion_slot = _promotion_slot_from_completion_arm(
        promotion_completion.get("arm")
    )
    return {
        "mode": "continuation" if continuation else "screen",
        "raw_completion": raw_completion,
        "raw_completion_binding": raw_completion_binding,
        "raw_trajectory_binding": raw_trajectory_binding,
        "promotion_completion": promotion_completion,
        "promotion_completion_binding": promotion_completion_binding,
        "score": score,
        "score_binding": score_binding,
        "verified_score": verified_score,
        "scored_trajectory_binding": scored_trajectory_binding,
        "promotion_slot": promotion_slot,
        "extra_bindings": extra_bindings,
    }


def _validate_ladder_and_frontier(
    *,
    repository_root: Path,
    epoch: Mapping[str, Any],
    ladder: Mapping[str, Any],
    ladder_binding: Mapping[str, Any],
    frontier: Mapping[str, Any],
) -> None:
    if (
        ladder.get("schema") != SIAM_NB4_LADDER_PLAN_V1
        or ladder.get("source_epoch_id") != epoch["source_epoch_id"]
        or ladder.get("source_revision") != epoch["source_revision"]
    ):
        raise SIAMStableBindingError("ladder plan is not bound to the source epoch")
    planned = _sequence(ladder.get("next_actions"), label="ladder next_actions")
    planned_by_id = {
        str(_mapping(row, label="ladder action").get("action_id") or ""): dict(row)
        for row in planned
    }
    if "" in planned_by_id or len(planned_by_id) != len(planned):
        raise SIAMStableBindingError("ladder plan needs unique nonempty next actions")
    for action in planned_by_id.values():
        if (
            action.get("exact_reference_blind") is not True
            or action.get("qiskit_compilation_requested") is not False
        ):
            raise SIAMStableBindingError(
                "ladder next action violates exact-blind/Qiskit-free policy"
            )

    frontier_status = str(frontier.get("status") or "")
    allowed_statuses = (
        {"PREPARED", "SUBMITTED"}
        if planned_by_id
        else {"IDLE", "CLOSED"}
    )
    if (
        frontier.get("schema") != SIAM_PREPARED_FRONTIER_V1
        or frontier.get("source_epoch_id") != epoch["source_epoch_id"]
        or frontier.get("source_revision") != epoch["source_revision"]
        or frontier.get("plan_sha256") != ladder_binding["sha256"]
        or frontier.get("qiskit_compilation_requested") is not False
        or frontier_status not in allowed_statuses
    ):
        raise SIAMStableBindingError("prepared frontier is not bound to the ladder")
    plan_path = _resolve_regular_file(
        repository_root, str(frontier.get("plan") or "")
    )
    if _relative_path(repository_root, plan_path) != ladder_binding["path"]:
        raise SIAMStableBindingError("prepared frontier names a different ladder plan")

    prepared_rows = _sequence(
        frontier.get("prepared_actions"), label="prepared frontier actions"
    )
    prepared_by_id: dict[str, Mapping[str, Any]] = {}
    for row_value in prepared_rows:
        row = _mapping(row_value, label="prepared action")
        action = dict(_mapping(row.get("action"), label="prepared action contract"))
        action_id = str(action.get("action_id") or "")
        if action_id in prepared_by_id or action_id not in planned_by_id:
            raise SIAMStableBindingError("prepared frontier action set mismatch")
        if action != planned_by_id[action_id] or row.get("source_epoch") != epoch:
            raise SIAMStableBindingError("prepared action is not source/plan locked")
        prepared_by_id[action_id] = row
    if set(prepared_by_id) != set(planned_by_id):
        raise SIAMStableBindingError("prepared frontier omits a planned action")

    submissions = _sequence(frontier.get("submissions"), label="frontier submissions")
    submitted_id_rows = [
        str(_mapping(row, label="frontier submission").get("action_id") or "")
        for row in submissions
    ]
    submitted_ids = set(submitted_id_rows)
    if (
        "" in submitted_ids
        or len(submitted_ids) != len(submitted_id_rows)
        or not submitted_ids.issubset(planned_by_id)
    ):
        raise SIAMStableBindingError("frontier submission action set mismatch")
    if frontier_status == "SUBMITTED" and submitted_ids != set(planned_by_id):
        raise SIAMStableBindingError("submitted frontier does not cover the plan")
    if not planned_by_id and (prepared_by_id or submitted_ids):
        raise SIAMStableBindingError(
            "idle/closed frontier must not contain prepared or submitted actions"
        )


def _discover_stale_backlink_exclusions(
    *,
    repository_root: Path,
    search_roots: Sequence[Path],
    current_result_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    exclusions: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for candidate in sorted(search_root.rglob("*supersession_receipt.json")):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                payload, binding, _path = _read_json(
                    repository_root, resolved, label="historical supersession receipt"
                )
            except SIAMStableBindingError:
                continue
            increment = payload.get("increment_receipt")
            if not isinstance(increment, Mapping):
                continue
            try:
                linked = _resolve_regular_file(
                    repository_root, str(increment.get("path") or "")
                )
            except SIAMStableBindingError:
                continue
            if (
                _relative_path(repository_root, linked)
                != current_result_binding["path"]
            ):
                continue
            declared = str(increment.get("sha256") or "")
            if declared == current_result_binding["sha256"]:
                continue
            exclusions.append(
                {
                    "artifact": binding,
                    "json_pointer": "/increment_receipt/sha256",
                    "declared_sha256": declared,
                    "current_sha256": current_result_binding["sha256"],
                    "reason": (
                        "historical supersession receipt predates final current-result "
                        "receipt backlinks and is not authoritative current state"
                    ),
                }
            )
    return exclusions


def _with_self_digest(core: Mapping[str, Any], field: str) -> dict[str, Any]:
    document = dict(core)
    document[field] = _canonical_sha256(core)
    return document


def build_siam_stable_binding_documents(
    *,
    repository_root: str | Path,
    source_epoch_receipt: str | Path,
    ladder_plan: str | Path,
    prepared_frontier_receipt: str | Path,
    result_receipt: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate current SIAM evidence and return the two stable documents."""

    root = Path(repository_root).resolve()
    source, source_binding, source_path = _read_json(
        root, source_epoch_receipt, label="source-epoch receipt"
    )
    epoch = _source_epoch_from_receipt(source)
    ladder, ladder_binding, _ladder_path = _read_json(
        root, ladder_plan, label="ladder plan"
    )
    frontier, frontier_binding, _frontier_path = _read_json(
        root, prepared_frontier_receipt, label="prepared frontier receipt"
    )
    result, result_binding, result_path = _read_json(
        root, result_receipt, label="current result receipt"
    )

    _validate_ladder_and_frontier(
        repository_root=root,
        epoch=epoch,
        ladder=ladder,
        ladder_binding=ladder_binding,
        frontier=frontier,
    )
    if (
        result.get("schema") != SIAM_INCREMENT_RECEIPT_V2
        or result.get("source_epoch") != epoch["source_epoch_id"]
        or result.get("source_revision") != epoch["source_revision"]
        or not str(result.get("status") or "").startswith("INGESTED")
    ):
        raise SIAMStableBindingError("result receipt is not current-epoch ingestion")
    scheduler = _mapping(result.get("scheduler"), label="result scheduler outcome")
    if scheduler.get("job_status") != 4 or scheduler.get("exit_code") != 0:
        raise SIAMStableBindingError("result receipt is not a successful terminal job")
    qiskit = _mapping(result.get("qiskit_compilation"), label="Qiskit declaration")
    if qiskit.get("required") is not False or qiskit.get("performed") is not False:
        raise SIAMStableBindingError("current SIAM result unexpectedly uses Qiskit")

    evidence = _validated_result_evidence(
        repository_root=root,
        result=result,
        epoch=epoch,
    )
    raw_completion = evidence["raw_completion"]
    raw_completion_binding = evidence["raw_completion_binding"]
    promotion_completion_binding = evidence["promotion_completion_binding"]
    score_binding = evidence["score_binding"]
    trajectory_sha256 = evidence["scored_trajectory_binding"]["sha256"]
    results = _mapping(result.get("results"), label="result artifacts")
    promotion, promotion_binding, _promotion_path = _read_bound_json(
        root, results.get("promotion"), label="results promotion"
    )
    report_receipt, report_receipt_binding, report_receipt_path = _read_bound_json(
        root, results.get("pdf_receipt"), label="report receipt"
    )
    pdf_bytes, pdf_binding, pdf_path = _read_bound_bytes(
        root, results.get("pdf"), label="report PDF"
    )

    history = _sequence(ladder.get("history"), label="ladder history")
    action_id = str(result.get("action_id") or "")
    history_rows = [
        _mapping(row, label="ladder history row")
        for row in history
        if isinstance(row, Mapping) and row.get("action_id") == action_id
    ]
    if len(history_rows) != 1:
        raise SIAMStableBindingError("ladder history does not uniquely bind the result")
    _require_same_artifact(
        root,
        {
            "path": history_rows[0].get("completion_receipt"),
            "sha256": history_rows[0].get("completion_receipt_sha256"),
        },
        raw_completion_binding,
        label="ladder completion receipt",
    )
    _require_same_artifact(
        root,
        {
            "path": history_rows[0].get("score"),
            "sha256": history_rows[0].get("score_sha256"),
        },
        score_binding,
        label="ladder score",
    )

    next_frontier = _mapping(result.get("next_frontier"), label="result next frontier")
    _require_same_artifact(
        root,
        {
            "path": next_frontier.get("plan"),
            "sha256": next_frontier.get("plan_sha256"),
        },
        ladder_binding,
        label="result ladder plan",
    )
    _require_same_artifact(
        root,
        {
            "path": next_frontier.get("prepared_frontier_receipt"),
            "sha256": next_frontier.get("prepared_frontier_receipt_sha256"),
        },
        frontier_binding,
        label="result prepared frontier",
    )

    validate_siam_results_pdf_binding(promotion, report_receipt, pdf_bytes)
    if results.get("counts") != promotion.get("counts"):
        raise SIAMStableBindingError("result and promotion counts differ")
    promotion_slot_name = str(evidence["promotion_slot"])
    promotion_slot = _mapping(
        _mapping(promotion.get("slots"), label="promotion slots").get(
            promotion_slot_name
        ),
        label=f"{promotion_slot_name} promotion slot",
    )
    promotion_inputs = _mapping(
        promotion_slot.get("inputs"), label=f"{promotion_slot_name} promotion inputs"
    )
    if (
        _mapping(
            promotion_inputs.get("completion_receipt"),
            label="promotion completion input",
        ).get("file_sha256")
        != promotion_completion_binding["sha256"]
        or _mapping(
            promotion_inputs.get("score"), label="promotion score input"
        ).get("file_sha256")
        != score_binding["sha256"]
    ):
        raise SIAMStableBindingError("promotion inputs do not bind current evidence")

    campaign_root = source_path.parents[2]
    exclusions = _discover_stale_backlink_exclusions(
        repository_root=root,
        search_roots=(campaign_root, pdf_path.parent / "status"),
        current_result_binding=result_binding,
    )
    shared_bindings = {
        "source_epoch_receipt": source_binding,
        "ladder_plan": ladder_binding,
        "prepared_frontier_receipt": frontier_binding,
        "result_receipt": result_binding,
        "completion_receipt": promotion_completion_binding,
        "raw_completion_receipt": raw_completion_binding,
        "raw_trajectory": evidence["raw_trajectory_binding"],
        "scored_trajectory": evidence["scored_trajectory_binding"],
        "offline_score": score_binding,
        "results_promotion": promotion_binding,
        "report_receipt": report_receipt_binding,
        "report_pdf": pdf_binding,
        **evidence["extra_bindings"],
    }
    provenance_policy = {
        "current_authority": "direct_regular_file_sha256",
        "source_epoch_receipt_planner_snapshot_authoritative": False,
        "source_epoch_receipt_planner_snapshot_reason": (
            "the receipt freezes epoch creation; current ladder_plan is the frontier"
        ),
        "historical_mutable_backlink_exclusions": exclusions,
    }

    scientific_core = {
        "schema": SIAM_STABLE_SCIENTIFIC_STATE_V1,
        "campaign_id": "siam_nb4_pilot_v1",
        "deterministic": True,
        "source_epoch": epoch,
        "state": {
            "phase": ladder.get("phase"),
            "complete": ladder.get("complete"),
            "blocked": ladder.get("blocked"),
            "cost_comparison_admissible": ladder.get(
                "cost_comparison_admissible"
            ),
            "frontier_status": frontier.get("status"),
            "next_action_ids": [
                str(_mapping(row, label="next action").get("action_id") or "")
                for row in _sequence(
                    ladder.get("next_actions"), label="ladder next_actions"
                )
            ],
            "submitted_cluster_ids": sorted(
                int(_mapping(row, label="submission")["cluster_id"])
                for row in _sequence(
                    frontier.get("submissions"), label="frontier submissions"
                )
            ),
            "latest_ingested_action_id": action_id,
            "latest_ingested_cluster_id": int(result["cluster_id"]),
            "latest_ingested_status": str(result["status"]),
            "latest_ingested_evidence_mode": evidence["mode"],
            "latest_ingested_promotion_slot": evidence["promotion_slot"],
            "raw_trajectory_sha256": evidence["raw_trajectory_binding"][
                "sha256"
            ],
            "trajectory_sha256": trajectory_sha256,
            "result_counts": dict(
                _mapping(promotion.get("counts"), label="promotion counts")
            ),
            "pair_classification": _mapping(
                promotion.get("pair_comparison"), label="pair comparison"
            ).get("classification"),
        },
        "bindings": shared_bindings,
        "provenance_policy": provenance_policy,
    }
    report_core = {
        "schema": SIAM_STABLE_REPORT_CONTRACT_V1,
        "campaign_id": "siam_nb4_pilot_v1",
        "deterministic": True,
        "source_epoch": epoch,
        "report": {
            "receipt_path": report_receipt_binding["path"],
            "receipt_sha256": report_receipt_binding["sha256"],
            "pdf_path": pdf_binding["path"],
            "pdf_sha256": pdf_binding["sha256"],
            "pdf_bytes": pdf_binding["bytes"],
            "pdf_page_count": _mapping(
                report_receipt.get("pdf"), label="report PDF record"
            ).get("page_count"),
            "promotion_sha256": promotion.get("promotion_sha256"),
            "cell_id": promotion.get("cell_id"),
            "promotion_role": promotion.get("promotion_role"),
            "pair_classification": _mapping(
                promotion.get("pair_comparison"), label="pair comparison"
            ).get("classification"),
            "result_counts": dict(
                _mapping(promotion.get("counts"), label="promotion counts")
            ),
            "qiskit_compilation_used": False,
        },
        "bindings": shared_bindings,
        "provenance_policy": provenance_policy,
    }
    return (
        _with_self_digest(scientific_core, "scientific_state_sha256"),
        _with_self_digest(report_core, "report_contract_sha256"),
    )


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_replace_if_changed(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_siam_stable_binding_documents(
    *,
    output_directory: str | Path,
    scientific_state: Mapping[str, Any],
    report_contract: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Atomically update the two regular-file stable SIAM bindings."""

    output = Path(output_directory)
    scientific_path = output / DEFAULT_SCIENTIFIC_STATE_NAME
    report_path = output / DEFAULT_REPORT_CONTRACT_NAME
    _atomic_replace_if_changed(scientific_path, _pretty_json_bytes(scientific_state))
    _atomic_replace_if_changed(report_path, _pretty_json_bytes(report_contract))
    return scientific_path, report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-epoch-receipt", type=Path, required=True)
    parser.add_argument("--ladder-plan", type=Path, required=True)
    parser.add_argument("--prepared-frontier-receipt", type=Path, required=True)
    parser.add_argument("--result-receipt", type=Path, required=True)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    output_directory = args.output_directory
    if not output_directory.is_absolute():
        output_directory = repository_root / output_directory
    scientific_state, report_contract = build_siam_stable_binding_documents(
        repository_root=repository_root,
        source_epoch_receipt=args.source_epoch_receipt,
        ladder_plan=args.ladder_plan,
        prepared_frontier_receipt=args.prepared_frontier_receipt,
        result_receipt=args.result_receipt,
    )
    scientific_path, report_path = write_siam_stable_binding_documents(
        output_directory=output_directory,
        scientific_state=scientific_state,
        report_contract=report_contract,
    )
    print(
        json.dumps(
            {
                "scientific_state": str(scientific_path),
                "scientific_state_sha256": _file_sha256(
                    scientific_path.read_bytes()
                ),
                "report_contract": str(report_path),
                "report_contract_sha256": _file_sha256(report_path.read_bytes()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
