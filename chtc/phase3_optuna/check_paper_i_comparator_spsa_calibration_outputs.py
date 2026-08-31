#!/usr/bin/env python3
"""Validate fetched Paper-I comparator SPSA calibration outputs.

Calibration artifacts checked here are schedule-search artifacts only.  They are
not manuscript/table evidence and are not promoted by this checker.  When asked,
the checker writes a reviewable schedule-lock *candidate* keyed by method-target
for a later approved production-rerun workflow.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna.generate_paper_i_comparator_spsa_calibration_records import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUEUE_OUTPUT_ROOT,
    RECORDS_TSV,
)
from pipelines.exact_bench.paper_i_comparator_spsa_calibration import (  # noqa: E402
    HH_GEO_QEB_TABLEIII_REPAIR_SCOPE,
    HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY,
    PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
    target_by_id,
    validate_hh_tableiii_repair_records,
    validate_method_id,
)

DEFAULT_RECORDS = DEFAULT_OUTPUT_DIR / RECORDS_TSV
DEFAULT_OUTPUT_ROOT = DEFAULT_QUEUE_OUTPUT_ROOT
DEFAULT_SUMMARY = Path("raw_outputs/paper_i_comparator_spsa_calibration_v1_output_check.json")

SUMMARY_SCHEMA = "paper_i_comparator_spsa_calibration_output_check_v1"
SCHEDULE_LOCK_SCHEMA = "paper_i_comparator_spsa_schedule_lock_candidate_v1"
CALIBRATION_EVIDENCE_ROLE = "calibration_only_not_manuscript_table_evidence"
SCHEDULE_LOCK_EVIDENCE_ROLE = "calibration_schedule_lock_candidate_only_not_table_evidence"
_INT_SCHEDULE_FIELDS = {"family_informed_spsa_eval_repeats", "family_informed_spsa_avg_last"}
_STRICT_STATUS_USABLE_POLICY = "strict_status_completed_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[Mapping[str, Any]] | None:
    if not path.exists():
        return None
    events: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL line {line_no} is not an object in {path}")
        events.append(payload)
    return events


def _load_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing records file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [{str(key): "" if value is None else str(value) for key, value in row.items()} for row in csv.DictReader(handle, delimiter="\t")]
    required = {"record_id", "profile_id", "method_id", "algorithm_id", "target_id", "case_ids_json", "record_output_dir"}
    missing = required - set(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"records file {path} missing columns: {sorted(missing)}")
    return rows


def _case_ids_from_row(row: Mapping[str, str]) -> list[str]:
    payload = json.loads(str(row.get("case_ids_json") or "[]"))
    if not isinstance(payload, list) or not payload or not all(isinstance(item, str) and item.strip() for item in payload):
        raise ValueError(f"record {row.get('record_id')} has malformed case_ids_json")
    return [str(item).strip() for item in payload]


def _record_dir(output_root: Path, row: Mapping[str, str]) -> Path:
    rel = str(row.get("record_output_dir") or "").strip() or f"records/{row['record_id']}"
    return Path(output_root) / rel


def _finite_number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _expected_schedule_fields(row: Mapping[str, str]) -> set[str]:
    try:
        payload = json.loads(str(row.get("search_space_fields_json") or "[]"))
    except Exception:
        return set()
    if not isinstance(payload, list):
        return set()
    return {str(item) for item in payload}


def _status_usable_policy(row: Mapping[str, str]) -> str:
    raw = str(row.get("calibration_usable_status_policy") or "").strip()
    if raw in {"", "none", "strict"}:
        return _STRICT_STATUS_USABLE_POLICY
    return raw


def _event_calibration_usable(event: Mapping[str, Any]) -> bool:
    if "all_cases_calibration_usable" in event:
        return bool(event.get("all_cases_calibration_usable"))
    return bool(event.get("all_cases_usable"))


def _schedule_errors(
    schedule: Any,
    *,
    method_id: str,
    label: str,
    expected_fields: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(schedule, Mapping):
        return [f"{label}:schedule_not_mapping"]
    allowed = set(PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD.get(method_id, ()))
    keys = {str(key) for key in schedule}
    extra = sorted(keys - allowed)
    if extra:
        errors.append(f"{label}:schedule_fields_not_allowed:{extra}")
    if expected_fields is not None:
        missing = sorted(expected_fields - keys)
        unexpected = sorted(keys - expected_fields)
        if missing or unexpected:
            errors.append(f"{label}:schedule_fields_mismatch:missing={missing}:unexpected={unexpected}")
    for key, value in schedule.items():
        numeric = _finite_number(value)
        if numeric is None or numeric <= 0.0:
            errors.append(f"{label}:schedule_value_not_positive_finite:{key}={value!r}")
        if str(key) in _INT_SCHEDULE_FIELDS and numeric is not None and not float(numeric).is_integer():
            errors.append(f"{label}:integer_schedule_value_not_integral:{key}={value!r}")
    return errors


def _identity_errors(payload: Mapping[str, Any], *, row: Mapping[str, str], case_ids: Sequence[str], label: str) -> list[str]:
    errors: list[str] = []
    for field in ("record_id", "method_id", "target_id"):
        if str(payload.get(field) or "") != str(row.get(field) or ""):
            errors.append(f"{label}:{field}_mismatch:{payload.get(field)!r}!={row.get(field)!r}")
    for field in ("rerun_scope", "adaptive_refit_engine", "warm_start_schedule_lock_json", "warm_start_schedule_key"):
        expected = str(row.get(field) or "").strip()
        if expected and str(payload.get(field) or "") != expected:
            errors.append(f"{label}:{field}_mismatch:{payload.get(field)!r}!={expected!r}")
    payload_cases = payload.get("case_ids")
    if payload_cases is not None and list(payload_cases) != list(case_ids):
        errors.append(f"{label}:case_ids_mismatch:{payload_cases!r}!={list(case_ids)!r}")
    return errors


def _warm_start_schedule_from_row(row: Mapping[str, str], *, method_id: str, target_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    lock_value = str(row.get("warm_start_schedule_lock_json") or "").strip()
    key_value = str(row.get("warm_start_schedule_key") or "").strip()
    if not lock_value and not key_value:
        return None, []
    errors: list[str] = []
    expected_key = f"{method_id}::{target_id}"
    if not lock_value:
        return None, ["warm_start_schedule_key_set_without_lock_json"]
    if key_value != expected_key:
        errors.append(f"warm_start_schedule_key_mismatch:{key_value!r}!={expected_key!r}")
    lock_path = Path(lock_value) if Path(lock_value).is_absolute() else REPO_ROOT / lock_value
    if not lock_path.exists():
        errors.append(f"warm_start_schedule_lock_missing:{lock_value}")
        return None, errors
    try:
        payload = _read_json(lock_path)
        entries = (payload or {}).get("method_target_schedules")
        if not isinstance(entries, Mapping):
            errors.append("warm_start_schedule_lock_missing_method_target_schedules")
            return None, errors
        entry = entries.get(key_value)
        if not isinstance(entry, Mapping):
            errors.append(f"warm_start_schedule_lock_missing_key:{key_value}")
            return None, errors
        schedule = entry.get("schedule")
        if not isinstance(schedule, Mapping):
            errors.append(f"warm_start_schedule_lock_missing_schedule:{key_value}")
            return None, errors
        schedule_errors = _schedule_errors(
            schedule,
            method_id=method_id,
            label="warm_start_schedule_lock",
            expected_fields=_expected_schedule_fields(row),
        )
        errors.extend(schedule_errors)
        if schedule_errors:
            return None, errors
        return {str(key): value for key, value in schedule.items()}, errors
    except Exception as exc:
        errors.append(f"warm_start_schedule_lock_invalid:{type(exc).__name__}:{exc}")
        return None, errors


def _event_errors(events: Sequence[Mapping[str, Any]], *, row: Mapping[str, str], case_ids: Sequence[str]) -> list[str]:
    errors: list[str] = []
    method_id = str(row["method_id"])
    expected_fields = _expected_schedule_fields(row)
    for index, event in enumerate(events):
        label = f"trial_events[{index}]"
        if str(event.get("schema") or "") != "paper_i_comparator_spsa_trial_event_v1":
            errors.append(f"{label}:schema_mismatch:{event.get('schema')!r}")
        errors.extend(_identity_errors(event, row=row, case_ids=case_ids, label=label))
        errors.extend(_schedule_errors(event.get("schedule"), method_id=method_id, label=label, expected_fields=expected_fields))
        event_cases = event.get("case_results")
        if not isinstance(event_cases, list) or len(event_cases) != len(case_ids):
            errors.append(f"{label}:case_results_count_mismatch")
        else:
            seen = sorted(str(case.get("case_id")) for case in event_cases if isinstance(case, Mapping))
            if seen != sorted(case_ids):
                errors.append(f"{label}:case_results_case_ids_mismatch:{seen!r}!={sorted(case_ids)!r}")
    return errors


def validate_record_output(row: Mapping[str, str], *, output_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    record_id = str(row["record_id"])
    status_policy = _status_usable_policy(row)
    if status_policy not in {_STRICT_STATUS_USABLE_POLICY, HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY}:
        errors.append(f"calibration_usable_status_policy_unknown:{status_policy!r}")
    try:
        method_id = validate_method_id(str(row.get("method_id") or row.get("algorithm_id") or ""))
        target = target_by_id(str(row.get("target_id") or ""))
        case_ids = _case_ids_from_row(row)
        if case_ids != list(target.case_ids):
            errors.append(f"record_case_ids_mismatch:{case_ids!r}!={list(target.case_ids)!r}")
    except Exception as exc:
        method_id = str(row.get("method_id") or "")
        case_ids = []
        errors.append(f"record_contract_invalid:{type(exc).__name__}:{exc}")
    expected_schedule_fields = _expected_schedule_fields(row)
    expected_warm_start_schedule, warm_start_errors = _warm_start_schedule_from_row(
        row,
        method_id=method_id,
        target_id=str(row.get("target_id") or ""),
    )
    errors.extend(warm_start_errors)
    if str(row.get("profile_id") or "") != PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID:
        errors.append(f"profile_id_mismatch:{row.get('profile_id')!r}")
    if str(row.get("algorithm_id") or "") != method_id:
        errors.append(f"algorithm_method_mismatch:{row.get('algorithm_id')!r}!={method_id!r}")
    if str(row.get("run_class") or "") not in {"smoke", "calibration_candidate_not_table_evidence"}:
        errors.append(f"run_class_invalid:{row.get('run_class')!r}")

    out_dir = _record_dir(output_root, row)
    if not out_dir.exists():
        return {
            "record_id": record_id,
            "method_id": method_id,
            "target_id": row.get("target_id"),
            "case_ids": case_ids,
            "status": "missing",
            "ok": False,
            "errors": [*errors, f"output_dir_missing:{out_dir}"],
            "warnings": warnings,
            "output_dir": str(out_dir),
        }

    paths = {
        "summary": out_dir / "summary.json",
        "best_schedule": out_dir / "best_schedule.json",
        "heartbeat": out_dir / "heartbeat.json",
        "current_best": out_dir / "progress" / "current_best.json",
        "trial_events": out_dir / "progress" / "trial_events.jsonl",
    }
    payloads: dict[str, Mapping[str, Any] | None] = {}
    for label, path in paths.items():
        if label == "trial_events":
            continue
        try:
            payloads[label] = _read_json(path)
        except Exception as exc:
            payloads[label] = None
            errors.append(f"{label}_json_invalid:{type(exc).__name__}:{exc}")
        if payloads[label] is None:
            errors.append(f"{label}_missing:{path}")
    try:
        events = _read_jsonl(paths["trial_events"])
    except Exception as exc:
        events = None
        errors.append(f"trial_events_jsonl_invalid:{type(exc).__name__}:{exc}")
    if events is None:
        errors.append(f"trial_events_missing:{paths['trial_events']}")
        events = []

    summary = payloads.get("summary") or {}
    best_schedule = payloads.get("best_schedule") or {}
    heartbeat = payloads.get("heartbeat") or {}
    current_best = payloads.get("current_best") or {}
    if summary:
        if str(summary.get("schema") or "") != "paper_i_comparator_spsa_calibration_summary_v1":
            errors.append(f"summary:schema_mismatch:{summary.get('schema')!r}")
        if str(summary.get("profile_id") or "") != PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID:
            errors.append(f"summary:profile_id_mismatch:{summary.get('profile_id')!r}")
        if str(summary.get("evidence_role") or "") != CALIBRATION_EVIDENCE_ROLE:
            errors.append(f"summary:evidence_role_mismatch:{summary.get('evidence_role')!r}")
        errors.extend(_identity_errors(summary, row=row, case_ids=case_ids, label="summary"))
    if best_schedule:
        if str(best_schedule.get("schema") or "") != "paper_i_comparator_spsa_best_schedule_v1":
            errors.append(f"best_schedule:schema_mismatch:{best_schedule.get('schema')!r}")
        errors.extend(_identity_errors(best_schedule, row=row, case_ids=case_ids, label="best_schedule"))
        errors.extend(
            _schedule_errors(
                best_schedule.get("schedule"),
                method_id=method_id,
                label="best_schedule",
                expected_fields=expected_schedule_fields,
            )
        )
    if heartbeat:
        if str(heartbeat.get("schema") or "") != "paper_i_comparator_spsa_calibration_heartbeat_v1":
            errors.append(f"heartbeat:schema_mismatch:{heartbeat.get('schema')!r}")
        errors.extend(_identity_errors(heartbeat, row=row, case_ids=case_ids, label="heartbeat"))
    if current_best:
        if str(current_best.get("schema") or "") != "paper_i_comparator_spsa_current_best_v1":
            errors.append(f"current_best:schema_mismatch:{current_best.get('schema')!r}")
        errors.extend(_identity_errors(current_best, row=row, case_ids=case_ids, label="current_best"))
        best = current_best.get("best")
        if isinstance(best, Mapping):
            errors.extend(
                _schedule_errors(
                    best.get("schedule"),
                    method_id=method_id,
                    label="current_best.best",
                    expected_fields=expected_schedule_fields,
                )
            )
        elif best is not None:
            errors.append("current_best:best_not_mapping_or_null")
        for required in ("completed_trial_count", "usable_trial_count", "progress_events_jsonl"):
            if required not in current_best:
                errors.append(f"current_best:missing_{required}")
        if status_policy == HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY and "calibration_usable_trial_count" not in current_best:
            errors.append("current_best:missing_calibration_usable_trial_count")
    errors.extend(_event_errors(events, row=row, case_ids=case_ids))
    if expected_warm_start_schedule is not None:
        for label, payload in (
            ("summary", summary),
            ("best_schedule", best_schedule),
            ("heartbeat", heartbeat),
            ("current_best", current_best),
        ):
            if payload:
                if int(payload.get("warm_start_enqueued_trial_count") or -1) != 1:
                    errors.append(f"{label}:warm_start_enqueued_trial_count_mismatch:{payload.get('warm_start_enqueued_trial_count')!r}")
                if dict(payload.get("warm_start_schedule") or {}) != expected_warm_start_schedule:
                    errors.append(f"{label}:warm_start_schedule_mismatch")
        trial_zero = next((event for event in events if _finite_number(event.get("trial_number")) == 0.0), None)
        if trial_zero is None:
            errors.append("warm_start_trial_zero_missing")
        else:
            if not bool(trial_zero.get("warm_start_trial")):
                errors.append("trial_events[0]:warm_start_trial_false")
            if dict(trial_zero.get("schedule") or {}) != expected_warm_start_schedule:
                errors.append("trial_events[0]:warm_start_schedule_mismatch")
    events_by_trial = {int(event.get("trial_number")): event for event in events if _finite_number(event.get("trial_number")) is not None}
    if summary and best_schedule:
        if best_schedule.get("trial_number") != summary.get("best_trial_number"):
            errors.append(
                f"best_schedule_trial_mismatch:{best_schedule.get('trial_number')!r}!={summary.get('best_trial_number')!r}"
            )
        if dict(best_schedule.get("schedule") or {}) != dict(summary.get("best_schedule") or {}):
            errors.append("best_schedule_summary_schedule_mismatch")
        selected_trial = _finite_number(summary.get("best_trial_number"))
        if selected_trial is not None:
            selected_event = events_by_trial.get(int(selected_trial))
            if selected_event is None:
                errors.append(f"best_trial_missing_from_events:{int(selected_trial)}")
            elif dict(selected_event.get("schedule") or {}) != dict(summary.get("best_schedule") or {}):
                errors.append("best_trial_event_summary_schedule_mismatch")
    if current_best and isinstance(current_best.get("best"), Mapping) and summary:
        current_best_event = current_best["best"]
        if current_best_event.get("trial_number") != summary.get("best_trial_number"):
            errors.append(
                f"current_best_trial_mismatch:{current_best_event.get('trial_number')!r}!={summary.get('best_trial_number')!r}"
            )
        if dict(current_best_event.get("schedule") or {}) != dict(summary.get("best_schedule") or {}):
            errors.append("current_best_summary_schedule_mismatch")

    completed = int(summary.get("completed_trial_count") or 0) if summary else 0
    usable = int(summary.get("usable_trial_count") or 0) if summary else 0
    calibration_usable = (
        int(summary.get("calibration_usable_trial_count") or usable) if summary else sum(1 for event in events if _event_calibration_usable(event))
    )
    requested = int(summary.get("n_trials_requested") or row.get("n_trials") or 0) if summary else int(row.get("n_trials") or 0)
    if completed != len(events):
        errors.append(f"completed_trial_count_mismatch:summary={completed}:events={len(events)}")
    if current_best and int(current_best.get("completed_trial_count") or -1) != len(events):
        errors.append(f"current_best_completed_trial_count_mismatch:{current_best.get('completed_trial_count')}:{len(events)}")
    if heartbeat and int(heartbeat.get("completed_trial_count") or -1) != len(events):
        errors.append(f"heartbeat_completed_trial_count_mismatch:{heartbeat.get('completed_trial_count')}:{len(events)}")
    event_usable = sum(1 for event in events if bool(event.get("all_cases_usable")))
    if usable != event_usable:
        errors.append(f"usable_trial_count_mismatch:summary={usable}:events={event_usable}")
    event_calibration_usable = sum(1 for event in events if _event_calibration_usable(event))
    if calibration_usable != event_calibration_usable:
        errors.append(f"calibration_usable_trial_count_mismatch:summary={calibration_usable}:events={event_calibration_usable}")
    if requested > 0 and completed < requested:
        warnings.append(f"trial_count_incomplete:completed={completed}:requested={requested}")
    summary_ok = bool(summary.get("ok")) if summary else False
    best_usable = bool(best_schedule.get("usable")) if best_schedule else False
    best_calibration_usable = bool(best_schedule.get("calibration_usable", best_usable)) if best_schedule else False
    if summary_ok and not best_calibration_usable:
        errors.append("summary_ok_but_best_schedule_not_calibration_usable")
    if status_policy == _STRICT_STATUS_USABLE_POLICY and usable <= 0:
        errors.append("no_usable_trials")
    if status_policy == HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY and calibration_usable <= 0:
        errors.append("no_calibration_usable_trials")

    status = "usable"
    if any(error.startswith(("summary_missing", "best_schedule_missing", "heartbeat_missing", "current_best_missing", "trial_events_missing")) for error in errors):
        status = "incomplete"
    if errors and status != "incomplete":
        status = "failed"
    if status == "usable" and warnings:
        status = "usable_with_warnings"
    return {
        "record_id": record_id,
        "method_id": method_id,
        "target_id": row.get("target_id"),
        "case_ids": case_ids,
        "status": status,
        "ok": status in {"usable", "usable_with_warnings"},
        "errors": errors,
        "warnings": warnings,
        "output_dir": str(out_dir),
        "summary_json": str(paths["summary"]),
        "best_schedule_json": str(paths["best_schedule"]),
        "progress_current_best_json": str(paths["current_best"]),
        "progress_trial_events_jsonl": str(paths["trial_events"]),
        "completed_trial_count": completed,
        "usable_trial_count": usable,
        "strict_usable_trial_count": usable,
        "calibration_usable_trial_count": calibration_usable,
        "status_usable_policy": status_policy,
        "n_trials_requested": requested,
        "best_schedule": dict(best_schedule.get("schedule") or {}) if isinstance(best_schedule, Mapping) else {},
        "best_objective": summary.get("best_objective") if isinstance(summary, Mapping) else None,
        "best_trial_number": summary.get("best_trial_number") if isinstance(summary, Mapping) else None,
    }


def _schedule_lock_payload(*, records_path: Path, output_root: Path, record_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    schedules: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = defaultdict(dict)
    for report in record_reports:
        if not bool(report.get("ok")):
            continue
        method_id = str(report["method_id"])
        target_id = str(report["target_id"])
        entry = {
            "method_id": method_id,
            "target_id": target_id,
            "case_ids": list(report.get("case_ids") or ()),
            "schedule": dict(report.get("best_schedule") or {}),
            "best_objective": report.get("best_objective"),
            "best_trial_number": report.get("best_trial_number"),
            "status_usable_policy": report.get("status_usable_policy"),
            "strict_usable_trial_count": report.get("strict_usable_trial_count"),
            "calibration_usable_trial_count": report.get("calibration_usable_trial_count"),
            "source_record_id": report.get("record_id"),
            "source_summary_json": report.get("summary_json"),
            "source_best_schedule_json": report.get("best_schedule_json"),
            "promotion_status": "candidate_not_promoted_user_approval_required",
            "table_evidence_status": "not_table_evidence",
        }
        key = f"{method_id}::{target_id}"
        schedules[key] = entry
        nested[method_id][target_id] = entry
    return {
        "schema": SCHEDULE_LOCK_SCHEMA,
        "profile_id": PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
        "evidence_role": SCHEDULE_LOCK_EVIDENCE_ROLE,
        "table_evidence_status": "not_table_evidence",
        "promotion_status": "not_promoted_user_approval_required",
        "records_path": str(records_path),
        "output_root": str(output_root),
        "generated_utc": _utc_now(),
        "usable_schedule_count": len(schedules),
        "method_target_schedules": schedules,
        "schedules_by_method": {method: dict(targets) for method, targets in sorted(nested.items())},
    }


def validate_outputs(
    *,
    records_path: Path = DEFAULT_RECORDS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    summary_path: Path | None = None,
    schedule_lock_candidate_path: Path | None = None,
    validate_schedule_lock_path: Path | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    records = _load_records(records_path)
    bundle_errors: list[str] = []
    repair_rows = [row for row in records if str(row.get("repair_scope") or "").strip() == HH_GEO_QEB_TABLEIII_REPAIR_SCOPE]
    if repair_rows:
        try:
            validate_hh_tableiii_repair_records(repair_rows)
        except Exception as exc:
            bundle_errors.append(f"hh_tableiii_repair_matrix_invalid:{type(exc).__name__}:{exc}")
    reports = [validate_record_output(row, output_root=output_root) for row in records]
    status_counts = Counter(str(report["status"]) for report in reports)
    status_by_method: dict[str, Counter[str]] = defaultdict(Counter)
    for report in reports:
        status_by_method[str(report.get("method_id"))][str(report.get("status"))] += 1
    missing = [dict(report) for report in reports if str(report.get("status")) == "missing"]
    incomplete = [dict(report) for report in reports if str(report.get("status")) == "incomplete"]
    failed = [dict(report) for report in reports if str(report.get("status")) == "failed"]
    usable = [dict(report) for report in reports if str(report.get("status")) in {"usable", "usable_with_warnings"}]
    warnings = [warning for report in reports for warning in report.get("warnings", [])]
    schedule_lock = _schedule_lock_payload(records_path=records_path, output_root=output_root, record_reports=reports)
    schedule_lock_validation: dict[str, Any] | None = None
    if schedule_lock_candidate_path is not None:
        schedule_lock_candidate_path.parent.mkdir(parents=True, exist_ok=True)
        schedule_lock_candidate_path.write_text(json.dumps(schedule_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if validate_schedule_lock_path is not None:
        existing = _read_json(validate_schedule_lock_path)
        expected_schedules = schedule_lock["method_target_schedules"]
        actual_schedules = dict((existing or {}).get("method_target_schedules") or {})
        schedule_lock_validation = {
            "path": str(validate_schedule_lock_path),
            "ok": actual_schedules == expected_schedules,
            "expected_keys": sorted(expected_schedules),
            "actual_keys": sorted(actual_schedules),
        }
    ok = (not failed) if bool(allow_incomplete) else (not failed and not missing and not incomplete and len(usable) == len(records))
    if bundle_errors:
        ok = False
    if schedule_lock_validation is not None and not bool(schedule_lock_validation["ok"]):
        ok = False
    summary = {
        "schema": SUMMARY_SCHEMA,
        "profile_id": PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
        "evidence_role": CALIBRATION_EVIDENCE_ROLE,
        "table_evidence_status": "not_table_evidence",
        "promotion_status": "not_promoted",
        "records_path": str(records_path),
        "output_root": str(output_root),
        "ok": bool(ok),
        "allow_incomplete": bool(allow_incomplete),
        "record_count": len(records),
        "usable_count": len(usable),
        "missing_count": len(missing),
        "incomplete_count": len(incomplete),
        "failed_count": len(failed),
        "bundle_errors": bundle_errors,
        "status_counts": dict(sorted(status_counts.items())),
        "status_by_method": {method: dict(counter) for method, counter in sorted(status_by_method.items())},
        "missing": missing,
        "incomplete": incomplete,
        "failed": failed,
        "usable": usable,
        "warnings": warnings,
        "schedule_lock_candidate": schedule_lock,
        "schedule_lock_candidate_path": None if schedule_lock_candidate_path is None else str(schedule_lock_candidate_path),
        "schedule_lock_validation": schedule_lock_validation,
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Paper-I comparator SPSA calibration fetched outputs.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-root", "--root", dest="output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--schedule-lock-candidate", type=Path, default=None)
    parser.add_argument("--validate-schedule-lock", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_outputs(
        records_path=args.records,
        output_root=args.output_root,
        summary_path=args.summary,
        schedule_lock_candidate_path=args.schedule_lock_candidate,
        validate_schedule_lock_path=args.validate_schedule_lock,
        allow_incomplete=bool(args.allow_incomplete),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if bool(summary.get("ok")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
