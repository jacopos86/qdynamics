#!/usr/bin/env python3
"""Build Hubbard-only SNAKE recalibration candidates from Paper-I table audits.

The input is the repaired fixed-accuracy table audit emitted by
``pipelines/reporting/build_paper_i_fixed_accuracy_table_pdf.py``.  The output is
an intentionally narrow handoff manifest for Item 7: only the expected
``(hubbard, SNAKE, weak)`` and ``(hubbard, SNAKE, strong)`` rows may become
Route-A recalibration records, and only when the audit shows stale/poor or
still-running current-best-not-reached evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna.paper_i_clean_ladder_contract import PAPER_I_CLEAN_TAU_PHYS  # noqa: E402
from pipelines.exact_bench.table_i_canonical_cases import TABLE_I_CLEAN_NPH3_REF4_PROFILE  # noqa: E402
from pipelines.static_adapt.route_identity import ROUTE_ID_A  # noqa: E402

AUDIT_SCHEMA = "paper_i_fixed_accuracy_table_audit_v1"
CANDIDATE_SCHEMA = "paper_i_hubbard_snake_recalibration_candidates_v1"
ALGORITHM_ID = "static_family_native_adapt_phase3"
METHOD = "SNAKE"
FAMILY = "hubbard"
EXPECTED_KEYS = ((FAMILY, METHOD, "weak"), (FAMILY, METHOD, "strong"))
EXPECTED_CASE_ID_BY_REGIME = {
    "weak": "hubbard_L2_clean_weak",
    "strong": "hubbard_L2_clean_strong",
}
ELIGIBLE_TERMINAL_STATUSES = {"not_reached", "failed"}
ELIGIBLE_RUNNING_THRESHOLD_STATUSES = {"running_current_best_not_reached"}
STALE_TARGET_TOKENS = ("stale", "invalid_target", "threshold_mismatch")
REACHED_THRESHOLD_STATUSES = {
    "ok_native_first_hit",
    "ok_terminal_only_method",
    "running_current_best_reached",
    "reached",
}
RECALIBRATION_TSV_FIELDS = (
    "paper_i_hubbard_snake_recalibration",
    "paper_i_hubbard_snake_recalibration_candidate_manifest_json",
    "paper_i_hubbard_snake_recalibration_candidate_key",
    "paper_i_hubbard_snake_recalibration_source_audit_json",
    "paper_i_hubbard_snake_recalibration_source_audit_sha256",
    "paper_i_hubbard_snake_recalibration_source_case_id",
    "paper_i_hubbard_snake_recalibration_source_status",
    "paper_i_hubbard_snake_recalibration_source_threshold_status",
    "paper_i_hubbard_snake_recalibration_source_target_profile",
    "paper_i_hubbard_snake_recalibration_source_threshold",
    "paper_i_hubbard_snake_recalibration_source_payload_path",
    "paper_i_hubbard_snake_recalibration_source_payload_sha256",
    "paper_i_hubbard_snake_recalibration_source_record_id",
    "paper_i_hubbard_snake_recalibration_source_row_index",
    "paper_i_hubbard_snake_recalibration_source_payload_path_kind",
    "paper_i_hubbard_snake_recalibration_source_running_state",
    "paper_i_hubbard_snake_recalibration_source_terminal_state",
    "paper_i_hubbard_snake_recalibration_source_complete_trial_count",
    "paper_i_hubbard_snake_recalibration_source_trial_count",
    "paper_i_hubbard_snake_recalibration_source_condor_job",
    "paper_i_hubbard_snake_recalibration_reason",
)


def _sha256_file(path: Path) -> str | None:
    try:
        resolved = Path(path)
    except TypeError:
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_matches(value: Any, expected: float) -> bool:
    try:
        parsed = float(value)
    except Exception:
        return False
    return math.isfinite(parsed) and math.isclose(parsed, float(expected), rel_tol=0.0, abs_tol=1e-15)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _raw_key(row: Mapping[str, Any]) -> list[Any] | None:
    value = row.get("expected_key") or row.get("normalized_key")
    return list(value) if isinstance(value, (list, tuple)) else None


def _family(row: Mapping[str, Any]) -> str:
    return str(row.get("canonical_family") or row.get("family_normalized") or row.get("family") or "").strip()


def _method(row: Mapping[str, Any]) -> str:
    return str(row.get("method") or row.get("method_normalized") or "").strip()


def _regime(row: Mapping[str, Any]) -> str:
    return str(row.get("regime") or "").strip()


def _lane(row: Mapping[str, Any]) -> str:
    return str(row.get("lane") or "").strip()


def _case_id(row: Mapping[str, Any]) -> str:
    return str(row.get("case_id") or row.get("benchmark_id") or "").strip()


def _source_path(row: Mapping[str, Any]) -> str:
    return str(row.get("source_payload_path") or row.get("payload_path") or "").strip()


def _is_phonon_row(row: Mapping[str, Any]) -> bool:
    stage = str(row.get("paper_i_ladder_stage") or "").strip()
    if stage and stage != "not_applicable_nonphonon":
        return True
    for field in ("n_ph_work", "n_ph_ref"):
        value = row.get(field)
        if value not in {None, ""}:
            return True
    return False


def _is_missing_source(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or "").strip().lower()
    threshold_status = str(row.get("threshold_status") or "").strip().lower()
    if status == "missing" or threshold_status == "missing":
        return True
    if str(row.get("missing_reason") or "").strip():
        return True
    if str(row.get("source_payload_missing_reason") or "").strip():
        return True
    return not bool(_source_path(row))


def _is_stale_target(row: Mapping[str, Any]) -> bool:
    values = " ".join(
        str(row.get(field) or "").strip().lower()
        for field in ("status", "threshold_status", "eligibility_reason", "skip_reason")
    )
    if any(token in values for token in STALE_TARGET_TOKENS):
        return True
    return row.get("threshold_matches_requested") is False


def _source_algorithm_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("algorithm_id") or row.get("source_algorithm_id") or "").strip()
    if explicit:
        return explicit
    if _lane(row) == "snake" and _method(row) == METHOD:
        return ALGORITHM_ID
    record_id = str(row.get("source_record_id") or row.get("record_id") or "").strip()
    if "__" in record_id:
        tail = record_id.rsplit("__", 1)[-1].strip()
        if tail:
            return tail
    return ""


def _candidate_key(*, regime: str, case_id: str) -> str:
    return "|".join((FAMILY, METHOD, regime, case_id, ALGORITHM_ID))


def _rejection(row: Mapping[str, Any], *, reason: str, detail: str | None = None) -> dict[str, Any]:
    return {
        "reason": reason,
        "detail": detail,
        "expected_key": _raw_key(row),
        "family": _family(row),
        "method": _method(row),
        "regime": _regime(row),
        "lane": _lane(row),
        "case_id": _case_id(row),
        "status": row.get("status"),
        "threshold_status": row.get("threshold_status"),
        "cost_included": row.get("cost_included"),
        "n_ph_work": row.get("n_ph_work"),
        "n_ph_ref": row.get("n_ph_ref"),
        "paper_i_ladder_stage": row.get("paper_i_ladder_stage"),
        "missing_reason": row.get("missing_reason"),
        "source_payload_missing_reason": row.get("source_payload_missing_reason"),
        "source_payload_path": row.get("source_payload_path") or row.get("payload_path"),
        "source_row_index": row.get("source_row_index") or row.get("row_index"),
        "source_record_id": row.get("source_record_id") or row.get("record_id"),
    }


def _eligibility_reason(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    status = str(row.get("status") or "").strip().lower()
    threshold_status = str(row.get("threshold_status") or "").strip().lower()

    if threshold_status in REACHED_THRESHOLD_STATUSES or _truthy(row.get("snake_first_crossing_reached")):
        if row.get("cost_included") is False or str(row.get("cost_included") or "").lower() == "false":
            return None, "cost_excluded_only_reached_rejected"
        return None, "reached_rejected"
    if status in ELIGIBLE_TERMINAL_STATUSES or threshold_status in ELIGIBLE_TERMINAL_STATUSES:
        terminal = status if status in ELIGIBLE_TERMINAL_STATUSES else threshold_status
        return f"terminal_{terminal}", None
    if threshold_status in ELIGIBLE_RUNNING_THRESHOLD_STATUSES:
        return "running_current_best_not_reached", None
    if _is_stale_target(row):
        return "stale_target", None
    return None, "not_eligible_status"


def _candidate_from_row(
    row: Mapping[str, Any],
    *,
    target_profile: str,
    threshold: float,
    source_audit_json: str | None,
    source_audit_sha256: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    row_threshold = row.get("threshold")
    if row_threshold not in {None, ""} and not _float_matches(row_threshold, threshold) and not _is_stale_target(row):
        return None, _rejection(row, reason="stale_threshold_rejected", detail=f"row_threshold={row_threshold}")

    if _is_phonon_row(row):
        return None, _rejection(row, reason="phonon_row_rejected")
    family = _family(row)
    if family != FAMILY:
        return None, _rejection(row, reason="non_hubbard_rejected")
    lane = _lane(row)
    method = _method(row)
    if lane != "snake" or method != METHOD:
        return None, _rejection(row, reason="comparator_or_non_snake_rejected")
    regime = _regime(row)
    expected_key = (family, method, regime)
    if expected_key not in set(EXPECTED_KEYS):
        return None, _rejection(row, reason="unexpected_hubbard_snake_key_rejected")
    case_id = _case_id(row)
    expected_case_id = EXPECTED_CASE_ID_BY_REGIME.get(regime, "")
    if case_id != expected_case_id:
        return None, _rejection(row, reason="unexpected_hubbard_snake_case_id", detail=f"expected={expected_case_id}")
    if _is_missing_source(row):
        return None, _rejection(row, reason="missing_no_source_rejected")

    eligibility_reason, rejection_reason = _eligibility_reason(row)
    if rejection_reason is not None:
        return None, _rejection(row, reason=rejection_reason)
    assert eligibility_reason is not None

    algorithm_id = _source_algorithm_id(row)
    if algorithm_id != ALGORITHM_ID:
        return None, _rejection(row, reason="snake_algorithm_mismatch", detail=algorithm_id)

    candidate = {
        "candidate_key": _candidate_key(regime=regime, case_id=case_id),
        "family": FAMILY,
        "method": METHOD,
        "regime": regime,
        "lane": "snake",
        "algorithm_id": ALGORITHM_ID,
        "case_id": case_id,
        "benchmark_id": case_id,
        "target_case_id": case_id,
        "target_suite_profile": TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        "target_profile": target_profile,
        "target_threshold": float(threshold),
        "static_route_id": ROUTE_ID_A,
        "fixed_inner_optimizer": "SPSA",
        "discovery_objective_mode": "discovery_first_crossing",
        "recalibration_reason": eligibility_reason,
        "source_expected_key": _raw_key(row),
        "source_status": str(row.get("status") or ""),
        "source_threshold_status": str(row.get("threshold_status") or ""),
        "source_display_delta_e": row.get("display_delta_e"),
        "source_cost_included": bool(row.get("cost_included")),
        "source_record_id": row.get("source_record_id") or row.get("record_id"),
        "source_row_index": row.get("source_row_index") or row.get("row_index"),
        "source_payload_path": _source_path(row),
        "source_payload_path_kind": row.get("source_payload_path_kind"),
        "source_payload_sha256": row.get("payload_sha256") or row.get("source_payload_sha256"),
        "source_audit_json": source_audit_json,
        "source_audit_sha256": source_audit_sha256,
        "source_target_profile": target_profile,
        "source_threshold": float(threshold),
        "snake_current_state": row.get("snake_current_state"),
        "snake_terminal_state": row.get("snake_terminal_state"),
        "snake_running_state": row.get("snake_running_state"),
        "snake_not_reached_state": row.get("snake_not_reached_state"),
        "snake_first_crossing_status": row.get("snake_first_crossing_status"),
        "snake_first_crossing_reached": bool(row.get("snake_first_crossing_reached")),
        "snake_best_trial_number": row.get("snake_best_trial_number"),
        "snake_complete_trial_count": row.get("snake_complete_trial_count"),
        "snake_running_trial_count": row.get("snake_running_trial_count"),
        "snake_trial_count": row.get("snake_trial_count"),
        "snake_source_condor_job": row.get("snake_source_condor_job"),
    }
    return candidate, None


def build_candidate_manifest(
    audit: Mapping[str, Any],
    *,
    source_audit_json: str | Path | None = None,
) -> dict[str, Any]:
    if str(audit.get("schema") or "") != AUDIT_SCHEMA:
        raise ValueError(f"audit schema must be {AUDIT_SCHEMA!r}, got {audit.get('schema')!r}")
    target_profile = str(audit.get("target_profile") or "").strip()
    if target_profile != "paper_i_phys_v1":
        raise ValueError(f"audit target_profile must be paper_i_phys_v1, got {target_profile!r}")
    threshold = audit.get("threshold")
    if not _float_matches(threshold, PAPER_I_CLEAN_TAU_PHYS):
        raise ValueError(f"audit threshold must be {PAPER_I_CLEAN_TAU_PHYS}, got {threshold!r}")

    source_path = None if source_audit_json is None else Path(source_audit_json)
    source_text = None if source_path is None else str(source_path)
    source_sha = None if source_path is None else _sha256_file(source_path)

    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_keys: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []

    for row in audit.get("expected_cell_audits") or ():
        if not isinstance(row, Mapping):
            continue
        candidate, rejection = _candidate_from_row(
            row,
            target_profile=target_profile,
            threshold=float(threshold),
            source_audit_json=source_text,
            source_audit_sha256=source_sha,
        )
        if rejection is not None:
            rejections.append(rejection)
            continue
        assert candidate is not None
        key = str(candidate["candidate_key"])
        if key in seen_keys:
            duplicates.append(key)
        else:
            seen_keys[key] = candidate
            candidates.append(candidate)

    missing_expected = [
        list(key)
        for key in EXPECTED_KEYS
        if _candidate_key(regime=key[2], case_id=EXPECTED_CASE_ID_BY_REGIME[key[2]]) not in seen_keys
    ]
    if duplicates:
        raise ValueError(f"duplicate Hubbard SNAKE recalibration candidate key(s): {sorted(duplicates)}")

    candidates.sort(key=lambda item: str(item["candidate_key"]))
    rejections.sort(
        key=lambda item: (
            str(item.get("reason") or ""),
            str(item.get("family") or ""),
            str(item.get("method") or ""),
            str(item.get("regime") or ""),
            str(item.get("case_id") or ""),
        )
    )
    by_regime = Counter(str(item["regime"]) for item in candidates)
    rejection_counts = Counter(str(item.get("reason") or "unknown") for item in rejections)
    return {
        "schema": CANDIDATE_SCHEMA,
        "generated_by": "chtc/phase3_optuna/paper_i_hubbard_snake_recalibration.py",
        "source_audit_json": source_text,
        "source_audit_sha256": source_sha,
        "source_audit_schema": AUDIT_SCHEMA,
        "target_profile": target_profile,
        "threshold": float(threshold),
        "target_suite_profile": TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        "static_route_id": ROUTE_ID_A,
        "fixed_inner_optimizer": "SPSA",
        "discovery_objective_mode": "discovery_first_crossing",
        "expected_keys": [list(key) for key in EXPECTED_KEYS],
        "expected_case_ids": dict(EXPECTED_CASE_ID_BY_REGIME),
        "candidate_count": len(candidates),
        "candidate_counts_by_regime": {key: by_regime.get(key, 0) for key in ("weak", "strong")},
        "missing_expected_keys": missing_expected,
        "rejected_candidate_count": len(rejections),
        "rejection_counts_by_reason": dict(sorted(rejection_counts.items())),
        "candidates": candidates,
        "rejections": rejections,
    }


def load_audit(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_candidate_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("schema") or "") != CANDIDATE_SCHEMA:
        raise ValueError(f"candidate manifest schema must be {CANDIDATE_SCHEMA!r}, got {payload.get('schema')!r}")
    if str(payload.get("target_profile") or "") != "paper_i_phys_v1":
        raise ValueError("candidate manifest target_profile must be paper_i_phys_v1")
    if str(payload.get("target_suite_profile") or "") != TABLE_I_CLEAN_NPH3_REF4_PROFILE:
        raise ValueError(f"candidate manifest target_suite_profile must be {TABLE_I_CLEAN_NPH3_REF4_PROFILE!r}")
    if not _float_matches(payload.get("threshold"), PAPER_I_CLEAN_TAU_PHYS):
        raise ValueError(f"candidate manifest threshold must be {PAPER_I_CLEAN_TAU_PHYS}")
    return payload


def candidates_for_recalibration(manifest: Mapping[str, Any] | str | Path) -> list[dict[str, Any]]:
    payload = load_candidate_manifest(manifest) if not isinstance(manifest, Mapping) else dict(manifest)
    if str(payload.get("schema") or "") != CANDIDATE_SCHEMA:
        raise ValueError(f"candidate manifest schema must be {CANDIDATE_SCHEMA!r}")
    selected = [dict(item) for item in payload.get("candidates") or () if isinstance(item, Mapping)]
    selected.sort(key=lambda item: str(item.get("candidate_key") or ""))
    return selected


def target_case_ids(candidates: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        case_id = str(candidate.get("target_case_id") or candidate.get("case_id") or "").strip()
        if case_id and case_id not in seen:
            seen.add(case_id)
            out.append(case_id)
    return tuple(out)


def source_metadata_fields(
    candidate: Mapping[str, Any],
    *,
    candidate_manifest_json: str | Path | None = None,
) -> dict[str, str]:
    manifest_path = None if candidate_manifest_json is None else Path(candidate_manifest_json)
    return {
        "paper_i_hubbard_snake_recalibration": "true",
        "paper_i_hubbard_snake_recalibration_candidate_manifest_json": "" if manifest_path is None else str(manifest_path),
        "paper_i_hubbard_snake_recalibration_candidate_key": str(candidate.get("candidate_key") or ""),
        "paper_i_hubbard_snake_recalibration_source_audit_json": str(candidate.get("source_audit_json") or ""),
        "paper_i_hubbard_snake_recalibration_source_audit_sha256": str(candidate.get("source_audit_sha256") or ""),
        "paper_i_hubbard_snake_recalibration_source_case_id": str(candidate.get("case_id") or ""),
        "paper_i_hubbard_snake_recalibration_source_status": str(candidate.get("source_status") or ""),
        "paper_i_hubbard_snake_recalibration_source_threshold_status": str(candidate.get("source_threshold_status") or ""),
        "paper_i_hubbard_snake_recalibration_source_target_profile": str(candidate.get("source_target_profile") or ""),
        "paper_i_hubbard_snake_recalibration_source_threshold": str(float(candidate.get("source_threshold") or PAPER_I_CLEAN_TAU_PHYS)),
        "paper_i_hubbard_snake_recalibration_source_payload_path": str(candidate.get("source_payload_path") or ""),
        "paper_i_hubbard_snake_recalibration_source_payload_sha256": str(candidate.get("source_payload_sha256") or ""),
        "paper_i_hubbard_snake_recalibration_source_record_id": str(candidate.get("source_record_id") or ""),
        "paper_i_hubbard_snake_recalibration_source_row_index": str(candidate.get("source_row_index") or ""),
        "paper_i_hubbard_snake_recalibration_source_payload_path_kind": str(candidate.get("source_payload_path_kind") or ""),
        "paper_i_hubbard_snake_recalibration_source_running_state": str(candidate.get("snake_running_state") or ""),
        "paper_i_hubbard_snake_recalibration_source_terminal_state": str(candidate.get("snake_terminal_state") or ""),
        "paper_i_hubbard_snake_recalibration_source_complete_trial_count": str(candidate.get("snake_complete_trial_count") or ""),
        "paper_i_hubbard_snake_recalibration_source_trial_count": str(candidate.get("snake_trial_count") or ""),
        "paper_i_hubbard_snake_recalibration_source_condor_job": str(candidate.get("snake_source_condor_job") or ""),
        "paper_i_hubbard_snake_recalibration_reason": str(candidate.get("recalibration_reason") or ""),
    }


def _resolve_path(path_text: str, *, repo_root: Path) -> Path:
    path = Path(str(path_text))
    return path if path.is_absolute() else repo_root / path


def validate_candidate_row_authorization(
    row: Mapping[str, Any],
    *,
    target_case_id: str,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Validate that a marked Route-A row is authorized by its candidate manifest."""

    blockers: list[str] = []
    marker = _truthy(row.get("paper_i_hubbard_snake_recalibration"))
    manifest_value = str(row.get("paper_i_hubbard_snake_recalibration_candidate_manifest_json") or "").strip()
    candidate_key = str(row.get("paper_i_hubbard_snake_recalibration_candidate_key") or "").strip()
    if not marker and not manifest_value and not candidate_key:
        return blockers
    if not marker:
        blockers.append("paper_i_hubbard_snake_recalibration_marker_missing")
    if not manifest_value:
        blockers.append("paper_i_hubbard_snake_recalibration_candidate_manifest_missing")
    if not candidate_key:
        blockers.append("paper_i_hubbard_snake_recalibration_candidate_key_missing")
    if not manifest_value or not candidate_key:
        return blockers

    manifest_path = _resolve_path(manifest_value, repo_root=repo_root)
    try:
        manifest = load_candidate_manifest(manifest_path)
    except Exception as exc:
        return [f"paper_i_hubbard_snake_recalibration_candidate_manifest_unreadable:{type(exc).__name__}:{exc}"]
    candidates = {
        str(candidate.get("candidate_key") or ""): candidate
        for candidate in manifest.get("candidates") or ()
        if isinstance(candidate, Mapping)
    }
    candidate = candidates.get(candidate_key)
    if not isinstance(candidate, Mapping):
        return [f"paper_i_hubbard_snake_recalibration_candidate_key_missing:{candidate_key}"]

    if str(candidate.get("family") or "") != FAMILY:
        blockers.append(f"paper_i_hubbard_snake_recalibration_candidate_family_mismatch:{candidate.get('family')}:{FAMILY}")
    if str(candidate.get("method") or "") != METHOD:
        blockers.append(f"paper_i_hubbard_snake_recalibration_candidate_method_mismatch:{candidate.get('method')}:{METHOD}")
    if str(candidate.get("lane") or "") != "snake":
        blockers.append(f"paper_i_hubbard_snake_recalibration_candidate_lane_mismatch:{candidate.get('lane')}:snake")
    if str(candidate.get("algorithm_id") or "") != ALGORITHM_ID:
        blockers.append(
            f"paper_i_hubbard_snake_recalibration_candidate_algorithm_id_mismatch:{candidate.get('algorithm_id')}:{ALGORITHM_ID}"
        )
    if str(candidate.get("target_case_id") or candidate.get("case_id") or "") != str(target_case_id):
        blockers.append(
            "paper_i_hubbard_snake_recalibration_candidate_target_case_mismatch:"
            f"{candidate.get('target_case_id') or candidate.get('case_id')}:{target_case_id}"
        )
    if str(candidate.get("target_suite_profile") or "") != str(row.get("suite_profile") or ""):
        blockers.append(
            "paper_i_hubbard_snake_recalibration_candidate_suite_profile_mismatch:"
            f"{candidate.get('target_suite_profile')}:{row.get('suite_profile')}"
        )

    comparisons = (
        ("case_id", "paper_i_hubbard_snake_recalibration_source_case_id"),
        ("source_status", "paper_i_hubbard_snake_recalibration_source_status"),
        ("source_threshold_status", "paper_i_hubbard_snake_recalibration_source_threshold_status"),
        ("source_target_profile", "paper_i_hubbard_snake_recalibration_source_target_profile"),
        ("source_payload_path", "paper_i_hubbard_snake_recalibration_source_payload_path"),
        ("source_payload_sha256", "paper_i_hubbard_snake_recalibration_source_payload_sha256"),
        ("source_record_id", "paper_i_hubbard_snake_recalibration_source_record_id"),
        ("source_payload_path_kind", "paper_i_hubbard_snake_recalibration_source_payload_path_kind"),
        ("snake_running_state", "paper_i_hubbard_snake_recalibration_source_running_state"),
        ("snake_terminal_state", "paper_i_hubbard_snake_recalibration_source_terminal_state"),
        ("snake_complete_trial_count", "paper_i_hubbard_snake_recalibration_source_complete_trial_count"),
        ("snake_trial_count", "paper_i_hubbard_snake_recalibration_source_trial_count"),
        ("snake_source_condor_job", "paper_i_hubbard_snake_recalibration_source_condor_job"),
        ("recalibration_reason", "paper_i_hubbard_snake_recalibration_reason"),
        ("source_audit_json", "paper_i_hubbard_snake_recalibration_source_audit_json"),
        ("source_audit_sha256", "paper_i_hubbard_snake_recalibration_source_audit_sha256"),
    )
    for candidate_field, row_field in comparisons:
        expected = candidate.get(candidate_field)
        actual = str(row.get(row_field) or "").strip()
        blocker_field = str(row_field).removeprefix("paper_i_hubbard_snake_recalibration_")
        if expected in {None, ""}:
            continue
        if not actual:
            blockers.append(f"paper_i_hubbard_snake_recalibration_{blocker_field}_missing")
        elif actual != str(expected):
            blockers.append(f"paper_i_hubbard_snake_recalibration_{blocker_field}_mismatch:{actual}:{expected}")
    source_threshold = row.get("paper_i_hubbard_snake_recalibration_source_threshold")
    if candidate.get("source_threshold") not in {None, ""} and source_threshold in {None, ""}:
        blockers.append("paper_i_hubbard_snake_recalibration_source_threshold_missing")
    elif source_threshold not in {None, ""} and not _float_matches(source_threshold, candidate.get("source_threshold")):
        blockers.append(
            "paper_i_hubbard_snake_recalibration_source_threshold_mismatch:"
            f"{source_threshold}:{candidate.get('source_threshold')}"
        )
    audit_value = str(row.get("paper_i_hubbard_snake_recalibration_source_audit_json") or "").strip()
    audit_sha = str(row.get("paper_i_hubbard_snake_recalibration_source_audit_sha256") or "").strip()
    if audit_value and audit_sha:
        actual_sha = _sha256_file(_resolve_path(audit_value, repo_root=repo_root))
        if actual_sha is not None and actual_sha != audit_sha:
            blockers.append(f"paper_i_hubbard_snake_recalibration_source_audit_sha256_mismatch:{actual_sha}:{audit_sha}")
    return blockers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--expect-candidate-count", type=int, default=None)
    args = parser.parse_args(argv)
    audit = load_audit(args.audit_json)
    manifest = build_candidate_manifest(audit, source_audit_json=args.audit_json)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.expect_candidate_count is not None and int(manifest.get("candidate_count") or 0) != int(args.expect_candidate_count):
        print(
            f"expected {args.expect_candidate_count} candidates, got {manifest.get('candidate_count')}; "
            "see rejections in manifest",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
