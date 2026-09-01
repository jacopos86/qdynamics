#!/usr/bin/env python3
"""Build Paper-I phonon cutoff-ladder escalation candidates from table audits.

The input is the repaired fixed-accuracy table audit emitted by
``pipelines/reporting/build_paper_i_fixed_accuracy_table_pdf.py``.  The output is
an intentionally deterministic handoff manifest: downstream generators may use
it to emit only approved nph3/ref4 escalation rows and to carry prior nph2/ref4
source metadata into preflight.
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

from chtc.phase3_optuna.paper_i_clean_ladder_contract import (  # noqa: E402
    PAPER_I_CLEAN_TAU_PHYS,
    PAPER_I_LADDER_STAGE_CONFIGS,
    PAPER_I_PHONON_FAMILIES,
    normalize_paper_i_ladder_stage,
)

AUDIT_SCHEMA = "paper_i_fixed_accuracy_table_audit_v1"
CANDIDATE_SCHEMA = "paper_i_phonon_ladder_escalation_candidates_v1"
SOURCE_STAGE = "nph2_ref4_screen"
TARGET_STAGE = "nph3_ref4_escalation"
VALID_SOURCE_STATUSES = {"not_reached", "failed"}

_METHOD_TO_ALGORITHM_ID = {
    "HEA VQE": "static_hea_qiskit_vqe",
    "family-informed VQE": "static_family_informed_vqe",
    "append-only ADAPT": "static_full_meta_append_adapt_vqe",
    "TETRIS-ADAPT": "static_tetris_qubit_adapt_vqe",
    "TETRIS-ADAPT-VQE": "static_tetris_qubit_adapt_vqe",
    "Pos-Geo-ADAPT": "static_pos_geo_adapt_vqe",
    "Pos-Geo-ADAPT-VQE": "static_pos_geo_adapt_vqe",
    "Qubit/QEB-ADAPT-VQE": "static_qubit_qeb_adapt_vqe",
    "SNAKE": "static_family_native_adapt_phase3",
}


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


def _int_value(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        parsed = int(float(str(value)))
    except Exception:
        return None
    return parsed


def _source_algorithm_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("algorithm_id") or row.get("source_algorithm_id") or "").strip()
    if explicit:
        return explicit
    record_id = str(row.get("source_record_id") or row.get("record_id") or "").strip()
    if "__" in record_id:
        tail = record_id.rsplit("__", 1)[-1].strip()
        if tail:
            return tail
    method = str(row.get("method") or row.get("method_normalized") or "").strip()
    return _METHOD_TO_ALGORITHM_ID.get(method, "")


def _candidate_key(*, lane: str, family: str, method: str, regime: str, next_case_id: str, algorithm_id: str) -> str:
    return "|".join((lane, family, method, regime, next_case_id, algorithm_id))


def _raw_key(row: Mapping[str, Any]) -> list[Any] | None:
    value = row.get("expected_key") or row.get("normalized_key")
    return list(value) if isinstance(value, (list, tuple)) else None


def _rejection(row: Mapping[str, Any], *, reason: str, detail: str | None = None) -> dict[str, Any]:
    return {
        "reason": reason,
        "detail": detail,
        "expected_key": _raw_key(row),
        "family": row.get("canonical_family") or row.get("family_normalized") or row.get("family"),
        "method": row.get("method") or row.get("method_normalized"),
        "regime": row.get("regime"),
        "lane": row.get("lane"),
        "case_id": row.get("case_id"),
        "status": row.get("status"),
        "threshold_status": row.get("threshold_status"),
        "n_ph_work": row.get("n_ph_work"),
        "n_ph_ref": row.get("n_ph_ref"),
        "paper_i_ladder_stage": row.get("paper_i_ladder_stage"),
        "next_stage": row.get("next_stage"),
        "next_stage_case_id": row.get("next_stage_case_id"),
        "eligible_for_escalation": bool(row.get("eligible_for_escalation")),
        "source_row_index": row.get("source_row_index") or row.get("row_index"),
        "source_record_id": row.get("source_record_id") or row.get("record_id"),
    }


def _candidate_from_row(
    row: Mapping[str, Any],
    *,
    target_profile: str,
    threshold: float,
    source_audit_json: str | None,
    source_audit_sha256: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    family = str(row.get("canonical_family") or row.get("family_normalized") or row.get("family") or "").strip()
    method = str(row.get("method") or row.get("method_normalized") or "").strip()
    regime = str(row.get("regime") or "").strip()
    lane = str(row.get("lane") or "").strip()
    status = str(row.get("status") or "").strip().lower()
    source_stage = str(row.get("paper_i_ladder_stage") or "").strip()
    source_work = _int_value(row.get("n_ph_work"))
    source_ref = _int_value(row.get("n_ph_ref"))
    next_stage = str(row.get("next_stage") or "").strip()
    next_case_id = str(row.get("next_stage_case_id") or "").strip()
    source_case_id = str(row.get("case_id") or "").strip()
    row_threshold = row.get("threshold")

    if row_threshold not in {None, ""} and not _float_matches(row_threshold, threshold):
        return None, _rejection(row, reason="stale_threshold_rejected", detail=f"row_threshold={row_threshold}")
    if lane not in {"comparator", "snake"}:
        return None, _rejection(row, reason="unsupported_lane")
    if family not in set(PAPER_I_PHONON_FAMILIES):
        return None, _rejection(row, reason="non_phonon_family")
    if status == "running":
        return None, _rejection(row, reason="running_not_terminal")
    if status == "missing":
        return None, _rejection(row, reason="missing_source")
    if status == "invalid_target":
        return None, _rejection(row, reason="invalid_target")
    if source_stage == "nph3_ref4_escalation":
        return None, _rejection(row, reason="nph3_to_nph4_ref5_requires_explicit_approval")
    if source_stage == "nph4_ref5_optional":
        return None, _rejection(row, reason="nph4_ref5_source_not_escalated")
    if source_stage != SOURCE_STAGE or source_work != 2 or source_ref != 4:
        return None, _rejection(row, reason="not_nph2_ref4_source")
    if status not in VALID_SOURCE_STATUSES:
        return None, _rejection(row, reason="terminal_status_not_failed_or_not_reached")
    if row.get("eligible_for_escalation") is not True:
        return None, _rejection(row, reason="audit_not_eligible_for_escalation")
    if next_stage != TARGET_STAGE:
        return None, _rejection(row, reason="next_stage_not_nph3_ref4")
    if not next_case_id:
        return None, _rejection(row, reason="missing_next_stage_case_id")
    if not source_case_id:
        return None, _rejection(row, reason="missing_source_case_id")

    algorithm_id = _source_algorithm_id(row)
    if not algorithm_id:
        return None, _rejection(row, reason="missing_algorithm_id")
    if lane == "snake" and algorithm_id != "static_family_native_adapt_phase3":
        return None, _rejection(row, reason="snake_lane_algorithm_mismatch", detail=algorithm_id)
    if lane == "comparator" and algorithm_id == "static_family_native_adapt_phase3":
        return None, _rejection(row, reason="comparator_lane_contains_snake_algorithm")

    _target_stage, target_config = normalize_paper_i_ladder_stage(TARGET_STAGE)
    escalation_reason = str(row.get("escalation_reason") or row.get("eligibility_reason") or "").strip()
    if not escalation_reason:
        escalation_reason = f"completed_{status}_phonon_ladder_row"
    candidate_key = _candidate_key(
        lane=lane,
        family=family,
        method=method,
        regime=regime,
        next_case_id=next_case_id,
        algorithm_id=algorithm_id,
    )
    candidate = {
        "candidate_key": candidate_key,
        "lane": lane,
        "family": family,
        "method": method,
        "regime": regime,
        "algorithm_id": algorithm_id,
        "source_case_id": source_case_id,
        "source_stage": source_stage,
        "source_n_ph_work": int(source_work),
        "source_n_ph_ref": int(source_ref),
        "source_status": status,
        "source_threshold_status": row.get("threshold_status"),
        "source_record_id": row.get("source_record_id") or row.get("record_id"),
        "source_row_index": row.get("source_row_index") or row.get("row_index"),
        "source_payload_path": row.get("source_payload_path") or row.get("payload_path"),
        "source_payload_sha256": row.get("payload_sha256"),
        "source_audit_json": source_audit_json,
        "source_audit_sha256": source_audit_sha256,
        "source_target_profile": target_profile,
        "source_threshold": float(threshold),
        "next_stage": TARGET_STAGE,
        "next_stage_case_id": next_case_id,
        "target_n_ph_work": int(target_config.n_ph_work or 0),
        "target_n_ph_ref": int(target_config.n_ph_ref or 0),
        "target_suite_profile": target_config.suite_profile,
        "escalation_reason": escalation_reason,
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

    for raw in audit.get("raw_row_audits") or ():
        if not isinstance(raw, Mapping):
            continue
        if raw.get("threshold_matches_requested") is False or str(raw.get("skip_reason") or "") == "threshold_mismatch":
            rejections.append(_rejection(raw, reason="stale_threshold_rejected"))

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
    if duplicates:
        raise ValueError(f"duplicate phonon escalation candidate key(s): {sorted(duplicates)}")

    candidates.sort(key=lambda item: str(item["candidate_key"]))
    rejections.sort(
        key=lambda item: (
            str(item.get("reason") or ""),
            str(item.get("lane") or ""),
            str(item.get("family") or ""),
            str(item.get("method") or ""),
            str(item.get("regime") or ""),
            str(item.get("case_id") or ""),
        )
    )
    by_lane = Counter(str(item["lane"]) for item in candidates)
    rejection_counts = Counter(str(item.get("reason") or "unknown") for item in rejections)
    return {
        "schema": CANDIDATE_SCHEMA,
        "generated_by": "chtc/phase3_optuna/paper_i_table_i_audit_escalation.py",
        "source_audit_json": source_text,
        "source_audit_sha256": source_sha,
        "source_audit_schema": AUDIT_SCHEMA,
        "target_profile": target_profile,
        "threshold": float(threshold),
        "source_stage": SOURCE_STAGE,
        "next_stage": TARGET_STAGE,
        "source_n_ph_work": 2,
        "source_n_ph_ref": 4,
        "target_n_ph_work": int(PAPER_I_LADDER_STAGE_CONFIGS[TARGET_STAGE].n_ph_work or 0),
        "target_n_ph_ref": int(PAPER_I_LADDER_STAGE_CONFIGS[TARGET_STAGE].n_ph_ref or 0),
        "candidate_count": len(candidates),
        "candidate_counts_by_lane": {key: by_lane.get(key, 0) for key in ("comparator", "snake")},
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
    if str(payload.get("next_stage") or "") != TARGET_STAGE:
        raise ValueError(f"candidate manifest next_stage must be {TARGET_STAGE!r}")
    if not _float_matches(payload.get("threshold"), PAPER_I_CLEAN_TAU_PHYS):
        raise ValueError(f"candidate manifest threshold must be {PAPER_I_CLEAN_TAU_PHYS}")
    return payload


def candidates_for_lane(
    manifest: Mapping[str, Any] | str | Path,
    *,
    lane: str,
    target_stage: str = TARGET_STAGE,
) -> list[dict[str, Any]]:
    payload = load_candidate_manifest(manifest) if not isinstance(manifest, Mapping) else dict(manifest)
    if str(payload.get("schema") or "") != CANDIDATE_SCHEMA:
        raise ValueError(f"candidate manifest schema must be {CANDIDATE_SCHEMA!r}")
    selected = [
        dict(item)
        for item in payload.get("candidates") or ()
        if isinstance(item, Mapping)
        and str(item.get("lane") or "") == lane
        and str(item.get("next_stage") or "") == target_stage
    ]
    if not selected:
        raise ValueError(f"candidate manifest contains no {lane} candidates for {target_stage}")
    selected.sort(key=lambda item: str(item.get("candidate_key") or ""))
    return selected


def generic_target_record_id(candidate: Mapping[str, Any]) -> str:
    return "static_table__{family}__{case_id}__{algorithm_id}".format(
        family=str(candidate.get("family") or ""),
        case_id=str(candidate.get("next_stage_case_id") or ""),
        algorithm_id=str(candidate.get("algorithm_id") or ""),
    )


def target_case_ids(candidates: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        case_id = str(candidate.get("next_stage_case_id") or "").strip()
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
    fields = {
        "paper_i_ladder_candidate_manifest_json": "" if manifest_path is None else str(manifest_path),
        "paper_i_ladder_candidate_key": str(candidate.get("candidate_key") or ""),
        "paper_i_ladder_source_audit_json": str(candidate.get("source_audit_json") or ""),
        "paper_i_ladder_source_audit_sha256": str(candidate.get("source_audit_sha256") or ""),
        "paper_i_ladder_source_case_id": str(candidate.get("source_case_id") or ""),
        "paper_i_ladder_source_stage": str(candidate.get("source_stage") or ""),
        "paper_i_ladder_source_status": str(candidate.get("source_status") or ""),
        "paper_i_ladder_source_threshold_status": str(candidate.get("source_threshold_status") or ""),
        "paper_i_ladder_source_n_ph_work": str(candidate.get("source_n_ph_work") or ""),
        "paper_i_ladder_source_n_ph_ref": str(candidate.get("source_n_ph_ref") or ""),
        "paper_i_ladder_source_threshold": str(float(candidate.get("source_threshold") or PAPER_I_CLEAN_TAU_PHYS)),
        "paper_i_ladder_source_target_profile": str(candidate.get("source_target_profile") or ""),
        "paper_i_ladder_source_payload_path": str(candidate.get("source_payload_path") or ""),
        "paper_i_ladder_source_payload_sha256": str(candidate.get("source_payload_sha256") or ""),
        "paper_i_ladder_source_record_id": str(candidate.get("source_record_id") or ""),
    }
    return fields


def _resolve_path(path_text: str, *, repo_root: Path) -> Path:
    path = Path(str(path_text))
    return path if path.is_absolute() else repo_root / path


def validate_candidate_row_authorization(
    row: Mapping[str, Any],
    *,
    target_case_id: str,
    lane: str,
    algorithm_id: str,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Validate that an escalation row is authorized by its candidate manifest."""

    blockers: list[str] = []
    manifest_value = str(row.get("paper_i_ladder_candidate_manifest_json") or "").strip()
    candidate_key = str(row.get("paper_i_ladder_candidate_key") or "").strip()
    if not manifest_value or not candidate_key:
        return blockers
    manifest_path = _resolve_path(manifest_value, repo_root=repo_root)
    try:
        manifest = load_candidate_manifest(manifest_path)
    except Exception as exc:
        return [f"paper_i_ladder_candidate_manifest_unreadable:{type(exc).__name__}:{exc}"]
    candidates = {
        str(candidate.get("candidate_key") or ""): candidate
        for candidate in manifest.get("candidates") or ()
        if isinstance(candidate, Mapping)
    }
    candidate = candidates.get(candidate_key)
    if not isinstance(candidate, Mapping):
        return [f"paper_i_ladder_candidate_key_missing:{candidate_key}"]

    if str(candidate.get("lane") or "") != lane:
        blockers.append(f"paper_i_ladder_candidate_lane_mismatch:{candidate.get('lane')}:{lane}")
    if lane == "snake":
        policy = str(row.get("paper_i_ladder_snake_policy") or "").strip()
        if policy != "snake_only":
            blockers.append(f"paper_i_ladder_candidate_lane_mismatch:{policy}:snake_only")
    if str(candidate.get("algorithm_id") or "") != str(algorithm_id):
        blockers.append(f"paper_i_ladder_candidate_algorithm_id_mismatch:{candidate.get('algorithm_id')}:{algorithm_id}")
    if str(candidate.get("next_stage_case_id") or "") != str(target_case_id):
        blockers.append(
            f"paper_i_ladder_candidate_target_case_mismatch:{candidate.get('next_stage_case_id')}:{target_case_id}"
        )
    if str(candidate.get("next_stage") or "") != str(row.get("paper_i_cutoff_ladder_stage") or ""):
        blockers.append(
            f"paper_i_ladder_candidate_stage_mismatch:{candidate.get('next_stage')}:{row.get('paper_i_cutoff_ladder_stage')}"
        )

    comparisons = (
        ("source_case_id", "paper_i_ladder_source_case_id"),
        ("source_stage", "paper_i_ladder_source_stage"),
        ("source_status", "paper_i_ladder_source_status"),
        ("source_threshold_status", "paper_i_ladder_source_threshold_status"),
        ("source_n_ph_work", "paper_i_ladder_source_n_ph_work"),
        ("source_n_ph_ref", "paper_i_ladder_source_n_ph_ref"),
        ("source_target_profile", "paper_i_ladder_source_target_profile"),
        ("source_payload_path", "paper_i_ladder_source_payload_path"),
        ("source_payload_sha256", "paper_i_ladder_source_payload_sha256"),
        ("source_record_id", "paper_i_ladder_source_record_id"),
        ("source_audit_json", "paper_i_ladder_source_audit_json"),
        ("source_audit_sha256", "paper_i_ladder_source_audit_sha256"),
    )
    for candidate_field, row_field in comparisons:
        expected = candidate.get(candidate_field)
        actual = str(row.get(row_field) or "").strip()
        if expected not in {None, ""} and actual and actual != str(expected):
            blockers.append(f"paper_i_ladder_candidate_{row_field}_mismatch:{actual}:{expected}")
    source_threshold = row.get("paper_i_ladder_source_threshold")
    if source_threshold not in {None, ""} and not _float_matches(source_threshold, candidate.get("source_threshold")):
        blockers.append(
            f"paper_i_ladder_candidate_source_threshold_mismatch:{source_threshold}:{candidate.get('source_threshold')}"
        )
    audit_value = str(row.get("paper_i_ladder_source_audit_json") or "").strip()
    audit_sha = str(row.get("paper_i_ladder_source_audit_sha256") or "").strip()
    if audit_value and audit_sha:
        actual_sha = _sha256_file(_resolve_path(audit_value, repo_root=repo_root))
        if actual_sha is not None and actual_sha != audit_sha:
            blockers.append(f"paper_i_ladder_source_audit_sha256_mismatch:{actual_sha}:{audit_sha}")
    return blockers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    audit = load_audit(args.audit_json)
    manifest = build_candidate_manifest(audit, source_audit_json=args.audit_json)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
