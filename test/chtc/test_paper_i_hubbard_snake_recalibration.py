from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna import paper_i_hubbard_snake_recalibration as recalibration


def _row(
    *,
    family: str = "hubbard",
    method: str = "SNAKE",
    lane: str = "snake",
    regime: str = "weak",
    status: str = "running",
    threshold_status: str = "running_current_best_not_reached",
    case_id: str | None = None,
    source_path: str | None = "raw_outputs/live_snake_best/summary.json",
    cost_included: bool = False,
    n_ph_work: int | None = None,
    n_ph_ref: int | None = None,
    paper_i_ladder_stage: str = "not_applicable_nonphonon",
    missing_reason: str | None = None,
) -> dict:
    resolved_case_id = case_id or f"{family}_L2_clean_{regime}"
    if family == "hubbard":
        resolved_case_id = case_id or recalibration.EXPECTED_CASE_ID_BY_REGIME[regime]
    return {
        "canonical_family": family,
        "method": method,
        "lane": lane,
        "regime": regime,
        "case_id": resolved_case_id,
        "expected_key": [family, method, regime],
        "status": status,
        "threshold_status": threshold_status,
        "cost_included": cost_included,
        "display_delta_e": "2.36e-01",
        "missing_reason": missing_reason,
        "n_ph_work": n_ph_work,
        "n_ph_ref": n_ph_ref,
        "paper_i_ladder_stage": paper_i_ladder_stage,
        "source_payload_path": source_path,
        "payload_sha256": "abc123",
        "source_payload_path_kind": "live_snake_overlay_summary",
        "source_record_id": f"live_snake_current_best__{resolved_case_id}",
        "source_row_index": 17,
        "snake_first_crossing_reached": threshold_status in {"running_current_best_reached", "reached"},
        "snake_first_crossing_status": "not_reached" if "not_reached" in threshold_status else "reached",
        "snake_running_state": "running_with_completed_trials",
        "snake_terminal_state": "not_terminal_running" if status == "running" else "terminal",
        "snake_complete_trial_count": 1,
        "snake_running_trial_count": 1,
        "snake_trial_count": 2,
        "snake_source_condor_job": "6635207.0",
    }


def _audit(*rows: dict, threshold: float = 2e-4) -> dict:
    return {
        "schema": recalibration.AUDIT_SCHEMA,
        "target_profile": "paper_i_phys_v1",
        "threshold": threshold,
        "expected_cell_audits": list(rows),
    }


def test_hubbard_snake_recalibration_manifest_selects_exact_expected_keys(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    audit = _audit(
        _row(regime="weak", threshold_status="running_current_best_not_reached"),
        _row(regime="strong", status="failed", threshold_status="failed"),
        _row(family="ionic_hubbard", regime="weak", status="failed", threshold_status="failed"),
        _row(method="HEA VQE", lane="comparator", regime="weak", status="not_reached", threshold_status="not_reached"),
        _row(family="hh", regime="weak", n_ph_work=2, n_ph_ref=4, paper_i_ladder_stage="nph2_ref4_screen"),
        _row(regime="weak", status="missing", threshold_status="missing", source_path=None, missing_reason="no_source_row_for_expected_key"),
        _row(regime="strong", threshold_status="running_current_best_reached", cost_included=False),
    )
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    payload = recalibration.build_candidate_manifest(audit, source_audit_json=audit_path)

    assert payload["schema"] == recalibration.CANDIDATE_SCHEMA
    assert payload["candidate_count"] == 2
    assert payload["candidate_counts_by_regime"] == {"weak": 1, "strong": 1}
    assert [candidate["case_id"] for candidate in payload["candidates"]] == [
        "hubbard_L2_clean_strong",
        "hubbard_L2_clean_weak",
    ]
    assert {candidate["recalibration_reason"] for candidate in payload["candidates"]} == {
        "running_current_best_not_reached",
        "terminal_failed",
    }
    reasons = payload["rejection_counts_by_reason"]
    assert reasons["non_hubbard_rejected"] >= 1
    assert reasons["comparator_or_non_snake_rejected"] >= 1
    assert reasons["phonon_row_rejected"] >= 1
    assert reasons["missing_no_source_rejected"] >= 1
    assert reasons["cost_excluded_only_reached_rejected"] >= 1


def test_hubbard_snake_recalibration_accepts_stale_target_with_source() -> None:
    stale = _row(regime="weak", status="invalid_target", threshold_status="threshold_mismatch")
    stale["threshold"] = 0.001
    payload = recalibration.build_candidate_manifest(
        _audit(
            stale,
            _row(regime="strong", status="not_reached", threshold_status="not_reached"),
        )
    )

    assert payload["candidate_count"] == 2
    assert {candidate["recalibration_reason"] for candidate in payload["candidates"]} == {
        "stale_target",
        "terminal_not_reached",
    }


def test_hubbard_snake_recalibration_rejects_bad_audit_threshold() -> None:
    with pytest.raises(ValueError, match="audit threshold"):
        recalibration.build_candidate_manifest(_audit(_row(), threshold=0.001))


def test_hubbard_snake_recalibration_authorization_detects_tampered_row(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    audit = _audit(
        _row(regime="weak"),
        _row(regime="strong"),
    )
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest = recalibration.build_candidate_manifest(audit, source_audit_json=audit_path)
    manifest_path = tmp_path / "candidates.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = manifest["candidates"][0]
    row = recalibration.source_metadata_fields(candidate, candidate_manifest_json=manifest_path)
    row.update(
        {
            "suite_profile": recalibration.TABLE_I_CLEAN_NPH3_REF4_PROFILE,
            "benchmark_ids": candidate["case_id"],
        }
    )

    assert recalibration.validate_candidate_row_authorization(
        row,
        target_case_id=str(candidate["case_id"]),
        repo_root=REPO_ROOT,
    ) == []

    row["paper_i_hubbard_snake_recalibration_source_status"] = "displayed"
    blockers = recalibration.validate_candidate_row_authorization(
        row,
        target_case_id=str(candidate["case_id"]),
        repo_root=REPO_ROOT,
    )
    assert any("source_status" in blocker for blocker in blockers)

    row = recalibration.source_metadata_fields(candidate, candidate_manifest_json=manifest_path)
    row.update(
        {
            "suite_profile": recalibration.TABLE_I_CLEAN_NPH3_REF4_PROFILE,
            "benchmark_ids": candidate["case_id"],
            "paper_i_hubbard_snake_recalibration_source_payload_path": "",
        }
    )
    blockers = recalibration.validate_candidate_row_authorization(
        row,
        target_case_id=str(candidate["case_id"]),
        repo_root=REPO_ROOT,
    )
    assert "paper_i_hubbard_snake_recalibration_source_payload_path_missing" in blockers
