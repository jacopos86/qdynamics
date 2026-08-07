from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna import check_paper_i_comparator_spsa_calibration_outputs as checker
from chtc.phase3_optuna import generate_paper_i_comparator_spsa_calibration_records as generator

SMOKE_CONFIG = Path("chtc/phase3_optuna/config/paper_i_comparator_spsa_calibration_v1_smoke.json")
REPAIR_SMOKE_CONFIG = Path("chtc/phase3_optuna/config/paper_i_hh_geo_qeb_spsa_repair_v1_smoke.json")


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _row_for_method(method_id: str = "static_hea_qiskit_vqe") -> dict[str, str]:
    rows, _smoke, _config = generator.build_rows(config_path=SMOKE_CONFIG, generation_mode="smoke")
    return {row["method_id"]: row for row in rows}[method_id]


def _repair_row_for(method_id: str = "static_geo_adapt_vqe", target_id: str = "hh_sym_weak_weak") -> dict[str, str]:
    rows, _smoke, _config = generator.build_rows(
        config_path=REPAIR_SMOKE_CONFIG,
        generation_mode="smoke",
        repair_scope=generator.HH_GEO_QEB_TABLEIII_REPAIR_SCOPE,
    )
    return {(row["method_id"], row["target_id"]): row for row in rows}[(method_id, target_id)]


def _schedule_for(row: dict[str, str], *, bad_field: bool = False) -> dict[str, float]:
    if bad_field:
        return {"not_allowed_spsa_field": 0.1}
    fields = json.loads(row["search_space_fields_json"])
    return {str(field): 0.01 if "perturbation" not in str(field) and not str(field).endswith("_c") else 0.005 for field in fields}


def _write_good_output(
    output_root: Path,
    row: dict[str, str],
    *,
    bad_schedule: bool = False,
    omit_current_best: bool = False,
    quality_nonpassing: bool = False,
) -> Path:
    record_id = row["record_id"]
    method_id = row["method_id"]
    target_id = row["target_id"]
    case_ids = json.loads(row["case_ids_json"])
    schedule = _schedule_for(row, bad_field=bad_schedule)
    scope_metadata = {
        field: row[field]
        for field in ("rerun_scope", "adaptive_refit_engine", "warm_start_schedule_lock_json", "warm_start_schedule_key")
        if row.get(field)
    }
    if row.get("warm_start_schedule_lock_json"):
        scope_metadata["warm_start_enqueued_trial_count"] = 1
        scope_metadata["warm_start_schedule"] = dict(schedule)
    case_status = "completed_quality_nonpassing" if quality_nonpassing else "completed"
    strict_usable = not quality_nonpassing
    out_dir = output_root / row["record_output_dir"]
    progress = out_dir / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    case_results = [
        {
            "schema": "paper_i_comparator_spsa_case_summary_v1",
            "record_id": record_id,
            "method_id": method_id,
            "target_id": target_id,
            "case_id": case_id,
            "trial_number": 0,
            **scope_metadata,
            "status": case_status,
            "usable": strict_usable,
            "strict_status_usable": strict_usable,
            "calibration_usable": True,
            "status_usable_policy": row.get("calibration_usable_status_policy") or "strict_status_completed_v1",
            "quality_nonpassing_usable": quality_nonpassing,
            "abs_delta_e": 1.0e-4,
            "case_loss": -0.30103,
            "resource_metric": 10.0,
            "resource_metric_field": "compiled_count_2q_total",
            "output_dir": str(out_dir / "trial_0000" / "cases" / case_id),
        }
        for case_id in case_ids
    ]
    event = {
        "schema": "paper_i_comparator_spsa_trial_event_v1",
        "study_name": f"study::{record_id}",
        "record_id": record_id,
        "method_id": method_id,
        "target_id": target_id,
        "case_ids": case_ids,
        **scope_metadata,
        "trial_number": 0,
        "trial_state": "COMPLETE",
        "objective": -0.25,
        "mean_case_loss": -0.30,
        "mean_log1p_resource_metric": 2.0,
        "resource_tiebreak_weight": 0.001,
        "all_cases_usable": strict_usable,
        "all_cases_calibration_usable": True,
        "status_usable_policy": row.get("calibration_usable_status_policy") or "strict_status_completed_v1",
        "schedule": schedule,
        "case_results": case_results,
        "case_abs_delta_e": {case_id: 1.0e-4 for case_id in case_ids},
        "case_output_dirs": {case_id: str(out_dir / "trial_0000" / "cases" / case_id) for case_id in case_ids},
        "log_paths": {"stdout": None, "stderr": None},
        "warm_start_trial": bool(row.get("warm_start_schedule_lock_json")),
        "elapsed_s": 1.0,
        "updated_utc": "2026-05-31T00:00:00Z",
    }
    (progress / "trial_events.jsonl").write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    current_best = {
        "schema": "paper_i_comparator_spsa_current_best_v1",
        "study_name": event["study_name"],
        "record_id": record_id,
        "method_id": method_id,
        "target_id": target_id,
        "case_ids": case_ids,
        **scope_metadata,
        "updated_utc": "2026-05-31T00:00:00Z",
        "source_schema_version": "paper_i_comparator_spsa_calibration_runner_v1",
        "best": event,
        "completed_trial_count": 1,
        "usable_trial_count": 0 if quality_nonpassing else 1,
        "strict_usable_trial_count": 0 if quality_nonpassing else 1,
        "calibration_usable_trial_count": 1,
        "progress_events_jsonl": str(progress / "trial_events.jsonl"),
    }
    if not omit_current_best:
        (progress / "current_best.json").write_text(json.dumps(current_best, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    best_schedule = {
        "schema": "paper_i_comparator_spsa_best_schedule_v1",
        "study_name": event["study_name"],
        "record_id": record_id,
        "method_id": method_id,
        "target_id": target_id,
        "case_ids": case_ids,
        **scope_metadata,
        "updated_utc": "2026-05-31T00:00:00Z",
        "usable": strict_usable,
        "strict_status_usable": strict_usable,
        "calibration_usable": True,
        "status_usable_policy": row.get("calibration_usable_status_policy") or "strict_status_completed_v1",
        "trial_number": 0,
        "objective": -0.25,
        "schedule": schedule,
        "case_abs_delta_e": {case_id: 1.0e-4 for case_id in case_ids},
        "source_trial_summary_json": str(out_dir / "trial_0000" / "trial_summary.json"),
    }
    (out_dir / "best_schedule.json").write_text(json.dumps(best_schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema": "paper_i_comparator_spsa_calibration_summary_v1",
        "ok": True,
        "status": "completed",
        "evidence_role": "calibration_only_not_manuscript_table_evidence",
        "study_name": event["study_name"],
        "record_id": record_id,
        "profile_id": row["profile_id"],
        "method_id": method_id,
        "target_id": target_id,
        **scope_metadata,
        "family": row["family"],
        "case_ids": case_ids,
        "config_path": row["config_path"],
        "config_sha256": row["config_sha256"],
        "n_trials_requested": 1,
        "completed_trial_count": 1,
        "usable_trial_count": 0 if quality_nonpassing else 1,
        "strict_usable_trial_count": 0 if quality_nonpassing else 1,
        "calibration_usable_trial_count": 1,
        "failed_or_incomplete_trial_count": 0,
        "strict_failed_or_incomplete_trial_count": 1 if quality_nonpassing else 0,
        "best_trial_number": 0,
        "best_objective": -0.25,
        "best_schedule": schedule,
        "summary_json": str(out_dir / "summary.json"),
        "best_schedule_json": str(out_dir / "best_schedule.json"),
        "progress_trial_events_jsonl": str(progress / "trial_events.jsonl"),
        "progress_current_best_json": str(progress / "current_best.json"),
        "heartbeat_json": str(out_dir / "heartbeat.json"),
        "status_usable_policy": row.get("calibration_usable_status_policy") or "strict_status_completed_v1",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    heartbeat = {
        "schema": "paper_i_comparator_spsa_calibration_heartbeat_v1",
        "status": "completed",
        "updated_utc": "2026-05-31T00:00:00Z",
        "started_utc": "2026-05-31T00:00:00Z",
        "record_id": record_id,
        "method_id": method_id,
        "target_id": target_id,
        "case_ids": case_ids,
        **scope_metadata,
        "completed_trial_count": 1,
        "usable_trial_count": 0 if quality_nonpassing else 1,
        "strict_usable_trial_count": 0 if quality_nonpassing else 1,
        "calibration_usable_trial_count": 1,
        "status_usable_policy": row.get("calibration_usable_status_policy") or "strict_status_completed_v1",
        "summary_json": str(out_dir / "summary.json"),
        "best_schedule_json": str(out_dir / "best_schedule.json"),
    }
    (out_dir / "heartbeat.json").write_text(json.dumps(heartbeat, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir


def test_checker_accepts_complete_calibration_output_and_writes_non_promotional_lock(tmp_path: Path) -> None:
    row = _row_for_method("static_hea_qiskit_vqe")
    records = tmp_path / "records.tsv"
    _write_rows(records, [row])
    output_root = tmp_path / "outputs"
    _write_good_output(output_root, row)
    lock_path = tmp_path / "schedule_lock_candidate.json"

    summary = checker.validate_outputs(
        records_path=records,
        output_root=output_root,
        summary_path=tmp_path / "check.json",
        schedule_lock_candidate_path=lock_path,
    )

    assert summary["ok"] is True, summary
    assert summary["usable_count"] == 1
    assert summary["table_evidence_status"] == "not_table_evidence"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    key = f"{row['method_id']}::{row['target_id']}"
    assert lock["schema"] == "paper_i_comparator_spsa_schedule_lock_candidate_v1"
    assert lock["evidence_role"] == "calibration_schedule_lock_candidate_only_not_table_evidence"
    assert lock["method_target_schedules"][key]["promotion_status"] == "candidate_not_promoted_user_approval_required"
    assert lock["method_target_schedules"][key]["table_evidence_status"] == "not_table_evidence"


def test_checker_accepts_warm_start_trial_zero_metadata(tmp_path: Path) -> None:
    row = _row_for_method("static_hea_qiskit_vqe")
    warm_key = f"{row['method_id']}::{row['target_id']}"
    warm_lock = tmp_path / "warm_start_schedule_lock.json"
    row["warm_start_schedule_lock_json"] = str(warm_lock)
    row["warm_start_schedule_key"] = warm_key
    schedule = _schedule_for(row)
    warm_lock.write_text(
        json.dumps(
            {
                "schema": "paper_i_comparator_spsa_schedule_lock_candidate_v1",
                "method_target_schedules": {
                    warm_key: {"method_id": row["method_id"], "target_id": row["target_id"], "schedule": schedule}
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    records = tmp_path / "records.tsv"
    _write_rows(records, [row])
    output_root = tmp_path / "outputs"
    _write_good_output(output_root, row)

    summary = checker.validate_outputs(records_path=records, output_root=output_root)

    assert summary["ok"] is True, summary
    assert summary["usable_count"] == 1
    report = summary["usable"][0]
    assert report["errors"] == []
    assert report["best_schedule"] == schedule


def test_checker_accepts_repair_policy_quality_nonpassing_as_calibration_usable(tmp_path: Path) -> None:
    repair_rows, _smoke, _config = generator.build_rows(
        config_path=REPAIR_SMOKE_CONFIG,
        generation_mode="smoke",
        repair_scope=generator.HH_GEO_QEB_TABLEIII_REPAIR_SCOPE,
    )
    row = {(item["method_id"], item["target_id"]): item for item in repair_rows}[("static_geo_adapt_vqe", "hh_sym_weak_weak")]
    records = tmp_path / "records.tsv"
    _write_rows(records, repair_rows)
    output_root = tmp_path / "outputs"
    _write_good_output(output_root, row, quality_nonpassing=True)
    lock_path = tmp_path / "schedule_lock_candidate.json"

    summary = checker.validate_outputs(
        records_path=records,
        output_root=output_root,
        summary_path=tmp_path / "check.json",
        schedule_lock_candidate_path=lock_path,
        allow_incomplete=True,
    )

    assert summary["ok"] is True, summary
    assert summary["usable_count"] == 1
    assert summary["missing_count"] == 7
    assert summary["usable"][0]["usable_trial_count"] == 0
    assert summary["usable"][0]["calibration_usable_trial_count"] == 1
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    key = f"{row['method_id']}::{row['target_id']}"
    assert lock["method_target_schedules"][key]["strict_usable_trial_count"] == 0
    assert lock["method_target_schedules"][key]["calibration_usable_trial_count"] == 1


def test_checker_flags_missing_current_best_as_incomplete(tmp_path: Path) -> None:
    row = _row_for_method("static_hea_qiskit_vqe")
    records = tmp_path / "records.tsv"
    _write_rows(records, [row])
    output_root = tmp_path / "outputs"
    _write_good_output(output_root, row, omit_current_best=True)

    summary = checker.validate_outputs(records_path=records, output_root=output_root)

    assert summary["ok"] is False
    assert summary["incomplete_count"] == 1
    assert any("current_best_missing" in error for error in summary["incomplete"][0]["errors"])


def test_checker_rejects_best_schedule_with_disallowed_field(tmp_path: Path) -> None:
    row = _row_for_method("static_hea_qiskit_vqe")
    records = tmp_path / "records.tsv"
    _write_rows(records, [row])
    output_root = tmp_path / "outputs"
    _write_good_output(output_root, row, bad_schedule=True)

    summary = checker.validate_outputs(records_path=records, output_root=output_root)
    reasons = "\n".join(summary["failed"][0]["errors"])

    assert summary["ok"] is False
    assert summary["failed_count"] == 1
    assert "schedule_fields_not_allowed" in reasons


def test_checker_rejects_partial_best_schedule(tmp_path: Path) -> None:
    row = _row_for_method("static_hea_qiskit_vqe")
    records = tmp_path / "records.tsv"
    _write_rows(records, [row])
    output_root = tmp_path / "outputs"
    out_dir = _write_good_output(output_root, row)
    best_path = out_dir / "best_schedule.json"
    best = json.loads(best_path.read_text(encoding="utf-8"))
    best["schedule"] = {"hea_spsa_learning_rate": 0.01}
    best_path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = checker.validate_outputs(records_path=records, output_root=output_root)
    reasons = "\n".join(summary["failed"][0]["errors"])

    assert summary["ok"] is False
    assert "schedule_fields_mismatch" in reasons


def test_checker_rejects_stale_summary_best_schedule(tmp_path: Path) -> None:
    row = _row_for_method("static_hea_qiskit_vqe")
    records = tmp_path / "records.tsv"
    _write_rows(records, [row])
    output_root = tmp_path / "outputs"
    out_dir = _write_good_output(output_root, row)
    summary_path = out_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["best_schedule"] = {"hea_spsa_learning_rate": 0.02, "hea_spsa_perturbation": 0.005}
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = checker.validate_outputs(records_path=records, output_root=output_root)
    reasons = "\n".join(summary["failed"][0]["errors"])

    assert summary["ok"] is False
    assert "best_schedule_summary_schedule_mismatch" in reasons


def test_checker_cli_writes_summary_and_returns_nonzero_for_missing_outputs(tmp_path: Path) -> None:
    row = _row_for_method("static_family_informed_vqe")
    records = tmp_path / "records.tsv"
    _write_rows(records, [row])
    summary_path = tmp_path / "summary.json"

    rc = checker.main([
        "--records",
        str(records),
        "--output-root",
        str(tmp_path / "missing_outputs"),
        "--summary",
        str(summary_path),
    ])

    assert rc == 1
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["missing_count"] == 1
    assert payload["evidence_role"] == "calibration_only_not_manuscript_table_evidence"
