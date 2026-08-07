from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.paper_iii_excited_dynamics import preflight_inputs, run_task, validate_outputs


FIELDS = [
    "record_id",
    "queue",
    "mode",
    "source_artifact_json",
    "artifact_json",
    "source_existing_strict_output_json",
    "existing_strict_output_json",
    "t_final",
    "num_times",
    "timeout_seconds",
    "run_tag",
    "require_progress_json",
    "require_partial_payload_json",
    "require_science_benchmark",
    "notes",
]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _valid_strict_hh_payload(*, t_final: float = 0.2, num_times: int = 3) -> dict:
    times = [idx * t_final / (num_times - 1) for idx in range(num_times)]
    trajectory = []
    for idx, time_value in enumerate(times):
        trajectory.append(
            {
                "step_index": idx,
                "time": time_value,
                "trajectory_sample_kind": "state_sample",
                "advances_time": True,
                "controller_exact_input_mode": "off",
                "decision_backend": "oracle",
                "decision_data_flow": "ideal_observable_estimator",
                "diagnostic_exact_reference_mode": "benchmark_exact",
                "strict_measurement_oracle_certified": True,
                "uses_reference_for_decision": False,
                "uses_future_exact_forecast_for_decision": False,
                "energy_total_controller": 0.10 + 0.01 * idx,
                "energy_total_exact": 0.105 + 0.01 * idx,
                "abs_energy_total_error": 0.005,
                "abs_primary_density_error": 0.002 * idx,
                "site_occupations_abs_error_max": 0.003 * idx,
                "abs_staggered_error": 0.004 * idx,
                "abs_doublon_error": 0.001 * idx,
            }
        )
    summary = {
        "mode": "oracle_v1",
        "status": "completed",
        "decision_path_kind": "strict_qpu_faithful_oracle_v1",
        "strict_qpu_faithful": True,
        "strict_qpu_hh": True,
        "strict_fail_closed": False,
        "qpu_faithful_decisions_passed": True,
        "strict_decision_contract_passed": True,
        "strict_measurement_oracle_certified": True,
        "exact_decision_checkpoints": 0,
        "reference_enabled": False,
        "controller_reference_enabled": False,
        "controller_exact_input_mode": "off",
        "diagnostic_exact_reference_mode": "benchmark_exact",
        "diagnostic_exact_reference_enabled": True,
        "decision_backend": "oracle",
        "decision_data_flow": "ideal_observable_estimator",
        "uses_reference_for_decision": False,
        "uses_future_exact_forecast_for_decision": False,
        "uses_statevector_as_ideal_observable_estimator": True,
        "final_abs_energy_total_error": 0.005,
        "mean_abs_energy_total_error": 0.005,
        "max_abs_energy_total_error": 0.005,
    }
    return {
        "run_tag": "paper_iii_p9_unit_strict_payload",
        "artifact_json": "source_adapt.json",
        "route_config": {
            "problem_family": "hh",
            "strict_qpu_faithful": True,
            "strict_qpu_hh": True,
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_exact",
            "decision_data_flow": "ideal_observable_estimator",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "uses_statevector_as_ideal_observable_estimator": True,
            "strict_measurement_oracle_certified": True,
        },
        "summary": summary,
        "trajectory": trajectory,
        "ledger": [dict(row, ledger_row_kind="decision") for row in trajectory[1:]],
        "reference": {
            "reference_enabled": False,
            "controller_exact_input_mode": "off",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        },
        "diagnostic_reference": {
            "diagnostic_reference_mode": "benchmark_exact",
            "diagnostic_reference_enabled": True,
            "role": "diagnostic_exact_benchmark",
            "feeds_controller_decisions": False,
            "controller_reference_enabled": False,
            "controller_exact_input_mode": "off",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        },
    }


def _write_report_only_fixture(tmp_path: Path) -> tuple[Path, Path, str, Path]:
    repo = tmp_path / "repo"
    source = _write_json(repo / "source_adapt.json", {"schema_version": "test_seed_v1"})
    existing_source = _write_json(repo / "existing" / "hh_strict_realtime_pilot.json", _valid_strict_hh_payload())
    record_id = "paper_iii_p9_report_only_existing_p7b_smoke"
    staged_seed = repo / "chtc" / "paper_iii_excited_dynamics" / "input" / "seed_artifacts" / f"{record_id}.json"
    staged_existing = repo / "chtc" / "paper_iii_excited_dynamics" / "input" / "existing_strict_outputs" / f"{record_id}.json"
    records = repo / "records.tsv"
    row = {
        "record_id": record_id,
        "queue": "smoke",
        "mode": preflight_inputs.MODE_REPORT_ONLY_EXISTING_OUTPUT,
        "source_artifact_json": str(source),
        "artifact_json": str(staged_seed),
        "source_existing_strict_output_json": str(existing_source),
        "existing_strict_output_json": str(staged_existing),
        "t_final": "0.2",
        "num_times": "3",
        "timeout_seconds": "30",
        "run_tag": record_id,
        "require_progress_json": "false",
        "require_partial_payload_json": "false",
        "require_science_benchmark": "true",
        "notes": "unit report-only smoke",
    }
    records.parent.mkdir(parents=True, exist_ok=True)
    with records.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    queue = repo / "smoke_record_ids.txt"
    queue.write_text(record_id + "\n", encoding="utf-8")
    return repo, records, record_id, queue


def _write_chtc_status(record_dir: Path) -> None:
    task_result = json.loads((record_dir / "task_result.json").read_text(encoding="utf-8"))
    status = {
        "schema_version": "paper_iii_excited_dynamics_chtc_status_v1",
        "record_id": record_dir.name,
        "return_code": task_result["return_code"],
        "task_result_exists": True,
        "report_json_exists": task_result["report_exists"],
        "run_manifest_exists": task_result["run_manifest_exists"],
    }
    _write_json(record_dir / "chtc_status.json", status)


def _run_report_only_record(tmp_path: Path) -> tuple[Path, dict]:
    repo, records, record_id, queue = _write_report_only_fixture(tmp_path)
    preflight = preflight_inputs.preflight_records(records_path=records, record_list=queue, repo_root=repo)
    assert preflight["ok"] is True, preflight
    record_dir = tmp_path / "outputs" / record_id
    rc = run_task.run_record(record_id, records, record_dir)
    assert rc == 0
    _write_chtc_status(record_dir)
    report = validate_outputs.validate_outputs(tmp_path / "outputs", record_list=queue, write_report=False)
    assert report["ok"] is True, report
    return record_dir, report


def test_preflight_stages_seed_and_existing_strict_output(tmp_path: Path) -> None:
    repo, records, _record_id, queue = _write_report_only_fixture(tmp_path)

    report = preflight_inputs.preflight_records(records_path=records, record_list=queue, repo_root=repo)

    assert report["ok"] is True, report
    assert report["schema_version"] == "paper_iii_excited_dynamics_preflight_report_v1"
    assert report["record_count"] == 1
    rec = report["records"][0]
    assert rec["staged_artifact_exists"] is True
    assert rec["staged_existing_strict_output_exists"] is True
    assert rec["strict_existing_output_validation_passed"] is True
    assert (records.parent / "preflight_report.json").exists()


def test_report_only_run_task_and_validator_accept_clean_output(tmp_path: Path) -> None:
    record_dir, report = _run_report_only_record(tmp_path)

    assert report["strict_leakage_failure_count"] == 0
    assert not report["failed_records"]
    assert (record_dir / "record.json").exists()
    assert (record_dir / "command.sh").exists()
    assert (record_dir / "task_result.json").exists()
    assert (record_dir / "existing_strict_output.json").exists()
    assert (record_dir / "paper_iii_local_science_pilot" / "paper_iii_local_science_pilot_report.json").exists()
    assert (record_dir / "paper_iii_local_science_pilot" / "run_manifest.json").exists()


def test_report_only_validator_is_fetch_safe_without_staged_input_copy(tmp_path: Path) -> None:
    record_dir, _report = _run_report_only_record(tmp_path)
    task_result = json.loads((record_dir / "task_result.json").read_text(encoding="utf-8"))
    staged_source = Path(task_result["source_existing_strict_output_json"])
    assert staged_source.exists()
    staged_source.unlink()

    result = validate_outputs.validate_record_dir(record_dir)

    assert result.ok, result.errors
    assert (record_dir / "existing_strict_output.json").exists()


def test_validator_rejects_leakage_mutation(tmp_path: Path) -> None:
    record_dir, _report = _run_report_only_record(tmp_path)
    report_json = record_dir / "paper_iii_local_science_pilot" / "paper_iii_local_science_pilot_report.json"
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    payload["runs"]["strict_hh_runtime_dynamics"]["metrics"]["strict_route"]["uses_reference_for_decision"] = True
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_outputs.validate_record_dir(record_dir)

    assert not result.ok
    assert result.leakage_failure is True
    assert any("uses_reference_for_decision" in error for error in result.errors)


def test_strict_records_require_progress_and_partial_artifacts(tmp_path: Path) -> None:
    record_dir, _report = _run_report_only_record(tmp_path)
    record_path = record_dir / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["mode"] = preflight_inputs.MODE_STRICT_HH
    record["require_progress_json"] = "true"
    record["require_partial_payload_json"] = "true"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    task_result_path = record_dir / "task_result.json"
    task_result = json.loads(task_result_path.read_text(encoding="utf-8"))
    task_result["mode"] = preflight_inputs.MODE_STRICT_HH
    task_result["progress_json"] = str(record_dir / "progress.json")
    task_result["partial_payload_json"] = str(record_dir / "partial_payload.json")
    task_result["progress_exists"] = False
    task_result["partial_payload_exists"] = False
    task_result_path.write_text(json.dumps(task_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_json = record_dir / "paper_iii_local_science_pilot" / "paper_iii_local_science_pilot_report.json"
    report = json.loads(report_json.read_text(encoding="utf-8"))
    run = report["runs"]["strict_hh_runtime_dynamics"]
    run["command_status"] = "completed"
    run["progress_json"] = str(record_dir / "progress.json")
    run["partial_payload_json"] = str(record_dir / "partial_payload.json")
    run["progress_json_exists"] = False
    run["partial_payload_json_exists"] = False
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_outputs.validate_record_dir(record_dir)

    assert not result.ok
    assert any("progress" in error for error in result.errors)
    assert any("partial payload" in error for error in result.errors)


def test_run_task_rejects_compatibility_audit_only_without_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record_id = "paper_iii_unit_compatibility_audit_only"
    records = tmp_path / "records.tsv"
    row = {
        "record_id": record_id,
        "queue": "full",
        "mode": preflight_inputs.MODE_COMPATIBILITY_AUDIT_ONLY,
        "source_artifact_json": "",
        "artifact_json": "",
        "source_existing_strict_output_json": "",
        "existing_strict_output_json": "",
        "t_final": "",
        "num_times": "",
        "timeout_seconds": "",
        "run_tag": "",
        "require_progress_json": "false",
        "require_partial_payload_json": "false",
        "require_science_benchmark": "false",
        "notes": "audit-only unit guard",
    }
    with records.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    def _fail_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("compatibility_audit_only must not launch subprocess.run")

    monkeypatch.setattr(run_task.subprocess, "run", _fail_run)
    out = tmp_path / "outputs" / record_id

    rc = run_task.run_record(record_id, records, out)

    assert rc == 2
    assert (out / "record.json").exists()
    assert not (out / "command.sh").exists()
    result = json.loads((out / "task_result.json").read_text(encoding="utf-8"))
    assert result["return_code"] == 2
    assert result["command"] == []
    assert result["mode"] == preflight_inputs.MODE_COMPATIBILITY_AUDIT_ONLY
    assert "compatibility_audit_only" in result["error"]
    assert "no local science pilot or dynamics launched" in result["error"]


def test_default_records_and_submit_descriptors_are_ready() -> None:
    rows = preflight_inputs.load_records(preflight_inputs.DEFAULT_RECORDS)
    record_ids = [row["record_id"] for row in rows]

    assert record_ids == [
        "paper_iii_p9_report_only_existing_p7b_smoke",
        "paper_iii_p9_strict_hh_baseline_smoke",
    ]
    assert preflight_inputs.load_record_ids(REPO_ROOT / "chtc/paper_iii_excited_dynamics/input/smoke_record_ids.txt") == [
        "paper_iii_p9_report_only_existing_p7b_smoke"
    ]
    assert preflight_inputs.load_record_ids(REPO_ROOT / "chtc/paper_iii_excited_dynamics/input/full_record_ids.txt") == record_ids
    smoke_submit = (REPO_ROOT / "chtc/paper_iii_excited_dynamics/submit_smoke.sub").read_text(encoding="utf-8")
    full_submit = (REPO_ROOT / "chtc/paper_iii_excited_dynamics/submit_full.sub").read_text(encoding="utf-8")
    assert "queue record_id from chtc/paper_iii_excited_dynamics/input/smoke_record_ids.txt" in smoke_submit
    assert "queue record_id from chtc/paper_iii_excited_dynamics/input/full_record_ids.txt" in full_submit
    for text in (smoke_submit, full_submit):
        transfer_line = next(line for line in text.splitlines() if line.startswith("transfer_input_files"))
        assert "chtc/paper_iii_excited_dynamics" in transfer_line
        assert "pipelines" in transfer_line
        assert "src" in transfer_line
        assert "artifacts" not in transfer_line
        assert "raw_outputs" not in transfer_line
        assert "prompt-exports" not in transfer_line


def test_new_package_has_no_forbidden_imports() -> None:
    forbidden = ["qiskit", "remote_runner", "exact_bench", "pipelines.time_dynamics", "optuna"]
    for rel in (
        "chtc/paper_iii_excited_dynamics/preflight_inputs.py",
        "chtc/paper_iii_excited_dynamics/run_task.py",
        "chtc/paper_iii_excited_dynamics/validate_outputs.py",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        lowered = text.lower()
        for needle in forbidden:
            assert needle not in lowered, f"{needle} leaked into {rel}"


@pytest.mark.parametrize("script", ["run_task.sh", "run_task_apptainer.sh"])
def test_shell_wrappers_use_package_specific_env(script: str) -> None:
    text = (REPO_ROOT / "chtc" / "paper_iii_excited_dynamics" / script).read_text(encoding="utf-8")
    assert "PAPER_III_EXCITED_DYNAMICS_RECORDS_PATH" in text
    assert "PAPER_III_EXCITED_DYNAMICS_OUTPUT_ROOT" in text
    assert "remote-runner" not in text
    if script == "run_task.sh":
        assert "^[A-Za-z0-9_.-]+$" in text
