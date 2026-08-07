from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.excited_dynamics.io import write_json
from pipelines.excited_dynamics.paper_iii_local_evidence_batch import (
    JOB_MODE_REPORT_ONLY,
    JOB_MODE_STRICT_HH,
    PAPER_III_LOCAL_EVIDENCE_BATCH_SCHEMA_VERSION,
    PaperIIILocalEvidenceBatchConfig,
    build_default_job_plan,
    run_paper_iii_local_evidence_batch,
)
from pipelines.excited_dynamics.paper_iii_local_science_pilot import PaperIIILocalSciencePilotConfig


FORBIDDEN_MARKERS = (
    "amplitudes_qn_to_q0",
    "raw_physical_state",
    "basis_matrix_vectors",
    "exact_target_trajectories",
    "exact_step_forecast",
    "state_at(",
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fake_pilot_factory(*, leak_strict: bool = False):
    def _fake_pilot(config: PaperIIILocalSciencePilotConfig) -> dict:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        if not config.report_only_existing_output:
            _write_json(config.strict_output_json, {"schema_version": "fake_strict_hh_output_v1"})
        if config.progress_json is not None:
            _write_json(Path(config.progress_json), {"schema_version": "fake_progress_v1", "checkpoint_count": 2})
        if config.partial_payload_json is not None:
            _write_json(Path(config.partial_payload_json), {"schema_version": "fake_partial_payload_v1"})
        leak = bool(leak_strict and not config.report_only_existing_output)
        strict_route = {
            "mode": "oracle_v1",
            "decision_backend": "oracle",
            "decision_data_flow": "ideal_observable_estimator",
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_exact",
            "strict_qpu_faithful": True,
            "strict_qpu_hh": True,
            "strict_measurement_oracle_certified": True,
            "qpu_faithful_decisions_passed": not leak,
            "strict_decision_contract_passed": not leak,
            "exact_decision_checkpoints": 0,
            "uses_reference_for_decision": leak,
            "uses_future_exact_forecast_for_decision": False,
            "uses_statevector_as_ideal_observable_estimator": True,
        }
        blockers = ["strict_validation:uses_reference_for_decision=true"] if leak else []
        report = {
            "schema_version": "paper_iii_local_science_pilot_report_v1",
            "pipeline": "paper_iii_local_science_pilot",
            "generated_utc": "2026-05-17T04:00:00Z",
            "paper_iii_science_benchmark": not leak,
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "reference_comparisons_feed_controller_decisions": False,
            "controller_boundary": {
                "feeds_controller_decisions": False,
                "reference_comparisons_feed_controller_decisions": False,
            },
            "runs": {
                "strict_hh_runtime_dynamics": {
                    "status": "blocked" if leak else "completed",
                    "command_status": "existing_output" if config.report_only_existing_output else "completed",
                    "command_wallclock_seconds": 0.0 if config.report_only_existing_output else 0.01,
                    "strict_payload_source_json": str(
                        config.existing_strict_output_json or config.strict_output_json
                    ),
                    "output_json": str(config.existing_strict_output_json or config.strict_output_json),
                    "strict_validation": {
                        "passed": not leak,
                        "violations": ["uses_reference_for_decision"] if leak else [],
                        "violation_count": 1 if leak else 0,
                        "physical_row_count": 9,
                        "final_physical_time": 1.0,
                    },
                    "metrics": {
                        "horizon": {
                            "physical_row_count": 9,
                            "final_physical_time": 1.0,
                            "full_horizon_gate_passed": True,
                        },
                        "strict_route": strict_route,
                        "energy": {"final_abs_energy_total_error": 0.01, "max_abs_energy_total_error": 0.02},
                        "fidelity": {"fidelity_exact_status": "computed"},
                    },
                }
            },
            "blockers": blockers,
            "scope_guardrails": {
                "chtc_used": False,
                "optuna_used": False,
                "fresh_adapt_run": False,
            },
        }
        write_json(config.report_json, report)
        config.report_md.write_text("# fake report\n", encoding="utf-8")
        config.command_log_md.write_text("# fake command log\n", encoding="utf-8")
        write_json(config.run_manifest_json, {"schema_version": "agent_run_manifest_v1"})
        config.command_sh.write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
        config.stdout_log.write_text("", encoding="utf-8")
        config.stderr_log.write_text("", encoding="utf-8")
        return report

    return _fake_pilot


def _batch_config(tmp_path: Path, *, include_second_strict: bool = False) -> PaperIIILocalEvidenceBatchConfig:
    source = _write_json(tmp_path / "source_adapt.json", {"schema_version": "test_adapt_seed_v1"})
    existing = _write_json(tmp_path / "existing" / "hh_strict_realtime_pilot.json", {"schema_version": "fake_existing"})
    return PaperIIILocalEvidenceBatchConfig(
        output_dir=tmp_path / "batch",
        artifact_json=source,
        existing_strict_output_json=existing,
        scoreboard_md=tmp_path / "prompt-exports" / "optimize-paper-iii-local-jobs-runs.md",
        t_final=1.0,
        num_times=9,
        timeout_seconds=1800,
        include_second_strict=include_second_strict,
    )


def test_default_job_plan_is_report_strict_report(tmp_path: Path) -> None:
    config = _batch_config(tmp_path)

    jobs = build_default_job_plan(config)

    assert [job.job_id for job in jobs] == ["p8-report-only-001", "p8-strict-001", "p8-report-only-002"]
    assert [job.mode for job in jobs] == [JOB_MODE_REPORT_ONLY, JOB_MODE_STRICT_HH, JOB_MODE_REPORT_ONLY]
    assert jobs[1].progress_json == config.output_dir / "baseline_strict_hh_001" / "progress.json"
    assert jobs[1].partial_payload_json == config.output_dir / "baseline_strict_hh_001" / "partial_payload.json"
    assert jobs[2].existing_strict_output_json == config.output_dir / "baseline_strict_hh_001" / "hh_strict_realtime_pilot.json"


def test_batch_runs_default_sequence_and_writes_scoreboard(tmp_path: Path) -> None:
    config = _batch_config(tmp_path)

    summary = run_paper_iii_local_evidence_batch(
        config,
        pilot_fn=_fake_pilot_factory(),
        command_argv=["python", "-m", "pipelines.excited_dynamics.paper_iii_local_evidence_batch"],
    )

    assert summary["schema_version"] == PAPER_III_LOCAL_EVIDENCE_BATCH_SCHEMA_VERSION
    assert [job["job_id"] for job in summary["jobs"]] == [
        "p8-report-only-001",
        "p8-strict-001",
        "p8-report-only-002",
    ]
    assert summary["primary_metric"]["attempted_job_count"] == 3
    assert summary["primary_metric"]["validated_local_evidence_job_count"] == 3
    assert summary["primary_metric"]["validated_strict_runtime_job_count"] == 1
    assert summary["primary_metric"]["controller_leakage_failure_count"] == 0
    strict_job = summary["jobs"][1]
    assert strict_job["artifacts"]["progress_json_exists"] is True
    assert strict_job["artifacts"]["partial_payload_json_exists"] is True
    assert summary["stop_continue_decision"]["decision"] == "continue"

    assert config.batch_summary_json.exists()
    assert config.batch_summary_md.exists()
    assert config.run_manifest_json.exists()
    assert config.scoreboard_md.exists()
    scoreboard = config.scoreboard_md.read_text(encoding="utf-8")
    assert "p8-report-only-001" in scoreboard
    assert "p8-strict-001" in scoreboard
    assert "progress=yes, partial=yes" in scoreboard
    for marker in FORBIDDEN_MARKERS:
        assert marker not in json.dumps(summary, sort_keys=True)
        assert marker not in scoreboard


def test_batch_stops_fail_closed_on_strict_leakage(tmp_path: Path) -> None:
    config = _batch_config(tmp_path)

    summary = run_paper_iii_local_evidence_batch(
        config,
        pilot_fn=_fake_pilot_factory(leak_strict=True),
        command_argv=["python", "-m", "pipelines.excited_dynamics.paper_iii_local_evidence_batch"],
    )

    assert [job["job_id"] for job in summary["jobs"]] == ["p8-report-only-001", "p8-strict-001"]
    assert summary["skipped_job_ids"] == ["p8-report-only-002"]
    assert summary["primary_metric"]["controller_leakage_failure_count"] == 1
    assert summary["jobs"][1]["status"] == "blocked"
    assert summary["jobs"][1]["controller_leakage_failure"] is True
    assert summary["stop_continue_decision"]["decision"] == "stop"
    assert "p8-report-only-002" not in config.scoreboard_md.read_text(encoding="utf-8")


def test_second_strict_plan_uses_sequential_strict_job(tmp_path: Path) -> None:
    config = _batch_config(tmp_path, include_second_strict=True)

    jobs = build_default_job_plan(config)

    assert [job.job_id for job in jobs] == ["p8-report-only-001", "p8-strict-001", "p8-strict-002"]
    assert [job.mode for job in jobs] == [JOB_MODE_REPORT_ONLY, JOB_MODE_STRICT_HH, JOB_MODE_STRICT_HH]
    assert jobs[2].progress_json == config.output_dir / "baseline_strict_hh_002" / "progress.json"

