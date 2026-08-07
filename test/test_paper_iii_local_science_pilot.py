from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.excited_dynamics.io import sha256_file
from pipelines.excited_dynamics.paper_iii_local_science_pilot import (
    PAPER_III_LOCAL_SCIENCE_PILOT_REPORT_SCHEMA_VERSION,
    PaperIIILocalSciencePilotConfig,
    PaperIIILocalSciencePilotError,
    build_strict_hh_command,
    main,
    run_paper_iii_local_science_pilot,
)


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


def _decision_row(step_index: int, time_value: float, *, include_fidelity: bool = False) -> dict:
    row = {
        "step_index": step_index,
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
        "energy_total_controller": 0.10 + 0.01 * step_index,
        "energy_total_exact": 0.105 + 0.01 * step_index,
        "abs_energy_total_error": 0.005,
        "abs_primary_density_error": 0.002 * step_index,
        "site_occupations_abs_error_max": 0.003 * step_index,
        "abs_staggered_error": 0.004 * step_index,
        "abs_doublon_error": 0.001 * step_index,
    }
    if include_fidelity:
        row["fidelity_exact"] = 0.99 - 0.01 * step_index
    return row


def _valid_strict_hh_payload(*, t_final: float = 0.2, num_times: int = 3, include_fidelity: bool = False) -> dict:
    times = [idx * t_final / (num_times - 1) for idx in range(num_times)]
    trajectory = [_decision_row(idx, time_value, include_fidelity=include_fidelity) for idx, time_value in enumerate(times)]
    ledger = [dict(row, ledger_row_kind="decision") for row in trajectory[1:]]
    summary = {
        "mode": "oracle_v1",
        "status": "completed",
        "decision_path_kind": "strict_qpu_faithful_oracle_v1",
        "strict_qpu_faithful": True,
        "strict_qpu_hh": True,
        "strict_fail_closed": False,
        "qpu_faithful_decisions_expected": True,
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
        "run_tag": "paper_iii_p7b_hh_strict_realtime_pilot",
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
            "compile_audit_mode": "off",
        },
        "summary": summary,
        "trajectory": trajectory,
        "ledger": ledger,
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


def _source_and_config(tmp_path: Path, payload: dict, *, t_final: float = 0.2, num_times: int = 3) -> tuple[Path, PaperIIILocalSciencePilotConfig]:
    source = _write_json(tmp_path / "source_adapt.json", {"schema_version": "test_adapt_seed_v1"})
    output_dir = tmp_path / "p7b"
    _write_json(output_dir / "hh_strict_realtime_pilot.json", payload)
    return source, PaperIIILocalSciencePilotConfig(
        artifact_json=source,
        output_dir=output_dir,
        t_final=t_final,
        num_times=num_times,
        timeout_seconds=5,
        report_only_existing_output=True,
    )


def _strict_validation(report: dict) -> dict:
    return report["runs"]["strict_hh_runtime_dynamics"]["strict_validation"]


def test_strict_command_forwards_progress_and_partial_paths(tmp_path: Path) -> None:
    config = PaperIIILocalSciencePilotConfig(
        artifact_json=tmp_path / "source_adapt.json",
        output_dir=tmp_path / "p7b",
        progress_json=tmp_path / "p7b" / "progress.json",
        partial_payload_json=tmp_path / "p7b" / "partial_payload.json",
    )

    command = build_strict_hh_command(config)

    assert "--progress-json" in command
    assert command[command.index("--progress-json") + 1] == str(config.progress_json)
    assert "--partial-payload-json" in command
    assert command[command.index("--partial-payload-json") + 1] == str(config.partial_payload_json)


def test_strict_command_omits_progress_and_partial_paths_by_default(tmp_path: Path) -> None:
    command = build_strict_hh_command(
        PaperIIILocalSciencePilotConfig(
            artifact_json=tmp_path / "source_adapt.json",
            output_dir=tmp_path / "p7b",
        )
    )

    assert "--progress-json" not in command
    assert "--partial-payload-json" not in command


def test_report_builder_accepts_valid_strict_hh_payload_and_writes_artifacts(tmp_path: Path) -> None:
    source, config = _source_and_config(tmp_path, _valid_strict_hh_payload())

    report = run_paper_iii_local_science_pilot(config)

    assert report["schema_version"] == PAPER_III_LOCAL_SCIENCE_PILOT_REPORT_SCHEMA_VERSION
    assert report["paper_iii_science_benchmark"] is True
    assert report["production_claim"] is False
    assert report["controller_usable"] is False
    assert report["feeds_controller_decisions"] is False
    assert report["reference_comparisons_feed_controller_decisions"] is False
    assert report["case"]["family"] == "hubbard_holstein"
    assert report["case"]["chtc_used"] is False
    assert report["case"]["optuna_used"] is False
    assert report["case"]["downscaled_local_pilot"] is True
    assert report["evidence_classification"]["p5b_p6a_p7a_reclassified_as_science"] is False
    assert report["runs"]["strict_hh_runtime_dynamics"]["source_artifact_sha256"] == sha256_file(source)
    run = report["runs"]["strict_hh_runtime_dynamics"]
    assert run["command_started_utc"] is not None
    assert run["command_finished_utc"] is not None
    assert run["command_wallclock_seconds"] is not None
    assert run["command_wallclock_seconds"] >= 0.0
    assert run["report_only_existing_output"] is True
    assert run["progress_json"] is None
    assert run["progress_json_exists"] is False
    assert run["partial_payload_json"] is None
    assert run["partial_payload_json_exists"] is False
    assert _strict_validation(report)["passed"] is True
    assert report["post_run_comparisons"]["fidelity"]["fidelity_exact_status"] == "missing_not_computed"

    for path in (
        config.report_json,
        config.report_md,
        config.command_log_md,
        config.run_manifest_json,
        config.command_sh,
        config.stdout_log,
        config.stderr_log,
    ):
        assert path.exists(), path

    manifest = json.loads(config.run_manifest_json.read_text(encoding="utf-8"))
    assert manifest["paper_iii_science_benchmark"] is True
    assert manifest["output_summary"]["fidelity_exact_status"] == "missing_not_computed"
    assert manifest["commands"]["strict_hh_runtime_dynamics"]["wallclock_seconds"] is not None
    assert manifest["output_summary"]["command_wallclock_seconds"] is not None
    assert "Status: `existing_output`" in config.command_log_md.read_text(encoding="utf-8")
    assert "Wall-clock seconds:" in config.command_log_md.read_text(encoding="utf-8")


def test_cli_report_only_existing_output_writes_expected_files(tmp_path: Path) -> None:
    source, config = _source_and_config(tmp_path, _valid_strict_hh_payload(include_fidelity=True))

    assert main(
        [
            "--artifact-json",
            str(source),
            "--output-dir",
            str(config.output_dir),
            "--t-final",
            "0.2",
            "--num-times",
            "3",
            "--report-only-existing-output",
        ]
    ) == 0

    report = json.loads(config.report_json.read_text(encoding="utf-8"))
    assert report["paper_iii_science_benchmark"] is True
    assert report["post_run_comparisons"]["fidelity"]["fidelity_exact_status"] == "computed"
    assert report["post_run_comparisons"]["fidelity"]["min_fidelity_exact"] == pytest.approx(0.97)
    assert "strict_validation_passed: `true`" in config.report_md.read_text(encoding="utf-8")


def test_external_existing_strict_output_validates_into_new_output_dir_without_source_rewrite(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "source_adapt.json", {"schema_version": "test_adapt_seed_v1"})
    existing_strict = _write_json(tmp_path / "p7b_source" / "hh_strict_realtime_pilot.json", _valid_strict_hh_payload())
    before_hash = sha256_file(existing_strict)
    output_dir = tmp_path / "fresh_report_dir"
    config = PaperIIILocalSciencePilotConfig(
        artifact_json=source,
        output_dir=output_dir,
        t_final=0.2,
        num_times=3,
        timeout_seconds=5,
        report_only_existing_output=True,
        existing_strict_output_json=existing_strict,
    )

    report = run_paper_iii_local_science_pilot(config)

    assert sha256_file(existing_strict) == before_hash
    assert report["paper_iii_science_benchmark"] is True
    run = report["runs"]["strict_hh_runtime_dynamics"]
    assert run["command_status"] == "existing_output"
    assert run["existing_strict_output_json"] == str(existing_strict)
    assert run["strict_payload_source_json"] == str(existing_strict)
    assert config.report_json.exists()
    assert config.command_log_md.exists()
    assert config.run_manifest_json.exists()
    assert not config.strict_output_json.exists()


def test_missing_external_existing_strict_output_fails_closed_in_new_output_dir(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "source_adapt.json", {"schema_version": "test_adapt_seed_v1"})
    missing_existing = tmp_path / "p7b_source" / "missing_hh_strict_realtime_pilot.json"
    output_dir = tmp_path / "fresh_report_dir"
    config = PaperIIILocalSciencePilotConfig(
        artifact_json=source,
        output_dir=output_dir,
        t_final=0.2,
        num_times=3,
        timeout_seconds=5,
        report_only_existing_output=True,
        existing_strict_output_json=missing_existing,
    )

    report = run_paper_iii_local_science_pilot(config)

    assert report["paper_iii_science_benchmark"] is False
    assert "existing_strict_output_missing" in report["blockers"]
    assert config.strict_output_json.exists()
    assert not missing_existing.exists()
    assert "existing strict HH output missing" in config.stderr_log.read_text(encoding="utf-8")


def test_malformed_existing_strict_output_fails_closed_with_blocked_report(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "source_adapt.json", {"schema_version": "test_adapt_seed_v1"})
    malformed = tmp_path / "p7b_source" / "hh_strict_realtime_pilot.json"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{not json", encoding="utf-8")
    config = PaperIIILocalSciencePilotConfig(
        artifact_json=source,
        output_dir=tmp_path / "fresh_report_dir",
        t_final=0.2,
        num_times=3,
        timeout_seconds=5,
        report_only_existing_output=True,
        existing_strict_output_json=malformed,
    )

    report = run_paper_iii_local_science_pilot(config)

    assert report["paper_iii_science_benchmark"] is False
    assert any("strict_output_json_parse_error" in blocker for blocker in report["blockers"])
    assert config.strict_output_json.exists()
    assert json.loads(config.strict_output_json.read_text(encoding="utf-8"))["status"] == "blocked"


def test_existing_strict_output_path_requires_report_only_mode(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "source_adapt.json", {"schema_version": "test_adapt_seed_v1"})
    config = PaperIIILocalSciencePilotConfig(
        artifact_json=source,
        output_dir=tmp_path / "p7b",
        existing_strict_output_json=tmp_path / "source_p7b" / "hh_strict_realtime_pilot.json",
    )

    with pytest.raises(PaperIIILocalSciencePilotError, match="report_only_existing_output"):
        run_paper_iii_local_science_pilot(config)


def test_strict_run_unlinks_stale_progress_and_partial_files_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_json(tmp_path / "source_adapt.json", {"schema_version": "test_adapt_seed_v1"})
    progress_json = tmp_path / "p7b" / "progress.json"
    partial_payload_json = tmp_path / "p7b" / "partial_payload.json"
    _write_json(progress_json, {"schema_version": "stale_progress"})
    _write_json(partial_payload_json, {"schema_version": "stale_partial"})

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(
        "pipelines.excited_dynamics.paper_iii_local_science_pilot.subprocess.run",
        _fake_run,
    )
    config = PaperIIILocalSciencePilotConfig(
        artifact_json=source,
        output_dir=tmp_path / "p7b",
        t_final=0.2,
        num_times=3,
        timeout_seconds=5,
        progress_json=progress_json,
        partial_payload_json=partial_payload_json,
    )

    report = run_paper_iii_local_science_pilot(config)

    assert report["paper_iii_science_benchmark"] is False
    run = report["runs"]["strict_hh_runtime_dynamics"]
    assert run["progress_json_exists"] is False
    assert run["partial_payload_json_exists"] is False
    assert not progress_json.exists()
    assert not partial_payload_json.exists()


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda p: p["route_config"].__setitem__("controller_exact_input_mode", "benchmark_exact"), "controller_exact_input_mode"),
        (lambda p: p["summary"].__setitem__("uses_reference_for_decision", True), "uses_reference_for_decision"),
        (lambda p: p["trajectory"][0].__setitem__("uses_future_exact_forecast_for_decision", True), "uses_future_exact_forecast_for_decision"),
        (lambda p: p["diagnostic_reference"].__setitem__("feeds_controller_decisions", True), "feeds_controller_decisions"),
        (lambda p: p["summary"].__setitem__("exact_decision_checkpoints", 1), "exact_decision_checkpoints"),
        (lambda p: p["route_config"].__setitem__("problem_family", "hubbard"), "problem_family"),
        (lambda p: p["ledger"][0].__setitem__("decision_backend", "exact"), "decision_backend"),
        (lambda p: p["trajectory"][0].__setitem__("amplitudes_qn_to_q0", [1.0, 0.0]), "forbidden raw/reference payload"),
    ],
)
def test_fail_closed_for_strict_hh_contract_violations(
    tmp_path: Path,
    mutate: Callable[[dict], None],
    needle: str,
) -> None:
    payload = _valid_strict_hh_payload()
    mutate(payload)
    _source, config = _source_and_config(tmp_path, payload)

    report = run_paper_iii_local_science_pilot(config)

    assert report["paper_iii_science_benchmark"] is False
    assert report["evidence_classification"]["paper_iii_science_benchmark"] is False
    assert report["evidence_classification"]["production_claim"] is False
    violations = _strict_validation(report)["violations"]
    assert any(needle in violation for violation in violations)


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda p: p["summary"].__setitem__("reference_enabled", True), "reference_enabled"),
        (lambda p: p["summary"].__setitem__("qpu_faithful_decisions_passed", False), "qpu_faithful_decisions_passed"),
        (lambda p: p["summary"].__setitem__("strict_decision_contract_passed", False), "strict_decision_contract_passed"),
        (lambda p: p["summary"].__setitem__("strict_fail_closed", True), "strict_fail_closed"),
        (lambda p: p["route_config"].__setitem__("diagnostic_exact_reference_mode", "off"), "diagnostic_exact_reference_mode"),
        (lambda p: p["summary"].__setitem__("diagnostic_exact_reference_mode", "off"), "diagnostic_exact_reference_mode"),
        (
            lambda p: p["route_config"].__setitem__("strict_measurement_oracle_certified", False),
            "strict_measurement_oracle_certified",
        ),
        (
            lambda p: p["summary"].__setitem__("strict_measurement_oracle_certified", False),
            "strict_measurement_oracle_certified",
        ),
        (lambda p: p["summary"].__setitem__("decision_backend", "exact"), "decision_backend"),
        (lambda p: p["summary"].__setitem__("decision_data_flow", "exact_reference"), "decision_data_flow"),
        (lambda p: p["route_config"].__setitem__("decision_data_flow", "exact_reference"), "decision_data_flow"),
    ],
)
def test_fail_closed_for_summary_and_route_boundary_violations(
    tmp_path: Path,
    mutate: Callable[[dict], None],
    needle: str,
) -> None:
    payload = _valid_strict_hh_payload()
    mutate(payload)
    _source, config = _source_and_config(tmp_path, payload)

    report = run_paper_iii_local_science_pilot(config)

    assert report["paper_iii_science_benchmark"] is False
    assert any(needle in violation for violation in _strict_validation(report)["violations"])


def test_fail_closed_when_horizon_is_short_or_physical_rows_missing(tmp_path: Path) -> None:
    short_payload = _valid_strict_hh_payload(t_final=0.1, num_times=2)
    _source, short_config = _source_and_config(tmp_path / "short", short_payload, t_final=0.2, num_times=3)
    short_report = run_paper_iii_local_science_pilot(short_config)
    assert short_report["paper_iii_science_benchmark"] is False
    assert any("short of t_final" in violation for violation in _strict_validation(short_report)["violations"])

    no_physical = _valid_strict_hh_payload()
    for row in no_physical["trajectory"]:
        row["trajectory_sample_kind"] = "repair_event"
        row["advances_time"] = False
    _source, no_physical_config = _source_and_config(tmp_path / "no_physical", no_physical)
    no_physical_report = run_paper_iii_local_science_pilot(no_physical_config)
    assert no_physical_report["paper_iii_science_benchmark"] is False
    assert any("physical state-sample" in violation for violation in _strict_validation(no_physical_report)["violations"])


def test_missing_source_artifact_emits_blocked_stub_and_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "blocked"
    config = PaperIIILocalSciencePilotConfig(
        artifact_json=tmp_path / "missing_source.json",
        output_dir=output_dir,
        t_final=0.2,
        num_times=3,
        timeout_seconds=1,
    )

    report = run_paper_iii_local_science_pilot(config)

    assert report["paper_iii_science_benchmark"] is False
    assert config.strict_output_json.exists()
    stub = json.loads(config.strict_output_json.read_text(encoding="utf-8"))
    assert stub["status"] == "blocked"
    assert "latest_phase3_source_artifact_missing_locally" in report["blockers"]
    assert config.stderr_log.exists()


def test_report_outputs_do_not_emit_raw_physical_state_markers(tmp_path: Path) -> None:
    payload = _valid_strict_hh_payload()
    payload["trajectory"][0]["amplitudes_qn_to_q0"] = [1.0, 0.0]
    _source, config = _source_and_config(tmp_path, payload)

    report = run_paper_iii_local_science_pilot(config)

    assert report["paper_iii_science_benchmark"] is False
    for path in (config.report_json, config.report_md, config.command_log_md, config.run_manifest_json):
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_MARKERS:
            assert marker not in text


def test_success_report_preserves_scope_guardrails(tmp_path: Path) -> None:
    _source, config = _source_and_config(tmp_path, _valid_strict_hh_payload())

    report = run_paper_iii_local_science_pilot(config)

    guardrails = report["scope_guardrails"]
    assert guardrails["chtc_used"] is False
    assert guardrails["optuna_used"] is False
    assert guardrails["fresh_adapt_run"] is False
    assert guardrails["realtime_or_controller_route_changed"] is False
    assert guardrails["adapt_static_defaults_changed"] is False
    assert guardrails["mclachlan_realtime_defaults_changed"] is False
    assert guardrails["p5b_reclassified_as_science"] is False
    assert guardrails["p6a_reclassified_as_science"] is False
    assert guardrails["p7a_reclassified_as_science"] is False
