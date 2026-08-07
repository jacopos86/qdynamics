from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from chtc.phase3_optuna import full_noise_monitoring as monitor


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _compatible_preflight_report(selected_source: Path) -> dict:
    return {
        "schema": "adapt_selected_logical_pool_match_report_v1",
        "mode": "family_closure_fail_closed",
        "source_json": str(selected_source),
        "source_exists": True,
        "transfer_mode": "exact_match_v1",
        "applied": True,
        "fallback_to_full_pool": False,
        "selected_record_count": 1,
        "active_pool_label_count": 5,
        "matched_count": 1,
        "selected_label_family_samples": [{"label": "compatible_label", "family_ids": ["compatible_family"]}],
        "missing_label_family_samples": [],
        "active_pool_label_samples": ["compatible_label"],
        "match_method_counts": {"exact_label": 1},
        "status": "pass",
        "reason": None,
    }


def _no_match_preflight_report(selected_source: Path) -> dict:
    return {
        **_compatible_preflight_report(selected_source),
        "applied": False,
        "matched_count": 0,
        "match_method_counts": {},
        "missing_label_family_samples": [{"label": "not_in_this_pool"}],
        "status": "fail",
        "reason": "no_matches",
        "semantic_reason": "selected_logical_no_pool_matches",
    }


def _valid_depth1_payload(selected_source: Path) -> dict:
    return {
        "adapt_vqe": {
            "ansatz_depth": 1,
            "selected_logical_filter_meta": {
                "applied": True,
                "fallback_to_full_pool": False,
                "source_json": str(selected_source),
            },
            "oracle_backend_info": {
                "details": {
                    "synthetic_coherent": {"inserted_count": 2},
                }
            },
            "continuation": {
                "oracle_gradient_config": {
                    "value_noise": {
                        "model": "gaussian_iid_v1",
                        "sigma0_abs": 0.001,
                        "N_eff": 400,
                        "physical_shots_unchanged": True,
                        "fixed_gate_error_reduction_claimed": False,
                    },
                    "synthetic_depolarizing": {
                        "one_qubit_error": 1e-8,
                        "two_qubit_error": 1e-7,
                        "one_qubit_gates": ["x", "sx", "rx", "ry", "h"],
                        "two_qubit_gates": ["cx", "cz", "ecr"],
                    },
                    "synthetic_coherent": {
                        "one_qubit_angle_std": 2e-4,
                        "two_qubit_angle_std": 6e-4,
                    },
                }
            },
        }
    }


def test_build_depth1_adapt_smoke_command_uses_required_smoke_settings(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    selected = tmp_path / "hh_L2_nph2.selected_logical.json"

    command = monitor._build_depth1_adapt_smoke_command(root, selected_source=selected)

    assert command[:4] == [sys.executable, "-u", "-m", "pipelines.static_adapt.adapt_pipeline"]
    assert _arg_value(command, "--problem") == "hh"
    assert _arg_value(command, "--L") == "2"
    assert _arg_value(command, "--u") == "1.25"
    assert _arg_value(command, "--g-ep") == "0.3535533905932738"
    assert _arg_value(command, "--n-ph-max") == "2"
    assert _arg_value(command, "--adapt-pool") == "math_md_full_meta_v1"
    assert _arg_value(command, "--adapt-continuation-mode") == "phase3_v1"
    assert _arg_value(command, "--adapt-segment-target-depth") == "1"
    assert _arg_value(command, "--adapt-segment-max-new-admissions") == "1"
    assert _arg_value(command, "--adapt-current-json").endswith("json/current.json")
    assert _arg_value(command, "--output-json").endswith("json/result.json")
    assert _arg_value(command, "--adapt-selected-logical-source-json") == str(selected)
    assert _arg_value(command, "--adapt-selected-logical-mode") == "family_closure_fail_closed"
    assert _arg_value(command, "--phase3-oracle-synthetic-depolarizing-1q-gates") == "x,sx,rx,ry,h"
    assert _arg_value(command, "--phase3-oracle-synthetic-depolarizing-2q-gates") == "cx,cz,ecr"
    assert _arg_value(command, "--phase3-oracle-synthetic-coherent-1q-gates") == "x,sx,rx,ry,h"
    assert _arg_value(command, "--phase3-oracle-synthetic-coherent-2q-gates") == "cx,cz,ecr"
    assert "--skip-pdf" in command
    assert "--skip-trajectory" in command


def test_depth1_adapt_smoke_source_preflight_failure_appends_rejection_row(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "smoke"
    board = tmp_path / "scoreboard.md"
    missing_source = tmp_path / "missing.selected_logical.json"

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("ADAPT subprocess should not run when source preflight fails")

    monkeypatch.setattr(monitor, "_run_depth1_adapt_smoke_subprocess", fail_if_called)

    rc = monitor.main(
        [
            "depth1-adapt-smoke",
            "--output-root",
            str(root),
            "--selected-logical-source-json",
            str(missing_source),
            "--scoreboard",
            str(board),
            "--iteration-id",
            "unit_depth1_missing_source",
            "--attributed-change",
            "source_lock_pool_match_preflight_v1",
            "--case-id",
            "unit_case",
            "--expected-validity",
            "valid",
        ]
    )

    assert rc == 4
    evidence = json.loads((root / "source_lock_preflight_evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "fail"
    assert evidence["reason"] == "source_json_missing"
    stdout = (root / "stdout.log").read_text(encoding="utf-8")
    assert "AI_LOG" in stdout
    assert "depth1_adapt_smoke_source_lock_preflight_failed" in stdout
    board_text = board.read_text(encoding="utf-8")
    assert "unit_depth1_missing_source" in board_text
    assert "| valid | rejected |" in board_text
    assert "source_lock_preflight_failed:source_json_missing" in board_text
    assert "fail(applied=false; source_exists=false; fallback_to_full_pool=false)" in board_text


def test_depth1_adapt_smoke_fake_success_appends_valid_row(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "smoke"
    board = tmp_path / "scoreboard.md"
    selected = _write_json(tmp_path / "selected.json", {"records": [{"generator_id": "g0"}], "record_count": 1})
    calls: list[list[str]] = []

    def fake_run(command, output_root, *, timeout_s=None):  # noqa: ANN001
        calls.append(list(command))
        out = Path(output_root)
        _write_json(out / "json" / "current.json", _valid_depth1_payload(selected))
        _write_json(out / "json" / "result.json", _valid_depth1_payload(selected))
        (out / "stdout.log").write_text(
            "AI_LOG "
            + json.dumps({"event": "hardcoded_adapt_main_start", "ts_utc": "2026-06-10T00:00:00+00:00", "depth": 0})
            + "\n",
            encoding="utf-8",
        )
        (out / "stderr.log").write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(monitor, "_run_depth1_adapt_smoke_subprocess", fake_run)
    monkeypatch.setattr(monitor, "_depth1_source_lock_pool_match_report", lambda selected_source: _compatible_preflight_report(Path(selected_source)))
    monkeypatch.setattr(monitor, "capture_remote_runner_status", lambda: {"status": "ok"})

    rc = monitor.main(
        [
            "depth1-adapt-smoke",
            "--output-root",
            str(root),
            "--selected-logical-source-json",
            str(selected),
            "--scoreboard",
            str(board),
            "--iteration-id",
            "unit_depth1_valid",
            "--attributed-change",
            "source_lock_pool_match_preflight_v1",
            "--case-id",
            "unit_case",
            "--expected-validity",
            "valid",
            "--capture-remote-runner-status",
        ]
    )

    assert rc == 0
    assert calls
    assert _arg_value(calls[0], "--adapt-selected-logical-source-json") == str(selected)
    board_text = board.read_text(encoding="utf-8")
    assert "unit_depth1_valid" in board_text
    assert "| valid | valid |" in board_text
    assert "pass(value_noise=pass; depolarizing_split=pass; depolarizing_positive=pass; coherent_inserted_count=2)" in board_text
    assert "pass(applied=true; source_exists=true; fallback_to_full_pool=false)" in board_text
    assert "pass(stdout_ai_log_parseable=true; current_json_parseable=true; result_json_parseable=true; backend_metadata_present=true)" in board_text
    evidence = json.loads((root / "source_lock_preflight_evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "pass"
    assert evidence["matched_count"] == 1
    assert evidence["active_pool_label_count"] == 5
    assert (root / "command.sh").exists()


def test_depth1_adapt_smoke_no_match_preflight_appends_semantic_rejection(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "smoke"
    board = tmp_path / "scoreboard.md"
    selected = _write_json(tmp_path / "selected.json", {"records": [{"candidate_label": "not_in_this_pool"}], "record_count": 1})

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("ADAPT subprocess should not run when source-lock pool-match preflight has zero matches")

    monkeypatch.setattr(monitor, "_run_depth1_adapt_smoke_subprocess", fail_if_called)
    monkeypatch.setattr(monitor, "_depth1_source_lock_pool_match_report", lambda selected_source: _no_match_preflight_report(Path(selected_source)))

    rc = monitor.main(
        [
            "depth1-adapt-smoke",
            "--output-root",
            str(root),
            "--selected-logical-source-json",
            str(selected),
            "--scoreboard",
            str(board),
            "--iteration-id",
            "unit_depth1_no_match_preflight",
            "--attributed-change",
            "source_lock_pool_match_preflight_v1",
            "--case-id",
            "unit_case",
            "--expected-validity",
            "valid",
        ]
    )

    assert rc == 4
    evidence = json.loads((root / "source_lock_preflight_evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "fail"
    assert evidence["reason"] == "selected_logical_no_pool_matches"
    assert evidence["selected_record_count"] == 1
    assert evidence["active_pool_label_count"] == 5
    assert evidence["matched_count"] == 0
    board_text = board.read_text(encoding="utf-8")
    assert "unit_depth1_no_match_preflight" in board_text
    assert "source_lock_pool_match_preflight_v1" in board_text
    assert "| valid | rejected |" in board_text
    assert "source_lock_preflight_failed:selected_logical_no_pool_matches" in board_text
    assert "selected_record_count=1" in board_text
    assert "active_pool_label_count=5" in board_text
    assert "matched_count=0" in board_text
    assert "source_lock_preflight_evidence.json" in board_text
    assert "fail(applied=false; source_exists=true; fallback_to_full_pool=false; selected_record_count=1; active_pool_label_count=5; matched_count=0)" in board_text


def test_scoreboard_scaffold_is_created_once_and_rows_append(tmp_path: Path) -> None:
    board = tmp_path / "scoreboard.md"

    monitor.ensure_scoreboard(board)
    first = board.read_text(encoding="utf-8")
    assert "# Paper-I Full-Noise Monitoring Optimization Runs" in first
    assert "| timestamp_utc | iteration_id |" in first

    row = {column: "" for column in monitor.SCOREBOARD_COLUMNS}
    row.update(
        {
            "timestamp_utc": "2026-06-10T00:00:00+00:00",
            "iteration_id": "unit",
            "attributed_change": "baseline_no_change",
            "case_id": "fake_case",
            "expected_validity": "invalid",
            "observed_run_validity": "rejected",
        }
    )
    monitor.append_scoreboard_row(board, row)
    second = board.read_text(encoding="utf-8")

    assert second.startswith(first)
    assert second.count("| 2026-06-10T00:00:00+00:00 | unit |") == 1


def test_collect_artifact_detects_semantic_full_noise_rejection(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    stdout_events = [
        {
            "event": "hardcoded_adapt_main_start",
            "ts_utc": "2026-06-10T17:27:44+00:00",
            "settings": {
                "phase3_oracle_value_noise_model": "gaussian_iid_v1",
                "phase3_oracle_value_noise_sigma0_abs": 0.001,
                "phase3_oracle_value_noise_n_eff": 200,
                "physical_shots_unchanged": True,
                "fixed_gate_error_reduction_claimed": False,
                "phase3_oracle_synthetic_depolarizing_1q_error": 1e-8,
                "phase3_oracle_synthetic_depolarizing_2q_error": 1e-7,
                "phase3_oracle_synthetic_depolarizing_1q_gates": "x sx rx ry h",
                "phase3_oracle_synthetic_depolarizing_2q_gates": "cx cz ecr",
                "phase3_oracle_synthetic_coherent_1q_angle_std": 0.0002,
                "phase3_oracle_synthetic_coherent_2q_angle_std": 0.0006,
            },
        },
        {
            "event": "hardcoded_adapt_pool_built",
            "ts_utc": "2026-06-10T17:27:44.100000+00:00",
            "adapt_selected_logical_filter": {
                "applied": False,
                "fallback_to_full_pool": True,
                "source_json": "missing.selected.json",
                "fallback_reason": "malformed_source_json",
            },
        },
        {
            "event": "hardcoded_adapt_phase3_oracle_inner_eval_error",
            "ts_utc": "2026-06-10T17:27:45.250000+00:00",
            "depth": 0,
            "error_type": "RuntimeError",
            "error_repr": 'RuntimeError("Synthetic coherent overrotation was configured but inserted zero gate errors. This is not a valid full-noise circuit-channel diagnostic.")',
        },
    ]
    (root / "stdout.log").write_text(
        "\n".join("AI_LOG " + json.dumps(event, sort_keys=True) for event in stdout_events) + "\n",
        encoding="utf-8",
    )
    (root / "stderr.log").write_text("traceback tail", encoding="utf-8")

    metrics = monitor.collect_artifact_metrics(root)

    assert metrics["invalid_run_detection_latency_s"] == 1.25
    assert metrics["invalid_run_detection_depth"] == 0
    assert metrics["full_noise_active_gate"].startswith("fail(")
    assert "depolarizing_split=pass" in metrics["full_noise_active_gate"]
    assert "coherent_inserted_count=0" in metrics["full_noise_active_gate"]
    assert metrics["source_lock_active_gate"].startswith("fail(")
    assert "fallback_to_full_pool=true" in metrics["source_lock_active_gate"]
    assert metrics["json_recoverability_gate"].startswith("partial(")
    assert monitor.classify_observed_validity("invalid", metrics) == "rejected"


def test_collect_artifact_marks_current_json_without_backend_as_ambiguous(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_json(
        root / "retrieved_current.json",
        {
            "adapt_vqe": {
                "ansatz_depth": 8,
                "continuation": {
                    "oracle_gradient_config": {
                        "noise_mode": "aer_density_matrix_synthetic_coherent",
                        "execution_surface": "expectation_v1",
                        "value_noise": {
                            "enabled": True,
                            "model": "gaussian_iid_v1",
                            "sigma0_abs": 0.001,
                            "N_eff": 200.0,
                            "physical_shots_unchanged": True,
                            "fixed_gate_error_reduction_claimed": False,
                        },
                        "synthetic_depolarizing": {
                            "one_qubit_error": 1e-8,
                            "two_qubit_error": 1e-7,
                            "one_qubit_gates": ["x sx rx ry h"],
                            "two_qubit_gates": ["cx cz ecr"],
                        },
                        "synthetic_coherent": {
                            "one_qubit_angle_std": 2e-4,
                            "two_qubit_angle_std": 6e-4,
                            "one_qubit_gates": ["x sx rx ry h"],
                            "two_qubit_gates": ["cx cz ecr"],
                        },
                    }
                },
            }
        },
    )

    metrics = monitor.collect_artifact_metrics(root)
    row = monitor.build_scoreboard_row(
        metrics=metrics,
        iteration_id="unit",
        attributed_change="baseline_no_change",
        command="collect fake current",
        case_id="fake_current",
        expected_validity="invalid",
        follow_up="harden backend metadata capture",
    )

    assert metrics["invalid_run_detection_latency_s"] is None
    assert metrics["invalid_run_detection_depth"] == 8
    assert metrics["full_noise_active_gate"].startswith("fail(")
    assert "depolarizing_split=pass" in metrics["full_noise_active_gate"]
    assert metrics["source_lock_active_gate"].startswith("ambiguous(")
    assert metrics["json_recoverability_gate"].startswith("partial(")
    assert row["observed_run_validity"] == "ambiguous"
    assert "retrieved_current.json" in json.dumps(row["evidence_paths"])


def test_split_gate_backend_evidence_can_pass_full_noise_gate(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_json(
        root / "result.json",
        {
            "adapt_vqe": {
                "ansatz_depth": 1,
                "selected_logical_filter_meta": {
                    "applied": True,
                    "fallback_to_full_pool": False,
                    "source_json": str(root / "selected.json"),
                },
                "oracle_backend_info": {
                    "details": {
                        "synthetic_coherent": {"inserted_count": 2},
                    }
                },
                "continuation": {
                    "oracle_gradient_config": {
                        "value_noise": {
                            "model": "gaussian_iid_v1",
                            "sigma0_abs": 0.001,
                            "N_eff": 400,
                            "physical_shots_unchanged": True,
                            "fixed_gate_error_reduction_claimed": False,
                        },
                        "synthetic_depolarizing": {
                            "one_qubit_error": 1e-8,
                            "two_qubit_error": 1e-7,
                            "one_qubit_gates": ["x", "sx", "rx", "ry", "h"],
                            "two_qubit_gates": ["cx", "cz", "ecr"],
                        },
                        "synthetic_coherent": {
                            "one_qubit_angle_std": 2e-4,
                            "two_qubit_angle_std": 6e-4,
                        },
                    }
                },
            }
        },
    )
    (root / "selected.json").write_text("{}", encoding="utf-8")

    metrics = monitor.collect_artifact_metrics(root)

    assert metrics["full_noise_active_gate"].startswith("pass(")
    assert metrics["source_lock_active_gate"].startswith("pass(")
    assert metrics["json_recoverability_gate"].startswith("pass(")
