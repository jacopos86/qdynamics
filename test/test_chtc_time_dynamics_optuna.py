from __future__ import annotations

import csv
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.time_dynamics_optuna import preflight_inputs, promote_class_settings, run_task, validate_outputs


def _base_row(tmp_path: Path, *, profile: str = "generic_l2_exact_v1", validation: str = "generic_exact_v1") -> dict[str, str]:
    artifact = tmp_path / "seed.json"
    artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
    return {
        "record_id": "rec_exact",
        "validation_profile": validation,
        "family": "hubbard",
        "source_artifact_json": str(artifact),
        "artifact_json": str(artifact),
        "study_profile": profile,
        "loader_mode": "replay_family",
        "generator_family": "full_meta",
        "fallback_family": "full_meta",
        "append_pool_family": "full_meta",
        "t_final": "8.0",
        "num_times": "161",
        "exact_steps_multiplier": "2",
        "enable_drive": "true",
        "disable_drive": "false",
        "drive_A": "0.2",
        "drive_omega": "1.0",
        "drive_tbar": "1.0",
        "drive_phi": "0.0",
        "drive_pattern": "staggered",
        "drive_time_sampling": "midpoint",
        "drive_t0": "0.0",
        "n_trials": "24",
        "n_startup_trials": "8",
        "sampler_seed": "123",
        "n_jobs": "1",
        "pair": "auto",
        "spectra_detrend": "linear",
        "spectra_window": "hann",
        "max_peaks": "5",
        "max_harmonic": "3",
        "skip_spectra_pdf": "true",
        "min_completed_trials": "1",
        "require_full_horizon": "true",
    }


def test_build_exact_v1_command_is_sequential_without_storage(tmp_path: Path) -> None:
    row = _base_row(tmp_path)
    row["tuning_class"] = "fermionic"
    row["objective_weight_rk4_count"] = "0.002"
    cmd, meta = run_task.build_optuna_command(row, record_id="rec_exact", run_root=tmp_path / "run")
    assert cmd[:4] == [run_task.sys.executable, "-u", "-m", "pipelines.time_dynamics.optimization.hh_realtime_optuna"]
    assert cmd[cmd.index("--study-profile") + 1] == "generic_l2_exact_v1"
    assert "--enable-drive" in cmd
    assert cmd[cmd.index("--drive-A") + 1] == "0.2"
    assert cmd[cmd.index("--tuning-class") + 1] == "fermionic"
    assert "--class-settings-output" in cmd
    assert cmd[cmd.index("--objective-weight-rk4-count") + 1] == "0.002"
    assert "--storage" not in cmd
    assert "--n-jobs" not in cmd
    assert meta["tag"] == "chtc_td_optuna_rec_exact"


def test_build_command_forwards_integrator_policy_override(tmp_path: Path) -> None:
    row = _base_row(
        tmp_path,
        profile="strict_qpu_faithful_append_prune_recoverability_v1",
        validation="strict_qpu_faithful",
    )
    row["record_id"] = "rec_forced_rk4"
    row["tuning_class"] = "fermionic"
    row["integrator_policy_override"] = "rk4"
    cmd, _meta = run_task.build_optuna_command(
        row,
        record_id="rec_forced_rk4",
        run_root=tmp_path / "run",
    )

    assert "--integrator-policy-override" in cmd
    assert cmd[cmd.index("--integrator-policy-override") + 1] == "rk4"


def test_append_prune_profile_validates_as_generic_exact_v1(tmp_path: Path) -> None:
    row = _base_row(tmp_path, profile="append_prune_noharm_l2_v1", validation="generic_exact_v1")
    row["record_id"] = "rec_append_prune"
    cmd, _meta = run_task.build_optuna_command(
        row,
        record_id="rec_append_prune",
        run_root=tmp_path / "run",
    )
    assert cmd[cmd.index("--study-profile") + 1] == "append_prune_noharm_l2_v1"
    assert preflight_inputs.validate_profile(row) is None


def test_build_command_forwards_optional_relative_signal_gate(tmp_path: Path) -> None:
    row = _base_row(tmp_path, profile="append_prune_noharm_l2_v1", validation="generic_exact_v1")
    row["record_id"] = "rec_append_prune"
    row["invalid_max_primary_observable_mae_over_span"] = "0.55"
    cmd, _meta = run_task.build_optuna_command(
        row,
        record_id="rec_append_prune",
        run_root=tmp_path / "run",
    )
    assert "--invalid-max-primary-observable-mae-over-span" in cmd
    assert cmd[cmd.index("--invalid-max-primary-observable-mae-over-span") + 1] == "0.55"


def test_preflight_rejects_mismatched_tuning_class(tmp_path: Path) -> None:
    row = _base_row(tmp_path)
    row["tuning_class"] = "bosonic"
    result = preflight_inputs.validate_record_row(row, repo_root=tmp_path, stage=False)
    assert result["ok"] is False
    assert any("tuning_class" in error for error in result["errors"])


def test_build_strict_command_static_profile(tmp_path: Path) -> None:
    row = _base_row(
        tmp_path,
        profile="strict_qpu_faithful_recoverability_v1",
        validation="strict_qpu_faithful",
    )
    row.update({"record_id": "rec_strict", "enable_drive": "false", "drive_A": "0.0", "append_pool_family": "match_replay"})
    cmd, _meta = run_task.build_optuna_command(row, record_id="rec_strict", run_root=tmp_path / "run")
    assert cmd[cmd.index("--study-profile") + 1] == "strict_qpu_faithful_recoverability_v1"
    assert "--enable-drive" not in cmd
    assert "--disable-drive" not in cmd
    assert cmd[cmd.index("--append-pool-family") + 1] == "match_replay"
    assert "--n-jobs" not in cmd


def test_build_strict_append_prune_class_policy_command(tmp_path: Path) -> None:
    row = _base_row(
        tmp_path,
        profile="strict_qpu_faithful_append_prune_recoverability_v1",
        validation="strict_qpu_faithful",
    )
    row.update(
        {
            "record_id": "rec_strict_append_prune",
            "enable_drive": "false",
            "drive_A": "0.0",
            "append_pool_family": "match_replay",
            "tuning_class": "fermionic",
            "objective_weight_runtime_count": "0.004",
            "objective_weight_append_count": "0.08",
            "objective_weight_prune_count": "0.05",
        }
    )

    cmd, _meta = run_task.build_optuna_command(
        row,
        record_id="rec_strict_append_prune",
        run_root=tmp_path / "run",
    )

    assert cmd[cmd.index("--study-profile") + 1] == "strict_qpu_faithful_append_prune_recoverability_v1"
    assert cmd[cmd.index("--tuning-class") + 1] == "fermionic"
    assert preflight_inputs.validate_profile(row) is None


def test_build_aggressive_append_prune_profile_command(tmp_path: Path) -> None:
    row = _base_row(
        tmp_path,
        profile="strict_qpu_faithful_append_prune_aggressive_v1",
        validation="strict_qpu_faithful",
    )
    row.update(
        {
            "record_id": "rec_strict_append_prune_aggressive",
            "family": "hh",
            "enable_drive": "false",
            "drive_A": "0.0",
            "append_pool_family": "full_meta",
            "tuning_class": "hybrid",
            "objective_weight_runtime_count": "0.002",
            "objective_weight_append_count": "0.03",
            "objective_weight_prune_count": "1.0",
        }
    )

    cmd, _meta = run_task.build_optuna_command(
        row,
        record_id="rec_strict_append_prune_aggressive",
        run_root=tmp_path / "run",
    )

    assert cmd[cmd.index("--study-profile") + 1] == "strict_qpu_faithful_append_prune_aggressive_v1"
    assert cmd[cmd.index("--objective-weight-prune-count") + 1] == "1.0"
    assert preflight_inputs.validate_profile(row) is None


def test_strict_append_prune_class_policy_requires_tuning_class(tmp_path: Path) -> None:
    row = _base_row(
        tmp_path,
        profile="strict_qpu_faithful_append_prune_recoverability_v1",
        validation="strict_qpu_faithful",
    )
    row.update({"record_id": "rec_strict_append_prune", "enable_drive": "false", "drive_A": "0.0"})

    with pytest.raises(ValueError, match="tuning_class"):
        run_task.build_optuna_command(row, record_id="rec_strict_append_prune", run_root=tmp_path / "run")
    result = preflight_inputs.validate_record_row(row, repo_root=tmp_path, stage=False)
    assert result["ok"] is False
    assert any("tuning_class" in error for error in result["errors"])


def test_build_strict_command_rejects_exact_objective_weights(tmp_path: Path) -> None:
    row = _base_row(
        tmp_path,
        profile="strict_qpu_faithful_append_prune_recoverability_v1",
        validation="strict_qpu_faithful",
    )
    row.update({"record_id": "rec_strict", "enable_drive": "false", "drive_A": "0.0", "tuning_class": "fermionic"})
    row["objective_weight_energy_mae"] = "1.0"

    with pytest.raises(ValueError, match="exact-reference objective"):
        run_task.build_optuna_command(row, record_id="rec_strict", run_root=tmp_path / "run")
    result = preflight_inputs.validate_record_row(row, repo_root=tmp_path, stage=False)
    assert result["ok"] is False
    assert any("objective_weight_energy_mae" in error for error in result["errors"])


def test_build_strict_command_rejects_relative_signal_gate(tmp_path: Path) -> None:
    row = _base_row(
        tmp_path,
        profile="strict_qpu_faithful_recoverability_v1",
        validation="strict_qpu_faithful",
    )
    row.update({"record_id": "rec_strict", "enable_drive": "false", "drive_A": "0.0"})
    row["invalid_max_primary_observable_mae_over_span"] = "0.5"
    with pytest.raises(ValueError, match="diagnostic exact-v1 only"):
        run_task.build_optuna_command(row, record_id="rec_strict", run_root=tmp_path / "run")


def test_reject_n_jobs_not_one(tmp_path: Path) -> None:
    row = _base_row(tmp_path)
    row["n_jobs"] = "2"
    with pytest.raises(ValueError, match="n_jobs"):
        run_task.build_optuna_command(row, record_id="rec_exact", run_root=tmp_path / "run")


def test_reject_invalid_drive_time_sampling(tmp_path: Path) -> None:
    row = _base_row(tmp_path)
    row["drive_time_sampling"] = "continuous"
    with pytest.raises(ValueError, match="drive_time_sampling"):
        run_task.build_optuna_command(row, record_id="rec_exact", run_root=tmp_path / "run")


def test_preflight_stages_seed_artifact(tmp_path: Path) -> None:
    repo = tmp_path
    src = repo / "source.json"
    src.write_text(json.dumps({"seed": 1}), encoding="utf-8")
    records = repo / "records.tsv"
    record_list = repo / "queue.txt"
    artifact_rel = "chtc/time_dynamics_optuna/input/seed_artifacts/preflight_rec.json"
    fields = [
        "record_id", "validation_profile", "family", "source_artifact_json", "artifact_json", "study_profile",
        "enable_drive", "disable_drive", "n_jobs", "loader_mode", "generator_family", "fallback_family",
    ]
    with records.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "record_id": "preflight_rec",
            "validation_profile": "generic_exact_v1",
            "family": "hubbard",
            "source_artifact_json": str(src),
            "artifact_json": artifact_rel,
            "study_profile": "generic_l2_exact_v1",
            "enable_drive": "true",
            "disable_drive": "false",
            "n_jobs": "1",
            "loader_mode": "replay_family",
            "generator_family": "full_meta",
            "fallback_family": "full_meta",
        })
    record_list.write_text("preflight_rec\n", encoding="utf-8")
    report = preflight_inputs.preflight_records(records_path=records, record_list=record_list, repo_root=repo)
    staged = repo / artifact_rel
    assert report["ok"] is True
    assert staged.exists()
    assert json.loads(staged.read_text()) == {"seed": 1}
    assert (records.parent / "preflight_report.json").exists()


def test_preflight_rejects_invalid_optional_relative_signal_gate(tmp_path: Path) -> None:
    row = _base_row(tmp_path)
    row["invalid_max_primary_observable_mae_over_span"] = "not-a-float"
    result = preflight_inputs.validate_record_row(row, repo_root=tmp_path, stage=False)
    assert result["ok"] is False
    assert any("invalid_max_primary_observable_mae_over_span" in error for error in result["errors"])


def test_preflight_rejects_relative_signal_gate_for_strict_profile(tmp_path: Path) -> None:
    row = _base_row(
        tmp_path,
        profile="strict_qpu_faithful_recoverability_v1",
        validation="strict_qpu_faithful",
    )
    row["invalid_max_primary_observable_mae_over_span"] = "0.5"
    result = preflight_inputs.validate_record_row(row, repo_root=tmp_path, stage=False)
    assert result["ok"] is False
    assert any("strict/QPU-faithful" in error for error in result["errors"])


def _all_algorithm_smoke_rows() -> list[dict[str, str]]:
    records = REPO_ROOT / "chtc/time_dynamics_optuna/input/paper_ii_all_algorithm_class_calibration_v1_smoke_records.tsv"
    with records.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def test_all_algorithm_class_calibration_smoke_preflight_is_candidate_only() -> None:
    records = REPO_ROOT / "chtc/time_dynamics_optuna/input/paper_ii_all_algorithm_class_calibration_v1_smoke_records.tsv"
    record_ids = REPO_ROOT / "chtc/time_dynamics_optuna/input/paper_ii_all_algorithm_class_calibration_v1_smoke_record_ids.txt"

    report = preflight_inputs.preflight_records(
        records_path=records,
        record_list=record_ids,
        repo_root=REPO_ROOT,
        stage=False,
        clean_staged=False,
        write_report=False,
    )

    assert report["ok"] is True, report["failed_records"]
    assert report["record_count"] == 24
    assert report["clean_staged"] is False
    assert all(item["kind"] == "benchmark" for item in report["records"])
    assert all(item["candidate_only_not_promoted"] is True for item in report["records"])
    assert all(item["staged_artifact_exists"] is False for item in report["records"])


def test_all_algorithm_smoke_runner_falls_back_when_arg2_records_path_is_missing(tmp_path: Path) -> None:
    script_target = tmp_path / "chtc" / "time_dynamics_optuna" / "run_task.sh"
    script_target.parent.mkdir(parents=True)
    script_target.write_text(
        (REPO_ROOT / "chtc" / "time_dynamics_optuna" / "run_task.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    script_target.chmod(script_target.stat().st_mode | stat.S_IXUSR)
    input_dir = script_target.parent / "input"
    input_dir.mkdir()
    default_records = input_dir / "records.tsv"
    default_records.write_text("record_id\tkind\nnot_the_smoke_record\tbenchmark\n", encoding="utf-8")
    smoke_records = input_dir / "paper_ii_all_algorithm_class_calibration_v1_smoke_records.tsv"
    smoke_record_id = (
        "paper_ii_all_algorithm_class_calibration_v1_smoke__controller-full__"
        "table1_bose_hubbard_snake_A0p2_t8_dt321_seedtracks_v2"
    )
    smoke_rel = "chtc/time_dynamics_optuna/input/paper_ii_all_algorithm_class_calibration_v1_smoke_records.tsv"
    smoke_records.write_text(f"record_id\tkind\n{smoke_record_id}\tbenchmark\n", encoding="utf-8")

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    fake_python_log = tmp_path / "fake_python_args.log"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "{ printf 'argv:'; for arg in \"$@\"; do printf ' [%s]' \"$arg\"; done; printf '\\n'; } >> \"$FAKE_PYTHON_LOG\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_PYTHON_LOG"] = str(fake_python_log)

    result = subprocess.run(
        ["bash", "chtc/time_dynamics_optuna/run_task.sh", smoke_record_id],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "JOB ARGS argc=1" in result.stdout
    assert f"records_path={smoke_rel}" in result.stdout
    assert "records_path_source=paper_ii_all_algorithm_smoke_fallback" in result.stdout
    fake_invocations = fake_python_log.read_text(encoding="utf-8")
    assert f"[--records] [{smoke_rel}]" in fake_invocations


def test_all_algorithm_class_calibration_records_dispatch_to_generic_benchmark(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GENERIC_TD_QISKIT_DYNAMICS_MODE", raising=False)
    monkeypatch.delenv("GENERIC_TD_QISKIT_QUBIT_CAP", raising=False)
    rows = _all_algorithm_smoke_rows()
    fixed = next(row for row in rows if row["algorithm_id"] == "dyn_fixed_mclachlan")
    avqds = next(row for row in rows if row["algorithm_id"] == "dyn_avqds")

    fixed_cmd, fixed_meta = run_task.build_generic_benchmark_command(
        fixed,
        record_id=fixed["record_id"],
        run_root=tmp_path / "fixed",
    )
    avqds_cmd, _avqds_meta = run_task.build_generic_benchmark_command(
        avqds,
        record_id=avqds["record_id"],
        run_root=tmp_path / "avqds",
    )

    assert fixed_cmd[:4] == [run_task.sys.executable, "-u", "-m", "pipelines.time_dynamics.tables.generic_dynamics_benchmark"]
    assert fixed_cmd[fixed_cmd.index("--case-manifest") + 1].endswith(
        "paper_ii_all_algorithm_class_calibration_v1_cases.json"
    )
    assert fixed_cmd[fixed_cmd.index("--class-settings-manifest") + 1].endswith(
        "paper_ii_all_algorithm_class_settings_candidate_lock_v1.json"
    )
    assert "--require-locked-class-settings" not in fixed_cmd
    assert fixed_cmd[fixed_cmd.index("--qiskit-dynamics-mode") + 1] == "parity_required"
    assert fixed_cmd[fixed_cmd.index("--qiskit-qubit-cap") + 1] == "none"
    assert "--qiskit-dynamics-mode" not in avqds_cmd
    assert fixed_meta["kind"] == "benchmark"
    assert fixed_meta["algorithm_id"] == "dyn_fixed_mclachlan"


def test_preflight_prunes_stale_seed_artifacts_for_selected_queue(tmp_path: Path) -> None:
    repo = tmp_path
    seed_dir = repo / "chtc" / "time_dynamics_optuna" / "input" / "seed_artifacts"
    seed_dir.mkdir(parents=True)
    keep_src = repo / "keep_source.json"
    skip_src = repo / "skip_source.json"
    keep_src.write_text(json.dumps({"seed": "keep"}), encoding="utf-8")
    skip_src.write_text(json.dumps({"seed": "skip"}), encoding="utf-8")
    stale = seed_dir / "skip_rec.json"
    stale.write_text(json.dumps({"stale": True}), encoding="utf-8")
    records = repo / "records.tsv"
    record_list = repo / "queue.txt"
    fields = [
        "record_id", "validation_profile", "family", "source_artifact_json", "artifact_json", "study_profile",
        "enable_drive", "disable_drive", "n_jobs", "loader_mode", "generator_family", "fallback_family",
    ]
    rows = [
        {
            "record_id": "keep_rec",
            "validation_profile": "generic_exact_v1",
            "family": "hubbard",
            "source_artifact_json": str(keep_src),
            "artifact_json": "chtc/time_dynamics_optuna/input/seed_artifacts/keep_rec.json",
            "study_profile": "generic_l2_exact_v1",
            "enable_drive": "true",
            "disable_drive": "false",
            "n_jobs": "1",
            "loader_mode": "replay_family",
            "generator_family": "full_meta",
            "fallback_family": "full_meta",
        },
        {
            "record_id": "skip_rec",
            "validation_profile": "generic_exact_v1",
            "family": "hubbard",
            "source_artifact_json": str(skip_src),
            "artifact_json": "chtc/time_dynamics_optuna/input/seed_artifacts/skip_rec.json",
            "study_profile": "generic_l2_exact_v1",
            "enable_drive": "true",
            "disable_drive": "false",
            "n_jobs": "1",
            "loader_mode": "replay_family",
            "generator_family": "full_meta",
            "fallback_family": "full_meta",
        },
    ]
    with records.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    record_list.write_text("keep_rec\n", encoding="utf-8")
    report = preflight_inputs.preflight_records(records_path=records, record_list=record_list, repo_root=repo)
    assert report["ok"] is True
    assert report["clean_staged"] is True
    assert "chtc/time_dynamics_optuna/input/seed_artifacts/skip_rec.json" in report["removed_stale_artifacts"]
    assert (seed_dir / "keep_rec.json").exists()
    assert not stale.exists()


def _write_common_record(record_dir: Path, *, validation_profile: str, num_times: int = 3, return_code: int = 0) -> None:
    (record_dir / "run" / "trials" / "trial_0001").mkdir(parents=True)
    record = {
        "record_id": record_dir.name,
        "validation_profile": validation_profile,
        "min_completed_trials": "1",
        "require_full_horizon": "true",
        "num_times": str(num_times),
    }
    (record_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")
    (record_dir / "chtc_status.json").write_text(json.dumps({"record_id": record_dir.name, "return_code": return_code}), encoding="utf-8")
    (record_dir / "task_result.json").write_text(json.dumps({"record_id": record_dir.name, "return_code": return_code}), encoding="utf-8")
    (record_dir / "command.sh").write_text("python -m pipelines.time_dynamics.optimization.hh_realtime_optuna\n", encoding="utf-8")
    (record_dir / "run" / "progress.json").write_text(json.dumps({"done": True}), encoding="utf-8")


def _write_exact_summary(
    record_dir: Path,
    *,
    output_path_value: str | None = None,
    trajectory: list[dict[str, object]] | None = None,
    metrics_extra: dict[str, object] | None = None,
) -> Path:
    trial_dir = record_dir / "run" / "trials" / "trial_0001"
    output = trial_dir / "result.json"
    spectra = trial_dir / "spectra.json"
    output.write_text(
        json.dumps({"trajectory": trajectory or [{"time": 0}, {"time": 1}, {"time": 2}]}),
        encoding="utf-8",
    )
    spectra.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    obs_output = output_path_value or str(output)
    metrics = {
        "generic_exact_v1_family_objective": True,
        "full_horizon_gate_passed": True,
        "primary_observable_mae_over_exact_span": 0.2,
        "mean_abs_energy_total_error": 0.3,
        "mean_abs_site_occupations_error": 0.4,
        "min_fidelity_exact": 0.9,
        "final_runtime_parameter_count": 7,
    }
    metrics.update(metrics_extra or {})
    summary = {
        "pipeline": "hh_realtime_optuna_v1",
        "n_trials_requested": 1,
        "feasible_trial_count": 1,
        "best_objective_trial": {"trial_number": 1, "objective": 0.1, "metrics": {"objective": 0.1}},
        "observations": [{
            "trial_number": 1,
            "status": "completed",
            "objective": 0.1,
            "output_json": obs_output,
            "spectra_json": str(spectra),
            "metrics": metrics,
        }],
    }
    (record_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return output


def test_exact_v1_synthetic_summary_validates(tmp_path: Path) -> None:
    record_dir = tmp_path / "exact_rec"
    _write_common_record(record_dir, validation_profile="generic_exact_v1")
    _write_exact_summary(record_dir)
    result = validate_outputs.validate_record_dir(record_dir)
    assert result.ok, result.errors
    assert result.best_trial_number == 1


def test_best_observation_join_and_path_remap(tmp_path: Path) -> None:
    root = tmp_path / "fetched"
    record_dir = root / "remap_rec"
    _write_common_record(record_dir, validation_profile="generic_exact_v1")
    fake_execute_path = "/scratch/execute/raw_outputs/remap_rec/run/trials/trial_0001/result.json"
    output = _write_exact_summary(record_dir, output_path_value=fake_execute_path)
    summary = json.loads((record_dir / "summary.json").read_text())
    best = validate_outputs.find_best_observation(summary)
    assert best is not None
    assert best["output_json"] == fake_execute_path
    assert validate_outputs.resolve_fetched_path(fake_execute_path, record_dir) == output
    result = validate_outputs.validate_record_dir(record_dir)
    assert result.ok, result.errors


def test_exact_validation_accepts_controller_declared_stable_early_stop_short_trajectory(tmp_path: Path) -> None:
    record_dir = tmp_path / "stable_stop_rec"
    _write_common_record(record_dir, validation_profile="generic_exact_v1", num_times=5)
    _write_exact_summary(
        record_dir,
        trajectory=[{"time": 0.0}, {"time": 0.5}, {"time": 1.0}],
        metrics_extra={
            "full_horizon_gate_passed": True,
            "full_horizon_successful_early_stop": True,
            "full_horizon_completion_kind": "stable_early_stop",
            "full_horizon_early_stop_reason": "progress_observables_stable:site_span=0.0001",
        },
    )
    result = validate_outputs.validate_record_dir(record_dir)
    assert result.ok, result.errors


def test_exact_validation_still_rejects_unmarked_short_trajectory(tmp_path: Path) -> None:
    record_dir = tmp_path / "short_bad_rec"
    _write_common_record(record_dir, validation_profile="generic_exact_v1", num_times=5)
    _write_exact_summary(record_dir, trajectory=[{"time": 0.0}, {"time": 1.0}])
    result = validate_outputs.validate_record_dir(record_dir)
    assert not result.ok
    assert any("trajectory row count 2 != num_times 5" in error for error in result.errors)


def _write_strict_summary(
    record_dir: Path,
    *,
    exact_feedback: bool = False,
    metrics_extra: dict[str, object] | None = None,
) -> None:
    trial_dir = record_dir / "run" / "trials" / "trial_0001"
    output = trial_dir / "result.json"
    spectra = trial_dir / "spectra.json"
    output.write_text(json.dumps({
        "trajectory": [{"time": 0}, {"time": 1}, {"time": 2}],
        "diagnostic_reference": {"feeds_controller_decisions": False, "diagnostic_reference_mode": "benchmark_exact"},
    }), encoding="utf-8")
    spectra.write_text(json.dumps({"metadata": {"analysis_skipped": True}}), encoding="utf-8")
    (trial_dir / "input_tokens.json").write_text(json.dumps({"tokens": [
        "--artifact-json", "seed.json",
        "--checkpoint-controller-mode", "oracle_v1",
        "--checkpoint-controller-reference-mode", "off",
        "--checkpoint-controller-exact-input-mode", "off",
        "--checkpoint-controller-strict-qpu-faithful",
        "--checkpoint-controller-noise-mode", "ideal",
    ]}), encoding="utf-8")
    metrics = {
        "strict_qpu_faithful": True,
        "qpu_faithful_decisions_passed": True,
        "strict_decision_contract_passed": True,
        "strict_fail_closed": False,
        "exact_decision_checkpoints": 0,
        "reference_enabled": False,
        "reference_mode": "off",
        "decision_noise_mode": "ideal",
        "non_ideal_decision_noise_count": 0,
        "full_horizon_gate_passed": True,
        "final_runtime_parameter_count": 5,
    }
    metrics.update(metrics_extra or {})
    if exact_feedback:
        metrics["mean_abs_energy_total_error"] = 0.1
    best_metrics = {"objective": 0.2}
    if exact_feedback:
        best_metrics["mean_abs_energy_total_error"] = 0.1
    summary = {
        "pipeline": "hh_realtime_optuna_v1",
        "n_trials_requested": 1,
        "feasible_trial_count": 1,
        "best_objective_trial": {"trial_number": 1, "objective": 0.2, "metrics": best_metrics},
        "observations": [{
            "trial_number": 1,
            "status": "completed",
            "objective": 0.2,
            "output_json": str(output),
            "spectra_json": str(spectra),
            "metrics": metrics,
        }],
    }
    (record_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_strict_synthetic_summary_validates_with_diagnostic_side_channel(tmp_path: Path) -> None:
    record_dir = tmp_path / "strict_rec"
    _write_common_record(record_dir, validation_profile="strict_qpu_faithful")
    _write_strict_summary(record_dir)
    result = validate_outputs.validate_record_dir(record_dir)
    assert result.ok, result.errors


def test_strict_summary_with_exact_feedback_metrics_fails(tmp_path: Path) -> None:
    record_dir = tmp_path / "strict_bad"
    _write_common_record(record_dir, validation_profile="strict_qpu_faithful")
    _write_strict_summary(record_dir, exact_feedback=True)
    result = validate_outputs.validate_record_dir(record_dir)
    assert not result.ok
    assert any("exact-feedback" in error for error in result.errors)


def _mark_strict_append_prune_record(record_dir: Path, *, tuning_class: str = "fermionic") -> None:
    record = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
    record.update({
        "study_profile": "strict_qpu_faithful_append_prune_recoverability_v1",
        "tuning_class": tuning_class,
    })
    (record_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")
    summary = json.loads((record_dir / "summary.json").read_text(encoding="utf-8"))
    summary["tuning_class"] = tuning_class
    summary["class_tuning_provenance"] = {"tuning_class": tuning_class}
    (record_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _write_class_settings_candidate(record_dir: Path, *, tuning_class: str = "fermionic") -> None:
    (record_dir / "run").mkdir(parents=True, exist_ok=True)
    candidate = {
        "schema": "dynamics_class_settings_candidate_v1",
        "tuning_class": tuning_class,
        "algorithm_id": "dyn_controller_full",
        "settings_kind": "controller",
        "strict_online_feedback_exact_free": True,
        "settings_payload": {
            "checkpoint_controller_mode": "observable_v1",
            "checkpoint_controller_exact_input_mode": "off",
            "miss_threshold": 0.05,
            "prune_mode": "schur_projected_shadow_v1",
        },
    }
    (record_dir / "run" / "class_settings_candidate.json").write_text(json.dumps(candidate), encoding="utf-8")


def test_strict_append_prune_validation_requires_candidate_and_telemetry(tmp_path: Path) -> None:
    record_dir = tmp_path / "strict_append_prune_rec"
    _write_common_record(record_dir, validation_profile="strict_qpu_faithful")
    _write_strict_summary(
        record_dir,
        metrics_extra={
            "append_opportunity_count": 1,
            "proposed_append_count": 1,
            "prune_opportunity_count": 1,
            "prune_candidate_checkpoint_count": 1,
            "prune_candidate_count": 2,
        },
    )
    _mark_strict_append_prune_record(record_dir)

    missing = validate_outputs.validate_record_dir(record_dir)
    assert not missing.ok
    assert any("class-policy candidate missing" in error for error in missing.errors)

    _write_class_settings_candidate(record_dir)
    result = validate_outputs.validate_record_dir(record_dir)
    assert result.ok, result.errors


def test_strict_append_prune_validation_rejects_exact_candidate_payload(tmp_path: Path) -> None:
    record_dir = tmp_path / "strict_append_prune_bad_candidate"
    _write_common_record(record_dir, validation_profile="strict_qpu_faithful")
    _write_strict_summary(
        record_dir,
        metrics_extra={
            "append_opportunity_count": 1,
            "proposed_append_count": 1,
            "prune_opportunity_count": 1,
            "prune_candidate_checkpoint_count": 1,
            "prune_candidate_count": 2,
        },
    )
    _mark_strict_append_prune_record(record_dir)
    _write_class_settings_candidate(record_dir)
    candidate_path = record_dir / "run" / "class_settings_candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["settings_payload"]["checkpoint_controller_mode"] = "exact_v1"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = validate_outputs.validate_record_dir(record_dir)
    assert not result.ok
    assert any("exact-assisted value" in error for error in result.errors)


def test_promote_class_settings_requires_exact_three_strict_candidates(tmp_path: Path) -> None:
    paths = []
    for tuning_class in ("fermionic", "bosonic", "hybrid"):
        path = tmp_path / f"{tuning_class}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "dynamics_class_settings_candidate_v1",
                    "family": tuning_class,
                    "tuning_class": tuning_class,
                    "algorithm_id": "dyn_controller_full",
                    "settings_kind": "controller",
                    "settings_source": "unit_optuna",
                    "settings_payload": {
                        "checkpoint_controller_mode": "observable_v1",
                        "checkpoint_controller_exact_input_mode": "off",
                        "miss_threshold": 0.05,
                    },
                    "selected_trial_number": 1,
                    "strict_online_feedback_exact_free": True,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    output = tmp_path / "locked.json"

    manifest = promote_class_settings.promote(paths, output=output, note="unit")

    assert output.exists()
    assert len(manifest["settings"]) == 3
    assert manifest["require_canonical_controller_classes"] is True
    with pytest.raises(ValueError, match="exactly one locked"):
        promote_class_settings.promote(paths[:2], output=tmp_path / "missing.json")


def test_validate_outputs_requires_expected_records(tmp_path: Path) -> None:
    root = tmp_path / "fetched"
    present = root / "present_rec"
    _write_common_record(present, validation_profile="generic_exact_v1")
    _write_exact_summary(present)
    record_list = tmp_path / "expected.txt"
    record_list.write_text("present_rec\nmissing_rec\n", encoding="utf-8")
    report = validate_outputs.validate_outputs(root, record_list=record_list, write_report=False)
    assert report["ok"] is False
    assert report["record_count"] == 2
    assert "missing_rec" in report["failed_records"]


def test_validate_record_requires_task_result_and_command(tmp_path: Path) -> None:
    record_dir = tmp_path / "missing_contract_files"
    _write_common_record(record_dir, validation_profile="generic_exact_v1")
    _write_exact_summary(record_dir)
    (record_dir / "task_result.json").unlink()
    (record_dir / "command.sh").unlink()
    result = validate_outputs.validate_record_dir(record_dir)
    assert not result.ok
    assert any("task_result.json" in error for error in result.errors)
    assert any("command.sh" in error for error in result.errors)


def test_validate_record_rejects_stale_record_id_payload(tmp_path: Path) -> None:
    record_dir = tmp_path / "fresh_rec"
    _write_common_record(record_dir, validation_profile="generic_exact_v1")
    _write_exact_summary(record_dir)
    status = json.loads((record_dir / "chtc_status.json").read_text(encoding="utf-8"))
    status["record_id"] = "stale_rec"
    (record_dir / "chtc_status.json").write_text(json.dumps(status), encoding="utf-8")
    result = validate_outputs.validate_record_dir(record_dir)
    assert not result.ok
    assert any("stale_rec" in error for error in result.errors)


def test_strict_tokens_accept_raw_list_schema(tmp_path: Path) -> None:
    record_dir = tmp_path / "strict_raw_tokens"
    _write_common_record(record_dir, validation_profile="strict_qpu_faithful")
    _write_strict_summary(record_dir)
    token_path = record_dir / "run" / "trials" / "trial_0001" / "input_tokens.json"
    tokens = json.loads(token_path.read_text(encoding="utf-8"))["tokens"]
    token_path.write_text(json.dumps(tokens), encoding="utf-8")
    result = validate_outputs.validate_record_dir(record_dir)
    assert result.ok, result.errors
