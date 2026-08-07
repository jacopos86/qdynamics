#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (  # noqa: E402
    DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND,
    validate_dynamics_tuning_class,
)
DEFAULT_RECORDS = Path(__file__).resolve().parent / "input" / "records.tsv"
_PIPELINE_MODULE = "pipelines.time_dynamics.optimization.hh_realtime_optuna"
_EXPECTED_VALIDATION_PROFILE = {
    "generic_l2_exact_v1": "generic_exact_v1",
    "append_prune_noharm_l2_v1": "generic_exact_v1",
    "append_prune_recoverability_l2_v1": "generic_exact_v1",
    "append_live_guard_l2_v1": "generic_exact_v1",
    "strict_qpu_faithful_recoverability_v1": "strict_qpu_faithful",
    "strict_qpu_faithful_append_prune_recoverability_v1": "strict_qpu_faithful",
    "strict_qpu_faithful_append_prune_guardrail_only_v1": "strict_qpu_faithful",
    "strict_qpu_faithful_append_prune_aggressive_v1": "strict_qpu_faithful",
    "strict_qpu_hh_recoverability_v1": "strict_qpu_hh",
}
_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}
_DRIVE_TIME_SAMPLING = {"", "left", "midpoint", "right"}
_QISKIT_SUPPORTED_GENERIC_ALGORITHMS = {
    "dyn_fixed_mclachlan",
    "dyn_product_formula_envelope",
    "dyn_qdrift",
    "dyn_fixed_pvqd",
    "dyn_adaptive_pvqd",
}
_STRICT_APPEND_PRUNE_PROFILES = {
    "strict_qpu_faithful_append_prune_recoverability_v1",
    "strict_qpu_faithful_append_prune_guardrail_only_v1",
    "strict_qpu_faithful_append_prune_aggressive_v1",
}
_STRICT_EXACT_ONLY_OPTION_KEYS = {
    "invalid_max_mean_energy_mae",
    "invalid_max_final_energy_mae",
    "invalid_min_fidelity",
    "invalid_max_mean_total_occupation_mae",
    "invalid_max_primary_observable_mae_over_span",
    "objective_weight_site_mae",
    "objective_weight_energy_mae",
    "objective_weight_energy_bias_abs",
    "objective_weight_energy_max_abs",
    "objective_weight_energy_under_response_mean",
    "objective_weight_energy_under_response_max",
    "objective_weight_fidelity_defect",
    "objective_weight_pair_mae_over_span",
    "objective_weight_epsilon_osc",
    "objective_weight_peak_omega",
    "objective_weight_pair_corr_defect",
}
_OPTION_MAP = {
    "loader_mode": "--loader-mode",
    "generator_family": "--generator-family",
    "fallback_family": "--fallback-family",
    "append_pool_family": "--append-pool-family",
    "t_final": "--t-final",
    "num_times": "--num-times",
    "exact_steps_multiplier": "--exact-steps-multiplier",
    "pair": "--pair",
    "objective_window_start": "--objective-window-start",
    "objective_window_end": "--objective-window-end",
    "spectra_detrend": "--spectra-detrend",
    "spectra_window": "--spectra-window",
    "max_peaks": "--max-peaks",
    "max_harmonic": "--max-harmonic",
    "integrator_policy_override": "--integrator-policy-override",
    "objective_weight_pair_mae_over_span": "--objective-weight-pair-mae-over-span",
    "objective_weight_epsilon_osc": "--objective-weight-epsilon-osc",
    "objective_weight_peak_omega": "--objective-weight-peak-omega",
    "objective_weight_pair_corr_defect": "--objective-weight-pair-corr-defect",
    "objective_weight_site_mae": "--objective-weight-site-mae",
    "objective_weight_energy_mae": "--objective-weight-energy-mae",
    "objective_weight_energy_bias_abs": "--objective-weight-energy-bias-abs",
    "objective_weight_energy_max_abs": "--objective-weight-energy-max-abs",
    "objective_weight_energy_under_response_mean": "--objective-weight-energy-under-response-mean",
    "objective_weight_energy_under_response_max": "--objective-weight-energy-under-response-max",
    "objective_weight_fidelity_defect": "--objective-weight-fidelity-defect",
    "objective_weight_runtime_count": "--objective-weight-runtime-count",
    "objective_weight_append_count": "--objective-weight-append-count",
    "objective_weight_prune_count": "--objective-weight-prune-count",
    "objective_weight_rk4_count": "--objective-weight-rk4-count",
    "invalid_max_mean_energy_mae": "--invalid-max-mean-energy-mae",
    "invalid_max_final_energy_mae": "--invalid-max-final-energy-mae",
    "invalid_min_fidelity": "--invalid-min-fidelity",
    "invalid_max_mean_total_occupation_mae": "--invalid-max-mean-total-occupation-mae",
    "invalid_max_primary_observable_mae_over_span": "--invalid-max-primary-observable-mae-over-span",
    "class_settings_source": "--class-settings-source",
    "class_settings_lock_json": "--class-settings-lock-json",
}
_DRIVE_OPTION_MAP = {
    "drive_A": "--drive-A",
    "drive_omega": "--drive-omega",
    "drive_tbar": "--drive-tbar",
    "drive_phi": "--drive-phi",
    "drive_pattern": "--drive-pattern",
    "drive_custom_weights": "--drive-custom-weights",
    "drive_time_sampling": "--drive-time-sampling",
    "drive_t0": "--drive-t0",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _add_option(cmd: list[str], flag: str, value: Any) -> None:
    text = _clean(value)
    if text == "" or text.lower() in {"none", "null"}:
        return
    cmd.extend([flag, text])


def _safe_tag(record_id: str) -> str:
    return "chtc_td_optuna_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id).strip("_")


def _nonzero_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return True


def load_records(path: str | Path = DEFAULT_RECORDS) -> list[dict[str, str]]:
    records_path = Path(path)
    with records_path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def find_record(rows: Sequence[Mapping[str, str]], record_id: str) -> dict[str, str]:
    matches = [dict(row) for row in rows if str(row.get("record_id", "")) == str(record_id)]
    if not matches:
        raise KeyError(f"record_id {record_id!r} not found")
    if len(matches) > 1:
        raise ValueError(f"record_id {record_id!r} appears {len(matches)} times")
    return matches[0]


def _is_generic_benchmark_row(row: Mapping[str, str]) -> bool:
    return _clean(row.get("kind")).lower() == "benchmark" and bool(_clean(row.get("algorithm_id")))


def validate_generic_benchmark_row(row: Mapping[str, str], *, record_id: str | None = None) -> None:
    rid = record_id or str(row.get("record_id", ""))
    family = _clean(row.get("family"))
    case_id = _clean(row.get("case_id"))
    algorithm_id = _clean(row.get("algorithm_id"))
    settings_kind = _clean(row.get("settings_kind"))
    if not rid:
        raise ValueError("generic benchmark row missing record_id")
    if not family or not case_id:
        raise ValueError(f"record {rid}: family and case_id are required for generic benchmark records")
    if algorithm_id not in DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND:
        known = ", ".join(DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND)
        raise ValueError(f"record {rid}: unsupported Table-I algorithm_id {algorithm_id!r}; expected one of {known}")
    expected_kind = DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND[algorithm_id]
    if settings_kind != expected_kind:
        raise ValueError(
            f"record {rid}: settings_kind {settings_kind!r} does not match algorithm_id {algorithm_id!r} expected {expected_kind!r}"
        )
    tuning_class = _clean(row.get("tuning_class"))
    if tuning_class:
        validate_dynamics_tuning_class(family=family, tuning_class=tuning_class)
    if _clean(row.get("case_manifest")) == "":
        raise ValueError(f"record {rid}: missing case_manifest")
    if _clean(row.get("class_settings_manifest")) == "":
        raise ValueError(f"record {rid}: missing class_settings_manifest")
    if parse_bool(row.get("candidate_only_not_promoted"), default=False) is not True:
        raise ValueError(f"record {rid}: all-algorithm class-calibration records must be candidate-only/not-promoted")
    if parse_bool(row.get("require_algorithm_class_settings"), default=False) is not True:
        raise ValueError(f"record {rid}: require_algorithm_class_settings must be true")
    if parse_bool(row.get("require_parity_correctness_sidecars"), default=False) is not True:
        raise ValueError(f"record {rid}: require_parity_correctness_sidecars must be true")
    if _clean(row.get("seed_track")) != "snake":
        raise ValueError(f"record {rid}: all-algorithm class-calibration records must use seed_track='snake'")
    if not _clean(row.get("same_seed_comparator_group_id")):
        raise ValueError(f"record {rid}: missing same_seed_comparator_group_id")
    if not _clean(row.get("seed_artifact_sha256")):
        raise ValueError(f"record {rid}: missing seed_artifact_sha256")


def build_generic_benchmark_command(
    row: Mapping[str, str],
    *,
    record_id: str | None = None,
    run_root: str | Path,
) -> tuple[list[str], dict[str, Any]]:
    rid = record_id or str(row.get("record_id", ""))
    validate_generic_benchmark_row(row, record_id=rid)
    run_dir = Path(run_root)
    algorithm_id = _clean(row.get("algorithm_id"))
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "pipelines.time_dynamics.tables.generic_dynamics_benchmark",
        "--run-single",
        "--family",
        _clean(row.get("family")),
        "--case-id",
        _clean(row.get("case_id")),
        "--algorithm-id",
        algorithm_id,
        "--output-dir",
        str(run_dir),
        "--case-manifest",
        _clean(row.get("case_manifest")),
        "--class-settings-manifest",
        _clean(row.get("class_settings_manifest")),
    ]
    if parse_bool(row.get("require_locked_class_settings"), default=False):
        cmd.append("--require-locked-class-settings")
    qiskit_mode = os.environ.get("GENERIC_TD_QISKIT_DYNAMICS_MODE", "").strip()
    if not qiskit_mode and algorithm_id in _QISKIT_SUPPORTED_GENERIC_ALGORITHMS:
        qiskit_mode = "parity_required"
    if qiskit_mode:
        if algorithm_id not in _QISKIT_SUPPORTED_GENERIC_ALGORITHMS:
            raise ValueError(
                f"record {rid}: Qiskit dynamics mode {qiskit_mode!r} requested for unsupported algorithm {algorithm_id!r}"
            )
        cmd.extend(["--qiskit-dynamics-mode", qiskit_mode])
        qiskit_qubit_cap = os.environ.get("GENERIC_TD_QISKIT_QUBIT_CAP", "none").strip() or "none"
        cmd.extend(["--qiskit-qubit-cap", qiskit_qubit_cap])
    metadata = {
        "record_id": rid,
        "kind": "benchmark",
        "algorithm_id": algorithm_id,
        "case_id": _clean(row.get("case_id")),
        "run_root": str(run_dir),
    }
    return cmd, metadata


def validate_row(row: Mapping[str, str], *, record_id: str | None = None) -> None:
    if _is_generic_benchmark_row(row):
        validate_generic_benchmark_row(row, record_id=record_id)
        return
    rid = record_id or str(row.get("record_id", ""))
    n_jobs = _clean(row.get("n_jobs"))
    if n_jobs not in {"", "1"}:
        raise ValueError(f"record {rid}: n_jobs must be blank or 1; got {n_jobs!r}")
    enable_drive = parse_bool(row.get("enable_drive"), default=False)
    disable_drive = parse_bool(row.get("disable_drive"), default=False)
    if enable_drive and disable_drive:
        raise ValueError(f"record {rid}: enable_drive and disable_drive cannot both be true")
    study_profile = _clean(row.get("study_profile"))
    validation_profile = _clean(row.get("validation_profile"))
    expected = _EXPECTED_VALIDATION_PROFILE.get(study_profile)
    if expected is None:
        raise ValueError(f"record {rid}: unsupported study_profile {study_profile!r}")
    if validation_profile != expected:
        raise ValueError(
            f"record {rid}: validation_profile {validation_profile!r} does not match "
            f"study_profile {study_profile!r} expected {expected!r}"
        )
    is_strict_validation = validation_profile in {"strict_qpu_faithful", "strict_qpu_hh"}
    if is_strict_validation and _clean(row.get("invalid_max_primary_observable_mae_over_span")):
        raise ValueError(
            f"record {rid}: invalid_max_primary_observable_mae_over_span is diagnostic exact-v1 only"
        )
    if is_strict_validation:
        for key in sorted(_STRICT_EXACT_ONLY_OPTION_KEYS):
            if key == "invalid_max_primary_observable_mae_over_span":
                continue
            if _nonzero_text(row.get(key, "")):
                raise ValueError(
                    f"record {rid}: {key} is exact-reference objective/gate feedback; strict records must leave it blank or zero"
                )
    if not _clean(row.get("artifact_json")):
        raise ValueError(f"record {rid}: missing artifact_json")
    family = _clean(row.get("family"))
    tuning_class = _clean(row.get("tuning_class"))
    if study_profile in _STRICT_APPEND_PRUNE_PROFILES and not tuning_class:
        raise ValueError(f"record {rid}: strict append/prune class-policy records must set tuning_class")
    if tuning_class:
        validate_dynamics_tuning_class(family=family, tuning_class=tuning_class)
    drive_time_sampling = _clean(row.get("drive_time_sampling"))
    if drive_time_sampling not in _DRIVE_TIME_SAMPLING:
        allowed = sorted(item for item in _DRIVE_TIME_SAMPLING if item)
        raise ValueError(f"record {rid}: drive_time_sampling must be blank or one of {allowed}, got {drive_time_sampling!r}")


def build_optuna_command(
    row: Mapping[str, str],
    *,
    record_id: str | None = None,
    run_root: str | Path,
) -> tuple[list[str], dict[str, Any]]:
    rid = record_id or str(row.get("record_id", ""))
    validate_row(row, record_id=rid)
    run_dir = Path(run_root)
    enable_drive = parse_bool(row.get("enable_drive"), default=False)
    disable_drive = parse_bool(row.get("disable_drive"), default=False)
    skip_spectra_pdf = parse_bool(row.get("skip_spectra_pdf"), default=True)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        _PIPELINE_MODULE,
        "--artifact-json",
        _clean(row.get("artifact_json")),
        "--study-profile",
        _clean(row.get("study_profile")),
        "--tag",
        _safe_tag(rid),
        "--output-dir",
        str(run_dir),
        "--n-trials",
        _clean(row.get("n_trials") or "24"),
        "--n-startup-trials",
        _clean(row.get("n_startup_trials") or "8"),
        "--sampler-seed",
        _clean(row.get("sampler_seed") or "7"),
    ]
    _add_option(cmd, "--tuning-class", row.get("tuning_class"))
    _add_option(cmd, "--class-settings-output", str(run_dir / "class_settings_candidate.json"))
    for key, flag in _OPTION_MAP.items():
        _add_option(cmd, flag, row.get(key))
    if parse_bool(row.get("lock_fixed_manifold"), default=False):
        cmd.append("--lock-fixed-manifold")
    if parse_bool(row.get("allow_repeats"), default=False):
        cmd.append("--allow-repeats")
    if parse_bool(row.get("no_baseline_trial"), default=False):
        cmd.append("--no-baseline-trial")
    if disable_drive:
        cmd.append("--disable-drive")
    elif enable_drive:
        cmd.append("--enable-drive")
        for key, flag in _DRIVE_OPTION_MAP.items():
            _add_option(cmd, flag, row.get(key))
        if parse_bool(row.get("drive_include_identity"), default=False):
            cmd.append("--drive-include-identity")
    if skip_spectra_pdf:
        cmd.append("--skip-spectra-pdf")
    else:
        cmd.append("--with-spectra-pdf")
    if "--storage" in cmd or "--n-jobs" in cmd:
        raise AssertionError("TD Optuna CHTC command must not pass --storage or --n-jobs")
    metadata = {
        "record_id": rid,
        "tag": _safe_tag(rid),
        "validation_profile": _clean(row.get("validation_profile")),
        "study_profile": _clean(row.get("study_profile")),
        "run_root": str(run_dir),
    }
    return cmd, metadata


def _copy_summary_if_present(run_root: Path, out_root: Path) -> Path | None:
    summary = run_root / "summary.json"
    if summary.exists():
        target = out_root / "summary.json"
        target.write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
        return target
    nested = sorted(run_root.glob("**/summary.json"))
    if nested:
        target = out_root / "summary.json"
        target.write_text(nested[0].read_text(encoding="utf-8"), encoding="utf-8")
        return target
    return None


def run_record(record_id: str, records_path: str | Path, output_root: str | Path) -> int:
    started = _now_utc()
    records_file = Path(records_path)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_root = out_root / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    command: list[str] = []
    error: str | None = None
    rc = 1
    summary_target: Path | None = None
    try:
        row = find_record(load_records(records_file), record_id)
        (out_root / "record.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if _is_generic_benchmark_row(row):
            command, _metadata = build_generic_benchmark_command(row, record_id=record_id, run_root=run_root)
        else:
            command, _metadata = build_optuna_command(row, record_id=record_id, run_root=run_root)
        (out_root / "command.sh").write_text(shlex.join(command) + "\n", encoding="utf-8")
        print("RUN", shlex.join(command), flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        rc = int(completed.returncode)
        summary_target = _copy_summary_if_present(run_root, out_root)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, file=sys.stderr, flush=True)
        rc = 2
    finished = _now_utc()
    result = {
        "record_id": record_id,
        "return_code": rc,
        "command": command,
        "run_root": str(run_root),
        "summary_json": None if summary_target is None else str(summary_target),
        "summary_exists": bool(summary_target is not None and summary_target.exists()),
        "progress_json": str(run_root / "progress.json"),
        "progress_exists": bool((run_root / "progress.json").exists()),
        "started_utc": started,
        "finished_utc": finished,
    }
    if error is not None:
        result["error"] = error
    (out_root / "task_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one TD Optuna CHTC record.")
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return run_record(args.record_id, args.records, args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
