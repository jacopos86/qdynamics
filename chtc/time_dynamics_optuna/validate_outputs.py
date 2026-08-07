#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_ROOT = Path("raw_outputs/chtc_time_dynamics_optuna")
_PIPELINE_NAME = "hh_realtime_optuna_v1"
_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}
_STRICT_VALIDATION_PROFILES = {"strict_qpu_faithful", "strict_qpu_hh"}
_STRICT_EXACT_FEEDBACK_KEYS = {
    "pair_mae_over_exact_span",
    "epsilon_osc_pair",
    "dominant_peak_abs_omega_error",
    "mean_abs_energy_total_error",
    "max_abs_energy_total_error",
    "mean_abs_site_occupations_error",
    "min_fidelity_exact",
    "primary_observable_mae_over_exact_span",
    "objective_pair_mae_over_exact_span",
    "objective_mean_abs_energy_total_error",
    "objective_mean_abs_site_occupations_error",
    "objective_min_fidelity_exact",
}
_STABLE_EARLY_STOP_PREFIX = "progress_observables_stable:"

_EXACT_FINITE_METRICS = [
    "primary_observable_mae_over_exact_span",
    "mean_abs_energy_total_error",
    "mean_abs_site_occupations_error",
    "min_fidelity_exact",
    "final_runtime_parameter_count",
]


@dataclass
class ValidationResult:
    record_id: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_profile: str | None = None
    best_trial_number: int | None = None
    completed_trial_count: int = 0

    def error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "validation_profile": self.validation_profile,
            "best_trial_number": self.best_trial_number,
            "completed_trial_count": self.completed_trial_count,
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_record_ids(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def completed_observations(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(obs)
        for obs in summary.get("observations", []) or []
        if isinstance(obs, Mapping) and str(obs.get("status")) == "completed"
    ]


def find_best_observation(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    best = summary.get("best_objective_trial") or {}
    if not isinstance(best, Mapping):
        return None
    trial_number = best.get("trial_number")
    if trial_number is None:
        return None
    for obs in summary.get("observations", []) or []:
        if not isinstance(obs, Mapping):
            continue
        if _as_int(obs.get("trial_number"), -1) == _as_int(trial_number, -2):
            return dict(obs)
    return None


def resolve_fetched_path(path_value: Any, record_dir: str | Path) -> Path | None:
    if path_value in {None, ""}:
        return None
    record_root = Path(record_dir)
    record_id = record_root.name
    raw = Path(str(path_value))
    if raw.exists():
        return raw
    candidates: list[Path] = []
    parts = raw.parts
    for idx, part in enumerate(parts[:-1]):
        if part == "raw_outputs" and idx + 1 < len(parts) and parts[idx + 1] == record_id:
            candidates.append(record_root.joinpath(*parts[idx + 2 :]))
    for idx, part in enumerate(parts[:-1]):
        if part == record_id:
            candidates.append(record_root.joinpath(*parts[idx + 1 :]))
    if not raw.is_absolute():
        if parts and parts[0] == record_id:
            candidates.append(record_root.joinpath(*parts[1:]))
        if len(parts) >= 2 and parts[0] == "raw_outputs" and parts[1] == record_id:
            candidates.append(record_root.joinpath(*parts[2:]))
        candidates.append(Path.cwd() / raw)
        candidates.append(record_root.parent / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else raw


def _relative_for_error(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _require_common(record_dir: Path, result: ValidationResult) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]] | None:
    record_path = record_dir / "record.json"
    status_path = record_dir / "chtc_status.json"
    summary_path = record_dir / "summary.json"
    progress_path = record_dir / "run" / "progress.json"
    task_result_path = record_dir / "task_result.json"
    command_path = record_dir / "command.sh"
    for required in (record_path, status_path, summary_path, progress_path, task_result_path, command_path):
        if not required.exists():
            result.error(f"missing required file: {_relative_for_error(required, record_dir)}")
    if not result.ok:
        return None
    record = _load_json(record_path)
    status = _load_json(status_path)
    summary = _load_json(summary_path)
    progress = _load_json(progress_path)
    task_result = _load_json(task_result_path)
    result.validation_profile = str(record.get("validation_profile", ""))
    for label, payload in (("record.json", record), ("chtc_status.json", status), ("task_result.json", task_result)):
        payload_record_id = payload.get("record_id")
        if payload_record_id not in {None, result.record_id}:
            result.error(f"{label} record_id {payload_record_id!r} != directory record_id {result.record_id!r}")
    try:
        return_code = int(status.get("return_code"))
    except Exception:
        return_code = 1
    try:
        task_return_code = int(task_result.get("return_code"))
    except Exception:
        task_return_code = None
    if return_code != 0:
        result.error(f"chtc_status return_code is {return_code}")
    if task_return_code != return_code:
        result.error(f"task_result return_code {task_return_code} != chtc_status return_code {return_code}")
    if summary.get("pipeline") != _PIPELINE_NAME:
        result.error(f"summary.pipeline expected {_PIPELINE_NAME!r}, got {summary.get('pipeline')!r}")
    observations = summary.get("observations") or []
    if not isinstance(observations, list):
        result.error("summary.observations is not a list")
        observations = []
    requested = summary.get("n_trials_requested")
    if requested is None:
        result.error("summary.n_trials_requested missing")
    elif len(observations) != int(requested):
        result.error(f"observation count {len(observations)} != n_trials_requested {requested}")
    if progress.get("done") is not True:
        result.error("run/progress.json done is not true")
    completed = completed_observations(summary)
    result.completed_trial_count = len(completed)
    best_obs = find_best_observation(summary)
    if best_obs is None:
        result.error("best_objective_trial did not join to summary.observations")
    else:
        result.best_trial_number = _as_int(best_obs.get("trial_number"), -1)
        output_path = resolve_fetched_path(best_obs.get("output_json"), record_dir)
        if output_path is None or not output_path.exists():
            result.error(f"best observation output_json not found after remap: {best_obs.get('output_json')!r}")
        spectra_value = best_obs.get("spectra_json")
        if spectra_value not in {None, ""}:
            spectra_path = resolve_fetched_path(spectra_value, record_dir)
            if spectra_path is None or not spectra_path.exists():
                result.error(f"best observation spectra_json not found after remap: {spectra_value!r}")
    return record, summary, best_obs, completed, progress


def _is_successful_stable_early_stop_metrics(metrics: Mapping[str, Any]) -> bool:
    reason = metrics.get("full_horizon_early_stop_reason", None)
    if reason in {None, ""}:
        reason = metrics.get("early_stop_reason", None)
    if reason in {None, ""}:
        gate_reason = str(metrics.get("full_horizon_gate_reason", "")).strip()
        if gate_reason.startswith("stable_early_stop:"):
            reason = gate_reason.removeprefix("stable_early_stop:")
    reason_text = "" if reason is None else str(reason).strip()
    return bool(
        metrics.get("full_horizon_gate_passed") is True
        and metrics.get("full_horizon_successful_early_stop") is True
        and str(metrics.get("full_horizon_completion_kind", "")) == "stable_early_stop"
        and reason_text.startswith(_STABLE_EARLY_STOP_PREFIX)
    )


def _trajectory_times_are_monotone(trajectory: Sequence[Mapping[str, Any]]) -> bool:
    prev: float | None = None
    for row in trajectory:
        if not isinstance(row, Mapping):
            return False
        value = row.get("time")
        try:
            t_val = float(value)
        except Exception:
            return False
        if not math.isfinite(t_val):
            return False
        if prev is not None and t_val < prev - 1.0e-12:
            return False
        prev = t_val
    return True


def _validate_exact(record_dir: Path, record: Mapping[str, Any], summary: Mapping[str, Any], best_obs: Mapping[str, Any] | None, completed: Sequence[Mapping[str, Any]], result: ValidationResult) -> None:
    min_completed = _as_int(record.get("min_completed_trials"), 1)
    if len(completed) < min_completed:
        result.error(f"completed trial count {len(completed)} < min_completed_trials {min_completed}")
    if _as_int(summary.get("feasible_trial_count"), 0) < min_completed:
        result.error(
            f"feasible_trial_count {summary.get('feasible_trial_count')} < min_completed_trials {min_completed}"
        )
    if best_obs is None:
        return
    if str(best_obs.get("status")) != "completed":
        result.error("best joined observation is not completed")
    if not _finite(best_obs.get("objective")):
        result.error("best observation objective is not finite")
    metrics = best_obs.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        result.error("best observation metrics is not a mapping")
        return
    if metrics.get("generic_exact_v1_family_objective") is not True:
        result.error("generic_exact_v1_family_objective is not true")
    if parse_bool(record.get("require_full_horizon"), default=True) and metrics.get("full_horizon_gate_passed") is not True:
        result.error("full_horizon_gate_passed is not true")
    for key in _EXACT_FINITE_METRICS:
        if not _finite(metrics.get(key)):
            result.error(f"best observation metric {key!r} is missing or non-finite")
    output_path = resolve_fetched_path(best_obs.get("output_json"), record_dir)
    if output_path is None or not output_path.exists():
        return
    payload = _load_json(output_path)
    trajectory = payload.get("trajectory") or []
    if not isinstance(trajectory, list) or not trajectory:
        result.error("best output_json trajectory is empty or not a list")
        return
    if not isinstance(trajectory[-1], Mapping):
        result.error("best output_json final trajectory row is not a mapping")
    if parse_bool(record.get("require_full_horizon"), default=True):
        stable_success = _is_successful_stable_early_stop_metrics(metrics)
        if stable_success:
            if not _trajectory_times_are_monotone(trajectory):
                result.error("stable early-stop trajectory times are not finite monotone")
        else:
            expected_rows = _as_int(record.get("num_times"), -1)
            if expected_rows > 0 and len(trajectory) != expected_rows:
                result.error(f"trajectory row count {len(trajectory)} != num_times {expected_rows}")


def _token_value(tokens: Sequence[Any], flag: str) -> str | None:
    text = [str(token) for token in tokens]
    try:
        idx = text.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(text):
        return None
    return text[idx + 1]


_CANDIDATE_EXACT_LEAK_VALUES = {"exact", "exact_v1", "benchmark_exact"}


def _validate_strict_class_candidate(
    record_dir: Path,
    *,
    expected_tuning_class: str,
    result: ValidationResult,
) -> None:
    candidate_path = record_dir / "run" / "class_settings_candidate.json"
    if not candidate_path.exists():
        result.error(f"strict class-policy candidate missing: {_relative_for_error(candidate_path, record_dir)}")
        return
    payload = _load_json(candidate_path)
    candidate = payload
    if isinstance(payload, Mapping) and isinstance(payload.get("class_settings_candidate"), Mapping):
        candidate = payload["class_settings_candidate"]
    if not isinstance(candidate, Mapping):
        result.error("strict class-policy candidate is not a JSON object")
        return
    if candidate.get("schema") != "dynamics_class_settings_candidate_v1":
        result.error(f"strict class-policy candidate schema {candidate.get('schema')!r} is not dynamics_class_settings_candidate_v1")
    if expected_tuning_class and str(candidate.get("tuning_class", "")).strip() != expected_tuning_class:
        result.error(
            f"strict class-policy candidate tuning_class {candidate.get('tuning_class')!r} != record {expected_tuning_class!r}"
        )
    if str(candidate.get("algorithm_id", "dyn_controller_full")) != "dyn_controller_full":
        result.error("strict class-policy candidate algorithm_id is not dyn_controller_full")
    if str(candidate.get("settings_kind", "controller")) != "controller":
        result.error("strict class-policy candidate settings_kind is not controller")
    if candidate.get("strict_online_feedback_exact_free") is not True:
        result.error("strict class-policy candidate strict_online_feedback_exact_free is not true")
    settings_payload = candidate.get("settings_payload", {})
    if not isinstance(settings_payload, Mapping):
        result.error("strict class-policy candidate settings_payload is not a mapping")
        return
    for key, value in settings_payload.items():
        key_text = str(key).strip().lower()
        if "exact_forecast" in key_text or "exact-v1" in key_text or "exact_v1" in key_text:
            result.error(f"strict class-policy candidate contains exact-assisted payload key {key!r}")
        if isinstance(value, str) and value.strip().lower() in _CANDIDATE_EXACT_LEAK_VALUES:
            result.error(f"strict class-policy candidate payload key {key!r} uses exact-assisted value {value!r}")


def _as_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _span(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return None
    return float(max(finite) - min(finite))


def _trajectory_float_series(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = _as_float_or_none(row.get(key))
        if value is not None:
            out.append(value)
    return out


def _validate_not_hard_frozen(
    payload: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    trial_number: int,
    result: ValidationResult,
) -> None:
    rows = [dict(row) for row in payload.get("trajectory", []) if isinstance(row, Mapping)]
    if len(rows) < 3:
        return
    exact_span = _span(_trajectory_float_series(rows, "energy_total_exact"))
    algo_span = _span(_trajectory_float_series(rows, "energy_total"))
    if exact_span is None or algo_span is None:
        return
    mean_rho_num = _as_float_or_none(metrics.get("mean_rho_num"))
    mean_rho_miss = _as_float_or_none(metrics.get("mean_rho_miss"))
    if (
        exact_span > 1.0e-4
        and algo_span < 1.0e-8
        and mean_rho_num is not None
        and mean_rho_num < 1.0e-10
        and mean_rho_miss is not None
        and mean_rho_miss > 0.90
    ):
        result.error(
            "trial "
            f"{trial_number}: hard frozen trajectory signature: exact energy moves "
            f"span={exact_span:.3e}, algorithm energy span={algo_span:.3e}, "
            f"mean_rho_num={mean_rho_num:.3e}, mean_rho_miss={mean_rho_miss:.3e}"
        )


def _validate_strict_tokens(tokens_path: Path, result: ValidationResult) -> None:
    if not tokens_path.exists():
        result.error(f"strict input_tokens.json missing: {tokens_path}")
        return
    data = _load_json(tokens_path)
    if isinstance(data, Mapping):
        raw_tokens = data.get("tokens")
        if raw_tokens is None:
            raw_tokens = data.get("argv")
    elif isinstance(data, list):
        raw_tokens = data
    else:
        raw_tokens = []
    tokens = [str(token) for token in raw_tokens or []]
    if "--checkpoint-controller-strict-qpu-faithful" not in tokens:
        result.error("strict tokens missing --checkpoint-controller-strict-qpu-faithful")
    required_pairs = {
        "--checkpoint-controller-reference-mode": "off",
        "--checkpoint-controller-exact-input-mode": "off",
    }
    mode = _token_value(tokens, "--checkpoint-controller-mode")
    if mode not in {"oracle_v1", "observable_v1"}:
        result.error(f"strict token --checkpoint-controller-mode expected oracle_v1/observable_v1, got {mode!r}")
    for flag, expected in required_pairs.items():
        actual = _token_value(tokens, flag)
        if actual != expected:
            result.error(f"strict token {flag} expected {expected!r}, got {actual!r}")
    lowered = [token.lower() for token in tokens]
    if any("exact_v1" in token for token in lowered):
        result.error("strict input tokens contain exact_v1")
    if any("benchmark_exact" in token for token in lowered):
        result.error("strict input tokens contain benchmark_exact")
    if any("exact-forecast" in token for token in lowered):
        result.error("strict input tokens contain exact-forecast")


def _validate_strict(record_dir: Path, record: Mapping[str, Any], summary: Mapping[str, Any], best_obs: Mapping[str, Any] | None, completed: Sequence[Mapping[str, Any]], result: ValidationResult) -> None:
    strict_append_prune_profile = str(record.get("study_profile", "")).strip() in {
        "strict_qpu_faithful_append_prune_recoverability_v1",
        "strict_qpu_faithful_append_prune_guardrail_only_v1",
        "strict_qpu_faithful_append_prune_aggressive_v1",
    }
    expected_tuning_class = str(record.get("tuning_class", "")).strip()
    if expected_tuning_class and str(summary.get("tuning_class", "")).strip() != expected_tuning_class:
        result.error(
            f"summary tuning_class {summary.get('tuning_class')!r} != record {expected_tuning_class!r}"
        )
    provenance = summary.get("class_tuning_provenance", {})
    if expected_tuning_class and isinstance(provenance, Mapping):
        if str(provenance.get("tuning_class", "")).strip() != expected_tuning_class:
            result.error(
                "summary class_tuning_provenance.tuning_class "
                f"{provenance.get('tuning_class')!r} != record {expected_tuning_class!r}"
            )
    if strict_append_prune_profile:
        _validate_strict_class_candidate(
            record_dir,
            expected_tuning_class=expected_tuning_class,
            result=result,
        )
    min_completed = _as_int(record.get("min_completed_trials"), 1)
    if len(completed) < min_completed:
        result.error(f"completed trial count {len(completed)} < min_completed_trials {min_completed}")
    if _as_int(summary.get("feasible_trial_count"), 0) < min_completed:
        result.error(
            f"feasible_trial_count {summary.get('feasible_trial_count')} < min_completed_trials {min_completed}"
        )
    if best_obs is not None and not _finite(best_obs.get("objective")):
        result.error("best strict observation objective is not finite")
    best_metrics = {}
    best_trial = summary.get("best_objective_trial")
    if isinstance(best_trial, Mapping) and isinstance(best_trial.get("metrics"), Mapping):
        best_metrics = dict(best_trial.get("metrics") or {})
    exact_best_keys = sorted(_STRICT_EXACT_FEEDBACK_KEYS.intersection(best_metrics.keys()))
    if exact_best_keys:
        result.error(f"best_objective_trial strict metrics contain exact-feedback keys {exact_best_keys}")
    require_full = parse_bool(record.get("require_full_horizon"), default=True)
    for obs in completed:
        trial_number = _as_int(obs.get("trial_number"), -1)
        if not _finite(obs.get("objective")):
            result.error(f"trial {trial_number}: objective is not finite")
        metrics = obs.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            result.error(f"trial {trial_number}: metrics is not a mapping")
            continue
        required_metric_values = {
            "strict_qpu_faithful": True,
            "qpu_faithful_decisions_passed": True,
            "strict_decision_contract_passed": True,
            "strict_fail_closed": False,
            "reference_enabled": False,
        }
        for key, expected in required_metric_values.items():
            if metrics.get(key) is not expected:
                result.error(f"trial {trial_number}: metric {key} expected {expected!r}, got {metrics.get(key)!r}")
        if _as_int(metrics.get("exact_decision_checkpoints"), -1) != 0:
            result.error(f"trial {trial_number}: exact_decision_checkpoints is not 0")
        if str(metrics.get("reference_mode")) != "off":
            result.error(f"trial {trial_number}: reference_mode is not off")
        if str(metrics.get("decision_noise_mode")) != "ideal":
            result.error(f"trial {trial_number}: decision_noise_mode is not ideal")
        if _as_int(metrics.get("non_ideal_decision_noise_count"), -1) != 0:
            result.error(f"trial {trial_number}: non_ideal_decision_noise_count is not 0")
        if require_full and metrics.get("full_horizon_gate_passed") is not True:
            result.error(f"trial {trial_number}: full_horizon_gate_passed is not true")
        exact_keys = sorted(_STRICT_EXACT_FEEDBACK_KEYS.intersection(metrics.keys()))
        if exact_keys:
            result.error(f"trial {trial_number}: strict metrics contain exact-feedback keys {exact_keys}")
        if strict_append_prune_profile:
            for key in (
                "append_opportunity_count",
                "proposed_append_count",
                "prune_opportunity_count",
                "prune_candidate_checkpoint_count",
                "prune_candidate_count",
            ):
                if not _finite(metrics.get(key)):
                    result.error(f"trial {trial_number}: strict append/prune metric {key!r} is missing or non-finite")
            if _as_int(metrics.get("append_opportunity_count"), 0) <= 0:
                result.warnings.append(f"trial {trial_number}: no measured append opportunity was observed")
            if _as_int(metrics.get("prune_opportunity_count"), 0) <= 0:
                result.warnings.append(f"trial {trial_number}: no measured prune opportunity was observed")
        output_path = resolve_fetched_path(obs.get("output_json"), record_dir)
        if output_path is None or not output_path.exists():
            result.error(f"trial {trial_number}: output_json missing after remap")
            continue
        tokens_path = output_path.parent / "input_tokens.json"
        _validate_strict_tokens(tokens_path, result)
        payload = _load_json(output_path)
        if strict_append_prune_profile:
            _validate_not_hard_frozen(payload, metrics=metrics, trial_number=trial_number, result=result)
        diagnostic_reference = payload.get("diagnostic_reference")
        if isinstance(diagnostic_reference, Mapping):
            if diagnostic_reference.get("feeds_controller_decisions") is not False:
                result.error(
                    f"trial {trial_number}: diagnostic_reference feeds_controller_decisions is not false"
                )


def validate_record_dir(record_dir: str | Path) -> ValidationResult:
    root = Path(record_dir)
    result = ValidationResult(record_id=root.name)
    common = _require_common(root, result)
    if common is None:
        return result
    record, summary, best_obs, completed, _progress = common
    profile = str(record.get("validation_profile", ""))
    if profile == "generic_exact_v1":
        _validate_exact(root, record, summary, best_obs, completed, result)
    elif profile in _STRICT_VALIDATION_PROFILES:
        _validate_strict(root, record, summary, best_obs, completed, result)
    else:
        result.error(f"unsupported validation_profile {profile!r}")
    return result


def validate_outputs(
    root: str | Path = DEFAULT_ROOT,
    *,
    record_ids: Sequence[str] | None = None,
    record_list: str | Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    output_root = Path(root)
    expected_ids: list[str] | None
    if record_ids:
        expected_ids = [str(item) for item in record_ids]
    elif record_list is not None:
        expected_ids = load_record_ids(record_list)
    else:
        expected_ids = None
    if expected_ids is None:
        record_dirs = sorted(path for path in output_root.iterdir() if path.is_dir()) if output_root.exists() else []
        results = [
            validate_record_dir(record_dir)
            for record_dir in record_dirs
            if (record_dir / "record.json").exists() or (record_dir / "chtc_status.json").exists()
        ]
    else:
        results = []
        for record_id in expected_ids:
            record_dir = output_root / record_id
            if not record_dir.exists():
                missing = ValidationResult(record_id=record_id, ok=False)
                missing.error(f"missing expected record output directory: {record_id}")
                results.append(missing)
            else:
                results.append(validate_record_dir(record_dir))
    report = {
        "generated_utc": _now_utc(),
        "root": str(output_root),
        "record_list": None if record_list is None else str(record_list),
        "expected_records": expected_ids,
        "record_count": len(results),
        "ok": bool(results) and all(item.ok for item in results),
        "records": [item.as_dict() for item in results],
        "failed_records": [item.record_id for item in results if not item.ok],
    }
    if not results:
        report["ok"] = False
        report["errors"] = ["no record output directories found"]
    if write_report:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_root / "failed_records.txt").write_text("\n".join(report["failed_records"]) + ("\n" if report["failed_records"] else ""), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate fetched TD Optuna CHTC outputs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--record-list", type=Path, default=None)
    parser.add_argument("--record-id", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = validate_outputs(args.root, record_ids=args.record_id, record_list=args.record_list, write_report=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
