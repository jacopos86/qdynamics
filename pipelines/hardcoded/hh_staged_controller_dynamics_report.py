#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from docs.reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
from src.quantum.drives_time_potential import default_spatial_weights, evaluate_drive_waveform

_REQUIRED_PHYSICS_KEYS: tuple[str, ...] = (
    "L",
    "t",
    "u",
    "dv",
    "omega0",
    "g_ep",
    "n_ph_max",
    "boundary",
    "ordering",
)
_AI_LOG_PREFIX = "AI_LOG "


"baseline = {t_i, F_i, E_i, O_i}_{i=1..N}"
@dataclass(frozen=True)
class BaselineDynamics:
    times: np.ndarray
    fidelity: np.ndarray
    energy_total_exact: np.ndarray
    energy_total_trotter: np.ndarray
    abs_energy_total_error: np.ndarray
    staggered: np.ndarray
    doublon: np.ndarray


"noisy = {t_i, E_ideal_i, E_noisy_i, O_ideal_i, O_noisy_i}_{i=1..N}"
@dataclass(frozen=True)
class NoisyDynamics:
    times: np.ndarray
    energy_total_ideal: np.ndarray
    energy_total_noisy: np.ndarray
    energy_total_delta: np.ndarray
    staggered_ideal: np.ndarray
    staggered_noisy: np.ndarray
    doublon_ideal: np.ndarray
    doublon_noisy: np.ndarray


"compile_summary = aggregate(stdout AI_LOG backend_scheduled compile events)"
@dataclass(frozen=True)
class CompileLogSummary:
    log_path: Path | None
    compile_start_count: int
    compile_done_count: int
    compile_cache_hit_count: int
    unique_circuit_count: int
    mean_two_qubit_count: float
    max_two_qubit_count: int
    mean_depth: float
    max_depth: int
    first_ts_utc: str | None
    last_ts_utc: str | None


"controller = {t_i, metrics_i}_{i=1..N}"
@dataclass(frozen=True)
class ControllerTelemetry:
    status: str
    reason: str | None
    time_axis: np.ndarray
    fidelity_exact: np.ndarray
    abs_energy_total_error: np.ndarray
    rho_miss: np.ndarray
    motion_kink_score: np.ndarray
    logical_block_count: np.ndarray
    runtime_parameter_count: np.ndarray
    selected_noisy_energy_mean: np.ndarray
    stay_noisy_energy_mean: np.ndarray
    oracle_budget_scale: np.ndarray
    oracle_confirm_limit: np.ndarray
    action_kinds: tuple[str, ...]
    append_count: int
    stay_count: int
    oracle_decision_checkpoints: int
    exact_decision_checkpoints: int
    oracle_cache_hits: int
    oracle_cache_misses: int
    exact_cache_hits: int
    exact_cache_misses: int
    geometry_memo_hits: int
    geometry_memo_misses: int
    raw_group_cache_hits: int
    raw_group_cache_misses: int
    compile_summary: CompileLogSummary


"entry = workflow_json ⊕ baseline ⊕ noisy ⊕ controller"
@dataclass(frozen=True)
class WorkflowReportEntry:
    json_path: Path
    payload: dict[str, Any]
    label: str
    physics: dict[str, Any]
    dynamics: dict[str, Any]
    noise: dict[str, Any]
    drive_profile: dict[str, Any]
    baseline: BaselineDynamics
    noisy: NoisyDynamics
    controller: ControllerTelemetry


"payload = json(path)"
def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


"x = float(value) if finite else nan"
def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


"arr_i = row_i[key]"
def _array_from_rows(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    fallback: float = float("nan"),
) -> np.ndarray:
    return np.asarray([_float_or_nan(row.get(key, fallback)) for row in rows], dtype=float)


"arr_i = row_i[primary] if finite else row_i[fallback]"
def _time_axis_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary: str,
    fallback: str,
) -> np.ndarray:
    primary_values = _array_from_rows(rows, primary)
    if primary_values.size and np.all(np.isfinite(primary_values)):
        return primary_values
    return _array_from_rows(rows, fallback)


"logsafe(x) = max(|x|, eps)"
def _positive_floor(values: np.ndarray, eps: float = 1.0e-18) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.maximum(np.abs(arr), float(eps))


"label = human_readable_controller_identity(payload)"
def _default_label(payload: Mapping[str, Any], path: Path) -> str:
    settings = payload.get("settings", {}) if isinstance(payload.get("settings", {}), Mapping) else {}
    realtime = settings.get("realtime_checkpoint", {}) if isinstance(settings.get("realtime_checkpoint", {}), Mapping) else {}
    adaptive = (
        payload.get("adaptive_realtime_checkpoint", {})
        if isinstance(payload.get("adaptive_realtime_checkpoint", {}), Mapping)
        else {}
    )
    mode = str(realtime.get("mode") or adaptive.get("mode") or "unknown")
    status = str(adaptive.get("status") or ("disabled" if mode == "off" else "missing"))
    if mode == "off":
        return "controller off"
    return f"{mode} ({status})"


"summary = parse(stdout.log)"
def _load_compile_log_summary(stdout_log: Path | None) -> CompileLogSummary:
    if stdout_log is None or not stdout_log.exists():
        return CompileLogSummary(
            log_path=stdout_log,
            compile_start_count=0,
            compile_done_count=0,
            compile_cache_hit_count=0,
            unique_circuit_count=0,
            mean_two_qubit_count=float("nan"),
            max_two_qubit_count=0,
            mean_depth=float("nan"),
            max_depth=0,
            first_ts_utc=None,
            last_ts_utc=None,
        )
    compile_start_count = 0
    compile_done_count = 0
    compile_cache_hit_count = 0
    circuits: set[str] = set()
    two_qubit_counts: list[int] = []
    depths: list[int] = []
    ts_values: list[str] = []
    with stdout_log.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if _AI_LOG_PREFIX not in raw_line:
                continue
            _, _, payload_str = raw_line.partition(_AI_LOG_PREFIX)
            try:
                payload = json.loads(payload_str)
            except Exception:
                continue
            if not isinstance(payload, Mapping):
                continue
            event = str(payload.get("event", ""))
            if "noise_oracle_backend_scheduled_compile" not in event:
                continue
            circuit_name = payload.get("circuit_name")
            if circuit_name not in {None, ""}:
                circuits.add(str(circuit_name))
            ts_raw = payload.get("ts_utc")
            if ts_raw not in {None, ""}:
                ts_values.append(str(ts_raw))
            if event.endswith("compile_start"):
                compile_start_count += 1
            elif event.endswith("compile_done"):
                compile_done_count += 1
                two_qubit_counts.append(int(payload.get("compiled_two_qubit_count", 0) or 0))
                depths.append(int(payload.get("compiled_depth", 0) or 0))
            elif event.endswith("compile_cache_hit"):
                compile_cache_hit_count += 1
                two_qubit_counts.append(int(payload.get("compiled_two_qubit_count", 0) or 0))
                depths.append(int(payload.get("compiled_depth", 0) or 0))
    return CompileLogSummary(
        log_path=stdout_log.resolve(),
        compile_start_count=int(compile_start_count),
        compile_done_count=int(compile_done_count),
        compile_cache_hit_count=int(compile_cache_hit_count),
        unique_circuit_count=int(len(circuits)),
        mean_two_qubit_count=(float(np.mean(two_qubit_counts)) if two_qubit_counts else float("nan")),
        max_two_qubit_count=(int(max(two_qubit_counts)) if two_qubit_counts else 0),
        mean_depth=(float(np.mean(depths)) if depths else float("nan")),
        max_depth=(int(max(depths)) if depths else 0),
        first_ts_utc=(min(ts_values) if ts_values else None),
        last_ts_utc=(max(ts_values) if ts_values else None),
    )


"baseline = driven replay trajectory"
def _load_baseline_dynamics(
    payload: Mapping[str, Any],
    *,
    profile_name: str,
    method_name: str,
) -> BaselineDynamics:
    dynamics_noiseless = payload.get("dynamics_noiseless", {})
    if not isinstance(dynamics_noiseless, Mapping):
        raise KeyError("Missing dynamics_noiseless in staged workflow payload.")
    profiles = dynamics_noiseless.get("profiles", {})
    if not isinstance(profiles, Mapping) or profile_name not in profiles:
        raise KeyError(f"Missing dynamics_noiseless.profiles.{profile_name!r}.")
    profile = profiles[profile_name]
    if not isinstance(profile, Mapping):
        raise TypeError(f"dynamics_noiseless.profiles.{profile_name!r} must be a mapping.")
    methods = profile.get("methods", {})
    if not isinstance(methods, Mapping) or method_name not in methods:
        raise KeyError(f"Missing dynamics_noiseless.profiles.{profile_name}.{method_name}.")
    method_payload = methods[method_name]
    if not isinstance(method_payload, Mapping):
        raise TypeError("Noiseless method payload must be a mapping.")
    trajectory = method_payload.get("trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("Noiseless trajectory is empty; cannot render time-dynamics report.")
    rows = [row for row in trajectory if isinstance(row, Mapping)]
    if len(rows) != len(trajectory):
        raise TypeError("Noiseless trajectory rows must be mappings.")
    times = _array_from_rows(rows, "time")
    energy_total_exact = _array_from_rows(rows, "energy_total_exact")
    energy_total_trotter = _array_from_rows(rows, "energy_total_trotter")
    return BaselineDynamics(
        times=times,
        fidelity=_array_from_rows(rows, "fidelity"),
        energy_total_exact=energy_total_exact,
        energy_total_trotter=energy_total_trotter,
        abs_energy_total_error=np.abs(energy_total_trotter - energy_total_exact),
        staggered=_array_from_rows(rows, "staggered_exact"),
        doublon=_array_from_rows(rows, "doublon_exact"),
    )


"noisy = driven noisy trajectory"
def _load_noisy_dynamics(
    payload: Mapping[str, Any],
    *,
    profile_name: str,
    method_name: str,
    noise_mode: str,
) -> NoisyDynamics:
    dynamics_noisy = payload.get("dynamics_noisy", {})
    if not isinstance(dynamics_noisy, Mapping):
        raise KeyError("Missing dynamics_noisy in staged workflow payload.")
    profiles = dynamics_noisy.get("profiles", {})
    if not isinstance(profiles, Mapping) or profile_name not in profiles:
        raise KeyError(f"Missing dynamics_noisy.profiles.{profile_name!r}.")
    profile = profiles[profile_name]
    if not isinstance(profile, Mapping):
        raise TypeError(f"dynamics_noisy.profiles.{profile_name!r} must be a mapping.")
    methods = profile.get("methods", {})
    if not isinstance(methods, Mapping) or method_name not in methods:
        raise KeyError(f"Missing dynamics_noisy.profiles.{profile_name}.{method_name}.")
    method_payload = methods[method_name]
    if not isinstance(method_payload, Mapping):
        raise TypeError("Noisy method payload must be a mapping.")
    modes = method_payload.get("modes", {})
    if not isinstance(modes, Mapping) or noise_mode not in modes:
        raise KeyError(f"Missing noisy mode {noise_mode!r} for method {method_name!r}.")
    mode_payload = modes[noise_mode]
    if not isinstance(mode_payload, Mapping):
        raise TypeError("Noisy mode payload must be a mapping.")
    trajectory = mode_payload.get("trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("Noisy trajectory is empty; cannot render time-dynamics report.")
    rows = [row for row in trajectory if isinstance(row, Mapping)]
    if len(rows) != len(trajectory):
        raise TypeError("Noisy trajectory rows must be mappings.")
    return NoisyDynamics(
        times=_array_from_rows(rows, "time"),
        energy_total_ideal=_array_from_rows(rows, "energy_total_ideal"),
        energy_total_noisy=_array_from_rows(rows, "energy_total_noisy"),
        energy_total_delta=_array_from_rows(rows, "energy_total_delta_noisy_minus_ideal"),
        staggered_ideal=_array_from_rows(rows, "staggered_ideal"),
        staggered_noisy=_array_from_rows(rows, "staggered_noisy"),
        doublon_ideal=_array_from_rows(rows, "doublon_ideal"),
        doublon_noisy=_array_from_rows(rows, "doublon_noisy"),
    )


"controller = adaptive checkpoint trajectory + cache telemetry"
def _load_controller_telemetry(payload: Mapping[str, Any], *, stdout_log: Path | None) -> ControllerTelemetry:
    adaptive = (
        payload.get("adaptive_realtime_checkpoint", {})
        if isinstance(payload.get("adaptive_realtime_checkpoint", {}), Mapping)
        else {}
    )
    summary = adaptive.get("summary", {}) if isinstance(adaptive.get("summary", {}), Mapping) else {}
    trajectory = adaptive.get("trajectory", [])
    ledger = adaptive.get("ledger", [])
    trajectory_rows = [row for row in trajectory if isinstance(row, Mapping)]
    if len(trajectory_rows) != len(trajectory):
        raise TypeError("adaptive_realtime_checkpoint.trajectory rows must be mappings.")
    ledger_rows = [row for row in ledger if isinstance(row, Mapping)]
    if len(ledger_rows) != len(ledger):
        raise TypeError("adaptive_realtime_checkpoint.ledger rows must be mappings.")

    def _sum_int(rows: Sequence[Mapping[str, Any]], key: str) -> int:
        return int(sum(int(row.get(key, 0) or 0) for row in rows))

    if trajectory_rows:
        time_axis = _time_axis_from_rows(trajectory_rows, primary="physical_time", fallback="time")
        action_kinds = tuple(str(row.get("action_kind", "unknown")) for row in trajectory_rows)
        fidelity_exact = _array_from_rows(trajectory_rows, "fidelity_exact")
        abs_energy_total_error = _array_from_rows(trajectory_rows, "abs_energy_total_error")
        rho_miss = _array_from_rows(trajectory_rows, "rho_miss")
        motion_kink_score = _array_from_rows(trajectory_rows, "motion_kink_score")
        logical_block_count = _array_from_rows(trajectory_rows, "logical_block_count")
        runtime_parameter_count = _array_from_rows(trajectory_rows, "runtime_parameter_count")
        selected_noisy_energy_mean = _array_from_rows(trajectory_rows, "selected_noisy_energy_mean")
        stay_noisy_energy_mean = _array_from_rows(trajectory_rows, "stay_noisy_energy_mean")
        oracle_budget_scale = _array_from_rows(trajectory_rows, "oracle_budget_scale")
        oracle_confirm_limit = _array_from_rows(trajectory_rows, "oracle_confirm_limit")
    else:
        empty = np.asarray([], dtype=float)
        time_axis = empty
        action_kinds = tuple()
        fidelity_exact = empty
        abs_energy_total_error = empty
        rho_miss = empty
        motion_kink_score = empty
        logical_block_count = empty
        runtime_parameter_count = empty
        selected_noisy_energy_mean = empty
        stay_noisy_energy_mean = empty
        oracle_budget_scale = empty
        oracle_confirm_limit = empty
    return ControllerTelemetry(
        status=str(adaptive.get("status", "missing")),
        reason=(None if adaptive.get("reason") in {None, ""} else str(adaptive.get("reason"))),
        time_axis=time_axis,
        fidelity_exact=fidelity_exact,
        abs_energy_total_error=abs_energy_total_error,
        rho_miss=rho_miss,
        motion_kink_score=motion_kink_score,
        logical_block_count=logical_block_count,
        runtime_parameter_count=runtime_parameter_count,
        selected_noisy_energy_mean=selected_noisy_energy_mean,
        stay_noisy_energy_mean=stay_noisy_energy_mean,
        oracle_budget_scale=oracle_budget_scale,
        oracle_confirm_limit=oracle_confirm_limit,
        action_kinds=action_kinds,
        append_count=int(summary.get("append_count", 0) or 0),
        stay_count=int(summary.get("stay_count", 0) or 0),
        oracle_decision_checkpoints=int(summary.get("oracle_decision_checkpoints", 0) or 0),
        exact_decision_checkpoints=int(summary.get("exact_decision_checkpoints", 0) or 0),
        oracle_cache_hits=_sum_int(ledger_rows, "oracle_cache_hits"),
        oracle_cache_misses=_sum_int(ledger_rows, "oracle_cache_misses"),
        exact_cache_hits=_sum_int(ledger_rows, "exact_cache_hits"),
        exact_cache_misses=_sum_int(ledger_rows, "exact_cache_misses"),
        geometry_memo_hits=_sum_int(ledger_rows, "geometry_memo_hits"),
        geometry_memo_misses=_sum_int(ledger_rows, "geometry_memo_misses"),
        raw_group_cache_hits=_sum_int(ledger_rows, "raw_group_cache_hits"),
        raw_group_cache_misses=_sum_int(ledger_rows, "raw_group_cache_misses"),
        compile_summary=_load_compile_log_summary(stdout_log),
    )


"entry = load_workflow_report_entry(json_path)"
def load_workflow_report_entry(
    json_path: str | Path,
    *,
    profile_name: str = "drive",
) -> WorkflowReportEntry:
    path = Path(json_path).resolve()
    payload = _read_json(path)
    if str(payload.get("pipeline", "")) != "hh_staged_noise":
        raise ValueError(f"Expected hh_staged_noise payload, got pipeline={payload.get('pipeline')!r} for {path}.")
    settings = payload.get("settings", {})
    if not isinstance(settings, Mapping):
        raise KeyError(f"Missing settings in {path}.")
    physics = dict(settings.get("physics", {})) if isinstance(settings.get("physics", {}), Mapping) else {}
    missing_physics = [key for key in _REQUIRED_PHYSICS_KEYS if key not in physics]
    if missing_physics:
        raise KeyError(f"Missing physics setting keys in {path}: {', '.join(missing_physics)}")
    dynamics = dict(settings.get("dynamics", {})) if isinstance(settings.get("dynamics", {}), Mapping) else {}
    noise = dict(settings.get("noise", {})) if isinstance(settings.get("noise", {}), Mapping) else {}
    methods = dynamics.get("methods", [])
    if not isinstance(methods, Sequence) or not methods:
        raise KeyError(f"Missing settings.dynamics.methods in {path}.")
    method_name = str(methods[0])
    noise_modes = noise.get("modes", [])
    if not isinstance(noise_modes, Sequence) or not noise_modes:
        raise KeyError(f"Missing settings.noise.modes in {path}.")
    noise_mode = str(noise_modes[0])
    profiles = payload.get("dynamics_noiseless", {}).get("profiles", {}) if isinstance(payload.get("dynamics_noiseless", {}), Mapping) else {}
    drive_profile = {}
    if isinstance(profiles, Mapping):
        profile_payload = profiles.get(profile_name, {})
        if isinstance(profile_payload, Mapping) and isinstance(profile_payload.get("drive_profile", {}), Mapping):
            drive_profile = dict(profile_payload.get("drive_profile", {}))
    if not drive_profile:
        drive_profile = {
            "A": dynamics.get("drive_A"),
            "omega": dynamics.get("drive_omega"),
            "tbar": dynamics.get("drive_tbar"),
            "phi": dynamics.get("drive_phi"),
            "pattern": dynamics.get("drive_pattern"),
            "custom_weights": dynamics.get("drive_custom_s"),
            "time_sampling": dynamics.get("drive_time_sampling"),
            "t0": dynamics.get("drive_t0"),
        }
    stdout_log = path.parent / "logs" / "stdout.log"
    return WorkflowReportEntry(
        json_path=path,
        payload=payload,
        label=_default_label(payload, path),
        physics=physics,
        dynamics=dynamics,
        noise=noise,
        drive_profile=drive_profile,
        baseline=_load_baseline_dynamics(payload, profile_name=profile_name, method_name=method_name),
        noisy=_load_noisy_dynamics(
            payload,
            profile_name=profile_name,
            method_name=method_name,
            noise_mode=noise_mode,
        ),
        controller=_load_controller_telemetry(payload, stdout_log=stdout_log),
    )


"signature = common_physics_and_drive(entry)"
def _compatibility_signature(entry: WorkflowReportEntry) -> tuple[Any, ...]:
    return (
        *(entry.physics.get(key) for key in _REQUIRED_PHYSICS_KEYS),
        bool(entry.dynamics.get("enable_drive", False)),
        entry.dynamics.get("drive_A"),
        entry.dynamics.get("drive_omega"),
        entry.dynamics.get("drive_tbar"),
        entry.dynamics.get("drive_phi"),
        entry.dynamics.get("drive_pattern"),
        entry.dynamics.get("drive_time_sampling"),
        entry.dynamics.get("drive_t0"),
        entry.noise.get("backend_name"),
    )


"validate(entries) = same physics + same driven surface"
def _validate_compatibility(entries: Sequence[WorkflowReportEntry]) -> None:
    if not entries:
        raise ValueError("At least one staged workflow JSON is required.")
    signatures = {_compatibility_signature(entry) for entry in entries}
    if len(signatures) != 1:
        raise ValueError("All report inputs must share the same driven HH physics and fake-backend surface.")


"row = headline_metrics(entry)"
def _headline_row(entry: WorkflowReportEntry) -> list[str]:
    controller = entry.controller
    final_noisy_delta = (
        float(entry.noisy.energy_total_delta[-1]) if entry.noisy.energy_total_delta.size else float("nan")
    )
    final_controller_fidelity = (
        float(controller.fidelity_exact[-1]) if controller.fidelity_exact.size else float("nan")
    )
    final_controller_error = (
        float(controller.abs_energy_total_error[-1]) if controller.abs_energy_total_error.size else float("nan")
    )
    return [
        entry.label,
        controller.status,
        f"{float(entry.baseline.fidelity[-1]):.6f}",
        f"{float(entry.baseline.abs_energy_total_error[-1]):.3e}",
        f"{final_noisy_delta:.3e}",
        ("n/a" if not np.isfinite(final_controller_fidelity) else f"{final_controller_fidelity:.6f}"),
        ("n/a" if not np.isfinite(final_controller_error) else f"{final_controller_error:.3e}"),
        f"{controller.append_count}/{controller.stay_count}",
    ]


"extra = manifest_fields(entries)"
def _manifest_extra(entries: Sequence[WorkflowReportEntry]) -> dict[str, Any]:
    first = entries[0]
    controller_modes = []
    for entry in entries:
        realtime = entry.payload.get("settings", {}).get("realtime_checkpoint", {})
        mode = realtime.get("mode") if isinstance(realtime, Mapping) else None
        controller_modes.append(str(mode or "unknown"))
    return {
        "L": first.physics.get("L"),
        "omega0": first.physics.get("omega0"),
        "g_ep": first.physics.get("g_ep"),
        "n_ph_max": first.physics.get("n_ph_max"),
        "boundary": first.physics.get("boundary"),
        "ordering": first.physics.get("ordering"),
        "t_final": first.dynamics.get("t_final"),
        "num_times": first.dynamics.get("num_times"),
        "trotter_steps": first.dynamics.get("trotter_steps"),
        "exact_steps_multiplier": first.dynamics.get("exact_steps_multiplier"),
        "drive_pattern": first.dynamics.get("drive_pattern"),
        "drive_A": first.dynamics.get("drive_A"),
        "drive_omega": first.dynamics.get("drive_omega"),
        "drive_tbar": first.dynamics.get("drive_tbar"),
        "drive_phi": first.dynamics.get("drive_phi"),
        "drive_t0": first.dynamics.get("drive_t0"),
        "drive_sampling": first.dynamics.get("drive_time_sampling"),
        "noisy_method": ",".join(str(x) for x in first.dynamics.get("methods", [])),
        "noise_mode": ",".join(str(x) for x in first.noise.get("modes", [])),
        "controller_modes": ", ".join(controller_modes),
        "controller_noise_mode": first.noise.get("controller_noise_mode"),
        "backend_name": first.noise.get("backend_name"),
        "input_jsons": "; ".join(str(entry.json_path) for entry in entries),
    }


"lines = prose(summary(entries))"
def _overview_lines(entries: Sequence[WorkflowReportEntry]) -> list[str]:
    lines = [
        "Driven HH staged controller time-dynamics report",
        "",
        "How to read this PDF",
        "- The replay panels show the driven matched-family baseline propagated from the saved replay seed.",
        "- Fidelity is measured against the driven exact replay reference for the same initial state.",
        "- |ΔE_total| tracks the absolute total-energy mismatch between the propagated state and the driven exact replay reference.",
        "- The noisy panels compare backend-sampled observables against their noiseless driven baseline.",
        "- Controller panels show append/stay behavior only when adaptive_realtime_checkpoint produced a trajectory.",
        "- If a controller page is marked env_blocked or off, there is no adaptive controller time series to judge yet.",
        "",
        "Input workflow JSONs",
    ]
    for entry in entries:
        lines.append(f"- {entry.label}: {entry.json_path}")
        if entry.controller.compile_summary.log_path is not None:
            lines.append(f"  stdout log: {entry.controller.compile_summary.log_path}")
    return lines


"waveform_page = v(t), s_j v(t)"
def _render_drive_page(pdf: Any, entries: Sequence[WorkflowReportEntry]) -> None:
    require_matplotlib()
    plt = get_plt()
    first = entries[0]
    if not bool(first.dynamics.get("enable_drive", False)):
        return
    weights = default_spatial_weights(
        int(first.physics.get("L", 0)),
        mode=str(first.drive_profile.get("pattern", "staggered")),
        custom=first.drive_profile.get("custom_weights"),
    )
    waveform = evaluate_drive_waveform(
        first.baseline.times,
        first.drive_profile,
        float(first.drive_profile.get("A", 0.0) or 0.0),
    )
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    axes[0].plot(first.baseline.times, waveform, color="#1f77b4", linewidth=2.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_title("Driven scalar waveform v(t)")
    axes[0].set_ylabel("v(t)")
    axes[0].grid(True, alpha=0.25)
    for site_idx, weight in enumerate(weights.tolist()):
        axes[1].plot(
            first.baseline.times,
            float(weight) * waveform,
            linewidth=1.8,
            label=f"site {site_idx} (weight={float(weight):+.1f})",
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Site-resolved onsite-density drive s_j v(t)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Potential")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


"overlay_page = compare replay/noisy traces across runs"
def _render_overlay_page(pdf: Any, entries: Sequence[WorkflowReportEntry]) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
    ax_fid, ax_err, ax_noisy_e, ax_noisy_stag = axes.ravel()
    for entry in entries:
        ax_fid.plot(entry.baseline.times, entry.baseline.fidelity, linewidth=2.0, label=entry.label)
        ax_err.semilogy(
            entry.baseline.times,
            _positive_floor(entry.baseline.abs_energy_total_error),
            linewidth=2.0,
            label=entry.label,
        )
        ax_noisy_e.plot(entry.noisy.times, entry.noisy.energy_total_delta, linewidth=2.0, label=entry.label)
        ax_noisy_stag.plot(
            entry.noisy.times,
            entry.noisy.staggered_noisy - entry.noisy.staggered_ideal,
            linewidth=2.0,
            label=entry.label,
        )
    ax_fid.set_title("Driven replay fidelity vs exact reference")
    ax_fid.set_ylabel("Fidelity")
    ax_fid.grid(True, alpha=0.25)
    ax_err.set_title("Driven replay |ΔE_total|")
    ax_err.set_ylabel("|ΔE_total|")
    ax_err.grid(True, alpha=0.25)
    ax_noisy_e.set_title("Driven noisy-minus-ideal total energy")
    ax_noisy_e.set_xlabel("Time")
    ax_noisy_e.set_ylabel("E_noisy - E_ideal")
    ax_noisy_e.grid(True, alpha=0.25)
    ax_noisy_stag.set_title("Driven noisy-minus-ideal staggered density")
    ax_noisy_stag.set_xlabel("Time")
    ax_noisy_stag.set_ylabel("staggered_noisy - staggered_ideal")
    ax_noisy_stag.grid(True, alpha=0.25)
    handles, labels = ax_fid.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(entries)), fontsize=9)
    fig.suptitle("Driven replay baseline overlay", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


"run_page(entry) = driven replay + noisy traces"
def _render_run_page(pdf: Any, entry: WorkflowReportEntry) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.5), sharex=True)
    ax_fid, ax_energy, ax_err, ax_noisy_e, ax_stag, ax_doublon = axes.ravel()

    ax_fid.plot(entry.baseline.times, entry.baseline.fidelity, color="#1f77b4", linewidth=2.0)
    ax_fid.set_title("Driven replay fidelity vs exact reference")
    ax_fid.set_ylabel("Fidelity")
    ax_fid.grid(True, alpha=0.25)

    ax_energy.plot(
        entry.baseline.times,
        entry.baseline.energy_total_exact,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label="exact replay ref",
    )
    ax_energy.plot(
        entry.baseline.times,
        entry.baseline.energy_total_trotter,
        color="#1f77b4",
        linewidth=2.0,
        label="baseline replay",
    )
    ax_energy.set_title("Driven replay energy traces")
    ax_energy.set_ylabel("Energy")
    ax_energy.grid(True, alpha=0.25)
    ax_energy.legend(fontsize=8, loc="best")

    ax_err.semilogy(
        entry.baseline.times,
        _positive_floor(entry.baseline.abs_energy_total_error),
        color="#9467bd",
        linewidth=2.0,
    )
    ax_err.set_title("Driven replay |ΔE_total|")
    ax_err.set_ylabel("|ΔE_total|")
    ax_err.grid(True, alpha=0.25)

    ax_noisy_e.plot(entry.noisy.times, entry.noisy.energy_total_ideal, color="#2ca02c", linewidth=1.8, label="ideal")
    ax_noisy_e.plot(entry.noisy.times, entry.noisy.energy_total_noisy, color="#ff7f0e", linewidth=2.0, label="noisy")
    ax_noisy_e.set_title("Driven noisy total energy")
    ax_noisy_e.set_ylabel("Energy")
    ax_noisy_e.grid(True, alpha=0.25)
    ax_noisy_e.legend(fontsize=8, loc="best")

    ax_stag.plot(entry.noisy.times, entry.noisy.staggered_ideal, color="#2ca02c", linewidth=1.8, label="ideal")
    ax_stag.plot(entry.noisy.times, entry.noisy.staggered_noisy, color="#ff7f0e", linewidth=2.0, label="noisy")
    ax_stag.set_title("Driven staggered density")
    ax_stag.set_xlabel("Time")
    ax_stag.set_ylabel("Staggered")
    ax_stag.grid(True, alpha=0.25)
    ax_stag.legend(fontsize=8, loc="best")

    ax_doublon.plot(entry.noisy.times, entry.noisy.doublon_ideal, color="#2ca02c", linewidth=1.8, label="ideal")
    ax_doublon.plot(entry.noisy.times, entry.noisy.doublon_noisy, color="#ff7f0e", linewidth=2.0, label="noisy")
    ax_doublon.set_title("Driven doublon response")
    ax_doublon.set_xlabel("Time")
    ax_doublon.set_ylabel("Doublon")
    ax_doublon.grid(True, alpha=0.25)
    ax_doublon.legend(fontsize=8, loc="best")

    fig.suptitle(f"{entry.label} | driven replay baseline", fontsize=13, y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


"controller_page(entry) = telemetry plots if available, else diagnostic text"
def _render_controller_page(pdf: Any, entry: WorkflowReportEntry) -> None:
    controller = entry.controller
    if not controller.time_axis.size:
        lines = [
            f"{entry.label} | controller telemetry unavailable",
            "",
            f"status: {controller.status}",
            f"reason: {controller.reason}",
            f"append_count: {controller.append_count}",
            f"stay_count: {controller.stay_count}",
            f"oracle_decision_checkpoints: {controller.oracle_decision_checkpoints}",
            f"exact_decision_checkpoints: {controller.exact_decision_checkpoints}",
            "",
            "Backend-scheduled compile summary",
            f"- stdout_log: {controller.compile_summary.log_path}",
            f"- compile_start_count: {controller.compile_summary.compile_start_count}",
            f"- compile_done_count: {controller.compile_summary.compile_done_count}",
            f"- compile_cache_hit_count: {controller.compile_summary.compile_cache_hit_count}",
            f"- unique_circuit_count: {controller.compile_summary.unique_circuit_count}",
            f"- mean_two_qubit_count: {controller.compile_summary.mean_two_qubit_count}",
            f"- max_two_qubit_count: {controller.compile_summary.max_two_qubit_count}",
            f"- mean_depth: {controller.compile_summary.mean_depth}",
            f"- max_depth: {controller.compile_summary.max_depth}",
            f"- first_ts_utc: {controller.compile_summary.first_ts_utc}",
            f"- last_ts_utc: {controller.compile_summary.last_ts_utc}",
            "",
            "Interpretation",
            "- This run does not contain an adaptive controller trajectory yet.",
            "- The replay/noisy pages in this PDF are still valid driven baseline diagnostics.",
            "- A true controller-efficacy verdict requires a completed adaptive_realtime_checkpoint trajectory.",
        ]
        render_text_page(pdf, lines, fontsize=10, line_spacing=0.03)
        return

    require_matplotlib()
    plt = get_plt()
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.5), sharex=True)
    ax_fid, ax_err, ax_rho, ax_size, ax_energy, ax_table = axes.ravel()

    ax_fid.plot(controller.time_axis, controller.fidelity_exact, color="#1f77b4", linewidth=2.0)
    ax_fid.set_title("Controller fidelity vs exact reference")
    ax_fid.set_ylabel("Fidelity")
    ax_fid.grid(True, alpha=0.25)

    ax_err.semilogy(
        controller.time_axis,
        _positive_floor(controller.abs_energy_total_error),
        color="#9467bd",
        linewidth=2.0,
    )
    ax_err.set_title("Controller |ΔE_total|")
    ax_err.set_ylabel("|ΔE_total|")
    ax_err.grid(True, alpha=0.25)

    ax_rho.plot(controller.time_axis, controller.rho_miss, color="#2ca02c", linewidth=2.0, label="rho_miss")
    ax_rho.plot(
        controller.time_axis,
        controller.motion_kink_score,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label="motion_kink_score",
    )
    ax_rho.set_title("Controller motion regime diagnostics")
    ax_rho.set_ylabel("rho_miss / kink")
    ax_rho.grid(True, alpha=0.25)
    ax_rho.legend(fontsize=8, loc="best")

    ax_size.step(
        controller.time_axis,
        controller.logical_block_count,
        where="mid",
        color="#8c564b",
        linewidth=2.0,
        label="logical blocks",
    )
    ax_size.step(
        controller.time_axis,
        controller.runtime_parameter_count,
        where="mid",
        color="#ff7f0e",
        linewidth=2.0,
        label="runtime params",
    )
    ax_size.set_title("Controller ansatz size proxy")
    ax_size.set_ylabel("Count")
    ax_size.grid(True, alpha=0.25)
    ax_size.legend(fontsize=8, loc="best")

    ax_energy.plot(
        controller.time_axis,
        controller.stay_noisy_energy_mean,
        color="#7f7f7f",
        linewidth=1.8,
        linestyle="--",
        label="stay noisy mean",
    )
    ax_energy.plot(
        controller.time_axis,
        controller.selected_noisy_energy_mean,
        color="#1f77b4",
        linewidth=2.0,
        label="selected noisy mean",
    )
    for t_val, action in zip(controller.time_axis.tolist(), controller.action_kinds):
        if action == "append":
            ax_energy.axvline(float(t_val), color="#2ca02c", linewidth=1.0, alpha=0.35)
        elif action == "stay":
            ax_energy.axvline(float(t_val), color="#d62728", linewidth=1.0, alpha=0.25)
    ax_energy.set_title("Controller stay-vs-selected noisy energy")
    ax_energy.set_xlabel("Time")
    ax_energy.set_ylabel("Noisy energy mean")
    ax_energy.grid(True, alpha=0.25)
    ax_energy.legend(fontsize=8, loc="best")

    render_compact_table(
        ax_table,
        title="Spend, reuse, and hardware-cost proxies",
        col_labels=["Metric", "Value"],
        rows=[
            ["status", controller.status],
            ["append / stay", f"{controller.append_count} / {controller.stay_count}"],
            ["oracle / exact checkpoints", f"{controller.oracle_decision_checkpoints} / {controller.exact_decision_checkpoints}"],
            ["oracle cache hits / misses", f"{controller.oracle_cache_hits} / {controller.oracle_cache_misses}"],
            ["exact cache hits / misses", f"{controller.exact_cache_hits} / {controller.exact_cache_misses}"],
            ["geometry memo hits / misses", f"{controller.geometry_memo_hits} / {controller.geometry_memo_misses}"],
            ["group cache hits / misses", f"{controller.raw_group_cache_hits} / {controller.raw_group_cache_misses}"],
            ["compile done / cache hit", f"{controller.compile_summary.compile_done_count} / {controller.compile_summary.compile_cache_hit_count}"],
            ["mean / max 2Q count", f"{controller.compile_summary.mean_two_qubit_count:.1f} / {controller.compile_summary.max_two_qubit_count}"],
            ["mean / max depth", f"{controller.compile_summary.mean_depth:.1f} / {controller.compile_summary.max_depth}"],
        ],
        fontsize=7,
    )

    fig.suptitle(f"{entry.label} | adaptive controller telemetry", fontsize=13, y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


"write_pdf(entries, output_pdf)"
def write_staged_controller_dynamics_pdf(
    input_jsons: Sequence[str | Path],
    output_pdf: str | Path,
    *,
    run_command: str | None = None,
) -> Path:
    require_matplotlib()
    entries = [load_workflow_report_entry(path) for path in input_jsons]
    _validate_compatibility(entries)
    output_path = Path(output_pdf).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = entries[0]
    PdfPages = get_PdfPages()
    plt = get_plt()
    with PdfPages(str(output_path)) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz="staged HH replay baseline + adaptive realtime checkpoint controller",
            drive_enabled=bool(first.dynamics.get("enable_drive", False)),
            t=float(first.physics.get("t", 0.0)),
            U=float(first.physics.get("u", 0.0)),
            dv=float(first.physics.get("dv", 0.0)),
            extra=_manifest_extra(entries),
            command=run_command or current_command_string(),
        )
        render_text_page(pdf, _overview_lines(entries), fontsize=10, line_spacing=0.03)

        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        render_compact_table(
            ax,
            title="Driven headline metrics",
            col_labels=[
                "Run",
                "Ctrl status",
                "Replay final fidelity",
                "Replay final |ΔE|",
                "Noisy final ΔE",
                "Ctrl final fidelity",
                "Ctrl final |ΔE|",
                "Append/Stay",
            ],
            rows=[_headline_row(entry) for entry in entries],
            fontsize=8,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        _render_drive_page(pdf, entries)
        if len(entries) > 1:
            _render_overlay_page(pdf, entries)
        for entry in entries:
            _render_run_page(pdf, entry)
            _render_controller_page(pdf, entry)
        render_command_page(
            pdf,
            run_command or current_command_string(),
            script_name="pipelines/hardcoded/hh_staged_controller_dynamics_report.py",
            extra_header_lines=[
                f"output_pdf: {output_path}",
                *[f"input_json[{idx}]: {entry.json_path}" for idx, entry in enumerate(entries)],
            ],
        )
    return output_path


"argv -> cli_config"
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a physics-facing PDF for driven HH staged controller time-dynamics artifacts.",
    )
    parser.add_argument(
        "--input-json",
        action="append",
        required=True,
        help="Staged HH noise workflow JSON artifact. Repeat for multi-run comparisons.",
    )
    parser.add_argument(
        "--output-pdf",
        required=True,
        help="Output PDF path.",
    )
    return parser.parse_args(argv)


"main = write_pdf(parse_args(argv))"
def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return write_staged_controller_dynamics_pdf(
        input_jsons=args.input_json,
        output_pdf=args.output_pdf,
        run_command=current_command_string(),
    )


if __name__ == "__main__":
    main()
