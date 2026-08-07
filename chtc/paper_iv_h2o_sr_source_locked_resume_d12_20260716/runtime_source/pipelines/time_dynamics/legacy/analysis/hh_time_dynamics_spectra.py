#!/usr/bin/env python3
"""Windowed spectral post-processing for HH time-dynamics JSON artifacts.

V1 scope:
- current controller JSON with top-level `trajectory`
- staged controller JSON with nested `adaptive_realtime_checkpoint.trajectory`
- one-sided amplitude spectra for staggered density, pair imbalance, and
  per-site fluctuation signals
- JSON summary + PNG figure outputs
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pipelines.time_dynamics.legacy.checkpoint_types import (
    high_miss_no_admit_diagnostic_counts,
    high_miss_no_admit_soft_fallback_counts,
    physical_trajectory_rows,
    trajectory_repair_counts,
)


@dataclass(frozen=True)
class LoadedTrajectoryPayload:
    source_schema: str
    input_json: Path
    run_tag: str | None
    time_key: str
    times: np.ndarray
    site_occupations: np.ndarray
    site_occupations_exact: np.ndarray | None
    energy_total: np.ndarray | None
    energy_total_exact: np.ndarray | None
    staggered: np.ndarray | None
    staggered_exact: np.ndarray | None
    doublon: np.ndarray | None
    doublon_exact: np.ndarray | None
    drive_omega: float | None
    drive_amplitude: float | None
    raw_payload: dict[str, Any]
    raw_trajectory_row_count: int = 0
    repair_event_row_count: int = 0
    trajectory_state_sample_count: int = 0
    high_miss_no_admit_soft_fallback_count: int = 0
    high_miss_count: int = 0
    high_miss_no_admit_count: int = 0
    append_no_harm_veto_count: int = 0
    high_miss_no_admit_reason_counts: dict[str, int] = field(default_factory=dict)
    append_no_harm_veto_reason_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SpectrumResult:
    dt: float
    frequencies: np.ndarray
    omega: np.ndarray
    amplitude: np.ndarray
    detrended_signal: np.ndarray
    windowed_signal: np.ndarray
    top_peaks: list[dict[str, float]]
    harmonic_fit: list[dict[str, float]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute windowed one-sided amplitude spectra for HH controller "
            "time-dynamics JSON artifacts."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument(
        "--time-key",
        choices=["time", "physical_time", "auto"],
        default="time",
        help="Time axis to analyze. Use 'time' by default for current controller plots.",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Optional site-pair difference i,j. Defaults to 0,1 for two-site payloads.",
    )
    parser.add_argument(
        "--detrend",
        choices=["constant", "linear"],
        default="constant",
        help="Temporal detrending method before windowing.",
    )
    parser.add_argument(
        "--window",
        choices=["hann", "none"],
        default="hann",
        help="Taper window for FFT amplitude spectra.",
    )
    parser.add_argument("--max-peaks", type=int, default=5)
    parser.add_argument("--max-harmonic", type=int, default=3)
    parser.add_argument(
        "--plot-max-omega",
        type=float,
        default=None,
        help="Optional x-axis cap for spectrum panels.",
    )
    parser.add_argument(
        "--plot-max-omega-energy",
        type=float,
        default=None,
        help="Optional x-axis cap for the energy spectrum panel. Falls back to --plot-max-omega.",
    )
    parser.add_argument(
        "--plot-max-omega-primary",
        type=float,
        default=None,
        help="Optional x-axis cap for the primary-imbalance spectrum panel. Falls back to --plot-max-omega.",
    )
    parser.add_argument(
        "--plot-max-omega-error",
        type=float,
        default=None,
        help="Optional x-axis cap for the controller-minus-exact error spectrum panel. Falls back to --plot-max-omega.",
    )
    return parser


def _default_output_paths(input_json: Path) -> tuple[Path, Path]:
    stem = input_json.with_suffix("")
    return stem.with_name(f"{stem.name}_spectra.json"), stem.with_name(f"{stem.name}_spectra.png")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected top-level JSON object in {path}")
    return payload


def _resolve_controller_rows(payload: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]], Mapping[str, Any]]:
    direct_rows = payload.get("trajectory")
    if isinstance(direct_rows, list) and all(isinstance(row, Mapping) for row in direct_rows):
        return "controller_top_level", list(direct_rows), payload

    nested = payload.get("adaptive_realtime_checkpoint")
    if isinstance(nested, Mapping):
        nested_rows = nested.get("trajectory")
        if isinstance(nested_rows, list) and all(isinstance(row, Mapping) for row in nested_rows):
            return "staged_adaptive_realtime_checkpoint", list(nested_rows), nested

    raise ValueError("Could not find controller trajectory rows in input JSON.")


def _select_time_key(rows: Sequence[Mapping[str, Any]], requested: str) -> str:
    if not rows:
        raise ValueError("Cannot select time key from empty trajectory.")
    if requested != "auto":
        if requested not in rows[0]:
            raise ValueError(f"Requested time key '{requested}' is missing from trajectory rows.")
        return requested
    if "time" in rows[0]:
        return "time"
    if "physical_time" in rows[0]:
        return "physical_time"
    raise ValueError("Neither 'time' nor 'physical_time' exists in trajectory rows.")


def _extract_numeric_series(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray | None:
    values: list[float] = []
    for row in rows:
        if key not in row or row.get(key) is None:
            return None
        values.append(float(row[key]))
    return np.asarray(values, dtype=float)


def _extract_site_matrix(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray | None:
    matrix: list[list[float]] = []
    width: int | None = None
    for row in rows:
        raw = row.get(key)
        if not isinstance(raw, list):
            return None
        if width is None:
            width = len(raw)
        if len(raw) != width:
            raise ValueError(f"Inconsistent site vector width for key '{key}'.")
        matrix.append([float(x) for x in raw])
    return np.asarray(matrix, dtype=float)


def load_trajectory_payload(input_json: Path, *, time_key: str = "time") -> LoadedTrajectoryPayload:
    payload = _load_json(input_json)
    schema, rows_raw, container = _resolve_controller_rows(payload)
    repair_counts = trajectory_repair_counts(rows_raw)
    soft_fallback_counts = high_miss_no_admit_soft_fallback_counts(rows_raw)
    high_miss_no_admit_counts = high_miss_no_admit_diagnostic_counts(rows_raw)
    rows = physical_trajectory_rows(rows_raw)
    chosen_time_key = _select_time_key(rows, time_key)

    times = _extract_numeric_series(rows, chosen_time_key)
    if times is None:
        raise ValueError(f"Missing time series for key '{chosen_time_key}'.")

    site_occupations = _extract_site_matrix(rows, "site_occupations")
    if site_occupations is None:
        raise ValueError("Current V1 spectra tool requires 'site_occupations' in controller rows.")

    site_occupations_exact = _extract_site_matrix(rows, "site_occupations_exact")
    energy_total = _extract_numeric_series(rows, "energy_total")
    energy_total_exact = _extract_numeric_series(rows, "energy_total_exact")
    staggered = _extract_numeric_series(rows, "staggered")
    staggered_exact = _extract_numeric_series(rows, "staggered_exact")
    doublon = _extract_numeric_series(rows, "doublon")
    doublon_exact = _extract_numeric_series(rows, "doublon_exact")

    drive_meta: dict[str, Any] = {}
    if isinstance(container.get("reference"), Mapping):
        ref = container.get("reference", {})
        drive_profile = ref.get("drive_profile") if isinstance(ref.get("drive_profile"), Mapping) else {}
        if isinstance(drive_profile, Mapping):
            drive_meta.update(
                {
                    "drive_omega": drive_profile.get("drive_omega", drive_profile.get("omega")),
                    "drive_A": drive_profile.get("drive_A", drive_profile.get("A")),
                }
            )
    top_drive = payload.get("drive") if isinstance(payload.get("drive"), Mapping) else {}
    if isinstance(top_drive, Mapping):
        drive_meta.update(
            {
                "drive_omega": top_drive.get("drive_omega", top_drive.get("omega", drive_meta.get("drive_omega"))),
                "drive_A": top_drive.get("drive_A", top_drive.get("A", drive_meta.get("drive_A"))),
            }
        )

    run_tag = payload.get("run_tag")
    return LoadedTrajectoryPayload(
        source_schema=str(schema),
        input_json=input_json,
        run_tag=None if run_tag is None else str(run_tag),
        time_key=str(chosen_time_key),
        times=np.asarray(times, dtype=float),
        site_occupations=np.asarray(site_occupations, dtype=float),
        site_occupations_exact=None if site_occupations_exact is None else np.asarray(site_occupations_exact, dtype=float),
        energy_total=None if energy_total is None else np.asarray(energy_total, dtype=float),
        energy_total_exact=None if energy_total_exact is None else np.asarray(energy_total_exact, dtype=float),
        staggered=None if staggered is None else np.asarray(staggered, dtype=float),
        staggered_exact=None if staggered_exact is None else np.asarray(staggered_exact, dtype=float),
        doublon=None if doublon is None else np.asarray(doublon, dtype=float),
        doublon_exact=None if doublon_exact is None else np.asarray(doublon_exact, dtype=float),
        drive_omega=None if drive_meta.get("drive_omega") is None else float(drive_meta["drive_omega"]),
        drive_amplitude=None if drive_meta.get("drive_A") is None else float(drive_meta["drive_A"]),
        raw_payload=payload,
        raw_trajectory_row_count=int(repair_counts["raw_trajectory_row_count"]),
        repair_event_row_count=int(repair_counts["repair_event_row_count"]),
        trajectory_state_sample_count=int(repair_counts["trajectory_state_sample_count"]),
        high_miss_no_admit_soft_fallback_count=int(
            soft_fallback_counts["high_miss_no_admit_soft_fallback_count"]
        ),
        high_miss_count=int(high_miss_no_admit_counts["high_miss_count"]),
        high_miss_no_admit_count=int(high_miss_no_admit_counts["high_miss_no_admit_count"]),
        append_no_harm_veto_count=int(high_miss_no_admit_counts["append_no_harm_veto_count"]),
        high_miss_no_admit_reason_counts=dict(
            high_miss_no_admit_counts["high_miss_no_admit_reason_counts"]
        ),
        append_no_harm_veto_reason_counts=dict(
            high_miss_no_admit_counts["append_no_harm_veto_reason_counts"]
        ),
    )


def _infer_uniform_dt(times: np.ndarray, *, tol: float = 1.0e-10) -> float:
    if times.ndim != 1 or times.size < 2:
        raise ValueError("Need at least two time samples for spectral analysis.")
    dt = np.diff(times)
    dt0 = float(dt[0])
    if dt0 <= 0.0:
        raise ValueError("Time grid must be strictly increasing.")
    if not np.allclose(dt, dt0, atol=tol, rtol=tol):
        raise ValueError("Time grid is not uniform; current V1 tool expects uniform sampling.")
    return dt0


def _parse_pair(pair_text: str | None, *, num_sites: int) -> tuple[int, int] | None:
    if pair_text is None:
        return (0, 1) if int(num_sites) == 2 else None
    parts = [part.strip() for part in str(pair_text).split(",")]
    if len(parts) != 2:
        raise ValueError("--pair must have the form i,j")
    left, right = int(parts[0]), int(parts[1])
    if left == right:
        raise ValueError("--pair indices must be distinct.")
    if not (0 <= left < num_sites and 0 <= right < num_sites):
        raise ValueError(f"--pair indices must lie in [0, {num_sites - 1}]")
    return left, right


_SPATIAL_FLUCTUATION_FORMULA = "delta_n_j(t) = n_j(t) - (1/L) * sum_m n_m(t)"


def build_site_fluctuation_signals(site_occupations: np.ndarray) -> np.ndarray:
    site_occ = np.asarray(site_occupations, dtype=float)
    if site_occ.ndim != 2:
        raise ValueError("site_occupations must be a 2D array.")
    return np.asarray(site_occ - np.mean(site_occ, axis=1, keepdims=True), dtype=float)


_PAIR_IMBALANCE_FORMULA = "d_ij(t) = n_i(t) - n_j(t)"


def build_pair_difference_signal(site_occupations: np.ndarray, *, pair: tuple[int, int]) -> np.ndarray:
    site_occ = np.asarray(site_occupations, dtype=float)
    left, right = int(pair[0]), int(pair[1])
    return np.asarray(site_occ[:, left] - site_occ[:, right], dtype=float)


_STAGGERED_FORMULA = "m(t) = (1/L) * sum_j (-1)^j n_j(t)"


def build_staggered_signal(site_occupations: np.ndarray) -> np.ndarray:
    site_occ = np.asarray(site_occupations, dtype=float)
    num_sites = int(site_occ.shape[1])
    signs = np.asarray([1.0 if (j % 2 == 0) else -1.0 for j in range(num_sites)], dtype=float)
    return np.asarray((site_occ @ signs) / float(num_sites), dtype=float)


_DETREND_FORMULA = "x_fluct(t) = x(t) - <x> or x(t) - (a t + b)"


def detrend_signal(times: np.ndarray, signal: np.ndarray, *, method: str) -> np.ndarray:
    x = np.asarray(signal, dtype=float).reshape(-1)
    t = np.asarray(times, dtype=float).reshape(-1)
    if x.size != t.size:
        raise ValueError("Signal/time size mismatch in detrending.")
    if str(method) == "constant":
        return np.asarray(x - float(np.mean(x)), dtype=float)
    if str(method) == "linear":
        coeffs = np.polyfit(t, x, deg=1)
        trend = coeffs[0] * t + coeffs[1]
        residual = x - trend
        return np.asarray(residual - float(np.mean(residual)), dtype=float)
    raise ValueError(f"Unsupported detrend method '{method}'.")


def _window_weights(num_samples: int, *, window: str) -> np.ndarray:
    if int(num_samples) < 1:
        raise ValueError("num_samples must be >= 1")
    if str(window) == "hann":
        if int(num_samples) < 3:
            return np.ones(int(num_samples), dtype=float)
        return np.hanning(int(num_samples))
    if str(window) == "none":
        return np.ones(int(num_samples), dtype=float)
    raise ValueError(f"Unsupported window '{window}'.")


def _top_peaks(omega: np.ndarray, amplitude: np.ndarray, *, max_peaks: int) -> list[dict[str, float]]:
    if int(max_peaks) <= 0:
        return []
    positive_idx = [int(i) for i in range(len(omega)) if float(omega[i]) > 0.0]
    ranked = sorted(positive_idx, key=lambda idx: float(amplitude[idx]), reverse=True)
    peaks: list[dict[str, float]] = []
    for idx in ranked[: int(max_peaks)]:
        peaks.append(
            {
                "index": float(idx),
                "omega": float(omega[idx]),
                "frequency": float(omega[idx] / (2.0 * math.pi)),
                "amplitude": float(amplitude[idx]),
            }
        )
    return peaks


_FFT_FORMULA = "A_k = 2 |X_k| / sum_n w_n, X_k = sum_n w_n x_n exp(-i 2 pi k n / N)"


def compute_one_sided_amplitude_spectrum(
    times: np.ndarray,
    signal: np.ndarray,
    *,
    detrend: str = "constant",
    window: str = "hann",
    max_peaks: int = 5,
    drive_omega: float | None = None,
    max_harmonic: int = 3,
) -> SpectrumResult:
    t = np.asarray(times, dtype=float).reshape(-1)
    x = np.asarray(signal, dtype=float).reshape(-1)
    if t.size != x.size:
        raise ValueError("Signal/time size mismatch in spectrum computation.")
    dt = _infer_uniform_dt(t)
    detrended = detrend_signal(t, x, method=str(detrend))
    weights = _window_weights(int(x.size), window=str(window))
    weighted_mean = float(np.sum(weights * detrended) / np.sum(weights))
    centered = np.asarray(detrended - weighted_mean, dtype=float)
    windowed_signal = np.asarray(centered * weights, dtype=float)

    fft_vals = np.fft.rfft(windowed_signal)
    frequencies = np.fft.rfftfreq(windowed_signal.size, d=dt)
    omega = 2.0 * math.pi * frequencies
    amplitude = np.abs(fft_vals) / max(float(np.sum(weights)), 1.0e-15)
    if amplitude.size > 1:
        if windowed_signal.size % 2 == 0:
            if amplitude.size > 2:
                amplitude[1:-1] *= 2.0
        else:
            amplitude[1:] *= 2.0

    harmonic_fit: list[dict[str, float]] = []
    if drive_omega is not None and float(drive_omega) > 0.0 and int(max_harmonic) >= 1:
        harmonic_fit = harmonic_regression(
            t,
            centered,
            drive_omega=float(drive_omega),
            max_harmonic=int(max_harmonic),
        )

    return SpectrumResult(
        dt=float(dt),
        frequencies=np.asarray(frequencies, dtype=float),
        omega=np.asarray(omega, dtype=float),
        amplitude=np.asarray(amplitude, dtype=float),
        detrended_signal=np.asarray(centered, dtype=float),
        windowed_signal=np.asarray(windowed_signal, dtype=float),
        top_peaks=_top_peaks(omega, amplitude, max_peaks=int(max_peaks)),
        harmonic_fit=harmonic_fit,
    )


_HARMONIC_REGRESSION_FORMULA = (
    "x(t) ~= c0 + sum_n [a_n cos(n omega_d t) + b_n sin(n omega_d t)]"
)


def harmonic_regression(
    times: np.ndarray,
    signal: np.ndarray,
    *,
    drive_omega: float,
    max_harmonic: int,
) -> list[dict[str, float]]:
    t = np.asarray(times, dtype=float).reshape(-1)
    y = np.asarray(signal, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError("Signal/time size mismatch in harmonic regression.")
    if t.size < 3:
        return []
    columns = [np.ones_like(t)]
    for harmonic in range(1, int(max_harmonic) + 1):
        omega_n = float(harmonic) * float(drive_omega)
        columns.append(np.cos(omega_n * t))
        columns.append(np.sin(omega_n * t))
    design = np.column_stack(columns)
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    results: list[dict[str, float]] = []
    for harmonic in range(1, int(max_harmonic) + 1):
        a = float(coeffs[1 + 2 * (harmonic - 1)])
        b = float(coeffs[2 + 2 * (harmonic - 1)])
        amplitude = float(math.hypot(a, b))
        phase = float(math.atan2(b, a))
        results.append(
            {
                "harmonic": float(harmonic),
                "omega": float(harmonic) * float(drive_omega),
                "frequency": float(harmonic) * float(drive_omega) / (2.0 * math.pi),
                "cos_coeff": a,
                "sin_coeff": b,
                "amplitude": amplitude,
                "phase_radians": phase,
            }
        )
    return results


_MANIFEST_SKIP_KEYS = frozenset({"trajectory", "ledger", "raw_traces", "spectra"})


def _find_first_nested_with_path(value: Any, keys: Sequence[str], *, path: str = "") -> tuple[Any, str | None]:
    wanted = {str(key).lower() for key in keys}
    if isinstance(value, Mapping):
        for key, item in value.items():
            current_path = f"{path}.{key}" if path else str(key)
            if str(key).lower() in wanted and item is not None:
                return item, current_path
        for key, item in value.items():
            if str(key) in _MANIFEST_SKIP_KEYS:
                continue
            current_path = f"{path}.{key}" if path else str(key)
            found, found_path = _find_first_nested_with_path(item, keys, path=current_path)
            if found is not None:
                return found, found_path
    return None, None


def _find_first_nested(value: Any, keys: Sequence[str]) -> Any:
    found, _ = _find_first_nested_with_path(value, keys)
    return found


def _mapping_bool(mapping: Any, *keys: str) -> bool | None:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return bool(mapping[key])
    return None


def _numeric_manifest_value(raw: Mapping[str, Any], keys: Sequence[str]) -> tuple[Any, str | None]:
    value, source = _find_first_nested_with_path(raw, keys)
    if value is None:
        return None, None
    try:
        numeric = float(value)
    except Exception:
        return value, source
    if not math.isfinite(numeric):
        return None, None
    if abs(numeric - round(numeric)) <= 1.0e-9:
        return int(round(numeric)), source
    return numeric, source


def _load_manifest_artifact_payload(artifact_json: Any, *, base_dir: Path) -> dict[str, Any]:
    if artifact_json is None or artifact_json == "":
        return {}
    try:
        artifact_path = Path(str(artifact_json)).expanduser()
    except Exception:
        return {}

    candidates = (
        [artifact_path]
        if artifact_path.is_absolute()
        else [base_dir / artifact_path, Path.cwd() / artifact_path]
    )
    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            with candidate.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    return {}


def _manifest_lookup(raw: Mapping[str, Any], artifact_payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value = _find_first_nested(raw, keys)
    if value is not None:
        return value
    if artifact_payload:
        return _find_first_nested(artifact_payload, keys)
    return None


def _model_manifest_name(model_name: Any) -> str:
    if model_name is None or model_name == "":
        return "Hubbard-Holstein"
    normalized = str(model_name).strip().lower().replace("_", "-")
    if normalized in {"hh", "hubbard-holstein", "hubbardholstein"}:
        return "Hubbard-Holstein"
    if normalized == "hubbard":
        return "Hubbard"
    return str(model_name)


def _parameter_manifest_from_payload(payload: LoadedTrajectoryPayload) -> dict[str, Any]:
    raw = payload.raw_payload if isinstance(payload.raw_payload, Mapping) else {}
    artifact_payload = _load_manifest_artifact_payload(raw.get("artifact_json"), base_dir=payload.input_json.parent)
    route_config = raw.get("route_config") if isinstance(raw.get("route_config"), Mapping) else {}
    drive_config = raw.get("drive_config") if isinstance(raw.get("drive_config"), Mapping) else {}
    drive_payload = raw.get("drive") if isinstance(raw.get("drive"), Mapping) else {}
    loader_summary = raw.get("loader_summary") if isinstance(raw.get("loader_summary"), Mapping) else {}

    drive_enabled = _mapping_bool(route_config, "drive_enabled")
    if drive_enabled is None:
        drive_enabled = _mapping_bool(drive_config, "enabled")
    if drive_enabled is None:
        drive_enabled = _mapping_bool(drive_payload, "enabled")
    if drive_enabled is None and payload.drive_amplitude is not None:
        drive_enabled = bool(abs(float(payload.drive_amplitude)) > 0.0)

    ansatz_parts = []
    for key in ("resolved_family", "generator_family", "fallback_family", "handoff_state_kind"):
        if isinstance(loader_summary, Mapping) and loader_summary.get(key) not in {None, ""}:
            ansatz_parts.append(f"{key}={loader_summary[key]}")

    compile_2q, compile_2q_source = _numeric_manifest_value(
        raw,
        (
            "compiled_count_2q",
            "compiled_two_qubit_count",
            "transpiled_count_2q",
            "transpiled_two_qubit_count",
            "hw_2q",
        ),
    )
    compile_depth, compile_depth_source = _numeric_manifest_value(
        raw,
        (
            "compiled_depth",
            "transpiled_depth",
            "circuit_depth",
        ),
    )
    compile_size, compile_size_source = _numeric_manifest_value(
        raw,
        (
            "compiled_size",
            "transpiled_size",
            "circuit_size",
        ),
    )
    compile_backend = _find_first_nested(raw, ("backend_name", "transpile_backend", "target_backend"))
    transpile_seed = _find_first_nested(raw, ("transpile_seed", "seed_transpiler"))
    transpile_opt_level = _find_first_nested(raw, ("transpile_optimization_level", "optimization_level"))

    model_name = _manifest_lookup(raw, artifact_payload, ("model_family", "model_name", "problem", "problem_name"))
    return {
        "model_family_name": _model_manifest_name(model_name),
        "ansatz_types": ", ".join(ansatz_parts) if ansatz_parts else "unknown/not recorded",
        "drive_enabled": None if drive_enabled is None else bool(drive_enabled),
        "L": _manifest_lookup(raw, artifact_payload, ("L", "num_sites", "n_sites")),
        "t": _manifest_lookup(raw, artifact_payload, ("t", "hubbard_t", "hopping_t")),
        "U": _manifest_lookup(raw, artifact_payload, ("U", "u", "hubbard_U", "onsite_U")),
        "dv": _manifest_lookup(raw, artifact_payload, ("dv", "delta_v", "delta_v_ext", "bias_dv")),
        "omega0": _manifest_lookup(raw, artifact_payload, ("omega0", "omega_0", "phonon_omega")),
        "g_ep": _manifest_lookup(raw, artifact_payload, ("g_ep", "g_ep_coupling", "electron_phonon_g")),
        "n_ph_max": _manifest_lookup(raw, artifact_payload, ("n_ph_max", "nph_max", "n_phonon_max")),
        "time_final": None if payload.times.size == 0 else float(payload.times[-1]),
        "time_samples": int(payload.times.size),
        "drive_A": payload.drive_amplitude,
        "drive_omega": payload.drive_omega,
        "compiled_count_2q": compile_2q,
        "compiled_count_2q_source": compile_2q_source,
        "compiled_depth": compile_depth,
        "compiled_depth_source": compile_depth_source,
        "compiled_size": compile_size,
        "compiled_size_source": compile_size_source,
        "compile_backend": compile_backend,
        "transpile_seed": transpile_seed,
        "transpile_optimization_level": transpile_opt_level,
        "compile_note": (
            "not recorded for this run" if compile_2q is None and compile_depth is None else "recorded in payload"
        ),
        "artifact_json": raw.get("artifact_json") if isinstance(raw, Mapping) else None,
    }


def _analysis_trace_components(
    payload: LoadedTrajectoryPayload,
    *,
    pair: tuple[int, int] | None,
) -> tuple[int, dict[str, np.ndarray], dict[str, Any]]:
    site_occupations = np.asarray(payload.site_occupations, dtype=float)
    if site_occupations.ndim != 2:
        raise ValueError("site_occupations must be a 2D array.")
    num_sites = int(site_occupations.shape[1])
    site_occupations_exact = (
        None
        if payload.site_occupations_exact is None
        else np.asarray(payload.site_occupations_exact, dtype=float)
    )

    site_fluct = build_site_fluctuation_signals(site_occupations)
    site_fluct_exact = (
        None if site_occupations_exact is None else build_site_fluctuation_signals(site_occupations_exact)
    )

    staggered = (
        build_staggered_signal(site_occupations)
        if payload.staggered is None
        else np.asarray(payload.staggered, dtype=float)
    )
    staggered_exact = None
    if site_occupations_exact is not None:
        staggered_exact = (
            build_staggered_signal(site_occupations_exact)
            if payload.staggered_exact is None
            else np.asarray(payload.staggered_exact, dtype=float)
        )

    signal_map: dict[str, np.ndarray] = {"staggered": np.asarray(staggered, dtype=float)}
    if staggered_exact is not None:
        signal_map["staggered_exact"] = np.asarray(staggered_exact, dtype=float)
    if payload.energy_total is not None:
        signal_map["energy_total"] = np.asarray(payload.energy_total, dtype=float)
    if payload.energy_total_exact is not None:
        signal_map["energy_total_exact"] = np.asarray(payload.energy_total_exact, dtype=float)
    if payload.energy_total is not None and payload.energy_total_exact is not None:
        signal_map["energy_total_error"] = np.asarray(
            np.asarray(payload.energy_total, dtype=float) - np.asarray(payload.energy_total_exact, dtype=float),
            dtype=float,
        )
    if staggered_exact is not None:
        signal_map["staggered_error"] = np.asarray(
            np.asarray(staggered, dtype=float) - np.asarray(staggered_exact, dtype=float),
            dtype=float,
        )

    for site in range(num_sites):
        signal_map[f"site_occupation_{site}"] = np.asarray(site_occupations[:, site], dtype=float)
        if site_occupations_exact is not None:
            signal_map[f"site_occupation_{site}_exact"] = np.asarray(
                site_occupations_exact[:, site],
                dtype=float,
            )
        signal_map[f"site_fluctuation_{site}"] = np.asarray(site_fluct[:, site], dtype=float)
        if site_fluct_exact is not None:
            signal_map[f"site_fluctuation_{site}_exact"] = np.asarray(site_fluct_exact[:, site], dtype=float)

    pair_signal = None
    pair_signal_exact = None
    if pair is not None:
        pair_signal = build_pair_difference_signal(site_occupations, pair=pair)
        signal_map[f"pair_difference_{pair[0]}_{pair[1]}"] = pair_signal
        if site_occupations_exact is not None:
            pair_signal_exact = build_pair_difference_signal(site_occupations_exact, pair=pair)
            signal_map[f"pair_difference_{pair[0]}_{pair[1]}_exact"] = pair_signal_exact
            signal_map[f"pair_difference_{pair[0]}_{pair[1]}_error"] = np.asarray(
                pair_signal - pair_signal_exact,
                dtype=float,
            )

    if payload.doublon is not None:
        signal_map["doublon"] = np.asarray(payload.doublon, dtype=float)
    if payload.doublon_exact is not None:
        signal_map["doublon_exact"] = np.asarray(payload.doublon_exact, dtype=float)

    raw_traces = {
        "times": [float(x) for x in payload.times.tolist()],
        "site_occupations": [[float(x) for x in row] for row in site_occupations.tolist()],
        "site_occupations_exact": None
        if site_occupations_exact is None
        else [[float(x) for x in row] for row in site_occupations_exact.tolist()],
        "energy_total": None if payload.energy_total is None else [float(x) for x in payload.energy_total.tolist()],
        "energy_total_exact": None
        if payload.energy_total_exact is None
        else [float(x) for x in payload.energy_total_exact.tolist()],
        "energy_total_error": None
        if payload.energy_total is None or payload.energy_total_exact is None
        else [
            float(x)
            for x in (
                np.asarray(payload.energy_total, dtype=float)
                - np.asarray(payload.energy_total_exact, dtype=float)
            ).tolist()
        ],
        "staggered": [float(x) for x in staggered.tolist()],
        "staggered_exact": None if staggered_exact is None else [float(x) for x in staggered_exact.tolist()],
        "staggered_error": None
        if staggered_exact is None
        else [float(x) for x in (np.asarray(staggered, dtype=float) - np.asarray(staggered_exact, dtype=float)).tolist()],
        "pair_difference": None if pair_signal is None else [float(x) for x in pair_signal.tolist()],
        "pair_difference_exact": None
        if pair_signal_exact is None
        else [float(x) for x in pair_signal_exact.tolist()],
        "pair_difference_error": None
        if pair_signal is None or pair_signal_exact is None
        else [float(x) for x in (pair_signal - pair_signal_exact).tolist()],
        "doublon": None if payload.doublon is None else [float(x) for x in payload.doublon.tolist()],
        "doublon_exact": None
        if payload.doublon_exact is None
        else [float(x) for x in payload.doublon_exact.tolist()],
    }
    return num_sites, signal_map, raw_traces


def _analysis_metadata(
    payload: LoadedTrajectoryPayload,
    *,
    num_sites: int,
    pair: tuple[int, int] | None,
    detrend: str,
    window: str,
    dt: float | None,
    analysis_status: str,
    analysis_error: str | None,
) -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_json": str(payload.input_json),
        "run_tag": payload.run_tag,
        "source_schema": payload.source_schema,
        "time_key": payload.time_key,
        "num_samples": int(payload.times.size),
        "raw_trajectory_row_count": int(payload.raw_trajectory_row_count),
        "repair_event_row_count": int(payload.repair_event_row_count),
        "trajectory_state_sample_count": int(payload.trajectory_state_sample_count),
        "high_miss_no_admit_soft_fallback_count": int(
            payload.high_miss_no_admit_soft_fallback_count
        ),
        "high_miss_count": int(payload.high_miss_count),
        "high_miss_no_admit_count": int(payload.high_miss_no_admit_count),
        "append_no_harm_veto_count": int(payload.append_no_harm_veto_count),
        "high_miss_no_admit_reason_counts": dict(payload.high_miss_no_admit_reason_counts),
        "append_no_harm_veto_reason_counts": dict(payload.append_no_harm_veto_reason_counts),
        "num_sites": int(num_sites),
        "dt": None if dt is None else float(dt),
        "t_initial": None if payload.times.size == 0 else float(payload.times[0]),
        "t_final": None if payload.times.size == 0 else float(payload.times[-1]),
        "window": str(window),
        "detrend": str(detrend),
        "drive_omega": None if payload.drive_omega is None else float(payload.drive_omega),
        "drive_amplitude": None if payload.drive_amplitude is None else float(payload.drive_amplitude),
        "nyquist_omega": None if dt is None else float(math.pi / dt),
        "delta_omega_bin": None
        if dt is None or payload.times.size == 0
        else float(2.0 * math.pi / (float(payload.times.size) * dt)),
        "pair_difference": None if pair is None else [int(pair[0]), int(pair[1])],
        "analysis_status": str(analysis_status),
        "analysis_error": analysis_error,
        "parameter_manifest": _parameter_manifest_from_payload(payload),
    }


def build_trace_only_analysis(
    payload: LoadedTrajectoryPayload,
    *,
    pair: tuple[int, int] | None,
    detrend: str,
    window: str,
    analysis_error: str,
    analysis_status: str = "spectra_unavailable",
) -> dict[str, Any]:
    num_sites, _, raw_traces = _analysis_trace_components(payload, pair=pair)
    return {
        "metadata": _analysis_metadata(
            payload,
            num_sites=num_sites,
            pair=pair,
            detrend=str(detrend),
            window=str(window),
            dt=None,
            analysis_status=str(analysis_status),
            analysis_error=str(analysis_error),
        ),
        "raw_traces": raw_traces,
        "spectra": {},
    }


def analyze_payload(
    payload: LoadedTrajectoryPayload,
    *,
    pair: tuple[int, int] | None,
    detrend: str,
    window: str,
    max_peaks: int,
    max_harmonic: int,
    allow_short: bool = False,
) -> dict[str, Any]:
    num_sites, signal_map, raw_traces = _analysis_trace_components(payload, pair=pair)
    try:
        dt = _infer_uniform_dt(payload.times)
    except Exception as exc:
        if not bool(allow_short):
            raise
        return build_trace_only_analysis(
            payload,
            pair=pair,
            detrend=str(detrend),
            window=str(window),
            analysis_error=str(exc),
            analysis_status=(
                "spectra_unavailable" if int(payload.times.size) < 2 else "spectra_failed"
            ),
        )

    spectra: dict[str, Any] = {}
    try:
        for name, signal in signal_map.items():
            result = compute_one_sided_amplitude_spectrum(
                payload.times,
                signal,
                detrend=str(detrend),
                window=str(window),
                max_peaks=int(max_peaks),
                drive_omega=payload.drive_omega,
                max_harmonic=int(max_harmonic),
            )
            spectra[name] = {
                "dt": float(result.dt),
                "omega": [float(x) for x in result.omega.tolist()],
                "frequency": [float(x) for x in result.frequencies.tolist()],
                "amplitude": [float(x) for x in result.amplitude.tolist()],
                "detrended_signal": [float(x) for x in result.detrended_signal.tolist()],
                "windowed_signal": [float(x) for x in result.windowed_signal.tolist()],
                "top_peaks": result.top_peaks,
                "harmonic_fit": result.harmonic_fit,
            }
    except Exception as exc:
        if not bool(allow_short):
            raise
        return build_trace_only_analysis(
            payload,
            pair=pair,
            detrend=str(detrend),
            window=str(window),
            analysis_error=str(exc),
            analysis_status="spectra_failed",
        )

    return {
        "metadata": _analysis_metadata(
            payload,
            num_sites=num_sites,
            pair=pair,
            detrend=str(detrend),
            window=str(window),
            dt=float(dt),
            analysis_status="ok",
            analysis_error=None,
        ),
        "raw_traces": raw_traces,
        "spectra": spectra,
    }


def _plot_drive_harmonics(ax: Any, *, drive_omega: float | None, max_harmonic: int, ymax: float) -> None:
    if drive_omega is None or float(drive_omega) <= 0.0:
        return
    for harmonic in range(1, int(max_harmonic) + 1):
        omega_n = float(harmonic) * float(drive_omega)
        ax.axvline(
            omega_n,
            color="#999999",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )
        ax.text(
            omega_n,
            ymax,
            f"{harmonic}ωd",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#666666",
        )


def _resolve_panel_plot_max_omega(
    *,
    common: float | None,
    energy: float | None = None,
    primary: float | None = None,
    error: float | None = None,
) -> dict[str, float | None]:
    return {
        "energy": float(common if energy is None else energy) if (common if energy is None else energy) is not None else None,
        "primary": float(common if primary is None else primary) if (common if primary is None else primary) is not None else None,
        "error": float(common if error is None else error) if (common if error is None else error) is not None else None,
    }


def _legend_if_any(ax: Any, *, fontsize: int = 8, loc: str = "best") -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize=fontsize, loc=loc)


def _no_data_label(metadata: Mapping[str, Any]) -> str:
    error = metadata.get("analysis_error")
    if error in {None, ""}:
        return "No spectral data available"
    return f"Spectral analysis unavailable\n{str(error)[:180]}"


def _mark_no_data(ax: Any, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
        wrap=True,
    )


def _plot_trace_series(
    ax: Any,
    times: np.ndarray,
    values: Any,
    *,
    label: str,
    color: Any,
    linewidth: float,
    linestyle: str = "-",
) -> bool:
    if values is None:
        return False
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return False
    x_axis = times if times.size == arr.size else np.arange(arr.size, dtype=float)
    ax.plot(x_axis, arr, color=color, linewidth=linewidth, linestyle=linestyle, label=label)
    return True


def _plot_spectrum_keys(
    ax: Any,
    spectra: Mapping[str, Any],
    keys: Sequence[str],
    *,
    max_harmonic: int,
    drive_omega: float | None,
    no_data_message: str,
) -> None:
    ymax = 0.0
    plotted = False
    for key in keys:
        if key not in spectra or not isinstance(spectra.get(key), Mapping):
            continue
        spec = spectra[key]
        omega = np.asarray(spec.get("omega", []), dtype=float)
        amplitude = np.asarray(spec.get("amplitude", []), dtype=float)
        if omega.size == 0 or amplitude.size == 0:
            continue
        ymax = max(ymax, float(np.max(amplitude)) if amplitude.size > 0 else 0.0)
        linestyle = "--" if str(key).endswith("_exact") else "-"
        ax.plot(omega, amplitude, linewidth=1.8, linestyle=linestyle, label=key)
        plotted = True
    if plotted:
        _plot_drive_harmonics(
            ax,
            drive_omega=drive_omega,
            max_harmonic=int(max_harmonic),
            ymax=max(ymax, 1.0e-12),
        )
        _legend_if_any(ax)
    else:
        _mark_no_data(ax, no_data_message)


def _build_spectrum_figure(
    analysis: Mapping[str, Any],
    *,
    max_harmonic: int,
    plot_max_omega: float | None = None,
    plot_max_omega_energy: float | None = None,
    plot_max_omega_primary: float | None = None,
    plot_max_omega_error: float | None = None,
) -> Any:
    metadata = analysis.get("metadata", {}) if isinstance(analysis.get("metadata", {}), Mapping) else {}
    traces = analysis.get("raw_traces", {}) if isinstance(analysis.get("raw_traces", {}), Mapping) else {}
    spectra = analysis.get("spectra", {}) if isinstance(analysis.get("spectra", {}), Mapping) else {}
    times = np.asarray(traces.get("times", []), dtype=float).reshape(-1)
    site_occ = np.asarray(traces.get("site_occupations", []), dtype=float)
    if site_occ.ndim == 1:
        site_occ = site_occ.reshape((-1, 1)) if site_occ.size else np.empty((0, 0), dtype=float)
    if times.size == 0 and site_occ.shape[0] > 0:
        times = np.arange(site_occ.shape[0], dtype=float)
    site_occ_exact = (
        None
        if traces.get("site_occupations_exact") is None
        else np.asarray(traces.get("site_occupations_exact"), dtype=float)
    )
    if site_occ_exact is not None and site_occ_exact.ndim == 1:
        site_occ_exact = site_occ_exact.reshape((-1, 1))
    panel_plot_max_omega = _resolve_panel_plot_max_omega(
        common=plot_max_omega,
        energy=plot_max_omega_energy,
        primary=plot_max_omega_primary,
        error=plot_max_omega_error,
    )
    time_label = str(metadata.get("time_key", "time"))
    no_data_message = _no_data_label(metadata)

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 13.0))
    (
        ax_energy,
        ax_occ,
        ax_primary,
        ax_spec_energy,
        ax_spec_primary,
        ax_spec_error,
    ) = axes.reshape(-1)

    energy_plotted = False
    energy_plotted |= _plot_trace_series(
        ax_energy,
        times,
        traces.get("energy_total"),
        label="energy_total",
        color="#2ca02c",
        linewidth=2.0,
    )
    energy_plotted |= _plot_trace_series(
        ax_energy,
        times,
        traces.get("energy_total_exact"),
        label="energy_total_exact",
        color="#2ca02c",
        linewidth=1.2,
        linestyle="--",
    )
    if not energy_plotted:
        _mark_no_data(ax_energy, "No total-energy trace available")
    ax_energy.set_title("Raw total energy")
    ax_energy.set_xlabel(time_label)
    ax_energy.set_ylabel("energy")
    ax_energy.grid(alpha=0.25)
    _legend_if_any(ax_energy)

    colors = plt.cm.tab10.colors
    if site_occ.size:
        for site in range(site_occ.shape[1]):
            color = colors[site % len(colors)]
            _plot_trace_series(
                ax_occ,
                times,
                site_occ[:, site],
                label=f"n_{site}",
                color=color,
                linewidth=1.8,
            )
            if site_occ_exact is not None and site_occ_exact.shape == site_occ.shape:
                _plot_trace_series(
                    ax_occ,
                    times,
                    site_occ_exact[:, site],
                    label=f"n_{site} exact",
                    color=color,
                    linewidth=1.2,
                    linestyle="--",
                )
    else:
        _mark_no_data(ax_occ, "No site-occupation trace available")
    ax_occ.set_title("Raw site occupations")
    ax_occ.set_xlabel(time_label)
    ax_occ.set_ylabel("occupation")
    ax_occ.grid(alpha=0.25)
    _legend_if_any(ax_occ)

    primary_plotted = False
    primary_plotted |= _plot_trace_series(
        ax_primary,
        times,
        traces.get("staggered"),
        label="staggered",
        color="#1f77b4",
        linewidth=2.0,
    )
    primary_plotted |= _plot_trace_series(
        ax_primary,
        times,
        traces.get("staggered_exact"),
        label="staggered exact",
        color="#1f77b4",
        linewidth=1.2,
        linestyle="--",
    )
    pair = metadata.get("pair_difference")
    if pair is not None:
        key = f"pair_difference_{pair[0]}_{pair[1]}"
        primary_plotted |= _plot_trace_series(
            ax_primary,
            times,
            traces.get("pair_difference"),
            label=key,
            color="#ff7f0e",
            linewidth=1.6,
        )
        primary_plotted |= _plot_trace_series(
            ax_primary,
            times,
            traces.get("pair_difference_exact"),
            label=f"{key} exact",
            color="#ff7f0e",
            linewidth=1.1,
            linestyle="--",
        )
    if not primary_plotted:
        _mark_no_data(ax_primary, "No primary imbalance trace available")
    ax_primary.set_title("Primary imbalance traces")
    ax_primary.set_xlabel(time_label)
    ax_primary.set_ylabel("signal")
    ax_primary.grid(alpha=0.25)
    _legend_if_any(ax_primary)

    _plot_spectrum_keys(
        ax_spec_energy,
        spectra,
        ["energy_total", "energy_total_exact"],
        max_harmonic=int(max_harmonic),
        drive_omega=metadata.get("drive_omega"),
        no_data_message=no_data_message,
    )
    ax_spec_energy.set_title("Energy one-sided amplitude spectra")
    ax_spec_energy.set_xlabel("angular frequency ω")
    ax_spec_energy.set_ylabel("amplitude")
    ax_spec_energy.grid(alpha=0.25)
    if panel_plot_max_omega["energy"] is not None:
        ax_spec_energy.set_xlim(0.0, float(panel_plot_max_omega["energy"]))

    primary_keys = ["staggered"]
    if traces.get("staggered_exact") is not None:
        primary_keys.append("staggered_exact")
    if pair is not None:
        primary_keys.append(f"pair_difference_{pair[0]}_{pair[1]}")
        primary_keys.append(f"pair_difference_{pair[0]}_{pair[1]}_exact")
    _plot_spectrum_keys(
        ax_spec_primary,
        spectra,
        primary_keys,
        max_harmonic=int(max_harmonic),
        drive_omega=metadata.get("drive_omega"),
        no_data_message=no_data_message,
    )
    ax_spec_primary.set_title("Primary one-sided amplitude spectra")
    ax_spec_primary.set_xlabel("angular frequency ω")
    ax_spec_primary.set_ylabel("amplitude")
    ax_spec_primary.grid(alpha=0.25)
    if panel_plot_max_omega["primary"] is not None:
        ax_spec_primary.set_xlim(0.0, float(panel_plot_max_omega["primary"]))

    error_keys = ["energy_total_error", "staggered_error"]
    if pair is not None:
        error_keys.append(f"pair_difference_{pair[0]}_{pair[1]}_error")
    _plot_spectrum_keys(
        ax_spec_error,
        spectra,
        error_keys,
        max_harmonic=int(max_harmonic),
        drive_omega=metadata.get("drive_omega"),
        no_data_message=no_data_message,
    )
    ax_spec_error.set_title("Controller-minus-exact error spectra")
    ax_spec_error.set_xlabel("angular frequency ω")
    ax_spec_error.set_ylabel("amplitude")
    ax_spec_error.grid(alpha=0.25)
    if panel_plot_max_omega["error"] is not None:
        ax_spec_error.set_xlim(0.0, float(panel_plot_max_omega["error"]))

    manifest = (
        metadata.get("parameter_manifest", {})
        if isinstance(metadata.get("parameter_manifest", {}), Mapping)
        else {}
    )
    compiled_2q = _manifest_value(manifest.get("compiled_count_2q"))
    compiled_depth = _manifest_value(manifest.get("compiled_depth"))
    compile_backend = _manifest_value(manifest.get("compile_backend"))
    compile_line = f"2Q={compiled_2q} | depth={compiled_depth} | backend={compile_backend}"
    input_name = Path(str(metadata.get("input_json", "input.json"))).name
    title_text = (
        f"HH time-dynamics spectra | {input_name} | "
        f"{metadata.get('window', 'unknown')} window, {metadata.get('detrend', 'unknown')} detrend | "
        f"{metadata.get('analysis_status', 'ok')} | {compile_line}"
    )
    title_top = _set_wrapped_suptitle(fig, title_text, fontsize=11)
    fig.text(
        0.01,
        0.006,
        f"Circuit cost: {compile_line}; source 2Q={_manifest_value(manifest.get('compiled_count_2q_source'))}; "
        f"source depth={_manifest_value(manifest.get('compiled_depth_source'))}; note={_manifest_value(manifest.get('compile_note'))}",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, title_top))
    return fig


def _set_wrapped_suptitle(
    fig: plt.Figure,
    title: str,
    *,
    fontsize: float = 11,
    width: int = 108,
) -> float:
    """Render a long figure title without clipping and return a safe layout top.

    The spectra page title includes run filenames/tags that can be much wider than the
    page. Insert line breaks instead of truncating so the report content stays intact.
    """

    wrapped_lines = textwrap.wrap(
        str(title),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [str(title)]
    wrapped_title = "\n".join(wrapped_lines)
    fig.suptitle(wrapped_title, fontsize=fontsize, y=0.992)
    return max(0.80, 0.955 - 0.035 * (len(wrapped_lines) - 1))


def _manifest_value(value: Any) -> str:
    if value is None:
        return "unknown/not recorded"
    if isinstance(value, str) and value == "":
        return "unknown/not recorded"
    return str(value)


def _wrap_manifest_lines(lines: Sequence[str], *, width: int = 86) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if line == "":
            wrapped.append("")
            continue
        wrapped.extend(
            textwrap.wrap(
                line,
                width=width,
                subsequent_indent="  " if line.startswith("- ") else "",
                break_long_words=True,
                break_on_hyphens=False,
            )
            or [line]
        )
    return wrapped


def _render_pdf_manifest_page(pdf: PdfPages, analysis: Mapping[str, Any]) -> None:
    metadata = analysis.get("metadata", {}) if isinstance(analysis.get("metadata", {}), Mapping) else {}
    manifest = (
        metadata.get("parameter_manifest", {})
        if isinstance(metadata.get("parameter_manifest", {}), Mapping)
        else {}
    )
    rows = [
        ("Model family/name", manifest.get("model_family_name")),
        ("Ansatz type(s) used", manifest.get("ansatz_types")),
        ("Drive enabled (--enable-drive)", manifest.get("drive_enabled")),
        ("L", manifest.get("L")),
        ("t", manifest.get("t")),
        ("U", manifest.get("U")),
        ("dv", manifest.get("dv")),
        ("omega0", manifest.get("omega0")),
        ("g_ep", manifest.get("g_ep")),
        ("n_ph_max", manifest.get("n_ph_max")),
        ("time_final", manifest.get("time_final")),
        ("time_samples", manifest.get("time_samples")),
        ("drive_A", manifest.get("drive_A")),
        ("drive_omega", manifest.get("drive_omega")),
        ("compiled 2Q gates", manifest.get("compiled_count_2q")),
        ("compiled 2Q source", manifest.get("compiled_count_2q_source")),
        ("circuit depth", manifest.get("compiled_depth")),
        ("circuit depth source", manifest.get("compiled_depth_source")),
        ("compiled size", manifest.get("compiled_size")),
        ("compile backend", manifest.get("compile_backend")),
        ("transpile seed", manifest.get("transpile_seed")),
        ("transpile optimization level", manifest.get("transpile_optimization_level")),
        ("compile note", manifest.get("compile_note")),
        ("run_tag", metadata.get("run_tag")),
        ("input_json", metadata.get("input_json")),
        ("artifact_json", manifest.get("artifact_json")),
        ("analysis_status", metadata.get("analysis_status")),
        ("analysis_error", metadata.get("analysis_error")),
    ]
    lines = [
        "HH time-dynamics spectra PDF",
        "Parameter manifest",
        "",
        *[f"- {label}: {_manifest_value(value)}" for label, value in rows],
    ]
    lines = _wrap_manifest_lines(lines)
    fig = plt.figure(figsize=(8.5, 11.0))
    fig.text(
        0.08,
        0.95,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def render_spectrum_png(
    analysis: Mapping[str, Any],
    *,
    output_png: Path,
    max_harmonic: int,
    plot_max_omega: float | None = None,
    plot_max_omega_energy: float | None = None,
    plot_max_omega_primary: float | None = None,
    plot_max_omega_error: float | None = None,
) -> None:
    fig = _build_spectrum_figure(
        analysis,
        max_harmonic=int(max_harmonic),
        plot_max_omega=plot_max_omega,
        plot_max_omega_energy=plot_max_omega_energy,
        plot_max_omega_primary=plot_max_omega_primary,
        plot_max_omega_error=plot_max_omega_error,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def render_spectrum_pdf(
    analysis: Mapping[str, Any],
    *,
    output_pdf: Path,
    max_harmonic: int,
    plot_max_omega: float | None = None,
    plot_max_omega_energy: float | None = None,
    plot_max_omega_primary: float | None = None,
    plot_max_omega_error: float | None = None,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        _render_pdf_manifest_page(pdf, analysis)
        fig = _build_spectrum_figure(
            analysis,
            max_harmonic=int(max_harmonic),
            plot_max_omega=plot_max_omega,
            plot_max_omega_energy=plot_max_omega_energy,
            plot_max_omega_primary=plot_max_omega_primary,
            plot_max_omega_error=plot_max_omega_error,
        )
        pdf.savefig(fig)
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_json, output_png = _default_output_paths(args.input_json)
    output_pdf = None if args.output_pdf is None else Path(args.output_pdf)
    if args.output_json is not None:
        output_json = Path(args.output_json)
    if args.output_png is not None:
        output_png = Path(args.output_png)

    payload = load_trajectory_payload(Path(args.input_json), time_key=str(args.time_key))
    pair = _parse_pair(args.pair, num_sites=int(payload.site_occupations.shape[1]))
    analysis = analyze_payload(
        payload,
        pair=pair,
        detrend=str(args.detrend),
        window=str(args.window),
        max_peaks=int(args.max_peaks),
        max_harmonic=int(args.max_harmonic),
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    render_spectrum_png(
        analysis,
        output_png=output_png,
        max_harmonic=int(args.max_harmonic),
        plot_max_omega=args.plot_max_omega,
        plot_max_omega_energy=args.plot_max_omega_energy,
        plot_max_omega_primary=args.plot_max_omega_primary,
        plot_max_omega_error=args.plot_max_omega_error,
    )
    if output_pdf is not None:
        render_spectrum_pdf(
            analysis,
            output_pdf=output_pdf,
            max_harmonic=int(args.max_harmonic),
            plot_max_omega=args.plot_max_omega,
            plot_max_omega_energy=args.plot_max_omega_energy,
            plot_max_omega_primary=args.plot_max_omega_primary,
            plot_max_omega_error=args.plot_max_omega_error,
        )

    print(f"Wrote spectrum JSON: {output_json}")
    print(f"Wrote spectrum PNG:  {output_png}")
    if output_pdf is not None:
        print(f"Wrote spectrum PDF:  {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
