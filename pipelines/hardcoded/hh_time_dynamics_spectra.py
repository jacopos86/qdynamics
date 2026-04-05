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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
        if key not in row:
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
    schema, rows, container = _resolve_controller_rows(payload)
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


def analyze_payload(
    payload: LoadedTrajectoryPayload,
    *,
    pair: tuple[int, int] | None,
    detrend: str,
    window: str,
    max_peaks: int,
    max_harmonic: int,
) -> dict[str, Any]:
    num_sites = int(payload.site_occupations.shape[1])
    dt = _infer_uniform_dt(payload.times)

    site_fluct = build_site_fluctuation_signals(payload.site_occupations)
    site_fluct_exact = (
        None
        if payload.site_occupations_exact is None
        else build_site_fluctuation_signals(payload.site_occupations_exact)
    )

    staggered = (
        build_staggered_signal(payload.site_occupations)
        if payload.staggered is None
        else np.asarray(payload.staggered, dtype=float)
    )
    staggered_exact = None
    if payload.site_occupations_exact is not None:
        staggered_exact = (
            build_staggered_signal(payload.site_occupations_exact)
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
        signal_map[f"site_occupation_{site}"] = np.asarray(payload.site_occupations[:, site], dtype=float)
        if payload.site_occupations_exact is not None:
            signal_map[f"site_occupation_{site}_exact"] = np.asarray(payload.site_occupations_exact[:, site], dtype=float)
        signal_map[f"site_fluctuation_{site}"] = np.asarray(site_fluct[:, site], dtype=float)
        if site_fluct_exact is not None:
            signal_map[f"site_fluctuation_{site}_exact"] = np.asarray(site_fluct_exact[:, site], dtype=float)

    if pair is not None:
        signal_map[f"pair_difference_{pair[0]}_{pair[1]}"] = build_pair_difference_signal(
            payload.site_occupations,
            pair=pair,
        )
        if payload.site_occupations_exact is not None:
            signal_map[f"pair_difference_{pair[0]}_{pair[1]}_exact"] = build_pair_difference_signal(
                payload.site_occupations_exact,
                pair=pair,
            )
            signal_map[f"pair_difference_{pair[0]}_{pair[1]}_error"] = np.asarray(
                build_pair_difference_signal(payload.site_occupations, pair=pair)
                - build_pair_difference_signal(payload.site_occupations_exact, pair=pair),
                dtype=float,
            )

    if payload.doublon is not None:
        signal_map["doublon"] = np.asarray(payload.doublon, dtype=float)
    if payload.doublon_exact is not None:
        signal_map["doublon_exact"] = np.asarray(payload.doublon_exact, dtype=float)

    spectra: dict[str, Any] = {}
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

    return {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "input_json": str(payload.input_json),
            "run_tag": payload.run_tag,
            "source_schema": payload.source_schema,
            "time_key": payload.time_key,
            "num_samples": int(payload.times.size),
            "num_sites": int(num_sites),
            "dt": float(dt),
            "t_initial": float(payload.times[0]),
            "t_final": float(payload.times[-1]),
            "window": str(window),
            "detrend": str(detrend),
            "drive_omega": None if payload.drive_omega is None else float(payload.drive_omega),
            "drive_amplitude": None if payload.drive_amplitude is None else float(payload.drive_amplitude),
            "nyquist_omega": float(math.pi / dt),
            "delta_omega_bin": float(2.0 * math.pi / (float(payload.times.size) * dt)),
            "pair_difference": None if pair is None else [int(pair[0]), int(pair[1])],
        },
        "raw_traces": {
            "times": [float(x) for x in payload.times.tolist()],
            "site_occupations": [[float(x) for x in row] for row in payload.site_occupations.tolist()],
            "site_occupations_exact": None
            if payload.site_occupations_exact is None
            else [[float(x) for x in row] for row in payload.site_occupations_exact.tolist()],
            "energy_total": None if payload.energy_total is None else [float(x) for x in payload.energy_total.tolist()],
            "energy_total_exact": None
            if payload.energy_total_exact is None
            else [float(x) for x in payload.energy_total_exact.tolist()],
            "energy_total_error": None
            if payload.energy_total is None or payload.energy_total_exact is None
            else [
                float(x)
                for x in (np.asarray(payload.energy_total, dtype=float) - np.asarray(payload.energy_total_exact, dtype=float)).tolist()
            ],
            "staggered": [float(x) for x in staggered.tolist()],
            "staggered_exact": None if staggered_exact is None else [float(x) for x in staggered_exact.tolist()],
            "staggered_error": None
            if staggered_exact is None
            else [float(x) for x in (np.asarray(staggered, dtype=float) - np.asarray(staggered_exact, dtype=float)).tolist()],
            "pair_difference": None
            if pair is None
            else [float(x) for x in build_pair_difference_signal(payload.site_occupations, pair=pair).tolist()],
            "pair_difference_exact": None
            if pair is None or payload.site_occupations_exact is None
            else [
                float(x)
                for x in build_pair_difference_signal(payload.site_occupations_exact, pair=pair).tolist()
            ],
            "pair_difference_error": None
            if pair is None or payload.site_occupations_exact is None
            else [
                float(x)
                for x in (
                    build_pair_difference_signal(payload.site_occupations, pair=pair)
                    - build_pair_difference_signal(payload.site_occupations_exact, pair=pair)
                ).tolist()
            ],
            "doublon": None if payload.doublon is None else [float(x) for x in payload.doublon.tolist()],
            "doublon_exact": None
            if payload.doublon_exact is None
            else [float(x) for x in payload.doublon_exact.tolist()],
        },
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


def render_spectrum_png(
    analysis: Mapping[str, Any],
    *,
    output_png: Path,
    max_harmonic: int,
    plot_max_omega: float | None = None,
) -> None:
    metadata = analysis["metadata"]
    traces = analysis["raw_traces"]
    spectra = analysis["spectra"]
    times = np.asarray(traces["times"], dtype=float)
    site_occ = np.asarray(traces["site_occupations"], dtype=float)
    site_occ_exact = (
        None
        if traces.get("site_occupations_exact") is None
        else np.asarray(traces["site_occupations_exact"], dtype=float)
    )

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 13.0))
    (
        ax_energy,
        ax_occ,
        ax_primary,
        ax_spec_energy,
        ax_spec_primary,
        ax_spec_error,
    ) = axes.reshape(-1)

    if traces.get("energy_total") is not None:
        ax_energy.plot(
            times,
            np.asarray(traces["energy_total"], dtype=float),
            color="#2ca02c",
            linewidth=2.0,
            label="energy_total",
        )
    if traces.get("energy_total_exact") is not None:
        ax_energy.plot(
            times,
            np.asarray(traces["energy_total_exact"], dtype=float),
            color="#2ca02c",
            linewidth=1.2,
            linestyle="--",
            label="energy_total_exact",
        )
    ax_energy.set_title("Raw total energy")
    ax_energy.set_xlabel(metadata["time_key"])
    ax_energy.set_ylabel("energy")
    ax_energy.grid(alpha=0.25)
    ax_energy.legend(fontsize=8, loc="best")

    colors = plt.cm.tab10.colors
    for site in range(site_occ.shape[1]):
        ax_occ.plot(times, site_occ[:, site], color=colors[site % len(colors)], linewidth=1.8, label=f"n_{site}")
        if site_occ_exact is not None:
            ax_occ.plot(
                times,
                site_occ_exact[:, site],
                color=colors[site % len(colors)],
                linewidth=1.2,
                linestyle="--",
                label=f"n_{site} exact",
            )
    ax_occ.set_title("Raw site occupations")
    ax_occ.set_xlabel(metadata["time_key"])
    ax_occ.set_ylabel("occupation")
    ax_occ.grid(alpha=0.25)
    ax_occ.legend(fontsize=8, loc="best")

    staggered = np.asarray(traces["staggered"], dtype=float)
    ax_primary.plot(times, staggered, color="#1f77b4", linewidth=2.0, label="staggered")
    if traces.get("staggered_exact") is not None:
        ax_primary.plot(
            times,
            np.asarray(traces["staggered_exact"], dtype=float),
            color="#1f77b4",
            linewidth=1.2,
            linestyle="--",
            label="staggered exact",
        )
    pair = metadata.get("pair_difference")
    if pair is not None:
        key = f"pair_difference_{pair[0]}_{pair[1]}"
        pair_signal = traces.get("pair_difference")
        if pair_signal is not None:
            ax_primary.plot(times, np.asarray(pair_signal, dtype=float), color="#ff7f0e", linewidth=1.6, label=key)
        if traces.get("pair_difference_exact") is not None:
            ax_primary.plot(
                times,
                np.asarray(traces["pair_difference_exact"], dtype=float),
                color="#ff7f0e",
                linewidth=1.1,
                linestyle="--",
                label=f"{key} exact",
            )
    ax_primary.set_title("Primary imbalance traces")
    ax_primary.set_xlabel(metadata["time_key"])
    ax_primary.set_ylabel("signal")
    ax_primary.grid(alpha=0.25)
    ax_primary.legend(fontsize=8, loc="best")

    energy_keys = [key for key in ["energy_total", "energy_total_exact"] if key in spectra]
    ymax_energy = 0.0
    for key in energy_keys:
        spec = spectra[key]
        omega = np.asarray(spec["omega"], dtype=float)
        amplitude = np.asarray(spec["amplitude"], dtype=float)
        ymax_energy = max(ymax_energy, float(np.max(amplitude)) if amplitude.size > 0 else 0.0)
        linestyle = "--" if key.endswith("_exact") else "-"
        ax_spec_energy.plot(omega, amplitude, linewidth=1.8, linestyle=linestyle, label=key)
    _plot_drive_harmonics(
        ax_spec_energy,
        drive_omega=metadata.get("drive_omega"),
        max_harmonic=int(max_harmonic),
        ymax=max(ymax_energy, 1.0e-12),
    )
    ax_spec_energy.set_title("Energy one-sided amplitude spectra")
    ax_spec_energy.set_xlabel("angular frequency ω")
    ax_spec_energy.set_ylabel("amplitude")
    ax_spec_energy.grid(alpha=0.25)
    ax_spec_energy.legend(fontsize=8, loc="best")
    if plot_max_omega is not None:
        ax_spec_energy.set_xlim(0.0, float(plot_max_omega))

    primary_keys = ["staggered"]
    if traces.get("staggered_exact") is not None:
        primary_keys.append("staggered_exact")
    if pair is not None:
        primary_keys.append(f"pair_difference_{pair[0]}_{pair[1]}")
        if f"pair_difference_{pair[0]}_{pair[1]}_exact" in spectra:
            primary_keys.append(f"pair_difference_{pair[0]}_{pair[1]}_exact")

    ymax_primary = 0.0
    for idx, key in enumerate(primary_keys):
        spec = spectra[key]
        omega = np.asarray(spec["omega"], dtype=float)
        amplitude = np.asarray(spec["amplitude"], dtype=float)
        ymax_primary = max(ymax_primary, float(np.max(amplitude)) if amplitude.size > 0 else 0.0)
        linestyle = "--" if key.endswith("_exact") else "-"
        ax_spec_primary.plot(omega, amplitude, linewidth=1.8, linestyle=linestyle, label=key)
    _plot_drive_harmonics(
        ax_spec_primary,
        drive_omega=metadata.get("drive_omega"),
        max_harmonic=int(max_harmonic),
        ymax=max(ymax_primary, 1.0e-12),
    )
    ax_spec_primary.set_title("Primary one-sided amplitude spectra")
    ax_spec_primary.set_xlabel("angular frequency ω")
    ax_spec_primary.set_ylabel("amplitude")
    ax_spec_primary.grid(alpha=0.25)
    ax_spec_primary.legend(fontsize=8, loc="best")
    if plot_max_omega is not None:
        ax_spec_primary.set_xlim(0.0, float(plot_max_omega))

    error_keys = [key for key in ["energy_total_error", "staggered_error"] if key in spectra]
    if pair is not None and f"pair_difference_{pair[0]}_{pair[1]}_error" in spectra:
        error_keys.append(f"pair_difference_{pair[0]}_{pair[1]}_error")
    ymax_error = 0.0
    for key in error_keys:
        spec = spectra[key]
        omega = np.asarray(spec["omega"], dtype=float)
        amplitude = np.asarray(spec["amplitude"], dtype=float)
        ymax_error = max(ymax_error, float(np.max(amplitude)) if amplitude.size > 0 else 0.0)
        ax_spec_error.plot(omega, amplitude, linewidth=1.8, label=key)
    _plot_drive_harmonics(
        ax_spec_error,
        drive_omega=metadata.get("drive_omega"),
        max_harmonic=int(max_harmonic),
        ymax=max(ymax_error, 1.0e-12),
    )
    ax_spec_error.set_title("Controller-minus-exact error spectra")
    ax_spec_error.set_xlabel("angular frequency ω")
    ax_spec_error.set_ylabel("amplitude")
    ax_spec_error.grid(alpha=0.25)
    ax_spec_error.legend(fontsize=8, loc="best")
    if plot_max_omega is not None:
        ax_spec_error.set_xlim(0.0, float(plot_max_omega))

    fig.suptitle(
        f"HH time-dynamics spectra | {metadata['input_json'].split('/')[-1]} | "
        f"{metadata['window']} window, {metadata['detrend']} detrend",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_json, output_png = _default_output_paths(args.input_json)
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
    )

    print(f"Wrote spectrum JSON: {output_json}")
    print(f"Wrote spectrum PNG:  {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
