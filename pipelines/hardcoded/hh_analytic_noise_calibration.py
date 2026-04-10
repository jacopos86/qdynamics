#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _nearest_unique_indices(times: np.ndarray, targets: Iterable[float], *, limit: int) -> list[int]:
    picked: list[int] = []
    for target in targets:
        if times.size == 0:
            break
        idx = int(np.argmin(np.abs(times - float(target))))
        if idx not in picked:
            picked.append(idx)
        if len(picked) >= int(limit):
            break
    if not picked and times.size:
        picked.append(0)
    return picked[: int(limit)]


def _row_time(row: dict[str, Any]) -> float:
    for key in ("time", "time_start", "time_stop"):
        value = row.get(key, None)
        if value is not None:
            return float(value)
    return float("nan")


def _safe_float(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key, None)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def fit_hybrid_proxy_calibration(
    payload: dict[str, Any],
    *,
    sample_times: tuple[float, ...] = (0.5, 1.5, 2.0, 4.0, 6.0, 8.0),
    max_samples: int = 6,
) -> dict[str, Any]:
    trajectory = list(payload.get("trajectory", []) or [])
    if not trajectory:
        raise ValueError("diagnostic payload is missing trajectory rows")
    oracle_cfg = dict(payload.get("oracle_config", {}) or {})
    summary = dict(payload.get("summary", {}) or {})
    times = np.asarray([_row_time(row) for row in trajectory], dtype=float)
    indices = _nearest_unique_indices(times, sample_times, limit=max_samples)

    sampled_rows: list[dict[str, Any]] = []
    combined_scales: list[float] = []
    depth_terms: list[float] = []
    group_terms: list[float] = []
    energy_biases: list[float] = []
    doublon_biases: list[float] = []
    staggered_biases: list[float] = []

    nominal_shots = int(oracle_cfg.get("shots", 2048) or 2048)
    nominal_repeats = int(oracle_cfg.get("oracle_repeats", 1) or 1)
    runtime_default = float(summary.get("final_runtime_parameter_count", 1) or 1)

    for idx in indices:
        row = dict(trajectory[int(idx)])
        time_value = _row_time(row)
        energy_err = _safe_float(row, "abs_energy_total_error", default=0.0)
        doublon_err = _safe_float(row, "abs_doublon_error", default=0.0)
        staggered_err = _safe_float(
            row,
            "abs_staggered_error",
            "abs_pair_difference_error",
            default=0.0,
        )
        combined_err = float(np.mean([abs(energy_err), abs(doublon_err), abs(staggered_err)]))
        group_burden = max(
            1.0,
            _safe_float(
                row,
                "groups_new",
                "measurement_groups_new",
                "group_count",
                default=runtime_default,
            ),
        )
        depth_proxy = max(
            1.0,
            _safe_float(
                row,
                "runtime_parameter_count",
                default=runtime_default,
            ),
        )
        shots_eff = max(
            1.0,
            (float(nominal_shots) * float(nominal_repeats)) / float(group_burden),
        )
        combined_scale = float(combined_err * np.sqrt(shots_eff))
        combined_scales.append(float(combined_scale))
        depth_terms.append(float(depth_proxy / 32.0))
        group_terms.append(float(np.log1p(group_burden)))
        energy_biases.append(
            _safe_float(row, "energy_total", default=0.0)
            - _safe_float(row, "energy_total_exact", default=0.0)
        )
        doublon_biases.append(
            _safe_float(row, "doublon", default=0.0)
            - _safe_float(row, "doublon_exact", default=0.0)
        )
        staggered_biases.append(
            _safe_float(row, "staggered", "pair_difference", default=0.0)
            - _safe_float(row, "staggered_exact", "pair_difference_exact", default=0.0)
        )
        sampled_rows.append(
            {
                "checkpoint_index": int(row.get("checkpoint_index", idx)),
                "time": float(time_value),
                "energy_error": float(energy_err),
                "doublon_error": float(doublon_err),
                "staggered_or_pair_error": float(staggered_err),
                "group_burden": float(group_burden),
                "depth_proxy": float(depth_proxy),
                "shots_eff": float(shots_eff),
                "combined_scale": float(combined_scale),
            }
        )

    median_scale = float(np.median(np.asarray(combined_scales, dtype=float)))
    depth_slope = 0.0
    group_slope = 0.0
    if len(sampled_rows) >= 2:
        depth_coeffs = np.polyfit(np.asarray(depth_terms, dtype=float), np.asarray(combined_scales, dtype=float), 1)
        group_coeffs = np.polyfit(np.asarray(group_terms, dtype=float), np.asarray(combined_scales, dtype=float), 1)
        depth_slope = float(max(0.0, depth_coeffs[0] / max(median_scale, 1.0e-9)))
        group_slope = float(max(0.0, group_coeffs[0] / max(median_scale, 1.0e-9)))
    lag_corr = 0.0
    if len(energy_biases) >= 2:
        lhs = np.asarray(energy_biases[:-1], dtype=float)
        rhs = np.asarray(energy_biases[1:], dtype=float)
        if np.std(lhs) > 1.0e-12 and np.std(rhs) > 1.0e-12:
            lag_corr = float(np.clip(np.corrcoef(lhs, rhs)[0, 1], 0.0, 0.95))

    coefficients = {
        "analytic_noise_model": "hybrid_qpu_proxy_v1",
        "analytic_noise_std": float(max(1.0e-6, median_scale)),
        "analytic_noise_nominal_shots": int(nominal_shots),
        "analytic_noise_nominal_repeats": int(nominal_repeats),
        "analytic_noise_shot_scale": 1.0,
        "analytic_noise_two_qubit_depth_scale": float(depth_slope),
        "analytic_noise_groups_new_scale": float(group_slope),
        "analytic_noise_time_corr": float(lag_corr),
        "analytic_noise_bias_energy": float(np.median(np.asarray(energy_biases, dtype=float))),
        "analytic_noise_bias_doublon": float(np.median(np.asarray(doublon_biases, dtype=float))),
        "analytic_noise_bias_staggered": float(np.median(np.asarray(staggered_biases, dtype=float))),
        "analytic_noise_metric_scale": 1.0,
        "analytic_noise_force_psd": True,
    }

    return {
        "model": "hybrid_qpu_proxy_v1",
        "source": {
            "backend_name": oracle_cfg.get("backend_name", None),
            "noise_mode": oracle_cfg.get("noise_mode", None),
            "shots": int(nominal_shots),
            "oracle_repeats": int(nominal_repeats),
            "mitigation": dict(oracle_cfg.get("mitigation", {}) or {}),
        },
        "sampled_checkpoints": sampled_rows,
        "coefficients": coefficients,
        "fit_residual_summary": {
            "median_scale": float(median_scale),
            "depth_slope": float(depth_slope),
            "group_slope": float(group_slope),
            "lag1_energy_bias_corr": float(lag_corr),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sample-times", type=str, default="0.5,1.5,2.0,4.0,6.0,8.0")
    parser.add_argument("--max-samples", type=int, default=6)
    args = parser.parse_args(argv)

    payload = _load_json(args.diagnostic_json)
    sample_times = tuple(
        float(chunk.strip()) for chunk in str(args.sample_times).split(",") if chunk.strip()
    )
    out = fit_hybrid_proxy_calibration(
        payload,
        sample_times=sample_times,
        max_samples=int(args.max_samples),
    )
    out["diagnostic_json"] = str(args.diagnostic_json)
    _dump_json(args.output_json, out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
