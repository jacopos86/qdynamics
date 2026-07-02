#!/usr/bin/env python3
"""Manual consistency checker for hardcoded vs Qiskit pipeline JSON outputs.

This script is read-only: it loads two JSON payloads, computes cross-pipeline deltas
using the same definitions as pipelines/compare_hardcoded_vs_qiskit_pipeline.py,
prints a compact terminal report, and exits with status:

0: threshold-gated PASS (or thresholds unavailable)
1: threshold-gated FAIL
2: hard mismatch (settings/time-grid/trajectory schema)
3: metrics mismatch (only when --metrics is provided)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_THRESHOLDS = {
    "ground_state_energy_abs_delta": 1e-8,
    "fidelity_max_abs_delta": 1e-4,
    "energy_trotter_max_abs_delta": 1e-3,
    "n_up_site0_trotter_max_abs_delta": 5e-3,
    "n_dn_site0_trotter_max_abs_delta": 5e-3,
    "doublon_trotter_max_abs_delta": 1e-3,
}

TARGET_METRICS = [
    "fidelity",
    "energy_trotter",
    "n_up_site0_trotter",
    "n_dn_site0_trotter",
    "doublon_trotter",
]

SETTINGS_KEYS = [
    "L",
    "t",
    "u",
    "dv",
    "boundary",
    "ordering",
    "t_final",
    "num_times",
    "suzuki_order",
    "trotter_steps",
    "initial_state_source",
]


def _fp(x: float) -> str:
    return repr(float(x))


def _first_crossing(times: np.ndarray, vals: np.ndarray, thr: float) -> float | None:
    idx = np.where(vals > thr)[0]
    if idx.size == 0:
        return None
    return float(times[int(idx[0])])


def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_thresholds(path: Path | None) -> dict[str, float]:
    if path is None:
        return dict(DEFAULT_THRESHOLDS)
    raw = _load_json(path)
    if isinstance(raw, dict) and "thresholds" in raw and isinstance(raw["thresholds"], dict):
        raw_map = raw["thresholds"]
    elif isinstance(raw, dict):
        raw_map = raw
    else:
        raise ValueError(f"Unsupported thresholds JSON format in {path}")
    out = dict(DEFAULT_THRESHOLDS)
    for k in DEFAULT_THRESHOLDS:
        if k in raw_map:
            out[k] = float(raw_map[k])
    return out


def _settings_mismatches(hc: dict[str, Any], qk: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    hc_settings = hc.get("settings", {})
    qk_settings = qk.get("settings", {})
    diffs: list[tuple[str, Any, Any]] = []
    for key in SETTINGS_KEYS:
        hv = hc_settings.get(key)
        qv = qk_settings.get(key)
        if isinstance(hv, (int, float)) and isinstance(qv, (int, float)):
            if not np.isclose(float(hv), float(qv), rtol=0.0, atol=1e-12):
                diffs.append((key, hv, qv))
        else:
            if hv != qv:
                diffs.append((key, hv, qv))
    return diffs


def _compute_metrics(hc: dict[str, Any], qk: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    h_rows = hc.get("trajectory", [])
    q_rows = qk.get("trajectory", [])
    if len(h_rows) != len(q_rows):
        raise ValueError(
            f"Trajectory length mismatch: hardcoded={len(h_rows)} qiskit={len(q_rows)}"
        )
    if len(h_rows) == 0:
        raise ValueError("Trajectory arrays are empty.")

    t_h = _arr(h_rows, "time")
    t_q = _arr(q_rows, "time")
    if not np.allclose(t_h, t_q, atol=1e-12, rtol=0.0):
        raise ValueError("Time-grid mismatch: full sample check failed.")
    if not np.isclose(float(t_h[0]), float(t_q[0]), atol=1e-12, rtol=0.0):
        raise ValueError("Time-grid mismatch: t0 differs.")
    if not np.isclose(float(t_h[-1]), float(t_q[-1]), atol=1e-12, rtol=0.0):
        raise ValueError("Time-grid mismatch: t_final differs.")

    out: dict[str, Any] = {
        "time_grid": {
            "num_times": int(t_h.size),
            "t0": float(t_h[0]),
            "t_final": float(t_h[-1]),
            "dt": float(t_h[1] - t_h[0]) if t_h.size > 1 else 0.0,
        },
        "trajectory_deltas": {},
    }

    for key in TARGET_METRICS:
        h = _arr(h_rows, key)
        q = _arr(q_rows, key)
        d = np.abs(h - q)
        out["trajectory_deltas"][key] = {
            "max_abs_delta": float(np.max(d)),
            "mean_abs_delta": float(np.mean(d)),
            "final_abs_delta": float(d[-1]),
            "first_time_abs_delta_gt_1e-4": _first_crossing(t_h, d, 1e-4),
            "first_time_abs_delta_gt_1e-3": _first_crossing(t_h, d, 1e-3),
        }

    gs_h = float(hc["ground_state"]["exact_energy"])
    gs_q = float(qk["ground_state"]["exact_energy"])
    out["ground_state_energy"] = {
        "hardcoded_exact_energy": gs_h,
        "qiskit_exact_energy": gs_q,
        "abs_delta": float(abs(gs_h - gs_q)),
    }

    checks = {
        "ground_state_energy_abs_delta": out["ground_state_energy"]["abs_delta"] <= thresholds["ground_state_energy_abs_delta"],
        "fidelity_max_abs_delta": out["trajectory_deltas"]["fidelity"]["max_abs_delta"] <= thresholds["fidelity_max_abs_delta"],
        "energy_trotter_max_abs_delta": out["trajectory_deltas"]["energy_trotter"]["max_abs_delta"] <= thresholds["energy_trotter_max_abs_delta"],
        "n_up_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_up_site0_trotter"]["max_abs_delta"] <= thresholds["n_up_site0_trotter_max_abs_delta"],
        "n_dn_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_dn_site0_trotter"]["max_abs_delta"] <= thresholds["n_dn_site0_trotter_max_abs_delta"],
        "doublon_trotter_max_abs_delta": out["trajectory_deltas"]["doublon_trotter"]["max_abs_delta"] <= thresholds["doublon_trotter_max_abs_delta"],
    }
    out["acceptance"] = {
        "thresholds": thresholds,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    return out


def _vqe_diag(hc: dict[str, Any], qk: dict[str, Any]) -> dict[str, Any]:
    e_exact = float(hc.get("ground_state", {}).get("exact_energy", qk["ground_state"]["exact_energy"]))
    e_hc_raw = hc.get("vqe", {}).get("energy")
    e_qk_raw = qk.get("vqe", {}).get("energy")
    e_hc = float(e_hc_raw) if e_hc_raw is not None else np.nan
    e_qk = float(e_qk_raw) if e_qk_raw is not None else np.nan
    return {
        "exact_energy": e_exact,
        "hardcoded_vqe_energy": e_hc_raw,
        "qiskit_vqe_energy": e_qk_raw,
        "abs_hardcoded_minus_exact": float(abs(e_hc - e_exact)) if np.isfinite(e_hc) else None,
        "abs_qiskit_minus_exact": float(abs(e_qk - e_exact)) if np.isfinite(e_qk) else None,
        "abs_hardcoded_minus_qiskit": float(abs(e_hc - e_qk)) if np.isfinite(e_hc) and np.isfinite(e_qk) else None,
        "hardcoded_vqe_meta": {
            "success": hc.get("vqe", {}).get("success"),
            "nit": hc.get("vqe", {}).get("nit"),
            "nfev": hc.get("vqe", {}).get("nfev"),
            "reps": hc.get("vqe", {}).get("reps"),
            "message": hc.get("vqe", {}).get("message"),
        },
        "qiskit_vqe_meta": {
            "success": qk.get("vqe", {}).get("success"),
            "nit": qk.get("vqe", {}).get("nit"),
            "nfev": qk.get("vqe", {}).get("nfev"),
            "reps": qk.get("vqe", {}).get("reps"),
            "message": qk.get("vqe", {}).get("message"),
        },
    }


def _flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten_numeric(v, key))
            elif isinstance(v, (int, float)):
                out[key] = float(v)
    return out


def _metrics_match(computed: dict[str, Any], metrics_payload: dict[str, Any]) -> tuple[bool, list[str]]:
    expected = metrics_payload.get("metrics", metrics_payload)
    c_nums = _flatten_numeric(computed)
    e_nums = _flatten_numeric(expected)
    mismatches: list[str] = []

    for key, cval in c_nums.items():
        if key not in e_nums:
            continue
        eval_ = e_nums[key]
        if not np.isclose(cval, eval_, atol=1e-12, rtol=1e-12):
            mismatches.append(f"{key}: computed={_fp(cval)} expected={_fp(eval_)}")

    bool_paths = [
        "acceptance.pass",
        "acceptance.checks.ground_state_energy_abs_delta",
        "acceptance.checks.fidelity_max_abs_delta",
        "acceptance.checks.energy_trotter_max_abs_delta",
        "acceptance.checks.n_up_site0_trotter_max_abs_delta",
        "acceptance.checks.n_dn_site0_trotter_max_abs_delta",
        "acceptance.checks.doublon_trotter_max_abs_delta",
    ]
    for path in bool_paths:
        c = _get_path(computed, path)
        e = _get_path(expected, path)
        if e is None:
            continue
        if c != e:
            mismatches.append(f"{path}: computed={c} expected={e}")

    return len(mismatches) == 0, mismatches


def _get_path(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _print_report(
    *,
    hardcoded_path: Path,
    qiskit_path: Path,
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    vqe: dict[str, Any],
    metrics_match: bool | None,
    metrics_mismatches: list[str] | None,
) -> None:
    print("== Manual Consistency Report ==")
    print(f"hardcoded_json: {hardcoded_path}")
    print(f"qiskit_json:    {qiskit_path}")
    print("")
    tg = metrics["time_grid"]
    print("TIME GRID")
    print(f"  num_times={tg['num_times']} t0={_fp(tg['t0'])} t_final={_fp(tg['t_final'])} dt={_fp(tg['dt'])}")
    print("")

    print("TRAJECTORY DELTAS")
    for key in TARGET_METRICS:
        row = metrics["trajectory_deltas"][key]
        print(
            f"  {key}: max={_fp(row['max_abs_delta'])} mean={_fp(row['mean_abs_delta'])} final={_fp(row['final_abs_delta'])}"
        )
    print("")

    gs = metrics["ground_state_energy"]
    print("GROUND-STATE EXACT ENERGY DELTA")
    print(
        "  "
        f"|E_exact_hc - E_exact_qk| = {_fp(gs['abs_delta'])} "
        f"(hc={_fp(gs['hardcoded_exact_energy'])}, qk={_fp(gs['qiskit_exact_energy'])})"
    )
    print("")

    print("VQE DIAGNOSTICS (NOT GATE)")
    print(f"  E_exact_filtered={_fp(vqe['exact_energy'])}")
    print(f"  E_vqe_hc={vqe['hardcoded_vqe_energy']}")
    print(f"  E_vqe_qk={vqe['qiskit_vqe_energy']}")
    print(f"  |E_vqe_hc - E_exact|={vqe['abs_hardcoded_minus_exact']}")
    print(f"  |E_vqe_qk - E_exact|={vqe['abs_qiskit_minus_exact']}")
    print(f"  |E_vqe_hc - E_vqe_qk|={vqe['abs_hardcoded_minus_qiskit']}")
    print(f"  hardcoded_vqe_meta={vqe['hardcoded_vqe_meta']}")
    print(f"  qiskit_vqe_meta={vqe['qiskit_vqe_meta']}")
    print("")

    checks = metrics["acceptance"]["checks"]
    print("THRESHOLD GATING")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if bool(v) else 'FAIL'} (thr={_fp(thresholds[k])})")
    print(f"  overall: {'PASS' if bool(metrics['acceptance']['pass']) else 'FAIL'}")
    print("")

    if metrics_match is not None:
        print(f"METRICS MATCH: {metrics_match}")
        if metrics_mismatches:
            print("METRICS MISMATCH FIELDS")
            for line in metrics_mismatches:
                print(f"  - {line}")
        print("")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual JSON-to-JSON consistency checker.")
    p.add_argument("--hardcoded", type=Path, required=True, help="Hardcoded pipeline JSON path.")
    p.add_argument("--qiskit", type=Path, required=True, help="Qiskit pipeline JSON path.")
    p.add_argument("--thresholds", type=Path, default=None, help="Optional thresholds JSON path.")
    p.add_argument("--metrics", type=Path, default=None, help="Optional metrics JSON for verification.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hc = _load_json(args.hardcoded)
    qk = _load_json(args.qiskit)
    thresholds = _load_thresholds(args.thresholds)

    settings_diffs = _settings_mismatches(hc, qk)
    if settings_diffs:
        print("SETTINGS MISMATCH")
        for key, hv, qv in settings_diffs:
            print(f"  {key}: hardcoded={hv!r} qiskit={qv!r}")
        sys.exit(2)

    try:
        metrics = _compute_metrics(hc, qk, thresholds)
    except Exception as exc:
        print("TRAJECTORY/TIME-GRID MISMATCH")
        print(f"  {exc}")
        sys.exit(2)

    vqe = _vqe_diag(hc, qk)

    mmatch: bool | None = None
    mmismatches: list[str] | None = None
    if args.metrics is not None:
        expected_payload = _load_json(args.metrics)
        mmatch, mmismatches = _metrics_match(metrics, expected_payload)

    _print_report(
        hardcoded_path=args.hardcoded,
        qiskit_path=args.qiskit,
        metrics=metrics,
        thresholds=thresholds,
        vqe=vqe,
        metrics_match=mmatch,
        metrics_mismatches=mmismatches,
    )

    # Exit-code precedence:
    # 2 settings/shape mismatch (already exited) > 3 metrics mismatch > 1 threshold fail > 0 pass.
    if mmatch is False:
        sys.exit(3)
    if not bool(metrics["acceptance"]["pass"]):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
