#!/usr/bin/env python3
"""Post-analysis for L=3 HH exp fidelity gate runs.

DIAGNOSTIC ONLY (served role):
  - Companion analysis for the 2026-03-02 fidelity-wrapper experiment.
  - Retained for traceability of negative-result diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted(root.glob("L3_hh_exp_fidelity_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No fidelity run directory found under {root}")
    return candidates[0]


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _best_ok_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [r for r in rows if str(r.get("status")) == "ok" and _safe_float(r.get("delta_E_abs")) is not None]
    if len(ok) == 0:
        return None
    return min(ok, key=lambda r: float(r["delta_E_abs"]))


def _by_key(rows: list[dict[str, Any]], key_fn) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(key_fn(row))
        out.setdefault(key, []).append(row)
    return out


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# L=3 HH Exp Fidelity Analysis")
    lines.append("")
    lines.append("## Run Info")
    lines.append("")
    lines.append(f"- analyzed_at_utc: `{payload['analyzed_at_utc']}`")
    lines.append(f"- run_dir: `{payload['run_dir']}`")
    lines.append(f"- total_rows: `{payload['total_rows']}`")
    lines.append(f"- ok_rows: `{payload['ok_rows']}`")
    lines.append("")

    lines.append("## Best Per Method")
    lines.append("")
    lines.append("| method | config | seed | trotter | E_best | E_exact | |ΔE| | runtime_s | leak_flag |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["best_per_method"]:
        if row["best_delta"] is None:
            lines.append(f"| {row['method']} | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {row['method']} | {row['config']} | {row['seed']} | {row['trotter']} | "
            f"{row['E_best']:.12f} | {row['E_exact_sector']:.12f} | {row['best_delta']:.3e} | "
            f"{row['runtime_s']:.1f} | {row['sector_leak_flag']} |"
        )
    lines.append("")

    lines.append("## Best Per Method+Config")
    lines.append("")
    lines.append("| method | config | best ΔE | seed | trotter | status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in payload["best_per_method_config"]:
        best = "-" if row["best_delta"] is None else f"{row['best_delta']:.3e}"
        seed = "-" if row["seed"] is None else str(row["seed"])
        trotter = "-" if row["trotter"] is None else str(row["trotter"])
        status = row["status"] if row["status"] else "-"
        lines.append(f"| {row['method']} | {row['config']} | {best} | {seed} | {trotter} | {status} |")
    lines.append("")

    lines.append("## Ranked OK Runs")
    lines.append("")
    lines.append("| rank | method | config | seed | trotter | |ΔE| | runtime_s |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for i, row in enumerate(payload["ranked_ok_runs"], start=1):
        lines.append(
            f"| {i} | {row['method_id']} | {row['config']} | {row['seed']} | "
            f"{row['fidelity_residual_trotter_steps']} | {row['delta_E_abs']:.3e} | {row['runtime_s']:.1f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze exp-fidelity wrapper run outputs.")
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd() / "artifacts" / "l3_hh_exp_fidelity_gate",
        help="Root containing L3_hh_exp_fidelity_* run directories.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory. If omitted, latest under output-root is used.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of ranked OK runs to include.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir is not None else _latest_run_dir(Path(args.output_root))
    csv_path = run_dir / "summary" / "fidelity_gate_runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing run CSV: {csv_path}")
    rows = _load_rows(csv_path)

    ok_rows = [r for r in rows if str(r.get("status")) == "ok" and _safe_float(r.get("delta_E_abs")) is not None]
    for row in rows:
        row["config"] = f"{row.get('ordering')},{row.get('boundary')},{row.get('boson_encoding')}"

    method_keys = sorted(set(str(r.get("method_id", "")) for r in rows if str(r.get("method_id", "")) != ""))
    config_keys = sorted(set(str(r.get("config", "")) for r in rows if str(r.get("config", "")) != ""))

    best_per_method: list[dict[str, Any]] = []
    for method in method_keys:
        candidates = [r for r in ok_rows if str(r.get("method_id")) == method]
        if len(candidates) == 0:
            best_per_method.append(
                {
                    "method": method,
                    "best_delta": None,
                    "config": None,
                    "seed": None,
                    "trotter": None,
                    "E_best": None,
                    "E_exact_sector": None,
                    "runtime_s": None,
                    "sector_leak_flag": None,
                }
            )
            continue
        best = min(candidates, key=lambda r: float(r["delta_E_abs"]))
        best_per_method.append(
            {
                "method": method,
                "best_delta": float(best["delta_E_abs"]),
                "config": str(best["config"]),
                "seed": int(best["seed"]),
                "trotter": int(best["fidelity_residual_trotter_steps"]),
                "E_best": float(best["E_best"]),
                "E_exact_sector": float(best["E_exact_sector"]),
                "runtime_s": float(best["runtime_s"]),
                "sector_leak_flag": bool(str(best.get("sector_leak_flag", "")).lower() == "true"),
            }
        )

    best_per_method_config: list[dict[str, Any]] = []
    for method in method_keys:
        for cfg in config_keys:
            candidates = [r for r in rows if str(r.get("method_id")) == method and str(r.get("config")) == cfg]
            if len(candidates) == 0:
                continue
            ok_candidates = [r for r in candidates if str(r.get("status")) == "ok" and _safe_float(r.get("delta_E_abs")) is not None]
            if len(ok_candidates) == 0:
                best_per_method_config.append(
                    {
                        "method": method,
                        "config": cfg,
                        "best_delta": None,
                        "seed": None,
                        "trotter": None,
                        "status": "no_ok",
                    }
                )
            else:
                best = min(ok_candidates, key=lambda r: float(r["delta_E_abs"]))
                best_per_method_config.append(
                    {
                        "method": method,
                        "config": cfg,
                        "best_delta": float(best["delta_E_abs"]),
                        "seed": int(best["seed"]),
                        "trotter": int(best["fidelity_residual_trotter_steps"]),
                        "status": "ok",
                    }
                )

    ranked_ok = sorted(ok_rows, key=lambda r: float(r["delta_E_abs"]))[: max(1, int(args.top_n))]
    ranked_ok_slim = [
        {
            "run_id": str(r.get("run_id")),
            "method_id": str(r.get("method_id")),
            "config": str(r.get("config")),
            "seed": int(r["seed"]),
            "fidelity_residual_trotter_steps": int(r["fidelity_residual_trotter_steps"]),
            "delta_E_abs": float(r["delta_E_abs"]),
            "runtime_s": float(r["runtime_s"]),
            "status": str(r.get("status")),
        }
        for r in ranked_ok
    ]

    payload = {
        "analyzed_at_utc": _now_utc_iso(),
        "run_dir": str(run_dir),
        "total_rows": len(rows),
        "ok_rows": len(ok_rows),
        "best_per_method": best_per_method,
        "best_per_method_config": best_per_method_config,
        "ranked_ok_runs": ranked_ok_slim,
    }

    summary_dir = run_dir / "summary"
    json_path = summary_dir / "fidelity_gate_analysis.json"
    md_path = summary_dir / "fidelity_gate_analysis.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)

    print(f"analysis_json={json_path}")
    print(f"analysis_md={md_path}")


if __name__ == "__main__":
    main()
