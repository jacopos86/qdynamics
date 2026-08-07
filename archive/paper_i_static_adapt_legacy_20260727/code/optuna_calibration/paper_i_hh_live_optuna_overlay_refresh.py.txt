#!/usr/bin/env python3
"""Refresh a live Paper-I HH SNAKE Optuna overlay report.

This is a diagnostic/reporting helper.  It reads local supervisor status,
Optuna SQLite user attributes, and live ``current.json`` checkpoints, then
writes a standalone TeX/PDF/JSON report.  It does not edit manuscript files,
source maps, run artifacts, or Optuna storage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-hh-live-overlay"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output/pdf"
SEED_REPORT = OUT_DIR / "paper_i_hh_snake_optuna_overlay_review_20260616.json"
PAPER_I_TEX = ROOT / "MATH/paper_details/static_adapt_paper_I.tex"

ACTIVE_LANES = {
    "weak-weak": {
        "slug": "paper_i_hh_local_geo_graphdomshot_salg_v11_missing_weak_live_20260616",
        "db_name": "weak-weak.sqlite3",
        "geo": {"dE": 0.001133089052752031, "N2Q": 100.0, "D2Q": 69.0, "Salg": 203000.0},
        "visible_key": "weak-weak",
    },
    "intermediate-weak": {
        "slug": "paper_i_hh_local_geo_graphdomshot_salg_v11_missing_weak_live_20260616",
        "db_name": "intermediate-weak.sqlite3",
        "geo": {"dE": 0.000189, "N2Q": 2988.0, "D2Q": 2719.0, "Salg": 2120000.0},
        "visible_key": "intermediate-weak",
    },
    "weak-strong": {
        "slug": "paper_i_hh_local_geo_graphdomshot_salg_v5_remaining_20260616",
        "db_name": "weak-strong.sqlite3",
        "geo": {"dE": 0.0427, "N2Q": 776.0, "D2Q": 630.0, "Salg": 291000.0},
        "visible_key": "weak-strong",
    },
    "intermediate-strong": {
        "slug": "paper_i_hh_local_geo_graphdomshot_salg_v5_remaining_20260616",
        "db_name": "intermediate-strong.sqlite3",
        "geo": {"dE": 0.00858, "N2Q": 1904.0, "D2Q": 1617.0, "Salg": 393000.0},
        "visible_key": "intermediate-strong",
    },
    "strong-weak-u8": {
        "slug": "paper_i_hh_local_geo_graphdomshot_salg_v5_remaining_20260616",
        "db_name": "strong-weak-u8.sqlite3",
        "geo": {"dE": 1.2254429947899936e-05, "N2Q": 296.0, "D2Q": 264.0, "Salg": 163868.0},
        "visible_key": None,
    },
    "strong-strong-u8": {
        "slug": "paper_i_hh_local_geo_graphdomshot_salg_v12_u8_ss_structural_resume_20260616",
        "db_name": "strong-strong-u8.sqlite3",
        "geo": {"dE": 0.00011327391134974274, "N2Q": 840.0, "D2Q": 711.0, "Salg": 180704.0},
        "visible_key": None,
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def fmt_num(value: Any, *, missing: str = "--") -> str:
    val = maybe_float(value)
    if val is None:
        return missing
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    if abs(val) < 1e-3 and val != 0:
        return f"{val:.3e}"
    return f"{val:.4g}"


def tex_escape(text: Any) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def supervisor_root(slug: str) -> Path:
    return ROOT / "raw_outputs/local_hh_optuna_supervisor" / slug


def optuna_db(slug: str, db_name: str) -> Path:
    return ROOT / "raw_outputs/optuna_studies/local_hh_optuna_supervisor" / slug / db_name


def trial_attrs(con: sqlite3.Connection, trial_id: int) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key, value_json in con.execute("select key,value_json from trial_user_attributes where trial_id=?", (trial_id,)):
        try:
            attrs[str(key)] = json.loads(value_json)
        except Exception:
            attrs[str(key)] = value_json
    return attrs


def trial_value(con: sqlite3.Connection, trial_id: int) -> float | None:
    try:
        row = con.execute("select value from trial_values where trial_id=? order by objective limit 1", (trial_id,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return None if row is None else maybe_float(row[0])


def read_optuna_trials(db: Path) -> list[dict[str, Any]]:
    if not db.exists():
        return []
    con = sqlite3.connect(db)
    try:
        rows: list[dict[str, Any]] = []
        for trial_id, number, state, start, complete, study_id, study_name in con.execute(
            """
            select t.trial_id,t.number,t.state,t.datetime_start,t.datetime_complete,t.study_id,s.study_name
            from trials t
            left join studies s on s.study_id=t.study_id
            order by t.study_id,t.number
            """
        ):
            attrs = trial_attrs(con, int(trial_id))
            rows.append(
                {
                    "trial_id": int(trial_id),
                    "study_id": int(study_id) if study_id is not None else None,
                    "study_name": study_name,
                    "trial_number": int(number),
                    "state": state,
                    "datetime_start": start,
                    "datetime_complete": complete,
                    "objective_value": trial_value(con, int(trial_id)),
                    "dE": maybe_float(attrs.get("abs_delta_e")),
                    "k": attrs.get("adapt_iteration_count") or attrs.get("dominance_prefix_iteration"),
                    "N2Q_proxy": maybe_float(attrs.get("graph_count_2q")),
                    "D2Q_proxy": maybe_float(attrs.get("graph_depth")),
                    "N1Q_proxy": maybe_float(attrs.get("graph_count_1q")),
                    "Salg": maybe_float(attrs.get("paper_i_table_s_alg")),
                    "Salg_status": attrs.get("paper_i_table_shots_status"),
                    "Salg_fallback": maybe_float(attrs.get("paper_i_shot_cost_scalar")),
                    "graph_plus_shot_objective_scalar": maybe_float(attrs.get("graph_plus_shot_objective_scalar")),
                    "result_json": attrs.get("result_json"),
                    "case_dir": attrs.get("case_dir"),
                    "invalid_reasons": attrs.get("invalid_reasons") or [],
                }
            )
        return rows
    finally:
        con.close()


def optuna_trial_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts: dict[str, int] = {}
    for row in trials:
        state = str(row.get("state") or "UNKNOWN")
        state_counts[state] = state_counts.get(state, 0) + 1

    per_study: list[dict[str, Any]] = []
    study_ids = sorted({row.get("study_id") for row in trials}, key=lambda x: (-1 if x is None else int(x)))
    for study_id in study_ids:
        study_rows = [row for row in trials if row.get("study_id") == study_id]
        if not study_rows:
            continue
        study_state_counts: dict[str, int] = {}
        for row in study_rows:
            state = str(row.get("state") or "UNKNOWN")
            study_state_counts[state] = study_state_counts.get(state, 0) + 1
        numbers = [int(row["trial_number"]) for row in study_rows if row.get("trial_number") is not None]
        per_study.append(
            {
                "study_id": study_id,
                "study_name": study_rows[0].get("study_name"),
                "total_trial_rows": len(study_rows),
                "min_trial_number": min(numbers) if numbers else None,
                "max_trial_number": max(numbers) if numbers else None,
                "state_counts": study_state_counts,
            }
        )

    numbers = [int(row["trial_number"]) for row in trials if row.get("trial_number") is not None]
    return {
        "total_trial_rows": len(trials),
        "state_counts": state_counts,
        "complete_count": state_counts.get("COMPLETE", 0),
        "fail_count": state_counts.get("FAIL", 0),
        "running_count": state_counts.get("RUNNING", 0),
        "waiting_count": state_counts.get("WAITING", 0),
        "max_trial_number": max(numbers) if numbers else None,
        "study_count": len(per_study),
        "per_study": per_study,
        "counting_note": "Counts are SQLite rows across all configured studies for this regime. Trial numbers are 0-based within each Optuna study.",
    }


def read_supervisor_status(root: Path) -> dict[str, Any]:
    path = root / "supervisor_status.json"
    if not path.exists():
        return {"exists": False}
    payload = load_json(path)
    return {
        "exists": True,
        "path": str(path.relative_to(ROOT)),
        "active": payload.get("active"),
        "last_update": payload.get("last_update_utc") or payload.get("last_update"),
        "rows": payload.get("rows") or [],
    }


def latest_current_json(root: Path, regime: str) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        root.glob(f"cycle_*/{regime}/canonical/eps_*/trial_*/current.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = load_json(path)
        except Exception:
            continue
        return path, payload
    return None, None


def history_series(current_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(current_payload, Mapping):
        return {"points": [], "status": "missing_current_json"}
    adapt = current_payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        return {"points": [], "status": "missing_adapt_vqe"}
    history = adapt.get("history_tail") or adapt.get("history") or adapt.get("adapt_history") or []
    if not isinstance(history, list) or not history:
        return {"points": [], "status": "missing_history_tail"}
    ansatz_depth = maybe_float(adapt.get("ansatz_depth"))
    history_count = maybe_float(adapt.get("history_count")) or len(history)
    base_depth = int(round(ansatz_depth - history_count)) if ansatz_depth is not None else 0
    points: list[dict[str, Any]] = []
    cumulative_burden = 0.0
    for idx, entry in enumerate(h for h in history if isinstance(h, Mapping)):
        local_depth = maybe_float(entry.get("depth"))
        iteration = base_depth + int(round(local_depth if local_depth is not None else idx + 1))
        err = (
            maybe_float(entry.get("benchmark_target_abs_delta_current"))
            or maybe_float(entry.get("delta_abs_current"))
            or maybe_float(entry.get("abs_delta_e"))
        )
        burden = maybe_float(entry.get("selector_burden")) or 1.0
        cumulative_burden += max(0.0, burden)
        if err is None:
            continue
        points.append({"iteration": iteration, "dE": abs(err), "cost_proxy_local": cumulative_burden})
    return {
        "points": points,
        "status": "ok" if points else "no_error_points",
        "base_depth": base_depth,
        "ansatz_depth": ansatz_depth,
        "history_count": history_count,
        "terminal_dE": maybe_float(adapt.get("abs_delta_e")) or maybe_float(adapt.get("benchmark_target_abs_delta_e_current")),
    }


def plateau_from_points(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not points:
        return None
    for idx, point in enumerate(points):
        future_best = min(float(p["dE"]) for p in points[idx:])
        current = float(point["dE"])
        if current - future_best <= 0.05 * max(current, 1e-14):
            return {**point, "source": "live_history_first_future_improvement_le_5pct"}
    return {**points[-1], "source": "terminal_fallback"}


def choose_best_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [t for t in trials if t.get("state") == "COMPLETE"]
    by_delta = min((t for t in complete if t.get("dE") is not None), key=lambda t: float(t["dE"]), default=None)
    by_objective = min((t for t in complete if t.get("objective_value") is not None), key=lambda t: float(t["objective_value"]), default=None)
    by_graph = min((t for t in complete if t.get("N2Q_proxy") is not None and t.get("D2Q_proxy") is not None), key=graph_cost_key, default=None)
    by_shot = min((t for t in complete if t.get("Salg") is not None), key=lambda t: float(t["Salg"]), default=None)
    by_shot_fallback = min((t for t in complete if t.get("Salg_fallback") is not None), key=lambda t: float(t["Salg_fallback"]), default=None)
    return {
        "complete_count": len(complete),
        "best_by_delta_e": by_delta,
        "best_by_objective": by_objective,
        "best_by_graph_proxy": by_graph,
        "best_by_paper_i_salg": by_shot,
        "best_by_shot_fallback": by_shot_fallback,
    }


def graph_cost_key(row: Mapping[str, Any]) -> float:
    n2q = maybe_float(row.get("N2Q_proxy"))
    d2q = maybe_float(row.get("D2Q_proxy"))
    n1q = maybe_float(row.get("N1Q_proxy")) or 0.0
    if n2q is None or d2q is None:
        return float("inf")
    return n2q * 1.0e9 + d2q * 1.0e6 + n1q


def graph_plus_shot_key(row: Mapping[str, Any]) -> float:
    explicit = maybe_float(row.get("graph_plus_shot_objective_scalar"))
    if explicit is not None:
        return explicit
    shot = maybe_float(row.get("Salg")) or maybe_float(row.get("Salg_fallback")) or 1.0e12
    return graph_cost_key(row) + shot


def candidate_rows(trials: list[dict[str, Any]], geo: Mapping[str, Any]) -> list[dict[str, Any]]:
    complete = [t for t in trials if t.get("state") == "COMPLETE"]
    geo_de = maybe_float(geo.get("dE"))
    energy_feasible = [
        t for t in complete
        if t.get("dE") is not None and geo_de is not None and float(t["dE"]) <= float(geo_de)
    ]
    specs: list[tuple[str, Mapping[str, Any] | None, str]] = [
        ("energy-dominant", min((t for t in complete if t.get("dE") is not None), key=lambda t: float(t["dE"]), default=None), "lowest completed energy error"),
        ("objective-dominant", min((t for t in complete if t.get("objective_value") is not None), key=lambda t: float(t["objective_value"]), default=None), "lowest stored Optuna objective"),
        ("cost-dominant feasible", min(energy_feasible, key=graph_plus_shot_key, default=None), "lowest graph+shot proxy after beating Geo energy"),
        ("graph-dominant", min((t for t in complete if t.get("N2Q_proxy") is not None and t.get("D2Q_proxy") is not None), key=graph_cost_key, default=None), "lowest graph proxy regardless of energy"),
        ("Salg-dominant", min((t for t in complete if t.get("Salg") is not None), key=lambda t: float(t["Salg"]), default=None), "lowest valid Paper-I S_alg"),
        ("shot-fallback-dominant", min((t for t in complete if t.get("Salg_fallback") is not None), key=lambda t: float(t["Salg_fallback"]), default=None), "lowest measurement-work fallback when S_alg is blocked"),
    ]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int | None, str]] = set()
    for label, row, basis in specs:
        if not isinstance(row, Mapping):
            rows.append({"label": label, "basis": basis, "missing": True})
            continue
        key = (row.get("trial_number"), label)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"label": label, "basis": basis, **dict(row)})
    return rows


def visible_rows(seed: Mapping[str, Any]) -> dict[str, Any]:
    rows = seed.get("visible_rows") if isinstance(seed, Mapping) else {}
    return rows if isinstance(rows, dict) else {}


def u8_visible_rows_from_manuscript() -> dict[str, Any]:
    if not PAPER_I_TEX.exists():
        return {}
    for line in PAPER_I_TEX.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text.startswith("%") or "paper_i_hh_u8_current_evidence_v1" not in text:
            continue
        try:
            payload = json.loads(text.lstrip("%").strip())
        except Exception:
            continue
        rows = payload.get("rows")
        if not isinstance(rows, Mapping):
            return {}
        out: dict[str, Any] = {}
        labels = {
            "hh_u8_strong_weak": "strong--weak U/t=8 (lambda=0.25, n_ph=2)",
            "hh_u8_strong_strong": "strong--strong U/t=8 (lambda=1.25, n_ph=4)",
        }
        keys = {
            "hh_u8_strong_weak": "strong-weak-u8",
            "hh_u8_strong_strong": "strong-strong-u8",
        }
        for src_key, dst_key in keys.items():
            src_rows = rows.get(src_key)
            if not isinstance(src_rows, list):
                continue
            rendered: list[dict[str, Any]] = []
            for row in src_rows:
                if not isinstance(row, Mapping):
                    continue
                rendered.append(
                    {
                        "method": row.get("method"),
                        "n_ph": row.get("n_ph_work"),
                        "k": row.get("k"),
                        "dE": row.get("same_cutoff_abs_delta_e"),
                        "N2q": row.get("N2q"),
                        "D2q": row.get("D2q"),
                        "Dc": row.get("Dc"),
                        "S_alg": row.get("S_alg"),
                    }
                )
            out[dst_key] = {"label": labels[src_key], "rows": rendered}
        return out
    return {}


def plot_lane(regime: str, lane: Mapping[str, Any], stem: str) -> Path | None:
    points = (((lane.get("live") or {}).get("history") or {}).get("points") or [])
    if not points:
        return None
    png = OUT_DIR / f"{stem}_{regime.replace('-', '_')}_live_curve.png"
    xs = [p["iteration"] for p in points]
    ys = [max(float(p["dE"]), 1e-14) for p in points]
    costs = [p["cost_proxy_local"] for p in points]
    fig, ax = plt.subplots(figsize=(5.8, 3.2), dpi=180)
    ax.plot(xs, ys, color="#E45756", marker="*", linewidth=2.0, label="error vs iteration")
    ax.set_yscale("log")
    ax.set_xlabel("ADAPT iteration")
    ax.set_ylabel(r"$|\Delta E|$")
    ax.grid(True, which="major", alpha=0.25)
    ax2 = ax.twiny()
    ax2.plot(costs, ys, color="#4C78A8", marker="o", linestyle=":", linewidth=1.6, label="error vs local cost proxy")
    ax2.set_xlabel("local cumulative selector-burden proxy")
    plateau = (lane.get("live") or {}).get("plateau")
    if isinstance(plateau, Mapping):
        ax.scatter([plateau["iteration"]], [max(float(plateau["dE"]), 1e-14)], color="#E45756", edgecolors="black", s=95, zorder=5)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return png


def build_payload(stem: str, *, include_plots: bool = True) -> dict[str, Any]:
    seed = load_json(SEED_REPORT) if SEED_REPORT.exists() else {}
    payload: dict[str, Any] = {
        "schema": "paper_i_hh_live_optuna_overlay_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Auto-refreshed diagnostic overlay; no manuscript/source-map edits.",
        "cost_policy": {
            "terminal_current_proxy": "Optuna graph_count_2q/graph_depth and paper_i_table_s_alg attrs when available.",
            "plateau_prefix_proxy": "Live history-derived plateau marker plus Optuna prefix/status fields when available; missing cells remain explicit.",
            "live_curve_cost_proxy": "Cumulative selector_burden from current.json history_tail; local diagnostic scale, not a compiled gate count.",
            "competitor_rows": "Seeded from existing Paper-I overlay visible rows; treated as plateau comparison rows, not rewritten.",
        },
        "visible_rows": visible_rows(seed),
        "u8_visible_rows": u8_visible_rows_from_manuscript(),
        "u8_diagnostic_rows": seed.get("u8_diagnostic_rows", {}) if isinstance(seed, Mapping) else {},
        "replayed_candidate_rows": seed.get("replayed_candidate_rows", []) if isinstance(seed, Mapping) else [],
        "lanes": {},
    }
    for regime, cfg in ACTIVE_LANES.items():
        root = supervisor_root(str(cfg["slug"]))
        db = optuna_db(str(cfg["slug"]), str(cfg["db_name"]))
        trials = read_optuna_trials(db)
        current_path, current_payload = latest_current_json(root, regime)
        hist = history_series(current_payload)
        plateau = plateau_from_points(hist.get("points") or [])
        lane = {
            "regime": regime,
            "slug": cfg["slug"],
            "supervisor_status": read_supervisor_status(root),
            "geo_proxy_target": cfg["geo"],
            "trial_summary": optuna_trial_summary(trials),
            "optuna": choose_best_trials(trials),
            "candidate_rows": candidate_rows(trials, cfg["geo"]),
            "running_trials": [t for t in trials if t.get("state") == "RUNNING"],
            "live": {
                "current_json": None if current_path is None else str(current_path.relative_to(ROOT)),
                "history": hist,
                "plateau": plateau,
                "plateau_proxy_status": "available_live_history_marker_only" if plateau else "missing_live_history",
            },
        }
        payload["lanes"][regime] = lane
    if bool(include_plots):
        for regime, lane in payload["lanes"].items():
            png = plot_lane(regime, lane, stem)
            if png is not None:
                lane["live"]["plot_png"] = str(png.relative_to(ROOT))
    return payload


def tex_trial_row(label: str, row: Mapping[str, Any] | None, geo: Mapping[str, Any]) -> str:
    if not isinstance(row, Mapping):
        return rf"{tex_escape(label)} & -- & -- & -- & -- & -- & -- & missing \\"
    n2q = maybe_float(row.get("N2Q_proxy"))
    d2q = maybe_float(row.get("D2Q_proxy"))
    salg = maybe_float(row.get("Salg"))
    n2q_gap = None if n2q is None or geo.get("N2Q") is None else n2q - float(geo["N2Q"])
    d2q_gap = None if d2q is None or geo.get("D2Q") is None else d2q - float(geo["D2Q"])
    return (
        rf"{tex_escape(label)} & {row.get('trial_number', '--')} & {fmt_num(row.get('dE'))} & "
        rf"{fmt_num(row.get('k'))} & {fmt_num(n2q)} ({fmt_num(n2q_gap)}) & "
        rf"{fmt_num(d2q)} ({fmt_num(d2q_gap)}) & {fmt_num(salg, missing='blocked')} & "
        rf"{tex_escape(row.get('Salg_status') or ','.join(row.get('invalid_reasons') or []) or 'ok')} \\"
    )


def tex_candidate_row(row: Mapping[str, Any], geo: Mapping[str, Any]) -> str:
    if bool(row.get("missing")):
        return rf"{tex_escape(row.get('label'))} & -- & -- & -- & -- & -- & -- & -- & {tex_escape(row.get('basis'))} \\"
    n2q = maybe_float(row.get("N2Q_proxy"))
    d2q = maybe_float(row.get("D2Q_proxy"))
    salg = maybe_float(row.get("Salg"))
    shot_fallback = maybe_float(row.get("Salg_fallback"))
    n2q_gap = None if n2q is None or geo.get("N2Q") is None else n2q - float(geo["N2Q"])
    d2q_gap = None if d2q is None or geo.get("D2Q") is None else d2q - float(geo["D2Q"])
    shot_cell = fmt_num(salg, missing="blocked")
    if salg is None and shot_fallback is not None:
        shot_cell = rf"blocked / fb {fmt_num(shot_fallback)}"
    return (
        rf"{tex_escape(row.get('label'))} & {row.get('trial_number', '--')} & {fmt_num(row.get('dE'))} & "
        rf"{fmt_num(row.get('k'))} & {fmt_num(n2q)} ({fmt_num(n2q_gap)}) & "
        rf"{fmt_num(d2q)} ({fmt_num(d2q_gap)}) & {shot_cell} & "
        rf"{fmt_num(row.get('graph_plus_shot_objective_scalar'))} & {tex_escape(row.get('basis'))} \\"
    )


def render_tex(payload: Mapping[str, Any], tex_path: Path) -> None:
    lines: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{booktabs,longtable,array,float,graphicx,hyperref,xcolor}",
        r"\usepackage{amsmath}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.45em}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large Live Paper-I HH SNAKE Optuna Overlay}\\",
        rf"{{\small Auto-refreshed {tex_escape(payload['generated_utc'])}; diagnostic only.}}",
        r"\end{center}",
        r"\textbf{Scope.} This report refreshes local SNAKE Optuna status against plateau comparator rows. It does not edit manuscript tables, figures, source maps, Optuna studies, or run artifacts.",
        r"\textbf{Cost semantics.} Terminal/current proxy columns come from Optuna graph proxy attrs. Plateau-prefix proxy is reported only when a live/history marker or stored prefix row is available; missing plateau proxy is not replaced by terminal cost.",
    ]
    for regime, lane in (payload.get("lanes") or {}).items():
        geo = lane["geo_proxy_target"]
        opt = lane.get("optuna") or {}
        live = lane.get("live") or {}
        plateau = live.get("plateau")
        lines.extend(
            [
                rf"\section*{{{tex_escape(regime)}}}",
                rf"Geo proxy target: $|\Delta E|={fmt_num(geo.get('dE'))}$, $N_{{2q}}={fmt_num(geo.get('N2Q'))}$, $D_{{2q}}={fmt_num(geo.get('D2Q'))}$, $S_{{\rm alg}}={fmt_num(geo.get('Salg'))}$.",
                r"\begin{table}[H]\centering\small",
                r"\begin{tabular}{lrrrrrrrl}",
                r"\toprule",
                r"Candidate & Trial & $|\Delta E|$ & $k$ & $N_{2q}$ (gap) & $D_{2q}$ (gap) & $S_{\rm alg}$ & Obj. & Basis \\",
                r"\midrule",
                *[tex_candidate_row(row, geo) for row in lane.get("candidate_rows", [])],
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Completed Optuna candidate views. Parentheses show proxy gap versus the Geo target; positive is more expensive than Geo. A blocked $S_{\rm alg}$ may still have a measurement-work fallback used in diagnostic objective telemetry, shown as fb.}",
                r"\end{table}",
            ]
        )
        if isinstance(plateau, Mapping):
            lines.append(
                rf"Live plateau marker: $k={fmt_num(plateau.get('iteration'))}$, $|\Delta E|={fmt_num(plateau.get('dE'))}$, local selector-burden proxy ${fmt_num(plateau.get('cost_proxy_local'))}$; source {tex_escape(plateau.get('source'))}."
            )
        else:
            lines.append("Live plateau marker: missing live history; plateau-prefix proxy not reported.")
        plot_rel = live.get("plot_png")
        if plot_rel:
            lines.extend(
                [
                    r"\begin{figure}[H]\centering",
                    rf"\includegraphics[width=0.92\linewidth]{{{tex_escape(str(ROOT / plot_rel))}}}",
                    r"\caption{Live SNAKE error versus ADAPT iteration and local selector-burden proxy. The cost axis is diagnostic, not an exact compiled-gate axis.}",
                    r"\end{figure}",
                ]
            )
    lines.extend(
        [
            r"\section*{Comparator plateau rows}",
            r"Comparator rows are copied from the existing review seed and are included to keep the visible Paper-I plateau context in the same artifact.",
        ]
    )
    visible = payload.get("visible_rows") or {}
    for key, block in visible.items():
        rows = block.get("rows") if isinstance(block, Mapping) else None
        if not isinstance(rows, list):
            continue
        lines.extend(
            [
                rf"\subsection*{{{tex_escape(block.get('label') or key)}}}",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                r"Method & $k$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $S_{\rm alg}$ \\",
                r"\midrule",
            ]
        )
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                rf"{tex_escape(row.get('method'))} & {fmt_num(row.get('k'))} & {fmt_num(row.get('dE'))} & {fmt_num(row.get('N2q'))} & {fmt_num(row.get('D2q'))} & {fmt_num(row.get('S_alg'))} \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
    lines.extend(
        [
            r"\section*{Provenance}",
            rf"Sidecar JSON: \texttt{{{tex_escape(str(tex_path.with_suffix('.json').relative_to(ROOT)))}}}.",
            r"\end{document}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_visible_row(row: Mapping[str, Any], *, include_nph: bool = False) -> str:
    prefix = f"{fmt_num(row.get('n_ph'))} & " if include_nph else ""
    return (
        rf"{tex_escape(row.get('method'))} & {prefix}{fmt_num(row.get('k'))} & {fmt_num(row.get('dE'))} & "
        rf"{fmt_num(row.get('N2q'))} & {fmt_num(row.get('D2q'))} & {fmt_num(row.get('Dc'))} & {fmt_num(row.get('S_alg'))} \\"
    )


def tex_proxy_candidate_row(row: Mapping[str, Any], *, regime: str = "") -> str:
    if bool(row.get("missing")):
        return (
            rf"{tex_escape(regime)} & {tex_escape(row.get('label'))} & -- & -- & -- & -- & -- & -- & -- & "
            rf"{tex_escape(row.get('basis'))} \\"
        )
    shot = maybe_float(row.get("Salg"))
    fallback = maybe_float(row.get("Salg_fallback"))
    shot_cell = fmt_num(shot, missing="blocked")
    if shot is None and fallback is not None:
        shot_cell = rf"blocked/fb {fmt_num(fallback)}"
    return (
        rf"{tex_escape(regime)} & {tex_escape(row.get('label'))} & {fmt_num(row.get('trial_number'))} & "
        rf"{fmt_num(row.get('k'))} & {fmt_num(row.get('dE'))} & {fmt_num(row.get('N2Q_proxy'))} & "
        rf"{fmt_num(row.get('D2Q_proxy'))} & -- & {shot_cell} & {tex_escape(row.get('basis'))} \\"
    )


def tex_replay_row(row: Mapping[str, Any]) -> str:
    return (
        rf"{tex_escape(row.get('regime'))} & {tex_escape(row.get('basis'))} & {fmt_num(row.get('trial'))} & "
        rf"{fmt_num(row.get('k'))} & {fmt_num(row.get('dE'))} & {fmt_num(row.get('N2q'))} & "
        rf"{fmt_num(row.get('D2q'))} & {fmt_num(row.get('Dc'))} & {fmt_num(row.get('S_alg'))} \\"
    )


def candidate_rows_for_regime(payload: Mapping[str, Any], regime: str, *, limit: int = 6) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(row: Mapping[str, Any], *, source: str) -> None:
        trial = str(row.get("trial_number", row.get("trial", "")))
        label = str(row.get("label", row.get("basis", source)))
        key = (trial, label)
        if key in seen:
            return
        seen.add(key)
        rows.append({**dict(row), "candidate_source": source})

    lanes = payload.get("lanes") or {}
    lane = lanes.get(regime) if isinstance(lanes, Mapping) else None
    if isinstance(lane, Mapping):
        for row in lane.get("candidate_rows") or []:
            if isinstance(row, Mapping):
                add(row, source="optuna-proxy")

    replay_rows = payload.get("replayed_candidate_rows") or []
    if isinstance(replay_rows, list):
        for row in replay_rows:
            if isinstance(row, Mapping) and str(row.get("regime")) == str(regime):
                add(row, source="replay+compile")

    u8_diag = payload.get("u8_diagnostic_rows") or {}
    diag_rows = u8_diag.get(regime) if isinstance(u8_diag, Mapping) else None
    if isinstance(diag_rows, list):
        for row in diag_rows:
            if isinstance(row, Mapping):
                add(row, source="replay+compile")

    while len(rows) < int(limit):
        rows.append(
            {
                "label": f"candidate slot {len(rows) + 1}",
                "basis": "no completed candidate available yet",
                "missing": True,
                "candidate_source": "missing",
            }
        )
    return rows[: int(limit)]


def tex_local_candidate_row(row: Mapping[str, Any]) -> str:
    if bool(row.get("missing")):
        return (
            rf"{tex_escape(row.get('label'))} & -- & -- & -- & -- & -- & -- & -- & "
            rf"{tex_escape(row.get('basis'))} \\"
        )
    source = str(row.get("candidate_source") or "")
    if source == "replay+compile":
        label = row.get("basis") or row.get("label") or source
        trial = row.get("trial")
        n2q = row.get("N2q")
        d2q = row.get("D2q")
        dc = row.get("Dc")
        shot = row.get("S_alg")
        note = "replay+compile"
    else:
        label = row.get("label") or row.get("basis") or source
        trial = row.get("trial_number")
        n2q = row.get("N2Q_proxy")
        d2q = row.get("D2Q_proxy")
        dc = None
        shot = maybe_float(row.get("Salg"))
        fallback = maybe_float(row.get("Salg_fallback"))
        shot_cell = fmt_num(shot, missing="blocked")
        if shot is None and fallback is not None:
            shot_cell = f"fb {fmt_num(fallback)}"
        note = row.get("basis") or source
    if source == "replay+compile":
        shot_cell = fmt_num(shot)
    return (
        rf"{tex_escape(label)} & {fmt_num(trial)} & {fmt_num(row.get('k'))} & {fmt_num(row.get('dE'))} & "
        rf"{fmt_num(n2q)} & {fmt_num(d2q)} & {fmt_num(dc)} & {tex_escape(shot_cell)} & {tex_escape(note)} \\"
    )


def tex_trial_count_note(payload: Mapping[str, Any], regime: str) -> str:
    lanes = payload.get("lanes") or {}
    lane = lanes.get(regime) if isinstance(lanes, Mapping) else None
    summary = lane.get("trial_summary") if isinstance(lane, Mapping) else None
    if not isinstance(summary, Mapping):
        return r"\textbf{Optuna trial count.} No configured SQLite trial rows found for this regime."
    states = summary.get("state_counts") if isinstance(summary.get("state_counts"), Mapping) else {}
    return (
        r"\textbf{Optuna trial count.} "
        rf"Configured SQLite rows: {fmt_num(summary.get('total_trial_rows'))}; "
        rf"complete {fmt_num(states.get('COMPLETE', 0))}, "
        rf"failed {fmt_num(states.get('FAIL', 0))}, "
        rf"running/stale {fmt_num(states.get('RUNNING', 0))}, "
        rf"waiting {fmt_num(states.get('WAITING', 0))}. "
        rf"Highest 0-based trial index: {fmt_num(summary.get('max_trial_number'))}; "
        rf"Optuna studies in DB: {fmt_num(summary.get('study_count'))}. "
        r"Trial numbers are per-study, so the highest index need not equal the total row count."
    )


def append_candidate_block(lines: list[str], payload: Mapping[str, Any], regime: str) -> None:
    candidates = candidate_rows_for_regime(payload, regime, limit=6)
    lines.extend(
        [
            tex_trial_count_note(payload, regime),
            r"\begin{table}[H]\centering\scriptsize",
            r"\begin{tabular}{lrrrrrrll}",
            r"\toprule",
            r"Candidate & Trial & $k$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ & Note \\",
            r"\midrule",
            *[tex_local_candidate_row(row) for row in candidates],
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Top six SNAKE Optuna/replay candidate views for this regime. Proxy rows use Optuna graph/S-alg attributes; replay rows use literal compile sidecars. Missing rows mean no completed candidate in the configured source yet.}",
            r"\end{table}",
        ]
    )


def render_canonical_review_tex(payload: Mapping[str, Any], tex_path: Path) -> None:
    generated = payload.get("generated_utc") or datetime.now(timezone.utc).isoformat()
    lines: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{booktabs,longtable,array,float,xcolor,hyperref}",
        r"\usepackage{amsmath}",
        r"\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.45em}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large Paper-I Hubbard--Holstein SNAKE Optuna Overlay Review}\\",
        r"{\normalsize Standalone diagnostic PDF; no manuscript table, plot, or provenance files are changed.}\\",
        rf"{{\small Auto-refreshed {tex_escape(generated)}; sidecar JSON written beside this PDF.}}",
        r"\end{center}",
        r"\textbf{Scope.} This report copies current visible Paper-I Hubbard--Holstein table rows and adds SNAKE Optuna candidate views from persistent local studies. Rows are for review only.",
        r"\textbf{Cost semantics.} Visible/replay rows use the recorded Paper-I table or replay compile fields. Optuna candidate rows use graph proxy attributes ($N_{2q}$, $D_{2q}$) and $S_{\rm alg}$ when available; a blocked $S_{\rm alg}$ is shown explicitly and is not replaced by terminal cost.",
    ]

    visible = payload.get("visible_rows") or {}
    for key in ["weak-weak", "intermediate-weak", "weak-strong", "intermediate-strong"]:
        block = visible.get(key) if isinstance(visible, Mapping) else None
        rows = block.get("rows") if isinstance(block, Mapping) else None
        if not isinstance(rows, list):
            continue
        lines.extend(
            [
                rf"\section*{{{tex_escape(block.get('label') or key)}}}",
                r"\begin{table}[H]\centering\small",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                r"Method & $k$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ \\",
                r"\midrule",
                *[tex_visible_row(row) for row in rows if isinstance(row, Mapping)],
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Visible Paper-I plateau rows retained for comparison.}",
                r"\end{table}",
            ]
        )
        append_candidate_block(lines, payload, key)

    u8_visible = payload.get("u8_visible_rows") or {}
    for key in ["strong-weak-u8", "strong-strong-u8"]:
        block = u8_visible.get(key) if isinstance(u8_visible, Mapping) else None
        rows = block.get("rows") if isinstance(block, Mapping) else None
        if not isinstance(rows, list):
            continue
        lines.extend(
            [
                rf"\section*{{{tex_escape(block.get('label') or key)}}}",
                r"\begin{table}[H]\centering\small",
                r"\begin{tabular}{lrrrrrrr}",
                r"\toprule",
                r"Method & $n_{\rm ph}$ & $k$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ \\",
                r"\midrule",
                *[tex_visible_row(row, include_nph=True) for row in rows if isinstance(row, Mapping)],
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Visible U/t=8 Paper-I rows from the manuscript provenance block.}",
                r"\end{table}",
            ]
        )
        append_candidate_block(lines, payload, key)

    lines.extend(
        [
            r"\section*{Provenance note}",
            rf"Sidecar JSON: \texttt{{{tex_escape(str(tex_path.with_suffix('.json').relative_to(ROOT)))}}}.",
            r"Optuna candidate views are regenerated from local SQLite trial attributes. Replay rows retain the seed review's compile-sidecar semantics. This report is evidence for review and does not decide promotion into Paper I.",
            r"\end{document}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_tex(tex_path: Path) -> tuple[bool, str]:
    extra_paths = [
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
        Path("/Library/TeX/texbin"),
    ]

    def resolve_exe(name: str) -> str | None:
        found = shutil.which(name)
        if found:
            return found
        for parent in extra_paths:
            candidate = parent / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
        return None

    latexmk = resolve_exe("latexmk")
    tectonic = resolve_exe("tectonic")
    pdflatex = resolve_exe("pdflatex")
    if latexmk:
        cmd = [latexmk, "-pdf", "-interaction=nonstopmode", tex_path.name]
    elif tectonic:
        cmd = [tectonic, "--keep-logs", "--reruns", "2", tex_path.name]
    elif pdflatex:
        cmd = [pdflatex, "-interaction=nonstopmode", tex_path.name]
    else:
        return False, "no LaTeX engine found"
    proc = subprocess.run(cmd, cwd=tex_path.parent, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    return proc.returncode == 0, proc.stdout[-4000:]


def refresh_once(stem: str, *, canonical_review: bool = False, latest_stem: str | None = "paper_i_hh_snake_optuna_overlay_review_live_latest") -> dict[str, Any]:
    payload = build_payload(stem, include_plots=not bool(canonical_review))
    json_path = OUT_DIR / f"{stem}.json"
    tex_path = OUT_DIR / f"{stem}.tex"
    write_json(json_path, payload)
    if bool(canonical_review):
        render_canonical_review_tex(payload, tex_path)
    else:
        render_tex(payload, tex_path)
    ok, log = compile_tex(tex_path)
    payload["build"] = {"ok": ok, "log_tail": log}
    write_json(json_path, payload)
    if latest_stem:
        for suffix in [".json", ".tex", ".pdf", ".log", ".aux"]:
            src = OUT_DIR / f"{stem}{suffix}"
            dst = OUT_DIR / f"{latest_stem}{suffix}"
            if src.exists() and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
    return {"json": str(json_path), "tex": str(tex_path), "pdf": str(tex_path.with_suffix(".pdf")), "build_ok": ok}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default="paper_i_hh_snake_optuna_overlay_review_live")
    parser.add_argument("--latest-stem", default="paper_i_hh_snake_optuna_overlay_review_live_latest")
    parser.add_argument("--canonical-review", action="store_true", help="Render the table-only canonical review PDF format.")
    parser.add_argument("--no-latest-copy", action="store_true")
    parser.add_argument("--interval-s", type=float, default=900.0)
    parser.add_argument("--duration-hours", type=float, default=0.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deadline = None if args.once or args.duration_hours <= 0 else time.monotonic() + args.duration_hours * 3600.0
    while True:
        stamped = args.stem if args.once else f"{args.stem}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        latest_stem = None if args.no_latest_copy else str(args.latest_stem)
        result = refresh_once(stamped, canonical_review=bool(args.canonical_review), latest_stem=latest_stem)
        print(json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(), **result}, sort_keys=True), flush=True)
        if args.once or (deadline is not None and time.monotonic() >= deadline):
            return 0
        time.sleep(max(30.0, float(args.interval_s)))


if __name__ == "__main__":
    raise SystemExit(main())
