#!/usr/bin/env python3
"""All-regime status view for local Paper-I HH SNAKE Optuna supervisors.

This script is intentionally read-only.  It summarizes the persistent Optuna
SQLite studies plus the latest per-regime ``current_best.json`` files produced
by ``paper_i_hh_local_optuna_supervisor`` / ``hh_cost_energy_optuna``.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLUG = "paper_i_hh_local_geo_energy_graphcost_20260615_night_v1"
DEFAULT_TARGET_MANIFEST = REPO_ROOT / "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_geo_targets_20260615.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "raw_outputs/local_hh_optuna_supervisor"
DEFAULT_STORAGE_ROOT = REPO_ROOT / "raw_outputs/optuna_studies/local_hh_optuna_supervisor"


@dataclass(frozen=True)
class RegimeTarget:
    regime: str
    display_label: str
    geo_abs_delta_e: float | None
    geo_iteration: int | None
    geo_n2q: float | None
    geo_d2q: float | None
    geo_dc: float | None
    geo_s_alg: float | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except Exception:
        return None


def _loads_attr(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _graph_plus_shot_value(row: Mapping[str, Any]) -> float:
    explicit = _maybe_float(row.get("graph_plus_shot_objective_scalar"))
    if explicit is not None:
        return float(explicit)
    graph_count_2q = _maybe_float(row.get("graph_count_2q"))
    graph_depth = _maybe_float(row.get("graph_depth"))
    if graph_count_2q is None or graph_depth is None:
        return float("inf")
    s_alg = _maybe_float(row.get("paper_i_table_s_alg"))
    if s_alg is None:
        s_alg = _maybe_float(row.get("paper_i_shot_cost_scalar"))
    if s_alg is None:
        s_alg = _maybe_float(row.get("paper_i_table_shots_total"))
    if s_alg is None:
        s_alg = 1.0e12
    return float(max(0.0, graph_count_2q) * 1.0e9 + max(0.0, graph_depth) * 1.0e6 + max(0.0, s_alg))


def _targets(manifest_path: Path) -> tuple[list[str], dict[str, RegimeTarget]]:
    manifest = _load_json(manifest_path)
    regimes = manifest.get("regimes", {})
    ordered: list[str] = []
    for pair in manifest.get("cycle_pairs", []):
        for regime in pair:
            if regime not in ordered:
                ordered.append(str(regime))
    for regime in regimes:
        if str(regime) not in ordered:
            ordered.append(str(regime))
    targets: dict[str, RegimeTarget] = {}
    for regime in ordered:
        row = regimes.get(regime, {})
        targets[regime] = RegimeTarget(
            regime=regime,
            display_label=str(row.get("display_label") or regime),
            geo_abs_delta_e=_maybe_float(row.get("geo_abs_delta_e")),
            geo_iteration=_maybe_int(row.get("geo_iteration")),
            geo_n2q=_maybe_float(row.get("geo_N2q")),
            geo_d2q=_maybe_float(row.get("geo_D2q")),
            geo_dc=_maybe_float(row.get("geo_Dc")),
            geo_s_alg=_maybe_float(row.get("geo_S_alg")),
        )
    return ordered, targets


def _trial_values_schema(con: sqlite3.Connection) -> str | None:
    try:
        cols = [row[1] for row in con.execute("pragma table_info(trial_values)")]
    except Exception:
        return None
    if "value" in cols:
        return "value"
    if "value_json" in cols:
        return "value_json"
    return None


def _trial_objective_value(con: sqlite3.Connection, trial_id: int, value_col: str | None) -> float | None:
    if value_col is None:
        return None
    try:
        if value_col == "value":
            rows = [row[0] for row in con.execute("select value from trial_values where trial_id=? order by objective", (trial_id,))]
        else:
            rows = [_loads_attr(row[0]) for row in con.execute("select value_json from trial_values where trial_id=? order by objective", (trial_id,))]
    except Exception:
        return None
    return _maybe_float(rows[0]) if rows else None


def _trial_attrs(con: sqlite3.Connection, trial_id: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        for key, value_json in con.execute("select key,value_json from trial_user_attributes where trial_id=?", (trial_id,)):
            out[str(key)] = _loads_attr(value_json)
    except Exception:
        pass
    return out


def _study_ids_for_prefix(con: sqlite3.Connection, prefix: str) -> list[int]:
    try:
        rows = con.execute(
            "select study_id from studies where study_name like ? order by study_id",
            (f"{prefix}_%",),
        ).fetchall()
    except Exception:
        return []
    return [int(row[0]) for row in rows]


def _study_filter_clause(study_ids: list[int]) -> tuple[str, tuple[int, ...]]:
    if not study_ids:
        return "", ()
    placeholders = ",".join("?" for _ in study_ids)
    return f" and study_id in ({placeholders})", tuple(int(x) for x in study_ids)


def _graph_dominance_payload(row: Mapping[str, Any] | None, target: RegimeTarget) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {
            "graph_dominates_geo": None,
            "graph_count_2q_gap_vs_geo": None,
            "graph_depth_gap_vs_geo": None,
            "reason": "missing_energy_feasible_row",
        }
    n2q = _maybe_float(row.get("graph_count_2q"))
    d2q = _maybe_float(row.get("graph_depth"))
    n2q_gap = None if n2q is None or target.geo_n2q is None else float(n2q) - float(target.geo_n2q)
    d2q_gap = None if d2q is None or target.geo_d2q is None else float(d2q) - float(target.geo_d2q)
    dominates = None
    reason = None
    if n2q_gap is None or d2q_gap is None:
        reason = "missing_graph_or_geo_cost"
    else:
        dominates = bool(n2q_gap <= 0.0 and d2q_gap <= 0.0)
    return {
        "graph_dominates_geo": dominates,
        "graph_count_2q_gap_vs_geo": n2q_gap,
        "graph_depth_gap_vs_geo": d2q_gap,
        "reason": reason,
    }


def _summarize_db(db_path: Path, target: RegimeTarget, *, study_prefixes: list[str] | None = None) -> dict[str, Any]:
    if not db_path.exists():
        return {"db_exists": False, "states": {}, "complete_count": 0, "running_trials": []}
    con = sqlite3.connect(db_path)
    try:
        selected_ids: list[int] = []
        selected_prefix: str | None = None
        for prefix in study_prefixes or []:
            ids = _study_ids_for_prefix(con, prefix)
            if ids:
                selected_ids = ids
                selected_prefix = str(prefix)
                break
        study_clause, study_params = _study_filter_clause(selected_ids)
        states = {str(state): int(count) for state, count in con.execute(f"select state, count(*) from trials where 1=1{study_clause} group by state order by state", study_params)}
        value_col = _trial_values_schema(con)
        complete_rows: list[dict[str, Any]] = []
        for trial_id, number in con.execute(f"select trial_id, number from trials where state='COMPLETE'{study_clause} order by trial_id", study_params):
            attrs = _trial_attrs(con, int(trial_id))
            abs_delta_e = _maybe_float(attrs.get("abs_delta_e"))
            feasible = bool(attrs.get("feasible"))
            energy_feasible = (
                abs_delta_e is not None
                and target.geo_abs_delta_e is not None
                and float(abs_delta_e) <= float(target.geo_abs_delta_e)
            )
            graph_count_2q = _maybe_float(attrs.get("graph_count_2q"))
            graph_depth = _maybe_float(attrs.get("graph_depth"))
            graph_dominates_geo = (
                energy_feasible
                and graph_count_2q is not None
                and graph_depth is not None
                and target.geo_n2q is not None
                and target.geo_d2q is not None
                and float(graph_count_2q) <= float(target.geo_n2q)
                and float(graph_depth) <= float(target.geo_d2q)
            )
            paper_i_table_s_alg = _maybe_float(attrs.get("paper_i_table_s_alg"))
            s_alg_dominates_geo = (
                energy_feasible
                and paper_i_table_s_alg is not None
                and target.geo_s_alg is not None
                and float(paper_i_table_s_alg) <= float(target.geo_s_alg)
            )
            complete_rows.append(
                {
                    "trial_id": int(trial_id),
                    "trial_number": int(number),
                    "objective_value": _trial_objective_value(con, int(trial_id), value_col),
                    "abs_delta_e": abs_delta_e,
                    "geo_energy_gap": None if abs_delta_e is None or target.geo_abs_delta_e is None else float(abs_delta_e) - float(target.geo_abs_delta_e),
                    "feasible": feasible,
                    "energy_feasible": energy_feasible,
                    "graph_dominates_geo": bool(graph_dominates_geo),
                    "adapt_iteration_count": _maybe_int(attrs.get("adapt_iteration_count")),
                    "graph_count_2q": graph_count_2q,
                    "graph_depth": graph_depth,
                    "paper_i_table_shots_total": _maybe_float(attrs.get("paper_i_table_shots_total")),
                    "paper_i_table_s_alg": paper_i_table_s_alg,
                    "s_alg_dominates_geo": bool(s_alg_dominates_geo),
                    "s_alg_gap_vs_geo": None if paper_i_table_s_alg is None or target.geo_s_alg is None else float(paper_i_table_s_alg) - float(target.geo_s_alg),
                    "paper_i_table_shots_status": attrs.get("paper_i_table_shots_status"),
                    "graph_hardware_objective_scalar": _maybe_float(attrs.get("graph_hardware_objective_scalar")),
                    "paper_i_shot_cost_scalar": _maybe_float(attrs.get("paper_i_shot_cost_scalar")),
                    "graph_plus_shot_objective_scalar": _maybe_float(attrs.get("graph_plus_shot_objective_scalar")),
                    "result_json": attrs.get("result_json"),
                    "case_dir": attrs.get("case_dir"),
                }
            )
        running_trials = [
            {"trial_id": int(trial_id), "trial_number": int(number), "datetime_start": str(datetime_start)}
            for trial_id, number, datetime_start in con.execute(f"select trial_id, number, datetime_start from trials where state='RUNNING'{study_clause} order by trial_id", study_params)
        ]
    
    finally:
        con.close()
    best_by_delta_e = min((row for row in complete_rows if row["abs_delta_e"] is not None), key=lambda row: float(row["abs_delta_e"]), default=None)
    feasible_rows = [row for row in complete_rows if bool(row.get("energy_feasible"))]
    graph_dominant_rows = [row for row in feasible_rows if bool(row.get("graph_dominates_geo"))]
    graph_shot_dominant_rows = [
        row
        for row in graph_dominant_rows
        if target.geo_s_alg is None or bool(row.get("s_alg_dominates_geo"))
    ]
    best_feasible_by_graph = min(feasible_rows, key=_graph_plus_shot_value, default=None)
    best_graph_dominant_by_graph = min(graph_shot_dominant_rows, key=_graph_plus_shot_value, default=None)
    graph_dominance = _graph_dominance_payload(best_feasible_by_graph, target)
    return {
        "db_exists": True,
        "study_filter": {
            "selected_prefix": selected_prefix,
            "selected_study_ids": selected_ids,
        },
        "states": states,
        "complete_count": len(complete_rows),
        "running_trials": running_trials,
        "energy_feasible_count": len(feasible_rows),
        "geo_graph_dominant_count": len(graph_dominant_rows),
        "geo_graph_shot_dominant_count": len(graph_shot_dominant_rows),
        "best_by_delta_e": best_by_delta_e,
        "best_energy_feasible_by_graph_cost": best_feasible_by_graph,
        "best_energy_feasible_by_graph_plus_shot": best_feasible_by_graph,
        "best_energy_feasible_graph_dominance_vs_geo": graph_dominance,
        "best_geo_dominant_by_graph_plus_shot": best_graph_dominant_by_graph,
    }


def _latest_current_best(supervisor_root: Path, regime: str) -> dict[str, Any] | None:
    candidates = sorted(supervisor_root.glob(f"cycle_*/{regime}/current_best.json"), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    for path in candidates:
        try:
            payload = _load_json(path)
            payload["source_path"] = str(path)
            return payload
        except Exception:
            continue
    return None


def _trial_number_from_path(path: Path) -> int | None:
    match = re.search(r"trial_(\d+)", str(path))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _latest_live_current(supervisor_root: Path, regime: str) -> dict[str, Any] | None:
    """Return a compact view of the freshest live trial ``current.json``.

    The completed-trial incumbent lives in ``current_best.json``.  While a trial
    is still running, the most useful heartbeat is the per-trial ``current.json``
    written by ``adapt_pipeline``; it carries the current beam depth and current
    delta-E even before Optuna has a completed objective value.
    """
    candidates = sorted(
        supervisor_root.glob(f"cycle_*/{regime}/canonical/eps_*/trial_*/current.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(adapt_vqe, Mapping):
            continue
        beam = adapt_vqe.get("beam_replay_telemetry", {})
        if not isinstance(beam, Mapping):
            beam = {}
        leading = beam.get("leading_branch", {})
        if not isinstance(leading, Mapping):
            leading = {}
        current_delta = (
            _maybe_float(adapt_vqe.get("benchmark_target_abs_delta_e_current"))
            or _maybe_float(adapt_vqe.get("abs_delta_e"))
            or _maybe_float(leading.get("benchmark_target_abs_delta_current"))
        )
        prefix = (
            _maybe_int(beam.get("depth"))
            or _maybe_int(leading.get("depth_local"))
            or _maybe_int(adapt_vqe.get("ansatz_depth"))
        )
        ansatz_depth = _maybe_int(adapt_vqe.get("ansatz_depth") or leading.get("ansatz_depth"))
        return {
            "source_path": str(path),
            "mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "trial_number": _trial_number_from_path(path),
            "prefix": prefix,
            "ansatz_depth": ansatz_depth,
            "abs_delta_e": current_delta,
            "energy": _maybe_float(adapt_vqe.get("energy") or leading.get("energy")),
        }
    return None


def _completed_prefix_best(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    current_best = row.get("current_best", {}) if isinstance(row, Mapping) else {}
    if not isinstance(current_best, Mapping):
        return None
    prefix_best = current_best.get("prefix_best", {})
    if not isinstance(prefix_best, Mapping):
        return None
    prefix = prefix_best.get("global_current_best_prefix")
    return prefix if isinstance(prefix, Mapping) else None


def _active_rows(supervisor_root: Path) -> dict[str, dict[str, Any]]:
    path = supervisor_root / "supervisor_status.json"
    if not path.exists():
        return {}
    try:
        payload = _load_json(path)
    except Exception:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows", []) if isinstance(payload, Mapping) else []:
        regime = row.get("regime")
        if regime is not None:
            rows[str(regime)] = dict(row)
    return rows


def build_status(*, slug: str, target_manifest: Path, output_root: Path, storage_root: Path) -> dict[str, Any]:
    ordered, targets = _targets(target_manifest)
    supervisor_root = output_root / slug
    study_root = storage_root / slug
    active = _active_rows(supervisor_root)
    rows: list[dict[str, Any]] = []
    for regime in ordered:
        target = targets[regime]
        regime_key = regime.replace("-", "_")
        db_summary = _summarize_db(
            study_root / f"{regime}.sqlite3",
            target,
            study_prefixes=[
                f"{slug}_{regime_key}_graphdomshot_salg_v5",
                f"{slug}_{regime_key}_graphdomshot_v4",
                f"{slug}_{regime_key}_graphdom_v3",
                f"{slug}_{regime_key}_graphshot_v2",
                f"{slug}_{regime_key}",
            ],
        )
        current_best = _latest_current_best(supervisor_root, regime)
        live_current = _latest_live_current(supervisor_root, regime)
        active_row = active.get(regime)
        row = {
            "regime": regime,
            "display_label": target.display_label,
            "target": {
                "geo_abs_delta_e": target.geo_abs_delta_e,
                "geo_iteration": target.geo_iteration,
                "geo_N2q": target.geo_n2q,
                "geo_D2q": target.geo_d2q,
                "geo_Dc": target.geo_dc,
                "geo_S_alg": target.geo_s_alg,
            },
            "active": active_row is not None,
            "active_state": None if active_row is None else active_row.get("state"),
            "active_pid": None if active_row is None else active_row.get("pid"),
            "db": db_summary,
            "current_best": current_best,
            "live_current": live_current,
        }
        rows.append(row)
    return {
        "schema": "paper_i_hh_local_optuna_all_regime_status_v2",
        "generated_utc": _utc_now(),
        "slug": slug,
        "supervisor_root": str(supervisor_root),
        "storage_root": str(study_root),
        "target_manifest": str(target_manifest),
        "rows": rows,
    }


def _fmt_float(value: Any, *, sig: int = 4) -> str:
    x = _maybe_float(value)
    if x is None:
        return "--"
    return f"{x:.{sig}g}"


def _best_cell(row: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    db = row.get("db", {}) if isinstance(row, Mapping) else {}
    value = db.get(key) if isinstance(db, Mapping) else None
    return value if isinstance(value, Mapping) else None


def status_markdown(status: Mapping[str, Any]) -> str:
    lines = [
        f"Status source: local Optuna SQLite + current_best JSON; fresh as of {status.get('generated_utc')}",
        "",
        "| Regime | State | complete/running/fail | best deltaE | best prefix | live prefix | Geo target | energy gate | Geo graph | best energy-feasible graph+shot | graph gate | best Geo-dominant graph+shot | S_alg | trial |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in status.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        db = row.get("db", {}) if isinstance(row.get("db"), Mapping) else {}
        states = db.get("states", {}) if isinstance(db.get("states"), Mapping) else {}
        complete = int(db.get("complete_count") or 0)
        running = len(db.get("running_trials") or [])
        fail = int(states.get("FAIL") or states.get("FAILED") or 0)
        best_de = _best_cell(row, "best_by_delta_e")
        best_feasible = _best_cell(row, "best_energy_feasible_by_graph_plus_shot") or _best_cell(row, "best_energy_feasible_by_graph_cost")
        best_geo_dominant = _best_cell(row, "best_geo_dominant_by_graph_plus_shot")
        completed_prefix = _completed_prefix_best(row)
        live_current = row.get("live_current") if isinstance(row.get("live_current"), Mapping) else None
        target = row.get("target", {}) if isinstance(row.get("target"), Mapping) else {}
        dominance = db.get("best_energy_feasible_graph_dominance_vs_geo", {}) if isinstance(db.get("best_energy_feasible_graph_dominance_vs_geo"), Mapping) else {}
        gate = "missing"
        if best_de and best_de.get("abs_delta_e") is not None and target.get("geo_abs_delta_e") is not None:
            gate = "pass" if float(best_de["abs_delta_e"]) <= float(target["geo_abs_delta_e"]) else "above"
        state = "running" if row.get("active") else ("not started" if not row.get("db", {}).get("db_exists") else "idle/done")
        trial = "--"
        if best_geo_dominant:
            trial = str(best_geo_dominant.get("trial_number") if best_geo_dominant.get("trial_number") is not None else "--")
        elif best_feasible:
            trial = str(best_feasible.get("trial_number") if best_feasible.get("trial_number") is not None else "--")
        elif best_de:
            trial = str(best_de.get("trial_number") if best_de.get("trial_number") is not None else "--")
        graph_gate = "--"
        if dominance.get("graph_dominates_geo") is True:
            graph_gate = "pass"
        elif dominance.get("graph_dominates_geo") is False:
            n2q_gap = _fmt_float(dominance.get("graph_count_2q_gap_vs_geo"), sig=4)
            d2q_gap = _fmt_float(dominance.get("graph_depth_gap_vs_geo"), sig=4)
            graph_gate = f"above dN={n2q_gap},dD={d2q_gap}"
        elif dominance.get("reason"):
            graph_gate = str(dominance.get("reason"))
        lines.append(
            "| {regime} | {state} | {counts} | {best_de} | {best_prefix} | {live_prefix} | {target} | {gate} | {geo_graph} | {graph} | {graph_gate} | {geo_dom} | {shots} | {trial} |".format(
                regime=row.get("display_label") or row.get("regime"),
                state=state,
                counts=f"{complete}/{running}/{fail}",
                best_de="--" if not best_de else _fmt_float(best_de.get("abs_delta_e")),
                best_prefix=(
                    "--"
                    if not completed_prefix
                    else "k={k},trial={trial},dE={de}".format(
                        k=completed_prefix.get("iteration", "--"),
                        trial=completed_prefix.get("trial_number", "--"),
                        de=_fmt_float(completed_prefix.get("abs_delta_e")),
                    )
                ),
                live_prefix=(
                    "--"
                    if not live_current
                    else "k={k},trial={trial},dE={de}".format(
                        k=live_current.get("prefix", "--"),
                        trial=live_current.get("trial_number", "--"),
                        de=_fmt_float(live_current.get("abs_delta_e")),
                    )
                ),
                target=_fmt_float(target.get("geo_abs_delta_e")),
                gate=gate,
                geo_graph=f"N2q={_fmt_float(target.get('geo_N2q'), sig=5)},D={_fmt_float(target.get('geo_D2q'), sig=5)}",
                graph="--" if not best_feasible else f"N2q={_fmt_float(best_feasible.get('graph_count_2q'), sig=5)},D={_fmt_float(best_feasible.get('graph_depth'), sig=5)}",
                graph_gate=graph_gate,
                geo_dom="--" if not best_geo_dominant else f"N2q={_fmt_float(best_geo_dominant.get('graph_count_2q'), sig=5)},D={_fmt_float(best_geo_dominant.get('graph_depth'), sig=5)}",
                shots="--" if not (best_geo_dominant or best_feasible) else _fmt_float((best_geo_dominant or best_feasible).get("paper_i_table_s_alg"), sig=5),
                trial=trial,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--target-manifest", type=Path, default=DEFAULT_TARGET_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    status = build_status(
        slug=str(args.slug),
        target_manifest=Path(args.target_manifest),
        output_root=Path(args.output_root),
        storage_root=Path(args.storage_root),
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if args.markdown:
        print(status_markdown(status), end="")
    else:
        print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
