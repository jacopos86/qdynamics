#!/usr/bin/env python3
"""Persistent Pareto tracking for staged HH noiseless workflow runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "hh_staged_pareto_v1"
OBJECTIVE_AXES = (
    "delta_E_abs",
    "measurement_groups_cumulative",
    "compile_gate_proxy_cumulative",
)
DIAGNOSTIC_AXES = (
    "measurement_shots_cumulative",
    "compile_cx_proxy_cumulative",
    "compile_sq_proxy_cumulative",
)


@dataclass(frozen=True)
class HHParetoRow:
    schema: str = SCHEMA_VERSION
    run_tag: str = ""
    run_id: str = ""
    recorded_utc: str = ""
    problem: str = "hh"
    stage_kind: str = ""
    method_kind: str = ""
    continuation_mode: str = ""
    ansatz_name: str = ""
    pool_name: str = ""
    selection_mode: str = ""
    status: str = "ok"
    L: int | None = None
    t: float | None = None
    u: float | None = None
    dv: float | None = None
    omega0: float | None = None
    g_ep: float | None = None
    n_ph_max: int | None = None
    stage_depth: int | None = None
    depth_proxy: int | None = None
    num_parameters: int | None = None
    energy: float | None = None
    exact_energy: float | None = None
    delta_E_abs: float | None = None
    delta_E_abs_drop_from_prev: float | None = None
    measurement_groups_new: int | None = None
    measurement_groups_cumulative: int | None = None
    measurement_shots_new: float | None = None
    measurement_shots_cumulative: float | None = None
    measurement_reuse_cost_new: float | None = None
    measurement_reuse_cost_cumulative: float | None = None
    compile_gate_proxy_new: float | None = None
    compile_gate_proxy_cumulative: float | None = None
    compile_cx_proxy_new: float | None = None
    compile_cx_proxy_cumulative: float | None = None
    compile_sq_proxy_new: float | None = None
    compile_sq_proxy_cumulative: float | None = None
    max_pauli_weight_new: float | None = None
    batch_size: int | None = None
    runtime_split_mode: str = ""
    runtime_split_child_count: int | None = None
    delta_E_drop_per_new_group: float | None = None
    delta_E_drop_per_new_gate_proxy: float | None = None
    pareto_eligible: bool = False


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


"""
ΔE_drop / cost = (ΔE_abs_prev - ΔE_abs_cur) / max(cost, 1)
"""
def _drop_per_cost(drop: float | None, cost: float | None) -> float | None:
    drop_f = _to_float(drop)
    cost_f = _to_float(cost)
    if drop_f is None or cost_f is None:
        return None
    if cost_f <= 0.0:
        return drop_f if drop_f > 0.0 else None
    return float(drop_f / cost_f)


def _base_row(
    *,
    run_tag: str,
    run_id: str,
    recorded_utc: str,
    stage_kind: str,
    method_kind: str,
    continuation_mode: str,
    ansatz_name: str,
    pool_name: str,
    physics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_tag": str(run_tag),
        "run_id": str(run_id),
        "recorded_utc": str(recorded_utc),
        "stage_kind": str(stage_kind),
        "method_kind": str(method_kind),
        "continuation_mode": str(continuation_mode),
        "ansatz_name": str(ansatz_name),
        "pool_name": str(pool_name),
        "L": _to_int(physics.get("L")),
        "t": _to_float(physics.get("t")),
        "u": _to_float(physics.get("u")),
        "dv": _to_float(physics.get("dv")),
        "omega0": _to_float(physics.get("omega0")),
        "g_ep": _to_float(physics.get("g_ep")),
        "n_ph_max": _to_int(physics.get("n_ph_max")),
    }


"""
N_init = N_final - Σ_d batch_size(d)
N_params(d) = N_init + Σ_{k<=d} batch_size(k)
M_cum(d) = Σ_{k<=d} M_new(k)
G_cum(d) = Σ_{k<=d} G_new(k)
"""
def extract_staged_hh_pareto_rows(
    *,
    run_tag: str,
    physics: Mapping[str, Any],
    warm_payload: Mapping[str, Any],
    adapt_payload: Mapping[str, Any],
    replay_payload: Mapping[str, Any],
    recorded_utc: str | None = None,
) -> list[HHParetoRow]:
    stamp = _now_utc() if recorded_utc is None else str(recorded_utc)
    rows: list[HHParetoRow] = []

    warm_base = _base_row(
        run_tag=run_tag,
        run_id=f"{run_tag}|warm_start",
        recorded_utc=stamp,
        stage_kind="warm_start",
        method_kind="warm_start",
        continuation_mode="",
        ansatz_name=str(warm_payload.get("ansatz", "")),
        pool_name="",
        physics=physics,
    )
    warm_energy = _to_float(warm_payload.get("energy"))
    warm_exact = _to_float(warm_payload.get("exact_filtered_energy"))
    rows.append(
        HHParetoRow(
            **warm_base,
            num_parameters=_to_int(warm_payload.get("num_parameters")),
            energy=warm_energy,
            exact_energy=warm_exact,
            delta_E_abs=(
                None
                if warm_energy is None or warm_exact is None
                else float(abs(float(warm_energy) - float(warm_exact)))
            ),
            pareto_eligible=False,
        )
    )

    history = adapt_payload.get("history", [])
    history_rows = list(history) if isinstance(history, Sequence) else []
    final_point = adapt_payload.get("optimal_point", [])
    final_num_parameters = len(final_point) if isinstance(final_point, Sequence) else None
    batch_sizes = [max(1, _to_int(_mapping_dict(row).get("batch_size")) or 1) for row in history_rows]
    initial_num_parameters = (
        max(0, int(final_num_parameters) - int(sum(batch_sizes)))
        if final_num_parameters is not None
        else 0
    )
    params_cumulative = int(initial_num_parameters)
    groups_cumulative = 0
    shots_cumulative = 0.0
    reuse_cumulative = 0.0
    gate_cumulative = 0.0
    cx_cumulative = 0.0
    sq_cumulative = 0.0

    for row in history_rows:
        rec = _mapping_dict(row)
        measurement_stats = _mapping_dict(rec.get("measurement_cache_stats"))
        compile_cost = _mapping_dict(rec.get("compile_cost_proxy"))
        batch_size = max(1, _to_int(rec.get("batch_size")) or 1)
        params_after_opt = _to_int(rec.get("num_parameters_after_opt"))
        if params_after_opt is not None:
            params_cumulative = int(params_after_opt)
        else:
            params_cumulative += int(batch_size)

        groups_new = _to_int(measurement_stats.get("groups_new")) or 0
        shots_new = _to_float(measurement_stats.get("shots_new")) or 0.0
        reuse_new = _to_float(measurement_stats.get("reuse_count_cost")) or 0.0
        gate_new = _to_float(compile_cost.get("gate_proxy_total")) or 0.0
        cx_new = _to_float(compile_cost.get("cx_proxy_total")) or 0.0
        sq_new = _to_float(compile_cost.get("sq_proxy_total")) or 0.0
        groups_cumulative += int(groups_new)
        shots_cumulative += float(shots_new)
        reuse_cumulative += float(reuse_new)
        gate_cumulative += float(gate_new)
        cx_cumulative += float(cx_new)
        sq_cumulative += float(sq_new)

        delta_cur = _to_float(rec.get("delta_abs_current"))
        delta_drop = _to_float(rec.get("delta_abs_drop_from_prev"))
        adapt_base = _base_row(
            run_tag=run_tag,
            run_id=f"{run_tag}|adapt_depth{_to_int(rec.get('depth')) or len(rows)}",
            recorded_utc=stamp,
            stage_kind="adapt_depth",
            method_kind="adapt",
            continuation_mode=str(
                rec.get("continuation_mode", adapt_payload.get("continuation_mode", ""))
            ),
            ansatz_name="staged_adapt",
            pool_name=str(rec.get("candidate_family", adapt_payload.get("pool_type", ""))),
            physics=physics,
        )
        rows.append(
            HHParetoRow(
                **adapt_base,
                selection_mode=str(rec.get("selection_mode", "")),
                stage_depth=_to_int(rec.get("depth")),
                depth_proxy=_to_int(rec.get("depth_cumulative", rec.get("depth"))),
                num_parameters=int(params_cumulative),
                energy=_to_float(rec.get("energy_after_opt")),
                exact_energy=_to_float(adapt_payload.get("exact_gs_energy")),
                delta_E_abs=delta_cur,
                delta_E_abs_drop_from_prev=delta_drop,
                measurement_groups_new=int(groups_new),
                measurement_groups_cumulative=int(groups_cumulative),
                measurement_shots_new=float(shots_new),
                measurement_shots_cumulative=float(shots_cumulative),
                measurement_reuse_cost_new=float(reuse_new),
                measurement_reuse_cost_cumulative=float(reuse_cumulative),
                compile_gate_proxy_new=float(gate_new),
                compile_gate_proxy_cumulative=float(gate_cumulative),
                compile_cx_proxy_new=float(cx_new),
                compile_cx_proxy_cumulative=float(cx_cumulative),
                compile_sq_proxy_new=float(sq_new),
                compile_sq_proxy_cumulative=float(sq_cumulative),
                max_pauli_weight_new=_to_float(compile_cost.get("max_pauli_weight")),
                batch_size=int(batch_size),
                runtime_split_mode=str(rec.get("runtime_split_mode", "")),
                runtime_split_child_count=_to_int(rec.get("runtime_split_child_count")),
                delta_E_drop_per_new_group=_drop_per_cost(delta_drop, groups_new),
                delta_E_drop_per_new_gate_proxy=_drop_per_cost(delta_drop, gate_new),
                pareto_eligible=(delta_cur is not None),
            )
        )

    replay_vqe = _mapping_dict(replay_payload.get("vqe"))
    replay_exact = _mapping_dict(replay_payload.get("exact"))
    replay_contract = _mapping_dict(replay_payload.get("replay_contract"))
    replay_base = _base_row(
        run_tag=run_tag,
        run_id=f"{run_tag}|conventional_replay",
        recorded_utc=stamp,
        stage_kind="conventional_replay",
        method_kind="conventional_replay",
        continuation_mode=str(replay_contract.get("continuation_mode", "")),
        ansatz_name="matched_family_replay",
        pool_name=str(_mapping_dict(replay_payload.get("generator_family")).get("resolved", "")),
        physics=physics,
    )
    rows.append(
        HHParetoRow(
            **replay_base,
            num_parameters=_to_int(replay_vqe.get("num_parameters")),
            energy=_to_float(replay_vqe.get("energy")),
            exact_energy=_to_float(replay_exact.get("E_exact_sector")),
            delta_E_abs=_to_float(replay_vqe.get("abs_delta_e")),
            pareto_eligible=False,
        )
    )

    return rows


def _row_value(row: Mapping[str, Any], key: str) -> float | None:
    return _to_float(row.get(key))


def _eligible_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        if not bool(rec.get("pareto_eligible", False)):
            continue
        if any(_row_value(rec, key) is None for key in OBJECTIVE_AXES):
            continue
        out.append(rec)
    return out


"""
x ≺ y iff all Pareto objectives of x are <= those of y and at least one is strict.
"""
def compute_pareto_frontier(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = _eligible_rows(rows)
    frontier: list[dict[str, Any]] = []
    for idx, candidate in enumerate(eligible):
        cand_vals = [_row_value(candidate, key) for key in OBJECTIVE_AXES]
        assert all(val is not None for val in cand_vals)
        dominated = False
        for jdx, other in enumerate(eligible):
            if idx == jdx:
                continue
            other_vals = [_row_value(other, key) for key in OBJECTIVE_AXES]
            assert all(val is not None for val in other_vals)
            no_worse = all(float(o) <= float(c) for o, c in zip(other_vals, cand_vals))
            strictly_better = any(float(o) < float(c) for o, c in zip(other_vals, cand_vals))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(
        key=lambda row: (
            float(_row_value(row, "delta_E_abs") or float("inf")),
            float(_row_value(row, "measurement_groups_cumulative") or float("inf")),
            float(_row_value(row, "compile_gate_proxy_cumulative") or float("inf")),
            int(_to_int(row.get("stage_depth")) or 0),
            str(row.get("run_tag", "")),
        )
    )
    return frontier


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).strip()
        if text == "":
            continue
        rows.append(dict(json.loads(text)))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _best_value(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    vals = [float(val) for val in (_row_value(row, key) for row in rows) if val is not None]
    return float(min(vals)) if vals else None


def _summary_payload(rows: Sequence[Mapping[str, Any]], frontier: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = _eligible_rows(rows)
    return {
        "schema": SCHEMA_VERSION,
        "objective_axes": list(OBJECTIVE_AXES),
        "diagnostic_axes": list(DIAGNOSTIC_AXES),
        "row_count": int(len(rows)),
        "eligible_row_count": int(len(eligible)),
        "frontier_count": int(len(frontier)),
        "best_delta_E_abs": _best_value(rows, "delta_E_abs"),
        "best_frontier_delta_E_abs": _best_value(frontier, "delta_E_abs"),
        "best_measurement_groups_cumulative": _best_value(frontier, "measurement_groups_cumulative"),
        "best_compile_gate_proxy_cumulative": _best_value(frontier, "compile_gate_proxy_cumulative"),
    }


def write_pareto_tracking(
    *,
    rows: Sequence[HHParetoRow],
    output_json_path: Path,
    run_tag: str,
    recorded_utc: str | None = None,
) -> dict[str, Any]:
    stamp = _now_utc() if recorded_utc is None else str(recorded_utc)
    output_json_path = Path(output_json_path)
    run_rows_path = output_json_path.with_name(f"{output_json_path.stem}_pareto_rows.json")
    run_frontier_path = output_json_path.with_name(f"{output_json_path.stem}_pareto_frontier.json")
    ledger_path = output_json_path.parent / "hh_staged_pareto_ledger.jsonl"
    rolling_frontier_path = output_json_path.parent / "hh_staged_pareto_frontier.json"

    row_dicts = [asdict(row) for row in rows]
    current_frontier = compute_pareto_frontier(row_dicts)

    run_rows_payload = {
        "schema": SCHEMA_VERSION,
        "generated_utc": stamp,
        "objective_axes": list(OBJECTIVE_AXES),
        "diagnostic_axes": list(DIAGNOSTIC_AXES),
        "rows": row_dicts,
    }
    run_frontier_payload = {
        "schema": SCHEMA_VERSION,
        "generated_utc": stamp,
        **_summary_payload(row_dicts, current_frontier),
        "frontier_rows": current_frontier,
    }
    run_rows_path.parent.mkdir(parents=True, exist_ok=True)
    run_rows_path.write_text(json.dumps(run_rows_payload, indent=2, sort_keys=True), encoding="utf-8")
    run_frontier_path.write_text(json.dumps(run_frontier_payload, indent=2, sort_keys=True), encoding="utf-8")

    ledger_rows = [row for row in _read_jsonl(ledger_path) if str(row.get("run_tag", "")) != str(run_tag)]
    ledger_rows.extend(row_dicts)
    ledger_rows.sort(
        key=lambda row: (
            str(row.get("run_tag", "")),
            int(_to_int(row.get("stage_depth")) or -1),
            str(row.get("stage_kind", "")),
            str(row.get("run_id", "")),
        )
    )
    _write_jsonl(ledger_path, ledger_rows)

    rolling_frontier = compute_pareto_frontier(ledger_rows)
    rolling_payload = {
        "schema": SCHEMA_VERSION,
        "generated_utc": stamp,
        **_summary_payload(ledger_rows, rolling_frontier),
        "frontier_rows": rolling_frontier,
    }
    rolling_frontier_path.write_text(json.dumps(rolling_payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "schema": SCHEMA_VERSION,
        "objective_axes": list(OBJECTIVE_AXES),
        "diagnostic_axes": list(DIAGNOSTIC_AXES),
        "paths": {
            "run_rows_json": run_rows_path,
            "run_frontier_json": run_frontier_path,
            "rolling_ledger_jsonl": ledger_path,
            "rolling_frontier_json": rolling_frontier_path,
        },
        "current_run": _summary_payload(row_dicts, current_frontier),
        "rolling": {
            "ledger_row_count": int(len(ledger_rows)),
            **_summary_payload(ledger_rows, rolling_frontier),
        },
    }
