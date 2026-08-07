#!/usr/bin/env python3
"""Build Paper-I HH prefix resources and matched SNAKE diagnostics.

This is a read-only support/audit script. It consumes a Paper-I HH support JSON
with method trajectories and source JSON pointers, reconstructs per-prefix
compiled resources when source metadata permits, and emits matched diagnostic
rows for SNAKE versus Append-ADAPT / Geo-ADAPT.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.hh_tableiii_prefix_resources import _compile_prefix_rows  # noqa: E402
from pipelines.reporting.build_paper_i_hh_pass2_costs_plots import (  # noqa: E402
    _compile_history_prefix as _compile_snake_history_prefix,
)

DEFAULT_SUPPORT_JSON = (
    REPO_ROOT
    / "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_manuscript_update_20260619.json"
)
DEFAULT_PLATEAU_JSON = (
    REPO_ROOT
    / "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_first_plateau_prefix_audit_20260619.json"
)
DEFAULT_OUTPUT_JSON = REPO_ROOT / "output/pdf/paper_i_hh_prefix_matched_diagnostics.json"
DEFAULT_PREFIX_CSV = REPO_ROOT / "output/pdf/paper_i_hh_prefix_matched_diagnostics_prefix_rows.csv"
DEFAULT_DIAGNOSTIC_CSV = REPO_ROOT / "output/pdf/paper_i_hh_prefix_matched_diagnostics_rows.csv"

METHOD_ORDER = ("Append-ADAPT", "Geo-ADAPT", "SNAKE")
COMPARATORS = ("Append-ADAPT", "Geo-ADAPT")
RESOURCE_AXES = ("N2q", "D_circ", "S")
EQUAL_ACCURACY_COSTS = ("N2q", "D2q", "D_circ", "S")
S_OK_STATUSES = {"ok"}
ENERGY_KEYS = ("energy_after_opt", "energy_after", "energy", "primary_energy_metric_after")
ERROR_KEYS = (
    "delta_abs_current",
    "benchmark_target_abs_delta_current",
    "abs_delta_e_same_cutoff_after",
    "abs_delta_e_after",
    "delta_E_abs_after",
    "abs_delta_e",
)


def _resolve(path: str | Path) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else REPO_ROOT / p


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int | None:
    x = _num(value)
    return None if x is None else int(round(x))


def _lower_better_stats(
    *,
    snake_value: float | None,
    comparator_value: float | None,
    comparator_label: str,
) -> dict[str, Any]:
    """Directional ratio summary for lower-is-better metrics.

    Negative percent improvements are intentionally not emitted. Percent-lower
    is always measured in the winner direction, and fold-higher is always
    measured from loser to winner.
    """

    if snake_value is None or comparator_value is None:
        return {
            "status": "blocked_missing_value",
            "winner": None,
            "snake_over_comparator_ratio": None,
            "winner_percent_lower": None,
            "loser_fold_higher": None,
        }
    snake = float(snake_value)
    comp = float(comparator_value)
    if snake < 0.0 or comp < 0.0:
        return {
            "status": "blocked_negative_value",
            "winner": None,
            "snake_over_comparator_ratio": None,
            "winner_percent_lower": None,
            "loser_fold_higher": None,
        }
    if snake == comp:
        return {
            "status": "tie",
            "winner": "tie",
            "snake_over_comparator_ratio": 1.0 if comp != 0.0 else None,
            "winner_percent_lower": 0.0,
            "loser_fold_higher": 1.0,
        }
    if snake == 0.0 or comp == 0.0:
        winner = "SNAKE" if snake < comp else comparator_label
        return {
            "status": "ok_zero_winner",
            "winner": winner,
            "snake_over_comparator_ratio": None if comp == 0.0 else snake / comp,
            "winner_percent_lower": 100.0,
            "loser_fold_higher": math.inf,
        }
    if snake < comp:
        return {
            "status": "ok",
            "winner": "SNAKE",
            "snake_over_comparator_ratio": snake / comp,
            "winner_percent_lower": 100.0 * (1.0 - snake / comp),
            "loser_fold_higher": comp / snake,
        }
    return {
        "status": "ok",
        "winner": comparator_label,
        "snake_over_comparator_ratio": snake / comp,
        "winner_percent_lower": 100.0 * (1.0 - comp / snake),
        "loser_fold_higher": snake / comp,
    }


def _add_lower_better_fields(
    target: dict[str, Any],
    *,
    prefix: str,
    snake_value: float | None,
    comparator_value: float | None,
    comparator_label: str,
) -> None:
    stats = _lower_better_stats(
        snake_value=snake_value,
        comparator_value=comparator_value,
        comparator_label=comparator_label,
    )
    target[f"{prefix}_comparison_status"] = stats["status"]
    target[f"{prefix}_winner"] = stats["winner"]
    target[f"{prefix}_snake_over_comparator_ratio"] = stats["snake_over_comparator_ratio"]
    target[f"{prefix}_winner_percent_lower"] = stats["winner_percent_lower"]
    target[f"{prefix}_loser_fold_higher"] = stats["loser_fold_higher"]


def _status_counts(rows: Iterable[Mapping[str, Any]], key: str = "status") -> dict[str, int]:
    counts: Counter[str] = Counter(str(row.get(key) or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def _history_from_source(payload: Mapping[str, Any], method: str) -> list[Mapping[str, Any]]:
    if method == "SNAKE":
        adapt = payload.get("adapt_vqe") if isinstance(payload.get("adapt_vqe"), Mapping) else {}
        history = adapt.get("history") or adapt.get("adapt_history") or adapt.get("history_tail") or []
    else:
        result = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
        history = result.get("adapt_history") or result.get("history") or []
    return [row for row in history if isinstance(row, Mapping)]


def _generic_prefix_s_values(payload: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> tuple[dict[int, float], str, dict[str, Any]]:
    """Recover cumulative comparator S_alg by prefix from native comparator history.

    This matches the current native comparator terminal convention:
    S_alg = N_H_refit_eval + N_grad_probe + N_metric_probe + N_other_quantum.
    """

    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
    prefix: dict[int, float] = {}
    refit = grad = metric = other = 0.0
    for idx, row in enumerate(history, start=1):
        refit += _num(row.get("optimizer_nfev")) or _num(row.get("nfev_opt")) or 0.0
        grad += _num(row.get("candidate_count_scored")) or 0.0
        metric += _num(row.get("selector_metric_probe_count")) or 0.0
        metric += _num(row.get("qngd_metric_operator_probe_count_total")) or 0.0
        metric += _num(row.get("qngd_metric_eval_count")) or 0.0
        other += _num(row.get("N_other_quantum")) or 0.0
        prefix[idx] = float(refit + grad + metric + other)

    terminal_source = _num(result.get("S_alg"))
    terminal_recovered = prefix.get(len(history))
    components = {
        "N_H_refit_eval": refit,
        "N_grad_probe": grad,
        "N_metric_probe": metric,
        "N_other_quantum": other,
        "terminal_recovered": terminal_recovered,
        "terminal_source": terminal_source,
    }
    if terminal_source is None:
        return prefix, "terminal_s_alg_missing_recovered_prefix_values", components
    if terminal_recovered is None or abs(float(terminal_recovered) - float(terminal_source)) > 1e-6:
        return prefix, "terminal_s_alg_mismatch_recovered_prefix_values", components
    return prefix, "ok", components


def _snake_prefix_s_values(
    *,
    regime: str,
    method: str,
    history: Sequence[Mapping[str, Any]],
    plateau_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[int, float], str, dict[str, Any]]:
    """Return only source-backed SNAKE prefix S values.

    Current native SNAKE source JSONs expose depth-scoped controller work, but
    the manuscript-facing S column uses the accepted Paper-I plateau audit value.
    Until the exact prefix S_alg reconstruction is formalized for every SNAKE
    prefix, expose only the audited plateau point and mark the route as
    single-point evidence.
    """

    plateau = plateau_by_key.get((regime, method), {})
    k_pl = _int(plateau.get("k_pl"))
    s_pl = _num(plateau.get("S_at_k_pl"))
    if k_pl is None or s_pl is None:
        return {}, "snake_prefix_s_blocked_missing_plateau_s", {}
    if k_pl < 1 or k_pl > len(history):
        return {}, "snake_prefix_s_blocked_plateau_outside_history", {"k_pl": k_pl, "history_len": len(history)}
    return {int(k_pl): float(s_pl)}, "single_point_from_plateau_audit", {"k_pl": k_pl, "S_at_k_pl": s_pl}


def _prefix_s_values(
    *,
    regime: str,
    method: str,
    payload: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    plateau_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[int, float], str, dict[str, Any]]:
    if method == "SNAKE":
        return _snake_prefix_s_values(regime=regime, method=method, history=history, plateau_by_key=plateau_by_key)
    return _generic_prefix_s_values(payload, history)


def _trajectory_fallback_rows(
    *,
    support_row: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    compile_status: str,
    compile_error: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in support_row.get("trajectory") or []:
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        k = _int(point[0])
        err = _num(point[1])
        if k is None or err is None:
            continue
        rows.append(
            {
                "schema": "paper_i_hh_prefix_matched_diagnostic_prefix_row_v1",
                "regime": support_row.get("regime"),
                "method": support_row.get("method"),
                "source_json": str(source_path.relative_to(REPO_ROOT)),
                "source_sha256": source_sha256,
                "prefix_k": k,
                "abs_delta_e": err,
                "compile_status": compile_status,
                "compile_error": compile_error,
                "N1q": None,
                "N2q": None,
                "D2q": None,
                "D_circ": None,
                "S": None,
                "S_status": "not_checked_due_compile_fallback",
            }
        )
    return rows


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[str | None, Any]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return key, value
    return None, None


def _snake_selected_labels(row: Mapping[str, Any]) -> list[str]:
    value = row.get("selected_ops")
    if isinstance(value, list):
        out = [str(item) for item in value if str(item).strip()]
        if out:
            return out
    for key in ("selected_op", "selected_logical_op", "selected_label"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
    return []


def _snake_compile_prefix_rows(
    *,
    regime: str,
    method: str,
    source_path: Path,
    source_sha256: str,
    payload: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    visible_cells: Mapping[str, Any],
    max_prefixes: int | None,
    progress: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    limit = len(history) if max_prefixes is None else min(len(history), int(max_prefixes))
    for idx, hist_row in enumerate(history[:limit], start=1):
        if progress:
            print(f"compile_prefix regime={regime} method={method} prefix={idx}", flush=True)
        try:
            compiled = _compile_snake_history_prefix(
                payload,
                history,
                idx,
                source_kind="paper_i_hh_native200_snake_history_prefix",
            )
            compile_status = "ok"
            compile_error = None
        except Exception as exc:
            compiled = {}
            compile_status = "prefix_compile_blocked"
            compile_error = str(exc)
        energy_key, energy = _first_present(hist_row, ENERGY_KEYS)
        error_key, error = _first_present(hist_row, ERROR_KEYS)
        rows.append(
            {
                "schema": "paper_i_hh_prefix_matched_diagnostic_prefix_row_v1",
                "regime": regime,
                "method": method,
                "source_json": str(source_path.relative_to(REPO_ROOT)),
                "source_sha256": source_sha256,
                "prefix_k": int(idx),
                "prefix_k_semantics": "adapt_vqe_history_row_index_1based",
                "adapt_iteration": hist_row.get("iteration", idx),
                "logical_operator_prefix_len": int(idx),
                "selected_batch_size": len(_snake_selected_labels(hist_row)),
                "selected_labels": _snake_selected_labels(hist_row),
                "selected_pauli_source": "adapt_vqe_history_parameterization_blocks",
                "prefix_order_semantics": "snake_history_parameterization_order",
                "energy": None if energy is None else float(energy),
                "energy_field": energy_key,
                "abs_delta_e": None if error is None else float(error),
                "abs_delta_e_field": error_key,
                "compile_status": compile_status,
                "compile_error": compile_error,
                "compile_convention": (compiled or {}).get("compile_convention"),
                "N1q": (compiled or {}).get("compiled_count_1q_total"),
                "N2q": (compiled or {}).get("compiled_count_2q_total"),
                "D_circ": (compiled or {}).get("compiled_depth_total"),
                "D2q": (compiled or {}).get("compiled_depth_2q_total"),
                "compiled_count_1q_semantics": (compiled or {}).get("compiled_count_1q_semantics"),
                "compiled_op_counts": (compiled or {}).get("compiled_op_counts"),
                "num_qubits": (compiled or {}).get("num_qubits"),
                "runtime_rotation_count": (compiled or {}).get("runtime_rotation_count"),
                "reference_state_status": "handled_by_snake_history_prefix_compiler",
                "visible_cells_terminal": dict(visible_cells),
            }
        )
    return rows


def _visible_cells(support_row: Mapping[str, Any], plateau_row: Mapping[str, Any] | None) -> dict[str, Any]:
    visible = {
        "support_reported_iteration": support_row.get("reported_iteration"),
        "support_active_depth": support_row.get("active_depth"),
        "terminal_same_cutoff_abs_delta_e": support_row.get("same_cutoff_abs_delta_e"),
        "terminal_N2q": support_row.get("N2q"),
        "terminal_D2q": support_row.get("D2q"),
        "terminal_D_circ": support_row.get("D_circ"),
        "terminal_S": support_row.get("S"),
    }
    if isinstance(plateau_row, Mapping):
        visible.update(
            {
                "k_pl": plateau_row.get("k_pl"),
                "plateau_same_cutoff_abs_delta_e": plateau_row.get("same_cutoff_abs_delta_e_at_k_pl"),
                "plateau_N2q": (plateau_row.get("compiled") or {}).get("N2q")
                if isinstance(plateau_row.get("compiled"), Mapping)
                else None,
                "plateau_D2q": (plateau_row.get("compiled") or {}).get("D2q")
                if isinstance(plateau_row.get("compiled"), Mapping)
                else None,
                "plateau_D_circ": (plateau_row.get("compiled") or {}).get("D_circ")
                if isinstance(plateau_row.get("compiled"), Mapping)
                else None,
                "plateau_S": plateau_row.get("S_at_k_pl"),
            }
        )
    return visible


def _normalize_prefix_rows(rows: Sequence[Mapping[str, Any]], *, source_rel: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["schema"] = "paper_i_hh_prefix_matched_diagnostic_prefix_row_v1"
        raw_source = str(item.get("source_json") or source_rel)
        try:
            source_path = _resolve(raw_source)
            item["source_json"] = str(source_path.relative_to(REPO_ROOT))
        except Exception:
            item["source_json"] = raw_source
        out.append(item)
    return out


def build_prefix_rows(
    *,
    support_json: Path,
    plateau_json: Path,
    include_regimes: set[str] | None,
    include_methods: set[str] | None,
    max_prefixes: int | None,
    progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    support = _read_json(support_json)
    plateau = _read_json(plateau_json)
    plateau_by_key = {
        (str(row.get("regime")), str(row.get("method"))): row
        for row in plateau.get("rows", [])
        if isinstance(row, Mapping)
    }

    prefix_rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    s_recovery: dict[str, Any] = {}

    for support_row in support.get("rows", []):
        if not isinstance(support_row, Mapping):
            continue
        regime = str(support_row.get("regime") or "")
        method = str(support_row.get("method") or "")
        if include_regimes is not None and regime not in include_regimes:
            continue
        if include_methods is not None and method not in include_methods:
            continue
        source_path = _resolve(str(support_row.get("source_json") or ""))
        source_rel = str(source_path.relative_to(REPO_ROOT)) if source_path.exists() else str(source_path)
        source_sha = str(support_row.get("source_sha256") or (_sha256(source_path) if source_path.exists() else ""))
        if not source_path.exists():
            blocked.append({"regime": regime, "method": method, "status": "source_json_missing", "source_json": source_rel})
            continue
        payload = _read_json(source_path)
        if not isinstance(payload, Mapping):
            blocked.append({"regime": regime, "method": method, "status": "source_json_not_object", "source_json": source_rel})
            continue
        history = _history_from_source(payload, method)
        s_values, s_status, s_components = _prefix_s_values(
            regime=regime,
            method=method,
            payload=payload,
            history=history,
            plateau_by_key=plateau_by_key,
        )
        s_recovery[f"{regime}:{method}"] = {"status": s_status, "components": s_components}
        try:
            visible_cells = _visible_cells(support_row, plateau_by_key.get((regime, method)))
            if method == "SNAKE":
                compiled_rows = _snake_compile_prefix_rows(
                    regime=regime,
                    method=method,
                    source_path=source_path,
                    source_sha256=source_sha,
                    payload=payload,
                    history=history,
                    visible_cells=visible_cells,
                    max_prefixes=max_prefixes,
                    progress=progress,
                )
            else:
                compiled_rows, _reference_meta = _compile_prefix_rows(
                    regime=regime,
                    method=method,
                    source_path=source_path,
                    source_sha256=source_sha,
                    visible_cells=visible_cells,
                    max_prefixes=max_prefixes,
                    progress=progress,
                )
            normalized = _normalize_prefix_rows(compiled_rows, source_rel=source_rel)
        except Exception as exc:
            blocked.append(
                {
                    "regime": regime,
                    "method": method,
                    "status": "prefix_compile_blocked",
                    "source_json": source_rel,
                    "error": str(exc),
                }
            )
            normalized = _trajectory_fallback_rows(
                support_row=support_row,
                source_path=source_path,
                source_sha256=source_sha,
                compile_status="prefix_compile_blocked",
                compile_error=str(exc),
            )
        for row in normalized:
            k = _int(row.get("prefix_k"))
            if k is not None and k in s_values:
                row["S"] = s_values[k]
                row["S_status"] = s_status
            else:
                row.setdefault("S", None)
                row["S_status"] = "missing_prefix_s" if s_status == "ok" else s_status
            row["support_json"] = str(support_json.relative_to(REPO_ROOT))
            row["plateau_json"] = str(plateau_json.relative_to(REPO_ROOT))
            prefix_rows.append(row)
    meta = {
        "support_json": str(support_json.relative_to(REPO_ROOT)),
        "support_json_sha256": _sha256(support_json),
        "plateau_json": str(plateau_json.relative_to(REPO_ROOT)),
        "plateau_json_sha256": _sha256(plateau_json),
        "s_recovery": s_recovery,
    }
    return prefix_rows, blocked, meta


def _rows_by_key(prefix_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prefix_rows:
        regime = str(row.get("regime") or "")
        method = str(row.get("method") or "")
        if regime and method:
            out.setdefault((regime, method), []).append(dict(row))
    for rows in out.values():
        rows.sort(key=lambda row: int(row.get("prefix_k") or 0))
    return out


def _usable_resource(row: Mapping[str, Any], axis: str) -> float | None:
    if axis == "S" and str(row.get("S_status") or "") not in S_OK_STATUSES:
        return None
    if axis != "S" and str(row.get("compile_status") or "") != "ok":
        return None
    return _num(row.get(axis))


def _plateau_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [row for row in rows if _usable_resource(row, "N2q") is not None and _num(row.get("abs_delta_e")) is not None]
    if not candidates:
        candidates = [row for row in rows if _num(row.get("abs_delta_e")) is not None]
    if not candidates:
        return None
    # The support/plateau audit puts the plateau point into visible_cells.
    k_pl = None
    visible = candidates[0].get("visible_cells_terminal")
    if isinstance(visible, Mapping):
        k_pl = _int(visible.get("k_pl"))
    if k_pl is not None:
        for row in candidates:
            if _int(row.get("prefix_k")) == k_pl:
                return row
    return candidates[-1]


def _first_reaching_error(rows: Sequence[Mapping[str, Any]], threshold: float) -> Mapping[str, Any] | None:
    for row in rows:
        err = _num(row.get("abs_delta_e"))
        if err is not None and err <= threshold:
            return row
    return None


def _best_under_budget(rows: Sequence[Mapping[str, Any]], axis: str, budget: float) -> Mapping[str, Any] | None:
    feasible: list[Mapping[str, Any]] = []
    for row in rows:
        value = _usable_resource(row, axis)
        err = _num(row.get("abs_delta_e"))
        if value is not None and err is not None and value <= budget:
            feasible.append(row)
    if not feasible:
        return None
    return min(feasible, key=lambda row: float(row.get("abs_delta_e")))


def _row_summary(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    return {
        "k": _int(row.get("prefix_k")),
        "abs_delta_e": _num(row.get("abs_delta_e")),
        "N2q": _num(row.get("N2q")),
        "D2q": _num(row.get("D2q")),
        "D_circ": _num(row.get("D_circ")),
        "S": _num(row.get("S")),
        "compile_status": row.get("compile_status"),
        "S_status": row.get("S_status"),
    }


def build_matched_diagnostics(prefix_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = _rows_by_key(prefix_rows)
    regimes = sorted({regime for regime, method in by_key if method == "SNAKE"})
    diagnostics: list[dict[str, Any]] = []
    for regime in regimes:
        snake_rows = by_key.get((regime, "SNAKE"), [])
        snake_plateau = _plateau_row(snake_rows)
        if snake_plateau is None:
            continue
        for comparator in COMPARATORS:
            comp_rows = by_key.get((regime, comparator), [])
            comp_plateau = _plateau_row(comp_rows)
            if comp_plateau is None:
                diagnostics.append(
                    {
                        "schema": "paper_i_hh_prefix_matched_diagnostic_row_v1",
                        "regime": regime,
                        "comparison": f"SNAKE_vs_{comparator}",
                        "diagnostic": "comparison_blocked",
                        "status": "missing_comparator_prefix_rows",
                    }
                )
                continue

            snake_pl_err = _num(snake_plateau.get("abs_delta_e"))
            comp_pl_err = _num(comp_plateau.get("abs_delta_e"))
            if snake_pl_err is None or comp_pl_err is None:
                status = "blocked_missing_plateau_error"
                threshold = None
                snake_equal = comp_equal = None
            else:
                threshold = max(snake_pl_err, comp_pl_err)
                snake_equal = _first_reaching_error(snake_rows, threshold)
                comp_equal = _first_reaching_error(comp_rows, threshold)
                status = "ok" if snake_equal is not None and comp_equal is not None else "blocked_unreached_equal_accuracy_threshold"
            equal_row: dict[str, Any] = {
                "schema": "paper_i_hh_prefix_matched_diagnostic_row_v1",
                "regime": regime,
                "comparison": f"SNAKE_vs_{comparator}",
                "diagnostic": "equal_accuracy_first_hit",
                "status": status,
                "shared_error_threshold": threshold,
                "snake": _row_summary(snake_equal),
                "comparator": _row_summary(comp_equal),
            }
            for axis in EQUAL_ACCURACY_COSTS:
                _add_lower_better_fields(
                    equal_row,
                    prefix=f"{axis}_at_equal_accuracy",
                    snake_value=_num((snake_equal or {}).get(axis)),
                    comparator_value=_num((comp_equal or {}).get(axis)),
                    comparator_label=comparator,
                )
            diagnostics.append(equal_row)

            for axis in RESOURCE_AXES:
                snake_budget = _usable_resource(snake_plateau, axis)
                comp_budget = _usable_resource(comp_plateau, axis)
                if snake_budget is None or comp_budget is None:
                    diagnostics.append(
                        {
                            "schema": "paper_i_hh_prefix_matched_diagnostic_row_v1",
                            "regime": regime,
                            "comparison": f"SNAKE_vs_{comparator}",
                            "diagnostic": f"equal_{axis}_budget_best_error",
                            "status": f"blocked_missing_plateau_{axis}",
                            "snake_plateau": _row_summary(snake_plateau),
                            "comparator_plateau": _row_summary(comp_plateau),
                        }
                    )
                    continue
                budget = min(float(snake_budget), float(comp_budget))
                snake_best = _best_under_budget(snake_rows, axis, budget)
                comp_best = _best_under_budget(comp_rows, axis, budget)
                status = "ok" if snake_best is not None and comp_best is not None else f"blocked_no_prefix_under_common_{axis}_budget"
                diagnostics.append(
                    {
                        "schema": "paper_i_hh_prefix_matched_diagnostic_row_v1",
                        "regime": regime,
                        "comparison": f"SNAKE_vs_{comparator}",
                        "diagnostic": f"equal_{axis}_budget_best_error",
                        "status": status,
                        "shared_budget_axis": axis,
                        "shared_budget": budget,
                        "snake": _row_summary(snake_best),
                        "comparator": _row_summary(comp_best),
                    }
                )
                _add_lower_better_fields(
                    diagnostics[-1],
                    prefix="abs_delta_e_at_equal_budget",
                    snake_value=_num((snake_best or {}).get("abs_delta_e")),
                    comparator_value=_num((comp_best or {}).get("abs_delta_e")),
                    comparator_label=comparator,
                )
    return diagnostics


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(value, sort_keys=True)
                else:
                    flat[key] = value
            writer.writerow(flat)


def build(
    *,
    support_json: Path,
    plateau_json: Path,
    output_json: Path,
    prefix_csv: Path | None,
    diagnostics_csv: Path | None,
    include_regimes: set[str] | None,
    include_methods: set[str] | None,
    max_prefixes: int | None,
    progress: bool,
) -> dict[str, Any]:
    prefix_rows, blocked_rows, meta = build_prefix_rows(
        support_json=support_json,
        plateau_json=plateau_json,
        include_regimes=include_regimes,
        include_methods=include_methods,
        max_prefixes=max_prefixes,
        progress=progress,
    )
    diagnostics = build_matched_diagnostics(prefix_rows)
    payload = {
        "schema": "paper_i_hh_prefix_matched_diagnostics_v1",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "contract": {
            "scope": "Paper-I Hubbard--Holstein prefix-level support diagnostic",
            "equal_accuracy_rule": "threshold=max(SNAKE plateau error, comparator plateau error); use first prefix reaching threshold",
            "equal_budget_rule": "budget=min(SNAKE plateau resource, comparator plateau resource); use best error among prefixes with resource <= budget",
            "comparison_rule": "lower-is-better fields are ratio-first; percent-lower is always reported in the winner direction and negative percent improvements are not emitted",
            "resource_axes": list(RESOURCE_AXES),
            "equal_accuracy_cost_axes": list(EQUAL_ACCURACY_COSTS),
            "s_status_policy": "equal-S diagnostics require prefix rows with S_status=ok for both methods",
            "manuscript_policy": "support artifact only; does not edit Paper_I.tex",
        },
        "source_meta": meta,
        "status_counts": {
            "prefix_compile": _status_counts(prefix_rows, key="compile_status"),
            "prefix_s": _status_counts(prefix_rows, key="S_status"),
            "diagnostics": _status_counts(diagnostics, key="status"),
            "blocked": _status_counts(blocked_rows, key="status"),
        },
        "blocked_rows": blocked_rows,
        "prefix_rows": prefix_rows,
        "diagnostics": diagnostics,
    }
    _write_json(output_json, payload)
    if prefix_csv is not None:
        _write_csv(prefix_csv, prefix_rows)
    if diagnostics_csv is not None:
        _write_csv(diagnostics_csv, diagnostics)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-json", type=Path, default=DEFAULT_SUPPORT_JSON)
    parser.add_argument("--plateau-json", type=Path, default=DEFAULT_PLATEAU_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--prefix-csv", type=Path, default=DEFAULT_PREFIX_CSV)
    parser.add_argument("--diagnostics-csv", type=Path, default=DEFAULT_DIAGNOSTIC_CSV)
    parser.add_argument("--regime", action="append", help="Restrict to a regime. Repeatable.")
    parser.add_argument("--method", action="append", help="Restrict to a method label. Repeatable.")
    parser.add_argument("--max-prefixes", type=int, help="Optional smoke-test cap per source.")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-prefix-csv", action="store_true")
    parser.add_argument("--no-diagnostics-csv", action="store_true")
    args = parser.parse_args(argv)

    include_methods = set(args.method) if args.method else set(METHOD_ORDER)
    # SNAKE is required for matched diagnostics even if the caller filters to one comparator.
    if args.method and any(method in include_methods for method in COMPARATORS):
        include_methods.add("SNAKE")

    payload = build(
        support_json=_resolve(args.support_json),
        plateau_json=_resolve(args.plateau_json),
        output_json=_resolve(args.output_json),
        prefix_csv=None if args.no_prefix_csv else _resolve(args.prefix_csv),
        diagnostics_csv=None if args.no_diagnostics_csv else _resolve(args.diagnostics_csv),
        include_regimes=set(args.regime) if args.regime else None,
        include_methods=include_methods,
        max_prefixes=args.max_prefixes,
        progress=bool(args.progress),
    )
    print(
        json.dumps(
            {
                "output_json": str(_resolve(args.output_json).relative_to(REPO_ROOT)),
                "prefix_rows": len(payload["prefix_rows"]),
                "diagnostics": len(payload["diagnostics"]),
                "status_counts": payload["status_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
