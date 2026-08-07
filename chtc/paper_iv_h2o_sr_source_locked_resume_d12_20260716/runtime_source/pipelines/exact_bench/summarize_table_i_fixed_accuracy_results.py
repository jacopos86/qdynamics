#!/usr/bin/env python3
"""Calibrated fixed-accuracy Table-I summary.

This script reports resource costs at declared accuracy thresholds.  It is
stricter than ``summarize_table_i_static_results.py``: adaptive final-row
fallbacks are upper bounds, raw shot totals are never promoted to table cost,
and promoted costs must pass the Table-I threshold classifier in
``generic_static_metric_enrichment``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.exact_bench import generic_static_metric_enrichment as enrich
from pipelines.exact_bench.summarize_table_i_static_results import (
    ENRICHMENT_SCHEMA_VERSION,
    _load_records,
    _num,
    _read_enrichment,
    _read_payload,
    _result,
)
from pipelines.exact_bench.table_i_static_benchmark import (
    TABLE_I_CLASS_BY_FAMILY,
    TABLE_I_METHOD_LABELS,
    table_i_method_label,
)

DEFAULT_RECORDS = Path("chtc/phase3_optuna/input/table_i_nph2_ref3_v1/generic_static_table_records.tsv")
DEFAULT_ROOT = Path("raw_outputs/generic_static_table_nph2_ref3_v1")
DEFAULT_OUTPUT_DIR = Path("raw_outputs/table_i_fixed_accuracy_calibrated")
DEFAULT_THRESHOLDS = (2e-4,)
CLASS_ORDER = ("fermionic", "bosonic", "fermion-boson", "all averaged")
ALGORITHM_ORDER = tuple(TABLE_I_METHOD_LABELS.keys())
FIXED_TERMINAL_METHOD_IDS = {"static_hea_qiskit_vqe", "static_family_informed_vqe"}


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return fmean(clean)


def _threshold_key(threshold: float) -> str:
    text = f"{float(threshold):.0e}".replace("-0", "-").replace("-", "_")
    return f"first_hit_{text}"


def _threshold_row(algorithm_id: str, result_row: Mapping[str, Any], threshold: float) -> tuple[Mapping[str, Any], str]:
    if str(algorithm_id) in FIXED_TERMINAL_METHOD_IDS:
        return result_row, "terminal_row_fixed_method"
    for key in _threshold_lookup_keys(float(threshold)):
        direct = result_row.get(key)
        if isinstance(direct, Mapping):
            return direct, str(key)
    hits = result_row.get("benchmark_first_hits")
    if isinstance(hits, Mapping):
        for key in _threshold_lookup_keys(float(threshold)):
            candidate = hits.get(key)
            if isinstance(candidate, Mapping):
                return candidate, f"benchmark_first_hits[{key}]"
    return result_row, "terminal_row"


def _s_alg_upper_bound_from_row(*, algorithm_id: str, row: Mapping[str, Any]) -> float | None:
    raw_proxy = {
        "shots_total": _num(row.get("shots_total")),
        "shot_cost_proxy": _num(row.get("shot_cost_proxy")),
        "measurement_shots_proxy": _num(row.get("measurement_shots_proxy")),
        "shot_proxy": _num(row.get("shot_proxy")),
    }
    try:
        replay_ledger, _replay_status = enrich._table_i_event_ledger_from_comparator_row(
            algorithm_id=str(algorithm_id),
            row=row,
        )
        alg_row = dict(row)
        if replay_ledger is not None and "table_i_measurement_event_ledger" not in alg_row:
            alg_row["table_i_measurement_event_ledger"] = replay_ledger
        _metric, updates, statuses = enrich.algorithmic_measurement_work_from_row(row=alg_row, raw_proxy=raw_proxy)
    except Exception:
        return None
    if str(statuses.get("S_alg") or "") != "ok":
        return None
    value = _num(updates.get("S_alg"))
    return None if value is None else float(value)


def _resource_value(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _num(row.get(key))
        if value is not None:
            return value
    return None


def _promoted_resource_value(
    *,
    cost_included: bool,
    cost: Mapping[str, Any],
    source_row: Mapping[str, Any],
    cost_keys: Sequence[str],
    source_keys: Sequence[str],
) -> float | None:
    if not cost_included:
        return None
    value = _resource_value(cost, *cost_keys)
    return value


def _method_cost_semantics(algorithm_id: str) -> str:
    alg = str(algorithm_id)
    if alg in FIXED_TERMINAL_METHOD_IDS:
        return "terminal_only_fixed_ansatz"
    if alg == "static_family_native_adapt_phase3":
        return "snake_first_hit_sidecar_required"
    return "adaptive_qiskit_compiled_first_hit_or_final_ansatz"


def _threshold_lookup_keys(threshold: float) -> tuple[str, ...]:
    sci = f"{float(threshold):.0e}"
    compact = sci.replace("-0", "-").replace("-", "_")
    legacy = compact.replace("e_", "e")
    keys = (
        f"first_hit_{compact}",
        f"first_hit_{legacy}",
        f"first_hit_abs_delta_e_le_{compact}",
        f"first_hit_abs_delta_e_le_{legacy}",
        sci,
        sci.replace("-0", "-"),
        str(float(threshold)),
        str(threshold),
    )
    return tuple(dict.fromkeys(keys))


def _threshold_metric_entry(
    enrichment: Mapping[str, Any],
    *,
    threshold: float,
) -> tuple[str | None, Mapping[str, Any] | None]:
    containers: list[Any] = []
    for key in ("threshold_metrics", "threshold_costs", "threshold_statevector_variance_metrics"):
        containers.append(enrichment.get(key))
    metrics = enrichment.get("metrics")
    if isinstance(metrics, Mapping):
        for key in ("threshold_metrics", "threshold_costs", "threshold_statevector_variance_metrics"):
            containers.append(metrics.get(key))
    lookup = _threshold_lookup_keys(float(threshold))
    for container in containers:
        if isinstance(container, Mapping):
            for key in lookup:
                entry = container.get(key)
                if isinstance(entry, Mapping):
                    return key, entry
        elif isinstance(container, list):
            for entry in container:
                if not isinstance(entry, Mapping):
                    continue
                entry_threshold = _num(entry.get("threshold"))
                if entry_threshold is not None and math.isclose(float(entry_threshold), float(threshold), rel_tol=0.0, abs_tol=0.0):
                    return str(entry.get("threshold_key") or entry.get("source") or threshold), entry
    return None, None


def _threshold_entry_has_local_state_scope(entry: Mapping[str, Any]) -> bool:
    nested = entry.get("statevector_variance_metric")
    nested_map = nested if isinstance(nested, Mapping) else {}
    scope = str(entry.get("state_scope") or nested_map.get("state_scope") or "")
    provenance = str(
        entry.get("provenance")
        or nested_map.get("provenance")
        or entry.get("source_kind")
        or nested_map.get("source_kind")
        or ""
    )
    if scope in {"threshold_first_hit_state", "event_local", "event_local_threshold_state"}:
        return True
    if scope == "terminal_final_state":
        return False
    lowered = provenance.lower()
    return "threshold" in lowered or "first_hit" in lowered or "event_local" in lowered


def _s_var_from_threshold_entry(entry: Mapping[str, Any]) -> tuple[float | None, str, Any]:
    if not _threshold_entry_has_local_state_scope(entry):
        return None, "missing_threshold_state", entry.get("S_var_components") or entry.get("components")
    nested = entry.get("statevector_variance_metric")
    if isinstance(nested, Mapping):
        nested_status = str(nested.get("status") or "")
        nested_value = _num(nested.get("S_var"))
        if nested_value is None:
            nested_value = _num(nested.get("S_phys_var"))
        if nested_status == "ok" and nested_value is not None:
            return float(nested_value), "ok", nested.get("components")
        metric, updates, status = enrich._statevector_variance_metric_from_components(nested)
        value = _num(updates.get("S_var"))
        if status == "ok" and value is not None:
            return float(value), "ok", metric.get("components")
        if nested_status and nested_status != "ok":
            return None, nested_status, nested.get("components")
        return None, status, nested.get("components")

    status = str(entry.get("S_var_status") or entry.get("S_phys_var_status") or entry.get("status") or "")
    value = _num(entry.get("S_var"))
    if value is None:
        value = _num(entry.get("S_phys_var"))
    if value is None:
        if status and status != "ok":
            return None, status, entry.get("S_var_components") or entry.get("components")
        return None, "missing_statevector_variance_event_components", entry.get("S_var_components") or entry.get("components")
    if status and status != "ok":
        return None, status, entry.get("S_var_components") or entry.get("components")
    return float(value), "ok", entry.get("S_var_components") or entry.get("components")


def _terminal_s_var_from_enrichment(enrichment: Mapping[str, Any], *, record_id: str) -> tuple[float | None, str]:
    if enrichment.get("schema") != ENRICHMENT_SCHEMA_VERSION:
        return None, "invalid_schema"
    if str(enrichment.get("record_id") or "") != str(record_id):
        return None, "record_id_mismatch"
    if str(enrichment.get("status") or "") in {"failed", "payload_missing"}:
        return None, str(enrichment.get("status") or "failed")
    statuses = enrichment.get("metric_statuses")
    updates = enrichment.get("row_updates")
    if not isinstance(statuses, Mapping) or not isinstance(updates, Mapping):
        return None, "missing_s_phys_var_status"
    status = str(statuses.get("S_phys_var") or statuses.get("S_var") or "missing_s_phys_var_status")
    if status != "ok":
        return None, status
    value = _num(updates.get("S_var"))
    if value is None:
        return None, "missing_S_var_value"
    return float(value), "ok"


def _threshold_s_var_from_enrichment(
    *,
    enrichment: Mapping[str, Any] | None,
    record_id: str,
    threshold: float,
    threshold_status: str,
    threshold_cost: Mapping[str, Any],
) -> dict[str, Any]:
    cost_status = str(threshold_cost.get("S_var_status") or "missing_statevector_variance_event_components")
    if not isinstance(enrichment, Mapping):
        return {
            "S_var": None,
            "status": cost_status if cost_status != "ok" else "threshold_s_var_requires_enrichment_sidecar",
            "provenance": None,
            "components": None,
            "source_key": None,
            "terminal_S_var_upper_bound": None,
            "terminal_S_var_upper_bound_status": "no_enrichment",
            "terminal_S_var_upper_bound_provenance": None,
        }
    if enrichment.get("schema") != ENRICHMENT_SCHEMA_VERSION:
        return {
            "S_var": None,
            "status": "invalid_schema",
            "provenance": None,
            "components": None,
            "source_key": None,
            "terminal_S_var_upper_bound": None,
            "terminal_S_var_upper_bound_status": "invalid_schema",
            "terminal_S_var_upper_bound_provenance": None,
        }
    if str(enrichment.get("record_id") or "") != str(record_id):
        return {
            "S_var": None,
            "status": "record_id_mismatch",
            "provenance": None,
            "components": None,
            "source_key": None,
            "terminal_S_var_upper_bound": None,
            "terminal_S_var_upper_bound_status": "record_id_mismatch",
            "terminal_S_var_upper_bound_provenance": None,
        }
    if str(enrichment.get("status") or "") in {"failed", "payload_missing"}:
        status = str(enrichment.get("status") or "failed")
        return {
            "S_var": None,
            "status": status,
            "provenance": None,
            "components": None,
            "source_key": None,
            "terminal_S_var_upper_bound": None,
            "terminal_S_var_upper_bound_status": status,
            "terminal_S_var_upper_bound_provenance": None,
        }

    entry_key, entry = _threshold_metric_entry(enrichment, threshold=float(threshold))
    if entry is not None:
        value, status, components = _s_var_from_threshold_entry(entry)
        if status == "ok" and value is not None:
            return {
                "S_var": float(value),
                "status": "ok",
                "provenance": str(entry.get("provenance") or "threshold_local_enrichment"),
                "components": components,
                "source_key": entry_key,
                "terminal_S_var_upper_bound": None,
                "terminal_S_var_upper_bound_status": None,
                "terminal_S_var_upper_bound_provenance": None,
            }
        terminal_value, terminal_status = _terminal_s_var_from_enrichment(enrichment, record_id=record_id)
        return {
            "S_var": None,
            "status": status,
            "provenance": None,
            "components": components,
            "source_key": entry_key,
            "terminal_S_var_upper_bound": terminal_value,
            "terminal_S_var_upper_bound_status": terminal_status,
            "terminal_S_var_upper_bound_provenance": "terminal_enrichment_S_var" if terminal_value is not None else None,
        }

    terminal_value, terminal_status = _terminal_s_var_from_enrichment(enrichment, record_id=record_id)
    if threshold_status == "not_reached":
        status = "not_reached"
    elif terminal_value is not None:
        status = "missing_threshold_state"
    else:
        status = cost_status if cost_status != "ok" else "missing_threshold_state"
    return {
        "S_var": None,
        "status": status,
        "provenance": None,
        "components": None,
        "source_key": None,
        "terminal_S_var_upper_bound": terminal_value,
        "terminal_S_var_upper_bound_status": terminal_status,
        "terminal_S_var_upper_bound_provenance": "terminal_enrichment_S_var" if terminal_value is not None else None,
    }


def _valid_enrichment_row_updates(
    enrichment: Mapping[str, Any] | None,
    *,
    record_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(enrichment, Mapping):
        return {}, {}
    if enrichment.get("schema") != ENRICHMENT_SCHEMA_VERSION:
        return {}, {}
    if str(enrichment.get("record_id") or "") != str(record_id):
        return {}, {}
    if str(enrichment.get("status") or "") in {"failed", "payload_missing"}:
        return {}, {}
    updates = enrichment.get("row_updates")
    statuses = enrichment.get("metric_statuses")
    if not isinstance(updates, Mapping) or not isinstance(statuses, Mapping):
        return {}, {}
    return updates, statuses


def _source_row_with_terminal_enrichment(
    source_row: Mapping[str, Any],
    *,
    algorithm_id: str,
    threshold: float,
    record_id: str,
    enrichment: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Merge post-hoc enrichment into terminal miss rows before cost gating.

    The enrichment sidecar is reporting-only: it reconstructs the final state and
    Qiskit-compiled final ansatz after the algorithm has already terminated.  It
    may therefore supply the displayed-ansatz cost for a miss, but it must not
    convert a target hit into a terminal-cost row or replace native first-hit
    costs.
    """

    if str(algorithm_id) == "static_family_native_adapt_phase3":
        return source_row
    delta = _num(source_row.get("delta_E_abs", source_row.get("abs_delta_e")))
    if delta is None or float(delta) <= float(threshold):
        return source_row
    updates, statuses = _valid_enrichment_row_updates(enrichment, record_id=record_id)
    if not updates:
        return source_row
    required = (
        "compiled_count_2q_total",
        "compiled_depth_2q_total",
        "compiled_depth_total",
    )
    if any(str(statuses.get(key) or "") != "ok" or _num(updates.get(key)) is None for key in required):
        return source_row
    out = dict(source_row)
    for key in required:
        out[key] = float(_num(updates.get(key)))
    source_kind = str(
        updates.get("compiled_resource_source_kind")
        or updates.get("first_hit_cost_source_kind")
        or "qiskit_compiled_final_ansatz_circuit"
    )
    out["compiled_circuit_stats_status"] = "ok"
    out["compiled_resource_source_kind"] = source_kind
    out["first_hit_cost_source_kind"] = source_kind
    out["compiled_resource_qiskit_validated"] = True
    out["qiskit_first_hit_cost_validated"] = False
    out.setdefault("compiled_resource_recovery_path", enrichment.get("source_payload_path"))
    out["compiled_resource_recovery_source"] = "generic_static_metric_enrichment_final_ansatz"
    return out


def _infidelity_from_enrichment_or_row(
    *,
    source_row: Mapping[str, Any],
    record_id: str,
    enrichment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    updates, statuses = _valid_enrichment_row_updates(enrichment, record_id=record_id)
    candidates = [
        ("infidelity_reference", source_row.get("infidelity_reference")),
        ("infidelity_4", source_row.get("infidelity_4")),
        ("infidelity_same", source_row.get("infidelity_same")),
        ("infidelity_exact", source_row.get("infidelity_exact")),
        ("infidelity", source_row.get("infidelity")),
    ]
    if updates:
        candidates = [
            ("infidelity_reference", updates.get("infidelity_reference")),
            ("infidelity_4", updates.get("infidelity_4")),
            ("infidelity_same", updates.get("infidelity_same")),
            ("infidelity_exact", updates.get("infidelity_exact")),
            ("infidelity", updates.get("infidelity")),
            *candidates,
        ]
    for key, raw in candidates:
        value = _num(raw)
        if value is None:
            continue
        status_key = key
        if key == "infidelity_reference":
            status_key = "infidelity_4"
        status = str(statuses.get(status_key) or statuses.get(key) or "ok")
        if updates and key in updates and status != "ok":
            continue
        infidelity = float(max(0.0, min(1.0, value)))
        out = {
            "infidelity": infidelity,
            "one_minus_fidelity": infidelity,
            "fidelity": float(max(0.0, min(1.0, 1.0 - infidelity))),
            "infidelity_source_key": key,
            "infidelity_status": status,
        }
        if key in {"infidelity_reference", "infidelity_4"}:
            out["infidelity_reference"] = infidelity
        elif key == "infidelity_same":
            out["infidelity_same"] = infidelity
        elif key == "infidelity_exact":
            out["infidelity_exact"] = infidelity
        return out
    return {
        "infidelity": None,
        "one_minus_fidelity": None,
        "fidelity": None,
        "infidelity_source_key": None,
        "infidelity_status": None,
    }


def _record_threshold_rows(
    *,
    record: Mapping[str, str],
    payload_path: Path,
    payload: Mapping[str, Any],
    thresholds: Sequence[float],
    enrichment_path: Path | None = None,
    enrichment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    result_row = _result(payload)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        source_row_raw, source_path = _threshold_row(str(record["algorithm_id"]), result_row, float(threshold))
        source_row = _source_row_with_terminal_enrichment(
            source_row_raw,
            algorithm_id=str(record["algorithm_id"]),
            threshold=float(threshold),
            record_id=str(record["record_id"]),
            enrichment=enrichment,
        )
        fidelity_fields = _infidelity_from_enrichment_or_row(
            source_row=source_row,
            record_id=str(record["record_id"]),
            enrichment=enrichment,
        )
        cost = enrich.table_i_threshold_cost_from_row(
            algorithm_id=str(record["algorithm_id"]),
            row=source_row,
            threshold=float(threshold),
            record=record,
            result_path=payload_path,
            enrichment_path=enrichment_path,
        )
        threshold_status = str(cost.get("threshold_status") or "unknown")
        cost_included = (
            threshold_status in {"ok_native_first_hit", "ok_terminal_only_method", "not_reached_final_ansatz"}
            and cost.get("resource_display_allowed") is True
            and str(cost.get("compiled_resource_validation_status") or "") == "ok"
        )
        count_2q = _promoted_resource_value(
            cost_included=cost_included,
            cost=cost,
            source_row=source_row,
            cost_keys=("count_2q", "compiled_count_2q_total"),
            source_keys=("compiled_count_2q_total", "count_2q"),
        )
        depth_2q = _promoted_resource_value(
            cost_included=cost_included,
            cost=cost,
            source_row=source_row,
            cost_keys=("depth_2q", "compiled_depth_2q_total"),
            source_keys=("compiled_depth_2q_total", "depth_2q"),
        )
        circuit_depth = _promoted_resource_value(
            cost_included=cost_included,
            cost=cost,
            source_row=source_row,
            cost_keys=("circuit_depth", "compiled_depth_total"),
            source_keys=("compiled_depth_total", "circuit_depth"),
        )
        terminal_s_alg = None if cost_included else _s_alg_upper_bound_from_row(
            algorithm_id=str(record["algorithm_id"]),
            row=source_row,
        )
        s_var_info = _threshold_s_var_from_enrichment(
            enrichment=enrichment,
            record_id=str(record["record_id"]),
            threshold=float(threshold),
            threshold_status=threshold_status,
            threshold_cost=cost,
        )
        threshold_s_var_included = (
            cost_included
            and str(s_var_info.get("status") or "") == "ok"
            and s_var_info.get("source_key") is not None
        )
        rows.append(
            {
                "record_id": record["record_id"],
                "family": record["family"],
                "case_id": record["case_id"],
                "algorithm_id": record["algorithm_id"],
                "method": table_i_method_label(record["algorithm_id"]),
                "class": TABLE_I_CLASS_BY_FAMILY.get(record["family"], "unmapped"),
                "threshold": float(threshold),
                "threshold_source": source_path,
                "threshold_status": threshold_status,
                "cost_included": bool(cost_included),
                "abs_delta_e": cost.get("abs_delta_e"),
                **fidelity_fields,
                "S_alg": cost.get("S_alg") if cost_included else None,
                "S_norm": None,
                "terminal_S_alg_upper_bound": terminal_s_alg,
                "S_var": s_var_info.get("S_var") if threshold_s_var_included else None,
                "S_phys_var": s_var_info.get("S_var") if threshold_s_var_included else None,
                "S_var_status": s_var_info.get("status"),
                "S_var_provenance": s_var_info.get("provenance") if threshold_s_var_included else None,
                "S_var_source_key": s_var_info.get("source_key"),
                "S_var_components": s_var_info.get("components") if threshold_s_var_included else None,
                "terminal_S_var_upper_bound": s_var_info.get("terminal_S_var_upper_bound"),
                "terminal_S_var_upper_bound_status": s_var_info.get("terminal_S_var_upper_bound_status"),
                "terminal_S_var_upper_bound_provenance": s_var_info.get("terminal_S_var_upper_bound_provenance"),
                "N_metric": cost.get("N_metric") if cost_included else None,
                "metric_fraction": cost.get("metric_fraction") if cost_included else None,
                "count_2q": count_2q,
                "depth_2q": depth_2q,
                "circuit_depth": circuit_depth,
                "payload_path": str(payload_path),
                "cost_source": cost.get("cost_source"),
                "source": cost.get("source"),
                "first_hit_semantics": cost.get("first_hit_semantics"),
                "method_cost_semantics": cost.get("method_cost_semantics") or _method_cost_semantics(str(record["algorithm_id"])),
                "resource_display_allowed": bool(cost.get("resource_display_allowed") is True and cost_included),
                "compiled_resource_validation_status": cost.get("compiled_resource_validation_status"),
                "compiled_resource_validation_reason": cost.get("compiled_resource_validation_reason"),
                "first_hit_cost_source_kind": cost.get("first_hit_cost_source_kind"),
                "source_resource_fields_present": cost.get("source_resource_fields_present"),
                "sidecar_validation_status": cost.get("sidecar_validation_status"),
                "sidecar_validation_reason": cost.get("sidecar_validation_reason"),
                "sidecar_hash_verified": cost.get("sidecar_hash_verified"),
                "sidecar_source_kind": cost.get("sidecar_source_kind"),
                "snake_first_crossing_cost_sidecar_key": cost.get("snake_first_crossing_cost_sidecar_key"),
                "snake_first_crossing_history_position_tau": cost.get("snake_first_crossing_history_position_tau"),
                "source_result_sha256": cost.get("source_result_sha256"),
                "source_result_path": cost.get("source_result_path"),
                "S_alg_missing_reason": cost.get("S_alg_missing_reason"),
                "reconstructability_status": cost.get("reconstructability_status"),
                "components": cost.get("components"),
            }
        )
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        threshold = float(row["threshold"])
        method = str(row["method"])
        klass = str(row["class"])
        alg = str(row["algorithm_id"])
        grouped[(threshold, klass, method, alg)].append(row)
        grouped[(threshold, "all averaged", method, alg)].append(row)

    algorithm_order = list(ALGORITHM_ORDER)
    for row in rows:
        alg = str(row["algorithm_id"])
        if alg not in algorithm_order:
            algorithm_order.append(alg)

    thresholds = sorted({float(row["threshold"]) for row in rows}, reverse=True)
    out: list[dict[str, Any]] = []
    for threshold in thresholds:
        for klass in CLASS_ORDER:
            for alg in algorithm_order:
                method = table_i_method_label(alg)
                items = grouped.get((threshold, klass, method, alg), [])
                if not items:
                    continue
                included = [item for item in items if bool(item.get("cost_included"))]
                upper = [item for item in items if str(item.get("threshold_status")) == "terminal_upper_bound_missing_native_first_hit"]
                out.append(
                    {
                        "threshold": float(threshold),
                        "class": klass,
                        "method": method,
                        "algorithm_id": alg,
                        "n": len(items),
                        "hit_count": len(included),
                        "hit_rate": float(len(included) / len(items)) if items else None,
                        "terminal_upper_bound_count": len(upper),
                        "status_counts": dict(Counter(str(item.get("threshold_status") or "unknown") for item in items)),
                        "delta_e_mean_included": _mean([item.get("abs_delta_e") for item in included]),
                        "delta_e_mean_all_terminal_or_hit": _mean([item.get("abs_delta_e") for item in items]),
                        "count_2q_mean": _mean([item.get("count_2q") for item in included]),
                        "depth_2q_mean": _mean([item.get("depth_2q") for item in included]),
                        "circuit_depth_mean": _mean([item.get("circuit_depth") for item in included]),
                        "S_alg_mean": _mean([item.get("S_alg") for item in included]),
                        "S_norm_mean": None,
                        "S_var_mean": _mean([item.get("S_var") for item in included]),
                        "S_var_available_n": sum(1 for item in included if item.get("S_var") is not None),
                        "S_var_status_counts": dict(Counter(str(item.get("S_var_status") or "none") for item in items)),
                        "N_metric_mean_support_only": _mean([item.get("N_metric") for item in included]),
                        "metric_fraction_mean_support_only": _mean([item.get("metric_fraction") for item in included]),
                        "terminal_upper_bound_S_alg_mean": _mean([item.get("terminal_S_alg_upper_bound") for item in upper]),
                        "terminal_S_var_upper_bound_mean": _mean([item.get("terminal_S_var_upper_bound") for item in items]),
                        "terminal_S_var_upper_bound_count": sum(1 for item in items if item.get("terminal_S_var_upper_bound") is not None),
                    }
                )
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "class",
        "method",
        "algorithm_id",
        "n",
        "hit_count",
        "hit_rate",
        "terminal_upper_bound_count",
        "status_counts",
        "delta_e_mean_included",
        "delta_e_mean_all_terminal_or_hit",
        "count_2q_mean",
        "depth_2q_mean",
        "circuit_depth_mean",
        "S_alg_mean",
        "S_norm_mean",
        "S_var_mean",
        "S_var_available_n",
        "S_var_status_counts",
        "N_metric_mean_support_only",
        "metric_fraction_mean_support_only",
        "terminal_upper_bound_S_alg_mean",
        "terminal_S_var_upper_bound_mean",
        "terminal_S_var_upper_bound_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _parse_thresholds(raw: str | None) -> tuple[float, ...]:
    if raw is None or not str(raw).strip():
        return DEFAULT_THRESHOLDS
    values = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    return tuple(sorted(set(values), reverse=True))


def _clean_thresholds_selected(thresholds: Sequence[float]) -> bool:
    return len(thresholds) == 1 and math.isclose(float(thresholds[0]), 2e-4, rel_tol=0.0, abs_tol=1e-12)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize calibrated fixed-accuracy Table-I benchmark outputs.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--enrichment-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--thresholds", default=",".join(f"{x:.0e}" for x in DEFAULT_THRESHOLDS))
    parser.add_argument("--allow-incomplete", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = _parse_thresholds(args.thresholds)
    records = _load_records(args.records)
    missing: list[dict[str, str]] = []
    threshold_rows: list[dict[str, Any]] = []
    enrichment_available = 0
    enrichment_missing = 0
    enrichment_failed = 0
    for record in records:
        payload_path, payload = _read_payload(args.root, record["record_id"])
        if payload is None:
            missing.append({**record, "expected_payload": str(payload_path)})
            continue
        enrichment_path, enrichment = _read_enrichment(args.enrichment_root, record["record_id"])
        if args.enrichment_root is not None:
            if enrichment is None:
                enrichment_missing += 1
            elif str(enrichment.get("status")) == "failed":
                enrichment_failed += 1
            else:
                enrichment_available += 1
        threshold_rows.extend(
            _record_threshold_rows(
                record=record,
                payload_path=payload_path,
                payload=payload,
                thresholds=thresholds,
                enrichment_path=enrichment_path,
                enrichment=enrichment,
            )
        )

    aggregate_rows = _aggregate(threshold_rows)
    clean_thresholds = _clean_thresholds_selected(thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "table_i_fixed_accuracy_calibrated_v1",
        "records_path": str(args.records),
        "output_root": str(args.root),
        "enrichment_root": str(args.enrichment_root) if args.enrichment_root is not None else None,
        "enrichment_available_count": enrichment_available,
        "enrichment_missing_count": enrichment_missing,
        "enrichment_failed_count": enrichment_failed,
        "thresholds": [float(x) for x in thresholds],
        "target_profile": "paper_i_phys_v1" if clean_thresholds else "support_diagnostic",
        "threshold_policy": "clean_paper_i_tau_phys" if clean_thresholds else "explicit_support_thresholds",
        "expected_count": len(records),
        "payload_count": len(records) - len(missing),
        "missing_count": len(missing),
        "missing": missing,
        "row_results": threshold_rows,
        "aggregate_rows": aggregate_rows,
    }
    json_path = args.output_dir / "table_i_fixed_accuracy_calibrated_summary.json"
    csv_path = args.output_dir / "table_i_fixed_accuracy_calibrated_rows.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, aggregate_rows)
    print(json.dumps({
        "summary_json": str(json_path),
        "rows_csv": str(csv_path),
        "expected_count": summary["expected_count"],
        "payload_count": summary["payload_count"],
        "missing_count": summary["missing_count"],
    }, indent=2, sort_keys=True))
    if missing and not bool(args.allow_incomplete):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
