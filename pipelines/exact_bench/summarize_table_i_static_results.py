#!/usr/bin/env python3
"""Aggregate Table-I static benchmark outputs into manuscript-facing rows.

This script is intentionally downstream of execution. It does not decide whether
an algorithm "succeeded" scientifically; if a job produced benchmark metrics, the
row is included. Quality-gate nonpasses are preserved as annotations so expected
stress-case behavior, such as harmonic/Kerr leakage, remains visible rather than
being treated as a missing benchmark.
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

from pipelines.exact_bench.table_i_static_benchmark import (
    TABLE_I_CLASS_BY_FAMILY,
    TABLE_I_METHOD_LABELS,
    TABLE_I_NONDEFAULT_METHOD_LABELS,
    table_i_method_label,
)

DEFAULT_RECORDS = Path("chtc/phase3_optuna/input/generic_static_table_records.tsv")
DEFAULT_ROOT = Path("raw_outputs/chtc_phase3_optuna/generic_static_table")
DEFAULT_OUTPUT_DIR = Path("raw_outputs/table_i_static_paper")

CLASS_ORDER = ("fermionic", "bosonic", "fermion-boson", "all averaged")
ALGORITHM_ORDER = tuple(TABLE_I_METHOD_LABELS.keys())
ENRICHMENT_FILENAME = "generic_static_metric_enrichment.json"
ENRICHMENT_SCHEMA_VERSION = "generic_static_metric_enrichment_v1"


def _load_records(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    required = {"record_id", "family", "case_id", "algorithm_id"}
    missing = required - set(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"records file {path} missing columns: {sorted(missing)}")
    return rows


def _read_payload(root: Path, record_id: str) -> tuple[Path, Mapping[str, Any] | None]:
    result_dir = root / record_id / "result"
    for name in (
        "generic_static_single.json",
        "result.json",
        "manifest.json",
        "skip.json",
        "hh_static_benchmark_result.json",
        "hh_static_benchmark_rows.json",
    ):
        path = result_dir / name
        if path.exists():
            return path, json.loads(path.read_text(encoding="utf-8"))
    return result_dir / "generic_static_single.json", None


def _read_enrichment(enrichment_root: Path | None, record_id: str) -> tuple[Path | None, Mapping[str, Any] | None]:
    if enrichment_root is None:
        return None, None
    path = enrichment_root / record_id / "result" / ENRICHMENT_FILENAME
    if not path.exists():
        return path, None
    return path, json.loads(path.read_text(encoding="utf-8"))


def _enrichment_update(enrichment: Mapping[str, Any] | None, key: str, *, record_id: str) -> float | None:
    if not isinstance(enrichment, Mapping):
        return None
    if enrichment.get("schema") != ENRICHMENT_SCHEMA_VERSION:
        return None
    if str(enrichment.get("record_id") or "") != str(record_id):
        return None
    if str(enrichment.get("status") or "") in {"failed", "payload_missing"}:
        return None
    row_updates = enrichment.get("row_updates")
    if not isinstance(row_updates, Mapping):
        return None
    statuses = enrichment.get("metric_statuses")
    if not isinstance(statuses, Mapping) or str(statuses.get(key) or "") != "ok":
        return None
    return _num(row_updates.get(key))


def _s_norm_from_enrichment(enrichment: Mapping[str, Any] | None, *, record_id: str) -> tuple[float | None, str]:
    if not isinstance(enrichment, Mapping):
        return None, "no_enrichment"
    if enrichment.get("schema") != ENRICHMENT_SCHEMA_VERSION:
        return None, "invalid_schema"
    if str(enrichment.get("record_id") or "") != str(record_id):
        return None, "record_id_mismatch"
    if str(enrichment.get("status") or "") in {"failed", "payload_missing"}:
        return None, str(enrichment.get("status") or "failed")
    row_updates = enrichment.get("row_updates")
    statuses = enrichment.get("metric_statuses")
    if not isinstance(row_updates, Mapping) or not isinstance(statuses, Mapping):
        return None, "missing_s_norm_status"
    status = str(statuses.get("S_norm") or "missing_s_norm_status")
    if status != "ok":
        return None, status
    value = _num(row_updates.get("S_norm"))
    if value is None:
        return None, "missing_s_norm_value"
    return value, "ok"


def _s_grp_from_enrichment(enrichment: Mapping[str, Any] | None, *, record_id: str) -> tuple[float | None, str]:
    if not isinstance(enrichment, Mapping):
        return None, "no_enrichment"
    if enrichment.get("schema") != ENRICHMENT_SCHEMA_VERSION:
        return None, "invalid_schema"
    if str(enrichment.get("record_id") or "") != str(record_id):
        return None, "record_id_mismatch"
    if str(enrichment.get("status") or "") in {"failed", "payload_missing"}:
        return None, str(enrichment.get("status") or "failed")
    row_updates = enrichment.get("row_updates")
    statuses = enrichment.get("metric_statuses")
    if not isinstance(row_updates, Mapping) or not isinstance(statuses, Mapping):
        return None, "missing_s_grp_status"
    status = str(statuses.get("S_grp") or "missing_s_grp_status")
    if status != "ok":
        return None, status
    value = _num(row_updates.get("S_grp_total"))
    if value is None:
        return None, "missing_s_grp_value"
    return value, "ok"


def _metric_from_enrichment(
    enrichment: Mapping[str, Any] | None,
    *,
    record_id: str,
    metric_key: str,
    row_update_key: str,
    missing_status: str,
) -> tuple[float | None, str]:
    if not isinstance(enrichment, Mapping):
        return None, "no_enrichment"
    if enrichment.get("schema") != ENRICHMENT_SCHEMA_VERSION:
        return None, "invalid_schema"
    if str(enrichment.get("record_id") or "") != str(record_id):
        return None, "record_id_mismatch"
    if str(enrichment.get("status") or "") in {"failed", "payload_missing"}:
        return None, str(enrichment.get("status") or "failed")
    row_updates = enrichment.get("row_updates")
    statuses = enrichment.get("metric_statuses")
    if not isinstance(row_updates, Mapping) or not isinstance(statuses, Mapping):
        return None, missing_status
    status = str(statuses.get(metric_key) or missing_status)
    if status != "ok":
        return None, status
    value = _num(row_updates.get(row_update_key))
    if value is None:
        return None, f"missing_{row_update_key}_value"
    return value, "ok"


def _s_alg_from_enrichment(enrichment: Mapping[str, Any] | None, *, record_id: str) -> tuple[float | None, str]:
    return _metric_from_enrichment(
        enrichment,
        record_id=record_id,
        metric_key="S_alg",
        row_update_key="S_alg",
        missing_status="missing_s_alg_status",
    )


def _s_phys_from_enrichment(enrichment: Mapping[str, Any] | None, *, record_id: str) -> tuple[float | None, str]:
    return _metric_from_enrichment(
        enrichment,
        record_id=record_id,
        metric_key="S_phys",
        row_update_key="S_phys",
        missing_status="missing_s_phys_status",
    )


def _s_l2_from_enrichment(enrichment: Mapping[str, Any] | None, *, record_id: str) -> tuple[float | None, str]:
    return _metric_from_enrichment(
        enrichment,
        record_id=record_id,
        metric_key="S_l2",
        row_update_key="S_l2",
        missing_status="missing_s_l2_status",
    )


def _s_var_from_enrichment(enrichment: Mapping[str, Any] | None, *, record_id: str) -> tuple[float | None, str]:
    value, status = _metric_from_enrichment(
        enrichment,
        record_id=record_id,
        metric_key="S_phys_var",
        row_update_key="S_var",
        missing_status="missing_s_phys_var_status",
    )
    if status == "missing_s_phys_var_status":
        return _metric_from_enrichment(
            enrichment,
            record_id=record_id,
            metric_key="S_var",
            row_update_key="S_var",
            missing_status="missing_s_var_status",
        )
    return value, status


def _legacy_shot_proxy(row: Mapping[str, Any]) -> tuple[float | None, str | None]:
    for key in ("shot_cost_proxy", "measurement_shots_proxy", "shots_total"):
        value = _num(row.get(key))
        if value is not None:
            return value, key
    return None, None


def _result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result")
    if isinstance(result, Mapping):
        return result
    rows = payload.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return rows[0]
    return payload


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return fmean(clean)


def _mean_complete(values: Sequence[float | None]) -> float | None:
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return None
    return fmean(float(v) for v in values)


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "--"
    av = abs(value)
    if av == 0:
        return "$0$"
    if av < 1e-2 or av >= 1e3:
        exponent = math.floor(math.log10(av))
        mantissa = value / (10 ** exponent)
        return f"${mantissa:.3g}\\!\\times\\!10^{{{exponent}}}$"
    if av < 1:
        return f"${value:.4f}$"
    return f"${value:.3g}$"


def _fmt_resource(value: float | None) -> str:
    if value is None:
        return "--"
    if abs(value) >= 100000:
        exponent = math.floor(math.log10(abs(value)))
        mantissa = value / (10 ** exponent)
        return f"${mantissa:.3g}\\!\\times\\!10^{{{exponent}}}$"
    if abs(value) >= 1000:
        return f"${value:.0f}$"
    if abs(value - round(value)) < 1e-9:
        return f"${int(round(value))}$"
    return f"${value:.1f}$"


def _metric_row(
    record: Mapping[str, str],
    payload_path: Path,
    payload: Mapping[str, Any],
    enrichment: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    r = _result(payload)
    same = _num(r.get("abs_delta_e_same_cutoff"))
    if same is None:
        same = _num(r.get("abs_delta_e"))
    enriched_same = _enrichment_update(enrichment, "abs_delta_e_same_cutoff", record_id=record["record_id"])
    if enriched_same is not None:
        same = enriched_same
    delta4 = _enrichment_update(enrichment, "abs_delta_e_reference", record_id=record["record_id"])
    if delta4 is None:
        delta4 = _num(r.get("abs_delta_e_reference"))
    if delta4 is None:
        delta4 = _num(r.get("delta_e_4"))
    count_2q = _enrichment_update(enrichment, "compiled_count_2q_total", record_id=record["record_id"])
    if count_2q is None:
        count_2q = _num(r.get("compiled_count_2q_total"))
    if count_2q is None:
        count_2q = _num(r.get("count_2q"))
    depth = _enrichment_update(enrichment, "compiled_depth_total", record_id=record["record_id"])
    if depth is None:
        depth = _num(r.get("compiled_depth_total"))
    if depth is None:
        depth = _num(r.get("circuit_depth"))
    legacy_shot, legacy_shot_source = _legacy_shot_proxy(r)
    s_norm, s_norm_status = _s_norm_from_enrichment(enrichment, record_id=record["record_id"])
    s_grp_total, s_grp_status = _s_grp_from_enrichment(enrichment, record_id=record["record_id"])
    s_alg, s_alg_status = _s_alg_from_enrichment(enrichment, record_id=record["record_id"])
    s_phys, s_phys_status = _s_phys_from_enrichment(enrichment, record_id=record["record_id"])
    s_l2, s_l2_status = _s_l2_from_enrichment(enrichment, record_id=record["record_id"])
    s_var, s_var_status = _s_var_from_enrichment(enrichment, record_id=record["record_id"])
    if s_alg is not None:
        measurement_work_proxy = s_alg
        measurement_work_proxy_source = "S_alg"
        measurement_work_proxy_status = "ok"
    else:
        measurement_work_proxy = None
        measurement_work_proxy_source = None
        measurement_work_proxy_status = s_alg_status
    if s_norm is not None:
        legacy_measurement_work_proxy = s_norm
        legacy_measurement_work_proxy_source = "S_norm"
        legacy_measurement_work_proxy_status = "legacy_normalized"
    else:
        legacy_measurement_work_proxy = None
        legacy_measurement_work_proxy_source = None
        legacy_measurement_work_proxy_status = f"unavailable:{s_norm_status}"
    depth_2q = _enrichment_update(enrichment, "compiled_depth_2q_total", record_id=record["record_id"])
    if depth_2q is None:
        depth_2q = _num(r.get("compiled_depth_2q_total")) if _num(r.get("compiled_depth_2q_total")) is not None else _num(r.get("depth_2q"))
    infidelity_same = _enrichment_update(enrichment, "infidelity_same", record_id=record["record_id"])
    if infidelity_same is None:
        infidelity_same = _num(r.get("infidelity_exact"))
    infidelity_4 = _enrichment_update(enrichment, "infidelity_4", record_id=record["record_id"])
    if same is None and count_2q is None and depth is None and measurement_work_proxy is None:
        return None
    status = str(payload.get("status") or r.get("status") or "unknown")
    return {
        "record_id": record["record_id"],
        "family": record["family"],
        "case_id": record["case_id"],
        "algorithm_id": record["algorithm_id"],
        "method": table_i_method_label(record["algorithm_id"]),
        "class": TABLE_I_CLASS_BY_FAMILY.get(record["family"], "unmapped"),
        "payload_path": str(payload_path),
        "payload_status": status,
        "quality_gate_reason": r.get("quality_gate_reason"),
        "failure_reason": r.get("failure_reason"),
        "delta_e_same": same,
        "delta_e_4": delta4,
        "infidelity_same": infidelity_same,
        "infidelity_4": infidelity_4,
        "count_2q": count_2q,
        "depth_2q": depth_2q,
        "circuit_depth": depth,
        "raw_shot_cost_proxy": _num(r.get("shot_cost_proxy")),
        "raw_measurement_shots_proxy": _num(r.get("measurement_shots_proxy")),
        "raw_shots_total": _num(r.get("shots_total")),
        "legacy_shot_proxy": legacy_shot,
        "legacy_shot_proxy_source": legacy_shot_source,
        "raw_shot_proxy_fallback_forbidden": legacy_shot is not None,
        "shot_cost_proxy": None,
        "shot_cost_proxy_status": "raw_fallback_forbidden" if legacy_shot is not None else "unavailable",
        "S_norm": s_norm,
        "S_norm_status": s_norm_status,
        "S_grp_total": s_grp_total,
        "S_grp_status": s_grp_status,
        "S_grp_source": "S_grp" if s_grp_total is not None else None,
        "S_alg": s_alg,
        "S_alg_status": s_alg_status,
        "S_phys": s_phys,
        "S_phys_status": s_phys_status,
        "S_l2": s_l2,
        "S_l2_status": s_l2_status,
        "S_var": s_var,
        "S_phys_var_status": s_var_status,
        "measurement_work_proxy": measurement_work_proxy,
        "measurement_work_proxy_source": measurement_work_proxy_source,
        "measurement_work_proxy_status": measurement_work_proxy_status,
        "legacy_measurement_work_proxy": legacy_measurement_work_proxy,
        "legacy_measurement_work_proxy_source": legacy_measurement_work_proxy_source,
        "legacy_measurement_work_proxy_status": legacy_measurement_work_proxy_status,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        method = str(row["method"])
        klass = str(row["class"])
        alg = str(row["algorithm_id"])
        grouped[(klass, method, alg)].append(row)
        grouped[("all averaged", method, alg)].append(row)

    out: list[dict[str, Any]] = []
    algorithm_order = list(ALGORITHM_ORDER)
    for row in rows:
        alg = str(row["algorithm_id"])
        if alg not in algorithm_order:
            algorithm_order.append(alg)

    for klass in CLASS_ORDER:
        for alg in algorithm_order:
            method = table_i_method_label(alg)
            items = grouped.get((klass, method, alg), [])
            if not items:
                continue
            out.append(
                {
                    "class": klass,
                    "method": method,
                    "algorithm_id": alg,
                    "n": len(items),
                    "quality_nonpassing_n": sum(1 for x in items if x.get("quality_gate_reason") or str(x.get("payload_status")) == "failed"),
                    "delta_e_same_mean": _mean([x.get("delta_e_same") for x in items]),
                    "delta_e_4_mean": _mean_complete([x.get("delta_e_4") for x in items]),
                    "infidelity_same_mean": _mean_complete([x.get("infidelity_same") for x in items]),
                    "infidelity_4_mean": _mean_complete([x.get("infidelity_4") for x in items]),
                    "count_2q_mean": _mean([x.get("count_2q") for x in items]),
                    "depth_2q_mean": _mean_complete([x.get("depth_2q") for x in items]),
                    "circuit_depth_mean": _mean([x.get("circuit_depth") for x in items]),
                    "shot_cost_proxy_mean": _mean([x.get("shot_cost_proxy") for x in items]),
                    "legacy_shot_proxy_mean": _mean([x.get("legacy_shot_proxy") for x in items]),
                    "S_norm_mean": _mean([x.get("S_norm") for x in items]),
                    "S_norm_available_n": sum(1 for x in items if x.get("S_norm") is not None),
                    "S_norm_status_counts": dict(Counter(str(x.get("S_norm_status") or "none") for x in items)),
                    "S_alg_mean": _mean([x.get("S_alg") for x in items]),
                    "S_alg_available_n": sum(1 for x in items if x.get("S_alg") is not None),
                    "S_alg_status_counts": dict(Counter(str(x.get("S_alg_status") or "none") for x in items)),
                    "S_phys_mean": _mean([x.get("S_phys") for x in items]),
                    "S_phys_available_n": sum(1 for x in items if x.get("S_phys") is not None),
                    "S_phys_status_counts": dict(Counter(str(x.get("S_phys_status") or "none") for x in items)),
                    "S_l2_mean": _mean([x.get("S_l2") for x in items]),
                    "S_l2_available_n": sum(1 for x in items if x.get("S_l2") is not None),
                    "S_l2_status_counts": dict(Counter(str(x.get("S_l2_status") or "none") for x in items)),
                    "S_var_mean": _mean([x.get("S_var") for x in items]),
                    "S_var_available_n": sum(1 for x in items if x.get("S_var") is not None),
                    "S_phys_var_status_counts": dict(Counter(str(x.get("S_phys_var_status") or "none") for x in items)),
                    "S_grp_total_mean": _mean([x.get("S_grp_total") for x in items]),
                    "S_grp_available_n": sum(1 for x in items if x.get("S_grp_total") is not None),
                    "S_grp_status_counts": dict(Counter(str(x.get("S_grp_status") or "none") for x in items)),
                    "S_grp_source_counts": dict(Counter(str(x.get("S_grp_source") or "none") for x in items)),
                    "raw_shot_fallback_n": sum(1 for x in items if x.get("raw_shot_proxy_fallback_forbidden")),
                    "measurement_work_proxy_mean": _mean([x.get("measurement_work_proxy") for x in items]),
                    "measurement_work_source_counts": dict(Counter(str(x.get("measurement_work_proxy_source") or "none") for x in items)),
                    "legacy_measurement_work_proxy_mean": _mean([x.get("legacy_measurement_work_proxy") for x in items]),
                    "legacy_measurement_work_source_counts": dict(Counter(str(x.get("legacy_measurement_work_proxy_source") or "none") for x in items)),
                }
            )
    return out


def _latex_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "% Auto-generated by pipelines/exact_bench/summarize_table_i_static_results.py.",
        "% Paste into tab:static_claims after checking the run coverage and captions.",
        "% Last column uses measurement_work_proxy_mean: S_alg only; -- means apples-to-apples event telemetry is missing.",
    ]
    prev_class = None
    for row in rows:
        klass = str(row["class"])
        if prev_class is not None and klass != prev_class:
            lines.append(r"\colrule")
        prev_class = klass
        note = ""
        if int(row.get("quality_nonpassing_n") or 0):
            note = f" % includes {int(row['quality_nonpassing_n'])} quality-gate nonpassing stress outcome(s)"
        lines.append(
            f"{klass} & {row['method']} & "
            f"{_fmt_num(row.get('delta_e_same_mean'))} & "
            f"{_fmt_num(row.get('delta_e_4_mean'))} & "
            f"{_fmt_num(row.get('infidelity_same_mean'))} & "
            f"{_fmt_num(row.get('infidelity_4_mean'))} & "
            f"{_fmt_resource(row.get('count_2q_mean'))} & "
            f"{_fmt_resource(row.get('depth_2q_mean'))} & "
            f"{_fmt_resource(row.get('circuit_depth_mean'))} & "
            f"{_fmt_resource(row.get('measurement_work_proxy_mean'))} \\\\{note}"
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class", "method", "algorithm_id", "n", "quality_nonpassing_n",
        "delta_e_same_mean", "delta_e_4_mean", "infidelity_same_mean", "infidelity_4_mean",
        "count_2q_mean", "depth_2q_mean", "circuit_depth_mean",
        "measurement_work_proxy_mean", "S_alg_mean", "S_alg_available_n", "S_alg_status_counts",
        "S_phys_mean", "S_phys_available_n", "S_phys_status_counts",
        "S_l2_mean", "S_l2_available_n", "S_l2_status_counts",
        "S_var_mean", "S_var_available_n", "S_phys_var_status_counts",
        "S_norm_mean", "S_norm_available_n", "S_norm_status_counts", "raw_shot_fallback_n",
        "S_grp_total_mean", "S_grp_available_n", "S_grp_status_counts", "S_grp_source_counts",
        "shot_cost_proxy_mean", "legacy_shot_proxy_mean", "measurement_work_source_counts",
        "legacy_measurement_work_proxy_mean", "legacy_measurement_work_source_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize generic static Table-I benchmark outputs for Paper I.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--enrichment-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-incomplete", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = _load_records(args.records)
    missing: list[dict[str, str]] = []
    unusable: list[dict[str, str]] = []
    metric_rows: list[dict[str, Any]] = []
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
        row = _metric_row(record, payload_path, payload, enrichment)
        if row is None:
            unusable.append({**record, "payload_path": str(payload_path)})
            continue
        metric_rows.append(row)

    aggregate_rows = _aggregate(metric_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "table_i_static_paper_results_v1",
        "records_path": str(args.records),
        "output_root": str(args.root),
        "enrichment_root": str(args.enrichment_root) if args.enrichment_root is not None else None,
        "enrichment_available_count": enrichment_available,
        "enrichment_missing_count": enrichment_missing,
        "enrichment_failed_count": enrichment_failed,
        "expected_count": len(records),
        "benchmarked_count": len(metric_rows),
        "missing_count": len(missing),
        "unusable_count": len(unusable),
        "quality_nonpassing_count": sum(1 for row in metric_rows if row.get("quality_gate_reason") or row.get("payload_status") == "failed"),
        "missing": missing,
        "unusable": unusable,
        "aggregate_rows": aggregate_rows,
        "row_results": metric_rows,
    }
    json_path = args.output_dir / "table_i_static_results_summary.json"
    csv_path = args.output_dir / "table_i_static_results_rows.csv"
    tex_path = args.output_dir / "table_i_static_claim_rows.tex"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, aggregate_rows)
    tex_path.write_text(_latex_rows(aggregate_rows), encoding="utf-8")
    print(json.dumps({
        "summary_json": str(json_path),
        "rows_csv": str(csv_path),
        "latex_rows": str(tex_path),
        "expected_count": summary["expected_count"],
        "benchmarked_count": summary["benchmarked_count"],
        "missing_count": summary["missing_count"],
        "unusable_count": summary["unusable_count"],
        "quality_nonpassing_count": summary["quality_nonpassing_count"],
    }, indent=2, sort_keys=True))
    if not bool(args.allow_incomplete) and (missing or unusable):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
