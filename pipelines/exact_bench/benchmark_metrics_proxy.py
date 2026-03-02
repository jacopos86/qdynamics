#!/usr/bin/env python3
"""Wrapper-level HH benchmark proxy metrics sidecars.

This module intentionally stays outside core operator/model code and provides
sidecar artifacts only.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "hh_bench_metrics_v1"


@dataclass(frozen=True)
class ProxyMetricRow:
    run_id: str = ""
    status: str = ""
    method_id: str = ""
    method_kind: str = ""
    ansatz_name: str = ""
    pool_name: str = ""
    problem: str = ""
    L: int | None = None
    runtime_s: float | None = None
    started_utc: str = ""
    finished_utc: str = ""
    wallclock_cap_s: int | None = None
    nfev: int | None = None
    nit: int | None = None
    num_parameters: int | None = None
    vqe_reps: int | None = None
    vqe_restarts: int | None = None
    vqe_maxiter: int | None = None
    depth_proxy: int | None = None
    operator_family_proxy: str = ""
    pool_family_proxy: str = ""
    fidelity_kernel_proxy: str = ""
    delta_E_abs: float | None = None
    sector_leak_flag: bool | None = None
    adapt_stop_reason: str = ""
    adapt_depth_reached: int | None = None


PROXY_FIELD_ORDER = list(ProxyMetricRow.__dataclass_fields__.keys())


def _to_int(x: Any) -> int | None:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


def _to_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _to_bool(x: Any) -> bool | None:
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    return None


def _first_nonempty(*vals: Any) -> Any:
    for v in vals:
        if v is not None and v != "":
            return v
    return None


def _infer_depth_proxy(merged: Mapping[str, Any], num_parameters: int | None) -> int | None:
    return _to_int(
        _first_nonempty(
            merged.get("depth_proxy"),
            merged.get("adapt_depth_reached"),
            merged.get("adapt_depth"),
            merged.get("ansatz_depth"),
            num_parameters,
            merged.get("num_params"),
        )
    )


def _infer_operator_family_proxy(
    method_kind: str,
    method_id: str,
    ansatz_name: str,
    pool_name: str,
    run_id: str,
) -> str:
    hay = " ".join(
        [
            str(method_id).lower(),
            str(ansatz_name).lower(),
            str(pool_name).lower(),
            str(run_id).lower(),
        ]
    )
    tags: list[str] = []
    if method_kind:
        tags.append(str(method_kind).lower())
    for token in ("uccsd", "paop", "hva"):
        if token in hay:
            tags.append(token)
    if not tags:
        label = _first_nonempty(pool_name, ansatz_name, method_id)
        if label is not None:
            tags.append(str(label).strip().lower())
    # Stable order and de-duplication
    out: list[str] = []
    for tag in tags:
        if tag and tag not in out:
            out.append(tag)
    return "+".join(out)


def _infer_pool_family_proxy(method_id: str, ansatz_name: str, pool_name: str) -> str:
    if str(pool_name).strip() != "":
        return str(pool_name).strip().lower()
    hay = f"{method_id} {ansatz_name}".lower()
    if "paop" in hay:
        return "paop"
    if "hva" in hay:
        return "hva"
    if "uccsd" in hay:
        return "uccsd"
    return ""


def _infer_fidelity_kernel_proxy(merged: Mapping[str, Any]) -> str:
    mode = _first_nonempty(merged.get("fidelity_grouping_mode"))
    steps = _to_int(_first_nonempty(merged.get("fidelity_residual_trotter_steps")))
    tol = _to_float(_first_nonempty(merged.get("fidelity_coeff_tolerance")))
    if mode is None and steps is None and tol is None:
        return ""
    parts: list[str] = []
    if mode is not None:
        parts.append(f"mode={mode}")
    if steps is not None:
        parts.append(f"steps={steps}")
    if tol is not None:
        parts.append(f"tol={tol:.3e}")
    return "|".join(parts)


def extract_proxy_metric_row(
    row: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> ProxyMetricRow:
    merged: dict[str, Any] = {}
    if defaults is not None:
        merged.update(defaults)
    merged.update(dict(row))

    run_id = str(_first_nonempty(merged.get("run_id"), merged.get("name"), "") or "")
    status = str(_first_nonempty(merged.get("status"), "") or "")
    method_id = str(_first_nonempty(merged.get("method_id"), merged.get("name"), run_id) or "")
    method_kind = str(_first_nonempty(merged.get("method_kind"), merged.get("category"), "") or "")
    ansatz_name = str(_first_nonempty(merged.get("ansatz_name"), merged.get("name"), "") or "")
    pool_name = str(_first_nonempty(merged.get("pool_name"), "") or "")
    problem = str(_first_nonempty(merged.get("problem"), merged.get("model"), "") or "")
    l_value = _to_int(_first_nonempty(merged.get("L"), merged.get("num_sites")))
    runtime_s = _to_float(_first_nonempty(merged.get("runtime_s"), merged.get("elapsed_s")))
    started_utc = str(_first_nonempty(merged.get("started_utc"), "") or "")
    finished_utc = str(_first_nonempty(merged.get("finished_utc"), "") or "")
    wallclock_cap_s = _to_int(merged.get("wallclock_cap_s"))

    nfev = _to_int(merged.get("nfev"))
    nit = _to_int(merged.get("nit"))
    num_parameters = _to_int(_first_nonempty(merged.get("num_parameters"), merged.get("num_params")))
    vqe_reps = _to_int(_first_nonempty(merged.get("vqe_reps"), merged.get("vqe_reps_used"), merged.get("reps")))
    vqe_restarts = _to_int(merged.get("vqe_restarts"))
    vqe_maxiter = _to_int(_first_nonempty(merged.get("vqe_maxiter"), merged.get("vqe_maxiter_used"), merged.get("maxiter")))

    adapt_depth_reached = _to_int(_first_nonempty(merged.get("adapt_depth_reached"), merged.get("adapt_depth")))
    depth_proxy = _infer_depth_proxy(merged, num_parameters)
    operator_family_proxy = _infer_operator_family_proxy(
        method_kind=method_kind,
        method_id=method_id,
        ansatz_name=ansatz_name,
        pool_name=pool_name,
        run_id=run_id,
    )
    pool_family_proxy = _infer_pool_family_proxy(method_id=method_id, ansatz_name=ansatz_name, pool_name=pool_name)
    fidelity_kernel_proxy = _infer_fidelity_kernel_proxy(merged)

    delta_e_abs = _to_float(_first_nonempty(merged.get("delta_E_abs"), merged.get("abs_delta_e")))
    sector_leak_flag = _to_bool(merged.get("sector_leak_flag"))
    adapt_stop_reason = str(_first_nonempty(merged.get("adapt_stop_reason"), "") or "")

    return ProxyMetricRow(
        run_id=run_id,
        status=status,
        method_id=method_id,
        method_kind=method_kind,
        ansatz_name=ansatz_name,
        pool_name=pool_name,
        problem=problem,
        L=l_value,
        runtime_s=runtime_s,
        started_utc=started_utc,
        finished_utc=finished_utc,
        wallclock_cap_s=wallclock_cap_s,
        nfev=nfev,
        nit=nit,
        num_parameters=num_parameters,
        vqe_reps=vqe_reps,
        vqe_restarts=vqe_restarts,
        vqe_maxiter=vqe_maxiter,
        depth_proxy=depth_proxy,
        operator_family_proxy=operator_family_proxy,
        pool_family_proxy=pool_family_proxy,
        fidelity_kernel_proxy=fidelity_kernel_proxy,
        delta_E_abs=delta_e_abs,
        sector_leak_flag=sector_leak_flag,
        adapt_stop_reason=adapt_stop_reason,
        adapt_depth_reached=adapt_depth_reached,
    )


def summarize_proxy_rows(
    rows: Sequence[ProxyMetricRow],
    *,
    summary_extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_vals = [float(r.runtime_s) for r in rows if r.runtime_s is not None]
    depth_vals = [int(r.depth_proxy) for r in rows if r.depth_proxy is not None]
    delta_vals = [float(r.delta_E_abs) for r in rows if r.delta_E_abs is not None]

    status_counts: dict[str, int] = {}
    for r in rows:
        key = str(r.status).strip().lower()
        if key == "":
            continue
        status_counts[key] = int(status_counts.get(key, 0) + 1)

    ok_rows = []
    for r in rows:
        if str(r.status).strip().lower() == "ok":
            ok_rows.append(r)
        elif str(r.status).strip() == "" and r.delta_E_abs is not None:
            ok_rows.append(r)

    op_counts: dict[str, int] = {}
    pool_counts: dict[str, int] = {}
    method_best_delta: dict[str, float] = {}
    for r in rows:
        if r.operator_family_proxy:
            op_counts[r.operator_family_proxy] = int(op_counts.get(r.operator_family_proxy, 0) + 1)
        if r.pool_family_proxy:
            pool_counts[r.pool_family_proxy] = int(pool_counts.get(r.pool_family_proxy, 0) + 1)
        if r.method_id and r.delta_E_abs is not None:
            old = method_best_delta.get(r.method_id)
            val = float(r.delta_E_abs)
            if old is None or val < old:
                method_best_delta[r.method_id] = val

    summary: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(rows)),
        "ok_row_count": int(len(ok_rows)),
        "status_counts": status_counts,
        "runtime_s": {
            "sum": float(sum(runtime_vals)) if runtime_vals else None,
            "median": float(statistics.median(runtime_vals)) if runtime_vals else None,
            "max": float(max(runtime_vals)) if runtime_vals else None,
        },
        "depth_proxy": {
            "min": int(min(depth_vals)) if depth_vals else None,
            "median": float(statistics.median(depth_vals)) if depth_vals else None,
            "max": int(max(depth_vals)) if depth_vals else None,
        },
        "delta_E_abs": {
            "best": float(min(delta_vals)) if delta_vals else None,
            "median": float(statistics.median(delta_vals)) if delta_vals else None,
        },
        "method_best_delta": method_best_delta,
        "operator_family_counts": op_counts,
        "pool_family_counts": pool_counts,
    }
    if summary_extras is not None:
        for key, value in dict(summary_extras).items():
            summary[key] = value
    return summary


def write_proxy_sidecars(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    defaults: Mapping[str, Any] | None = None,
    summary_extras: Mapping[str, Any] | None = None,
    csv_name: str = "metrics_proxy_runs.csv",
    jsonl_name: str = "metrics_proxy_runs.jsonl",
    summary_name: str = "metrics_proxy_summary.json",
) -> dict[str, Path]:
    normalized_rows = [extract_proxy_metric_row(row, defaults=defaults) for row in rows]
    out_rows = [asdict(r) for r in normalized_rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / str(csv_name)
    jsonl_path = output_dir / str(jsonl_name)
    summary_path = output_dir / str(summary_name)

    with csv_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=PROXY_FIELD_ORDER)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, None) for k in PROXY_FIELD_ORDER})

    with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        for row in out_rows:
            f_jsonl.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    summary_payload = summarize_proxy_rows(normalized_rows, summary_extras=summary_extras)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "csv": csv_path,
        "jsonl": jsonl_path,
        "summary_json": summary_path,
    }

