#!/usr/bin/env python3
"""Paper-I HH backend compile-cost latency diagnostic.

This diagnostic compares the strict full-trial backend compile path against the
incremental prefix/suffix path for recorded HH SNAKE candidate-position
admissions. It is reporting-only and does not mutate paper tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.generic_static_adapt_variants import _resolve_context_from_spec
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
    table_i_canonical_spec_by_case_id,
)
from pipelines.static_adapt.builders.pool_resolution import (
    resolve_pool_plan,
    resolve_requested_pool_filters,
)
from pipelines.static_adapt.engine_support import _adapt_energy_fn
from pipelines.static_adapt.hh_backend_compile_oracle import (
    BackendCompileConfig,
    BackendCompileOracle,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_polynomial import compile_polynomial_action
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


DEFAULT_CASE_ID = "hh_L2_nph4_three_model_sym_strong_strong"
DEFAULT_SOURCE_JSON = (
    "raw_outputs/chtc_fetches/hh_snake_all_time_best_ws_ss_20260531/"
    "hh_ss_trial0001_result.json"
)
DEFAULT_MEASUREMENT_JSON = (
    "raw_outputs/chtc_fetches/"
    "paper_i_hh_tableiii_snake_shot_proxy_repair_20260612_v1_fetch_20260613T1634Z/"
    "raw_outputs/paper_i_hh_tableiii_snake_shot_proxy_repair_20260612_v1/"
    "paper_i_hh_tableiii_snake_shot_proxy_repair_20260612_v1_strong_strong/result/"
    "generic_static_single.json"
)
DEFAULT_CLASS_FILTER_JSON = "agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json"
COMPILE_MODES = ("transpile_single_v1", "incremental_prefix_suffix_v1")
SCHEMA = "paper_i_hh_backend_compile_latency_diagnostic_v1"


@dataclass(frozen=True)
class AdmissionRecord:
    step_index: int
    label: str
    position_id: int
    source_path: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_path(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _payload_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result")
    return result if isinstance(result, Mapping) else payload


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _median(values: Sequence[float]) -> float | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(clean)) if clean else None


def _p90(values: Sequence[float]) -> float | None:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return None
    idx = int(math.ceil(0.9 * len(clean))) - 1
    return float(clean[max(0, min(idx, len(clean) - 1))])


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(val):
        return "--"
    if val == 0.0:
        return "0"
    if abs(val) >= 1e4 or abs(val) < 1e-3:
        return f"{val:.{digits}e}"
    return f"{val:.{digits}g}"


def _candidate_label_from_record(record: Mapping[str, Any]) -> str | None:
    for key in ("generator_label", "candidate_label", "selected_label", "operator_label", "label"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _position_from_record(record: Mapping[str, Any], default: int) -> int:
    for key in ("position_id", "append_position", "selected_position"):
        value = record.get(key)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except Exception:
            continue
    return int(default)


def extract_admission_records(payload: Mapping[str, Any]) -> list[AdmissionRecord]:
    """Extract candidate-position admissions from a SNAKE result payload."""

    sources: list[tuple[str, Any]] = [
        (
            "adapt_vqe.continuation.selected_scaffold_record_chain",
            _get_path(payload, ("adapt_vqe", "continuation", "selected_scaffold_record_chain")),
        ),
        (
            "adapt_vqe.continuation.selected_scaffold_history.selected_records",
            _get_path(payload, ("adapt_vqe", "continuation", "selected_scaffold_history")),
        ),
        ("adapt_vqe.history.admitted_records", _get_path(payload, ("adapt_vqe", "history"))),
    ]
    for source_path, raw in sources:
        out: list[AdmissionRecord] = []
        if not isinstance(raw, list):
            continue
        if source_path.endswith("selected_records"):
            for idx, step in enumerate(raw, start=1):
                if not isinstance(step, Mapping):
                    continue
                records = step.get("selected_records")
                if not isinstance(records, list):
                    continue
                for rec in records:
                    if not isinstance(rec, Mapping):
                        continue
                    label = _candidate_label_from_record(rec)
                    if label is None:
                        continue
                    out.append(
                        AdmissionRecord(
                            step_index=int(step.get("step_index") or idx),
                            label=str(label),
                            position_id=_position_from_record(rec, len(out)),
                            source_path=source_path,
                        )
                    )
        elif source_path.endswith("admitted_records"):
            for idx, step in enumerate(raw, start=1):
                if not isinstance(step, Mapping):
                    continue
                records = step.get("admitted_records")
                if not isinstance(records, list):
                    continue
                for rec in records:
                    if not isinstance(rec, Mapping):
                        continue
                    label = _candidate_label_from_record(rec)
                    if label is None:
                        continue
                    out.append(
                        AdmissionRecord(
                            step_index=int(step.get("step_index") or step.get("depth") or idx),
                            label=str(label),
                            position_id=_position_from_record(rec, len(out)),
                            source_path=source_path,
                        )
                    )
        else:
            for idx, rec in enumerate(raw, start=1):
                if not isinstance(rec, Mapping):
                    continue
                label = _candidate_label_from_record(rec)
                if label is None:
                    continue
                out.append(
                    AdmissionRecord(
                        step_index=int(rec.get("step_index") or rec.get("record_index") or idx),
                        label=str(label),
                        position_id=_position_from_record(rec, len(out)),
                        source_path=source_path,
                    )
                )
        if out:
            return out
    return []


def _theta_from_payload(payload: Mapping[str, Any], expected_len: int) -> tuple[np.ndarray, str]:
    paths = (
        ("adapt_vqe", "continuation", "selected_scaffold_summary", "theta_adapt"),
        ("adapt_vqe", "theta"),
        ("adapt_vqe", "optimal_point"),
        ("result", "theta"),
    )
    for path in paths:
        raw = _get_path(payload, path)
        if not isinstance(raw, list):
            continue
        try:
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except Exception:
            continue
        if arr.size == int(expected_len):
            return arr, ".".join(path)
    return np.zeros(int(expected_len), dtype=float), "zeros_fallback_theta_length_mismatch_or_missing"


def _build_pool(
    *,
    case_id: str,
    profile: str,
    class_filter_json: Path,
) -> tuple[Any, list[AnsatzTerm], dict[str, Any], dict[str, Any]]:
    spec = table_i_canonical_spec_by_case_id("hh", str(case_id), profile=str(profile))
    context = _resolve_context_from_spec(spec)
    request = context.request
    filter_resolution = resolve_requested_pool_filters(
        problem_key=str(context.family_key),
        num_sites=int(request.num_sites),
        n_ph_max=int(request.n_ph_max),
        adapt_pool="full_meta",
        adapt_pool_class_filter_json=class_filter_json,
        adapt_pool_label_filter_json=None,
        adapt_selected_logical_source_json=None,
        adapt_selected_logical_mode="off",
        adapt_selected_logical_transfer_mode="exact_match_v1",
    )
    plan = resolve_pool_plan(
        resolved_problem=context,
        continuation_mode="phase3_v1",
        adapt_pool="full_meta",
        paop_r=3,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        phase3_symmetry_mitigation_mode="off",
        filter_resolution=filter_resolution,
        ai_log=None,
    )
    return (
        context,
        list(plan.pool),
        {
            "pool_key": str(plan.pool_key),
            "method_name": str(plan.method_name),
            "pool_size": int(len(plan.pool)),
            "qpb": int(plan.qpb),
            "full_meta_class_filter_meta": plan.full_meta_class_filter_meta,
            "full_meta_label_filter_meta": plan.full_meta_label_filter_meta,
            "pool_legal_subspace_filter_meta": plan.pool_legal_subspace_filter_meta,
        },
        {
            "benchmark_id": str(spec.benchmark_id),
            "family": str(spec.family),
            "base_pipeline_args": list(spec.base_pipeline_args),
            "exact_reference_n_ph_max": getattr(spec, "exact_reference_n_ph_max", None),
            "tags": list(getattr(spec, "tags", ())),
        },
    )


def _normalize_state(raw: Any) -> np.ndarray:
    state = np.asarray(raw, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise ValueError("reference state has zero norm")
    return state / norm


def _candidate_sequence(
    admissions: Sequence[AdmissionRecord],
    pool_by_label: Mapping[str, AnsatzTerm],
    *,
    record_limit: int | None,
) -> tuple[list[dict[str, Any]], list[AnsatzTerm], list[str]]:
    current: list[AnsatzTerm] = []
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    limit = len(admissions) if record_limit is None else min(int(record_limit), len(admissions))
    for idx, rec in enumerate(admissions[:limit]):
        term = pool_by_label.get(str(rec.label))
        if term is None:
            missing.append(str(rec.label))
            continue
        pos = max(0, min(int(rec.position_id), len(current)))
        rows.append(
            {
                "sequence_index": int(idx),
                "step_index": int(rec.step_index),
                "candidate_label": str(rec.label),
                "position_id": int(pos),
                "base_depth": int(len(current)),
                "source_path": str(rec.source_path),
            }
        )
        current.insert(pos, term)
    return rows, current, missing


def _estimate_selected_metadata(estimate: Any) -> dict[str, Any]:
    row = getattr(estimate, "selected_backend_row", None)
    selected = dict(row) if isinstance(row, Mapping) else {}
    inc = selected.get("incremental_prefix_suffix")
    return {
        "compile_gate_open": bool(getattr(estimate, "compile_gate_open", False)),
        "source_mode": str(getattr(estimate, "source_mode", "")),
        "hardware_cost_source": str(getattr(estimate, "hardware_cost_source", "")),
        "selected_backend_name": getattr(estimate, "selected_backend_name", None),
        "penalty_total": getattr(estimate, "penalty_total", None),
        "delta_compiled_count_2q": getattr(estimate, "delta_compiled_count_2q", None),
        "delta_compiled_depth_2q": getattr(estimate, "delta_compiled_depth_2q", None),
        "delta_compiled_depth": getattr(estimate, "delta_compiled_depth", None),
        "delta_compiled_size": getattr(estimate, "delta_compiled_size", None),
        "incremental_prefix_suffix": inc if isinstance(inc, Mapping) else None,
    }


def time_compile_mode(
    *,
    mode: str,
    admissions: Sequence[AdmissionRecord],
    pool_by_label: Mapping[str, AnsatzTerm],
    num_qubits: int,
    ref_state: np.ndarray,
    backend_name: str,
    seed_transpiler: int,
    optimization_level: int,
    structure_theta_value: float,
    weight_2q: float,
    weight_depth: float,
    weight_size: float,
    record_limit: int | None,
) -> dict[str, Any]:
    config = BackendCompileConfig(
        mode=str(mode),
        requested_backend_name=str(backend_name),
        seed_transpiler=int(seed_transpiler),
        optimization_level=int(optimization_level),
        structure_theta_value=float(structure_theta_value),
        weight_2q=float(weight_2q),
        weight_depth=float(weight_depth),
        weight_size=float(weight_size),
    )
    oracle = BackendCompileOracle(config=config, num_qubits=int(num_qubits), ref_state=ref_state)
    current_ops: list[AnsatzTerm] = []
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    limit = len(admissions) if record_limit is None else min(int(record_limit), len(admissions))
    for seq_index, rec in enumerate(admissions[:limit]):
        candidate = pool_by_label.get(str(rec.label))
        if candidate is None:
            missing.append(str(rec.label))
            continue
        pos = max(0, min(int(rec.position_id), len(current_ops)))
        snapshot_t0 = time.perf_counter()
        snapshot = oracle.snapshot_base(list(current_ops))
        snapshot_s = time.perf_counter() - snapshot_t0
        before_cache = dict(oracle.cache_summary())
        estimate_t0 = time.perf_counter()
        estimate = oracle.estimate_insertion(snapshot, candidate_term=candidate, position_id=pos, proxy_baseline=None)
        estimate_s = time.perf_counter() - estimate_t0
        after_cache = dict(oracle.cache_summary())
        rows.append(
            {
                "mode": str(mode),
                "sequence_index": int(seq_index),
                "step_index": int(rec.step_index),
                "candidate_label": str(rec.label),
                "position_id": int(pos),
                "base_depth": int(len(current_ops)),
                "snapshot_s": float(snapshot_s),
                "estimate_s": float(estimate_s),
                "total_s": float(snapshot_s + estimate_s),
                "cache_before": before_cache,
                "cache_after": after_cache,
                **_estimate_selected_metadata(estimate),
            }
        )
        current_ops.insert(pos, candidate)
    snapshot_times = [float(row["snapshot_s"]) for row in rows]
    estimate_times = [float(row["estimate_s"]) for row in rows]
    total_times = [float(row["total_s"]) for row in rows]
    source_modes = sorted({str(row.get("source_mode")) for row in rows if row.get("source_mode")})
    return {
        "mode": str(mode),
        "record_count": int(len(rows)),
        "missing_labels": sorted(set(missing)),
        "resolution_audit": list(getattr(oracle, "resolution_audit", [])),
        "cache_summary": dict(oracle.cache_summary()),
        "source_modes": source_modes,
        "snapshot_total_s": float(sum(snapshot_times)),
        "snapshot_median_s": _median(snapshot_times),
        "snapshot_p90_s": _p90(snapshot_times),
        "estimate_total_s": float(sum(estimate_times)),
        "estimate_median_s": _median(estimate_times),
        "estimate_p90_s": _p90(estimate_times),
        "compile_path_total_s": float(sum(total_times)),
        "compile_path_median_s": _median(total_times),
        "compile_path_p90_s": _p90(total_times),
        "rows": rows,
    }


def _time_energy_objective(
    *,
    context: Any,
    selected_ops: Sequence[AnsatzTerm],
    theta: np.ndarray,
    ref_state: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    compile_t0 = time.perf_counter()
    h_compiled = compile_polynomial_action(context.hamiltonian)
    h_compile_s = time.perf_counter() - compile_t0
    layout = build_parameter_layout(
        list(selected_ops),
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    if theta_arr.size != int(layout.logical_parameter_count):
        theta_arr = np.zeros(int(layout.logical_parameter_count), dtype=float)
        theta_source_status = "zeros_fallback_theta_layout_mismatch"
    else:
        theta_source_status = "source_theta"
    # Warm one call outside the timing set.
    _adapt_energy_fn(
        context.hamiltonian,
        ref_state,
        list(selected_ops),
        theta_arr,
        h_compiled=h_compiled,
        parameter_layout=layout,
    )
    times: list[float] = []
    energies: list[float] = []
    for _idx in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        energy = _adapt_energy_fn(
            context.hamiltonian,
            ref_state,
            list(selected_ops),
            theta_arr,
            h_compiled=h_compiled,
            parameter_layout=layout,
        )
        times.append(float(time.perf_counter() - t0))
        energies.append(float(energy))
    median = _median(times)
    return {
        "hamiltonian_compile_s": float(h_compile_s),
        "repeats": int(max(1, int(repeats))),
        "median_s": median,
        "p90_s": _p90(times),
        "total_s": float(sum(times)),
        "times_s": times,
        "last_energy": energies[-1] if energies else None,
        "theta_source_status": theta_source_status,
        "nfev_8000_median_walltime_s": (None if median is None else float(8000.0 * median)),
        "spsa_maxiter_8000_nfev_eval_repeats1_avg_last0": 16001,
    }


def _measurement_proxy(
    *,
    measurement_payload: Mapping[str, Any] | None,
    context: Any,
    nfev: int,
    shot_rates: Sequence[float],
) -> dict[str, Any]:
    row = _payload_result(measurement_payload or {})
    h_count = _to_float(row.get("hamiltonian_pauli_term_count"))
    if h_count is None:
        try:
            h_count = float(len(context.hamiltonian.return_polynomial()))
        except Exception:
            h_count = None
    shots_per_term = _to_float(row.get("shots_per_pauli_term_proxy")) or 1024.0
    shots_per_energy_eval = None if h_count is None else float(shots_per_term * h_count)
    shots_total = None if shots_per_energy_eval is None else float(shots_per_energy_eval * int(nfev))
    rate_rows = []
    for rate in shot_rates:
        rate_f = float(rate)
        if not math.isfinite(rate_f) or rate_f <= 0.0:
            continue
        rate_rows.append(
            {
                "shot_rate_shots_per_s": rate_f,
                "nfev": int(nfev),
                "measurement_time_s": None if shots_total is None else float(shots_total / rate_f),
            }
        )
    return {
        "schema": "parameterized_qpu_measurement_time_proxy_v1",
        "nfev": int(nfev),
        "hamiltonian_pauli_term_count": None if h_count is None else int(round(h_count)),
        "shots_per_pauli_term_proxy": int(round(shots_per_term)),
        "shots_per_energy_eval_proxy": shots_per_energy_eval,
        "shots_total_for_nfev": shots_total,
        "shot_rate_rows": rate_rows,
        "note": "Parameterized timing proxy only; no physical QPU runtime is claimed.",
    }


def _source_backend_cost_mode(payload: Mapping[str, Any]) -> str | None:
    paths = (
        ("settings", "phase3_backend_cost_mode"),
        ("adapt_vqe", "settings", "phase3_backend_cost_mode"),
        ("result", "policy_roundtrip_audit", "emitted_options", "phase3_backend_cost_mode"),
        ("result", "policy_roundtrip_audit", "normalized_policy_summary", "phase3_backend_cost_mode"),
    )
    for path in paths:
        value = _get_path(payload, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _source_warning(source_payload: Mapping[str, Any], measurement_payload: Mapping[str, Any] | None) -> str | None:
    modes = []
    for payload in (source_payload, measurement_payload or {}):
        mode = _source_backend_cost_mode(payload)
        if mode:
            modes.append(str(mode))
    if any(mode in {"auto", "transpile_single_v1"} for mode in modes):
        return "source_artifact_backend_mode_auto_or_full_transpile"
    if not any(mode == "incremental_prefix_suffix_v1" for mode in modes):
        return "source_artifact_does_not_show_incremental_prefix_suffix_mode"
    return None


def _write_csv(path: Path, mode_results: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "sequence_index",
        "step_index",
        "candidate_label",
        "position_id",
        "base_depth",
        "snapshot_s",
        "estimate_s",
        "total_s",
        "source_mode",
        "hardware_cost_source",
        "selected_backend_name",
        "penalty_total",
        "delta_compiled_count_2q",
        "delta_compiled_depth_2q",
        "delta_compiled_depth",
        "delta_compiled_size",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in mode_results:
            for row in result.get("rows", []):
                writer.writerow({key: row.get(key) for key in fieldnames})


def _pdf_table(data: Sequence[Sequence[Any]], *, col_widths: Sequence[float] | None = None) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table([[str(cell) for cell in row] for row in data], colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8C0CC")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _write_pdf(path: Path, payload: Mapping[str, Any]) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph

    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story: list[Any] = []
    story.append(Paragraph("HH Backend Compile-Cost Timing Diagnostic", styles["Title"]))
    summary = payload["summary"]
    story.append(
        Paragraph(
            "Diagnostic only. Source is HH intermediate-strong; optimizer workload is 8000 nfev.",
            styles["BodyText"],
        )
    )
    if summary.get("source_warning"):
        story.append(Paragraph(f"Warning: {summary['source_warning']}.", styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Compile Timing", styles["Heading2"]))
    rows = [["Mode", "Records", "Snapshot total s", "Estimate total s", "Path total s", "Estimate median s", "Source modes"]]
    for mode_row in payload["mode_summaries"]:
        rows.append(
            [
                mode_row["mode"],
                mode_row["record_count"],
                _fmt(mode_row["snapshot_total_s"]),
                _fmt(mode_row["estimate_total_s"]),
                _fmt(mode_row["compile_path_total_s"]),
                _fmt(mode_row["estimate_median_s"]),
                ", ".join(mode_row.get("source_modes") or ()),
            ]
        )
    story.append(_pdf_table(rows))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Measurement Comparison", styles["Heading2"]))
    energy = payload["local_energy_objective"]
    meas = payload["measurement_proxy"]
    rows = [
        ["Quantity", "Value"],
        ["Local objective median s", _fmt(energy.get("median_s"))],
        ["Local objective x 8000 s", _fmt(energy.get("nfev_8000_median_walltime_s"))],
        ["Hamiltonian Pauli terms", meas.get("hamiltonian_pauli_term_count")],
        ["Shots per energy eval proxy", _fmt(meas.get("shots_per_energy_eval_proxy"))],
        ["Shots for 8000 nfev", _fmt(meas.get("shots_total_for_nfev"))],
    ]
    story.append(_pdf_table(rows, col_widths=[220, 240]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Parameterized QPU Shot-Time Proxy", styles["Heading2"]))
    rows = [["Shot rate shots/s", "Measurement time s", "Measurement time h"]]
    for row in meas.get("shot_rate_rows", []):
        t_s = row.get("measurement_time_s")
        rows.append([_fmt(row.get("shot_rate_shots_per_s")), _fmt(t_s), _fmt(None if t_s is None else float(t_s) / 3600.0)])
    story.append(_pdf_table(rows))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Per-Record Timing", styles["Heading2"]))
    rows = [["Mode", "Step", "Base k", "Position", "Estimate s", "Delta N2q", "Delta D2q", "Label"]]
    for mode_row in payload["mode_summaries"]:
        for row in mode_row.get("rows", [])[:24]:
            rows.append(
                [
                    row.get("mode"),
                    row.get("step_index"),
                    row.get("base_depth"),
                    row.get("position_id"),
                    _fmt(row.get("estimate_s")),
                    _fmt(row.get("delta_compiled_count_2q")),
                    _fmt(row.get("delta_compiled_depth_2q")),
                    str(row.get("candidate_label"))[:48],
                ]
            )
    story.append(_pdf_table(rows))
    doc.build(story)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _parse_shot_rates(raw: str) -> tuple[float, ...]:
    out: list[float] = []
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"invalid shot rate {token!r}")
        out.append(value)
    return tuple(out or [1_000.0, 10_000.0, 100_000.0])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--measurement-json", default=DEFAULT_MEASUREMENT_JSON)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--profile", default=TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE)
    parser.add_argument("--class-filter-json", default=DEFAULT_CLASS_FILTER_JSON)
    parser.add_argument("--backend-name", default="FakeMarrakesh")
    parser.add_argument("--seed-transpiler", type=int, default=7)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--structure-theta-value", type=float, default=1.0)
    parser.add_argument("--weight-2q", type=float, default=1.0)
    parser.add_argument("--weight-depth", type=float, default=0.1)
    parser.add_argument("--weight-size", type=float, default=0.01)
    parser.add_argument("--record-limit", type=int, default=None)
    parser.add_argument("--energy-repeats", type=int, default=5)
    parser.add_argument("--nfev", type=int, default=8000)
    parser.add_argument("--shot-rates", default="1000,10000,100000")
    parser.add_argument("--output-dir", default="output/pdf")
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--no-pdf", action="store_true")
    return parser


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("STATIC_ADAPT_HH_POOL_CACHE", "disk")
    source_path = Path(args.source_json)
    measurement_path = Path(args.measurement_json)
    if not source_path.exists():
        raise FileNotFoundError(f"source JSON not found: {source_path}")
    source_payload = _read_json(source_path)
    measurement_payload = _read_json(measurement_path) if measurement_path.exists() else None
    admissions_all = extract_admission_records(source_payload)
    if not admissions_all:
        raise ValueError(f"no admission records found in {source_path}")

    setup_t0 = time.perf_counter()
    context, pool, pool_meta, spec_meta = _build_pool(
        case_id=str(args.case_id),
        profile=str(args.profile),
        class_filter_json=Path(args.class_filter_json),
    )
    setup_s = time.perf_counter() - setup_t0
    pool_by_label = {str(term.label): term for term in pool}
    candidate_rows, selected_ops, missing_labels = _candidate_sequence(
        admissions_all,
        pool_by_label,
        record_limit=args.record_limit,
    )
    if missing_labels:
        raise ValueError(f"source admissions missing from rebuilt pool: {sorted(set(missing_labels))[:8]}")
    ref_state = _normalize_state(context.reference_state.build_state())
    theta, theta_source = _theta_from_payload(source_payload, len(selected_ops))
    mode_summaries = []
    for mode in COMPILE_MODES:
        mode_summaries.append(
            time_compile_mode(
                mode=mode,
                admissions=admissions_all,
                pool_by_label=pool_by_label,
                num_qubits=int(context.layout.total_qubits),
                ref_state=ref_state,
                backend_name=str(args.backend_name),
                seed_transpiler=int(args.seed_transpiler),
                optimization_level=int(args.optimization_level),
                structure_theta_value=float(args.structure_theta_value),
                weight_2q=float(args.weight_2q),
                weight_depth=float(args.weight_depth),
                weight_size=float(args.weight_size),
                record_limit=args.record_limit,
            )
        )
    local_energy = _time_energy_objective(
        context=context,
        selected_ops=selected_ops,
        theta=theta,
        ref_state=ref_state,
        repeats=int(args.energy_repeats),
    )
    local_energy["theta_source"] = theta_source
    if str(theta_source).startswith("zeros_fallback"):
        local_energy["theta_source_status"] = str(theta_source)
    measurement_proxy = _measurement_proxy(
        measurement_payload=measurement_payload,
        context=context,
        nfev=int(args.nfev),
        shot_rates=_parse_shot_rates(args.shot_rates),
    )
    summary = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_class": "diagnostic",
        "case_id": str(args.case_id),
        "display_regime": "intermediate-strong",
        "legacy_regime_key": "strong_strong",
        "source_warning": _source_warning(source_payload, measurement_payload),
        "nfev": int(args.nfev),
        "spsa_maxiter_8000_note": "In this repo SPSA maxiter=8000 is about 16001 objective evaluations when eval_repeats=1 and avg_last=0.",
        "setup_pool_context_s": float(setup_s),
        "record_count_total": int(len(admissions_all)),
        "record_count_used": int(len(candidate_rows)),
        "compile_modes": list(COMPILE_MODES),
    }
    return {
        "schema": SCHEMA,
        "summary": summary,
        "source": {
            "source_json": str(source_path),
            "source_sha256": _sha256(source_path),
            "source_backend_cost_mode": _source_backend_cost_mode(source_payload),
            "measurement_json": str(measurement_path) if measurement_path.exists() else None,
            "measurement_sha256": _sha256(measurement_path) if measurement_path.exists() else None,
            "measurement_backend_cost_mode": (
                _source_backend_cost_mode(measurement_payload) if measurement_payload is not None else None
            ),
        },
        "settings": {
            "case_id": str(args.case_id),
            "profile": str(args.profile),
            "class_filter_json": str(args.class_filter_json),
            "backend_name": str(args.backend_name),
            "seed_transpiler": int(args.seed_transpiler),
            "optimization_level": int(args.optimization_level),
            "structure_theta_value": float(args.structure_theta_value),
            "weights": {
                "weight_2q": float(args.weight_2q),
                "weight_depth": float(args.weight_depth),
                "weight_size": float(args.weight_size),
            },
            "record_limit": args.record_limit,
            "energy_repeats": int(args.energy_repeats),
            "STATIC_ADAPT_HH_POOL_CACHE": os.environ.get("STATIC_ADAPT_HH_POOL_CACHE"),
            "STATIC_ADAPT_HH_POOL_CACHE_DIR": os.environ.get("STATIC_ADAPT_HH_POOL_CACHE_DIR"),
        },
        "spec": spec_meta,
        "pool": pool_meta,
        "candidate_records": candidate_rows,
        "mode_summaries": mode_summaries,
        "local_energy_objective": local_energy,
        "measurement_proxy": measurement_proxy,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_diagnostic(args)
    out_dir = Path(args.output_dir)
    stem = args.output_stem or f"paper_i_hh_backend_compile_latency_{_utc_stamp()}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    pdf_path = out_dir / f"{stem}.pdf"
    _write_json(json_path, payload)
    _write_csv(csv_path, payload["mode_summaries"])
    if not args.no_pdf:
        _write_pdf(pdf_path, payload)
    payload["artifacts"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "pdf": None if args.no_pdf else str(pdf_path),
    }
    _write_json(json_path, payload)
    print(json.dumps(payload["artifacts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
