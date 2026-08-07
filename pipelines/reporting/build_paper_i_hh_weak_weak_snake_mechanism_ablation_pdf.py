#!/usr/bin/env python3
"""Build Paper-I HH weak-weak SNAKE mechanism-ablation support PDF.

This report consumes the 2026-07-08 CHTC ablation matrix plus existing local
reference anchors.  It is a support artifact for choosing plateau prefixes; it
does not edit or promote manuscript tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.snake_table_i_measurement_work import (  # noqa: E402
    snake_algorithmic_work_from_payload,
    snake_mechanism_resolved_work_from_payload,
)
from pipelines.exact_bench.table_i_qiskit_resource_compile import (  # noqa: E402
    TABLE_I_QISKIT_COMPILE_CONVENTION,
    TableICompileUnavailable,
    compile_table_i_pauli_label_groups,
)
from pipelines.reporting.build_paper_i_hh_powell_snake_batchcap3_ablation_pdf import (  # noqa: E402
    _adapt,
    _compile_prefix,
    _compile_terminal,
    _float_or_none,
    _history,
    _history_error,
    _history_points,
    _int_or_none,
    _max_batch_size,
    _num_qubits_from_groups,
    _plateau_k,
    _reference_state,
    _selected_labels,
    _selected_positions,
    _tex_escape,
)


BATCH_ID = "paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v2"
TSV_PATH = REPO_ROOT / "chtc/phase3_optuna/input" / BATCH_ID / "paper_i_hh_spsa_budget_ladder_records.tsv"
FETCH_ROOT = (
    REPO_ROOT
    / "output/chtc_retrievals/paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v2_fetch"
    / "Holstein_phase3_optuna_chtc/raw_outputs"
    / BATCH_ID
)
OUT_DIR = REPO_ROOT / "output/pdf/paper_i_hh_weak_weak_snake_mechanism_ablation_20260708"
STEM = "paper_i_hh_weak_weak_snake_mechanism_ablation_20260708"
PLATEAU_REL_TOL = 0.10


FAMILY_LABELS = {
    "batch_cap3_combinatorial": "Combinatorial cap-3 source family",
    "physical_operator_lane": "Physical-operator lane source family",
}

VARIANT_LABELS = {
    "full_anchor_reference": "Source anchor",
    "no_batching_reference": "No batching reference",
    "greedy_cap3": "Greedy batch cap 3",
    "combinatorial_cap3": "Combinatorial batch cap 3",
    "no_prune": "No prune",
    "no_cost_term": "No cost term",
    "no_novelty": "No novelty",
    "phase2_novelty_only_no_second_order": "Novelty only; no second order",
    "phase2_second_order_only_no_novelty": "Second order only; no novelty",
    "no_phase3": "No Phase 3",
    "phase1_only_macro_pool": "Phase 1 only; macro pool",
    "phase1_only_singleton_pool": "Phase 1 only; singleton pool",
    "full_geometry_window": "Full geometry window",
    "no_shortlisting": "No shortlisting",
}


@dataclass(frozen=True)
class ManifestRow:
    record_id: str
    family: str
    variant: str
    feature: str
    role: str
    runnable: bool
    blocker: str
    result_json: Path | None
    reference_json: Path | None
    changed_fields_vs_anchor: str
    overrides_json: str


@dataclass
class ReportRow:
    family: str
    family_label: str
    variant: str
    variant_label: str
    feature: str
    role: str
    status: str
    record_id: str
    result_kind: str
    source_json: str | None
    source_sha256: str | None
    k_plateau: int | None
    plateau_abs_delta_e: float | None
    plateau_d_ans: int | None
    plateau_n2q: int | None
    plateau_d2q: int | None
    plateau_dc: int | None
    plateau_s_alg: float | None
    plateau_s_grad: float | None
    plateau_s_h: float | None
    plateau_s_metric: float | None
    plateau_coarse_s_alg: float | None
    plateau_coarse_s_metric: float | None
    plateau_s_alg_source: str
    plateau_phase2_formula_reconstruction: str
    plateau_mechanism_status: str
    plateau_requires_formula_reconstruction: bool | None
    terminal_k_iter: int | None
    terminal_abs_delta_e: float | None
    terminal_one_minus_f: float | None
    terminal_d_ans: int | None
    terminal_n2q: int | None
    terminal_d2q: int | None
    terminal_dc: int | None
    terminal_s_alg: float | None
    terminal_s_grad: float | None
    terminal_s_h: float | None
    terminal_s_metric: float | None
    terminal_coarse_s_alg: float | None
    terminal_coarse_s_metric: float | None
    terminal_s_alg_source: str
    terminal_phase2_formula_reconstruction: str
    terminal_mechanism_status: str
    terminal_requires_formula_reconstruction: bool | None
    s_beam_search_total: float | None
    max_observed_batch_size: int | None
    batch_sequence: str
    plateau_rule: str
    plateau_compile_status: str
    terminal_compile_status: str
    s_alg_status_plateau: str
    s_alg_status_terminal: str
    note: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _fmt_int(value: int | float | None) -> str:
    if value is None:
        return "--"
    try:
        return str(int(round(float(value))))
    except Exception:
        return "--"


def _fmt_err(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3e}"


def _fmt_fidelity(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3e}"


def _short_hash(value: str | None) -> str:
    return "--" if not value else value[:12]


def _load_manifest() -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with TSV_PATH.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            runnable = raw.get("runnable") == "true"
            record_id = raw["record_id"]
            reference = raw.get("hh_mechanism_ablation_reference_json") or ""
            reference_json = (REPO_ROOT / reference) if reference else None
            blocker = raw.get("blocker") or raw.get("blocked_reason") or ""
            if runnable:
                result_json = FETCH_ROOT / record_id / "json/result.json"
            elif blocker.startswith("reference_existing"):
                result_json = reference_json
            else:
                result_json = None
            rows.append(
                ManifestRow(
                    record_id=record_id,
                    family=raw.get("source_anchor_family") or "unknown",
                    variant=raw.get("hh_mechanism_ablation_variant") or raw.get("route_variant") or "unknown",
                    feature=raw.get("hh_mechanism_ablation_feature") or "",
                    role=raw.get("hh_mechanism_ablation_role") or raw.get("matrix_role") or "",
                    runnable=runnable,
                    blocker=blocker,
                    result_json=result_json,
                    reference_json=reference_json,
                    changed_fields_vs_anchor=raw.get("changed_fields_vs_anchor") or "",
                    overrides_json=raw.get("hh_mechanism_ablation_overrides_json") or "",
                )
            )
    return rows


def _runtime_split_paulis(record: Mapping[str, Any]) -> list[str]:
    labels = record.get("runtime_split_child_labels")
    if not isinstance(labels, list):
        return []
    out: list[str] = []
    for label in labels:
        tail = str(label).rsplit("::", 1)[-1].strip().lower()
        if tail and set(tail) <= set("exyz") and any(char != "e" for char in tail):
            out.append(tail)
    return out


def _record_paulis(record: Mapping[str, Any]) -> list[str]:
    for key in ("pauli_labels_exyz", "pauli_labels", "pauli_strings"):
        value = record.get(key)
        if isinstance(value, list):
            clean = [str(item).strip().lower() for item in value if str(item).strip()]
            if clean:
                return clean
    terms = record.get("runtime_terms_exyz")
    if isinstance(terms, list):
        clean: list[str] = []
        for term in terms:
            if isinstance(term, Mapping):
                text = str(term.get("pauli_exyz") or "").strip().lower()
            else:
                text = str(term).strip().lower()
            if text:
                clean.append(text)
        if clean:
            return clean
    return _runtime_split_paulis(record)


def _candidate_record_lookup(row: Mapping[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in ("admitted_records", "selected_feature_rows", "top_candidates", "retained_shortlist_records", "shortlisted_records"):
        records = row.get(key)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            label = record.get("label") or record.get("candidate_label") or record.get("selected_label")
            if not isinstance(label, str):
                continue
            group = _record_paulis(record)
            if group:
                out.setdefault(label, group)
    return out


def _compile_prefix_from_runtime_split_labels(payload: Mapping[str, Any], result_path: Path, k: int) -> dict[str, Any]:
    history = _history(payload)
    if k < 1 or k > len(history):
        raise ValueError(f"prefix k={k} outside history length {len(history)}")
    reference_state, reference_status = _reference_state(payload)
    pauli_groups: list[list[str]] = []
    selected_error: float | None = None
    for idx, row in enumerate(history, start=1):
        if idx > k:
            break
        labels = _selected_labels(row)
        lookup = _candidate_record_lookup(row)
        step_groups: list[list[str]] = []
        for label in labels:
            group = lookup.get(label)
            if not group:
                raise ValueError(f"runtime-split Pauli group missing for {label}")
            step_groups.append(group)
        for group, position in zip(step_groups, _selected_positions(row, len(step_groups))):
            if position is None or position < 0 or position > len(pauli_groups):
                pauli_groups.append(group)
            else:
                pauli_groups.insert(int(position), group)
        if idx == k:
            selected_error = _history_error(row)
    num_qubits = _num_qubits_from_groups(pauli_groups, reference_state)
    try:
        compiled = compile_table_i_pauli_label_groups(
            pauli_label_groups=tuple(tuple(group) for group in pauli_groups),
            num_qubits=int(num_qubits),
            reference_state=reference_state,
            source_kind="paper_i_hh_mechanism_ablation_runtime_split_prefix",
        )
        compile_status = "ok"
        compile_error = None
    except TableICompileUnavailable as exc:
        compiled = {}
        compile_status = exc.status
        compile_error = exc.reason
    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=int(k),
        source_label=str(result_path),
    )
    return {
        "k_iter": int(k),
        "d_ans": int(len(pauli_groups)),
        "abs_delta_e": selected_error,
        "n2q": _int_or_none(compiled.get("compiled_count_2q_total")),
        "d2q": _int_or_none(compiled.get("compiled_depth_2q_total")),
        "dc": _int_or_none(compiled.get("compiled_depth_total")),
        "qiskit_cost_status": compile_status,
        "qiskit_cost_source": "qiskit_runtime_split_child_label_prefix_compile",
        "compile_error": compile_error,
        "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
        "reference_state_status": reference_status,
        "s_alg": _float_or_none(work.get("S_alg")),
        "s_alg_status": str(work.get("S_alg_status") or audit.get("status") or "unknown"),
        "s_alg_source": "snake_algorithmic_work_from_payload(scope=display_prefix)",
        "s_work_audit_status": audit.get("status"),
        "s_grad": _float_or_none(work.get("S_alg_N_grad_probe")),
        "s_metric": _float_or_none(work.get("S_alg_N_metric_probe")),
        "s_h": (_float_or_none(work.get("S_alg_N_H_refit_eval")) or 0.0)
        + (_float_or_none(work.get("S_alg_N_H_outer_eval")) or 0.0),
        "s_beam_search_total": _float_or_none(work.get("S_beam_search_total")),
    }


def _s_work_fields(
    payload: Mapping[str, Any],
    result_path: Path,
    *,
    scope: str,
    history_position: int | None = None,
) -> dict[str, Any]:
    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope=scope,
        history_position=history_position,
        source_label=str(result_path),
    )
    mechanism, mechanism_audit = snake_mechanism_resolved_work_from_payload(
        payload,
        scope=scope,
        history_position=history_position,
        source_label=str(result_path),
    )
    mechanism_work = mechanism.get("mechanism_algorithmic_work")
    mechanism_work = mechanism_work if isinstance(mechanism_work, Mapping) else {}
    mechanism_publishable = bool(mechanism_work.get("publishable"))
    mechanism_status = str(
        mechanism.get("mechanism_resolution_status")
        or mechanism_audit.get("mechanism_resolution_status")
        or mechanism.get("status")
        or "unknown"
    )
    mechanism_s_alg_status = str(mechanism_work.get("status") or work.get("S_alg_status") or audit.get("status") or "unknown")
    if mechanism_publishable:
        visible_s_alg = _float_or_none(mechanism_work.get("S_alg"))
        visible_s_grad = _float_or_none(mechanism_work.get("S_alg_N_grad_probe"))
        visible_s_metric = _float_or_none(mechanism_work.get("S_alg_N_metric_probe"))
        h_refit = _float_or_none(mechanism_work.get("S_alg_N_H_refit_eval")) or 0.0
        h_outer = _float_or_none(mechanism_work.get("S_alg_N_H_outer_eval")) or 0.0
        visible_s_h = h_refit + h_outer
        s_alg_source = f"snake_mechanism_resolved_work_from_payload(scope={scope})"
    else:
        visible_s_alg = None
        visible_s_grad = None
        visible_s_metric = None
        visible_s_h = None
        s_alg_source = f"withheld:{mechanism_s_alg_status}"
    return {
        "s_alg": visible_s_alg,
        "s_alg_status": mechanism_s_alg_status,
        "s_alg_source": s_alg_source,
        "s_work_audit_status": audit.get("status"),
        "s_grad": visible_s_grad,
        "s_metric": visible_s_metric,
        "s_h": visible_s_h,
        "coarse_s_alg": _float_or_none(work.get("S_alg")),
        "coarse_s_grad": _float_or_none(work.get("S_alg_N_grad_probe")),
        "coarse_s_metric": _float_or_none(work.get("S_alg_N_metric_probe")),
        "coarse_s_h": (_float_or_none(work.get("S_alg_N_H_refit_eval")) or 0.0)
        + (_float_or_none(work.get("S_alg_N_H_outer_eval")) or 0.0),
        "mechanism_s_alg_publishable": mechanism_publishable,
        "phase2_formula_reconstruction": json.dumps(
            mechanism_work.get("phase2_formula_reconstruction") or {},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "s_beam_search_total": _float_or_none(work.get("S_beam_search_total")),
        "mechanism_status": mechanism_status,
        "requires_formula_reconstruction": bool(mechanism.get("requires_formula_reconstruction"))
        if mechanism.get("requires_formula_reconstruction") is not None
        else None,
        "mechanism_reconstruction_status": str(mechanism.get("status") or mechanism_audit.get("status") or "unknown"),
        "mechanism_resolution_detail": mechanism.get("mechanism_resolution_detail"),
    }


def _apply_s_work_fields(
    metrics: dict[str, Any],
    payload: Mapping[str, Any],
    result_path: Path,
    *,
    scope: str,
    history_position: int | None = None,
) -> dict[str, Any]:
    updated = dict(metrics)
    updated.update(_s_work_fields(payload, result_path, scope=scope, history_position=history_position))
    return updated


def _compile_prefix_safe(payload: Mapping[str, Any], result_path: Path, k: int) -> dict[str, Any]:
    try:
        metrics = _compile_prefix(payload, result_path, k)
        metrics["prefix_compile_fallback"] = "standard"
    except Exception as exc:
        metrics = _compile_prefix_from_runtime_split_labels(payload, result_path, k)
        metrics["prefix_compile_fallback"] = f"runtime_split_child_labels_after:{type(exc).__name__}"
    return _apply_s_work_fields(metrics, payload, result_path, scope="display_prefix", history_position=int(k))


def _row_from_payload(row: ManifestRow, payload: Mapping[str, Any], result_kind: str) -> ReportRow:
    points = _history_points(payload)
    k_plateau, plateau_meta = _plateau_k(points)
    max_batch, batch_sequence = _max_batch_size(payload)
    if k_plateau is None:
        plateau = {}
        plateau_status = "missing_positive_trajectory"
        plateau_note = json.dumps(plateau_meta, sort_keys=True)
    else:
        try:
            plateau = _compile_prefix_safe(payload, row.result_json or Path("<unknown>"), int(k_plateau))
            plateau_status = str(plateau.get("qiskit_cost_status") or "unknown")
            plateau_note = json.dumps(
                {
                    "plateau": plateau_meta,
                    "prefix_compile_fallback": plateau.get("prefix_compile_fallback"),
                },
                sort_keys=True,
            )
        except Exception as exc:
            plateau = {
                "k_iter": k_plateau,
                "abs_delta_e": dict(points).get(int(k_plateau)),
                "qiskit_cost_status": f"blocked:{type(exc).__name__}",
                "s_alg_status": "blocked",
            }
            plateau_status = str(plateau["qiskit_cost_status"])
            plateau_note = f"plateau_prefix_compile_failed:{exc}"
    try:
        terminal = _compile_terminal(payload, row.result_json or Path("<unknown>"))
        terminal = _apply_s_work_fields(
            terminal,
            payload,
            row.result_json or Path("<unknown>"),
            scope="terminal",
        )
    except Exception as exc:
        adapt = _adapt(payload)
        terminal = {
            "k_iter": len(_history(payload)),
            "abs_delta_e": _float_or_none(adapt.get("abs_delta_e")),
            "d_ans": _int_or_none(adapt.get("ansatz_depth")),
            "qiskit_cost_status": f"blocked:{type(exc).__name__}",
            "s_alg_status": "blocked",
        }
    return ReportRow(
        family=row.family,
        family_label=FAMILY_LABELS.get(row.family, row.family),
        variant=row.variant,
        variant_label=VARIANT_LABELS.get(row.variant, row.variant.replace("_", " ")),
        feature=row.feature,
        role=row.role,
        status="done" if result_kind == "chtc_result" else "reference",
        record_id=row.record_id,
        result_kind=result_kind,
        source_json=_rel(row.result_json),
        source_sha256=_sha256(row.result_json),
        k_plateau=_int_or_none(plateau.get("k_iter")),
        plateau_abs_delta_e=_float_or_none(plateau.get("abs_delta_e")),
        plateau_d_ans=_int_or_none(plateau.get("d_ans")),
        plateau_n2q=_int_or_none(plateau.get("n2q")),
        plateau_d2q=_int_or_none(plateau.get("d2q")),
        plateau_dc=_int_or_none(plateau.get("dc")),
        plateau_s_alg=_float_or_none(plateau.get("s_alg")),
        plateau_s_grad=_float_or_none(plateau.get("s_grad")),
        plateau_s_h=_float_or_none(plateau.get("s_h")),
        plateau_s_metric=_float_or_none(plateau.get("s_metric")),
        plateau_coarse_s_alg=_float_or_none(plateau.get("coarse_s_alg")),
        plateau_coarse_s_metric=_float_or_none(plateau.get("coarse_s_metric")),
        plateau_s_alg_source=str(plateau.get("s_alg_source") or "unknown"),
        plateau_phase2_formula_reconstruction=str(plateau.get("phase2_formula_reconstruction") or ""),
        plateau_mechanism_status=str(plateau.get("mechanism_status") or "unknown"),
        plateau_requires_formula_reconstruction=plateau.get("requires_formula_reconstruction"),
        terminal_k_iter=_int_or_none(terminal.get("k_iter")),
        terminal_abs_delta_e=_float_or_none(terminal.get("abs_delta_e")),
        terminal_one_minus_f=_float_or_none(terminal.get("one_minus_f")),
        terminal_d_ans=_int_or_none(terminal.get("d_ans")),
        terminal_n2q=_int_or_none(terminal.get("n2q")),
        terminal_d2q=_int_or_none(terminal.get("d2q")),
        terminal_dc=_int_or_none(terminal.get("dc")),
        terminal_s_alg=_float_or_none(terminal.get("s_alg")),
        terminal_s_grad=_float_or_none(terminal.get("s_grad")),
        terminal_s_h=_float_or_none(terminal.get("s_h")),
        terminal_s_metric=_float_or_none(terminal.get("s_metric")),
        terminal_coarse_s_alg=_float_or_none(terminal.get("coarse_s_alg")),
        terminal_coarse_s_metric=_float_or_none(terminal.get("coarse_s_metric")),
        terminal_s_alg_source=str(terminal.get("s_alg_source") or "unknown"),
        terminal_phase2_formula_reconstruction=str(terminal.get("phase2_formula_reconstruction") or ""),
        terminal_mechanism_status=str(terminal.get("mechanism_status") or "unknown"),
        terminal_requires_formula_reconstruction=terminal.get("requires_formula_reconstruction"),
        s_beam_search_total=_float_or_none(terminal.get("s_beam_search_total")) or _float_or_none(plateau.get("s_beam_search_total")),
        max_observed_batch_size=max_batch,
        batch_sequence=batch_sequence,
        plateau_rule=json.dumps(plateau_meta, sort_keys=True),
        plateau_compile_status=plateau_status,
        terminal_compile_status=str(terminal.get("qiskit_cost_status") or "unknown"),
        s_alg_status_plateau=str(plateau.get("s_alg_status") or "unknown"),
        s_alg_status_terminal=str(terminal.get("s_alg_status") or "unknown"),
        note=plateau_note,
    )


def _blocked_row(row: ManifestRow, status: str, note: str) -> ReportRow:
    return ReportRow(
        family=row.family,
        family_label=FAMILY_LABELS.get(row.family, row.family),
        variant=row.variant,
        variant_label=VARIANT_LABELS.get(row.variant, row.variant.replace("_", " ")),
        feature=row.feature,
        role=row.role,
        status=status,
        record_id=row.record_id,
        result_kind="blocked",
        source_json=_rel(row.reference_json),
        source_sha256=_sha256(row.reference_json),
        k_plateau=None,
        plateau_abs_delta_e=None,
        plateau_d_ans=None,
        plateau_n2q=None,
        plateau_d2q=None,
        plateau_dc=None,
        plateau_s_alg=None,
        plateau_s_grad=None,
        plateau_s_h=None,
        plateau_s_metric=None,
        plateau_coarse_s_alg=None,
        plateau_coarse_s_metric=None,
        plateau_s_alg_source="not_applicable",
        plateau_phase2_formula_reconstruction="",
        plateau_mechanism_status="not_applicable",
        plateau_requires_formula_reconstruction=None,
        terminal_k_iter=None,
        terminal_abs_delta_e=None,
        terminal_one_minus_f=None,
        terminal_d_ans=None,
        terminal_n2q=None,
        terminal_d2q=None,
        terminal_dc=None,
        terminal_s_alg=None,
        terminal_s_grad=None,
        terminal_s_h=None,
        terminal_s_metric=None,
        terminal_coarse_s_alg=None,
        terminal_coarse_s_metric=None,
        terminal_s_alg_source="not_applicable",
        terminal_phase2_formula_reconstruction="",
        terminal_mechanism_status="not_applicable",
        terminal_requires_formula_reconstruction=None,
        s_beam_search_total=None,
        max_observed_batch_size=None,
        batch_sequence="",
        plateau_rule="",
        plateau_compile_status="not_applicable",
        terminal_compile_status="not_applicable",
        s_alg_status_plateau="not_applicable",
        s_alg_status_terminal="not_applicable",
        note=note,
    )


def _load_rows() -> list[ReportRow]:
    rows: list[ReportRow] = []
    for manifest_row in _load_manifest():
        if manifest_row.result_json is not None and manifest_row.result_json.exists():
            payload = _read_json(manifest_row.result_json)
            result_kind = "chtc_result" if manifest_row.runnable else "reference_existing"
            rows.append(_row_from_payload(manifest_row, payload, result_kind))
        elif manifest_row.runnable:
            rows.append(_blocked_row(manifest_row, "missing", "runnable row has no fetched result JSON"))
        else:
            rows.append(_blocked_row(manifest_row, "blocked", manifest_row.blocker))
    return rows


def _write_csv(rows: Sequence[ReportRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _color_for(index: int) -> str:
    palette = plt.get_cmap("tab20").colors
    rgb = palette[index % len(palette)]
    return "#{:02x}{:02x}{:02x}".format(*(int(255 * channel) for channel in rgb[:3]))


def _plot_overlay(rows: Sequence[ReportRow], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    plotted = 0
    for index, row in enumerate([r for r in rows if r.source_json and r.status in {"done", "reference"}]):
        payload = _read_json(REPO_ROOT / row.source_json)
        points = _history_points(payload)
        if not points:
            continue
        xs, ys = zip(*points)
        color = _color_for(index)
        ax.plot(xs, ys, color=color, linewidth=1.35, alpha=0.90, label=row.variant_label)
        if row.k_plateau is not None and row.plateau_abs_delta_e is not None:
            ax.scatter([row.k_plateau], [row.plateau_abs_delta_e], color=color, marker="o", s=30, zorder=5)
        plotted += 1
    ax.set_yscale("log")
    ax.set_xlabel("ADAPT iteration k")
    ax.set_ylabel(r"$|\Delta E|$")
    ax.grid(True, which="both", alpha=0.22)
    ax.set_title(rows[0].family_label if rows else "Ablation family")
    if plotted:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.5, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_small_multiples(rows: Sequence[ReportRow], out_path: Path) -> None:
    data_rows = [r for r in rows if r.source_json and r.status in {"done", "reference"}]
    ncols = 2
    nrows = max(1, math.ceil(len(data_rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.2, max(1.95 * nrows, 3.0)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for index, row in enumerate(data_rows):
        ax = axes.flat[index]
        ax.axis("on")
        payload = _read_json(REPO_ROOT / row.source_json)
        points = _history_points(payload)
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=_color_for(index), linewidth=1.3)
            if row.k_plateau is not None and row.plateau_abs_delta_e is not None:
                ax.scatter([row.k_plateau], [row.plateau_abs_delta_e], color="black", s=22, zorder=5)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.18)
        ax.set_title(f"{row.variant_label} (k_pl={_fmt_int(row.k_plateau)})", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
    fig.suptitle(rows[0].family_label if rows else "Ablation family", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _summary_table(rows: Sequence[ReportRow], *, terminal: bool) -> list[str]:
    title = "Terminal-depth metrics" if terminal else "Plateau-prefix metrics"
    lines = [
        rf"\subsection*{{{title}}}",
        r"\scriptsize",
        r"\begin{longtable}{lrrrrrrr}",
        r"\toprule",
        r"Variant & $k$ & $|\Delta E|$ & $d_{\rm ans}$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Variant & $k$ & $|\Delta E|$ & $d_{\rm ans}$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        if terminal:
            cells = [
                row.variant_label,
                _fmt_int(row.terminal_k_iter),
                _fmt_err(row.terminal_abs_delta_e),
                _fmt_int(row.terminal_d_ans),
                _fmt_int(row.terminal_n2q),
                _fmt_int(row.terminal_d2q),
                _fmt_int(row.terminal_dc),
                _fmt_int(row.terminal_s_alg),
            ]
        else:
            cells = [
                row.variant_label,
                _fmt_int(row.k_plateau),
                _fmt_err(row.plateau_abs_delta_e),
                _fmt_int(row.plateau_d_ans),
                _fmt_int(row.plateau_n2q),
                _fmt_int(row.plateau_d2q),
                _fmt_int(row.plateau_dc),
                _fmt_int(row.plateau_s_alg),
            ]
        lines.append(" & ".join(_tex_escape(cell) for cell in cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    return lines


def _audit_table(rows: Sequence[ReportRow]) -> list[str]:
    lines = [
        r"\subsection*{Provenance audit}",
        r"\scriptsize",
        r"\begin{longtable}{lllll}",
        r"\toprule",
        r"Variant & Result & Plateau compile & Terminal compile & Source hash \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Variant & Result & Plateau compile & Terminal compile & Source hash \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        cells = [
            row.variant_label,
            row.result_kind,
            row.plateau_compile_status,
            row.terminal_compile_status,
            _short_hash(row.source_sha256),
        ]
        lines.append(" & ".join(_tex_escape(cell) for cell in cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    return lines


def _tex_document(rows: Sequence[ReportRow], generated_utc: str, figures: Mapping[str, Mapping[str, Any]]) -> str:
    done = sum(1 for row in rows if row.status in {"done", "reference"})
    blocked = sum(1 for row in rows if row.status == "blocked")
    missing = sum(1 for row in rows if row.status == "missing")
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.55in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
        r"\title{Paper-I HH Weak-Weak SNAKE Mechanism Ablation Matrix}",
        r"\author{}",
        r"\date{}",
        r"\maketitle",
        r"\section*{Manifest}",
        r"\scriptsize",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Field & Value \\",
        r"\midrule",
    ]
    manifest = [
        ("Report", STEM),
        ("Generated UTC", generated_utc),
        ("Result source", f"CHTC v2 fetch root for {BATCH_ID}; see JSON sidecar"),
        ("Record manifest", "chtc/phase3_optuna/input/.../paper_i_hh_spsa_budget_ladder_records.tsv"),
        ("Rows", f"{len(rows)} total; {done} plotted; {blocked} blocked; {missing} missing"),
        ("Regime", "weak-weak Hubbard-Holstein"),
        ("Optimizer", "POWELL; maxiter 200; final/refit maxiter 200"),
        ("Pool", "Paper-I visible POWELL full_meta/HVA-included source contract"),
        ("Primary anchors", "combinatorial batch cap 3 and physical-operator lane"),
        ("Plateau marker", "first prefix within 10 percent of that row's best trajectory error"),
        ("Cost convention", "Qiskit-compiled N2q/D2q/Dc; S_alg is exact mechanism-resolved winner-lineage work when available"),
        ("S-work convention", "S_alg is withheld as -- when Phase-II mechanism work requires formula reconstruction; coarse branch-local totals remain in CSV/JSON sidecars"),
    ]
    for key, value in manifest:
        lines.append(f"{_tex_escape(key)} & {_tex_escape(value)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\clearpage"])

    for family in FAMILY_LABELS:
        family_rows = [row for row in rows if row.family == family]
        if not family_rows:
            continue
        lines.extend(
            [
                rf"\section*{{{_tex_escape(FAMILY_LABELS[family])}}}",
                r"\subsection*{Overlay trajectory}",
                rf"\includegraphics[width=\linewidth]{{{figures[family]['overlay']}}}",
                r"\clearpage",
            ]
        )
        for idx, small_fig in enumerate(figures[family]["small"], start=1):
            lines.extend(
                [
                    rf"\subsection*{{Per-row trajectories {idx}}}",
                    rf"\includegraphics[width=\linewidth]{{{small_fig}}}",
                    r"\clearpage",
                ]
            )
        lines.extend(
            [
                *_summary_table(family_rows, terminal=False),
                r"\clearpage",
                *_summary_table(family_rows, terminal=True),
                r"\clearpage",
                *_audit_table(family_rows),
                r"\clearpage",
            ]
        )
    lines.extend(
        [
            r"\section*{Notes}",
            r"\small",
            r"The no-shortlisting stress row is omitted from this human-facing matrix because that route has not been audited to open all relevant shortlist, controller-cap, and lane gates. "
            r"Plateau markers are decision aids, not manuscript commitments; use the trajectories and sidecar metrics to choose final reported prefixes.",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def build() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUT_DIR / "figures"
    rows = [row for row in _load_rows() if row.variant != "no_shortlisting"]
    generated_utc = datetime.now(timezone.utc).isoformat()
    figures: dict[str, dict[str, Any]] = {}
    for family in FAMILY_LABELS:
        family_rows = [row for row in rows if row.family == family]
        if not family_rows:
            continue
        overlay = figures_dir / f"{family}_overlay.png"
        small_figs: list[str] = []
        _plot_overlay(family_rows, overlay)
        plottable = [row for row in family_rows if row.source_json and row.status in {"done", "reference"}]
        for chunk_index in range(0, len(plottable), 6):
            chunk = plottable[chunk_index : chunk_index + 6]
            small = figures_dir / f"{family}_small_multiples_{chunk_index // 6 + 1}.png"
            _plot_small_multiples(chunk, small)
            small_figs.append(str(small.relative_to(OUT_DIR)))
        figures[family] = {
            "overlay": str(overlay.relative_to(OUT_DIR)),
            "small": small_figs,
        }

    csv_path = OUT_DIR / f"{STEM}.csv"
    json_path = OUT_DIR / f"{STEM}.json"
    tex_path = OUT_DIR / f"{STEM}.tex"
    pdf_path = OUT_DIR / f"{STEM}.pdf"
    _write_csv(rows, csv_path)
    payload = {
        "schema": "paper_i_hh_weak_weak_snake_mechanism_ablation_report_v1",
        "generated_utc": generated_utc,
        "batch_id": BATCH_ID,
        "fetch_root": str(FETCH_ROOT.relative_to(REPO_ROOT)),
        "record_manifest": str(TSV_PATH.relative_to(REPO_ROOT)),
        "report_pdf": str(pdf_path.relative_to(REPO_ROOT)),
        "report_tex": str(tex_path.relative_to(REPO_ROOT)),
        "report_csv": str(csv_path.relative_to(REPO_ROOT)),
        "figures": figures,
        "plateau_rel_tol": PLATEAU_REL_TOL,
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tex_path.write_text(_tex_document(rows, generated_utc, figures), encoding="utf-8")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=OUT_DIR,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return payload


def main() -> None:
    payload = build()
    print(payload["report_pdf"])


if __name__ == "__main__":
    main()
