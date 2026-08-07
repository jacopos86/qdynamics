#!/usr/bin/env python3
"""Build the Paper-I HH physical-operator-lane comparison report.

This support artifact compares the local physical-operator-lane SNAKE rerun
against the duplicate no-batch Paper-I PDF source.  It intentionally consumes
the machine-readable block embedded in the duplicate PDF's LaTeX source rather
than reinterpreting the baseline rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.snake_table_i_measurement_work import (  # noqa: E402
    snake_algorithmic_work_from_payload,
)
from pipelines.exact_bench.table_i_qiskit_resource_compile import (  # noqa: E402
    TABLE_I_QISKIT_COMPILE_CONVENTION,
    TableICompileUnavailable,
    compile_table_i_pauli_label_groups,
)
from pipelines.reporting.build_paper_i_hh_powell_snake_batchcap3_ablation_pdf import (  # noqa: E402
    PLATEAU_REL_TOL,
    _fallback_step_groups,
    _float_or_none,
    _history,
    _history_points,
    _int_or_none,
    _num_qubits_from_groups,
    _plateau_k,
    _reference_state,
    _selected_labels,
    _selected_positions,
)


RUN_ROOT = REPO_ROOT / "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708"
RUN_STATUS = RUN_ROOT / "run_status.json"
SOURCE_LOCK = RUN_ROOT / "source_lock_manifest.json"
COMMANDS_JSON = RUN_ROOT / "commands.json"
SOURCE_TEX = REPO_ROOT / "MATH/paper_details/static_adapt_paper_I_snake_nobatch_promoted_20260707.tex"
SOURCE_PDF = REPO_ROOT / "MATH/paper_details/static_adapt_paper_I_snake_nobatch_promoted_20260707.pdf"
SOURCE_BLOCK = "BEGIN_MACHINE_READABLE_HH_SNAKE_NOBATCH_DUPLICATE_PROMOTION_20260707"
OUT_DIR = REPO_ROOT / "output/pdf/paper_i_hh_physical_operator_lane_comparison_20260708"
STEM = "paper_i_hh_physical_operator_lane_comparison_20260708"

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)

PHYSICAL_LANE_K_PL_OVERRIDES = {
    "weak-weak": 13,
    "intermediate-weak": 10,
    "strong-weak": 11,
    "strong-strong": 13,
}


@dataclass
class ReportRow:
    regime: str
    row_source: str
    route_label: str
    status: str
    k_pl: int | None
    d_pl: int | None
    abs_delta_e: float | None
    one_minus_f: float | None
    n2q: int | None
    d2q: int | None
    dc: int | None
    s_alg: float | None
    s_grad: float | None
    s_h: float | None
    s_metric: float | None
    s_beam_search_total: float | None
    qiskit_cost_source: str
    qiskit_cost_status: str
    qiskit_compile_convention: str
    s_alg_source: str
    s_alg_status: str
    s_alg_policy: str
    plateau_rule: str
    plateau_best_error: float | None
    plateau_threshold: float | None
    plateau_selected_error: float | None
    selected_pauli_source: str
    reference_state_status: str
    source_json: str
    source_json_sha256: str | None
    source_pdf: str
    source_pdf_sha256: str | None
    source_tex: str
    source_tex_sha256: str | None
    run_status_json: str
    run_status_json_sha256: str | None
    source_lock_manifest: str
    source_lock_manifest_sha256: str | None
    commands_json: str
    commands_json_sha256: str | None
    baseline_result_json: str
    baseline_result_sha256: str | None
    local_git_branch: str
    local_git_head: str
    static_lane_route: str
    route_variant_id: str
    physical_lane_shortlist_aggressiveness: float | None
    phase1_shortlist_size_source: int | None
    phase1_shortlist_size_effective: int | None
    phase2_shortlist_size_source: int | None
    phase2_shortlist_size_effective: int | None
    phase2_shortlist_fraction_source: float | None
    phase2_shortlist_fraction_effective: float | None
    delta_abs_delta_e_vs_baseline: float | None
    ratio_abs_delta_e_vs_baseline: float | None
    delta_n2q_vs_baseline: int | None
    ratio_n2q_vs_baseline: float | None
    delta_d2q_vs_baseline: int | None
    ratio_d2q_vs_baseline: float | None
    delta_dc_vs_baseline: int | None
    ratio_dc_vs_baseline: float | None
    delta_s_alg_vs_baseline: float | None
    ratio_s_alg_vs_baseline: float | None
    note: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _git_value(args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def _tex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _tex_manifest_value(value: Any) -> str:
    text = str(value)
    if " " not in text and ("/" in text or "_" in text) and "{" not in text and "}" not in text:
        delimiter = "|"
        if delimiter in text:
            delimiter = "!"
        return rf"\path{delimiter}{text}{delimiter}"
    return _tex_escape(text)


def _fmt_err(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3e}"


def _fmt_int(value: int | float | None) -> str:
    if value is None:
        return "--"
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return "--"


def _fmt_ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.3g}"


def _safe_ratio(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None:
        return None
    try:
        den_f = float(den)
        if den_f == 0.0:
            return None
        out = float(num) / den_f
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _safe_delta_int(num: int | None, den: int | None) -> int | None:
    if num is None or den is None:
        return None
    return int(num) - int(den)


def _safe_delta_float(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    out = float(num) - float(den)
    return out if math.isfinite(out) else None


def _pauli_group_from_record(record: Mapping[str, Any]) -> list[str]:
    for key in ("pauli_labels_exyz", "pauli_labels", "pauli_strings"):
        value = record.get(key)
        if isinstance(value, list):
            out = [str(item).strip().lower() for item in value if str(item).strip()]
            if out:
                return out
    metadata = record.get("generator_metadata")
    if isinstance(metadata, Mapping):
        compile_metadata = metadata.get("compile_metadata")
        if isinstance(compile_metadata, Mapping):
            for key in ("serialized_terms_exyz", "runtime_terms_exyz"):
                value = compile_metadata.get(key)
                if isinstance(value, list):
                    out: list[str] = []
                    for item in value:
                        if isinstance(item, Mapping):
                            label = item.get("pauli_exyz") or item.get("pauli_label") or item.get("pauli")
                        else:
                            label = item
                        text = str(label).strip().lower()
                        if text:
                            out.append(text)
                    if out:
                        return out
    return []


def _pauli_map_for_physical_lane(row: Mapping[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in ("top_candidates", "admitted_records", "selected_feature_rows"):
        records = row.get(key)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            label = record.get("label") or record.get("candidate_label") or record.get("selected_label")
            group = _pauli_group_from_record(record)
            if isinstance(label, str) and label and group:
                out[label] = group
    return out


def _compile_prefix_for_report(payload: Mapping[str, Any], result_path: Path, k: int) -> dict[str, Any]:
    history = _history(payload)
    if k < 1 or k > len(history):
        raise ValueError(f"prefix k={k} outside history length {len(history)}")
    fallback_groups, fallback_source = _fallback_step_groups(payload, history, max_prefix=k)
    reference_state, reference_status = _reference_state(payload)
    pauli_groups: list[list[str]] = []
    selected_error: float | None = None
    selected_labels: list[str] = []
    selected_pauli_source = ""
    for idx, row in enumerate(history, start=1):
        if idx > k:
            break
        labels = _selected_labels(row)
        lookup = _pauli_map_for_physical_lane(row)
        if labels and all(lookup.get(label) for label in labels):
            step_groups = [list(lookup[label]) for label in labels]
            step_source = "history_selected_candidate_generator_metadata_serialized_terms_exyz"
        elif fallback_groups is not None and idx <= len(fallback_groups):
            step_groups = [list(group) for group in fallback_groups[idx - 1]]
            step_source = str(fallback_source)
        else:
            missing = [label for label in labels if label not in lookup]
            raise ValueError(f"missing Pauli group at prefix {idx}: {missing or labels}")
        for group, position in zip(step_groups, _selected_positions(row, len(step_groups))):
            if position is None or position < 0 or position > len(pauli_groups):
                pauli_groups.append(group)
            else:
                pauli_groups.insert(int(position), group)
        if idx == k:
            from pipelines.reporting.build_paper_i_hh_powell_snake_batchcap3_ablation_pdf import _history_error

            selected_error = _history_error(row)
            selected_labels = labels
            selected_pauli_source = step_source
    num_qubits = _num_qubits_from_groups(pauli_groups, reference_state)
    try:
        compiled = compile_table_i_pauli_label_groups(
            pauli_label_groups=tuple(tuple(group) for group in pauli_groups),
            num_qubits=int(num_qubits),
            reference_state=reference_state,
            source_kind="paper_i_hh_physical_operator_lane_selected_snake_prefix",
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
        "qiskit_cost_source": "qiskit_selected_snake_history_prefix_compile",
        "compile_error": compile_error,
        "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
        "selected_labels": selected_labels,
        "selected_pauli_source": selected_pauli_source,
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
        "s_beam_search_total_status": work.get("S_beam_search_total_status"),
        "s_alg_work_scope": work.get("S_alg_work_scope"),
        "s_alg_row_policy": work.get("S_alg_row_policy"),
    }


def _iter_percent_json_lines(path: Path, marker: str) -> Iterable[str]:
    in_block = False
    end_marker = marker.replace("BEGIN_", "END_", 1)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == f"% {marker}" or line == f"%{marker}":
            in_block = True
            continue
        if in_block and (line == f"% {end_marker}" or line == f"%{end_marker}"):
            return
        if not in_block:
            continue
        if raw.startswith("%"):
            stripped = raw[1:]
            if stripped.startswith(" "):
                stripped = stripped[1:]
            yield stripped
        else:
            yield raw
    raise ValueError(f"machine-readable block not closed: {marker}")


def _source_manifest() -> Mapping[str, Any]:
    text = "\n".join(_iter_percent_json_lines(SOURCE_TEX, SOURCE_BLOCK))
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{SOURCE_BLOCK} did not parse to an object")
    return payload


def _run_rows(*, require_complete: bool) -> list[Mapping[str, Any]]:
    status = _read_json(RUN_STATUS)
    rows = status.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{RUN_STATUS} missing rows[]")
    complete: list[Mapping[str, Any]] = []
    incomplete: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        regime = str(row.get("regime") or "")
        result_json = REPO_ROOT / str(row.get("output_json") or "")
        if row.get("returncode") != 0 or not result_json.exists():
            incomplete.append(regime or str(row.get("index") or "unknown"))
            continue
        complete.append(row)
    if require_complete and incomplete:
        raise RuntimeError(f"incomplete run rows: {', '.join(incomplete)}")
    return complete


def _row_order(row: ReportRow) -> tuple[int, int]:
    try:
        regime_idx = REGIME_ORDER.index(row.regime)
    except ValueError:
        regime_idx = len(REGIME_ORDER)
    source_idx = 0 if row.row_source == "paper_pdf_baseline" else 1
    return regime_idx, source_idx


def _source_lock_by_regime() -> dict[str, Mapping[str, Any]]:
    payload = _read_json(SOURCE_LOCK)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}
    return {str(row.get("regime")): row for row in rows if isinstance(row, Mapping)}


def _changed_value(row: Mapping[str, Any], key: str, side: str) -> Any:
    changed = row.get("settings_changed")
    if not isinstance(changed, Mapping):
        return None
    item = changed.get(key)
    if not isinstance(item, Mapping):
        return None
    return item.get(side)


def _baseline_rows(
    manifest: Mapping[str, Any],
    source_lock_rows: Mapping[str, Mapping[str, Any]],
    common: Mapping[str, Any],
) -> list[ReportRow]:
    cells = manifest.get("changed_snake_cells")
    if not isinstance(cells, Mapping):
        raise TypeError("source manifest missing changed_snake_cells")
    rows: list[ReportRow] = []
    for regime in REGIME_ORDER:
        cell = cells.get(regime)
        if not isinstance(cell, Mapping):
            continue
        lock_row = source_lock_rows.get(regime, {})
        rows.append(
            ReportRow(
                regime=regime,
                row_source="paper_pdf_baseline",
                route_label="PDF no-batch SNAKE",
                status="source_pdf_recorded",
                k_pl=_int_or_none(cell.get("k_pl")),
                d_pl=_int_or_none(cell.get("d_pl")),
                abs_delta_e=_float_or_none(cell.get("abs_delta_e")),
                one_minus_f=_float_or_none(cell.get("one_minus_f")),
                n2q=_int_or_none(cell.get("N2q")),
                d2q=_int_or_none(cell.get("D2q")),
                dc=_int_or_none(cell.get("Dc")),
                s_alg=_float_or_none(cell.get("S_alg")),
                s_grad=None,
                s_h=None,
                s_metric=None,
                s_beam_search_total=None,
                qiskit_cost_source=str(cell.get("qiskit_cost_source") or ""),
                qiskit_cost_status="source_pdf_recorded",
                qiskit_compile_convention=str(manifest.get("qiskit_compile_convention") or ""),
                s_alg_source=str(cell.get("S_alg_source") or ""),
                s_alg_status="source_pdf_recorded",
                s_alg_policy=str(manifest.get("s_alg_policy") or ""),
                plateau_rule="source PDF recorded k_pl from duplicate-promotion block",
                plateau_best_error=None,
                plateau_threshold=None,
                plateau_selected_error=_float_or_none(cell.get("abs_delta_e")),
                selected_pauli_source="source PDF recorded prefix",
                reference_state_status=str(cell.get("fidelity_status") or ""),
                source_json=str(cell.get("result_json") or ""),
                source_json_sha256=str(cell.get("result_sha256") or "") or None,
                baseline_result_json=str(cell.get("result_json") or ""),
                baseline_result_sha256=str(cell.get("result_sha256") or "") or None,
                static_lane_route=str(_changed_value(lock_row, "static_lane_route", "source") or "algebraic_lane_shortlist"),
                route_variant_id="source_pdf_route",
                physical_lane_shortlist_aggressiveness=None,
                phase1_shortlist_size_source=_int_or_none(_changed_value(lock_row, "phase1_shortlist_size_effective", "source")),
                phase1_shortlist_size_effective=_int_or_none(_changed_value(lock_row, "phase1_shortlist_size_effective", "source")),
                phase2_shortlist_size_source=_int_or_none(_changed_value(lock_row, "phase2_shortlist_size_effective", "source")),
                phase2_shortlist_size_effective=_int_or_none(_changed_value(lock_row, "phase2_shortlist_size_effective", "source")),
                phase2_shortlist_fraction_source=_float_or_none(_changed_value(lock_row, "phase2_shortlist_fraction_effective", "source")),
                phase2_shortlist_fraction_effective=_float_or_none(_changed_value(lock_row, "phase2_shortlist_fraction_effective", "source")),
                delta_abs_delta_e_vs_baseline=None,
                ratio_abs_delta_e_vs_baseline=None,
                delta_n2q_vs_baseline=None,
                ratio_n2q_vs_baseline=None,
                delta_d2q_vs_baseline=None,
                ratio_d2q_vs_baseline=None,
                delta_dc_vs_baseline=None,
                ratio_dc_vs_baseline=None,
                delta_s_alg_vs_baseline=None,
                ratio_s_alg_vs_baseline=None,
                note="baseline metrics read from SOURCE_BLOCK in the PDF LaTeX source",
                **common,
            )
        )
    return rows


def _candidate_rows(
    run_rows: Sequence[Mapping[str, Any]],
    baseline_by_regime: Mapping[str, ReportRow],
    source_lock_rows: Mapping[str, Mapping[str, Any]],
    common: Mapping[str, Any],
) -> list[ReportRow]:
    rows: list[ReportRow] = []
    for run_row in run_rows:
        regime = str(run_row.get("regime") or "")
        result_path = REPO_ROOT / str(run_row.get("output_json") or "")
        payload = _read_json(result_path)
        points = _history_points(payload)
        auto_k_pl, plateau_meta = _plateau_k(points)
        if auto_k_pl is None:
            raise RuntimeError(f"{regime}: cannot locate first plateau prefix")
        k_pl = PHYSICAL_LANE_K_PL_OVERRIDES.get(regime, auto_k_pl)
        point_by_k = {int(k): float(err) for k, err in points}
        if int(k_pl) not in point_by_k:
            raise RuntimeError(f"{regime}: requested physical-lane prefix k={k_pl} not present in trajectory")
        if regime in PHYSICAL_LANE_K_PL_OVERRIDES:
            plateau_meta = {
                **plateau_meta,
                "status": "manual_physical_lane_prefix_override",
                "rule": "user_requested_physical_lane_prefix_for_cost_and_plot_marker",
                "auto_k_pl": int(auto_k_pl),
                "selected_error": point_by_k[int(k_pl)],
                "user_selected_k_pl": int(k_pl),
            }
        metrics = _compile_prefix_for_report(payload, result_path, int(k_pl))
        adapt = payload.get("adapt_vqe") if isinstance(payload.get("adapt_vqe"), Mapping) else {}
        route = adapt.get("static_route_identity") if isinstance(adapt.get("static_route_identity"), Mapping) else {}
        observed = route.get("observed_components") if isinstance(route.get("observed_components"), Mapping) else {}
        lock_row = source_lock_rows.get(regime, {})
        baseline = baseline_by_regime.get(regime)
        n2q = _int_or_none(metrics.get("n2q"))
        d2q = _int_or_none(metrics.get("d2q"))
        dc = _int_or_none(metrics.get("dc"))
        s_alg = _float_or_none(metrics.get("s_alg"))
        abs_delta_e = _float_or_none(metrics.get("abs_delta_e"))
        status = "done"
        q_status = str(metrics.get("qiskit_cost_status") or "")
        s_status = str(metrics.get("s_alg_status") or "")
        if q_status != "ok" or s_status != "ok":
            status = f"blocked:qiskit={q_status};s_alg={s_status}"
        rows.append(
            ReportRow(
                regime=regime,
                row_source="local_physical_operator_lanes_x3",
                route_label="local physical lanes x3",
                status=status,
                k_pl=_int_or_none(metrics.get("k_iter")),
                d_pl=_int_or_none(metrics.get("d_ans")),
                abs_delta_e=abs_delta_e,
                one_minus_f=None,
                n2q=n2q,
                d2q=d2q,
                dc=dc,
                s_alg=s_alg,
                s_grad=_float_or_none(metrics.get("s_grad")),
                s_h=_float_or_none(metrics.get("s_h")),
                s_metric=_float_or_none(metrics.get("s_metric")),
                s_beam_search_total=_float_or_none(metrics.get("s_beam_search_total")),
                qiskit_cost_source=str(metrics.get("qiskit_cost_source") or ""),
                qiskit_cost_status=q_status,
                qiskit_compile_convention=str(metrics.get("compile_convention") or TABLE_I_QISKIT_COMPILE_CONVENTION),
                s_alg_source=str(metrics.get("s_alg_source") or ""),
                s_alg_status=s_status,
                s_alg_policy="row-facing winner-lineage/display-prefix S_alg; not S_beam_search_total",
                plateau_rule=str(plateau_meta.get("rule") or plateau_meta.get("status") or ""),
                plateau_best_error=_float_or_none(plateau_meta.get("best_error")),
                plateau_threshold=_float_or_none(plateau_meta.get("threshold")),
                plateau_selected_error=_float_or_none(plateau_meta.get("selected_error")),
                selected_pauli_source=str(metrics.get("selected_pauli_source") or ""),
                reference_state_status=str(metrics.get("reference_state_status") or ""),
                source_json=_rel(result_path),
                source_json_sha256=_sha256(result_path),
                baseline_result_json=baseline.source_json if baseline else str(lock_row.get("source_json") or ""),
                baseline_result_sha256=baseline.source_json_sha256 if baseline else str(lock_row.get("source_sha256_tex") or "") or None,
                static_lane_route=str(observed.get("static_lane_route") or _changed_value(lock_row, "static_lane_route", "new") or ""),
                route_variant_id=str(observed.get("route_variant_id") or ""),
                physical_lane_shortlist_aggressiveness=_float_or_none(
                    _changed_value(lock_row, "physical_lane_shortlist_aggressiveness", "new")
                ),
                phase1_shortlist_size_source=_int_or_none(_changed_value(lock_row, "phase1_shortlist_size_effective", "source")),
                phase1_shortlist_size_effective=_int_or_none(_changed_value(lock_row, "phase1_shortlist_size_effective", "new")),
                phase2_shortlist_size_source=_int_or_none(_changed_value(lock_row, "phase2_shortlist_size_effective", "source")),
                phase2_shortlist_size_effective=_int_or_none(_changed_value(lock_row, "phase2_shortlist_size_effective", "new")),
                phase2_shortlist_fraction_source=_float_or_none(_changed_value(lock_row, "phase2_shortlist_fraction_effective", "source")),
                phase2_shortlist_fraction_effective=_float_or_none(_changed_value(lock_row, "phase2_shortlist_fraction_effective", "new")),
                delta_abs_delta_e_vs_baseline=_safe_delta_float(abs_delta_e, baseline.abs_delta_e if baseline else None),
                ratio_abs_delta_e_vs_baseline=_safe_ratio(abs_delta_e, baseline.abs_delta_e if baseline else None),
                delta_n2q_vs_baseline=_safe_delta_int(n2q, baseline.n2q if baseline else None),
                ratio_n2q_vs_baseline=_safe_ratio(n2q, baseline.n2q if baseline else None),
                delta_d2q_vs_baseline=_safe_delta_int(d2q, baseline.d2q if baseline else None),
                ratio_d2q_vs_baseline=_safe_ratio(d2q, baseline.d2q if baseline else None),
                delta_dc_vs_baseline=_safe_delta_int(dc, baseline.dc if baseline else None),
                ratio_dc_vs_baseline=_safe_ratio(dc, baseline.dc if baseline else None),
                delta_s_alg_vs_baseline=_safe_delta_float(s_alg, baseline.s_alg if baseline else None),
                ratio_s_alg_vs_baseline=_safe_ratio(s_alg, baseline.s_alg if baseline else None),
                note=json.dumps({"plateau": plateau_meta, "elapsed_sec": run_row.get("elapsed_sec")}, sort_keys=True),
                **common,
            )
        )
    return rows


def _write_csv(rows: Sequence[ReportRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _main_table(rows: Sequence[ReportRow]) -> list[str]:
    lines = [
        r"\section*{First Plateau Point Costs}",
        r"\scriptsize",
        r"\begin{longtable}{p{0.13\linewidth}p{0.17\linewidth}rrrrrrrr}",
        r"\toprule",
        r"Regime & Route & $k_{\rm pl}$ & $d_{\rm pl}$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ & $S$ ratio \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Regime & Route & $k_{\rm pl}$ & $d_{\rm pl}$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ & $S$ ratio \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in sorted(rows, key=_row_order):
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.regime),
                    _tex_escape(row.route_label),
                    _fmt_int(row.k_pl),
                    _fmt_int(row.d_pl),
                    _fmt_err(row.abs_delta_e),
                    _fmt_int(row.n2q),
                    _fmt_int(row.d2q),
                    _fmt_int(row.dc),
                    _fmt_int(row.s_alg),
                    _fmt_ratio(row.ratio_s_alg_vs_baseline),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    return lines


def _delta_table(rows: Sequence[ReportRow]) -> list[str]:
    candidates = [row for row in rows if row.row_source == "local_physical_operator_lanes_x3"]
    lines = [
        r"\section*{Local Physical-Lane Delta Against Source PDF}",
        r"\scriptsize",
        r"\begin{longtable}{lrrrrrrrr}",
        r"\toprule",
        r"Regime & $|\Delta E|$ ratio & $\Delta N_{2q}$ & $N_{2q}$ ratio & $\Delta D_{2q}$ & $D_{2q}$ ratio & $\Delta D_c$ & $S_{\rm alg}$ ratio & $\Delta S_{\rm alg}$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Regime & $|\Delta E|$ ratio & $\Delta N_{2q}$ & $N_{2q}$ ratio & $\Delta D_{2q}$ & $D_{2q}$ ratio & $\Delta D_c$ & $S_{\rm alg}$ ratio & $\Delta S_{\rm alg}$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in sorted(candidates, key=_row_order):
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.regime),
                    _fmt_ratio(row.ratio_abs_delta_e_vs_baseline),
                    _fmt_int(row.delta_n2q_vs_baseline),
                    _fmt_ratio(row.ratio_n2q_vs_baseline),
                    _fmt_int(row.delta_d2q_vs_baseline),
                    _fmt_ratio(row.ratio_d2q_vs_baseline),
                    _fmt_int(row.delta_dc_vs_baseline),
                    _fmt_ratio(row.ratio_s_alg_vs_baseline),
                    _fmt_int(row.delta_s_alg_vs_baseline),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    return lines


def _provenance_table(rows: Sequence[ReportRow]) -> list[str]:
    candidates = [row for row in rows if row.row_source == "local_physical_operator_lanes_x3"]

    def _compact_path(path_text: str) -> str:
        parts = Path(path_text).parts
        if len(parts) >= 3:
            return str(Path(*parts[-3:]))
        return path_text

    lines = [
        r"\section*{Provenance Pointers}",
        r"\scriptsize",
        r"\begin{longtable}{llllp{0.34\linewidth}}",
        r"\toprule",
        r"Regime & Route variant & Result SHA & Baseline SHA & Result JSON \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Regime & Route variant & Result SHA & Baseline SHA & Result JSON \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in sorted(candidates, key=_row_order):
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.regime),
                    _tex_escape(row.route_variant_id),
                    _tex_escape((row.source_json_sha256 or "")[:12]),
                    _tex_escape((row.baseline_result_sha256 or "")[:12]),
                    _tex_escape(_compact_path(row.source_json)),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    return lines


def _plot_history_error(row: Mapping[str, Any]) -> float | None:
    for key in (
        "delta_abs_current",
        "abs_delta_e_same_cutoff_after",
        "abs_delta_e_after",
        "delta_E_abs_after",
        "benchmark_target_abs_delta_current",
        "exact_abs_delta_e_from_final_state",
        "abs_delta_e",
        "delta_E_abs",
    ):
        value = _float_or_none(row.get(key))
        if value is not None and value > 0.0:
            return float(value)
    return None


def _plot_history_points(payload: Mapping[str, Any]) -> list[tuple[int, float]]:
    history = _history(payload)
    points: list[tuple[int, float]] = []
    if history:
        initial = _float_or_none(history[0].get("delta_abs_prev"))
        if initial is not None and initial > 0.0:
            points.append((0, float(initial)))
    for idx, row in enumerate(history, start=1):
        err = _plot_history_error(row)
        if err is None:
            continue
        depth = _int_or_none(row.get("depth"))
        points.append((int(depth) if depth is not None else idx, float(err)))
    return points


def _point_at_k(points: Sequence[tuple[int, float]], k: int | None) -> tuple[int, float]:
    if k is not None:
        for x_val, y_val in points:
            if int(x_val) == int(k):
                return int(x_val), float(y_val)
    if not points:
        raise ValueError("cannot mark an empty curve")
    return int(points[-1][0]), float(points[-1][1])


def _rows_by_regime_source(rows: Sequence[ReportRow]) -> dict[tuple[str, str], ReportRow]:
    return {(row.regime, row.row_source): row for row in rows}


def _make_error_curve_plots(rows: Sequence[ReportRow]) -> tuple[list[dict[str, Any]], list[Path]]:
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    indexed = _rows_by_regime_source(rows)
    summaries: list[dict[str, Any]] = []
    paths: list[Path] = []
    for regime in REGIME_ORDER:
        baseline = indexed.get((regime, "paper_pdf_baseline"))
        candidate = indexed.get((regime, "local_physical_operator_lanes_x3"))
        if baseline is None or candidate is None:
            continue
        baseline_payload = _read_json(REPO_ROOT / baseline.source_json)
        candidate_payload = _read_json(REPO_ROOT / candidate.source_json)
        baseline_points = _plot_history_points(baseline_payload)
        candidate_points = _plot_history_points(candidate_payload)
        if not baseline_points or not candidate_points:
            summaries.append(
                {
                    "regime": regime,
                    "status": "blocked:missing_history_points",
                    "baseline_points": len(baseline_points),
                    "candidate_points": len(candidate_points),
                }
            )
            continue
        baseline_marker = _point_at_k(baseline_points, baseline.k_pl)
        candidate_marker = _point_at_k(candidate_points, candidate.k_pl)
        png = plot_dir / f"{STEM}_{regime.replace('-', '_')}_error_vs_iteration.png"

        fig, ax = plt.subplots(figsize=(4.8, 3.35), dpi=220)
        ax.plot(
            [x for x, _y in baseline_points],
            [y for _x, y in baseline_points],
            color="#4B5563",
            linestyle="-",
            linewidth=1.95,
        )
        ax.plot(
            [x for x, _y in candidate_points],
            [y for _x, y in candidate_points],
            color="#E45756",
            linestyle="-",
            linewidth=2.35,
        )
        ax.scatter(
            [baseline_marker[0]],
            [baseline_marker[1]],
            marker="o",
            s=58,
            color="#4B5563",
            edgecolors="black",
            linewidths=0.7,
            zorder=8,
        )
        ax.scatter(
            [candidate_marker[0]],
            [candidate_marker[1]],
            marker="*",
            s=92,
            color="#E45756",
            edgecolors="black",
            linewidths=0.7,
            zorder=8,
        )
        y_values = [y for _x, y in baseline_points + candidate_points if y > 0.0]
        ax.set_yscale("log")
        ax.set_ylim(max(1e-8, min(y_values) / 2.5), max(y_values) * 2.5)
        ax.set_xlim(left=0, right=max(max(x for x, _y in baseline_points), max(x for x, _y in candidate_points)) + 1)
        ax.set_xlabel("ADAPT iteration")
        ax.set_ylabel("Same-cutoff energy error")
        ax.set_title(f"HH {regime}: SNAKE descent")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=80))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.25)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.16)
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#4B5563",
                    linestyle="-",
                    linewidth=1.95,
                    marker="o",
                    markersize=5.5,
                    markerfacecolor="#4B5563",
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    label=f"PDF no-batch SNAKE, k={baseline_marker[0]}",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#E45756",
                    linestyle="-",
                    linewidth=2.35,
                    marker="*",
                    markersize=8,
                    markerfacecolor="#E45756",
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    label=f"physical lanes x3, k={candidate_marker[0]}",
                ),
            ],
            frameon=True,
            framealpha=0.94,
            fontsize=7.2,
            title="marker = first plateau prefix",
        )
        fig.tight_layout()
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        paths.append(png)
        summaries.append(
            {
                "regime": regime,
                "status": "ok",
                "plot_png": _rel(png),
                "baseline_source_json": baseline.source_json,
                "baseline_source_json_sha256": baseline.source_json_sha256,
                "candidate_source_json": candidate.source_json,
                "candidate_source_json_sha256": candidate.source_json_sha256,
                "baseline_points": len(baseline_points),
                "candidate_points": len(candidate_points),
                "baseline_marker_k": baseline_marker[0],
                "baseline_marker_error": baseline_marker[1],
                "candidate_marker_k": candidate_marker[0],
                "candidate_marker_error": candidate_marker[1],
                "candidate_marker_error_ratio": _safe_ratio(candidate_marker[1], baseline_marker[1]),
                "marker_policy": "first plateau prefix from comparison rows",
                "error_metric": "adapt_vqe.history same-cutoff delta_abs_current; x=0 initial error from delta_abs_prev when available",
            }
        )
    return summaries, paths


def _plot_summary_table(plot_summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        r"\section*{Error-Versus-Iteration Overlay Summary}",
        r"\scriptsize",
        r"\begin{longtable}{lrrrrr}",
        r"\toprule",
        r"Regime & Base pts & Phys pts & Base marker err & Phys marker err & Phys/Base err \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Regime & Base pts & Phys pts & Base marker err & Phys marker err & Phys/Base err \\",
        r"\midrule",
        r"\endhead",
    ]
    for item in plot_summaries:
        if item.get("status") != "ok":
            continue
        lines.append(
            " & ".join(
                [
                    _tex_escape(item.get("regime", "")),
                    _fmt_int(_int_or_none(item.get("baseline_points"))),
                    _fmt_int(_int_or_none(item.get("candidate_points"))),
                    _fmt_err(_float_or_none(item.get("baseline_marker_error"))),
                    _fmt_err(_float_or_none(item.get("candidate_marker_error"))),
                    _fmt_ratio(_float_or_none(item.get("candidate_marker_error_ratio"))),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    return lines


def _plot_include_path(plot_png: Any) -> str:
    path = REPO_ROOT / str(plot_png)
    try:
        return str(path.resolve().relative_to(OUT_DIR.resolve()))
    except ValueError:
        return str(path)


def _cost_panel_table(regime: str, rows_by_key: Mapping[tuple[str, str], ReportRow]) -> list[str]:
    baseline = rows_by_key[(regime, "paper_pdf_baseline")]
    candidate = rows_by_key[(regime, "local_physical_operator_lanes_x3")]
    out = [
        r"{\tiny",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Row & $k$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ \\",
        r"\midrule",
    ]
    for label, row in (("PDF", baseline), ("Phys", candidate)):
        out.append(
            " & ".join(
                [
                    label,
                    _fmt_int(row.k_pl),
                    _fmt_err(row.abs_delta_e),
                    _fmt_int(row.n2q),
                    _fmt_int(row.d2q),
                    _fmt_int(row.dc),
                    _fmt_int(row.s_alg),
                ]
            )
            + r" \\"
        )
    out.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    return out


def _regime_panel(regime: str, summary: Mapping[str, Any], rows_by_key: Mapping[tuple[str, str], ReportRow]) -> list[str]:
    lines = [
        r"\begin{minipage}[t]{0.475\linewidth}",
        rf"\textbf{{{_tex_escape(regime)}}}\\[-0.4em]",
        r"\begin{center}",
        rf"\includegraphics[width=\linewidth]{{\detokenize{{{_plot_include_path(summary['plot_png'])}}}}}",
        r"\vspace{-0.8em}",
        *_cost_panel_table(regime, rows_by_key),
        r"\end{center}",
        r"\end{minipage}",
    ]
    return lines


def _plot_panel_pages(rows: Sequence[ReportRow], plot_summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    rows_by_key = _rows_by_regime_source(rows)
    summary_by_regime = {str(item.get("regime")): item for item in plot_summaries if item.get("status") == "ok"}
    lines: list[str] = [
        r"\section*{Physical-Lane Error Curves and Prefix Costs}",
        r"\small",
        r"Each two-column panel overlays the source-PDF no-batch SNAKE curve with the local physical-operator-lane x3 curve. The compact table under each panel reports the Qiskit prefix costs and \(S_{\rm alg}\) at the displayed marker prefix.",
        r"\normalsize",
    ]
    for left_idx in range(0, len(REGIME_ORDER), 2):
        pair = REGIME_ORDER[left_idx : left_idx + 2]
        lines.append(r"\par\medskip")
        for idx, regime in enumerate(pair):
            summary = summary_by_regime.get(regime)
            if summary is None or (regime, "paper_pdf_baseline") not in rows_by_key or (regime, "local_physical_operator_lanes_x3") not in rows_by_key:
                continue
            if idx == 1:
                lines.append(r"\hfill")
            lines.extend(_regime_panel(regime, summary, rows_by_key))
        lines.append(r"\par\bigskip")
        if left_idx == 2:
            lines.append(r"\clearpage")
    return lines


def _plot_pages(plot_summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    def _include_path(plot_png: Any) -> str:
        path = REPO_ROOT / str(plot_png)
        try:
            return str(path.resolve().relative_to(OUT_DIR.resolve()))
        except ValueError:
            return str(path)

    lines: list[str] = [
        r"\section*{Error-Versus-Iteration Overlays}",
        r"\small",
        r"Each curve uses the same completed JSON as the cost table. The grey curve is the source-PDF no-batch SNAKE run; the red curve is the local physical-operator-lane x3 run. Markers denote the first plateau prefix used for the table row.",
        r"\normalsize",
    ]
    for item in plot_summaries:
        if item.get("status") != "ok":
            continue
        lines.extend(
            [
                r"\clearpage",
                rf"\subsection*{{{_tex_escape(item['regime'])}}}",
                r"\begin{center}",
                rf"\includegraphics[width=0.82\linewidth]{{\detokenize{{{_include_path(item['plot_png'])}}}}}",
                r"\end{center}",
            ]
        )
    return lines


def _tex_document(
    rows: Sequence[ReportRow],
    manifest: Mapping[str, Any],
    generated: str,
    csv_path: Path,
    plot_summaries: Sequence[Mapping[str, Any]],
) -> str:
    candidate_rows = [row for row in rows if row.row_source == "local_physical_operator_lanes_x3"]
    common = candidate_rows[0] if candidate_rows else rows[0]
    manifest_rows = [
        ("Report", STEM),
        ("Generated UTC", generated),
        ("Run root", _rel(RUN_ROOT)),
        ("Source PDF", _rel(SOURCE_PDF)),
        ("Source PDF SHA-256", common.source_pdf_sha256 or ""),
        ("Source TEX", _rel(SOURCE_TEX)),
        ("Source TEX SHA-256", common.source_tex_sha256 or ""),
        ("Source block", SOURCE_BLOCK),
        ("CSV sidecar", _rel(csv_path)),
        ("Local branch", common.local_git_branch),
        ("Local HEAD", common.local_git_head),
        ("Candidate route", "physical_operator_type"),
        (
            "Shortlist aggressiveness",
            "3x means caps divided by 3: phase1 24 to 8, phase2 12 to 4, phase2 fraction 0.25 to 0.0833333333",
        ),
        ("Batching", "disabled: phase-2 batching false, phase-3 batching false, runtime split max subset size 1"),
        ("Physical-lane prefix overrides", json.dumps(PHYSICAL_LANE_K_PL_OVERRIDES, sort_keys=True)),
        ("Unchanged route contract", json.dumps(manifest.get("snake_runtime_contract", {}), sort_keys=True)),
        ("Plateau rule", "source-PDF baseline k_pl as recorded; physical-lane k_pl uses listed overrides where present"),
        ("Qiskit scope", str(manifest.get("qiskit_circuit_scope") or "")),
        ("Qiskit convention", str(manifest.get("qiskit_compile_convention") or TABLE_I_QISKIT_COMPILE_CONVENTION)),
        ("S_alg policy", str(manifest.get("s_alg_policy") or "")),
    ]
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.62in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{amsmath}",
        r"\usepackage{graphicx}",
        r"\usepackage{url}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\begin{document}",
        r"\title{Paper-I HH Physical Operator Lane Comparison}",
        r"\author{}",
        r"\date{}",
        r"\maketitle",
        r"\section*{Manifest}",
        r"\scriptsize",
        r"\begin{longtable}{p{0.25\linewidth}p{0.68\linewidth}}",
        r"\toprule",
        r"Field & Value \\",
        r"\midrule",
    ]
    for key, value in manifest_rows:
        lines.append(f"{_tex_escape(key)} & {_tex_manifest_value(value)} \\\\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    lines.extend(_plot_panel_pages(rows, plot_summaries))
    lines.extend(_delta_table(rows))
    lines.extend(_provenance_table(rows))
    lines.extend(
        [
            r"\section*{Notes}",
            r"\small",
            r"The baseline rows are the SNAKE cells recorded in the source PDF's duplicate-promotion LaTeX block. "
            r"The local rows are reconstructed from the completed physical-operator-lane JSONs at the physical-lane prefixes listed in the manifest. "
            r"The local physical-operator-lane runs did not enable batching: the unchanged runtime contract records \texttt{phase2\_enable\_batching=false}, \texttt{phase3\_enable\_batching=false}, and \texttt{runtime\_split\_max\_subset\_size=1}. "
            r"Here ``x3'' is only the shortlist-aggressiveness factor, meaning the source lane caps were divided by three. "
            r"The CSV sidecar carries full paths, SHA-256 hashes, Qiskit/S work-source fields, route identity, and cap provenance for each row.",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def _build_pdf(tex_path: Path) -> None:
    commands = (
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        ["tectonic", "--keep-logs", "--reruns", "2", tex_path.name],
    )
    errors: list[str] = []
    for cmd in commands:
        try:
            repeat = 2 if cmd[0] == "pdflatex" else 1
            for _ in range(repeat):
                subprocess.run(
                    cmd,
                    cwd=tex_path.parent,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            return
        except FileNotFoundError:
            errors.append(f"{cmd[0]} not found")
        except subprocess.CalledProcessError as exc:
            errors.append(exc.stdout[-4000:] if exc.stdout else str(exc))
    raise RuntimeError("LaTeX build failed:\n" + "\n".join(errors))


def build(*, require_complete: bool = True) -> Mapping[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    manifest = _source_manifest()
    source_lock_rows = _source_lock_by_regime()
    common = {
        "source_pdf": _rel(SOURCE_PDF),
        "source_pdf_sha256": _sha256(SOURCE_PDF),
        "source_tex": _rel(SOURCE_TEX),
        "source_tex_sha256": _sha256(SOURCE_TEX),
        "run_status_json": _rel(RUN_STATUS),
        "run_status_json_sha256": _sha256(RUN_STATUS),
        "source_lock_manifest": _rel(SOURCE_LOCK),
        "source_lock_manifest_sha256": _sha256(SOURCE_LOCK),
        "commands_json": _rel(COMMANDS_JSON),
        "commands_json_sha256": _sha256(COMMANDS_JSON),
        "local_git_branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "local_git_head": _git_value(["rev-parse", "HEAD"]),
    }
    baseline = _baseline_rows(manifest, source_lock_rows, common)
    baseline_by_regime = {row.regime: row for row in baseline}
    candidates = _candidate_rows(_run_rows(require_complete=require_complete), baseline_by_regime, source_lock_rows, common)
    rows = sorted([*baseline, *candidates], key=_row_order)
    if not candidates:
        raise RuntimeError("no completed candidate rows available")
    plot_summaries, plot_paths = _make_error_curve_plots(rows)
    csv_path = OUT_DIR / f"{STEM}_provenance.csv"
    json_path = OUT_DIR / f"{STEM}_provenance.json"
    tex_path = OUT_DIR / f"{STEM}.tex"
    pdf_path = OUT_DIR / f"{STEM}.pdf"
    _write_csv(rows, csv_path)
    payload = {
        "schema": "paper_i_hh_physical_operator_lane_comparison_report_v1",
        "generated_utc": generated,
        "report_pdf": _rel(pdf_path),
        "report_tex": _rel(tex_path),
        "report_csv": _rel(csv_path),
        "report_json": _rel(json_path),
        "source_machine_readable_block": SOURCE_BLOCK,
        "source_pdf": _rel(SOURCE_PDF),
        "source_pdf_sha256": common["source_pdf_sha256"],
        "source_tex": _rel(SOURCE_TEX),
        "source_tex_sha256": common["source_tex_sha256"],
        "run_root": _rel(RUN_ROOT),
        "run_status_json_sha256": common["run_status_json_sha256"],
        "source_lock_manifest_sha256": common["source_lock_manifest_sha256"],
        "commands_json_sha256": common["commands_json_sha256"],
        "plateau_rel_tol": PLATEAU_REL_TOL,
        "error_curve_plots": [_rel(path) for path in plot_paths],
        "error_curve_summaries": plot_summaries,
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tex_path.write_text(_tex_document(rows, manifest, generated, csv_path, plot_summaries), encoding="utf-8")
    _build_pdf(tex_path)
    payload["report_pdf_sha256"] = _sha256(pdf_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build from completed rows only. Default requires all six run rows to finish.",
    )
    args = parser.parse_args()
    payload = build(require_complete=not args.allow_incomplete)
    print(payload["report_pdf"])
    print(payload["report_csv"])


if __name__ == "__main__":
    main()
