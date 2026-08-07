#!/usr/bin/env python3
"""Build a Powell pool-exposure support PDF for Paper-I HH results.

This report reads existing full-meta singleton symmetry matrix report JSON
sidecars, filters to Powell rows, and builds a compact two-column LaTeX support
PDF.  It does not launch runs.  For selected-prefix display rows, it reloads the
linked source JSONs to compute canonical SNAKE prefix work and Qiskit prefix
resource columns at the displayed prefix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

_IMPORT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_IMPORT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_REPO_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pipelines.exact_bench.snake_table_i_measurement_work import snake_algorithmic_work_from_payload
from pipelines.reporting.paper_i_run_summary import (
    EFFECTIVE_PLATEAU_POLICY,
    PaperIErrorTracePoint,
    PaperIEffectivePlateauSelection,
    canonical_paper_i_algorithmic_work,
    select_paper_i_effective_plateau,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630"
)
DEFAULT_STEM = (
    "paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_"
    "powell_pool_exposure_support"
)
SCHEMA = "paper_i_hh_powell_pool_exposure_support_v1"
SOURCE_SCHEMA = "paper_i_hh_fullmeta_singleton_symmetry_matrix_report_v1"
ENERGY_KEYS = (
    "energy_after",
    "energy_after_opt",
    "energy",
    "primary_energy_metric_after",
)
ERROR_KEYS = (
    "abs_delta_e_same_cutoff_after",
    "abs_delta_e_after",
    "benchmark_target_abs_delta_current",
    "delta_abs_current",
    "delta_E_abs_after",
)

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
REGIME_DISPLAY = {
    "weak-weak": "Weak--weak",
    "intermediate-weak": "Intermediate--weak",
    "strong-weak": "Strong--weak",
    "weak-strong": "Weak--strong",
    "intermediate-strong": "Intermediate--strong",
    "strong-strong": "Strong--strong",
}

A1 = "A_native_staged_singleton_hard_guard"
B1 = "B_common_phase0_singleton_hard_guard"
C_MACRO = "C_macro_only"
@dataclass(frozen=True)
class RoleSpec:
    key: str
    display: str
    method: str
    matrix_label: str
    color: str
    linestyle: str
    marker: str
    description: str


ROLE_SPECS: tuple[RoleSpec, ...] = (
    RoleSpec(
        "snake_native_a1",
        "SNAKE",
        "snake",
        A1,
        "#E45756",
        "-",
        "*",
        "native SNAKE with Phase-III hard-guard singleton replacement",
    ),
    RoleSpec(
        "geo_macro_c",
        "Geo macro",
        "geo",
        C_MACRO,
        "#54A24B",
        ":",
        "^",
        "Geo-ADAPT with macro-generator-only pool",
    ),
    RoleSpec(
        "geo_singleton_b1",
        "Geo singleton",
        "geo",
        B1,
        "#54A24B",
        "-",
        "^",
        "Geo-ADAPT with common Phase-0 hard-guard singleton pool",
    ),
    RoleSpec(
        "append_macro_c",
        "Append macro",
        "append",
        C_MACRO,
        "#4C78A8",
        ":",
        "o",
        "append-only ADAPT with macro-generator-only pool",
    ),
    RoleSpec(
        "append_singleton_b1",
        "Append singleton",
        "append",
        B1,
        "#4C78A8",
        "-",
        "o",
        "append-only ADAPT with common Phase-0 hard-guard singleton pool",
    ),
)


@dataclass
class InputSidecar:
    path: str
    sha256: str
    schema: str | None
    generated_utc: str | None
    report_mode: str | None
    optimizer: str | None
    output_csv: str | None
    output_csv_sha256: str | None
    records_tsv: str | None
    records_tsv_sha256: str | None


@dataclass
class DerivedRow:
    role_key: str
    role_display: str
    role_description: str
    matrix_label: str
    regime: str
    method: str
    optimizer: str
    status: str
    plotted: bool
    missing_reason: str
    selection_policy: str
    selection_status: str
    selected_prefix_k: int | None
    record_id: str
    input_report_json: str
    input_report_json_sha256: str
    iteration: int | None
    depth: int | None
    abs_delta_e: float | None
    fidelity: float | None
    fidelity_status: str
    fidelity_source: str
    fidelity_status_detail: str
    n2q: int | None
    d2q: int | None
    dc: int | None
    cost_source: str
    cost_status: str
    s_grad: float | None
    s_refit: float | None
    s_outer: float | None
    s_h: float | None
    s_metric: float | None
    s_alg: float | None
    s_work_status: str
    s_work_source: str
    s_work_status_detail: str
    trajectory_points: list[list[float]]
    source_json: str
    source_sha256: str
    source_dir: str
    note: str


@dataclass
class DuplicateRecord:
    key: list[str]
    kept_record_id: str
    discarded_record_id: str
    kept_input_report_json: str
    discarded_input_report_json: str
    reason: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path | str) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return int(round(v))


def _selected_labels(row: Mapping[str, Any]) -> list[str]:
    value = row.get("selected_batch_labels")
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    value = row.get("selected_ops")
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    for key in ("selected_op", "selected_logical_op", "selected_label"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
    return []


def _tableiii_prefix_resources_module() -> Any:
    """Load exact-prefix compilation support only for selected-prefix work."""

    from pipelines.exact_bench import hh_tableiii_prefix_resources

    return hh_tableiii_prefix_resources


def _compile_snake_history_prefix_lazily(
    payload: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    selected_k: int,
    *,
    source_kind: str,
) -> Mapping[str, Any] | None:
    """Keep optional historical SNAKE compilation out of builder import time."""

    from pipelines.reporting.build_paper_i_hh_pass2_costs_plots import (
        _compile_history_prefix,
    )

    return _compile_history_prefix(
        payload,
        history,
        selected_k,
        source_kind=source_kind,
    )


def fmt_float(value: float | None, sig: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    av = abs(value)
    # Display any table error whose scientific exponent is e-2 or smaller
    # in explicit scientific notation; keep e-1 scale values in compact
    # decimal form for readability.
    if av < 1.0e-1 or av >= 1.0e3:
        return f"{value:.{sig}e}"
    return f"{value:.{sig}g}"


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "--"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(v):
        return "--"
    return f"{int(round(v)):,}"


def tex_escape(text: Any) -> str:
    s = "" if text is None else str(text)
    return (
        s.replace("\\", r"\textbackslash{}")
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def discover_input_jsons(input_dir: Path, explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        return [p if p.is_absolute() else REPO_ROOT / p for p in explicit]
    paths: list[Path] = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            data = load_json(path)
        except Exception:
            continue
        if data.get("schema") != SOURCE_SCHEMA:
            continue
        rows = data.get("rows")
        if not isinstance(rows, list):
            continue
        contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
        contract_optimizer = str(contract.get("optimizer") or "").upper()
        has_powell = contract_optimizer == "POWELL" or any(
            str(row.get("optimizer") or "").upper() == "POWELL"
            for row in rows
            if isinstance(row, Mapping)
        )
        if has_powell:
            paths.append(path)
    return paths


def input_sidecar_from(path: Path, data: Mapping[str, Any]) -> InputSidecar:
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    output_csv = data.get("output_csv")
    output_csv_sha = None
    if output_csv:
        csv_path = REPO_ROOT / str(output_csv)
        if csv_path.exists():
            output_csv_sha = sha256(csv_path)
    return InputSidecar(
        path=rel(path),
        sha256=sha256(path),
        schema=str(data.get("schema") or "") or None,
        generated_utc=str(data.get("generated_utc") or "") or None,
        report_mode=str(data.get("report_mode") or "") or None,
        optimizer=str(contract.get("optimizer") or "") or None,
        output_csv=str(output_csv or "") or None,
        output_csv_sha256=output_csv_sha,
        records_tsv=str(data.get("records_tsv") or "") or None,
        records_tsv_sha256=str(data.get("records_tsv_sha256") or "") or None,
    )


def normalize_points(raw: Any) -> list[list[float]]:
    points: list[list[float]] = []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return points
    for item in raw:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
            continue
        x = as_float(item[0])
        y = as_float(item[1])
        if x is None or y is None or y <= 0:
            continue
        points.append([float(x), float(y)])
    return points


def resolve_repo_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json_if_available(path_text: str | None) -> tuple[Path | None, Mapping[str, Any] | None, str | None]:
    path = resolve_repo_path(path_text)
    if path is None or not path.exists() or not path.is_file():
        return path, None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        return path, None, sha256(path)
    return path, data, sha256(path)


def trajectory_error_at_k(points: Sequence[Sequence[float]], k: int | None) -> float | None:
    if k is None:
        return None
    for point in points:
        if len(point) >= 2 and as_int(point[0]) == int(k):
            return as_float(point[1])
    return None


@dataclass(frozen=True)
class SourceLockedHistoricalPlateauAdapter:
    """Typed, fail-closed projection of a preserved comparator trace."""

    error_trace: tuple[PaperIErrorTracePoint, ...]
    effective_plateau: PaperIEffectivePlateauSelection


def source_locked_historical_plateau_adapter(
    points: Sequence[Sequence[float]],
) -> SourceLockedHistoricalPlateauAdapter:
    """Delegate preserved trace selection to the canonical summary policy."""

    error_trace = tuple(
        PaperIErrorTracePoint(
            controller_round=_source_locked_controller_round(point, index),
            absolute_energy_error=_source_locked_absolute_error(point, index),
        )
        for index, point in enumerate(points, start=1)
    )
    return SourceLockedHistoricalPlateauAdapter(
        error_trace=error_trace,
        effective_plateau=select_paper_i_effective_plateau(error_trace),
    )


def _source_locked_controller_round(
    point: Sequence[float],
    expected_round: int,
) -> int:
    if len(point) < 2:
        raise ValueError(
            "source-locked comparator trajectory row lacks controller round "
            "and same-cutoff error."
        )
    observed = as_int(point[0])
    if observed != expected_round:
        raise ValueError(
            "source-locked comparator trajectory must contain complete ordered "
            f"controller rounds 1..N; expected {expected_round}, got {observed}."
        )
    return int(observed)


def _source_locked_absolute_error(
    point: Sequence[float],
    controller_round: int,
) -> float:
    error = as_float(point[1]) if len(point) >= 2 else None
    if error is None or error < 0.0:
        raise ValueError(
            "source-locked comparator trajectory has no finite nonnegative "
            f"same-cutoff error at controller round {controller_round}."
        )
    return float(error)


def generic_history(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
    history = result.get("adapt_history") or result.get("history") or payload.get("adapt_history") or payload.get("history") or []
    return [row for row in history if isinstance(row, Mapping)]


def _history_depth_after(row: Mapping[str, Any], depth_before: int) -> int:
    explicit = as_int(row.get("depth_after"))
    if explicit is not None:
        return int(explicit)
    appended = as_int(row.get("appended_operator_count"))
    if appended is None:
        appended = as_int(row.get("batch_size"))
    if appended is None:
        labels = _selected_labels(row)
        appended = len(labels)
    return int(depth_before) + max(0, int(appended))


def _history_rows_through_depth(
    history: Sequence[Mapping[str, Any]],
    selected_depth: int,
) -> tuple[list[Mapping[str, Any]], int]:
    if int(selected_depth) < 1:
        return [], 0
    consumed: list[Mapping[str, Any]] = []
    depth = 0
    for index, row in enumerate(history):
        depth_after = _history_depth_after(row, depth)
        if depth_after > int(selected_depth):
            raise ValueError(
                f"selected depth {selected_depth} cuts through adaptive batch {depth}->{depth_after}"
            )
        consumed.append(row)
        depth = int(depth_after)
        if depth == int(selected_depth):
            probe_depth = depth
            for later in history[index + 1 :]:
                later_depth = _history_depth_after(later, probe_depth)
                if later_depth == probe_depth:
                    raise ValueError(
                        "logical depth maps to multiple history rows after a Geo immediate-repeat skip; "
                        "an explicit history position is required"
                    )
                break
            return consumed, depth
    return consumed, depth


def snake_history(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    adapt = payload.get("adapt_vqe") if isinstance(payload.get("adapt_vqe"), Mapping) else {}
    history = adapt.get("history") or adapt.get("adapt_history") or adapt.get("history_tail") or []
    return [row for row in history if isinstance(row, Mapping)]


def generic_prefix_s_alg(payload: Mapping[str, Any], selected_k: int) -> tuple[float | None, str, dict[str, Any]]:
    history = generic_history(payload)
    try:
        consumed, reached_depth = _history_rows_through_depth(history, int(selected_k))
    except ValueError as exc:
        return None, "generic_prefix_s_alg_blocked_batch_cut", {"reason": str(exc)}
    if int(selected_k) < 1 or reached_depth != int(selected_k):
        return None, "generic_prefix_s_alg_blocked_k_outside_depth", {
            "history_len": len(history),
            "max_depth": reached_depth,
        }
    outer = refit = grad = metric = other = 0.0
    for row in consumed:
        outer += as_float(row.get("outer_hamiltonian_eval_count")) or 1.0
        refit += as_float(row.get("optimizer_nfev")) or as_float(row.get("nfev_opt")) or 0.0
        grad += as_float(row.get("candidate_count_scored")) or 0.0
        qngd_steps = as_float(row.get("qngd_metric_eval_count")) or 0.0
        qngd_gradient = as_float(row.get("qngd_gradient_operator_probe_count_total"))
        if qngd_steps > 0.0 and qngd_gradient is None:
            return None, "generic_prefix_s_alg_blocked_missing_qngd_gradient_ledger", {
                "history_position": row.get("history_position", row.get("iteration")),
                "qngd_metric_eval_count": qngd_steps,
            }
        grad += qngd_gradient or 0.0
        metric += as_float(row.get("selector_metric_probe_count")) or 0.0
        metric += as_float(row.get("qngd_metric_operator_probe_count_total")) or 0.0
        other += as_float(row.get("N_other_quantum")) or 0.0
    metadata = {
        "N_H_outer": outer,
        "N_H_refit": refit,
        "N_grad": grad,
        "N_metric": metric,
        "N_other_quantum": other,
        "history_len": len(history),
        "history_rows_consumed": len(consumed),
        "selected_logical_depth": int(selected_k),
    }
    if other != 0.0:
        return None, "generic_prefix_s_alg_blocked_noncanonical_other_work", metadata
    try:
        work = canonical_paper_i_algorithmic_work(
            n_h_outer=_nonnegative_integral_count(outer, name="N_H_outer"),
            n_h_refit=_nonnegative_integral_count(refit, name="N_H_refit"),
            n_grad=_nonnegative_integral_count(grad, name="N_grad"),
            n_metric=_nonnegative_integral_count(metric, name="N_metric"),
        )
    except ValueError as exc:
        metadata["reason"] = str(exc)
        return None, "generic_prefix_s_alg_blocked_invalid_component", metadata
    return float(work.s_alg), "ok", metadata


def _nonnegative_integral_count(value: float, *, name: str) -> int:
    if not math.isfinite(value) or value < 0.0 or int(value) != value:
        raise ValueError(f"{name} must be a nonnegative integral count.")
    return int(value)


def compile_generic_selected_prefix_row(
    *,
    regime: str,
    method: str,
    source_path: Path,
    source_sha256: str | None,
    payload: Mapping[str, Any],
    selected_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile exactly one generic ADAPT prefix row.

    ``selected_k`` is logical ansatz depth, not history-row position.  Geo
    immediate-repeat skips still consume estimator work and history rows, but
    do not add a circuit generator.
    """
    tableiii = _tableiii_prefix_resources_module()
    history = tableiii._history(payload)
    consumed, reached_depth = _history_rows_through_depth(history, int(selected_k))
    if int(selected_k) < 1 or reached_depth != int(selected_k):
        raise ValueError(
            f"selected logical depth {selected_k} outside reconstructed depth {reached_depth}"
        )
    reference_state, reference_meta = tableiii._reference_state_from_source(
        source_path,
        payload,
    )
    if reference_state is None:
        raise ValueError(f"reference state unavailable for {source_path}: {reference_meta}")
    fallback_groups, fallback_source = tableiii._fallback_step_groups(payload, history)
    pauli_groups: list[list[str]] = []
    execution_modes: list[str] = []
    selected_hist_row: Mapping[str, Any] | None = None
    selected_labels: list[str] = []
    selected_pauli_source = ""
    appended_group_cursor = 0
    reconstructed_depth = 0
    for history_index, hist_row in enumerate(consumed, start=1):
        labels = _selected_labels(hist_row)
        pauli_lookup = tableiii._pauli_map(hist_row)
        if labels and all(pauli_lookup.get(label) for label in labels):
            step_groups = [list(pauli_lookup[label]) for label in labels]
            step_pauli_source = "history_selected_candidate_pauli_labels"
        elif labels and fallback_groups is not None and appended_group_cursor < len(fallback_groups):
            step_groups = [list(group) for group in fallback_groups[appended_group_cursor]]
            step_pauli_source = str(fallback_source)
        elif not labels and _history_depth_after(hist_row, reconstructed_depth) == reconstructed_depth:
            step_groups = []
            step_pauli_source = "geo_immediate_repeat_no_append"
        else:
            missing = [label for label in labels if label not in pauli_lookup]
            raise ValueError(
                f"missing Pauli group for {regime}/{method} history row {history_index}: {missing or labels}"
            )
        positions = tableiii._selected_positions(hist_row, len(step_groups))
        raw_modes = hist_row.get("selected_batch_execution_modes") or hist_row.get(
            "selected_candidate_execution_modes"
        )
        if isinstance(raw_modes, Sequence) and not isinstance(raw_modes, (str, bytes)):
            step_modes = [str(mode) for mode in raw_modes]
        else:
            step_modes = []
        if len(step_modes) != len(step_groups):
            step_modes = ["unknown_legacy_execution_mode"] * len(step_groups)
        for group, mode, position in zip(step_groups, step_modes, positions):
            if position is None or position < 0 or position > len(pauli_groups):
                pauli_groups.append(group)
                execution_modes.append(mode)
            else:
                pauli_groups.insert(int(position), group)
                execution_modes.insert(int(position), mode)
        if step_groups:
            appended_group_cursor += 1
        reconstructed_depth = _history_depth_after(hist_row, reconstructed_depth)
        if reconstructed_depth == int(selected_k):
            selected_hist_row = hist_row
            selected_labels = labels
            selected_pauli_source = step_pauli_source
    if selected_hist_row is None or not pauli_groups:
        raise ValueError(f"selected prefix {selected_k} produced no logical groups")
    num_qubits = tableiii._num_qubits_from_groups(pauli_groups, reference_state)
    unsupported_modes = sorted({mode for mode in execution_modes if mode != "termwise_product"})
    if unsupported_modes:
        compiled = {}
        compile_status = "grouped_or_unknown_generator_synthesis_unavailable"
        compile_error = (
            "refusing termwise Pauli-rotation cost substitution for execution modes "
            + ", ".join(unsupported_modes)
        )
    else:
        try:
            compiled = tableiii.compile_table_i_pauli_label_groups(
                pauli_label_groups=tuple(tuple(group) for group in pauli_groups),
                num_qubits=num_qubits,
                reference_state=reference_state,
                source_kind="paper_i_hh_powell_pool_exposure_selected_generic_prefix",
            )
            compile_status = "ok"
            compile_error = None
        except tableiii.TableICompileUnavailable as exc:
            compiled = {}
            compile_status = exc.status
            compile_error = exc.reason
    energy_key, energy = tableiii._first_present(selected_hist_row, ENERGY_KEYS)
    error_key, error = tableiii._first_present(selected_hist_row, ERROR_KEYS)
    return {
        "schema": "paper_i_hh_powell_pool_exposure_selected_generic_prefix_resource_row_v1",
        "regime": regime,
        "method": method,
        "source_json": str(source_path),
        "source_sha256": source_sha256,
        "prefix_k": int(selected_k),
        "prefix_k_semantics": "logical_operator_depth",
        "history_rows_consumed": int(len(consumed)),
        "adapt_iteration": selected_hist_row.get("iteration", selected_k),
        "logical_operator_prefix_len": int(len(pauli_groups)),
        "selected_batch_size": int(len(selected_labels)),
        "selected_labels": selected_labels,
        "selected_pauli_source": selected_pauli_source,
        "operator_execution_modes": execution_modes,
        "prefix_order_semantics": "history_selected_positions_when_serialized_else_append_order",
        "energy": None if energy is None else float(energy),
        "energy_field": energy_key,
        "abs_delta_e": None if error is None else float(error),
        "abs_delta_e_field": error_key,
        "compile_status": compile_status,
        "compile_error": compile_error,
        "compile_convention": tableiii.TABLE_I_QISKIT_COMPILE_CONVENTION,
        "N1q": compiled.get("compiled_count_1q_total"),
        "N2q": compiled.get("compiled_count_2q_total"),
        "D_circ": compiled.get("compiled_depth_total"),
        "D2q": compiled.get("compiled_depth_2q_total"),
        "compiled_count_1q_semantics": compiled.get("compiled_count_1q_semantics"),
        "compiled_op_counts": compiled.get("compiled_op_counts"),
        "num_qubits": compiled.get("num_qubits", num_qubits),
        "runtime_rotation_count": compiled.get("runtime_rotation_count"),
        "reference_state_status": reference_meta.get("status"),
    }, reference_meta


def selected_prefix_payload(raw: Mapping[str, Any], spec: RoleSpec, regime: str, points: Sequence[Sequence[float]]) -> dict[str, Any]:
    try:
        source_locked = source_locked_historical_plateau_adapter(points)
        selection = source_locked.effective_plateau
        selected_k = selection.controller_round
        policy = selection.policy
        policy_meta: dict[str, Any] = {
            "status": "ok",
            "adapter": "source_locked_historical_plateau_adapter_v1",
            "selected_trace_index": selection.selected_trace_index,
            "best_error": selection.best_observed_error,
            "threshold": selection.selection_threshold,
            "selected_error": selection.absolute_energy_error,
            "horizon_controller_rounds": selection.horizon_controller_rounds,
        }
    except ValueError as exc:
        selected_k = None
        policy = EFFECTIVE_PLATEAU_POLICY
        policy_meta = {
            "status": "blocked_source_locked_error_trace",
            "reason": str(exc),
        }
    out: dict[str, Any] = {
        "selection_policy": policy,
        "selection_status": str(policy_meta.get("status") or "ok"),
        "selected_prefix_k": selected_k,
        "selection_meta": policy_meta,
        "iteration": selected_k,
        "depth": selected_k,
        "abs_delta_e": trajectory_error_at_k(points, selected_k),
        "fidelity": None,
        "fidelity_status": "blocked_selected_prefix_state_not_serialized",
        "fidelity_source": "not_computed",
        "fidelity_status_detail": "selected-prefix optimized state/theta is not serialized in the source artifact",
        "n2q": None,
        "d2q": None,
        "dc": None,
        "cost_source": "selected_prefix_compile",
        "cost_status": "not_attempted",
        "s_alg": None,
        "s_work_status": "not_attempted",
        "s_work_source": "selected_prefix",
        "s_work_status_detail": "",
        "note": "selected-prefix row",
    }
    source_path, payload, source_sha = load_json_if_available(str(raw.get("source_json") or ""))
    if source_path is None or payload is None:
        out.update(
            fidelity_status="blocked_missing_source_json",
            fidelity_source="not_computed",
            fidelity_status_detail="source_json unavailable for selected-prefix fidelity reconstruction",
            cost_status="blocked_missing_source_json",
            s_work_status="blocked_missing_source_json",
            s_work_status_detail="source_json unavailable for selected-prefix reconstruction",
        )
        return out
    if selected_k is None:
        out.update(
            fidelity_status="blocked_missing_selected_prefix",
            fidelity_source="not_computed",
            fidelity_status_detail="no selected prefix was resolved",
            cost_status="blocked_missing_selected_prefix",
            s_work_status="blocked_missing_selected_prefix",
            s_work_status_detail="no selected prefix was resolved",
        )
        return out
    try:
        if spec.method == "snake":
            history = snake_history(payload)
            compiled = _compile_snake_history_prefix_lazily(
                payload,
                history,
                int(selected_k),
                source_kind="paper_i_hh_powell_pool_exposure_selected_snake_prefix",
            ) or {}
            out.update(
                n2q=as_int(compiled.get("compiled_count_2q_total")),
                d2q=as_int(compiled.get("compiled_depth_2q_total")),
                dc=as_int(compiled.get("compiled_depth_total")),
                cost_status="ok",
                cost_source="qiskit_selected_snake_history_prefix_compile",
            )
            work, audit = snake_algorithmic_work_from_payload(
                payload,
                scope="display_prefix",
                history_position=int(selected_k),
                source_label=rel(source_path),
            )
            out.update(
                s_alg=as_float(work.get("S_alg")) if isinstance(work, Mapping) else None,
                s_work_status=str(work.get("S_alg_status") or audit.get("status") or "unknown") if isinstance(work, Mapping) else str(audit.get("status") or "unknown"),
                s_work_source="snake_algorithmic_work_from_payload(scope=display_prefix)",
                s_work_status_detail=json.dumps({"audit_status": audit.get("status"), "history_position": selected_k}, sort_keys=True),
            )
            if int(selected_k) == as_int(raw.get("iteration")):
                terminal_fidelity = as_float(raw.get("fidelity"))
                if terminal_fidelity is not None:
                    out["fidelity"] = terminal_fidelity
                    out["fidelity_status"] = "ok_terminal_selected_prefix"
                    out["fidelity_source"] = "input_report_row_terminal_fidelity"
                    out["fidelity_status_detail"] = "selected prefix equals terminal row iteration"
                else:
                    out["fidelity_status"] = "blocked_terminal_fidelity_missing"
                    out["fidelity_source"] = "not_computed"
                    out["fidelity_status_detail"] = "selected prefix equals terminal row iteration, but input report row lacks fidelity"
        else:
            selected, reference_meta = compile_generic_selected_prefix_row(
                regime=regime,
                method=spec.display,
                source_path=source_path,
                source_sha256=source_sha,
                payload=payload,
                selected_k=int(selected_k),
            )
            out.update(
                depth=as_int(selected.get("logical_operator_prefix_len")) or selected_k,
                abs_delta_e=as_float(selected.get("abs_delta_e")) or out["abs_delta_e"],
                n2q=as_int(selected.get("N2q")),
                d2q=as_int(selected.get("D2q")),
                dc=as_int(selected.get("D_circ")),
                cost_status=str(selected.get("compile_status") or "unknown"),
                cost_source="qiskit_selected_generic_prefix_compile",
            )
            s_alg, s_status, s_meta = generic_prefix_s_alg(payload, int(selected_k))
            out.update(
                s_alg=s_alg,
                s_work_status=s_status,
                s_work_source=(
                    "source_locked_generic_history_to_"
                    "canonical_paper_i_algorithmic_work"
                ),
                s_work_status_detail=json.dumps({"components": s_meta, "reference_state_status": reference_meta.get("status")}, sort_keys=True),
            )
    except Exception as exc:
        out.update(
            fidelity_status="blocked_selected_prefix_reconstruction_error",
            fidelity_source="not_computed",
            fidelity_status_detail=str(exc),
            cost_status="selected_prefix_reconstruction_blocked",
            s_work_status="selected_prefix_reconstruction_blocked",
            s_work_status_detail=str(exc),
        )
    if out["abs_delta_e"] is None:
        out["abs_delta_e"] = as_float(raw.get("abs_delta_e")) if selected_k == as_int(raw.get("iteration")) else None
    out["note"] = f"selected_prefix_k={selected_k}; policy={policy}; source={rel(source_path)}"
    return out


def status_rank(row: Mapping[str, Any]) -> int:
    status = str(row.get("status") or "").lower()
    if status in {"done", "reference"}:
        return 4
    if status in {"evidence-invalid", "invalid"}:
        return 3
    if status == "pending":
        return 2
    if status:
        return 1
    return 0


def row_score(row: Mapping[str, Any], generated_utc: str) -> tuple[int, int, int, int, str]:
    note = str(row.get("note") or "").lower()
    repair = 1 if "repair" in note or "depth-zero" in note or "depth_zero" in note else 0
    points = normalize_points(row.get("trajectory_points"))
    has_cost = int(all(as_int(row.get(k)) is not None for k in ("n2q", "d2q", "dc")))
    has_work = int(as_float(row.get("s_alg")) is not None)
    return (repair, status_rank(row), int(bool(points)), has_cost + has_work, generated_utc)


def select_rows(
    input_paths: Sequence[Path],
) -> tuple[list[InputSidecar], dict[tuple[str, str, str], tuple[dict[str, Any], InputSidecar]], list[DuplicateRecord]]:
    sidecars: list[InputSidecar] = []
    selected: dict[tuple[str, str, str], tuple[dict[str, Any], InputSidecar]] = {}
    selected_scores: dict[tuple[str, str, str], tuple[int, int, int, int, str]] = {}
    duplicates: list[DuplicateRecord] = []
    for path in input_paths:
        data = load_json(path)
        if data.get("schema") != SOURCE_SCHEMA:
            continue
        sidecar = input_sidecar_from(path, data)
        sidecars.append(sidecar)
        generated = str(data.get("generated_utc") or "")
        contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
        contract_optimizer = str(contract.get("optimizer") or "").upper()
        rows = data.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_optimizer = str(row.get("optimizer") or contract_optimizer or "").upper()
            if row_optimizer != "POWELL":
                continue
            key = (
                str(row.get("matrix_label") or ""),
                str(row.get("regime") or ""),
                str(row.get("method") or ""),
            )
            if not all(key):
                continue
            score = row_score(row, generated)
            if key not in selected:
                selected[key] = (row, sidecar)
                selected_scores[key] = score
                continue
            old_row, old_sidecar = selected[key]
            old_score = selected_scores[key]
            if score > old_score:
                selected[key] = (row, sidecar)
                selected_scores[key] = score
                duplicates.append(
                    DuplicateRecord(
                        key=list(key),
                        kept_record_id=str(row.get("record_id") or ""),
                        discarded_record_id=str(old_row.get("record_id") or ""),
                        kept_input_report_json=sidecar.path,
                        discarded_input_report_json=old_sidecar.path,
                        reason="new row ranked higher by repair/status/trajectory/cost/work/generated_utc",
                    )
                )
            else:
                duplicates.append(
                    DuplicateRecord(
                        key=list(key),
                        kept_record_id=str(old_row.get("record_id") or ""),
                        discarded_record_id=str(row.get("record_id") or ""),
                        kept_input_report_json=old_sidecar.path,
                        discarded_input_report_json=sidecar.path,
                        reason="existing row ranked higher or tied by repair/status/trajectory/cost/work/generated_utc",
                    )
                )
    return sidecars, selected, duplicates


def make_derived_rows(
    selected: Mapping[tuple[str, str, str], tuple[dict[str, Any], InputSidecar]],
    regimes: Sequence[str],
) -> tuple[list[DerivedRow], list[dict[str, Any]]]:
    out: list[DerivedRow] = []
    missing: list[dict[str, Any]] = []
    for regime in regimes:
        for spec in ROLE_SPECS:
            key = (spec.matrix_label, regime, spec.method)
            if key not in selected:
                reason = "missing_source_row"
                missing.append(
                    {
                        "role_key": spec.key,
                        "matrix_label": spec.matrix_label,
                        "regime": regime,
                        "method": spec.method,
                        "reason": reason,
                    }
                )
                out.append(
                    DerivedRow(
                        role_key=spec.key,
                        role_display=spec.display,
                        role_description=spec.description,
                        matrix_label=spec.matrix_label,
                        regime=regime,
                        method=spec.method,
                        optimizer="POWELL",
                        status="missing",
                        plotted=False,
                        missing_reason=reason,
                        selection_policy="missing",
                        selection_status=reason,
                        selected_prefix_k=None,
                        record_id="",
                        input_report_json="",
                        input_report_json_sha256="",
                        iteration=None,
                        depth=None,
                        abs_delta_e=None,
                        fidelity=None,
                        fidelity_status="missing",
                        fidelity_source="",
                        fidelity_status_detail="missing expected Powell pool-exposure source row",
                        n2q=None,
                        d2q=None,
                        dc=None,
                        cost_source="",
                        cost_status="missing",
                        s_grad=None,
                        s_refit=None,
                        s_outer=None,
                        s_h=None,
                        s_metric=None,
                        s_alg=None,
                        s_work_status="missing",
                        s_work_source="",
                        s_work_status_detail="",
                        trajectory_points=[],
                        source_json="",
                        source_sha256="",
                        source_dir="",
                        note="missing expected Powell pool-exposure source row",
                    )
                )
                continue
            raw, sidecar = selected[key]
            s_refit = as_float(raw.get("s_refit"))
            s_outer = as_float(raw.get("s_outer"))
            s_h = None if s_refit is None or s_outer is None else s_refit + s_outer
            points = normalize_points(raw.get("trajectory_points"))
            status = str(raw.get("status") or "")
            plotted = status.lower() in {"done", "reference"} and bool(points)
            missing_reason = "" if plotted else ("no_positive_trajectory" if not points else f"status:{status}")
            prefix = selected_prefix_payload(raw, spec, regime, points) if plotted else {
                "selection_policy": "not_plotted",
                "selection_status": missing_reason,
                "selected_prefix_k": None,
                "iteration": None,
                "depth": None,
                "abs_delta_e": None,
                "fidelity": None,
                "fidelity_status": "not_plotted",
                "fidelity_source": "",
                "fidelity_status_detail": missing_reason,
                "n2q": None,
                "d2q": None,
                "dc": None,
                "cost_source": "",
                "cost_status": "not_plotted",
                "s_alg": None,
                "s_work_status": "not_plotted",
                "s_work_source": "",
                "s_work_status_detail": "",
                "note": "not plotted",
            }
            out.append(
                DerivedRow(
                    role_key=spec.key,
                    role_display=spec.display,
                    role_description=spec.description,
                    matrix_label=spec.matrix_label,
                    regime=regime,
                    method=spec.method,
                    optimizer=str(raw.get("optimizer") or "POWELL"),
                    status=status,
                    plotted=plotted,
                    missing_reason=missing_reason,
                    selection_policy=str(prefix.get("selection_policy") or ""),
                    selection_status=str(prefix.get("selection_status") or ""),
                    selected_prefix_k=as_int(prefix.get("selected_prefix_k")),
                    record_id=str(raw.get("record_id") or ""),
                    input_report_json=sidecar.path,
                    input_report_json_sha256=sidecar.sha256,
                    iteration=as_int(prefix.get("iteration")),
                    depth=as_int(prefix.get("depth")),
                    abs_delta_e=as_float(prefix.get("abs_delta_e")),
                    fidelity=as_float(prefix.get("fidelity")),
                    fidelity_status=str(prefix.get("fidelity_status") or ""),
                    fidelity_source=str(prefix.get("fidelity_source") or ""),
                    fidelity_status_detail=str(prefix.get("fidelity_status_detail") or ""),
                    n2q=as_int(prefix.get("n2q")),
                    d2q=as_int(prefix.get("d2q")),
                    dc=as_int(prefix.get("dc")),
                    cost_source=str(prefix.get("cost_source") or ""),
                    cost_status=str(prefix.get("cost_status") or ""),
                    s_grad=as_float(raw.get("s_grad")),
                    s_refit=s_refit,
                    s_outer=s_outer,
                    s_h=s_h,
                    s_metric=as_float(raw.get("s_metric")),
                    s_alg=as_float(prefix.get("s_alg")),
                    s_work_status=str(prefix.get("s_work_status") or ""),
                    s_work_source=str(prefix.get("s_work_source") or ""),
                    s_work_status_detail=str(prefix.get("s_work_status_detail") or ""),
                    trajectory_points=points,
                    source_json=str(raw.get("source_json") or ""),
                    source_sha256=str(raw.get("source_sha256") or ""),
                    source_dir=str(raw.get("source_dir") or ""),
                    note=str(prefix.get("note") or raw.get("note") or ""),
                )
            )
    return out, missing


def compare_values(a: Any, b: Any, tol: float = 1.0e-10) -> bool:
    fa = as_float(a)
    fb = as_float(b)
    if fa is not None or fb is not None:
        return fa is not None and fb is not None and math.isclose(fa, fb, rel_tol=tol, abs_tol=tol)
    return str(a or "") == str(b or "")


def singleton_equivalence(
    selected: Mapping[tuple[str, str, str], tuple[dict[str, Any], InputSidecar]],
    regimes: Sequence[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    fields = ("abs_delta_e", "iteration", "depth", "n2q", "d2q", "dc", "s_alg")
    for regime in regimes:
        for method in ("geo", "append"):
            a = selected.get((A1, regime, method))
            b = selected.get((B1, regime, method))
            if a is None or b is None:
                checks.append(
                    {
                        "regime": regime,
                        "method": method,
                        "a1_present": a is not None,
                        "b1_present": b is not None,
                        "equivalent": False,
                        "differing_fields": ["missing_a1_or_b1"],
                    }
                )
                continue
            row_a, _ = a
            row_b, _ = b
            diff = [field for field in fields if not compare_values(row_a.get(field), row_b.get(field))]
            checks.append(
                {
                    "regime": regime,
                    "method": method,
                    "a1_record_id": str(row_a.get("record_id") or ""),
                    "b1_record_id": str(row_b.get("record_id") or ""),
                    "equivalent": not diff,
                    "differing_fields": diff,
                }
            )
    return checks


def plot_regime(
    rows: Sequence[DerivedRow],
    regime: str,
    fig_dir: Path,
    stem: str,
    omit_plot_roles: set[str] | None = None,
) -> dict[str, Any]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    plotted: list[list[str]] = []
    omitted: list[dict[str, str]] = []
    omit_plot_roles = omit_plot_roles or set()
    row_by_role = {row.role_key: row for row in rows if row.regime == regime}
    spec_by_role = {spec.key: spec for spec in ROLE_SPECS}
    for spec in ROLE_SPECS:
        row = row_by_role.get(spec.key)
        if spec.key in omit_plot_roles:
            omitted.append(
                {
                    "role_key": spec.key,
                    "matrix_label": spec.matrix_label,
                    "method": spec.method,
                    "reason": "plot_role_omitted",
                }
            )
            continue
        if row is None or not row.plotted:
            omitted.append(
                {
                    "role_key": spec.key,
                    "matrix_label": spec.matrix_label,
                    "method": spec.method,
                    "reason": "missing" if row is None else row.missing_reason,
                }
            )
            continue
        xs = [point[0] for point in row.trajectory_points]
        ys = [point[1] for point in row.trajectory_points]
        ax.plot(xs, ys, color=spec.color, linestyle=spec.linestyle, linewidth=1.55, alpha=0.95)
        marker_x = row.iteration if row.iteration is not None else xs[-1]
        marker_y = trajectory_error_at_k(row.trajectory_points, row.iteration) or row.abs_delta_e or ys[-1]
        ax.scatter(
            [marker_x],
            [marker_y],
            color=spec.color,
            marker=spec.marker,
            s=54 if spec.marker == "*" else 30,
            edgecolor="black",
            linewidth=0.35,
            zorder=4,
        )
        plotted.append([spec.matrix_label, regime, spec.method])
    ax.set_yscale("log")
    ax.set_xlabel("ADAPT outer iteration $k$", fontsize=8)
    ax.set_ylabel(r"$|\Delta E|$", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(REGIME_DISPLAY.get(regime, regime), fontsize=9)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
    legend_handles = [
        Line2D([0], [0], color=spec_by_role[spec.key].color, marker=spec_by_role[spec.key].marker,
               linestyle=spec_by_role[spec.key].linestyle, label=spec_by_role[spec.key].display,
               markersize=7 if spec_by_role[spec.key].marker == "*" else 5)
        for spec in ROLE_SPECS
        if spec.key not in omit_plot_roles
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=5.7, frameon=False)
    fig.tight_layout(pad=0.55)
    safe_regime = regime.replace("-", "_")
    png = fig_dir / f"{stem}__{safe_regime}.png"
    pdf = fig_dir / f"{stem}__{safe_regime}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "regime": regime,
        "figure_png": rel(png),
        "figure_png_sha256": sha256(png),
        "figure_pdf": rel(pdf),
        "figure_pdf_sha256": sha256(pdf),
        "plotted_curve_keys": plotted,
        "omitted_curve_keys": omitted,
        "role_styles": {spec.key: {"color": spec.color, "linestyle": spec.linestyle, "marker": spec.marker} for spec in ROLE_SPECS},
    }


def write_csv(path: Path, rows: Sequence[DerivedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "role_key",
        "role_display",
        "role_description",
        "matrix_label",
        "regime",
        "method",
        "optimizer",
        "status",
        "plotted",
        "missing_reason",
        "selection_policy",
        "selection_status",
        "selected_prefix_k",
        "record_id",
        "input_report_json",
        "input_report_json_sha256",
        "iteration",
        "depth",
        "abs_delta_e",
        "fidelity",
        "fidelity_status",
        "fidelity_source",
        "fidelity_status_detail",
        "N2q",
        "D2q",
        "Dc",
        "cost_source",
        "cost_status",
        "S_grad",
        "S_refit",
        "S_outer",
        "S_H",
        "S_metric",
        "S_alg",
        "s_work_status",
        "s_work_source",
        "s_work_status_detail",
        "trajectory_points_json",
        "source_json",
        "source_sha256",
        "source_dir",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "role_key": row.role_key,
                    "role_display": row.role_display,
                    "role_description": row.role_description,
                    "matrix_label": row.matrix_label,
                    "regime": row.regime,
                    "method": row.method,
                    "optimizer": row.optimizer,
                    "status": row.status,
                    "plotted": str(row.plotted).lower(),
                    "missing_reason": row.missing_reason,
                    "selection_policy": row.selection_policy,
                    "selection_status": row.selection_status,
                    "selected_prefix_k": "" if row.selected_prefix_k is None else row.selected_prefix_k,
                    "record_id": row.record_id,
                    "input_report_json": row.input_report_json,
                    "input_report_json_sha256": row.input_report_json_sha256,
                    "iteration": "" if row.iteration is None else row.iteration,
                    "depth": "" if row.depth is None else row.depth,
                    "abs_delta_e": "" if row.abs_delta_e is None else row.abs_delta_e,
                    "fidelity": "" if row.fidelity is None else row.fidelity,
                    "fidelity_status": row.fidelity_status,
                    "fidelity_source": row.fidelity_source,
                    "fidelity_status_detail": row.fidelity_status_detail,
                    "N2q": "" if row.n2q is None else row.n2q,
                    "D2q": "" if row.d2q is None else row.d2q,
                    "Dc": "" if row.dc is None else row.dc,
                    "cost_source": row.cost_source,
                    "cost_status": row.cost_status,
                    "S_grad": "" if row.s_grad is None else row.s_grad,
                    "S_refit": "" if row.s_refit is None else row.s_refit,
                    "S_outer": "" if row.s_outer is None else row.s_outer,
                    "S_H": "" if row.s_h is None else row.s_h,
                    "S_metric": "" if row.s_metric is None else row.s_metric,
                    "S_alg": "" if row.s_alg is None else row.s_alg,
                    "s_work_status": row.s_work_status,
                    "s_work_source": row.s_work_source,
                    "s_work_status_detail": row.s_work_status_detail,
                    "trajectory_points_json": json.dumps(row.trajectory_points, separators=(",", ":")),
                    "source_json": row.source_json,
                    "source_sha256": row.source_sha256,
                    "source_dir": row.source_dir,
                    "note": row.note,
                }
            )


def cost_table_tex(rows: Sequence[DerivedRow]) -> str:
    lines = [
        r"\begin{adjustbox}{max width=\columnwidth}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Role & $|\Delta E|$ & $k$ & $1-F$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S$ \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            tex_escape(row.role_display),
            fmt_float(row.abs_delta_e),
            "--" if row.iteration is None else str(row.iteration),
            "--" if row.fidelity is None else f"{1.0 - row.fidelity:.3e}",
            fmt_int(row.n2q),
            fmt_int(row.d2q),
            fmt_int(row.dc),
            fmt_int(row.s_alg),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"])
    return "\n".join(lines)


def work_table_tex(rows: Sequence[DerivedRow]) -> str:
    lines = [
        r"\begin{adjustbox}{max width=\columnwidth}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Role & $S_{\rm alg}$ & $S_{\rm grad}$ & $S_{\rm metric}$ & $S_H$ \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            tex_escape(row.role_display),
            fmt_int(row.s_alg),
            fmt_int(row.s_grad),
            fmt_int(row.s_metric),
            fmt_int(row.s_h),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"])
    return "\n".join(lines)


def tex_path(path: str, output_dir: Path) -> str:
    p = REPO_ROOT / path
    try:
        return str(p.relative_to(output_dir)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _manifest_value(
    value: Any,
    *,
    unavailable_reason: str = "not serialized by the source row",
) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return (
            r"\textbf{unavailable:} "
            + tex_escape(unavailable_reason)
        )
    return r"\textbf{source-bound:} " + tex_escape(value)


def _fixed_manifest_value(value: Any) -> str:
    return r"\textbf{fixed:} " + tex_escape(value)


def parameter_manifest_tex(rows: Sequence[DerivedRow], stem: str) -> str:
    """Render the final agent-facing parameter and provenance appendix."""

    lines = [
        r"\clearpage",
        r"\onecolumn",
        r"\appendix",
        r"\section*{Parameter and provenance manifest}",
        (
            r"\small Values are labelled \textbf{fixed} when set by this "
            r"report builder, \textbf{source-bound} when copied from an input "
            r"row, and \textbf{unavailable} when the input does not serialize "
            r"the field. Unavailable values are not inferred."
        ),
        r"\begin{longtable}{@{}p{0.24\textwidth}p{0.72\textwidth}@{}}",
        r"\toprule",
        r"Field & Normalized value \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Field & Normalized value (continued) \\",
        r"\midrule",
        r"\endhead",
        "Report schema & " + _fixed_manifest_value(SCHEMA) + r" \\",
        "Report identifier & " + _fixed_manifest_value(stem) + r" \\",
        "Report scope & "
        + _fixed_manifest_value(
            "Paper-I Hubbard--Holstein Powell pool-exposure support report; "
            "existing report sidecars only; no scientific rerun"
        )
        + r" \\",
        "Model scope & "
        + _fixed_manifest_value(
            "static Hubbard--Holstein adaptive comparison; regime labels are "
            "reported per source-bound row"
        )
        + r" \\",
        "Drive status & "
        + _fixed_manifest_value(
            "drive_enabled=false; static energy-report scope"
        )
        + r" \\",
        "Optimizer filter & " + _fixed_manifest_value("POWELL") + r" \\",
        "Selection contract & "
        + _fixed_manifest_value(
            f"{EFFECTIVE_PLATEAU_POLICY} through "
            "source_locked_historical_plateau_adapter_v1"
        )
        + r" \\",
        "Hamiltonian parameters & "
        + _manifest_value(
            None,
            unavailable_reason=(
                "explicit numeric Hamiltonian parameters are not serialized "
                "by this builder's normalized row"
            ),
        )
        + r" \\",
        "Working cutoff and seed & "
        + _manifest_value(
            None,
            unavailable_reason=(
                "cutoff and seed are not serialized by this builder's "
                "normalized row"
            ),
        )
        + r" \\",
        r"\midrule",
    ]
    for row in rows:
        record_label = (
            f"{row.regime} / {row.role_display or row.role_key or 'unavailable'}"
        )
        lines.extend(
            [
                r"\multicolumn{2}{@{}l}{\textbf{Record: "
                + tex_escape(record_label)
                + r"}} \\",
                "Method / role identity & "
                + _manifest_value(
                    (
                        f"method={row.method}; role_key={row.role_key}; "
                        f"matrix_label={row.matrix_label}"
                    )
                )
                + r" \\",
                "Route identity & "
                + _manifest_value(
                    None,
                    unavailable_reason=(
                        "route identity is not serialized by the source row; "
                        "the source-bound role identity is reported instead"
                    ),
                )
                + r" \\",
                "Optimizer & " + _manifest_value(row.optimizer) + r" \\",
                "Selection policy & "
                + _manifest_value(row.selection_policy)
                + r" \\",
                "Selected prefix & "
                + _manifest_value(
                    (
                        None
                        if row.selected_prefix_k is None
                        else f"selected_prefix_k={row.selected_prefix_k}"
                    ),
                    unavailable_reason=(
                        "no selected prefix is serialized for this row"
                    ),
                )
                + r" \\",
                "Selection status & "
                + _manifest_value(row.selection_status)
                + r" \\",
                "Source result path & "
                + _manifest_value(row.source_json)
                + r" \\",
                "Source result SHA-256 & "
                + _manifest_value(row.source_sha256)
                + r" \\",
                "Input report sidecar path & "
                + _manifest_value(row.input_report_json)
                + r" \\",
                "Input report sidecar SHA-256 & "
                + _manifest_value(row.input_report_json_sha256)
                + r" \\",
                "Cost provenance & "
                + _manifest_value(
                    (
                        f"source={row.cost_source}; status={row.cost_status}"
                        if row.cost_source or row.cost_status
                        else None
                    )
                )
                + r" \\",
                "Algorithmic-work provenance & "
                + _manifest_value(
                    (
                        f"source={row.s_work_source}; "
                        f"status={row.s_work_status}"
                        if row.s_work_source or row.s_work_status
                        else None
                    )
                )
                + r" \\",
                r"\addlinespace",
                r"\midrule",
            ]
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def write_tex(path: Path, rows: Sequence[DerivedRow], figures: Sequence[Mapping[str, Any]], stem: str) -> None:
    fig_by_regime = {str(fig["regime"]): fig for fig in figures}
    row_by_regime: dict[str, list[DerivedRow]] = {regime: [] for regime in REGIME_ORDER}
    role_index = {spec.key: idx for idx, spec in enumerate(ROLE_SPECS)}
    for row in rows:
        row_by_regime.setdefault(row.regime, []).append(row)
    for regime_rows in row_by_regime.values():
        regime_rows.sort(key=lambda row: role_index.get(row.role_key, 999))

    body: list[str] = []
    body.extend(
        [
            r"\documentclass[twocolumn,9pt]{article}",
            r"\usepackage[margin=0.43in,columnsep=0.24in]{geometry}",
            r"\usepackage{graphicx}",
            r"\usepackage{booktabs}",
            r"\usepackage{adjustbox}",
            r"\usepackage{longtable}",
            r"\usepackage{amsmath}",
            r"\usepackage{hyperref}",
            r"\usepackage{xcolor}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{2pt}",
            r"\pagestyle{plain}",
            r"\begin{document}",
            "% BEGIN_MACHINE_READABLE_POWELL_POOL_EXPOSURE_SUPPORT",
            f"% schema={SCHEMA}",
            "% source_policy=Powell-only existing full-meta singleton symmetry matrix report sidecars; no CHTC runs or resource recomputation.",
            "% row_policy=SNAKE=A_native_staged_singleton_hard_guard/snake; singleton Geo/append=B_common_phase0_singleton_hard_guard; macro Geo/append=C_macro_only.",
            f"% value_policy=selected-prefix diagnostics use {EFFECTIVE_PLATEAU_POLICY} through the source-locked historical adapter; Qiskit costs are compiled at selected prefix.",
            "% visible_policy=result pages retain the compact plot/table layout; a rendered parameter/provenance manifest is the final appendix.",
            "% END_MACHINE_READABLE_POWELL_POOL_EXPOSURE_SUPPORT",
            r"\scriptsize",
        ]
    )
    for regime in REGIME_ORDER:
        regime_rows = row_by_regime.get(regime, [])
        fig = fig_by_regime.get(regime)
        if regime == "strong-strong":
            # In two-column mode this advances to the next column, keeping the
            # final two regimes side-by-side after the standalone title block
            # is removed.
            body.append(r"\newpage")
        body.append(r"\begin{minipage}{\columnwidth}")
        body.append(r"\subsection*{" + tex_escape(REGIME_DISPLAY.get(regime, regime)) + "}")
        if fig:
            body.append(
                r"\includegraphics[width=\columnwidth]{"
                + tex_path(str(fig["figure_pdf"]), path.parent)
                + "}"
            )
        body.append(r"\vspace{-0.35em}")
        body.append(r"\textbf{Qiskit compiled cost / accuracy.}\par\noindent")
        body.append(cost_table_tex(regime_rows))
        notes = []
        for row in regime_rows:
            if not row.plotted:
                notes.append(f"{row.role_display}: {row.missing_reason or row.status}")
        if notes:
            body.append(r"{\tiny Missing/omitted: " + tex_escape("; ".join(notes)) + r".}")
        body.append(r"\end{minipage}\par")
        body.append(r"\vspace{0.75em}")
    body.append(parameter_manifest_tex(rows, stem))
    body.append(r"\end{document}")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def compile_tex(tex_path_: Path) -> tuple[str, str]:
    tools: list[tuple[str, list[str]]] = []
    if shutil.which("latexmk"):
        tools.append(("latexmk", ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path_.name]))
    if shutil.which("tectonic"):
        tools.append(("tectonic", ["tectonic", "--keep-logs", "--reruns", "2", tex_path_.name]))
    if shutil.which("pdflatex"):
        tools.append(("pdflatex", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path_.name]))
    if not tools:
        raise RuntimeError("No LaTeX compiler found: need latexmk, tectonic, or pdflatex")
    errors: list[str] = []
    for name, cmd in tools:
        try:
            if name == "pdflatex":
                for _ in range(2):
                    result = subprocess.run(
                        cmd,
                        cwd=tex_path_.parent,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(result.stdout[-4000:])
            else:
                result = subprocess.run(
                    cmd,
                    cwd=tex_path_.parent,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stdout[-4000:])
            return name, "ok"
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("LaTeX compilation failed:\n" + "\n".join(errors))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    stem = str(args.stem)
    regimes = tuple(args.regime or REGIME_ORDER)
    input_paths = discover_input_jsons(input_dir, tuple(Path(p) for p in (args.input_json or ())))
    if not input_paths:
        raise FileNotFoundError(f"No Powell matrix-report JSON sidecars found in {input_dir}")
    sidecars, selected, duplicates = select_rows(input_paths)
    rows, missing = make_derived_rows(selected, regimes)
    equivalence = singleton_equivalence(selected, regimes)
    omit_plot_roles = set(args.omit_plot_role or ())
    valid_role_keys = {spec.key for spec in ROLE_SPECS}
    unknown_omit = sorted(omit_plot_roles - valid_role_keys)
    if unknown_omit:
        raise ValueError(f"Unknown --omit-plot-role value(s): {', '.join(unknown_omit)}")

    fig_dir = output_dir / "figures" / stem
    figures = [plot_regime(rows, regime, fig_dir, stem, omit_plot_roles) for regime in regimes]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    tex_path_ = output_dir / f"{stem}.tex"
    json_path = output_dir / f"{stem}.json"
    pdf_path = output_dir / f"{stem}.pdf"
    write_csv(csv_path, rows)
    write_tex(tex_path_, rows, figures, stem)

    compile_status = "not_requested"
    compile_tool = None
    if not args.no_compile:
        compile_tool, compile_status = compile_tex(tex_path_)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Expected compiled PDF missing: {pdf_path}")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_policy": {
            "source": "existing_matrix_report_sidecars_only",
            "optimizer_filter": "POWELL",
            "cthc_or_run_launches": False,
            "raw_result_recompute": False,
            "trajectory_source": "input_report_json.rows[].trajectory_points",
            "selected_prefix_policy": {
                "source_locked_adapter": "source_locked_historical_plateau_adapter_v1",
                "policy": EFFECTIVE_PLATEAU_POLICY,
            },
            "cost_policy": "Qiskit costs are recompiled from linked source_json at selected prefix",
            "fidelity_policy": "Fidelity is emitted only when the selected prefix is the terminal row and terminal fidelity is present; nonterminal selected-prefix fidelities are blocked because prefix-optimized state/theta is not serialized in these artifacts.",
            "work_policy": "SNAKE S_alg uses snake_algorithmic_work_from_payload(scope=display_prefix); comparator S_alg is source-locked from emitted adapt_history and closed through canonical_paper_i_algorithmic_work",
            "manuscript_status": "support_artifact_only_not_inserted_into_paper",
        },
        "selection": {
            "regimes": list(regimes),
            "roles": [asdict(spec) for spec in ROLE_SPECS],
            "omitted_plot_role_keys": sorted(omit_plot_roles),
            "native_snake_source": A1,
            "singleton_comparator_source": B1,
            "macro_comparator_source": C_MACRO,
        },
        "input_sidecars": [asdict(item) for item in sidecars],
        "outputs": {
            "pdf": rel(pdf_path),
            "pdf_sha256": sha256(pdf_path) if pdf_path.exists() else None,
            "tex": rel(tex_path_),
            "tex_sha256": sha256(tex_path_),
            "csv": rel(csv_path),
            "csv_sha256": sha256(csv_path),
            "json": rel(json_path),
            "compile_tool": compile_tool,
            "compile_status": compile_status,
        },
        "figures": figures,
        "rows": [asdict(row) for row in rows],
        "missing_slots": missing,
        "duplicate_rows": [asdict(item) for item in duplicates],
        "singleton_hard_guard_equivalence_checks": equivalence,
        "interpretation_notes": [],
        "validation_warnings": [],
    }
    warnings = payload["validation_warnings"]
    notes = payload["interpretation_notes"]
    if missing:
        warnings.append(f"{len(missing)} expected role/regime rows missing")
    differing = [item for item in equivalence if not item.get("equivalent")]
    if differing:
        notes.append(
            f"{len(differing)} A1-vs-B1 singleton hard-guard comparator checks differ or are missing; "
            "B1 is used intentionally as the canonical common Phase-0 singleton comparator source."
        )
    write_json(json_path, payload)
    # Refresh output hash after JSON exists.
    payload["outputs"]["json_sha256"] = sha256(json_path)
    write_json(json_path, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--input-json", action="append", default=[])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--regime", action="append", choices=REGIME_ORDER)
    parser.add_argument(
        "--omit-plot-role",
        action="append",
        default=[],
        help="Role key to omit from iteration plots only; rows remain in CSV/JSON/Tex support tables.",
    )
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build(args)
    print(json.dumps(payload["outputs"], indent=2, sort_keys=True))
    if payload.get("validation_warnings"):
        print("validation_warnings:", json.dumps(payload["validation_warnings"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
