"""Read-only normalization for ADAPT scaffold plotting artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    deserialize_layout,
    expand_legacy_logical_theta,
    project_runtime_theta_block_mean,
)


@dataclass(frozen=True)
class AdaptHistoryStep:
    step_index: int
    selected_labels: tuple[str, ...]
    selection_mode: str | None
    batch_size: int
    energy_before_opt: float
    energy_after_opt: float
    delta_energy: float
    delta_abs_drop_from_prev: float | None
    selected_feature_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AdaptAdmissionRecord:
    admission_order_index: int
    step_index: int
    record_index: int
    generator_label: str
    generator_id: str | None
    template_id: str | None
    family_id: str | None
    runtime_split_mode: str | None
    support_sites: tuple[int, ...]
    support_qubits: tuple[int, ...]
    num_polynomial_terms: int | None
    has_boson_support: bool | None
    has_fermion_support: bool | None


@dataclass(frozen=True)
class FinalScaffoldRecord:
    final_order_index: int
    generator_label: str
    generator_id: str | None
    template_id: str | None
    family_id: str | None
    runtime_start: int | None
    runtime_stop: int | None
    runtime_split_mode: str | None
    support_sites: tuple[int, ...]
    support_qubits: tuple[int, ...]
    num_polynomial_terms: int | None
    has_boson_support: bool | None
    has_fermion_support: bool | None
    admission_order_index: int | None = None
    admission_step_index: int | None = None
    admission_record_index: int | None = None


@dataclass(frozen=True)
class AdaptPlotArtifact:
    json_path: Path
    payload: dict[str, Any]
    settings: dict[str, Any]
    adapt_vqe: dict[str, Any]
    continuation: dict[str, Any]
    history_steps: tuple[AdaptHistoryStep, ...]
    admission_records: tuple[AdaptAdmissionRecord, ...]
    final_scaffold_records: tuple[FinalScaffoldRecord, ...]
    layout: AnsatzParameterLayout | None
    final_theta_runtime: np.ndarray | None
    final_theta_logical: np.ndarray | None
    final_operator_labels: tuple[str, ...]
    exact_gs_energy: float | None
    ansatz_input_state_vector: np.ndarray | None
    final_matching_complete: bool
    warnings: tuple[str, ...]


def _is_sequence_like(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _json_equivalent(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    return json.dumps(lhs, sort_keys=True, default=str) == json.dumps(rhs, sort_keys=True, default=str)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _string_list_from_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    labels_raw = row.get("selected_ops")
    if _is_sequence_like(labels_raw):
        labels = tuple(str(x) for x in labels_raw if str(x).strip() != "")
        if labels:
            return labels
    label_single = str(row.get("selected_op", "")).strip()
    if label_single:
        return (label_single,)
    raise ValueError("history row missing selected_op/selected_ops.")


def _feature_rows_from_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    feature_rows_raw = row.get("selected_feature_rows", [])
    if not _is_sequence_like(feature_rows_raw):
        return tuple()
    out: list[dict[str, Any]] = []
    for raw in feature_rows_raw:
        if isinstance(raw, Mapping):
            out.append(dict(raw))
    return tuple(out)


def _metadata_fields(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    meta = raw if isinstance(raw, Mapping) else {}
    generator_meta = meta.get("generator_metadata", {})
    if not isinstance(generator_meta, Mapping):
        generator_meta = {}
    compile_meta = generator_meta.get("compile_metadata", {})
    if not isinstance(compile_meta, Mapping):
        compile_meta = {}
    support_sites_raw = generator_meta.get("support_sites", [])
    support_qubits_raw = generator_meta.get("support_qubits", [])
    runtime_split = compile_meta.get("runtime_split", {})
    return {
        "generator_id": (
            str(meta.get("generator_id"))
            if meta.get("generator_id") is not None
            else (
                str(generator_meta.get("generator_id"))
                if generator_meta.get("generator_id") is not None
                else None
            )
        ),
        "template_id": (
            str(meta.get("template_id"))
            if meta.get("template_id") is not None
            else (
                str(generator_meta.get("template_id"))
                if generator_meta.get("template_id") is not None
                else None
            )
        ),
        "family_id": (
            str(meta.get("candidate_family"))
            if meta.get("candidate_family") is not None
            else (
                str(generator_meta.get("family_id"))
                if generator_meta.get("family_id") is not None
                else None
            )
        ),
        "runtime_split_mode": (
            str(meta.get("runtime_split_mode"))
            if meta.get("runtime_split_mode") is not None
            else (
                str(runtime_split.get("mode"))
                if isinstance(runtime_split, Mapping) and runtime_split.get("mode") is not None
                else None
            )
        ),
        "support_sites": tuple(int(x) for x in support_sites_raw) if _is_sequence_like(support_sites_raw) else tuple(),
        "support_qubits": tuple(int(x) for x in support_qubits_raw) if _is_sequence_like(support_qubits_raw) else tuple(),
        "num_polynomial_terms": (
            int(compile_meta.get("num_polynomial_terms"))
            if compile_meta.get("num_polynomial_terms") is not None
            else None
        ),
        "has_boson_support": (
            bool(compile_meta.get("has_boson_support"))
            if compile_meta.get("has_boson_support") is not None
            else None
        ),
        "has_fermion_support": (
            bool(compile_meta.get("has_fermion_support"))
            if compile_meta.get("has_fermion_support") is not None
            else None
        ),
    }


def _resolve_adapt_vqe(payload: Mapping[str, Any]) -> dict[str, Any]:
    nested = payload.get("adapt_vqe")
    if isinstance(nested, Mapping):
        return dict(nested)
    return dict(payload)


def _resolve_continuation(payload: Mapping[str, Any], adapt_vqe: Mapping[str, Any]) -> dict[str, Any]:
    top_level = payload.get("continuation")
    nested = adapt_vqe.get("continuation")
    top_map = dict(top_level) if isinstance(top_level, Mapping) else None
    nested_map = dict(nested) if isinstance(nested, Mapping) else None
    if top_map is not None and nested_map is not None and not _json_equivalent(top_map, nested_map):
        raise ValueError("Top-level continuation and adapt_vqe.continuation differ.")
    if nested_map is not None:
        return nested_map
    if top_map is not None:
        return top_map
    return {}


def _resolve_history_rows(payload: Mapping[str, Any], adapt_vqe: Mapping[str, Any]) -> list[dict[str, Any]]:
    history_raw = adapt_vqe.get("history")
    if not _is_sequence_like(history_raw):
        history_raw = payload.get("history")
    if not _is_sequence_like(history_raw):
        raise ValueError("Artifact is missing history.")
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(history_raw, start=1):
        if not isinstance(raw, Mapping):
            raise TypeError(f"history[{idx}] must be a mapping.")
        rows.append(dict(raw))
    if not rows:
        raise ValueError("Artifact history is empty.")
    return rows


def _extract_history_steps(history_rows: Sequence[Mapping[str, Any]]) -> tuple[AdaptHistoryStep, ...]:
    out: list[AdaptHistoryStep] = []
    seen_steps: set[int] = set()
    for idx, row in enumerate(history_rows, start=1):
        step_index = int(row.get("depth", idx))
        if step_index in seen_steps:
            raise ValueError(f"Duplicate history depth={step_index}.")
        seen_steps.add(step_index)
        labels = _string_list_from_row(row)
        energy_before_opt = _float_or_none(row.get("energy_before_opt"))
        energy_after_opt = _float_or_none(row.get("energy_after_opt"))
        if energy_before_opt is None or energy_after_opt is None:
            raise ValueError(f"history[{idx}] is missing finite energy_before_opt/energy_after_opt.")
        delta_energy = _float_or_none(row.get("delta_energy"))
        if delta_energy is None:
            delta_energy = float(energy_after_opt - energy_before_opt)
        batch_size = max(int(len(labels)), int(row.get("batch_size", 0) or 0), 1)
        out.append(
            AdaptHistoryStep(
                step_index=int(step_index),
                selected_labels=tuple(labels),
                selection_mode=(
                    str(row.get("selection_mode"))
                    if row.get("selection_mode") is not None
                    else None
                ),
                batch_size=int(batch_size),
                energy_before_opt=float(energy_before_opt),
                energy_after_opt=float(energy_after_opt),
                delta_energy=float(delta_energy),
                delta_abs_drop_from_prev=_float_or_none(row.get("delta_abs_drop_from_prev")),
                selected_feature_rows=_feature_rows_from_row(row),
            )
        )
    out.sort(key=lambda item: item.step_index)
    return tuple(out)


def _history_feature_lookup(history_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for idx, row in enumerate(history_rows, start=1):
        step_index = int(row.get("depth", idx))
        labels = _string_list_from_row(row)
        feature_rows = _feature_rows_from_row(row)
        for record_index, _label in enumerate(labels, start=1):
            if record_index <= len(feature_rows):
                lookup[(step_index, record_index)] = dict(feature_rows[record_index - 1])
    return lookup


def _validate_chain_counts(
    chain_rows: Sequence[Mapping[str, Any]],
    history_steps: Sequence[AdaptHistoryStep],
) -> None:
    counts_chain: dict[int, int] = {}
    for row in chain_rows:
        step_index = int(row.get("step_index", 0))
        counts_chain[step_index] = int(counts_chain.get(step_index, 0)) + 1
    for step in history_steps:
        expected = int(step.batch_size)
        actual = int(counts_chain.get(step.step_index, 0))
        if actual != expected:
            raise ValueError(
                f"selected_scaffold_record_chain disagrees with history for step {step.step_index}: "
                f"chain={actual}, history={expected}."
            )


def _build_admission_record(
    *,
    admission_order_index: int,
    step_index: int,
    record_index: int,
    generator_label: str,
    metadata_source: Mapping[str, Any] | None,
) -> AdaptAdmissionRecord:
    meta = _metadata_fields(metadata_source)
    return AdaptAdmissionRecord(
        admission_order_index=int(admission_order_index),
        step_index=int(step_index),
        record_index=int(record_index),
        generator_label=str(generator_label),
        generator_id=meta["generator_id"],
        template_id=meta["template_id"],
        family_id=meta["family_id"],
        runtime_split_mode=meta["runtime_split_mode"],
        support_sites=tuple(meta["support_sites"]),
        support_qubits=tuple(meta["support_qubits"]),
        num_polynomial_terms=meta["num_polynomial_terms"],
        has_boson_support=meta["has_boson_support"],
        has_fermion_support=meta["has_fermion_support"],
    )


def _extract_admission_records(
    *,
    continuation: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    history_steps: Sequence[AdaptHistoryStep],
) -> tuple[AdaptAdmissionRecord, ...]:
    feature_lookup = _history_feature_lookup(history_rows)
    chain_raw = continuation.get("selected_scaffold_record_chain")
    if _is_sequence_like(chain_raw):
        chain_rows = [dict(row) for row in chain_raw if isinstance(row, Mapping)]
        _validate_chain_counts(chain_rows, history_steps)
        out: list[AdaptAdmissionRecord] = []
        for admission_order_index, row in enumerate(chain_rows, start=1):
            step_index = int(row.get("step_index", 0))
            record_index = int(row.get("record_index", admission_order_index))
            feature_meta = feature_lookup.get((step_index, record_index), {})
            merged_meta = {**feature_meta, **dict(row)}
            out.append(
                _build_admission_record(
                    admission_order_index=admission_order_index,
                    step_index=step_index,
                    record_index=record_index,
                    generator_label=str(row.get("generator_label", "")),
                    metadata_source=merged_meta,
                )
            )
        return tuple(out)

    scaffold_history_raw = continuation.get("selected_scaffold_history")
    if _is_sequence_like(scaffold_history_raw):
        out = []
        admission_order_index = 1
        for step in scaffold_history_raw:
            if not isinstance(step, Mapping):
                continue
            step_index = int(step.get("step_index", admission_order_index))
            records_raw = step.get("selected_records", [])
            if not _is_sequence_like(records_raw):
                continue
            for per_step_index, record in enumerate(records_raw, start=1):
                if not isinstance(record, Mapping):
                    continue
                record_index = int(record.get("record_index", per_step_index))
                feature_meta = feature_lookup.get((step_index, record_index), {})
                merged_meta = {**feature_meta, **dict(record)}
                out.append(
                    _build_admission_record(
                        admission_order_index=admission_order_index,
                        step_index=step_index,
                        record_index=record_index,
                        generator_label=str(record.get("generator_label", "")),
                        metadata_source=merged_meta,
                    )
                )
                admission_order_index += 1
        return tuple(out)

    out = []
    admission_order_index = 1
    for step in history_steps:
        for record_index, label in enumerate(step.selected_labels, start=1):
            feature_meta = {}
            if record_index <= len(step.selected_feature_rows):
                feature_meta = dict(step.selected_feature_rows[record_index - 1])
            out.append(
                _build_admission_record(
                    admission_order_index=admission_order_index,
                    step_index=step.step_index,
                    record_index=record_index,
                    generator_label=str(label),
                    metadata_source=feature_meta,
                )
            )
            admission_order_index += 1
    return tuple(out)


def _resolve_exact_gs_energy(payload: Mapping[str, Any], adapt_vqe: Mapping[str, Any]) -> float | None:
    exact = _float_or_none(adapt_vqe.get("exact_gs_energy"))
    if exact is not None:
        return exact
    exact = _float_or_none(payload.get("exact_gs_energy"))
    if exact is not None:
        return exact
    ground_state = payload.get("ground_state", {})
    if isinstance(ground_state, Mapping):
        return _float_or_none(ground_state.get("exact_energy"))
    return None


def _resolve_ansatz_input_state_vector(payload: Mapping[str, Any]) -> np.ndarray | None:
    state_block = payload.get("ansatz_input_state")
    if not isinstance(state_block, Mapping):
        return None
    amplitudes = state_block.get("amplitudes_qn_to_q0")
    if not isinstance(amplitudes, Mapping) or not amplitudes:
        return None
    sample_key = str(next(iter(amplitudes.keys())))
    nq_total = int(state_block.get("nq_total", len(sample_key)))
    if nq_total <= 0:
        return None
    out = np.zeros(1 << int(nq_total), dtype=complex)
    for bitstr_raw, coeff_raw in amplitudes.items():
        bitstr = str(bitstr_raw).strip()
        if len(bitstr) != int(nq_total) or not isinstance(coeff_raw, Mapping):
            return None
        out[int(bitstr, 2)] = complex(float(coeff_raw.get("re", 0.0)), float(coeff_raw.get("im", 0.0)))
    norm = float(np.linalg.norm(out))
    if norm <= 0.0:
        return None
    return np.asarray(out / norm, dtype=complex)


def _resolve_layout_and_theta(
    adapt_vqe: Mapping[str, Any],
    *,
    warnings: list[str],
) -> tuple[AnsatzParameterLayout | None, np.ndarray | None, np.ndarray | None]:
    parameterization = adapt_vqe.get("parameterization")
    if not isinstance(parameterization, Mapping):
        return None, None, None
    layout = deserialize_layout(parameterization)
    optimal_point = np.asarray(adapt_vqe.get("optimal_point", []), dtype=float).reshape(-1)
    logical_optimal_point = np.asarray(adapt_vqe.get("logical_optimal_point", []), dtype=float).reshape(-1)

    theta_runtime: np.ndarray | None = None
    theta_logical: np.ndarray | None = None
    if int(optimal_point.size) == int(layout.runtime_parameter_count):
        theta_runtime = np.asarray(optimal_point, dtype=float)
    elif int(logical_optimal_point.size) == int(layout.logical_parameter_count):
        theta_runtime = np.asarray(expand_legacy_logical_theta(logical_optimal_point, layout), dtype=float)
    elif int(optimal_point.size) == int(layout.logical_parameter_count):
        warnings.append("optimal_point matched logical count; expanded as legacy logical theta.")
        theta_runtime = np.asarray(expand_legacy_logical_theta(optimal_point, layout), dtype=float)

    if int(logical_optimal_point.size) == int(layout.logical_parameter_count):
        theta_logical = np.asarray(logical_optimal_point, dtype=float)
    elif theta_runtime is not None:
        theta_logical = np.asarray(project_runtime_theta_block_mean(theta_runtime, layout), dtype=float)

    return layout, theta_runtime, theta_logical


def _resolve_final_operator_labels(adapt_vqe: Mapping[str, Any]) -> tuple[str, ...]:
    operators = adapt_vqe.get("operators", [])
    if not _is_sequence_like(operators):
        return tuple()
    return tuple(str(x) for x in operators)


def _align_selected_generator_metadata(
    final_operator_labels: Sequence[str],
    continuation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    selected_meta_raw = continuation.get("selected_generator_metadata", [])
    if not _is_sequence_like(selected_meta_raw):
        return [{} for _ in final_operator_labels], warnings
    selected_meta = [dict(row) for row in selected_meta_raw if isinstance(row, Mapping)]
    if not selected_meta:
        return [{} for _ in final_operator_labels], warnings
    if len(selected_meta) == len(final_operator_labels):
        aligned: list[dict[str, Any]] = []
        for idx, label in enumerate(final_operator_labels):
            meta = dict(selected_meta[idx])
            candidate_label = str(meta.get("candidate_label", label))
            if candidate_label != label:
                warnings.append(
                    f"selected_generator_metadata[{idx}] label mismatch: {candidate_label!r} != {label!r}; using label-based fallback."
                )
                break
            aligned.append(meta)
        else:
            return aligned, warnings

    queues: dict[str, list[dict[str, Any]]] = {}
    for meta in selected_meta:
        label = str(meta.get("candidate_label", "")).strip()
        if label == "":
            continue
        queues.setdefault(label, []).append(dict(meta))
    aligned = []
    for label in final_operator_labels:
        bucket = queues.get(label, [])
        aligned.append(bucket.pop(0) if bucket else {})
    return aligned, warnings


def _build_final_scaffold_records(
    *,
    final_operator_labels: Sequence[str],
    continuation: Mapping[str, Any],
    layout: AnsatzParameterLayout | None,
) -> tuple[tuple[FinalScaffoldRecord, ...], tuple[str, ...]]:
    aligned_meta, warning_list = _align_selected_generator_metadata(final_operator_labels, continuation)
    if layout is not None and int(layout.logical_parameter_count) != int(len(final_operator_labels)):
        raise ValueError(
            "parameterization logical count does not match final operator count: "
            f"{layout.logical_parameter_count} != {len(final_operator_labels)}."
        )
    records: list[FinalScaffoldRecord] = []
    for idx, label in enumerate(final_operator_labels):
        block = layout.blocks[idx] if layout is not None else None
        if block is not None and str(block.candidate_label) != str(label):
            raise ValueError(
                f"parameterization block {idx} label mismatch: {block.candidate_label!r} != {label!r}."
            )
        meta = aligned_meta[idx] if idx < len(aligned_meta) else {}
        meta_fields = _metadata_fields(meta)
        records.append(
            FinalScaffoldRecord(
                final_order_index=int(idx + 1),
                generator_label=str(label),
                generator_id=meta_fields["generator_id"],
                template_id=meta_fields["template_id"],
                family_id=meta_fields["family_id"],
                runtime_start=(int(block.runtime_start) if block is not None else None),
                runtime_stop=(int(block.runtime_stop) if block is not None else None),
                runtime_split_mode=meta_fields["runtime_split_mode"],
                support_sites=tuple(meta_fields["support_sites"]),
                support_qubits=tuple(meta_fields["support_qubits"]),
                num_polynomial_terms=meta_fields["num_polynomial_terms"],
                has_boson_support=meta_fields["has_boson_support"],
                has_fermion_support=meta_fields["has_fermion_support"],
            )
        )
    return tuple(records), tuple(warning_list)


def _match_final_records_to_admissions(
    final_records: Sequence[FinalScaffoldRecord],
    admission_records: Sequence[AdaptAdmissionRecord],
) -> tuple[tuple[FinalScaffoldRecord, ...], tuple[str, ...]]:
    warnings: list[str] = []
    matched: list[FinalScaffoldRecord] = []
    cursor = 0
    for record in final_records:
        match_pos: int | None = None
        if record.generator_id is not None:
            for pos in range(cursor, len(admission_records)):
                if admission_records[pos].generator_id == record.generator_id:
                    match_pos = pos
                    break
        if match_pos is None:
            for pos in range(cursor, len(admission_records)):
                if admission_records[pos].generator_label == record.generator_label:
                    match_pos = pos
                    break
        if match_pos is None:
            warnings.append(
                f"Could not match final scaffold record {record.final_order_index}:{record.generator_label} to admission history."
            )
            matched.append(record)
            continue
        admission = admission_records[match_pos]
        cursor = match_pos + 1
        matched.append(
            replace(
                record,
                admission_order_index=int(admission.admission_order_index),
                admission_step_index=int(admission.step_index),
                admission_record_index=int(admission.record_index),
            )
        )
    return tuple(matched), tuple(warnings)


def load_adapt_plot_artifact(json_path: str | Path) -> AdaptPlotArtifact:
    """Load a raw or exported ADAPT JSON artifact for plotting."""

    path = Path(json_path).expanduser().resolve()
    payload = _require_mapping(json.loads(path.read_text(encoding="utf-8")), name="artifact payload")
    adapt_vqe = _resolve_adapt_vqe(payload)
    continuation = _resolve_continuation(payload, adapt_vqe)
    history_rows = _resolve_history_rows(payload, adapt_vqe)
    history_steps = _extract_history_steps(history_rows)
    admission_records = _extract_admission_records(
        continuation=continuation,
        history_rows=history_rows,
        history_steps=history_steps,
    )
    warnings: list[str] = []
    layout, theta_runtime, theta_logical = _resolve_layout_and_theta(adapt_vqe, warnings=warnings)
    final_operator_labels = _resolve_final_operator_labels(adapt_vqe)
    final_scaffold_records, align_warnings = _build_final_scaffold_records(
        final_operator_labels=final_operator_labels,
        continuation=continuation,
        layout=layout,
    )
    matched_final_records, match_warnings = _match_final_records_to_admissions(
        final_scaffold_records,
        admission_records,
    )
    warnings.extend(list(align_warnings))
    warnings.extend(list(match_warnings))

    return AdaptPlotArtifact(
        json_path=path,
        payload=dict(payload),
        settings=_require_mapping(payload.get("settings", {}), name="settings"),
        adapt_vqe=dict(adapt_vqe),
        continuation=dict(continuation),
        history_steps=tuple(history_steps),
        admission_records=tuple(admission_records),
        final_scaffold_records=tuple(matched_final_records),
        layout=layout,
        final_theta_runtime=(None if theta_runtime is None else np.asarray(theta_runtime, dtype=float)),
        final_theta_logical=(None if theta_logical is None else np.asarray(theta_logical, dtype=float)),
        final_operator_labels=tuple(final_operator_labels),
        exact_gs_energy=_resolve_exact_gs_energy(payload, adapt_vqe),
        ansatz_input_state_vector=_resolve_ansatz_input_state_vector(payload),
        final_matching_complete=(len(match_warnings) == 0),
        warnings=tuple(warnings),
    )


__all__ = [
    "AdaptAdmissionRecord",
    "AdaptHistoryStep",
    "AdaptPlotArtifact",
    "FinalScaffoldRecord",
    "load_adapt_plot_artifact",
]
