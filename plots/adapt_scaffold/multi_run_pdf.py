"""Standalone multi-run PDF report for ADAPT scaffold artifacts."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from docs.reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_text_page,
    require_matplotlib,
)
from plots.adapt_scaffold.artifact_reader import AdaptPlotArtifact, load_adapt_plot_artifact
from plots.adapt_scaffold.landscape import (
    build_scaffold_trajectory_dataset,
    trajectory_prerequisite_failure_reason,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_TERM_TOL = 1.0e-12


@dataclass(frozen=True)
class ReportRunTarget:
    label: str
    json_path: str
    math_reference: str


@dataclass(frozen=True)
class LoadedReportRun:
    target: ReportRunTarget
    artifact: AdaptPlotArtifact
    resolved_json_path: Path
    final_gap_abs: float | None
    final_energy: float | None
    ansatz_depth: int
    runtime_params: int | None
    logical_ops: int | None


@dataclass(frozen=True)
class StepEnergyRow:
    step_index: int
    energy_before_opt: float
    energy_after_opt: float
    shared_step_drop: float
    admission_count: int


@dataclass(frozen=True)
class TypeCatalogEntry:
    type_key: str
    type_token_math_body: str
    display_math_body: str
    short_math_body: str
    support_summary: str


@dataclass(frozen=True)
class AdmissionPoint:
    step_index: int
    record_index: int
    generator_id: str | None
    instance_math_body: str
    type_key: str
    type_token_math_body: str
    display_math_body: str
    shared_step_drop: float


@dataclass(frozen=True)
class FinalRayPoint:
    final_order_index: int
    ray_math_body: str
    type_key: str
    type_token_math_body: str
    display_math_body: str
    admission_step_index: int | None


@dataclass(frozen=True)
class RunSymbolicContext:
    step_rows: tuple[StepEnergyRow, ...]
    type_entries: tuple[TypeCatalogEntry, ...]
    admission_points: tuple[AdmissionPoint, ...]
    final_points: tuple[FinalRayPoint, ...]


DEFAULT_MATH_MD_STARTERS: tuple[ReportRunTarget, ...] = (
    ReportRunTarget(
        label="Raw HF-start winner",
        json_path="artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json",
        math_reference="Math.md / best raw HF-start winner",
    ),
    ReportRunTarget(
        label="Reduced-pool direct line",
        json_path="artifacts/json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_currentcode_20260321T171615Z.json",
        math_reference="Math.md / best reduced-pool direct line",
    ),
    ReportRunTarget(
        label="Windowed diagnostic sweep",
        json_path="artifacts/phase3_windowed_cost_sweep_20260320_143517/json/windowed_w4__seed7.json",
        math_reference="Math.md / windowed diagnostic family",
    ),
    ReportRunTarget(
        label="Strong beam best line",
        json_path=(
            "artifacts/agent_runs/20260403_hh_l2_u4_g05_beam_sweep_stage1/cases/"
            "beam_runtime_split-off__batching-off__lifetime-off__profile-tight__gamma-1.0/json/"
            "beam_runtime_split-off__batching-off__lifetime-off__profile-tight__gamma-1.0.json"
        ),
        math_reference="Math.md / strong-coupling beam sweep best case",
    ),
    ReportRunTarget(
        label="Weak beam diagnostic",
        json_path=(
            "artifacts/agent_runs/20260403_hh_l2_u05_g02_beam_sweep_stage1_parent_full_meta/cases/"
            "beam_runtime_split-off__batching-off__lifetime-off__profile-tight__gamma-1.0/json/"
            "beam_runtime_split-off__batching-off__lifetime-off__profile-tight__gamma-1.0.json"
        ),
        math_reference="Math.md / weak-coupling beam diagnostic",
    ),
)


def _resolve_artifact_json(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        repo_path = (REPO_ROOT / path).resolve()
        if repo_path.exists():
            path = repo_path
        else:
            path = path.resolve()
    if path.is_dir():
        result_json = path / "json" / "result.json"
        if result_json.exists():
            return result_json.resolve()
        raise FileNotFoundError(f"Could not resolve result.json under artifact directory: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Artifact JSON not found: {path}")
    return path.resolve()


def _abbreviate_path(path_like: str | Path, *, keep_tail_segments: int = 4) -> str:
    path = Path(path_like)
    parts = path.parts
    if len(parts) <= keep_tail_segments + 1:
        return str(path)
    return "/".join([parts[0], "...", *parts[-int(keep_tail_segments):]])


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    if limit <= 3 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _final_energy_and_gap(artifact: AdaptPlotArtifact) -> tuple[float | None, float | None]:
    adapt_vqe = artifact.adapt_vqe
    final_energy = None
    if adapt_vqe.get("energy") is not None:
        final_energy = float(adapt_vqe.get("energy"))
    elif artifact.history_steps:
        final_energy = float(artifact.history_steps[-1].energy_after_opt)

    gap = None
    if adapt_vqe.get("abs_delta_e") is not None:
        gap = float(adapt_vqe.get("abs_delta_e"))
    elif final_energy is not None and artifact.exact_gs_energy is not None:
        gap = abs(float(final_energy) - float(artifact.exact_gs_energy))
    return final_energy, gap


def load_report_runs(targets: Sequence[ReportRunTarget]) -> list[LoadedReportRun]:
    loaded: list[LoadedReportRun] = []
    for target in targets:
        resolved = _resolve_artifact_json(target.json_path)
        artifact = load_adapt_plot_artifact(resolved)
        final_energy, final_gap = _final_energy_and_gap(artifact)
        adapt_vqe = artifact.adapt_vqe
        loaded.append(
            LoadedReportRun(
                target=target,
                artifact=artifact,
                resolved_json_path=resolved,
                final_gap_abs=final_gap,
                final_energy=final_energy,
                ansatz_depth=int(adapt_vqe.get("ansatz_depth", len(artifact.final_operator_labels) or len(artifact.history_steps))),
                runtime_params=(
                    int(adapt_vqe.get("num_parameters"))
                    if adapt_vqe.get("num_parameters") is not None
                    else None
                ),
                logical_ops=(
                    int(len(artifact.final_operator_labels))
                    if artifact.final_operator_labels
                    else (
                        int(adapt_vqe.get("logical_num_parameters"))
                        if adapt_vqe.get("logical_num_parameters") is not None
                        else None
                    )
                ),
            )
        )
    return loaded


def _common_setting(runs: Sequence[LoadedReportRun], key: str) -> Any:
    values = [run.artifact.settings.get(key) for run in runs]
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else "mixed"


def _is_sequence_like(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _normalized_term_rows(raw_terms: Any) -> tuple[dict[str, Any], ...]:
    if not _is_sequence_like(raw_terms):
        return tuple()
    out: list[dict[str, Any]] = []
    for raw in raw_terms:
        if not isinstance(raw, Mapping):
            continue
        pauli = str(raw.get("pauli_exyz", "")).strip().lower()
        if pauli == "":
            continue
        out.append(
            {
                "pauli_exyz": pauli,
                "coeff_re": float(raw.get("coeff_re", 0.0) or 0.0),
                "coeff_im": float(raw.get("coeff_im", 0.0) or 0.0),
                "nq": int(raw.get("nq", len(pauli))),
            }
        )
    return tuple(out)


def _normalize_metadata_source(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "generator_id": None,
            "candidate_label": None,
            "template_id": None,
            "family_id": None,
            "support_sites": tuple(),
            "support_qubits": tuple(),
            "compile_metadata": {},
        }
    generator_meta = raw.get("generator_metadata", {})
    if not isinstance(generator_meta, Mapping):
        generator_meta = {}
    compile_metadata: dict[str, Any] = {}
    nested_compile = generator_meta.get("compile_metadata", {})
    if isinstance(nested_compile, Mapping):
        compile_metadata.update(dict(nested_compile))
    top_compile = raw.get("compile_metadata", {})
    if isinstance(top_compile, Mapping):
        compile_metadata.update(dict(top_compile))
    support_sites_raw = raw.get("support_sites", generator_meta.get("support_sites", []))
    support_qubits_raw = raw.get("support_qubits", generator_meta.get("support_qubits", []))
    return {
        "generator_id": (
            str(raw.get("generator_id"))
            if raw.get("generator_id") is not None
            else (
                str(generator_meta.get("generator_id"))
                if generator_meta.get("generator_id") is not None
                else None
            )
        ),
        "candidate_label": (
            str(raw.get("candidate_label"))
            if raw.get("candidate_label") is not None
            else (
                str(raw.get("generator_label"))
                if raw.get("generator_label") is not None
                else None
            )
        ),
        "template_id": (
            str(raw.get("template_id"))
            if raw.get("template_id") is not None
            else (
                str(generator_meta.get("template_id"))
                if generator_meta.get("template_id") is not None
                else None
            )
        ),
        "family_id": (
            str(raw.get("family_id"))
            if raw.get("family_id") is not None
            else (
                str(raw.get("candidate_family"))
                if raw.get("candidate_family") is not None
                else (
                    str(generator_meta.get("family_id"))
                    if generator_meta.get("family_id") is not None
                    else None
                )
            )
        ),
        "support_sites": tuple(int(x) for x in support_sites_raw) if _is_sequence_like(support_sites_raw) else tuple(),
        "support_qubits": tuple(int(x) for x in support_qubits_raw) if _is_sequence_like(support_qubits_raw) else tuple(),
        "compile_metadata": dict(compile_metadata),
    }


def _merge_normalized_metadata(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in ("generator_id", "candidate_label", "template_id", "family_id"):
        if update.get(key) is not None:
            merged[key] = update.get(key)
    for key in ("support_sites", "support_qubits"):
        if update.get(key):
            merged[key] = tuple(update.get(key))
        else:
            merged.setdefault(key, tuple())
    compile_metadata = {}
    if isinstance(base.get("compile_metadata"), Mapping):
        compile_metadata.update(dict(base.get("compile_metadata", {})))
    if isinstance(update.get("compile_metadata"), Mapping):
        compile_metadata.update(dict(update.get("compile_metadata", {})))
    merged["compile_metadata"] = compile_metadata
    return merged


def _history_metadata_lookup(artifact: AdaptPlotArtifact) -> dict[tuple[int, int], dict[str, Any]]:
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for step in artifact.history_steps:
        for record_index, raw in enumerate(step.selected_feature_rows, start=1):
            lookup[(int(step.step_index), int(record_index))] = _normalize_metadata_source(raw)
    return lookup


def _selected_metadata_lookups(artifact: AdaptPlotArtifact) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_label: dict[str, list[dict[str, Any]]] = {}
    raw_rows = artifact.continuation.get("selected_generator_metadata", [])
    if not _is_sequence_like(raw_rows):
        return by_id, by_label
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        normalized = _normalize_metadata_source(raw)
        generator_id = normalized.get("generator_id")
        label = normalized.get("candidate_label")
        if generator_id:
            by_id[str(generator_id)] = normalized
        if label:
            by_label.setdefault(str(label), []).append(normalized)
    return by_id, by_label


def _parameterization_blocks(artifact: AdaptPlotArtifact) -> list[dict[str, Any]]:
    parameterization = artifact.adapt_vqe.get("parameterization", {})
    if not isinstance(parameterization, Mapping):
        return []
    blocks_raw = parameterization.get("blocks", [])
    if not _is_sequence_like(blocks_raw):
        return []
    return [dict(block) for block in blocks_raw if isinstance(block, Mapping)]


def _terms_from_metadata(meta: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    compile_metadata = meta.get("compile_metadata", {}) if isinstance(meta, Mapping) else {}
    if not isinstance(compile_metadata, Mapping):
        return tuple()
    return _normalized_term_rows(compile_metadata.get("serialized_terms_exyz", []))


def _terms_from_parameterization_block(block: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    if not isinstance(block, Mapping):
        return tuple()
    return _normalized_term_rows(block.get("runtime_terms_exyz", []))


def _coeff_abs_to_latex(value: float) -> str:
    common = {
        1.0: "1",
        0.5: r"\frac{1}{2}",
        0.25: r"\frac{1}{4}",
        0.75: r"\frac{3}{4}",
        0.125: r"\frac{1}{8}",
    }
    abs_value = abs(float(value))
    for candidate, latex in common.items():
        if math.isclose(abs_value, candidate, rel_tol=0.0, abs_tol=1.0e-10):
            return latex
    if math.isclose(abs_value, round(abs_value), rel_tol=0.0, abs_tol=1.0e-10):
        return str(int(round(abs_value)))
    return f"{abs_value:.3g}"


def _pauli_word_to_math_body(pauli_word: str) -> str:
    word = str(pauli_word).strip().lower()
    if word == "":
        return "I"
    pieces: list[str] = []
    for idx, letter in enumerate(word):
        if letter == "e":
            continue
        qidx = int(len(word) - 1 - idx)
        pieces.append(f"{letter.upper()}_{{{qidx}}}")
    return r"\,".join(pieces) if pieces else "I"


def _serialized_term_to_math_body(term: Mapping[str, Any]) -> str:
    coeff_re = float(term.get("coeff_re", 0.0) or 0.0)
    coeff_im = float(term.get("coeff_im", 0.0) or 0.0)
    pauli_body = _pauli_word_to_math_body(str(term.get("pauli_exyz", "")))
    if abs(coeff_im) <= _TERM_TOL:
        if math.isclose(coeff_re, 1.0, rel_tol=0.0, abs_tol=1.0e-10):
            return pauli_body
        if math.isclose(coeff_re, -1.0, rel_tol=0.0, abs_tol=1.0e-10):
            return f"-{pauli_body}"
        coeff_body = _coeff_abs_to_latex(coeff_re)
        sign = "-" if coeff_re < 0.0 else ""
        return sign + coeff_body if pauli_body == "I" else f"{sign}{coeff_body}\\,{pauli_body}"
    if abs(coeff_re) <= _TERM_TOL:
        imag_abs = _coeff_abs_to_latex(coeff_im)
        imag_prefix = "" if imag_abs == "1" else imag_abs
        sign = "-" if coeff_im < 0.0 else ""
        imag_body = f"{sign}{imag_prefix}i"
        return imag_body if pauli_body == "I" else f"{imag_body}\\,{pauli_body}"
    coeff_body = f"({_coeff_abs_to_latex(coeff_re)}{'+' if coeff_im >= 0.0 else '-'}{_coeff_abs_to_latex(coeff_im)}i)"
    return coeff_body if pauli_body == "I" else f"{coeff_body}\\,{pauli_body}"


def _serialized_terms_to_math_body(terms: Sequence[Mapping[str, Any]], *, max_terms: int = 3) -> str:
    normalized_terms = list(_normalized_term_rows(terms))
    if not normalized_terms:
        return r"\mathcal{G}"
    pieces: list[str] = []
    for idx, term in enumerate(normalized_terms[: max(int(max_terms), 1)]):
        body = _serialized_term_to_math_body(term)
        if idx == 0:
            pieces.append(body)
        elif body.startswith("-"):
            pieces.append(" - " + body[1:].lstrip())
        else:
            pieces.append(" + " + body)
    if len(normalized_terms) > max(int(max_terms), 1):
        pieces.append(r" + \cdots")
    return "".join(pieces)


def _fallback_math_body(meta: Mapping[str, Any]) -> str:
    sites = meta.get("support_sites") if isinstance(meta, Mapping) else ()
    if sites:
        site_body = ",".join(str(int(site)) for site in sites)
        return rf"\mathcal{{G}}\left[\mathrm{{sites}}=\{{{site_body}\}}\right]"
    return r"\mathcal{G}"


def _support_summary(meta: Mapping[str, Any], terms: Sequence[Mapping[str, Any]]) -> str:
    compile_metadata = meta.get("compile_metadata", {}) if isinstance(meta, Mapping) else {}
    if not isinstance(compile_metadata, Mapping):
        compile_metadata = {}
    support_bits: list[str] = []
    if compile_metadata.get("has_boson_support") is True:
        support_bits.append("B")
    if compile_metadata.get("has_fermion_support") is True:
        support_bits.append("F")
    support = "+".join(support_bits) if support_bits else "-"
    sites = meta.get("support_sites") if isinstance(meta, Mapping) else ()
    site_text = "{" + ",".join(str(int(site)) for site in sites) + "}" if sites else "-"
    n_terms = int(compile_metadata.get("num_polynomial_terms", len(terms) or 0) or 0)
    return f"sites={site_text}, support={support}, terms={n_terms}"


def _canonical_type_key(terms: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], *, fallback_label: str) -> str:
    normalized_terms = list(_normalized_term_rows(terms))
    if normalized_terms:
        canonical = tuple(
            sorted(
                (
                    str(term["pauli_exyz"]),
                    round(float(term["coeff_re"]), 12),
                    round(float(term["coeff_im"]), 12),
                )
                for term in normalized_terms
            )
        )
        return repr(canonical)
    family = meta.get("family_id") if isinstance(meta, Mapping) else None
    template = meta.get("template_id") if isinstance(meta, Mapping) else None
    return f"fallback::{family}::{template}::{fallback_label}"


def _build_step_rows(artifact: AdaptPlotArtifact) -> tuple[StepEnergyRow, ...]:
    rows: list[StepEnergyRow] = []
    for step in artifact.history_steps:
        rows.append(
            StepEnergyRow(
                step_index=int(step.step_index),
                energy_before_opt=float(step.energy_before_opt),
                energy_after_opt=float(step.energy_after_opt),
                shared_step_drop=float(step.energy_before_opt - step.energy_after_opt),
                admission_count=int(max(len(step.selected_labels), step.batch_size, 1)),
            )
        )
    return tuple(rows)


def _build_run_symbolic_context(artifact: AdaptPlotArtifact) -> RunSymbolicContext:
    step_rows = _build_step_rows(artifact)
    step_drop_lookup = {row.step_index: row.shared_step_drop for row in step_rows}
    history_lookup = _history_metadata_lookup(artifact)
    selected_by_id, selected_by_label = _selected_metadata_lookups(artifact)
    parameterization_blocks = _parameterization_blocks(artifact)

    type_entries: list[TypeCatalogEntry] = []
    type_entries_by_key: dict[str, TypeCatalogEntry] = {}

    def ensure_type_entry(terms: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], *, fallback_label: str) -> TypeCatalogEntry:
        key = _canonical_type_key(terms, meta, fallback_label=fallback_label)
        entry = type_entries_by_key.get(key)
        if entry is not None:
            return entry
        index = len(type_entries) + 1
        full_math = _serialized_terms_to_math_body(terms, max_terms=3) if terms else _fallback_math_body(meta)
        short_math = _serialized_terms_to_math_body(terms, max_terms=2) if terms else full_math
        entry = TypeCatalogEntry(
            type_key=key,
            type_token_math_body=rf"\tau_{{{index}}}",
            display_math_body=full_math,
            short_math_body=short_math,
            support_summary=_support_summary(meta, terms),
        )
        type_entries_by_key[key] = entry
        type_entries.append(entry)
        return entry

    admission_points: list[AdmissionPoint] = []
    for record in artifact.admission_records:
        meta = history_lookup.get((int(record.step_index), int(record.record_index)), {})
        if record.generator_id is not None and str(record.generator_id) in selected_by_id:
            meta = _merge_normalized_metadata(meta, selected_by_id[str(record.generator_id)])
        elif record.generator_label and selected_by_label.get(str(record.generator_label)):
            meta = _merge_normalized_metadata(meta, selected_by_label[str(record.generator_label)][0])
        terms = _terms_from_metadata(meta)
        entry = ensure_type_entry(terms, meta, fallback_label=str(record.generator_label))
        admission_points.append(
            AdmissionPoint(
                step_index=int(record.step_index),
                record_index=int(record.record_index),
                generator_id=record.generator_id,
                instance_math_body=rf"G_{{{int(record.step_index)},{int(record.record_index)}}}",
                type_key=entry.type_key,
                type_token_math_body=entry.type_token_math_body,
                display_math_body=entry.display_math_body,
                shared_step_drop=float(step_drop_lookup.get(int(record.step_index), 0.0)),
            )
        )

    final_points: list[FinalRayPoint] = []
    for idx, record in enumerate(artifact.final_scaffold_records):
        meta: dict[str, Any] = {}
        if record.generator_id is not None and str(record.generator_id) in selected_by_id:
            meta = _merge_normalized_metadata(meta, selected_by_id[str(record.generator_id)])
        elif record.generator_label and selected_by_label.get(str(record.generator_label)):
            meta = _merge_normalized_metadata(meta, selected_by_label[str(record.generator_label)][0])
        block = parameterization_blocks[idx] if idx < len(parameterization_blocks) else None
        if isinstance(block, Mapping):
            meta = _merge_normalized_metadata(meta, _normalize_metadata_source(block))
        terms = _terms_from_parameterization_block(block)
        if not terms:
            terms = _terms_from_metadata(meta)
        entry = ensure_type_entry(terms, meta, fallback_label=str(record.generator_label))
        final_points.append(
            FinalRayPoint(
                final_order_index=int(record.final_order_index),
                ray_math_body=rf"G_{{{int(record.final_order_index)}}}",
                type_key=entry.type_key,
                type_token_math_body=entry.type_token_math_body,
                display_math_body=entry.display_math_body,
                admission_step_index=record.admission_step_index,
            )
        )

    return RunSymbolicContext(
        step_rows=tuple(step_rows),
        type_entries=tuple(type_entries),
        admission_points=tuple(admission_points),
        final_points=tuple(final_points),
    )


def _admission_points_by_step(context: RunSymbolicContext) -> dict[int, list[AdmissionPoint]]:
    grouped: dict[int, list[AdmissionPoint]] = {}
    for point in context.admission_points:
        grouped.setdefault(int(point.step_index), []).append(point)
    for points in grouped.values():
        points.sort(key=lambda item: item.record_index)
    return grouped


def _final_points_by_order(context: RunSymbolicContext) -> dict[int, FinalRayPoint]:
    return {int(point.final_order_index): point for point in context.final_points}


def _apply_symlog_for_step_drops(ax: Any, values: Sequence[float]) -> None:
    finite = np.asarray([abs(float(value)) for value in values if np.isfinite(float(value))], dtype=float)
    finite = finite[finite > 0.0]
    linthresh = max(float(np.min(finite)) / 2.0, 1.0e-8) if finite.size else 1.0e-8
    ax.set_yscale("symlog", linthresh=linthresh)


def _trim_lines(lines: Sequence[str], *, max_lines: int) -> list[str]:
    if len(lines) <= int(max_lines):
        return list(lines)
    omitted = int(len(lines) - max_lines + 1)
    return [*list(lines[: max_lines - 1]), f"... ({omitted} more lines)"]


def _render_text_columns(ax: Any, *, left_lines: Sequence[str], right_lines: Sequence[str], fontsize: float = 7.0) -> None:
    ax.axis("off")
    ax.text(
        0.01,
        0.98,
        "\n".join(_trim_lines(left_lines, max_lines=26)),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
    )
    ax.text(
        0.51,
        0.98,
        "\n".join(_trim_lines(right_lines, max_lines=26)),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
    )


def _report_source_label(runs: Sequence[LoadedReportRun]) -> str:
    if runs and all(run.target.math_reference == "explicit_cli_input" for run in runs):
        return "Explicit ADAPT artifact inputs"
    return "Diverse Math.md-linked ADAPT subset"


def _render_manifest_page(pdf: Any, runs: Sequence[LoadedReportRun]) -> None:
    lines = [
        "ADAPT scaffold multi-run report",
        "",
        "Parameter manifest",
        "",
        "Model family/name : Hubbard-Holstein",
        "Ansatz type(s)    : ADAPT scaffold / appended-generator history",
        "Drive enabled     : False",
        f"t                 : {_common_setting(runs, 't')}",
        f"U                 : {_common_setting(runs, 'u')}",
        f"dv                : {_common_setting(runs, 'dv')}",
        f"L                 : {_common_setting(runs, 'L')}",
        f"omega0            : {_common_setting(runs, 'omega0')}",
        f"g_ep              : {_common_setting(runs, 'g_ep')}",
        f"n_ph_max          : {_common_setting(runs, 'n_ph_max')}",
        f"boson_encoding    : {_common_setting(runs, 'boson_encoding')}",
        f"ordering          : {_common_setting(runs, 'ordering')}",
        f"boundary          : {_common_setting(runs, 'boundary')}",
        "",
        "Source set",
        f" - {_report_source_label(runs)} used for plotting",
        " - Includes raw winner, reduced-pool line, windowed diagnostic, strong beam, weak beam",
        "",
        "Included runs",
    ]
    for run in runs:
        rel_path = run.resolved_json_path
        try:
            rel_path = run.resolved_json_path.relative_to(REPO_ROOT)
        except ValueError:
            pass
        lines.append(f" - {run.target.label} | {run.target.math_reference} | {_abbreviate_path(rel_path)}")
    render_text_page(pdf, lines, fontsize=9.5, line_spacing=0.029, max_line_width=140)


def _render_summary_page(pdf: Any, runs: Sequence[LoadedReportRun]) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    rows = []
    for run in runs:
        rows.append(
            [
                _truncate(run.target.label, 24),
                _truncate(run.target.math_reference.replace("Math.md / ", ""), 28),
                ("--" if run.final_gap_abs is None else f"{run.final_gap_abs:.3e}"),
                str(run.logical_ops if run.logical_ops is not None else "--"),
                str(run.runtime_params if run.runtime_params is not None else "--"),
                str(run.ansatz_depth),
                str(len(run.artifact.history_steps)),
            ]
        )
    render_compact_table(
        ax,
        title=f"{_report_source_label(runs)} summary",
        col_labels=["Run", "Math.md ref", "|ΔE|", "Logical ops", "Runtime params", "Ansatz depth", "History steps"],
        rows=rows,
        fontsize=7.5,
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_energy_overlay_page(pdf: Any, runs: Sequence[LoadedReportRun]) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), height_ratios=[1.8, 1.0])

    ax = axes[0]
    for run in runs:
        step_rows = _build_step_rows(run.artifact)
        steps = np.asarray([row.step_index for row in step_rows], dtype=float)
        before = np.asarray([row.energy_before_opt for row in step_rows], dtype=float)
        after = np.asarray([row.energy_after_opt for row in step_rows], dtype=float)
        line = ax.plot(steps, after, marker="o", linewidth=1.8, label=run.target.label)[0]
        ax.plot(steps, before, linestyle="--", marker="o", linewidth=1.0, color=line.get_color(), alpha=0.55)
    ax.set_title("Per-step energy traces across runs")
    ax.set_xlabel("ADAPT selection step")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    ax.text(
        0.01,
        0.02,
        "solid = after local reoptimization, dashed = before local reoptimization",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    ax = axes[1]
    step_drop_values: list[float] = []
    for run in runs:
        step_rows = _build_step_rows(run.artifact)
        steps = np.asarray([row.step_index for row in step_rows], dtype=float)
        drops = np.asarray([row.shared_step_drop for row in step_rows], dtype=float)
        step_drop_values.extend(float(x) for x in drops)
        ax.plot(steps, drops, marker="o", linewidth=1.5, label=run.target.label)
    _apply_symlog_for_step_drops(ax, step_drop_values)
    ax.set_title("Shared step drop by step (zoom-friendly scale)")
    ax.set_xlabel("ADAPT selection step")
    ax.set_ylabel("shared step drop = E_before - E_after")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_single_run_analysis_page(pdf: Any, run: LoadedReportRun) -> None:
    require_matplotlib()
    plt = get_plt()
    context = _build_run_symbolic_context(run.artifact)
    step_points = _admission_points_by_step(context)
    type_entries = list(context.type_entries)
    type_index_by_key = {entry.type_key: idx for idx, entry in enumerate(type_entries, start=1)}

    fig = plt.figure(figsize=(11.0, 8.5))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.95], width_ratios=[1.0, 1.2])
    ax_energy = fig.add_subplot(grid[0, 0])
    ax_drop = fig.add_subplot(grid[1, 0])
    ax_3d = fig.add_subplot(grid[0:2, 1], projection="3d")
    ax_text = fig.add_subplot(grid[2, :])

    steps = np.asarray([row.step_index for row in context.step_rows], dtype=float)
    before = np.asarray([row.energy_before_opt for row in context.step_rows], dtype=float)
    after = np.asarray([row.energy_after_opt for row in context.step_rows], dtype=float)
    drops = np.asarray([row.shared_step_drop for row in context.step_rows], dtype=float)

    ax_energy.plot(steps, before, linestyle="--", marker="o", color="0.60", linewidth=1.1, label="Before")
    ax_energy.plot(steps, after, linestyle="-", marker="o", color="tab:blue", linewidth=1.8, label="After")
    if run.artifact.exact_gs_energy is not None:
        ax_energy.axhline(float(run.artifact.exact_gs_energy), linestyle=":", color="tab:orange", linewidth=1.4, label="Exact GS")
    for row in context.step_rows:
        points = step_points.get(int(row.step_index), [])
        if not points:
            continue
        label_body = ", ".join(point.instance_math_body for point in points)
        ax_energy.annotate(
            rf"${label_body}$",
            xy=(float(row.step_index), float(row.energy_after_opt)),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax_energy.set_title(f"{run.target.label}\nBefore vs after local reoptimization")
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("Energy")
    ax_energy.grid(alpha=0.25)
    ax_energy.legend(fontsize=7, loc="best")

    ax_drop.bar(steps, drops, color="tab:purple", alpha=0.75, width=0.65)
    ax_drop.plot(steps, drops, marker="o", linewidth=1.2, color="black")
    _apply_symlog_for_step_drops(ax_drop, drops)
    ax_drop.set_title("Shared step drop by step")
    ax_drop.set_xlabel("Step")
    ax_drop.set_ylabel("E_before - E_after")
    ax_drop.grid(alpha=0.25, axis="y")

    grouped_bars: dict[tuple[int, str], dict[str, Any]] = {}
    for point in context.admission_points:
        key = (int(point.step_index), str(point.type_key))
        bucket = grouped_bars.setdefault(
            key,
            {
                "step_index": int(point.step_index),
                "type_key": str(point.type_key),
                "shared_step_drop": float(point.shared_step_drop),
                "count": 0,
            },
        )
        bucket["count"] = int(bucket["count"]) + 1
    bar_rows = list(grouped_bars.values())
    for row in bar_rows:
        x = float(row["step_index"]) - 0.3
        y = float(type_index_by_key[row["type_key"]]) - 0.3
        z = 0.0
        dz = float(row["shared_step_drop"])
        color = "tab:blue" if dz >= 0.0 else "tab:red"
        ax_3d.bar3d(x, y, z, 0.6, 0.6, dz, color=color, alpha=0.65, shade=True)
        ax_3d.text(
            x + 0.3,
            y + 0.3,
            dz,
            f"x{int(row['count'])}",
            fontsize=7,
            ha="center",
            va="bottom",
        )
    ax_3d.set_title("Operator type vs step vs duplicated shared step drop")
    ax_3d.set_xlabel("Step")
    ax_3d.set_ylabel("Operator type")
    ax_3d.set_zlabel("duplicated shared step drop")
    ax_3d.set_xticks(list(range(1, len(context.step_rows) + 1)))
    ax_3d.set_yticks(list(range(1, len(type_entries) + 1)))
    ax_3d.set_yticklabels([rf"${entry.type_token_math_body}$" for entry in type_entries], fontsize=8)
    ax_3d.view_init(elev=24, azim=-58)
    ax_3d.text2D(
        0.02,
        0.96,
        "duplicate comparison only; no within-step attribution",
        transform=ax_3d.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )

    rel_path = run.resolved_json_path
    try:
        rel_path = run.resolved_json_path.relative_to(REPO_ROOT)
    except ValueError:
        pass
    step_lines = [
        run.target.label,
        f"math ref: {run.target.math_reference}",
        f"path: {rel_path}",
        f"|ΔE|: {'--' if run.final_gap_abs is None else f'{run.final_gap_abs:.6e}'}",
        f"ansatz_depth: {run.ansatz_depth}",
        f"logical_ops: {run.logical_ops if run.logical_ops is not None else '--'}",
        f"runtime_params: {run.runtime_params if run.runtime_params is not None else '--'}",
        "",
        "Step-to-generator map",
    ]
    for row in context.step_rows:
        points = step_points.get(int(row.step_index), [])
        if not points:
            continue
        point_parts = [rf"${point.instance_math_body} \in {point.type_token_math_body}$" for point in points]
        step_lines.append(f"s={int(row.step_index)}: " + ", ".join(point_parts))
    type_lines = [
        "Type catalog",
        r"$\tau_i$ is the operator type and bars use the shared step drop for the whole admitted step.",
        "",
    ]
    for entry in type_entries:
        type_lines.append(rf"${entry.type_token_math_body} \equiv {entry.display_math_body}$")
        type_lines.append(f"  {entry.support_summary}")
    _render_text_columns(ax_text, left_lines=step_lines, right_lines=type_lines, fontsize=6.7)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05, hspace=0.42, wspace=0.28)
    pdf.savefig(fig)
    plt.close(fig)


def _render_single_run_trajectory_page(pdf: Any, run: LoadedReportRun) -> None:
    reason = trajectory_prerequisite_failure_reason(run.artifact)
    if reason is not None:
        rel_path = run.resolved_json_path
        try:
            rel_path = run.resolved_json_path.relative_to(REPO_ROOT)
        except ValueError:
            pass
        render_text_page(
            pdf,
            [
                f"{run.target.label} - projected scaffold trajectory unavailable",
                "",
                f"math ref: {run.target.math_reference}",
                f"path: {rel_path}",
                f"skip reason: {reason}",
                "",
                "This page is preserved intentionally so the PDF page count stays stable even when older artifacts",
                "do not carry the parameterization/runtime-theta bundle required for the projected scaffold view.",
            ],
            fontsize=11,
            line_spacing=0.04,
            max_line_width=120,
        )
        return

    require_matplotlib()
    plt = get_plt()
    context = _build_run_symbolic_context(run.artifact)
    final_by_order = _final_points_by_order(context)
    dataset = build_scaffold_trajectory_dataset(run.artifact)

    fig = plt.figure(figsize=(11.0, 8.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.9, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    ax_text = fig.add_subplot(grid[1, 0])

    points = np.asarray(dataset.projected_points, dtype=float)
    color_values = (
        dataset.actual_step_energies
        if dataset.actual_step_energies is not None and np.isfinite(dataset.actual_step_energies[1:]).any()
        else np.arange(points.shape[0], dtype=float)
    )
    scatter = ax.scatter(points[:, 0], points[:, 1], c=color_values, cmap="viridis", s=82, zorder=3)
    fig.colorbar(scatter, ax=ax, shrink=0.82, label="Recorded step energy / point index")
    ax.scatter(points[0, 0], points[0, 1], color="black", marker="*", s=140, label="Origin", zorder=4)
    norm_denom = max(float(np.max(dataset.block_l2_norms)), 1.0e-12) if dataset.block_l2_norms.size else 1.0
    for idx, record in enumerate(dataset.final_scaffold_records, start=1):
        start = points[idx - 1]
        end = points[idx]
        width = 1.0 + 2.5 * (float(dataset.block_l2_norms[idx - 1]) / norm_denom)
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={"arrowstyle": "->", "linewidth": width, "alpha": 0.85, "color": "tab:blue"},
            zorder=2,
        )
        final_point = final_by_order.get(int(record.final_order_index))
        label_body = final_point.ray_math_body if final_point is not None else rf"G_{{{int(record.final_order_index)}}}"
        ax.text(end[0], end[1], rf"${label_body}$", fontsize=8, ha="left", va="bottom")
    ax.set_title(f"{run.target.label}\nProjected final scaffold trajectory")
    ax.set_xlabel("Projected axis 1")
    ax.set_ylabel("Projected axis 2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    trajectory_lines = [
        run.target.label,
        f"math ref: {run.target.math_reference}",
        f"projection: {dataset.projection_method}",
        f"runtime dimension: {int(dataset.prefix_runtime_points.shape[1])}",
        f"warnings: {', '.join(dataset.warnings) if dataset.warnings else '--'}",
        "",
        "Final scaffold ray map",
    ]
    for record in dataset.final_scaffold_records:
        final_point = final_by_order.get(int(record.final_order_index))
        if final_point is None:
            continue
        admission_text = (
            f"admitted at step {int(final_point.admission_step_index)}"
            if final_point.admission_step_index is not None
            else "admission step unavailable"
        )
        trajectory_lines.append(
            rf"${final_point.ray_math_body} \in {final_point.type_token_math_body}$ : {admission_text}"
        )
    type_lines = ["Type catalog", ""]
    seen_type_keys: set[str] = set()
    for point in context.final_points:
        if point.type_key in seen_type_keys:
            continue
        seen_type_keys.add(point.type_key)
        matching = next((entry for entry in context.type_entries if entry.type_key == point.type_key), None)
        if matching is None:
            continue
        type_lines.append(rf"${matching.type_token_math_body} \equiv {matching.display_math_body}$")
        type_lines.append(f"  {matching.support_summary}")
    _render_text_columns(ax_text, left_lines=trajectory_lines, right_lines=type_lines, fontsize=6.8)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.05, hspace=0.36)
    pdf.savefig(fig)
    plt.close(fig)


def build_multi_run_pdf(
    *,
    runs: Sequence[LoadedReportRun],
    output_pdf: str | Path,
) -> Path:
    require_matplotlib()
    output_path = Path(output_pdf).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PdfPages = get_PdfPages()
    with PdfPages(output_path) as pdf:
        _render_manifest_page(pdf, runs)
        _render_summary_page(pdf, runs)
        _render_energy_overlay_page(pdf, runs)
        for run in runs:
            _render_single_run_analysis_page(pdf, run)
            _render_single_run_trajectory_page(pdf, run)
        render_command_page(
            pdf,
            current_command_string(),
            script_name="plots.adapt_scaffold.multi_run_pdf",
            extra_header_lines=[
                f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
                f"Runs included: {len(runs)}",
            ],
        )
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multi-run ADAPT scaffold PDF report.")
    parser.add_argument(
        "--artifact-json",
        action="append",
        default=[],
        help="Optional explicit artifact JSON(s). If omitted, the Math.md starter set is used.",
    )
    parser.add_argument(
        "--output-pdf",
        default="output/pdf/adapt_scaffold_math_starter_runs.pdf",
        help="Output PDF path.",
    )
    return parser.parse_args(argv)


def _targets_from_args(args: argparse.Namespace) -> list[ReportRunTarget]:
    if args.artifact_json:
        return [
            ReportRunTarget(
                label=Path(path).stem,
                json_path=str(path),
                math_reference="explicit_cli_input",
            )
            for path in args.artifact_json
        ]
    return list(DEFAULT_MATH_MD_STARTERS)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        runs = load_report_runs(_targets_from_args(args))
        out = build_multi_run_pdf(
            runs=runs,
            output_pdf=args.output_pdf,
        )
        print(f"Wrote PDF report: {out}")
        print(f"Runs included: {len(runs)}")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
