#!/usr/bin/env python3
"""Paper III QSE table aggregation: per-manifest rows and multi-run tables.

RECONSTRUCTION (2026-08-18): the original module was never committed and was
lost when snapshot commit 6442fbb5 captured its importers without it. This
implementation is reconstructed against the committed behavioral specs in
``test/test_qse_table_aggregate.py`` and ``test/test_qse_exact_reference.py``.

``summarize_qse_manifest`` reduces one ``qse_spectra_v1`` manifest to a flat
table row: method identity, basis/rank/conditioning diagnostics, a matrix
measurement-work proxy, spectral-window reference errors, Paper III contract
and production-gate fields, response/conductivity/Green-function summaries,
and a controller-boundary check. ``build_qse_table_aggregate`` collects rows
across manifests; the CLI writes JSON/TSV/Markdown outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

QSE_TABLE_AGGREGATE_SCHEMA_VERSION = "paper_iii_qse_table_aggregate_v1"

_BOUNDARY_VIOLATION_FLAGS = (
    "feeds_controller_decisions",
    "controller_usable",
    "uses_exact_reference_for_decision",
    "uses_future_exact_forecast_for_decision",
    "reference_comparisons_feed_controller_decisions",
    "decision_path_allowed",
    "controller_decision_input",
    "uses_reference_for_decision",
)

_TSV_COLUMNS = (
    "row_id",
    "method_id",
    "run_class",
    "compatibility_tier",
    "approval_status",
    "basis_size",
    "retained_rank",
    "condition_number",
    "eigenvalue_count",
    "lowest_energy",
    "max_generalized_residual_norm",
    "matrix_measurement_proxy_total",
    "spectral_reference_l2_error_max",
    "spectral_reference_max_abs_error_max",
    "controller_boundary_status",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _opt_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _opt_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _opt_str(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"QSE manifest must be a JSON object: {path}")
    return dict(payload)


def _method_id(payload: Mapping[str, Any]) -> str | None:
    selection = _mapping(payload.get("static_record_selection"))
    mode = _opt_str(_mapping(selection.get("selection_config")).get("mode"))
    if mode is not None:
        return f"qse_selection::{mode}"
    basis_input = _mapping(_mapping(payload.get("input")).get("operator_basis"))
    source_schema = _opt_str(basis_input.get("source_schema"))
    if source_schema is not None:
        return f"qse_basis::{source_schema}"
    return None


def _matrix_measurement_proxy(payload: Mapping[str, Any]) -> dict[str, int] | None:
    diagnostics = _mapping(payload.get("diagnostics"))
    basis_size = _opt_int(diagnostics.get("basis_size"))
    if basis_size is None:
        return None
    pairs = basis_size * (basis_size + 1) // 2
    hamiltonian_terms = _opt_int(
        _mapping(_mapping(payload.get("input")).get("hamiltonian")).get("term_count_output")
    ) or 0
    basis_term_proxy = 0
    for element in _sequence(payload.get("operator_basis")):
        record = _mapping(element)
        if record.get("kind") == "pauli_polynomial":
            basis_term_proxy += len(_sequence(record.get("terms")))
        else:
            basis_term_proxy += 1
    transition_count = len(_sequence(payload.get("transition_observables")))
    overlap_entries = pairs
    hamiltonian_entries = pairs * hamiltonian_terms
    transition_entries = transition_count * basis_size
    return {
        "basis_pairs_upper_triangle": pairs,
        "hamiltonian_term_count": hamiltonian_terms,
        "basis_term_proxy": basis_term_proxy,
        "overlap_entries": overlap_entries,
        "hamiltonian_entries": hamiltonian_entries,
        "transition_entries": transition_entries,
        "total": overlap_entries + hamiltonian_entries + transition_entries,
    }


def _spectral_reference_errors(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _mapping(payload.get("spectral_window_metrics"))
    l1: list[float] = []
    l2: list[float] = []
    max_abs: list[float] = []
    metric_count = 0
    for observable in _sequence(metrics.get("observables")):
        for window in _sequence(_mapping(observable).get("window_metrics")):
            metric_count += 1
            comparison = _mapping(_mapping(window).get("reference_comparison"))
            for source, sink in (("l1_error", l1), ("l2_error", l2), ("max_abs_error", max_abs)):
                value = _opt_float(comparison.get(source))
                if value is not None:
                    sink.append(value)
    windows = metrics.get("windows")
    return {
        "spectral_reference_l1_error_max": _max_or_none(l1),
        "spectral_reference_l2_error_max": _max_or_none(l2),
        "spectral_reference_max_abs_error_max": _max_or_none(max_abs),
        "spectral_window_metric_count": metric_count,
        "spectral_window_count": len(_sequence(windows)) if windows is not None else None,
    }


def _controller_boundary_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    checked: list[str] = []
    failures: list[str] = []
    for key, section in payload.items():
        if not isinstance(section, Mapping):
            continue
        boundary = section.get("controller_boundary")
        if not isinstance(boundary, Mapping):
            continue
        checked.append(str(key))
        for flag in _BOUNDARY_VIOLATION_FLAGS:
            if boundary.get(flag) is True:
                failures.append(f"{key}.controller_boundary sets {flag}=true")
    if not checked:
        status = "missing"
        passed = None
    elif failures:
        status = "fail"
        passed = False
    else:
        status = "pass"
        passed = True
    return {
        "controller_boundary": {"checked_sections": checked, "failures": failures},
        "controller_boundary_status": status,
        "controller_boundary_passed": passed,
    }


def _response_function_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    response = payload.get("qse_response_functions_v1")
    if not isinstance(response, Mapping):
        return {
            "response_channel_count": None,
            "response_observable_count": None,
            "moment_deficit_status": "missing",
            "moment_deficit_summary": None,
            "moment_deficit_evaluated_channel_count": None,
            "moment_deficit_not_evaluated_channel_count": None,
            "moment_deficit_m0_abs_max": None,
            "moment_deficit_m1_abs_max": None,
        }
    channels = [_mapping(channel) for channel in _sequence(response.get("channels"))]
    status_counts: Counter[str] = Counter()
    m0: list[float] = []
    m1: list[float] = []
    for channel in channels:
        deficits = _mapping(channel.get("sum_rule_deficits"))
        status = _opt_str(deficits.get("status"))
        if status is not None:
            status_counts[status] += 1
        for order, sink in (("m0", m0), ("m1", m1)):
            value = _opt_float(_mapping(deficits.get(order)).get("deficit_abs"))
            if value is not None:
                sink.append(value)
    return {
        "response_channel_count": len(channels),
        "response_observable_count": len(_sequence(response.get("observables"))),
        "moment_deficit_status": "present",
        "moment_deficit_summary": {"status_counts": dict(status_counts)},
        "moment_deficit_evaluated_channel_count": status_counts.get("evaluated", 0),
        "moment_deficit_not_evaluated_channel_count": status_counts.get("not_evaluated", 0),
        "moment_deficit_m0_abs_max": _max_or_none(m0),
        "moment_deficit_m1_abs_max": _max_or_none(m1),
    }


def _conductivity_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    conductivity = payload.get("qse_conductivity_response_v1")
    if not isinstance(conductivity, Mapping):
        return {
            "conductivity_response_present": False,
            "conductivity_schema_version": None,
            "conductivity_channel_count": None,
            "conductivity_observable_count": None,
            "conductivity_contact_supplied_channel_count": None,
            "conductivity_zero_current_source_count": None,
            "conductivity_summary": None,
        }
    channels = [_mapping(channel) for channel in _sequence(conductivity.get("channels"))]
    contact_counts: Counter[str] = Counter()
    drude_counts: Counter[str] = Counter()
    contact_supplied = 0
    zero_current = 0
    for channel in channels:
        contact_status = _opt_str(_mapping(channel.get("contact_term")).get("status"))
        if contact_status is not None:
            contact_counts[contact_status] += 1
            if contact_status == "evaluated":
                contact_supplied += 1
        drude_status = _opt_str(_mapping(channel.get("drude_weight")).get("status"))
        if drude_status is not None:
            drude_counts[drude_status] += 1
        if _mapping(channel.get("current_source")).get("zero_current_source") is True:
            zero_current += 1
    return {
        "conductivity_response_present": True,
        "conductivity_schema_version": _opt_str(conductivity.get("schema_version")),
        "conductivity_channel_count": len(channels),
        "conductivity_observable_count": len(_sequence(conductivity.get("observables"))),
        "conductivity_contact_supplied_channel_count": contact_supplied,
        "conductivity_zero_current_source_count": zero_current,
        "conductivity_summary": {
            "contact_status_counts": dict(contact_counts),
            "drude_status_counts": dict(drude_counts),
        },
    }


def _green_function_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    green = payload.get("qse_green_function_v1")
    if not isinstance(green, Mapping):
        return {
            "green_function_present": False,
            "green_function_schema_version": None,
            "green_function_mode_count": None,
            "green_function_sector_count": None,
            "green_function_solved_sector_count": None,
            "green_function_zero_source_sector_count": None,
            "green_function_source_norm_canonical_deficit_abs_max": None,
            "green_function_residue_canonical_deficit_abs_max": None,
        }
    summary = _mapping(green.get("summary"))
    source_norm: list[float] = []
    residue: list[float] = []
    for mode in _sequence(green.get("modes")):
        diagnostics = _mapping(_mapping(mode).get("diagonal_sum_rule_diagnostics"))
        for key, sink in (
            ("source_norm_canonical_deficit_abs", source_norm),
            ("residue_canonical_deficit_abs", residue),
        ):
            value = _opt_float(diagnostics.get(key))
            if value is not None:
                sink.append(value)
    return {
        "green_function_present": True,
        "green_function_schema_version": _opt_str(green.get("schema_version")),
        "green_function_mode_count": _opt_int(summary.get("mode_count")),
        "green_function_sector_count": _opt_int(summary.get("sector_count")),
        "green_function_solved_sector_count": _opt_int(summary.get("solved_sector_count")),
        "green_function_zero_source_sector_count": _opt_int(summary.get("zero_source_sector_count")),
        "green_function_source_norm_canonical_deficit_abs_max": _max_or_none(source_norm),
        "green_function_residue_canonical_deficit_abs_max": _max_or_none(residue),
    }


def summarize_qse_manifest(path: Any) -> dict[str, Any]:
    """Reduce one qse_spectra manifest to a flat aggregate table row."""

    manifest_path = Path(path)
    payload = _load_manifest(manifest_path)
    diagnostics = _mapping(payload.get("diagnostics"))
    eigenvalues = [_mapping(entry) for entry in _sequence(payload.get("eigenvalues"))]
    energies = [value for value in (_opt_float(entry.get("energy")) for entry in eigenvalues) if value is not None]
    residuals = [
        value
        for value in (_opt_float(entry.get("generalized_residual_norm")) for entry in eigenvalues)
        if value is not None
    ]

    contract = _mapping(payload.get("paper_iii_contract"))
    selection = _mapping(payload.get("static_record_selection"))
    selection_config = _mapping(selection.get("selection_config"))
    cutoff = _mapping(payload.get("cutoff_boundary_diagnostics"))
    cutoff_layout = _mapping(cutoff.get("layout"))
    cutoff_roots = [_mapping(root) for root in _sequence(cutoff.get("roots"))]
    gate = _mapping(payload.get("paper_iii_production_gate"))
    spectral_functions = _mapping(payload.get("spectral_functions"))

    n_ph_max = _opt_int(
        _mapping(_mapping(contract.get("hh_full_meta_provenance")).get("layout")).get("n_ph_max")
    )
    target_root_count = _opt_int(selection_config.get("geometry_target_roots"))
    overlap_condition = _opt_float(diagnostics.get("overlap_condition_estimate"))

    row: dict[str, Any] = {
        "row_id": manifest_path.stem,
        "source_path": str(manifest_path),
        "schema_version": _opt_str(payload.get("schema_version")),
        "method_id": _method_id(payload),
        "basis_size": _opt_int(diagnostics.get("basis_size")),
        "retained_rank": _opt_int(diagnostics.get("retained_rank")),
        "discarded_rank": _opt_int(diagnostics.get("discarded_rank")),
        "overlap_condition_estimate": overlap_condition,
        "condition_number": overlap_condition,
        "eigenvalue_count": len(eigenvalues),
        "lowest_energy": min(energies) if energies else None,
        "max_generalized_residual_norm": _max_or_none(residuals),
        "matrix_measurement_proxy": _matrix_measurement_proxy(payload),
        "run_class": _opt_str(contract.get("run_class")),
        "approval_status": _opt_str(contract.get("approval_status")),
        "compatibility_tier": _opt_str(contract.get("compatibility_tier")),
        "visible_target": _opt_str(contract.get("visible_target")),
        "n_ph_max": n_ph_max,
        "n_ph_source": (
            "paper_iii_contract.hh_full_meta_provenance.layout.n_ph_max"
            if n_ph_max is not None
            else None
        ),
        "target_root_count": target_root_count,
        "target_root_count_source": (
            "static_record_selection.selection_config.geometry_target_roots"
            if target_root_count is not None
            else None
        ),
        "cutoff_n_ph_max": _opt_int(cutoff_layout.get("n_ph_max")),
        "cutoff_num_sites": _opt_int(cutoff_layout.get("num_sites")),
        "cutoff_boson_encoding": _opt_str(cutoff_layout.get("boson_encoding")),
        "cutoff_root_count": len(cutoff_roots) if cutoff else None,
        "cutoff_ell_cut_max": _max_or_none(
            [value for value in (_opt_float(root.get("ell_cut")) for root in cutoff_roots) if value is not None]
        ),
        "spectral_function_observable_count": (
            len(_sequence(spectral_functions.get("observables"))) if spectral_functions else None
        ),
        "production_gate_ok": gate.get("ok") if gate else None,
        "production_gate_production_ready": gate.get("production_ready") if gate else None,
        "production_gate_required_target_excited_roots": (
            _opt_int(gate.get("required_target_excited_roots")) if gate else None
        ),
        "production_gate_exact_reference_boundary_status": (
            _opt_str(_mapping(gate.get("exact_reference_boundary_status")).get("status")) if gate else None
        ),
    }
    row.update(_spectral_reference_errors(payload))
    row.update(_controller_boundary_row(payload))
    row.update(_response_function_row(payload))
    row.update(_conductivity_row(payload))
    row.update(_green_function_row(payload))
    return row


@dataclass(frozen=True)
class QSETableAggregateConfig:
    """Aggregate-build configuration: manifest paths and optional outputs."""

    qse_manifest_paths: tuple[Any, ...] = ()
    output_json: Any | None = None
    output_tsv: Any | None = None
    output_md: Any | None = None
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


def _write_json_output(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tsv_cell(row: Mapping[str, Any], column: str) -> str:
    if column == "matrix_measurement_proxy_total":
        proxy = row.get("matrix_measurement_proxy")
        value = _mapping(proxy).get("total") if isinstance(proxy, Mapping) else None
    else:
        value = row.get(column)
    return "" if value is None else str(value)


def _write_tsv_output(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(_TSV_COLUMNS)]
    for row in rows:
        lines.append("\t".join(_tsv_cell(row, column) for column in _TSV_COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_md_output(path: Path, aggregate: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_mapping(row) for row in _sequence(aggregate.get("rows"))]
    lines = [
        "# Paper III QSE table aggregate",
        "",
        f"Rows: {len(rows)}",
        "",
        "| " + " | ".join(_TSV_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(_TSV_COLUMNS)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_tsv_cell(row, column) for column in _TSV_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_qse_table_aggregate(config: QSETableAggregateConfig) -> dict[str, Any]:
    """Build the aggregate payload (and write configured outputs)."""

    rows = [summarize_qse_manifest(path) for path in config.qse_manifest_paths]
    method_ids = sorted({row["method_id"] for row in rows if row.get("method_id")})
    aggregate: dict[str, Any] = {
        "schema_version": QSE_TABLE_AGGREGATE_SCHEMA_VERSION,
        "pipeline": "qse_table_aggregate",
        "rows": rows,
        "summary": {
            "row_count": len(rows),
            "method_ids": method_ids,
            "run_classes": sorted({row["run_class"] for row in rows if row.get("run_class")}),
            "compatibility_tiers": sorted(
                {row["compatibility_tier"] for row in rows if row.get("compatibility_tier")}
            ),
            "rows_with_conductivity_payload": sum(
                1 for row in rows if row.get("conductivity_response_present") is True
            ),
            "rows_with_green_function_payload": sum(
                1 for row in rows if row.get("green_function_present") is True
            ),
        },
    }
    if config.extra_metadata:
        aggregate["metadata"] = dict(config.extra_metadata)
    if config.output_json is not None:
        _write_json_output(Path(config.output_json), aggregate)
    if config.output_tsv is not None:
        _write_tsv_output(Path(config.output_tsv), rows)
    if config.output_md is not None:
        _write_md_output(Path(config.output_md), aggregate)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate qse_spectra manifests into a Paper III table payload."
    )
    parser.add_argument("--qse-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-tsv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_qse_table_aggregate(
        QSETableAggregateConfig(
            qse_manifest_paths=tuple(args.qse_manifest),
            output_json=args.output_json,
            output_tsv=args.output_tsv,
            output_md=args.output_md,
        )
    )
    return 0
