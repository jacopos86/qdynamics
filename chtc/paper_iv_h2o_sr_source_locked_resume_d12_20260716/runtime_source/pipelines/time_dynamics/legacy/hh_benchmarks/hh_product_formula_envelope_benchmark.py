#!/usr/bin/env python3
"""Benchmark-local product-formula envelope row for the HH L=2 t=8 anchor.

This module wraps ``hh_realtime_suzuki_overlay.run_overlay`` once with Suzuki
orders 1 and 2 on the validated 160-interval grid, then performs an offline
benchmark-local family selection.  Exact/reference errors are used only for this
row-level family selection and never feed controller decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.legacy.analysis import hh_realtime_suzuki_overlay as overlay
from pipelines.time_dynamics.legacy.hh_benchmarks import hh_suzuki_benchmark as suzuki_bench


SCHEMA_VERSION = "hh_product_formula_envelope_benchmark_v1"
DEFAULT_CASE_ID = "hh_l2_t8_anchor_v1"
METHOD_ID = "hh_td_pf_envelope_suzuki12_v1"
METHOD_KIND = "product_formula_envelope"
CANDIDATE_ORDERS: tuple[int, ...] = (1, 2)
SELECTION_METRIC = "mean_abs_energy_total_error"
SELECTION_RULE_FIELDS: tuple[str, ...] = (
    "mean_abs_energy_total_error",
    "max_abs_energy_total_error",
    "final_abs_energy_total_error",
    "state_at_time_2q",
    "full_horizon_2q",
    "order",
)
EXACT_SELECTION_POLICY = "benchmark_local_offline_family_selection_only_not_controller_input"


@dataclass(frozen=True)
class ProductFormulaEnvelopeBenchmarkCase:
    case_id: str
    controller_json: Path
    source_pdf: Path
    trotter_steps: int
    candidate_orders: tuple[int, ...] = CANDIDATE_ORDERS
    skip_pdf: bool = True
    export_compiled_circuits: bool = False
    backend_name: str | None = None
    seed_transpiler: int | None = None
    optimization_level: int | None = None
    preferred_fake_backends: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductFormulaEnvelopeBenchmarkRow:
    case_id: str
    method_id: str
    method_kind: str
    status: str
    selected_overlay_method: str
    selected_order: int
    selected_candidate_method_id: str | None
    selected_candidate_method_kind: str | None
    selection_metric: str
    selection_uses_exact_reference: bool
    selection_rule: tuple[str, ...]
    selection_policy: str
    candidate_count: int
    successful_candidate_count: int
    controller_json: str | None
    source_pdf: str | None
    seed_artifact_json: str | None
    drive_enabled: bool | None
    t_final: float | None
    num_times: int | None
    trotter_steps: int | None
    final_energy_total: float | None
    final_energy_total_exact: float | None
    final_abs_energy_total_error: float | None
    mean_abs_energy_total_error: float | None
    max_abs_energy_total_error: float | None
    state_at_time_scope: str | None
    state_at_time_basis: str | None
    state_at_time_2q: int | None
    state_at_time_depth: int | None
    state_at_time_size: int | None
    full_horizon_scope: str | None
    full_horizon_basis: str | None
    full_horizon_2q: int | None
    full_horizon_depth: int | None
    full_horizon_size: int | None
    full_horizon_horizon_2q: int | None
    full_horizon_depth_serial: int | None
    controller_state_scope: str | None
    controller_state_basis: str | None
    controller_state_2q: int | None
    controller_state_depth: int | None
    controller_state_size: int | None
    backend_name: str | None
    seed_transpiler: int | None
    optimization_level: int | None
    preferred_fake_backends: tuple[str, ...]
    export_compiled_circuits: bool
    artifact_overlay_json: str | None
    artifact_candidate_rows_json: str | None
    artifact_manifest_json: str | None
    artifact_rows_json: str | None
    artifact_summary_json: str | None
    compiled_circuit_dir: str | None
    exact_reference_method: str | None
    exact_steps_multiplier: Any
    exact_reference_controller_inputs: bool = False
    controller_decisions_modified: bool = False


@dataclass(frozen=True)
class _CaseRunRecord:
    case: ProductFormulaEnvelopeBenchmarkCase
    overlay_config: overlay.SuzukiOverlayConfig
    overlay_json: Path
    candidate_rows: tuple[dict[str, Any], ...]
    selected_candidate: dict[str, Any]
    row: dict[str, Any]


"Built Math: PF-envelope row = argmin_{order in {1,2}} (mean |dE|, max |dE|, final |dE|, state 2Q, horizon 2Q, order) on the offline exact-benchmark overlay; controller decisions are unchanged."
def _now_utc() -> str:
    return suzuki_bench._now_utc()


def _write_json(path: Path, payload: Any) -> Path:
    return suzuki_bench._write_json(path, payload)


def _maybe_float(value: Any) -> float | None:
    return suzuki_bench._maybe_float(value)


def _maybe_int(value: Any) -> int | None:
    return suzuki_bench._maybe_int(value)


def _maybe_bool(value: Any) -> bool | None:
    return suzuki_bench._maybe_bool(value)


def _as_optional_str(value: Any) -> str | None:
    return suzuki_bench._as_optional_str(value)


def default_cases() -> tuple[ProductFormulaEnvelopeBenchmarkCase, ...]:
    return (
        ProductFormulaEnvelopeBenchmarkCase(
            case_id=DEFAULT_CASE_ID,
            controller_json=overlay.DEFAULT_CONTROLLER_JSON,
            source_pdf=overlay.DEFAULT_SOURCE_PDF,
            trotter_steps=160,
            candidate_orders=CANDIDATE_ORDERS,
            skip_pdf=True,
            export_compiled_circuits=False,
        ),
    )


def _case_by_id(case_id: str) -> ProductFormulaEnvelopeBenchmarkCase:
    for case in default_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in default_cases())
    raise ValueError(f"unknown product-formula envelope benchmark case_id={case_id!r}; known cases: {known}")


def _parse_int_tuple(raw: str | Sequence[int] | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    chunks: Sequence[Any]
    if isinstance(raw, str):
        chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    else:
        chunks = list(raw)
    out: list[int] = []
    seen: set[int] = set()
    for chunk in chunks:
        order = int(chunk)
        if order not in {1, 2}:
            raise ValueError("Only Suzuki orders 1 and 2 are supported.")
        if order not in seen:
            seen.add(order)
            out.append(order)
    if not out:
        raise ValueError("At least one candidate order is required.")
    return tuple(out)


def _validate_case(case: ProductFormulaEnvelopeBenchmarkCase) -> None:
    orders = tuple(int(order) for order in case.candidate_orders)
    if orders != CANDIDATE_ORDERS:
        raise ValueError(f"{METHOD_ID} expects candidate_orders={CANDIDATE_ORDERS}, got {orders}")
    if int(case.trotter_steps) != 160:
        raise ValueError(f"{METHOD_ID} uses the validated 160-interval grid; got trotter_steps={case.trotter_steps!r}")


def _suzuki_case_for_case(case: ProductFormulaEnvelopeBenchmarkCase) -> suzuki_bench.SuzukiBenchmarkCase:
    _validate_case(case)
    return suzuki_bench.SuzukiBenchmarkCase(
        case_id=str(case.case_id),
        controller_json=Path(case.controller_json),
        source_pdf=Path(case.source_pdf),
        trotter_steps=int(case.trotter_steps),
        suzuki_orders=tuple(int(order) for order in case.candidate_orders),
        skip_pdf=bool(case.skip_pdf),
        export_compiled_circuits=bool(case.export_compiled_circuits),
        backend_name=case.backend_name,
        seed_transpiler=case.seed_transpiler,
        optimization_level=case.optimization_level,
        preferred_fake_backends=tuple(str(x) for x in case.preferred_fake_backends),
    )


def _candidate_method_id(order: int) -> str:
    if int(order) == 2:
        return suzuki_bench.METHOD_ID
    return f"hh_td_suzuki{int(order)}"


def _candidate_rows_from_payload(
    payload: Mapping[str, Any],
    *,
    case: ProductFormulaEnvelopeBenchmarkCase,
    overlay_json: Path,
    manifest_json: Path,
    candidate_rows_json: Path,
    summary_json: Path,
    preferred_fake_backends: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    _validate_case(case)
    rows: list[dict[str, Any]] = []
    for order in case.candidate_orders:
        order_int = int(order)
        rows.append(
            suzuki_bench._row_from_overlay_method_payload(
                payload,
                case_id=str(case.case_id),
                overlay_method=f"suzuki{order_int}",
                method_id=_candidate_method_id(order_int),
                method_kind=suzuki_bench.METHOD_KIND,
                expected_order=order_int,
                artifact_overlay_json=overlay_json,
                artifact_manifest_json=manifest_json,
                artifact_rows_json=candidate_rows_json,
                artifact_summary_json=summary_json,
                preferred_fake_backends=preferred_fake_backends,
            )
        )
    return tuple(rows)


def _finite_selection_value(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"candidate {row.get('method_id')!r} missing/non-finite selection field {field!r}: {value!r}"
        ) from exc
    if not math.isfinite(out):
        raise ValueError(
            f"candidate {row.get('method_id')!r} missing/non-finite selection field {field!r}: {value!r}"
        )
    return float(out)


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, int]:
    order = _maybe_int(row.get("order"))
    if order is None:
        raise ValueError(f"candidate {row.get('method_id')!r} missing order")
    return (
        _finite_selection_value(row, "mean_abs_energy_total_error"),
        _finite_selection_value(row, "max_abs_energy_total_error"),
        _finite_selection_value(row, "final_abs_energy_total_error"),
        _finite_selection_value(row, "state_at_time_2q"),
        _finite_selection_value(row, "full_horizon_2q"),
        int(order),
    )


def _select_candidate_row(candidate_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_order: dict[int, Mapping[str, Any]] = {}
    for row in candidate_rows:
        order = _maybe_int(row.get("order"))
        if order is None:
            raise ValueError(f"candidate {row.get('method_id')!r} missing order")
        if int(order) in by_order:
            raise ValueError(f"duplicate product-formula candidate for order={int(order)}")
        by_order[int(order)] = row
    for order in CANDIDATE_ORDERS:
        if int(order) not in by_order:
            raise ValueError(f"missing required product-formula candidate suzuki{int(order)}")
    ordered_rows = [by_order[int(order)] for order in CANDIDATE_ORDERS]
    return dict(min(ordered_rows, key=_selection_key))


def _final_row_from_selected_candidate(
    *,
    case: ProductFormulaEnvelopeBenchmarkCase,
    selected_candidate: Mapping[str, Any],
    candidate_count: int,
    successful_candidate_count: int,
    overlay_json: Path,
    manifest_json: Path,
    candidate_rows_json: Path,
    rows_json: Path,
    summary_json: Path,
) -> dict[str, Any]:
    selected_order = int(_finite_selection_value(selected_candidate, "order"))
    row = ProductFormulaEnvelopeBenchmarkRow(
        case_id=str(case.case_id),
        method_id=METHOD_ID,
        method_kind=METHOD_KIND,
        status="ok",
        selected_overlay_method=f"suzuki{selected_order}",
        selected_order=int(selected_order),
        selected_candidate_method_id=_as_optional_str(selected_candidate.get("method_id")),
        selected_candidate_method_kind=_as_optional_str(selected_candidate.get("method_kind")),
        selection_metric=SELECTION_METRIC,
        selection_uses_exact_reference=True,
        selection_rule=SELECTION_RULE_FIELDS,
        selection_policy=EXACT_SELECTION_POLICY,
        candidate_count=int(candidate_count),
        successful_candidate_count=int(successful_candidate_count),
        controller_json=_as_optional_str(selected_candidate.get("controller_json")),
        source_pdf=_as_optional_str(selected_candidate.get("source_pdf")),
        seed_artifact_json=_as_optional_str(selected_candidate.get("seed_artifact_json")),
        drive_enabled=_maybe_bool(selected_candidate.get("drive_enabled")),
        t_final=_maybe_float(selected_candidate.get("t_final")),
        num_times=_maybe_int(selected_candidate.get("num_times")),
        trotter_steps=_maybe_int(selected_candidate.get("trotter_steps")),
        final_energy_total=_maybe_float(selected_candidate.get("final_energy_total")),
        final_energy_total_exact=_maybe_float(selected_candidate.get("final_energy_total_exact")),
        final_abs_energy_total_error=_maybe_float(selected_candidate.get("final_abs_energy_total_error")),
        mean_abs_energy_total_error=_maybe_float(selected_candidate.get("mean_abs_energy_total_error")),
        max_abs_energy_total_error=_maybe_float(selected_candidate.get("max_abs_energy_total_error")),
        state_at_time_scope=_as_optional_str(selected_candidate.get("state_at_time_scope")),
        state_at_time_basis=_as_optional_str(selected_candidate.get("state_at_time_basis")),
        state_at_time_2q=_maybe_int(selected_candidate.get("state_at_time_2q")),
        state_at_time_depth=_maybe_int(selected_candidate.get("state_at_time_depth")),
        state_at_time_size=_maybe_int(selected_candidate.get("state_at_time_size")),
        full_horizon_scope=_as_optional_str(selected_candidate.get("full_horizon_scope")),
        full_horizon_basis=_as_optional_str(selected_candidate.get("full_horizon_basis")),
        full_horizon_2q=_maybe_int(selected_candidate.get("full_horizon_2q")),
        full_horizon_depth=_maybe_int(selected_candidate.get("full_horizon_depth")),
        full_horizon_size=_maybe_int(selected_candidate.get("full_horizon_size")),
        full_horizon_horizon_2q=_maybe_int(selected_candidate.get("full_horizon_horizon_2q")),
        full_horizon_depth_serial=_maybe_int(selected_candidate.get("full_horizon_depth_serial")),
        controller_state_scope=_as_optional_str(selected_candidate.get("controller_state_scope")),
        controller_state_basis=_as_optional_str(selected_candidate.get("controller_state_basis")),
        controller_state_2q=_maybe_int(selected_candidate.get("controller_state_2q")),
        controller_state_depth=_maybe_int(selected_candidate.get("controller_state_depth")),
        controller_state_size=_maybe_int(selected_candidate.get("controller_state_size")),
        backend_name=_as_optional_str(selected_candidate.get("backend_name")),
        seed_transpiler=_maybe_int(selected_candidate.get("seed_transpiler")),
        optimization_level=_maybe_int(selected_candidate.get("optimization_level")),
        preferred_fake_backends=tuple(str(x) for x in (selected_candidate.get("preferred_fake_backends") or ())),
        export_compiled_circuits=bool(selected_candidate.get("export_compiled_circuits", False)),
        artifact_overlay_json=str(overlay_json),
        artifact_candidate_rows_json=str(candidate_rows_json),
        artifact_manifest_json=str(manifest_json),
        artifact_rows_json=str(rows_json),
        artifact_summary_json=str(summary_json),
        compiled_circuit_dir=_as_optional_str(selected_candidate.get("compiled_circuit_dir")),
        exact_reference_method=_as_optional_str(selected_candidate.get("exact_reference_method")),
        exact_steps_multiplier=selected_candidate.get("exact_steps_multiplier"),
    )
    return suzuki_bench._jsonable(row)


def _run_case(
    case: ProductFormulaEnvelopeBenchmarkCase,
    *,
    output_dir: Path,
    manifest_json: Path,
    candidate_rows_json: Path,
    rows_json: Path,
    summary_json: Path,
    command: str,
) -> _CaseRunRecord:
    _validate_case(case)
    overlay_dir = Path(output_dir) / "overlay"
    overlay_json = overlay_dir / f"{case.case_id}.json"
    overlay_pdf = None if bool(case.skip_pdf) else overlay_dir / f"{case.case_id}.pdf"
    compiled_dir = (
        Path(output_dir) / "compiled_circuits" / case.case_id
        if bool(case.export_compiled_circuits)
        else None
    )
    config = suzuki_bench._overlay_config_for_case(
        _suzuki_case_for_case(case),
        output_json=overlay_json,
        output_pdf=overlay_pdf,
        compiled_circuit_dir=compiled_dir,
    )
    payload = overlay.run_overlay(config, command=command)
    _write_json(overlay_json, payload)
    candidate_rows = _candidate_rows_from_payload(
        payload,
        case=case,
        overlay_json=overlay_json,
        manifest_json=manifest_json,
        candidate_rows_json=candidate_rows_json,
        summary_json=summary_json,
        preferred_fake_backends=config.preferred_fake_backends,
    )
    selected_candidate = _select_candidate_row(candidate_rows)
    row = _final_row_from_selected_candidate(
        case=case,
        selected_candidate=selected_candidate,
        candidate_count=len(case.candidate_orders),
        successful_candidate_count=len(candidate_rows),
        overlay_json=overlay_json,
        manifest_json=manifest_json,
        candidate_rows_json=candidate_rows_json,
        rows_json=rows_json,
        summary_json=summary_json,
    )
    return _CaseRunRecord(
        case=case,
        overlay_config=config,
        overlay_json=overlay_json,
        candidate_rows=tuple(dict(row) for row in candidate_rows),
        selected_candidate=dict(selected_candidate),
        row=dict(row),
    )


def _required_hardware_scope_manifest() -> list[dict[str, str]]:
    return [
        {"method": f"suzuki{order}", "scope": scope}
        for order in CANDIDATE_ORDERS
        for scope in ("seed_plus_one_step_additive", "full_horizon_with_seed_prep")
    ] + [{"method": "controller", "scope": "controller_state_at_time"}]


def _manifest_payload(
    *,
    records: Sequence[_CaseRunRecord],
    output_dir: Path,
    manifest_json: Path,
    candidate_rows_json: Path,
    rows_json: Path,
    summary_json: Path,
    command: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "benchmark": "hh_product_formula_envelope_time_dynamics",
        "method_contract": {
            "method_id": METHOD_ID,
            "method_kind": METHOD_KIND,
            "candidate_overlay_methods": [f"suzuki{order}" for order in CANDIDATE_ORDERS],
            "candidate_orders": list(CANDIDATE_ORDERS),
            "trotter_steps": 160,
            "selection_rule": list(SELECTION_RULE_FIELDS),
            "selection_metric": SELECTION_METRIC,
            "selection_uses_exact_reference": True,
            "selection_policy": EXACT_SELECTION_POLICY,
            "controller_decisions_modified": False,
            "exact_reference_controller_inputs": False,
            "hardware_scope_policy": "fail_closed_required_report_rows",
            "required_hardware_scopes": _required_hardware_scope_manifest(),
        },
        "command": command,
        "output_dir": str(output_dir),
        "paths": {
            "manifest_json": str(manifest_json),
            "candidate_rows_json": str(candidate_rows_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
            "overlay_dir": str(Path(output_dir) / "overlay"),
        },
        "cases": [
            {
                "case": suzuki_bench._jsonable(record.case),
                "resolved_overlay_config": suzuki_bench._jsonable(record.overlay_config),
                "artifact_overlay_json": str(record.overlay_json),
                "candidate_count": len(record.candidate_rows),
                "selected_overlay_method": record.row.get("selected_overlay_method"),
                "selected_order": record.row.get("selected_order"),
            }
            for record in records
        ],
    }


def _summary_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    manifest_json: Path,
    candidate_rows_json: Path,
    rows_json: Path,
    summary_json: Path,
    command: str,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "benchmark": "hh_product_formula_envelope_time_dynamics",
        "command": command,
        "output_dir": str(output_dir),
        "row_count": int(len(rows)),
        "candidate_row_count": int(len(candidate_rows)),
        "status_counts": dict(sorted(status_counts.items())),
        "case_ids": [str(row.get("case_id")) for row in rows],
        "method_ids": [str(row.get("method_id")) for row in rows],
        "selected_candidates": [
            {
                "case_id": row.get("case_id"),
                "method_id": row.get("method_id"),
                "selected_overlay_method": row.get("selected_overlay_method"),
                "selected_order": row.get("selected_order"),
                "selected_candidate_method_id": row.get("selected_candidate_method_id"),
                "selection_metric": row.get("selection_metric"),
                "selection_uses_exact_reference": row.get("selection_uses_exact_reference"),
            }
            for row in rows
        ],
        "paths": {
            "manifest_json": str(manifest_json),
            "candidate_rows_json": str(candidate_rows_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
        },
        "key_metrics": [
            {
                "case_id": row.get("case_id"),
                "method_id": row.get("method_id"),
                "selected_order": row.get("selected_order"),
                "selected_overlay_method": row.get("selected_overlay_method"),
                "final_abs_energy_total_error": row.get("final_abs_energy_total_error"),
                "mean_abs_energy_total_error": row.get("mean_abs_energy_total_error"),
                "max_abs_energy_total_error": row.get("max_abs_energy_total_error"),
                "state_at_time_2q": row.get("state_at_time_2q"),
                "state_at_time_depth": row.get("state_at_time_depth"),
                "full_horizon_2q": row.get("full_horizon_2q"),
                "full_horizon_depth": row.get("full_horizon_depth"),
            }
            for row in rows
        ],
    }


def run_benchmark(
    *,
    cases: Sequence[ProductFormulaEnvelopeBenchmarkCase],
    output_dir: Path,
    command: str = "",
) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_json = root / "manifest.json"
    candidate_rows_json = root / "candidate_rows.json"
    rows_json = root / "rows.json"
    summary_json = root / "summary.json"
    root.mkdir(parents=True, exist_ok=True)

    records = [
        _run_case(
            case,
            output_dir=root,
            manifest_json=manifest_json,
            candidate_rows_json=candidate_rows_json,
            rows_json=rows_json,
            summary_json=summary_json,
            command=command,
        )
        for case in cases
    ]
    candidate_rows = [dict(row) for record in records for row in record.candidate_rows]
    rows = [dict(record.row) for record in records]
    manifest = _manifest_payload(
        records=records,
        output_dir=root,
        manifest_json=manifest_json,
        candidate_rows_json=candidate_rows_json,
        rows_json=rows_json,
        summary_json=summary_json,
        command=command,
    )
    summary = _summary_payload(
        rows=rows,
        candidate_rows=candidate_rows,
        output_dir=root,
        manifest_json=manifest_json,
        candidate_rows_json=candidate_rows_json,
        rows_json=rows_json,
        summary_json=summary_json,
        command=command,
    )

    _write_json(manifest_json, manifest)
    _write_json(candidate_rows_json, candidate_rows)
    _write_json(rows_json, rows)
    _write_json(summary_json, summary)
    return {
        "manifest": manifest,
        "candidate_rows": candidate_rows,
        "rows": rows,
        "summary": summary,
        "paths": {
            "manifest_json": str(manifest_json),
            "candidate_rows_json": str(candidate_rows_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the benchmark-local HH L2 t=8 Suzuki-1/2 product-formula envelope row."
    )
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-json", type=Path, default=None)
    parser.add_argument("--source-pdf", type=Path, default=None)
    parser.add_argument("--trotter-steps", type=int, default=None)
    parser.add_argument("--candidate-orders", type=str, default=None)
    parser.add_argument("--compile-backend-name", type=str, default=None)
    parser.add_argument("--compile-seed-transpiler", type=int, default=None)
    parser.add_argument("--compile-optimization-level", type=int, default=None)
    parser.add_argument("--compile-preferred-fake-backends", type=str, default=None)
    parser.add_argument("--export-compiled-circuits", action="store_true")
    parser.add_argument("--write-pdf", action="store_true", help="Write the overlay PDF; default is JSON-only.")
    return parser


def _case_from_args(args: argparse.Namespace) -> ProductFormulaEnvelopeBenchmarkCase:
    case = _case_by_id(str(args.case_id))
    preferred = suzuki_bench._parse_string_tuple(args.compile_preferred_fake_backends)
    candidate_orders = _parse_int_tuple(args.candidate_orders) if args.candidate_orders is not None else case.candidate_orders
    return replace(
        case,
        controller_json=Path(args.controller_json) if args.controller_json is not None else case.controller_json,
        source_pdf=Path(args.source_pdf) if args.source_pdf is not None else case.source_pdf,
        trotter_steps=int(args.trotter_steps) if args.trotter_steps is not None else case.trotter_steps,
        candidate_orders=candidate_orders,
        skip_pdf=not bool(args.write_pdf),
        export_compiled_circuits=bool(args.export_compiled_circuits),
        backend_name=str(args.compile_backend_name) if args.compile_backend_name is not None else case.backend_name,
        seed_transpiler=(
            int(args.compile_seed_transpiler)
            if args.compile_seed_transpiler is not None
            else case.seed_transpiler
        ),
        optimization_level=(
            int(args.compile_optimization_level)
            if args.compile_optimization_level is not None
            else case.optimization_level
        ),
        preferred_fake_backends=preferred or case.preferred_fake_backends,
    )


def _command_from_argv(argv: Sequence[str] | None) -> str:
    if argv is None:
        return " ".join([sys.executable, "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_product_formula_envelope_benchmark", *sys.argv[1:]])
    return " ".join(["python", "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_product_formula_envelope_benchmark", *map(str, argv)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    case = _case_from_args(args)
    command = _command_from_argv(argv)
    result = run_benchmark(cases=(case,), output_dir=Path(args.output_dir), command=command)
    row = result["rows"][0]
    print(f"manifest_json={result['paths']['manifest_json']}")
    print(f"candidate_rows_json={result['paths']['candidate_rows_json']}")
    print(f"rows_json={result['paths']['rows_json']}")
    print(f"summary_json={result['paths']['summary_json']}")
    print(f"artifact_overlay_json={row.get('artifact_overlay_json')}")
    print(f"selected_overlay_method={row.get('selected_overlay_method')}")
    print(f"selected_order={row.get('selected_order')}")
    print(f"final_abs_energy_total_error={row.get('final_abs_energy_total_error')}")
    print(f"mean_abs_energy_total_error={row.get('mean_abs_energy_total_error')}")
    print(f"max_abs_energy_total_error={row.get('max_abs_energy_total_error')}")
    print(f"state_at_time_2q={row.get('state_at_time_2q')}")
    print(f"state_at_time_depth={row.get('state_at_time_depth')}")
    print(f"full_horizon_2q={row.get('full_horizon_2q')}")
    print(f"full_horizon_depth={row.get('full_horizon_depth')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
