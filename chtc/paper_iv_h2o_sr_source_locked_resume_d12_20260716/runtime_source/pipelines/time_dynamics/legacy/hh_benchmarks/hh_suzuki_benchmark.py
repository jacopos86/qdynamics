#!/usr/bin/env python3
"""Benchmark-local Suzuki-2 dynamics baseline for the validated HH L=2 t=8 anchor.

This module is intentionally a thin row/manifest wrapper around
``pipelines.time_dynamics.legacy.analysis.hh_realtime_suzuki_overlay.run_overlay``.  It does not
reimplement Suzuki evolution or alter controller decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.legacy.analysis import hh_realtime_suzuki_overlay as overlay


SCHEMA_VERSION = "hh_suzuki_benchmark_v1"
DEFAULT_CASE_ID = "hh_l2_t8_anchor_v1"
METHOD_ID = "hh_td_suzuki2"
METHOD_KIND = "suzuki_trotter"
OVERLAY_METHOD = "suzuki2"
REQUIRED_HARDWARE_SCOPES: tuple[tuple[str, str], ...] = (
    ("suzuki2", "seed_plus_one_step_additive"),
    ("suzuki2", "full_horizon_with_seed_prep"),
    ("controller", "controller_state_at_time"),
)


@dataclass(frozen=True)
class SuzukiBenchmarkCase:
    case_id: str
    controller_json: Path
    source_pdf: Path
    trotter_steps: int
    suzuki_orders: tuple[int, ...] = (2,)
    skip_pdf: bool = True
    export_compiled_circuits: bool = False
    backend_name: str | None = None
    seed_transpiler: int | None = None
    optimization_level: int | None = None
    preferred_fake_backends: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuzukiBenchmarkRow:
    case_id: str
    method_id: str
    method_kind: str
    order: int
    status: str
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
    state_at_time_scope: str
    state_at_time_basis: str | None
    state_at_time_2q: int | None
    state_at_time_depth: int | None
    state_at_time_size: int | None
    full_horizon_scope: str
    full_horizon_basis: str | None
    full_horizon_2q: int | None
    full_horizon_depth: int | None
    full_horizon_size: int | None
    full_horizon_horizon_2q: int | None
    full_horizon_depth_serial: int | None
    controller_state_scope: str
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
    artifact_overlay_pdf: str | None
    artifact_manifest_json: str | None
    artifact_rows_json: str | None
    artifact_summary_json: str | None
    compiled_circuit_dir: str | None
    exact_reference_method: str | None
    exact_steps_multiplier: Any
    exact_fields_reporting_only: bool = True


@dataclass(frozen=True)
class _CaseRunRecord:
    case: SuzukiBenchmarkCase
    overlay_config: overlay.SuzukiOverlayConfig
    overlay_json: Path
    row: dict[str, Any]


"Built Math: benchmark row = report(summary_suzuki2, selected_hardware_scopes); exact fields are diagnostics only."
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _parse_string_tuple(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = [str(x) for x in raw]
    return tuple(part.strip() for part in parts if part.strip())


def _maybe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def default_cases() -> tuple[SuzukiBenchmarkCase, ...]:
    return (
        SuzukiBenchmarkCase(
            case_id=DEFAULT_CASE_ID,
            controller_json=overlay.DEFAULT_CONTROLLER_JSON,
            source_pdf=overlay.DEFAULT_SOURCE_PDF,
            trotter_steps=160,
            suzuki_orders=(2,),
            skip_pdf=True,
            export_compiled_circuits=False,
        ),
    )


def _case_by_id(case_id: str) -> SuzukiBenchmarkCase:
    for case in default_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in default_cases())
    raise ValueError(f"unknown Suzuki benchmark case_id={case_id!r}; known cases: {known}")


def _compile_defaults_for_case(case: SuzukiBenchmarkCase) -> dict[str, Any]:
    need_source_defaults = (
        case.backend_name is None
        or case.seed_transpiler is None
        or case.optimization_level is None
        or not case.preferred_fake_backends
    )
    if need_source_defaults:
        source_payload = overlay._load_source_payload(Path(case.controller_json))
        defaults = dict(overlay._source_compile_defaults(source_payload))
    else:
        defaults = {}
    preferred = case.preferred_fake_backends or tuple(defaults.get("preferred_fake_backends", ()))
    return {
        "backend_name": str(case.backend_name if case.backend_name is not None else defaults.get("backend_name")),
        "seed_transpiler": int(
            case.seed_transpiler if case.seed_transpiler is not None else defaults.get("seed_transpiler")
        ),
        "optimization_level": int(
            case.optimization_level if case.optimization_level is not None else defaults.get("optimization_level")
        ),
        "preferred_fake_backends": tuple(str(x) for x in preferred),
    }


def _overlay_config_for_case(
    case: SuzukiBenchmarkCase,
    *,
    output_json: Path,
    output_pdf: Path | None,
    compiled_circuit_dir: Path | None,
) -> overlay.SuzukiOverlayConfig:
    defaults = _compile_defaults_for_case(case)
    return overlay.SuzukiOverlayConfig(
        controller_json=Path(case.controller_json),
        output_json=Path(output_json),
        output_pdf=None if output_pdf is None else Path(output_pdf),
        source_pdf=Path(case.source_pdf),
        trotter_steps=int(case.trotter_steps),
        suzuki_orders=tuple(int(order) for order in case.suzuki_orders),
        backend_name=str(defaults["backend_name"]),
        seed_transpiler=int(defaults["seed_transpiler"]),
        optimization_level=int(defaults["optimization_level"]),
        preferred_fake_backends=tuple(str(x) for x in defaults["preferred_fake_backends"]),
        export_compiled_circuits=bool(case.export_compiled_circuits),
        compiled_circuit_dir=None if compiled_circuit_dir is None else Path(compiled_circuit_dir),
        skip_pdf=bool(case.skip_pdf),
    )


def _required_hardware_report_row(
    payload: Mapping[str, Any],
    *,
    method: str,
    scope: str,
) -> Mapping[str, Any]:
    raw_rows = payload.get("hardware_report_rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("overlay payload is missing list field hardware_report_rows")
    matches = [
        row
        for row in raw_rows
        if isinstance(row, Mapping)
        and str(row.get("method", "")) == str(method)
        and str(row.get("scope", "")) == str(scope)
    ]
    if not matches:
        raise ValueError(
            "required hardware_report_rows entry missing: "
            f"method={method!r} scope={scope!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            "required hardware_report_rows entry is ambiguous: "
            f"method={method!r} scope={scope!r} count={len(matches)}"
        )
    return matches[0]


def _last_trajectory_time(method_payload: Mapping[str, Any]) -> float | None:
    trajectory = method_payload.get("trajectory", [])
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)) or not trajectory:
        return None
    last = trajectory[-1]
    if not isinstance(last, Mapping):
        return None
    return _maybe_float(last.get("time"))


def _trajectory_count(method_payload: Mapping[str, Any]) -> int | None:
    trajectory = method_payload.get("trajectory", [])
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        return None
    return int(len(trajectory))


def _required_hardware_scopes_for_overlay_method(overlay_method: str) -> tuple[tuple[str, str], ...]:
    method = str(overlay_method)
    return (
        (method, "seed_plus_one_step_additive"),
        (method, "full_horizon_with_seed_prep"),
        ("controller", "controller_state_at_time"),
    )


def _row_from_overlay_method_payload(
    payload: Mapping[str, Any],
    *,
    case_id: str,
    overlay_method: str,
    method_id: str,
    method_kind: str,
    expected_order: int,
    artifact_overlay_json: Path | str | None = None,
    artifact_manifest_json: Path | str | None = None,
    artifact_rows_json: Path | str | None = None,
    artifact_summary_json: Path | str | None = None,
    preferred_fake_backends: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Extract one Suzuki overlay method as a benchmark row.

    The hardware-cost scope checks intentionally stay fail-closed and are
    parameterized by ``overlay_method`` so benchmark wrappers can extract either
    ``suzuki1`` or ``suzuki2`` without duplicating row-contract logic.
    """

    method_name = str(overlay_method)
    order_expected = int(expected_order)
    methods = _maybe_mapping(payload.get("methods"))
    method_payload = _maybe_mapping(methods.get(method_name))
    if not method_payload:
        raise ValueError(f"overlay payload missing methods[{method_name!r}]")
    summary = _maybe_mapping(method_payload.get("summary"))
    if not summary:
        raise ValueError(f"overlay payload missing methods[{method_name!r}].summary")

    order = _maybe_int(method_payload.get("order")) or _maybe_int(summary.get("order")) or order_expected
    if int(order) != order_expected:
        raise ValueError(
            f"benchmark row expects Suzuki order {order_expected}, got order={order!r}"
        )

    for method, scope in _required_hardware_scopes_for_overlay_method(method_name):
        _required_hardware_report_row(payload, method=method, scope=scope)
    state_cost = _required_hardware_report_row(
        payload,
        method=method_name,
        scope="seed_plus_one_step_additive",
    )
    full_cost = _required_hardware_report_row(
        payload,
        method=method_name,
        scope="full_horizon_with_seed_prep",
    )
    controller_cost = _required_hardware_report_row(
        payload,
        method="controller",
        scope="controller_state_at_time",
    )

    manifest = _maybe_mapping(payload.get("parameter_manifest"))
    source = _maybe_mapping(payload.get("source"))
    config = _maybe_mapping(payload.get("config"))
    written = _maybe_mapping(payload.get("written"))

    overlay_json = artifact_overlay_json or written.get("output_json") or manifest.get("output_json")
    preferred = tuple(str(x) for x in (preferred_fake_backends or ()))

    row = SuzukiBenchmarkRow(
        case_id=str(case_id),
        method_id=str(method_id),
        method_kind=str(method_kind),
        order=int(order),
        status="ok",
        controller_json=_as_optional_str(manifest.get("controller_json") or source.get("controller_json")),
        source_pdf=_as_optional_str(manifest.get("source_pdf") or source.get("source_pdf")),
        seed_artifact_json=_as_optional_str(manifest.get("seed_artifact_json") or source.get("artifact_json")),
        drive_enabled=_maybe_bool(manifest.get("drive_enabled")),
        t_final=_maybe_float(manifest.get("t_final")) or _last_trajectory_time(method_payload),
        num_times=_maybe_int(manifest.get("num_times"))
        or _maybe_int(summary.get("row_count"))
        or _trajectory_count(method_payload),
        trotter_steps=_maybe_int(manifest.get("trotter_steps")) or _maybe_int(config.get("trotter_steps")),
        final_energy_total=_maybe_float(summary.get("final_energy_total")),
        final_energy_total_exact=_maybe_float(summary.get("final_energy_total_exact")),
        final_abs_energy_total_error=_maybe_float(summary.get("final_abs_energy_total_error")),
        mean_abs_energy_total_error=_maybe_float(summary.get("mean_abs_energy_total_error")),
        max_abs_energy_total_error=_maybe_float(summary.get("max_abs_energy_total_error")),
        state_at_time_scope=str(state_cost.get("scope")),
        state_at_time_basis=_as_optional_str(state_cost.get("basis")),
        state_at_time_2q=_maybe_int(state_cost.get("compiled_count_2q")),
        state_at_time_depth=_maybe_int(state_cost.get("compiled_depth")),
        state_at_time_size=_maybe_int(state_cost.get("compiled_size")),
        full_horizon_scope=str(full_cost.get("scope")),
        full_horizon_basis=_as_optional_str(full_cost.get("basis")),
        full_horizon_2q=_maybe_int(full_cost.get("compiled_count_2q")),
        full_horizon_depth=_maybe_int(full_cost.get("compiled_depth")),
        full_horizon_size=_maybe_int(full_cost.get("compiled_size")),
        full_horizon_horizon_2q=_maybe_int(full_cost.get("horizon_count_2q")),
        full_horizon_depth_serial=_maybe_int(full_cost.get("horizon_depth_serial")),
        controller_state_scope=str(controller_cost.get("scope")),
        controller_state_basis=_as_optional_str(controller_cost.get("basis")),
        controller_state_2q=_maybe_int(controller_cost.get("compiled_count_2q")),
        controller_state_depth=_maybe_int(controller_cost.get("compiled_depth")),
        controller_state_size=_maybe_int(controller_cost.get("compiled_size")),
        backend_name=_as_optional_str(manifest.get("compile_backend")),
        seed_transpiler=_maybe_int(manifest.get("compile_seed_transpiler")),
        optimization_level=_maybe_int(manifest.get("compile_optimization_level")),
        preferred_fake_backends=preferred,
        export_compiled_circuits=bool(config.get("export_compiled_circuits", False)),
        artifact_overlay_json=_as_optional_str(overlay_json),
        artifact_overlay_pdf=_as_optional_str(written.get("output_pdf") or manifest.get("output_pdf")),
        artifact_manifest_json=_as_optional_str(artifact_manifest_json),
        artifact_rows_json=_as_optional_str(artifact_rows_json),
        artifact_summary_json=_as_optional_str(artifact_summary_json),
        compiled_circuit_dir=_as_optional_str(written.get("compiled_circuit_dir") or config.get("compiled_circuit_dir")),
        exact_reference_method=_as_optional_str(manifest.get("exact_reference_method")),
        exact_steps_multiplier=manifest.get("exact_steps_multiplier"),
    )
    return _jsonable(row)


def _row_from_overlay_payload(
    payload: Mapping[str, Any],
    *,
    case_id: str,
    artifact_overlay_json: Path | str | None = None,
    artifact_manifest_json: Path | str | None = None,
    artifact_rows_json: Path | str | None = None,
    artifact_summary_json: Path | str | None = None,
    preferred_fake_backends: Sequence[str] | None = None,
) -> dict[str, Any]:
    return _row_from_overlay_method_payload(
        payload,
        case_id=case_id,
        overlay_method=OVERLAY_METHOD,
        method_id=METHOD_ID,
        method_kind=METHOD_KIND,
        expected_order=2,
        artifact_overlay_json=artifact_overlay_json,
        artifact_manifest_json=artifact_manifest_json,
        artifact_rows_json=artifact_rows_json,
        artifact_summary_json=artifact_summary_json,
        preferred_fake_backends=preferred_fake_backends,
    )


def _run_case(
    case: SuzukiBenchmarkCase,
    *,
    output_dir: Path,
    manifest_json: Path,
    rows_json: Path,
    summary_json: Path,
    command: str,
) -> _CaseRunRecord:
    overlay_dir = Path(output_dir) / "overlay"
    overlay_json = overlay_dir / f"{case.case_id}.json"
    overlay_pdf = None if bool(case.skip_pdf) else overlay_dir / f"{case.case_id}.pdf"
    compiled_dir = (
        Path(output_dir) / "compiled_circuits" / case.case_id
        if bool(case.export_compiled_circuits)
        else None
    )
    config = _overlay_config_for_case(
        case,
        output_json=overlay_json,
        output_pdf=overlay_pdf,
        compiled_circuit_dir=compiled_dir,
    )
    payload = overlay.run_overlay(config, command=command)
    # The overlay writes this path itself; write it again here so mocked tests and
    # future overlay no-op modes still satisfy the benchmark artifact contract.
    _write_json(overlay_json, payload)
    row = _row_from_overlay_payload(
        payload,
        case_id=case.case_id,
        artifact_overlay_json=overlay_json,
        artifact_manifest_json=manifest_json,
        artifact_rows_json=rows_json,
        artifact_summary_json=summary_json,
        preferred_fake_backends=config.preferred_fake_backends,
    )
    return _CaseRunRecord(case=case, overlay_config=config, overlay_json=overlay_json, row=row)


def _manifest_payload(
    *,
    records: Sequence[_CaseRunRecord],
    output_dir: Path,
    manifest_json: Path,
    rows_json: Path,
    summary_json: Path,
    command: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "benchmark": "hh_suzuki_time_dynamics",
        "method_contract": {
            "method_id": METHOD_ID,
            "method_kind": METHOD_KIND,
            "overlay_method": OVERLAY_METHOD,
            "suzuki_orders": [2],
            "exact_reference_policy": "reporting_only",
            "hardware_scope_policy": "fail_closed_required_report_rows",
            "required_hardware_scopes": [
                {"method": method, "scope": scope} for method, scope in REQUIRED_HARDWARE_SCOPES
            ],
        },
        "command": command,
        "output_dir": str(output_dir),
        "paths": {
            "manifest_json": str(manifest_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
            "overlay_dir": str(Path(output_dir) / "overlay"),
        },
        "cases": [
            {
                "case": _jsonable(record.case),
                "resolved_overlay_config": _jsonable(record.overlay_config),
                "artifact_overlay_json": str(record.overlay_json),
            }
            for record in records
        ],
    }


def _summary_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    manifest_json: Path,
    rows_json: Path,
    summary_json: Path,
    command: str,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "benchmark": "hh_suzuki_time_dynamics",
        "command": command,
        "output_dir": str(output_dir),
        "row_count": int(len(rows)),
        "status_counts": dict(sorted(status_counts.items())),
        "case_ids": [str(row.get("case_id")) for row in rows],
        "method_ids": [str(row.get("method_id")) for row in rows],
        "paths": {
            "manifest_json": str(manifest_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
        },
        "key_metrics": [
            {
                "case_id": row.get("case_id"),
                "method_id": row.get("method_id"),
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
    cases: Sequence[SuzukiBenchmarkCase],
    output_dir: Path,
    command: str = "",
) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_json = root / "manifest.json"
    rows_json = root / "rows.json"
    summary_json = root / "summary.json"
    root.mkdir(parents=True, exist_ok=True)

    records = [
        _run_case(
            case,
            output_dir=root,
            manifest_json=manifest_json,
            rows_json=rows_json,
            summary_json=summary_json,
            command=command,
        )
        for case in cases
    ]
    rows = [dict(record.row) for record in records]
    manifest = _manifest_payload(
        records=records,
        output_dir=root,
        manifest_json=manifest_json,
        rows_json=rows_json,
        summary_json=summary_json,
        command=command,
    )
    summary = _summary_payload(
        rows=rows,
        output_dir=root,
        manifest_json=manifest_json,
        rows_json=rows_json,
        summary_json=summary_json,
        command=command,
    )

    _write_json(manifest_json, manifest)
    _write_json(rows_json, rows)
    _write_json(summary_json, summary)
    return {
        "manifest": manifest,
        "rows": rows,
        "summary": summary,
        "paths": {
            "manifest_json": str(manifest_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the benchmark-local HH L2 t=8 Suzuki-2 dynamics baseline."
    )
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-json", type=Path, default=None)
    parser.add_argument("--source-pdf", type=Path, default=None)
    parser.add_argument("--trotter-steps", type=int, default=None)
    parser.add_argument("--compile-backend-name", type=str, default=None)
    parser.add_argument("--compile-seed-transpiler", type=int, default=None)
    parser.add_argument("--compile-optimization-level", type=int, default=None)
    parser.add_argument("--compile-preferred-fake-backends", type=str, default=None)
    parser.add_argument("--export-compiled-circuits", action="store_true")
    parser.add_argument("--write-pdf", action="store_true", help="Write the overlay PDF; default is JSON-only.")
    return parser


def _case_from_args(args: argparse.Namespace) -> SuzukiBenchmarkCase:
    case = _case_by_id(str(args.case_id))
    preferred = _parse_string_tuple(args.compile_preferred_fake_backends)
    return replace(
        case,
        controller_json=Path(args.controller_json) if args.controller_json is not None else case.controller_json,
        source_pdf=Path(args.source_pdf) if args.source_pdf is not None else case.source_pdf,
        trotter_steps=int(args.trotter_steps) if args.trotter_steps is not None else case.trotter_steps,
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
        return " ".join([sys.executable, "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_suzuki_benchmark", *sys.argv[1:]])
    return " ".join(["python", "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_suzuki_benchmark", *map(str, argv)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    case = _case_from_args(args)
    command = _command_from_argv(argv)
    result = run_benchmark(cases=(case,), output_dir=Path(args.output_dir), command=command)
    row = result["rows"][0]
    print(f"manifest_json={result['paths']['manifest_json']}")
    print(f"rows_json={result['paths']['rows_json']}")
    print(f"summary_json={result['paths']['summary_json']}")
    print(f"artifact_overlay_json={row.get('artifact_overlay_json')}")
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
