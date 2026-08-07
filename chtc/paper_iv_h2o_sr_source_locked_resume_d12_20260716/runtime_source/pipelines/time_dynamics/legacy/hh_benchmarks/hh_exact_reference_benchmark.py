#!/usr/bin/env python3
"""Benchmark-local exact-reference dynamics row for the HH L=2 t=8 anchor.

This module extracts the exact trajectory already embedded in the validated
Chapter 17A controller artifact and emits it as a standalone benchmark row.  It
is intentionally read-only relative to controller logic: it does not call the
Suzuki overlay runner, fixed-manifold runner, or any controller path.
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


SCHEMA_VERSION = "hh_exact_reference_benchmark_v1"
REFERENCE_SCHEMA_VERSION = "hh_exact_reference_payload_v1"
DEFAULT_CASE_ID = "hh_l2_t8_anchor_v1"
METHOD_ID = "hh_td_exact_reference_v1"
METHOD_KIND = "exact_reference"
HARDWARE_COST_POLICY = "not_applicable_for_classical_exact_reference_null_fields"


@dataclass(frozen=True)
class ExactReferenceBenchmarkCase:
    case_id: str
    controller_json: Path


@dataclass(frozen=True)
class ExactReferenceBenchmarkRow:
    case_id: str
    method_id: str
    method_kind: str
    status: str
    exact_reference_method: str
    qpu_faithful: bool
    diagnostic_exact_assisted: bool
    hardware_cost_applicable: bool
    hardware_cost_policy: str
    controller_json: str | None
    seed_artifact_json: str | None
    drive_enabled: bool | None
    t_final: float | None
    num_times: int | None
    final_energy_total: float | None
    final_energy_total_exact: float | None
    final_abs_energy_total_error: float
    mean_abs_energy_total_error: float
    max_abs_energy_total_error: float
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
    reference_steps_multiplier: Any
    exact_steps_multiplier: Any
    artifact_reference_json: str | None
    artifact_manifest_json: str | None
    artifact_rows_json: str | None
    artifact_summary_json: str | None
    exact_error_policy: str = "reference_self_comparison_exactly_zero"
    controller_decisions_modified: bool = False
    controller_energy_fields_read: bool = False


@dataclass(frozen=True)
class _CaseRunRecord:
    case: ExactReferenceBenchmarkCase
    reference_json: Path
    reference_payload: Mapping[str, Any]
    row: dict[str, Any]


"Built Math: exact-reference row = identity comparison of the source exact energy trajectory, so |E_ref - E_ref| = 0 at every retained state-sample time."
def _now_utc() -> str:
    return suzuki_bench._now_utc()


def _write_json(path: Path, payload: Any) -> Path:
    return suzuki_bench._write_json(path, payload)


def _jsonable(value: Any) -> Any:
    return suzuki_bench._jsonable(value)


def _maybe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_optional_str(value: Any) -> str | None:
    return suzuki_bench._as_optional_str(value)


def _maybe_bool(value: Any) -> bool | None:
    return suzuki_bench._maybe_bool(value)


def _required_finite_float(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be finite; got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return float(out)


def _optional_finite_float(value: Any, *, field: str, default: float | None = None) -> float | None:
    if value is None:
        return default
    return _required_finite_float(value, field=field)


def _reference_method_from_source(source_payload: Mapping[str, Any]) -> str:
    reference = _maybe_mapping(source_payload.get("reference"))
    method_raw = reference.get("reference_method")
    method = "" if method_raw is None else str(method_raw).strip()
    if not method:
        raise ValueError("missing/blank exact reference provenance: reference.reference_method")
    return method


def _selected_drive_config(source_payload: Mapping[str, Any]) -> dict[str, Any]:
    drive = _maybe_mapping(source_payload.get("drive_config"))
    keys = (
        "enabled",
        "drive_A",
        "drive_omega",
        "drive_tbar",
        "drive_phi",
        "drive_pattern",
        "drive_include_identity",
        "drive_time_sampling",
        "drive_t0",
        "exact_steps_multiplier",
    )
    return {key: drive.get(key) for key in keys if key in drive}


def _selected_reference_provenance(source_payload: Mapping[str, Any], *, reference_method: str) -> dict[str, Any]:
    reference = _maybe_mapping(source_payload.get("reference"))
    keys = (
        "reference_mode",
        "reference_enabled",
        "kind",
        "reference_steps_multiplier",
        "projection_time_sampling",
        "geometry_sample_time_policy",
    )
    provenance = {key: reference.get(key) for key in keys if key in reference}
    provenance["reference_method"] = str(reference_method)
    return provenance


def _reference_payload_from_source(
    source_payload: Mapping[str, Any],
    *,
    case: ExactReferenceBenchmarkCase,
) -> dict[str, Any]:
    reference_method = _reference_method_from_source(source_payload)
    source_rows = overlay._state_sample_rows(source_payload)
    times = overlay._source_times(source_payload, source_rows)
    num_times = int(times.size)
    if num_times < 2:
        raise ValueError(f"exact reference benchmark requires num_times >= 2; got {num_times}")

    trajectory: list[dict[str, Any]] = []
    exact_energies: list[float] = []
    for idx, row in enumerate(source_rows):
        energy_exact = _required_finite_float(
            row.get("energy_total_exact"),
            field=f"state-sample row {idx} energy_total_exact",
        )
        time = _required_finite_float(times[idx], field=f"source time {idx}")
        physical_time = _optional_finite_float(
            row.get("physical_time"),
            field=f"state-sample row {idx} physical_time",
            default=time,
        )
        checkpoint_raw = row.get("checkpoint_index", idx)
        try:
            checkpoint_index = int(checkpoint_raw)
        except (TypeError, ValueError):
            checkpoint_index = int(idx)
        exact_energies.append(float(energy_exact))
        trajectory.append(
            {
                "checkpoint_index": checkpoint_index,
                "source_row_index": int(idx),
                "time": float(time),
                "physical_time": float(physical_time if physical_time is not None else time),
                "energy_total": float(energy_exact),
                "energy_total_exact": float(energy_exact),
                "abs_energy_total_error": 0.0,
            }
        )

    final_exact = float(exact_energies[-1])
    reference = _selected_reference_provenance(source_payload, reference_method=reference_method)
    drive_config = _selected_drive_config(source_payload)
    summary = {
        "row_count": int(num_times),
        "t_final": float(times[-1]),
        "num_times": int(num_times),
        "final_energy_total": final_exact,
        "final_energy_total_exact": final_exact,
        "final_abs_energy_total_error": 0.0,
        "mean_abs_energy_total_error": 0.0,
        "max_abs_energy_total_error": 0.0,
    }
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "case_id": str(case.case_id),
        "method_id": METHOD_ID,
        "method_kind": METHOD_KIND,
        "source": {
            "controller_json": str(case.controller_json),
            "run_tag": source_payload.get("run_tag"),
            "artifact_json": source_payload.get("artifact_json"),
        },
        "reference": reference,
        "drive_config": drive_config,
        "trajectory": trajectory,
        "summary": summary,
        "contract": {
            "qpu_faithful": False,
            "diagnostic_exact_assisted": False,
            "hardware_cost_applicable": False,
            "controller_energy_fields_read": False,
            "controller_paths_called": False,
            "exact_error_policy": "reference_self_comparison_exactly_zero",
        },
    }


def _row_from_reference_payload(
    reference_payload: Mapping[str, Any],
    *,
    case: ExactReferenceBenchmarkCase,
    reference_json: Path | str | None = None,
    manifest_json: Path | str | None = None,
    rows_json: Path | str | None = None,
    summary_json: Path | str | None = None,
) -> dict[str, Any]:
    summary = _maybe_mapping(reference_payload.get("summary"))
    reference = _maybe_mapping(reference_payload.get("reference"))
    source = _maybe_mapping(reference_payload.get("source"))
    drive_config = _maybe_mapping(reference_payload.get("drive_config"))
    exact_reference_method = str(reference.get("reference_method", "")).strip()
    if not exact_reference_method:
        raise ValueError("missing/blank exact reference provenance: reference.reference_method")

    row = ExactReferenceBenchmarkRow(
        case_id=str(case.case_id),
        method_id=METHOD_ID,
        method_kind=METHOD_KIND,
        status="ok",
        exact_reference_method=exact_reference_method,
        qpu_faithful=False,
        diagnostic_exact_assisted=False,
        hardware_cost_applicable=False,
        hardware_cost_policy=HARDWARE_COST_POLICY,
        controller_json=_as_optional_str(source.get("controller_json") or case.controller_json),
        seed_artifact_json=_as_optional_str(source.get("artifact_json")),
        drive_enabled=_maybe_bool(drive_config.get("enabled")),
        t_final=_required_finite_float(summary.get("t_final"), field="summary.t_final"),
        num_times=int(summary.get("num_times")),
        final_energy_total=_required_finite_float(
            summary.get("final_energy_total"), field="summary.final_energy_total"
        ),
        final_energy_total_exact=_required_finite_float(
            summary.get("final_energy_total_exact"), field="summary.final_energy_total_exact"
        ),
        final_abs_energy_total_error=0.0,
        mean_abs_energy_total_error=0.0,
        max_abs_energy_total_error=0.0,
        state_at_time_scope=None,
        state_at_time_basis=None,
        state_at_time_2q=None,
        state_at_time_depth=None,
        state_at_time_size=None,
        full_horizon_scope=None,
        full_horizon_basis=None,
        full_horizon_2q=None,
        full_horizon_depth=None,
        full_horizon_size=None,
        full_horizon_horizon_2q=None,
        full_horizon_depth_serial=None,
        controller_state_scope=None,
        controller_state_basis=None,
        controller_state_2q=None,
        controller_state_depth=None,
        controller_state_size=None,
        reference_steps_multiplier=reference.get("reference_steps_multiplier"),
        exact_steps_multiplier=drive_config.get(
            "exact_steps_multiplier", reference.get("reference_steps_multiplier")
        ),
        artifact_reference_json=_as_optional_str(reference_json),
        artifact_manifest_json=_as_optional_str(manifest_json),
        artifact_rows_json=_as_optional_str(rows_json),
        artifact_summary_json=_as_optional_str(summary_json),
    )
    if int(row.num_times or 0) < 2:
        raise ValueError(f"exact reference benchmark requires num_times >= 2; got {row.num_times}")
    if row.final_energy_total != row.final_energy_total_exact:
        raise ValueError("exact reference row must have final_energy_total == final_energy_total_exact")
    return _jsonable(row)


def default_cases() -> tuple[ExactReferenceBenchmarkCase, ...]:
    return (
        ExactReferenceBenchmarkCase(
            case_id=DEFAULT_CASE_ID,
            controller_json=overlay.DEFAULT_CONTROLLER_JSON,
        ),
    )


def _case_by_id(case_id: str) -> ExactReferenceBenchmarkCase:
    for case in default_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in default_cases())
    raise ValueError(f"unknown exact-reference benchmark case_id={case_id!r}; known cases: {known}")


def _run_case(
    case: ExactReferenceBenchmarkCase,
    *,
    output_dir: Path,
    manifest_json: Path,
    rows_json: Path,
    summary_json: Path,
) -> _CaseRunRecord:
    source_payload = overlay._load_source_payload(Path(case.controller_json))
    reference_payload = _reference_payload_from_source(source_payload, case=case)
    reference_json = Path(output_dir) / "reference" / f"{case.case_id}.json"
    _write_json(reference_json, reference_payload)
    row = _row_from_reference_payload(
        reference_payload,
        case=case,
        reference_json=reference_json,
        manifest_json=manifest_json,
        rows_json=rows_json,
        summary_json=summary_json,
    )
    return _CaseRunRecord(
        case=case,
        reference_json=reference_json,
        reference_payload=reference_payload,
        row=row,
    )


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
        "benchmark": "hh_exact_reference_time_dynamics",
        "method_contract": {
            "method_id": METHOD_ID,
            "method_kind": METHOD_KIND,
            "default_case_id": DEFAULT_CASE_ID,
            "qpu_faithful": False,
            "diagnostic_exact_assisted": False,
            "hardware_cost_applicable": False,
            "hardware_cost_policy": HARDWARE_COST_POLICY,
            "source_policy": "read_existing_controller_artifact_only",
            "forbidden_calls": [
                "hh_realtime_suzuki_overlay.run_overlay",
                "hh_fixed_manifold_mclachlan.run_fixed_manifold_exact",
                "controller_decision_paths",
            ],
            "exact_error_policy": "reference_self_comparison_exactly_zero",
            "fail_closed_requirements": [
                "reference.reference_method nonblank",
                "nonempty retained state-sample rows",
                "strictly increasing source time grid matching retained rows",
                "finite energy_total_exact on every retained state-sample row",
                "num_times >= 2",
            ],
        },
        "command": command,
        "output_dir": str(output_dir),
        "paths": {
            "manifest_json": str(manifest_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
            "reference_dir": str(Path(output_dir) / "reference"),
        },
        "cases": [
            {
                "case": _jsonable(record.case),
                "artifact_reference_json": str(record.reference_json),
                "exact_reference_method": record.row.get("exact_reference_method"),
                "num_times": record.row.get("num_times"),
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
        "benchmark": "hh_exact_reference_time_dynamics",
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
                "exact_reference_method": row.get("exact_reference_method"),
                "num_times": row.get("num_times"),
                "final_energy_total_exact": row.get("final_energy_total_exact"),
                "final_abs_energy_total_error": row.get("final_abs_energy_total_error"),
                "mean_abs_energy_total_error": row.get("mean_abs_energy_total_error"),
                "max_abs_energy_total_error": row.get("max_abs_energy_total_error"),
                "hardware_cost_applicable": row.get("hardware_cost_applicable"),
            }
            for row in rows
        ],
    }


def run_benchmark(
    *,
    cases: Sequence[ExactReferenceBenchmarkCase],
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
            "reference_dir": str(root / "reference"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the benchmark-local exact-reference row for the HH L2 t=8 anchor."
    )
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-json", type=Path, default=None)
    return parser


def _case_from_args(args: argparse.Namespace) -> ExactReferenceBenchmarkCase:
    case = _case_by_id(str(args.case_id))
    return replace(
        case,
        controller_json=Path(args.controller_json) if args.controller_json is not None else case.controller_json,
    )


def _command_from_argv(argv: Sequence[str] | None) -> str:
    if argv is None:
        return " ".join(
            [sys.executable, "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_exact_reference_benchmark", *sys.argv[1:]]
        )
    return " ".join(["python", "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_exact_reference_benchmark", *map(str, argv)])


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
    print(f"artifact_reference_json={row.get('artifact_reference_json')}")
    print(f"method_id={row.get('method_id')}")
    print(f"exact_reference_method={row.get('exact_reference_method')}")
    print(f"num_times={row.get('num_times')}")
    print(f"final_energy_total_exact={row.get('final_energy_total_exact')}")
    print(f"final_abs_energy_total_error={row.get('final_abs_energy_total_error')}")
    print(f"mean_abs_energy_total_error={row.get('mean_abs_energy_total_error')}")
    print(f"max_abs_energy_total_error={row.get('max_abs_energy_total_error')}")
    print(f"hardware_cost_applicable={row.get('hardware_cost_applicable')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
