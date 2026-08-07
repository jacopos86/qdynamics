#!/usr/bin/env python3
"""Benchmark-local fixed-manifold McLachlan row for the HH L=2 t=8 anchor.

This module is intentionally a thin row/manifest wrapper around
``pipelines.time_dynamics.fixed_manifold.mclachlan.run_fixed_manifold_exact``.
It does not alter controller decisions or production realtime code.  The row is
an exact-assisted diagnostic control, not a QPU-faithful controller route.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.fixed_manifold import mclachlan as fixed
from pipelines.time_dynamics.legacy.analysis import hh_realtime_suzuki_overlay as overlay


SCHEMA_VERSION = "hh_fixed_mclachlan_benchmark_v1"
DEFAULT_CASE_ID = "hh_l2_t8_anchor_v1"
METHOD_ID = "hh_td_fixed_mclachlan_pareto_lean_l2_exactv1"
METHOD_KIND = "fixed_mclachlan"
DECISION_MODE = "exact_v1"
SEED_FAMILY = "pareto_lean_l2"
STATE_SCOPE = "state_scaffold_source"
HORIZON_SCOPE = "repeated_state_scaffold_budget"
CONTROLLER_STATE_SCOPE = "controller_state_at_time"


@dataclass(frozen=True)
class FixedMclachlanBenchmarkCase:
    case_id: str
    controller_json: Path
    source_artifact_json: Path
    spec_name: str
    loader_mode: str
    generator_family: str
    fallback_family: str
    append_pool_family: str
    miss_threshold: float
    gain_ratio_threshold: float
    append_margin_abs: float
    backend_name: str | None = None
    seed_transpiler: int | None = None
    optimization_level: int | None = None
    preferred_fake_backends: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixedMclachlanBenchmarkRow:
    case_id: str
    method_id: str
    method_kind: str
    status: str
    decision_mode: str
    diagnostic_exact_assisted: bool
    qpu_faithful: bool
    seed_family: str
    controller_json: str | None
    source_artifact_json: str | None
    drive_enabled: bool | None
    t_final: float | None
    num_times: int | None
    final_energy_total: float | None
    final_energy_total_exact: float | None
    final_abs_energy_total_error: float | None
    mean_abs_energy_total_error: float | None
    max_abs_energy_total_error: float | None
    fidelity_min: float | None
    rho_miss_max: float | None
    final_logical_block_count: int | None
    final_runtime_parameter_count: int | None
    state_at_time_scope: str
    state_at_time_basis: str | None
    state_at_time_2q: int | None
    state_at_time_depth: int | None
    state_at_time_size: int | None
    full_horizon_scope: str
    full_horizon_basis: str | None
    full_horizon_intervals: int | None
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
    artifact_run_json: str | None
    artifact_manifest_json: str | None
    artifact_rows_json: str | None
    artifact_summary_json: str | None
    exact_assisted_contract: str = "diagnostic_exact_assisted_not_qpu_faithful"


@dataclass(frozen=True)
class _CaseRunRecord:
    case: FixedMclachlanBenchmarkCase
    spec: fixed.FixedManifoldRunSpec
    run_record: Mapping[str, Any]
    run_json: Path
    row: dict[str, Any]
    raw_compile_rows: Sequence[Mapping[str, Any]]


"Built Math: fixed-manifold row = exact_v1 McLachlan trajectory + one compiled fixed state scaffold; full-horizon cost is repeated serial state preparation over num_times-1 intervals."
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"re": _jsonable(float(np.real(value))), "im": _jsonable(float(np.imag(value)))}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}.")
    return dict(payload)


def _maybe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


def _mul_optional_int(value: int | None, factor: int) -> int | None:
    return None if value is None else int(value) * int(factor)


def _finite_values(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        maybe = _maybe_float(value)
        if maybe is not None:
            out.append(float(maybe))
    return out


def _nan_safe_min(values: Sequence[Any]) -> float | None:
    finite = _finite_values(values)
    return None if not finite else float(min(finite))


def _nan_safe_max(values: Sequence[Any]) -> float | None:
    finite = _finite_values(values)
    return None if not finite else float(max(finite))


def _nan_safe_mean(values: Sequence[Any]) -> float | None:
    finite = _finite_values(values)
    return None if not finite else float(sum(finite) / len(finite))


def _parse_string_tuple(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = [str(x) for x in raw]
    return tuple(part.strip() for part in parts if part.strip())


def default_cases() -> tuple[FixedMclachlanBenchmarkCase, ...]:
    return (
        FixedMclachlanBenchmarkCase(
            case_id=DEFAULT_CASE_ID,
            controller_json=overlay.DEFAULT_CONTROLLER_JSON,
            source_artifact_json=fixed.DEFAULT_PARETO_ARTIFACT,
            spec_name=SEED_FAMILY,
            loader_mode="replay_family",
            generator_family="match_adapt",
            fallback_family="full_meta",
            append_pool_family="match_replay",
            miss_threshold=1.0e9,
            gain_ratio_threshold=1.0e-9,
            append_margin_abs=1.0e-12,
        ),
    )


def _case_by_id(case_id: str) -> FixedMclachlanBenchmarkCase:
    for case in default_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in default_cases())
    raise ValueError(f"unknown fixed-McLachlan benchmark case_id={case_id!r}; known cases: {known}")


def _run_spec_for_case(case: FixedMclachlanBenchmarkCase) -> fixed.FixedManifoldRunSpec:
    return fixed.FixedManifoldRunSpec(
        name=str(case.spec_name),
        artifact_json=Path(case.source_artifact_json),
        loader_mode=str(case.loader_mode),
        generator_family=str(case.generator_family),
        fallback_family=str(case.fallback_family),
        append_pool_family=str(case.append_pool_family),
    )


def _compile_defaults_for_case(
    case: FixedMclachlanBenchmarkCase,
    source_payload: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = dict(overlay._source_compile_defaults(source_payload))
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


def _drive_kwargs_from_source(source_payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = source_payload.get("drive_config", {})
    drive = raw if isinstance(raw, Mapping) else {}
    custom_raw = drive.get("drive_custom_s", drive.get("drive_custom_weights", None))
    if isinstance(custom_raw, (list, tuple)):
        custom_s = json.dumps([float(x) for x in custom_raw])
    elif custom_raw is None:
        custom_s = None
    else:
        custom_s = str(custom_raw)
    return {
        "enable_drive": bool(drive.get("enabled", False)),
        "drive_A": float(drive.get("drive_A", 0.0)),
        "drive_omega": float(drive.get("drive_omega", 1.0)),
        "drive_tbar": float(drive.get("drive_tbar", 1.0)),
        "drive_phi": float(drive.get("drive_phi", 0.0)),
        "drive_pattern": str(drive.get("drive_pattern", "staggered")),
        "drive_custom_s": custom_s,
        "drive_include_identity": bool(drive.get("drive_include_identity", False)),
        "drive_time_sampling": str(drive.get("drive_time_sampling", "midpoint")),
        "drive_t0": float(drive.get("drive_t0", 0.0)),
        "exact_steps_multiplier": int(drive.get("exact_steps_multiplier", 1)),
    }


def _required_controller_cost_row(source_payload: Mapping[str, Any]) -> overlay.CircuitCostRow:
    row = overlay._source_controller_cost_row(source_payload)
    if row is None:
        raise ValueError("source controller compile reference row is absent")
    if row.compiled_count_2q is None or row.compiled_depth is None:
        raise ValueError("source controller compile reference row is missing compiled 2q/depth metrics")
    return row


def _num_qubits_from_state(state: Any) -> int:
    size = int(np.asarray(state, dtype=complex).reshape(-1).size)
    if size <= 0:
        raise ValueError("cannot infer qubit count from an empty reference state")
    nq = int(round(math.log2(size)))
    if 2**nq != size:
        raise ValueError(f"reference state length {size} is not a power of two")
    return nq


def _compile_fixed_state_scaffold(
    *,
    spec: fixed.FixedManifoldRunSpec,
    case: FixedMclachlanBenchmarkCase,
    compile_defaults: Mapping[str, Any],
) -> tuple[overlay.CircuitCostRow, list[dict[str, Any]]]:
    loaded = fixed.load_run_context(
        spec,
        tag=f"{case.case_id}_fixed_mclachlan_benchmark_compile",
        lock_fixed_manifold=True,
    )
    nq = _num_qubits_from_state(loaded.replay_context.psi_ref)
    scaffold_circuit = overlay.build_ansatz_circuit(
        loaded.replay_context.base_layout,
        np.asarray(loaded.replay_context.adapt_theta_runtime, dtype=float).reshape(-1),
        int(nq),
        ref_state=np.asarray(loaded.replay_context.psi_ref, dtype=complex).reshape(-1),
    )
    cost, raw_rows = overlay._compile_one_circuit_cost(
        method="fixed_mclachlan",
        order=None,
        scope=STATE_SCOPE,
        trotter_steps=None,
        includes_seed_prep=True,
        circuit=scaffold_circuit,
        backend_name=str(compile_defaults["backend_name"]),
        preferred_fake_backends=tuple(str(x) for x in compile_defaults["preferred_fake_backends"]),
        seed_transpiler=int(compile_defaults["seed_transpiler"]),
        optimization_level=int(compile_defaults["optimization_level"]),
    )
    if cost.compiled_count_2q is None or cost.compiled_depth is None:
        raise ValueError(f"fixed state scaffold compile failed: {cost.error or cost.transpile_status}")
    return cost, [dict(row) for row in raw_rows if isinstance(row, Mapping)]


def _trajectory_energy_errors(trajectory: Sequence[Mapping[str, Any]]) -> list[float | None]:
    out: list[float | None] = []
    for row in trajectory:
        explicit = _maybe_float(row.get("abs_energy_total_error"))
        if explicit is not None:
            out.append(explicit)
            continue
        energy = _maybe_float(row.get("energy_total"))
        exact = _maybe_float(row.get("energy_total_exact"))
        out.append(None if energy is None or exact is None else abs(float(energy) - float(exact)))
    return out


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _artifact_trajectory_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    trajectory_raw = payload.get("trajectory", [])
    trajectory = [dict(row) for row in trajectory_raw if isinstance(row, Mapping)]
    summary = _maybe_mapping(payload.get("summary"))
    extra = _maybe_mapping(payload.get("extra_summary"))
    final_row = trajectory[-1] if trajectory else {}
    energy_errors = _trajectory_energy_errors(trajectory)
    final_error = _maybe_float(final_row.get("abs_energy_total_error"))
    if final_error is None:
        energy = _maybe_float(final_row.get("energy_total"))
        exact = _maybe_float(final_row.get("energy_total_exact"))
        final_error = None if energy is None or exact is None else abs(float(energy) - float(exact))
    mean_error = _nan_safe_mean(energy_errors) if energy_errors else None
    max_error = _nan_safe_max(energy_errors) if energy_errors else None
    fidelity_min = _nan_safe_min([row.get("fidelity_exact") for row in trajectory])
    rho_miss_max = _nan_safe_max([row.get("rho_miss") for row in trajectory])
    return {
        "trajectory_points": int(len(trajectory)),
        "final_energy_total": _first_not_none(
            _maybe_float(final_row.get("energy_total")),
            _maybe_float(summary.get("final_energy_total")),
        ),
        "final_energy_total_exact": _first_not_none(
            _maybe_float(final_row.get("energy_total_exact")),
            _maybe_float(summary.get("final_energy_total_exact")),
        ),
        "final_abs_energy_total_error": _first_not_none(
            final_error,
            _maybe_float(summary.get("final_abs_energy_total_error")),
        ),
        "mean_abs_energy_total_error": _first_not_none(
            mean_error,
            _maybe_float(summary.get("mean_abs_energy_total_error")),
        ),
        "max_abs_energy_total_error": _first_not_none(
            max_error,
            _maybe_float(summary.get("max_abs_energy_total_error")),
        ),
        "fidelity_min": _first_not_none(fidelity_min, _maybe_float(extra.get("fidelity_min"))),
        "rho_miss_max": _first_not_none(rho_miss_max, _maybe_float(extra.get("rho_miss_max"))),
        "final_logical_block_count": _first_not_none(
            _maybe_int(summary.get("final_logical_block_count")),
            _maybe_int(extra.get("final_logical_block_count")),
            _maybe_int(final_row.get("logical_block_count")),
        ),
        "final_runtime_parameter_count": _first_not_none(
            _maybe_int(summary.get("final_runtime_parameter_count")),
            _maybe_int(extra.get("final_runtime_parameter_count")),
            _maybe_int(final_row.get("runtime_parameter_count")),
        ),
    }


def _row_from_run_artifact(
    payload: Mapping[str, Any],
    *,
    case: FixedMclachlanBenchmarkCase,
    source_payload: Mapping[str, Any],
    t_final: float,
    num_times: int,
    drive_kwargs: Mapping[str, Any],
    state_cost: overlay.CircuitCostRow,
    controller_cost: overlay.CircuitCostRow,
    artifact_run_json: Path | str | None,
    artifact_manifest_json: Path | str | None = None,
    artifact_rows_json: Path | str | None = None,
    artifact_summary_json: Path | str | None = None,
    preferred_fake_backends: Sequence[str] | None = None,
) -> dict[str, Any]:
    metrics = _artifact_trajectory_metrics(payload)
    intervals = max(int(num_times) - 1, 0)
    row = FixedMclachlanBenchmarkRow(
        case_id=str(case.case_id),
        method_id=METHOD_ID,
        method_kind=METHOD_KIND,
        status="ok",
        decision_mode=DECISION_MODE,
        diagnostic_exact_assisted=True,
        qpu_faithful=False,
        seed_family=SEED_FAMILY,
        controller_json=str(case.controller_json),
        source_artifact_json=str(case.source_artifact_json),
        drive_enabled=_maybe_bool(drive_kwargs.get("enable_drive")),
        t_final=float(t_final),
        num_times=int(num_times),
        final_energy_total=_maybe_float(metrics.get("final_energy_total")),
        final_energy_total_exact=_maybe_float(metrics.get("final_energy_total_exact")),
        final_abs_energy_total_error=_maybe_float(metrics.get("final_abs_energy_total_error")),
        mean_abs_energy_total_error=_maybe_float(metrics.get("mean_abs_energy_total_error")),
        max_abs_energy_total_error=_maybe_float(metrics.get("max_abs_energy_total_error")),
        fidelity_min=_maybe_float(metrics.get("fidelity_min")),
        rho_miss_max=_maybe_float(metrics.get("rho_miss_max")),
        final_logical_block_count=_maybe_int(metrics.get("final_logical_block_count")),
        final_runtime_parameter_count=_maybe_int(metrics.get("final_runtime_parameter_count")),
        state_at_time_scope=STATE_SCOPE,
        state_at_time_basis="fixed McLachlan state scaffold",
        state_at_time_2q=_maybe_int(state_cost.compiled_count_2q),
        state_at_time_depth=_maybe_int(state_cost.compiled_depth),
        state_at_time_size=_maybe_int(state_cost.compiled_size),
        full_horizon_scope=HORIZON_SCOPE,
        full_horizon_basis=f"{intervals} repeated fixed state scaffolds",
        full_horizon_intervals=int(intervals),
        full_horizon_horizon_2q=_mul_optional_int(state_cost.compiled_count_2q, intervals),
        full_horizon_depth_serial=_mul_optional_int(state_cost.compiled_depth, intervals),
        controller_state_scope=CONTROLLER_STATE_SCOPE,
        controller_state_basis="controller state-at-time compile reference",
        controller_state_2q=_maybe_int(controller_cost.compiled_count_2q),
        controller_state_depth=_maybe_int(controller_cost.compiled_depth),
        controller_state_size=_maybe_int(controller_cost.compiled_size),
        backend_name=_as_optional_str(state_cost.backend_name or controller_cost.backend_name),
        seed_transpiler=_maybe_int(state_cost.seed_transpiler),
        optimization_level=_maybe_int(state_cost.optimization_level),
        preferred_fake_backends=tuple(str(x) for x in (preferred_fake_backends or ())),
        artifact_run_json=_as_optional_str(artifact_run_json),
        artifact_manifest_json=_as_optional_str(artifact_manifest_json),
        artifact_rows_json=_as_optional_str(artifact_rows_json),
        artifact_summary_json=_as_optional_str(artifact_summary_json),
    )
    out = _jsonable(row)
    # Preserve a small provenance seam for drive inheritance without bloating the flat row.
    out["source_controller_run_tag"] = _as_optional_str(source_payload.get("run_tag"))
    out["exact_steps_multiplier"] = _maybe_int(drive_kwargs.get("exact_steps_multiplier"))
    return out


def _run_case(
    case: FixedMclachlanBenchmarkCase,
    *,
    output_dir: Path,
    manifest_json: Path,
    rows_json: Path,
    summary_json: Path,
) -> _CaseRunRecord:
    source_payload = overlay._load_source_payload(Path(case.controller_json))
    source_rows = overlay._state_sample_rows(source_payload)
    times = overlay._source_times(source_payload, source_rows)
    t_final = float(times[-1])
    num_times = int(times.size)
    compile_defaults = _compile_defaults_for_case(case, source_payload)
    controller_cost = _required_controller_cost_row(source_payload)
    drive_kwargs = _drive_kwargs_from_source(source_payload)
    spec = _run_spec_for_case(case)

    run_dir = Path(output_dir) / "runs"
    run_record = fixed.run_fixed_manifold_exact(
        spec,
        tag=str(case.case_id),
        output_dir=run_dir,
        t_final=t_final,
        num_times=num_times,
        miss_threshold=float(case.miss_threshold),
        gain_ratio_threshold=float(case.gain_ratio_threshold),
        append_margin_abs=float(case.append_margin_abs),
        **drive_kwargs,
    )
    raw_engine_json = Path(str(run_record.get("output_json", run_dir / f"{spec.name}.json")))
    stable_run_json = run_dir / f"{case.case_id}.json"
    if not raw_engine_json.exists():
        raise ValueError(f"fixed-manifold engine did not write expected run artifact: {raw_engine_json}")
    stable_run_json.parent.mkdir(parents=True, exist_ok=True)
    if raw_engine_json.resolve() != stable_run_json.resolve():
        shutil.copy2(raw_engine_json, stable_run_json)

    run_payload = _read_json(stable_run_json)
    state_cost, raw_compile_rows = _compile_fixed_state_scaffold(
        spec=spec,
        case=case,
        compile_defaults=compile_defaults,
    )
    row = _row_from_run_artifact(
        run_payload,
        case=case,
        source_payload=source_payload,
        t_final=t_final,
        num_times=num_times,
        drive_kwargs=drive_kwargs,
        state_cost=state_cost,
        controller_cost=controller_cost,
        artifact_run_json=stable_run_json,
        artifact_manifest_json=manifest_json,
        artifact_rows_json=rows_json,
        artifact_summary_json=summary_json,
        preferred_fake_backends=compile_defaults["preferred_fake_backends"],
    )
    return _CaseRunRecord(
        case=case,
        spec=spec,
        run_record=dict(run_record),
        run_json=stable_run_json,
        row=row,
        raw_compile_rows=raw_compile_rows,
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
        "benchmark": "hh_fixed_mclachlan_time_dynamics",
        "method_contract": {
            "method_id": METHOD_ID,
            "method_kind": METHOD_KIND,
            "decision_mode": DECISION_MODE,
            "diagnostic_exact_assisted": True,
            "qpu_faithful": False,
            "compile_cost_policy": (
                "compile one representative fixed state scaffold; repeated-horizon budget is "
                "state cost multiplied by num_times-1 intervals"
            ),
            "controller_reference_policy": "fail_closed_required_source_compile_reference",
        },
        "command": command,
        "output_dir": str(output_dir),
        "paths": {
            "manifest_json": str(manifest_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
            "runs_dir": str(Path(output_dir) / "runs"),
        },
        "cases": [
            {
                "case": _jsonable(record.case),
                "resolved_run_spec": _jsonable(record.spec),
                "run_record": _jsonable(record.run_record),
                "artifact_run_json": str(record.run_json),
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
        "benchmark": "hh_fixed_mclachlan_time_dynamics",
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
                "fidelity_min": row.get("fidelity_min"),
                "rho_miss_max": row.get("rho_miss_max"),
                "state_at_time_2q": row.get("state_at_time_2q"),
                "state_at_time_depth": row.get("state_at_time_depth"),
                "full_horizon_horizon_2q": row.get("full_horizon_horizon_2q"),
                "full_horizon_depth_serial": row.get("full_horizon_depth_serial"),
                "controller_state_2q": row.get("controller_state_2q"),
                "controller_state_depth": row.get("controller_state_depth"),
            }
            for row in rows
        ],
    }


def run_benchmark(
    *,
    cases: Sequence[FixedMclachlanBenchmarkCase],
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
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the benchmark-local HH L2 t=8 fixed-manifold McLachlan row."
    )
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-json", type=Path, default=None)
    parser.add_argument("--source-artifact-json", type=Path, default=None)
    parser.add_argument("--compile-backend-name", type=str, default=None)
    parser.add_argument("--compile-seed-transpiler", type=int, default=None)
    parser.add_argument("--compile-optimization-level", type=int, default=None)
    parser.add_argument("--compile-preferred-fake-backends", type=str, default=None)
    return parser


def _case_from_args(args: argparse.Namespace) -> FixedMclachlanBenchmarkCase:
    case = _case_by_id(str(args.case_id))
    preferred = _parse_string_tuple(args.compile_preferred_fake_backends)
    return replace(
        case,
        controller_json=Path(args.controller_json) if args.controller_json is not None else case.controller_json,
        source_artifact_json=(
            Path(args.source_artifact_json)
            if args.source_artifact_json is not None
            else case.source_artifact_json
        ),
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
        return " ".join([sys.executable, "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_fixed_mclachlan_benchmark", *sys.argv[1:]])
    return " ".join(["python", "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_fixed_mclachlan_benchmark", *map(str, argv)])


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
    print(f"artifact_run_json={row.get('artifact_run_json')}")
    print(f"final_abs_energy_total_error={row.get('final_abs_energy_total_error')}")
    print(f"mean_abs_energy_total_error={row.get('mean_abs_energy_total_error')}")
    print(f"max_abs_energy_total_error={row.get('max_abs_energy_total_error')}")
    print(f"state_at_time_2q={row.get('state_at_time_2q')}")
    print(f"state_at_time_depth={row.get('state_at_time_depth')}")
    print(f"full_horizon_horizon_2q={row.get('full_horizon_horizon_2q')}")
    print(f"full_horizon_depth_serial={row.get('full_horizon_depth_serial')}")
    print(f"controller_state_2q={row.get('controller_state_2q')}")
    print(f"controller_state_depth={row.get('controller_state_depth')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
