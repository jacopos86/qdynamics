#!/usr/bin/env python3
"""Benchmark-local generic static ED reference runner.

This row exposes the existing problem-local exact target machinery as a
benchmark artifact producer.  It intentionally does not call Phase3/static ADAPT
controller paths and does not implement a new diagonalization backend; the only
physics call is ``ResolvedProblemContext.exact_target.resolve_energy(ai_log=None)``.
"""

from __future__ import annotations

import json
import time
from argparse import Namespace
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.exact_bench.benchmark_metrics_proxy import write_proxy_sidecars
from pipelines.exact_bench.table_i_canonical_cases import (
    table_i_canonical_case_ids,
    table_i_canonical_spec_by_case_id,
)
from pipelines.static_adapt.builders.problem_registry import (
    ProblemRequest,
    ResolvedProblemContext,
    resolve_problem_context,
)
from pipelines.exact_bench.static_benchmark_runtime import (
    HamiltonianBenchmarkSpec,
)

SCHEMA_VERSION = "generic_static_ed_reference_v1"
_METHOD_ID = "static_ed_reference"
_DENSE_EIGH_MAX_DIM_DEFAULT = 1024

GENERIC_STATIC_ED_REFERENCE_FAMILIES = (
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_restricted_closed_shell",
    "molecular_vibronic_h2",
)


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            return str(value)
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_static_ed_reference_case_ids(family: str) -> tuple[str, ...]:
    """Return canonical Paper-I Table-I case IDs implemented by this ED row."""
    family_key = str(family).strip()
    if family_key not in GENERIC_STATIC_ED_REFERENCE_FAMILIES:
        return ()
    if family_key == "molecular_restricted_closed_shell":
        return ("molecular_restricted_closed_shell_L2",)
    return table_i_canonical_case_ids(family_key)


def _spec_by_case_id(family: str, case_id: str) -> HamiltonianBenchmarkSpec:
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    if family_key not in GENERIC_STATIC_ED_REFERENCE_FAMILIES:
        raise ValueError(f"static_ed_reference is not implemented for family={family_key!r}")
    return table_i_canonical_spec_by_case_id(family_key, case_key)


def _namespace_from_base_args(argv: Sequence[str]) -> Namespace:
    defaults: dict[str, Any] = {
        "problem": "hubbard",
        "L": 2,
        "t": 1.0,
        "u": 4.0,
        "dv": 0.0,
        "omega0": 1.0,
        "g_ep": 0.5,
        "n_ph_max": 1,
        "boson_encoding": "binary",
        "ordering": "blocked",
        "boundary": "periodic",
        "include_zero_point": True,
        "molecular_problem_json": None,
        "v_nn": 0.0,
        "t_prime": 0.0,
        "n_fermions": None,
        "dense_eigh_max_dim": _DENSE_EIGH_MAX_DIM_DEFAULT,
    }
    key_map = {
        "--problem": "problem",
        "--L": "L",
        "--t": "t",
        "--u": "u",
        "--dv": "dv",
        "--omega0": "omega0",
        "--g-ep": "g_ep",
        "--n-ph-max": "n_ph_max",
        "--boson-encoding": "boson_encoding",
        "--ordering": "ordering",
        "--boundary": "boundary",
        "--molecular-problem-json": "molecular_problem_json",
        "--v-nn": "v_nn",
        "--t-prime": "t_prime",
        "--n-fermions": "n_fermions",
        "--dense-eigh-max-dim": "dense_eigh_max_dim",
    }
    int_keys = {"L", "n_ph_max", "n_fermions", "dense_eigh_max_dim"}
    float_keys = {"t", "u", "dv", "omega0", "g_ep", "v_nn", "t_prime"}
    values = dict(defaults)
    idx = 0
    argv_tuple = tuple(str(x) for x in argv)
    while idx < len(argv_tuple):
        token = argv_tuple[idx]
        if token == "--include-zero-point":
            values["include_zero_point"] = True
            idx += 1
            continue
        if token == "--no-include-zero-point":
            values["include_zero_point"] = False
            idx += 1
            continue
        if token not in key_map:
            idx += 1
            continue
        if idx + 1 >= len(argv_tuple):
            raise ValueError(f"Missing value for {token}")
        key = key_map[token]
        raw = argv_tuple[idx + 1]
        if key in int_keys and raw not in {"", "None", "none"}:
            values[key] = int(raw)
        elif key in float_keys:
            values[key] = float(raw)
        elif key in {"n_fermions", "dense_eigh_max_dim"} and raw in {"", "None", "none"}:
            values[key] = None
        else:
            values[key] = raw
        idx += 2
    return Namespace(**values)


def _resolve_context_from_spec(spec: HamiltonianBenchmarkSpec) -> ResolvedProblemContext:
    request = ProblemRequest.from_namespace(_namespace_from_base_args(spec.base_pipeline_args))
    return resolve_problem_context(request)


def _spec_metadata(spec: HamiltonianBenchmarkSpec) -> dict[str, Any]:
    features = getattr(spec, "features", None)
    return {
        "benchmark_id": str(spec.benchmark_id),
        "family": str(spec.family),
        "base_pipeline_args": list(spec.base_pipeline_args),
        "split": str(spec.split),
        "tags": list(spec.tags),
        "features": asdict(features) if is_dataclass(features) else _json_default(features),
    }


def _molecular_resource_guard(spec: HamiltonianBenchmarkSpec) -> dict[str, Any] | None:
    if str(spec.family) != "molecular_restricted_closed_shell":
        return None
    namespace = _namespace_from_base_args(spec.base_pipeline_args)
    max_dim_raw = getattr(namespace, "dense_eigh_max_dim", _DENSE_EIGH_MAX_DIM_DEFAULT)
    max_dim = _DENSE_EIGH_MAX_DIM_DEFAULT if max_dim_raw is None else int(max_dim_raw)
    n_qubits = int(getattr(spec.features, "n_qubits"))
    dense_dim = int(2**n_qubits)
    if dense_dim <= max_dim:
        return None
    return {
        "resource_guard": True,
        "reason": "molecular dense Hilbert dimension exceeds --dense-eigh-max-dim",
        "dense_hilbert_dimension": dense_dim,
        "dense_eigh_max_dim": max_dim,
        "num_qubits": n_qubits,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_artifacts(output_dir: Path, payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "result.json", payload)
    _write_json(output_dir / "rows.json", {"schema": f"{SCHEMA_VERSION}_rows", "rows": list(rows)})
    _write_json(
        output_dir / "manifest.json",
        {"schema": f"{SCHEMA_VERSION}_manifest", **{k: v for k, v in dict(payload).items() if k != "schema"}},
    )
    _write_json(output_dir / "generic_static_single.json", payload)
    write_proxy_sidecars(rows, output_dir, summary_extras={"schema_source": SCHEMA_VERSION})
    return dict(payload)


def _guardrails(*, exact_reference_usage: str) -> dict[str, Any]:
    return {
        "uses_exact_for_decision": False,
        "exact_reference_usage": str(exact_reference_usage),
        "phase3_controller_called": False,
        "qiskit_boundary": "not_used",
        "static_adapt_controller_boundary": "not_called",
        "new_ed_solver_written": False,
    }


def _resource_guard_payload(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    spec: HamiltonianBenchmarkSpec,
    started_utc: str,
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    finished = _utc_now()
    feature_l = getattr(spec.features, "L", None)
    row = {
        "run_id": f"{case_id}::{_METHOD_ID}",
        "schema": SCHEMA_VERSION,
        "family": family,
        "problem": family,
        "L": None if feature_l is None else int(feature_l),
        "hamiltonian_id": case_id,
        "case_id": case_id,
        "method_id": _METHOD_ID,
        "method_kind": "classical_reference",
        "ansatz_name": "not_applicable_exact_diagonalization",
        "algorithm_origin": "resolved_problem_context_exact_target",
        "status": "skipped_resource_guard",
        "reason": str(guard["reason"]),
        "energy": None,
        "exact_energy": None,
        "exact_gs_energy": None,
        "same_cutoff_exact_gs_energy": None,
        "delta_E_abs": None,
        "abs_delta_e": None,
        "num_qubits": int(guard["num_qubits"]),
        "sector_label": "not_resolved_resource_guard",
        "exact_target_kind": "not_resolved_resource_guard",
        "exact_comparison_space_label": "not_resolved_resource_guard",
        "runtime_s": 0.0,
        "shots_total": None,
        "num_parameters": 0,
        "count_2q": 0,
        "depth_proxy": 0,
        "circuit_depth": 0,
        "uses_exact_for_decision": False,
        "exact_reference_usage": "not_resolved_resource_guard",
        "phase3_controller_called": False,
        "qiskit_boundary": "not_used",
        "resource_guard": True,
        "resource_guard_kind": "molecular_dense_hilbert_dimension",
        "dense_hilbert_dimension": int(guard["dense_hilbert_dimension"]),
        "dense_eigh_max_dim": int(guard["dense_eigh_max_dim"]),
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "skipped_resource_guard",
        "reason": str(guard["reason"]),
        "runner": "pipelines.exact_bench.generic_static_ed_reference.run_static_ed_reference_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(exact_reference_usage="not_resolved_resource_guard"),
        "resource_guard": dict(guard),
        "spec": _spec_metadata(spec),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    return _write_artifacts(output_dir, payload, [row])


def _failure_payload(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    reason: str,
    exception_type: str,
    started_utc: str | None = None,
) -> dict[str, Any]:
    started = started_utc or _utc_now()
    finished = _utc_now()
    row = {
        "run_id": f"{case_id}::{_METHOD_ID}",
        "schema": SCHEMA_VERSION,
        "family": family,
        "problem": family,
        "L": None,
        "hamiltonian_id": case_id,
        "case_id": case_id,
        "method_id": _METHOD_ID,
        "method_kind": "classical_reference",
        "ansatz_name": "not_applicable_exact_diagonalization",
        "algorithm_origin": "resolved_problem_context_exact_target",
        "status": "failed",
        "reason": reason,
        "exception_type": exception_type,
        "energy": None,
        "exact_energy": None,
        "exact_gs_energy": None,
        "same_cutoff_exact_gs_energy": None,
        "delta_E_abs": None,
        "abs_delta_e": None,
        "runtime_s": None,
        "shots_total": None,
        "num_parameters": 0,
        "count_2q": 0,
        "depth_proxy": 0,
        "circuit_depth": 0,
        "uses_exact_for_decision": False,
        "exact_reference_usage": "failed_before_or_during_exact_target_resolution",
        "phase3_controller_called": False,
        "qiskit_boundary": "not_used",
        "started_utc": started,
        "finished_utc": finished,
    }
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "failed",
        "reason": reason,
        "exception_type": exception_type,
        "runner": "pipelines.exact_bench.generic_static_ed_reference.run_static_ed_reference_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(exact_reference_usage="failed_before_or_during_exact_target_resolution"),
        "result": row,
        "rows": [row],
        "started_utc": started,
        "finished_utc": finished,
    }
    return _write_artifacts(output_dir, payload, [row])


def _run_static_ed_reference_single_impl(*, family: str, case_id: str, output_dir: Path) -> dict[str, Any]:
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    output = Path(output_dir)
    started_utc = _utc_now()
    t0 = time.perf_counter()

    spec = _spec_by_case_id(family_key, case_key)
    guard = _molecular_resource_guard(spec)
    if guard is not None:
        return _resource_guard_payload(
            family=family_key,
            case_id=case_key,
            output_dir=output,
            spec=spec,
            started_utc=started_utc,
            guard=guard,
        )

    context = _resolve_context_from_spec(spec)
    exact_energy = float(context.exact_target.resolve_energy(ai_log=None))
    walltime = float(time.perf_counter() - t0)
    finished_utc = _utc_now()
    comparison_space = str(
        getattr(context, "exact_comparison_space_label", None)
        or getattr(context.exact_target, "comparison_space_label", "")
    )
    row = {
        "run_id": f"{case_key}::{_METHOD_ID}",
        "schema": SCHEMA_VERSION,
        "family": family_key,
        "problem": family_key,
        "L": int(context.request.num_sites),
        "hamiltonian_id": case_key,
        "case_id": case_key,
        "method_id": _METHOD_ID,
        "method_kind": "classical_reference",
        "ansatz_name": "not_applicable_exact_diagonalization",
        "algorithm_origin": "resolved_problem_context_exact_target",
        "status": "ok",
        "energy": exact_energy,
        "exact_energy": exact_energy,
        "exact_gs_energy": exact_energy,
        "same_cutoff_exact_gs_energy": exact_energy,
        "delta_E_abs": 0.0,
        "abs_delta_e": 0.0,
        "infidelity_exact": None,
        "infidelity_status": "not_available_exact_state_not_exposed_by_problem_context",
        "observable_error_status": "not_computed_energy_reference_only",
        "static_shot_estimate_status": "not_applicable_classical_reference",
        "shots_total": None,
        "num_qubits": int(context.layout.total_qubits),
        "num_parameters": 0,
        "nfev": None,
        "nit": None,
        "runtime_s": walltime,
        "count_2q": 0,
        "depth_proxy": 0,
        "circuit_depth": 0,
        "sector_label": str(context.sector.label),
        "exact_target_kind": str(context.exact_target.kind),
        "exact_comparison_space_label": comparison_space,
        "uses_exact_for_decision": False,
        "exact_reference_usage": "primary_result_exact_diagonalization",
        "phase3_controller_called": False,
        "pauli_ordering": "left-to-right q_(n-1)...q_0; qubit 0 rightmost",
        "internal_pauli_alphabet": "e/x/y/z",
        "qiskit_boundary": "not_used",
        "resource_guard": False,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family_key,
        "case_id": case_key,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "completed",
        "runner": "pipelines.exact_bench.generic_static_ed_reference.run_static_ed_reference_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(exact_reference_usage="primary_result_exact_diagonalization"),
        "spec": _spec_metadata(spec),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }
    return _write_artifacts(output, payload, [row])


def run_static_ed_reference_single(*, family: str, case_id: str, output_dir: Path) -> dict[str, Any]:
    """Run one generic static ED reference case and always emit artifacts."""
    started_utc = _utc_now()
    try:
        return _run_static_ed_reference_single_impl(
            family=family,
            case_id=case_id,
            output_dir=output_dir,
        )
    except Exception as exc:
        return _failure_payload(
            family=str(family).strip(),
            case_id=str(case_id).strip(),
            output_dir=Path(output_dir),
            reason=str(exc),
            exception_type=type(exc).__name__,
            started_utc=started_utc,
        )


__all__ = [
    "GENERIC_STATIC_ED_REFERENCE_FAMILIES",
    "SCHEMA_VERSION",
    "default_static_ed_reference_case_ids",
    "run_static_ed_reference_single",
]
