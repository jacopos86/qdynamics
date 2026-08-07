#!/usr/bin/env python3
"""Generic static Qiskit Algorithms AdaptVQE benchmark runner.

This is an exact-bench-local library ADAPT reference row.  It uses
``qiskit_algorithms.AdaptVQE`` with the problem-local ``full_meta`` pool
converted to Qiskit ``SparsePauliOp`` generators, does not call the
Phase3/SNAKE/static_adapt controller, and resolves exact references only after
the optimizer/adaptive loop has returned.
"""

from __future__ import annotations

import json
import os
import time
from argparse import Namespace
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.benchmark_metrics_proxy import write_proxy_sidecars
from pipelines.exact_bench.comparator_provenance import comparator_source_fields
from pipelines.exact_bench.molecular_vibronic_h2_fixture_override import (
    with_molecular_vibronic_h2_fixture_override,
)
from pipelines.exact_bench.generic_static_hea_qiskit_vqe import sector_probability
from pipelines.exact_bench.generic_static_adapt_variants import build_full_meta_candidate_pool
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_STATIC_SUITE_PROFILE_ENV,
    table_i_canonical_case_ids,
    table_i_canonical_spec_by_case_id,
)
from pipelines.exact_bench.qiskit_adaptvqe_adapter import (
    QiskitAdaptVQEUnavailable,
    build_reference_state_circuit,
    hamiltonian_term_pool_labels,
    has_qiskit_adaptvqe_support,
    import_qiskit_adaptvqe_components,
    pauli_poly_to_sparse_pauli_op,
)
from pipelines.static_adapt.builders.problem_registry import (
    ProblemRequest,
    ResolvedProblemContext,
    resolve_problem_context,
)
from pipelines.static_adapt.optimization.phase3_policy_optuna import (
    HamiltonianBenchmarkSpec,
)

SCHEMA_VERSION = "generic_static_qiskit_adapt_vqe_v2"
_METHOD_ID = "static_qiskit_adapt_vqe"
_RUNNER_MODULE = "pipelines.exact_bench.generic_static_qiskit_adapt_vqe"
_QUBIT_CAP = 10
_POOL_TERM_CAP = 128
_RESOURCE_QUBIT_CAP_ENV = "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP"
_RESOURCE_POOL_TERM_CAP_ENV = "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP"
_DEFAULT_MAX_ADAPT_ITERATIONS = 1000
_DEFAULT_OPTIMIZER_MAXITER = 200
_DEFAULT_SHOTS_PER_PAULI_TERM_PROXY = 1024
_SHOT_PROXY_FORMULA = (
    "shots_total = shots_per_pauli_term_proxy * hamiltonian_pauli_term_count * "
    "(energy_eval_count_proxy + gradient_operator_probe_count_proxy)"
)
_COMPILED_BASIS_GATES = (
    "id",
    "x",
    "sx",
    "rx",
    "ry",
    "rz",
    "h",
    "s",
    "sdg",
    "cx",
    "cz",
)


def _resource_cap_from_env(name: str, default: int | None) -> int | None:
    raw = os.environ.get(str(name), "")
    if raw is None or str(raw).strip() == "":
        return default
    key = str(raw).strip().lower()
    if key in {"0", "none", "off", "false", "unbounded", "unlimited"}:
        return None
    value = int(key)
    if value < 1:
        return None
    return int(value)

GENERIC_STATIC_QISKIT_ADAPTVQE_FAMILIES = (
    "hh",
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_vibronic_h2",
)


def _suite_profile_from_env() -> str | None:
    raw = os.environ.get(TABLE_I_STATIC_SUITE_PROFILE_ENV, "")
    value = str(raw or "").strip()
    return value or None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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


def default_static_qiskit_adapt_vqe_case_ids(family: str) -> tuple[str, ...]:
    """Return canonical Table-I cases for this AdaptVQE row."""
    family_key = str(family).strip()
    if family_key not in GENERIC_STATIC_QISKIT_ADAPTVQE_FAMILIES:
        return ()
    return table_i_canonical_case_ids(family_key, _suite_profile_from_env())


def _spec_by_case_id(family: str, case_id: str) -> HamiltonianBenchmarkSpec:
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    profile = _suite_profile_from_env()
    if family_key not in GENERIC_STATIC_QISKIT_ADAPTVQE_FAMILIES:
        raise ValueError(f"{_METHOD_ID} is not implemented for family={family_key!r}")
    if case_key not in default_static_qiskit_adapt_vqe_case_ids(family_key):
        raise ValueError(f"{_METHOD_ID} is not implemented for {family_key}/{case_key}")
    return with_molecular_vibronic_h2_fixture_override(
        table_i_canonical_spec_by_case_id(family_key, case_key, profile),
        family=family_key,
    )


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
        "molecular_vibronic_h2_fixture_json": None,
        "v_nn": 0.0,
        "t_prime": 0.0,
        "n_fermions": None,
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
        "--molecular-vibronic-h2-fixture-json": "molecular_vibronic_h2_fixture_json",
        "--v-nn": "v_nn",
        "--t-prime": "t_prime",
        "--n-fermions": "n_fermions",
    }
    int_keys = {"L", "n_ph_max", "n_fermions"}
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
        elif key == "n_fermions" and raw in {"", "None", "none"}:
            values[key] = None
        else:
            values[key] = raw
        idx += 2
    return Namespace(**values)


def _resolve_context_from_spec(spec: HamiltonianBenchmarkSpec) -> ResolvedProblemContext:
    request = ProblemRequest.from_namespace(_namespace_from_base_args(spec.base_pipeline_args))
    return resolve_problem_context(request)


def _safe_exact_energy(context: ResolvedProblemContext) -> float | None:
    try:
        return float(context.exact_target.resolve_energy(ai_log=None))
    except TypeError:
        try:
            return float(context.exact_target.resolve_energy())
        except Exception:
            return None
    except Exception:
        return None


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _source_fields(**overrides: Any) -> dict[str, Any]:
    return comparator_source_fields(_METHOD_ID, runner_module=_RUNNER_MODULE, **overrides)


def _write_artifacts(output_dir: Path, payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_with_source = dict(payload)
    payload_with_source.setdefault("comparator_source", _source_fields())
    _write_json(output_dir / "result.json", payload_with_source)
    _write_json(output_dir / "rows.json", {"schema": f"{SCHEMA_VERSION}_rows", "rows": list(rows)})
    _write_json(
        output_dir / "manifest.json",
        {"schema": f"{SCHEMA_VERSION}_manifest", **{k: v for k, v in payload_with_source.items() if k != "schema"}},
    )
    _write_json(output_dir / "generic_static_single.json", payload_with_source)
    write_proxy_sidecars(rows, output_dir, summary_extras={"schema_source": SCHEMA_VERSION})
    return dict(payload_with_source)


def _guardrails(*, exact_reference_usage: str) -> dict[str, Any]:
    return {
        "uses_exact_for_decision": False,
        "exact_reference_usage": str(exact_reference_usage),
        "phase3_controller_called": False,
        "static_adapt_controller_boundary": "not_called",
        "qiskit_boundary": "pipelines.exact_bench_only",
        "qiskit_algorithms_boundary": "pipelines.exact_bench_only",
        "adapt_append_only": True,
        "phase3_emulation": False,
        "pool_source": "problem_local_full_meta_pool",
        "pool_name": "full_meta",
        "taxonomy_role": "same_pool_controller_comparator",
    }


def _base_row(
    *,
    family: str,
    case_id: str,
    status: str,
    started_utc: str,
    finished_utc: str,
) -> dict[str, Any]:
    return {
        "run_id": f"{case_id}::{_METHOD_ID}",
        "schema": SCHEMA_VERSION,
        "family": family,
        "problem": family,
        "hamiltonian_id": case_id,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "method_kind": "adapt_reference",
        "ansatz_name": "qiskit_algorithms_adaptvqe_full_meta_pool",
        "algorithm_origin": "qiskit_algorithms_adaptvqe_full_meta_exact_bench",
        "status": status,
        "qiskit_available": None,
        "uses_exact_for_decision": False,
        "exact_reference_usage": "reporting_only_after_optimization",
        "phase3_controller_called": False,
        "static_adapt_controller_boundary": "not_called",
        "qiskit_boundary": "pipelines.exact_bench_only",
        "qiskit_algorithms_boundary": "pipelines.exact_bench_only",
        "adapt_append_only": True,
        "phase3_emulation": False,
        "pool_source": "problem_local_full_meta_pool",
        "pool_name": "full_meta",
        "taxonomy_role": "same_pool_controller_comparator",
        "pauli_ordering": "left-to-right q_(n-1)...q_0; qubit 0 rightmost",
        "internal_pauli_alphabet": "e/x/y/z",
        "shots_total": 0,
        "static_shot_estimate_status": "not_applicable_not_completed",
        "shot_proxy_formula": _SHOT_PROXY_FORMULA,
        "shots_per_pauli_term_proxy": _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
        "hamiltonian_pauli_term_count": 0,
        "energy_eval_count_proxy": 0,
        "gradient_scan_count_proxy": 0,
        "gradient_operator_probe_count_proxy": 0,
        **_source_fields(),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }


def _skip_payload(*, family: str, case_id: str, output_dir: Path, reason: str, started_utc: str) -> dict[str, Any]:
    finished = _utc_now()
    row = _base_row(family=family, case_id=case_id, status="skipped_optional_dependency", started_utc=started_utc, finished_utc=finished)
    row.update(
        {
            "qiskit_available": False,
            "reason": reason,
            "exact_reference_usage": "not_resolved_for_dependency_skip",
            "energy": None,
            "exact_energy": None,
            "delta_E_abs": None,
            "num_parameters": 0,
            "selected_operator_count": 0,
            "adapt_depth_reached": 0,
            "adapt_stop_reason": "optional_dependency_unavailable",
        }
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "skipped_optional_dependency",
        "qiskit_available": False,
        "reason": reason,
        "runner": "pipelines.exact_bench.generic_static_qiskit_adapt_vqe.run_static_qiskit_adapt_vqe_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(exact_reference_usage="not_resolved_for_dependency_skip"),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    return _write_artifacts(output_dir, payload, [row])


def _resource_guard_payload(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    spec: HamiltonianBenchmarkSpec,
    started_utc: str,
    reason: str,
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    finished = _utc_now()
    row = _base_row(family=family, case_id=case_id, status="skipped_resource_guard", started_utc=started_utc, finished_utc=finished)
    row.update(
        {
            "qiskit_available": True,
            "reason": reason,
            "exact_reference_usage": "not_resolved_resource_guard",
            "energy": None,
            "exact_energy": None,
            "delta_E_abs": None,
            "num_qubits": guard.get("num_qubits"),
            "pool_term_count": guard.get("pool_term_count"),
            "num_parameters": 0,
            "selected_operator_count": 0,
            "adapt_depth_reached": 0,
            "adapt_stop_reason": "resource_guard",
            "resource_guard": True,
        }
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "skipped_resource_guard",
        "qiskit_available": True,
        "reason": reason,
        "runner": "pipelines.exact_bench.generic_static_qiskit_adapt_vqe.run_static_qiskit_adapt_vqe_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
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
    qiskit_available: bool,
    started_utc: str,
) -> dict[str, Any]:
    finished = _utc_now()
    row = _base_row(family=family, case_id=case_id, status="failed", started_utc=started_utc, finished_utc=finished)
    row.update(
        {
            "qiskit_available": bool(qiskit_available),
            "reason": reason,
            "exception_type": exception_type,
            "exact_reference_usage": "reporting_only_after_optimization_or_not_reached",
            "energy": None,
            "exact_energy": None,
            "delta_E_abs": None,
            "runtime_s": None,
            "num_parameters": None,
            "selected_operator_count": None,
            "adapt_depth_reached": None,
            "adapt_stop_reason": "failed",
        }
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "failed",
        "qiskit_available": bool(qiskit_available),
        "reason": reason,
        "exception_type": exception_type,
        "runner": "pipelines.exact_bench.generic_static_qiskit_adapt_vqe.run_static_qiskit_adapt_vqe_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(exact_reference_usage="reporting_only_after_optimization_or_not_reached"),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    return _write_artifacts(output_dir, payload, [row])


def _circuit_stats(circuit: Any) -> dict[str, Any]:
    depth: int | None
    try:
        depth = int(circuit.depth())
    except Exception:
        depth = None
    try:
        op_counts = {str(k): int(v) for k, v in dict(circuit.count_ops()).items()}
    except Exception:
        op_counts = {}
    count_2q = 0
    try:
        for item in circuit.data:
            operation = getattr(item, "operation", None)
            if operation is None and isinstance(item, (tuple, list)) and item:
                operation = item[0]
            if int(getattr(operation, "num_qubits", 0)) == 2:
                count_2q += 1
    except Exception:
        count_2q = int(op_counts.get("cx", 0) + op_counts.get("cz", 0))
    try:
        from pipelines.qiskit_backend_tools import safe_two_qubit_depth

        depth_2q = int(safe_two_qubit_depth(circuit))
    except Exception:
        depth_2q = None
    return {"depth": depth, "depth_2q": depth_2q, "count_2q": int(count_2q), "op_counts": op_counts}


def _bind_result_circuit(result: Any, fallback_circuit: Any) -> Any:
    circuit = getattr(result, "optimal_circuit", None) or fallback_circuit
    optimal_parameters = getattr(result, "optimal_parameters", None) or {}
    try:
        if optimal_parameters:
            return circuit.assign_parameters(optimal_parameters, inplace=False)
        point = getattr(result, "optimal_point", None)
        if point is not None and getattr(circuit, "num_parameters", 0):
            params = list(getattr(circuit, "parameters", []))
            values = np.asarray(point, dtype=float).reshape(-1)
            if len(params) == int(values.size):
                return circuit.assign_parameters({p: float(values[i]) for i, p in enumerate(params)}, inplace=False)
    except Exception:
        return circuit
    return circuit


def _final_statevector_from_circuit(components: Any, circuit: Any) -> np.ndarray | None:
    try:
        state = components.Statevector.from_instruction(circuit)
        return np.asarray(getattr(state, "data", state), dtype=complex).reshape(-1)
    except Exception:
        return None


def _final_statevector_from_result(components: Any, result: Any) -> np.ndarray | None:
    circuit = getattr(result, "optimal_circuit", None)
    if circuit is None:
        return None
    return _final_statevector_from_circuit(components, _bind_result_circuit(result, circuit))


def _statevector_re_im(psi: np.ndarray | None) -> list[list[float]] | None:
    """JSON-safe final-state encoding for post-hoc reporting metrics."""
    if psi is None:
        return None
    arr = np.asarray(psi, dtype=complex).reshape(-1)
    return [[float(np.real(x)), float(np.imag(x))] for x in arr]


def _compiled_circuit_stats(circuit: Any) -> dict[str, Any]:
    """Return optional Qiskit compiled-circuit stats without making Qiskit required."""
    empty = {
        "compiled_depth_total": None,
        "compiled_depth_2q_total": None,
        "compiled_depth_2q_semantics": None,
        "compiled_count_2q_total": None,
        "compiled_op_counts": None,
        "compiled_circuit_stats_status": "not_available",
        "compiled_circuit_stats_error": None,
        "compiled_basis_gates": list(_COMPILED_BASIS_GATES),
    }
    try:
        from qiskit import transpile
    except Exception as exc:  # pragma: no cover - optional-dep failure varies
        out = dict(empty)
        out.update(
            {
                "compiled_circuit_stats_status": "qiskit_transpile_unavailable",
                "compiled_circuit_stats_error": str(exc),
            }
        )
        return out
    try:
        try:
            decomposed = circuit.decompose(reps=10)
        except Exception:
            decomposed = circuit
        compiled = transpile(
            decomposed,
            basis_gates=list(_COMPILED_BASIS_GATES),
            optimization_level=0,
        )
        stats = _circuit_stats(compiled)
        return {
            "compiled_depth_total": stats["depth"],
            "compiled_depth_2q_total": stats["depth_2q"],
            "compiled_depth_2q_semantics": "qiskit_compiled_two_qubit_layer_depth_final_adapt_circuit",
            "compiled_count_2q_total": stats["count_2q"],
            "compiled_op_counts": stats["op_counts"],
            "compiled_circuit_stats_status": "ok",
            "compiled_circuit_stats_error": None,
            "compiled_basis_gates": list(_COMPILED_BASIS_GATES),
        }
    except Exception as exc:
        out = dict(empty)
        out.update(
            {
                "compiled_circuit_stats_status": "failed",
                "compiled_circuit_stats_error": str(exc),
            }
        )
        return out


def _shot_proxy_fields(
    *,
    hamiltonian_pauli_term_count: int,
    pool_term_count: int,
    energy_eval_count: int | None,
    adapt_num_iterations: int,
    shots_per_pauli_term_proxy: int,
) -> dict[str, Any]:
    h_count = max(0, int(hamiltonian_pauli_term_count))
    p_count = max(0, int(pool_term_count))
    energy_count = max(1, int(energy_eval_count or 0))
    gradient_scan_count = max(1, int(adapt_num_iterations or 0))
    gradient_probe_count = int(gradient_scan_count * p_count)
    shots_per_term = max(0, int(shots_per_pauli_term_proxy))
    shots_total = int(shots_per_term * h_count * (energy_count + gradient_probe_count))
    return {
        "shots_total": shots_total,
        "static_shot_estimate_status": "deterministic_proxy_not_physical_shots",
        "shot_proxy_formula": _SHOT_PROXY_FORMULA,
        "shot_proxy_note": (
            "Benchmark-table deterministic proxy only; it is not a hardware shot allocation. "
            "Initial-gradient-converged rows count one full gradient scan over the pool."
        ),
        "shots_per_pauli_term_proxy": shots_per_term,
        "hamiltonian_pauli_term_count": h_count,
        "pool_term_count": p_count,
        "energy_eval_count_proxy": energy_count,
        "gradient_scan_count_proxy": gradient_scan_count,
        "gradient_operator_probe_count_proxy": gradient_probe_count,
    }


def _sector_or_unavailable(context: ResolvedProblemContext, psi: np.ndarray | None) -> dict[str, Any]:
    if psi is None:
        return {
            "sector_probability": None,
            "sector_leak_probability": None,
            "sector_leak_flag": None,
            "sector_leak_threshold": None,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": None,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
            "policy": "not_available_final_state_not_reconstructed",
        }
    try:
        return sector_probability(context, psi)
    except Exception as exc:
        return {
            "sector_probability": None,
            "sector_leak_probability": None,
            "sector_leak_flag": None,
            "sector_leak_threshold": None,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": None,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
            "policy": "failed_sector_diagnostic",
            "sector_diagnostic_error": str(exc),
        }


def _resource_guard_for_context(
    context: ResolvedProblemContext,
    pool_labels: Sequence[str],
    *,
    qubit_cap: int | None,
    pool_term_cap: int | None,
) -> dict[str, Any] | None:
    num_qubits = int(context.layout.total_qubits)
    pool_count = int(len(tuple(pool_labels)))
    if qubit_cap is not None and num_qubits > int(qubit_cap):
        return {
            "resource_guard": True,
            "resource_guard_kind": "qiskit_adaptvqe_qubit_cap",
            "reason": "Qiskit AdaptVQE canonical case qubit count exceeds cap",
            "num_qubits": num_qubits,
            "qubit_cap": int(qubit_cap),
            "pool_term_count": pool_count,
            "pool_term_cap": None if pool_term_cap is None else int(pool_term_cap),
        }
    if pool_count <= 0:
        return {
            "resource_guard": True,
            "resource_guard_kind": "qiskit_adaptvqe_empty_pool",
            "reason": "full_meta pool is empty after identity/zero filtering",
            "num_qubits": num_qubits,
            "qubit_cap": None if qubit_cap is None else int(qubit_cap),
            "pool_term_count": pool_count,
            "pool_term_cap": None if pool_term_cap is None else int(pool_term_cap),
        }
    if pool_term_cap is not None and pool_count > int(pool_term_cap):
        return {
            "resource_guard": True,
            "resource_guard_kind": "qiskit_adaptvqe_full_meta_pool_term_cap",
            "reason": "Qiskit AdaptVQE full_meta pool exceeds cap",
            "num_qubits": num_qubits,
            "qubit_cap": None if qubit_cap is None else int(qubit_cap),
            "pool_term_count": pool_count,
            "pool_term_cap": int(pool_term_cap),
        }
    return None


def _is_initial_gradient_convergence_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "All gradients have been evaluated to lie below the convergence threshold" in msg
        and "first iteration" in msg
    )


def _reference_state_energy(components: Any, psi_ref: np.ndarray, hamiltonian_op: Any) -> float:
    state = components.Statevector(np.asarray(psi_ref, dtype=complex).reshape(-1))
    value = state.expectation_value(hamiltonian_op)
    return float(np.real(complex(value)))


def _run_static_qiskit_adapt_vqe_single_impl(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    max_adapt_iterations: int = _DEFAULT_MAX_ADAPT_ITERATIONS,
    optimizer_maxiter: int = _DEFAULT_OPTIMIZER_MAXITER,
    gradient_threshold: float = 1e-5,
    eigenvalue_threshold: float = 1e-6,
    seed: int = 42,
    shots_per_pauli_term_proxy: int = _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
) -> dict[str, Any]:
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    output = Path(output_dir)
    started_utc = _utc_now()
    t0 = time.perf_counter()

    # Validate case support first, but do not resolve exact/problem state before
    # optional dependency checks.  Missing Qiskit must remain a normalized skip.
    spec = _spec_by_case_id(family_key, case_key)

    if not has_qiskit_adaptvqe_support():
        return _skip_payload(
            family=family_key,
            case_id=case_key,
            output_dir=output,
            reason="optional Qiskit/qiskit_algorithms AdaptVQE dependencies are not importable",
            started_utc=started_utc,
        )

    components = import_qiskit_adaptvqe_components()
    context = _resolve_context_from_spec(spec)
    qubit_cap = _resource_cap_from_env(_RESOURCE_QUBIT_CAP_ENV, _QUBIT_CAP)
    pool_term_cap = _resource_cap_from_env(_RESOURCE_POOL_TERM_CAP_ENV, _POOL_TERM_CAP)
    try:
        pool_candidates = build_full_meta_candidate_pool(context, max_terms=pool_term_cap)
    except ValueError as exc:
        if "full_meta pool exceeds cap" not in str(exc):
            raise
        pool_candidates = tuple()
        pool_labels = tuple()
        guard = {
            "resource_guard": True,
            "resource_guard_kind": "qiskit_adaptvqe_full_meta_pool_term_cap",
            "reason": "Qiskit AdaptVQE full_meta pool exceeds cap",
            "num_qubits": int(context.layout.total_qubits),
            "qubit_cap": None if qubit_cap is None else int(qubit_cap),
            "pool_term_count": None if pool_term_cap is None else int(pool_term_cap + 1),
            "pool_term_cap": None if pool_term_cap is None else int(pool_term_cap),
        }
    else:
        pool_labels = tuple(str(candidate.label) for candidate in pool_candidates)
        guard = _resource_guard_for_context(
            context,
            pool_labels,
            qubit_cap=qubit_cap,
            pool_term_cap=pool_term_cap,
        )
    if guard is not None:
        return _resource_guard_payload(
            family=family_key,
            case_id=case_key,
            output_dir=output,
            spec=spec,
            started_utc=started_utc,
            reason=str(guard["reason"]),
            guard=guard,
        )

    psi_ref = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
    hamiltonian_op = pauli_poly_to_sparse_pauli_op(
        context.hamiltonian,
        sparse_pauli_op_cls=components.SparsePauliOp,
    )
    pool_ops = tuple(
        pauli_poly_to_sparse_pauli_op(
            candidate.polynomial,
            sparse_pauli_op_cls=components.SparsePauliOp,
        )
        for candidate in pool_candidates
    )
    pool_labels = tuple(str(candidate.label) for candidate in pool_candidates)
    pool_operator_pauli_labels_exyz = {
        str(candidate.label): list(candidate.pauli_labels_exyz)
        for candidate in pool_candidates
    }
    hamiltonian_pauli_term_count = int(len(hamiltonian_term_pool_labels(context.hamiltonian, max_terms=None)))
    reference_circuit = build_reference_state_circuit(
        psi_ref,
        num_qubits=int(context.layout.total_qubits),
        quantum_circuit_cls=components.QuantumCircuit,
    )

    try:
        np.random.seed(int(seed))
    except Exception:
        pass

    estimator = components.StatevectorEstimator()
    optimizer = components.COBYLA(maxiter=int(optimizer_maxiter))
    solver = components.VQE(estimator, reference_circuit, optimizer)
    adapt = components.AdaptVQE(
        solver=solver,
        operators=list(pool_ops),
        initial_state=reference_circuit,
        max_iterations=int(max_adapt_iterations),
        gradient_threshold=float(gradient_threshold),
        eigenvalue_threshold=float(eigenvalue_threshold),
        reps=1,
        flatten=True,
    )

    # Exact references are intentionally not resolved until after AdaptVQE
    # returns or after its first-gradient convergence check fails closed.
    initial_gradient_converged = False
    qiskit_algorithm_error: str | None = None
    try:
        result = adapt.compute_minimum_eigenvalue(hamiltonian_op)
        bound_circuit = _bind_result_circuit(result, reference_circuit)
        psi_final = _final_statevector_from_circuit(components, bound_circuit)
        stats = _circuit_stats(bound_circuit)
        compiled_stats = _compiled_circuit_stats(bound_circuit)
        optimal_point = np.asarray(getattr(result, "optimal_point", np.zeros(0)), dtype=float).reshape(-1)
        selected_ops = tuple(getattr(adapt, "_excitation_list", ()) or ())
        selected_operator_count = int(len(selected_ops))
        termination = str(getattr(result, "termination_criterion", None))
        adapt_num_iterations = int(getattr(result, "num_iterations", len(selected_ops)) or 0)
        selected_operator_labels_status = "qiskit_private_excitation_list"
        if selected_operator_count <= 0 and int(optimal_point.size) > 0:
            selected_operator_count = int(optimal_point.size)
            selected_operator_labels_status = "count_from_optimal_point_labels_unavailable"
        final_max_gradient = (
            None
            if getattr(result, "final_max_gradient", None) is None
            else float(np.real(getattr(result, "final_max_gradient")))
        )
        eigenvalue_history = [float(np.real(complex(v))) for v in getattr(result, "eigenvalue_history", [])]
        nfev = None if getattr(result, "cost_function_evals", None) is None else int(getattr(result, "cost_function_evals"))
        nit = (
            None
            if getattr(getattr(result, "optimizer_result", None), "nit", None) is None
            else int(getattr(getattr(result, "optimizer_result", None), "nit"))
        )
        energy = float(np.real(complex(getattr(result, "eigenvalue", np.nan))))
    except Exception as exc:
        if not _is_initial_gradient_convergence_error(exc):
            raise
        # Qiskit Algorithms raises AlgorithmError when the first gradient scan is
        # already below threshold.  For a benchmark row this is a legitimate
        # append-only ADAPT terminal state (zero selected operators), so normalize
        # it as a completed reference-state result rather than an infrastructure
        # failure.  Exact data is still resolved only after this AdaptVQE call.
        initial_gradient_converged = True
        qiskit_algorithm_error = str(exc)
        psi_final = psi_ref
        bound_circuit = reference_circuit
        stats = _circuit_stats(reference_circuit)
        compiled_stats = _compiled_circuit_stats(reference_circuit)
        optimal_point = np.zeros(0, dtype=float)
        selected_ops = ()
        selected_operator_count = 0
        selected_operator_labels_status = "initial_gradient_converged_no_selected_operators"
        termination = "initial_gradients_below_threshold"
        adapt_num_iterations = 0
        final_max_gradient = 0.0
        eigenvalue_history = []
        nfev = 0
        nit = 0
        energy = _reference_state_energy(components, psi_ref, hamiltonian_op)

    exact_energy = _safe_exact_energy(context)
    abs_delta = None if exact_energy is None else abs(float(energy) - float(exact_energy))
    sector = _sector_or_unavailable(context, psi_final)
    shot_proxy = _shot_proxy_fields(
        hamiltonian_pauli_term_count=hamiltonian_pauli_term_count,
        pool_term_count=len(pool_labels),
        energy_eval_count=nfev,
        adapt_num_iterations=adapt_num_iterations,
        shots_per_pauli_term_proxy=int(shots_per_pauli_term_proxy),
    )
    walltime = float(time.perf_counter() - t0)
    finished_utc = _utc_now()

    row = _base_row(family=family_key, case_id=case_key, status="ok", started_utc=started_utc, finished_utc=finished_utc)
    row.update(
        {
            "L": int(context.request.num_sites),
            "qiskit_available": True,
            "energy": energy,
            "exact_energy": exact_energy,
            "exact_gs_energy": exact_energy,
            "delta_E_abs": abs_delta,
            "abs_delta_e": abs_delta,
            "infidelity_exact": None,
            "infidelity_status": "available_post_hoc_from_final_statevector",
            "final_statevector_re_im": _statevector_re_im(psi_final),
            "final_statevector_status": "ok" if psi_final is not None else "not_reconstructed",
            "observable_error_status": "not_implemented_static_train_suite",
            "num_qubits": int(context.layout.total_qubits),
            "num_parameters": int(optimal_point.size),
            "selected_operator_count": int(selected_operator_count),
            "pool_term_count": int(len(pool_labels)),
            "hamiltonian_pauli_term_count": hamiltonian_pauli_term_count,
            "pool_name": "full_meta",
            "pool_source": "problem_local_full_meta_pool",
            "taxonomy_role": "same_pool_controller_comparator",
            "pool_labels": list(pool_labels),
            "pool_operator_pauli_labels_exyz": pool_operator_pauli_labels_exyz,
            "selected_operators": [str(op) for op in selected_ops],
            "selected_operator_labels_status": selected_operator_labels_status,
            "adapt_depth_reached": int(len(selected_ops)),
            "adapt_num_iterations": int(adapt_num_iterations),
            "adapt_max_iterations": int(max_adapt_iterations),
            "adapt_stop_reason": str(termination),
            "adapt_final_max_gradient": final_max_gradient,
            "adapt_eigenvalue_history": eigenvalue_history,
            "initial_gradient_converged": bool(initial_gradient_converged),
            "qiskit_algorithm_error": qiskit_algorithm_error,
            "optimizer": "COBYLA",
            "optimizer_maxiter": int(optimizer_maxiter),
            "gradient_threshold": float(gradient_threshold),
            "eigenvalue_threshold": float(eigenvalue_threshold),
            "seed": int(seed),
            "nfev": nfev,
            "nit": nit,
            "runtime_s": walltime,
            "count_2q": stats["count_2q"],
            "depth_proxy": stats["depth"],
            "circuit_depth": stats["depth"],
            "qiskit_op_counts": stats["op_counts"],
            "circuit_stat_basis": "qiskit_high_level_undecomposed_proxy",
            "count_2q_semantics": "counts explicit 2q instructions before Qiskit basis-gate decomposition",
            **compiled_stats,
            **shot_proxy,
            "sector_probability": sector.get("sector_probability"),
            "sector_leak_probability": sector.get("sector_leak_probability"),
            "sector_leak_flag": sector.get("sector_leak_flag"),
            "sector_leak_threshold": sector.get("sector_leak_threshold"),
            "boson_legal_probability_min": sector.get("boson_legal_probability_min"),
            "boson_illegal_probability_max": sector.get("boson_illegal_probability_max"),
            "boson_truncation_leak_flag": sector.get("boson_truncation_leak_flag"),
            "boson_subspace_diagnostics": sector.get("boson_subspace_diagnostics"),
            "truncation_diagnostics": sector.get("truncation_constraints_evaluated"),
            "sector_diagnostics": sector,
            "theta": optimal_point.tolist(),
        }
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family_key,
        "case_id": case_key,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "completed",
        "qiskit_available": True,
        "runner": "pipelines.exact_bench.generic_static_qiskit_adapt_vqe.run_static_qiskit_adapt_vqe_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(exact_reference_usage="reporting_only_after_optimization"),
        "spec": _spec_metadata(spec),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }
    return _write_artifacts(output, payload, [row])


def run_static_qiskit_adapt_vqe_single(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    max_adapt_iterations: int = _DEFAULT_MAX_ADAPT_ITERATIONS,
    optimizer_maxiter: int = _DEFAULT_OPTIMIZER_MAXITER,
    gradient_threshold: float = 1e-5,
    eigenvalue_threshold: float = 1e-6,
    seed: int = 42,
    shots_per_pauli_term_proxy: int = _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
) -> dict[str, Any]:
    """Run one generic Qiskit AdaptVQE benchmark case and always emit artifacts."""
    started_utc = _utc_now()
    try:
        return _run_static_qiskit_adapt_vqe_single_impl(
            family=family,
            case_id=case_id,
            output_dir=output_dir,
            max_adapt_iterations=max_adapt_iterations,
            optimizer_maxiter=optimizer_maxiter,
            gradient_threshold=gradient_threshold,
            eigenvalue_threshold=eigenvalue_threshold,
            seed=seed,
            shots_per_pauli_term_proxy=shots_per_pauli_term_proxy,
        )
    except QiskitAdaptVQEUnavailable as exc:
        return _skip_payload(
            family=str(family).strip(),
            case_id=str(case_id).strip(),
            output_dir=Path(output_dir),
            reason=str(exc),
            started_utc=started_utc,
        )
    except Exception as exc:
        return _failure_payload(
            family=str(family).strip(),
            case_id=str(case_id).strip(),
            output_dir=Path(output_dir),
            reason=str(exc),
            exception_type=type(exc).__name__,
            qiskit_available=has_qiskit_adaptvqe_support(),
            started_utc=started_utc,
        )


__all__ = [
    "GENERIC_STATIC_QISKIT_ADAPTVQE_FAMILIES",
    "SCHEMA_VERSION",
    "default_static_qiskit_adapt_vqe_case_ids",
    "run_static_qiskit_adapt_vqe_single",
]
