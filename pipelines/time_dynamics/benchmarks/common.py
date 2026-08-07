#!/usr/bin/env python3
"""Shared row I/O and statevector helpers for isolated dynamics benchmarks.

This module is benchmark-only plumbing.  It owns generic row serialization,
metric/resource extraction, fixture loading, and small statevector helpers used
by the algorithm-specific modules in :mod:`pipelines.time_dynamics.benchmarks`.
It must not import compatibility shims or HH legacy wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DYNAMICS_TABLE_BUNDLE_SCHEMA,
    DYNAMICS_CASE_METADATA_OVERRIDE_SOURCE,
    DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
    DYNAMICS_LEGACY_MISSING_TUNING_SOURCE,
    DYNAMICS_SKIPPED_TUNING_SOURCE,
    DYNAMICS_TUNING_CLASS_SOURCE,
    DynamicsBenchmarkCase,
    DynamicsBenchmarkRow,
    DynamicsTableFields,
    build_dynamics_tuning_provenance,
    dynamics_tuning_class,
    dynamics_table_bundle_payload,
    json_safe,
    validate_dynamics_metric_contract,
)
from pipelines.time_dynamics.tables.table_lock_contract import (
    build_locked_or_default_tuning_provenance,
    case_with_class_settings_overrides,
    class_settings_manifest_path,
    table_lock_provenance_for_case,
)
from pipelines.time_dynamics.benchmarks.qiskit_native import (
    QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS,
    QISKIT_COMMUNITY_METHOD_LABELS,
    QISKIT_COMMUNITY_RUNNER_MODULE_BY_ALGORITHM,
    QISKIT_COMMUNITY_TABLE_LABELS,
)
from pipelines.time_dynamics.adapters.observables import (
    observable_snapshot_for_state,
    primary_density_value_from_snapshot,
)
from pipelines.time_dynamics.adapters.drive_terms import (
    count_nonidentity_pauli_terms,
    resolve_realtime_drive_model,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    TimeDependentHamiltonian,
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.drive_aligned import (
    augment_state_with_drive_aligned_generator,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    APMcLachlanState,
    state_from_scaffold_runtime_input,
)
from src.quantum.ansatz_parameterization import build_parameter_layout, iter_runtime_rotation_terms
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_actions import apply_exp_term, compile_pauli_action_exyz
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

PAPER_II_FIXED_MCLACHLAN_INTEGRATOR_POLICY = "rk4"
PAPER_II_FIXED_MCLACHLAN_INTEGRATOR_SOURCE = "paper_ii_default_fixed_rk4"

REALTIME_GENERIC_DYNAMICS_ALGORITHMS: tuple[str, ...] = (
    "dyn_exact_reference",
    "dyn_fixed_mclachlan",
)
NATIVE_GENERIC_COMPARATOR_ALGORITHMS: tuple[str, ...] = (
    "dyn_product_formula_envelope",
    "dyn_qdrift",
    "dyn_fixed_pvqd",
    "dyn_adaptive_pvqd",
    "dyn_avqds",
    "dyn_avqds_t",
    "dyn_avqds_tetris",
)
SUPPORTED_GENERIC_DYNAMICS_ALGORITHMS: tuple[str, ...] = (
    *REALTIME_GENERIC_DYNAMICS_ALGORITHMS,
    *NATIVE_GENERIC_COMPARATOR_ALGORITHMS,
    *QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS,
)
CANDIDATE_POOL_REQUIRED_GENERIC_ALGORITHMS: tuple[str, ...] = (
    "dyn_adaptive_pvqd",
    "dyn_avqds",
    "dyn_avqds_t",
)

_METHOD_LABELS = {
    "dyn_exact_reference": "Exact/Krylov reference dynamics",
    "dyn_fixed_mclachlan": "Fixed-scaffold McLachlan dynamics",
    "dyn_product_formula_envelope": "Product-formula/Suzuki envelope",
    "dyn_qdrift": "qDRIFT/randomized product formula",
    "dyn_fixed_pvqd": "Fixed pVQD projection dynamics",
    "dyn_adaptive_pvqd": "Adaptive pVQD projection dynamics",
    "dyn_avqds": "AVQDS RHS-tangent dynamics",
    "dyn_avqds_t": "Product-formula-target adaptive tangent diagnostic",
    "dyn_avqds_tetris": "AVQDS(T) TETRIS dynamics",
    **QISKIT_COMMUNITY_METHOD_LABELS,
    "dyn_controller_full": "Full strict checkpoint controller",
    "dyn_controller_fixed_scaffold": "Fixed scaffold control",
    "dyn_controller_no_append": "No append ablation",
    "dyn_controller_no_pruning": "No pruning ablation",
    "dyn_controller_fixed_integrator": "Appendix diagnostic: fixed Euler integrator",
    "dyn_controller_no_residual_split": "No residual-split confirmation ablation",
    "dyn_hh_recovery_s1_rk4_no_append_no_prune": "HH recovery S1: RK4 fixed scaffold",
    "dyn_hh_recovery_s2_rk4_append_only": "HH recovery S2: RK4 append only",
    "dyn_hh_recovery_s3_rk4_append_prune": "HH recovery S3: RK4 append+prune",
    "dyn_hh_recovery_s4_auto_append_prune": "HH recovery S4: auto append+prune",
    "dyn_hh_recovery_s5_auto_no_append_no_prune": "HH recovery S5: auto fixed scaffold",
    "dyn_controller_ablation_matrix": "Generic controller ablation matrix",
}

_BENCHMARK_RUNNER_MODULE_BY_ALGORITHM = {
    "dyn_exact_reference": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_fixed_mclachlan": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_product_formula_envelope": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_qdrift": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_fixed_pvqd": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_adaptive_pvqd": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_avqds": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_avqds_t": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_avqds_tetris": "pipelines.time_dynamics.benchmarks.avqds_tetris",
    **QISKIT_COMMUNITY_RUNNER_MODULE_BY_ALGORITHM,
    "dyn_controller_ablation_matrix": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_controller_full": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_controller_fixed_scaffold": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_controller_no_append": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_controller_no_pruning": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_controller_fixed_integrator": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_controller_no_residual_split": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_hh_recovery_s1_rk4_no_append_no_prune": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_hh_recovery_s2_rk4_append_only": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_hh_recovery_s3_rk4_append_prune": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_hh_recovery_s4_auto_append_prune": "pipelines.time_dynamics.benchmarks.legacy_native",
    "dyn_hh_recovery_s5_auto_no_append_no_prune": "pipelines.time_dynamics.benchmarks.legacy_native",
}

NATIVE_RESOURCE_POLICY = "repo_native_pauli_rotation_proxy_no_qiskit"
QISKIT_COMMUNITY_RESOURCE_POLICY = "qiskit_community_compiled_circuit_accumulated_v1"

CORRECTNESS_SIDECAR_FILENAMES: dict[str, str] = {
    "dyn_fixed_mclachlan": "mclachlan_correctness.json",
    "dyn_avqds": "avqds_correctness.json",
    "dyn_avqds_t": "avqds_t_correctness.json",
    "dyn_avqds_tetris": "avqds_tetris_correctness.json",
}
CORRECTNESS_SIDECAR_KEYS: dict[str, str] = {
    "dyn_fixed_mclachlan": "mclachlan_correctness",
    "dyn_avqds": "avqds_correctness",
    "dyn_avqds_t": "avqds_t_correctness",
    "dyn_avqds_tetris": "avqds_tetris_correctness",
}
QISKIT_PARITY_SIDECAR_REQUIRED_ALGORITHMS: tuple[str, ...] = (
    "dyn_fixed_mclachlan",
    "dyn_product_formula_envelope",
    "dyn_qdrift",
    "dyn_fixed_pvqd",
    "dyn_adaptive_pvqd",
)
MCLACHLAN_CORRECTNESS_SCHEMA = "fixed_mclachlan_correctness_v1"


class _LazyRealtimeModule:
    """Compatibility proxy for historical ``common.realtime`` monkeypatches."""

    def _module(self) -> Any:
        from pipelines.time_dynamics.runners import generic_from_adapt_artifact

        return generic_from_adapt_artifact

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module(), name)


realtime: Any = _LazyRealtimeModule()


def metadata_override_settings_source(case: DynamicsBenchmarkCase, keys: Sequence[str]) -> str:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    if any(str(key) in metadata for key in keys):
        return DYNAMICS_CASE_METADATA_OVERRIDE_SOURCE
    return DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE


def _case_smoke_fast_mode(case: DynamicsBenchmarkCase) -> bool:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    smoke_only = _boolish(metadata.get("smoke_only_not_paper_evidence", False))
    requested = _boolish(
        metadata.get("local_smoke_fast_mode", metadata.get("smoke_fast_mode", False))
    )
    return bool(smoke_only and requested)


def _product_formula_sequence(terms: Sequence[Any], *, order: int) -> tuple[tuple[Any, float], ...]:
    """Return a QPU-preparable first/second-order product-formula sequence.

    This helper is intentionally exact-reference-free.  It describes the
    circuit target used by product-formula and measurement-compatible pVQD /
    target-tangent benchmarks.  Exact ED trajectories may still be computed by
    callers, but only for diagnostic output.
    """

    if int(order) == 1:
        return tuple((term, 1.0) for term in terms)
    if int(order) == 2:
        return tuple((term, 0.5) for term in terms) + tuple((term, 0.5) for term in reversed(tuple(terms)))
    raise ValueError(f"unsupported Suzuki/product-formula order {order!r}; expected 1 or 2")


def _compiled_pauli_actions_by_label(terms: Sequence[Any]) -> dict[str, Any]:
    return {
        str(term.pauli_exyz): compile_pauli_action_exyz(str(term.pauli_exyz), int(term.nq))
        for term in terms
    }


def _apply_product_formula_step(
    *,
    terms: Sequence[Any],
    action_by_label: Mapping[str, Any],
    psi: np.ndarray,
    dt: float,
    order: int,
) -> np.ndarray:
    """Apply a product-formula circuit step to the prepared state.

    The resulting target state is QPU-faithful in the repo sense: it is the
    ideal/infinite-shot simulation of a state that a circuit could prepare, not
    an ED evolved target trajectory.
    """

    state = _normalize_state(psi)
    for term, factor in _product_formula_sequence(terms, order=int(order)):
        label = str(term.pauli_exyz)
        state = apply_exp_term(
            state,
            action_by_label[label],
            complex(float(term.coeff_real)),
            float(dt) * float(factor),
        )
    return _normalize_state(state)




def _runner_module_for_algorithm(algorithm_id: str) -> str | None:
    return _BENCHMARK_RUNNER_MODULE_BY_ALGORITHM.get(str(algorithm_id))


def _qiskit_dynamics_adapter() -> Any:
    """Import the exact-bench Qiskit parity adapter lazily."""

    return importlib.import_module("pipelines.exact_bench.qiskit_dynamics_adapter")


def _qiskit_community_dynamics_adapter() -> Any:
    """Import the exact-bench Qiskit-community primary adapter lazily."""

    return importlib.import_module("pipelines.exact_bench.qiskit_community_dynamics_adapter")


def _qiskit_dynamics_config_for_case(case: DynamicsBenchmarkCase) -> Any:
    return _qiskit_dynamics_adapter().qiskit_dynamics_config_from_case(case)


def _qiskit_parity_requested_for_case(case: DynamicsBenchmarkCase) -> bool:
    adapter = _qiskit_dynamics_adapter()
    config = adapter.qiskit_dynamics_config_from_case(case)
    return bool(adapter.parity_requested(config))


def _fixed_mclachlan_integrator_policy_override(case: DynamicsBenchmarkCase) -> str | None:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    for key in (
        "fixed_mclachlan_integrator_policy",
        "benchmark_fixed_mclachlan_integrator_policy",
        "integrator_policy",
    ):
        raw = metadata.get(key)
        if raw in {None, ""}:
            continue
        policy = str(raw).strip().lower()
        if policy not in {"euler", "rk4", "auto_euler_rk4"}:
            raise ValueError(
                f"case {case.case_id}: {key} must be one of euler, rk4, or auto_euler_rk4; got {raw!r}"
            )
        return policy
    return None


def run_native_generic_comparator_row(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    output_dir: Path,
    payload_builder: Callable[..., Mapping[str, Any]],
) -> DynamicsBenchmarkRow:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        case = case_with_class_settings_overrides(
            case,
            algorithm_id=str(algorithm_id),
            settings_kind="comparator",
        )
    except ValueError as exc:
        return write_skipped_generic_dynamics_row(
            case=case,
            algorithm_id=algorithm_id,
            output_dir=root,
            status="failed",
            reason=str(exc),
        )
    raw_payload_json = root / "raw_payload.json"
    command = _native_comparator_command(case=case, algorithm_id=algorithm_id, output_dir=root)
    _write_json(root / "command.json", command)
    runtime_input = _load_runtime_input_for_case(case)
    try:
        _assert_native_case_supported(case, runtime_input)
    except ValueError as exc:
        return write_skipped_generic_dynamics_row(
            case=case,
            algorithm_id=algorithm_id,
            output_dir=root,
            status="skipped_unsupported",
            reason=str(exc),
        )
    if (
        algorithm_id in CANDIDATE_POOL_REQUIRED_GENERIC_ALGORITHMS
        and not _candidate_pool_is_complete(runtime_input)
    ):
        comparator_label = {
            "dyn_adaptive_pvqd": "generic adaptive pVQD comparator",
            "dyn_avqds": "generic AVQDS comparator",
            "dyn_avqds_t": "product-formula-target adaptive tangent diagnostic",
        }.get(str(algorithm_id), "generic dynamics comparator")
        return write_skipped_generic_dynamics_row(
            case=case,
            algorithm_id=algorithm_id,
            output_dir=root,
            status="skipped_unsupported",
            reason=(
                f"{comparator_label} requires complete candidate pool; "
                f"got {_candidate_pool_completeness(runtime_input)!r}"
            ),
        )
    payload = dict(payload_builder(case=case, runtime_input=runtime_input, command=command))
    if class_settings_manifest_path(case) is not None:
        tuning = build_locked_or_default_tuning_provenance(
            case=case,
            algorithm_id=str(algorithm_id),
            settings_kind="comparator",
            settings_source=DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
            locked=False,
        )
        payload["tuning_provenance"] = tuning
        parameter_manifest = (
            dict(payload.get("parameter_manifest", {}))
            if isinstance(payload.get("parameter_manifest"), Mapping)
            else {}
        )
        parameter_manifest["tuning_provenance"] = dict(tuning)
        payload["parameter_manifest"] = parameter_manifest
    _write_json(raw_payload_json, payload)
    row = _row_from_payload(
        case=case,
        algorithm_id=algorithm_id,
        payload=payload,
        artifact_json=raw_payload_json,
        command=command,
    )
    write_dynamics_row_bundle(row=row, output_dir=root, raw_payload=payload)
    return row


def _qiskit_community_resources(
    *,
    algorithm_id: str,
    circuit_records: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records = [dict(record) for record in circuit_records if isinstance(record, Mapping)]
    final = records[-1] if records else {}
    compiled_count_2q_total = int(sum(_int_or_none(record.get("count_2q")) or 0 for record in records))
    compiled_depth_total = int(sum(_int_or_none(record.get("depth")) or 0 for record in records))
    compiled_size_total = int(sum(_int_or_none(record.get("size")) or 0 for record in records))
    resources = {
        "resource_policy": QISKIT_COMMUNITY_RESOURCE_POLICY,
        "state_at_time_scope": "qiskit_community_bound_circuit",
        "state_at_time_resource_basis": "qiskit_circuit_stats_after_decomposition",
        "full_horizon_scope": "qiskit_community_trajectory_circuit_records",
        "full_horizon_resource_basis": "sum_of_qiskit_circuit_stats_over_reported_time_points",
        "state_at_time_2q": _int_or_none(final.get("count_2q")),
        "state_at_time_depth_2q": _int_or_none(final.get("count_2q")),
        "state_at_time_depth": _int_or_none(final.get("depth")),
        "state_at_time_size": _int_or_none(final.get("size")),
        "compiled_count_2q_total": int(compiled_count_2q_total),
        "compiled_depth_2q_total": int(compiled_count_2q_total),
        "compiled_depth_total": int(compiled_depth_total),
        "compiled_size_total": int(compiled_size_total),
        "qiskit_circuit_record_count": int(len(records)),
        "compiled_backend_name": "qiskit_statevector_default",
        "algorithm_id": str(algorithm_id),
    }
    if extra:
        resources.update(dict(extra))
    return resources


def _qiskit_compile_audit_from_resources(resources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selected_backend": {
            "backend_name": resources.get("compiled_backend_name", "qiskit_statevector_default"),
            "compiled_count_2q": _int_or_none(resources.get("compiled_count_2q_total")),
            "compiled_depth_2q": _int_or_none(resources.get("compiled_depth_2q_total")),
            "compiled_depth": _int_or_none(resources.get("compiled_depth_total")),
            "compiled_size": _int_or_none(resources.get("compiled_size_total")),
        }
    }


def _shared_benchmark_surface_payload(
    *,
    case: DynamicsBenchmarkCase,
    flow: NativeHamiltonianFlow,
    runtime_input: Any,
) -> dict[str, Any]:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    seed_lock = table_lock_provenance_for_case(case)
    seed_sha = metadata.get(
        "static_seed_artifact_sha256",
        metadata.get(
            "seed_artifact_sha256",
            metadata.get("source_artifact_sha256", seed_lock.get("static_seed_artifact_sha256")),
        ),
    )
    if seed_sha in {None, ""}:
        seed_sha = str(case.artifact_json)
    seed_artifact = seed_lock.get("static_seed_artifact_json", str(case.artifact_json))
    group_id = metadata.get(
        "same_seed_comparator_group_id",
        seed_lock.get("same_seed_comparator_group_id", f"{case.family}:{case.case_id}"),
    )
    drive_payload = {
        "drive_included": bool(flow.drive_enabled),
        "drive_time_sampling": str(flow.drive_time_sampling),
        "drive_t0": float(flow.drive_t0),
        "drive_model": dict(getattr(flow.drive_model, "profile_payload", {}) or {}),
    }
    surface = {
        "schema": "dynamics_benchmark_shared_surface_v1",
        "same_seed_comparator_group_id": str(group_id),
        "static_seed_artifact_sha256": str(seed_sha),
        "static_seed_artifact_json": str(seed_artifact),
        "drive_signature": json.dumps(json_safe(drive_payload), sort_keys=True),
        "time_grid_signature": json.dumps([float(x) for x in np.asarray(flow.times, dtype=float)], sort_keys=True),
        "observable_set_signature": json.dumps(
            {
                "resolved_problem_family": str(
                    getattr(getattr(runtime_input, "resolved_problem", None), "family_key", case.family)
                ),
                "observable_context_keys": sorted(str(key) for key in dict(flow.observable_context or {})),
            },
            sort_keys=True,
        ),
        "diagnostic_reference_signature": "benchmark_exact_reporting_only",
        "compile_target_signature": "qiskit_statevector_default",
    }
    surface["surface_signature"] = json.dumps(
        {
            key: surface[key]
            for key in (
                "same_seed_comparator_group_id",
                "static_seed_artifact_sha256",
                "drive_signature",
                "time_grid_signature",
                "observable_set_signature",
                "diagnostic_reference_signature",
                "compile_target_signature",
            )
        },
        sort_keys=True,
    )
    return surface


def run_qiskit_community_comparator_row(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    output_dir: Path,
) -> DynamicsBenchmarkRow:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if str(algorithm_id) not in QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS:
        return write_skipped_generic_dynamics_row(
            case=case,
            algorithm_id=algorithm_id,
            output_dir=root,
            status="skipped_no_runner",
            reason=f"{algorithm_id!r} is not a Qiskit-community comparator algorithm",
        )
    try:
        case = case_with_class_settings_overrides(
            case,
            algorithm_id=str(algorithm_id),
            settings_kind="comparator",
        )
    except ValueError as exc:
        return write_skipped_generic_dynamics_row(
            case=case,
            algorithm_id=algorithm_id,
            output_dir=root,
            status="failed",
            reason=str(exc),
        )
    raw_payload_json = root / "raw_payload.json"
    command = _native_comparator_command(case=case, algorithm_id=algorithm_id, output_dir=root)
    _write_json(root / "command.json", command)
    runtime_input = _load_runtime_input_for_case(case)
    try:
        flow = _native_hamiltonian_flow(case, runtime_input)
        _terms, layout, theta, psi_ref, _executor, drive_aligned_scaffold = _runtime_variational_bundle(
            runtime_input,
            hamiltonian=flow.hamiltonian,
            drive_aligned_ansatz=(
                bool(flow.drive_enabled)
                and str(algorithm_id) in {"dyn_qiskit_pvqd", "dyn_qiskit_varqrte"}
            ),
        )
        _assert_native_case_supported(case, runtime_input)
        adapter = _qiskit_community_dynamics_adapter()
        config = adapter.qiskit_community_config_from_case(case)
        result = adapter.run_qiskit_community_dynamics(
            config=config,
            case=case,
            algorithm_id=str(algorithm_id),
            terms_for_interval=flow.terms_for_interval,
            times=flow.times,
            layout=layout,
            theta_runtime=theta,
            psi_ref=psi_ref,
            progress_json=root / "qiskit_progress.json",
        )
    except Exception as exc:
        status = "skipped_unsupported"
        adapter_module = None
        try:
            adapter_module = _qiskit_community_dynamics_adapter()
        except Exception:
            adapter_module = None
        if adapter_module is not None and isinstance(
            exc,
            (
                getattr(adapter_module, "QiskitCommunityDynamicsUnavailable"),
                getattr(adapter_module, "QiskitCommunityDynamicsUnsupported"),
            ),
        ):
            status = "skipped_unsupported"
        else:
            status = "failed"
        return write_skipped_generic_dynamics_row(
            case=case,
            algorithm_id=algorithm_id,
            output_dir=root,
            status=status,
            reason=str(exc),
        )
    states = tuple(np.asarray(state, dtype=complex).reshape(-1) for state in result.states_by_time)
    trajectory = _trajectory_from_states(
        times=flow.times,
        states=states,
        exact_states=flow.exact_states,
        hmat=flow.static_hmat,
        hmat_sequence=flow.hmat_sequence_for_trajectory_samples(),
        method=str(algorithm_id),
        **dict(flow.observable_context or {}),
    )
    summary = _trajectory_summary(trajectory)
    public_payload = dict(result.public_payload)
    resources = _qiskit_community_resources(
        algorithm_id=str(algorithm_id),
        circuit_records=result.circuit_records,
        extra={
            "compiled_backend_name": public_payload.get("config", {}).get(
                "compile_backend_name", "qiskit_statevector_default"
            )
            if isinstance(public_payload.get("config"), Mapping)
            else "qiskit_statevector_default",
        },
    )
    tuning = build_locked_or_default_tuning_provenance(
        case=case,
        algorithm_id=str(algorithm_id),
        settings_kind="comparator",
        settings_payload={
            "qiskit_community_config": public_payload.get("config", {}),
            "qiskit_algorithm_name": public_payload.get("qiskit_algorithm_name"),
        },
        settings_source=DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
        locked=False,
    )
    terms = flow.terms_for_interval(float(flow.times[0]), float(flow.times[min(1, len(flow.times) - 1)]))
    parameter_manifest = _generic_parameter_manifest(
        case=case,
        runtime_input=runtime_input,
        algorithm_id=str(algorithm_id),
        times=flow.times,
        terms=terms,
        flow=flow,
    )
    parameter_manifest["tuning_provenance"] = dict(tuning)
    shared_surface = _shared_benchmark_surface_payload(
        case=case,
        flow=flow,
        runtime_input=runtime_input,
    )
    payload = json_safe(
        {
            "schema_version": "generic_qiskit_community_dynamics_benchmark_v1",
            "case": case.to_dict(),
            "status": "completed",
            "row_contract": {
                "qpu_faithful": True,
                "exact_assisted": False,
                "diagnostic": True,
            },
            "parameter_manifest": parameter_manifest,
            "tuning_provenance": tuning,
            "command": list(command),
            "trajectory": trajectory,
            "summary": summary,
            "qiskit_community": public_payload,
            "drive_aligned_ansatz": drive_aligned_scaffold.to_json_dict(),
            "metrics": {
                "method_kind": str(public_payload.get("qiskit_algorithm_name", algorithm_id)),
                "decision_data_flow": "qiskit_community_algorithm_without_exact_reference_inputs",
                "qiskit_primary_mode": True,
                "qiskit_parity_sidecar": False,
                "exact_fields_reporting_only": True,
                "exact_reference_controller_inputs": False,
                "controller_decisions_modified": False,
            },
            "resources": resources,
            "compile_audit": _qiskit_compile_audit_from_resources(resources),
            "benchmark_surface": shared_surface,
            "provenance": {
                "route_module": _runner_module_for_algorithm(algorithm_id),
                "benchmark_only": True,
                "runner_module": _runner_module_for_algorithm(algorithm_id),
                "execution_surface": "pinned_qiskit_community_time_evolver",
                "execution_surface_role": "primary_execution_surface",
                "qiskit_primary_mode": True,
                "qiskit_parity_sidecar": False,
                "qiskit_boundary": "pipelines.exact_bench.qiskit_community_dynamics_adapter",
                "repo_native_comparator": False,
                "resource_policy": QISKIT_COMMUNITY_RESOURCE_POLICY,
                "controller_paths_called": False,
                "controller_decisions_modified": False,
                "exact_reference_controller_inputs": False,
                "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_qiskit_algorithm_input",
                "benchmark_surface": shared_surface,
            },
        }
    )
    if class_settings_manifest_path(case) is not None:
        payload["tuning_provenance"] = tuning
        parameter_manifest = (
            dict(payload.get("parameter_manifest", {}))
            if isinstance(payload.get("parameter_manifest"), Mapping)
            else {}
        )
        parameter_manifest["tuning_provenance"] = dict(tuning)
        payload["parameter_manifest"] = parameter_manifest
    _write_json(raw_payload_json, payload)
    row = _row_from_payload(
        case=case,
        algorithm_id=algorithm_id,
        payload=payload,
        artifact_json=raw_payload_json,
        command=command,
    )
    write_dynamics_row_bundle(row=row, output_dir=root, raw_payload=payload)
    return row


def run_realtime_generic_dynamics_row(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    output_dir: Path,
) -> DynamicsBenchmarkRow:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if str(algorithm_id) == "dyn_fixed_mclachlan":
        try:
            case = case_with_class_settings_overrides(
                case,
                algorithm_id=str(algorithm_id),
                settings_kind="mclachlan",
            )
        except ValueError as exc:
            return write_skipped_generic_dynamics_row(
                case=case,
                algorithm_id=algorithm_id,
                output_dir=root,
                status="failed",
                reason=str(exc),
            )
    raw_payload_json = root / "raw_payload.json"
    argv = _realtime_argv(case=case, algorithm_id=algorithm_id, raw_payload_json=raw_payload_json)
    command = [sys.executable, "-m", "pipelines.time_dynamics.runners.generic_from_adapt_artifact", *argv]
    _write_json(root / "command.json", command)
    args = realtime.build_parser().parse_args(argv)
    payload = realtime.run_from_args(args)
    if not isinstance(payload, Mapping):
        raise ValueError("generic realtime route returned a non-mapping payload")
    payload = dict(payload)
    if str(algorithm_id) == "dyn_fixed_mclachlan":
        fixed_integrator_policy = _fixed_mclachlan_integrator_policy_override(case)
        settings_payload = {
            "checkpoint_controller_mode": "observable_v1",
            "checkpoint_controller_exact_input_mode": "off",
            "lock_fixed_manifold": True,
            "noise_mode": "ideal",
            "integrator_policy": fixed_integrator_policy or PAPER_II_FIXED_MCLACHLAN_INTEGRATOR_POLICY,
        }
        if fixed_integrator_policy is not None:
            settings_payload["integrator_policy_override_source"] = "case_metadata"
        else:
            settings_payload["integrator_policy_override_source"] = (
                PAPER_II_FIXED_MCLACHLAN_INTEGRATOR_SOURCE
            )
        tuning = build_locked_or_default_tuning_provenance(
            case=case,
            algorithm_id=algorithm_id,
            settings_kind="mclachlan",
            settings_payload=settings_payload,
            settings_source=DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
            locked=False,
        )
    elif str(algorithm_id) == "dyn_exact_reference":
        tuning = build_locked_or_default_tuning_provenance(
            case=case,
            algorithm_id=algorithm_id,
            settings_kind="reference",
            settings_payload={"reference_mode": "benchmark_exact"},
            settings_source=DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
            locked=False,
        )
    else:
        tuning = build_locked_or_default_tuning_provenance(
            case=case,
            algorithm_id=algorithm_id,
            settings_kind="benchmark",
            settings_source=DYNAMICS_LEGACY_MISSING_TUNING_SOURCE,
            locked=False,
        )
    payload["tuning_provenance"] = tuning
    parameter_manifest = (
        dict(payload.get("parameter_manifest", {}))
        if isinstance(payload.get("parameter_manifest"), Mapping)
        else {}
    )
    parameter_manifest["tuning_provenance"] = dict(tuning)
    payload["parameter_manifest"] = parameter_manifest
    provenance = dict(payload.get("provenance", {})) if isinstance(payload.get("provenance"), Mapping) else {}
    existing_controller_decisions_modified = _boolish(provenance.get("controller_decisions_modified", False))
    existing_exact_reference_controller_inputs = _boolish(
        provenance.get("exact_reference_controller_inputs", False)
    )
    exact_data_policy = (
        "diagnostic_exact_reference_reporting_only_not_controller_input"
        if str(algorithm_id) == "dyn_fixed_mclachlan"
        else "exact_reference_trajectory_diagnostic_algorithm"
    )
    provenance.update(
        {
            "route_module": _runner_module_for_algorithm(algorithm_id),
            "benchmark_only": True,
            "runner_module": _runner_module_for_algorithm(algorithm_id),
            "exact_data_policy": exact_data_policy,
            "controller_paths_called": False,
            "controller_decisions_modified": existing_controller_decisions_modified,
            "exact_reference_controller_inputs": existing_exact_reference_controller_inputs,
        }
    )
    payload["provenance"] = provenance
    if str(algorithm_id) == "dyn_fixed_mclachlan":
        payload["row_contract"] = {
            "qpu_faithful": True,
            "exact_assisted": False,
            "diagnostic": True,
        }
        payload["mclachlan_correctness"] = build_fixed_mclachlan_correctness_sidecar(
            case=case,
            payload=payload,
        )
        try:
            adapter = _qiskit_dynamics_adapter()
            config = adapter.qiskit_dynamics_config_from_case(case)
            payload["qiskit_parity"] = adapter.fixed_mclachlan_post_run_parity_result(
                config=config,
                case=case,
                payload=payload,
            )
        except ValueError:
            raise
        except Exception:
            # Optional post-run parity must not perturb the repo-native realtime route.
            pass
    elif str(algorithm_id) == "dyn_exact_reference":
        payload["row_contract"] = {
            "qpu_faithful": False,
            "exact_assisted": True,
            "diagnostic": True,
        }
    _write_json(raw_payload_json, payload)
    row = _row_from_payload(
        case=case,
        algorithm_id=algorithm_id,
        payload=payload,
        artifact_json=raw_payload_json,
        command=command,
    )
    write_dynamics_row_bundle(row=row, output_dir=root, raw_payload=payload)
    return row


__all__ = [
    "DYNAMICS_TABLE_BUNDLE_SCHEMA",
    "CANDIDATE_POOL_REQUIRED_GENERIC_ALGORITHMS",
    "CORRECTNESS_SIDECAR_FILENAMES",
    "CORRECTNESS_SIDECAR_KEYS",
    "NATIVE_GENERIC_COMPARATOR_ALGORITHMS",
    "NATIVE_RESOURCE_POLICY",
    "QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS",
    "QISKIT_COMMUNITY_RESOURCE_POLICY",
    "REALTIME_GENERIC_DYNAMICS_ALGORITHMS",
    "SUPPORTED_GENERIC_DYNAMICS_ALGORITHMS",
    "run_native_generic_comparator_row",
    "run_qiskit_community_comparator_row",
    "run_realtime_generic_dynamics_row",
    "skipped_generic_dynamics_row",
    "write_dynamics_row_bundle",
    "write_skipped_generic_dynamics_row",
]


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_values(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        maybe = _float_or_none(value)
        if maybe is not None:
            out.append(float(maybe))
    return out


def _json_numeric_values_are_finite(value: Any) -> bool:
    """Return True when all numeric leaves are finite JSON-safe scalars."""

    if value is None or isinstance(value, str):
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(math.isfinite(float(value)))
    if isinstance(value, Mapping):
        return all(_json_numeric_values_are_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_json_numeric_values_are_finite(item) for item in value)
    return True


def _check_payload(
    *,
    check_id: str,
    passed: bool | None,
    status: str | None = None,
    check_type: str = "invariant_correctness",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = str(status or ("passed" if passed is True else "failed" if passed is False else "not_applicable"))
    return {
        "check_id": str(check_id),
        "check_type": str(check_type),
        "status": state,
        "passed": None if passed is None else bool(passed),
        "details": json_safe(dict(details or {})),
    }


def _checks_pass(checks: Sequence[Mapping[str, Any]]) -> bool:
    return all(check.get("passed") is not False for check in checks)


def _numeric_vector_or_none(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        finite = _finite_values(value)
        return tuple(float(item) for item in finite) if finite else None
    finite_value = _float_or_none(value)
    return None if finite_value is None else (float(finite_value),)


def _aligned_vector_delta(left: Sequence[float], right: Sequence[float]) -> float:
    shared = min(len(left), len(right))
    deltas = [abs(float(left[idx]) - float(right[idx])) for idx in range(shared)]
    if len(left) > shared:
        deltas.extend(abs(float(item)) for item in left[shared:])
    if len(right) > shared:
        deltas.extend(abs(float(item)) for item in right[shared:])
    return float(max(deltas, default=0.0))


def _movement_from_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float | None:
    """Return maximum same-observable motion across checkpoints.

    Heterogeneous observables can have different absolute scales, so movement is
    computed per key over time rather than by flattening all row fields together.
    """

    movements: list[float] = []
    for key in keys:
        series: list[tuple[float, ...]] = []
        for row in rows:
            numeric = _numeric_vector_or_none(row.get(key))
            if numeric is not None:
                series.append(numeric)
        if len(series) < 2:
            continue
        base = series[0]
        movements.extend(_aligned_vector_delta(base, item) for item in series[1:])
    if not movements:
        return None
    return float(max(movements))


def _rows_from_payload(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    raw = payload.get(key, [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [dict(row) for row in raw if isinstance(row, Mapping)]


def _metadata_drive_amplitude(case: DynamicsBenchmarkCase, payload: Mapping[str, Any]) -> float | None:
    drive_config = payload.get("drive_config", {}) if isinstance(payload.get("drive_config", {}), Mapping) else {}
    route_config = payload.get("route_config", {}) if isinstance(payload.get("route_config", {}), Mapping) else {}
    case_metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    drive_meta = case_metadata.get("drive", {}) if isinstance(case_metadata.get("drive", {}), Mapping) else {}
    for value in (
        drive_config.get("A"),
        (drive_config.get("profile_payload", {}) or {}).get("A") if isinstance(drive_config.get("profile_payload", {}), Mapping) else None,
        route_config.get("drive_A"),
        drive_meta.get("A"),
        case_metadata.get("drive_A"),
    ):
        maybe = _float_or_none(value)
        if maybe is not None:
            return float(maybe)
    return None


def build_fixed_mclachlan_correctness_sidecar(
    *,
    case: DynamicsBenchmarkCase,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the repo-native fixed-McLachlan correctness sidecar.

    This is separate from the Qiskit scaffold/state parity sidecar.  It audits
    telemetry emitted by the fixed-manifold McLachlan controller: geometry/RHS
    fields, regularized solve metadata, integrator policy, A=0/no-drive
    consistency, and frozen-trajectory diagnostics.  Exact reference values are
    read only as diagnostic row fields and never feed controller decisions.
    """

    summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), Mapping) else {}
    route_config = payload.get("route_config", {}) if isinstance(payload.get("route_config", {}), Mapping) else {}
    runtime_contract = payload.get("runtime_contract", {}) if isinstance(payload.get("runtime_contract", {}), Mapping) else {}
    provenance = payload.get("provenance", {}) if isinstance(payload.get("provenance", {}), Mapping) else {}
    decision_flag_locations: tuple[tuple[str, Mapping[str, Any]], ...] = (
        ("runtime_contract", runtime_contract),
        ("summary", summary),
        ("route_config", route_config),
        ("provenance", provenance),
    )
    exact_decision_flag_keys = (
        "uses_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
        "exact_reference_controller_inputs",
    )
    exact_decision_truthy: list[dict[str, Any]] = []
    exact_decision_missing: list[str] = []
    for key in exact_decision_flag_keys:
        observed = False
        for location, mapping in decision_flag_locations:
            if key not in mapping:
                continue
            observed = True
            if _boolish(mapping.get(key)):
                exact_decision_truthy.append(
                    {"location": location, "field": key, "value": mapping.get(key)}
                )
        if not observed:
            exact_decision_missing.append(key)
    exact_decision_free = not exact_decision_truthy and not exact_decision_missing
    rows = _rows_from_payload(payload, "trajectory")
    ledger = _rows_from_payload(payload, "ledger")
    sample_rows = [row for row in rows if str(row.get("trajectory_sample_kind", "state_sample")) == "state_sample"]
    geometry_rows = [row for row in sample_rows if isinstance(row.get("baseline_geometry"), Mapping)]
    checks: list[dict[str, Any]] = []

    required_geometry_keys = (
        "rho_miss",
        "rho_real",
        "rho_num",
        "epsilon_proj_sq",
        "epsilon_step_sq",
        "theta_dot_l2",
        "matrix_rank",
        "condition_number",
        "regularization_lambda",
        "solve_mode",
        "runtime_parameter_count",
    )
    geometry_missing: list[dict[str, Any]] = []
    geometry_bad_numeric: list[int] = []
    regularization_bad: list[int] = []
    for row in sample_rows:
        idx = int(row.get("checkpoint_index", len(geometry_missing)))
        geom = row.get("baseline_geometry")
        if not isinstance(geom, Mapping):
            geometry_missing.append({"checkpoint_index": idx, "reason": "missing_baseline_geometry"})
            continue
        missing = [key for key in required_geometry_keys if key not in geom]
        if missing:
            geometry_missing.append({"checkpoint_index": idx, "missing": missing})
        if not _json_numeric_values_are_finite(geom):
            geometry_bad_numeric.append(idx)
        rank = _int_or_none(geom.get("matrix_rank"))
        runtime_count = _int_or_none(geom.get("runtime_parameter_count"))
        regularization = _float_or_none(geom.get("regularization_lambda"))
        solve_mode = str(geom.get("solve_mode", ""))
        if (
            rank is None
            or runtime_count is None
            or rank < 0
            or rank > max(runtime_count, 0)
            or regularization is None
            or regularization < 0.0
            or solve_mode not in {"pinv_reg", "grouped_raw_measured"}
        ):
            regularization_bad.append(idx)
    checks.append(
        _check_payload(
            check_id="metric_force_rhs_geometry_fields",
            check_type="mclachlan_dense_geometry_telemetry",
            passed=bool(sample_rows) and not geometry_missing and not geometry_bad_numeric,
            details={
                "sample_row_count": len(sample_rows),
                "geometry_row_count": len(geometry_rows),
                "required_geometry_keys": list(required_geometry_keys),
                "missing_or_incomplete": geometry_missing,
                "nonfinite_geometry_checkpoint_indices": geometry_bad_numeric,
            },
        )
    )
    checks.append(
        _check_payload(
            check_id="regularized_pseudoinverse_solve_metadata",
            check_type="mclachlan_linear_solve_correctness",
            passed=bool(sample_rows) and not regularization_bad,
            details={
                "bad_checkpoint_indices": regularization_bad,
                "accepted_solve_modes": ["pinv_reg", "grouped_raw_measured"],
            },
        )
    )

    integrator_bad: list[dict[str, Any]] = []
    integrator_used_values: set[str] = set()
    terminal_sample_pos = len(sample_rows) - 1
    for pos, row in enumerate(sample_rows):
        idx = int(row.get("checkpoint_index", len(integrator_bad)))
        policy = str(row.get("integrator_policy", summary.get("integrator_policy", route_config.get("integrator_policy", ""))))
        used = str(row.get("integrator_used", ""))
        if used:
            integrator_used_values.add(used)
        terminal_no_step = pos == terminal_sample_pos and used == "none"
        if policy not in {"euler", "rk4", "auto_euler_rk4"}:
            integrator_bad.append({"checkpoint_index": idx, "reason": "bad_policy", "value": policy})
        if used not in {"euler", "rk4", "stay", "no_advance"} and not terminal_no_step:
            integrator_bad.append({"checkpoint_index": idx, "reason": "bad_used", "value": used})
        if policy == "auto_euler_rk4" and row.get("integrator_auto_policy_schema") in {None, ""}:
            integrator_bad.append({"checkpoint_index": idx, "reason": "missing_auto_policy_schema"})
    checks.append(
        _check_payload(
            check_id="integrator_policy_and_step_semantics",
            check_type="mclachlan_integrator_correctness",
            passed=bool(sample_rows) and not integrator_bad,
            details={
                "bad_rows": integrator_bad,
                "integrator_used_values": sorted(integrator_used_values),
                "covers_euler_rk4_auto_contract": True,
            },
        )
    )

    fixed_manifold_bad: list[dict[str, Any]] = []
    for pos, row in enumerate(sample_rows):
        idx = int(row.get("checkpoint_index", pos))
        if "runtime_parameter_count_delta" not in row:
            fixed_manifold_bad.append({"checkpoint_index": idx, "reason": "missing_runtime_parameter_count_delta"})
        elif _int_or_none(row.get("runtime_parameter_count_delta")) != 0:
            fixed_manifold_bad.append(
                {
                    "checkpoint_index": idx,
                    "reason": "nonzero_runtime_parameter_count_delta",
                    "value": row.get("runtime_parameter_count_delta"),
                }
            )
        if "action_kind" not in row:
            fixed_manifold_bad.append({"checkpoint_index": idx, "reason": "missing_action_kind"})
        elif str(row.get("action_kind")) in {"append_candidate", "prune_coordinate", "repair_miss"}:
            fixed_manifold_bad.append(
                {"checkpoint_index": idx, "reason": "forbidden_action_kind", "value": row.get("action_kind")}
            )
    checks.append(
        _check_payload(
            check_id="fixed_manifold_no_append_prune_semantics",
            check_type="mclachlan_fixed_manifold_correctness",
            passed=bool(sample_rows) and not fixed_manifold_bad,
            details={"bad_rows": fixed_manifold_bad},
        )
    )

    drive_a = _metadata_drive_amplitude(case, payload)
    if drive_a is None or abs(float(drive_a)) > 1.0e-12:
        a0_passed: bool | None = None
        a0_status = "not_applicable_nonzero_or_unspecified_drive"
    else:
        a0_passed = bool(exact_decision_free)
        a0_status = "passed" if a0_passed else "failed_exact_decision_leakage_on_a0_check"
    checks.append(
        _check_payload(
            check_id="a0_no_drive_consistency_boundary",
            check_type="mclachlan_a0_no_drive_invariant",
            passed=a0_passed,
            status=a0_status,
            details={
                "drive_A": drive_a,
                "note": "Nonzero-drive rows mark this check not_applicable; A=0 rows must remain exact-decision-free and finite.",
                "truthy_exact_decision_flags": exact_decision_truthy,
                "missing_exact_decision_flags": exact_decision_missing,
            },
        )
    )

    exact_motion = _movement_from_rows(
        sample_rows,
        (
            "energy_total_exact",
            "primary_density_exact",
            "staggered_exact",
            "doublon_exact",
            "site_occupations_exact",
        ),
    )
    controller_motion = _movement_from_rows(
        sample_rows,
        (
            "theta_update_l2",
            "energy_total_controller",
            "primary_density",
            "staggered",
            "doublon",
            "site_occupations",
        ),
    )
    theta_updates = _finite_values([row.get("theta_update_l2") for row in sample_rows])
    theta_motion = max(theta_updates, default=0.0)
    exact_moves = exact_motion is not None and float(exact_motion) > 1.0e-12
    controller_moves = max(float(controller_motion or 0.0), float(theta_motion)) > 1.0e-12
    if not exact_moves:
        frozen_passed: bool | None = None
        frozen_status = "not_applicable_no_moving_exact_diagnostic"
    else:
        frozen_passed = bool(controller_moves)
        frozen_status = "passed" if frozen_passed else "failed_controller_frozen_while_exact_moves"
    checks.append(
        _check_payload(
            check_id="non_frozen_when_exact_diagnostic_moves",
            check_type="mclachlan_trajectory_invariant",
            passed=frozen_passed,
            status=frozen_status,
            details={
                "exact_diagnostic_motion": exact_motion,
                "controller_motion": controller_motion,
                "max_theta_update_l2": theta_motion,
            },
        )
    )

    checks.append(
        _check_payload(
            check_id="exact_reference_diagnostic_only",
            check_type="decision_data_flow_correctness",
            passed=bool(exact_decision_free),
            details={
                "controller_exact_input_mode": summary.get(
                    "controller_exact_input_mode",
                    runtime_contract.get("controller_exact_input_mode", route_config.get("controller_exact_input_mode")),
                ),
                "diagnostic_exact_reference_mode": summary.get(
                    "diagnostic_exact_reference_mode",
                    runtime_contract.get("diagnostic_exact_reference_mode", route_config.get("diagnostic_exact_reference_mode")),
                ),
                "truthy_exact_decision_flags": exact_decision_truthy,
                "missing_exact_decision_flags": exact_decision_missing,
                "required_explicit_false_fields": list(exact_decision_flag_keys),
            },
        )
    )

    finite_payload = _json_numeric_values_are_finite(json_safe({"trajectory": sample_rows, "ledger": ledger}))
    checks.append(
        _check_payload(
            check_id="finite_trajectory_and_ledger_diagnostics",
            check_type="trajectory_invariant_correctness",
            passed=bool(sample_rows) and bool(finite_payload),
            details={"trajectory_rows": len(sample_rows), "ledger_rows": len(ledger)},
        )
    )

    passed = _checks_pass(checks)
    return json_safe(
        {
            "schema": MCLACHLAN_CORRECTNESS_SCHEMA,
            "algorithm_id": "dyn_fixed_mclachlan",
            "family": str(case.family),
            "case_id": str(case.case_id),
            "sidecar_name": CORRECTNESS_SIDECAR_FILENAMES["dyn_fixed_mclachlan"],
            "support_scope": "fixed_mclachlan_metric_rhs_solve_integrator_and_nonfrozen_correctness",
            "sidecar_kind": "dense_telemetry_and_invariant_correctness",
            "status": "ok" if passed else "failed",
            "passed": bool(passed),
            "required_status": "passed",
            "check_count": int(len(checks)),
            "checks": checks,
            "exact_data_policy": "benchmark_exact_fields_diagnostic_reporting_only_not_controller_input",
            "physical_error_policy": "additive_correctness_provenance_not_a_physical_error_column",
            "controller_decisions_modified": False,
            "exact_reference_controller_inputs": False,
            "qiskit_scaffold_parity_is_separate": True,
        }
    )


def _mean_or_none(values: Sequence[Any]) -> float | None:
    finite = _finite_values(values)
    return None if not finite else float(sum(finite) / len(finite))


def _max_or_none(values: Sequence[Any]) -> float | None:
    finite = _finite_values(values)
    return None if not finite else float(max(finite))


def _min_or_none(values: Sequence[Any]) -> float | None:
    finite = _finite_values(values)
    return None if not finite else float(min(finite))


def _trajectory(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("trajectory", [])
    return [row for row in raw if isinstance(row, Mapping)] if isinstance(raw, list) else []


def _row_energy_error(row: Mapping[str, Any]) -> float | None:
    explicit = _float_or_none(row.get("abs_energy_total_error"))
    if explicit is not None:
        return explicit
    energy = _float_or_none(row.get("energy_total_controller", row.get("energy_total")))
    exact = _float_or_none(row.get("energy_total_exact"))
    return None if energy is None or exact is None else abs(float(energy) - float(exact))


def _energy_metrics(payload: Mapping[str, Any], *, algorithm_id: str) -> dict[str, Any]:
    rows = _trajectory(payload)
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    final = rows[-1] if rows else {}
    exact_values = [_float_or_none(row.get("energy_total_exact")) for row in rows]

    if algorithm_id == "dyn_exact_reference" and any(value is not None for value in exact_values):
        final_exact = next((value for value in reversed(exact_values) if value is not None), None)
        errors: list[float | None] = [0.0 for value in exact_values if value is not None]
        return {
            "trajectory_points": int(len(rows)),
            "final_energy_total": final_exact,
            "final_energy_total_exact": final_exact,
            "final_abs_energy_total_error": 0.0,
            "mean_abs_energy_total_error": 0.0,
            "max_abs_energy_total_error": 0.0,
            "energy_error_policy": "exact_reference_self_comparison",
        }

    errors = [_row_energy_error(row) for row in rows]
    final_error = _row_energy_error(final) if final else None
    return {
        "trajectory_points": int(len(rows)),
        "final_energy_total": _float_or_none(
            final.get("energy_total_controller", final.get("energy_total", summary.get("final_energy_total")))
        ),
        "final_energy_total_exact": _float_or_none(
            final.get("energy_total_exact", summary.get("final_energy_total_exact"))
        ),
        "final_abs_energy_total_error": final_error,
        "mean_abs_energy_total_error": _mean_or_none(errors),
        "max_abs_energy_total_error": _max_or_none(errors),
        "energy_error_policy": "controller_minus_benchmark_exact_when_available",
    }


def _observable_metrics(payload: Mapping[str, Any], *, algorithm_id: str) -> dict[str, Any]:
    rows = _trajectory(payload)
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    if algorithm_id == "dyn_exact_reference" and rows:
        return {
            "max_abs_primary_density_error": 0.0,
            "mean_abs_primary_density_error": 0.0,
            "max_abs_site_occupations_error": 0.0,
            "epsilon_obs_2_policy": "exact_reference_self_comparison",
        }
    primary_errors = [row.get("abs_primary_density_error") for row in rows]
    site_errors = [row.get("site_occupations_abs_error_max") for row in rows]
    site_max = _float_or_none(summary.get("max_abs_site_occupations_error"))
    if site_max is None:
        site_max = _max_or_none(site_errors)
    primary_max = _float_or_none(summary.get("max_abs_primary_density_error"))
    if primary_max is None:
        primary_max = _max_or_none(primary_errors)
    return {
        "max_abs_primary_density_error": primary_max,
        "mean_abs_primary_density_error": _float_or_none(
            summary.get("mean_abs_primary_density_error")
        )
        if summary.get("mean_abs_primary_density_error") is not None
        else _mean_or_none(primary_errors),
        "max_abs_site_occupations_error": site_max,
        "epsilon_obs_2_policy": "site_occupations_max_else_primary_density_max",
    }


def _fidelity_metrics(payload: Mapping[str, Any], *, algorithm_id: str) -> dict[str, Any]:
    rows = _trajectory(payload)
    if algorithm_id == "dyn_exact_reference" and rows:
        return {
            "final_fidelity_exact": 1.0,
            "min_fidelity_exact": 1.0,
            "one_minus_final_fidelity_exact": 0.0,
            "one_minus_min_fidelity_exact": 0.0,
            "fidelity_policy": "exact_reference_self_comparison",
        }
    final_fidelity = _float_or_none(rows[-1].get("fidelity_exact")) if rows else None
    min_fidelity = _min_or_none([row.get("fidelity_exact") for row in rows])
    return {
        "final_fidelity_exact": final_fidelity,
        "min_fidelity_exact": min_fidelity,
        "one_minus_final_fidelity_exact": (
            None if final_fidelity is None else float(max(0.0, 1.0 - final_fidelity))
        ),
        "one_minus_min_fidelity_exact": (
            None if min_fidelity is None else float(max(0.0, 1.0 - min_fidelity))
        ),
        "fidelity_policy": "controller_minus_benchmark_exact_when_available",
    }


def _compile_resources(payload: Mapping[str, Any]) -> dict[str, Any]:
    compile_audit = payload.get("compile_audit", {})
    if not isinstance(compile_audit, Mapping):
        compile_audit = {}
    selected = compile_audit.get("selected_backend", {})
    if not isinstance(selected, Mapping):
        selected = {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    route = payload.get("route_config", {}) if isinstance(payload.get("route_config"), Mapping) else {}
    return {
        "compiled_count_2q_total": _int_or_none(selected.get("compiled_count_2q")),
        "compiled_depth_2q_total": _int_or_none(
            selected.get("compiled_depth_2q", selected.get("compiled_depth_2q_total"))
        ),
        "compiled_depth_total": _int_or_none(selected.get("compiled_depth")),
        "compiled_size_total": _int_or_none(selected.get("compiled_size")),
        "compiled_backend_name": selected.get("backend_name"),
        "shots_total": _int_or_none(summary.get("shots_total", route.get("shots_total"))),
    }


_PROTECTED_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "trajectory_points",
        "final_energy_total",
        "final_energy_total_exact",
        "final_abs_energy_total_error",
        "mean_abs_energy_total_error",
        "max_abs_energy_total_error",
        "energy_error_policy",
        "max_abs_primary_density_error",
        "mean_abs_primary_density_error",
        "max_abs_site_occupations_error",
        "epsilon_obs_2_policy",
        "final_fidelity_exact",
        "min_fidelity_exact",
        "one_minus_final_fidelity_exact",
        "one_minus_min_fidelity_exact",
        "fidelity_policy",
    }
)

_PROTECTED_PROVENANCE_KEYS: frozenset[str] = frozenset(
    {
        "benchmark_only",
        "controller_decisions_modified",
        "exact_reference_controller_inputs",
        "uses_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
    }
)


def _merge_auxiliary_metrics(
    *,
    computed: Mapping[str, Any],
    payload_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge payload metrics without letting stale payloads clobber contract fields."""

    merged = dict(computed)
    ignored: dict[str, Any] = {}
    for key, value in dict(payload_metrics).items():
        if key in _PROTECTED_METRIC_KEYS and key in merged:
            if value != merged.get(key):
                ignored[str(key)] = value
            continue
        merged[str(key)] = value
    if ignored:
        merged["ignored_payload_metric_overrides"] = json_safe(ignored)
    return merged


def _method_table_label(algorithm_id: str) -> str:
    if algorithm_id == "dyn_exact_reference":
        return "diagnostic exact reference"
    if algorithm_id == "dyn_fixed_mclachlan":
        return "fixed-scaffold McLachlan"
    if algorithm_id == "dyn_product_formula_envelope":
        return "product-formula/Suzuki envelope"
    if algorithm_id == "dyn_qdrift":
        return "qDRIFT randomized product formula"
    if algorithm_id == "dyn_fixed_pvqd":
        return "fixed pVQD product-formula target"
    if algorithm_id == "dyn_adaptive_pvqd":
        return "adaptive pVQD product-formula target"
    if algorithm_id == "dyn_avqds":
        return "AVQDS tangent diagnostic"
    if algorithm_id == "dyn_avqds_t":
        return "PF-target adaptive tangent diagnostic"
    if algorithm_id == "dyn_avqds_tetris":
        return "AVQDS(T) Method-3 TETRIS"
    if algorithm_id in QISKIT_COMMUNITY_TABLE_LABELS:
        return QISKIT_COMMUNITY_TABLE_LABELS[algorithm_id]
    if algorithm_id == "dyn_controller_full":
        return "full strict controller"
    if algorithm_id == "dyn_controller_fixed_scaffold":
        return "fixed scaffold control"
    if algorithm_id == "dyn_controller_no_append":
        return "no append"
    if algorithm_id == "dyn_controller_no_pruning":
        return "no pruning"
    if algorithm_id == "dyn_controller_fixed_integrator":
        return "appendix fixed Euler diagnostic"
    if algorithm_id == "dyn_controller_no_residual_split":
        return "no residual-split confirmation"
    return "skipped"


def _table_fields(*, algorithm_id: str, metrics: Mapping[str, Any], resources: Mapping[str, Any]) -> DynamicsTableFields:
    epsilon_obs = metrics.get("max_abs_site_occupations_error")
    if epsilon_obs is None:
        epsilon_obs = metrics.get("max_abs_primary_density_error")
    return DynamicsTableFields(
        mean_abs_energy_total_error=_float_or_none(metrics.get("mean_abs_energy_total_error")),
        epsilon_obs_2=_float_or_none(epsilon_obs),
        one_minus_min_fidelity_exact=_float_or_none(metrics.get("one_minus_min_fidelity_exact")),
        epsilon_spec=None,
        compiled_count_2q_total=_int_or_none(resources.get("compiled_count_2q_total")),
        compiled_depth_2q_total=_int_or_none(resources.get("compiled_depth_2q_total")),
        compiled_depth_total=_int_or_none(resources.get("compiled_depth_total")),
        shots_total=_int_or_none(resources.get("shots_total")),
        table_status_label=_method_table_label(algorithm_id),
    )


def _realtime_argv(*, case: DynamicsBenchmarkCase, algorithm_id: str, raw_payload_json: Path) -> list[str]:
    smoke_fast = _case_smoke_fast_mode(case)
    if str(algorithm_id) == "dyn_fixed_mclachlan":
        controller_mode = "observable_v1"
        controller_exact_input_mode = "off"
        diagnostic_exact_reference_mode = "off" if smoke_fast else "benchmark_exact"
    elif str(algorithm_id) == "dyn_exact_reference":
        controller_mode = "exact_v1"
        controller_exact_input_mode = "benchmark_exact"
        diagnostic_exact_reference_mode = "benchmark_exact"
    else:
        controller_mode = "off"
        controller_exact_input_mode = "benchmark_exact"
        diagnostic_exact_reference_mode = "off"
    argv = [
        "--artifact-json",
        str(case.artifact_json),
        "--output-json",
        str(raw_payload_json),
        "--run-tag",
        f"{case.case_id}_{algorithm_id}",
        "--loader-mode",
        str(case.loader_mode),
        "--generator-family",
        str(case.generator_family),
        "--fallback-family",
        str(case.fallback_family),
        "--append-pool-family",
        str(case.append_pool_family),
        "--checkpoint-controller-mode",
        str(controller_mode),
        "--checkpoint-controller-exact-input-mode",
        str(controller_exact_input_mode),
        "--diagnostic-exact-reference-mode",
        str(diagnostic_exact_reference_mode),
        "--checkpoint-controller-noise-mode",
        "ideal",
        "--num-times",
        str(int(case.num_times)),
        "--t-final",
        str(float(case.t_final)),
        "--compile-audit-mode",
        "off" if smoke_fast else "final_scaffold",
        "--compile-audit-backend-name",
        "FakeMarrakesh",
        "--compile-audit-seed-transpiler",
        "7",
        "--compile-audit-optimization-level",
        "2",
        "--compile-audit-preferred-fake-backends",
        "FakeMarrakesh",
    ]
    if algorithm_id == "dyn_fixed_mclachlan":
        argv.append("--checkpoint-controller-strict-qpu-faithful")
        fixed_integrator_policy = _fixed_mclachlan_integrator_policy_override(case)
        argv.extend(
            [
                "--checkpoint-controller-integrator-policy",
                fixed_integrator_policy or PAPER_II_FIXED_MCLACHLAN_INTEGRATOR_POLICY,
            ]
        )
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    drive = metadata.get("drive", {}) if isinstance(metadata.get("drive", {}), Mapping) else {}
    enable_drive = bool(drive.get("enable_drive", metadata.get("enable_drive", False)))
    disable_drive = bool(drive.get("disable_drive", metadata.get("disable_drive", False)))
    if enable_drive and disable_drive:
        raise ValueError(f"case {case.case_id}: enable_drive and disable_drive cannot both be true")
    if disable_drive:
        argv.append("--disable-drive")
    elif enable_drive:
        argv.extend(
            [
                "--enable-drive",
                "--drive-A",
                str(float(drive.get("A", drive.get("drive_A", metadata.get("drive_A", 0.0))))),
                "--drive-omega",
                str(float(drive.get("omega", drive.get("drive_omega", metadata.get("drive_omega", 1.0))))),
                "--drive-tbar",
                str(float(drive.get("tbar", drive.get("drive_tbar", metadata.get("drive_tbar", 1.0))))),
                "--drive-phi",
                str(float(drive.get("phi", drive.get("drive_phi", metadata.get("drive_phi", 0.0))))),
                "--drive-pattern",
                str(drive.get("pattern", drive.get("drive_pattern", metadata.get("drive_pattern", "staggered")))),
                "--drive-custom-weights",
                str(drive.get("custom_weights", drive.get("drive_custom_weights", metadata.get("drive_custom_weights", "")))),
                "--drive-time-sampling",
                str(drive.get("time_sampling", drive.get("drive_time_sampling", metadata.get("drive_time_sampling", "midpoint")))),
                "--drive-t0",
                str(float(drive.get("t0", drive.get("drive_t0", metadata.get("drive_t0", 0.0))))),
            ]
        )
        if bool(drive.get("include_identity", metadata.get("drive_include_identity", False))):
            argv.append("--drive-include-identity")
    if algorithm_id == "dyn_fixed_mclachlan":
        argv.extend(["--lock-fixed-manifold", "--no-checkpoint-controller-append-enabled"])
        if _qiskit_parity_requested_for_case(case):
            argv.append("--emit-fixed-scaffold-qiskit-parity-payload")
    return argv


def _metadata_int(
    case: DynamicsBenchmarkCase,
    key: str,
    default: int,
    *,
    minimum: int = 1,
) -> int:
    raw = case.metadata.get(key, default) if isinstance(case.metadata, Mapping) else default
    value = int(raw)
    if value < int(minimum):
        raise ValueError(f"generic dynamics case metadata {key!r} must be >= {minimum}, got {value}")
    return int(value)


def _metadata_optional_int(
    case: DynamicsBenchmarkCase,
    key: str,
    default: int | None,
    *,
    minimum: int = 0,
) -> int | None:
    raw = case.metadata.get(key, default) if isinstance(case.metadata, Mapping) else default
    if raw is None or raw == "":
        return None
    value = int(raw)
    if value < int(minimum):
        raise ValueError(f"generic dynamics case metadata {key!r} must be >= {minimum}, got {value}")
    return int(value)


def _metadata_float(
    case: DynamicsBenchmarkCase,
    key: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    raw = case.metadata.get(key, default) if isinstance(case.metadata, Mapping) else default
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"generic dynamics case metadata {key!r} must be finite, got {raw!r}")
    if minimum is not None and value < float(minimum):
        raise ValueError(f"generic dynamics case metadata {key!r} must be >= {minimum}, got {value}")
    return float(value)


def _read_case_artifact_payload(case: DynamicsBenchmarkCase) -> Mapping[str, Any]:
    artifact_path = Path(case.artifact_json).expanduser()
    if not artifact_path.exists():
        return {}
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _parity_correctness_sidecars_required_for_case(case: DynamicsBenchmarkCase) -> bool:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    for key in (
        "require_parity_correctness_sidecars",
        "paper_ii_table_i_require_parity_correctness_sidecars",
        "paper_ii_table_i_parity_correctness_required",
    ):
        if key in metadata:
            return _boolish(metadata.get(key))
    requirements = metadata.get("parity_correctness_requirements")
    if isinstance(requirements, Mapping):
        for key in ("required", "require_sidecars", "fail_closed"):
            if key in requirements:
                return _boolish(requirements.get(key))
    return False


@dataclass(frozen=True)
class _NativeDriveConfig:
    enabled: bool
    n_sites: int
    ordering: str
    drive_A: float
    drive_omega: float
    drive_tbar: float
    drive_phi: float
    drive_pattern: str
    drive_custom_weights: tuple[float, ...] | None = None
    drive_include_identity: bool = False
    drive_time_sampling: str = "midpoint"
    drive_t0: float = 0.0
    exact_steps_multiplier: int = 1


@dataclass(frozen=True)
class NativeHamiltonianFlow:
    """Static or driven Hamiltonian context shared by repo-native comparators.

    The exact/reference trajectory is diagnostic-only.  Comparator propagation
    uses ``terms_for_interval`` / ``hmat_for_interval`` sampled from the same
    drive metadata as the checkpoint-controller case.
    """

    case: DynamicsBenchmarkCase
    runtime_input: Any
    hamiltonian: TimeDependentHamiltonian
    static_h_poly: Any
    static_hmat: np.ndarray
    drive_model: Any | None
    drive_time_sampling: str
    drive_t0: float
    exact_steps_multiplier: int
    times: np.ndarray
    exact_states: tuple[np.ndarray, ...]
    observable_context: Mapping[str, Any]

    @property
    def drive_enabled(self) -> bool:
        return self.drive_model is not None

    def physical_time(self, time_value: float) -> float:
        return float(time_value) + float(self.drive_t0)

    def interval_sample_time(self, left: float, right: float) -> float:
        sampling = str(self.drive_time_sampling).strip().lower()
        if sampling == "midpoint":
            base = 0.5 * (float(left) + float(right))
        elif sampling == "left":
            base = float(left)
        elif sampling == "right":
            base = float(right)
        else:
            raise ValueError(f"unsupported drive_time_sampling={self.drive_time_sampling!r}")
        return self.physical_time(base)

    def h_poly_at_physical_time(self, physical_time: float) -> Any:
        return self.hamiltonian.polynomial_at(float(physical_time))

    def hmat_at_time(self, time_value: float) -> np.ndarray:
        return self.hamiltonian.matrix_at(self.physical_time(float(time_value)))

    def hmat_for_interval(self, left: float, right: float) -> np.ndarray:
        return self.hamiltonian.matrix_at(self.interval_sample_time(left, right))

    def terms_for_interval(self, left: float, right: float) -> tuple[Any, ...]:
        return _active_hamiltonian_terms(
            self.hamiltonian.polynomial_at(self.interval_sample_time(left, right))
        )

    def hmat_sequence_for_times(self) -> tuple[np.ndarray, ...]:
        return tuple(self.hmat_at_time(float(t)) for t in self.times)

    def hmat_sequence_for_trajectory_samples(self) -> tuple[np.ndarray, ...]:
        """Hamiltonians used for emitted trajectory-energy diagnostics.

        Checkpoint-controller rows report the state at grid time ``t_k`` but
        evaluate driven-energy diagnostics at the interval sample time for
        nonterminal intervals.  Matching that convention keeps comparator
        ``energy_total`` fields on the same observable definition.
        """

        times = [float(t) for t in np.asarray(self.times, dtype=float).reshape(-1)]
        if not times:
            return tuple()
        if not self.drive_enabled:
            return tuple(np.asarray(self.static_hmat, dtype=complex) for _ in times)
        mats: list[np.ndarray] = []
        for idx, time_value in enumerate(times):
            if idx < len(times) - 1:
                physical_time = self.interval_sample_time(time_value, times[idx + 1])
            else:
                physical_time = self.physical_time(time_value)
            mats.append(
                np.asarray(
                    hamiltonian_matrix(self.h_poly_at_physical_time(float(physical_time))),
                    dtype=complex,
                )
            )
        return tuple(mats)


def _drive_metadata(case: DynamicsBenchmarkCase) -> Mapping[str, Any]:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    drive = metadata.get("drive", {}) if isinstance(metadata.get("drive", {}), Mapping) else {}
    return drive


def _drive_is_enabled(case: DynamicsBenchmarkCase) -> bool:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    drive = _drive_metadata(case)
    if _boolish(drive.get("disable_drive", metadata.get("disable_drive", False))):
        return False
    return _boolish(drive.get("enable_drive", metadata.get("enable_drive", metadata.get("drive_enabled", False))))


def _custom_weights_tuple(raw: Any) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        return tuple(float(item) for item in parts) if parts else None
    if isinstance(raw, Sequence):
        return tuple(float(item) for item in raw)
    return None


def _native_drive_config_for_case(case: DynamicsBenchmarkCase, runtime_input: Any) -> _NativeDriveConfig | None:
    if not _drive_is_enabled(case):
        return None
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    drive = _drive_metadata(case)
    resolved = getattr(runtime_input, "resolved_problem", None)
    request = getattr(resolved, "request", None)
    n_sites = int(
        drive.get(
            "n_sites",
            metadata.get(
                "n_sites",
                getattr(request, "num_sites", getattr(runtime_input, "num_sites", 1)),
            ),
        )
    )
    ordering = str(
        drive.get(
            "ordering",
            metadata.get(
                "ordering",
                getattr(request, "ordering", getattr(runtime_input, "ordering", "blocked")),
            ),
        )
    )
    return _NativeDriveConfig(
        enabled=True,
        n_sites=int(n_sites),
        ordering=str(ordering),
        drive_A=float(drive.get("A", drive.get("drive_A", metadata.get("drive_A", 0.0)))),
        drive_omega=float(drive.get("omega", drive.get("drive_omega", metadata.get("drive_omega", 1.0)))),
        drive_tbar=float(drive.get("tbar", drive.get("drive_tbar", metadata.get("drive_tbar", 1.0)))),
        drive_phi=float(drive.get("phi", drive.get("drive_phi", metadata.get("drive_phi", 0.0)))),
        drive_pattern=str(drive.get("pattern", drive.get("drive_pattern", metadata.get("drive_pattern", "staggered")))),
        drive_custom_weights=_custom_weights_tuple(
            drive.get("custom_weights", drive.get("drive_custom_weights", metadata.get("drive_custom_weights", "")))
        ),
        drive_include_identity=_boolish(
            drive.get("include_identity", metadata.get("drive_include_identity", False))
        ),
        drive_time_sampling=str(
            drive.get("time_sampling", drive.get("drive_time_sampling", metadata.get("drive_time_sampling", "midpoint")))
        ),
        drive_t0=float(drive.get("t0", drive.get("drive_t0", metadata.get("drive_t0", 0.0)))),
        exact_steps_multiplier=int(
            drive.get("exact_steps_multiplier", metadata.get("exact_steps_multiplier", 1))
        ),
    )


def _resolve_native_drive_model(case: DynamicsBenchmarkCase, runtime_input: Any) -> tuple[Any | None, _NativeDriveConfig | None]:
    drive_config = _native_drive_config_for_case(case, runtime_input)
    if drive_config is None:
        return None, None
    resolved = getattr(runtime_input, "resolved_problem", None)
    return resolve_realtime_drive_model(
        resolved_problem=resolved,
        drive_config=drive_config,
    ), drive_config


def _driven_exact_states_for_times(
    *,
    flow: NativeHamiltonianFlow,
    psi_initial: np.ndarray,
) -> tuple[np.ndarray, ...]:
    times = np.asarray(flow.times, dtype=float).reshape(-1)
    states: list[np.ndarray] = [_normalize_state(psi_initial)]
    psi = _normalize_state(psi_initial)
    multiplier = max(1, int(flow.exact_steps_multiplier))
    for left, right in zip(times[:-1], times[1:]):
        left_f = float(left)
        right_f = float(right)
        dt_total = float(right_f - left_f)
        if dt_total <= 0.0:
            states.append(np.asarray(psi, dtype=complex))
            continue
        dt_micro = dt_total / float(multiplier)
        for micro in range(multiplier):
            micro_left = left_f + float(micro) * dt_micro
            micro_right = micro_left + dt_micro
            hmat_step = flow.hmat_for_interval(micro_left, micro_right)
            hermitian = 0.5 * (np.asarray(hmat_step, dtype=complex) + np.asarray(hmat_step, dtype=complex).conj().T)
            evals, evecs = np.linalg.eigh(hermitian)
            psi = _exact_step_from_eigendecomp(
                evals=np.asarray(evals, dtype=float),
                evecs=np.asarray(evecs, dtype=complex),
                psi=psi,
                dt=float(dt_micro),
            )
        states.append(np.asarray(psi, dtype=complex))
    return tuple(states)


def _native_hamiltonian_flow(case: DynamicsBenchmarkCase, runtime_input: Any) -> NativeHamiltonianFlow:
    times = np.linspace(0.0, float(case.t_final), int(case.num_times))
    drive_model, drive_config = _resolve_native_drive_model(case, runtime_input)
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_model=drive_model,
    )
    h_poly_static = hamiltonian.static_poly
    hmat_static = np.asarray(hamiltonian_matrix(h_poly_static), dtype=complex)
    observable_context = _observable_context_from_runtime_input(runtime_input)
    placeholder = NativeHamiltonianFlow(
        case=case,
        runtime_input=runtime_input,
        hamiltonian=hamiltonian,
        static_h_poly=h_poly_static,
        static_hmat=hmat_static,
        drive_model=drive_model,
        drive_time_sampling="left" if drive_config is None else str(drive_config.drive_time_sampling),
        drive_t0=0.0 if drive_config is None else float(drive_config.drive_t0),
        exact_steps_multiplier=1 if drive_config is None else int(drive_config.exact_steps_multiplier),
        times=np.asarray(times, dtype=float),
        exact_states=(),
        observable_context=observable_context,
    )
    psi_initial = _normalize_state(runtime_input.psi_initial)
    if _case_smoke_fast_mode(case):
        exact_states = tuple(np.asarray(psi_initial, dtype=complex).reshape(-1) for _ in times)
    elif drive_model is None:
        exact_states = tuple(_exact_states_for_times(hmat=hmat_static, psi_initial=psi_initial, times=times))
    else:
        exact_states = _driven_exact_states_for_times(flow=placeholder, psi_initial=psi_initial)
    return NativeHamiltonianFlow(
        case=case,
        runtime_input=runtime_input,
        hamiltonian=hamiltonian,
        static_h_poly=h_poly_static,
        static_hmat=hmat_static,
        drive_model=drive_model,
        drive_time_sampling="left" if drive_config is None else str(drive_config.drive_time_sampling),
        drive_t0=0.0 if drive_config is None else float(drive_config.drive_t0),
        exact_steps_multiplier=1 if drive_config is None else int(drive_config.exact_steps_multiplier),
        times=np.asarray(times, dtype=float),
        exact_states=tuple(exact_states),
        observable_context=observable_context,
    )


def _assert_native_case_supported(case: DynamicsBenchmarkCase, runtime_input: Any) -> None:
    """Validate that the repo-native comparator can honor the case time dependence."""

    payload = _read_case_artifact_payload(case)
    settings = payload.get("settings", {}) if isinstance(payload.get("settings"), Mapping) else {}
    drive_config = (
        payload.get("drive_config", {}) if isinstance(payload.get("drive_config"), Mapping) else {}
    )
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    drive_enabled = _drive_is_enabled(case) or any(
        _boolish(value)
        for value in (
            drive_config.get("enabled"),
            settings.get("drive_enabled"),
            settings.get("enable_drive"),
            metadata.get("drive_enabled"),
            metadata.get("enable_drive"),
        )
        if value is not None
    )
    time_dependence = str(metadata.get("time_dependence", settings.get("time_dependence", "static"))).lower()
    if drive_enabled:
        _resolve_native_drive_model(case, runtime_input)
        return
    if time_dependence not in {"", "none", "static", "static_only", "time_independent"}:
        raise ValueError(
            "repo-native generic dynamics comparator requires either static "
            "time dependence or a supported drive metadata block"
        )


def _load_runtime_input_for_case(case: DynamicsBenchmarkCase) -> Any:
    kwargs: dict[str, Any] = {
        "loader_mode": case.loader_mode,
        "tag": f"generic_dynamics_{case.case_id}",
        "generator_family": case.generator_family,
        "fallback_family": case.fallback_family,
    }
    try:
        parameters = inspect.signature(load_scaffold_runtime_input).parameters
    except (TypeError, ValueError):
        parameters = {}
    supports_append_pool = "append_pool_family" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if supports_append_pool:
        kwargs["append_pool_family"] = case.append_pool_family
    return load_scaffold_runtime_input(case.artifact_json, **kwargs)


def _observable_context_from_runtime_input(runtime_input: Any) -> dict[str, Any]:
    resolved = getattr(runtime_input, "resolved_problem", None)
    request = getattr(resolved, "request", None)
    raw_num_sites = getattr(request, "num_sites", None)
    if raw_num_sites in {None, ""}:
        raw_num_sites = getattr(runtime_input, "num_sites", None)
    if raw_num_sites in {None, ""}:
        raw_num_sites = 2
    raw_ordering = getattr(request, "ordering", None)
    if raw_ordering in {None, ""}:
        raw_ordering = getattr(runtime_input, "ordering", "site_major")
    return {
        "resolved_problem": resolved,
        "num_sites": int(raw_num_sites),
        "ordering": str(raw_ordering),
    }


def _active_hamiltonian_terms(h_poly: Any) -> tuple[Any, ...]:
    return iter_runtime_rotation_terms(
        h_poly,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )


def _candidate_pool_completeness(runtime_input: Any) -> str:
    source = getattr(runtime_input, "candidate_pool_source", None)
    completeness = getattr(source, "completeness", None)
    if completeness is not None:
        return str(completeness)
    complete = getattr(source, "candidate_pool_complete", None)
    if complete is not None:
        return "complete" if bool(complete) else "selected_only"
    return "selected_only"


def _candidate_pool_is_complete(runtime_input: Any) -> bool:
    return _candidate_pool_completeness(runtime_input) == "complete"


def _term_label(term: Any, fallback_index: int) -> str:
    raw = getattr(term, "label", None)
    if raw is None:
        raw = getattr(term, "candidate_label", None)
    return str(raw if raw is not None else f"term_{int(fallback_index)}")


def _term_label_set(terms: Sequence[Any]) -> set[str]:
    return {_term_label(term, idx) for idx, term in enumerate(terms)}


def _layout_rotation_labels(layout: Any) -> list[str]:
    labels: list[str] = []
    for block in getattr(layout, "blocks", ()):
        for spec in getattr(block, "terms", ()):
            labels.append(str(getattr(spec, "pauli_exyz")))
    return labels


def _build_layout_for_terms(terms: Sequence[Any], *, reference_layout: Any | None) -> Any:
    return build_parameter_layout(
        tuple(terms),
        ignore_identity=bool(getattr(reference_layout, "ignore_identity", True)),
        coefficient_tolerance=float(getattr(reference_layout, "coefficient_tolerance", 1.0e-12)),
        sort_terms=(str(getattr(reference_layout, "term_order", "sorted")).strip().lower() == "sorted"),
    )


def _copy_theta_by_layout_blocks(
    *,
    old_theta: np.ndarray,
    old_layout: Any,
    new_layout: Any,
) -> np.ndarray:
    """Transfer matching scaffold parameters by stable logical block labels."""

    old_vec = np.asarray(old_theta, dtype=float).reshape(-1)
    new_vec = np.zeros(int(getattr(new_layout, "runtime_parameter_count")), dtype=float)
    old_by_label = {
        str(getattr(block, "candidate_label")): block
        for block in getattr(old_layout, "blocks", ())
    }
    for new_block in getattr(new_layout, "blocks", ()):
        label = str(getattr(new_block, "candidate_label"))
        old_block = old_by_label.get(label)
        if old_block is None:
            continue
        old_start = int(getattr(old_block, "runtime_start"))
        old_stop = int(getattr(old_block, "runtime_stop"))
        new_start = int(getattr(new_block, "runtime_start"))
        new_stop = int(getattr(new_block, "runtime_stop"))
        old_slice = old_vec[old_start:old_stop]
        if int(old_slice.size) != int(new_stop - new_start):
            raise ValueError(
                f"cannot transfer theta for block {label!r}: runtime width changed "
                f"from {old_slice.size} to {new_stop - new_start}"
            )
        new_vec[new_start:new_stop] = old_slice
    return new_vec


def _compiled_executor_for_terms(terms: Sequence[Any], layout: Any) -> CompiledAnsatzExecutor:
    return CompiledAnsatzExecutor(
        tuple(terms),
        coefficient_tolerance=float(getattr(layout, "coefficient_tolerance", 1.0e-12)),
        ignore_identity=bool(getattr(layout, "ignore_identity", True)),
        sort_terms=(str(getattr(layout, "term_order", "sorted")).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )


def _ap_state_for_runtime_input(runtime_input: Any) -> APMcLachlanState:
    return state_from_scaffold_runtime_input(runtime_input)


def _runtime_variational_bundle(
    runtime_input: Any,
    *,
    hamiltonian: TimeDependentHamiltonian | None = None,
    drive_aligned_ansatz: bool = False,
) -> tuple[tuple[Any, ...], Any, np.ndarray, np.ndarray, CompiledAnsatzExecutor, Any]:
    state = _ap_state_for_runtime_input(runtime_input)
    augmentation = augment_state_with_drive_aligned_generator(
        state,
        hamiltonian=hamiltonian,
        enabled=bool(drive_aligned_ansatz) and hamiltonian is not None,
    )
    state = augmentation.state
    terms = tuple(state.terms)
    if not terms:
        raise ValueError("generic variational dynamics comparator requires selected scaffold terms")
    layout = state.layout
    theta = np.asarray(state.theta_runtime, dtype=float).reshape(-1)
    expected = int(getattr(layout, "runtime_parameter_count"))
    if int(theta.size) != expected:
        raise ValueError(
            f"theta_runtime length mismatch for generic variational comparator: got {theta.size}, expected {expected}"
        )
    psi_ref = _normalize_state(state.psi_ref)
    return terms, layout, theta, psi_ref, state.executor, augmentation


def _prepare_scaffold_state(
    executor: CompiledAnsatzExecutor,
    psi_ref: Any,
    theta_runtime: Any,
) -> np.ndarray:
    return _normalize_state(
        executor.prepare_state(
            np.asarray(theta_runtime, dtype=float).reshape(-1),
            np.asarray(psi_ref, dtype=complex).reshape(-1),
        )
    )


def _normalize_state(psi: Any) -> np.ndarray:
    arr = np.asarray(psi, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("statevector norm must be positive")
    return np.asarray(arr / norm, dtype=complex)


def _exact_states_for_times(
    *,
    hmat: np.ndarray,
    psi_initial: np.ndarray,
    times: np.ndarray,
) -> list[np.ndarray]:
    hermitian = 0.5 * (np.asarray(hmat, dtype=complex) + np.asarray(hmat, dtype=complex).conj().T)
    evals, evecs = np.linalg.eigh(hermitian)
    coeff = evecs.conj().T @ np.asarray(psi_initial, dtype=complex).reshape(-1)
    states: list[np.ndarray] = []
    for time_value in np.asarray(times, dtype=float).reshape(-1):
        phase = np.exp(-1j * evals * float(time_value))
        states.append(_normalize_state(evecs @ (phase * coeff)))
    return states


def _energy_from_matrix(psi: np.ndarray, hmat: np.ndarray) -> float:
    value = np.vdot(np.asarray(psi, dtype=complex), np.asarray(hmat, dtype=complex) @ np.asarray(psi, dtype=complex))
    return float(np.real(value))


def _observable_comparison_fields(
    *,
    state: np.ndarray,
    exact_state: np.ndarray,
    resolved_problem: Any | None = None,
    num_sites: int | None = None,
    ordering: str | None = None,
    compiled_poly_cache: dict[str, Any] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if resolved_problem is None and num_sites is None:
        return {}
    try:
        n_sites = int(2 if num_sites is None else num_sites)
        order = str("site_major" if ordering in {None, ""} else ordering)
        snapshot = observable_snapshot_for_state(
            np.asarray(state, dtype=complex).reshape(-1),
            resolved_problem=resolved_problem,
            num_sites=n_sites,
            ordering=order,
            compiled_poly_cache=compiled_poly_cache,
            pauli_action_cache=pauli_action_cache,
        )
        exact_snapshot = observable_snapshot_for_state(
            np.asarray(exact_state, dtype=complex).reshape(-1),
            resolved_problem=resolved_problem,
            num_sites=n_sites,
            ordering=order,
            compiled_poly_cache=compiled_poly_cache,
            pauli_action_cache=pauli_action_cache,
        )
        primary_mode = str(
            snapshot.get("primary_density_mode", "auto")
            or exact_snapshot.get("primary_density_mode", "auto")
            or "auto"
        )
        primary = primary_density_value_from_snapshot(
            snapshot,
            resolved_problem=resolved_problem,
            num_sites=n_sites,
            requested_mode=primary_mode,
        )
        primary_exact = primary_density_value_from_snapshot(
            exact_snapshot,
            resolved_problem=resolved_problem,
            num_sites=n_sites,
            requested_mode=primary_mode,
        )
        fields: dict[str, Any] = {
            "observable_family": snapshot.get("observable_family"),
            "site_occupations_label": snapshot.get("site_occupations_label"),
            "site_occupations_component_labels": snapshot.get(
                "site_occupations_component_labels"
            ),
            "site_occupations": snapshot.get("site_occupations"),
            "site_occupations_exact": exact_snapshot.get("site_occupations"),
            "primary_density_mode": primary_mode,
            "primary_density": float(primary),
            "primary_density_exact": float(primary_exact),
            "abs_primary_density_error": float(abs(float(primary) - float(primary_exact))),
        }
        site = snapshot.get("site_occupations")
        site_exact = exact_snapshot.get("site_occupations")
        if isinstance(site, Sequence) and isinstance(site_exact, Sequence):
            errors = [
                abs(float(a) - float(b))
                for a, b in zip(site, site_exact)
                if _float_or_none(a) is not None and _float_or_none(b) is not None
            ]
            fields["site_occupations_abs_error"] = [float(x) for x in errors]
            fields["site_occupations_abs_error_max"] = (
                None if not errors else float(max(errors))
            )
        for key in ("staggered", "doublon", "boson_number_total", "site0_occupation"):
            lhs = _float_or_none(snapshot.get(key))
            rhs = _float_or_none(exact_snapshot.get(key))
            if lhs is not None:
                fields[key] = lhs
            if rhs is not None:
                fields[f"{key}_exact"] = rhs
            if lhs is not None and rhs is not None:
                fields[f"abs_{key}_error"] = float(abs(lhs - rhs))
        return fields
    except Exception as exc:
        return {
            "observable_diagnostic_status": "failed",
            "observable_diagnostic_error": f"{type(exc).__name__}: {exc}",
        }


def _exact_step_from_eigendecomp(
    *,
    evals: np.ndarray,
    evecs: np.ndarray,
    psi: np.ndarray,
    dt: float,
) -> np.ndarray:
    coeff = np.asarray(evecs, dtype=complex).conj().T @ np.asarray(psi, dtype=complex).reshape(-1)
    phase = np.exp(-1j * np.asarray(evals, dtype=float).reshape(-1) * float(dt))
    return _normalize_state(np.asarray(evecs, dtype=complex) @ (phase * coeff))


def _state_diagnostic_row(
    *,
    checkpoint_index: int,
    time_value: float,
    method: str,
    method_kind: str,
    state: np.ndarray,
    exact_state: np.ndarray,
    hmat: np.ndarray,
    extra: Mapping[str, Any] | None = None,
    resolved_problem: Any | None = None,
    num_sites: int | None = None,
    ordering: str | None = None,
    compiled_poly_cache: dict[str, Any] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    energy = _energy_from_matrix(state, hmat)
    exact_energy = _energy_from_matrix(exact_state, hmat)
    fidelity = float(abs(np.vdot(_normalize_state(exact_state), _normalize_state(state))) ** 2)
    row: dict[str, Any] = {
        "checkpoint_index": int(checkpoint_index),
        "time": float(time_value),
        "method": str(method),
        "method_kind": str(method_kind),
        "energy_total": float(energy),
        "energy_total_exact": float(exact_energy),
        "abs_energy_total_error": float(abs(energy - exact_energy)),
        "fidelity_exact": float(min(1.0, max(0.0, fidelity))),
        "state_norm": float(np.linalg.norm(state)),
    }
    row.update(
        _observable_comparison_fields(
            state=state,
            exact_state=exact_state,
            resolved_problem=resolved_problem,
            num_sites=num_sites,
            ordering=ordering,
            compiled_poly_cache=compiled_poly_cache,
            pauli_action_cache=pauli_action_cache,
        )
    )
    if extra:
        row.update(dict(extra))
    return row


def _trajectory_from_states(
    *,
    times: Sequence[float],
    states: Sequence[np.ndarray],
    exact_states: Sequence[np.ndarray],
    hmat: np.ndarray,
    hmat_sequence: Sequence[np.ndarray] | None = None,
    method: str,
    resolved_problem: Any | None = None,
    num_sites: int | None = None,
    ordering: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    compiled_poly_cache: dict[str, Any] = {}
    pauli_action_cache: dict[str, Any] = {}
    for idx, (time_value, state, exact_state) in enumerate(zip(times, states, exact_states)):
        hmat_i = (
            np.asarray(hmat_sequence[int(idx)], dtype=complex)
            if hmat_sequence is not None and int(idx) < len(hmat_sequence)
            else np.asarray(hmat, dtype=complex)
        )
        energy = _energy_from_matrix(state, hmat_i)
        exact_energy = _energy_from_matrix(exact_state, hmat_i)
        fidelity = float(abs(np.vdot(np.asarray(exact_state, dtype=complex), np.asarray(state, dtype=complex))) ** 2)
        row = {
            "checkpoint_index": int(idx),
            "time": float(time_value),
            "method": str(method),
            "energy_total": float(energy),
            "energy_total_exact": float(exact_energy),
            "abs_energy_total_error": float(abs(energy - exact_energy)),
            "fidelity_exact": float(min(1.0, max(0.0, fidelity))),
            "state_norm": float(np.linalg.norm(state)),
        }
        row.update(
            _observable_comparison_fields(
                state=state,
                exact_state=exact_state,
                resolved_problem=resolved_problem,
                num_sites=num_sites,
                ordering=ordering,
                compiled_poly_cache=compiled_poly_cache,
                pauli_action_cache=pauli_action_cache,
            )
        )
        rows.append(row)
    return rows


def _trajectory_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "final_energy_total": None,
            "final_energy_total_exact": None,
            "final_abs_energy_total_error": None,
            "mean_abs_energy_total_error": None,
            "max_abs_energy_total_error": None,
            "min_fidelity_exact": None,
        }
    final = rows[-1]
    errors = [row.get("abs_energy_total_error") for row in rows]
    fidelities = [row.get("fidelity_exact") for row in rows]
    primary_errors = [row.get("abs_primary_density_error") for row in rows]
    site_errors = [row.get("site_occupations_abs_error_max") for row in rows]
    min_fidelity = _min_or_none(fidelities)
    return {
        "row_count": int(len(rows)),
        "final_energy_total": _float_or_none(final.get("energy_total")),
        "final_energy_total_exact": _float_or_none(final.get("energy_total_exact")),
        "final_abs_energy_total_error": _float_or_none(final.get("abs_energy_total_error")),
        "mean_abs_energy_total_error": _mean_or_none(errors),
        "max_abs_energy_total_error": _max_or_none(errors),
        "min_fidelity_exact": min_fidelity,
        "one_minus_min_fidelity_exact": (
            None if min_fidelity is None else float(max(0.0, 1.0 - min_fidelity))
        ),
        "mean_abs_primary_density_error": _mean_or_none(primary_errors),
        "max_abs_primary_density_error": _max_or_none(primary_errors),
        "max_abs_site_occupations_error": _max_or_none(site_errors),
    }


def _pauli_weight(label_exyz: str) -> int:
    return int(sum(1 for char in str(label_exyz) if char != "e"))


def _rotation_twoq_count(label_exyz: str) -> int:
    return int(2 * max(0, _pauli_weight(label_exyz) - 1))


def _rotation_depth(label_exyz: str) -> int:
    weight = _pauli_weight(label_exyz)
    return int(max(1, 2 * max(0, weight - 1) + 1))


def _sequence_resource_totals(labels: Sequence[str]) -> dict[str, int]:
    label_list = [str(label) for label in labels]
    twoq = int(sum(_rotation_twoq_count(label) for label in label_list))
    depth = int(sum(_rotation_depth(label) for label in label_list))
    return {
        "rotation_count": int(len(label_list)),
        "compiled_count_2q": twoq,
        "compiled_depth_2q": twoq,
        "compiled_depth": depth,
        "compiled_size": int(depth + twoq),
    }


def _full_horizon_resources(
    *,
    per_interval_labels: Sequence[str],
    interval_count: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    per_interval = _sequence_resource_totals(per_interval_labels)
    intervals = int(interval_count)
    full = {
        "resource_policy": NATIVE_RESOURCE_POLICY,
        "state_at_time_scope": "one_product_formula_interval",
        "state_at_time_resource_basis": "deterministic_pauli_rotation_sequence",
        "full_horizon_scope": "all_product_formula_intervals",
        "full_horizon_resource_basis": "serial_interval_sequence_repeated",
        "state_at_time_rotation_count": int(per_interval["rotation_count"]),
        "state_at_time_2q": int(per_interval["compiled_count_2q"]),
        "state_at_time_depth_2q": int(per_interval["compiled_depth_2q"]),
        "state_at_time_depth": int(per_interval["compiled_depth"]),
        "state_at_time_size": int(per_interval["compiled_size"]),
        "compiled_count_2q_total": int(per_interval["compiled_count_2q"] * intervals),
        "compiled_depth_2q_total": int(per_interval["compiled_depth_2q"] * intervals),
        "compiled_depth_total": int(per_interval["compiled_depth"] * intervals),
        "compiled_size_total": int(per_interval["compiled_size"] * intervals),
        "rotation_count_total": int(per_interval["rotation_count"] * intervals),
        "interval_count": int(intervals),
        "compiled_backend_name": "repo_native_statevector_proxy",
    }
    if extra:
        full.update(dict(extra))
    return full


def _compile_audit_from_resources(resources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selected_backend": {
            "backend_name": "repo_native_statevector_proxy",
            "compiled_count_2q": _int_or_none(resources.get("compiled_count_2q_total")),
            "compiled_depth_2q": _int_or_none(resources.get("compiled_depth_2q_total")),
            "compiled_depth": _int_or_none(resources.get("compiled_depth_total")),
            "compiled_size": _int_or_none(resources.get("compiled_size_total")),
        }
    }


def _scaffold_resources_for_layouts(
    *,
    state_layout: Any,
    interval_layouts: Sequence[Any],
    state_scope: str,
    horizon_scope: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state_labels = _layout_rotation_labels(state_layout)
    state_cost = _sequence_resource_totals(state_labels)
    interval_costs = [_sequence_resource_totals(_layout_rotation_labels(layout)) for layout in interval_layouts]
    full = {
        "resource_policy": NATIVE_RESOURCE_POLICY,
        "state_at_time_scope": str(state_scope),
        "state_at_time_resource_basis": "selected_scaffold_pauli_rotations",
        "full_horizon_scope": str(horizon_scope),
        "full_horizon_resource_basis": "serial_state_scaffold_repetitions",
        "state_at_time_rotation_count": int(state_cost["rotation_count"]),
        "state_at_time_2q": int(state_cost["compiled_count_2q"]),
        "state_at_time_depth_2q": int(state_cost["compiled_depth_2q"]),
        "state_at_time_depth": int(state_cost["compiled_depth"]),
        "state_at_time_size": int(state_cost["compiled_size"]),
        "compiled_count_2q_total": int(sum(cost["compiled_count_2q"] for cost in interval_costs)),
        "compiled_depth_2q_total": int(sum(cost["compiled_depth_2q"] for cost in interval_costs)),
        "compiled_depth_total": int(sum(cost["compiled_depth"] for cost in interval_costs)),
        "compiled_size_total": int(sum(cost["compiled_size"] for cost in interval_costs)),
        "rotation_count_total": int(sum(cost["rotation_count"] for cost in interval_costs)),
        "interval_count": int(len(interval_costs)),
        "final_logical_block_count": int(getattr(state_layout, "logical_parameter_count")),
        "final_runtime_parameter_count": int(getattr(state_layout, "runtime_parameter_count")),
        "compiled_backend_name": "repo_native_statevector_proxy",
    }
    if extra:
        full.update(dict(extra))
    return full


def _generic_parameter_manifest(
    *,
    case: DynamicsBenchmarkCase,
    runtime_input: Any,
    algorithm_id: str,
    times: np.ndarray,
    terms: Sequence[Any],
    flow: NativeHamiltonianFlow | None = None,
) -> dict[str, Any]:
    resolved = getattr(runtime_input, "resolved_problem", None)
    request = getattr(resolved, "request", None)
    manifest = {
        "family": str(case.family),
        "table_class": str(case.table_class),
        "source_table_class": str(case.table_class),
        "tuning_class": dynamics_tuning_class(case),
        "tuning_class_source": DYNAMICS_TUNING_CLASS_SOURCE,
        "case_id": str(case.case_id),
        "algorithm_id": str(algorithm_id),
        "artifact_json": str(case.artifact_json),
        "problem_family": str(getattr(resolved, "family_key", case.family)),
        "num_sites": _int_or_none(getattr(request, "num_sites", None)),
        "nq": int(round(math.log2(np.asarray(runtime_input.psi_initial).size))),
        "t_final": float(times[-1]) if len(times) else None,
        "num_times": int(len(times)),
        "active_pauli_terms": int(len(terms)),
        "time_dependence": "driven_hamiltonian" if flow is not None and flow.drive_enabled else "static_hamiltonian_only",
        "drive_included": bool(flow is not None and flow.drive_enabled),
        "exact_reference_policy": "benchmark_exact_reporting_and_offline_row_scoring_only",
        "controller_decisions_modified": False,
        "static_scaffold_scope": "benchmark_point",
        "static_scaffold_source": str(case.artifact_json),
    }
    if flow is not None and flow.drive_enabled:
        drive_model = getattr(flow, "drive_model", None)
        manifest["drive_profile"] = dict(getattr(drive_model, "profile_payload", {}) or {})
        manifest["drive_term_count"] = _int_or_none(getattr(drive_model, "drive_term_count", None))
        manifest["drive_time_sampling"] = str(flow.drive_time_sampling)
        manifest["drive_t0"] = float(flow.drive_t0)
    return manifest


def _native_comparator_command(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pipelines.time_dynamics.tables.generic_dynamics_benchmark",
        "--run-single",
        "--family",
        str(case.family),
        "--case-id",
        str(case.case_id),
        "--algorithm-id",
        str(algorithm_id),
        "--output-dir",
        str(output_dir),
    ]
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    qcfg = metadata.get("qiskit_dynamics", {}) if isinstance(metadata.get("qiskit_dynamics", {}), Mapping) else {}
    mode = qcfg.get("mode", metadata.get("qiskit_dynamics_mode", "off"))
    if mode not in {None, "", "off"} and str(algorithm_id) not in QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS:
        command.extend(["--qiskit-dynamics-mode", str(mode)])
        cap_key_present = "qubit_cap" in qcfg or "qiskit_qubit_cap" in metadata
        cap = qcfg.get("qubit_cap", metadata.get("qiskit_qubit_cap", None))
        if cap_key_present:
            command.extend(["--qiskit-qubit-cap", "none" if cap in {None, "none", "None"} else str(cap)])
        if bool(qcfg.get("export_circuits", metadata.get("qiskit_export_circuits", False))):
            command.append("--qiskit-export-circuits")
    return command


def _row_from_payload(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    payload: Mapping[str, Any],
    artifact_json: Path,
    command: Sequence[str],
) -> DynamicsBenchmarkRow:
    extra_metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), Mapping) else {}
    computed_metrics = {
        **_energy_metrics(payload, algorithm_id=algorithm_id),
        **_observable_metrics(payload, algorithm_id=algorithm_id),
        **_fidelity_metrics(payload, algorithm_id=algorithm_id),
    }
    metrics = _merge_auxiliary_metrics(
        computed=computed_metrics,
        payload_metrics=extra_metrics,
    )
    qiskit_parity = payload.get("qiskit_parity", {})
    if not isinstance(qiskit_parity, Mapping):
        qiskit_parity = {}
    if qiskit_parity:
        metrics.setdefault("qiskit_parity_status", qiskit_parity.get("status"))
        metrics.setdefault("qiskit_parity_passed", qiskit_parity.get("passed"))
        metrics.setdefault("qiskit_parity_mode", qiskit_parity.get("mode"))
        metrics.setdefault("qiskit_max_state_l2", _float_or_none(qiskit_parity.get("max_state_l2")))
        metrics.setdefault("qiskit_max_infidelity", _float_or_none(qiskit_parity.get("max_infidelity")))
        metrics.setdefault("qiskit_max_energy_abs_delta", _float_or_none(qiskit_parity.get("max_energy_abs_delta")))
        metrics.setdefault(
            "qiskit_max_projection_loss_abs_delta",
            _float_or_none(qiskit_parity.get("max_projection_loss_abs_delta")),
        )
    correctness_key = CORRECTNESS_SIDECAR_KEYS.get(str(algorithm_id))
    correctness = payload.get(correctness_key, {}) if correctness_key is not None else {}
    if not isinstance(correctness, Mapping):
        correctness = {}
    if correctness_key is not None and correctness:
        metrics.setdefault("correctness_status", correctness.get("status"))
        metrics.setdefault("correctness_passed", correctness.get("passed"))
        metrics.setdefault("correctness_support_scope", correctness.get("support_scope"))
        metrics.setdefault(f"{correctness_key}_status", correctness.get("status"))
        metrics.setdefault(f"{correctness_key}_passed", correctness.get("passed"))
        metrics.setdefault(f"{correctness_key}_check_count", _int_or_none(correctness.get("check_count")))
    extra_resources = (
        payload.get("resources", {}) if isinstance(payload.get("resources"), Mapping) else {}
    )
    resources = {
        **_compile_resources(payload),
        **dict(extra_resources),
    }
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    runtime_contract = (
        payload.get("runtime_contract", {}) if isinstance(payload.get("runtime_contract"), Mapping) else {}
    )
    protected_provenance = {
        "command": list(command),
        "route_module": "pipelines.time_dynamics.runners.generic_from_adapt_artifact",
        "controller_mode": summary.get("mode"),
        "controller_exact_input_mode": summary.get(
            "controller_exact_input_mode", runtime_contract.get("controller_exact_input_mode")
        ),
        "diagnostic_exact_reference_mode": summary.get(
            "diagnostic_exact_reference_mode", runtime_contract.get("diagnostic_exact_reference_mode")
        ),
        "decision_data_flow": summary.get("decision_data_flow", runtime_contract.get("decision_data_flow")),
        "uses_reference_for_decision": summary.get(
            "uses_reference_for_decision", runtime_contract.get("uses_reference_for_decision")
        ),
        "uses_future_exact_forecast_for_decision": summary.get(
            "uses_future_exact_forecast_for_decision",
            runtime_contract.get("uses_future_exact_forecast_for_decision"),
        ),
        "case_metadata": dict(case.metadata) if isinstance(case.metadata, Mapping) else {},
        **table_lock_provenance_for_case(case),
    }
    protected_provenance.update(
        {
            "benchmark_only": True,
            "runner_module": _runner_module_for_algorithm(algorithm_id),
            "exact_data_policy": "benchmark_exact_reporting_and_diagnostics_only_not_controller_input",
            "controller_decisions_modified": False,
            "exact_reference_controller_inputs": False,
        }
    )
    extra_provenance = (
        payload.get("provenance", {}) if isinstance(payload.get("provenance"), Mapping) else {}
    )
    for leakage_key in (
        "controller_decisions_modified",
        "exact_reference_controller_inputs",
        "uses_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
    ):
        if _boolish(protected_provenance.get(leakage_key, False)) or _boolish(
            extra_provenance.get(leakage_key, False)
        ):
            protected_provenance[leakage_key] = True
        elif leakage_key in protected_provenance or leakage_key in extra_provenance:
            protected_provenance[leakage_key] = False
    provenance = dict(protected_provenance)
    provenance.update(dict(extra_provenance))
    provenance.update(
        {
            key: protected_provenance.get(key)
            for key in _PROTECTED_PROVENANCE_KEYS
            if key in protected_provenance
        }
    )
    if qiskit_parity:
        provenance.update(
            {
                "qiskit_boundary": "pipelines.exact_bench_only",
                "qiskit_parity_schema": qiskit_parity.get("schema"),
                "qiskit_parity_status": qiskit_parity.get("status"),
                "qiskit_parity_passed": qiskit_parity.get("passed"),
                "qiskit_parity_mode": qiskit_parity.get("mode"),
                "qiskit_parity_json": str(Path(artifact_json).with_name("qiskit_parity.json")),
                "qiskit_exact_data_policy": "diagnostic_only_not_decision_input",
                "qiskit_primary_mode": False,
            }
        )
    if correctness_key is not None and correctness:
        correctness_filename = CORRECTNESS_SIDECAR_FILENAMES[str(algorithm_id)]
        provenance.update(
            {
                "correctness_sidecar_schema": correctness.get("schema"),
                "correctness_sidecar_name": correctness_filename,
                "correctness_sidecar_status": correctness.get("status"),
                "correctness_sidecar_passed": correctness.get("passed"),
                "correctness_sidecar_json": str(Path(artifact_json).with_name(correctness_filename)),
                "correctness_exact_data_policy": correctness.get("exact_data_policy"),
                "correctness_support_scope": correctness.get("support_scope"),
                "correctness_primary_mode": False,
            }
        )
    tuning_provenance = (
        payload.get("tuning_provenance", {})
        if isinstance(payload.get("tuning_provenance"), Mapping)
        else {}
    )
    variant_id = provenance.get("ablation_variant") if isinstance(provenance, Mapping) else None
    if tuning_provenance and class_settings_manifest_path(case) is not None:
        tuning_algorithm_id = str(tuning_provenance.get("algorithm_id", algorithm_id))
        tuning_variant_id = tuning_provenance.get("variant_id", variant_id)
        rebuild_payload: dict[str, Any] = {}
        rebuild_source = str(tuning_provenance.get("settings_source", DYNAMICS_LEGACY_MISSING_TUNING_SOURCE))
        if str(tuning_algorithm_id) == "dyn_fixed_mclachlan":
            fixed_integrator_policy = _fixed_mclachlan_integrator_policy_override(case)
            rebuild_payload = {
                "integrator_policy": fixed_integrator_policy
                or PAPER_II_FIXED_MCLACHLAN_INTEGRATOR_POLICY,
                "integrator_policy_override_source": (
                    "case_metadata"
                    if fixed_integrator_policy is not None
                    else PAPER_II_FIXED_MCLACHLAN_INTEGRATOR_SOURCE
                ),
            }
            rebuild_source = (
                DYNAMICS_CASE_METADATA_OVERRIDE_SOURCE
                if fixed_integrator_policy is not None
                else DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE
            )
        tuning_provenance = build_locked_or_default_tuning_provenance(
            case=case,
            algorithm_id=tuning_algorithm_id,
            settings_kind=str(tuning_provenance.get("settings_kind", "benchmark")),
            settings_payload=rebuild_payload,
            settings_source=rebuild_source,
            variant_id=None if tuning_variant_id in {None, ""} else str(tuning_variant_id),
            locked=bool(tuning_provenance.get("class_tuned_result_locked", False)),
        )
    elif not tuning_provenance:
        tuning_provenance = build_locked_or_default_tuning_provenance(
            case=case,
            algorithm_id=algorithm_id,
            settings_kind="benchmark",
            settings_source=DYNAMICS_LEGACY_MISSING_TUNING_SOURCE,
            locked=False,
        )
    provenance.update(dict(tuning_provenance))
    provenance["tuning_provenance"] = dict(tuning_provenance)
    row_contract = payload.get("row_contract", {}) if isinstance(payload.get("row_contract"), Mapping) else {}
    row_status = str(payload.get("status", "completed"))
    if row_status not in {"completed", "skipped_unsupported", "skipped_not_implemented", "skipped_no_runner", "failed"}:
        row_status = "completed"
    if str(algorithm_id) in NATIVE_GENERIC_COMPARATOR_ALGORITHMS:
        default_reason = "generic repo-native comparator completed"
    elif str(algorithm_id) in QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS:
        default_reason = "generic qiskit-community comparator completed"
    else:
        default_reason = "generic neutral realtime route completed"
    row_reason = str(payload.get("reason", default_reason))
    case_metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    qiskit_metadata = (
        case_metadata.get("qiskit_dynamics", {})
        if isinstance(case_metadata.get("qiskit_dynamics", {}), Mapping)
        else {}
    )
    requested_qiskit_mode = str(
        qiskit_metadata.get("mode", case_metadata.get("qiskit_dynamics_mode", qiskit_parity.get("mode", "off")))
    )
    parity_matrix_requires_sidecars = _parity_correctness_sidecars_required_for_case(case)
    qiskit_parity_required = (
        str(algorithm_id) not in QISKIT_COMMUNITY_GENERIC_COMPARATOR_ALGORITHMS
        and (
            requested_qiskit_mode == "parity_required"
            or (
                parity_matrix_requires_sidecars
                and str(algorithm_id) in QISKIT_PARITY_SIDECAR_REQUIRED_ALGORITHMS
            )
        )
    )
    if qiskit_parity_required:
        if not qiskit_parity:
            row_status = "failed"
            row_reason = "qiskit parity required but no parity sidecar was produced"
        elif qiskit_parity.get("passed") is not True:
            row_status = "failed"
            row_reason = f"qiskit parity required but did not pass: {qiskit_parity.get('status')}"
    if correctness_key is not None:
        filename = CORRECTNESS_SIDECAR_FILENAMES[str(algorithm_id)]
        if not correctness:
            if row_status != "failed":
                row_reason = f"{filename} required but no correctness sidecar was produced"
            row_status = "failed"
        elif correctness.get("passed") is not True:
            if row_status != "failed":
                row_reason = f"{filename} required but did not pass: {correctness.get('status')}"
            row_status = "failed"
    return DynamicsBenchmarkRow(
        family=str(case.family),
        table_class=str(case.table_class),
        case_id=str(case.case_id),
        algorithm_id=str(algorithm_id),
        method_label=_METHOD_LABELS[str(algorithm_id)],
        status=row_status,
        reason=row_reason,
        qpu_faithful=bool(row_contract.get("qpu_faithful", False)) and row_status == "completed",
        exact_assisted=bool(row_contract.get("exact_assisted", str(algorithm_id) == "dyn_exact_reference")),
        diagnostic=bool(row_contract.get("diagnostic", True)),
        artifact_json=str(artifact_json),
        metrics=metrics,
        resources=resources,
        provenance=provenance,
        table_fields=_table_fields(algorithm_id=algorithm_id, metrics=metrics, resources=resources),
    )


def write_dynamics_row_bundle(
    *,
    row: DynamicsBenchmarkRow,
    output_dir: Path,
    raw_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    row_payload = row.to_dict()
    validate_dynamics_metric_contract(row_payload, strict=True)
    rows = [row_payload]
    summary = dynamics_table_bundle_payload(rows=rows)
    summary["paths"] = {
        "rows_json": str(root / "rows.json"),
        "summary_json": str(root / "summary.json"),
        "result_json": str(root / "result.json"),
    }
    if raw_payload is not None:
        summary["paths"]["raw_payload_json"] = str(root / "raw_payload.json")
        if not (root / "raw_payload.json").exists():
            _write_json(root / "raw_payload.json", raw_payload)
        qiskit_parity = raw_payload.get("qiskit_parity") if isinstance(raw_payload, Mapping) else None
        if isinstance(qiskit_parity, Mapping):
            summary["paths"]["qiskit_parity_json"] = str(root / "qiskit_parity.json")
            _write_json(root / "qiskit_parity.json", qiskit_parity)
        for algorithm_id, sidecar_key in CORRECTNESS_SIDECAR_KEYS.items():
            sidecar = raw_payload.get(sidecar_key)
            if isinstance(sidecar, Mapping):
                filename = CORRECTNESS_SIDECAR_FILENAMES[algorithm_id]
                summary["paths"][f"{sidecar_key}_json"] = str(root / filename)
                _write_json(root / filename, sidecar)
    _write_json(root / "rows.json", rows)
    _write_json(root / "summary.json", summary)
    _write_json(root / "result.json", row_payload)
    return {"row": row_payload, "summary": summary, "paths": dict(summary["paths"])}


def skipped_generic_dynamics_row(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    status: str,
    reason: str,
) -> DynamicsBenchmarkRow:
    tuning = build_dynamics_tuning_provenance(
        case=case,
        algorithm_id=algorithm_id,
        settings_kind="skip",
        settings_source=DYNAMICS_SKIPPED_TUNING_SOURCE,
        locked=False,
    )
    return DynamicsBenchmarkRow(
        family=str(case.family),
        table_class=str(case.table_class),
        case_id=str(case.case_id),
        algorithm_id=str(algorithm_id),
        method_label=_METHOD_LABELS.get(str(algorithm_id), str(algorithm_id)),
        status=str(status),
        reason=str(reason),
        qpu_faithful=False,
        exact_assisted=False,
        diagnostic=True,
        artifact_json=None,
        metrics={},
        resources={},
        provenance={
            "skip_reason": str(reason),
            "benchmark_only": True,
            "runner_module": _runner_module_for_algorithm(algorithm_id),
            "controller_decisions_modified": False,
            "exact_reference_controller_inputs": False,
            "case_metadata": dict(case.metadata) if isinstance(case.metadata, Mapping) else {},
            **dict(tuning),
            "tuning_provenance": dict(tuning),
        },
        table_fields=DynamicsTableFields(table_status_label=str(status)),
    )


def write_skipped_generic_dynamics_row(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    output_dir: Path,
    status: str,
    reason: str,
) -> DynamicsBenchmarkRow:
    row = skipped_generic_dynamics_row(
        case=case,
        algorithm_id=algorithm_id,
        status=status,
        reason=reason,
    )
    bundle = write_dynamics_row_bundle(row=row, output_dir=Path(output_dir))
    skip_payload = dict(bundle["row"])
    skip_payload["schema"] = "generic_dynamics_benchmark_skip_v1"
    _write_json(Path(output_dir) / "skip.json", skip_payload)
    return row
