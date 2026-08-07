#!/usr/bin/env python3
"""Collapsed legacy repo-native dynamics benchmark implementations.

This file intentionally collects the old repo-native benchmark implementations
so the top-level benchmark package exposes one legacy-native surface next to the
Paper-II Qiskit-native comparator surface.  Do not add new Paper-II Qiskit
comparators here; use ``qiskit_native.py`` for those.
"""

from __future__ import annotations



# ---- exact_reference.py (collapsed legacy implementation) ----
"""Exact/Krylov reference generic dynamics benchmark wrapper."""


from pathlib import Path

from pipelines.time_dynamics.benchmarks import common
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DynamicsBenchmarkCase, DynamicsBenchmarkRow

EXACT_REFERENCE_ALGORITHM_ID = "dyn_exact_reference"


def run_exact_reference_benchmark_row(*, case: DynamicsBenchmarkCase, output_dir: Path) -> DynamicsBenchmarkRow:
    return common.run_realtime_generic_dynamics_row(
        case=case,
        algorithm_id=EXACT_REFERENCE_ALGORITHM_ID,
        output_dir=Path(output_dir),
    )




# ---- fixed_mclachlan.py (collapsed legacy implementation) ----
"""Fixed-scaffold McLachlan generic dynamics benchmark wrapper."""


from pathlib import Path

from pipelines.time_dynamics.benchmarks import common
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DynamicsBenchmarkCase, DynamicsBenchmarkRow

FIXED_MCLACHLAN_ALGORITHM_ID = "dyn_fixed_mclachlan"


def run_fixed_mclachlan_benchmark_row(*, case: DynamicsBenchmarkCase, output_dir: Path) -> DynamicsBenchmarkRow:
    return common.run_realtime_generic_dynamics_row(
        case=case,
        algorithm_id=FIXED_MCLACHLAN_ALGORITHM_ID,
        output_dir=Path(output_dir),
    )




# ---- product_formula.py (collapsed legacy implementation) ----
"""Repo-native product-formula/Suzuki generic dynamics benchmark."""


from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.benchmarks import common
from pipelines.time_dynamics.benchmarks.common import (
    _apply_product_formula_step,
    _compile_audit_from_resources,
    _compiled_pauli_actions_by_label,
    _float_or_none,
    _generic_parameter_manifest,
    _int_or_none,
    _metadata_int,
    _native_hamiltonian_flow,
    _normalize_state,
    _product_formula_sequence,
    _sequence_resource_totals,
    _trajectory_from_states,
    _trajectory_summary,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DynamicsBenchmarkCase,
    DynamicsBenchmarkRow,
    build_dynamics_tuning_provenance,
    json_safe,
)

PRODUCT_FORMULA_CANDIDATE_ORDERS: tuple[int, ...] = (1, 2)


def _simulate_product_formula_candidate(
    *,
    flow: common.NativeHamiltonianFlow,
    psi_initial: np.ndarray,
    times: np.ndarray,
    order: int,
    include_states: bool = False,
) -> dict[str, Any]:
    psi = _normalize_state(psi_initial)
    states = [np.asarray(psi, dtype=complex)]
    action_by_label: dict[str, Any] = {}
    all_interval_labels: list[str] = []
    first_interval_labels: list[str] | None = None
    for left, right in zip(times[:-1], times[1:]):
        dt = float(right - left)
        terms_step = flow.terms_for_interval(float(left), float(right))
        for label, action in _compiled_pauli_actions_by_label(terms_step).items():
            action_by_label.setdefault(str(label), action)
        psi = _apply_product_formula_step(
            terms=terms_step,
            action_by_label=action_by_label,
            psi=psi,
            dt=float(dt),
            order=int(order),
        )
        interval_labels = [
            str(term.pauli_exyz)
            for term, _factor in _product_formula_sequence(terms_step, order=int(order))
        ]
        if first_interval_labels is None:
            first_interval_labels = list(interval_labels)
        all_interval_labels.extend(interval_labels)
        states.append(np.asarray(psi, dtype=complex))
    method = f"suzuki{int(order)}"
    trajectory = _trajectory_from_states(
        times=times,
        states=states,
        exact_states=flow.exact_states,
        hmat=flow.static_hmat,
        hmat_sequence=flow.hmat_sequence_for_trajectory_samples(),
        method=method,
        **dict(flow.observable_context or {}),
    )
    state_cost = _sequence_resource_totals(first_interval_labels or [])
    full_cost = _sequence_resource_totals(all_interval_labels)
    resources = {
        "resource_policy": common.NATIVE_RESOURCE_POLICY,
        "state_at_time_scope": "one_product_formula_interval",
        "state_at_time_resource_basis": "deterministic_pauli_rotation_sequence_interval0",
        "full_horizon_scope": "all_product_formula_intervals",
        "full_horizon_resource_basis": "serial_interval_sequence_actual_driven_terms",
        "state_at_time_rotation_count": int(state_cost["rotation_count"]),
        "state_at_time_2q": int(state_cost["compiled_count_2q"]),
        "state_at_time_depth_2q": int(state_cost["compiled_depth_2q"]),
        "state_at_time_depth": int(state_cost["compiled_depth"]),
        "state_at_time_size": int(state_cost["compiled_size"]),
        "compiled_count_2q_total": int(full_cost["compiled_count_2q"]),
        "compiled_depth_2q_total": int(full_cost["compiled_depth_2q"]),
        "compiled_depth_total": int(full_cost["compiled_depth"]),
        "compiled_size_total": int(full_cost["compiled_size"]),
        "rotation_count_total": int(full_cost["rotation_count"]),
        "interval_count": max(0, int(len(times) - 1)),
        "compiled_backend_name": "repo_native_statevector_proxy",
        "order": int(order),
        "method": method,
        "drive_included": bool(flow.drive_enabled),
    }
    payload: dict[str, Any] = {
        "method": method,
        "order": int(order),
        "trajectory": trajectory,
        "summary": _trajectory_summary(trajectory),
        "resources": resources,
    }
    if include_states:
        payload["_states"] = tuple(np.asarray(state, dtype=complex) for state in states)
    return payload


def _selection_value(row: Mapping[str, Any], field: str) -> float:
    value = _float_or_none(row.get(field))
    if value is None:
        raise ValueError(f"candidate {row.get('method')} missing finite selection field {field!r}")
    return float(value)


def _product_formula_selection_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    summary = candidate.get("summary", {}) if isinstance(candidate.get("summary"), Mapping) else {}
    resources = candidate.get("resources", {}) if isinstance(candidate.get("resources"), Mapping) else {}
    return (
        _selection_value(summary, "mean_abs_energy_total_error"),
        _selection_value(summary, "max_abs_energy_total_error"),
        _selection_value(summary, "final_abs_energy_total_error"),
        float(_int_or_none(resources.get("compiled_count_2q_total")) or 0),
        int(candidate.get("order", 0)),
    )


def _selected_product_formula_order(case: DynamicsBenchmarkCase) -> int:
    order = _metadata_int(
        case,
        "product_formula_order",
        max(PRODUCT_FORMULA_CANDIDATE_ORDERS),
        minimum=1,
    )
    if int(order) not in PRODUCT_FORMULA_CANDIDATE_ORDERS:
        raise ValueError(
            f"product_formula_order={order!r} is not in supported orders "
            f"{PRODUCT_FORMULA_CANDIDATE_ORDERS!r}"
        )
    return int(order)


def _build_product_formula_payload(
    *,
    case: DynamicsBenchmarkCase,
    runtime_input: Any,
    command: Sequence[str],
) -> dict[str, Any]:
    flow = _native_hamiltonian_flow(case, runtime_input)
    terms = flow.terms_for_interval(float(flow.times[0]), float(flow.times[min(1, len(flow.times) - 1)]))
    hmat = np.asarray(flow.static_hmat, dtype=complex)
    psi_initial = _normalize_state(runtime_input.psi_initial)
    if hmat.shape[0] != psi_initial.size:
        raise ValueError(
            f"Hamiltonian dimension {hmat.shape} does not match state size {psi_initial.size}"
        )
    times = flow.times
    selected_order = _selected_product_formula_order(case)
    candidates = [
        _simulate_product_formula_candidate(
            flow=flow,
            psi_initial=psi_initial,
            times=times,
            order=order,
            include_states=True,
        )
        for order in PRODUCT_FORMULA_CANDIDATE_ORDERS
    ]
    selected_private = dict(next(candidate for candidate in candidates if int(candidate["order"]) == int(selected_order)))
    selected_states = tuple(selected_private.get("_states", ()))
    qiskit_parity = None
    adapter = common._qiskit_dynamics_adapter()
    qiskit_config = adapter.qiskit_dynamics_config_from_case(case)
    if adapter.parity_requested(qiskit_config):
        qiskit_parity = adapter.product_formula_parity_result(
            config=qiskit_config,
            case=case,
            flow=flow,
            initial_state=psi_initial,
            times=times,
            order=int(selected_order),
            native_states=selected_states,
        )
    public_candidates = [
        {key: value for key, value in dict(candidate).items() if not str(key).startswith("_")}
        for candidate in candidates
    ]
    selected = {key: value for key, value in selected_private.items() if not str(key).startswith("_")}
    summary = dict(selected["summary"])
    resources = dict(selected["resources"])
    metrics = {
        "method_kind": "product_formula_envelope",
        "candidate_orders": list(PRODUCT_FORMULA_CANDIDATE_ORDERS),
        "selected_order": int(selected["order"]),
        "selected_method": str(selected["method"]),
        "selection_metric": "metadata_product_formula_order",
        "selection_uses_exact_reference": False,
        "selection_policy": "qpu_faithful_fixed_order_from_case_metadata",
        "candidate_count": int(len(candidates)),
        "exact_fields_reporting_only": True,
    }
    tuning = build_dynamics_tuning_provenance(
        case=case,
        algorithm_id=PRODUCT_FORMULA_ALGORITHM_ID,
        settings_kind="comparator",
        settings_payload={
            "candidate_orders": list(PRODUCT_FORMULA_CANDIDATE_ORDERS),
            "selected_order": int(selected["order"]),
            "selection_policy": "qpu_faithful_fixed_order_from_case_metadata",
        },
        settings_source=common.metadata_override_settings_source(
            case,
            ("product_formula_order",),
        ),
        locked=False,
    )
    parameter_manifest = _generic_parameter_manifest(
        case=case,
        runtime_input=runtime_input,
        algorithm_id=PRODUCT_FORMULA_ALGORITHM_ID,
        times=times,
        terms=terms,
        flow=flow,
    )
    parameter_manifest["tuning_provenance"] = dict(tuning)
    return json_safe(
        {
            "schema_version": "generic_product_formula_envelope_benchmark_v1",
            "case": case.to_dict(),
            "row_contract": {
                "qpu_faithful": True,
                "exact_assisted": False,
                "diagnostic": True,
            },
            "parameter_manifest": parameter_manifest,
            "tuning_provenance": tuning,
            "command": list(command),
            "methods": {str(candidate["method"]): candidate for candidate in public_candidates},
            "candidate_rows": public_candidates,
            "selected_candidate": selected,
            "qiskit_parity": qiskit_parity,
            "trajectory": selected["trajectory"],
            "summary": summary,
            "metrics": metrics,
            "resources": resources,
            "compile_audit": _compile_audit_from_resources(resources),
            "provenance": {
                "route_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "benchmark_only": True,
                "runner_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_benchmark_selection",
                "comparator_kernel": "repo_native_suzuki_product_formula",
                "decision_data_flow": "ideal_product_formula_circuit_state",
                "controller_paths_called": False,
                "controller_decisions_modified": False,
                "exact_reference_controller_inputs": False,
                "selection_uses_exact_reference": False,
                "exact_fields_reporting_only": True,
                "qiskit_boundary": "pipelines.exact_bench_only" if qiskit_parity else "not_requested",
            },
        }
    )



PRODUCT_FORMULA_ALGORITHM_ID = "dyn_product_formula_envelope"


def run_product_formula_benchmark_row(*, case: DynamicsBenchmarkCase, output_dir: Path) -> DynamicsBenchmarkRow:
    return common.run_native_generic_comparator_row(
        case=case,
        algorithm_id=PRODUCT_FORMULA_ALGORITHM_ID,
        output_dir=Path(output_dir),
        payload_builder=_build_product_formula_payload,
    )




# ---- qdrift.py (collapsed legacy implementation) ----
"""Repo-native qDRIFT generic dynamics benchmark."""


from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.benchmarks import common
from pipelines.time_dynamics.benchmarks.common import (
    _compile_audit_from_resources,
    _generic_parameter_manifest,
    _max_or_none,
    _mean_or_none,
    _metadata_int,
    _native_hamiltonian_flow,
    _normalize_state,
    _sequence_resource_totals,
    _trajectory_from_states,
    _trajectory_summary,
    NATIVE_RESOURCE_POLICY,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DynamicsBenchmarkCase,
    DynamicsBenchmarkRow,
    build_dynamics_tuning_provenance,
    json_safe,
)
from src.quantum.pauli_actions import apply_exp_term, compile_pauli_action_exyz

DEFAULT_QDRIFT_SAMPLES_PER_INTERVAL = 16
DEFAULT_QDRIFT_RNG_SEED = 7


def _simulate_qdrift(
    *,
    flow: common.NativeHamiltonianFlow,
    psi_initial: np.ndarray,
    times: np.ndarray,
    samples_per_interval: int,
    rng_seed: int,
    include_states: bool = False,
) -> dict[str, Any]:
    if int(samples_per_interval) <= 0:
        raise ValueError("qDRIFT samples_per_interval must be positive")
    rng = np.random.default_rng(int(rng_seed))
    psi = _normalize_state(psi_initial)
    states = [np.asarray(psi, dtype=complex)]
    intervals: list[dict[str, Any]] = []
    sampled_labels_full: list[str] = []
    action_by_label: dict[str, Any] = {}
    for interval_index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        dt = float(right - left)
        terms_step = flow.terms_for_interval(float(left), float(right))
        labels = [str(term.pauli_exyz) for term in terms_step]
        coeffs = [float(term.coeff_real) for term in terms_step]
        for term in terms_step:
            label = str(term.pauli_exyz)
            if label not in action_by_label:
                action_by_label[label] = compile_pauli_action_exyz(label, int(term.nq))
        lam = float(sum(abs(coeff) for coeff in coeffs))
        sampled_labels: list[str] = []
        sampled_signs: list[float] = []
        tau = 0.0 if lam <= 1.0e-15 else float(lam) * float(dt) / float(samples_per_interval)
        if lam > 1.0e-15 and labels:
            probabilities = np.asarray([abs(coeff) / lam for coeff in coeffs], dtype=float)
            sampled_indices = rng.choice(
                len(labels),
                size=int(samples_per_interval),
                replace=True,
                p=probabilities,
            )
            for raw_idx in sampled_indices:
                idx = int(raw_idx)
                label = labels[idx]
                sign = 1.0 if float(coeffs[idx]) >= 0.0 else -1.0
                sampled_labels.append(label)
                sampled_signs.append(float(sign))
                sampled_labels_full.append(label)
                psi = apply_exp_term(psi, action_by_label[label], complex(sign), float(tau))
        else:
            probabilities = np.zeros(len(labels), dtype=float)
        psi = _normalize_state(psi)
        intervals.append(
            {
                "interval_index": int(interval_index),
                "time_start": float(left),
                "time_stop": float(right),
                "dt": float(dt),
                "lambda": float(lam),
                "tau": float(tau),
                "samples_per_interval": int(samples_per_interval),
                "sampled_labels": sampled_labels,
                "sampled_signs": sampled_signs,
                "probabilities": [float(x) for x in probabilities.tolist()],
            }
        )
        states.append(np.asarray(psi, dtype=complex))
    trajectory = _trajectory_from_states(
        times=times,
        states=states,
        exact_states=flow.exact_states,
        hmat=flow.static_hmat,
        hmat_sequence=flow.hmat_sequence_for_trajectory_samples(),
        method="qdrift",
        **dict(flow.observable_context or {}),
    )
    first_interval_labels = (
        intervals[0]["sampled_labels"]
        if intervals
        else [label for label in labels for _ in range(int(samples_per_interval))]
    )
    state_cost = _sequence_resource_totals(first_interval_labels)
    full_cost = _sequence_resource_totals(sampled_labels_full)
    resources = {
        "resource_policy": NATIVE_RESOURCE_POLICY,
        "randomization_family": "qdrift",
        "samples_per_interval": int(samples_per_interval),
        "rng_seed": int(rng_seed),
        "state_at_time_scope": "representative_interval0_qdrift_sample",
        "state_at_time_resource_basis": "first_interval_sampled_labels",
        "full_horizon_scope": "all_sampled_qdrift_microsteps",
        "full_horizon_resource_basis": "realized_sampled_labels_all_intervals",
        "state_at_time_rotation_count": int(state_cost["rotation_count"]),
        "state_at_time_2q": int(state_cost["compiled_count_2q"]),
        "state_at_time_depth_2q": int(state_cost["compiled_depth_2q"]),
        "state_at_time_depth": int(state_cost["compiled_depth"]),
        "state_at_time_size": int(state_cost["compiled_size"]),
        "compiled_count_2q_total": int(full_cost["compiled_count_2q"]),
        "compiled_depth_2q_total": int(full_cost["compiled_depth_2q"]),
        "compiled_depth_total": int(full_cost["compiled_depth"]),
        "compiled_size_total": int(full_cost["compiled_size"]),
        "rotation_count_total": int(full_cost["rotation_count"]),
        "interval_count": max(0, int(len(times) - 1)),
        "sampled_rotation_count": int(len(sampled_labels_full)),
        "lambda_mean": _mean_or_none([row.get("lambda") for row in intervals]),
        "lambda_max": _max_or_none([row.get("lambda") for row in intervals]),
        "compiled_backend_name": "repo_native_statevector_proxy",
        "drive_included": bool(flow.drive_enabled),
    }
    payload: dict[str, Any] = {
        "method": "qdrift",
        "trajectory": trajectory,
        "summary": _trajectory_summary(trajectory),
        "resources": resources,
        "qdrift_intervals": intervals,
    }
    if include_states:
        payload["_states"] = tuple(np.asarray(state, dtype=complex) for state in states)
    return payload


def _build_qdrift_payload(
    *,
    case: DynamicsBenchmarkCase,
    runtime_input: Any,
    command: Sequence[str],
) -> dict[str, Any]:
    flow = _native_hamiltonian_flow(case, runtime_input)
    terms = flow.terms_for_interval(float(flow.times[0]), float(flow.times[min(1, len(flow.times) - 1)]))
    hmat = np.asarray(flow.static_hmat, dtype=complex)
    psi_initial = _normalize_state(runtime_input.psi_initial)
    if hmat.shape[0] != psi_initial.size:
        raise ValueError(
            f"Hamiltonian dimension {hmat.shape} does not match state size {psi_initial.size}"
        )
    times = flow.times
    samples_per_interval = _metadata_int(
        case,
        "qdrift_samples_per_interval",
        DEFAULT_QDRIFT_SAMPLES_PER_INTERVAL,
    )
    rng_seed = _metadata_int(case, "qdrift_rng_seed", DEFAULT_QDRIFT_RNG_SEED, minimum=0)
    simulation = _simulate_qdrift(
        flow=flow,
        psi_initial=psi_initial,
        times=times,
        samples_per_interval=samples_per_interval,
        rng_seed=rng_seed,
        include_states=True,
    )
    adapter = common._qiskit_dynamics_adapter()
    qiskit_config = adapter.qiskit_dynamics_config_from_case(case)
    qiskit_parity = None
    if adapter.parity_requested(qiskit_config):
        qiskit_parity = adapter.qdrift_parity_result(
            config=qiskit_config,
            case=case,
            initial_state=psi_initial,
            intervals=simulation["qdrift_intervals"],
            native_states=tuple(simulation.get("_states", ())),
            hmat_sequence=flow.hmat_sequence_for_trajectory_samples(),
        )
    simulation_public = {key: value for key, value in dict(simulation).items() if not str(key).startswith("_")}
    resources = dict(simulation_public["resources"])
    metrics = {
        "method_kind": "randomized_product_formula",
        "randomization_family": "qdrift",
        "samples_per_interval": int(samples_per_interval),
        "rng_seed": int(rng_seed),
        "sampled_rotation_count": int(resources.get("sampled_rotation_count", 0)),
        "lambda_mean": resources.get("lambda_mean"),
        "lambda_max": resources.get("lambda_max"),
        "exact_fields_reporting_only": True,
    }
    tuning = build_dynamics_tuning_provenance(
        case=case,
        algorithm_id=QDRIFT_ALGORITHM_ID,
        settings_kind="comparator",
        settings_payload={
            "samples_per_interval": int(samples_per_interval),
            "rng_seed": int(rng_seed),
            "randomization_family": "qdrift",
        },
        settings_source=common.metadata_override_settings_source(
            case,
            ("qdrift_samples_per_interval", "qdrift_rng_seed"),
        ),
        locked=False,
    )
    parameter_manifest = _generic_parameter_manifest(
        case=case,
        runtime_input=runtime_input,
        algorithm_id=QDRIFT_ALGORITHM_ID,
        times=times,
        terms=terms,
        flow=flow,
    )
    parameter_manifest["tuning_provenance"] = dict(tuning)
    return json_safe(
        {
            "schema_version": "generic_qdrift_benchmark_v1",
            "case": case.to_dict(),
            "row_contract": {
                "qpu_faithful": True,
                "exact_assisted": False,
                "diagnostic": True,
            },
            "parameter_manifest": parameter_manifest,
            "tuning_provenance": tuning,
            "command": list(command),
            "trajectory": simulation_public["trajectory"],
            "summary": simulation_public["summary"],
            "qdrift_intervals": simulation_public["qdrift_intervals"],
            "qiskit_parity": qiskit_parity,
            "metrics": metrics,
            "resources": resources,
            "compile_audit": _compile_audit_from_resources(resources),
            "provenance": {
                "route_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "benchmark_only": True,
                "runner_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_benchmark_selection",
                "comparator_kernel": "repo_native_qdrift",
                "decision_data_flow": "ideal_randomized_product_formula_circuit_state",
                "controller_paths_called": False,
                "controller_decisions_modified": False,
                "exact_reference_controller_inputs": False,
                "exact_fields_reporting_only": True,
                "qdrift_sampling_depends_on_exact_fields": False,
                "qiskit_boundary": "pipelines.exact_bench_only" if qiskit_parity else "not_requested",
            },
        }
    )



QDRIFT_ALGORITHM_ID = "dyn_qdrift"


def run_qdrift_benchmark_row(*, case: DynamicsBenchmarkCase, output_dir: Path) -> DynamicsBenchmarkRow:
    return common.run_native_generic_comparator_row(
        case=case,
        algorithm_id=QDRIFT_ALGORITHM_ID,
        output_dir=Path(output_dir),
        payload_builder=_build_qdrift_payload,
    )




# ---- fixed_pvqd.py (collapsed legacy implementation) ----
"""Repo-native fixed-pVQD generic dynamics benchmark."""


from pathlib import Path
from typing import Any, Sequence
import math

import numpy as np

from pipelines.time_dynamics.benchmarks import common
from pipelines.time_dynamics.benchmarks.common import (
    _apply_product_formula_step,
    _compile_audit_from_resources,
    _compiled_pauli_actions_by_label,
    _generic_parameter_manifest,
    _metadata_float,
    _metadata_int,
    _native_hamiltonian_flow,
    _normalize_state,
    _prepare_scaffold_state,
    _runtime_variational_bundle,
    _scaffold_resources_for_layouts,
    _state_diagnostic_row,
    _trajectory_summary,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DynamicsBenchmarkCase,
    DynamicsBenchmarkRow,
    build_dynamics_tuning_provenance,
    json_safe,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor


def _projection_loss_for_state(psi_trial: np.ndarray, target_state: np.ndarray) -> tuple[float, float]:
    trial = _normalize_state(psi_trial)
    target = _normalize_state(target_state)
    overlap = float(abs(np.vdot(target, trial)) ** 2)
    if not math.isfinite(overlap):
        return float(1.0e12), 0.0
    overlap = min(1.0, max(0.0, float(overlap)))
    return float(max(0.0, 1.0 - overlap)), float(overlap)


def _coordinate_refine_projection(
    objective: Any,
    theta_init: np.ndarray,
    *,
    initial_loss: float,
    maxiter: int,
    overlap_tol: float,
) -> tuple[np.ndarray, float, str, int]:
    theta_best = np.asarray(theta_init, dtype=float).reshape(-1).copy()
    best = float(initial_loss)
    step = 0.1
    sweeps = 0
    for sweep in range(max(1, int(maxiter))):
        sweeps = int(sweep) + 1
        improved = False
        for idx in range(int(theta_best.size)):
            for sign in (-1.0, 1.0):
                candidate = theta_best.copy()
                candidate[int(idx)] += float(sign) * float(step)
                loss = float(objective(candidate))
                if loss + 1.0e-15 < best:
                    theta_best = candidate
                    best = float(loss)
                    improved = True
        if best <= float(overlap_tol):
            return theta_best, best, "coordinate_refine_overlap_tol", sweeps
        if not improved:
            step *= 0.5
            if step < 1.0e-8:
                break
    return theta_best, best, "coordinate_refine_done", sweeps


def _fit_pvqd_projection_step(
    *,
    executor: CompiledAnsatzExecutor,
    psi_ref: np.ndarray,
    theta_start: np.ndarray,
    target_state: np.ndarray,
    maxiter: int,
    overlap_tol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    theta_init = np.asarray(theta_start, dtype=float).reshape(-1)
    target = _normalize_state(target_state)
    eval_count = 0

    def objective(theta_vec: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        try:
            psi_trial = _prepare_scaffold_state(executor, psi_ref, theta_vec)
            loss, _overlap = _projection_loss_for_state(psi_trial, target)
        except Exception:
            return float(1.0e12)
        return float(loss) if math.isfinite(float(loss)) else float(1.0e12)

    initial_loss = float(objective(theta_init))
    initial_overlap = float(max(0.0, min(1.0, 1.0 - initial_loss)))
    if initial_loss <= float(overlap_tol):
        return theta_init, {
            "initial_projection_loss": float(initial_loss),
            "final_projection_loss": float(initial_loss),
            "initial_overlap": float(initial_overlap),
            "final_overlap": float(initial_overlap),
            "nfev": int(eval_count),
            "nit": 0,
            "success": True,
            "status": "skipped_overlap_tol",
            "message": "warm start already satisfies overlap tolerance",
        }

    theta_best, final_loss, status, nit = _coordinate_refine_projection(
        objective,
        theta_init,
        initial_loss=float(initial_loss),
        maxiter=int(maxiter),
        overlap_tol=float(overlap_tol),
    )
    final_overlap = float(max(0.0, min(1.0, 1.0 - final_loss)))
    return np.asarray(theta_best, dtype=float).reshape(-1), {
        "initial_projection_loss": float(initial_loss),
        "final_projection_loss": float(final_loss),
        "initial_overlap": float(initial_overlap),
        "final_overlap": float(final_overlap),
        "nfev": int(eval_count),
        "nit": int(nit),
        "success": bool(final_loss <= initial_loss),
        "status": str(status),
        "message": "repo-native coordinate-refine projection fit",
    }


def _build_fixed_pvqd_payload(
    *,
    case: DynamicsBenchmarkCase,
    runtime_input: Any,
    command: Sequence[str],
) -> dict[str, Any]:
    flow = _native_hamiltonian_flow(case, runtime_input)
    terms = flow.terms_for_interval(float(flow.times[0]), float(flow.times[min(1, len(flow.times) - 1)]))
    hmat = np.asarray(flow.static_hmat, dtype=complex)
    psi_initial = _normalize_state(runtime_input.psi_initial)
    if hmat.shape[0] != psi_initial.size:
        raise ValueError(
            f"Hamiltonian dimension {hmat.shape} does not match state size {psi_initial.size}"
        )
    scaffold_terms, layout, theta, psi_ref, executor, _drive_aligned_scaffold = _runtime_variational_bundle(runtime_input)
    times = flow.times
    if int(times.size) < 2:
        raise ValueError("fixed pVQD comparator requires at least two time points")
    exact_states = flow.exact_states
    observable_context = dict(flow.observable_context or {})
    maxiter = _metadata_int(case, "pvqd_optimizer_maxiter", 24, minimum=1)
    overlap_tol = _metadata_float(case, "pvqd_overlap_tol", 1.0e-8, minimum=0.0)
    target_order = _metadata_int(case, "pvqd_target_product_formula_order", 2, minimum=1)
    action_by_label: dict[str, Any] = {}
    qiskit_component_inputs: list[dict[str, Any]] = []

    theta_current = np.asarray(theta, dtype=float).reshape(-1)
    current_state = _prepare_scaffold_state(executor, psi_ref, theta_current)
    trajectory: list[dict[str, Any]] = [
        _state_diagnostic_row(
            checkpoint_index=0,
            time_value=float(times[0]),
            method="generic_fixed_pvqd",
            method_kind="fixed_pvqd",
            state=current_state,
            exact_state=exact_states[0],
            hmat=flow.hmat_at_time(float(times[0])),
            **observable_context,
            extra={
                "runtime_parameter_count": int(theta_current.size),
                "logical_block_count": int(getattr(layout, "logical_parameter_count")),
                "pvqd_step_index": None,
                "pvqd_nfev": 0,
                "projection_loss_initial": None,
                "projection_loss_final": None,
                "projection_overlap_initial": None,
                "projection_overlap_final": None,
                "optimizer_status": None,
                "optimizer_success": None,
            },
        )
    ]
    pvqd_steps: list[dict[str, Any]] = []
    for interval_index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        dt = float(right - left)
        terms_step = flow.terms_for_interval(float(left), float(right))
        theta_start = np.asarray(theta_current, dtype=float).reshape(-1).copy()
        start_state = np.asarray(current_state, dtype=complex).reshape(-1).copy()
        for label, action in _compiled_pauli_actions_by_label(terms_step).items():
            action_by_label.setdefault(str(label), action)
        target = _apply_product_formula_step(
            terms=terms_step,
            action_by_label=action_by_label,
            psi=current_state,
            dt=float(dt),
            order=int(target_order),
        )
        theta_next, fit = _fit_pvqd_projection_step(
            executor=executor,
            psi_ref=psi_ref,
            theta_start=theta_current,
            target_state=target,
            maxiter=maxiter,
            overlap_tol=overlap_tol,
        )
        theta_current = np.asarray(theta_next, dtype=float).reshape(-1)
        current_state = _prepare_scaffold_state(executor, psi_ref, theta_current)
        qiskit_component_inputs.append(
            {
                "interval_index": int(interval_index),
                "psi_ref": np.asarray(psi_ref, dtype=complex).reshape(-1).copy(),
                "start_layout": layout,
                "final_layout": layout,
                "theta_start": theta_start,
                "theta_final": np.asarray(theta_current, dtype=float).reshape(-1).copy(),
                "native_start_state": start_state,
                "native_target_state": np.asarray(target, dtype=complex).reshape(-1).copy(),
                "native_final_state": np.asarray(current_state, dtype=complex).reshape(-1).copy(),
                "target_terms": tuple(terms_step),
                "dt": float(dt),
                "target_order": int(target_order),
                "fit": dict(fit),
            }
        )
        step_payload = {
            "interval_index": int(interval_index),
            "time_start": float(left),
            "time_stop": float(right),
            "dt": float(dt),
            "optimizer_method": "coordinate_refine",
            "optimizer_maxiter": int(maxiter),
            "overlap_tol": float(overlap_tol),
            "target_policy": "product_formula_circuit_step",
            "target_product_formula_order": int(target_order),
            **fit,
        }
        pvqd_steps.append(step_payload)
        trajectory.append(
            _state_diagnostic_row(
                checkpoint_index=int(interval_index) + 1,
                time_value=float(right),
                method="generic_fixed_pvqd",
                method_kind="fixed_pvqd",
                state=current_state,
                exact_state=exact_states[int(interval_index) + 1],
                hmat=flow.hmat_at_time(float(right)),
                **observable_context,
                extra={
                    "runtime_parameter_count": int(theta_current.size),
                    "logical_block_count": int(getattr(layout, "logical_parameter_count")),
                    "pvqd_step_index": int(interval_index),
                    "pvqd_nfev": int(fit["nfev"]),
                    "projection_loss_initial": float(fit["initial_projection_loss"]),
                    "projection_loss_final": float(fit["final_projection_loss"]),
                    "projection_overlap_initial": float(fit["initial_overlap"]),
                    "projection_overlap_final": float(fit["final_overlap"]),
                    "optimizer_status": str(fit["status"]),
                    "optimizer_success": bool(fit["success"]),
                },
            )
        )

    summary = _trajectory_summary(trajectory)
    pvqd_nfev_total = int(sum(int(step.get("nfev", 0)) for step in pvqd_steps))
    resources = _scaffold_resources_for_layouts(
        state_layout=layout,
        interval_layouts=[layout for _ in range(max(0, int(len(times) - 1)))],
        state_scope="generic_fixed_pvqd_state_scaffold",
        horizon_scope="repeated_generic_fixed_pvqd_state_scaffold",
        extra={
            "pvqd_step_count": int(len(pvqd_steps)),
            "pvqd_nfev_total": int(pvqd_nfev_total),
        },
    )
    adapter = common._qiskit_dynamics_adapter()
    qiskit_config = adapter.qiskit_dynamics_config_from_case(case)
    qiskit_parity = adapter.pvqd_component_parity_result(
        config=qiskit_config,
        case=case,
        algorithm_id=FIXED_PVQD_ALGORITHM_ID,
        component_inputs=qiskit_component_inputs,
        support_scope="fixed_pvqd_scaffold_target_projection_component_parity",
    )

    metrics = {
        "method_kind": "fixed_pvqd",
        "decision_data_flow": "ideal_overlap_estimator_for_product_formula_target_circuit",
        "pvqd_step_count": int(len(pvqd_steps)),
        "pvqd_nfev_total": int(pvqd_nfev_total),
        "final_runtime_parameter_count": int(theta_current.size),
        "final_logical_block_count": int(getattr(layout, "logical_parameter_count")),
        "pvqd_target_depends_on_exact_interval_propagation": False,
        "pvqd_target_policy": "product_formula_circuit_step",
        "pvqd_target_product_formula_order": int(target_order),
        "uses_statevector_as_ideal_overlap_estimator": True,
        "exact_fields_reporting_only": True,
    }
    tuning = build_dynamics_tuning_provenance(
        case=case,
        algorithm_id=FIXED_PVQD_ALGORITHM_ID,
        settings_kind="comparator",
        settings_payload={
            "optimizer_method": "coordinate_refine",
            "optimizer_maxiter": int(maxiter),
            "overlap_tol": float(overlap_tol),
            "target_policy": "product_formula_circuit_step",
            "target_product_formula_order": int(target_order),
        },
        settings_source=common.metadata_override_settings_source(
            case,
            ("pvqd_optimizer_maxiter", "pvqd_overlap_tol", "pvqd_target_product_formula_order"),
        ),
        locked=False,
    )
    parameter_manifest = _generic_parameter_manifest(
        case=case,
        runtime_input=runtime_input,
        algorithm_id=FIXED_PVQD_ALGORITHM_ID,
        times=times,
        terms=terms,
        flow=flow,
    )
    parameter_manifest["tuning_provenance"] = dict(tuning)
    return json_safe(
        {
            "schema_version": "generic_fixed_pvqd_benchmark_v1",
            "case": case.to_dict(),
            "row_contract": {
                "qpu_faithful": True,
                "exact_assisted": False,
                "diagnostic": True,
            },
            "parameter_manifest": parameter_manifest,
            "tuning_provenance": tuning,
            "command": list(command),
            "trajectory": trajectory,
            "pvqd_steps": pvqd_steps,
            "qiskit_parity": qiskit_parity,
            "summary": summary,
            "metrics": metrics,
            "resources": resources,
            "compile_audit": _compile_audit_from_resources(resources),
            "provenance": {
                "route_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "benchmark_only": True,
                "runner_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_pvqd_target",
                "comparator_kernel": "repo_native_fixed_pvqd_product_formula_target_projection",
                "decision_data_flow": "ideal_overlap_estimator_for_product_formula_target_circuit",
                "controller_paths_called": False,
                "controller_decisions_modified": False,
                "exact_reference_controller_inputs": False,
                "exact_interval_targets_used_by_comparator": False,
                "uses_statevector_as_ideal_overlap_estimator": True,
                "exact_fields_reporting_only": True,
                "qiskit_boundary": "pipelines.exact_bench_only" if qiskit_parity else "not_requested",
            },
        }
    )



FIXED_PVQD_ALGORITHM_ID = "dyn_fixed_pvqd"


def run_fixed_pvqd_benchmark_row(*, case: DynamicsBenchmarkCase, output_dir: Path) -> DynamicsBenchmarkRow:
    return common.run_native_generic_comparator_row(
        case=case,
        algorithm_id=FIXED_PVQD_ALGORITHM_ID,
        output_dir=Path(output_dir),
        payload_builder=_build_fixed_pvqd_payload,
    )




# ---- adaptive_pvqd.py (collapsed legacy implementation) ----
"""Repo-native adaptive-pVQD generic dynamics benchmark.

The pVQD target is a product-formula circuit step applied to the current
prepared variational state.  Statevectors are used only as an ideal overlap
estimator for that circuit-preparable target, never as ED target trajectories.
"""


from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.benchmarks import common
from pipelines.time_dynamics.benchmarks.common import (
    _apply_product_formula_step,
    _build_layout_for_terms,
    _candidate_pool_completeness,
    _compile_audit_from_resources,
    _compiled_executor_for_terms,
    _compiled_pauli_actions_by_label,
    _copy_theta_by_layout_blocks,
    _float_or_none,
    _generic_parameter_manifest,
    _metadata_float,
    _metadata_int,
    _metadata_optional_int,
    _native_hamiltonian_flow,
    _normalize_state,
    _prepare_scaffold_state,
    _runtime_variational_bundle,
    _scaffold_resources_for_layouts,
    _state_diagnostic_row,
    _term_label,
    _term_label_set,
    _trajectory_summary,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DynamicsBenchmarkCase,
    DynamicsBenchmarkRow,
    build_dynamics_tuning_provenance,
    json_safe,
)

ADAPTIVE_PVQD_ALGORITHM_ID = "dyn_adaptive_pvqd"


def _candidate_indices_for_adaptive_pvqd(
    *,
    candidate_pool: Sequence[Any],
    used_labels: set[str],
    candidate_limit: int | None,
) -> list[int]:
    if candidate_limit is not None and int(candidate_limit) <= 0:
        return []
    out: list[int] = []
    for idx, term in enumerate(candidate_pool):
        if _term_label(term, idx) in used_labels:
            continue
        if candidate_limit is not None and len(out) >= int(candidate_limit):
            break
        out.append(int(idx))
    return out


def _loss_from_fit(fit: Mapping[str, Any]) -> float:
    value = _float_or_none(fit.get("final_projection_loss"))
    return float("inf") if value is None else float(value)


def _build_adaptive_pvqd_payload(
    *,
    case: DynamicsBenchmarkCase,
    runtime_input: Any,
    command: Sequence[str],
) -> dict[str, Any]:
    flow = _native_hamiltonian_flow(case, runtime_input)
    terms = flow.terms_for_interval(float(flow.times[0]), float(flow.times[min(1, len(flow.times) - 1)]))
    hmat = np.asarray(flow.static_hmat, dtype=complex)
    psi_initial = _normalize_state(runtime_input.psi_initial)
    if hmat.shape[0] != psi_initial.size:
        raise ValueError(
            f"Hamiltonian dimension {hmat.shape} does not match state size {psi_initial.size}"
        )
    current_terms, layout, theta, psi_ref, executor, _drive_aligned_scaffold = _runtime_variational_bundle(runtime_input)
    candidate_pool = tuple(getattr(runtime_input, "candidate_pool_terms", ()) or ())
    times = flow.times
    if int(times.size) < 2:
        raise ValueError("adaptive pVQD comparator requires at least two time points")
    exact_states = flow.exact_states
    observable_context = dict(flow.observable_context or {})
    action_by_label: dict[str, Any] = {}
    qiskit_component_inputs: list[dict[str, Any]] = []

    maxiter_default = _metadata_int(case, "pvqd_optimizer_maxiter", 24, minimum=1)
    overlap_tol_default = _metadata_float(case, "pvqd_overlap_tol", 1.0e-8, minimum=0.0)
    maxiter = _metadata_int(case, "adaptive_pvqd_optimizer_maxiter", maxiter_default, minimum=1)
    overlap_tol = _metadata_float(case, "adaptive_pvqd_overlap_tol", overlap_tol_default, minimum=0.0)
    target_order = _metadata_int(case, "adaptive_pvqd_target_product_formula_order", 2, minimum=1)
    append_loss_threshold = _metadata_float(
        case,
        "adaptive_pvqd_append_loss_threshold",
        1.0e-3,
        minimum=0.0,
    )
    append_min_improvement = _metadata_float(
        case,
        "adaptive_pvqd_append_min_loss_improvement",
        1.0e-5,
        minimum=0.0,
    )
    candidate_limit = _metadata_optional_int(
        case,
        "adaptive_pvqd_append_candidate_limit",
        4,
        minimum=0,
    )

    theta_current = np.asarray(theta, dtype=float).reshape(-1)
    current_state = _prepare_scaffold_state(executor, psi_ref, theta_current)
    trajectory: list[dict[str, Any]] = [
        _state_diagnostic_row(
            checkpoint_index=0,
            time_value=float(times[0]),
            method="generic_adaptive_pvqd",
            method_kind="adaptive_pvqd",
            state=current_state,
            exact_state=exact_states[0],
            hmat=flow.hmat_at_time(float(times[0])),
            **observable_context,
            extra={
                "runtime_parameter_count": int(theta_current.size),
                "logical_block_count": int(getattr(layout, "logical_parameter_count")),
                "adaptive_pvqd_step_index": None,
                "pvqd_nfev": 0,
                "projection_loss_initial": None,
                "projection_loss_final": None,
                "projection_overlap_initial": None,
                "projection_overlap_final": None,
                "append_accepted": None,
                "append_candidate_evaluations": 0,
            },
        )
    ]
    pvqd_steps: list[dict[str, Any]] = []
    append_events: list[dict[str, Any]] = []
    append_candidate_evaluations: list[dict[str, Any]] = []
    interval_layouts: list[Any] = []

    for interval_index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        dt = float(right - left)
        terms_step = flow.terms_for_interval(float(left), float(right))
        start_layout = layout
        theta_start = np.asarray(theta_current, dtype=float).reshape(-1).copy()
        start_state = np.asarray(current_state, dtype=complex).reshape(-1).copy()
        for label, action in _compiled_pauli_actions_by_label(terms_step).items():
            action_by_label.setdefault(str(label), action)
        target = _apply_product_formula_step(
            terms=terms_step,
            action_by_label=action_by_label,
            psi=current_state,
            dt=float(dt),
            order=int(target_order),
        )
        base_theta, base_fit = _fit_pvqd_projection_step(
            executor=executor,
            psi_ref=psi_ref,
            theta_start=theta_current,
            target_state=target,
            maxiter=maxiter,
            overlap_tol=overlap_tol,
        )
        base_state = _prepare_scaffold_state(executor, psi_ref, base_theta)
        base_loss = _loss_from_fit(base_fit)
        selected = {
            "theta": base_theta,
            "state": base_state,
            "fit": base_fit,
            "loss": base_loss,
            "terms": current_terms,
            "layout": layout,
            "executor": executor,
            "candidate_pool_index": None,
            "candidate_label": None,
        }
        candidate_eval_count = 0
        if base_loss > float(append_loss_threshold):
            used_labels = _term_label_set(current_terms)
            for candidate_index in _candidate_indices_for_adaptive_pvqd(
                candidate_pool=candidate_pool,
                used_labels=used_labels,
                candidate_limit=candidate_limit,
            ):
                candidate = candidate_pool[int(candidate_index)]
                candidate_terms = tuple(current_terms) + (candidate,)
                candidate_layout = _build_layout_for_terms(candidate_terms, reference_layout=layout)
                candidate_theta0 = _copy_theta_by_layout_blocks(
                    old_theta=theta_current,
                    old_layout=layout,
                    new_layout=candidate_layout,
                )
                candidate_executor = _compiled_executor_for_terms(candidate_terms, candidate_layout)
                cand_theta, cand_fit = _fit_pvqd_projection_step(
                    executor=candidate_executor,
                    psi_ref=psi_ref,
                    theta_start=candidate_theta0,
                    target_state=target,
                    maxiter=maxiter,
                    overlap_tol=overlap_tol,
                )
                cand_state = _prepare_scaffold_state(candidate_executor, psi_ref, cand_theta)
                cand_loss = _loss_from_fit(cand_fit)
                candidate_eval_count += 1
                eval_row = {
                    "interval_index": int(interval_index),
                    "candidate_pool_index": int(candidate_index),
                    "candidate_label": _term_label(candidate, candidate_index),
                    "base_projection_loss": float(base_loss),
                    "candidate_projection_loss": float(cand_loss),
                    "projection_loss_improvement": float(base_loss - cand_loss),
                    "runtime_parameter_count": int(np.asarray(cand_theta).size),
                    "logical_block_count": int(getattr(candidate_layout, "logical_parameter_count")),
                }
                append_candidate_evaluations.append(eval_row)
                if cand_loss + 1.0e-15 < float(selected["loss"]):
                    selected = {
                        "theta": cand_theta,
                        "state": cand_state,
                        "fit": cand_fit,
                        "loss": cand_loss,
                        "terms": candidate_terms,
                        "layout": candidate_layout,
                        "executor": candidate_executor,
                        "candidate_pool_index": int(candidate_index),
                        "candidate_label": _term_label(candidate, candidate_index),
                    }

        improvement = float(base_loss - float(selected["loss"]))
        append_accepted = (
            selected["candidate_pool_index"] is not None
            and improvement >= float(append_min_improvement)
        )
        if not append_accepted:
            selected = {
                "theta": base_theta,
                "state": base_state,
                "fit": base_fit,
                "loss": base_loss,
                "terms": current_terms,
                "layout": layout,
                "executor": executor,
                "candidate_pool_index": None,
                "candidate_label": None,
            }
            improvement = 0.0

        qiskit_component_inputs.append(
            {
                "interval_index": int(interval_index),
                "psi_ref": np.asarray(psi_ref, dtype=complex).reshape(-1).copy(),
                "start_layout": start_layout,
                "final_layout": selected["layout"],
                "theta_start": theta_start,
                "theta_final": np.asarray(selected["theta"], dtype=float).reshape(-1).copy(),
                "native_start_state": start_state,
                "native_target_state": np.asarray(target, dtype=complex).reshape(-1).copy(),
                "native_final_state": np.asarray(selected["state"], dtype=complex).reshape(-1).copy(),
                "target_terms": tuple(terms_step),
                "dt": float(dt),
                "target_order": int(target_order),
                "fit": dict(selected["fit"]),
            }
        )

        theta_current = np.asarray(selected["theta"], dtype=float).reshape(-1)
        current_state = np.asarray(selected["state"], dtype=complex).reshape(-1)
        current_terms = tuple(selected["terms"])
        layout = selected["layout"]
        executor = selected["executor"]
        interval_layouts.append(layout)
        fit = dict(selected["fit"])
        step_payload = {
            "interval_index": int(interval_index),
            "time_start": float(left),
            "time_stop": float(right),
            "dt": float(dt),
            "optimizer_method": "coordinate_refine",
            "optimizer_maxiter": int(maxiter),
            "overlap_tol": float(overlap_tol),
            "target_policy": "product_formula_circuit_step",
            "target_product_formula_order": int(target_order),
            "append_loss_threshold": float(append_loss_threshold),
            "append_min_loss_improvement": float(append_min_improvement),
            "base_projection_loss": float(base_loss),
            "selected_projection_loss": float(selected["loss"]),
            "selected_loss_improvement": float(improvement),
            "append_accepted": bool(append_accepted),
            "append_candidate_evaluations": int(candidate_eval_count),
            **fit,
        }
        if append_accepted:
            event = {
                "interval_index": int(interval_index),
                "candidate_pool_index": int(selected["candidate_pool_index"]),
                "candidate_label": str(selected["candidate_label"]),
                "base_projection_loss": float(base_loss),
                "selected_projection_loss": float(selected["loss"]),
                "projection_loss_improvement": float(improvement),
                "runtime_parameter_count": int(theta_current.size),
                "logical_block_count": int(getattr(layout, "logical_parameter_count")),
            }
            append_events.append(event)
            step_payload["append_event"] = event
        pvqd_steps.append(step_payload)
        trajectory.append(
            _state_diagnostic_row(
                checkpoint_index=int(interval_index) + 1,
                time_value=float(right),
                method="generic_adaptive_pvqd",
                method_kind="adaptive_pvqd",
                state=current_state,
                exact_state=exact_states[int(interval_index) + 1],
                hmat=flow.hmat_at_time(float(right)),
                **observable_context,
                extra={
                    "runtime_parameter_count": int(theta_current.size),
                    "logical_block_count": int(getattr(layout, "logical_parameter_count")),
                    "adaptive_pvqd_step_index": int(interval_index),
                    "pvqd_nfev": int(fit["nfev"]),
                    "projection_loss_initial": float(fit["initial_projection_loss"]),
                    "projection_loss_final": float(fit["final_projection_loss"]),
                    "projection_overlap_initial": float(fit["initial_overlap"]),
                    "projection_overlap_final": float(fit["final_overlap"]),
                    "optimizer_status": str(fit["status"]),
                    "optimizer_success": bool(fit["success"]),
                    "append_accepted": bool(append_accepted),
                    "append_candidate_evaluations": int(candidate_eval_count),
                },
            )
        )

    summary = _trajectory_summary(trajectory)
    pvqd_nfev_total = int(sum(int(step.get("nfev", 0)) for step in pvqd_steps))
    resources = _scaffold_resources_for_layouts(
        state_layout=layout,
        interval_layouts=interval_layouts,
        state_scope="generic_adaptive_pvqd_state_scaffold",
        horizon_scope="generic_adaptive_pvqd_adaptive_scaffold_epoch_sum",
        extra={
            "adaptive_pvqd_step_count": int(len(pvqd_steps)),
            "pvqd_nfev_total": int(pvqd_nfev_total),
            "append_events_total": int(len(append_events)),
            "append_candidate_evaluations_total": int(len(append_candidate_evaluations)),
        },
    )
    adapter = common._qiskit_dynamics_adapter()
    qiskit_config = adapter.qiskit_dynamics_config_from_case(case)
    qiskit_parity = adapter.pvqd_component_parity_result(
        config=qiskit_config,
        case=case,
        algorithm_id=ADAPTIVE_PVQD_ALGORITHM_ID,
        component_inputs=qiskit_component_inputs,
        support_scope="adaptive_pvqd_component_parity_only_no_native_qiskit_adaptive_update",
    )

    metrics = {
        "method_kind": "adaptive_pvqd",
        "decision_mode": "generic_product_formula_target_projection_v1",
        "decision_data_flow": "ideal_overlap_estimator_for_product_formula_target_circuit",
        "candidate_pool_complete": True,
        "candidate_pool_completeness": _candidate_pool_completeness(runtime_input),
        "candidate_pool_size": int(len(candidate_pool)),
        "adaptive_pvqd_step_count": int(len(pvqd_steps)),
        "pvqd_nfev_total": int(pvqd_nfev_total),
        "append_events_total": int(len(append_events)),
        "append_candidate_evaluations_total": int(len(append_candidate_evaluations)),
        "final_runtime_parameter_count": int(theta_current.size),
        "final_logical_block_count": int(getattr(layout, "logical_parameter_count")),
        "pvqd_target_depends_on_exact_interval_propagation": False,
        "pvqd_target_policy": "product_formula_circuit_step",
        "pvqd_target_product_formula_order": int(target_order),
        "append_scoring_uses_exact_reference": False,
        "qiskit_adaptive_support": "component_parity_only_no_native_qiskit_adaptive_update",
        "uses_statevector_as_ideal_overlap_estimator": True,
        "exact_fields_reporting_only": True,
    }
    tuning = build_dynamics_tuning_provenance(
        case=case,
        algorithm_id=ADAPTIVE_PVQD_ALGORITHM_ID,
        settings_kind="comparator",
        settings_payload={
            "optimizer_method": "coordinate_refine",
            "optimizer_maxiter": int(maxiter),
            "overlap_tol": float(overlap_tol),
            "target_policy": "product_formula_circuit_step",
            "target_product_formula_order": int(target_order),
            "append_loss_threshold": float(append_loss_threshold),
            "append_min_loss_improvement": float(append_min_improvement),
            "append_candidate_limit": None if candidate_limit is None else int(candidate_limit),
        },
        settings_source=common.metadata_override_settings_source(
            case,
            (
                "pvqd_optimizer_maxiter",
                "pvqd_overlap_tol",
                "adaptive_pvqd_optimizer_maxiter",
                "adaptive_pvqd_overlap_tol",
                "adaptive_pvqd_target_product_formula_order",
                "adaptive_pvqd_append_loss_threshold",
                "adaptive_pvqd_append_min_loss_improvement",
                "adaptive_pvqd_append_candidate_limit",
            ),
        ),
        locked=False,
    )
    parameter_manifest = _generic_parameter_manifest(
        case=case,
        runtime_input=runtime_input,
        algorithm_id=ADAPTIVE_PVQD_ALGORITHM_ID,
        times=times,
        terms=terms,
        flow=flow,
    )
    parameter_manifest["tuning_provenance"] = dict(tuning)
    return json_safe(
        {
            "schema_version": "generic_adaptive_pvqd_benchmark_v1",
            "case": case.to_dict(),
            "row_contract": {
                "qpu_faithful": True,
                "exact_assisted": False,
                "diagnostic": True,
            },
            "parameter_manifest": parameter_manifest,
            "tuning_provenance": tuning,
            "command": list(command),
            "trajectory": trajectory,
            "pvqd_steps": pvqd_steps,
            "append_events": append_events,
            "append_candidate_evaluations": append_candidate_evaluations,
            "qiskit_parity": qiskit_parity,
            "summary": summary,
            "metrics": metrics,
            "resources": resources,
            "compile_audit": _compile_audit_from_resources(resources),
            "provenance": {
                "route_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "benchmark_only": True,
                "runner_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_pvqd_target",
                "comparator_kernel": "repo_native_adaptive_pvqd_product_formula_target_projection",
                "decision_data_flow": "ideal_overlap_estimator_for_product_formula_target_circuit",
                "controller_paths_called": False,
                "controller_decisions_modified": False,
                "exact_reference_controller_inputs": False,
                "exact_interval_targets_used_by_comparator": False,
                "append_scoring_uses_exact_reference": False,
                "qiskit_adaptive_support": "component_parity_only_no_native_qiskit_adaptive_update",
                "uses_statevector_as_ideal_overlap_estimator": True,
                "exact_fields_reporting_only": True,
                "qiskit_boundary": "pipelines.exact_bench_only" if qiskit_parity else "not_requested",
            },
        }
    )


def run_adaptive_pvqd_benchmark_row(*, case: DynamicsBenchmarkCase, output_dir: Path) -> DynamicsBenchmarkRow:
    return common.run_native_generic_comparator_row(
        case=case,
        algorithm_id=ADAPTIVE_PVQD_ALGORITHM_ID,
        output_dir=Path(output_dir),
        payload_builder=_build_adaptive_pvqd_payload,
    )




# ---- avqds.py (collapsed legacy implementation) ----
"""Repo-native AVQDS RHS-tangent generic dynamics benchmark."""


from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.benchmarks import common
from pipelines.time_dynamics.benchmarks.common import (
    _build_layout_for_terms,
    _candidate_pool_completeness,
    _compile_audit_from_resources,
    _compiled_executor_for_terms,
    _copy_theta_by_layout_blocks,
    _float_or_none,
    _generic_parameter_manifest,
    _max_or_none,
    _metadata_float,
    _metadata_optional_int,
    _native_hamiltonian_flow,
    _normalize_state,
    _prepare_scaffold_state,
    _runtime_variational_bundle,
    _scaffold_resources_for_layouts,
    _state_diagnostic_row,
    _term_label,
    _term_label_set,
    _trajectory_summary,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DynamicsBenchmarkCase,
    DynamicsBenchmarkRow,
    build_dynamics_tuning_provenance,
    json_safe,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor


def _solve_avqds_tangent_step(
    *,
    executor: CompiledAnsatzExecutor,
    psi_ref: np.ndarray,
    theta_start: np.ndarray,
    hmat: np.ndarray,
    dt: float,
    regularization_lambda: float,
    pinv_relative_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    theta_vec = np.asarray(theta_start, dtype=float).reshape(-1)
    psi, tangents = executor.prepare_state_with_runtime_tangents(theta_vec, psi_ref)
    psi = _normalize_state(psi)
    rhs_state = -1.0j * (np.asarray(hmat, dtype=complex) @ psi)
    rhs_norm = float(np.linalg.norm(rhs_state))
    param_count = int(theta_vec.size)
    if param_count == 0:
        residual_norm = rhs_norm
        return theta_vec, psi, {
            "theta_dot": [],
            "rhs_norm": float(rhs_norm),
            "rhs_residual_norm": float(residual_norm),
            "rhs_residual_ratio": 0.0 if rhs_norm <= 1.0e-15 else float(residual_norm / rhs_norm),
            "projected_rhs_norm": 0.0,
            "delta_norm": 0.0,
            "linear_solve_status": "no_parameters",
            "linear_solve_count": 0,
            "regularization_lambda": float(regularization_lambda),
            "pinv_relative_cutoff": float(pinv_relative_cutoff),
            "retained_rank": 0,
            "parameter_count": 0,
            "tangent_condition_estimate": None,
            "metric_symmetry_max_abs": 0.0,
            "force_norm": 0.0,
            "solve_residual_norm": 0.0,
            "dense_reference_kind": "direct_dense_tangent_matrix_no_parameters",
            "state_prep_count": 1,
            "success": True,
            "message": "no runtime parameters available",
        }

    tangent_matrix = np.column_stack(
        [np.asarray(tangents[idx], dtype=complex).reshape(-1) for idx in range(param_count)]
    )
    metric = np.real(tangent_matrix.conj().T @ tangent_matrix)
    force = np.real(tangent_matrix.conj().T @ rhs_state)
    reg = float(max(0.0, regularization_lambda))
    solve_matrix = metric + reg * np.eye(param_count)
    singular_values = np.linalg.svd(solve_matrix, compute_uv=False)
    cutoff = float(max(0.0, pinv_relative_cutoff))
    max_sv = float(np.max(singular_values)) if singular_values.size else 0.0
    retained_rank = int(sum(float(sv) > cutoff * max_sv for sv in singular_values)) if max_sv > 0.0 else 0
    condition = None
    positive = [float(sv) for sv in singular_values if float(sv) > 1.0e-15]
    if positive:
        condition = float(max(positive) / min(positive))
    theta_dot = np.asarray(np.linalg.pinv(solve_matrix, rcond=cutoff) @ force, dtype=float).reshape(-1)
    solve_residual = np.asarray(solve_matrix @ theta_dot - force, dtype=float).reshape(-1)
    projected = tangent_matrix @ theta_dot
    residual = rhs_state - projected
    delta = float(dt) * theta_dot
    theta_next = theta_vec + delta
    final_state = _prepare_scaffold_state(executor, psi_ref, theta_next)
    residual_norm = float(np.linalg.norm(residual))
    projected_norm = float(np.linalg.norm(projected))
    ratio = 0.0 if rhs_norm <= 1.0e-15 else float(residual_norm / rhs_norm)
    return theta_next, final_state, {
        "theta_dot": [float(x) for x in theta_dot.tolist()],
        "rhs_norm": float(rhs_norm),
        "rhs_residual_norm": float(residual_norm),
        "rhs_residual_ratio": float(ratio),
        "projected_rhs_norm": float(projected_norm),
        "delta_norm": float(np.linalg.norm(delta)),
        "linear_solve_status": "ok",
        "linear_solve_count": 1,
        "regularization_lambda": float(reg),
        "pinv_relative_cutoff": float(cutoff),
        "retained_rank": int(retained_rank),
        "parameter_count": int(param_count),
        "tangent_condition_estimate": condition,
        "metric_symmetry_max_abs": float(np.max(np.abs(metric - metric.T))) if metric.size else 0.0,
        "force_norm": float(np.linalg.norm(force)),
        "solve_residual_norm": float(np.linalg.norm(solve_residual)),
        "dense_reference_kind": "direct_dense_tangent_matrix_regularized_pinv",
        "state_prep_count": 1,
        "success": True,
        "message": "repo-native regularized tangent solve",
    }


def _candidate_indices_for_avqds(
    *,
    candidate_pool: Sequence[Any],
    used_labels: set[str],
    candidate_limit: int | None,
) -> list[int]:
    if candidate_limit is not None and int(candidate_limit) <= 0:
        return []
    out: list[int] = []
    for idx, term in enumerate(candidate_pool):
        if _term_label(term, idx) in used_labels:
            continue
        execution_mode = str(
            getattr(term, "execution_mode", "termwise_product") or "termwise_product"
        ).strip().lower()
        if execution_mode == "grouped_exact":
            # AVQDS and AVQDS-T currently require runtime Pauli tangents for
            # each candidate.  grouped_exact generators deliberately do not
            # expose per-Pauli runtime parameters, so they are not legal append
            # candidates for this comparator even though they are valid static
            # ADAPT generators under logical-shared execution.
            continue
        if candidate_limit is not None and len(out) >= int(candidate_limit):
            break
        out.append(int(idx))
    return out


def _build_avqds_correctness_sidecar(
    *,
    case: DynamicsBenchmarkCase,
    avqds_steps: Sequence[Mapping[str, Any]],
    append_events: Sequence[Mapping[str, Any]],
    append_candidate_evaluations: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
    candidate_pool: Sequence[Any],
    state_norms: Sequence[float],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    finite_payload = common._json_numeric_values_are_finite(
        {
            "avqds_steps": list(avqds_steps),
            "append_events": list(append_events),
            "append_candidate_evaluations": list(append_candidate_evaluations),
            "trajectory": list(trajectory),
            "state_norms": list(state_norms),
        }
    )
    checks.append(
        common._check_payload(
            check_id="finite_trajectory_diagnostics",
            check_type="invariant_correctness",
            passed=bool(avqds_steps) and bool(finite_payload),
            details={"step_count": len(avqds_steps), "trajectory_points": len(trajectory)},
        )
    )

    ratio_bad: list[int] = []
    solve_bad: list[int] = []
    for step in avqds_steps:
        idx = int(step.get("interval_index", len(ratio_bad)))
        rhs_norm = _float_or_none(step.get("rhs_norm"))
        residual_norm = _float_or_none(step.get("rhs_residual_norm"))
        ratio = _float_or_none(step.get("rhs_residual_ratio"))
        if rhs_norm is None or residual_norm is None or ratio is None:
            ratio_bad.append(idx)
        else:
            expected = 0.0 if abs(rhs_norm) <= 1.0e-15 else float(residual_norm / rhs_norm)
            if abs(float(ratio) - expected) > 1.0e-9:
                ratio_bad.append(idx)
        parameter_count = int(step.get("parameter_count", 0) or 0)
        rank = int(step.get("retained_rank", 0) or 0)
        if (
            str(step.get("linear_solve_status")) not in {"ok", "no_parameters"}
            or rank < 0
            or rank > max(parameter_count, 0)
            or _float_or_none(step.get("regularization_lambda")) is None
            or float(step.get("regularization_lambda", 0.0)) < 0.0
            or _float_or_none(step.get("pinv_relative_cutoff")) is None
            or float(step.get("pinv_relative_cutoff", 0.0)) < 0.0
            or _float_or_none(step.get("metric_symmetry_max_abs")) is None
            or float(step.get("metric_symmetry_max_abs", 0.0)) > 1.0e-8
            or _float_or_none(step.get("solve_residual_norm")) is None
        ):
            solve_bad.append(idx)
    checks.append(
        common._check_payload(
            check_id="rhs_tangent_dense_reference_solve",
            check_type="dense_reference_component_parity",
            passed=bool(avqds_steps) and not ratio_bad and not solve_bad,
            details={
                "ratio_bad_interval_indices": ratio_bad,
                "solve_bad_interval_indices": solve_bad,
                "dense_reference_kind": "direct_dense_tangent_matrix_regularized_pinv",
            },
        )
    )

    grouped_labels = {
        _term_label(term, idx)
        for idx, term in enumerate(candidate_pool)
        if str(getattr(term, "execution_mode", "termwise_product") or "termwise_product").strip().lower()
        == "grouped_exact"
    }
    grouped_eval_labels = [
        str(item.get("candidate_label"))
        for item in append_candidate_evaluations
        if str(item.get("candidate_label")) in grouped_labels
    ]
    checks.append(
        common._check_payload(
            check_id="grouped_exact_candidate_exclusion",
            check_type="append_admission_correctness",
            passed=not grouped_eval_labels,
            details={
                "grouped_exact_candidate_excluded_count": len(grouped_labels),
                "grouped_exact_labels_evaluated": grouped_eval_labels,
            },
        )
    )

    evals_by_interval: dict[int, list[Mapping[str, Any]]] = {}
    for item in append_candidate_evaluations:
        evals_by_interval.setdefault(int(item.get("interval_index", -1)), []).append(item)
    events_by_interval = {int(event.get("interval_index", -1)): event for event in append_events}
    append_bad: list[int] = []
    for step in avqds_steps:
        idx = int(step.get("interval_index", -1))
        evals = evals_by_interval.get(idx, [])
        threshold = float(step.get("append_rhs_residual_ratio_threshold", 0.0))
        min_gain = float(step.get("append_min_residual_ratio_gain", 0.0))
        accepted = bool(step.get("append_accepted", False))
        event = events_by_interval.get(idx)
        base_ratio = None
        if event is not None:
            base_ratio = _float_or_none(event.get("rhs_residual_ratio_base"))
        if base_ratio is None and evals:
            base_ratio = _float_or_none(evals[0].get("rhs_residual_ratio_base"))
        if base_ratio is None:
            base_ratio = _float_or_none(step.get("rhs_residual_ratio"))
        if int(step.get("append_candidate_evaluations", 0)) != len(evals):
            append_bad.append(idx)
            continue
        if base_ratio is not None and float(base_ratio) <= threshold and evals:
            append_bad.append(idx)
            continue
        eligible = [
            ev
            for ev in evals
            if base_ratio is not None
            and _float_or_none(ev.get("rhs_residual_ratio")) is not None
            and float(base_ratio) - float(ev["rhs_residual_ratio"]) >= min_gain
        ]
        if accepted:
            selected_ratio = None if event is None else _float_or_none(event.get("rhs_residual_ratio_selected"))
            best_ratio = min((float(ev["rhs_residual_ratio"]) for ev in eligible), default=None)
            if event is None or selected_ratio is None or best_ratio is None or abs(float(selected_ratio) - best_ratio) > 1.0e-9:
                append_bad.append(idx)
        elif eligible:
            append_bad.append(idx)
    checks.append(
        common._check_payload(
            check_id="append_threshold_min_gain_semantics",
            check_type="append_admission_correctness",
            passed=not append_bad,
            details={"bad_interval_indices": append_bad},
        )
    )

    norm_deviations = [abs(float(value) - 1.0) for value in state_norms]
    max_norm_deviation = max(norm_deviations, default=0.0)
    checks.append(
        common._check_payload(
            check_id="state_norm_preservation",
            check_type="invariant_correctness",
            passed=bool(state_norms) and max_norm_deviation <= 1.0e-10,
            details={"state_norm_count": len(state_norms), "max_norm_deviation": max_norm_deviation},
        )
    )

    passed = common._checks_pass(checks)
    return json_safe(
        {
            "schema": "avqds_correctness_v1",
            "algorithm_id": AVQDS_ALGORITHM_ID,
            "family": str(case.family),
            "case_id": str(case.case_id),
            "sidecar_name": common.CORRECTNESS_SIDECAR_FILENAMES[AVQDS_ALGORITHM_ID],
            "support_scope": "avqds_rhs_tangent_solve_append_and_invariant_correctness",
            "sidecar_kind": "dense_reference_component_parity_and_invariant_correctness",
            "status": "ok" if passed else "failed",
            "passed": bool(passed),
            "required_status": "passed",
            "check_count": int(len(checks)),
            "checks": checks,
            "exact_data_policy": "benchmark_exact_fields_reporting_only_not_rhs_tangent_or_append_decision",
            "physical_error_policy": "additive_correctness_provenance_not_a_physical_error_column",
            "controller_decisions_modified": False,
            "exact_reference_controller_inputs": False,
        }
    )


def _build_avqds_payload(
    *,
    case: DynamicsBenchmarkCase,
    runtime_input: Any,
    command: Sequence[str],
) -> dict[str, Any]:
    flow = _native_hamiltonian_flow(case, runtime_input)
    terms = flow.terms_for_interval(float(flow.times[0]), float(flow.times[min(1, len(flow.times) - 1)]))
    hmat = np.asarray(flow.static_hmat, dtype=complex)
    psi_initial = _normalize_state(runtime_input.psi_initial)
    if hmat.shape[0] != psi_initial.size:
        raise ValueError(
            f"Hamiltonian dimension {hmat.shape} does not match state size {psi_initial.size}"
        )
    current_terms, layout, theta, psi_ref, executor, _drive_aligned_scaffold = _runtime_variational_bundle(runtime_input)
    times = flow.times
    if int(times.size) < 2:
        raise ValueError("AVQDS comparator requires at least two time points")
    exact_states = flow.exact_states
    observable_context = dict(flow.observable_context or {})
    regularization_lambda = _metadata_float(case, "avqds_regularization_lambda", 1.0e-8, minimum=0.0)
    pinv_relative_cutoff = _metadata_float(case, "avqds_pinv_relative_cutoff", 1.0e-10, minimum=0.0)
    append_threshold = _metadata_float(
        case,
        "avqds_append_rhs_residual_ratio_threshold",
        1.0e-3,
        minimum=0.0,
    )
    append_min_gain = _metadata_float(case, "avqds_append_min_residual_ratio_gain", 1.0e-5, minimum=0.0)
    candidate_limit = _metadata_optional_int(case, "avqds_append_candidate_limit", 4, minimum=0)
    candidate_pool = tuple(getattr(runtime_input, "candidate_pool_terms", ()) or ())

    theta_current = np.asarray(theta, dtype=float).reshape(-1)
    current_state = _prepare_scaffold_state(executor, psi_ref, theta_current)
    trajectory: list[dict[str, Any]] = [
        _state_diagnostic_row(
            checkpoint_index=0,
            time_value=float(times[0]),
            method="generic_avqds",
            method_kind="avqds",
            state=current_state,
            exact_state=exact_states[0],
            hmat=flow.hmat_at_time(float(times[0])),
            **observable_context,
            extra={
                "runtime_parameter_count": int(theta_current.size),
                "logical_block_count": int(getattr(layout, "logical_parameter_count")),
                "avqds_step_index": None,
                "avqds_linear_solve_count": 0,
                "avqds_state_prep_count": 0,
                "rhs_residual_ratio": None,
                "append_accepted": None,
                "append_candidate_evaluations": 0,
            },
        )
    ]
    avqds_steps: list[dict[str, Any]] = []
    append_events: list[dict[str, Any]] = []
    append_candidate_evaluations: list[dict[str, Any]] = []
    interval_layouts: list[Any] = []
    state_norms: list[float] = [float(np.linalg.norm(current_state))]

    for interval_index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        dt = float(right - left)
        hmat_step = flow.hmat_for_interval(float(left), float(right))
        base_theta, base_state, base_fit = _solve_avqds_tangent_step(
            executor=executor,
            psi_ref=psi_ref,
            theta_start=theta_current,
            hmat=hmat_step,
            dt=dt,
            regularization_lambda=regularization_lambda,
            pinv_relative_cutoff=pinv_relative_cutoff,
        )
        selected = {
            "theta": base_theta,
            "state": base_state,
            "fit": base_fit,
            "terms": current_terms,
            "layout": layout,
            "executor": executor,
            "candidate_pool_index": None,
            "candidate_label": None,
        }
        candidate_eval_count = 0
        used_labels = _term_label_set(current_terms)
        if float(base_fit["rhs_residual_ratio"]) > float(append_threshold):
            for candidate_index in _candidate_indices_for_avqds(
                candidate_pool=candidate_pool,
                used_labels=used_labels,
                candidate_limit=candidate_limit,
            ):
                candidate = candidate_pool[int(candidate_index)]
                candidate_terms = tuple(current_terms) + (candidate,)
                candidate_layout = _build_layout_for_terms(candidate_terms, reference_layout=layout)
                candidate_theta = _copy_theta_by_layout_blocks(
                    old_theta=theta_current,
                    old_layout=layout,
                    new_layout=candidate_layout,
                )
                candidate_executor = _compiled_executor_for_terms(candidate_terms, candidate_layout)
                cand_theta, cand_state, cand_fit = _solve_avqds_tangent_step(
                    executor=candidate_executor,
                    psi_ref=psi_ref,
                    theta_start=candidate_theta,
                    hmat=hmat_step,
                    dt=dt,
                    regularization_lambda=regularization_lambda,
                    pinv_relative_cutoff=pinv_relative_cutoff,
                )
                candidate_eval_count += 1
                eval_row = {
                    "interval_index": int(interval_index),
                    "candidate_pool_index": int(candidate_index),
                    "candidate_label": _term_label(candidate, candidate_index),
                    "rhs_residual_ratio": float(cand_fit["rhs_residual_ratio"]),
                    "rhs_residual_ratio_base": float(base_fit["rhs_residual_ratio"]),
                    "runtime_parameter_count": int(cand_theta.size),
                    "logical_block_count": int(getattr(candidate_layout, "logical_parameter_count")),
                }
                append_candidate_evaluations.append(eval_row)
                if float(base_fit["rhs_residual_ratio"]) - float(cand_fit["rhs_residual_ratio"]) >= float(append_min_gain):
                    if selected["candidate_pool_index"] is None or float(cand_fit["rhs_residual_ratio"]) < float(selected["fit"]["rhs_residual_ratio"]):
                        selected = {
                            "theta": cand_theta,
                            "state": cand_state,
                            "fit": cand_fit,
                            "terms": candidate_terms,
                            "layout": candidate_layout,
                            "executor": candidate_executor,
                            "candidate_pool_index": int(candidate_index),
                            "candidate_label": _term_label(candidate, candidate_index),
                        }

        append_accepted = selected["candidate_pool_index"] is not None
        theta_current = np.asarray(selected["theta"], dtype=float).reshape(-1)
        current_state = np.asarray(selected["state"], dtype=complex).reshape(-1)
        current_terms = tuple(selected["terms"])
        layout = selected["layout"]
        executor = selected["executor"]
        interval_layouts.append(layout)
        fit = dict(selected["fit"])
        state_norm_after = float(np.linalg.norm(current_state))
        state_norms.append(state_norm_after)
        step_payload = {
            "interval_index": int(interval_index),
            "time_start": float(left),
            "time_stop": float(right),
            "dt": float(dt),
            "append_accepted": bool(append_accepted),
            "append_candidate_evaluations": int(candidate_eval_count),
            "append_rhs_residual_ratio_threshold": float(append_threshold),
            "append_min_residual_ratio_gain": float(append_min_gain),
            "state_norm_after": float(state_norm_after),
            **fit,
        }
        if append_accepted:
            event = {
                "interval_index": int(interval_index),
                "candidate_pool_index": int(selected["candidate_pool_index"]),
                "candidate_label": str(selected["candidate_label"]),
                "rhs_residual_ratio_base": float(base_fit["rhs_residual_ratio"]),
                "rhs_residual_ratio_selected": float(fit["rhs_residual_ratio"]),
                "runtime_parameter_count": int(theta_current.size),
                "logical_block_count": int(getattr(layout, "logical_parameter_count")),
            }
            append_events.append(event)
            step_payload["append_event"] = event
        avqds_steps.append(step_payload)
        trajectory.append(
            _state_diagnostic_row(
                checkpoint_index=int(interval_index) + 1,
                time_value=float(right),
                method="generic_avqds",
                method_kind="avqds",
                state=current_state,
                exact_state=exact_states[int(interval_index) + 1],
                hmat=flow.hmat_at_time(float(right)),
                **observable_context,
                extra={
                    "runtime_parameter_count": int(theta_current.size),
                    "logical_block_count": int(getattr(layout, "logical_parameter_count")),
                    "avqds_step_index": int(interval_index),
                    "avqds_linear_solve_count": int(fit["linear_solve_count"]),
                    "avqds_state_prep_count": int(fit["state_prep_count"]),
                    "rhs_residual_ratio": float(fit["rhs_residual_ratio"]),
                    "rhs_residual_norm": float(fit["rhs_residual_norm"]),
                    "append_accepted": bool(append_accepted),
                    "append_candidate_evaluations": int(candidate_eval_count),
                },
            )
        )

    summary = _trajectory_summary(trajectory)
    rhs_ratios = [step.get("rhs_residual_ratio") for step in avqds_steps]
    linear_solve_total = int(sum(int(step.get("linear_solve_count", 0)) for step in avqds_steps))
    state_prep_total = int(sum(int(step.get("state_prep_count", 0)) for step in avqds_steps))
    resources = _scaffold_resources_for_layouts(
        state_layout=layout,
        interval_layouts=interval_layouts,
        state_scope="generic_avqds_state_scaffold",
        horizon_scope="generic_avqds_scaffold_epoch_sum",
        extra={
            "avqds_step_count": int(len(avqds_steps)),
            "avqds_linear_solve_total": int(linear_solve_total),
            "avqds_state_prep_total": int(state_prep_total),
            "append_events_total": int(len(append_events)),
            "append_candidate_evaluations_total": int(len(append_candidate_evaluations)),
        },
    )
    metrics = {
        "method_kind": "avqds",
        "decision_mode": "generic_rhs_tangent_v1",
        "decision_data_flow": "ideal_mclachlan_tangent_observable_estimator",
        "candidate_pool_complete": True,
        "candidate_pool_completeness": _candidate_pool_completeness(runtime_input),
        "candidate_pool_size": int(len(candidate_pool)),
        "append_events_total": int(len(append_events)),
        "append_candidate_evaluations_total": int(len(append_candidate_evaluations)),
        "final_runtime_parameter_count": int(theta_current.size),
        "final_logical_block_count": int(getattr(layout, "logical_parameter_count")),
        "avqds_linear_solve_total": int(linear_solve_total),
        "avqds_step_count": int(len(avqds_steps)),
        "avqds_state_prep_total": int(state_prep_total),
        "rhs_residual_ratio_final": _float_or_none(rhs_ratios[-1]) if rhs_ratios else None,
        "rhs_residual_ratio_max": _max_or_none(rhs_ratios),
        "uses_statevector_as_ideal_observable_estimator": True,
        "exact_fields_reporting_only": True,
    }
    tuning = build_dynamics_tuning_provenance(
        case=case,
        algorithm_id=AVQDS_ALGORITHM_ID,
        settings_kind="comparator",
        settings_payload={
            "regularization_lambda": float(regularization_lambda),
            "pinv_relative_cutoff": float(pinv_relative_cutoff),
            "append_rhs_residual_ratio_threshold": float(append_threshold),
            "append_min_residual_ratio_gain": float(append_min_gain),
            "append_candidate_limit": None if candidate_limit is None else int(candidate_limit),
        },
        settings_source=common.metadata_override_settings_source(
            case,
            (
                "avqds_regularization_lambda",
                "avqds_pinv_relative_cutoff",
                "avqds_append_rhs_residual_ratio_threshold",
                "avqds_append_min_residual_ratio_gain",
                "avqds_append_candidate_limit",
            ),
        ),
        locked=False,
    )
    parameter_manifest = _generic_parameter_manifest(
        case=case,
        runtime_input=runtime_input,
        algorithm_id=AVQDS_ALGORITHM_ID,
        times=times,
        terms=terms,
        flow=flow,
    )
    parameter_manifest["tuning_provenance"] = dict(tuning)
    avqds_correctness = _build_avqds_correctness_sidecar(
        case=case,
        avqds_steps=avqds_steps,
        append_events=append_events,
        append_candidate_evaluations=append_candidate_evaluations,
        trajectory=trajectory,
        candidate_pool=candidate_pool,
        state_norms=state_norms,
    )
    return json_safe(
        {
            "schema_version": "generic_avqds_benchmark_v1",
            "case": case.to_dict(),
            "row_contract": {
                "qpu_faithful": True,
                "exact_assisted": False,
                "diagnostic": True,
            },
            "parameter_manifest": parameter_manifest,
            "tuning_provenance": tuning,
            "command": list(command),
            "trajectory": trajectory,
            "avqds_steps": avqds_steps,
            "append_events": append_events,
            "append_candidate_evaluations": append_candidate_evaluations,
            "avqds_correctness": avqds_correctness,
            "summary": summary,
            "metrics": metrics,
            "resources": resources,
            "compile_audit": _compile_audit_from_resources(resources),
            "provenance": {
                "route_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "benchmark_only": True,
                "runner_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_tangent_decision",
                "comparator_kernel": "repo_native_avqds_rhs_tangent",
                "decision_data_flow": "ideal_mclachlan_tangent_observable_estimator",
                "controller_paths_called": False,
                "controller_decisions_modified": False,
                "exact_reference_controller_inputs": False,
                "exact_fields_reporting_only": True,
                "append_scoring_uses_exact_reference": False,
                "uses_statevector_as_ideal_observable_estimator": True,
            },
        }
    )



AVQDS_ALGORITHM_ID = "dyn_avqds"


def run_avqds_benchmark_row(*, case: DynamicsBenchmarkCase, output_dir: Path) -> DynamicsBenchmarkRow:
    return common.run_native_generic_comparator_row(
        case=case,
        algorithm_id=AVQDS_ALGORITHM_ID,
        output_dir=Path(output_dir),
        payload_builder=_build_avqds_payload,
    )




# ---- avqds_t.py (collapsed legacy implementation) ----
"""Repo-native AVQDS(T) target-tangent generic dynamics benchmark.

The target tangent is built from a product-formula circuit step applied to the
current prepared state.  Exact ED/reference trajectories are diagnostics only.
"""


from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.benchmarks import common
from pipelines.time_dynamics.benchmarks.common import (
    _apply_product_formula_step,
    _build_layout_for_terms,
    _candidate_pool_completeness,
    _compile_audit_from_resources,
    _compiled_executor_for_terms,
    _compiled_pauli_actions_by_label,
    _copy_theta_by_layout_blocks,
    _float_or_none,
    _generic_parameter_manifest,
    _max_or_none,
    _metadata_float,
    _metadata_optional_int,
    _native_hamiltonian_flow,
    _normalize_state,
    _prepare_scaffold_state,
    _runtime_variational_bundle,
    _scaffold_resources_for_layouts,
    _state_diagnostic_row,
    _term_label,
    _term_label_set,
    _trajectory_summary,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DynamicsBenchmarkCase,
    DynamicsBenchmarkRow,
    build_dynamics_tuning_provenance,
    json_safe,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor

AVQDS_T_ALGORITHM_ID = "dyn_avqds_t"


def _phase_align_target(*, psi: np.ndarray, target_state: np.ndarray) -> np.ndarray:
    psi_vec = _normalize_state(psi)
    target = _normalize_state(target_state)
    overlap = np.vdot(psi_vec, target)
    if abs(overlap) <= 1.0e-15:
        return target
    return _normalize_state(target * np.exp(-1.0j * float(np.angle(overlap))))


def _target_tangent(*, psi: np.ndarray, target_state: np.ndarray, dt: float) -> np.ndarray:
    if float(dt) <= 0.0:
        raise ValueError("AVQDS-T target tangent requires positive dt")
    aligned = _phase_align_target(psi=psi, target_state=target_state)
    tangent = (aligned - _normalize_state(psi)) / float(dt)
    # Remove the state-parallel component so the comparator focuses on the
    # representable projective tangent, not global phase/norm drift.
    psi_vec = _normalize_state(psi)
    tangent = tangent - psi_vec * np.vdot(psi_vec, tangent)
    return np.asarray(tangent, dtype=complex).reshape(-1)


def _solve_avqds_t_target_tangent_step(
    *,
    executor: CompiledAnsatzExecutor,
    psi_ref: np.ndarray,
    theta_start: np.ndarray,
    target_state: np.ndarray,
    dt: float,
    regularization_lambda: float,
    pinv_relative_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    theta_vec = np.asarray(theta_start, dtype=float).reshape(-1)
    psi, tangents = executor.prepare_state_with_runtime_tangents(theta_vec, psi_ref)
    psi = _normalize_state(psi)
    rhs_state = _target_tangent(psi=psi, target_state=target_state, dt=dt)
    rhs_norm = float(np.linalg.norm(rhs_state))
    param_count = int(theta_vec.size)
    if param_count == 0:
        residual_norm = rhs_norm
        return theta_vec, psi, {
            "theta_dot": [],
            "target_tangent_norm": float(rhs_norm),
            "target_tangent_residual_norm": float(residual_norm),
            "target_tangent_residual_ratio": 0.0 if rhs_norm <= 1.0e-15 else float(residual_norm / rhs_norm),
            "projected_target_tangent_norm": 0.0,
            "delta_norm": 0.0,
            "linear_solve_status": "no_parameters",
            "linear_solve_count": 0,
            "regularization_lambda": float(regularization_lambda),
            "pinv_relative_cutoff": float(pinv_relative_cutoff),
            "retained_rank": 0,
            "parameter_count": 0,
            "tangent_condition_estimate": None,
            "metric_symmetry_max_abs": 0.0,
            "force_norm": 0.0,
            "solve_residual_norm": 0.0,
            "target_tangent_state_parallel_abs": float(abs(np.vdot(psi, rhs_state))),
            "dense_reference_kind": "direct_dense_target_tangent_matrix_no_parameters",
            "state_prep_count": 1,
            "success": True,
            "message": "no runtime parameters available",
        }

    tangent_matrix = np.column_stack(
        [np.asarray(tangents[idx], dtype=complex).reshape(-1) for idx in range(param_count)]
    )
    metric = np.real(tangent_matrix.conj().T @ tangent_matrix)
    force = np.real(tangent_matrix.conj().T @ rhs_state)
    reg = float(max(0.0, regularization_lambda))
    solve_matrix = metric + reg * np.eye(param_count)
    singular_values = np.linalg.svd(solve_matrix, compute_uv=False)
    cutoff = float(max(0.0, pinv_relative_cutoff))
    max_sv = float(np.max(singular_values)) if singular_values.size else 0.0
    retained_rank = int(sum(float(sv) > cutoff * max_sv for sv in singular_values)) if max_sv > 0.0 else 0
    positive = [float(sv) for sv in singular_values if float(sv) > 1.0e-15]
    condition = float(max(positive) / min(positive)) if positive else None
    theta_dot = np.asarray(np.linalg.pinv(solve_matrix, rcond=cutoff) @ force, dtype=float).reshape(-1)
    solve_residual = np.asarray(solve_matrix @ theta_dot - force, dtype=float).reshape(-1)
    projected = tangent_matrix @ theta_dot
    residual = rhs_state - projected
    delta = float(dt) * theta_dot
    theta_next = theta_vec + delta
    final_state = _prepare_scaffold_state(executor, psi_ref, theta_next)
    residual_norm = float(np.linalg.norm(residual))
    projected_norm = float(np.linalg.norm(projected))
    ratio = 0.0 if rhs_norm <= 1.0e-15 else float(residual_norm / rhs_norm)
    return theta_next, final_state, {
        "theta_dot": [float(x) for x in theta_dot.tolist()],
        "target_tangent_norm": float(rhs_norm),
        "target_tangent_residual_norm": float(residual_norm),
        "target_tangent_residual_ratio": float(ratio),
        "projected_target_tangent_norm": float(projected_norm),
        "delta_norm": float(np.linalg.norm(delta)),
        "linear_solve_status": "ok",
        "linear_solve_count": 1,
        "regularization_lambda": float(reg),
        "pinv_relative_cutoff": float(cutoff),
        "retained_rank": int(retained_rank),
        "parameter_count": int(param_count),
        "tangent_condition_estimate": condition,
        "metric_symmetry_max_abs": float(np.max(np.abs(metric - metric.T))) if metric.size else 0.0,
        "force_norm": float(np.linalg.norm(force)),
        "solve_residual_norm": float(np.linalg.norm(solve_residual)),
        "target_tangent_state_parallel_abs": float(abs(np.vdot(psi, rhs_state))),
        "dense_reference_kind": "direct_dense_target_tangent_matrix_regularized_pinv",
        "state_prep_count": 1,
        "success": True,
        "message": "repo-native regularized target-tangent solve",
    }


def _build_avqds_t_correctness_sidecar(
    *,
    case: DynamicsBenchmarkCase,
    avqds_t_steps: Sequence[Mapping[str, Any]],
    append_events: Sequence[Mapping[str, Any]],
    append_candidate_evaluations: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
    candidate_pool: Sequence[Any],
    state_norms: Sequence[float],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    finite_payload = common._json_numeric_values_are_finite(
        {
            "avqds_t_steps": list(avqds_t_steps),
            "append_events": list(append_events),
            "append_candidate_evaluations": list(append_candidate_evaluations),
            "trajectory": list(trajectory),
            "state_norms": list(state_norms),
        }
    )
    checks.append(
        common._check_payload(
            check_id="finite_trajectory_diagnostics",
            check_type="invariant_correctness",
            passed=bool(avqds_t_steps) and bool(finite_payload),
            details={"step_count": len(avqds_t_steps), "trajectory_points": len(trajectory)},
        )
    )

    ratio_bad: list[int] = []
    solve_bad: list[int] = []
    tangent_parallel_bad: list[int] = []
    target_order_bad: list[int] = []
    for step in avqds_t_steps:
        idx = int(step.get("interval_index", len(ratio_bad)))
        rhs_norm = _float_or_none(step.get("target_tangent_norm"))
        residual_norm = _float_or_none(step.get("target_tangent_residual_norm"))
        ratio = _float_or_none(step.get("target_tangent_residual_ratio"))
        if rhs_norm is None or residual_norm is None or ratio is None:
            ratio_bad.append(idx)
        else:
            expected = 0.0 if abs(rhs_norm) <= 1.0e-15 else float(residual_norm / rhs_norm)
            if abs(float(ratio) - expected) > 1.0e-9:
                ratio_bad.append(idx)
        parameter_count = int(step.get("parameter_count", 0) or 0)
        rank = int(step.get("retained_rank", 0) or 0)
        if (
            str(step.get("linear_solve_status")) not in {"ok", "no_parameters"}
            or rank < 0
            or rank > max(parameter_count, 0)
            or _float_or_none(step.get("regularization_lambda")) is None
            or float(step.get("regularization_lambda", 0.0)) < 0.0
            or _float_or_none(step.get("pinv_relative_cutoff")) is None
            or float(step.get("pinv_relative_cutoff", 0.0)) < 0.0
            or _float_or_none(step.get("metric_symmetry_max_abs")) is None
            or float(step.get("metric_symmetry_max_abs", 0.0)) > 1.0e-8
            or _float_or_none(step.get("solve_residual_norm")) is None
        ):
            solve_bad.append(idx)
        tangent_parallel = _float_or_none(step.get("target_tangent_state_parallel_abs"))
        if tangent_parallel is None or float(tangent_parallel) > 1.0e-8:
            tangent_parallel_bad.append(idx)
        if int(step.get("target_product_formula_order", 0) or 0) not in {1, 2}:
            target_order_bad.append(idx)
    checks.append(
        common._check_payload(
            check_id="target_tangent_dense_reference_solve",
            check_type="dense_reference_component_parity",
            passed=bool(avqds_t_steps) and not ratio_bad and not solve_bad and not tangent_parallel_bad,
            details={
                "ratio_bad_interval_indices": ratio_bad,
                "solve_bad_interval_indices": solve_bad,
                "target_tangent_parallel_bad_interval_indices": tangent_parallel_bad,
                "dense_reference_kind": "direct_dense_target_tangent_matrix_regularized_pinv",
            },
        )
    )
    checks.append(
        common._check_payload(
            check_id="product_formula_target_order_policy",
            check_type="target_tangent_construction_correctness",
            passed=bool(avqds_t_steps) and not target_order_bad,
            details={"bad_interval_indices": target_order_bad, "allowed_orders": [1, 2]},
        )
    )

    grouped_labels = {
        _term_label(term, idx)
        for idx, term in enumerate(candidate_pool)
        if str(getattr(term, "execution_mode", "termwise_product") or "termwise_product").strip().lower()
        == "grouped_exact"
    }
    grouped_eval_labels = [
        str(item.get("candidate_label"))
        for item in append_candidate_evaluations
        if str(item.get("candidate_label")) in grouped_labels
    ]
    checks.append(
        common._check_payload(
            check_id="grouped_exact_candidate_exclusion",
            check_type="append_admission_correctness",
            passed=not grouped_eval_labels,
            details={
                "grouped_exact_candidate_excluded_count": len(grouped_labels),
                "grouped_exact_labels_evaluated": grouped_eval_labels,
            },
        )
    )

    evals_by_interval: dict[int, list[Mapping[str, Any]]] = {}
    for item in append_candidate_evaluations:
        evals_by_interval.setdefault(int(item.get("interval_index", -1)), []).append(item)
    events_by_interval = {int(event.get("interval_index", -1)): event for event in append_events}
    append_bad: list[int] = []
    for step in avqds_t_steps:
        idx = int(step.get("interval_index", -1))
        evals = evals_by_interval.get(idx, [])
        threshold = float(step.get("append_target_tangent_residual_ratio_threshold", 0.0))
        min_gain = float(step.get("append_min_residual_ratio_gain", 0.0))
        accepted = bool(step.get("append_accepted", False))
        event = events_by_interval.get(idx)
        base_ratio = None
        if event is not None:
            base_ratio = _float_or_none(event.get("target_tangent_residual_ratio_base"))
        if base_ratio is None and evals:
            base_ratio = _float_or_none(evals[0].get("target_tangent_residual_ratio_base"))
        if base_ratio is None:
            base_ratio = _float_or_none(step.get("target_tangent_residual_ratio"))
        if int(step.get("append_candidate_evaluations", 0)) != len(evals):
            append_bad.append(idx)
            continue
        if base_ratio is not None and float(base_ratio) <= threshold and evals:
            append_bad.append(idx)
            continue
        eligible = [
            ev
            for ev in evals
            if base_ratio is not None
            and _float_or_none(ev.get("target_tangent_residual_ratio")) is not None
            and float(base_ratio) - float(ev["target_tangent_residual_ratio"]) >= min_gain
        ]
        if accepted:
            selected_ratio = None if event is None else _float_or_none(event.get("target_tangent_residual_ratio_selected"))
            best_ratio = min((float(ev["target_tangent_residual_ratio"]) for ev in eligible), default=None)
            if event is None or selected_ratio is None or best_ratio is None or abs(float(selected_ratio) - best_ratio) > 1.0e-9:
                append_bad.append(idx)
        elif eligible:
            append_bad.append(idx)
    checks.append(
        common._check_payload(
            check_id="append_ranking_acceptance_semantics",
            check_type="append_admission_correctness",
            passed=not append_bad,
            details={"bad_interval_indices": append_bad},
        )
    )

    norm_deviations = [abs(float(value) - 1.0) for value in state_norms]
    max_norm_deviation = max(norm_deviations, default=0.0)
    checks.append(
        common._check_payload(
            check_id="state_norm_preservation",
            check_type="invariant_correctness",
            passed=bool(state_norms) and max_norm_deviation <= 1.0e-10,
            details={"state_norm_count": len(state_norms), "max_norm_deviation": max_norm_deviation},
        )
    )

    passed = common._checks_pass(checks)
    return json_safe(
        {
            "schema": "avqds_t_correctness_v1",
            "algorithm_id": AVQDS_T_ALGORITHM_ID,
            "family": str(case.family),
            "case_id": str(case.case_id),
            "sidecar_name": common.CORRECTNESS_SIDECAR_FILENAMES[AVQDS_T_ALGORITHM_ID],
            "support_scope": "avqds_t_target_tangent_append_and_invariant_correctness",
            "sidecar_kind": "dense_reference_component_parity_and_invariant_correctness",
            "status": "ok" if passed else "failed",
            "passed": bool(passed),
            "required_status": "passed",
            "check_count": int(len(checks)),
            "checks": checks,
            "exact_data_policy": "benchmark_exact_fields_reporting_only_not_target_tangent_or_append_decision",
            "physical_error_policy": "additive_correctness_provenance_not_a_physical_error_column",
            "controller_decisions_modified": False,
            "exact_reference_controller_inputs": False,
        }
    )


def _build_avqds_t_payload(
    *,
    case: DynamicsBenchmarkCase,
    runtime_input: Any,
    command: Sequence[str],
) -> dict[str, Any]:
    flow = _native_hamiltonian_flow(case, runtime_input)
    terms = flow.terms_for_interval(float(flow.times[0]), float(flow.times[min(1, len(flow.times) - 1)]))
    hmat = np.asarray(flow.static_hmat, dtype=complex)
    psi_initial = _normalize_state(runtime_input.psi_initial)
    if hmat.shape[0] != psi_initial.size:
        raise ValueError(
            f"Hamiltonian dimension {hmat.shape} does not match state size {psi_initial.size}"
        )
    current_terms, layout, theta, psi_ref, executor, _drive_aligned_scaffold = _runtime_variational_bundle(runtime_input)
    candidate_pool = tuple(getattr(runtime_input, "candidate_pool_terms", ()) or ())
    times = flow.times
    if int(times.size) < 2:
        raise ValueError("AVQDS-T comparator requires at least two time points")
    exact_states = flow.exact_states
    observable_context = dict(flow.observable_context or {})
    action_by_label: dict[str, Any] = {}

    regularization_lambda = _metadata_float(case, "avqds_t_regularization_lambda", 1.0e-8, minimum=0.0)
    pinv_relative_cutoff = _metadata_float(case, "avqds_t_pinv_relative_cutoff", 1.0e-10, minimum=0.0)
    target_order = _metadata_optional_int(case, "avqds_t_target_product_formula_order", 2, minimum=1)
    if target_order is None:
        target_order = 2
    append_threshold = _metadata_float(
        case,
        "avqds_t_append_target_tangent_residual_ratio_threshold",
        1.0e-3,
        minimum=0.0,
    )
    append_min_gain = _metadata_float(case, "avqds_t_append_min_residual_ratio_gain", 1.0e-5, minimum=0.0)
    candidate_limit = _metadata_optional_int(case, "avqds_t_append_candidate_limit", 4, minimum=0)

    theta_current = np.asarray(theta, dtype=float).reshape(-1)
    current_state = _prepare_scaffold_state(executor, psi_ref, theta_current)
    trajectory: list[dict[str, Any]] = [
        _state_diagnostic_row(
            checkpoint_index=0,
            time_value=float(times[0]),
            method="generic_avqds_t",
            method_kind="avqds_t",
            state=current_state,
            exact_state=exact_states[0],
            hmat=flow.hmat_at_time(float(times[0])),
            **observable_context,
            extra={
                "runtime_parameter_count": int(theta_current.size),
                "logical_block_count": int(getattr(layout, "logical_parameter_count")),
                "avqds_t_step_index": None,
                "avqds_t_linear_solve_count": 0,
                "avqds_t_state_prep_count": 0,
                "target_tangent_residual_ratio": None,
                "append_accepted": None,
                "append_candidate_evaluations": 0,
            },
        )
    ]
    avqds_t_steps: list[dict[str, Any]] = []
    append_events: list[dict[str, Any]] = []
    append_candidate_evaluations: list[dict[str, Any]] = []
    interval_layouts: list[Any] = []
    state_norms: list[float] = [float(np.linalg.norm(current_state))]

    for interval_index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        dt = float(right - left)
        terms_step = flow.terms_for_interval(float(left), float(right))
        for label, action in _compiled_pauli_actions_by_label(terms_step).items():
            action_by_label.setdefault(str(label), action)
        target = _apply_product_formula_step(
            terms=terms_step,
            action_by_label=action_by_label,
            psi=current_state,
            dt=float(dt),
            order=int(target_order),
        )
        base_theta, base_state, base_fit = _solve_avqds_t_target_tangent_step(
            executor=executor,
            psi_ref=psi_ref,
            theta_start=theta_current,
            target_state=target,
            dt=dt,
            regularization_lambda=regularization_lambda,
            pinv_relative_cutoff=pinv_relative_cutoff,
        )
        selected = {
            "theta": base_theta,
            "state": base_state,
            "fit": base_fit,
            "terms": current_terms,
            "layout": layout,
            "executor": executor,
            "candidate_pool_index": None,
            "candidate_label": None,
        }
        candidate_eval_count = 0
        used_labels = _term_label_set(current_terms)
        base_ratio = float(base_fit["target_tangent_residual_ratio"])
        if base_ratio > float(append_threshold):
            for candidate_index in _candidate_indices_for_avqds(
                candidate_pool=candidate_pool,
                used_labels=used_labels,
                candidate_limit=candidate_limit,
            ):
                candidate = candidate_pool[int(candidate_index)]
                candidate_terms = tuple(current_terms) + (candidate,)
                candidate_layout = _build_layout_for_terms(candidate_terms, reference_layout=layout)
                candidate_theta = _copy_theta_by_layout_blocks(
                    old_theta=theta_current,
                    old_layout=layout,
                    new_layout=candidate_layout,
                )
                candidate_executor = _compiled_executor_for_terms(candidate_terms, candidate_layout)
                cand_theta, cand_state, cand_fit = _solve_avqds_t_target_tangent_step(
                    executor=candidate_executor,
                    psi_ref=psi_ref,
                    theta_start=candidate_theta,
                    target_state=target,
                    dt=dt,
                    regularization_lambda=regularization_lambda,
                    pinv_relative_cutoff=pinv_relative_cutoff,
                )
                candidate_eval_count += 1
                cand_ratio = float(cand_fit["target_tangent_residual_ratio"])
                eval_row = {
                    "interval_index": int(interval_index),
                    "candidate_pool_index": int(candidate_index),
                    "candidate_label": _term_label(candidate, candidate_index),
                    "target_tangent_residual_ratio": float(cand_ratio),
                    "target_tangent_residual_ratio_base": float(base_ratio),
                    "target_tangent_residual_ratio_improvement": float(base_ratio - cand_ratio),
                    "runtime_parameter_count": int(cand_theta.size),
                    "logical_block_count": int(getattr(candidate_layout, "logical_parameter_count")),
                }
                append_candidate_evaluations.append(eval_row)
                if base_ratio - cand_ratio >= float(append_min_gain):
                    current_selected_ratio = float(selected["fit"]["target_tangent_residual_ratio"])
                    if selected["candidate_pool_index"] is None or cand_ratio < current_selected_ratio:
                        selected = {
                            "theta": cand_theta,
                            "state": cand_state,
                            "fit": cand_fit,
                            "terms": candidate_terms,
                            "layout": candidate_layout,
                            "executor": candidate_executor,
                            "candidate_pool_index": int(candidate_index),
                            "candidate_label": _term_label(candidate, candidate_index),
                        }

        append_accepted = selected["candidate_pool_index"] is not None
        theta_current = np.asarray(selected["theta"], dtype=float).reshape(-1)
        current_state = np.asarray(selected["state"], dtype=complex).reshape(-1)
        current_terms = tuple(selected["terms"])
        layout = selected["layout"]
        executor = selected["executor"]
        interval_layouts.append(layout)
        fit = dict(selected["fit"])
        state_norm_after = float(np.linalg.norm(current_state))
        state_norms.append(state_norm_after)
        step_payload = {
            "interval_index": int(interval_index),
            "time_start": float(left),
            "time_stop": float(right),
            "dt": float(dt),
            "append_accepted": bool(append_accepted),
            "append_candidate_evaluations": int(candidate_eval_count),
            "target_policy": "product_formula_circuit_step",
            "target_product_formula_order": int(target_order),
            "append_target_tangent_residual_ratio_threshold": float(append_threshold),
            "append_min_residual_ratio_gain": float(append_min_gain),
            "state_norm_after": float(state_norm_after),
            **fit,
        }
        if append_accepted:
            event = {
                "interval_index": int(interval_index),
                "candidate_pool_index": int(selected["candidate_pool_index"]),
                "candidate_label": str(selected["candidate_label"]),
                "target_tangent_residual_ratio_base": float(base_ratio),
                "target_tangent_residual_ratio_selected": float(fit["target_tangent_residual_ratio"]),
                "runtime_parameter_count": int(theta_current.size),
                "logical_block_count": int(getattr(layout, "logical_parameter_count")),
            }
            append_events.append(event)
            step_payload["append_event"] = event
        avqds_t_steps.append(step_payload)
        trajectory.append(
            _state_diagnostic_row(
                checkpoint_index=int(interval_index) + 1,
                time_value=float(right),
                method="generic_avqds_t",
                method_kind="avqds_t",
                state=current_state,
                exact_state=exact_states[int(interval_index) + 1],
                hmat=flow.hmat_at_time(float(right)),
                **observable_context,
                extra={
                    "runtime_parameter_count": int(theta_current.size),
                    "logical_block_count": int(getattr(layout, "logical_parameter_count")),
                    "avqds_t_step_index": int(interval_index),
                    "avqds_t_linear_solve_count": int(fit["linear_solve_count"]),
                    "avqds_t_state_prep_count": int(fit["state_prep_count"]),
                    "target_tangent_residual_ratio": float(fit["target_tangent_residual_ratio"]),
                    "target_tangent_residual_norm": float(fit["target_tangent_residual_norm"]),
                    "append_accepted": bool(append_accepted),
                    "append_candidate_evaluations": int(candidate_eval_count),
                },
            )
        )

    summary = _trajectory_summary(trajectory)
    residual_ratios = [step.get("target_tangent_residual_ratio") for step in avqds_t_steps]
    linear_solve_total = int(sum(int(step.get("linear_solve_count", 0)) for step in avqds_t_steps))
    state_prep_total = int(sum(int(step.get("state_prep_count", 0)) for step in avqds_t_steps))
    resources = _scaffold_resources_for_layouts(
        state_layout=layout,
        interval_layouts=interval_layouts,
        state_scope="generic_avqds_t_state_scaffold",
        horizon_scope="generic_avqds_t_scaffold_epoch_sum",
        extra={
            "avqds_t_step_count": int(len(avqds_t_steps)),
            "avqds_t_linear_solve_total": int(linear_solve_total),
            "avqds_t_state_prep_total": int(state_prep_total),
            "append_events_total": int(len(append_events)),
            "append_candidate_evaluations_total": int(len(append_candidate_evaluations)),
        },
    )
    metrics = {
        "method_kind": "avqds_t",
        "decision_mode": "generic_product_formula_target_tangent_v1",
        "decision_data_flow": "ideal_target_tangent_estimator_for_product_formula_target_circuit",
        "candidate_pool_complete": True,
        "candidate_pool_completeness": _candidate_pool_completeness(runtime_input),
        "candidate_pool_size": int(len(candidate_pool)),
        "append_events_total": int(len(append_events)),
        "append_candidate_evaluations_total": int(len(append_candidate_evaluations)),
        "final_runtime_parameter_count": int(theta_current.size),
        "final_logical_block_count": int(getattr(layout, "logical_parameter_count")),
        "avqds_t_linear_solve_total": int(linear_solve_total),
        "avqds_t_step_count": int(len(avqds_t_steps)),
        "avqds_t_state_prep_total": int(state_prep_total),
        "target_tangent_residual_ratio_final": _float_or_none(residual_ratios[-1]) if residual_ratios else None,
        "target_tangent_residual_ratio_max": _max_or_none(residual_ratios),
        "exact_tangent_target_depends_on_exact_interval_propagation": False,
        "target_tangent_policy": "product_formula_circuit_step",
        "target_product_formula_order": int(target_order),
        "append_scoring_uses_exact_reference": False,
        "uses_statevector_as_ideal_target_tangent_estimator": True,
        "exact_fields_reporting_only": True,
    }
    tuning = build_dynamics_tuning_provenance(
        case=case,
        algorithm_id=AVQDS_T_ALGORITHM_ID,
        settings_kind="comparator",
        settings_payload={
            "regularization_lambda": float(regularization_lambda),
            "pinv_relative_cutoff": float(pinv_relative_cutoff),
            "target_policy": "product_formula_circuit_step",
            "target_product_formula_order": int(target_order),
            "append_target_tangent_residual_ratio_threshold": float(append_threshold),
            "append_min_residual_ratio_gain": float(append_min_gain),
            "append_candidate_limit": None if candidate_limit is None else int(candidate_limit),
        },
        settings_source=common.metadata_override_settings_source(
            case,
            (
                "avqds_t_regularization_lambda",
                "avqds_t_pinv_relative_cutoff",
                "avqds_t_target_product_formula_order",
                "avqds_t_append_target_tangent_residual_ratio_threshold",
                "avqds_t_append_min_residual_ratio_gain",
                "avqds_t_append_candidate_limit",
            ),
        ),
        locked=False,
    )
    parameter_manifest = _generic_parameter_manifest(
        case=case,
        runtime_input=runtime_input,
        algorithm_id=AVQDS_T_ALGORITHM_ID,
        times=times,
        terms=terms,
        flow=flow,
    )
    parameter_manifest["tuning_provenance"] = dict(tuning)
    avqds_t_correctness = _build_avqds_t_correctness_sidecar(
        case=case,
        avqds_t_steps=avqds_t_steps,
        append_events=append_events,
        append_candidate_evaluations=append_candidate_evaluations,
        trajectory=trajectory,
        candidate_pool=candidate_pool,
        state_norms=state_norms,
    )
    return json_safe(
        {
            "schema_version": "generic_avqds_t_benchmark_v1",
            "case": case.to_dict(),
            "row_contract": {
                "qpu_faithful": True,
                "exact_assisted": False,
                "diagnostic": True,
            },
            "parameter_manifest": parameter_manifest,
            "tuning_provenance": tuning,
            "command": list(command),
            "trajectory": trajectory,
            "avqds_t_steps": avqds_t_steps,
            "append_events": append_events,
            "append_candidate_evaluations": append_candidate_evaluations,
            "avqds_t_correctness": avqds_t_correctness,
            "summary": summary,
            "metrics": metrics,
            "resources": resources,
            "compile_audit": _compile_audit_from_resources(resources),
            "provenance": {
                "route_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "benchmark_only": True,
                "runner_module": "pipelines.time_dynamics.benchmarks.legacy_native",
                "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_target_tangent",
                "comparator_kernel": "repo_native_avqds_t_product_formula_target_tangent",
                "decision_data_flow": "ideal_target_tangent_estimator_for_product_formula_target_circuit",
                "controller_paths_called": False,
                "controller_decisions_modified": False,
                "exact_reference_controller_inputs": False,
                "exact_tangent_targets_used_by_comparator": False,
                "append_scoring_uses_exact_reference": False,
                "uses_statevector_as_ideal_target_tangent_estimator": True,
                "exact_fields_reporting_only": True,
            },
        }
    )


def run_avqds_t_benchmark_row(*, case: DynamicsBenchmarkCase, output_dir: Path) -> DynamicsBenchmarkRow:
    return common.run_native_generic_comparator_row(
        case=case,
        algorithm_id=AVQDS_T_ALGORITHM_ID,
        output_dir=Path(output_dir),
        payload_builder=_build_avqds_t_payload,
    )




# ---- controller_ablation_matrix.py (collapsed legacy implementation) ----
"""Generic strict controller ablation matrix runner.

The matrix is a reporting-side benchmark surface for Hamiltonian-generic
realtime controller variants.  Adaptive variants run with controller exact
inputs off and with post-run exact references marked diagnostic-only.  If a
variant's decision telemetry leaks exact target/reference data, the runner fails
closed for that variant.
"""


import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.benchmarks import common as rows_mod
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
    DYNAMICS_SKIPPED_TUNING_SOURCE,
    DynamicsBenchmarkCase,
    DynamicsBenchmarkRow,
    DynamicsTableFields,
    build_dynamics_tuning_provenance,
    dynamics_table_bundle_payload,
    dynamics_tuning_class,
    json_safe,
)
from pipelines.time_dynamics.tables.generic_dynamics_cases import get_generic_dynamics_case
from pipelines.time_dynamics.tables.table_lock_contract import (
    build_locked_or_default_tuning_provenance,
    controller_cli_tokens_for_case,
    table_lock_provenance_for_case,
    with_class_settings_lock_manifest,
)
from pipelines.time_dynamics.tables.generic_dynamics_tables import (
    FULL_CONTROLLER_ABLATION_VARIANT,
    FULL_CONTROLLER_ALGORITHM_ID,
    write_generic_dynamics_table_summaries,
)
from pipelines.time_dynamics.legacy.checkpoint_types import (
    DECISION_DATA_FLOW_EXACT_ASSISTED,
    strict_qpu_faithful_decision_contract,
)

GENERIC_CONTROLLER_ABLATION_MATRIX_SCHEMA = "generic_controller_ablation_matrix_v1"
GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM = "dyn_controller_ablation_matrix"
PAPER_II_PRIMARY_INTEGRATOR_POLICY = "rk4"
AUTO_EULER_RK4_INTEGRATOR_POLICY = "auto_euler_rk4"
HH_AUTO_EULER_GUARDRAIL_ID = "hh_auto_euler_observable_guardrails_v1"
HH_AUTO_EULER_GUARDRAIL_THRESHOLDS = {
    "integrator_euler_site_span_max": 1.0e-2,
    "integrator_euler_primary_density_span_max": 2.0e-2,
    "integrator_euler_energy_span_max": 2.0e-3,
}


class _LazyRealtimeModule:
    """Lazy proxy for the heavy native realtime controller module.

    Tests and compatibility shims may replace the module-level ``realtime``
    object with a fake.  Keeping this variable lazy prevents benchmark registry
    imports from paying the HH controller import cost before a row actually runs.
    """

    _module: Any | None = None

    def _load(self) -> Any:
        if self._module is None:
            self._module = importlib.import_module(
                "pipelines.time_dynamics.runners.generic_from_adapt_artifact"
            )
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


realtime: Any = _LazyRealtimeModule()


@dataclass(frozen=True)
class GenericControllerAblationVariant:
    variant_id: str
    algorithm_id: str
    method_label: str
    disabled_feature: str | None
    controller_mode: str = "observable_v1"
    strict_qpu_faithful: bool = True
    append_enabled: bool = True
    prune_mode: str = "schur_projected_shadow_v1"
    integrator_policy: str = PAPER_II_PRIMARY_INTEGRATOR_POLICY
    confirm_score_mode: str = "compressed_whitened_v1"
    description: str = ""
    diagnostic_ladder_id: str | None = None
    diagnostic_ladder_stage: int | None = None
    include_in_default_matrix: bool = True
    force_requested_knobs: bool = False
    hh_only: bool = False
    paper_promotion_eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "variant_id": self.variant_id,
                "algorithm_id": self.algorithm_id,
                "method_label": self.method_label,
                "disabled_feature": self.disabled_feature,
                "controller_mode": self.controller_mode,
                "strict_qpu_faithful": bool(self.strict_qpu_faithful),
                "append_enabled": bool(self.append_enabled),
                "prune_mode": self.prune_mode,
                "integrator_policy": self.integrator_policy,
                "confirm_score_mode": self.confirm_score_mode,
                "description": self.description,
                "diagnostic_ladder_id": self.diagnostic_ladder_id,
                "diagnostic_ladder_stage": self.diagnostic_ladder_stage,
                "include_in_default_matrix": bool(self.include_in_default_matrix),
                "force_requested_knobs": bool(self.force_requested_knobs),
                "hh_only": bool(self.hh_only),
                "paper_promotion_eligible": bool(self.paper_promotion_eligible),
            }
        )


_CONTROLLER_ABLATION_VARIANTS: tuple[GenericControllerAblationVariant, ...] = (
    GenericControllerAblationVariant(
        variant_id=FULL_CONTROLLER_ABLATION_VARIANT,
        algorithm_id=FULL_CONTROLLER_ALGORITHM_ID,
        method_label="Full strict checkpoint controller",
        disabled_feature=None,
        description="Strict ideal-observable controller with append, prune, fixed RK4 integration, and compressed confirmation enabled.",
    ),
    GenericControllerAblationVariant(
        variant_id="fixed_scaffold",
        algorithm_id="dyn_controller_fixed_scaffold",
        method_label="Fixed scaffold control",
        disabled_feature="adaptive_controller",
        controller_mode="off",
        strict_qpu_faithful=False,
        append_enabled=False,
        prune_mode="off",
        integrator_policy=PAPER_II_PRIMARY_INTEGRATOR_POLICY,
        confirm_score_mode="exact_gain_ratio",
        description="No adaptive controller decisions; propagates the seed scaffold with fixed RK4, exact inputs off, and diagnostic exact overlays only.",
    ),
    GenericControllerAblationVariant(
        variant_id="no_append",
        algorithm_id="dyn_controller_no_append",
        method_label="No append ablation",
        disabled_feature="append",
        append_enabled=False,
        description="Full strict controller with append candidate generation disabled.",
    ),
    GenericControllerAblationVariant(
        variant_id="no_pruning",
        algorithm_id="dyn_controller_no_pruning",
        method_label="No pruning ablation",
        disabled_feature="pruning",
        prune_mode="off",
        description="Full strict controller with pruning disabled.",
    ),
    GenericControllerAblationVariant(
        variant_id="fixed_integrator_policy",
        algorithm_id="dyn_controller_fixed_integrator",
        method_label="Appendix diagnostic: fixed Euler integrator",
        disabled_feature="auto_integrator_policy",
        integrator_policy="euler",
        description="Appendix-only legacy diagnostic using fixed Euler instead of the Paper-II primary fixed RK4 convention.",
        include_in_default_matrix=False,
        paper_promotion_eligible=False,
    ),
    GenericControllerAblationVariant(
        variant_id="no_residual_split",
        algorithm_id="dyn_controller_no_residual_split",
        method_label="No residual-split confirmation ablation",
        disabled_feature="compressed_residual_split_confirmation",
        confirm_score_mode="exact_gain_ratio",
        description="Full strict controller with compressed whitened confirmation disabled in favor of raw gain-ratio confirmation.",
    ),
    GenericControllerAblationVariant(
        variant_id="hh_recovery_s1_rk4_no_append_no_prune",
        algorithm_id="dyn_hh_recovery_s1_rk4_no_append_no_prune",
        method_label="HH recovery S1: RK4 fixed scaffold",
        disabled_feature="append_and_prune",
        append_enabled=False,
        prune_mode="off",
        integrator_policy="rk4",
        description="HH-only diagnostic ladder stage 1: exact-free observable controller, fixed RK4, append disabled, prune disabled.",
        diagnostic_ladder_id="hh_recovery_ladder_v1",
        diagnostic_ladder_stage=1,
        include_in_default_matrix=False,
        force_requested_knobs=True,
        hh_only=True,
        paper_promotion_eligible=False,
    ),
    GenericControllerAblationVariant(
        variant_id="hh_recovery_s2_rk4_append_only",
        algorithm_id="dyn_hh_recovery_s2_rk4_append_only",
        method_label="HH recovery S2: RK4 append only",
        disabled_feature="pruning",
        append_enabled=True,
        prune_mode="off",
        integrator_policy="rk4",
        description="HH-only diagnostic ladder stage 2: exact-free observable controller, fixed RK4, append enabled, prune disabled.",
        diagnostic_ladder_id="hh_recovery_ladder_v1",
        diagnostic_ladder_stage=2,
        include_in_default_matrix=False,
        force_requested_knobs=True,
        hh_only=True,
        paper_promotion_eligible=False,
    ),
    GenericControllerAblationVariant(
        variant_id="hh_recovery_s3_rk4_append_prune",
        algorithm_id="dyn_hh_recovery_s3_rk4_append_prune",
        method_label="HH recovery S3: RK4 append+prune",
        disabled_feature=None,
        append_enabled=True,
        prune_mode="schur_projected_shadow_v1",
        integrator_policy="rk4",
        description="HH-only diagnostic ladder stage 3: exact-free observable controller, fixed RK4, append and prune enabled.",
        diagnostic_ladder_id="hh_recovery_ladder_v1",
        diagnostic_ladder_stage=3,
        include_in_default_matrix=False,
        force_requested_knobs=True,
        hh_only=True,
        paper_promotion_eligible=False,
    ),
    GenericControllerAblationVariant(
        variant_id="hh_recovery_s4_auto_append_prune",
        algorithm_id="dyn_hh_recovery_s4_auto_append_prune",
        method_label="HH recovery S4: auto append+prune",
        disabled_feature=None,
        append_enabled=True,
        prune_mode="schur_projected_shadow_v1",
        integrator_policy=AUTO_EULER_RK4_INTEGRATOR_POLICY,
        description="HH-only diagnostic ladder stage 4: exact-free observable controller, adaptive Euler/RK4, append and prune enabled.",
        diagnostic_ladder_id="hh_recovery_ladder_v1",
        diagnostic_ladder_stage=4,
        include_in_default_matrix=False,
        force_requested_knobs=True,
        hh_only=True,
        paper_promotion_eligible=False,
    ),
    GenericControllerAblationVariant(
        variant_id="hh_recovery_s5_auto_no_append_no_prune",
        algorithm_id="dyn_hh_recovery_s5_auto_no_append_no_prune",
        method_label="HH recovery S5: auto fixed scaffold",
        disabled_feature="append_and_prune",
        append_enabled=False,
        prune_mode="off",
        integrator_policy=AUTO_EULER_RK4_INTEGRATOR_POLICY,
        description="HH-only diagnostic ladder stage 5: exact-free observable controller, adaptive Euler/RK4, append disabled, prune disabled. This isolates integrator-policy failure from scaffold mutation.",
        diagnostic_ladder_id="hh_recovery_ladder_v1",
        diagnostic_ladder_stage=5,
        include_in_default_matrix=False,
        force_requested_knobs=True,
        hh_only=True,
        paper_promotion_eligible=False,
    ),
)


def controller_ablation_variants() -> tuple[GenericControllerAblationVariant, ...]:
    return tuple(_CONTROLLER_ABLATION_VARIANTS)


def default_controller_ablation_variants() -> tuple[GenericControllerAblationVariant, ...]:
    return tuple(variant for variant in _CONTROLLER_ABLATION_VARIANTS if bool(variant.include_in_default_matrix))


def get_controller_ablation_variant(variant_id: str) -> GenericControllerAblationVariant:
    for variant in _CONTROLLER_ABLATION_VARIANTS:
        if variant.variant_id == str(variant_id):
            return variant
    known = ", ".join(variant.variant_id for variant in _CONTROLLER_ABLATION_VARIANTS)
    raise ValueError(f"unknown generic controller ablation variant {variant_id!r}; known: {known}")


def _drive_argv_for_case(case: DynamicsBenchmarkCase) -> list[str]:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    drive = metadata.get("drive", {}) if isinstance(metadata.get("drive", {}), Mapping) else {}
    enable_drive = bool(drive.get("enable_drive", metadata.get("enable_drive", False)))
    disable_drive = bool(drive.get("disable_drive", metadata.get("disable_drive", False)))
    if enable_drive and disable_drive:
        raise ValueError(f"case {case.case_id}: enable_drive and disable_drive cannot both be true")
    if disable_drive:
        return ["--disable-drive"]
    if not enable_drive:
        return []
    argv = [
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
    if bool(drive.get("include_identity", metadata.get("drive_include_identity", False))):
        argv.append("--drive-include-identity")
    return argv


def _hh_auto_euler_guardrail_tokens(
    *,
    case: DynamicsBenchmarkCase,
    variant: GenericControllerAblationVariant,
) -> list[str]:
    if str(case.family).strip().lower() != "hh":
        return []
    if str(variant.integrator_policy).strip().lower() != "auto_euler_rk4":
        return []
    return [
        "--checkpoint-controller-integrator-euler-site-span-max",
        str(float(HH_AUTO_EULER_GUARDRAIL_THRESHOLDS["integrator_euler_site_span_max"])),
        "--checkpoint-controller-integrator-euler-primary-density-span-max",
        str(float(HH_AUTO_EULER_GUARDRAIL_THRESHOLDS["integrator_euler_primary_density_span_max"])),
        "--checkpoint-controller-integrator-euler-energy-span-max",
        str(float(HH_AUTO_EULER_GUARDRAIL_THRESHOLDS["integrator_euler_energy_span_max"])),
    ]


def _base_realtime_argv(
    *,
    case: DynamicsBenchmarkCase,
    variant: GenericControllerAblationVariant,
    raw_payload_json: Path,
) -> list[str]:
    full_policy_tokens: list[str] = []
    if bool(variant.strict_qpu_faithful):
        # Disabled-feature variants inherit the single class-locked full policy
        # first.  Variant-specific disabling tokens are appended after this
        # block, so a locked policy cannot accidentally re-enable append/prune
        # or undo a fixed-integrator ablation.
        full_policy_tokens = controller_cli_tokens_for_case(
            case,
            algorithm_id=FULL_CONTROLLER_ALGORITHM_ID,
            settings_kind="controller",
            variant_id=FULL_CONTROLLER_ABLATION_VARIANT,
        )
    argv = [
        "--artifact-json",
        str(case.artifact_json),
        "--output-json",
        str(raw_payload_json),
        "--run-tag",
        f"{case.case_id}_{variant.algorithm_id}",
        "--loader-mode",
        str(case.loader_mode),
        "--generator-family",
        str(case.generator_family),
        "--fallback-family",
        str(case.fallback_family),
        "--append-pool-family",
        str(case.append_pool_family),
        "--num-times",
        str(int(case.num_times)),
        "--t-final",
        str(float(case.t_final)),
        "--checkpoint-controller-mode",
        "observable_v1",
        "--checkpoint-controller-exact-input-mode",
        "off",
        "--diagnostic-exact-reference-mode",
        "benchmark_exact",
        "--checkpoint-controller-prune-mode",
        "schur_projected_shadow_v1",
        "--checkpoint-controller-integrator-policy",
        PAPER_II_PRIMARY_INTEGRATOR_POLICY,
        "--checkpoint-controller-confirm-score-mode",
        "compressed_whitened_v1",
        "--compile-audit-mode",
        "final_scaffold",
        "--compile-audit-backend-name",
        "FakeMarrakesh",
        "--compile-audit-seed-transpiler",
        "7",
        "--compile-audit-optimization-level",
        "2",
        "--compile-audit-preferred-fake-backends",
        "FakeMarrakesh",
    ]
    argv.extend(_drive_argv_for_case(case))
    argv.extend(full_policy_tokens)
    if bool(variant.force_requested_knobs):
        argv.extend(["--checkpoint-controller-mode", str(variant.controller_mode)])
        argv.extend(["--checkpoint-controller-prune-mode", str(variant.prune_mode)])
        argv.extend(["--checkpoint-controller-integrator-policy", str(variant.integrator_policy)])
        argv.extend(["--checkpoint-controller-confirm-score-mode", str(variant.confirm_score_mode)])
        argv.append(
            "--checkpoint-controller-append-enabled"
            if bool(variant.append_enabled)
            else "--no-checkpoint-controller-append-enabled"
        )
    elif str(variant.controller_mode) != "observable_v1":
        argv.extend(["--checkpoint-controller-mode", str(variant.controller_mode)])
    if not bool(variant.force_requested_knobs) and str(variant.prune_mode) == "off":
        argv.extend(["--checkpoint-controller-prune-mode", "off"])
    if (
        not bool(variant.force_requested_knobs)
        and str(variant.integrator_policy) != PAPER_II_PRIMARY_INTEGRATOR_POLICY
    ):
        argv.extend(["--checkpoint-controller-integrator-policy", str(variant.integrator_policy)])
    if not bool(variant.force_requested_knobs) and str(variant.confirm_score_mode) != "compressed_whitened_v1":
        argv.extend(["--checkpoint-controller-confirm-score-mode", str(variant.confirm_score_mode)])
    if bool(variant.strict_qpu_faithful):
        argv.append("--checkpoint-controller-strict-qpu-faithful")
    if (not bool(variant.force_requested_knobs)) and not bool(variant.append_enabled):
        argv.append("--no-checkpoint-controller-append-enabled")
    argv.extend(_hh_auto_euler_guardrail_tokens(case=case, variant=variant))
    # Final guardrails are deliberately last so class locks or stage overrides
    # cannot reintroduce controller exact inputs.
    argv.extend(["--checkpoint-controller-exact-input-mode", "off"])
    argv.extend(["--diagnostic-exact-reference-mode", "benchmark_exact"])
    return argv


def _policy_tuning_algorithm_id(variant: GenericControllerAblationVariant) -> str:
    return (
        FULL_CONTROLLER_ALGORITHM_ID
        if bool(variant.strict_qpu_faithful)
        else str(variant.algorithm_id)
    )


def _policy_tuning_variant_id(variant: GenericControllerAblationVariant) -> str:
    return (
        FULL_CONTROLLER_ABLATION_VARIANT
        if bool(variant.strict_qpu_faithful)
        else str(variant.variant_id)
    )


def _command_for_variant(
    *,
    case: DynamicsBenchmarkCase,
    variant: GenericControllerAblationVariant,
    raw_payload_json: Path,
) -> tuple[list[str], list[str]]:
    argv = _base_realtime_argv(case=case, variant=variant, raw_payload_json=raw_payload_json)
    command = [sys.executable, "-m", "pipelines.time_dynamics.runners.generic_from_adapt_artifact", *argv]
    return command, argv


def _case_drive_mapping(case: DynamicsBenchmarkCase) -> Mapping[str, Any]:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    drive = metadata.get("drive", {}) if isinstance(metadata.get("drive", {}), Mapping) else {}
    return drive


def _case_drive_enabled(case: DynamicsBenchmarkCase) -> bool:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    drive = _case_drive_mapping(case)
    return bool(drive.get("enable_drive", metadata.get("enable_drive", False))) and not bool(
        drive.get("disable_drive", metadata.get("disable_drive", False))
    )


def _case_drive_amplitude(case: DynamicsBenchmarkCase) -> float | None:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    drive = _case_drive_mapping(case)
    value = drive.get("A", drive.get("drive_A", metadata.get("drive_A", None)))
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _case_seed_track(case: DynamicsBenchmarkCase) -> str:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    seed_lock = metadata.get("seed_lock", {}) if isinstance(metadata.get("seed_lock", {}), Mapping) else {}
    return str(seed_lock.get("seed_track", metadata.get("seed_track", ""))).strip().lower()


def _validate_recovery_variant_case(*, case: DynamicsBenchmarkCase, variant: GenericControllerAblationVariant) -> None:
    """Keep explicit diagnostic recovery variants off all-family production matrices."""

    if not bool(variant.hh_only):
        return
    violations: list[str] = []
    if str(case.family) != "hh":
        violations.append(f"family={case.family!r}")
    if abs(float(case.t_final) - 8.0) > 1.0e-9:
        violations.append(f"t_final={case.t_final!r}")
    if int(case.num_times) != 321:
        violations.append(f"num_times={case.num_times!r}")
    if not _case_drive_enabled(case):
        violations.append("drive_enabled=false")
    amplitude = _case_drive_amplitude(case)
    if amplitude is None or all(abs(amplitude - allowed) > 1.0e-9 for allowed in (0.2, 0.6)):
        violations.append(f"drive_A={amplitude!r}")
    seed_track = _case_seed_track(case)
    if seed_track and seed_track not in {"snake", "oldgood", "old_good", "legacy_goodseed", "goodseed"}:
        violations.append(f"seed_track={seed_track!r}")
    if violations:
        joined = "; ".join(violations)
        raise ValueError(
            f"HH recovery variant {variant.variant_id!r} is diagnostic-only and requires HH L2 "
            f"t=8 dt321 driven A=0.2/A=0.6 SNAKE/old-good cases; got {joined}"
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _decision_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    ledger = payload.get("ledger", [])
    if isinstance(ledger, list):
        return [dict(row) for row in ledger if isinstance(row, Mapping)]
    trajectory = payload.get("trajectory", [])
    return [dict(row) for row in trajectory if isinstance(row, Mapping)] if isinstance(trajectory, list) else []


_EXACT_CONTROLLER_INPUT_MODES = {"benchmark_exact", "benchmark", "exact"}


def _mode_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()



def _summary_decision_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract decision-path summary fields without serialized exact-forecast defaults.

    Realtime payload summaries currently include many `exact_forecast_*`
    configuration/default counters even when exact-forecast decision logic is
    disabled.  Those serialized defaults/counters are not decision data when
    the exact forecast guardrail is off and the route reports no future-exact
    decision use.  The strict contract should still see active exact-decision
    flags, exact audit helpers, non-off exact forecast guardrails, and
    exact-forecast veto counts only when exact forecast logic is active.
    """

    summary = dict(_mapping(payload.get("summary", {})))
    out: dict[str, Any] = {}
    for key in (
        "mode",
        "controller_exact_input_mode",
        "controller_reference_mode",
        "reference_mode",
        "decision_data_flow",
        "uses_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
        "uses_statevector_as_ideal_observable_estimator",
        "strict_measurement_oracle_certified",
        "reference_enabled",
        "exact_decision_checkpoints",
        "exact_audit_helper_active",
        "exact_audit_active",
        "exact_audit_enabled",
        "exact_step_forecast_active",
        "state_at_active",
    ):
        if key in summary:
            out[key] = summary[key]

    guardrail_mode = _mode_text(summary.get("exact_forecast_guardrail_mode", "off"))
    exact_forecast_active = guardrail_mode not in {"", "off", "none", "null", "false"}
    if exact_forecast_active:
        out["exact_forecast_guardrail_mode"] = guardrail_mode
    try:
        veto_count = int(summary.get("exact_forecast_veto_count", 0) or 0)
    except Exception:
        veto_count = 1
    uses_future_exact = bool(summary.get("uses_future_exact_forecast_for_decision", False))
    if veto_count > 0 and (exact_forecast_active or uses_future_exact):
        out["exact_forecast_veto_count"] = veto_count
    return out

def _reference_decision_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only controller-decision fields from a reference block.

    Diagnostic exact-reference payloads may legitimately carry benchmark-exact
    observable data.  The ablation guard audits decision-flow flags and
    controller exact-input modes, not diagnostic overlay values.
    """

    reference = dict(_mapping(payload.get("reference", {})))
    out: dict[str, Any] = {}
    uses_reference = bool(reference.get("uses_reference_for_decision", False))
    uses_future_forecast = bool(reference.get("uses_future_exact_forecast_for_decision", False))
    for key in (
        "uses_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
    ):
        if key in reference:
            out[key] = reference[key]
    if uses_reference or uses_future_forecast:
        for key in (
            "controller_exact_input_mode",
            "controller_reference_mode",
            "reference_enabled",
            "reference_mode",
        ):
            if key in reference:
                out[key] = reference[key]
    return out


def _custom_no_exact_decision_guard(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = _decision_rows(payload)
    violations: list[str] = []

    def _violate(reason: str) -> None:
        if reason not in violations:
            violations.append(str(reason))

    decision_metadata = (
        ("summary", dict(_mapping(payload.get("summary", {})))),
        ("runtime_contract", dict(_mapping(payload.get("runtime_contract", {})))),
        ("route_config", dict(_mapping(payload.get("route_config", {})))),
        ("controller_config", dict(_mapping(payload.get("controller_config", {})))),
        ("reference", _reference_decision_mapping(payload)),
    )
    for label, mapping in decision_metadata:
        for key in ("controller_exact_input_mode", "controller_reference_mode", "reference_mode"):
            if key not in mapping:
                continue
            exact_mode = _mode_text(mapping.get(key))
            if exact_mode in _EXACT_CONTROLLER_INPUT_MODES:
                _violate(f"{label}.{key}={exact_mode}")
        flow = _mode_text(mapping.get("decision_data_flow", ""))
        if flow == DECISION_DATA_FLOW_EXACT_ASSISTED:
            _violate(f"{label}.decision_data_flow={flow}")
        if bool(mapping.get("uses_reference_for_decision", False)):
            _violate(f"{label}.uses_reference_for_decision=true")
        if bool(mapping.get("uses_future_exact_forecast_for_decision", False)):
            _violate(f"{label}.uses_future_exact_forecast_for_decision=true")
        try:
            exact_decisions = int(mapping.get("exact_decision_checkpoints", 0) or 0)
        except Exception:
            exact_decisions = 1
        if exact_decisions > 0:
            _violate(f"{label}.exact_decision_checkpoints={exact_decisions}")
    for idx, row in enumerate(rows):
        backend = _mode_text(row.get("decision_backend", ""))
        if backend == "exact":
            _violate(f"row[{idx}].decision_backend=exact")
        flow = _mode_text(row.get("decision_data_flow", ""))
        if flow == DECISION_DATA_FLOW_EXACT_ASSISTED:
            _violate(f"row[{idx}].decision_data_flow={flow}")
        if bool(row.get("uses_reference_for_decision", False)):
            _violate(f"row[{idx}].uses_reference_for_decision=true")
        if bool(row.get("uses_future_exact_forecast_for_decision", False)):
            _violate(f"row[{idx}].uses_future_exact_forecast_for_decision=true")
    return {
        "passed": not violations,
        "violations": list(violations),
        "violation_count": int(len(violations)),
        "guard": "controller_exact_inputs_off_no_exact_decision_fields",
    }


def _merge_contracts(*contracts: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(contracts[0]) if contracts else {}
    violations: list[str] = []
    for contract in contracts:
        for violation in contract.get("violations", ()):  # type: ignore[union-attr]
            if str(violation) not in violations:
                violations.append(str(violation))
    merged["passed"] = all(bool(contract.get("passed", False)) for contract in contracts)
    merged["violations"] = list(violations)
    merged["violation_count"] = int(len(violations))
    if len(contracts) > 1:
        merged["subguards"] = [dict(contract) for contract in contracts]
    return merged


def validate_ablation_decision_data_flow(
    *,
    payload: Mapping[str, Any],
    variant: GenericControllerAblationVariant,
) -> dict[str, Any]:
    """Return a guardrail audit and raise on exact decision leakage."""

    metadata_contract = _custom_no_exact_decision_guard(payload)
    if bool(variant.strict_qpu_faithful):
        strict_contract = strict_qpu_faithful_decision_contract(
            summary=_summary_decision_mapping(payload),
            reference=_reference_decision_mapping(payload),
            decision_rows=_decision_rows(payload),
        )
        contract = _merge_contracts(strict_contract, metadata_contract)
    else:
        contract = metadata_contract
    if not bool(contract.get("passed", False)):
        violations = "; ".join(str(item) for item in contract.get("violations", ()))
        raise ValueError(
            f"generic controller ablation {variant.variant_id!r} exact-leakage guard failed: {violations}"
        )
    return dict(contract)


def _count_actions(rows: Sequence[Mapping[str, Any]], action_kind: str) -> int:
    return int(sum(1 for row in rows if str(row.get("action_kind", "")) == str(action_kind)))


def _summary_int(summary: Mapping[str, Any], key: str, fallback: int) -> int:
    try:
        value = summary.get(key, None)
        return int(fallback if value in {None, ""} else value)
    except Exception:
        return int(fallback)


def _observed_config_value(payload: Mapping[str, Any], key: str) -> Any:
    for section in ("summary", "route_config", "controller_config", "runtime_contract"):
        mapping = _mapping(payload.get(section, {}))
        if key in mapping:
            return mapping.get(key)
    return None


def _count_integrator_rows(rows: Sequence[Mapping[str, Any]], integrator: str) -> int:
    return int(
        sum(1 for row in rows if _mode_text(row.get("integrator_used", "")) == str(integrator))
    )


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value in {None, ""}:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _v2_auto_euler_row_violations(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    violations: list[str] = []
    required_true_fields = (
        "integrator_geometry_gate_pass",
        "integrator_euler_error_pass",
        "integrator_condition_pass",
        "integrator_rho_miss_pass",
        "integrator_euler_time_gate_pass",
        "integrator_euler_observable_gate_pass",
        "integrator_auto_admit_euler",
    )
    for index, row in enumerate(rows):
        if _mode_text(row.get("integrator_used", "")) != "euler":
            continue
        if _mode_text(row.get("integrator_forced_policy", "")) == "euler":
            continue
        if str(row.get("integrator_auto_policy_schema", "")) != "auto_euler_rk4_policy_v2":
            continue
        bad_fields = [field for field in required_true_fields if not _is_truthy(row.get(field))]
        blockers = row.get("integrator_euler_blockers", [])
        if isinstance(blockers, str):
            blocker_values = [blockers] if blockers.strip() else []
        elif isinstance(blockers, Sequence):
            blocker_values = [str(value) for value in blockers if value not in {None, ""}]
        else:
            blocker_values = []
        if bad_fields or blocker_values:
            violations.append(
                f"auto_euler_v2_gate_violation[row={index},bad_fields={bad_fields},blockers={blocker_values}]"
            )
    return violations


def validate_ablation_variant_runtime(
    *,
    payload: Mapping[str, Any],
    variant: GenericControllerAblationVariant,
) -> dict[str, Any]:
    """Fail closed when a requested ablation knob was not honored at runtime."""

    summary = _mapping(payload.get("summary", {}))
    rows = _decision_rows(payload)
    append_actions = _count_actions(rows, "append_candidate")
    prune_actions = _count_actions(rows, "prune_coordinate")
    append_count = _summary_int(summary, "append_count", append_actions)
    prune_count = _summary_int(summary, "prune_count", prune_actions)
    euler_count = _summary_int(summary, "integrator_euler_count", _count_integrator_rows(rows, "euler"))
    rk4_count = _summary_int(summary, "integrator_rk4_count", _count_integrator_rows(rows, "rk4"))
    forced_euler_count = int(
        sum(1 for row in rows if _mode_text(row.get("integrator_forced_policy", "")) == "euler")
    )
    violations: list[str] = []

    def _violate(reason: str) -> None:
        if reason not in violations:
            violations.append(str(reason))

    if not bool(variant.append_enabled):
        if append_count != 0:
            _violate(f"append_count={append_count}")
        if append_actions != 0:
            _violate(f"append_action_count={append_actions}")
        observed_append_enabled = _observed_config_value(payload, "append_enabled")
        if observed_append_enabled is True:
            _violate("observed append_enabled=true")
    elif bool(variant.force_requested_knobs):
        observed_append_enabled = _observed_config_value(payload, "append_enabled")
        if observed_append_enabled is False:
            _violate("observed append_enabled=false")
    if str(variant.prune_mode) == "off":
        if prune_count != 0:
            _violate(f"prune_count={prune_count}")
        if prune_actions != 0:
            _violate(f"prune_action_count={prune_actions}")
        observed_prune_mode = _mode_text(_observed_config_value(payload, "prune_mode"))
        if observed_prune_mode and observed_prune_mode != "off":
            _violate(f"observed prune_mode={observed_prune_mode}")
    elif bool(variant.force_requested_knobs):
        observed_prune_mode = _mode_text(_observed_config_value(payload, "prune_mode"))
        if observed_prune_mode and observed_prune_mode != str(variant.prune_mode):
            _violate(f"observed prune_mode={observed_prune_mode}")
    if str(variant.integrator_policy) == "euler":
        if rk4_count != 0:
            _violate(f"integrator_rk4_count={rk4_count}")
        observed_integrator = _mode_text(_observed_config_value(payload, "integrator_policy"))
        if observed_integrator and observed_integrator != "euler":
            _violate(f"observed integrator_policy={observed_integrator}")
    elif str(variant.integrator_policy) == "rk4":
        if euler_count != 0:
            _violate(f"integrator_euler_count={euler_count}")
        if forced_euler_count != 0:
            _violate(f"integrator_forced_euler_count={forced_euler_count}")
        observed_integrator = _mode_text(_observed_config_value(payload, "integrator_policy"))
        if observed_integrator and observed_integrator != "rk4":
            _violate(f"observed integrator_policy={observed_integrator}")
        if rk4_count == 0 and rows:
            _violate("integrator_rk4_count=0")
    elif bool(variant.force_requested_knobs):
        observed_integrator = _mode_text(_observed_config_value(payload, "integrator_policy"))
        if observed_integrator and observed_integrator != str(variant.integrator_policy):
            _violate(f"observed integrator_policy={observed_integrator}")
    for violation in _v2_auto_euler_row_violations(rows):
        _violate(violation)
    if str(variant.controller_mode) == "off":
        observed_mode = _mode_text(_observed_config_value(payload, "mode"))
        if observed_mode and observed_mode != "off":
            _violate(f"observed controller mode={observed_mode}")
    observed_confirm = _mode_text(_observed_config_value(payload, "confirm_score_mode"))
    if observed_confirm and observed_confirm != str(variant.confirm_score_mode):
        _violate(f"observed confirm_score_mode={observed_confirm}")

    return {
        "passed": not violations,
        "violations": list(violations),
        "violation_count": int(len(violations)),
        "guard": "requested_ablation_knobs_match_runtime_telemetry",
        "append_count": int(append_count),
        "prune_count": int(prune_count),
        "integrator_euler_count": int(euler_count),
        "integrator_rk4_count": int(rk4_count),
        "integrator_forced_euler_count": int(forced_euler_count),
    }


def _final_runtime_parameter_count(payload: Mapping[str, Any]) -> int | None:
    summary = _mapping(payload.get("summary", {}))
    value = summary.get("final_runtime_parameter_count", None)
    if value not in {None, ""}:
        try:
            return int(value)
        except Exception:
            pass
    trajectory = payload.get("trajectory", [])
    if isinstance(trajectory, list) and trajectory:
        last = trajectory[-1]
        if isinstance(last, Mapping) and last.get("runtime_parameter_count", None) not in {None, ""}:
            try:
                return int(last["runtime_parameter_count"])
            except Exception:
                return None
    return None


def _augment_payload_for_ablation(
    *,
    payload: Mapping[str, Any],
    case: DynamicsBenchmarkCase,
    variant: GenericControllerAblationVariant,
    command: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(payload)
    summary = dict(_mapping(out.get("summary", {})))
    rows = _decision_rows(out)
    append_count = _summary_int(summary, "append_count", _count_actions(rows, "append_candidate"))
    prune_count = _summary_int(summary, "prune_count", _count_actions(rows, "prune_coordinate"))
    metrics = dict(_mapping(out.get("metrics", {})))
    metrics.update(
        {
            "ablation_parent_algorithm_id": GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM,
            "ablation_variant": str(variant.variant_id),
            "disabled_feature": variant.disabled_feature,
            "append_count": int(append_count),
            "prune_count": int(prune_count),
            "final_runtime_parameter_count": _final_runtime_parameter_count(out),
            "integrator_euler_count": _summary_int(summary, "integrator_euler_count", 0),
            "integrator_rk4_count": _summary_int(summary, "integrator_rk4_count", 0),
            "strict_decision_contract_passed": bool(contract.get("passed", False)),
            "strict_decision_contract_violation_count": int(contract.get("violation_count", 0)),
            "diagnostic_ladder_id": variant.diagnostic_ladder_id,
            "diagnostic_ladder_stage": variant.diagnostic_ladder_stage,
            "diagnostic_recovery_candidate": bool(variant.diagnostic_ladder_id),
            "paper_promotion_eligible": bool(variant.paper_promotion_eligible),
        }
    )
    tuning = build_locked_or_default_tuning_provenance(
        case=case,
        algorithm_id=_policy_tuning_algorithm_id(variant),
        settings_kind="controller",
        settings_payload={
            "controller_mode": str(variant.controller_mode),
            "strict_qpu_faithful": bool(variant.strict_qpu_faithful),
            "append_enabled": bool(variant.append_enabled),
            "prune_mode": str(variant.prune_mode),
            "integrator_policy": str(variant.integrator_policy),
            "confirm_score_mode": str(variant.confirm_score_mode),
            "controller_exact_input_mode": "off",
        },
        settings_source=DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
        variant_id=_policy_tuning_variant_id(variant),
        locked=False,
    )
    provenance = dict(_mapping(out.get("provenance", {})))
    provenance.update(
        {
            "route_module": "pipelines.time_dynamics.benchmarks.legacy_native",
            "benchmark_only": True,
            "runner_module": "pipelines.time_dynamics.benchmarks.legacy_native",
            "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_controller_input",
            "controller_decisions_modified": bool(
                _hh_auto_euler_guardrail_tokens(case=case, variant=variant)
            ),
            "exact_reference_controller_inputs": False,
            "command": list(command),
            "ablation_parent_algorithm_id": GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM,
            "ablation_base_policy_algorithm_id": _policy_tuning_algorithm_id(variant),
            "ablation_base_policy_variant_id": _policy_tuning_variant_id(variant),
            "ablation_group_id": f"{dynamics_tuning_class(case)}::{case.family}::{case.case_id}",
            "ablation_variant": str(variant.variant_id),
            "disabled_feature": variant.disabled_feature,
            "controller_mode_requested": str(variant.controller_mode),
            "append_enabled": bool(variant.append_enabled),
            "prune_mode": str(variant.prune_mode),
            "integrator_policy": str(variant.integrator_policy),
            "confirm_score_mode": str(variant.confirm_score_mode),
            "diagnostic_ladder_id": variant.diagnostic_ladder_id,
            "diagnostic_ladder_stage": variant.diagnostic_ladder_stage,
            "diagnostic_recovery_candidate": bool(variant.diagnostic_ladder_id),
            "paper_promotion_eligible": bool(variant.paper_promotion_eligible),
            "hh_auto_euler_guardrail_applied": bool(
                _hh_auto_euler_guardrail_tokens(case=case, variant=variant)
            ),
            "hh_auto_euler_guardrail_id": (
                HH_AUTO_EULER_GUARDRAIL_ID
                if _hh_auto_euler_guardrail_tokens(case=case, variant=variant)
                else None
            ),
            "hh_auto_euler_guardrail_thresholds": (
                dict(HH_AUTO_EULER_GUARDRAIL_THRESHOLDS)
                if _hh_auto_euler_guardrail_tokens(case=case, variant=variant)
                else {}
            ),
            "stage_knob_source": (
                f"{variant.diagnostic_ladder_id}_explicit_stage_override"
                if variant.diagnostic_ladder_id
                else "generic_controller_ablation_variant"
            ),
            "class_lock_role": (
                "baseline_hybrid_policy_only"
                if variant.diagnostic_ladder_id
                else "ablation_base_policy"
            ),
            "controller_exact_input_mode": summary.get("controller_exact_input_mode", "off"),
            "diagnostic_exact_reference_mode": summary.get("diagnostic_exact_reference_mode", "benchmark_exact"),
            "exact_references_reporting_only": True,
            **table_lock_provenance_for_case(case),
            "strict_decision_contract": dict(contract),
            "strict_decision_contract_passed": bool(contract.get("passed", False)),
            **dict(tuning),
            "controller_decisions_modified": bool(
                _hh_auto_euler_guardrail_tokens(case=case, variant=variant)
            ),
            "tuning_provenance": dict(tuning),
        }
    )
    if _hh_auto_euler_guardrail_tokens(case=case, variant=variant):
        provenance["controller_decisions_modified"] = True
        provenance["decision_policy_guardrail_applied"] = True
        provenance["decision_policy_guardrail_id"] = HH_AUTO_EULER_GUARDRAIL_ID
    parameter_manifest = dict(_mapping(out.get("parameter_manifest", {})))
    parameter_manifest["tuning_provenance"] = dict(tuning)
    out["parameter_manifest"] = parameter_manifest
    out["tuning_provenance"] = tuning
    out["metrics"] = metrics
    out["provenance"] = provenance
    out["row_contract"] = {
        "qpu_faithful": bool(contract.get("passed", False)),
        "exact_assisted": False,
        "diagnostic": True,
        "paper_promotion_eligible": bool(variant.paper_promotion_eligible),
    }
    return json_safe(out)


def _failed_ablation_row(
    *,
    case: DynamicsBenchmarkCase,
    variant: GenericControllerAblationVariant,
    reason: str,
) -> DynamicsBenchmarkRow:
    tuning = build_locked_or_default_tuning_provenance(
        case=case,
        algorithm_id=_policy_tuning_algorithm_id(variant),
        settings_kind="controller",
        settings_payload={
            "controller_mode": str(variant.controller_mode),
            "strict_qpu_faithful": bool(variant.strict_qpu_faithful),
            "append_enabled": bool(variant.append_enabled),
            "prune_mode": str(variant.prune_mode),
            "integrator_policy": str(variant.integrator_policy),
            "confirm_score_mode": str(variant.confirm_score_mode),
            "controller_exact_input_mode": "off",
        },
        settings_source=DYNAMICS_SKIPPED_TUNING_SOURCE,
        variant_id=_policy_tuning_variant_id(variant),
        locked=False,
    )
    return DynamicsBenchmarkRow(
        family=str(case.family),
        table_class=str(case.table_class),
        case_id=str(case.case_id),
        algorithm_id=str(variant.algorithm_id),
        method_label=str(variant.method_label),
        status="failed",
        reason=str(reason),
        qpu_faithful=False,
        exact_assisted=False,
        diagnostic=True,
        artifact_json=None,
        metrics={
            "ablation_parent_algorithm_id": GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM,
            "ablation_variant": str(variant.variant_id),
            "disabled_feature": variant.disabled_feature,
            "diagnostic_ladder_id": variant.diagnostic_ladder_id,
            "diagnostic_ladder_stage": variant.diagnostic_ladder_stage,
            "diagnostic_recovery_candidate": bool(variant.diagnostic_ladder_id),
            "paper_promotion_eligible": bool(variant.paper_promotion_eligible),
        },
        resources={},
        provenance={
            "route_module": "pipelines.time_dynamics.benchmarks.legacy_native",
            "benchmark_only": True,
            "runner_module": "pipelines.time_dynamics.benchmarks.legacy_native",
            "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_controller_input",
            "controller_decisions_modified": bool(
                _hh_auto_euler_guardrail_tokens(case=case, variant=variant)
            ),
            "exact_reference_controller_inputs": False,
            "ablation_parent_algorithm_id": GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM,
            "ablation_base_policy_algorithm_id": _policy_tuning_algorithm_id(variant),
            "ablation_base_policy_variant_id": _policy_tuning_variant_id(variant),
            "ablation_group_id": f"{dynamics_tuning_class(case)}::{case.family}::{case.case_id}",
            "ablation_variant": str(variant.variant_id),
            "disabled_feature": variant.disabled_feature,
            "controller_mode_requested": str(variant.controller_mode),
            "append_enabled": bool(variant.append_enabled),
            "prune_mode": str(variant.prune_mode),
            "integrator_policy": str(variant.integrator_policy),
            "confirm_score_mode": str(variant.confirm_score_mode),
            "diagnostic_ladder_id": variant.diagnostic_ladder_id,
            "diagnostic_ladder_stage": variant.diagnostic_ladder_stage,
            "diagnostic_recovery_candidate": bool(variant.diagnostic_ladder_id),
            "paper_promotion_eligible": bool(variant.paper_promotion_eligible),
            "hh_auto_euler_guardrail_applied": bool(
                _hh_auto_euler_guardrail_tokens(case=case, variant=variant)
            ),
            "hh_auto_euler_guardrail_id": (
                HH_AUTO_EULER_GUARDRAIL_ID
                if _hh_auto_euler_guardrail_tokens(case=case, variant=variant)
                else None
            ),
            "hh_auto_euler_guardrail_thresholds": (
                dict(HH_AUTO_EULER_GUARDRAIL_THRESHOLDS)
                if _hh_auto_euler_guardrail_tokens(case=case, variant=variant)
                else {}
            ),
            "stage_knob_source": (
                f"{variant.diagnostic_ladder_id}_explicit_stage_override"
                if variant.diagnostic_ladder_id
                else "generic_controller_ablation_variant"
            ),
            "class_lock_role": (
                "baseline_hybrid_policy_only"
                if variant.diagnostic_ladder_id
                else "ablation_base_policy"
            ),
            "failure_reason": str(reason),
            **table_lock_provenance_for_case(case),
            **dict(tuning),
            "controller_decisions_modified": bool(
                _hh_auto_euler_guardrail_tokens(case=case, variant=variant)
            ),
            "tuning_provenance": dict(tuning),
        },
        table_fields=DynamicsTableFields(table_status_label="failed"),
    )


def run_generic_controller_ablation_row(
    *,
    case: DynamicsBenchmarkCase,
    variant_id: str,
    output_dir: Path,
) -> DynamicsBenchmarkRow:
    variant = get_controller_ablation_variant(variant_id)
    _validate_recovery_variant_case(case=case, variant=variant)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw_payload_json = root / "raw_payload.json"
    command, argv = _command_for_variant(case=case, variant=variant, raw_payload_json=raw_payload_json)
    rows_mod._write_json(root / "command.json", command)
    args = realtime.build_parser().parse_args(argv)
    payload = realtime.run_from_args(args)
    if not isinstance(payload, Mapping):
        raise ValueError("generic controller ablation realtime route returned a non-mapping payload")
    decision_contract = validate_ablation_decision_data_flow(payload=payload, variant=variant)
    runtime_contract = validate_ablation_variant_runtime(payload=payload, variant=variant)
    if not bool(runtime_contract.get("passed", False)):
        violations = "; ".join(str(item) for item in runtime_contract.get("violations", ()))
        raise ValueError(
            f"generic controller ablation {variant.variant_id!r} knob guard failed: {violations}"
        )
    contract = _merge_contracts(decision_contract, runtime_contract)
    augmented = _augment_payload_for_ablation(
        payload=payload,
        case=case,
        variant=variant,
        command=command,
        contract=contract,
    )
    if not raw_payload_json.exists():
        rows_mod._write_json(raw_payload_json, augmented)
    else:
        rows_mod._write_json(raw_payload_json, augmented)
    row = rows_mod._row_from_payload(
        case=case,
        algorithm_id=variant.algorithm_id,
        payload=augmented,
        artifact_json=raw_payload_json,
        command=command,
    )
    rows_mod.write_dynamics_row_bundle(row=row, output_dir=root, raw_payload=augmented)
    return row


def run_generic_controller_ablation_matrix(
    *,
    case: DynamicsBenchmarkCase,
    output_dir: Path,
    variants: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    variant_ids = (
        tuple(variant.variant_id for variant in default_controller_ablation_variants())
        if variants is None
        else tuple(str(item) for item in variants)
    )
    rows: list[dict[str, Any]] = []
    for variant_id in variant_ids:
        variant = get_controller_ablation_variant(variant_id)
        variant_dir = root / variant.variant_id
        try:
            row = run_generic_controller_ablation_row(
                case=case,
                variant_id=variant.variant_id,
                output_dir=variant_dir,
            )
        except Exception as exc:
            row = _failed_ablation_row(
                case=case,
                variant=variant,
                reason=f"{type(exc).__name__}: {exc}",
            )
            rows_mod.write_dynamics_row_bundle(row=row, output_dir=variant_dir)
        rows.append(row.to_dict())
    bundle = dynamics_table_bundle_payload(
        rows=rows,
        label=GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM,
    )
    bundle.update(
        {
            "schema": GENERIC_CONTROLLER_ABLATION_MATRIX_SCHEMA,
            "algorithm_id": GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM,
            "case": case.to_dict(),
            "variants": [get_controller_ablation_variant(variant_id).to_dict() for variant_id in variant_ids],
            "qpu_faithful_contract": "controller exact inputs off; diagnostic exact references reporting-only; strict variants audited with strict_qpu_faithful_decision_contract",
        }
    )
    rows_mod._write_json(root / "rows.json", rows)
    table_payload = write_generic_dynamics_table_summaries(
        rows=rows,
        output_dir=root,
        label=GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM,
    )
    bundle["tables"] = table_payload["tables"]
    bundle["paths"] = {
        "rows_json": str(root / "rows.json"),
        "summary_json": str(root / "summary.json"),
        "tab_dyn_claims_json": str(root / "tab_dyn_claims.json"),
        "tab_dyn_ablation_matrix_json": str(root / "tab_dyn_ablation_matrix.json"),
        "tables_summary_json": str(root / "tables_summary.json"),
    }
    rows_mod._write_json(root / "summary.json", bundle)
    return json_safe(bundle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a generic strict controller ablation matrix.")
    parser.add_argument("--family", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-manifest", type=Path, default=None)
    parser.add_argument("--class-settings-manifest", type=Path, default=None)
    parser.add_argument("--require-locked-class-settings", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", dest="variants", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case = get_generic_dynamics_case(
        str(args.case_id),
        family=str(args.family),
        case_manifest=args.case_manifest,
    )
    case = with_class_settings_lock_manifest(
        case,
        manifest_path=args.class_settings_manifest,
        require_locked=bool(args.require_locked_class_settings),
    )
    result = run_generic_controller_ablation_matrix(
        case=case,
        output_dir=Path(args.output_dir),
        variants=args.variants,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

HH_DYNAMICS_CASES: tuple[str, ...] = ("hh_l2_t8_anchor_v1",)
HH_MODULE_MAP: dict[str, str] = {
    "dyn_exact_reference": "pipelines.time_dynamics.legacy.hh_benchmarks.hh_exact_reference_benchmark",
    "dyn_product_formula_envelope": "pipelines.time_dynamics.legacy.hh_benchmarks.hh_product_formula_envelope_benchmark",
    "dyn_qdrift": "pipelines.time_dynamics.legacy.hh_benchmarks.hh_qdrift_benchmark",
    "dyn_fixed_mclachlan": "pipelines.time_dynamics.legacy.hh_benchmarks.hh_fixed_mclachlan_benchmark",
    "dyn_fixed_pvqd": "pipelines.time_dynamics.legacy.hh_benchmarks.hh_fixed_pvqd_benchmark",
    "dyn_adaptive_pvqd": "pipelines.time_dynamics.legacy.hh_benchmarks.hh_adaptive_pvqd_benchmark",
    "dyn_avqds": "pipelines.time_dynamics.legacy.hh_benchmarks.hh_avqds_benchmark",
    "dyn_avqds_t": "pipelines.time_dynamics.legacy.hh_benchmarks.hh_avqds_t_benchmark",
    "dyn_vff_like": "pipelines.time_dynamics.legacy.hh_benchmarks.hh_vff_like_benchmark",
}


def has_legacy_runner(algorithm_id: str) -> bool:
    return str(algorithm_id) in HH_MODULE_MAP


def module_for_algorithm(algorithm_id: str) -> str:
    try:
        return HH_MODULE_MAP[str(algorithm_id)]
    except KeyError as exc:
        raise KeyError(f"no concrete HH dynamics module mapping for {algorithm_id!r}") from exc


def run_legacy_hh_wrapper(*, case_id: str, algorithm_id: str, output_dir: Path) -> dict[str, str]:
    if str(case_id) not in HH_DYNAMICS_CASES:
        raise ValueError(f"Unknown HH dynamics benchmark case_id={case_id!r}")
    module = module_for_algorithm(algorithm_id)
    cmd = [sys.executable, "-m", module, "--case-id", str(case_id), "--output-dir", str(output_dir)]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    (root / "generic_dispatch_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (root / "generic_dispatch_stderr.log").write_text(completed.stderr, encoding="utf-8")
    (root / "generic_dispatch_exit_code.txt").write_text(f"{completed.returncode}\n", encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return {
        "schema": "generic_dynamics_benchmark_single_v1",
        "family": "hh",
        "case_id": str(case_id),
        "algorithm_id": str(algorithm_id),
        "status": "ok",
        "module": module,
        "output_dir": str(output_dir),
    }


LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE = "pipelines.time_dynamics.benchmarks.legacy_native"

LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE_BY_ALGORITHM: dict[str, str] = {
    EXACT_REFERENCE_ALGORITHM_ID: LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE,
    FIXED_MCLACHLAN_ALGORITHM_ID: LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE,
    PRODUCT_FORMULA_ALGORITHM_ID: LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE,
    QDRIFT_ALGORITHM_ID: LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE,
    FIXED_PVQD_ALGORITHM_ID: LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE,
    ADAPTIVE_PVQD_ALGORITHM_ID: LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE,
    AVQDS_ALGORITHM_ID: LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE,
    AVQDS_T_ALGORITHM_ID: LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE,
}

LEGACY_NATIVE_RUNNER_BY_ALGORITHM = {
    EXACT_REFERENCE_ALGORITHM_ID: run_exact_reference_benchmark_row,
    FIXED_MCLACHLAN_ALGORITHM_ID: run_fixed_mclachlan_benchmark_row,
    PRODUCT_FORMULA_ALGORITHM_ID: run_product_formula_benchmark_row,
    QDRIFT_ALGORITHM_ID: run_qdrift_benchmark_row,
    FIXED_PVQD_ALGORITHM_ID: run_fixed_pvqd_benchmark_row,
    ADAPTIVE_PVQD_ALGORITHM_ID: run_adaptive_pvqd_benchmark_row,
    AVQDS_ALGORITHM_ID: run_avqds_benchmark_row,
    AVQDS_T_ALGORITHM_ID: run_avqds_t_benchmark_row,
}

LEGACY_NATIVE_GENERIC_ALGORITHMS: tuple[str, ...] = tuple(LEGACY_NATIVE_RUNNER_BY_ALGORITHM)


def has_legacy_native_runner(algorithm_id: str) -> bool:
    return str(algorithm_id) in LEGACY_NATIVE_RUNNER_BY_ALGORITHM


def run_benchmark_row(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    output_dir: Path,
) -> DynamicsBenchmarkRow:
    try:
        runner = LEGACY_NATIVE_RUNNER_BY_ALGORITHM[str(algorithm_id)]
    except KeyError as exc:
        raise KeyError(str(algorithm_id)) from exc
    return runner(case=case, output_dir=Path(output_dir))


__all__ = [
    "ADAPTIVE_PVQD_ALGORITHM_ID",
    "AVQDS_ALGORITHM_ID",
    "AVQDS_T_ALGORITHM_ID",
    "DEFAULT_QDRIFT_RNG_SEED",
    "DEFAULT_QDRIFT_SAMPLES_PER_INTERVAL",
    "EXACT_REFERENCE_ALGORITHM_ID",
    "FIXED_MCLACHLAN_ALGORITHM_ID",
    "FIXED_PVQD_ALGORITHM_ID",
    "GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM",
    "GENERIC_CONTROLLER_ABLATION_MATRIX_SCHEMA",
    "GenericControllerAblationVariant",
    "HH_DYNAMICS_CASES",
    "HH_MODULE_MAP",
    "LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE",
    "LEGACY_NATIVE_BENCHMARK_RUNNER_MODULE_BY_ALGORITHM",
    "LEGACY_NATIVE_GENERIC_ALGORITHMS",
    "PRODUCT_FORMULA_ALGORITHM_ID",
    "PRODUCT_FORMULA_CANDIDATE_ORDERS",
    "QDRIFT_ALGORITHM_ID",
    "controller_ablation_variants",
    "default_controller_ablation_variants",
    "get_controller_ablation_variant",
    "has_legacy_runner",
    "has_legacy_native_runner",
    "module_for_algorithm",
    "run_benchmark_row",
    "run_generic_controller_ablation_matrix",
    "run_generic_controller_ablation_row",
    "run_legacy_hh_wrapper",
    "validate_ablation_decision_data_flow",
    "validate_ablation_variant_runtime",
]


if __name__ == "__main__":
    raise SystemExit(main())
