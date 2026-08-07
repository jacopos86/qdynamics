#!/usr/bin/env python3
"""Pinned Qiskit-community primary dynamics comparators for Paper II.

This module is deliberately separate from :mod:`qiskit_dynamics_adapter`, which
is parity-only.  The functions here run Qiskit Algorithms time evolvers as
primary comparator implementations for benchmark rows.  Exact/reference states
are not accepted as inputs; physical scoring against exact diagonalization is
performed by the caller after the Qiskit trajectory is produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.qiskit_pauli_tools import (
    QiskitPauliUnavailable,
    build_parameterized_runtime_layout_circuit,
    build_runtime_layout_circuit,
    circuit_stats,
    import_qiskit_pauli_components,
    to_ixyz_label,
)

QISKIT_COMMUNITY_DYNAMICS_SCHEMA = "qiskit_community_dynamics_primary_v1"
QISKIT_COMMUNITY_RESOURCE_POLICY = "qiskit_community_compiled_circuit_accumulated_v1"
QISKIT_COMMUNITY_ALGORITHMS: tuple[str, ...] = (
    "dyn_qiskit_trotter_qrte",
    "dyn_qiskit_pvqd",
    "dyn_qiskit_varqrte",
)
QISKIT_COMMUNITY_LABELS: dict[str, str] = {
    "dyn_qiskit_trotter_qrte": "Qiskit TrotterQRTE",
    "dyn_qiskit_pvqd": "Qiskit PVQD",
    "dyn_qiskit_varqrte": "Qiskit VarQRTE",
}


class QiskitCommunityDynamicsUnavailable(QiskitPauliUnavailable):
    """Raised when pinned Qiskit-community dynamics support is unavailable."""


class QiskitCommunityDynamicsUnsupported(ValueError):
    """Raised when a benchmark case cannot be represented by the Qiskit row."""


@dataclass(frozen=True)
class QiskitCommunityDynamicsConfig:
    qubit_cap: int | None = 12
    pvqd_optimizer_maxiter: int = 24
    varqrte_num_timesteps_per_interval: int = 1
    varqrte_max_runtime_parameters: int | None = 64
    varqrte_max_qgt_entries: int | None = 4096
    trotter_num_timesteps_per_interval: int = 1
    time_segmentation: str = "match_case_time_grid_piecewise_constant"
    dependency_pin_policy: str = "qiskit_algorithms_0_4_0"
    compile_backend_name: str = "qiskit_statevector_default"
    compile_optimization_level: int = 0
    seed_transpiler: int = 0
    export_circuits: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QiskitCommunityDynamicsRunResult:
    public_payload: Mapping[str, Any]
    states_by_time: tuple[np.ndarray, ...]
    circuit_records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _QiskitCommunityComponents:
    TimeEvolutionProblem: Any
    TrotterQRTE: Any
    PVQD: Any
    VarQRTE: Any
    Statevector: Any
    StatevectorEstimator: Any
    StatevectorSampler: Any
    ComputeUncompute: Any
    COBYLA: Any
    qiskit_version: str
    qiskit_algorithms_version: str


@dataclass(frozen=True)
class _QiskitProgressWriter:
    path: Path | None
    algorithm_id: str
    total_intervals: int

    def write(
        self,
        *,
        interval_index: int,
        time_value: float,
        status: str = "running",
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if self.path is None:
            return
        payload = {
            "schema": "qiskit_community_dynamics_progress_v1",
            "algorithm_id": str(self.algorithm_id),
            "status": str(status),
            "interval_index": int(interval_index),
            "total_intervals": int(self.total_intervals),
            "time_value": float(time_value),
            "updated_unix": float(time.time()),
        }
        if extra:
            payload.update(dict(extra))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)


def qiskit_community_config_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> QiskitCommunityDynamicsConfig:
    """Resolve Qiskit-community primary-comparator options from case metadata."""

    meta = metadata if isinstance(metadata, Mapping) else {}
    nested = (
        meta.get("qiskit_community_dynamics", {})
        if isinstance(meta.get("qiskit_community_dynamics", {}), Mapping)
        else {}
    )

    def _optional_int(key: str, default: int | None) -> int | None:
        raw = nested.get(key, meta.get(f"qiskit_community_{key}", default))
        if raw in {None, "", "none", "None"}:
            return None
        return int(raw)

    return QiskitCommunityDynamicsConfig(
        qubit_cap=_optional_int("qubit_cap", 12),
        pvqd_optimizer_maxiter=int(
            nested.get("pvqd_optimizer_maxiter", meta.get("qiskit_community_pvqd_optimizer_maxiter", 24))
        ),
        varqrte_num_timesteps_per_interval=int(
            nested.get(
                "varqrte_num_timesteps_per_interval",
                meta.get("qiskit_community_varqrte_num_timesteps_per_interval", 1),
            )
        ),
        varqrte_max_runtime_parameters=_optional_int("varqrte_max_runtime_parameters", 64),
        varqrte_max_qgt_entries=_optional_int("varqrte_max_qgt_entries", 4096),
        trotter_num_timesteps_per_interval=int(
            nested.get(
                "trotter_num_timesteps_per_interval",
                meta.get("qiskit_community_trotter_num_timesteps_per_interval", 1),
            )
        ),
        export_circuits=bool(nested.get("export_circuits", meta.get("qiskit_community_export_circuits", False))),
    )


def qiskit_community_config_from_case(case: Any) -> QiskitCommunityDynamicsConfig:
    return qiskit_community_config_from_metadata(getattr(case, "metadata", {}) or {})


def import_qiskit_community_components() -> _QiskitCommunityComponents:
    """Import optional Qiskit-community algorithm components lazily."""

    try:
        import qiskit
        import qiskit_algorithms
        from qiskit.primitives import StatevectorEstimator, StatevectorSampler
        from qiskit_algorithms import PVQD, TrotterQRTE, VarQRTE, TimeEvolutionProblem
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_algorithms.state_fidelities import ComputeUncompute
    except Exception as exc:  # pragma: no cover - optional dependency state
        raise QiskitCommunityDynamicsUnavailable(
            "Qiskit-community dynamics comparators require qiskit-algorithms "
            "with TrotterQRTE, PVQD, and VarQRTE."
        ) from exc

    pauli_components = import_qiskit_pauli_components()
    return _QiskitCommunityComponents(
        TimeEvolutionProblem=TimeEvolutionProblem,
        TrotterQRTE=TrotterQRTE,
        PVQD=PVQD,
        VarQRTE=VarQRTE,
        Statevector=pauli_components.Statevector,
        StatevectorEstimator=StatevectorEstimator,
        StatevectorSampler=StatevectorSampler,
        ComputeUncompute=ComputeUncompute,
        COBYLA=COBYLA,
        qiskit_version=str(getattr(qiskit, "__version__", "unknown")),
        qiskit_algorithms_version=str(getattr(qiskit_algorithms, "__version__", "unknown")),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    return value


def _num_qubits_from_state(state: Any) -> int:
    size = int(np.asarray(state, dtype=complex).reshape(-1).size)
    nq = int(round(np.log2(size)))
    if (1 << nq) != size:
        raise QiskitCommunityDynamicsUnsupported(f"state dimension {size} is not a power of two")
    return int(nq)


def _assert_resource_guard(config: QiskitCommunityDynamicsConfig, num_qubits: int) -> None:
    if config.qubit_cap is not None and int(num_qubits) > int(config.qubit_cap):
        raise QiskitCommunityDynamicsUnsupported(
            f"num_qubits={int(num_qubits)} exceeds qiskit_community_qubit_cap={int(config.qubit_cap)}"
        )


def _varqrte_qgt_preflight(
    *,
    config: QiskitCommunityDynamicsConfig,
    algorithm_id: str,
    theta_runtime: np.ndarray,
) -> Mapping[str, Any] | None:
    """Fail fast before Qiskit VarQRTE builds an expensive QGT job."""

    if str(algorithm_id) != "dyn_qiskit_varqrte":
        return None
    parameter_count = int(np.asarray(theta_runtime, dtype=float).reshape(-1).size)
    qgt_entries = int(parameter_count * parameter_count)
    payload = {
        "schema": "qiskit_varqrte_qgt_preflight_v1",
        "runtime_parameter_count": int(parameter_count),
        "qgt_entry_count": int(qgt_entries),
        "max_runtime_parameters": None
        if config.varqrte_max_runtime_parameters is None
        else int(config.varqrte_max_runtime_parameters),
        "max_qgt_entries": None
        if config.varqrte_max_qgt_entries is None
        else int(config.varqrte_max_qgt_entries),
        "passed": True,
    }
    if parameter_count <= 0:
        raise QiskitCommunityDynamicsUnsupported(
            "Qiskit VarQRTE requires a parameterized static seed ansatz; "
            f"runtime_parameter_count={parameter_count}. This is a diagnostic preflight skip, "
            "not a physical trajectory result."
        )
    if (
        config.varqrte_max_runtime_parameters is not None
        and parameter_count > int(config.varqrte_max_runtime_parameters)
    ):
        raise QiskitCommunityDynamicsUnsupported(
            "Qiskit VarQRTE QGT preflight skipped diagnostic row: "
            f"runtime_parameter_count={parameter_count} exceeds "
            f"varqrte_max_runtime_parameters={int(config.varqrte_max_runtime_parameters)}; "
            f"qgt_entry_count={qgt_entries}, "
            f"varqrte_max_qgt_entries={payload['max_qgt_entries']}. "
            "This is a diagnostic preflight skip, not a physical trajectory result."
        )
    if config.varqrte_max_qgt_entries is not None and qgt_entries > int(config.varqrte_max_qgt_entries):
        raise QiskitCommunityDynamicsUnsupported(
            "Qiskit VarQRTE QGT preflight skipped diagnostic row: "
            f"runtime_parameter_count={parameter_count}, qgt_entry_count={qgt_entries} exceeds "
            f"varqrte_max_qgt_entries={int(config.varqrte_max_qgt_entries)}; "
            f"varqrte_max_runtime_parameters={payload['max_runtime_parameters']}. "
            "This is a diagnostic preflight skip, not a physical trajectory result."
        )
    return payload


def _terms_to_sparse_pauli_op(terms: Sequence[Any], *, sparse_pauli_op_cls: Any) -> Any:
    items: list[tuple[str, complex]] = []
    for term in terms:
        coeff = complex(float(getattr(term, "coeff_real")))
        if abs(coeff) <= 1.0e-12:
            continue
        items.append((to_ixyz_label(str(getattr(term, "pauli_exyz"))), coeff))
    if not items:
        raise QiskitCommunityDynamicsUnsupported("Qiskit dynamics interval has no active Pauli terms")
    return sparse_pauli_op_cls.from_list(items).simplify(atol=1.0e-12)


def _statevector_from_circuit(components: _QiskitCommunityComponents, circuit: Any) -> np.ndarray:
    return np.asarray(components.Statevector.from_instruction(circuit).data, dtype=complex).reshape(-1)


def _bound_circuit(ansatz_bundle: Any, values: Sequence[float]) -> Any:
    value_arr = np.asarray(values, dtype=float).reshape(-1)
    if int(value_arr.size) != len(ansatz_bundle.parameters):
        raise QiskitCommunityDynamicsUnsupported(
            f"parameter length mismatch: got {value_arr.size}, expected {len(ansatz_bundle.parameters)}"
        )
    return ansatz_bundle.circuit.assign_parameters(
        dict(zip(ansatz_bundle.parameters, value_arr)),
        inplace=False,
    )


def _stats_record(*, index: int, time_value: float, circuit: Any) -> dict[str, Any]:
    try:
        stats_circuit = circuit.decompose(reps=10)
    except Exception:
        stats_circuit = circuit
    stats = circuit_stats(stats_circuit)
    stats.update({"index": int(index), "time": float(time_value)})
    return stats


def _dependency_payload(components: _QiskitCommunityComponents) -> dict[str, Any]:
    return {
        "qiskit": components.qiskit_version,
        "qiskit_algorithms": components.qiskit_algorithms_version,
    }


def _run_trotter_qrte(
    *,
    config: QiskitCommunityDynamicsConfig,
    components: _QiskitCommunityComponents,
    terms_for_interval: Callable[[float, float], Sequence[Any]],
    times: np.ndarray,
    layout: Any,
    theta_runtime: np.ndarray,
    psi_ref: np.ndarray,
    num_qubits: int,
    progress: _QiskitProgressWriter,
) -> tuple[tuple[np.ndarray, ...], tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    current = build_runtime_layout_circuit(
        layout,
        theta_runtime,
        int(num_qubits),
        reference_state=psi_ref,
    )
    states: list[np.ndarray] = [_statevector_from_circuit(components, current)]
    records: list[Mapping[str, Any]] = [_stats_record(index=0, time_value=float(times[0]), circuit=current)]
    progress.write(interval_index=0, time_value=float(times[0]))
    for interval_index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        hamiltonian = _terms_to_sparse_pauli_op(
            terms_for_interval(float(left), float(right)),
            sparse_pauli_op_cls=import_qiskit_pauli_components().SparsePauliOp,
        )
        evolver = components.TrotterQRTE(num_timesteps=max(1, int(config.trotter_num_timesteps_per_interval)))
        problem = components.TimeEvolutionProblem(
            hamiltonian=hamiltonian,
            time=float(right - left),
            initial_state=current,
        )
        result = evolver.evolve(problem)
        current = result.evolved_state.decompose(reps=1)
        states.append(_statevector_from_circuit(components, current))
        records.append(_stats_record(index=int(interval_index) + 1, time_value=float(right), circuit=current))
        progress.write(interval_index=int(interval_index) + 1, time_value=float(right))
    details = {
        "qiskit_algorithm_name": "TrotterQRTE",
        "num_timesteps_per_interval": int(config.trotter_num_timesteps_per_interval),
        "circuit_flattening_policy": "decompose_one_level_after_each_interval_to_avoid_recursive_circuit_nesting",
    }
    return tuple(states), tuple(records), details


def _run_pvqd(
    *,
    config: QiskitCommunityDynamicsConfig,
    components: _QiskitCommunityComponents,
    terms_for_interval: Callable[[float, float], Sequence[Any]],
    times: np.ndarray,
    layout: Any,
    theta_runtime: np.ndarray,
    psi_ref: np.ndarray,
    num_qubits: int,
    progress: _QiskitProgressWriter,
) -> tuple[tuple[np.ndarray, ...], tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    ansatz = build_parameterized_runtime_layout_circuit(
        layout,
        num_qubits=int(num_qubits),
        reference_state=psi_ref,
        parameter_prefix="theta",
    )
    theta_current = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if int(theta_current.size) == 0:
        raise QiskitCommunityDynamicsUnsupported("Qiskit PVQD requires a parameterized static seed ansatz")
    states: list[np.ndarray] = [_statevector_from_circuit(components, _bound_circuit(ansatz, theta_current))]
    records: list[Mapping[str, Any]] = [
        _stats_record(index=0, time_value=float(times[0]), circuit=_bound_circuit(ansatz, theta_current))
    ]
    progress.write(interval_index=0, time_value=float(times[0]), extra={"estimator_call_proxy": 0})
    fidelities: list[float] = []
    estimator_calls = 0
    for interval_index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        hamiltonian = _terms_to_sparse_pauli_op(
            terms_for_interval(float(left), float(right)),
            sparse_pauli_op_cls=import_qiskit_pauli_components().SparsePauliOp,
        )
        fidelity = components.ComputeUncompute(components.StatevectorSampler())
        optimizer = components.COBYLA(maxiter=max(1, int(config.pvqd_optimizer_maxiter)))
        evolver = components.PVQD(
            fidelity=fidelity,
            ansatz=ansatz.circuit,
            initial_parameters=theta_current,
            optimizer=optimizer,
            num_timesteps=1,
            use_parameter_shift=False,
            initial_guess=np.zeros_like(theta_current),
        )
        problem = components.TimeEvolutionProblem(
            hamiltonian=hamiltonian,
            time=float(right - left),
        )
        result = evolver.evolve(problem)
        theta_current = np.asarray(result.parameters[-1], dtype=float).reshape(-1)
        if getattr(result, "fidelities", None):
            fidelities.append(float(result.fidelities[-1]))
        estimator_calls += int(config.pvqd_optimizer_maxiter)
        bound = _bound_circuit(ansatz, theta_current)
        states.append(_statevector_from_circuit(components, bound))
        records.append(_stats_record(index=int(interval_index) + 1, time_value=float(right), circuit=bound))
        progress.write(
            interval_index=int(interval_index) + 1,
            time_value=float(right),
            extra={"estimator_call_proxy": int(estimator_calls)},
        )
    details = {
        "qiskit_algorithm_name": "PVQD",
        "pvqd_optimizer": "COBYLA",
        "pvqd_optimizer_maxiter": int(config.pvqd_optimizer_maxiter),
        "pvqd_step_count": int(max(0, len(times) - 1)),
        "pvqd_fidelities": [float(x) for x in fidelities],
        "pvqd_estimator_call_proxy": int(estimator_calls),
        "final_runtime_parameter_count": int(theta_current.size),
    }
    return tuple(states), tuple(records), details


def _run_varqrte(
    *,
    config: QiskitCommunityDynamicsConfig,
    components: _QiskitCommunityComponents,
    terms_for_interval: Callable[[float, float], Sequence[Any]],
    times: np.ndarray,
    layout: Any,
    theta_runtime: np.ndarray,
    psi_ref: np.ndarray,
    num_qubits: int,
    progress: _QiskitProgressWriter,
) -> tuple[tuple[np.ndarray, ...], tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    ansatz = build_parameterized_runtime_layout_circuit(
        layout,
        num_qubits=int(num_qubits),
        reference_state=psi_ref,
        parameter_prefix="theta",
    )
    theta_current = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if int(theta_current.size) == 0:
        raise QiskitCommunityDynamicsUnsupported("Qiskit VarQRTE requires a parameterized static seed ansatz")
    states: list[np.ndarray] = [_statevector_from_circuit(components, _bound_circuit(ansatz, theta_current))]
    records: list[Mapping[str, Any]] = [
        _stats_record(index=0, time_value=float(times[0]), circuit=_bound_circuit(ansatz, theta_current))
    ]
    progress.write(interval_index=0, time_value=float(times[0]))
    parameter_history: list[list[float]] = [theta_current.astype(float).tolist()]
    for interval_index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        hamiltonian = _terms_to_sparse_pauli_op(
            terms_for_interval(float(left), float(right)),
            sparse_pauli_op_cls=import_qiskit_pauli_components().SparsePauliOp,
        )
        evolver = components.VarQRTE(
            ansatz=ansatz.circuit,
            initial_parameters=theta_current,
            num_timesteps=max(1, int(config.varqrte_num_timesteps_per_interval)),
        )
        problem = components.TimeEvolutionProblem(
            hamiltonian=hamiltonian,
            time=float(right - left),
        )
        result = evolver.evolve(problem)
        theta_current = np.asarray(result.parameter_values[-1], dtype=float).reshape(-1)
        parameter_history.append(theta_current.astype(float).tolist())
        bound = _bound_circuit(ansatz, theta_current)
        states.append(_statevector_from_circuit(components, bound))
        records.append(_stats_record(index=int(interval_index) + 1, time_value=float(right), circuit=bound))
        progress.write(interval_index=int(interval_index) + 1, time_value=float(right))
    details = {
        "qiskit_algorithm_name": "VarQRTE",
        "variational_principle": "RealMcLachlanPrinciple",
        "ode_solver": "ForwardEulerSolver",
        "num_timesteps_per_interval": int(config.varqrte_num_timesteps_per_interval),
        "varqrte_step_count": int(max(0, len(times) - 1)),
        "final_runtime_parameter_count": int(theta_current.size),
        "parameter_history": parameter_history,
    }
    return tuple(states), tuple(records), details


def run_qiskit_community_dynamics(
    *,
    config: QiskitCommunityDynamicsConfig,
    case: Any,
    algorithm_id: str,
    terms_for_interval: Callable[[float, float], Sequence[Any]],
    times: Sequence[float],
    layout: Any,
    theta_runtime: Sequence[float],
    psi_ref: Sequence[complex],
    progress_json: str | Path | None = None,
) -> QiskitCommunityDynamicsRunResult:
    """Run one pinned Qiskit-community comparator without exact-reference inputs."""

    algorithm = str(algorithm_id)
    if algorithm not in QISKIT_COMMUNITY_ALGORITHMS:
        raise QiskitCommunityDynamicsUnsupported(f"unsupported Qiskit-community algorithm {algorithm!r}")
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if int(times_arr.size) < 1:
        raise QiskitCommunityDynamicsUnsupported("Qiskit-community dynamics requires at least one time point")
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    psi_ref_arr = np.asarray(psi_ref, dtype=complex).reshape(-1)
    num_qubits = _num_qubits_from_state(psi_ref_arr)
    _assert_resource_guard(config, num_qubits)
    varqrte_preflight = _varqrte_qgt_preflight(
        config=config,
        algorithm_id=algorithm,
        theta_runtime=theta_arr,
    )
    components = import_qiskit_community_components()

    runner = {
        "dyn_qiskit_trotter_qrte": _run_trotter_qrte,
        "dyn_qiskit_pvqd": _run_pvqd,
        "dyn_qiskit_varqrte": _run_varqrte,
    }[algorithm]
    progress = _QiskitProgressWriter(
        None if progress_json in {None, ""} else Path(progress_json),
        algorithm_id=algorithm,
        total_intervals=max(0, int(times_arr.size) - 1),
    )
    states, records, details = runner(
        config=config,
        components=components,
        terms_for_interval=terms_for_interval,
        times=times_arr,
        layout=layout,
        theta_runtime=theta_arr,
        psi_ref=psi_ref_arr,
        num_qubits=int(num_qubits),
        progress=progress,
    )
    progress.write(
        interval_index=max(0, int(times_arr.size) - 1),
        time_value=float(times_arr[-1]),
        status="completed",
    )
    payload = {
        "schema": QISKIT_COMMUNITY_DYNAMICS_SCHEMA,
        "algorithm_id": algorithm,
        "family": str(getattr(case, "family", "")),
        "case_id": str(getattr(case, "case_id", "")),
        "status": "completed",
        "method_label": QISKIT_COMMUNITY_LABELS[algorithm],
        "mapping_convention": "repo_exyz_left_to_right_qnminus1_to_q0__qiskit_qubit0_rightmost",
        "resource_policy": QISKIT_COMMUNITY_RESOURCE_POLICY,
        "time_segmentation": str(config.time_segmentation),
        "config": config.to_dict(),
        "dependencies": _dependency_payload(components),
        "controller_decisions_modified": False,
        "exact_reference_controller_inputs": False,
        "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_qiskit_algorithm_input",
        "circuit_records": [dict(record) for record in records],
        **dict(details),
    }
    if varqrte_preflight is not None:
        payload["varqrte_qgt_preflight"] = dict(varqrte_preflight)
    return QiskitCommunityDynamicsRunResult(
        public_payload=_json_safe(payload),
        states_by_time=tuple(np.asarray(state, dtype=complex).reshape(-1) for state in states),
        circuit_records=tuple(dict(record) for record in records),
    )


__all__ = [
    "QISKIT_COMMUNITY_ALGORITHMS",
    "QISKIT_COMMUNITY_DYNAMICS_SCHEMA",
    "QISKIT_COMMUNITY_LABELS",
    "QISKIT_COMMUNITY_RESOURCE_POLICY",
    "QiskitCommunityDynamicsConfig",
    "QiskitCommunityDynamicsRunResult",
    "QiskitCommunityDynamicsUnavailable",
    "QiskitCommunityDynamicsUnsupported",
    "import_qiskit_community_components",
    "qiskit_community_config_from_case",
    "qiskit_community_config_from_metadata",
    "run_qiskit_community_dynamics",
]
