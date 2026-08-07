#!/usr/bin/env python3
"""Reusable static Hubbard-Holstein conventional VQE trial helper.

This module is intentionally narrow: it owns the existing non-ADAPT HH fixed-
ansatz VQE single-trial logic used by ``cross_check_suite`` and by the static HH
benchmark runner.  It does not build HH Hamiltonians or resolve exact targets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.benchmark_decision_noise import (
    BenchmarkDecisionNoiseConfig,
    BenchmarkDecisionNoiseRecorder,
    coerce_config as coerce_benchmark_decision_noise_config,
    copy_decision_noise_metadata,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import compile_polynomial_action, energy_via_one_apply
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    HubbardHolsteinTermwiseAnsatz,
    half_filled_num_particles,
    vqe_minimize,
)

HHConventionalAnsatzKind = Literal["termwise", "layerwise", "qiskit_hea"]
CompiledOperatorParameterizationMode = Literal["logical_shared", "per_pauli_term"]

# Same HH VQE defaults that cross_check_suite.py historically used for the
# HH-Termwise and HH-Layerwise trials.  Trotter trajectory settings remain in
# cross_check_suite because they are not part of a ground-state VQE trial.
_HH_CONVENTIONAL_VQE_DEFAULTS: dict[tuple[int, int], dict[str, Any]] = {
    (2, 1): {"reps": 2, "restarts": 3, "maxiter": 800, "method": "COBYLA"},
    (2, 2): {"reps": 3, "restarts": 4, "maxiter": 1500, "method": "COBYLA"},
    (3, 1): {"reps": 2, "restarts": 4, "maxiter": 2400, "method": "COBYLA"},
}

_ANSATZ_DISPLAY_NAMES: dict[str, str] = {
    "termwise": "HH-Termwise",
    "layerwise": "HH-Layerwise",
    "qiskit_hea": "HH-HEA-Qiskit",
}

_ANSATZ_MACHINE_NAMES: dict[str, str] = {
    "termwise": "hh_hva_termwise",
    "layerwise": "hh_hva_layerwise",
    "qiskit_hea": "hh_hea_qiskit",
}

_QISKIT_HEA_DEPENDENCY_MESSAGE = (
    "Qiskit benchmark-only HEA support requires qiskit.circuit.QuantumCircuit, "
    "qiskit.circuit.ParameterVector, and qiskit.quantum_info.Statevector."
)
_QISKIT_HEA_SHOTS_PER_PAULI_TERM_PROXY = 1024
_QISKIT_HEA_SHOT_PROXY_FORMULA = (
    "shots_total = shots_per_pauli_term_proxy * hamiltonian_pauli_term_count * energy_eval_count_proxy"
)


def default_hh_conventional_vqe_config(num_sites: int, n_ph_max: int) -> dict[str, Any]:
    """Return the legacy HH fixed-ansatz VQE defaults for one HH size/cutoff."""
    key = (int(num_sites), int(n_ph_max))
    if key in _HH_CONVENTIONAL_VQE_DEFAULTS:
        return dict(_HH_CONVENTIONAL_VQE_DEFAULTS[key])
    known = ", ".join(f"L={l},n_ph_max={nph}" for l, nph in sorted(_HH_CONVENTIONAL_VQE_DEFAULTS))
    raise ValueError(
        "No legacy HH conventional VQE defaults for "
        f"L={int(num_sites)}, n_ph_max={int(n_ph_max)}; known: {known}"
    )


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Zero-norm state.")
    return psi / nrm


def _import_qiskit_hea_components() -> tuple[Any, Any, Any]:
    """Import the optional benchmark-only Qiskit components lazily."""
    try:
        from qiskit.circuit import ParameterVector, QuantumCircuit
        from qiskit.quantum_info import Statevector
    except Exception as exc:  # pragma: no cover - exact exception varies by install
        raise ImportError(_QISKIT_HEA_DEPENDENCY_MESSAGE) from exc
    return QuantumCircuit, ParameterVector, Statevector


def has_qiskit_hea_support() -> bool:
    """Return whether optional benchmark-only Qiskit HEA support is importable."""
    try:
        _import_qiskit_hea_components()
    except Exception:
        return False
    return True


def _num_qubits_from_state_vector(psi_ref: np.ndarray) -> int:
    size = int(np.asarray(psi_ref).size)
    if size <= 0:
        raise ValueError("psi_ref must be a non-empty state vector.")
    if size & (size - 1):
        raise ValueError("psi_ref length must be a power of two for Qiskit HEA evolution.")
    return int(size.bit_length() - 1)


def _qiskit_circuit_stats(circuit: Any) -> dict[str, Any]:
    if circuit is None:
        return {
            "compiled_depth_total": None,
            "compiled_count_2q_total": None,
            "compiled_op_counts": {},
            "compiled_circuit_stats_status": "unavailable_no_circuit",
        }
    compiled = circuit
    try:
        from qiskit import transpile

        compiled = transpile(circuit, basis_gates=["id", "x", "sx", "rx", "ry", "rz", "h", "s", "sdg", "cx", "cz"], optimization_level=1)
        status = "qiskit_transpile_basis_proxy"
    except Exception:
        status = "qiskit_circuit_proxy_untranspiled"
    try:
        op_counts_raw = dict(compiled.count_ops())
    except Exception:
        op_counts_raw = {}
    op_counts = {str(key): int(value) for key, value in op_counts_raw.items()}
    try:
        depth = int(compiled.depth())
    except Exception:
        depth = None
    count_2q = int(sum(op_counts.get(gate, 0) for gate in ("cx", "cz", "swap", "iswap", "ecr")))
    return {
        "compiled_depth_total": depth,
        "compiled_count_2q_total": count_2q,
        "compiled_op_counts": op_counts,
        "compiled_circuit_stats_status": status,
    }


def _final_exact_energy_from_state(hamiltonian: Any, psi: np.ndarray) -> float:
    compiled = compile_polynomial_action(hamiltonian, tol=1e-12)
    energy, _ = energy_via_one_apply(np.asarray(psi, dtype=complex).ravel(), compiled)
    return float(energy)


def _hamiltonian_pauli_term_count(hamiltonian: Any, *, tol: float = 1e-12) -> int:
    try:
        terms = list(hamiltonian.return_polynomial())
    except Exception:
        return 0
    labels: set[str] = set()
    for term in terms:
        try:
            coeff = complex(term.p_coeff)
            label = str(term.pw2strng()).lower()
        except Exception:
            continue
        if abs(coeff) <= float(tol) or not label or label == "e" * len(label):
            continue
        labels.add(label)
    return int(len(labels))


class _QiskitStatevectorAnsatzAdapter:
    """Minimal Qiskit circuit adapter for the repo-native VQE minimizer."""

    def __init__(self, *, circuit: Any, parameters: Sequence[Any], statevector_cls: Any) -> None:
        self._circuit = circuit
        self._parameters = tuple(parameters)
        self._statevector_cls = statevector_cls
        self.num_qubits = int(getattr(circuit, "num_qubits"))
        self.num_parameters = int(len(self._parameters))

    def circuit_stats(self) -> dict[str, Any]:
        return _qiskit_circuit_stats(self._circuit)

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float).ravel()
        if int(theta_arr.size) != int(self.num_parameters):
            raise ValueError(
                "theta size mismatch for Qiskit HEA ansatz: "
                f"expected {int(self.num_parameters)}, got {int(theta_arr.size)}"
            )
        psi_ref_arr = np.asarray(psi_ref, dtype=complex).ravel()
        num_qubits = _num_qubits_from_state_vector(psi_ref_arr)
        if num_qubits != int(self.num_qubits):
            raise ValueError(
                "psi_ref qubit count does not match Qiskit HEA circuit: "
                f"state has {num_qubits}, circuit has {int(self.num_qubits)}"
            )
        assignments = {param: float(theta_arr[idx]) for idx, param in enumerate(self._parameters)}
        bound_circuit = self._circuit.assign_parameters(assignments, inplace=False)
        evolved = self._statevector_cls(psi_ref_arr).evolve(bound_circuit)
        return np.asarray(getattr(evolved, "data", evolved), dtype=complex).ravel()


def _build_qiskit_hea_ansatz(*, num_qubits: int, reps: int) -> _QiskitStatevectorAnsatzAdapter:
    """Build a simple benchmark-only linear HEA circuit adapter."""
    n_qubits = int(num_qubits)
    n_reps = int(reps)
    if n_qubits <= 0:
        raise ValueError("num_qubits must be positive for Qiskit HEA ansatz.")
    if n_reps <= 0:
        raise ValueError("reps must be positive for Qiskit HEA ansatz.")
    QuantumCircuit, ParameterVector, Statevector = _import_qiskit_hea_components()
    circuit = QuantumCircuit(n_qubits)
    parameter_count = int(2 * n_qubits * (n_reps + 1))
    theta = ParameterVector("theta", parameter_count)
    ordered_parameters = [theta[idx] for idx in range(parameter_count)]
    cursor = 0
    for _layer in range(n_reps):
        for qubit in range(n_qubits):
            circuit.ry(theta[cursor], qubit)
            cursor += 1
            circuit.rz(theta[cursor], qubit)
            cursor += 1
        for qubit in range(n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
    for qubit in range(n_qubits):
        circuit.ry(theta[cursor], qubit)
        cursor += 1
        circuit.rz(theta[cursor], qubit)
        cursor += 1
    if cursor != parameter_count:  # defensive guard for future edits
        raise RuntimeError("internal Qiskit HEA parameter-count mismatch")
    return _QiskitStatevectorAnsatzAdapter(
        circuit=circuit,
        parameters=ordered_parameters,
        statevector_cls=Statevector,
    )


def _coerce_ansatz_kind(ansatz_kind: str) -> HHConventionalAnsatzKind:
    kind = str(ansatz_kind).strip().lower().replace("hh-", "").replace("_", "-")
    if kind in {"termwise", "term-wise", "hh-termwise", "hh-term-wise"}:
        return "termwise"
    if kind in {"layerwise", "layer-wise", "hh-layerwise", "hh-layer-wise"}:
        return "layerwise"
    if kind in {"qiskit-hea", "hea-qiskit", "hea", "hh-qiskit-hea", "hh-hea-qiskit"}:
        return "qiskit_hea"
    raise ValueError("ansatz_kind must be 'termwise', 'layerwise', or 'qiskit_hea'")


def _build_hh_conventional_ansatz(
    *,
    ansatz_kind: HHConventionalAnsatzKind,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    ordering: str,
    reps: int,
    include_zero_point: bool,
) -> Any:
    if ansatz_kind not in {"termwise", "layerwise"}:
        raise ValueError("Native HH conventional builder only supports termwise/layerwise ansatz kinds.")
    cls = (
        HubbardHolsteinTermwiseAnsatz
        if ansatz_kind == "termwise"
        else HubbardHolsteinLayerwiseAnsatz
    )
    return cls(
        dims=int(num_sites),
        J=float(t),
        U=float(u),
        omega0=float(omega0),
        g=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        reps=int(reps),
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary) == "periodic"),
        include_zero_point=bool(include_zero_point),
    )


def run_hh_conventional_vqe_trial(
    *,
    ansatz_kind: str,
    h_poly: Any,
    exact_gs: float,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    ordering: str,
    reps: int,
    optimizer: str,
    maxiter: int,
    restarts: int,
    seed: int,
    include_zero_point: bool = True,
    psi_ref: np.ndarray | None = None,
    ai_log: Callable[..., None] | None = None,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | Mapping[str, Any] | None = None,
    benchmark_decision_noise_scope: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Run one existing HH fixed-ansatz VQE trial and return normalized data.

    The caller owns Hamiltonian construction and exact-reference resolution.  A
    completed optimizer call returns ``success=True`` at the runner level even if
    the optimizer's own success flag is false; optimizer convergence is exposed
    separately as ``optimizer_success``/``converged``.
    """
    kind = _coerce_ansatz_kind(ansatz_kind)
    display_name = _ANSATZ_DISPLAY_NAMES[kind]
    machine_name = _ANSATZ_MACHINE_NAMES[kind]
    if psi_ref is None:
        psi_ref_arr = np.asarray(
            hubbard_holstein_reference_state(
                dims=int(num_sites),
                num_particles=half_filled_num_particles(int(num_sites)),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                indexing=str(ordering),
            ),
            dtype=complex,
        ).ravel()
    else:
        psi_ref_arr = np.asarray(psi_ref, dtype=complex).ravel()

    if ai_log is not None:
        ai_log("building_ansatz", name=display_name)
    if kind == "qiskit_hea":
        ansatz = _build_qiskit_hea_ansatz(
            num_qubits=_num_qubits_from_state_vector(psi_ref_arr),
            reps=int(reps),
        )
    else:
        ansatz = _build_hh_conventional_ansatz(
            ansatz_kind=kind,
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            omega0=float(omega0),
            g_ep=float(g_ep),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            boundary=str(boundary),
            ordering=str(ordering),
            reps=int(reps),
            include_zero_point=bool(include_zero_point),
        )

    noise_scope = {
        "family": "hh",
        "algorithm_id": machine_name,
        "surface_family": "hh_conventional_vqe",
        **dict(benchmark_decision_noise_scope or {}),
    }
    decision_noise_config = coerce_benchmark_decision_noise_config(
        benchmark_decision_noise_config,
        family=str(noise_scope.get("family") or "hh"),
        case_id=str(noise_scope.get("case_id") or ""),
        algorithm_id=str(noise_scope.get("algorithm_id") or machine_name),
    )
    decision_noise_recorder = BenchmarkDecisionNoiseRecorder(
        decision_noise_config,
        base_scope=noise_scope,
    )

    def _objective_value_transform(event: Mapping[str, Any]) -> float:
        return float(
            decision_noise_recorder.apply(
                float(event["energy_ideal"]),
                surface="hh_vqe_objective",
                value_kind="energy",
                phase="optimizer",
                extra_scope={
                    "restart_index": int(event.get("restart_index", 0) or 0),
                    "nfev_restart": int(event.get("nfev_restart", 0) or 0),
                    "nfev_total_estimate": int(event.get("nfev_total_estimate", 0) or 0),
                    "ansatz_kind": kind,
                },
            )
        )

    if ai_log is not None:
        ai_log(
            "trial_start",
            name=display_name,
            category="conventional_vqe",
            num_params=int(ansatz.num_parameters),
        )
    t0 = time.perf_counter()
    vqe_kwargs: dict[str, Any] = {
        "restarts": int(restarts),
        "seed": int(seed),
        "maxiter": int(maxiter),
        "method": str(optimizer),
    }
    if bool(decision_noise_config.enabled):
        vqe_kwargs["objective_value_transform"] = _objective_value_transform
    result = vqe_minimize(
        h_poly,
        ansatz,
        psi_ref_arr,
        **vqe_kwargs,
    )
    theta = np.asarray(result.theta, dtype=float)
    psi_vqe = np.asarray(ansatz.prepare_state(theta, psi_ref_arr), dtype=complex).ravel()
    psi_vqe = _normalize_state(psi_vqe)
    elapsed_s = float(time.perf_counter() - t0)
    optimizer_decision_energy = float(result.energy)
    energy = (
        _final_exact_energy_from_state(h_poly, psi_vqe)
        if bool(decision_noise_config.enabled)
        else optimizer_decision_energy
    )
    exact_energy = float(exact_gs)
    delta_e = float(energy - exact_energy)
    decision_noise_metadata = None
    if bool(decision_noise_config.enabled):
        decision_noise_metadata = copy_decision_noise_metadata(
            decision_noise_recorder.summary(
                status="ok",
                supported=True,
                extra={"runner": "hh_conventional_vqe", "ansatz_kind": kind},
            )
        )
    optimizer_success = bool(getattr(result, "success", False))
    optimizer_message = str(getattr(result, "message", ""))
    compiled_stats = ansatz.circuit_stats() if kind == "qiskit_hea" and hasattr(ansatz, "circuit_stats") else {}
    hamiltonian_pauli_term_count = _hamiltonian_pauli_term_count(h_poly)
    energy_eval_count_proxy = int(getattr(result, "nfev", 0))
    shots_total = int(_QISKIT_HEA_SHOTS_PER_PAULI_TERM_PROXY * hamiltonian_pauli_term_count * max(energy_eval_count_proxy, 0))
    if ai_log is not None:
        ai_log(
            "trial_done",
            name=display_name,
            energy=energy,
            delta_e=delta_e,
            elapsed_s=round(elapsed_s, 2),
            optimizer_success=optimizer_success,
        )

    payload = {
        "success": True,
        "method_kind": "conventional_vqe",
        "display_name": display_name,
        "name": display_name,
        "ansatz_kind": kind,
        "ansatz_name": machine_name,
        "energy": energy,
        "exact_gs_energy": exact_energy,
        "exact_energy": exact_energy,
        "delta_e": delta_e,
        "abs_delta_e": abs(delta_e),
        "delta_E_abs": abs(delta_e),
        "nfev": energy_eval_count_proxy,
        "nit": int(getattr(result, "nit", 0)),
        "num_parameters": int(ansatz.num_parameters),
        "num_params": int(ansatz.num_parameters),
        "vqe_reps_used": int(reps),
        "vqe_reps": int(reps),
        "vqe_restarts": int(restarts),
        "vqe_maxiter_used": int(maxiter),
        "vqe_maxiter": int(maxiter),
        "optimizer": str(optimizer),
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "optimizer_decision_energy": optimizer_decision_energy,
        "optimizer_reported_energy": optimizer_decision_energy,
        "converged": optimizer_success,
        "best_restart": int(getattr(result, "best_restart", -1)),
        "runtime_s": elapsed_s,
        "phase3_controller_called": False,
        "phase3_emulation": False,
        "uses_exact_for_decision": False,
        "algorithm_origin": "qiskit_hea_exact_bench" if kind == "qiskit_hea" else "repo_native_hh_conventional_vqe",
        "hamiltonian_pauli_term_count": hamiltonian_pauli_term_count,
        "energy_eval_count_proxy": energy_eval_count_proxy,
        "shots_per_pauli_term_proxy": _QISKIT_HEA_SHOTS_PER_PAULI_TERM_PROXY,
        "shot_proxy_formula": _QISKIT_HEA_SHOT_PROXY_FORMULA,
        "shots_total": shots_total,
        "static_shot_estimate_status": "deterministic_proxy_not_physical_shots",
        "compiled_depth_total": compiled_stats.get("compiled_depth_total"),
        "compiled_count_2q_total": compiled_stats.get("compiled_count_2q_total"),
        "compiled_op_counts": compiled_stats.get("compiled_op_counts", {}),
        "compiled_circuit_stats_status": compiled_stats.get("compiled_circuit_stats_status"),
        "circuit_depth": compiled_stats.get("compiled_depth_total"),
        "count_2q": compiled_stats.get("compiled_count_2q_total"),
        "depth_proxy": compiled_stats.get("compiled_depth_total"),
        "theta": theta.tolist(),
        "restart_summaries": getattr(result, "restart_summaries", None),
        "num_sites": int(num_sites),
        "t": float(t),
        "u": float(u),
        "dv": float(dv),
        "omega0": float(omega0),
        "g_ep": float(g_ep),
        "n_ph_max": int(n_ph_max),
        "boson_encoding": str(boson_encoding),
        "boundary": str(boundary),
        "ordering": str(ordering),
        "include_zero_point": bool(include_zero_point),
        "_psi_vqe": psi_vqe,
    }
    if decision_noise_metadata is not None:
        payload.update(
            {
                "benchmark_decision_noise_status": "ok",
                "benchmark_decision_noise": copy_decision_noise_metadata(decision_noise_metadata),
            }
        )
    return payload


def _coerce_parameterization_mode(parameterization_mode: str) -> CompiledOperatorParameterizationMode:
    mode = str(parameterization_mode).strip().lower()
    if mode in {"logical_shared", "per_pauli_term"}:
        return mode  # type: ignore[return-value]
    raise ValueError("parameterization_mode must be 'logical_shared' or 'per_pauli_term'")


@dataclass(frozen=True)
class _CompiledOperatorTrialSetup:
    terms: list[Any]
    mode: CompiledOperatorParameterizationMode
    psi_ref: np.ndarray
    selected_operator_labels: tuple[str, ...]
    executor: CompiledAnsatzExecutor
    logical_parameter_count: int
    runtime_parameter_count: int
    num_parameters: int


def _build_compiled_operator_trial_setup(
    *,
    operator_terms: Sequence[Any],
    psi_ref: np.ndarray,
    parameterization_mode: str,
) -> _CompiledOperatorTrialSetup:
    terms = list(operator_terms)
    if not terms:
        raise ValueError("operator_terms must contain at least one operator.")
    mode = _coerce_parameterization_mode(parameterization_mode)
    psi_ref_arr = np.asarray(psi_ref, dtype=complex).ravel()
    if psi_ref_arr.size <= 0:
        raise ValueError("psi_ref must be a non-empty state vector.")

    selected_operator_labels = tuple(
        str(getattr(term, "label", f"operator_{idx}"))
        for idx, term in enumerate(terms)
    )
    executor = CompiledAnsatzExecutor(
        terms,
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode=mode,
    )
    logical_parameter_count = int(getattr(executor, "logical_parameter_count", len(terms)))
    runtime_parameter_count = int(
        getattr(executor, "runtime_parameter_count", getattr(executor, "num_parameters", logical_parameter_count))
    )
    num_parameters = int(getattr(executor, "num_parameters", logical_parameter_count))
    return _CompiledOperatorTrialSetup(
        terms=terms,
        mode=mode,
        psi_ref=psi_ref_arr,
        selected_operator_labels=selected_operator_labels,
        executor=executor,
        logical_parameter_count=logical_parameter_count,
        runtime_parameter_count=runtime_parameter_count,
        num_parameters=num_parameters,
    )


def _solve_regularized_mclachlan_direction(
    A: np.ndarray,
    b: np.ndarray,
    *,
    regularization: float,
) -> np.ndarray:
    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1] or A_arr.shape[0] != b_arr.size:
        raise ValueError("Invalid McLachlan system shape.")
    if not np.all(np.isfinite(A_arr)) or not np.all(np.isfinite(b_arr)):
        raise ValueError("Non-finite McLachlan system encountered.")
    reg = max(0.0, float(regularization))
    lhs = A_arr + reg * np.eye(int(b_arr.size), dtype=float)
    rhs = -b_arr
    try:
        direction = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        direction = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    direction = np.asarray(direction, dtype=float).reshape(-1)
    if not np.all(np.isfinite(direction)):
        direction = np.asarray(np.linalg.pinv(lhs) @ rhs, dtype=float).reshape(-1)
    if not np.all(np.isfinite(direction)):
        raise ValueError("McLachlan solve produced no finite AVQITE direction.")
    return direction


def _sector_basis_index_lookup(
    sector_basis_full_indices: Sequence[int],
    *,
    full_state_dimension: int,
) -> tuple[np.ndarray, dict[int, int]]:
    basis_full = np.asarray(sector_basis_full_indices, dtype=int).reshape(-1)
    if basis_full.size <= 1:
        raise ValueError("QSCI requires a sector basis with dimension greater than one.")
    if np.any(basis_full < 0) or np.any(basis_full >= int(full_state_dimension)):
        raise ValueError("sector_basis_full_indices contains indices outside the full state vector.")
    lookup: dict[int, int] = {}
    for local_idx, full_idx in enumerate(basis_full.tolist()):
        full_i = int(full_idx)
        if full_i in lookup:
            raise ValueError(f"Duplicate sector basis full index {full_i} encountered.")
        lookup[full_i] = int(local_idx)
    return basis_full, lookup


def _project_sector_hamiltonian_dense(
    sector_hamiltonian: Any,
    selected_sector_indices: Sequence[int],
) -> np.ndarray:
    selected = np.asarray(selected_sector_indices, dtype=int).reshape(-1)
    if selected.size <= 0:
        raise ValueError("QSCI selected no sector basis states.")
    h_shape = getattr(sector_hamiltonian, "shape", None)
    if h_shape is None or len(tuple(h_shape)) != 2 or int(h_shape[0]) != int(h_shape[1]):
        raise ValueError("sector_hamiltonian must be a square matrix.")
    dim = int(h_shape[0])
    if np.any(selected < 0) or np.any(selected >= dim):
        raise ValueError("selected_sector_indices contains out-of-range sector positions.")
    if hasattr(sector_hamiltonian, "tocsr"):
        h_csr = sector_hamiltonian.tocsr()
        projected = h_csr[selected, :][:, selected].toarray()
    else:
        h_dense = np.asarray(sector_hamiltonian, dtype=complex)
        projected = h_dense[np.ix_(selected, selected)]
    projected_arr = np.asarray(projected, dtype=complex)
    if projected_arr.shape != (int(selected.size), int(selected.size)):
        raise ValueError("Projected QSCI Hamiltonian has an unexpected shape.")
    if not np.all(np.isfinite(projected_arr)):
        raise ValueError("Projected QSCI Hamiltonian contains non-finite entries.")
    return projected_arr


def _compiled_operator_trial_labels(
    *,
    setup: _CompiledOperatorTrialSetup,
    operator_labels: Sequence[str] | None,
) -> list[str]:
    labels = (
        [str(label) for label in operator_labels]
        if operator_labels is not None
        else list(setup.selected_operator_labels)
    )
    if len(labels) != len(setup.terms):
        raise ValueError("operator_labels must match operator_terms length.")
    return labels


def _reference_full_index_from_state(
    psi_ref_arr: np.ndarray,
    *,
    method_name: str,
) -> int:
    ref_amps = np.abs(np.asarray(psi_ref_arr, dtype=complex).reshape(-1))
    if ref_amps.size <= 0 or float(np.max(ref_amps)) <= 0.0:
        raise ValueError("psi_ref must contain a nonzero reference basis amplitude.")
    return int(np.argmax(ref_amps))


def _projected_sector_context(
    *,
    sector_hamiltonian: Any,
    sector_basis_full_indices: Sequence[int],
    psi_ref_arr: np.ndarray,
    method_name: str,
) -> tuple[int, np.ndarray, dict[int, int], int]:
    ref_full_index = _reference_full_index_from_state(psi_ref_arr, method_name=method_name)
    sector_full_indices, sector_lookup = _sector_basis_index_lookup(
        sector_basis_full_indices,
        full_state_dimension=int(psi_ref_arr.size),
    )
    full_sector_dimension = int(sector_full_indices.size)
    h_shape = getattr(sector_hamiltonian, "shape", None)
    if h_shape is None or int(h_shape[0]) != full_sector_dimension or int(h_shape[1]) != full_sector_dimension:
        raise ValueError(
            "sector_hamiltonian shape must match len(sector_basis_full_indices)."
        )
    if ref_full_index not in sector_lookup:
        raise ValueError(
            "The reference basis state is not present in the supplied HH sector basis."
        )
    return ref_full_index, sector_full_indices, sector_lookup, full_sector_dimension


def _prepare_single_operator_probe_states(
    *,
    setup: _CompiledOperatorTrialSetup,
    psi_ref_arr: np.ndarray,
    basis_probe_angle: float,
    method_name: str,
) -> list[np.ndarray]:
    angle = float(basis_probe_angle)
    if not np.isfinite(angle):
        raise ValueError("basis_probe_angle must be finite.")
    num_parameters = int(setup.num_parameters)
    if len(setup.terms) > num_parameters:
        raise ValueError(
            f"{method_name} requires one logical-shared parameter per probe operator."
        )
    probe_states: list[np.ndarray] = []
    for op_idx in range(len(setup.terms)):
        theta = np.zeros(num_parameters, dtype=float)
        theta[int(op_idx)] = angle
        probe_state = np.asarray(
            setup.executor.prepare_state(theta, psi_ref_arr),
            dtype=complex,
        ).reshape(-1)
        probe_state = _normalize_state(probe_state)
        if not np.all(np.isfinite(probe_state)):
            raise ValueError(f"{method_name} compiled operator probe produced a non-finite state.")
        probe_states.append(probe_state)
    return probe_states


def _diagonalize_projected_sector_subspace(
    *,
    sector_hamiltonian: Any,
    sector_lookup: Mapping[int, int],
    selected_full_indices: Sequence[int],
    psi_template: np.ndarray,
    method_name: str,
) -> tuple[float, np.ndarray, list[int]]:
    selected_full = [int(idx) for idx in selected_full_indices]
    if not selected_full:
        raise ValueError(f"{method_name} selected no sector basis states.")
    if len(set(selected_full)) != len(selected_full):
        raise ValueError(f"{method_name} selected duplicate sector basis states.")
    selected_sector_indices = []
    for full_idx in selected_full:
        if full_idx not in sector_lookup:
            raise ValueError(f"{method_name} selected a basis state outside the HH sector.")
        selected_sector_indices.append(int(sector_lookup[int(full_idx)]))
    projected_hamiltonian = _project_sector_hamiltonian_dense(
        sector_hamiltonian,
        selected_sector_indices,
    )
    evals, evecs = np.linalg.eigh(projected_hamiltonian)
    ground_pos = int(np.argmin(np.real(evals)))
    energy = float(np.real(evals[ground_pos]))
    psi_projected = np.zeros_like(psi_template, dtype=complex)
    ground_vec = np.asarray(evecs[:, ground_pos], dtype=complex).reshape(-1)
    for coeff, full_idx in zip(ground_vec.tolist(), selected_full):
        psi_projected[int(full_idx)] = complex(coeff)
    psi_projected = _normalize_state(psi_projected)
    return energy, psi_projected, selected_sector_indices


def run_compiled_operator_qsci_trial(
    *,
    operator_terms: Sequence[Any],
    operator_labels: Sequence[str] | None = None,
    ansatz_name: str,
    display_name: str,
    sector_hamiltonian: Any,
    sector_basis_full_indices: Sequence[int],
    psi_ref: np.ndarray,
    exact_gs: float,
    basis_probe_angle: float = np.pi / 2,
    basis_amp_cutoff: float = 1e-9,
    qsci_max_basis_states: int = 32,
    ai_log: Callable[..., None] | None = None,
) -> Mapping[str, Any]:
    """Run one benchmark-local QSCI projected-subspace diagonalization trial.

    The exact ground-state energy is reporting-only: the selected basis is built
    exclusively from the reference state and single-operator compiled probes,
    then the sector Hamiltonian is projected and diagonalized.  ``exact_gs`` is
    read only after the projected ground energy is computed.
    """
    setup = _build_compiled_operator_trial_setup(
        operator_terms=operator_terms,
        psi_ref=psi_ref,
        parameterization_mode="logical_shared",
    )
    terms = list(setup.terms)
    labels = _compiled_operator_trial_labels(setup=setup, operator_labels=operator_labels)

    angle = float(basis_probe_angle)
    if not np.isfinite(angle):
        raise ValueError("basis_probe_angle must be finite.")
    amp_cutoff = float(basis_amp_cutoff)
    if amp_cutoff < 0.0 or not np.isfinite(amp_cutoff):
        raise ValueError("basis_amp_cutoff must be non-negative and finite.")
    max_basis_states = int(qsci_max_basis_states)
    if max_basis_states <= 0:
        raise ValueError("qsci_max_basis_states must be positive.")

    psi_ref_arr = np.asarray(setup.psi_ref, dtype=complex).reshape(-1)
    ref_full_index, _sector_full_indices, sector_lookup, full_sector_dimension = _projected_sector_context(
        sector_hamiltonian=sector_hamiltonian,
        sector_basis_full_indices=sector_basis_full_indices,
        psi_ref_arr=psi_ref_arr,
        method_name="QSCI",
    )
    ref_amps = np.abs(psi_ref_arr)

    if ai_log is not None:
        ai_log(
            "trial_start",
            name=str(display_name),
            category="compiled_operator_qsci",
            selected_operator_count=int(len(terms)),
            full_sector_dimension=full_sector_dimension,
            qsci_max_basis_states=max_basis_states,
        )

    t0 = time.perf_counter()
    max_amp_by_full_index: dict[int, float] = {
        ref_full_index: float(ref_amps[ref_full_index])
    }
    for probe_state in _prepare_single_operator_probe_states(
        setup=setup,
        psi_ref_arr=psi_ref_arr,
        basis_probe_angle=angle,
        method_name="QSCI",
    ):
        probe_amps = np.abs(probe_state)
        support = np.flatnonzero(probe_amps >= amp_cutoff)
        for full_idx_raw in support.tolist():
            full_idx = int(full_idx_raw)
            if full_idx not in sector_lookup:
                continue
            amp = float(probe_amps[full_idx])
            old_amp = max_amp_by_full_index.get(full_idx)
            if old_amp is None or amp > old_amp:
                max_amp_by_full_index[full_idx] = amp

    max_subspace_dimension = min(max_basis_states, full_sector_dimension - 1)
    if max_subspace_dimension <= 1:
        raise ValueError("QSCI anti-full-sector cap leaves fewer than two basis states.")
    ranked_non_ref = [
        full_idx for full_idx in max_amp_by_full_index if int(full_idx) != ref_full_index
    ]
    ranked_non_ref.sort(
        key=lambda full_idx: (
            -float(max_amp_by_full_index[int(full_idx)]),
            int(full_idx),
        )
    )
    selected_full_indices = [ref_full_index] + ranked_non_ref[: max_subspace_dimension - 1]
    if len(selected_full_indices) <= 1:
        raise ValueError("QSCI selected subspace must contain more than one sector basis state.")
    if len(selected_full_indices) >= full_sector_dimension:
        raise ValueError(
            "QSCI selected subspace reached the full HH sector; refusing full-sector ED."
        )

    energy, psi_qsci, selected_sector_indices = _diagonalize_projected_sector_subspace(
        sector_hamiltonian=sector_hamiltonian,
        sector_lookup=sector_lookup,
        selected_full_indices=selected_full_indices,
        psi_template=psi_ref_arr,
        method_name="QSCI",
    )
    elapsed_s = float(time.perf_counter() - t0)

    exact_energy = float(exact_gs)
    delta_e = float(energy - exact_energy)
    if ai_log is not None:
        ai_log(
            "trial_done",
            name=str(display_name),
            category="compiled_operator_qsci",
            energy=energy,
            delta_e=delta_e,
            subspace_dimension=int(len(selected_full_indices)),
            full_sector_dimension=full_sector_dimension,
            elapsed_s=round(elapsed_s, 2),
        )

    basis_probe_count = int(len(terms))
    return {
        "success": True,
        "method_kind": "qsci",
        "display_name": str(display_name),
        "name": str(display_name),
        "ansatz_kind": "compiled_operator_qsci",
        "ansatz_name": str(ansatz_name),
        "energy": energy,
        "exact_gs_energy": exact_energy,
        "exact_energy": exact_energy,
        "delta_e": delta_e,
        "abs_delta_e": abs(delta_e),
        "delta_E_abs": abs(delta_e),
        "nfev_total": basis_probe_count,
        "nfev": basis_probe_count,
        "nit": 0,
        "num_parameters": None,
        "num_params": None,
        "logical_parameter_count": int(setup.logical_parameter_count),
        "runtime_parameter_count": int(setup.runtime_parameter_count),
        "vqe_reps": None,
        "vqe_restarts": None,
        "vqe_maxiter_used": None,
        "vqe_maxiter": None,
        "optimizer": "projected_diagonalization",
        "optimizer_success": True,
        "optimizer_message": "projected_diag",
        "converged": True,
        "runtime_s": elapsed_s,
        "selected_operator_labels": labels,
        "selected_operator_count": int(len(terms)),
        "selected_basis_full_indices": [int(idx) for idx in selected_full_indices],
        "selected_sector_indices": [int(idx) for idx in selected_sector_indices],
        "subspace_dimension": int(len(selected_full_indices)),
        "full_sector_dimension": full_sector_dimension,
        "qsci_basis_probe_count": basis_probe_count,
        "qsci_candidate_basis_count": int(len(max_amp_by_full_index)),
        "qsci_basis_selection_mode": "single_operator_support_union_top_amp",
        "qsci_stop_reason": "projected_diag",
        "qsci_basis_probe_angle": angle,
        "qsci_basis_amp_cutoff": amp_cutoff,
        "qsci_max_basis_states": max_basis_states,
        "_psi_vqe": psi_qsci,
    }


def run_compiled_operator_sqd_trial(
    *,
    operator_terms: Sequence[Any],
    operator_labels: Sequence[str] | None = None,
    ansatz_name: str,
    display_name: str,
    sector_hamiltonian: Any,
    sector_basis_full_indices: Sequence[int],
    psi_ref: np.ndarray,
    exact_gs: float,
    basis_probe_angle: float = np.pi / 2,
    sqd_shots_per_probe: int = 256,
    sqd_max_basis_states: int = 32,
    sqd_seed: int = 7,
    ai_log: Callable[..., None] | None = None,
) -> Mapping[str, Any]:
    """Run one benchmark-local sampled quantum diagonalization trial.

    SQD selection is built only from samples of the reference-prepared
    single-operator probe states.  The exact ground-state energy is consumed
    after projected diagonalization solely to populate reporting deltas.
    """
    setup = _build_compiled_operator_trial_setup(
        operator_terms=operator_terms,
        psi_ref=psi_ref,
        parameterization_mode="logical_shared",
    )
    terms = list(setup.terms)
    labels = _compiled_operator_trial_labels(setup=setup, operator_labels=operator_labels)

    angle = float(basis_probe_angle)
    if not np.isfinite(angle):
        raise ValueError("basis_probe_angle must be finite.")
    shots_per_probe = int(sqd_shots_per_probe)
    if shots_per_probe <= 0:
        raise ValueError("sqd_shots_per_probe must be positive.")
    max_basis_states = int(sqd_max_basis_states)
    if max_basis_states <= 0:
        raise ValueError("sqd_max_basis_states must be positive.")
    seed = int(sqd_seed)

    psi_ref_arr = np.asarray(setup.psi_ref, dtype=complex).reshape(-1)
    ref_full_index, _sector_full_indices, sector_lookup, full_sector_dimension = _projected_sector_context(
        sector_hamiltonian=sector_hamiltonian,
        sector_basis_full_indices=sector_basis_full_indices,
        psi_ref_arr=psi_ref_arr,
        method_name="SQD",
    )
    ref_probability = float(abs(psi_ref_arr[ref_full_index]) ** 2)

    if ai_log is not None:
        ai_log(
            "trial_start",
            name=str(display_name),
            category="compiled_operator_sqd",
            selected_operator_count=int(len(terms)),
            full_sector_dimension=full_sector_dimension,
            sqd_shots_per_probe=shots_per_probe,
            sqd_max_basis_states=max_basis_states,
            sqd_seed=seed,
        )

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    sample_count_by_full_index: dict[int, int] = {}
    max_prob_by_full_index: dict[int, float] = {ref_full_index: ref_probability}
    probe_states = _prepare_single_operator_probe_states(
        setup=setup,
        psi_ref_arr=psi_ref_arr,
        basis_probe_angle=angle,
        method_name="SQD",
    )
    sampled_sector_shots = 0
    for probe_state in probe_states:
        probabilities = np.asarray(np.abs(probe_state) ** 2, dtype=float).reshape(-1)
        total_probability = float(np.sum(probabilities))
        if total_probability <= 0.0 or not np.isfinite(total_probability):
            raise ValueError("SQD probe state produced invalid sampling probabilities.")
        probabilities = probabilities / total_probability
        sampled_full_indices = rng.choice(
            int(probabilities.size),
            size=shots_per_probe,
            replace=True,
            p=probabilities,
        )
        for full_idx_raw in sampled_full_indices.tolist():
            full_idx = int(full_idx_raw)
            if full_idx not in sector_lookup:
                continue
            sampled_sector_shots += 1
            sample_count_by_full_index[full_idx] = int(sample_count_by_full_index.get(full_idx, 0) + 1)
            prob = float(probabilities[full_idx])
            old_prob = max_prob_by_full_index.get(full_idx)
            if old_prob is None or prob > old_prob:
                max_prob_by_full_index[full_idx] = prob

    max_subspace_dimension = min(max_basis_states, full_sector_dimension - 1)
    if max_subspace_dimension < 1:
        raise ValueError("SQD anti-full-sector cap leaves no basis states.")
    ranked_non_ref = [
        full_idx for full_idx in sample_count_by_full_index if int(full_idx) != ref_full_index
    ]
    ranked_non_ref.sort(
        key=lambda full_idx: (
            -int(sample_count_by_full_index[int(full_idx)]),
            -float(max_prob_by_full_index.get(int(full_idx), 0.0)),
            int(full_idx),
        )
    )
    selected_full_indices = [ref_full_index] + ranked_non_ref[: max_subspace_dimension - 1]
    if len(selected_full_indices) >= full_sector_dimension:
        selected_full_indices = selected_full_indices[: max(1, full_sector_dimension - 1)]
    if len(selected_full_indices) >= full_sector_dimension:
        raise ValueError("SQD selected subspace reached the full HH sector; refusing full-sector ED.")
    stop_reason = "reference_only" if len(selected_full_indices) == 1 else "projected_diag"

    energy, psi_sqd, selected_sector_indices = _diagonalize_projected_sector_subspace(
        sector_hamiltonian=sector_hamiltonian,
        sector_lookup=sector_lookup,
        selected_full_indices=selected_full_indices,
        psi_template=psi_ref_arr,
        method_name="SQD",
    )
    elapsed_s = float(time.perf_counter() - t0)

    exact_energy = float(exact_gs)
    delta_e = float(energy - exact_energy)
    basis_probe_count = int(len(terms))
    shots_total = int(shots_per_probe * basis_probe_count)
    if ai_log is not None:
        ai_log(
            "trial_done",
            name=str(display_name),
            category="compiled_operator_sqd",
            energy=energy,
            delta_e=delta_e,
            subspace_dimension=int(len(selected_full_indices)),
            full_sector_dimension=full_sector_dimension,
            shots_total=shots_total,
            elapsed_s=round(elapsed_s, 2),
        )

    sample_counts_payload = {
        str(int(idx)): int(count)
        for idx, count in sorted(sample_count_by_full_index.items(), key=lambda item: int(item[0]))
    }
    max_probs_payload = {
        str(int(idx)): float(max_prob_by_full_index[int(idx)])
        for idx in sorted(max_prob_by_full_index)
        if int(idx) in sample_count_by_full_index or int(idx) == ref_full_index
    }
    return {
        "success": True,
        "method_kind": "sqd",
        "display_name": str(display_name),
        "name": str(display_name),
        "ansatz_kind": "compiled_operator_sqd",
        "ansatz_name": str(ansatz_name),
        "energy": energy,
        "exact_gs_energy": exact_energy,
        "exact_energy": exact_energy,
        "delta_e": delta_e,
        "abs_delta_e": abs(delta_e),
        "delta_E_abs": abs(delta_e),
        "nfev_total": basis_probe_count,
        "nfev": basis_probe_count,
        "nit": 0,
        "num_parameters": None,
        "num_params": None,
        "logical_parameter_count": int(setup.logical_parameter_count),
        "runtime_parameter_count": int(setup.runtime_parameter_count),
        "vqe_reps": None,
        "vqe_restarts": None,
        "vqe_maxiter_used": None,
        "vqe_maxiter": None,
        "optimizer": "projected_diagonalization",
        "optimizer_success": True,
        "optimizer_message": str(stop_reason),
        "converged": True,
        "runtime_s": elapsed_s,
        "selected_operator_labels": labels,
        "selected_operator_count": int(len(terms)),
        "selected_basis_full_indices": [int(idx) for idx in selected_full_indices],
        "selected_sector_indices": [int(idx) for idx in selected_sector_indices],
        "subspace_dimension": int(len(selected_full_indices)),
        "full_sector_dimension": full_sector_dimension,
        "shots_total": shots_total,
        "sqd_basis_probe_count": basis_probe_count,
        "sqd_shots_per_probe": shots_per_probe,
        "sqd_sampled_sector_shots": int(sampled_sector_shots),
        "sqd_candidate_basis_count": int(len(sample_count_by_full_index)),
        "sqd_basis_selection_mode": "single_operator_probe_shot_counts",
        "sqd_seed": seed,
        "sqd_stop_reason": str(stop_reason),
        "sqd_basis_probe_angle": angle,
        "sqd_max_basis_states": max_basis_states,
        "sqd_sample_counts_by_full_index": sample_counts_payload,
        "sqd_max_observed_probability_by_full_index": max_probs_payload,
        "_psi_vqe": psi_sqd,
    }


def run_compiled_operator_vqe_trial(
    *,
    operator_terms: Sequence[Any],
    ansatz_name: str,
    display_name: str,
    h_poly: Any,
    exact_gs: float,
    psi_ref: np.ndarray,
    optimizer: str,
    maxiter: int,
    restarts: int,
    seed: int,
    parameterization_mode: str = "logical_shared",
    ai_log: Callable[..., None] | None = None,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | Mapping[str, Any] | None = None,
    benchmark_decision_noise_scope: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Run one benchmark-only fixed operator-list VQE trial.

    The operator list is compiled through ``CompiledAnsatzExecutor`` and then
    optimized with the same ``vqe_minimize`` boundary as the existing HH fixed
    ansatz rows.  For the lifted-UCCSD benchmark row, ``logical_shared`` means
    one variational parameter per lifted UCCSD generator.
    """
    setup = _build_compiled_operator_trial_setup(
        operator_terms=operator_terms,
        psi_ref=psi_ref,
        parameterization_mode=parameterization_mode,
    )
    if ai_log is not None:
        ai_log(
            "building_ansatz",
            name=str(display_name),
            category="compiled_operator_vqe",
            selected_operator_count=int(len(setup.terms)),
            parameterization_mode=str(setup.mode),
        )

    executor = setup.executor
    logical_parameter_count = int(setup.logical_parameter_count)
    runtime_parameter_count = int(setup.runtime_parameter_count)
    num_parameters = int(setup.num_parameters)
    noise_scope = {
        "family": "hh",
        "algorithm_id": str(ansatz_name),
        "surface_family": "hh_compiled_operator_vqe",
        **dict(benchmark_decision_noise_scope or {}),
    }
    decision_noise_config = coerce_benchmark_decision_noise_config(
        benchmark_decision_noise_config,
        family=str(noise_scope.get("family") or "hh"),
        case_id=str(noise_scope.get("case_id") or ""),
        algorithm_id=str(noise_scope.get("algorithm_id") or ansatz_name),
    )
    decision_noise_recorder = BenchmarkDecisionNoiseRecorder(
        decision_noise_config,
        base_scope=noise_scope,
    )

    def _objective_value_transform(event: Mapping[str, Any]) -> float:
        return float(
            decision_noise_recorder.apply(
                float(event["energy_ideal"]),
                surface="hh_compiled_vqe_objective",
                value_kind="energy",
                phase="optimizer",
                extra_scope={
                    "restart_index": int(event.get("restart_index", 0) or 0),
                    "nfev_restart": int(event.get("nfev_restart", 0) or 0),
                    "nfev_total_estimate": int(event.get("nfev_total_estimate", 0) or 0),
                    "ansatz_name": str(ansatz_name),
                    "parameterization_mode": str(setup.mode),
                },
            )
        )

    if ai_log is not None:
        ai_log(
            "trial_start",
            name=str(display_name),
            category="compiled_operator_vqe",
            num_params=num_parameters,
            logical_parameter_count=logical_parameter_count,
            runtime_parameter_count=runtime_parameter_count,
        )

    t0 = time.perf_counter()
    vqe_kwargs: dict[str, Any] = {
        "restarts": int(restarts),
        "seed": int(seed),
        "maxiter": int(maxiter),
        "method": str(optimizer),
    }
    if bool(decision_noise_config.enabled):
        vqe_kwargs["objective_value_transform"] = _objective_value_transform
    result = vqe_minimize(
        h_poly,
        executor,
        setup.psi_ref,
        **vqe_kwargs,
    )
    theta = np.asarray(result.theta, dtype=float)
    psi_vqe = np.asarray(executor.prepare_state(theta, setup.psi_ref), dtype=complex).ravel()
    psi_vqe = _normalize_state(psi_vqe)
    elapsed_s = float(time.perf_counter() - t0)
    optimizer_decision_energy = float(result.energy)
    energy = (
        _final_exact_energy_from_state(h_poly, psi_vqe)
        if bool(decision_noise_config.enabled)
        else optimizer_decision_energy
    )
    exact_energy = float(exact_gs)
    delta_e = float(energy - exact_energy)
    decision_noise_metadata = None
    if bool(decision_noise_config.enabled):
        decision_noise_metadata = copy_decision_noise_metadata(
            decision_noise_recorder.summary(
                status="ok",
                supported=True,
                extra={"runner": "hh_compiled_operator_vqe", "ansatz_name": str(ansatz_name)},
            )
        )
    optimizer_success = bool(getattr(result, "success", False))
    optimizer_message = str(getattr(result, "message", ""))
    if ai_log is not None:
        ai_log(
            "trial_done",
            name=str(display_name),
            energy=energy,
            delta_e=delta_e,
            elapsed_s=round(elapsed_s, 2),
            optimizer_success=optimizer_success,
        )

    payload = {
        "success": True,
        "method_kind": "conventional_vqe",
        "display_name": str(display_name),
        "name": str(display_name),
        "ansatz_kind": "compiled_operator",
        "ansatz_name": str(ansatz_name),
        "energy": energy,
        "exact_gs_energy": exact_energy,
        "exact_energy": exact_energy,
        "delta_e": delta_e,
        "abs_delta_e": abs(delta_e),
        "delta_E_abs": abs(delta_e),
        "nfev": int(getattr(result, "nfev", 0)),
        "nit": int(getattr(result, "nit", 0)),
        "num_parameters": num_parameters,
        "num_params": num_parameters,
        "logical_parameter_count": logical_parameter_count,
        "runtime_parameter_count": runtime_parameter_count,
        "vqe_reps": None,
        "vqe_restarts": int(restarts),
        "vqe_maxiter_used": int(maxiter),
        "vqe_maxiter": int(maxiter),
        "optimizer": str(optimizer),
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "optimizer_decision_energy": optimizer_decision_energy,
        "optimizer_reported_energy": optimizer_decision_energy,
        "converged": optimizer_success,
        "best_restart": int(getattr(result, "best_restart", -1)),
        "runtime_s": elapsed_s,
        "theta": theta.tolist(),
        "restart_summaries": getattr(result, "restart_summaries", None),
        "parameterization_mode": str(setup.mode),
        "selected_operator_labels": list(setup.selected_operator_labels),
        "selected_operator_count": int(len(setup.terms)),
        "_psi_vqe": psi_vqe,
    }
    if decision_noise_metadata is not None:
        payload.update(
            {
                "benchmark_decision_noise_status": "ok",
                "benchmark_decision_noise": copy_decision_noise_metadata(decision_noise_metadata),
            }
        )
    return payload


def run_compiled_operator_avqite_trial(
    *,
    operator_terms: Sequence[Any],
    ansatz_name: str,
    display_name: str,
    h_poly: Any,
    exact_gs: float,
    psi_ref: np.ndarray,
    parameterization_mode: str = "logical_shared",
    avqite_step_size: float = 0.1,
    avqite_max_steps: int = 80,
    avqite_energy_tol: float = 1e-8,
    avqite_residual_tol: float = 1e-6,
    avqite_derivative_eps: float = 1e-4,
    avqite_regularization: float = 1e-8,
    avqite_backtrack_factor: float = 0.5,
    avqite_max_backtracks: int = 6,
    ai_log: Callable[..., None] | None = None,
) -> Mapping[str, Any]:
    """Run one benchmark-local compiled-operator AVQITE trial.

    The exact ground energy is deliberately diagnostic-only: the AVQITE control
    loop uses only the prepared ansatz state, finite-difference tangents, and
    Hamiltonian expectation data.  ``exact_gs`` is consumed only after the final
    state is selected to populate reporting deltas.
    """
    setup = _build_compiled_operator_trial_setup(
        operator_terms=operator_terms,
        psi_ref=psi_ref,
        parameterization_mode=parameterization_mode,
    )
    executor = setup.executor
    num_parameters = int(setup.num_parameters)
    if num_parameters <= 0:
        raise ValueError("compiled-operator AVQITE requires at least one active parameter.")
    step_size_initial = float(avqite_step_size)
    if step_size_initial <= 0.0 or not np.isfinite(step_size_initial):
        raise ValueError("avqite_step_size must be positive and finite.")
    max_steps = int(avqite_max_steps)
    if max_steps < 0:
        raise ValueError("avqite_max_steps must be non-negative.")
    derivative_eps = float(avqite_derivative_eps)
    if derivative_eps <= 0.0 or not np.isfinite(derivative_eps):
        raise ValueError("avqite_derivative_eps must be positive and finite.")
    backtrack_factor = float(avqite_backtrack_factor)
    if not (0.0 < backtrack_factor < 1.0):
        raise ValueError("avqite_backtrack_factor must be in the open interval (0, 1).")
    max_backtracks = int(avqite_max_backtracks)
    if max_backtracks < 0:
        raise ValueError("avqite_max_backtracks must be non-negative.")
    energy_tol = max(0.0, float(avqite_energy_tol))
    residual_tol = max(0.0, float(avqite_residual_tol))
    regularization = max(0.0, float(avqite_regularization))

    if ai_log is not None:
        ai_log(
            "building_ansatz",
            name=str(display_name),
            category="compiled_operator_avqite",
            selected_operator_count=int(len(setup.terms)),
            parameterization_mode=str(setup.mode),
        )
        ai_log(
            "trial_start",
            name=str(display_name),
            category="compiled_operator_avqite",
            num_params=num_parameters,
            logical_parameter_count=int(setup.logical_parameter_count),
            runtime_parameter_count=int(setup.runtime_parameter_count),
        )

    compiled_h = compile_polynomial_action(h_poly, tol=1e-12)
    state_preparations_total = 0
    energy_evaluations_total = 0

    def _prepare_state(theta_vec: np.ndarray) -> np.ndarray:
        nonlocal state_preparations_total
        psi = np.asarray(executor.prepare_state(theta_vec, setup.psi_ref), dtype=complex).ravel()
        psi = _normalize_state(psi)
        if not np.all(np.isfinite(psi)):
            raise ValueError("Compiled-operator AVQITE prepared a non-finite state.")
        state_preparations_total += 1
        return psi

    def _energy_state_hpsi(theta_vec: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        nonlocal energy_evaluations_total
        psi = _prepare_state(theta_vec)
        energy, hpsi = energy_via_one_apply(psi, compiled_h)
        if not np.isfinite(float(energy)) or not np.all(np.isfinite(hpsi)):
            raise ValueError("Compiled-operator AVQITE produced non-finite Hamiltonian data.")
        energy_evaluations_total += 1
        return float(energy), psi, np.asarray(hpsi, dtype=complex).ravel()

    def _finite_difference_tangents(theta_vec: np.ndarray, psi_current: np.ndarray) -> list[np.ndarray]:
        tangents: list[np.ndarray] = []
        tangent_norm_max = 0.0
        for idx in range(num_parameters):
            shift = np.zeros(num_parameters, dtype=float)
            shift[idx] = derivative_eps
            psi_plus = _prepare_state(theta_vec + shift)
            psi_minus = _prepare_state(theta_vec - shift)
            tangent = np.asarray((psi_plus - psi_minus) / (2.0 * derivative_eps), dtype=complex).ravel()
            if tangent.size != psi_current.size or not np.all(np.isfinite(tangent)):
                raise ValueError("Compiled-operator AVQITE finite-difference tangent is invalid.")
            tangent_norm_max = max(tangent_norm_max, float(np.linalg.norm(tangent)))
            tangents.append(tangent)
        if tangent_norm_max <= 1e-14:
            raise ValueError("No nonzero finite-difference tangent could be computed for AVQITE.")
        return tangents

    def _mclachlan_system(
        *,
        psi_current: np.ndarray,
        hpsi_current: np.ndarray,
        energy_current: float,
        tangents: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        overlaps = np.array(
            [float(np.real(np.vdot(tangent, psi_current))) for tangent in tangents],
            dtype=float,
        )
        A = np.empty((num_parameters, num_parameters), dtype=float)
        for i, tangent_i in enumerate(tangents):
            for j, tangent_j in enumerate(tangents):
                A[i, j] = float(np.real(np.vdot(tangent_i, tangent_j))) - overlaps[i] * overlaps[j]
        b = np.array(
            [
                float(np.real(np.vdot(tangent, hpsi_current)))
                - float(energy_current) * overlaps[idx]
                for idx, tangent in enumerate(tangents)
            ],
            dtype=float,
        )
        residual_max = float(np.max(np.abs(b))) if b.size else 0.0
        return A, b, residual_max

    t0 = time.perf_counter()
    theta = np.zeros(num_parameters, dtype=float)
    energy, psi_current, hpsi_current = _energy_state_hpsi(theta)
    history: list[dict[str, Any]] = [
        {
            "event": "initial",
            "step": 0,
            "energy": float(energy),
            "imaginary_time": 0.0,
        }
    ]
    steps_completed = 0
    imaginary_time_total = 0.0
    stop_reason = "max_steps"

    for step_idx in range(max_steps):
        tangents = _finite_difference_tangents(theta, psi_current)
        A, b, residual_max = _mclachlan_system(
            psi_current=psi_current,
            hpsi_current=hpsi_current,
            energy_current=energy,
            tangents=tangents,
        )
        if residual_max <= residual_tol:
            stop_reason = "residual_tol"
            history.append(
                {
                    "event": "stop",
                    "step": int(step_idx),
                    "energy": float(energy),
                    "residual_max": residual_max,
                    "stop_reason": stop_reason,
                }
            )
            break
        direction = _solve_regularized_mclachlan_direction(
            A,
            b,
            regularization=regularization,
        )
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 0.0 or not np.isfinite(direction_norm):
            raise ValueError("McLachlan solve produced a zero or non-finite AVQITE direction.")

        accepted = False
        trial_dt = step_size_initial
        backtrack_trials: list[dict[str, Any]] = []
        energy_before = float(energy)
        for backtrack_idx in range(max_backtracks + 1):
            theta_trial = theta + trial_dt * direction
            trial_energy, trial_psi, trial_hpsi = _energy_state_hpsi(theta_trial)
            trial_record = {
                "backtrack_index": int(backtrack_idx),
                "step_size": float(trial_dt),
                "energy": float(trial_energy),
                "accepted": bool(trial_energy <= energy_before + 1.0e-10),
            }
            backtrack_trials.append(trial_record)
            if bool(trial_record["accepted"]):
                theta = theta_trial
                energy = float(trial_energy)
                psi_current = trial_psi
                hpsi_current = trial_hpsi
                steps_completed += 1
                imaginary_time_total += float(trial_dt)
                accepted = True
                history.append(
                    {
                        "event": "accepted_step",
                        "step": int(steps_completed),
                        "energy_before": energy_before,
                        "energy": float(energy),
                        "energy_delta": float(energy - energy_before),
                        "step_size": float(trial_dt),
                        "backtracks": int(backtrack_idx),
                        "residual_max": residual_max,
                        "direction_norm": direction_norm,
                        "imaginary_time": float(imaginary_time_total),
                        "backtrack_trials": backtrack_trials,
                    }
                )
                if abs(float(energy_before) - float(energy)) <= energy_tol:
                    stop_reason = "energy_tol"
                break
            trial_dt *= backtrack_factor

        if not accepted:
            stop_reason = "step_underflow"
            history.append(
                {
                    "event": "rejected_step",
                    "step": int(step_idx + 1),
                    "energy": energy_before,
                    "residual_max": residual_max,
                    "direction_norm": direction_norm,
                    "backtrack_trials": backtrack_trials,
                    "stop_reason": stop_reason,
                }
            )
            break
        if stop_reason == "energy_tol":
            break
    else:
        stop_reason = "max_steps"

    elapsed_s = float(time.perf_counter() - t0)
    exact_energy = float(exact_gs)
    delta_e = float(energy - exact_energy)
    converged = stop_reason in {"residual_tol", "energy_tol"}
    if ai_log is not None:
        ai_log(
            "trial_done",
            name=str(display_name),
            energy=float(energy),
            elapsed_s=round(elapsed_s, 2),
            avqite_steps_completed=int(steps_completed),
            avqite_stop_reason=str(stop_reason),
        )

    return {
        "success": True,
        "method_kind": "avqite",
        "display_name": str(display_name),
        "name": str(display_name),
        "ansatz_kind": "compiled_operator",
        "ansatz_name": str(ansatz_name),
        "energy": float(energy),
        "exact_gs_energy": exact_energy,
        "exact_energy": exact_energy,
        "delta_e": delta_e,
        "abs_delta_e": abs(delta_e),
        "delta_E_abs": abs(delta_e),
        "nfev_total": int(state_preparations_total),
        "nfev": int(state_preparations_total),
        "energy_evaluations_total": int(energy_evaluations_total),
        "state_preparations_total": int(state_preparations_total),
        "nit": int(steps_completed),
        "num_parameters": int(num_parameters),
        "num_params": int(num_parameters),
        "logical_parameter_count": int(setup.logical_parameter_count),
        "runtime_parameter_count": int(setup.runtime_parameter_count),
        "vqe_reps": None,
        "vqe_restarts": None,
        "vqe_maxiter_used": None,
        "vqe_maxiter": None,
        "optimizer": "AVQITE",
        "optimizer_success": bool(converged),
        "optimizer_message": str(stop_reason),
        "converged": bool(converged),
        "runtime_s": elapsed_s,
        "theta": theta.tolist(),
        "parameterization_mode": str(setup.mode),
        "selected_operator_labels": list(setup.selected_operator_labels),
        "selected_operator_count": int(len(setup.terms)),
        "avqite_steps_completed": int(steps_completed),
        "imaginary_time_total": float(imaginary_time_total),
        "avqite_stop_reason": str(stop_reason),
        "avqite_step_size_initial": float(step_size_initial),
        "avqite_max_steps": int(max_steps),
        "avqite_energy_tol": float(energy_tol),
        "avqite_residual_tol": float(residual_tol),
        "avqite_derivative_eps": float(derivative_eps),
        "avqite_regularization": float(regularization),
        "avqite_backtrack_factor": float(backtrack_factor),
        "avqite_max_backtracks": int(max_backtracks),
        "history": history,
        "_psi_vqe": psi_current,
    }


__all__ = [
    "CompiledOperatorParameterizationMode",
    "HHConventionalAnsatzKind",
    "default_hh_conventional_vqe_config",
    "has_qiskit_hea_support",
    "run_compiled_operator_avqite_trial",
    "run_compiled_operator_qsci_trial",
    "run_compiled_operator_sqd_trial",
    "run_compiled_operator_vqe_trial",
    "run_hh_conventional_vqe_trial",
]
