#!/usr/bin/env python3
"""Benchmark-local Qiskit parity helpers for generic time-dynamics comparators.

This module is parity-only: it validates repo-native dynamics comparator pieces
against lazily imported Qiskit statevector circuits.  It does not provide a
Qiskit-primary controller or any online McLachlan decision backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.qiskit_pauli_tools import (
    QiskitPauliUnavailable,
    append_pauli_rotation_exyz,
    build_runtime_layout_circuit,
    circuit_stats,
    import_qiskit_pauli_components,
    to_ixyz_label,
)
from src.quantum.ansatz_parameterization import deserialize_layout

QISKIT_DYNAMICS_PARITY_SCHEMA = "dynamics_qiskit_parity_v1"
QISKIT_DYNAMICS_MODE_VALUES: tuple[str, ...] = ("off", "parity", "parity_required")
MAPPING_CONVENTION = "repo_exyz_left_to_right_qnminus1_to_q0__qiskit_qubit0_rightmost"


@dataclass(frozen=True)
class QiskitDynamicsConfig:
    mode: str = "off"
    qubit_cap: int | None = 12
    state_l2_tol: float = 1.0e-8
    infidelity_tol: float = 1.0e-10
    energy_abs_tol: float = 1.0e-8
    export_circuits: bool = False
    time_segmentation: str = "match_native_interval"

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


class QiskitDynamicsParityUnavailable(QiskitPauliUnavailable):
    """Raised when optional dynamics parity support cannot be imported."""


def qiskit_dynamics_config_from_metadata(metadata: Mapping[str, Any] | None) -> QiskitDynamicsConfig:
    """Resolve the additive Qiskit parity config from case metadata."""

    meta = metadata if isinstance(metadata, Mapping) else {}
    nested = meta.get("qiskit_dynamics", {}) if isinstance(meta.get("qiskit_dynamics", {}), Mapping) else {}
    mode = str(nested.get("mode", meta.get("qiskit_dynamics_mode", meta.get("qiskit_parity_mode", "off")))).strip()
    if not mode:
        mode = "off"
    if mode not in QISKIT_DYNAMICS_MODE_VALUES:
        raise ValueError(
            f"unsupported qiskit dynamics mode {mode!r}; parity support currently allows "
            f"{QISKIT_DYNAMICS_MODE_VALUES!r} only"
        )

    raw_cap = nested.get("qubit_cap", meta.get("qiskit_qubit_cap", 12))
    qubit_cap = None if raw_cap in {None, "", "none", "None"} else int(raw_cap)
    return QiskitDynamicsConfig(
        mode=mode,
        qubit_cap=qubit_cap,
        state_l2_tol=float(nested.get("state_l2_tol", meta.get("qiskit_state_l2_tol", 1.0e-8))),
        infidelity_tol=float(nested.get("infidelity_tol", meta.get("qiskit_infidelity_tol", 1.0e-10))),
        energy_abs_tol=float(nested.get("energy_abs_tol", meta.get("qiskit_energy_abs_tol", 1.0e-8))),
        export_circuits=bool(nested.get("export_circuits", meta.get("qiskit_export_circuits", False))),
        time_segmentation=str(nested.get("time_segmentation", meta.get("qiskit_time_segmentation", "match_native_interval"))),
    )


def qiskit_dynamics_config_from_case(case: Any) -> QiskitDynamicsConfig:
    return qiskit_dynamics_config_from_metadata(getattr(case, "metadata", {}) or {})


def parity_requested(config: QiskitDynamicsConfig) -> bool:
    return str(config.mode) in {"parity", "parity_required"}


def _base_result(
    *,
    config: QiskitDynamicsConfig,
    algorithm_id: str,
    family: str,
    case_id: str,
    status: str,
    passed: bool | None,
    reason: str | None = None,
    support_scope: str = "parity_only",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": QISKIT_DYNAMICS_PARITY_SCHEMA,
        "algorithm_id": str(algorithm_id),
        "family": str(family),
        "case_id": str(case_id),
        "mode": str(config.mode),
        "status": str(status),
        "passed": passed,
        "reason": reason,
        "support_scope": str(support_scope),
        "mapping_convention": MAPPING_CONVENTION,
        "internal_pauli_alphabet": "e/x/y/z",
        "qiskit_label_alphabet": "I/X/Y/Z",
        "time_segmentation": str(config.time_segmentation),
        "config": config.to_dict(),
        "controller_decisions_modified": False,
        "exact_reference_controller_inputs": False,
        "exact_data_policy": "diagnostic_only_not_decision_input",
    }
    if extra:
        payload.update(dict(extra))
    return _json_safe(payload)


def skipped_optional_dependency_result(
    *,
    config: QiskitDynamicsConfig,
    algorithm_id: str,
    family: str,
    case_id: str,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    return _base_result(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        status="skipped_optional_dependency",
        passed=None,
        reason=str(exc) if exc is not None else "optional Qiskit dependency is unavailable",
        extra={"qiskit_available": False},
    )


def skipped_resource_guard_result(
    *,
    config: QiskitDynamicsConfig,
    algorithm_id: str,
    family: str,
    case_id: str,
    num_qubits: int,
) -> dict[str, Any]:
    return _base_result(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        status="skipped_resource_guard",
        passed=None,
        reason=f"num_qubits={int(num_qubits)} exceeds qiskit_qubit_cap={config.qubit_cap}",
        extra={"qiskit_available": True, "num_qubits": int(num_qubits)},
    )


def failed_result(
    *,
    config: QiskitDynamicsConfig,
    algorithm_id: str,
    family: str,
    case_id: str,
    exc: BaseException | str,
    support_scope: str = "parity_only",
) -> dict[str, Any]:
    return _base_result(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        status="failed",
        passed=False,
        reason=str(exc),
        support_scope=support_scope,
        extra={"qiskit_available": True},
    )


def not_applicable_result(
    *,
    config: QiskitDynamicsConfig,
    algorithm_id: str,
    family: str,
    case_id: str,
    reason: str,
    support_scope: str = "post_run_validation_only",
) -> dict[str, Any]:
    return _base_result(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        status="not_applicable",
        passed=None,
        reason=str(reason),
        support_scope=support_scope,
        extra={"qiskit_available": None},
    )


def fixed_mclachlan_not_applicable_result(*, config: QiskitDynamicsConfig, case: Any) -> dict[str, Any] | None:
    if not parity_requested(config):
        return None
    return not_applicable_result(
        config=config,
        algorithm_id="dyn_fixed_mclachlan",
        family=str(getattr(case, "family", "")),
        case_id=str(getattr(case, "case_id", "")),
        reason="missing_serialized_fixed_scaffold_layout_or_theta_payload",
        support_scope="post_run_fixed_scaffold_parity_only_no_controller_decisions",
    )


def _complex_vector_from_payload(payload: Any) -> np.ndarray:
    if isinstance(payload, np.ndarray):
        return np.asarray(payload, dtype=complex).reshape(-1)
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("serialized complex vector must be a sequence")
    values: list[complex] = []
    for item in payload:
        if isinstance(item, Mapping):
            values.append(complex(float(item.get("re", 0.0) or 0.0), float(item.get("im", 0.0) or 0.0)))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            values.append(complex(float(item[0]), float(item[1])))
        else:
            values.append(complex(item))
    return np.asarray(values, dtype=complex).reshape(-1)


def _qiskit_energy_from_serialized_terms(
    *,
    state: Any,
    terms: Sequence[Mapping[str, Any]],
    components: Any,
) -> float | None:
    items: list[tuple[str, complex]] = []
    for raw in terms:
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("pauli_exyz", "")).strip().lower()
        if not label:
            continue
        coeff = complex(float(raw.get("coeff_re", 0.0) or 0.0), float(raw.get("coeff_im", 0.0) or 0.0))
        if abs(coeff) <= 1.0e-15:
            continue
        items.append((to_ixyz_label(label), coeff))
    if not items:
        return None
    op = components.SparsePauliOp.from_list(items).simplify(atol=1.0e-12)
    sv = components.Statevector(_normalize_state(state))
    return float(np.real(sv.expectation_value(op)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"re": _json_safe(float(value.real)), "im": _json_safe(float(value.imag))}
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


def _normalize_state(state: Any) -> np.ndarray:
    arr = np.asarray(state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("state vector has zero norm")
    return arr / norm


def _num_qubits_from_state(state: Any) -> int:
    size = int(np.asarray(state, dtype=complex).reshape(-1).size)
    if size <= 0 or size & (size - 1):
        raise ValueError(f"statevector length {size} is not a positive power of two")
    return int(size.bit_length() - 1)


def _check_qubit_cap(
    *,
    config: QiskitDynamicsConfig,
    algorithm_id: str,
    family: str,
    case_id: str,
    num_qubits: int,
) -> dict[str, Any] | None:
    if config.qubit_cap is not None and int(num_qubits) > int(config.qubit_cap):
        return skipped_resource_guard_result(
            config=config,
            algorithm_id=algorithm_id,
            family=family,
            case_id=case_id,
            num_qubits=int(num_qubits),
        )
    return None


def _term_label(term: Any) -> str:
    if hasattr(term, "pauli_exyz"):
        return str(getattr(term, "pauli_exyz")).lower()
    if hasattr(term, "pw2strng"):
        return str(term.pw2strng()).lower()
    raise ValueError(f"cannot extract Pauli label from term {term!r}")


def _term_coeff_real(term: Any) -> float:
    if hasattr(term, "coeff_real"):
        return float(getattr(term, "coeff_real"))
    if hasattr(term, "p_coeff"):
        coeff = complex(getattr(term, "p_coeff"))
        if abs(coeff.imag) > 1.0e-12:
            raise ValueError(f"imaginary Pauli coefficient is not supported by parity adapter: {coeff}")
        return float(coeff.real)
    raise ValueError(f"cannot extract real coefficient from term {term!r}")


def _product_formula_sequence(terms: Sequence[Any], *, order: int) -> tuple[tuple[Any, float], ...]:
    if int(order) == 1:
        return tuple((term, 1.0) for term in terms)
    if int(order) == 2:
        items = tuple(terms)
        return tuple((term, 0.5) for term in items) + tuple((term, 0.5) for term in reversed(items))
    raise ValueError(f"unsupported product formula order {order!r}; expected 1 or 2")


def _evolve_state_with_circuit(state: np.ndarray, circuit: Any, Statevector: Any) -> np.ndarray:
    evolved = Statevector(_normalize_state(state)).evolve(circuit)
    return _normalize_state(np.asarray(getattr(evolved, "data", evolved), dtype=complex).reshape(-1))


def product_formula_step_state(
    *,
    state: np.ndarray,
    terms: Sequence[Any],
    dt: float,
    order: int,
    QuantumCircuit: Any | None = None,
    Statevector: Any | None = None,
) -> np.ndarray:
    """Return Qiskit statevector for one native product-formula interval step."""

    components = None
    if QuantumCircuit is None or Statevector is None:
        components = import_qiskit_pauli_components()
    QuantumCircuit = QuantumCircuit or components.QuantumCircuit
    Statevector = Statevector or components.Statevector
    num_qubits = _num_qubits_from_state(state)
    circuit = QuantumCircuit(int(num_qubits))
    for term, factor in _product_formula_sequence(terms, order=int(order)):
        angle = 2.0 * _term_coeff_real(term) * float(dt) * float(factor)
        append_pauli_rotation_exyz(circuit, label_exyz=_term_label(term), angle=angle)
    return _evolve_state_with_circuit(np.asarray(state, dtype=complex), circuit, Statevector)


def product_formula_state_trajectory(
    *,
    flow: Any,
    initial_state: np.ndarray,
    times: Sequence[float],
    order: int,
) -> tuple[np.ndarray, ...]:
    components = import_qiskit_pauli_components()
    psi = _normalize_state(initial_state)
    states: list[np.ndarray] = [np.asarray(psi, dtype=complex)]
    for left, right in zip(times[:-1], times[1:]):
        terms = flow.terms_for_interval(float(left), float(right))
        psi = product_formula_step_state(
            state=psi,
            terms=terms,
            dt=float(right) - float(left),
            order=int(order),
            QuantumCircuit=components.QuantumCircuit,
            Statevector=components.Statevector,
        )
        states.append(np.asarray(psi, dtype=complex))
    return tuple(states)


def qdrift_state_trajectory(
    *,
    initial_state: np.ndarray,
    intervals: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, ...]:
    components = import_qiskit_pauli_components()
    psi = _normalize_state(initial_state)
    states: list[np.ndarray] = [np.asarray(psi, dtype=complex)]
    num_qubits = _num_qubits_from_state(psi)
    for interval in intervals:
        labels = list(interval.get("sampled_labels", []) or [])
        signs = list(interval.get("sampled_signs", []) or [])
        tau = float(interval.get("tau", 0.0) or 0.0)
        if len(labels) != len(signs):
            raise ValueError("qDRIFT sample plan has mismatched sampled_labels/sampled_signs lengths")
        for label, sign in zip(labels, signs):
            circuit = components.QuantumCircuit(int(num_qubits))
            append_pauli_rotation_exyz(
                circuit,
                label_exyz=str(label),
                angle=2.0 * float(sign) * float(tau),
            )
            psi = _evolve_state_with_circuit(psi, circuit, components.Statevector)
        states.append(np.asarray(psi, dtype=complex))
    return tuple(states)


def state_fidelity(left: Any, right: Any) -> float:
    lhs = _normalize_state(left)
    rhs = _normalize_state(right)
    return float(min(1.0, max(0.0, abs(np.vdot(lhs, rhs)) ** 2)))


def phase_aligned_l2(reference: Any, candidate: Any) -> float:
    ref = _normalize_state(reference)
    cand = _normalize_state(candidate)
    overlap = complex(np.vdot(ref, cand))
    if abs(overlap) > 0.0:
        cand = cand * np.conjugate(overlap / abs(overlap))
    return float(np.linalg.norm(ref - cand))


def _energy_from_matrix(state: np.ndarray, hmat: np.ndarray) -> float:
    psi = _normalize_state(state)
    mat = np.asarray(hmat, dtype=complex)
    return float(np.real(np.vdot(psi, mat @ psi)))


def compare_state_sequences(
    *,
    native_states: Sequence[Any],
    qiskit_states: Sequence[Any],
    hmat_sequence: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    if len(native_states) != len(qiskit_states):
        raise ValueError(
            f"state trajectory length mismatch: native={len(native_states)}, qiskit={len(qiskit_states)}"
        )
    comparisons: list[dict[str, Any]] = []
    for idx, (native, qiskit) in enumerate(zip(native_states, qiskit_states)):
        fidelity = state_fidelity(native, qiskit)
        row: dict[str, Any] = {
            "index": int(idx),
            "phase_aligned_l2": phase_aligned_l2(native, qiskit),
            "fidelity": fidelity,
            "infidelity": float(max(0.0, 1.0 - fidelity)),
        }
        if hmat_sequence is not None and idx < len(hmat_sequence):
            nrg_native = _energy_from_matrix(_normalize_state(native), np.asarray(hmat_sequence[idx], dtype=complex))
            nrg_qiskit = _energy_from_matrix(_normalize_state(qiskit), np.asarray(hmat_sequence[idx], dtype=complex))
            row["native_energy"] = nrg_native
            row["qiskit_energy"] = nrg_qiskit
            row["energy_abs_delta"] = abs(nrg_native - nrg_qiskit)
        comparisons.append(row)
    return _json_safe(
        {
            "comparisons": comparisons,
            "max_state_l2": max((float(row["phase_aligned_l2"]) for row in comparisons), default=None),
            "max_infidelity": max((float(row["infidelity"]) for row in comparisons), default=None),
            "max_energy_abs_delta": max(
                (float(row.get("energy_abs_delta", 0.0)) for row in comparisons if row.get("energy_abs_delta") is not None),
                default=None,
            ),
        }
    )


def _passes_tolerances(config: QiskitDynamicsConfig, comparison: Mapping[str, Any]) -> bool:
    max_l2 = comparison.get("max_state_l2")
    max_inf = comparison.get("max_infidelity")
    max_energy = comparison.get("max_energy_abs_delta")
    return (
        (max_l2 is None or float(max_l2) <= float(config.state_l2_tol))
        and (max_inf is None or float(max_inf) <= float(config.infidelity_tol))
        and (max_energy is None or float(max_energy) <= float(config.energy_abs_tol))
    )


def product_formula_parity_result(
    *,
    config: QiskitDynamicsConfig,
    case: Any,
    flow: Any,
    initial_state: np.ndarray,
    times: Sequence[float],
    order: int,
    native_states: Sequence[Any],
) -> dict[str, Any] | None:
    if not parity_requested(config):
        return None
    algorithm_id = "dyn_product_formula_envelope"
    family = str(getattr(case, "family", ""))
    case_id = str(getattr(case, "case_id", ""))
    num_qubits = _num_qubits_from_state(initial_state)
    guard = _check_qubit_cap(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        num_qubits=num_qubits,
    )
    if guard is not None:
        return guard
    try:
        qiskit_states = product_formula_state_trajectory(
            flow=flow,
            initial_state=initial_state,
            times=times,
            order=int(order),
        )
    except QiskitPauliUnavailable as exc:
        return skipped_optional_dependency_result(config=config, algorithm_id=algorithm_id, family=family, case_id=case_id, exc=exc)
    except Exception as exc:
        return failed_result(config=config, algorithm_id=algorithm_id, family=family, case_id=case_id, exc=exc)
    comparison = compare_state_sequences(
        native_states=native_states,
        qiskit_states=qiskit_states,
        hmat_sequence=flow.hmat_sequence_for_times(),
    )
    passed = _passes_tolerances(config, comparison)
    return _base_result(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        status="ok",
        passed=bool(passed),
        support_scope="product_formula_sequence_parity_only",
        extra={
            "qiskit_available": True,
            "num_qubits": int(num_qubits),
            "selected_order": int(order),
            "sequence_source": "native_product_formula_sequence_and_terms_for_interval",
            "max_state_l2": comparison.get("max_state_l2"),
            "max_infidelity": comparison.get("max_infidelity"),
            "max_energy_abs_delta": comparison.get("max_energy_abs_delta"),
            "state_comparisons": comparison.get("comparisons", []),
            "resources": {
                "statevector_qubits": int(num_qubits),
                "state_snapshot_count": int(len(qiskit_states)),
            },
        },
    )


def qdrift_parity_result(
    *,
    config: QiskitDynamicsConfig,
    case: Any,
    initial_state: np.ndarray,
    intervals: Sequence[Mapping[str, Any]],
    native_states: Sequence[Any],
    hmat_sequence: Sequence[np.ndarray] | None = None,
) -> dict[str, Any] | None:
    if not parity_requested(config):
        return None
    algorithm_id = "dyn_qdrift"
    family = str(getattr(case, "family", ""))
    case_id = str(getattr(case, "case_id", ""))
    num_qubits = _num_qubits_from_state(initial_state)
    guard = _check_qubit_cap(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        num_qubits=num_qubits,
    )
    if guard is not None:
        return guard
    try:
        qiskit_states = qdrift_state_trajectory(initial_state=initial_state, intervals=intervals)
    except QiskitPauliUnavailable as exc:
        return skipped_optional_dependency_result(config=config, algorithm_id=algorithm_id, family=family, case_id=case_id, exc=exc)
    except Exception as exc:
        return failed_result(config=config, algorithm_id=algorithm_id, family=family, case_id=case_id, exc=exc)
    comparison = compare_state_sequences(
        native_states=native_states,
        qiskit_states=qiskit_states,
        hmat_sequence=hmat_sequence,
    )
    passed = _passes_tolerances(config, comparison)
    return _base_result(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        status="ok",
        passed=bool(passed),
        support_scope="qdrift_realized_sample_plan_parity_only",
        extra={
            "qiskit_available": True,
            "num_qubits": int(num_qubits),
            "sample_plan_source": "native_qdrift_intervals",
            "sample_plan_matches": True,
            "sampled_rotation_count": int(sum(len(row.get("sampled_labels", []) or []) for row in intervals)),
            "max_state_l2": comparison.get("max_state_l2"),
            "max_infidelity": comparison.get("max_infidelity"),
            "max_energy_abs_delta": comparison.get("max_energy_abs_delta"),
            "state_comparisons": comparison.get("comparisons", []),
            "resources": {
                "statevector_qubits": int(num_qubits),
                "state_snapshot_count": int(len(qiskit_states)),
            },
        },
    )


def statevector_from_runtime_layout(
    *,
    layout: Any,
    theta_runtime: np.ndarray | Sequence[float],
    psi_ref: np.ndarray | Sequence[complex],
) -> np.ndarray:
    components = import_qiskit_pauli_components()
    num_qubits = _num_qubits_from_state(psi_ref)
    circuit = build_runtime_layout_circuit(
        layout,
        theta_runtime,
        num_qubits,
        reference_state=np.asarray(psi_ref, dtype=complex).reshape(-1),
        quantum_circuit_cls=components.QuantumCircuit,
    )
    zero = np.zeros(1 << num_qubits, dtype=complex)
    zero[0] = 1.0
    return _evolve_state_with_circuit(zero, circuit, components.Statevector)


def fixed_mclachlan_post_run_parity_result(
    *,
    config: QiskitDynamicsConfig,
    case: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate fixed-McLachlan prepared checkpoint states with Qiskit circuits.

    The input payload is produced after repo-native checkpoint telemetry; it is
    benchmark-local and never participates in online controller decisions.
    """

    if not parity_requested(config):
        return None
    algorithm_id = "dyn_fixed_mclachlan"
    family = str(getattr(case, "family", ""))
    case_id = str(getattr(case, "case_id", ""))
    sidecar = payload.get("fixed_scaffold_parity_payload") if isinstance(payload, Mapping) else None
    if not isinstance(sidecar, Mapping):
        return not_applicable_result(
            config=config,
            algorithm_id=algorithm_id,
            family=family,
            case_id=case_id,
            reason="missing_serialized_fixed_scaffold_layout_or_theta_payload",
            support_scope="post_run_fixed_scaffold_parity_only_no_controller_decisions",
        )
    checkpoints = sidecar.get("checkpoints", [])
    if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, (str, bytes)) or not checkpoints:
        return not_applicable_result(
            config=config,
            algorithm_id=algorithm_id,
            family=family,
            case_id=case_id,
            reason="fixed_scaffold_parity_payload_has_no_checkpoints",
            support_scope="post_run_fixed_scaffold_parity_only_no_controller_decisions",
        )
    fixed_layout_payload = sidecar.get("fixed_layout")
    if not isinstance(fixed_layout_payload, Mapping):
        return not_applicable_result(
            config=config,
            algorithm_id=algorithm_id,
            family=family,
            case_id=case_id,
            reason="fixed_scaffold_parity_payload_missing_fixed_layout",
            support_scope="post_run_fixed_scaffold_parity_only_no_controller_decisions",
        )
    try:
        psi_ref = _complex_vector_from_payload(sidecar.get("psi_ref", []))
        num_qubits = _num_qubits_from_state(psi_ref)
    except Exception as exc:
        return failed_result(
            config=config,
            algorithm_id=algorithm_id,
            family=family,
            case_id=case_id,
            exc=f"invalid fixed scaffold parity reference state: {exc}",
            support_scope="post_run_fixed_scaffold_parity_only_no_controller_decisions",
        )
    guard = _check_qubit_cap(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        num_qubits=num_qubits,
    )
    if guard is not None:
        return guard

    try:
        components = import_qiskit_pauli_components()
        comparisons: list[dict[str, Any]] = []
        circuits: list[dict[str, Any]] = []
        for idx, raw_checkpoint in enumerate(checkpoints):
            if not isinstance(raw_checkpoint, Mapping):
                raise ValueError(f"checkpoint {idx} is not a mapping")
            layout_payload = raw_checkpoint.get("layout", fixed_layout_payload)
            if not isinstance(layout_payload, Mapping):
                raise ValueError(f"checkpoint {idx} missing serialized layout")
            theta_runtime = np.asarray(raw_checkpoint.get("theta_runtime", []), dtype=float).reshape(-1)
            native_state = _complex_vector_from_payload(raw_checkpoint.get("native_state", []))
            layout = deserialize_layout(layout_payload)
            qiskit_state = statevector_from_runtime_layout(
                layout=layout,
                theta_runtime=theta_runtime,
                psi_ref=psi_ref,
            )
            fidelity = state_fidelity(native_state, qiskit_state)
            qiskit_energy = None
            energy_abs_delta = None
            terms = raw_checkpoint.get("hamiltonian_terms_exyz", [])
            controller_energy = raw_checkpoint.get("energy_total_controller")
            if isinstance(terms, Sequence) and not isinstance(terms, (str, bytes)) and controller_energy is not None:
                qiskit_energy = _qiskit_energy_from_serialized_terms(
                    state=qiskit_state,
                    terms=terms,
                    components=components,
                )
                if qiskit_energy is not None:
                    energy_abs_delta = abs(float(qiskit_energy) - float(controller_energy))
            row = {
                "index": int(idx),
                "checkpoint_index": int(raw_checkpoint.get("checkpoint_index", idx)),
                "time": raw_checkpoint.get("time"),
                "phase_aligned_l2": phase_aligned_l2(native_state, qiskit_state),
                "fidelity": fidelity,
                "infidelity": float(max(0.0, 1.0 - fidelity)),
                "native_energy": controller_energy,
                "qiskit_energy": qiskit_energy,
                "energy_abs_delta": energy_abs_delta,
                "observable_source": "controller_prepared_state_and_step_hamiltonian_terms",
            }
            comparisons.append(row)
            try:
                circuit = build_runtime_layout_circuit(
                    layout,
                    theta_runtime,
                    int(num_qubits),
                    reference_state=psi_ref,
                    quantum_circuit_cls=components.QuantumCircuit,
                )
                circuits.append({"checkpoint_index": row["checkpoint_index"], **circuit_stats(circuit)})
            except Exception:
                pass
    except QiskitPauliUnavailable as exc:
        return skipped_optional_dependency_result(
            config=config,
            algorithm_id=algorithm_id,
            family=family,
            case_id=case_id,
            exc=exc,
        )
    except Exception as exc:
        return failed_result(
            config=config,
            algorithm_id=algorithm_id,
            family=family,
            case_id=case_id,
            exc=exc,
            support_scope="post_run_fixed_scaffold_parity_only_no_controller_decisions",
        )

    max_l2 = max((float(row["phase_aligned_l2"]) for row in comparisons), default=None)
    max_infidelity = max((float(row["infidelity"]) for row in comparisons), default=None)
    max_energy = max(
        (
            float(row["energy_abs_delta"])
            for row in comparisons
            if row.get("energy_abs_delta") is not None
        ),
        default=None,
    )
    comparison_summary = {
        "max_state_l2": max_l2,
        "max_infidelity": max_infidelity,
        "max_energy_abs_delta": max_energy,
    }
    passed = _passes_tolerances(config, comparison_summary)
    return _base_result(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        status="ok",
        passed=bool(passed),
        support_scope="fixed_mclachlan_post_run_scaffold_state_and_observable_parity",
        extra={
            "qiskit_available": True,
            "num_qubits": int(num_qubits),
            "parity_payload_schema": sidecar.get("schema"),
            "checkpoint_count": int(len(comparisons)),
            "layout_stable": bool(sidecar.get("layout_stable", True)),
            "qiskit_used_in_online_controller": False,
            "controller_decisions_modified": False,
            "exact_reference_controller_inputs": False,
            "max_state_l2": max_l2,
            "max_infidelity": max_infidelity,
            "max_energy_abs_delta": max_energy,
            "state_comparisons": comparisons,
            "circuit_stats": circuits,
            "resources": {
                "statevector_qubits": int(num_qubits),
                "state_snapshot_count": int(len(comparisons)),
            },
        },
    )



def _projection_loss_and_overlap(state: Any, target: Any) -> tuple[float, float]:
    fidelity = state_fidelity(state, target)
    return float(max(0.0, 1.0 - fidelity)), float(fidelity)


def pvqd_component_parity_result(
    *,
    config: QiskitDynamicsConfig,
    case: Any,
    algorithm_id: str,
    component_inputs: Sequence[Mapping[str, Any]],
    support_scope: str,
) -> dict[str, Any] | None:
    if not parity_requested(config):
        return None
    family = str(getattr(case, "family", ""))
    case_id = str(getattr(case, "case_id", ""))
    if not component_inputs:
        return not_applicable_result(
            config=config,
            algorithm_id=algorithm_id,
            family=family,
            case_id=case_id,
            reason="no_pvqd_component_payloads_available",
            support_scope=support_scope,
        )
    num_qubits = _num_qubits_from_state(component_inputs[0]["psi_ref"])
    guard = _check_qubit_cap(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        num_qubits=num_qubits,
    )
    if guard is not None:
        return guard
    try:
        import_qiskit_pauli_components()
        comparisons: list[dict[str, Any]] = []
        circuits: list[dict[str, Any]] = []
        for item in component_inputs:
            interval_index = int(item.get("interval_index", len(comparisons)))
            psi_ref = np.asarray(item["psi_ref"], dtype=complex).reshape(-1)
            q_start = statevector_from_runtime_layout(
                layout=item["start_layout"],
                theta_runtime=item["theta_start"],
                psi_ref=psi_ref,
            )
            q_target = product_formula_step_state(
                state=q_start,
                terms=item["target_terms"],
                dt=float(item["dt"]),
                order=int(item["target_order"]),
            )
            q_final = statevector_from_runtime_layout(
                layout=item["final_layout"],
                theta_runtime=item["theta_final"],
                psi_ref=psi_ref,
            )
            native_start = np.asarray(item["native_start_state"], dtype=complex).reshape(-1)
            native_target = np.asarray(item["native_target_state"], dtype=complex).reshape(-1)
            native_final = np.asarray(item["native_final_state"], dtype=complex).reshape(-1)
            fit = item.get("fit", {}) if isinstance(item.get("fit", {}), Mapping) else {}
            q_initial_loss, q_initial_overlap = _projection_loss_and_overlap(q_start, q_target)
            q_final_loss, q_final_overlap = _projection_loss_and_overlap(q_final, q_target)
            comparisons.append(
                {
                    "interval_index": int(interval_index),
                    "start_state_l2": phase_aligned_l2(native_start, q_start),
                    "target_state_l2": phase_aligned_l2(native_target, q_target),
                    "final_state_l2": phase_aligned_l2(native_final, q_final),
                    "start_infidelity": max(0.0, 1.0 - state_fidelity(native_start, q_start)),
                    "target_infidelity": max(0.0, 1.0 - state_fidelity(native_target, q_target)),
                    "final_infidelity": max(0.0, 1.0 - state_fidelity(native_final, q_final)),
                    "qiskit_initial_projection_loss": q_initial_loss,
                    "qiskit_final_projection_loss": q_final_loss,
                    "qiskit_initial_overlap": q_initial_overlap,
                    "qiskit_final_overlap": q_final_overlap,
                    "native_initial_projection_loss": fit.get("initial_projection_loss"),
                    "native_final_projection_loss": fit.get("final_projection_loss"),
                    "initial_projection_loss_abs_delta": None
                    if fit.get("initial_projection_loss") is None
                    else abs(float(fit.get("initial_projection_loss")) - q_initial_loss),
                    "final_projection_loss_abs_delta": None
                    if fit.get("final_projection_loss") is None
                    else abs(float(fit.get("final_projection_loss")) - q_final_loss),
                    "target_policy": "product_formula_circuit_step",
                }
            )
            try:
                circuit = build_runtime_layout_circuit(
                    item["final_layout"],
                    item["theta_final"],
                    int(num_qubits),
                    reference_state=psi_ref,
                )
                circuits.append({"interval_index": interval_index, **circuit_stats(circuit)})
            except Exception:
                pass
    except QiskitPauliUnavailable as exc:
        return skipped_optional_dependency_result(config=config, algorithm_id=algorithm_id, family=family, case_id=case_id, exc=exc)
    except Exception as exc:
        return failed_result(
            config=config,
            algorithm_id=algorithm_id,
            family=family,
            case_id=case_id,
            exc=exc,
            support_scope=support_scope,
        )

    state_l2_values = [
        float(row[key])
        for row in comparisons
        for key in ("start_state_l2", "target_state_l2", "final_state_l2")
        if row.get(key) is not None
    ]
    infidelity_values = [
        float(row[key])
        for row in comparisons
        for key in ("start_infidelity", "target_infidelity", "final_infidelity")
        if row.get(key) is not None
    ]
    loss_delta_values = [
        float(row[key])
        for row in comparisons
        for key in ("initial_projection_loss_abs_delta", "final_projection_loss_abs_delta")
        if row.get(key) is not None
    ]
    max_l2 = max(state_l2_values, default=None)
    max_infidelity = max(infidelity_values, default=None)
    max_loss_delta = max(loss_delta_values, default=None)
    passed = (
        (max_l2 is None or max_l2 <= float(config.state_l2_tol))
        and (max_infidelity is None or max_infidelity <= float(config.infidelity_tol))
        and (max_loss_delta is None or max_loss_delta <= max(float(config.state_l2_tol), 1.0e-8))
    )
    return _base_result(
        config=config,
        algorithm_id=algorithm_id,
        family=family,
        case_id=case_id,
        status="ok",
        passed=bool(passed),
        support_scope=support_scope,
        extra={
            "qiskit_available": True,
            "num_qubits": int(num_qubits),
            "component_count": int(len(comparisons)),
            "max_state_l2": max_l2,
            "max_infidelity": max_infidelity,
            "max_projection_loss_abs_delta": max_loss_delta,
            "max_energy_abs_delta": None,
            "component_comparisons": comparisons,
            "circuit_stats": circuits,
            "resources": {
                "statevector_qubits": int(num_qubits),
                "component_count": int(len(comparisons)),
            },
        },
    )


def sparse_pauli_label_for_exyz(label_exyz: str) -> str:
    """Small public shim used by tests for convention auditing."""

    return to_ixyz_label(label_exyz)


__all__ = [
    "MAPPING_CONVENTION",
    "QISKIT_DYNAMICS_MODE_VALUES",
    "QISKIT_DYNAMICS_PARITY_SCHEMA",
    "QiskitDynamicsConfig",
    "QiskitDynamicsParityUnavailable",
    "compare_state_sequences",
    "failed_result",
    "fixed_mclachlan_not_applicable_result",
    "fixed_mclachlan_post_run_parity_result",
    "not_applicable_result",
    "parity_requested",
    "phase_aligned_l2",
    "product_formula_parity_result",
    "product_formula_state_trajectory",
    "product_formula_step_state",
    "pvqd_component_parity_result",
    "qdrift_parity_result",
    "qdrift_state_trajectory",
    "qiskit_dynamics_config_from_case",
    "qiskit_dynamics_config_from_metadata",
    "skipped_optional_dependency_result",
    "skipped_resource_guard_result",
    "sparse_pauli_label_for_exyz",
    "state_fidelity",
    "statevector_from_runtime_layout",
]
