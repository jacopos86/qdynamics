"""Compact offline QSE-root refit with honest static-scaffold composition.

The legacy refitter expands every QSE parent into every Pauli child before it
knows which directions help prepare the selected root.  This module keeps the
full QSE Ritz vector as the target but greedily selects individual Pauli
rotations by their exact one-angle fidelity gain, globally refitting the small
selected circuit after each append.  The fitted suffix is then composed after
the complete source scaffold, so the emitted circuit starts from the source
reference state rather than an injected prepared-state vector.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import QSEBasisElement, normalize_statevector
from pipelines.qse_spectra.io import load_polynomial_json, write_manifest_json
from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
from pipelines.scaffold.qse_root_refit import (
    HamiltonianResolution,
    PauliRotationRefitResult,
    PreparedStateResolution,
    QSERootRefitConfig,
    QSERootRefitError,
    build_qse_root_refit_artifact,
    reconstruct_qse_root_target,
)
from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    iter_runtime_rotation_terms,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import compile_polynomial_action, energy_via_one_apply
from src.quantum.pauli_actions import apply_compiled_pauli, compile_pauli_action_exyz
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


COMPACT_REFIT_SCHEMA_V1 = "qse_compact_greedy_pauli_refit_v1"
BASE_COMPOSITION_SCHEMA_V1 = "qse_base_scaffold_excitation_composition_v1"


@dataclass(frozen=True)
class CompactQSERootRefitConfig:
    qse_result_json: Path
    state_index: int
    output_json: Path
    base_scaffold_json: Path
    hamiltonian_json: Path | None = None
    allow_ground_state: bool = False
    max_selected_paulis: int = 30
    target_infidelity: float = 1.0e-8
    max_energy_error: float = 1.0e-6
    max_physical_residual: float = 1.0e-3
    optimizer_maxiter: int = 2000
    amplitude_cutoff: float = 1.0e-12


@dataclass(frozen=True)
class CompactFitDiagnostics:
    selected_labels: tuple[str, ...]
    depth_history: tuple[Mapping[str, Any], ...]
    candidate_count: int
    physical_residual_norm: float
    energy_error_vs_qse: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    import json

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QSERootRefitError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise QSERootRefitError(f"Expected a JSON mapping at {path}")
    return payload


def _validate_config(config: CompactQSERootRefitConfig) -> None:
    for path in (config.qse_result_json, config.base_scaffold_json):
        if not Path(path).is_file():
            raise QSERootRefitError(f"Required compact-refit input not found: {path}")
    if config.hamiltonian_json is not None and not Path(config.hamiltonian_json).is_file():
        raise QSERootRefitError(f"Hamiltonian JSON not found: {config.hamiltonian_json}")
    if int(config.state_index) < 0:
        raise QSERootRefitError("state_index must be non-negative")
    if int(config.max_selected_paulis) < 1:
        raise QSERootRefitError("max_selected_paulis must be positive")
    if int(config.optimizer_maxiter) < 1:
        raise QSERootRefitError("optimizer_maxiter must be positive")
    for name in ("target_infidelity", "max_energy_error", "max_physical_residual", "amplitude_cutoff"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value < 0.0:
            raise QSERootRefitError(f"{name} must be finite and non-negative")


def _normalize(state: np.ndarray, *, name: str) -> np.ndarray:
    vector, _norm, _nq = normalize_statevector(np.asarray(state, dtype=complex).reshape(-1))
    if not np.all(np.isfinite(vector)):
        raise QSERootRefitError(f"{name} contains non-finite amplitudes")
    return np.asarray(vector, dtype=complex)


def _fidelity(target: np.ndarray, state: np.ndarray) -> float:
    left = _normalize(target, name="target state")
    right = _normalize(state, name="fitted state")
    return float(max(0.0, min(1.0, abs(np.vdot(left, right)) ** 2)))


def _single_pauli_term(*, label: str, nq: int, ordinal: int) -> AnsatzTerm:
    return AnsatzTerm(
        label=f"qse_compact_excitation[{int(ordinal)}]::{label}",
        polynomial=PauliPolynomial("JW", [PauliTerm(int(nq), ps=str(label), pc=1.0)]),
        execution_mode="termwise_product",
    )


def unique_qse_pauli_candidates(
    basis: Sequence[QSEBasisElement],
    *,
    nq: int,
) -> tuple[str, ...]:
    """Return unique nonidentity QSE Pauli children in source order."""

    labels: list[str] = []
    seen: set[str] = set()
    for element in basis:
        if str(element.kind) == "pauli_string":
            raw_labels = (str(element.pauli_label_exyz),)
        elif element.polynomial is not None:
            raw_labels = tuple(
                str(spec.pauli_exyz)
                for spec in iter_runtime_rotation_terms(
                    element.polynomial,
                    ignore_identity=True,
                    coefficient_tolerance=1.0e-12,
                    sort_terms=True,
                )
            )
        else:
            continue
        for label in raw_labels:
            if len(label) != int(nq):
                raise QSERootRefitError(
                    f"QSE candidate label {label!r} has length {len(label)}, expected nq={nq}"
                )
            if set(label) <= {"e"} or label in seen:
                continue
            seen.add(label)
            labels.append(label)
    if not labels:
        raise QSERootRefitError("QSE basis contains no unique nonidentity Pauli candidates")
    return tuple(labels)


def _best_one_angle(
    *,
    target: np.ndarray,
    current: np.ndarray,
    pauli_state: np.ndarray,
) -> tuple[float, float]:
    """Maximize |<t|(cos(theta)I-i sin(theta)P)|psi>|^2 analytically."""

    overlap_identity = complex(np.vdot(target, current))
    overlap_pauli = complex(np.vdot(target, -1.0j * pauli_state))
    quadratic = np.asarray(
        [
            [abs(overlap_identity) ** 2, float(np.real(np.conj(overlap_identity) * overlap_pauli))],
            [float(np.real(np.conj(overlap_identity) * overlap_pauli)), abs(overlap_pauli) ** 2],
        ],
        dtype=float,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (quadratic + quadratic.T))
    vector = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
    angle = float(math.atan2(float(vector[1]), float(vector[0])))
    return float(max(0.0, min(1.0, float(np.max(eigenvalues))))), angle


def _optimize_selected(
    *,
    target: np.ndarray,
    reference: np.ndarray,
    terms: Sequence[AnsatzTerm],
    theta_initial: np.ndarray,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    layout = build_parameter_layout(
        terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    target_state = _normalize(target, name="compact target")
    reference_state = _normalize(reference, name="compact reference")
    x0 = np.asarray(theta_initial, dtype=float).reshape(-1)
    if int(x0.size) != int(layout.runtime_parameter_count):
        raise QSERootRefitError(
            f"Compact optimizer theta length {x0.size} != {layout.runtime_parameter_count}"
        )
    evaluations = 0

    def objective_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        psi, tangents = executor.prepare_state_with_parameter_tangents(
            np.asarray(theta, dtype=float),
            reference_state,
        )
        overlap = complex(np.vdot(target_state, psi))
        objective = float(max(0.0, 1.0 - abs(overlap) ** 2))
        gradient = np.asarray(
            [
                -2.0 * float(np.real(np.conj(overlap) * np.vdot(target_state, tangents[index])))
                for index in range(int(layout.runtime_parameter_count))
            ],
            dtype=float,
        )
        return objective, gradient

    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - SciPy is required by the scientific environment.
        raise QSERootRefitError("Compact QSE refit requires scipy.optimize") from exc
    result = minimize(
        objective_and_gradient,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(maxiter),
            "ftol": 1.0e-15,
            "gtol": 1.0e-12,
            "maxls": 50,
            "maxcor": 20,
        },
    )
    theta = np.asarray(result.x, dtype=float).reshape(-1)
    fitted = np.asarray(executor.prepare_state(theta, reference_state), dtype=complex).reshape(-1)
    return theta, fitted, {
        "method": "deterministic_greedy_one_angle_then_global_lbfgsb_exact_gradient",
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "objective_evaluations": int(evaluations),
        "objective": float(1.0 - _fidelity(target_state, fitted)),
    }


def _state_metrics(
    *,
    state: np.ndarray,
    target: np.ndarray,
    compiled_hamiltonian: Any,
    qse_energy: float,
) -> dict[str, float]:
    fitted = _normalize(state, name="compact fitted state")
    energy, hpsi = energy_via_one_apply(fitted, compiled_hamiltonian)
    residual = float(np.linalg.norm(hpsi - float(energy) * fitted))
    fidelity = _fidelity(target, fitted)
    return {
        "fidelity": float(fidelity),
        "infidelity": float(max(0.0, 1.0 - fidelity)),
        "energy": float(energy),
        "energy_error_vs_qse": float(abs(float(energy) - float(qse_energy))),
        "physical_residual_norm": residual,
    }


def fit_compact_greedy_pauli_ansatz(
    *,
    target_state: np.ndarray,
    prepared_state: np.ndarray,
    basis: Sequence[QSEBasisElement],
    nq: int,
    hamiltonian: Any,
    qse_energy: float,
    max_selected_paulis: int,
    target_infidelity: float,
    max_energy_error: float,
    max_physical_residual: float,
    optimizer_maxiter: int,
) -> tuple[PauliRotationRefitResult, CompactFitDiagnostics]:
    """Greedily build and globally refit a compact single-Pauli suffix."""

    target = _normalize(target_state, name="QSE target")
    reference = _normalize(prepared_state, name="QSE prepared reference")
    candidate_labels = unique_qse_pauli_candidates(basis, nq=int(nq))
    compiled_actions = {
        label: compile_pauli_action_exyz(label, int(nq)) for label in candidate_labels
    }
    compiled_h = compile_polynomial_action(hamiltonian)
    remaining = list(candidate_labels)
    selected_labels: list[str] = []
    selected_terms: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    current = reference.copy()
    history: list[Mapping[str, Any]] = []
    last_optimizer: dict[str, Any] = {}
    accepted = False

    for depth in range(1, int(max_selected_paulis) + 1):
        best_label: str | None = None
        best_angle = 0.0
        best_fidelity = -1.0
        for label in remaining:
            pauli_state = apply_compiled_pauli(current, compiled_actions[label])
            candidate_fidelity, angle = _best_one_angle(
                target=target,
                current=current,
                pauli_state=pauli_state,
            )
            if candidate_fidelity > best_fidelity + 1.0e-15:
                best_label = label
                best_angle = float(angle)
                best_fidelity = float(candidate_fidelity)
        if best_label is None:
            raise QSERootRefitError("Compact greedy selector exhausted its candidate set")
        remaining.remove(best_label)
        selected_labels.append(best_label)
        selected_terms.append(
            _single_pauli_term(label=best_label, nq=int(nq), ordinal=depth - 1)
        )
        theta_initial = np.concatenate([theta, np.asarray([best_angle], dtype=float)])
        theta, current, last_optimizer = _optimize_selected(
            target=target,
            reference=reference,
            terms=selected_terms,
            theta_initial=theta_initial,
            maxiter=int(optimizer_maxiter),
        )
        metrics = _state_metrics(
            state=current,
            target=target,
            compiled_hamiltonian=compiled_h,
            qse_energy=float(qse_energy),
        )
        history.append(
            {
                "depth": int(depth),
                "selected_pauli_exyz": str(best_label),
                "greedy_one_angle_fidelity": float(best_fidelity),
                **metrics,
                "optimizer": dict(last_optimizer),
            }
        )
        accepted = bool(
            float(metrics["infidelity"]) <= float(target_infidelity)
            and float(metrics["energy_error_vs_qse"]) <= float(max_energy_error)
            and float(metrics["physical_residual_norm"]) <= float(max_physical_residual)
        )
        if accepted:
            break

    final_metrics = dict(history[-1])
    if not accepted:
        raise QSERootRefitError(
            "Compact QSE root refit did not meet thresholds by depth "
            f"{int(max_selected_paulis)}: infidelity={final_metrics['infidelity']:.6e} "
            f"(max {float(target_infidelity):.6e}), "
            f"energy_error={final_metrics['energy_error_vs_qse']:.6e} "
            f"(max {float(max_energy_error):.6e}), "
            f"physical_residual={final_metrics['physical_residual_norm']:.6e} "
            f"(max {float(max_physical_residual):.6e})."
        )
    layout = build_parameter_layout(
        selected_terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    fit = PauliRotationRefitResult(
        terms=tuple(selected_terms),
        layout=layout,
        theta_runtime=np.asarray(theta, dtype=float),
        theta_logical=np.asarray(theta, dtype=float),
        fitted_state=np.asarray(current, dtype=complex),
        fidelity=float(final_metrics["fidelity"]),
        infidelity=float(final_metrics["infidelity"]),
        optimizer_summary={
            **dict(last_optimizer),
            "selected_pauli_count": int(len(selected_labels)),
            "candidate_count": int(len(candidate_labels)),
            "acceptance_thresholds": {
                "target_infidelity": float(target_infidelity),
                "max_energy_error": float(max_energy_error),
                "max_physical_residual": float(max_physical_residual),
            },
        },
    )
    diagnostics = CompactFitDiagnostics(
        selected_labels=tuple(selected_labels),
        depth_history=tuple(dict(row) for row in history),
        candidate_count=int(len(candidate_labels)),
        physical_residual_norm=float(final_metrics["physical_residual_norm"]),
        energy_error_vs_qse=float(final_metrics["energy_error_vs_qse"]),
    )
    return fit, diagnostics


def compose_base_scaffold_and_excitation(
    *,
    runtime_input: Any,
    qse_prepared_state: np.ndarray,
    excitation_fit: PauliRotationRefitResult,
    parity_infidelity_tolerance: float = 1.0e-12,
) -> tuple[PauliRotationRefitResult, PreparedStateResolution, dict[str, Any]]:
    """Compose the complete source circuit before a fitted excitation suffix."""

    base_initial = _normalize(runtime_input.psi_initial, name="base scaffold prepared state")
    qse_prepared = _normalize(qse_prepared_state, name="QSE source prepared state")
    base_qse_fidelity = _fidelity(base_initial, qse_prepared)
    if 1.0 - float(base_qse_fidelity) > float(parity_infidelity_tolerance):
        raise QSERootRefitError(
            "Base scaffold prepared state does not match the QSE source state: "
            f"infidelity={1.0 - base_qse_fidelity:.6e}."
        )
    base_terms = tuple(runtime_input.selected_terms)
    excitation_terms = tuple(excitation_fit.terms)
    terms = base_terms + excitation_terms
    layout = build_parameter_layout(
        terms,
        ignore_identity=bool(runtime_input.base_layout.ignore_identity),
        coefficient_tolerance=float(runtime_input.base_layout.coefficient_tolerance),
        sort_terms=(str(runtime_input.base_layout.term_order).strip().lower() == "sorted"),
    )
    base_theta = np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1)
    excitation_theta = np.asarray(excitation_fit.theta_runtime, dtype=float).reshape(-1)
    theta_runtime = np.concatenate([base_theta, excitation_theta])
    base_theta_logical = getattr(runtime_input, "theta_logical", None)
    if base_theta_logical is None:
        if int(runtime_input.base_layout.logical_parameter_count) != int(base_theta.size):
            raise QSERootRefitError("Base scaffold lacks a valid logical theta alias for composition")
        base_theta_logical = base_theta.copy()
    theta_logical = np.concatenate(
        [
            np.asarray(base_theta_logical, dtype=float).reshape(-1),
            np.asarray(excitation_fit.theta_logical, dtype=float).reshape(-1),
        ]
    )
    executor = CompiledAnsatzExecutor(
        terms,
        ignore_identity=bool(layout.ignore_identity),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    psi_ref = _normalize(runtime_input.psi_ref, name="base scaffold reference state")
    replayed = _normalize(
        executor.prepare_state(theta_runtime, psi_ref),
        name="composed scaffold replay",
    )
    replay_fidelity = _fidelity(replayed, excitation_fit.fitted_state)
    if 1.0 - replay_fidelity > float(parity_infidelity_tolerance):
        raise QSERootRefitError(
            "HF -> base -> excitation replay does not reproduce the fitted root: "
            f"infidelity={1.0 - replay_fidelity:.6e}."
        )
    combined = replace(
        excitation_fit,
        terms=terms,
        layout=layout,
        theta_runtime=theta_runtime,
        theta_logical=theta_logical,
        fitted_state=replayed,
        optimizer_summary={
            **dict(excitation_fit.optimizer_summary),
            "composition": "complete_base_scaffold_prefix_then_compact_excitation_suffix",
            "base_runtime_parameter_count": int(base_theta.size),
            "excitation_runtime_parameter_count": int(excitation_theta.size),
            "total_runtime_parameter_count": int(theta_runtime.size),
            "base_qse_prepared_state_fidelity": float(base_qse_fidelity),
            "composed_replay_fidelity": float(replay_fidelity),
        },
    )
    prepared = PreparedStateResolution(
        state=psi_ref,
        provenance={
            "source_schema": BASE_COMPOSITION_SCHEMA_V1,
            "source": "base_scaffold_reference_state",
            "base_runtime_provenance": dict(getattr(runtime_input, "provenance", {}) or {}),
            "nq_total": int(round(math.log2(int(psi_ref.size)))),
        },
        override_used=False,
    )
    composition = {
        "schema": BASE_COMPOSITION_SCHEMA_V1,
        "base_runtime_parameter_count": int(base_theta.size),
        "base_logical_parameter_count": int(len(base_terms)),
        "excitation_runtime_parameter_count": int(excitation_theta.size),
        "excitation_logical_parameter_count": int(len(excitation_terms)),
        "total_runtime_parameter_count": int(theta_runtime.size),
        "total_logical_parameter_count": int(len(terms)),
        "base_qse_prepared_state_fidelity": float(base_qse_fidelity),
        "composed_replay_fidelity": float(replay_fidelity),
        "reference_state_role": "source_scaffold_reference_hf",
        "prepared_state_injection_used": False,
    }
    return combined, prepared, composition


def run_compact_qse_root_refit(config: CompactQSERootRefitConfig) -> dict[str, Any]:
    _validate_config(config)
    qse_path = Path(config.qse_result_json)
    qse_payload = _read_json(qse_path)
    target, qse_prepared, basis, nq = reconstruct_qse_root_target(
        qse_payload,
        qse_result_json=qse_path,
        state_index=int(config.state_index),
        allow_ground_state=bool(config.allow_ground_state),
        amplitude_cutoff=float(config.amplitude_cutoff),
    )
    hamiltonian_path = (
        Path(config.hamiltonian_json)
        if config.hamiltonian_json is not None
        else Path(config.base_scaffold_json)
    )
    hamiltonian, hamiltonian_provenance = load_polynomial_json(hamiltonian_path)
    suffix_fit, compact_diagnostics = fit_compact_greedy_pauli_ansatz(
        target_state=target.state,
        prepared_state=qse_prepared.state,
        basis=basis,
        nq=int(nq),
        hamiltonian=hamiltonian,
        qse_energy=float(target.qse_energy),
        max_selected_paulis=int(config.max_selected_paulis),
        target_infidelity=float(config.target_infidelity),
        max_energy_error=float(config.max_energy_error),
        max_physical_residual=float(config.max_physical_residual),
        optimizer_maxiter=int(config.optimizer_maxiter),
    )
    runtime_input = load_scaffold_runtime_input(Path(config.base_scaffold_json))
    fit, circuit_reference, composition = compose_base_scaffold_and_excitation(
        runtime_input=runtime_input,
        qse_prepared_state=qse_prepared.state,
        excitation_fit=suffix_fit,
    )
    standard_config = QSERootRefitConfig(
        qse_result_json=qse_path,
        state_index=int(config.state_index),
        output_json=Path(config.output_json),
        allow_ground_state=bool(config.allow_ground_state),
        hamiltonian_json=hamiltonian_path,
        max_infidelity=float(config.target_infidelity),
        max_energy_error=float(config.max_energy_error),
        maxiter=int(config.optimizer_maxiter),
        amplitude_cutoff=float(config.amplitude_cutoff),
    )
    hamiltonian_resolution = HamiltonianResolution(
        polynomial=hamiltonian,
        provenance={
            **dict(hamiltonian_provenance),
            "resolved_path": str(hamiltonian_path),
            "available": True,
        },
        explicit_override_used=True,
    )
    artifact = build_qse_root_refit_artifact(
        config=standard_config,
        qse_payload=qse_payload,
        target=target,
        prepared=circuit_reference,
        basis=basis,
        nq=int(nq),
        fit=fit,
        hamiltonian=hamiltonian_resolution,
    )
    artifact["base_scaffold_composition"] = {
        **composition,
        "base_scaffold_json": str(Path(config.base_scaffold_json)),
        "base_scaffold_sha256": _sha256_file(Path(config.base_scaffold_json)),
        "qse_source_prepared_state_provenance": dict(qse_prepared.provenance),
    }
    artifact["compact_refit"] = {
        "schema": COMPACT_REFIT_SCHEMA_V1,
        "candidate_source": "unique_nonidentity_pauli_children_of_full_qse_operator_basis",
        "candidate_count": int(compact_diagnostics.candidate_count),
        "selected_pauli_count": int(len(compact_diagnostics.selected_labels)),
        "selected_pauli_labels_exyz": list(compact_diagnostics.selected_labels),
        "selection_without_replacement": True,
        "selection_uses_exact_one_angle_fidelity_gain": True,
        "global_refit_uses_exact_analytic_fidelity_gradient": True,
        "depth_history": [dict(row) for row in compact_diagnostics.depth_history],
    }
    physical_pass = bool(
        float(compact_diagnostics.physical_residual_norm)
        <= float(config.max_physical_residual)
    )
    artifact["fit_summary"]["physical_residual_norm"] = float(
        compact_diagnostics.physical_residual_norm
    )
    artifact["fit_summary"]["thresholds"]["max_physical_residual"] = float(
        config.max_physical_residual
    )
    artifact["fit_summary"]["passes"]["physical_residual"] = physical_pass
    artifact["fit_summary"]["passes"]["all_thresholds"] = bool(
        artifact["fit_summary"]["passes"]["all_thresholds"] and physical_pass
    )
    artifact["ansatz_payload"]["prepared_state"] = build_statevector_manifest(
        psi_state=fit.fitted_state,
        source="qse_compact_root_refit.composed_fitted_root",
        handoff_state_kind="prepared_state",
        amplitude_cutoff=float(config.amplitude_cutoff),
    )
    artifact["warnings"] = [
        *list(artifact.get("warnings", [])),
        "compact_pauli_selection_is_offline_target_aware_and_forbidden_to_runtime_controller",
        "complete_base_scaffold_circuit_is_composed_before_excitation_suffix",
    ]
    artifact["source"]["selected_operator_basis"]["selection_mode"] = (
        "compact_greedy_pauli_children_full_qse_target"
    )
    write_manifest_json(Path(config.output_json), artifact)
    return artifact


__all__ = [
    "BASE_COMPOSITION_SCHEMA_V1",
    "COMPACT_REFIT_SCHEMA_V1",
    "CompactFitDiagnostics",
    "CompactQSERootRefitConfig",
    "compose_base_scaffold_and_excitation",
    "fit_compact_greedy_pauli_ansatz",
    "run_compact_qse_root_refit",
    "unique_qse_pauli_candidates",
]
