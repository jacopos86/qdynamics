"""Standard residual-synthesizing adaptive QSE benchmark.

This module implements the external quantum-Davidson comparator used by the
Paper III matched-accuracy campaign.  Unlike the production selector, it does
not choose records from a fixed alphabet: after a declared seed frame, every
new physical direction is synthesized from a preconditioned Ritz residual.

The projected pencil is solved through :func:`solve_qse_generalized_eigenproblem`
so overlap stabilization and retained-support semantics are identical to the
record-based QSE path.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from pipelines.qse_spectra.core import (
    QSEBasisElement,
    QSEMatrices,
    QSEObservable,
    QSEPruningConfig,
    apply_qse_observable,
    normalize_statevector,
    pauli_string_basis_element,
    solve_qse_generalized_eigenproblem,
)
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial


ADAPTIVE_QSE_BENCHMARK_SCHEMA_VERSION = "adaptive_qse_benchmark_v1"
STOP_RESIDUAL_CONVERGED = "RESIDUAL_CONVERGED"
STOP_MAX_DIMENSION = "MAX_DIMENSION"
STOP_POOL_EXHAUSTED_EQUIVALENT = "POOL_EXHAUSTED_EQUIVALENT"
_STOP_REASONS = {
    STOP_RESIDUAL_CONVERGED,
    STOP_MAX_DIMENSION,
    STOP_POOL_EXHAUSTED_EQUIVALENT,
}
DEFAULT_COSTING_CONVENTION = (
    "one_first_order_trotter_step_of_H_per_admitted_direction;"
    "marrakesh_graph_span_v1;two_qubit_only_v1"
)


def _finite_vector(value: Any, *, name: str, expected_size: int | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=complex).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if expected_size is not None and int(vector.size) != int(expected_size):
        raise ValueError(f"{name} has size {vector.size}; expected {expected_size}.")
    if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def guarded_davidson_correction(
    diagonal: np.ndarray,
    energy: float,
    residual: np.ndarray,
    *,
    denominator_floor: float = 1.0e-8,
) -> np.ndarray:
    """Return ``(diag(H) - energy I)^-1 residual`` with guarded denominators."""

    delta = float(denominator_floor)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("denominator_floor must be positive and finite.")
    diag = np.asarray(diagonal, dtype=float).reshape(-1)
    res = _finite_vector(residual, name="Davidson residual", expected_size=diag.size)
    if not np.all(np.isfinite(diag)):
        raise ValueError("Hamiltonian diagonal contains non-finite values.")
    eps = float(energy)
    if not math.isfinite(eps):
        raise ValueError("Ritz energy must be finite.")

    denominator = diag - eps
    signs = np.where(denominator < 0.0, -1.0, 1.0)
    guarded = np.where(np.abs(denominator) < delta, signs * delta, denominator)
    correction = res / guarded
    return _finite_vector(correction, name="guarded Davidson correction", expected_size=diag.size)


def orthogonalize_adaptive_direction(
    candidate: np.ndarray,
    retained_frame: np.ndarray,
    *,
    independence_floor: float = 1.0e-12,
) -> tuple[np.ndarray | None, float]:
    """Project a candidate against the retained frame and return its novelty.

    Novelty is the same squared orthogonal-component fraction used by the
    production record selector, ``||(I-P)t||^2 / ||t||^2``.  Two projection
    passes match that selector's reorthogonalized Gram--Schmidt convention.
    """

    floor = float(independence_floor)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("independence_floor must be finite and non-negative.")
    vector = _finite_vector(candidate, name="adaptive candidate")
    frame = np.asarray(retained_frame, dtype=complex)
    if frame.ndim != 2 or int(frame.shape[0]) != int(vector.size):
        raise ValueError("retained_frame must be a 2D matrix with candidate-sized rows.")
    if not np.all(np.isfinite(frame.real)) or not np.all(np.isfinite(frame.imag)):
        raise ValueError("retained_frame contains non-finite values.")

    norm_sq = float(np.vdot(vector, vector).real)
    if norm_sq <= 0.0:
        return None, 0.0
    projected = vector.copy()
    for _pass in range(2):
        for column in range(frame.shape[1]):
            unit = frame[:, column]
            projected = projected - complex(np.vdot(unit, projected)) * unit
    projected_norm_sq = max(0.0, float(np.vdot(projected, projected).real))
    novelty = max(0.0, projected_norm_sq / norm_sq)
    if projected_norm_sq <= 0.0 or novelty < floor:
        return None, float(novelty)
    return np.asarray(projected / math.sqrt(projected_norm_sq), dtype=complex), float(novelty)


def _pauli_diagonal(hamiltonian: PauliPolynomial, *, dim: int, nq: int) -> np.ndarray:
    indices = np.arange(int(dim), dtype=np.uint64)
    diagonal = np.zeros(int(dim), dtype=complex)
    for term in hamiltonian.return_polynomial():
        coefficient = complex(term.p_coeff)
        word = str(term.pw2strng()).lower()
        if len(word) != int(nq):
            raise ValueError(f"Hamiltonian term {word!r} does not have nq={nq}.")
        if set(word) - set("exyz"):
            raise ValueError(f"Hamiltonian term uses unsupported Pauli symbols: {word!r}.")
        if "x" in word or "y" in word:
            continue
        values = np.ones(int(dim), dtype=float)
        for position, symbol in enumerate(word):
            if symbol != "z":
                continue
            qubit = int(nq) - 1 - int(position)
            values *= 1.0 - 2.0 * ((indices >> qubit) & 1).astype(float)
        diagonal += coefficient * values
    imag_max = float(np.max(np.abs(diagonal.imag))) if diagonal.size else 0.0
    if imag_max > 1.0e-12:
        raise ValueError(f"Hamiltonian diagonal has imaginary component {imag_max}.")
    return np.asarray(diagonal.real, dtype=float)


def _hamiltonian_operator(
    hamiltonian: Any,
    *,
    dim: int,
    nq: int,
    config: QSEPruningConfig,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, str]:
    if isinstance(hamiltonian, np.ndarray):
        dense = np.asarray(hamiltonian, dtype=complex)
        if dense.shape != (int(dim), int(dim)):
            raise ValueError(f"dense Hamiltonian has shape {dense.shape}; expected {(dim, dim)}.")
        if not np.all(np.isfinite(dense.real)) or not np.all(np.isfinite(dense.imag)):
            raise ValueError("dense Hamiltonian contains non-finite values.")
        residual = float(np.max(np.abs(dense - dense.conj().T)))
        allowed = max(
            float(config.hermitian_absolute_tolerance),
            float(config.hermitian_relative_tolerance)
            * max(1.0, float(np.max(np.abs(dense)))),
        )
        if residual > allowed:
            raise ValueError(f"dense Hamiltonian is non-Hermitian: residual {residual} > {allowed}.")
        dense = 0.5 * (dense + dense.conj().T)

        def _apply(vector: np.ndarray) -> np.ndarray:
            return np.asarray(dense @ vector, dtype=complex)

        diagonal = np.asarray(np.diag(dense), dtype=complex)
        imag_max = float(np.max(np.abs(diagonal.imag))) if diagonal.size else 0.0
        if imag_max > float(config.hamiltonian_coeff_imag_absolute_tolerance):
            raise ValueError(f"dense Hamiltonian diagonal has imaginary component {imag_max}.")
        return _apply, np.asarray(diagonal.real, dtype=float), "dense_hermitian"

    if isinstance(hamiltonian, PauliPolynomial):
        if int(hamiltonian.get_nq()) != int(nq):
            raise ValueError(f"Hamiltonian has nq={hamiltonian.get_nq()}; state has nq={nq}.")
        coefficient_imag_max = max(
            (abs(float(complex(term.p_coeff).imag)) for term in hamiltonian.return_polynomial()),
            default=0.0,
        )
        if coefficient_imag_max > float(config.hamiltonian_coeff_imag_absolute_tolerance):
            raise ValueError(
                f"Hamiltonian coefficient imaginary part {coefficient_imag_max} exceeds tolerance."
            )
        compiled = compile_polynomial_action(
            hamiltonian, tol=float(config.polynomial_drop_abs_tol)
        )

        def _apply(vector: np.ndarray) -> np.ndarray:
            result = apply_compiled_polynomial(np.asarray(vector, dtype=complex), compiled)
            return _finite_vector(result, name="Hamiltonian-applied vector", expected_size=dim)

        diagonal = _pauli_diagonal(hamiltonian, dim=int(dim), nq=int(nq))
        return _apply, diagonal, "pauli_polynomial_compiled_action"

    raise TypeError("hamiltonian must be a dense numpy array or PauliPolynomial.")


def _seed_action(
    seed: Any,
    prepared_state: np.ndarray,
    *,
    nq: int,
    config: QSEPruningConfig,
) -> tuple[np.ndarray, str]:
    if isinstance(seed, QSEBasisElement):
        observable = QSEObservable(
            name=str(seed.name),
            kind=str(seed.kind),
            pauli_label_exyz=seed.pauli_label_exyz,
            polynomial=seed.polynomial,
            metadata=seed.metadata,
        )
        vector = apply_qse_observable(
            observable,
            prepared_state,
            config=config,
            normalize_state=False,
            expected_nq=int(nq),
        )
        return np.asarray(vector, dtype=complex), str(seed.name)
    return (
        _finite_vector(seed, name="seed vector", expected_size=prepared_state.size),
        "explicit_physical_vector",
    )


def _projected_ritz_solve(
    frame: Sequence[np.ndarray],
    h_frame: Sequence[np.ndarray],
    *,
    nq: int,
    reference_energy: float,
    config: QSEPruningConfig,
) -> tuple[Any, np.ndarray, np.ndarray]:
    vectors = np.column_stack(tuple(frame))
    h_vectors = np.column_stack(tuple(h_frame))
    overlap = vectors.conj().T @ vectors
    projected_hamiltonian = vectors.conj().T @ h_vectors
    overlap = 0.5 * (overlap + overlap.conj().T)
    projected_hamiltonian = 0.5 * (
        projected_hamiltonian + projected_hamiltonian.conj().T
    )
    identity = "e" * int(nq)
    basis = tuple(
        pauli_string_basis_element(identity, nq=int(nq), name=f"adaptive_direction_{index}")
        for index in range(len(frame))
    )
    matrices = QSEMatrices(
        nq=int(nq),
        hilbert_dim=int(vectors.shape[0]),
        basis_elements=basis,
        reference_energy=float(reference_energy),
        reference_energy_imag_abs=0.0,
        basis_vector_norms=tuple(float(np.linalg.norm(vector)) for vector in frame),
        overlap=np.asarray(overlap, dtype=complex),
        hamiltonian=np.asarray(projected_hamiltonian, dtype=complex),
        overlap_hermitian_residual_max_abs_raw=0.0,
        hamiltonian_hermitian_residual_max_abs_raw=0.0,
        hamiltonian_coeff_imag_max_abs=0.0,
        basis_matrix_vectors=tuple(np.asarray(vector, dtype=complex) for vector in frame),
    )
    result = solve_qse_generalized_eigenproblem(matrices, config=config)
    ritz_vectors = vectors @ np.asarray(result.eigenvectors_basis, dtype=complex)
    h_ritz_vectors = h_vectors @ np.asarray(result.eigenvectors_basis, dtype=complex)
    return result, np.asarray(ritz_vectors, dtype=complex), np.asarray(h_ritz_vectors, dtype=complex)


def _resource_triple(value: Mapping[str, Any] | None) -> dict[str, float]:
    raw = {} if value is None else dict(value)
    resources: dict[str, float] = {}
    for key in ("n2q", "d2q", "dc"):
        number = float(raw.get(key, 0.0))
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"direction_resources[{key!r}] must be finite and non-negative.")
        resources[key] = number
    return resources


def _scaled_resources(per_direction: Mapping[str, float], count: int) -> dict[str, float]:
    return {key: float(value) * int(count) for key, value in per_direction.items()}


def run_adaptive_qse_benchmark(
    hamiltonian: Any,
    prepared_state: np.ndarray,
    *,
    target_roots: int,
    eps_residual: float,
    max_dimension: int,
    seed_elements: Sequence[Any] = (),
    denominator_floor: float = 1.0e-8,
    independence_floor: float = 1.0e-12,
    qse_config: QSEPruningConfig | None = None,
    direction_resources: Mapping[str, Any] | None = None,
    costing_convention: str = DEFAULT_COSTING_CONVENTION,
) -> dict[str, Any]:
    """Run standard adaptive QSE/quantum Davidson and return an audit dict.

    ``prepared_state`` is the free reference seed.  Every admitted declared
    seed image and every residual-synthesized correction is charged once under
    ``direction_resources``; the reference itself is not charged.
    """

    roots_requested = int(target_roots)
    dimension_cap = int(max_dimension)
    residual_target = float(eps_residual)
    if roots_requested <= 0:
        raise ValueError("target_roots must be positive.")
    if dimension_cap <= 0:
        raise ValueError("max_dimension must be positive.")
    if not math.isfinite(residual_target) or residual_target <= 0.0:
        raise ValueError("eps_residual must be positive and finite.")
    convention = str(costing_convention).strip()
    if not convention:
        raise ValueError("costing_convention must be non-empty.")

    cfg = qse_config if qse_config is not None else QSEPruningConfig()
    psi, _norm, nq = normalize_statevector(prepared_state)
    dim = int(psi.size)
    if roots_requested > dim:
        raise ValueError(f"target_roots={roots_requested} exceeds Hilbert dimension {dim}.")
    apply_h, diagonal, hamiltonian_representation = _hamiltonian_operator(
        hamiltonian, dim=dim, nq=int(nq), config=cfg
    )
    h_psi = apply_h(psi)
    reference_energy_complex = complex(np.vdot(psi, h_psi))
    if abs(float(reference_energy_complex.imag)) > float(cfg.hermitian_absolute_tolerance):
        raise ValueError("prepared-state energy has a non-negligible imaginary part.")
    reference_energy = float(reference_energy_complex.real)

    per_direction = _resource_triple(direction_resources)
    frame: list[np.ndarray] = [np.asarray(psi, dtype=complex)]
    h_frame: list[np.ndarray] = [np.asarray(h_psi, dtype=complex)]
    charged_directions = 0
    seed_rows: list[dict[str, Any]] = []
    declared_seeds = tuple(seed_elements)
    for seed_index, seed in enumerate(declared_seeds):
        if len(frame) >= dimension_cap:
            break
        raw, name = _seed_action(seed, psi, nq=int(nq), config=cfg)
        q0_projected = raw - complex(np.vdot(psi, raw)) * psi
        retained_frame = np.column_stack(tuple(frame))
        admitted, novelty = orthogonalize_adaptive_direction(
            q0_projected,
            retained_frame,
            independence_floor=float(independence_floor),
        )
        accepted = admitted is not None
        seed_rows.append(
            {
                "seed_index": int(seed_index),
                "name": str(name),
                "q0_projected_norm": float(np.linalg.norm(q0_projected)),
                "novelty_fraction": float(novelty),
                "admitted": bool(accepted),
            }
        )
        if admitted is not None:
            frame.append(admitted)
            h_frame.append(apply_h(admitted))
            charged_directions += 1

    iterations: list[dict[str, Any]] = []
    preceding_admission: dict[str, Any] | None = None
    stop_reason: str | None = None
    final_result: Any = None
    final_residuals: list[float] = []

    while stop_reason is None:
        result, ritz_vectors, h_ritz_vectors = _projected_ritz_solve(
            frame,
            h_frame,
            nq=int(nq),
            reference_energy=float(reference_energy),
            config=cfg,
        )
        final_result = result
        evaluated_roots = min(int(roots_requested), int(result.eigenvalues.size))
        residual_vectors: list[np.ndarray] = []
        residual_norms: list[float] = []
        for root in range(evaluated_roots):
            residual = (
                h_ritz_vectors[:, root]
                - float(result.eigenvalues[root]) * ritz_vectors[:, root]
            )
            residual = _finite_vector(
                residual, name=f"Ritz residual {root}", expected_size=dim
            )
            residual_vectors.append(residual)
            residual_norms.append(float(np.linalg.norm(residual)))
        max_residual = max(residual_norms, default=float("inf"))
        final_residuals = residual_norms
        resources = _scaled_resources(per_direction, charged_directions)
        iterations.append(
            {
                "iteration": int(len(iterations)),
                "dimension": int(len(frame)),
                "retained_rank": int(result.retained_rank),
                "target_window_complete": bool(result.eigenvalues.size >= roots_requested),
                "root_energies": [
                    float(value) for value in result.eigenvalues[:roots_requested]
                ],
                "root_residual_norms": [float(value) for value in residual_norms],
                "max_root_residual": float(max_residual),
                "admitted_direction_novelty_fraction": (
                    None
                    if preceding_admission is None
                    else float(preceding_admission["novelty_fraction"])
                ),
                "admitted_from_root": (
                    None
                    if preceding_admission is None
                    else int(preceding_admission["root_index"])
                ),
                "resources": resources,
            }
        )

        if result.eigenvalues.size >= roots_requested and max_residual <= residual_target:
            stop_reason = STOP_RESIDUAL_CONVERGED
            break
        if len(frame) >= dimension_cap:
            stop_reason = STOP_MAX_DIMENSION
            break

        retained_frame = np.asarray(ritz_vectors, dtype=complex)
        accepted_direction: np.ndarray | None = None
        accepted_metadata: dict[str, Any] | None = None
        root_order = sorted(
            range(evaluated_roots), key=lambda root: (-residual_norms[root], root)
        )
        for root in root_order:
            correction = guarded_davidson_correction(
                diagonal,
                float(result.eigenvalues[root]),
                residual_vectors[root],
                denominator_floor=float(denominator_floor),
            )
            direction, novelty = orthogonalize_adaptive_direction(
                correction,
                retained_frame,
                independence_floor=float(independence_floor),
            )
            if direction is None:
                continue
            accepted_direction = direction
            accepted_metadata = {
                "root_index": int(root),
                "root_residual_norm": float(residual_norms[root]),
                "novelty_fraction": float(novelty),
            }
            break
        if accepted_direction is None or accepted_metadata is None:
            stop_reason = STOP_POOL_EXHAUSTED_EQUIVALENT
            break
        frame.append(accepted_direction)
        h_frame.append(apply_h(accepted_direction))
        charged_directions += 1
        preceding_admission = accepted_metadata

    if stop_reason not in _STOP_REASONS:
        raise RuntimeError(f"unexpected adaptive-QSE stop reason {stop_reason!r}.")
    if final_result is None:
        raise RuntimeError("adaptive-QSE solve produced no iterations.")
    final_resources = _scaled_resources(per_direction, charged_directions)
    return {
        "schema_version": ADAPTIVE_QSE_BENCHMARK_SCHEMA_VERSION,
        "method": "standard_adaptive_qse_quantum_davidson",
        "construction_family": "residual_synthesized_no_record_alphabet",
        "record_alphabet_consumed": False,
        "hamiltonian_representation": str(hamiltonian_representation),
        "config": {
            "target_roots": int(roots_requested),
            "eps_residual": float(residual_target),
            "max_dimension": int(dimension_cap),
            "denominator_floor": float(denominator_floor),
            "independence_floor": float(independence_floor),
            "overlap_relative_cutoff": float(cfg.overlap_relative_cutoff),
            "overlap_absolute_cutoff": float(cfg.overlap_absolute_cutoff),
            "retained_support_convention": "core.solve_qse_generalized_eigenproblem",
        },
        "seed_policy": {
            "reference_state_included": True,
            "reference_state_charged": False,
            "reference_projection": "q0",
            "basis_vector_normalization": "orthogonal_component_normalized_on_admission",
            "declared_seed_set_size": int(len(declared_seeds)),
            "evaluated_seed_set_size": int(len(seed_rows)),
            "admitted_seed_direction_count": int(sum(row["admitted"] for row in seed_rows)),
            "seed_rows": seed_rows,
        },
        "costing": {
            "convention": convention,
            "per_admitted_direction": dict(per_direction),
            "charged_direction_count": int(charged_directions),
            "resources": final_resources,
        },
        "iterations": iterations,
        "terminal_dimension": int(len(frame)),
        "terminal_retained_rank": int(final_result.retained_rank),
        "root_energies": [
            float(value) for value in final_result.eigenvalues[:roots_requested]
        ],
        "root_residual_norms": [float(value) for value in final_residuals],
        "max_root_residual": float(max(final_residuals, default=float("inf"))),
        "target_window_complete": bool(final_result.eigenvalues.size >= roots_requested),
        "resources": final_resources,
        "stop_reason": str(stop_reason),
    }


__all__ = [
    "ADAPTIVE_QSE_BENCHMARK_SCHEMA_VERSION",
    "DEFAULT_COSTING_CONVENTION",
    "STOP_MAX_DIMENSION",
    "STOP_POOL_EXHAUSTED_EQUIVALENT",
    "STOP_RESIDUAL_CONVERGED",
    "guarded_davidson_correction",
    "orthogonalize_adaptive_direction",
    "run_adaptive_qse_benchmark",
]
