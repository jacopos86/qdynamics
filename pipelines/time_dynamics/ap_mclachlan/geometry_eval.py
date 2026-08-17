"""Statevector geometry evaluator for AP-McLachlan fixed-support dynamics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.geometry import McLachlanGeometry
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.performance import phase
from pipelines.time_dynamics.ap_mclachlan.state import APMcLachlanState


GEOMETRY_EVALUATION_SCHEMA_V1 = "ap_mclachlan_geometry_evaluation_v1"


@dataclass(frozen=True)
class GeometryEvaluation:
    """Prepared state plus measured/evaluated McLachlan geometry at one time."""

    geometry: McLachlanGeometry
    psi: np.ndarray
    h_psi: np.ndarray
    energy_expectation: float
    theta_runtime: np.ndarray
    tangent_matrix: np.ndarray | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": GEOMETRY_EVALUATION_SCHEMA_V1,
            "time": None if self.geometry.time is None else float(self.geometry.time),
            "runtime_parameter_count": int(np.asarray(self.theta_runtime).size),
            "energy_expectation": float(self.energy_expectation),
            "geometry": self.geometry.to_json_dict(),
        }


def evaluate_mclachlan_geometry(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray | Sequence[float] | None = None,
    time: float = 0.0,
    runtime_indices: Sequence[int] | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_tangent_matrix: bool = False,
) -> GeometryEvaluation:
    """Evaluate ``G``, ``f``, and ``V`` for Eq. (8) from a prepared scaffold state."""

    theta = (
        np.asarray(state.theta_runtime, dtype=float).reshape(-1)
        if theta_runtime is None
        else np.asarray(theta_runtime, dtype=float).reshape(-1)
    )
    if int(theta.size) != int(state.runtime_parameter_count):
        raise ValueError(
            f"theta_runtime length mismatch: got {theta.size}, expected {state.runtime_parameter_count}."
        )
    indices = _runtime_indices(runtime_indices, size=int(state.runtime_parameter_count))

    with phase("geometry.state_and_tangents"):
        psi, tangents_by_index = state.executor.prepare_state_with_parameter_tangents(
            theta,
            np.asarray(state.psi_ref, dtype=complex).reshape(-1),
            parameter_indices=indices,
        )
        psi = _normalize_state(psi, name="psi")

    with phase("geometry.hamiltonian_action"):
        hmat = np.asarray(hamiltonian.matrix_at(float(time)), dtype=complex)
        if hmat.shape != (int(psi.size), int(psi.size)):
            raise ValueError(
                f"Hamiltonian matrix shape {hmat.shape} does not match state dimension {psi.size}."
            )
        h_psi = np.asarray(hmat @ psi, dtype=complex).reshape(-1)
        energy = float(np.real(np.vdot(psi, h_psi)))
        b_bar = np.asarray(-1.0j * (h_psi - energy * psi), dtype=complex)

    with phase("geometry.horizontal_projection"):
        tangent_columns = [
            _horizontalize(
                np.asarray(tangents_by_index[int(idx)], dtype=complex).reshape(-1),
                psi=psi,
            )
            for idx in indices
        ]
        tangent_matrix = (
            np.column_stack(tangent_columns)
            if tangent_columns
            else np.zeros((int(psi.size), 0), dtype=complex)
        )

    with phase("geometry.gram_force"):
        if tangent_columns:
            K = np.asarray(np.real(tangent_matrix.conj().T @ tangent_matrix), dtype=float)
            f = np.asarray(np.real(tangent_matrix.conj().T @ b_bar), dtype=float).reshape(-1)
        else:
            K = np.zeros((0, 0), dtype=float)
            f = np.zeros(0, dtype=float)

    geometry_metadata = {
        "geometry_evaluation_schema": GEOMETRY_EVALUATION_SCHEMA_V1,
        "hamiltonian_drive_enabled": bool(hamiltonian.drive_enabled),
        **dict(metadata or {}),
    }
    geometry = McLachlanGeometry(
        K=0.5 * (K + K.T),
        f=f,
        norm_b_sq=float(np.real(np.vdot(b_bar, b_bar))),
        support_indices=indices,
        support_labels=_labels_for_indices(state.runtime_coordinate_labels, indices),
        time=float(time),
        metadata=geometry_metadata,
    )
    return GeometryEvaluation(
        geometry=geometry,
        psi=psi,
        h_psi=h_psi,
        energy_expectation=energy,
        theta_runtime=theta,
        tangent_matrix=tangent_matrix if bool(include_tangent_matrix) else None,
    )


def state_space_velocity_from_evaluation(
    evaluation: GeometryEvaluation,
    theta_dot: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return the Hilbert-space tangent velocity ``T theta_dot``.

    This helper is for changed-support patch checks. Same-support diagnostics can
    continue to use ``K`` without retaining the tangent matrix.
    """

    tangent_matrix = evaluation.tangent_matrix
    if tangent_matrix is None:
        raise ValueError("GeometryEvaluation does not include a tangent_matrix.")
    matrix = np.asarray(tangent_matrix, dtype=complex)
    theta = np.asarray(theta_dot, dtype=float).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("tangent_matrix must be a rank-2 array.")
    if int(theta.size) != int(matrix.shape[1]):
        raise ValueError(
            "theta_dot length must match tangent_matrix columns: "
            f"got {theta.size}, expected {matrix.shape[1]}."
        )
    if int(theta.size) != int(evaluation.geometry.dimension):
        raise ValueError(
            "theta_dot length must match McLachlan geometry dimension: "
            f"got {theta.size}, expected {evaluation.geometry.dimension}."
        )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(theta)):
        raise ValueError("tangent_matrix and theta_dot must contain only finite values.")
    velocity = np.asarray(matrix @ theta, dtype=complex).reshape(-1)
    if not np.all(np.isfinite(velocity)):
        raise ValueError("state-space velocity contains non-finite values.")
    return velocity


def geometry_evaluation_without_tangent_matrix(
    evaluation: GeometryEvaluation,
) -> GeometryEvaluation:
    """Drop optional tangent data before storing trajectory points."""

    if evaluation.tangent_matrix is None:
        return evaluation
    return replace(evaluation, tangent_matrix=None)


def _horizontalize(vector: np.ndarray, *, psi: np.ndarray) -> np.ndarray:
    return np.asarray(vector - psi * np.vdot(psi, vector), dtype=complex)


def _normalize_state(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have positive finite norm.")
    return np.asarray(arr / norm, dtype=complex)


def _runtime_indices(indices: Sequence[int] | None, *, size: int) -> tuple[int, ...]:
    if indices is None:
        return tuple(range(int(size)))
    out = tuple(int(idx) for idx in indices)
    if len(set(out)) != len(out):
        raise ValueError("runtime_indices must not contain duplicates.")
    for idx in out:
        if idx < 0 or idx >= int(size):
            raise ValueError(f"runtime index {idx} out of bounds for size {size}.")
    return out


def _labels_for_indices(labels: Sequence[str], indices: Sequence[int]) -> tuple[str, ...]:
    labels_tuple = tuple(str(label) for label in labels)
    if not labels_tuple:
        return tuple()
    return tuple(labels_tuple[int(idx)] for idx in indices)


__all__ = [
    "GEOMETRY_EVALUATION_SCHEMA_V1",
    "GeometryEvaluation",
    "evaluate_mclachlan_geometry",
    "geometry_evaluation_without_tangent_matrix",
    "state_space_velocity_from_evaluation",
]
