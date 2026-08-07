#!/usr/bin/env python3
"""Degeneracy-safe, reporting-only ground-space fidelity utilities.

This module is deliberately outside every controller and optimizer path.  Its
public entry point requires the working/reference cutoff pair, the fixed-sector
basis, and the legal binary-code basis explicitly.  It then restricts the
Hamiltonian to their intersection before resolving the ground eigenspace.

The reported fidelity is

    <psi|P_0|psi> = sum_j |<phi_j|psi>|^2,

where the ``phi_j`` span every eigenvector within the named degeneracy
tolerance of the physical-sector ground energy.  The projector convention is
therefore invariant under arbitrary rotations and phases inside a degenerate
ground space.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


GROUND_SPACE_FIDELITY_SCHEMA = "ground_space_projector_fidelity_v1"
GROUND_SPACE_TOLERANCE_SCHEMA = "ground_space_degeneracy_tolerance_v1"


class GroundSpaceFidelityError(ValueError):
    """Fail-closed error with a stable machine-readable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class GroundSpaceTolerance:
    """Absolute-plus-relative ground-cluster tolerance.

    The threshold is ``max(absolute, relative * max(1, |E0|))``.  Both values
    are serialized so a later audit can reproduce the same cluster exactly.
    """

    absolute: float = 1.0e-10
    relative: float = 1.0e-10

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.absolute)) or float(self.absolute) < 0.0:
            raise GroundSpaceFidelityError(
                "invalid_degeneracy_absolute_tolerance",
                "absolute degeneracy tolerance must be finite and nonnegative",
            )
        if not math.isfinite(float(self.relative)) or float(self.relative) < 0.0:
            raise GroundSpaceFidelityError(
                "invalid_degeneracy_relative_tolerance",
                "relative degeneracy tolerance must be finite and nonnegative",
            )
        if float(self.absolute) == 0.0 and float(self.relative) == 0.0:
            raise GroundSpaceFidelityError(
                "zero_degeneracy_tolerance",
                "at least one degeneracy tolerance component must be positive",
            )

    def threshold(self, ground_energy: float) -> float:
        return float(
            max(
                float(self.absolute),
                float(self.relative) * max(1.0, abs(float(ground_energy))),
            )
        )

    def as_dict(self, *, ground_energy: float) -> dict[str, Any]:
        return {
            "schema": GROUND_SPACE_TOLERANCE_SCHEMA,
            "absolute": float(self.absolute),
            "relative": float(self.relative),
            "resolved_threshold": float(self.threshold(float(ground_energy))),
            "formula": "max(absolute, relative * max(1, abs(E0)))",
        }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _normalized_state(state: np.ndarray, *, label: str) -> tuple[np.ndarray, float]:
    value = np.asarray(state, dtype=complex).reshape(-1)
    if value.size == 0 or not np.all(np.isfinite(value.real)) or not np.all(
        np.isfinite(value.imag)
    ):
        raise GroundSpaceFidelityError(
            f"invalid_{label}", f"{label} must contain finite amplitudes"
        )
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise GroundSpaceFidelityError(
            f"zero_norm_{label}", f"{label} must have positive finite norm"
        )
    return np.asarray(value / norm, dtype=complex), norm


def _validated_basis_indices(
    values: Sequence[int], *, dimension: int, label: str
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise GroundSpaceFidelityError(
            f"invalid_{label}", f"{label} must be an integer sequence"
        )
    try:
        indices = tuple(int(value) for value in values)
    except Exception as exc:  # pragma: no cover - defensive type boundary
        raise GroundSpaceFidelityError(
            f"invalid_{label}", f"{label} must be an integer sequence"
        ) from exc
    if not indices:
        raise GroundSpaceFidelityError(f"empty_{label}", f"{label} must not be empty")
    if len(set(indices)) != len(indices):
        raise GroundSpaceFidelityError(
            f"duplicate_{label}", f"{label} must not contain duplicate indices"
        )
    if min(indices) < 0 or max(indices) >= int(dimension):
        raise GroundSpaceFidelityError(
            f"out_of_range_{label}",
            f"{label} contains an index outside [0, {int(dimension)})",
        )
    return tuple(sorted(indices))


def _dense_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    matrix = np.asarray(value, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise GroundSpaceFidelityError(
            "hamiltonian_not_square", "Hamiltonian must be a square matrix"
        )
    if matrix.shape[0] == 0:
        raise GroundSpaceFidelityError(
            "hamiltonian_empty", "Hamiltonian dimension must be positive"
        )
    if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
        raise GroundSpaceFidelityError(
            "hamiltonian_nonfinite", "Hamiltonian contains nonfinite entries"
        )
    return matrix


def _complex_matrix_sha256(matrix: np.ndarray, *, decimals: int = 14) -> str:
    value = np.asarray(matrix, dtype=complex)
    real = np.round(value.real, decimals=int(decimals))
    imag = np.round(value.imag, decimals=int(decimals))
    real[np.abs(real) < 10.0 ** (-int(decimals))] = 0.0
    imag[np.abs(imag) < 10.0 ** (-int(decimals))] = 0.0
    payload = {
        "shape": [int(x) for x in value.shape],
        "round_decimals": int(decimals),
        "real": real.tolist(),
        "imag": imag.tolist(),
    }
    return _canonical_sha256(payload)


def projector_fidelity(
    variational_state: np.ndarray,
    ground_vectors: np.ndarray,
) -> float:
    """Return fidelity with the span of ``ground_vectors``.

    The vectors may use any basis for the same subspace.  QR orthonormalization
    makes the result insensitive to their phases, ordering, or unitary mixing.
    """

    state, _ = _normalized_state(variational_state, label="variational_state")
    vectors = np.asarray(ground_vectors, dtype=complex)
    if vectors.ndim == 1:
        vectors = vectors.reshape(-1, 1)
    if vectors.ndim != 2 or vectors.shape[0] != state.size or vectors.shape[1] < 1:
        raise GroundSpaceFidelityError(
            "ground_vectors_shape_mismatch",
            "ground vectors must have shape (state_dimension, positive_rank)",
        )
    if not np.all(np.isfinite(vectors.real)) or not np.all(np.isfinite(vectors.imag)):
        raise GroundSpaceFidelityError(
            "ground_vectors_nonfinite", "ground vectors contain nonfinite entries"
        )
    q, r = np.linalg.qr(vectors)
    diagonal = np.abs(np.diag(r))
    if diagonal.size != vectors.shape[1] or np.any(diagonal <= 1.0e-14):
        raise GroundSpaceFidelityError(
            "ground_vectors_rank_deficient", "ground vectors are linearly dependent"
        )
    overlaps = np.asarray(q.conj().T @ state, dtype=complex).reshape(-1)
    value = float(np.sum(np.abs(overlaps) ** 2).real)
    return float(min(1.0, max(0.0, value)))


def evaluate_ground_space_fidelity(
    *,
    hamiltonian: Any,
    variational_state: np.ndarray,
    working_cutoff: int,
    reference_cutoff: int,
    fixed_sector_basis_indices: Sequence[int],
    legal_binary_basis_indices: Sequence[int],
    fixed_sector_label: str,
    legal_binary_basis_label: str,
    tolerance: GroundSpaceTolerance | None = None,
    state_leakage_tolerance: float = 1.0e-10,
    hermiticity_tolerance: float = 1.0e-10,
    subspace_invariance_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Evaluate same-cutoff ground-space fidelity in an explicit physical basis.

    This function fails closed on cutoff mismatch, missing basis metadata,
    non-Hermitian input, or variational leakage beyond the requested tolerance.
    It never performs optimization, refitting, controller selection, or query
    accounting.
    """

    if int(working_cutoff) != int(reference_cutoff):
        raise GroundSpaceFidelityError(
            "cutoff_mismatch",
            "working and exact-reference cutoffs must be identical for fidelity",
        )
    if not str(fixed_sector_label).strip():
        raise GroundSpaceFidelityError(
            "missing_fixed_sector_label", "fixed-sector label is required"
        )
    if not str(legal_binary_basis_label).strip():
        raise GroundSpaceFidelityError(
            "missing_legal_binary_basis_label",
            "legal-binary basis label is required",
        )
    if not math.isfinite(float(state_leakage_tolerance)) or float(
        state_leakage_tolerance
    ) < 0.0:
        raise GroundSpaceFidelityError(
            "invalid_state_leakage_tolerance",
            "state leakage tolerance must be finite and nonnegative",
        )
    if not math.isfinite(float(hermiticity_tolerance)) or float(
        hermiticity_tolerance
    ) < 0.0:
        raise GroundSpaceFidelityError(
            "invalid_hermiticity_tolerance",
            "Hermiticity tolerance must be finite and nonnegative",
        )
    if not math.isfinite(float(subspace_invariance_tolerance)) or float(
        subspace_invariance_tolerance
    ) < 0.0:
        raise GroundSpaceFidelityError(
            "invalid_subspace_invariance_tolerance",
            "subspace-invariance tolerance must be finite and nonnegative",
        )

    matrix = _dense_matrix(hamiltonian)
    dimension = int(matrix.shape[0])
    state, input_norm = _normalized_state(
        variational_state, label="variational_state"
    )
    if int(state.size) != int(dimension):
        raise GroundSpaceFidelityError(
            "state_hamiltonian_dimension_mismatch",
            "variational-state and Hamiltonian dimensions differ",
        )

    sector_basis = _validated_basis_indices(
        fixed_sector_basis_indices,
        dimension=dimension,
        label="fixed_sector_basis_indices",
    )
    legal_basis = _validated_basis_indices(
        legal_binary_basis_indices,
        dimension=dimension,
        label="legal_binary_basis_indices",
    )
    physical_basis = tuple(sorted(set(sector_basis).intersection(legal_basis)))
    if not physical_basis:
        raise GroundSpaceFidelityError(
            "empty_physical_basis_intersection",
            "fixed-sector and legal-binary bases have an empty intersection",
        )

    scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    hermiticity_error = float(np.linalg.norm(matrix - matrix.conj().T, ord=np.inf))
    if hermiticity_error > float(hermiticity_tolerance) * scale:
        raise GroundSpaceFidelityError(
            "hamiltonian_not_hermitian",
            "Hamiltonian exceeds the configured Hermiticity tolerance",
        )
    matrix = 0.5 * (matrix + matrix.conj().T)

    physical_indices = np.asarray(physical_basis, dtype=int)
    physical_set = set(int(value) for value in physical_basis)
    complement_indices = np.asarray(
        [index for index in range(dimension) if index not in physical_set],
        dtype=int,
    )
    if complement_indices.size:
        physical_coupling = np.asarray(
            matrix[np.ix_(complement_indices, physical_indices)], dtype=complex
        )
        subspace_invariance_error = float(
            np.linalg.norm(physical_coupling, ord=np.inf)
        )
    else:
        subspace_invariance_error = 0.0
    if subspace_invariance_error > float(subspace_invariance_tolerance) * scale:
        raise GroundSpaceFidelityError(
            "hamiltonian_does_not_preserve_physical_basis",
            "Hamiltonian couples the supplied physical basis to its complement",
        )
    physical_probability = float(np.sum(np.abs(state[physical_indices]) ** 2).real)
    physical_probability = float(min(1.0, max(0.0, physical_probability)))
    leakage_probability = float(max(0.0, 1.0 - physical_probability))
    if leakage_probability > float(state_leakage_tolerance):
        raise GroundSpaceFidelityError(
            "variational_state_outside_physical_basis",
            "variational state has fixed-sector or padding leakage above tolerance",
        )
    if physical_probability <= 0.0:
        raise GroundSpaceFidelityError(
            "zero_physical_state_norm",
            "variational state has zero norm in the physical basis",
        )
    sector_hamiltonian = np.asarray(
        matrix[np.ix_(physical_indices, physical_indices)], dtype=complex
    )
    eigenvalues, eigenvectors = np.linalg.eigh(sector_hamiltonian)
    eigenvalues = np.asarray(eigenvalues.real, dtype=float)
    ground_energy = float(eigenvalues[0])
    tolerance_policy = tolerance or GroundSpaceTolerance()
    cluster_tolerance = float(tolerance_policy.threshold(ground_energy))
    ground_mask = np.asarray(
        (eigenvalues - ground_energy) <= cluster_tolerance, dtype=bool
    )
    ground_vectors = np.asarray(eigenvectors[:, ground_mask], dtype=complex)
    multiplicity = int(ground_vectors.shape[1])
    if multiplicity < 1:  # pragma: no cover - eigh always includes E0
        raise GroundSpaceFidelityError(
            "ground_space_empty", "eigensolver produced no ground-space vectors"
        )

    # Embed the physical-sector eigenspace back into the full register before
    # evaluating <psi|P0|psi>.  Do not renormalize the restricted state: even
    # tolerated numerical leakage must remain visible in the reported fidelity.
    ground_vectors_full = np.zeros(
        (dimension, multiplicity), dtype=complex
    )
    ground_vectors_full[physical_indices, :] = ground_vectors
    fidelity = projector_fidelity(state, ground_vectors_full)
    infidelity = float(max(0.0, 1.0 - fidelity))
    next_eigenvalue = (
        None if multiplicity >= eigenvalues.size else float(eigenvalues[multiplicity])
    )
    gap = (
        None
        if next_eigenvalue is None
        else float(max(0.0, next_eigenvalue - ground_energy))
    )
    unique_proved = bool(
        multiplicity == 1
        and (eigenvalues.size == 1 or (gap is not None and gap > cluster_tolerance))
    )

    projector = np.asarray(
        ground_vectors @ ground_vectors.conj().T, dtype=complex
    )
    basis_payload = {
        "full_dimension": int(dimension),
        "fixed_sector_label": str(fixed_sector_label),
        "legal_binary_basis_label": str(legal_binary_basis_label),
        "fixed_sector_basis_indices": [int(x) for x in sector_basis],
        "legal_binary_basis_indices": [int(x) for x in legal_basis],
        "physical_basis_indices": [int(x) for x in physical_basis],
        "working_cutoff": int(working_cutoff),
        "reference_cutoff": int(reference_cutoff),
    }
    basis_sha256 = _canonical_sha256(basis_payload)
    fixed_sector_basis_sha256 = _canonical_sha256(
        {
            "full_dimension": int(dimension),
            "label": str(fixed_sector_label),
            "indices": [int(x) for x in sector_basis],
        }
    )
    legal_binary_basis_sha256 = _canonical_sha256(
        {
            "full_dimension": int(dimension),
            "label": str(legal_binary_basis_label),
            "indices": [int(x) for x in legal_basis],
        }
    )
    physical_basis_index_sha256 = _canonical_sha256(
        {
            "full_dimension": int(dimension),
            "indices": [int(x) for x in physical_basis],
        }
    )
    projector_sha256 = _canonical_sha256(
        {
            "basis_sha256": basis_sha256,
            "projector_matrix_sha256": _complex_matrix_sha256(projector),
            "projector_rank": int(multiplicity),
        }
    )
    hamiltonian_sha256 = _complex_matrix_sha256(sector_hamiltonian)

    residuals = [
        float(
            np.linalg.norm(
                sector_hamiltonian @ ground_vectors[:, index]
                - eigenvalues[index] * ground_vectors[:, index]
            )
        )
        for index in range(multiplicity)
    ]
    return {
        "schema": GROUND_SPACE_FIDELITY_SCHEMA,
        "status": "ok",
        "usage_scope": "post_run_reporting_only",
        "controller_decision_eligible": False,
        "optimizer_input_eligible": False,
        "stopping_input_eligible": False,
        "s_alg_charged": False,
        "same_cutoff_verified": True,
        "working_cutoff": int(working_cutoff),
        "reference_cutoff": int(reference_cutoff),
        "fixed_sector_label": str(fixed_sector_label),
        "legal_binary_basis_label": str(legal_binary_basis_label),
        "full_dimension": int(dimension),
        "fixed_sector_basis_count": int(len(sector_basis)),
        "legal_binary_basis_count": int(len(legal_basis)),
        "physical_basis_count": int(len(physical_basis)),
        "physical_basis_sha256": basis_sha256,
        "fixed_sector_basis_sha256": fixed_sector_basis_sha256,
        "legal_binary_basis_sha256": legal_binary_basis_sha256,
        "physical_basis_index_sha256": physical_basis_index_sha256,
        "physical_sector_hamiltonian_sha256": hamiltonian_sha256,
        "projector_sha256": projector_sha256,
        "reference_convention": (
            "unique_ground_state_vector"
            if unique_proved
            else "degenerate_ground_space_projector"
        ),
        "ground_energy": float(ground_energy),
        "ground_space_multiplicity": int(multiplicity),
        "ground_space_unique_proved": bool(unique_proved),
        "ground_space_next_eigenvalue": next_eigenvalue,
        "ground_space_gap": gap,
        "ground_space_eigenvalues": [
            float(x) for x in eigenvalues[:multiplicity].tolist()
        ],
        "ground_space_max_residual": float(max(residuals, default=0.0)),
        "degeneracy_tolerance": tolerance_policy.as_dict(
            ground_energy=ground_energy
        ),
        "variational_state_input_norm": float(input_norm),
        "variational_state_physical_probability": float(physical_probability),
        "variational_state_leakage_probability": float(leakage_probability),
        "state_leakage_tolerance": float(state_leakage_tolerance),
        "hamiltonian_hermiticity_error": float(hermiticity_error),
        "hamiltonian_hermiticity_tolerance": float(hermiticity_tolerance),
        "hamiltonian_physical_subspace_invariance_error": float(
            subspace_invariance_error
        ),
        "hamiltonian_physical_subspace_invariance_tolerance": float(
            subspace_invariance_tolerance
        ),
        "fidelity": float(fidelity),
        "infidelity": float(infidelity),
    }


__all__ = [
    "GROUND_SPACE_FIDELITY_SCHEMA",
    "GROUND_SPACE_TOLERANCE_SCHEMA",
    "GroundSpaceFidelityError",
    "GroundSpaceTolerance",
    "evaluate_ground_space_fidelity",
    "projector_fidelity",
]
