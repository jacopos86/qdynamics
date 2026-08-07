"""Numerically robust solvers for joint active-ansatz and batch models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable

import numpy as np
from scipy.linalg import eigvalsh as generalized_eigvalsh


JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1 = (
    "supported_metric_whitened_eigh_v1"
)
JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2 = (
    "supported_metric_global_trust_eigh_v2"
)
JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1 = (
    "supported_metric_projected_generalized_trust_v1"
)
JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1 = "block_pinv_legacy_v1"
JOINT_LINEAR_SOLVE_POLICIES = frozenset(
    {
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
        JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1,
    }
)


@dataclass(frozen=True)
class JointLinearSolveConfig:
    policy: str = JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
    rank_relative_tolerance: float = 1e-6
    metric_regularization: float = 1e-9
    energy_regularization: float = 1e-9
    max_fubini_study_step: float = 0.25
    global_trust_kkt_residual_accuracy: float = 1e-8
    global_trust_metric_distortion_budget: float = 5e-2

    def __post_init__(self) -> None:
        if str(self.policy) not in JOINT_LINEAR_SOLVE_POLICIES:
            raise ValueError(
                f"policy must be one of {sorted(JOINT_LINEAR_SOLVE_POLICIES)}."
            )
        for name in (
            "rank_relative_tolerance",
            "metric_regularization",
            "energy_regularization",
            "max_fubini_study_step",
            "global_trust_kkt_residual_accuracy",
            "global_trust_metric_distortion_budget",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if float(self.max_fubini_study_step) <= 0.0:
            raise ValueError("max_fubini_study_step must be positive.")
        if float(self.global_trust_kkt_residual_accuracy) <= 0.0:
            raise ValueError(
                "global_trust_kkt_residual_accuracy must be positive."
            )
        if float(self.global_trust_kkt_residual_accuracy) > 1.0:
            raise ValueError(
                "global_trust_kkt_residual_accuracy must not exceed one."
            )
        if float(self.global_trust_metric_distortion_budget) >= 1.0:
            raise ValueError(
                "global_trust_metric_distortion_budget must be less than one."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": str(self.policy),
            "rank_relative_tolerance": float(self.rank_relative_tolerance),
            "metric_regularization": float(self.metric_regularization),
            "energy_regularization": float(self.energy_regularization),
            "max_fubini_study_step": float(self.max_fubini_study_step),
            "global_trust_kkt_residual_accuracy": float(
                self.global_trust_kkt_residual_accuracy
            ),
            "global_trust_metric_distortion_budget": float(
                self.global_trust_metric_distortion_budget
            ),
        }


@dataclass(frozen=True)
class JointLinearSolveResult:
    feasible: bool
    reason: str
    active_parameter_relaxation: np.ndarray
    batch_coordinate_step: np.ndarray
    joint_step: np.ndarray
    predicted_reduction: float
    fubini_study_displacement_sq: float
    trust_lambda: float
    telemetry: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "feasible": bool(self.feasible),
            "reason": str(self.reason),
            "active_parameter_relaxation": [
                float(value) for value in self.active_parameter_relaxation.tolist()
            ],
            "batch_coordinate_step": [
                float(value) for value in self.batch_coordinate_step.tolist()
            ],
            "joint_step": [float(value) for value in self.joint_step.tolist()],
            "predicted_reduction": float(self.predicted_reduction),
            "fubini_study_displacement_sq": float(
                self.fubini_study_displacement_sq
            ),
            "trust_lambda": float(self.trust_lambda),
            **dict(self.telemetry),
        }


@dataclass(frozen=True)
class SupportedMetricWhitening:
    """One supported-metric factorization shared by selector and optimizer routes.

    The support decision is made from the *raw* metric before applying the
    ridge.  ``whitening`` therefore whitens the retained regularized metric,
    while ``raw_orthonormalizer`` whitens the retained raw Fubini--Study
    metric.  ``regularized_to_raw_frame`` is the explicit bridge between those
    two coordinate systems.
    """

    feasible: bool
    reason: str
    raw_metric: np.ndarray
    raw_eigenvalues: np.ndarray
    retained_mask: np.ndarray
    retained_eigenvalues: np.ndarray
    retained_vectors: np.ndarray
    whitening: np.ndarray
    whitening_pseudoinverse: np.ndarray
    raw_orthonormalizer: np.ndarray
    regularized_to_raw_frame: np.ndarray
    raw_whitened_metric: np.ndarray
    regularized_supported_metric: np.ndarray
    raw_metric_pseudoinverse: np.ndarray
    support_threshold: float
    negative_eigenvalue_tolerance: float
    metric_ridge: float
    raw_condition_number: float | None
    retained_condition_number: float | None
    provenance_id: str

    @property
    def dimension(self) -> int:
        return int(self.raw_metric.shape[0])

    @property
    def rank(self) -> int:
        return int(self.retained_eigenvalues.size)

    def lift_whitened_vector(self, value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size != self.rank:
            raise ValueError("whitened vector length must match the supported rank.")
        return np.asarray(self.whitening @ vector, dtype=float)

    def pull_supported_vector(self, value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size != self.dimension:
            raise ValueError("coordinate vector length must match the metric dimension.")
        return np.asarray(self.whitening_pseudoinverse @ vector, dtype=float)

    def pull_covector(self, value: np.ndarray) -> np.ndarray:
        covector = np.asarray(value, dtype=float).reshape(-1)
        if covector.size != self.dimension:
            raise ValueError("coordinate covector length must match the metric dimension.")
        return np.asarray(self.whitening.T @ covector, dtype=float)

    def pull_bilinear(self, value: np.ndarray) -> np.ndarray:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("bilinear form shape must match the metric dimension.")
        return _symmetrize(self.whitening.T @ matrix @ self.whitening)

    def telemetry(self) -> dict[str, Any]:
        raw_in_raw_basis = _symmetrize(
            self.raw_orthonormalizer.T
            @ self.raw_metric
            @ self.raw_orthonormalizer
        )
        raw_in_regularized_basis = _symmetrize(
            self.whitening.T @ self.raw_metric @ self.whitening
        )
        regularized_in_regularized_basis = _symmetrize(
            self.whitening.T
            @ self.regularized_supported_metric
            @ self.whitening
        )
        return {
            "supported_metric_whitening_schema": (
                "supported_metric_whitening_factorization_v1"
            ),
            "supported_metric_whitening_policy": (
                JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
            ),
            "supported_metric_whitening_feasible": bool(self.feasible),
            "supported_metric_whitening_reason": str(self.reason),
            "supported_metric_whitening_provenance_id": str(self.provenance_id),
            "raw_metric_eigenvalues": [
                float(value) for value in self.raw_eigenvalues.tolist()
            ],
            "metric_support_threshold": float(self.support_threshold),
            "metric_retained_mask": [
                bool(value) for value in self.retained_mask.tolist()
            ],
            "retained_metric_eigenvalues": [
                float(value) for value in self.retained_eigenvalues.tolist()
            ],
            "metric_negative_eigenvalue_tolerance": float(
                self.negative_eigenvalue_tolerance
            ),
            "raw_metric_condition_number": self.raw_condition_number,
            "retained_metric_condition_number": self.retained_condition_number,
            "metric_support_rank": int(self.rank),
            "metric_whitening_ridge": float(self.metric_ridge),
            "raw_whitened_metric_eigenvalues": [
                float(value)
                for value in np.linalg.eigvalsh(raw_in_regularized_basis).tolist()
            ],
            "raw_metric_in_raw_orthonormal_basis_eigenvalues": [
                float(value) for value in np.linalg.eigvalsh(raw_in_raw_basis).tolist()
            ],
            "raw_metric_in_regularized_whitened_basis_eigenvalues": [
                float(value)
                for value in np.linalg.eigvalsh(raw_in_regularized_basis).tolist()
            ],
            "regularized_whitened_metric_eigenvalues": (
                [
                    float(value)
                    for value in np.linalg.eigvalsh(
                        regularized_in_regularized_basis
                    ).tolist()
                ]
                if self.feasible
                else []
            ),
            "raw_orthonormal_identity_residual": float(
                np.linalg.norm(raw_in_raw_basis - np.eye(self.rank), ord="fro")
            ),
            "regularized_whitening_identity_residual": float(
                np.linalg.norm(
                    regularized_in_regularized_basis - np.eye(self.rank),
                    ord="fro",
                )
            ),
            "classical_quantum_query_charge": 0,
        }


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    return 0.5 * (arr + arr.T)


def _validated_inputs(
    *,
    gram: np.ndarray,
    hessian: np.ndarray,
    gradient: np.ndarray,
    active_coordinate_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    G = np.asarray(gram, dtype=float)
    H = np.asarray(hessian, dtype=float)
    g = np.asarray(gradient, dtype=float).reshape(-1)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("gram must be a square matrix.")
    if H.shape != G.shape:
        raise ValueError("hessian must have the same square shape as gram.")
    if g.size != G.shape[0]:
        raise ValueError("gradient length must match the joint matrix dimension.")
    active_count = int(active_coordinate_count)
    if active_count < 0 or active_count > g.size:
        raise ValueError("active_coordinate_count is out of range.")
    if g.size == 0:
        raise ValueError("joint model must contain at least one coordinate.")
    if not (
        np.all(np.isfinite(G))
        and np.all(np.isfinite(H))
        and np.all(np.isfinite(g))
    ):
        raise ValueError("joint model contains nonfinite values.")
    return _symmetrize(G), _symmetrize(H), g, active_count


def _empty_result(
    *,
    reason: str,
    dimension: int,
    active_coordinate_count: int,
    telemetry: dict[str, Any],
) -> JointLinearSolveResult:
    step = np.zeros(int(dimension), dtype=float)
    active_count = int(active_coordinate_count)
    return JointLinearSolveResult(
        feasible=False,
        reason=str(reason),
        active_parameter_relaxation=step[:active_count].copy(),
        batch_coordinate_step=step[active_count:].copy(),
        joint_step=step,
        predicted_reduction=0.0,
        fubini_study_displacement_sq=0.0,
        trust_lambda=0.0,
        telemetry={
            "joint_linear_solve_policy_effective": telemetry.get(
                "joint_linear_solve_policy_effective"
            ),
            "classical_quantum_query_charge": 0,
            **dict(telemetry),
        },
    )


def _condition_number_from_positive(values: np.ndarray) -> float | None:
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return None
    return float(np.max(positive) / np.min(positive))


def factor_supported_metric(
    gram: np.ndarray,
    *,
    rank_relative_tolerance: float = 1e-6,
    metric_regularization: float = 1e-9,
) -> SupportedMetricWhitening:
    """Compatibility spelling for the neutral RA support owner."""

    from pipelines.static_adapt.ra_adapt.support import (
        factor_retained_support,
    )

    return factor_retained_support(
        gram,
        rank_relative_tolerance=float(rank_relative_tolerance),
        metric_regularization=float(metric_regularization),
    ).factorization


def _global_trust_support_provenance_id(
    *,
    raw_metric: np.ndarray,
    raw_eigenvalues: np.ndarray,
    retained_mask: np.ndarray,
    retained_vectors: np.ndarray,
    eta_G: float,
    epsilon_G: float,
    eta_KKT: float,
    effective_backward_error_level: float,
    target_condition_number: float,
    metric_ridge: float,
    metric_distortion_budget: float,
    metric_distortion_upper_bound: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"stable_raw_metric_support_factorization_v2\0")
    for array in (
        np.asarray(raw_metric, dtype="<f8"),
        np.asarray(raw_eigenvalues, dtype="<f8"),
        np.asarray(retained_mask, dtype=np.uint8),
        np.asarray(retained_vectors, dtype="<f8"),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes())
        digest.update(b"\0")
    for value in (
        eta_G,
        epsilon_G,
        eta_KKT,
        effective_backward_error_level,
        target_condition_number,
        metric_ridge,
        metric_distortion_budget,
        metric_distortion_upper_bound,
    ):
        digest.update(float(value).hex().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _factor_global_trust_raw_metric_support(
    gram: np.ndarray,
    *,
    eta_G: float,
    eta_KKT: float,
    metric_distortion_budget: float,
) -> tuple[SupportedMetricWhitening | None, str, dict[str, Any]]:
    """Select a stable raw-Gram support cluster for the v2 trust solver.

    ``eta_G`` is a dimensionless relative whitening-error allocation.  It is
    deliberately not multiplied by the largest Gram eigenvalue to create an
    eigenvalue cutoff.  The support interval and gap decisions instead use a
    derived floating-point backward-error radius ``epsilon_G``.  Only after
    that raw support is fixed is the smallest conditioning ridge derived from
    ``eta_KKT / u_eff``.  If that ridge exceeds the metric-distortion budget,
    the resolved raw support may instead be whitened with zero ridge, but only
    as a candidate for the unchanged downstream null-compatibility, propagated
    model-error, and global KKT certificates.  The legacy fixed
    ``metric_regularization`` is never applied by this v2 factorization.
    """

    G = _symmetrize(np.asarray(gram, dtype=float))
    dimension = int(G.shape[0])
    eta = float(eta_G)
    kkt_accuracy = float(eta_KKT)
    distortion_budget = float(metric_distortion_budget)
    unit_roundoff = float(np.finfo(float).eps)
    arithmetic_numerator = float(64.0 * max(1, dimension) * unit_roundoff)
    arithmetic_denominator = float(max(1.0 - arithmetic_numerator, 0.5))
    arithmetic_backward_error_level = float(
        arithmetic_numerator / arithmetic_denominator
    )
    stabilization_base = {
        "metric_stabilization_schema": (
            "derived_kkt_conditioning_ridge_with_zero_ridge_fallback_v2"
        ),
        "metric_stabilization_eta_KKT": kkt_accuracy,
        "metric_stabilization_arithmetic_backward_error_level": (
            arithmetic_backward_error_level
        ),
        "metric_stabilization_distortion_budget": distortion_budget,
        "metric_stabilization_condition_target_formula": (
            "eta_KKT_over_effective_backward_error"
        ),
        "metric_stabilization_lambda_formula": (
            "max_zero_lambda_max_minus_kappa_lambda_min_over_kappa_minus_one"
        ),
        "metric_stabilization_ridge_source": (
            "derived_after_stable_raw_support"
        ),
        "metric_stabilization_fixed_metric_regularization_applied": False,
        "metric_stabilization_legacy_metric_regularization_role": (
            "v1_only_not_applied_by_global_trust_v2"
        ),
    }
    try:
        raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(G)
    except np.linalg.LinAlgError:
        telemetry = {
            "supported_metric_whitening_schema": (
                "stable_raw_metric_support_factorization_v2"
            ),
            "supported_metric_whitening_policy": (
                JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
            ),
            "raw_metric_support_status": "unresolved",
            "raw_metric_support_selection_status": "unresolved",
            "raw_metric_support_reason": "metric_eigendecomposition_failed",
            "raw_metric_support_eta_G": eta,
            "raw_metric_support_epsilon_G": None,
            "raw_metric_support_cluster_gap": None,
            "raw_metric_support_rotation_bound": None,
            "metric_stabilization_status": "unresolved",
            "metric_stabilization_reason": "metric_eigendecomposition_failed",
            "metric_stabilization_effective_backward_error_level": None,
            "metric_stabilization_target_condition_number": None,
            **stabilization_base,
            "classical_quantum_query_charge": 0,
        }
        return None, "metric_eigendecomposition_failed", telemetry

    raw_eigenvalues = np.asarray(raw_eigenvalues, dtype=float)
    raw_eigenvectors = np.asarray(raw_eigenvectors, dtype=float)
    reconstructed = _symmetrize(
        raw_eigenvectors @ np.diag(raw_eigenvalues) @ raw_eigenvectors.T
    )
    gram_norm = float(np.linalg.norm(G, ord=2))
    reconstruction_residual = float(np.linalg.norm(G - reconstructed, ord=2))
    arithmetic_backward_error = float(
        arithmetic_backward_error_level * gram_norm
    )
    epsilon_G = float(reconstruction_residual + arithmetic_backward_error)
    effective_backward_error_level = float(
        epsilon_G / gram_norm
        if gram_norm > np.finfo(float).tiny
        else arithmetic_backward_error_level
    )
    target_condition_number = float(
        kkt_accuracy / effective_backward_error_level
    )
    minimum_eigenvalue = float(raw_eigenvalues[0])
    candidate_checks: list[dict[str, Any]] = []
    common = {
        "supported_metric_whitening_schema": (
            "stable_raw_metric_support_factorization_v2"
        ),
        "supported_metric_whitening_policy": (
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
        ),
        "raw_metric_eigenvalues": [
            float(value) for value in raw_eigenvalues.tolist()
        ],
        "raw_metric_support_eta_G": eta,
        "raw_metric_support_epsilon_G": epsilon_G,
        "raw_gram_eigendecomposition_backward_error": reconstruction_residual,
        "raw_gram_arithmetic_backward_error": arithmetic_backward_error,
        "raw_gram_effective_backward_error_level": (
            effective_backward_error_level
        ),
        "metric_stabilization_effective_backward_error_level": (
            effective_backward_error_level
        ),
        "metric_stabilization_target_condition_number": (
            target_condition_number
        ),
        "raw_metric_minimum_eigenvalue_upper_bound": (
            minimum_eigenvalue + epsilon_G
        ),
        "raw_metric_minimum_eigenvalue_lower_bound": (
            minimum_eigenvalue - epsilon_G
        ),
        "raw_metric_support_selection_semantics": (
            "spectrally_separated_positive_cluster_with_relative_whitening_budget"
        ),
        "raw_metric_support_candidate_checks": candidate_checks,
        **stabilization_base,
        "classical_quantum_query_charge": 0,
    }
    if minimum_eigenvalue + epsilon_G < 0.0:
        telemetry = {
            **common,
            "raw_metric_support_status": "invalid_geometry",
            "raw_metric_support_selection_status": "invalid_geometry",
            "raw_metric_support_reason": "materially_negative_metric_eigenvalue",
            "raw_metric_support_cluster_gap": None,
            "raw_metric_support_rotation_bound": None,
            "metric_stabilization_status": "not_attempted",
            "metric_stabilization_reason": "invalid_raw_metric_geometry",
        }
        return None, "materially_negative_metric_eigenvalue", telemetry

    selected_start: int | None = None
    selected_gap = math.inf
    selected_relative_error = math.inf
    for start in range(dimension):
        minimum_supported = float(raw_eigenvalues[start])
        positive_lower_bound = float(minimum_supported - epsilon_G)
        if start == 0:
            gap = math.inf
        else:
            gap = float(raw_eigenvalues[start] - raw_eigenvalues[start - 1])
        positive = bool(positive_lower_bound > 0.0)
        separated = bool(math.isinf(gap) or gap > 2.0 * epsilon_G)
        relative_error: float | None = (
            float(epsilon_G / positive_lower_bound)
            if positive
            else None
        )
        within_budget = bool(
            relative_error is not None and relative_error <= eta
        )
        accepted = bool(positive and separated and within_budget)
        candidate_checks.append(
            {
                "cluster_start_index": int(start),
                "cluster_rank": int(dimension - start),
                "minimum_supported_eigenvalue": minimum_supported,
                "minimum_supported_lower_bound": positive_lower_bound,
                "boundary_gap": None if math.isinf(gap) else gap,
                "boundary_gap_is_infinite": bool(math.isinf(gap)),
                "gap_exceeds_twice_epsilon_G": separated,
                "relative_whitening_error_bound": relative_error,
                "within_eta_G_budget": within_budget,
                "accepted": accepted,
            }
        )
        if accepted:
            selected_start = int(start)
            selected_gap = float(gap)
            assert relative_error is not None
            selected_relative_error = float(relative_error)
            break

    if selected_start is None:
        telemetry = {
            **common,
            "raw_metric_support_status": "unresolved",
            "raw_metric_support_selection_status": "unresolved",
            "raw_metric_support_reason": "raw_metric_support_unresolved",
            "raw_metric_support_cluster_gap": None,
            "raw_metric_support_rotation_bound": None,
            "raw_metric_support_relative_whitening_error_bound": None,
            "metric_stabilization_status": "not_attempted",
            "metric_stabilization_reason": "raw_metric_support_unresolved",
        }
        return None, "raw_metric_support_unresolved", telemetry

    retained_mask = np.zeros(dimension, dtype=bool)
    retained_mask[selected_start:] = True
    retained_eigenvalues = np.asarray(
        raw_eigenvalues[retained_mask],
        dtype=float,
    )
    retained_vectors = np.asarray(
        raw_eigenvectors[:, retained_mask],
        dtype=float,
    )
    retained_minimum = float(retained_eigenvalues[0])
    retained_maximum = float(retained_eigenvalues[-1])
    raw_supported_condition = float(retained_maximum / retained_minimum)
    selected_support_telemetry = {
        "raw_metric_support_selection_status": "resolved",
        "raw_metric_support_cluster_start_index": selected_start,
        "raw_metric_support_cluster_gap": (
            None if math.isinf(selected_gap) else selected_gap
        ),
        "raw_metric_support_cluster_gap_is_infinite": bool(
            math.isinf(selected_gap)
        ),
        "raw_metric_support_relative_whitening_error_bound": (
            selected_relative_error
        ),
        "raw_metric_support_minimum_eigenvalue_lower_bound": float(
            retained_minimum - epsilon_G
        ),
        "metric_support_rank": int(retained_eigenvalues.size),
        "metric_retained_mask": [
            bool(value) for value in retained_mask.tolist()
        ],
        "retained_metric_eigenvalues": [
            float(value) for value in retained_eigenvalues.tolist()
        ],
        "metric_stabilization_raw_supported_minimum_eigenvalue": (
            retained_minimum
        ),
        "metric_stabilization_raw_supported_maximum_eigenvalue": (
            retained_maximum
        ),
        "metric_stabilization_raw_supported_condition_number": (
            raw_supported_condition
        ),
    }
    if (
        not math.isfinite(target_condition_number)
        or target_condition_number <= 1.0
    ):
        telemetry = {
            **common,
            **selected_support_telemetry,
            "raw_metric_support_status": "unresolved",
            "raw_metric_support_reason": "kkt_accuracy_below_arithmetic_floor",
            "raw_metric_support_rotation_bound": None,
            "metric_stabilization_status": "unresolved",
            "metric_stabilization_reason": (
                "kkt_accuracy_below_arithmetic_floor"
            ),
            "metric_stabilization_lambda_G": None,
            "metric_stabilization_stabilized_condition_number": None,
            "metric_stabilization_distortion_lower_bound": None,
            "metric_stabilization_distortion_upper_bound": None,
            "metric_stabilization_distortion_within_budget": False,
        }
        return None, "kkt_accuracy_below_arithmetic_floor", telemetry

    proposed_ridge = float(
        max(
            0.0,
            (
                retained_maximum
                - target_condition_number * retained_minimum
            )
            / (target_condition_number - 1.0),
        )
    )
    proposed_stabilized_condition = float(
        (retained_maximum + proposed_ridge)
        / (retained_minimum + proposed_ridge)
    )
    proposed_distortion_lower_bound = float(
        proposed_ridge / (retained_maximum + proposed_ridge)
    )
    proposed_distortion_upper_bound = float(
        proposed_ridge / (retained_minimum + proposed_ridge)
    )
    proposed_condition_target_met = bool(
        proposed_stabilized_condition
        <= target_condition_number
        * (1.0 + 16.0 * np.finfo(float).eps)
    )
    proposed_stabilization_telemetry = {
        "metric_stabilization_proposed_lambda_G": proposed_ridge,
        "metric_stabilization_proposed_condition_number": (
            proposed_stabilized_condition
        ),
        "metric_stabilization_proposed_condition_target_met": (
            proposed_condition_target_met
        ),
        "metric_stabilization_proposed_distortion_lower_bound": (
            proposed_distortion_lower_bound
        ),
        "metric_stabilization_proposed_distortion_upper_bound": (
            proposed_distortion_upper_bound
        ),
        "metric_stabilization_proposed_distortion_within_budget": bool(
            proposed_distortion_upper_bound <= distortion_budget
        ),
    }
    if (
        not math.isfinite(proposed_ridge)
        or not math.isfinite(proposed_stabilized_condition)
        or not proposed_condition_target_met
    ):
        telemetry = {
            **common,
            **selected_support_telemetry,
            **proposed_stabilization_telemetry,
            "raw_metric_support_status": "unresolved",
            "raw_metric_support_reason": "metric_stabilization_unresolved",
            "raw_metric_support_rotation_bound": None,
            "metric_stabilization_status": "unresolved",
            "metric_stabilization_reason": "metric_stabilization_unresolved",
            "metric_stabilization_zero_ridge_fallback_eligible": False,
            "metric_stabilization_zero_ridge_fallback_attempted": False,
        }
        return None, "metric_stabilization_unresolved", telemetry

    zero_ridge_fallback = bool(
        proposed_distortion_upper_bound > distortion_budget
    )
    ridge = 0.0 if zero_ridge_fallback else proposed_ridge
    denominators = np.asarray(retained_eigenvalues + ridge, dtype=float)
    stabilized_condition = float(
        (retained_maximum + ridge) / (retained_minimum + ridge)
    )
    distortion_lower_bound = float(ridge / (retained_maximum + ridge))
    distortion_upper_bound = float(ridge / (retained_minimum + ridge))
    stabilization_telemetry = {
        **proposed_stabilization_telemetry,
        "metric_stabilization_lambda_G": ridge,
        "metric_stabilization_stabilized_condition_number": (
            stabilized_condition
        ),
        "metric_stabilization_condition_target_met": bool(
            stabilized_condition
            <= target_condition_number
            * (1.0 + 16.0 * np.finfo(float).eps)
        ),
        "metric_stabilization_distortion_definition": (
            "spectral_bounds_of_lambda_G_over_raw_eigenvalue_plus_lambda_G"
        ),
        "metric_stabilization_distortion_lower_bound": (
            distortion_lower_bound
        ),
        "metric_stabilization_distortion_upper_bound": (
            distortion_upper_bound
        ),
        "metric_stabilization_distortion_within_budget": bool(
            distortion_upper_bound <= distortion_budget
        ),
        "metric_stabilization_zero_ridge_fallback_eligible": (
            zero_ridge_fallback
        ),
        "metric_stabilization_zero_ridge_fallback_attempted": False,
        "metric_stabilization_zero_ridge_fallback_trigger": (
            "metric_stabilization_distortion_budget_exceeded"
            if zero_ridge_fallback
            else None
        ),
        "metric_stabilization_zero_ridge_fallback_schema": (
            "resolved_raw_support_aposteriori_certificate_v1"
        ),
        "metric_stabilization_zero_ridge_fallback_acceptance_contract": (
            "raw_null_compatibility_then_unchanged_model_and_global_kkt_certificates"
        ),
        "metric_stabilization_zero_ridge_fallback_condition_target_role": (
            "proposed_ridge_sufficient_precondition_not_zero_ridge_acceptance"
        ),
        "metric_stabilization_zero_ridge_fallback_solver_status": (
            "awaiting_raw_metric_null_compatibility"
            if zero_ridge_fallback
            else "not_applicable"
        ),
    }
    if np.any(denominators <= 0.0):
        telemetry = {
            **common,
            **selected_support_telemetry,
            **stabilization_telemetry,
            "raw_metric_support_status": "unresolved",
            "raw_metric_support_reason": "nonpositive_whitening_denominator",
            "raw_metric_support_rotation_bound": None,
            "metric_stabilization_status": "unresolved",
            "metric_stabilization_reason": "nonpositive_whitening_denominator",
        }
        return None, "nonpositive_whitening_denominator", telemetry

    whitening = retained_vectors @ np.diag(denominators ** -0.5)
    whitening_pseudoinverse = (
        np.diag(denominators ** 0.5) @ retained_vectors.T
    )
    raw_orthonormalizer = retained_vectors @ np.diag(
        retained_eigenvalues ** -0.5
    )
    regularized_to_raw_frame = np.diag(
        np.sqrt(retained_eigenvalues / denominators)
    )
    raw_whitened_metric = _symmetrize(whitening.T @ G @ whitening)
    regularized_supported_metric = _symmetrize(
        retained_vectors @ np.diag(denominators) @ retained_vectors.T
    )
    raw_metric_pseudoinverse = _symmetrize(
        retained_vectors
        @ np.diag(retained_eigenvalues ** -1.0)
        @ retained_vectors.T
    )
    raw_positive = np.asarray(
        raw_eigenvalues[raw_eigenvalues > epsilon_G],
        dtype=float,
    )
    provenance_id = _global_trust_support_provenance_id(
        raw_metric=G,
        raw_eigenvalues=raw_eigenvalues,
        retained_mask=retained_mask,
        retained_vectors=retained_vectors,
        eta_G=eta,
        epsilon_G=epsilon_G,
        eta_KKT=kkt_accuracy,
        effective_backward_error_level=effective_backward_error_level,
        target_condition_number=target_condition_number,
        metric_ridge=ridge,
        metric_distortion_budget=distortion_budget,
        metric_distortion_upper_bound=distortion_upper_bound,
    )
    factorization = SupportedMetricWhitening(
        feasible=True,
        reason="stable_positive_raw_metric_support",
        raw_metric=G.copy(),
        raw_eigenvalues=raw_eigenvalues.copy(),
        retained_mask=retained_mask.copy(),
        retained_eigenvalues=retained_eigenvalues.copy(),
        retained_vectors=retained_vectors.copy(),
        whitening=np.asarray(whitening, dtype=float).copy(),
        whitening_pseudoinverse=np.asarray(
            whitening_pseudoinverse, dtype=float
        ).copy(),
        raw_orthonormalizer=np.asarray(
            raw_orthonormalizer, dtype=float
        ).copy(),
        regularized_to_raw_frame=np.asarray(
            regularized_to_raw_frame, dtype=float
        ).copy(),
        raw_whitened_metric=np.asarray(raw_whitened_metric, dtype=float).copy(),
        regularized_supported_metric=np.asarray(
            regularized_supported_metric, dtype=float
        ).copy(),
        raw_metric_pseudoinverse=np.asarray(
            raw_metric_pseudoinverse, dtype=float
        ).copy(),
        support_threshold=epsilon_G,
        negative_eigenvalue_tolerance=epsilon_G,
        metric_ridge=ridge,
        raw_condition_number=_condition_number_from_positive(raw_positive),
        retained_condition_number=_condition_number_from_positive(
            retained_eigenvalues
        ),
        provenance_id=provenance_id,
    )
    rotation_bound = (
        0.0
        if math.isinf(selected_gap)
        else float(epsilon_G / (selected_gap - epsilon_G))
    )
    telemetry = {
        **factorization.telemetry(),
        **common,
        "supported_metric_whitening_schema": (
            "stable_raw_metric_support_factorization_v2"
        ),
        "supported_metric_whitening_policy": (
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
        ),
        "supported_metric_whitening_reason": factorization.reason,
        "raw_metric_support_status": "resolved",
        "raw_metric_support_reason": "stable_positive_raw_metric_support",
        **selected_support_telemetry,
        **stabilization_telemetry,
        "metric_stabilization_status": (
            "pending_end_to_end_certificate"
            if zero_ridge_fallback
            else "resolved"
        ),
        "metric_stabilization_reason": (
            "zero_ridge_fallback_candidate_after_distortion_breach"
            if zero_ridge_fallback
            else "derived_minimum_conditioning_ridge"
        ),
        "raw_metric_support_rotation_bound": rotation_bound,
        "metric_support_selection_is_relative_budget_not_cutoff": True,
    }
    return factorization, factorization.reason, telemetry


def _solve_supported_metric_projected_generalized_trust(
    *,
    G: np.ndarray,
    H: np.ndarray,
    g: np.ndarray,
    active_count: int,
    config: JointLinearSolveConfig,
) -> JointLinearSolveResult:
    """Solve directly in the supported raw-metric eigenspace.

    This policy deliberately stops after support projection.  It never forms
    a Gram inverse, pseudoinverse, or inverse square root.  The retained raw
    eigenvalues remain the trust metric in the generalized KKT system

        (H_s + lambda Lambda_s) q = g_s,

    and the lifted physical step is ``V_s q``.  This keeps null-direction
    removal separate from the supported-FS whitening used by the subsequent
    accepted-ansatz Powell refit.
    """

    dimension = int(g.size)
    policy = JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
    base: dict[str, Any] = {
        "joint_linear_solve_policy_requested": str(config.policy),
        "joint_linear_solve_policy_effective": policy,
        "joint_coordinate_count": dimension,
        "active_coordinate_count": int(active_count),
        "batch_coordinate_count": int(dimension - active_count),
        "supported_metric_projection_schema": (
            "raw_metric_supported_eigenspace_projection_v1"
        ),
        "supported_metric_projection_active": True,
        "supported_metric_whitening_active": False,
        "supported_metric_inverse_sqrt_constructed": False,
        "supported_metric_inverse_constructed": False,
        "metric_regularization_applied": False,
        "metric_regularization_configured_inactive": float(
            config.metric_regularization
        ),
        "trust_metric_coordinate_system": "raw_supported_metric_eigenbasis",
        "classical_quantum_query_charge": 0,
    }

    import pipelines.static_adapt.ra_adapt.support as ra_support

    retained_support = ra_support.factor_retained_support(
        G,
        rank_relative_tolerance=float(config.rank_relative_tolerance),
        # Phase III projects the raw Gram and never applies a metric ridge.
        metric_regularization=0.0,
        source_provenance_id="phase3_projected_source_gram_v1",
    )
    factorization = retained_support.factorization
    support_receipt = retained_support.receipt
    raw_eigenvalues = np.asarray(
        factorization.raw_eigenvalues,
        dtype=float,
    )
    retained_mask = np.asarray(
        factorization.retained_mask,
        dtype=bool,
    )
    retained_eigenvalues = np.asarray(
        factorization.retained_eigenvalues,
        dtype=float,
    )
    retained_vectors = np.asarray(
        factorization.retained_vectors,
        dtype=float,
    )
    negative_tolerance = float(
        factorization.negative_eigenvalue_tolerance
    )
    support_threshold = float(factorization.support_threshold)
    projection_provenance_id = str(
        support_receipt.factorization_provenance_id
    )
    base.update(
        {
            "retained_support_receipt": support_receipt.as_dict(),
            "raw_metric_eigenvalues": [
                float(value) for value in raw_eigenvalues.tolist()
            ],
            "metric_support_threshold": float(support_threshold),
            "metric_retained_mask": [
                bool(value) for value in retained_mask.tolist()
            ],
            "retained_metric_eigenvalues": [
                float(value) for value in retained_eigenvalues.tolist()
            ],
            "metric_negative_eigenvalue_tolerance": float(
                negative_tolerance
            ),
            "metric_support_rank": int(retained_eigenvalues.size),
            "supported_metric_projection_provenance_id": str(
                projection_provenance_id
            ),
            "raw_metric_condition_number": (
                factorization.raw_condition_number
            ),
            "retained_metric_condition_number": (
                factorization.retained_condition_number
            ),
        }
    )
    if not factorization.feasible:
        return _empty_result(
            reason=str(factorization.reason),
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=base,
        )

    V_s = retained_vectors
    Lambda_s = np.diag(retained_eigenvalues)
    H_s = _symmetrize(V_s.T @ H @ V_s)
    g_s = np.asarray(V_s.T @ g, dtype=float)
    try:
        generalized_curvatures = np.asarray(
            generalized_eigvalsh(H_s, Lambda_s, check_finite=False),
            dtype=float,
        )
    except (ValueError, np.linalg.LinAlgError):
        return _empty_result(
            reason="supported_generalized_hessian_eigendecomposition_failed",
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=base,
        )
    curvature_scale = float(
        max(np.max(np.abs(generalized_curvatures)), np.finfo(float).tiny)
    )
    denominator_floor = float(
        max(
            float(config.energy_regularization),
            64.0
            * np.finfo(float).eps
            * max(1, retained_eigenvalues.size)
            * curvature_scale,
            np.finfo(float).tiny,
        )
    )
    curvature_shift = float(
        max(0.0, denominator_floor - float(np.min(generalized_curvatures)))
    )
    radius = float(config.max_fubini_study_step)
    radius_sq = float(radius * radius)
    trust_tolerance = 1e-10
    discarded_gradient = np.asarray(g - V_s @ g_s, dtype=float)

    def solve_at(trust_lambda: float) -> dict[str, Any] | None:
        system = _symmetrize(H_s + float(trust_lambda) * Lambda_s)
        try:
            q = np.asarray(np.linalg.solve(system, g_s), dtype=float)
        except np.linalg.LinAlgError:
            return None
        z = np.asarray(V_s @ q, dtype=float)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(z))):
            return None
        supported_displacement_sq = float(q.T @ Lambda_s @ q)
        displacement_sq = float(max(0.0, z.T @ G @ z))
        predicted_reduction = float(g.T @ z - 0.5 * z.T @ H @ z)
        supported_residual_vector = np.asarray(
            system @ q - g_s, dtype=float
        )
        full_residual_vector = np.asarray(
            (H + float(trust_lambda) * G) @ z - g, dtype=float
        )
        return {
            "trust_lambda": float(trust_lambda),
            "q": q,
            "z": z,
            "supported_displacement_sq": float(supported_displacement_sq),
            "displacement_sq": float(displacement_sq),
            "predicted_reduction": float(predicted_reduction),
            "supported_kkt_residual": float(
                np.linalg.norm(supported_residual_vector)
            ),
            "full_direct_residual": float(np.linalg.norm(full_residual_vector)),
        }

    def trust_feasible(solution: dict[str, Any] | None) -> bool:
        return bool(
            solution is not None
            and math.isfinite(float(solution["predicted_reduction"]))
            and float(solution["supported_displacement_sq"])
            <= radius_sq * (1.0 + trust_tolerance)
            and float(solution["displacement_sq"])
            <= radius_sq * (1.0 + trust_tolerance)
        )

    lower = float(curvature_shift)
    applied = solve_at(lower)
    trust_radius_clipped = False
    bracket_iterations = 0
    bisection_iterations = 0
    if not trust_feasible(applied):
        high = float(max(lower, denominator_floor))
        if high <= lower:
            high = float(lower + denominator_floor)
        feasible_high: dict[str, Any] | None = None
        for bracket_iterations in range(1, 81):
            trial = solve_at(high)
            if trust_feasible(trial):
                feasible_high = trial
                break
            high = float(2.0 * high + denominator_floor)
        if feasible_high is None:
            return _empty_result(
                reason="supported_generalized_trust_bracket_failed",
                dimension=dimension,
                active_coordinate_count=active_count,
                telemetry={
                    **base,
                    "supported_generalized_hessian_eigenvalues": [
                        float(value) for value in generalized_curvatures.tolist()
                    ],
                    "denominator_floor": float(denominator_floor),
                    "curvature_shift": float(curvature_shift),
                },
            )
        low = float(lower)
        for bisection_iterations in range(1, 65):
            midpoint = float(0.5 * (low + high))
            trial = solve_at(midpoint)
            if trust_feasible(trial):
                high = midpoint
                feasible_high = trial
            else:
                low = midpoint
        applied = feasible_high
        trust_radius_clipped = True

    if applied is None:
        return _empty_result(
            reason="supported_generalized_trust_solve_failed",
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=base,
        )
    predicted_reduction_raw = float(applied["predicted_reduction"])
    prediction_tolerance = float(
        max(
            float(config.energy_regularization),
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(predicted_reduction_raw)),
        )
    )
    if predicted_reduction_raw < -prediction_tolerance:
        return _empty_result(
            reason="negative_predicted_reduction",
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry={
                **base,
                "applied_predicted_reduction_raw": predicted_reduction_raw,
            },
        )
    predicted_reduction = float(max(0.0, predicted_reduction_raw))
    step = np.asarray(applied["z"], dtype=float)
    total_metric_multiplier_mu = float(applied["trust_lambda"])
    trust_boundary_multiplier_lambda = float(
        max(0.0, total_metric_multiplier_mu - curvature_shift)
    )
    trust_boundary_active = bool(trust_boundary_multiplier_lambda > 0.0)
    trust_radius_binding = bool(
        trust_radius_clipped
        and abs(float(applied["supported_displacement_sq"]) - radius_sq)
        <= max(1e-12, radius_sq * 1e-8)
    )
    telemetry = {
        **base,
        "supported_generalized_hessian_eigenvalues": [
            float(value) for value in generalized_curvatures.tolist()
        ],
        "supported_metric_diagonal": [
            float(value) for value in retained_eigenvalues.tolist()
        ],
        "curvature_shift": float(curvature_shift),
        # ``trust_lambda`` is retained as the historical spelling for the
        # total metric multiplier mu.  Canonical RA-ADAPT receipts distinguish
        # curvature stabilization kappa from the trust-boundary increment
        # lambda in (H_s + (kappa + lambda) Lambda_s) q = g_s.
        "trust_lambda": float(total_metric_multiplier_mu),
        "kappa_stabilization_shift": float(curvature_shift),
        "trust_boundary_multiplier_lambda": float(
            trust_boundary_multiplier_lambda
        ),
        "total_metric_multiplier_mu": float(total_metric_multiplier_mu),
        "trust_boundary_active": bool(trust_boundary_active),
        "curvature_stabilization_applied": bool(curvature_shift > 0.0),
        "legacy_total_metric_regularization_applied": bool(
            total_metric_multiplier_mu > 0.0
        ),
        "denominator_floor": float(denominator_floor),
        "supported_coordinate_step_norm": float(
            np.linalg.norm(np.asarray(applied["q"], dtype=float))
        ),
        "supported_metric_displacement_sq": float(
            applied["supported_displacement_sq"]
        ),
        "joint_fubini_study_displacement_sq": float(
            applied["displacement_sq"]
        ),
        "full_direct_residual": float(applied["full_direct_residual"]),
        "supported_generalized_kkt_residual": float(
            applied["supported_kkt_residual"]
        ),
        "discarded_gradient_norm": float(np.linalg.norm(discarded_gradient)),
        "trust_regularization_applied": bool(trust_boundary_active),
        "trust_clipped": bool(trust_radius_clipped),
        "trust_radius_binding": bool(trust_radius_binding),
        "trust_bracket_iterations": int(bracket_iterations),
        "trust_bisection_iterations": int(bisection_iterations),
        "active_parameter_relaxation": [
            float(value) for value in step[:active_count].tolist()
        ],
        "batch_coordinate_step": [
            float(value) for value in step[active_count:].tolist()
        ],
        "applied_predicted_reduction": float(predicted_reduction),
    }
    return JointLinearSolveResult(
        feasible=True,
        reason="supported_metric_projected_generalized_trust_solve",
        active_parameter_relaxation=step[:active_count].copy(),
        batch_coordinate_step=step[active_count:].copy(),
        joint_step=step.copy(),
        predicted_reduction=float(predicted_reduction),
        fubini_study_displacement_sq=float(applied["displacement_sq"]),
        trust_lambda=float(total_metric_multiplier_mu),
        telemetry=telemetry,
    )


def _solve_supported_metric_whitened(
    *,
    G: np.ndarray,
    H: np.ndarray,
    g: np.ndarray,
    active_count: int,
    config: JointLinearSolveConfig,
) -> JointLinearSolveResult:
    dimension = int(g.size)
    policy = JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
    base: dict[str, Any] = {
        "joint_linear_solve_policy_requested": str(config.policy),
        "joint_linear_solve_policy_effective": policy,
        "joint_coordinate_count": dimension,
        "active_coordinate_count": int(active_count),
        "batch_coordinate_count": int(dimension - active_count),
        "classical_quantum_query_charge": 0,
    }
    factorization = factor_supported_metric(
        G,
        rank_relative_tolerance=float(config.rank_relative_tolerance),
        metric_regularization=float(config.metric_regularization),
    )
    base.update(factorization.telemetry())
    if not factorization.feasible:
        return _empty_result(
            reason=str(factorization.reason),
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=base,
        )
    metric_eigenvalues = factorization.raw_eigenvalues
    retained_eigenvalues = factorization.retained_eigenvalues
    retained_vectors = factorization.retained_vectors
    whitening_denominators = retained_eigenvalues + float(
        config.metric_regularization
    )
    W = factorization.whitening
    H_w = _symmetrize(W.T @ H @ W)
    g_w = np.asarray(W.T @ g, dtype=float)
    try:
        hessian_eigenvalues, hessian_eigenvectors = np.linalg.eigh(H_w)
    except np.linalg.LinAlgError:
        return _empty_result(
            reason="whitened_hessian_eigendecomposition_failed",
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=base,
        )

    hessian_scale = float(
        max(np.max(np.abs(hessian_eigenvalues)), np.finfo(float).tiny)
    )
    denominator_floor = float(
        max(
            float(config.energy_regularization),
            64.0
            * np.finfo(float).eps
            * max(1, retained_eigenvalues.size)
            * hessian_scale,
            np.finfo(float).tiny,
        )
    )
    curvature_shift = float(
        max(0.0, denominator_floor - float(np.min(hessian_eigenvalues)))
    )
    gradient_eigenbasis = np.asarray(hessian_eigenvectors.T @ g_w, dtype=float)
    radius = float(config.max_fubini_study_step)
    radius_sq = float(radius * radius)
    trust_tolerance = 1e-10

    def solve_at(trust_lambda: float) -> dict[str, Any] | None:
        denominators = np.asarray(
            hessian_eigenvalues + float(trust_lambda),
            dtype=float,
        )
        if np.any(denominators < denominator_floor * (1.0 - 1e-12)):
            return None
        x_eigenbasis = gradient_eigenbasis / denominators
        x = np.asarray(hessian_eigenvectors @ x_eigenbasis, dtype=float)
        z = np.asarray(W @ x, dtype=float)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
            return None
        whitened_norm = float(np.linalg.norm(x))
        displacement_sq = float(max(0.0, z.T @ G @ z))
        predicted_reduction = float(g.T @ z - 0.5 * z.T @ H @ z)
        original_residual_vector = np.asarray(
            (H + float(trust_lambda) * G) @ z - g,
            dtype=float,
        )
        supported_residual = float(np.linalg.norm(W.T @ original_residual_vector))
        discarded_gradient = np.asarray(
            g - retained_vectors @ (retained_vectors.T @ g),
            dtype=float,
        )
        solved_residual = float(
            np.linalg.norm(
                (H_w + float(trust_lambda) * np.eye(H_w.shape[0])) @ x - g_w
            )
        )
        return {
            "trust_lambda": float(trust_lambda),
            "denominators": denominators,
            "x": x,
            "z": z,
            "whitened_norm": float(whitened_norm),
            "displacement_sq": float(displacement_sq),
            "predicted_reduction": float(predicted_reduction),
            "full_residual": float(np.linalg.norm(original_residual_vector)),
            "supported_residual": float(supported_residual),
            "discarded_gradient_norm": float(np.linalg.norm(discarded_gradient)),
            "whitened_solve_residual": float(solved_residual),
        }

    def trust_feasible(solution: dict[str, Any] | None) -> bool:
        return bool(
            solution is not None
            and math.isfinite(float(solution["predicted_reduction"]))
            and float(solution["whitened_norm"])
            <= radius * (1.0 + trust_tolerance)
            and float(solution["displacement_sq"])
            <= radius_sq * (1.0 + trust_tolerance)
        )

    lower = float(curvature_shift)
    applied = solve_at(lower)
    trust_radius_clipped = False
    bracket_iterations = 0
    bisection_iterations = 0
    if not trust_feasible(applied):
        high = float(max(lower, denominator_floor))
        if high <= lower:
            high = float(lower + denominator_floor)
        feasible_high: dict[str, Any] | None = None
        for bracket_iterations in range(1, 81):
            trial = solve_at(high)
            if trust_feasible(trial):
                feasible_high = trial
                break
            high = float(2.0 * high + denominator_floor)
        if feasible_high is None:
            return _empty_result(
                reason="whitened_trust_bracket_failed",
                dimension=dimension,
                active_coordinate_count=active_count,
                telemetry={
                    **base,
                    "H_w_eigenvalues": [
                        float(value) for value in hessian_eigenvalues.tolist()
                    ],
                    "denominator_floor": float(denominator_floor),
                    "curvature_shift": float(curvature_shift),
                },
            )
        low = float(lower)
        for bisection_iterations in range(1, 65):
            midpoint = float(0.5 * (low + high))
            trial = solve_at(midpoint)
            if trust_feasible(trial):
                high = midpoint
                feasible_high = trial
            else:
                low = midpoint
        applied = feasible_high
        trust_radius_clipped = True

    if applied is None:
        return _empty_result(
            reason="whitened_trust_solve_failed",
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=base,
        )
    predicted_reduction = float(applied["predicted_reduction"])
    prediction_tolerance = float(
        max(
            float(config.energy_regularization),
            64.0 * np.finfo(float).eps * max(1.0, abs(predicted_reduction)),
        )
    )
    if predicted_reduction < -prediction_tolerance:
        return _empty_result(
            reason="negative_predicted_reduction",
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry={
                **base,
                "applied_predicted_reduction_raw": float(predicted_reduction),
            },
        )
    predicted_reduction = float(max(0.0, predicted_reduction))
    step = np.asarray(applied["z"], dtype=float)
    applied_denominators = np.asarray(applied["denominators"], dtype=float)
    applied_condition = float(
        np.max(applied_denominators) / np.min(applied_denominators)
    )
    trust_radius_binding = bool(
        trust_radius_clipped
        and abs(float(applied["whitened_norm"]) - radius)
        <= max(1e-12, radius * 1e-8)
    )
    telemetry = {
        **base,
        "whitening_denominators": [
            float(value) for value in whitening_denominators.tolist()
        ],
        "whitened_metric_eigenvalues": [
            float(value)
            for value in np.linalg.eigvalsh(_symmetrize(W.T @ G @ W)).tolist()
        ],
        "raw_metric_in_regularized_whitened_basis_eigenvalues": [
            float(value)
            for value in np.linalg.eigvalsh(_symmetrize(W.T @ G @ W)).tolist()
        ],
        "H_w_eigenvalues": [
            float(value) for value in hessian_eigenvalues.tolist()
        ],
        "curvature_shift": float(curvature_shift),
        "trust_lambda": float(applied["trust_lambda"]),
        "denominator_floor": float(denominator_floor),
        "applied_whitened_condition_number": float(applied_condition),
        "whitened_step_norm": float(applied["whitened_norm"]),
        "joint_fubini_study_displacement_sq": float(applied["displacement_sq"]),
        "full_direct_residual": float(applied["full_residual"]),
        "supported_direct_residual": float(applied["supported_residual"]),
        "whitened_solve_residual": float(applied["whitened_solve_residual"]),
        "discarded_gradient_norm": float(applied["discarded_gradient_norm"]),
        "trust_regularization_applied": bool(float(applied["trust_lambda"]) > 0.0),
        "trust_clipped": bool(trust_radius_clipped),
        "trust_radius_binding": bool(trust_radius_binding),
        "trust_bracket_iterations": int(bracket_iterations),
        "trust_bisection_iterations": int(bisection_iterations),
        "active_parameter_relaxation": [
            float(value) for value in step[:active_count].tolist()
        ],
        "batch_coordinate_step": [
            float(value) for value in step[active_count:].tolist()
        ],
        "applied_predicted_reduction": float(predicted_reduction),
    }
    return JointLinearSolveResult(
        feasible=True,
        reason="supported_metric_whitened_eigh_solve",
        active_parameter_relaxation=step[:active_count].copy(),
        batch_coordinate_step=step[active_count:].copy(),
        joint_step=step.copy(),
        predicted_reduction=float(predicted_reduction),
        fubini_study_displacement_sq=float(applied["displacement_sq"]),
        trust_lambda=float(applied["trust_lambda"]),
        telemetry=telemetry,
    )


def _canonical_eigenspace_direction(eigenspace: np.ndarray) -> np.ndarray:
    """Choose a basis-invariant deterministic unit vector in an eigenspace."""

    vectors = np.asarray(eigenspace, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("eigenspace must contain at least one vector.")
    projector = _symmetrize(vectors @ vectors.T)
    anchor = int(np.argmax(np.clip(np.diag(projector), 0.0, None)))
    direction = np.asarray(projector[:, anchor], dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= np.finfo(float).tiny:
        direction = np.asarray(vectors[:, 0], dtype=float)
        direction_norm = float(np.linalg.norm(direction))
    direction = np.asarray(direction / direction_norm, dtype=float)
    orientation_anchor = int(np.argmax(np.abs(direction)))
    if float(direction[orientation_anchor]) < 0.0:
        direction = -direction
    return direction


def _maximize_quadratic_on_sphere(
    *,
    quadratic: np.ndarray,
    linear: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Globally maximize ``y.T Q y + 2 q.T y`` on a Euclidean sphere."""

    Q = _symmetrize(np.asarray(quadratic, dtype=float))
    q = np.asarray(linear, dtype=float).reshape(-1)
    sphere_radius = float(radius)
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    maximum_eigenvalue = float(eigenvalues[-1])
    scale = float(max(1.0, np.max(np.abs(eigenvalues)), np.linalg.norm(q)))
    tolerance = float(
        256.0 * np.finfo(float).eps * max(1, q.size) * scale
    )
    maximum_mask = np.asarray(
        np.abs(eigenvalues - maximum_eigenvalue) <= tolerance,
        dtype=bool,
    )
    coefficients = np.asarray(eigenvectors.T @ q, dtype=float)
    maximum_gradient_norm = float(np.linalg.norm(coefficients[maximum_mask]))
    lower_coefficients = np.zeros(q.size, dtype=float)
    nonmaximum_mask = ~maximum_mask
    lower_coefficients[nonmaximum_mask] = (
        coefficients[nonmaximum_mask]
        / (maximum_eigenvalue - eigenvalues[nonmaximum_mask])
    )
    lower_solution = np.asarray(eigenvectors @ lower_coefficients, dtype=float)
    lower_norm = float(np.linalg.norm(lower_solution))
    iterations = 0
    solution_case = "sphere_secular_root"

    if maximum_gradient_norm <= tolerance and lower_norm <= sphere_radius + tolerance:
        maximum_eigenspace = np.asarray(
            eigenvectors[:, maximum_mask],
            dtype=float,
        )
        completion_direction = _canonical_eigenspace_direction(
            maximum_eigenspace
        )
        completion_radius = float(
            math.sqrt(max(0.0, sphere_radius**2 - lower_norm**2))
        )
        solution = np.asarray(
            lower_solution + completion_radius * completion_direction,
            dtype=float,
        )
        solution_case = "sphere_hard_case"
    else:
        def regular_solution(multiplier: float) -> np.ndarray | None:
            denominators = np.asarray(multiplier - eigenvalues, dtype=float)
            if np.any(denominators <= 0.0):
                return None
            value = np.asarray(
                eigenvectors @ (coefficients / denominators),
                dtype=float,
            )
            return value if np.all(np.isfinite(value)) else None

        representable_delta = float(
            np.nextafter(maximum_eigenvalue, math.inf) - maximum_eigenvalue
        )
        high_delta = float(
            max(
                2.0 * np.linalg.norm(coefficients) / sphere_radius,
                tolerance,
                representable_delta,
            )
        )
        high = float(maximum_eigenvalue + high_delta)
        high_solution = regular_solution(high)
        for _ in range(80):
            if (
                high_solution is not None
                and float(np.linalg.norm(high_solution)) <= sphere_radius
            ):
                break
            high_delta = float(2.0 * high_delta)
            high = float(maximum_eigenvalue + high_delta)
            high_solution = regular_solution(high)
        else:
            raise ArithmeticError("quotient orientation sphere bracket failed")
        low = float(maximum_eigenvalue)
        for iterations in range(1, 161):
            midpoint = float(0.5 * (low + high))
            if midpoint <= low or midpoint >= high:
                break
            midpoint_solution = regular_solution(midpoint)
            if (
                midpoint_solution is None
                or float(np.linalg.norm(midpoint_solution)) > sphere_radius
            ):
                low = midpoint
            else:
                high = midpoint
                high_solution = midpoint_solution
        assert high_solution is not None
        solution = np.asarray(high_solution, dtype=float)
        solution_norm = float(np.linalg.norm(solution))
        if solution_norm > 0.0:
            solution = np.asarray(
                solution * (sphere_radius / solution_norm),
                dtype=float,
            )

    objective = float(solution.T @ Q @ solution + 2.0 * q.T @ solution)
    return solution, {
        "solution_case": solution_case,
        "iterations": int(iterations),
        "maximum_eigenvalue": maximum_eigenvalue,
        "maximum_eigenspace_dimension": int(np.count_nonzero(maximum_mask)),
        "maximum_eigenspace_gradient_norm": maximum_gradient_norm,
        "objective": objective,
    }


def _candidate_quotient_hard_case_direction(
    *,
    factorization: SupportedMetricWhitening,
    active_coordinate_count: int,
    minimum_eigenspace: np.ndarray,
    particular_solution: np.ndarray,
    completion_radius: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Orient a degenerate hard case by maximal physical quotient fraction.

    The quotient construction is subordinate to the already certified raw-
    metric support.  Its active-block support decision reuses that
    factorization's backward-error radius and provenance.  If the induced
    active support is not spectrally separated, or if the Schur quotient is
    materially non-PSD, the orientation is unavailable instead of silently
    manufacturing a projected quotient.
    """

    G = _symmetrize(np.asarray(factorization.raw_metric, dtype=float))
    supported_G = _symmetrize(
        factorization.retained_vectors
        @ np.diag(factorization.retained_eigenvalues)
        @ factorization.retained_vectors.T
    )
    W = np.asarray(factorization.whitening, dtype=float)
    E = np.asarray(minimum_eigenspace, dtype=float)
    x_p = np.asarray(particular_solution, dtype=float).reshape(-1)
    active_count = int(active_coordinate_count)
    dimension = int(G.shape[0])
    epsilon_G = float(factorization.support_threshold)
    support_common: dict[str, Any] = {
        "hard_case_orientation_support_schema": (
            "shared_supported_metric_quotient_support_v1"
        ),
        "hard_case_orientation_shared_metric_provenance_id": str(
            factorization.provenance_id
        ),
        "hard_case_orientation_shared_metric_support_rank": int(
            factorization.rank
        ),
        "hard_case_orientation_shared_metric_epsilon_G": epsilon_G,
        "hard_case_orientation_shared_metric_discarded_residual_norm": float(
            np.linalg.norm(G - supported_G, ord=2)
        ),
    }
    if active_count:
        active_gram = _symmetrize(
            supported_G[:active_count, :active_count]
        )
        try:
            active_eigenvalues, active_eigenvectors = np.linalg.eigh(active_gram)
        except np.linalg.LinAlgError:
            return None, {
                **support_common,
                "hard_case_orientation_support_status": "unresolved",
                "hard_case_orientation_support_reason": (
                    "active_metric_eigendecomposition_failed"
                ),
            }
        active_eigenvalues = np.asarray(active_eigenvalues, dtype=float)
        active_eigenvectors = np.asarray(active_eigenvectors, dtype=float)
        active_lower_bounds = np.asarray(
            active_eigenvalues - epsilon_G,
            dtype=float,
        )
        active_upper_bounds = np.asarray(
            active_eigenvalues + epsilon_G,
            dtype=float,
        )
        if float(active_upper_bounds[0]) < 0.0:
            return None, {
                **support_common,
                "hard_case_orientation_support_status": "invalid_geometry",
                "hard_case_orientation_support_reason": (
                    "materially_negative_active_metric"
                ),
                "hard_case_orientation_active_metric_eigenvalues": [
                    float(value) for value in active_eigenvalues.tolist()
                ],
                "hard_case_orientation_active_metric_eigenvalue_lower_bounds": [
                    float(value) for value in active_lower_bounds.tolist()
                ],
                "hard_case_orientation_active_metric_eigenvalue_upper_bounds": [
                    float(value) for value in active_upper_bounds.tolist()
                ],
            }
        active_mask = np.asarray(
            active_lower_bounds > 0.0,
            dtype=bool,
        )
        active_boundary_gap: float | None = None
        if np.any(active_mask) and not np.all(active_mask):
            first_retained = int(np.flatnonzero(active_mask)[0])
            active_boundary_gap = float(
                active_eigenvalues[first_retained]
                - active_eigenvalues[first_retained - 1]
            )
            if active_boundary_gap <= 2.0 * epsilon_G:
                return None, {
                    **support_common,
                    "hard_case_orientation_support_status": "unresolved",
                    "hard_case_orientation_support_reason": (
                        "active_metric_support_boundary_unresolved"
                    ),
                    "hard_case_orientation_active_metric_eigenvalues": [
                        float(value) for value in active_eigenvalues.tolist()
                    ],
                    "hard_case_orientation_active_metric_eigenvalue_lower_bounds": [
                        float(value) for value in active_lower_bounds.tolist()
                    ],
                    "hard_case_orientation_active_metric_eigenvalue_upper_bounds": [
                        float(value) for value in active_upper_bounds.tolist()
                    ],
                    "hard_case_orientation_active_metric_support_boundary_gap": (
                        active_boundary_gap
                    ),
                }
        if np.any(active_mask):
            active_pseudoinverse = _symmetrize(
                active_eigenvectors[:, active_mask]
                @ np.diag(active_eigenvalues[active_mask] ** -1.0)
                @ active_eigenvectors[:, active_mask].T
            )
        else:
            active_pseudoinverse = np.zeros_like(active_gram)
        quotient_joint = _symmetrize(
            supported_G
            - supported_G[:, :active_count]
            @ active_pseudoinverse
            @ supported_G[:active_count, :]
        )
    else:
        active_eigenvalues = np.zeros(0, dtype=float)
        active_lower_bounds = np.zeros(0, dtype=float)
        active_upper_bounds = np.zeros(0, dtype=float)
        active_mask = np.zeros(0, dtype=bool)
        active_boundary_gap = None
        quotient_joint = supported_G.copy()
    quotient_eigenvalues, quotient_eigenvectors = np.linalg.eigh(quotient_joint)
    quotient_psd_tolerance = float(
        max(
            float(epsilon_G),
            256.0
            * np.finfo(float).eps
            * max(1, dimension)
            * max(1.0, np.linalg.norm(G, ord=2)),
        )
    )
    materially_negative_quotient = bool(
        float(quotient_eigenvalues[0]) < -quotient_psd_tolerance
    )
    if materially_negative_quotient:
        return None, {
            **support_common,
            "hard_case_orientation_support_status": "invalid_geometry",
            "hard_case_orientation_support_reason": (
                "materially_negative_active_quotient"
            ),
            "hard_case_orientation_active_metric_eigenvalues": [
                float(value) for value in active_eigenvalues.tolist()
            ],
            "hard_case_orientation_active_metric_retained_mask": [
                bool(value) for value in active_mask.tolist()
            ],
            "hard_case_orientation_quotient_joint_eigenvalues": [
                float(value) for value in quotient_eigenvalues.tolist()
            ],
            "hard_case_orientation_quotient_psd_tolerance": (
                quotient_psd_tolerance
            ),
            "hard_case_orientation_quotient_materially_negative": True,
            "hard_case_orientation_quotient_psd_projection_applied": False,
        }
    quotient_psd_projection_applied = bool(
        float(quotient_eigenvalues[0]) < 0.0
    )
    quotient_joint_psd = _symmetrize(
        quotient_eigenvectors
        @ np.diag(np.clip(quotient_eigenvalues, 0.0, None))
        @ quotient_eigenvectors.T
    )
    total_trust_form = _symmetrize(W.T @ G @ W)
    quotient_trust_form = _symmetrize(W.T @ quotient_joint_psd @ W)

    def quotient_fraction(value: np.ndarray) -> tuple[float, float, float]:
        vector = np.asarray(value, dtype=float)
        numerator = float(max(0.0, vector.T @ quotient_trust_form @ vector))
        denominator = float(max(0.0, vector.T @ total_trust_form @ vector))
        ratio = float(numerator / denominator) if denominator > 0.0 else 0.0
        return numerator, denominator, ratio

    if E.shape[1] == 1:
        direction = _canonical_eigenspace_direction(E)
        search_iterations = 0
        search_residual = 0.0
        search_case = "simple_minimum_eigenspace"
    else:
        numerator_quadratic = _symmetrize(E.T @ quotient_trust_form @ E)
        numerator_linear = np.asarray(
            float(completion_radius) * E.T @ quotient_trust_form @ x_p,
            dtype=float,
        )
        numerator_constant = float(x_p.T @ quotient_trust_form @ x_p)
        denominator_quadratic = _symmetrize(E.T @ total_trust_form @ E)
        denominator_linear = np.asarray(
            float(completion_radius) * E.T @ total_trust_form @ x_p,
            dtype=float,
        )
        denominator_constant = float(x_p.T @ total_trust_form @ x_p)
        radius_sq = float(completion_radius) ** 2
        numerator_quadratic = radius_sq * numerator_quadratic
        denominator_quadratic = radius_sq * denominator_quadratic
        low = 0.0
        high = 1.0

        def maximum_difference(
            ratio: float,
        ) -> tuple[float, np.ndarray, dict[str, Any]]:
            y, details = _maximize_quadratic_on_sphere(
                quadratic=numerator_quadratic - ratio * denominator_quadratic,
                linear=numerator_linear - ratio * denominator_linear,
                radius=1.0,
            )
            constant = float(
                numerator_constant - ratio * denominator_constant
            )
            return float(details["objective"] + constant), y, details

        high_difference, _, _ = maximum_difference(high)
        for _ in range(16):
            if high_difference <= quotient_psd_tolerance:
                break
            high *= 2.0
            high_difference, _, _ = maximum_difference(high)
        best_y = _canonical_eigenspace_direction(np.eye(E.shape[1]))
        best_details: dict[str, Any] = {
            "solution_case": "quotient_zero_tie",
            "iterations": 0,
        }
        search_iterations = 0
        for search_iterations in range(1, 97):
            midpoint = float(0.5 * (low + high))
            difference, trial_y, trial_details = maximum_difference(midpoint)
            if difference > 0.0:
                low = midpoint
                best_y = trial_y
                best_details = trial_details
            else:
                high = midpoint
        direction = np.asarray(E @ best_y, dtype=float)
        direction_norm = float(np.linalg.norm(direction))
        direction = np.asarray(direction / direction_norm, dtype=float)
        plus_value = np.asarray(
            x_p + float(completion_radius) * direction,
            dtype=float,
        )
        selected_ratio = quotient_fraction(plus_value)[2]
        search_residual = float(abs(selected_ratio - low))
        search_case = str(best_details["solution_case"])

    plus_value = np.asarray(
        x_p + float(completion_radius) * direction,
        dtype=float,
    )
    minus_value = np.asarray(
        x_p - float(completion_radius) * direction,
        dtype=float,
    )
    plus_numerator, plus_denominator, plus_ratio = quotient_fraction(plus_value)
    minus_numerator, minus_denominator, minus_ratio = quotient_fraction(minus_value)
    telemetry = {
        **support_common,
        "hard_case_orientation_support_status": "resolved",
        "hard_case_orientation_support_reason": (
            "shared_metric_provenance_and_separated_active_support"
        ),
        "hard_case_orientation_policy": (
            "raw_joint_candidate_quotient_max_v1"
        ),
        "hard_case_orientation_active_coordinate_count": active_count,
        "hard_case_orientation_batch_coordinate_count": (
            dimension - active_count
        ),
        "hard_case_orientation_active_metric_eigenvalues": [
            float(value) for value in active_eigenvalues.tolist()
        ],
        "hard_case_orientation_active_metric_eigenvalue_lower_bounds": [
            float(value) for value in active_lower_bounds.tolist()
        ],
        "hard_case_orientation_active_metric_eigenvalue_upper_bounds": [
            float(value) for value in active_upper_bounds.tolist()
        ],
        "hard_case_orientation_active_metric_support_boundary_gap": (
            active_boundary_gap
        ),
        "hard_case_orientation_active_metric_retained_mask": [
            bool(value) for value in active_mask.tolist()
        ],
        "hard_case_orientation_active_projection_rank": int(
            np.count_nonzero(active_mask)
        ),
        "hard_case_orientation_quotient_joint_eigenvalues": [
            float(value) for value in quotient_eigenvalues.tolist()
        ],
        "hard_case_orientation_quotient_materially_negative": (
            materially_negative_quotient
        ),
        "hard_case_orientation_quotient_psd_tolerance": (
            quotient_psd_tolerance
        ),
        "hard_case_orientation_quotient_psd_projection_applied": (
            quotient_psd_projection_applied
        ),
        "hard_case_orientation_minimum_eigenspace_dimension": int(E.shape[1]),
        "hard_case_orientation_search_case": search_case,
        "hard_case_orientation_search_iterations": int(search_iterations),
        "hard_case_orientation_search_ratio_residual": search_residual,
        "hard_case_orientation_direction_whitened": [
            float(value) for value in direction.tolist()
        ],
        "hard_case_orientation_plus_quotient_norm_sq": plus_numerator,
        "hard_case_orientation_plus_total_norm_sq": plus_denominator,
        "hard_case_orientation_plus_quotient_fraction": plus_ratio,
        "hard_case_orientation_minus_quotient_norm_sq": minus_numerator,
        "hard_case_orientation_minus_total_norm_sq": minus_denominator,
        "hard_case_orientation_minus_quotient_fraction": minus_ratio,
        "hard_case_orientation_exact_signs_retained": True,
    }
    return direction, telemetry


def _minimum_hessian_eigenspace_cluster(
    *,
    eigenvalues: np.ndarray,
    propagated_hessian_error: float,
    machine_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve the minimum-Hessian spectral cluster under a declared error.

    Adjacent eigenvalues remain in the minimum cluster until the first gap
    exceeds the two-sided propagated Hessian uncertainty (plus the ordinary
    eigensolver machine tolerance).  This avoids treating an uncertainty-
    mixed near-degeneracy as a stable one-dimensional eigenspace.
    """

    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("minimum eigenspace requires at least one eigenvalue.")
    propagated_error = float(max(0.0, propagated_hessian_error))
    machine_error = float(max(0.0, machine_tolerance))
    separation_threshold = float(2.0 * propagated_error + machine_error)
    adjacent_gaps = np.asarray(np.diff(values), dtype=float)
    cluster_size = int(values.size)
    boundary_gap: float | None = None
    for gap_index, gap in enumerate(adjacent_gaps.tolist()):
        if float(gap) > separation_threshold:
            cluster_size = int(gap_index + 1)
            boundary_gap = float(gap)
            break
    mask = np.zeros(values.size, dtype=bool)
    mask[:cluster_size] = True
    rotation_bound = (
        None
        if boundary_gap is None
        else float(
            propagated_error
            / max(boundary_gap - propagated_error, np.finfo(float).tiny)
        )
    )
    telemetry = {
        "minimum_eigenspace_cluster_schema": (
            "propagated_hessian_spectral_gap_cluster_v1"
        ),
        "minimum_eigenspace_cluster_status": "resolved",
        "minimum_eigenspace_cluster_reason": (
            "spectrally_separated_from_remainder"
            if boundary_gap is not None
            else "minimum_cluster_spans_supported_spectrum"
        ),
        "minimum_eigenspace_propagated_hessian_error_bound": (
            propagated_error
        ),
        "minimum_eigenspace_machine_tolerance": machine_error,
        "minimum_eigenspace_separation_threshold": separation_threshold,
        "minimum_eigenspace_adjacent_gaps": [
            float(value) for value in adjacent_gaps.tolist()
        ],
        "minimum_eigenspace_boundary_gap": boundary_gap,
        "minimum_eigenspace_rotation_bound": rotation_bound,
        "minimum_eigenspace_dimension": cluster_size,
    }
    return mask, telemetry


def _raw_metric_null_compatibility_certificate(
    *,
    factorization: SupportedMetricWhitening,
    hessian: np.ndarray,
    gradient: np.ndarray,
    energy_tolerance: float,
) -> tuple[bool, str, dict[str, Any]]:
    """Certify that the energy model descends to the raw-metric quotient.

    Directions excluded by the raw Gram support decision are nominally metric
    null for the trust model.  A coordinate-invariant quotient model therefore
    requires the gradient to be orthogonal to ``ker(G)`` and requires
    ``H ker(G) = 0``.  The latter includes
    support--null coupling as well as curvature wholly inside the discarded
    subspace.  This certificate is intentionally consumed only by the v2
    global-trust policy; the source-locked v1 policy is unchanged.
    """

    H = _symmetrize(np.asarray(hessian, dtype=float))
    g = np.asarray(gradient, dtype=float).reshape(-1)
    dimension = int(g.size)
    retained_vectors = np.asarray(factorization.retained_vectors, dtype=float)
    retained_projector = _symmetrize(retained_vectors @ retained_vectors.T)
    discarded_projector = _symmetrize(
        np.eye(dimension, dtype=float) - retained_projector
    )
    discarded_dimension = int(dimension - factorization.rank)

    discarded_gradient = np.asarray(discarded_projector @ g, dtype=float)
    support_null_hessian = np.asarray(
        retained_projector @ H @ discarded_projector,
        dtype=float,
    )
    null_null_hessian = _symmetrize(discarded_projector @ H @ discarded_projector)
    discarded_hessian = np.asarray(H @ discarded_projector, dtype=float)

    machine_factor = float(
        4096.0 * np.finfo(float).eps * max(1, dimension)
    )
    gradient_scale = float(max(1.0, np.linalg.norm(g)))
    hessian_scale = float(max(1.0, np.linalg.norm(H, ord=2)))
    gradient_tolerance = float(
        max(float(energy_tolerance), machine_factor * gradient_scale)
    )
    hessian_tolerance = float(
        max(float(energy_tolerance), machine_factor * hessian_scale)
    )
    gradient_residual = float(np.linalg.norm(discarded_gradient))
    support_null_residual = float(np.linalg.norm(support_null_hessian, ord=2))
    null_null_residual = float(np.linalg.norm(null_null_hessian, ord=2))
    hessian_residual = float(np.linalg.norm(discarded_hessian, ord=2))
    gradient_compatible = bool(gradient_residual <= gradient_tolerance)
    hessian_compatible = bool(hessian_residual <= hessian_tolerance)
    certified = bool(gradient_compatible and hessian_compatible)
    if not gradient_compatible:
        reason = "raw_metric_null_gradient_incompatible"
    elif not hessian_compatible:
        reason = "raw_metric_null_hessian_incompatible"
    else:
        reason = "raw_metric_null_compatible"

    telemetry = {
        "raw_metric_null_compatibility_schema": (
            "raw_metric_null_compatibility_certificate_v1"
        ),
        "raw_metric_discarded_support_dimension": discarded_dimension,
        "raw_metric_null_gradient_residual_norm": gradient_residual,
        "raw_metric_null_gradient_tolerance": gradient_tolerance,
        "raw_metric_null_gradient_compatible": gradient_compatible,
        "raw_metric_support_null_hessian_coupling_norm": support_null_residual,
        "raw_metric_null_null_hessian_norm": null_null_residual,
        "raw_metric_null_hessian_residual_norm": hessian_residual,
        "raw_metric_null_hessian_tolerance": hessian_tolerance,
        "raw_metric_null_hessian_compatible": hessian_compatible,
        "raw_metric_null_compatibility_certified": certified,
        "raw_metric_null_compatibility_reason": reason,
    }
    return certified, reason, telemetry


def _solve_supported_metric_global_trust(
    *,
    G: np.ndarray,
    H: np.ndarray,
    g: np.ndarray,
    active_count: int,
    config: JointLinearSolveConfig,
) -> JointLinearSolveResult:
    """Solve the supported, whitened quadratic trust problem globally.

    This v2 policy implements the singular More--Sorensen hard case rather
    than making the shifted Hessian artificially positive definite.  The v1
    policy remains available unchanged for source-locked historical routes.
    """

    dimension = int(g.size)
    policy = JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
    base: dict[str, Any] = {
        "joint_linear_solve_policy_requested": str(config.policy),
        "joint_linear_solve_policy_effective": policy,
        "global_trust_solver_schema": policy,
        "joint_coordinate_count": dimension,
        "active_coordinate_count": int(active_count),
        "batch_coordinate_count": int(dimension - active_count),
        "classical_quantum_query_charge": 0,
    }
    factorization, support_reason, support_telemetry = (
        _factor_global_trust_raw_metric_support(
            G,
            eta_G=float(config.rank_relative_tolerance),
            eta_KKT=float(config.global_trust_kkt_residual_accuracy),
            metric_distortion_budget=float(
                config.global_trust_metric_distortion_budget
            ),
        )
    )
    base.update(support_telemetry)
    if factorization is None:
        return _empty_result(
            reason=str(support_reason),
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=base,
        )

    (
        null_compatibility_certified,
        null_compatibility_reason,
        null_compatibility_telemetry,
    ) = _raw_metric_null_compatibility_certificate(
        factorization=factorization,
        hessian=H,
        gradient=g,
        energy_tolerance=float(config.energy_regularization),
    )
    base.update(null_compatibility_telemetry)
    if not null_compatibility_certified:
        if bool(
            base.get(
                "metric_stabilization_zero_ridge_fallback_eligible",
                False,
            )
        ):
            base[
                "metric_stabilization_zero_ridge_fallback_solver_status"
            ] = "not_attempted_raw_metric_null_incompatible"
            base["metric_stabilization_status"] = "unresolved"
            base["metric_stabilization_reason"] = (
                "zero_ridge_fallback_not_attempted_raw_metric_null_incompatible"
            )
        return _empty_result(
            reason=str(null_compatibility_reason),
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=base,
        )
    if bool(
        base.get(
            "metric_stabilization_zero_ridge_fallback_eligible",
            False,
        )
    ):
        base["metric_stabilization_zero_ridge_fallback_attempted"] = True
        base[
            "metric_stabilization_zero_ridge_fallback_solver_status"
        ] = "pending_downstream_certificates"

    retained_eigenvalues = factorization.retained_eigenvalues
    retained_vectors = factorization.retained_vectors
    whitening_denominators = retained_eigenvalues + float(
        factorization.metric_ridge
    )
    W = factorization.whitening
    H_w = _symmetrize(W.T @ H @ W)
    g_w = np.asarray(W.T @ g, dtype=float)
    try:
        hessian_eigenvalues, hessian_eigenvectors = np.linalg.eigh(H_w)
    except np.linalg.LinAlgError:
        return _empty_result(
            reason="whitened_hessian_eigendecomposition_failed",
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=base,
        )

    supported_rank = int(hessian_eigenvalues.size)
    hessian_minimum = float(hessian_eigenvalues[0])
    hessian_scale = float(
        max(np.max(np.abs(hessian_eigenvalues)), np.finfo(float).tiny)
    )
    radius = float(config.max_fubini_study_step)
    radius_sq = float(radius * radius)
    machine_factor = float(
        4096.0 * np.finfo(float).eps * max(1, supported_rank)
    )
    eigenspace_tolerance = float(
        256.0
        * np.finfo(float).eps
        * max(1, supported_rank)
        * max(1.0, hessian_scale)
    )
    norm_tolerance = float(
        max(
            machine_factor * max(1.0, radius),
            float(config.energy_regularization),
        )
    )
    gradient_scale = float(
        max(1.0, np.linalg.norm(g_w), hessian_scale * radius)
    )
    gradient_zero_tolerance = float(machine_factor * gradient_scale)
    effective_backward_error_level = float(
        support_telemetry["raw_gram_effective_backward_error_level"]
    )
    whitening_operator_norm = float(np.linalg.norm(W, ord=2))
    gradient_input_norm = float(np.linalg.norm(g))
    hessian_input_norm = float(np.linalg.norm(H, ord=2))
    primitive_gradient_error = float(
        effective_backward_error_level * gradient_input_norm
    )
    algebraic_gradient_error = float(
        effective_backward_error_level
        * whitening_operator_norm
        * gradient_input_norm
    )
    propagated_gradient_error = float(
        whitening_operator_norm * primitive_gradient_error
        + algebraic_gradient_error
    )
    primitive_hessian_error = float(
        effective_backward_error_level * hessian_input_norm
    )
    algebraic_hessian_error = float(
        2.0
        * effective_backward_error_level
        * whitening_operator_norm**2
        * hessian_input_norm
    )
    hessian_reconstruction = _symmetrize(
        hessian_eigenvectors
        @ np.diag(hessian_eigenvalues)
        @ hessian_eigenvectors.T
    )
    hessian_eigendecomposition_error = float(
        np.linalg.norm(H_w - hessian_reconstruction, ord=2)
    )
    propagated_hessian_error = float(
        whitening_operator_norm**2 * primitive_hessian_error
        + algebraic_hessian_error
        + hessian_eigendecomposition_error
    )
    model_energy_scale = float(
        np.linalg.norm(g_w) * radius
        + 0.5 * np.linalg.norm(H_w, ord=2) * radius_sq
    )
    energy_comparison_width = float(
        max(
            float(config.energy_regularization),
            effective_backward_error_level * model_energy_scale,
        )
    )
    gradient_resolution = float(energy_comparison_width / radius)
    supported_gradient_norm = float(np.linalg.norm(g_w))
    stationarity_upper_bound = float(
        supported_gradient_norm + propagated_gradient_error
    )
    stationarity_lower_bound = float(
        max(0.0, supported_gradient_norm - propagated_gradient_error)
    )
    if stationarity_upper_bound <= gradient_resolution:
        supported_stationarity_status = "stationary"
    elif stationarity_lower_bound > gradient_resolution:
        supported_stationarity_status = "certified_nonstationary"
    else:
        supported_stationarity_status = "unresolved"
    hessian_eigenvalue_lower_bounds = np.asarray(
        hessian_eigenvalues - propagated_hessian_error,
        dtype=float,
    )
    hessian_eigenvalue_upper_bounds = np.asarray(
        hessian_eigenvalues + propagated_hessian_error,
        dtype=float,
    )
    if float(hessian_eigenvalue_upper_bounds[0]) < 0.0:
        supported_inertia_status = "negative"
    elif float(hessian_eigenvalue_lower_bounds[0]) >= 0.0:
        supported_inertia_status = "psd"
    else:
        supported_inertia_status = "unresolved"
    supported_eigenvalue_statuses = [
        (
            "negative"
            if float(upper) < 0.0
            else "positive"
            if float(lower) > 0.0
            else "unresolved"
        )
        for lower, upper in zip(
            hessian_eigenvalue_lower_bounds.tolist(),
            hessian_eigenvalue_upper_bounds.tolist(),
        )
    ]
    model_certificate_telemetry = {
        "supported_model_certificate_schema": (
            "propagated_numerical_bounds_v1"
        ),
        "supported_model_effective_backward_error_level": (
            effective_backward_error_level
        ),
        "supported_model_whitening_operator_norm": whitening_operator_norm,
        "supported_gradient_primitive_error_bound": primitive_gradient_error,
        "supported_gradient_algebraic_error_bound": algebraic_gradient_error,
        "supported_gradient_propagated_error_bound": (
            propagated_gradient_error
        ),
        "supported_hessian_primitive_error_bound": primitive_hessian_error,
        "supported_hessian_algebraic_error_bound": algebraic_hessian_error,
        "supported_hessian_eigendecomposition_error_bound": (
            hessian_eigendecomposition_error
        ),
        "supported_hessian_propagated_error_bound": propagated_hessian_error,
        "supported_energy_comparison_width": energy_comparison_width,
        "supported_energy_comparison_width_source": (
            "max_configured_energy_resolution_and_model_backward_error"
        ),
        "supported_gradient_resolution": gradient_resolution,
        "supported_gradient_norm": supported_gradient_norm,
        "supported_gradient_norm_lower_bound": stationarity_lower_bound,
        "supported_gradient_norm_upper_bound": stationarity_upper_bound,
        "supported_stationarity_status": supported_stationarity_status,
        "supported_hessian_eigenvalue_lower_bounds": [
            float(value) for value in hessian_eigenvalue_lower_bounds.tolist()
        ],
        "supported_hessian_eigenvalue_upper_bounds": [
            float(value) for value in hessian_eigenvalue_upper_bounds.tolist()
        ],
        "supported_inertia_label_issued": False,
    }
    minimum_mask, minimum_cluster_telemetry = (
        _minimum_hessian_eigenspace_cluster(
            eigenvalues=hessian_eigenvalues,
            propagated_hessian_error=propagated_hessian_error,
            machine_tolerance=eigenspace_tolerance,
        )
    )
    minimum_eigenspace = np.asarray(
        hessian_eigenvectors[:, minimum_mask],
        dtype=float,
    )
    gradient_eigenbasis = np.asarray(hessian_eigenvectors.T @ g_w, dtype=float)
    minimum_eigenspace_gradient_norm = float(
        np.linalg.norm(gradient_eigenbasis[minimum_mask])
    )
    minimum_eigenspace_gradient_lower_bound = float(
        max(
            0.0,
            minimum_eigenspace_gradient_norm - propagated_gradient_error,
        )
    )
    minimum_eigenspace_gradient_upper_bound = float(
        minimum_eigenspace_gradient_norm + propagated_gradient_error
    )
    if minimum_eigenspace_gradient_norm <= gradient_zero_tolerance:
        minimum_eigenspace_gradient_status = "point_zero_compatible"
    elif minimum_eigenspace_gradient_lower_bound > gradient_resolution:
        minimum_eigenspace_gradient_status = "resolved_nonzero"
    else:
        minimum_eigenspace_gradient_status = "unresolved_from_zero"
    lambda_lower = float(max(0.0, -hessian_minimum))
    shifted_at_lower = np.asarray(
        hessian_eigenvalues + lambda_lower,
        dtype=float,
    )
    singular_mask = np.asarray(
        np.abs(shifted_at_lower) <= eigenspace_tolerance,
        dtype=bool,
    )
    exact_minimum_eigenspace = np.asarray(
        hessian_eigenvectors[:, singular_mask],
        dtype=float,
    )
    singular_gradient_norm = float(
        np.linalg.norm(gradient_eigenbasis[singular_mask])
    )

    hard_case_detected = False
    hard_case_boundary_completion = False
    hard_case_direction = np.zeros(0, dtype=float)
    hard_case_candidates_whitened: list[list[float]] = []
    hard_case_candidates_joint: list[list[float]] = []
    hard_case_candidate_reductions: list[float] = []
    hard_case_candidate_point_estimate_roles: list[str] = []
    hard_case_selected_sign: int | None = None
    hard_case_point_estimate_optimum_candidate_index: int | None = None
    hard_case_classification = "none"
    hard_case_uncertain_projection_reflection_retained = False
    hard_case_orientation_telemetry: dict[str, Any] = {}
    boundary_root_iterations = 0
    solution_case = "boundary_secular_root"
    applied_x: np.ndarray | None = None
    applied_lambda: float | None = None

    lower_solution = np.zeros(supported_rank, dtype=float)
    nonsingular_mask = ~singular_mask
    lower_solution_eigenbasis = np.zeros(supported_rank, dtype=float)
    lower_solution_eigenbasis[nonsingular_mask] = (
        gradient_eigenbasis[nonsingular_mask]
        / shifted_at_lower[nonsingular_mask]
    )
    lower_solution = np.asarray(
        hessian_eigenvectors @ lower_solution_eigenbasis,
        dtype=float,
    )
    lower_solution_norm = float(np.linalg.norm(lower_solution))
    lower_stationarity_compatible = bool(
        singular_gradient_norm <= gradient_zero_tolerance
    )

    if (
        lower_stationarity_compatible
        and lower_solution_norm <= radius + norm_tolerance
    ):
        if hessian_minimum < 0.0:
            hard_case_detected = True
            hard_case_classification = "exact_singular_more_sorensen"
            hard_case_radius = float(
                math.sqrt(max(0.0, radius_sq - lower_solution_norm**2))
            )
            (
                hard_case_direction,
                hard_case_orientation_telemetry,
            ) = _candidate_quotient_hard_case_direction(
                factorization=factorization,
                active_coordinate_count=active_count,
                minimum_eigenspace=exact_minimum_eigenspace,
                particular_solution=lower_solution,
                completion_radius=hard_case_radius,
            )
            if hard_case_direction is None:
                return _empty_result(
                    reason=str(
                        hard_case_orientation_telemetry.get(
                            "hard_case_orientation_support_reason",
                            "hard_case_orientation_support_unresolved",
                        )
                    ),
                    dimension=dimension,
                    active_coordinate_count=active_count,
                    telemetry={
                        **base,
                        **model_certificate_telemetry,
                        **minimum_cluster_telemetry,
                        **hard_case_orientation_telemetry,
                    },
                )
            hard_case_boundary_completion = bool(
                hard_case_radius > norm_tolerance
            )
            candidate_plus = np.asarray(
                lower_solution + hard_case_radius * hard_case_direction,
                dtype=float,
            )
            candidate_minus = np.asarray(
                lower_solution - hard_case_radius * hard_case_direction,
                dtype=float,
            )
            candidates = (candidate_plus, candidate_minus)
            for candidate in candidates:
                candidate_joint = np.asarray(W @ candidate, dtype=float)
                candidate_reduction = float(
                    g_w.T @ candidate - 0.5 * candidate.T @ H_w @ candidate
                )
                hard_case_candidates_whitened.append(
                    [float(value) for value in candidate.tolist()]
                )
                hard_case_candidates_joint.append(
                    [float(value) for value in candidate_joint.tolist()]
                )
                hard_case_candidate_reductions.append(candidate_reduction)
                hard_case_candidate_point_estimate_roles.append(
                    "point_estimate_global_optimum"
                )
            selected_index = int(np.argmax(hard_case_candidate_reductions))
            applied_x = np.asarray(candidates[selected_index], dtype=float)
            hard_case_selected_sign = 1 if selected_index == 0 else -1
            hard_case_point_estimate_optimum_candidate_index = selected_index
            applied_lambda = lambda_lower
            solution_case = "singular_hard_case"
        else:
            applied_x = lower_solution
            applied_lambda = 0.0
            solution_case = "positive_semidefinite_interior"

    def regular_solution(trust_lambda: float) -> np.ndarray | None:
        denominators = np.asarray(
            hessian_eigenvalues + float(trust_lambda),
            dtype=float,
        )
        if np.any(denominators <= 0.0):
            return None
        value_eigenbasis = gradient_eigenbasis / denominators
        value = np.asarray(hessian_eigenvectors @ value_eigenbasis, dtype=float)
        if not np.all(np.isfinite(value)):
            return None
        return value

    if applied_x is None:
        coefficient_norm = float(np.linalg.norm(gradient_eigenbasis))
        representable_delta = float(
            np.nextafter(lambda_lower, math.inf) - lambda_lower
        )
        high_delta = float(
            max(
                2.0 * coefficient_norm / radius,
                eigenspace_tolerance,
                representable_delta,
            )
        )
        high = float(lambda_lower + high_delta)
        high_solution = regular_solution(high)
        for _ in range(80):
            if (
                high_solution is not None
                and float(np.linalg.norm(high_solution)) <= radius
            ):
                break
            high_delta = float(2.0 * high_delta)
            high = float(lambda_lower + high_delta)
            high_solution = regular_solution(high)
        else:
            return _empty_result(
                reason="global_trust_bracket_failed",
                dimension=dimension,
                active_coordinate_count=active_count,
                telemetry={
                    **base,
                    "H_w_eigenvalues": [
                        float(value) for value in hessian_eigenvalues.tolist()
                    ],
                    "global_trust_lambda_lower_bound": lambda_lower,
                },
            )

        low = float(lambda_lower)
        for boundary_root_iterations in range(1, 161):
            midpoint = float(0.5 * (low + high))
            if midpoint <= low or midpoint >= high:
                break
            midpoint_solution = regular_solution(midpoint)
            if (
                midpoint_solution is None
                or float(np.linalg.norm(midpoint_solution)) > radius
            ):
                low = midpoint
            else:
                high = midpoint
                high_solution = midpoint_solution
        applied_x = np.asarray(high_solution, dtype=float)
        applied_x_norm = float(np.linalg.norm(applied_x))
        if applied_x_norm > 0.0:
            applied_x = np.asarray(
                applied_x * (radius / applied_x_norm),
                dtype=float,
            )
        applied_lambda = float(high)

    if applied_x is None or applied_lambda is None:
        return _empty_result(
            reason="global_trust_solve_failed",
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=base,
        )

    if (
        solution_case == "boundary_secular_root"
        and hessian_minimum < 0.0
        and minimum_eigenspace_gradient_status == "unresolved_from_zero"
    ):
        # The regular secular solution is the point-estimate optimum.  At the
        # declared model-comparison resolution, however, the sign preference
        # in the uncertainty-mixed minimum cluster is not resolved.  Preserve
        # the regular solution and its exact cluster reflection for the
        # downstream exact-energy guard; do not relabel the reflection as a
        # second point-estimate optimum.
        minimum_projector = _symmetrize(
            minimum_eigenspace @ minimum_eigenspace.T
        )
        reflected_x = np.asarray(
            applied_x - 2.0 * minimum_projector @ applied_x,
            dtype=float,
        )
        regular_reduction = float(
            g_w.T @ applied_x - 0.5 * applied_x.T @ H_w @ applied_x
        )
        reflected_reduction = float(
            g_w.T @ reflected_x - 0.5 * reflected_x.T @ H_w @ reflected_x
        )
        reflected_joint = np.asarray(W @ reflected_x, dtype=float)
        regular_joint = np.asarray(W @ applied_x, dtype=float)
        hard_case_detected = True
        hard_case_classification = (
            "unresolved_minimum_projection_reflection_pair"
        )
        hard_case_uncertain_projection_reflection_retained = True
        hard_case_candidates_whitened = [
            [float(value) for value in applied_x.tolist()],
            [float(value) for value in reflected_x.tolist()],
        ]
        hard_case_candidates_joint = [
            [float(value) for value in regular_joint.tolist()],
            [float(value) for value in reflected_joint.tolist()],
        ]
        hard_case_candidate_reductions = [
            regular_reduction,
            reflected_reduction,
        ]
        hard_case_candidate_point_estimate_roles = [
            "regular_point_estimate_global_optimum",
            "uncertainty_reflection_not_point_estimate_optimum",
        ]
        hard_case_selected_sign = 1
        hard_case_point_estimate_optimum_candidate_index = 0
        projected_component = np.asarray(
            minimum_projector @ applied_x,
            dtype=float,
        )
        projected_component_norm = float(np.linalg.norm(projected_component))
        hard_case_direction = (
            np.asarray(
                projected_component / projected_component_norm,
                dtype=float,
            )
            if projected_component_norm > 0.0
            else np.zeros_like(applied_x)
        )
        pairwise_comparison_tolerance = float(
            2.0 * energy_comparison_width
            + 2.0 * radius * propagated_gradient_error
        )
        hard_case_orientation_telemetry = {
            "hard_case_orientation_policy": (
                "minimum_eigenspace_reflection_under_unresolved_projection_v1"
            ),
            "hard_case_orientation_support_status": "resolved",
            "hard_case_orientation_support_reason": (
                "shared_minimum_cluster_reflection"
            ),
            "hard_case_orientation_shared_metric_provenance_id": str(
                factorization.provenance_id
            ),
            "hard_case_orientation_exact_signs_retained": True,
            "hard_case_reflection_is_point_estimate_optimum": False,
            "hard_case_reflection_predicted_reduction_difference": float(
                regular_reduction - reflected_reduction
            ),
            "hard_case_reflection_pairwise_comparison_tolerance": (
                pairwise_comparison_tolerance
            ),
            "hard_case_reflection_preference_resolved": bool(
                abs(regular_reduction - reflected_reduction)
                > pairwise_comparison_tolerance
            ),
            "hard_case_reflection_operator_schema": (
                "I_minus_2P_minimum_cluster_v1"
            ),
        }

    step = np.asarray(W @ applied_x, dtype=float)
    whitened_norm = float(np.linalg.norm(applied_x))
    displacement_sq = float(max(0.0, step.T @ G @ step))
    predicted_reduction_raw = float(
        g_w.T @ applied_x - 0.5 * applied_x.T @ H_w @ applied_x
    )
    shifted_eigenvalues = np.asarray(
        hessian_eigenvalues + applied_lambda,
        dtype=float,
    )
    stationarity_vector = np.asarray(
        (H_w + applied_lambda * np.eye(supported_rank)) @ applied_x - g_w,
        dtype=float,
    )
    stationarity_residual = float(np.linalg.norm(stationarity_vector))
    primal_violation = float(max(0.0, whitened_norm - radius))
    dual_violation = float(max(0.0, -applied_lambda))
    complementarity_residual = float(
        abs(applied_lambda * (whitened_norm - radius))
    )
    shifted_hessian_minimum = float(np.min(shifted_eigenvalues))
    psd_violation = float(max(0.0, -shifted_hessian_minimum))
    objective_identity_value = float(
        0.5 * g_w.T @ applied_x
        + 0.5 * applied_lambda * whitened_norm**2
    )
    objective_identity_residual = float(
        abs(predicted_reduction_raw - objective_identity_value)
    )
    stationarity_tolerance = float(
        propagated_gradient_error
        + radius * propagated_hessian_error
        + effective_backward_error_level
        * (
            np.linalg.norm(H_w, ord=2)
            + propagated_hessian_error
            + abs(applied_lambda)
        )
        * radius
    )
    dual_psd_tolerance = float(
        propagated_hessian_error
        + effective_backward_error_level
        * max(
            np.linalg.norm(H_w, ord=2) + abs(applied_lambda),
            np.finfo(float).tiny,
        )
    )
    complementarity_tolerance = float(
        abs(applied_lambda) * norm_tolerance
        + effective_backward_error_level
        * max(abs(applied_lambda) * radius, np.finfo(float).tiny)
    )
    objective_identity_tolerance = float(
        stationarity_tolerance * radius
        + effective_backward_error_level
        * max(abs(predicted_reduction_raw), np.finfo(float).tiny)
    )
    global_optimality_certified = bool(
        stationarity_residual <= stationarity_tolerance
        and primal_violation <= norm_tolerance
        and dual_violation <= dual_psd_tolerance
        and complementarity_residual <= complementarity_tolerance
        and psd_violation <= dual_psd_tolerance
        and objective_identity_residual <= objective_identity_tolerance
    )

    regularized_residual_vector = np.asarray(
        (
            H
            + applied_lambda * factorization.regularized_supported_metric
        )
        @ step
        - g,
        dtype=float,
    )
    discarded_gradient = np.asarray(
        g - retained_vectors @ (retained_vectors.T @ g),
        dtype=float,
    )
    if shifted_hessian_minimum <= eigenspace_tolerance:
        applied_condition: float | None = None
    else:
        applied_condition = float(
            np.max(shifted_eigenvalues) / shifted_hessian_minimum
        )
    trust_radius_binding = bool(
        abs(whitened_norm - radius) <= norm_tolerance
    )
    telemetry = {
        **base,
        **model_certificate_telemetry,
        **minimum_cluster_telemetry,
        **hard_case_orientation_telemetry,
        "whitening_denominators": [
            float(value) for value in whitening_denominators.tolist()
        ],
        "whitened_metric_eigenvalues": [
            float(value)
            for value in np.linalg.eigvalsh(_symmetrize(W.T @ G @ W)).tolist()
        ],
        "H_w_eigenvalues": [
            float(value) for value in hessian_eigenvalues.tolist()
        ],
        "global_trust_solution_case": solution_case,
        "global_trust_lambda_lower_bound": lambda_lower,
        "global_trust_eigenspace_tolerance": eigenspace_tolerance,
        "global_trust_gradient_zero_tolerance": gradient_zero_tolerance,
        "minimum_eigenspace_dimension": int(np.count_nonzero(minimum_mask)),
        "minimum_eigenspace_gradient_norm": minimum_eigenspace_gradient_norm,
        "minimum_eigenspace_gradient_norm_lower_bound": (
            minimum_eigenspace_gradient_lower_bound
        ),
        "minimum_eigenspace_gradient_norm_upper_bound": (
            minimum_eigenspace_gradient_upper_bound
        ),
        "minimum_eigenspace_gradient_comparison_resolution": (
            gradient_resolution
        ),
        "minimum_eigenspace_gradient_status": (
            minimum_eigenspace_gradient_status
        ),
        "hard_case_detected": bool(hard_case_detected),
        "hard_case_classification": hard_case_classification,
        "hard_case_boundary_completion": bool(hard_case_boundary_completion),
        "hard_case_uncertain_projection_reflection_retained": bool(
            hard_case_uncertain_projection_reflection_retained
        ),
        "hard_case_deterministic_direction_whitened": [
            float(value) for value in hard_case_direction.tolist()
        ],
        "hard_case_sign_candidates_whitened": hard_case_candidates_whitened,
        "hard_case_sign_candidates_joint": hard_case_candidates_joint,
        "hard_case_sign_candidate_predicted_reductions": (
            hard_case_candidate_reductions
        ),
        "hard_case_sign_candidate_point_estimate_roles": (
            hard_case_candidate_point_estimate_roles
        ),
        "hard_case_selected_sign": hard_case_selected_sign,
        "hard_case_point_estimate_optimum_candidate_index": (
            hard_case_point_estimate_optimum_candidate_index
        ),
        "trust_lambda": float(applied_lambda),
        "curvature_shift": lambda_lower,
        "applied_whitened_condition_number": applied_condition,
        "whitened_step_norm": whitened_norm,
        "joint_fubini_study_displacement_sq": displacement_sq,
        "full_direct_residual": float(np.linalg.norm(regularized_residual_vector)),
        "supported_direct_residual": float(
            np.linalg.norm(W.T @ regularized_residual_vector)
        ),
        "whitened_solve_residual": stationarity_residual,
        "discarded_gradient_norm": float(np.linalg.norm(discarded_gradient)),
        "trust_regularization_applied": bool(applied_lambda > 0.0),
        "trust_clipped": bool(applied_lambda > 0.0),
        "trust_radius_binding": bool(trust_radius_binding),
        "trust_boundary_root_iterations": int(boundary_root_iterations),
        "trust_kkt_stationarity_residual": stationarity_residual,
        "trust_kkt_stationarity_tolerance": stationarity_tolerance,
        "trust_kkt_primal_violation": primal_violation,
        "trust_kkt_primal_tolerance": norm_tolerance,
        "trust_kkt_dual_violation": dual_violation,
        "trust_kkt_dual_tolerance": dual_psd_tolerance,
        "trust_kkt_complementarity_residual": complementarity_residual,
        "trust_kkt_complementarity_tolerance": complementarity_tolerance,
        "trust_kkt_shifted_hessian_minimum_eigenvalue": (
            shifted_hessian_minimum
        ),
        "trust_kkt_psd_violation": psd_violation,
        "trust_kkt_psd_tolerance": dual_psd_tolerance,
        "trust_kkt_objective_identity_residual": objective_identity_residual,
        "trust_kkt_objective_identity_tolerance": (
            objective_identity_tolerance
        ),
        "trust_global_optimality_certified": global_optimality_certified,
        "active_parameter_relaxation": [
            float(value) for value in step[:active_count].tolist()
        ],
        "batch_coordinate_step": [
            float(value) for value in step[active_count:].tolist()
        ],
        "applied_predicted_reduction": float(predicted_reduction_raw),
    }
    if not global_optimality_certified:
        return _empty_result(
            reason="global_trust_kkt_certificate_failed",
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=telemetry,
        )
    prediction_tolerance = float(
        max(
            float(config.energy_regularization),
            machine_factor * max(1.0, abs(predicted_reduction_raw)),
        )
    )
    if predicted_reduction_raw < -prediction_tolerance:
        return _empty_result(
            reason="negative_predicted_reduction",
            dimension=dimension,
            active_coordinate_count=active_count,
            telemetry=telemetry,
        )
    predicted_reduction = float(max(0.0, predicted_reduction_raw))
    telemetry["applied_predicted_reduction"] = predicted_reduction
    telemetry["supported_inertia_label_issued"] = True
    telemetry["supported_inertia_status"] = supported_inertia_status
    telemetry["supported_hessian_eigenvalue_statuses"] = (
        supported_eigenvalue_statuses
    )
    return JointLinearSolveResult(
        feasible=True,
        reason="supported_metric_global_trust_eigh_solve",
        active_parameter_relaxation=step[:active_count].copy(),
        batch_coordinate_step=step[active_count:].copy(),
        joint_step=step.copy(),
        predicted_reduction=predicted_reduction,
        fubini_study_displacement_sq=displacement_sq,
        trust_lambda=float(applied_lambda),
        telemetry=telemetry,
    )


def _solve_block_pinv_legacy(
    *,
    G: np.ndarray,
    H: np.ndarray,
    g: np.ndarray,
    active_count: int,
    config: JointLinearSolveConfig,
) -> JointLinearSolveResult:
    dimension = int(g.size)
    energy_floor = float(max(config.energy_regularization, 1e-15))
    radius_sq = float(config.max_fubini_study_step) ** 2
    identity = np.eye(dimension, dtype=float)

    def solve_at(trust_lambda: float) -> dict[str, Any]:
        matrix = _symmetrize(H + float(trust_lambda) * G) + energy_floor * identity
        if active_count:
            M_AA = matrix[:active_count, :active_count]
            M_AB = matrix[:active_count, active_count:]
            M_BA = matrix[active_count:, :active_count]
            M_BB = matrix[active_count:, active_count:]
            M_AA_inverse = np.linalg.pinv(M_AA, rcond=energy_floor)
            M_effective = M_BB - M_BA @ M_AA_inverse @ M_AB
            g_A = g[:active_count]
            g_B = g[active_count:]
            g_effective = g_B - M_BA @ M_AA_inverse @ g_A
            batch_step = np.asarray(
                np.linalg.pinv(M_effective, rcond=energy_floor) @ g_effective,
                dtype=float,
            )
            active_step = np.asarray(
                M_AA_inverse @ (g_A - M_AB @ batch_step),
                dtype=float,
            )
        else:
            batch_step = np.asarray(np.linalg.pinv(matrix, rcond=energy_floor) @ g)
            active_step = np.zeros(0, dtype=float)
        step = np.concatenate([active_step, batch_step])
        displacement_sq = float(max(0.0, step.T @ G @ step))
        predicted_reduction = float(g.T @ step - 0.5 * step.T @ H @ step)
        eigenvalues = np.linalg.eigvalsh(matrix)
        return {
            "matrix": matrix,
            "step": step,
            "displacement_sq": displacement_sq,
            "predicted_reduction": predicted_reduction,
            "minimum_eigenvalue": float(np.min(eigenvalues)),
            "maximum_eigenvalue": float(np.max(eigenvalues)),
            "solved_residual": float(np.linalg.norm(matrix @ step - g)),
            "full_residual": float(
                np.linalg.norm((H + float(trust_lambda) * G) @ step - g)
            ),
            "trust_lambda": float(trust_lambda),
        }

    def feasible(solution: dict[str, Any]) -> bool:
        return bool(
            math.isfinite(float(solution["predicted_reduction"]))
            and float(solution["minimum_eigenvalue"]) > 0.0
            and float(solution["displacement_sq"]) <= radius_sq * (1.0 + 1e-10)
        )

    unconstrained = solve_at(0.0)
    applied = unconstrained
    trust_clipped = False
    if not feasible(applied):
        low = 0.0
        high = float(max(energy_floor, 1e-12))
        feasible_high: dict[str, Any] | None = None
        for _ in range(80):
            trial = solve_at(high)
            if feasible(trial):
                feasible_high = trial
                break
            low = high
            high *= 2.0
        if feasible_high is None:
            return _empty_result(
                reason="legacy_joint_trust_solve_failed",
                dimension=dimension,
                active_coordinate_count=active_count,
                telemetry={
                    "joint_linear_solve_policy_effective": (
                        JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1
                    )
                },
            )
        for _ in range(64):
            midpoint = 0.5 * (low + high)
            trial = solve_at(midpoint)
            if feasible(trial):
                high = midpoint
                feasible_high = trial
            else:
                low = midpoint
        applied = feasible_high
        trust_clipped = True

    step = np.asarray(applied["step"], dtype=float)
    predicted_reduction = float(max(0.0, applied["predicted_reduction"]))
    minimum_eigenvalue = float(applied["minimum_eigenvalue"])
    maximum_eigenvalue = float(applied["maximum_eigenvalue"])
    condition_number = float(maximum_eigenvalue / minimum_eigenvalue)
    telemetry = {
        "joint_linear_solve_policy_requested": str(config.policy),
        "joint_linear_solve_policy_effective": (
            JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1
        ),
        "classical_quantum_query_charge": 0,
        "trust_lambda": float(applied["trust_lambda"]),
        "trust_clipped": bool(trust_clipped),
        "trust_radius_binding": bool(
            trust_clipped
            and abs(float(applied["displacement_sq"]) - radius_sq)
            <= max(1e-14, radius_sq * 1e-8)
        ),
        "applied_whitened_condition_number": float(condition_number),
        "joint_fubini_study_displacement_sq": float(applied["displacement_sq"]),
        "full_direct_residual": float(applied["full_residual"]),
        "legacy_solved_residual": float(applied["solved_residual"]),
        "active_parameter_relaxation": [
            float(value) for value in step[:active_count].tolist()
        ],
        "batch_coordinate_step": [
            float(value) for value in step[active_count:].tolist()
        ],
        "applied_predicted_reduction": float(predicted_reduction),
    }
    return JointLinearSolveResult(
        feasible=True,
        reason="block_pinv_legacy_solve",
        active_parameter_relaxation=step[:active_count].copy(),
        batch_coordinate_step=step[active_count:].copy(),
        joint_step=step.copy(),
        predicted_reduction=float(predicted_reduction),
        fubini_study_displacement_sq=float(applied["displacement_sq"]),
        trust_lambda=float(applied["trust_lambda"]),
        telemetry=telemetry,
    )


def _primitive_hessian_symmetry_certificate(
    *,
    hessian: np.ndarray,
    energy_resolution: float,
    trust_radius: float,
) -> tuple[bool, dict[str, Any]]:
    """Bound primitive Hessian skew before the public input symmetrization."""

    primitive = np.asarray(hessian, dtype=float)
    if primitive.ndim != 2 or primitive.shape[0] != primitive.shape[1]:
        return False, {
            "primitive_hessian_symmetry_schema": (
                "primitive_hessian_antisymmetry_certificate_v1"
            ),
            "primitive_hessian_symmetry_status": "invalid_geometry",
            "primitive_hessian_symmetry_reason": "hessian_not_square",
            "primitive_hessian_antisymmetric_residual_norm": None,
            "primitive_hessian_antisymmetric_floating_error_bound": None,
            "primitive_hessian_antisymmetric_input_resolution_bound": None,
            "primitive_hessian_antisymmetric_total_bound": None,
        }
    dimension = int(primitive.shape[0])
    skew = np.asarray(0.5 * (primitive - primitive.T), dtype=float)
    residual = float(np.linalg.norm(skew, ord=2))
    hessian_scale = float(
        max(np.linalg.norm(primitive, ord=2), np.finfo(float).tiny)
    )
    arithmetic_numerator = float(
        128.0 * max(1, dimension) * np.finfo(float).eps
    )
    arithmetic_denominator = float(max(1.0 - arithmetic_numerator, 0.5))
    floating_error_bound = float(
        arithmetic_numerator / arithmetic_denominator * hessian_scale
    )
    radius_sq = float(max(float(trust_radius) ** 2, np.finfo(float).tiny))
    # Since the quadratic model contains (1/2) x^T H x, a Hessian
    # perturbation of 2*epsilon_E/rho^2 is below the declared energy
    # comparison width throughout the trust ball.
    input_resolution_bound = float(
        2.0 * max(0.0, float(energy_resolution)) / radius_sq
    )
    total_bound = float(floating_error_bound + input_resolution_bound)
    certified = bool(residual <= total_bound)
    telemetry = {
        "primitive_hessian_symmetry_schema": (
            "primitive_hessian_antisymmetry_certificate_v1"
        ),
        "primitive_hessian_symmetry_status": (
            "resolved" if certified else "invalid_geometry"
        ),
        "primitive_hessian_symmetry_reason": (
            "antisymmetry_within_derived_uncertainty"
            if certified
            else "primitive_hessian_antisymmetry_exceeds_uncertainty"
        ),
        "primitive_hessian_antisymmetric_residual_norm": residual,
        "primitive_hessian_antisymmetric_floating_error_bound": (
            floating_error_bound
        ),
        "primitive_hessian_antisymmetric_input_resolution_bound": (
            input_resolution_bound
        ),
        "primitive_hessian_antisymmetric_total_bound": total_bound,
        "primitive_hessian_antisymmetric_norm_convention": (
            "spectral_norm_of_half_H_minus_H_transpose"
        ),
    }
    return certified, telemetry


def solve_joint_linear_model(
    *,
    gram: np.ndarray,
    hessian: np.ndarray,
    gradient: np.ndarray,
    active_coordinate_count: int,
    config: JointLinearSolveConfig | None = None,
) -> JointLinearSolveResult:
    """Solve a joint quadratic model under a Fubini-Study trust radius."""

    resolved = config if config is not None else JointLinearSolveConfig()
    primitive_symmetry_certified = True
    primitive_symmetry_telemetry: dict[str, Any] = {}
    if (
        str(resolved.policy)
        == JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
    ):
        (
            primitive_symmetry_certified,
            primitive_symmetry_telemetry,
        ) = _primitive_hessian_symmetry_certificate(
            hessian=hessian,
            energy_resolution=float(resolved.energy_regularization),
            trust_radius=float(resolved.max_fubini_study_step),
        )
    G, H, g, active_count = _validated_inputs(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=active_coordinate_count,
    )
    if not primitive_symmetry_certified:
        return _empty_result(
            reason=str(
                primitive_symmetry_telemetry.get(
                    "primitive_hessian_symmetry_reason",
                    "primitive_hessian_antisymmetry_exceeds_uncertainty",
                )
            ),
            dimension=int(g.size),
            active_coordinate_count=active_count,
            telemetry={
                "joint_linear_solve_policy_requested": str(resolved.policy),
                "joint_linear_solve_policy_effective": str(resolved.policy),
                "joint_coordinate_count": int(g.size),
                "active_coordinate_count": active_count,
                "batch_coordinate_count": int(g.size - active_count),
                "classical_quantum_query_charge": 0,
                **primitive_symmetry_telemetry,
            },
        )
    solvers: dict[str, Callable[..., JointLinearSolveResult]] = {
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2: (
            _solve_supported_metric_global_trust
        ),
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1: (
            _solve_supported_metric_projected_generalized_trust
        ),
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1: (
            _solve_supported_metric_whitened
        ),
        JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1: _solve_block_pinv_legacy,
    }
    result = solvers[str(resolved.policy)](
        G=G,
        H=H,
        g=g,
        active_count=active_count,
        config=resolved,
    )
    if bool(
        result.telemetry.get(
            "metric_stabilization_zero_ridge_fallback_attempted",
            False,
        )
    ):
        result.telemetry[
            "metric_stabilization_zero_ridge_fallback_solver_status"
        ] = "certified" if result.feasible else "rejected"
        result.telemetry[
            "metric_stabilization_zero_ridge_fallback_solver_certified"
        ] = bool(result.feasible)
        result.telemetry[
            "metric_stabilization_zero_ridge_fallback_solver_reason"
        ] = str(result.reason)
        result.telemetry["metric_stabilization_status"] = (
            "resolved" if result.feasible else "unresolved"
        )
        result.telemetry["metric_stabilization_reason"] = (
            "zero_ridge_fallback_solver_certified"
            if result.feasible
            else "zero_ridge_fallback_solver_rejected"
        )
    if primitive_symmetry_telemetry:
        result.telemetry.update(primitive_symmetry_telemetry)
    return result


__all__ = [
    "JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1",
    "JOINT_LINEAR_SOLVE_POLICIES",
    "JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2",
    "JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1",
    "JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1",
    "JointLinearSolveConfig",
    "JointLinearSolveResult",
    "SupportedMetricWhitening",
    "factor_supported_metric",
    "solve_joint_linear_model",
]
