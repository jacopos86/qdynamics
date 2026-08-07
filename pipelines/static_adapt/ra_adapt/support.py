"""Canonical retained-support factorization for RA-ADAPT.

This is the neutral numerical owner used by the Phase-III selector, the
source-Gram trust transaction, and the independent full-ansatz accepted refit.
Compatibility callers may continue importing ``factor_supported_metric`` from
``joint_linear_solve``; that spelling delegates back to this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from pipelines.static_adapt.joint_linear_solve import (
        SupportedMetricWhitening,
    )


RETAINED_SUPPORT_RECEIPT_SCHEMA = "ra_adapt_retained_support_receipt_v1"
RETAINED_SUPPORT_IMPLEMENTATION = (
    "pipelines.static_adapt.ra_adapt.support.factor_retained_support"
)


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    return 0.5 * (array + array.T)


def _condition_number_from_positive(values: np.ndarray) -> float | None:
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return None
    return float(np.max(positive) / np.min(positive))


def _factorization_provenance_id(
    *,
    raw_metric: np.ndarray,
    raw_eigenvalues: np.ndarray,
    retained_mask: np.ndarray,
    retained_vectors: np.ndarray,
    rank_relative_tolerance: float,
    metric_regularization: float,
    reason: str,
) -> str:
    """Preserve the characterized factorization provenance byte contract."""

    digest = hashlib.sha256()
    digest.update(b"supported_metric_whitening_factorization_v1\0")
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
    digest.update(float(rank_relative_tolerance).hex().encode("ascii"))
    digest.update(b"\0")
    digest.update(float(metric_regularization).hex().encode("ascii"))
    digest.update(b"\0")
    digest.update(str(reason).encode("utf-8"))
    return digest.hexdigest()


def _factor_supported_metric(
    gram: np.ndarray,
    *,
    rank_relative_tolerance: float,
    metric_regularization: float,
) -> SupportedMetricWhitening:
    """Implement the one raw-metric support/ridge convention."""

    raw_metric = np.asarray(gram, dtype=float)
    if raw_metric.ndim != 2 or raw_metric.shape[0] != raw_metric.shape[1]:
        raise ValueError("gram must be a square matrix.")
    if not np.all(np.isfinite(raw_metric)):
        raise ValueError("gram must contain only finite values.")
    tolerance = float(rank_relative_tolerance)
    ridge = float(metric_regularization)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "rank_relative_tolerance must be finite and nonnegative."
        )
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("metric_regularization must be finite and nonnegative.")

    raw_metric = _symmetrize(raw_metric)
    dimension = int(raw_metric.shape[0])
    try:
        raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(raw_metric)
    except np.linalg.LinAlgError:
        raw_eigenvalues = np.zeros(dimension, dtype=float)
        raw_eigenvectors = np.eye(dimension, dtype=float)
        eigendecomposition_failed = True
    else:
        raw_eigenvalues = np.asarray(raw_eigenvalues, dtype=float)
        raw_eigenvectors = np.asarray(raw_eigenvectors, dtype=float)
        eigendecomposition_failed = False

    if raw_eigenvalues.size:
        metric_abs_scale = float(
            max(np.max(np.abs(raw_eigenvalues)), np.finfo(float).tiny)
        )
        metric_positive_max = float(max(np.max(raw_eigenvalues), 0.0))
    else:
        metric_abs_scale = float(np.finfo(float).tiny)
        metric_positive_max = 0.0
    negative_tolerance = float(
        64.0
        * np.finfo(float).eps
        * max(1, dimension)
        * metric_abs_scale
    )
    support_threshold = float(tolerance * metric_positive_max)
    retained_mask = np.asarray(
        raw_eigenvalues > support_threshold,
        dtype=bool,
    )
    retained_eigenvalues = np.asarray(
        raw_eigenvalues[retained_mask],
        dtype=float,
    )
    retained_vectors = np.asarray(
        raw_eigenvectors[:, retained_mask],
        dtype=float,
    )
    raw_positive = np.asarray(
        raw_eigenvalues[raw_eigenvalues > negative_tolerance],
        dtype=float,
    )

    reason = "supported_metric_factorization"
    feasible = True
    if eigendecomposition_failed:
        reason = "metric_eigendecomposition_failed"
        feasible = False
    elif (
        raw_eigenvalues.size
        and float(np.min(raw_eigenvalues)) < -negative_tolerance
    ):
        reason = "materially_negative_metric_eigenvalue"
        feasible = False
    elif metric_positive_max <= 0.0 or retained_eigenvalues.size == 0:
        reason = "empty_supported_metric_subspace"
        feasible = False

    denominators = retained_eigenvalues + ridge
    if feasible and np.any(denominators <= 0.0):
        reason = "nonpositive_whitening_denominator"
        feasible = False

    rank = int(retained_eigenvalues.size)
    if feasible:
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
        raw_whitened_metric = _symmetrize(
            whitening.T @ raw_metric @ whitening
        )
        regularized_supported_metric = _symmetrize(
            retained_vectors @ np.diag(denominators) @ retained_vectors.T
        )
        raw_metric_pseudoinverse = _symmetrize(
            retained_vectors
            @ np.diag(retained_eigenvalues ** -1.0)
            @ retained_vectors.T
        )
    else:
        whitening = np.zeros((dimension, rank), dtype=float)
        whitening_pseudoinverse = np.zeros((rank, dimension), dtype=float)
        raw_orthonormalizer = np.zeros((dimension, rank), dtype=float)
        regularized_to_raw_frame = np.zeros((rank, rank), dtype=float)
        raw_whitened_metric = np.zeros((rank, rank), dtype=float)
        regularized_supported_metric = np.zeros(
            (dimension, dimension),
            dtype=float,
        )
        raw_metric_pseudoinverse = np.zeros(
            (dimension, dimension),
            dtype=float,
        )

    provenance_id = _factorization_provenance_id(
        raw_metric=raw_metric,
        raw_eigenvalues=raw_eigenvalues,
        retained_mask=retained_mask,
        retained_vectors=retained_vectors,
        rank_relative_tolerance=tolerance,
        metric_regularization=ridge,
        reason=reason,
    )
    # Imported only after joint_linear_solve has initialized; this avoids a
    # module cycle while preserving the established public return type.
    from pipelines.static_adapt.joint_linear_solve import (
        SupportedMetricWhitening,
    )

    return SupportedMetricWhitening(
        feasible=bool(feasible),
        reason=str(reason),
        raw_metric=np.asarray(raw_metric, dtype=float).copy(),
        raw_eigenvalues=raw_eigenvalues.copy(),
        retained_mask=retained_mask.copy(),
        retained_eigenvalues=retained_eigenvalues.copy(),
        retained_vectors=retained_vectors.copy(),
        whitening=np.asarray(whitening, dtype=float).copy(),
        whitening_pseudoinverse=np.asarray(
            whitening_pseudoinverse,
            dtype=float,
        ).copy(),
        raw_orthonormalizer=np.asarray(
            raw_orthonormalizer,
            dtype=float,
        ).copy(),
        regularized_to_raw_frame=np.asarray(
            regularized_to_raw_frame,
            dtype=float,
        ).copy(),
        raw_whitened_metric=np.asarray(
            raw_whitened_metric,
            dtype=float,
        ).copy(),
        regularized_supported_metric=np.asarray(
            regularized_supported_metric,
            dtype=float,
        ).copy(),
        raw_metric_pseudoinverse=np.asarray(
            raw_metric_pseudoinverse,
            dtype=float,
        ).copy(),
        support_threshold=float(support_threshold),
        negative_eigenvalue_tolerance=float(negative_tolerance),
        metric_ridge=float(ridge),
        raw_condition_number=_condition_number_from_positive(raw_positive),
        retained_condition_number=_condition_number_from_positive(
            retained_eigenvalues
        ),
        provenance_id=str(provenance_id),
    )


@dataclass(frozen=True)
class RetainedSupportReceipt:
    """Deterministic description of one raw-Gram support decision."""

    feasible: bool
    reason: str
    dimension: int
    rank: int
    rank_relative_tolerance: float
    metric_regularization: float
    support_threshold: float
    negative_eigenvalue_tolerance: float
    raw_metric_eigenvalues: tuple[float, ...]
    retained_mask: tuple[bool, ...]
    retained_metric_eigenvalues: tuple[float, ...]
    retained_eigenvectors: tuple[tuple[float, ...], ...]
    raw_condition_number: float | None
    retained_condition_number: float | None
    factorization_provenance_id: str
    source_provenance_id: str | None
    receipt_provenance_id: str

    def as_dict(self) -> dict[str, Any]:
        retained_pairs = []
        for index, eigenvalue in enumerate(self.retained_metric_eigenvalues):
            retained_pairs.append(
                {
                    "eigenvalue": float(eigenvalue),
                    "eigenvector": [
                        float(row[index]) for row in self.retained_eigenvectors
                    ],
                }
            )
        return {
            "schema": RETAINED_SUPPORT_RECEIPT_SCHEMA,
            "implementation": RETAINED_SUPPORT_IMPLEMENTATION,
            "feasible": bool(self.feasible),
            "reason": str(self.reason),
            "dimension": int(self.dimension),
            "rank": int(self.rank),
            "rank_relative_tolerance": float(self.rank_relative_tolerance),
            "metric_regularization": float(self.metric_regularization),
            "support_threshold": float(self.support_threshold),
            "negative_eigenvalue_tolerance": float(
                self.negative_eigenvalue_tolerance
            ),
            "raw_metric_eigenvalues": [
                float(value) for value in self.raw_metric_eigenvalues
            ],
            "retained_mask": [bool(value) for value in self.retained_mask],
            "retained_metric_eigenvalues": [
                float(value) for value in self.retained_metric_eigenvalues
            ],
            "retained_eigenvectors": [
                [float(value) for value in row]
                for row in self.retained_eigenvectors
            ],
            "retained_eigenpairs": retained_pairs,
            "raw_condition_number": self.raw_condition_number,
            "retained_condition_number": self.retained_condition_number,
            "factorization_provenance_id": str(
                self.factorization_provenance_id
            ),
            "source_provenance_id": self.source_provenance_id,
            "classical_quantum_query_charge": 0,
            "receipt_provenance_id": str(self.receipt_provenance_id),
        }


@dataclass(frozen=True)
class RetainedSupportFactorization:
    """Numerical factorization paired with its canonical RA-ADAPT receipt."""

    factorization: SupportedMetricWhitening
    receipt: RetainedSupportReceipt


def _receipt_digest_payload(
    receipt: RetainedSupportReceipt,
) -> dict[str, Any]:
    payload = receipt.as_dict()
    payload.pop("receipt_provenance_id", None)
    # This is a human-useful redundant view, not part of the canonical digest.
    payload.pop("retained_eigenpairs", None)
    return payload


def factor_retained_support(
    gram: np.ndarray,
    *,
    rank_relative_tolerance: float = 1e-6,
    metric_regularization: float = 1e-9,
    source_provenance_id: str | None = None,
) -> RetainedSupportFactorization:
    """Factor ``gram`` once and emit the canonical retained-support receipt."""

    tolerance = float(rank_relative_tolerance)
    ridge = float(metric_regularization)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "rank_relative_tolerance must be finite and nonnegative."
        )
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("metric_regularization must be finite and nonnegative.")
    source_id = (
        None if source_provenance_id is None else str(source_provenance_id)
    )
    if source_id == "":
        raise ValueError("source_provenance_id must be nonempty when provided.")

    factorization = _factor_supported_metric(
        gram,
        rank_relative_tolerance=tolerance,
        metric_regularization=ridge,
    )
    receipt_payload: dict[str, Any] = {
        "schema": RETAINED_SUPPORT_RECEIPT_SCHEMA,
        "implementation": RETAINED_SUPPORT_IMPLEMENTATION,
        "feasible": bool(factorization.feasible),
        "reason": str(factorization.reason),
        "dimension": int(factorization.dimension),
        "rank": int(factorization.rank),
        "rank_relative_tolerance": tolerance,
        "metric_regularization": ridge,
        "support_threshold": float(factorization.support_threshold),
        "negative_eigenvalue_tolerance": float(
            factorization.negative_eigenvalue_tolerance
        ),
        "raw_metric_eigenvalues": [
            float(value) for value in factorization.raw_eigenvalues.tolist()
        ],
        "retained_mask": [
            bool(value) for value in factorization.retained_mask.tolist()
        ],
        "retained_metric_eigenvalues": [
            float(value)
            for value in factorization.retained_eigenvalues.tolist()
        ],
        "retained_eigenvectors": [
            [float(value) for value in row]
            for row in factorization.retained_vectors.tolist()
        ],
        "raw_condition_number": factorization.raw_condition_number,
        "retained_condition_number": factorization.retained_condition_number,
        "factorization_provenance_id": str(factorization.provenance_id),
        "source_provenance_id": source_id,
        "classical_quantum_query_charge": 0,
    }
    receipt_id = _canonical_digest(receipt_payload)
    receipt = RetainedSupportReceipt(
        feasible=bool(factorization.feasible),
        reason=str(factorization.reason),
        dimension=int(factorization.dimension),
        rank=int(factorization.rank),
        rank_relative_tolerance=tolerance,
        metric_regularization=ridge,
        support_threshold=float(factorization.support_threshold),
        negative_eigenvalue_tolerance=float(
            factorization.negative_eigenvalue_tolerance
        ),
        raw_metric_eigenvalues=tuple(
            float(value) for value in factorization.raw_eigenvalues.tolist()
        ),
        retained_mask=tuple(
            bool(value) for value in factorization.retained_mask.tolist()
        ),
        retained_metric_eigenvalues=tuple(
            float(value)
            for value in factorization.retained_eigenvalues.tolist()
        ),
        retained_eigenvectors=tuple(
            tuple(float(value) for value in row)
            for row in factorization.retained_vectors.tolist()
        ),
        raw_condition_number=factorization.raw_condition_number,
        retained_condition_number=factorization.retained_condition_number,
        factorization_provenance_id=str(factorization.provenance_id),
        source_provenance_id=source_id,
        receipt_provenance_id=receipt_id,
    )
    return RetainedSupportFactorization(
        factorization=factorization,
        receipt=receipt,
    )


def validate_retained_support_receipt(
    payload: Mapping[str, Any] | RetainedSupportReceipt,
) -> RetainedSupportReceipt:
    """Parse and fail closed on a selector/refit support receipt."""

    if isinstance(payload, RetainedSupportReceipt):
        receipt = payload
    elif isinstance(payload, Mapping):
        if str(payload.get("schema", "")) != RETAINED_SUPPORT_RECEIPT_SCHEMA:
            raise ValueError("Retained-support receipt schema is invalid.")
        if str(payload.get("implementation", "")) != (
            RETAINED_SUPPORT_IMPLEMENTATION
        ):
            raise ValueError("Retained-support implementation owner is invalid.")
        try:
            source_raw = payload.get("source_provenance_id")
            receipt = RetainedSupportReceipt(
                feasible=bool(payload["feasible"]),
                reason=str(payload["reason"]),
                dimension=int(payload["dimension"]),
                rank=int(payload["rank"]),
                rank_relative_tolerance=float(
                    payload["rank_relative_tolerance"]
                ),
                metric_regularization=float(
                    payload["metric_regularization"]
                ),
                support_threshold=float(payload["support_threshold"]),
                negative_eigenvalue_tolerance=float(
                    payload["negative_eigenvalue_tolerance"]
                ),
                raw_metric_eigenvalues=tuple(
                    float(value)
                    for value in payload["raw_metric_eigenvalues"]
                ),
                retained_mask=tuple(
                    bool(value) for value in payload["retained_mask"]
                ),
                retained_metric_eigenvalues=tuple(
                    float(value)
                    for value in payload["retained_metric_eigenvalues"]
                ),
                retained_eigenvectors=tuple(
                    tuple(float(value) for value in row)
                    for row in payload["retained_eigenvectors"]
                ),
                raw_condition_number=(
                    None
                    if payload.get("raw_condition_number") is None
                    else float(payload["raw_condition_number"])
                ),
                retained_condition_number=(
                    None
                    if payload.get("retained_condition_number") is None
                    else float(payload["retained_condition_number"])
                ),
                factorization_provenance_id=str(
                    payload["factorization_provenance_id"]
                ),
                source_provenance_id=(
                    None if source_raw is None else str(source_raw)
                ),
                receipt_provenance_id=str(
                    payload["receipt_provenance_id"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Retained-support receipt is incomplete or malformed."
            ) from exc
    else:
        raise TypeError("Retained-support receipt must be a mapping.")

    if receipt.dimension < 0 or receipt.rank < 0:
        raise ValueError("Retained-support dimensions must be nonnegative.")
    if len(receipt.raw_metric_eigenvalues) != receipt.dimension:
        raise ValueError("Retained-support spectrum dimension is invalid.")
    if len(receipt.retained_mask) != receipt.dimension:
        raise ValueError("Retained-support mask dimension is invalid.")
    if sum(receipt.retained_mask) != receipt.rank:
        raise ValueError("Retained-support mask and rank disagree.")
    if len(receipt.retained_metric_eigenvalues) != receipt.rank:
        raise ValueError("Retained-support retained spectrum is invalid.")
    if len(receipt.retained_eigenvectors) != receipt.dimension or any(
        len(row) != receipt.rank
        for row in receipt.retained_eigenvectors
    ):
        raise ValueError("Retained-support eigenvector shape is invalid.")
    expected_retained = tuple(
        eigenvalue
        for eigenvalue, retained in zip(
            receipt.raw_metric_eigenvalues,
            receipt.retained_mask,
        )
        if retained
    )
    if expected_retained != receipt.retained_metric_eigenvalues:
        raise ValueError("Retained-support mask and eigenvalues disagree.")
    if not receipt.factorization_provenance_id:
        raise ValueError("Retained-support provenance is missing.")
    expected_digest = _canonical_digest(_receipt_digest_payload(receipt))
    if receipt.receipt_provenance_id != expected_digest:
        raise ValueError("Retained-support receipt digest is invalid.")
    return receipt


__all__ = [
    "RETAINED_SUPPORT_IMPLEMENTATION",
    "RETAINED_SUPPORT_RECEIPT_SCHEMA",
    "RetainedSupportFactorization",
    "RetainedSupportReceipt",
    "factor_retained_support",
    "validate_retained_support_receipt",
]
