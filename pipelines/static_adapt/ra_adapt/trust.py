"""Source-Gram, no-endpoint-overlap trust transactions for RA-ADAPT."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import pipelines.static_adapt.ra_adapt.support as ra_support


SOURCE_GRAM_NO_OVERLAP_TRUST_SCHEMA = (
    "ra_adapt_source_gram_no_overlap_trust_receipt_v1"
)
DELEGATE_TRUST_TRANSACTION_SCHEMA = (
    "sr_projected_source_metric_accepted_path_transaction_v1"
)
PROJECTED_GENERALIZED_SOLVE_POLICY = (
    "supported_metric_projected_generalized_trust_v1"
)


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SourceGramNoOverlapTrustReceipt:
    """Typed accepted-path displacement receipt shared by both adapters."""

    adapter_id: str
    support_provenance_id: str
    retained_mask: tuple[bool, ...]
    supported_rank: int
    raw_metric_eigenvalue_reconstruction_residual: float
    raw_metric_eigenvalue_reconstruction_tolerance: float
    predicted_displacement: float
    predicted_displacement_sq: float
    realized_displacement: float
    realized_displacement_sq: float
    predicted_displacement_crosscheck_tolerance: float
    predicted_joint_step: tuple[float, ...]
    accepted_realized_joint_step: tuple[float, ...]
    certified_trust_radius_sq: float
    branch_trust_radius_before: float
    trust_radius_sq_match_tolerance: float
    receipt_provenance_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_GRAM_NO_OVERLAP_TRUST_SCHEMA,
            "delegate_schema": DELEGATE_TRUST_TRANSACTION_SCHEMA,
            "adapter_id": str(self.adapter_id),
            "joint_linear_solve_policy": PROJECTED_GENERALIZED_SOLVE_POLICY,
            "supported_metric_projection_provenance_id": str(
                self.support_provenance_id
            ),
            "metric_retained_mask": [
                bool(value) for value in self.retained_mask
            ],
            "supported_rank": int(self.supported_rank),
            "raw_metric_eigenvalue_reconstruction_residual": float(
                self.raw_metric_eigenvalue_reconstruction_residual
            ),
            "raw_metric_eigenvalue_reconstruction_tolerance": float(
                self.raw_metric_eigenvalue_reconstruction_tolerance
            ),
            "predicted_source_metric_displacement": float(
                self.predicted_displacement
            ),
            "predicted_source_metric_displacement_sq": float(
                self.predicted_displacement_sq
            ),
            "realized_source_metric_displacement": float(
                self.realized_displacement
            ),
            "realized_source_metric_displacement_sq": float(
                self.realized_displacement_sq
            ),
            "predicted_displacement_crosscheck_tolerance": float(
                self.predicted_displacement_crosscheck_tolerance
            ),
            "predicted_joint_step": [
                float(value) for value in self.predicted_joint_step
            ],
            "accepted_realized_joint_step": [
                float(value) for value in self.accepted_realized_joint_step
            ],
            "certified_trust_radius_sq": float(
                self.certified_trust_radius_sq
            ),
            "branch_trust_radius_before": float(
                self.branch_trust_radius_before
            ),
            "trust_radius_sq_match_tolerance": float(
                self.trust_radius_sq_match_tolerance
            ),
            "adaptive_radius_rescale_authority": (
                "supported_source_gram_parameter_displacement_v1"
            ),
            "supported_metric_whitening_active": False,
            "supported_metric_inverse_sqrt_constructed": False,
            "endpoint_overlap_required": False,
            "endpoint_overlap_query_charge": 0,
            "incremental_quantum_query_charge": 0,
            "transaction_complete": True,
            "receipt_provenance_id": str(self.receipt_provenance_id),
        }


def source_gram_no_overlap_trust_receipt_from_mapping(
    transaction: Mapping[str, Any],
    *,
    adapter_id: str,
) -> SourceGramNoOverlapTrustReceipt:
    """Validate and type one actual accepted-path trust transaction."""

    resolved_adapter_id = str(adapter_id)
    if not resolved_adapter_id:
        raise ValueError("adapter_id must be nonempty.")
    if not isinstance(transaction, Mapping):
        raise TypeError("Source-Gram trust transaction must be a mapping.")
    payload = dict(transaction)
    if str(payload.get("schema", "")) != DELEGATE_TRUST_TRANSACTION_SCHEMA:
        raise ValueError("Delegated source-Gram trust schema is invalid.")
    if (
        str(payload.get("joint_linear_solve_policy", ""))
        != PROJECTED_GENERALIZED_SOLVE_POLICY
    ):
        raise ValueError(
            "Delegated source-Gram trust solve policy is invalid."
        )
    if payload.get("endpoint_overlap_required") is not False:
        raise ValueError(
            "Canonical RA-ADAPT trust must not require endpoint overlap."
        )
    try:
        endpoint_overlap_charge = float(
            payload.get("endpoint_overlap_query_charge", -1)
        )
    except (TypeError, ValueError):
        endpoint_overlap_charge = math.nan
    if (
        not math.isfinite(endpoint_overlap_charge)
        or endpoint_overlap_charge != 0.0
    ):
        raise ValueError(
            "Canonical RA-ADAPT trust endpoint-overlap query charge must be zero."
        )
    if payload.get("transaction_complete") is not True:
        raise ValueError("Delegated source-Gram trust transaction is incomplete.")

    provenance_payload = {
        **payload,
        "schema": SOURCE_GRAM_NO_OVERLAP_TRUST_SCHEMA,
        "delegate_schema": DELEGATE_TRUST_TRANSACTION_SCHEMA,
        "adapter_id": resolved_adapter_id,
        "incremental_quantum_query_charge": 0,
    }
    receipt_id = _canonical_digest(provenance_payload)
    try:
        return SourceGramNoOverlapTrustReceipt(
            adapter_id=resolved_adapter_id,
            support_provenance_id=str(
                payload["supported_metric_projection_provenance_id"]
            ),
            retained_mask=tuple(
                bool(value) for value in payload["metric_retained_mask"]
            ),
            supported_rank=int(payload["supported_rank"]),
            raw_metric_eigenvalue_reconstruction_residual=float(
                payload["raw_metric_eigenvalue_reconstruction_residual"]
            ),
            raw_metric_eigenvalue_reconstruction_tolerance=float(
                payload["raw_metric_eigenvalue_reconstruction_tolerance"]
            ),
            predicted_displacement=float(
                payload["predicted_source_metric_displacement"]
            ),
            predicted_displacement_sq=float(
                payload["predicted_source_metric_displacement_sq"]
            ),
            realized_displacement=float(
                payload["realized_source_metric_displacement"]
            ),
            realized_displacement_sq=float(
                payload["realized_source_metric_displacement_sq"]
            ),
            predicted_displacement_crosscheck_tolerance=float(
                payload["predicted_displacement_crosscheck_tolerance"]
            ),
            predicted_joint_step=tuple(
                float(value) for value in payload["predicted_joint_step"]
            ),
            accepted_realized_joint_step=tuple(
                float(value)
                for value in payload["accepted_realized_joint_step"]
            ),
            certified_trust_radius_sq=float(
                payload["certified_trust_radius_sq"]
            ),
            branch_trust_radius_before=float(
                payload["branch_trust_radius_before"]
            ),
            trust_radius_sq_match_tolerance=float(
                payload["trust_radius_sq_match_tolerance"]
            ),
            receipt_provenance_id=receipt_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Delegated source-Gram trust transaction is incomplete."
        ) from exc


def build_source_gram_no_overlap_trust_transaction(
    selector_summary: Mapping[str, Any],
    *,
    realized_joint_step: Sequence[float],
    radius_before: float,
    adapter_id: str,
) -> SourceGramNoOverlapTrustReceipt:
    """Build one query-neutral accepted-path trust transaction.

    The numerical transaction is delegated to the proven singleton helper.
    This owner fixes the policy for both RA-ADAPT adapters and fails closed if
    the delegated receipt indicates any endpoint-overlap acquisition.
    """

    resolved_adapter_id = str(adapter_id)
    if not resolved_adapter_id:
        raise ValueError("adapter_id must be nonempty.")
    radius = float(radius_before)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_before must be finite and positive.")

    raw_support = selector_summary.get("retained_support_receipt")
    if not isinstance(raw_support, Mapping):
        raise ValueError(
            "Source-Gram trust requires the selector retained-support receipt."
        )
    support_receipt = ra_support.validate_retained_support_receipt(
        raw_support
    )
    if float(support_receipt.metric_regularization) != 0.0:
        raise ValueError(
            "Source-Gram trust requires a raw, zero-ridge selector support."
        )
    summary_provenance = str(
        selector_summary.get(
            "supported_metric_projection_provenance_id",
            "",
        )
    )
    if (
        not summary_provenance
        or summary_provenance
        != support_receipt.factorization_provenance_id
    ):
        raise ValueError(
            "Source-Gram trust support provenance disagrees with Phase III."
        )
    if tuple(
        bool(value)
        for value in selector_summary.get("metric_retained_mask", ())
    ) != support_receipt.retained_mask:
        raise ValueError(
            "Source-Gram trust support mask disagrees with Phase III."
        )
    if tuple(
        float(value)
        for value in selector_summary.get("raw_metric_eigenvalues", ())
    ) != support_receipt.raw_metric_eigenvalues:
        raise ValueError(
            "Source-Gram trust spectrum disagrees with Phase III."
        )

    # Lazy import avoids coupling route_a_trust_region's compatibility layer
    # back into this neutral RA-ADAPT owner during module initialization.
    from pipelines.static_adapt.route_a_trust_region import (
        _sr_projected_source_metric_trust_transaction,
    )

    transaction = _sr_projected_source_metric_trust_transaction(
        selector_summary,
        realized_joint_step=realized_joint_step,
        radius_before=radius,
    )
    if transaction is None:
        raise ValueError(
            "Source-Gram no-overlap trust requires the projected generalized "
            "Phase-III solve policy."
        )
    return source_gram_no_overlap_trust_receipt_from_mapping(
        transaction,
        adapter_id=resolved_adapter_id,
    )


# Short public spelling for callers that already describe their operation as a
# transaction.  Both names return the same typed receipt.
source_gram_no_overlap_trust_transaction = (
    build_source_gram_no_overlap_trust_transaction
)


__all__ = [
    "DELEGATE_TRUST_TRANSACTION_SCHEMA",
    "PROJECTED_GENERALIZED_SOLVE_POLICY",
    "SOURCE_GRAM_NO_OVERLAP_TRUST_SCHEMA",
    "SourceGramNoOverlapTrustReceipt",
    "build_source_gram_no_overlap_trust_transaction",
    "source_gram_no_overlap_trust_receipt_from_mapping",
    "source_gram_no_overlap_trust_transaction",
]
