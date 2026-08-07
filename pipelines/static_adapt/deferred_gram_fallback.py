"""Typed receipts for the retained deferred-Gram robustness fallback.

This module deliberately does not own ordinary novelty scoring.  It records
only the fail-closed path used when every Phase-III energy model is infeasible
and the already-measured deferred Gram geometry is used to expand the
candidate domain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1 = (
    "deferred_gram_all_models_infeasible_fallback_v1"
)
_FALLBACK_ONLY_POLICY = "fallback_only_v1"


def deferred_gram_fallback_enabled(
    route_contract: Mapping[str, Any],
) -> bool:
    """Return whether a resolved route authorizes the retained fallback."""

    execution_settings = route_contract.get("execution_settings")
    if not isinstance(execution_settings, Mapping):
        return False
    return bool(
        str(execution_settings.get("phase2_gram_novelty_policy", ""))
        == _FALLBACK_ONLY_POLICY
        and str(execution_settings.get("phase3_gram_novelty_policy", ""))
        == _FALLBACK_ONLY_POLICY
    )


def selected_admission_deferred_gram_fallback_receipt(
    selected_records: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    controller_round: int,
) -> dict[str, Any]:
    """Build the receipt from the record or records actually admitted."""

    if int(controller_round) <= 0:
        raise ValueError("controller_round must be positive.")
    fired_records: list[dict[str, Any]] = []
    for raw_record in selected_records:
        if not isinstance(raw_record, Mapping):
            raise RuntimeError(
                "Deferred-Gram admission telemetry requires mapping records."
            )
        mode = str(raw_record.get("route_a_geometry_expansion_mode", "") or "")
        reason = str(
            raw_record.get("route_a_geometry_expansion_reason", "") or ""
        )
        if not mode:
            continue
        if not reason:
            raise RuntimeError(
                "A selected deferred-Gram expansion lacks its reason."
            )
        fired_records.append(
            {
                "mode": mode,
                "reason": reason,
                "charge": int(
                    raw_record.get(
                        "route_a_geometry_expansion_query_charge",
                        0,
                    )
                    or 0
                ),
            }
        )

    if fired_records and not bool(enabled):
        raise RuntimeError(
            "The deferred-Gram all-models-infeasible fallback fired while "
            "its resolved protocol disabled it."
        )
    if len(fired_records) > 1:
        raise RuntimeError(
            "The deferred-Gram fallback must admit at most one singleton "
            "record per controller round."
        )

    fired = bool(fired_records)
    selected = fired_records[0] if fired else None
    return {
        "schema": DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1,
        "scope": "accepted_controller_round",
        "enabled": bool(enabled),
        "fired": fired,
        "rounds": [int(controller_round)] if fired else [],
        "charge": 0 if selected is None else int(selected["charge"]),
        "mode": None if selected is None else str(selected["mode"]),
        "reason": None if selected is None else str(selected["reason"]),
    }


def summarize_deferred_gram_fallback(
    history_rows: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    allow_missing_prefix_rounds: int = 0,
) -> dict[str, Any]:
    """Close accepted-round receipts into one run-level fallback receipt."""

    missing_prefix = max(0, int(allow_missing_prefix_rounds))
    if missing_prefix > len(history_rows):
        raise ValueError(
            "allow_missing_prefix_rounds exceeds the history length."
        )
    rounds: list[int] = []
    selected_operators: list[str] = []
    reason_counts: dict[str, int] = {}
    charge = 0

    for ordinal, raw_row in enumerate(history_rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise RuntimeError(
                "Deferred-Gram run telemetry requires mapping history rows."
            )
        raw_receipt = raw_row.get(
            DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1
        )
        if not isinstance(raw_receipt, Mapping):
            if ordinal <= missing_prefix:
                continue
            raise RuntimeError(
                "An accepted round lacks its deferred-Gram fallback receipt."
            )
        receipt = dict(raw_receipt)
        if (
            str(receipt.get("schema", ""))
            != DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1
        ):
            raise RuntimeError(
                "An accepted round has an invalid deferred-Gram fallback "
                "receipt schema."
            )
        if bool(receipt.get("enabled", False)) != bool(enabled):
            raise RuntimeError(
                "Deferred-Gram fallback authorization changed within a run."
            )
        if not bool(receipt.get("fired", False)):
            if receipt.get("rounds") not in ([], ()):
                raise RuntimeError(
                    "An unfired deferred-Gram receipt recorded rounds."
                )
            if int(receipt.get("charge", 0) or 0) != 0:
                raise RuntimeError(
                    "An unfired deferred-Gram receipt recorded a charge."
                )
            continue

        receipt_rounds = receipt.get("rounds")
        if not (
            isinstance(receipt_rounds, Sequence)
            and not isinstance(receipt_rounds, (str, bytes, bytearray))
            and len(receipt_rounds) == 1
        ):
            raise RuntimeError(
                "A fired deferred-Gram round receipt must identify one round."
            )
        controller_round = int(receipt_rounds[0])
        expected_round = int(
            raw_row.get(
                "depth_cumulative",
                raw_row.get("depth", ordinal),
            )
        )
        if controller_round != expected_round:
            raise RuntimeError(
                "Deferred-Gram receipt round disagrees with accepted history."
            )
        reason = str(receipt.get("reason", "") or "")
        mode = str(receipt.get("mode", "") or "")
        if not reason or not mode:
            raise RuntimeError(
                "A fired deferred-Gram receipt lacks mode or reason."
            )
        rounds.append(controller_round)
        selected_operators.append(
            str(
                raw_row.get(
                    "selected_logical_op",
                    raw_row.get("selected_op", ""),
                )
            )
        )
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        charge += int(receipt.get("charge", 0) or 0)

    fired = bool(rounds)
    if fired and not bool(enabled):
        raise RuntimeError(
            "The deferred-Gram all-models-infeasible fallback fired while "
            "its resolved protocol disabled it."
        )
    return {
        "schema": DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1,
        "scope": "run",
        "enabled": bool(enabled),
        "fired": fired,
        "rounds": rounds,
        "charge": int(charge),
        "activation_count": len(rounds),
        "selected_operators": selected_operators,
        "reason_counts": {
            key: int(value) for key, value in sorted(reason_counts.items())
        },
        "historical_prefix_rounds_without_new_receipt": missing_prefix,
    }


__all__ = [
    "DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1",
    "deferred_gram_fallback_enabled",
    "selected_admission_deferred_gram_fallback_receipt",
    "summarize_deferred_gram_fallback",
]
