"""Pure standard-ADAPT Phase-0 ranking for global singleton candidates.

This module deliberately has no selector-feature, metric, lane, or compile-cost
dependencies.  The numerical runtime supplies an already measured gradient
surface; Phase 0 only orders generator identities by ``abs(gradient)`` and
retains a fixed prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY = (
    "global_singleton_absolute_gradient_shortlist_v1"
)
GLOBAL_SINGLETON_GRADIENT_PHASE0_RECEIPT_SCHEMA = (
    "paper_i_global_singleton_gradient_phase0_receipt_v1"
)
GLOBAL_SINGLETON_GRADIENT_PHASE0_CONSUMER_SCOPE = (
    "phase0_global_singleton_gradient_surface"
)
MACRO_GRADIENT_PHASE0_RECEIPT_SCHEMA = (
    "paper_i_macro_gradient_phase0_receipt_v1"
)
MACRO_GRADIENT_PHASE0_CONSUMER_SCOPE = "phase0_macro_gradient_surface"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class GlobalSingletonGradientPhase0Shortlist:
    """One deterministic generator-level ``|gradient|`` shortlist."""

    input_indices: tuple[int, ...]
    ranked_indices: tuple[int, ...]
    retained_indices: tuple[int, ...]
    shortlist_size: int
    signed_gradients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.input_indices:
            raise ValueError("Phase-0 input must be non-empty.")
        if len(set(self.input_indices)) != len(self.input_indices):
            raise ValueError("Phase-0 input indices must be unique.")
        if set(self.ranked_indices) != set(self.input_indices):
            raise ValueError("Phase-0 ranking must cover its input exactly once.")
        if not 1 <= int(self.shortlist_size) <= len(self.input_indices):
            raise ValueError("Phase-0 shortlist size is out of range.")
        if self.retained_indices != self.ranked_indices[: self.shortlist_size]:
            raise ValueError("Phase-0 retained indices must be a ranked prefix.")
        if len(self.signed_gradients) != len(self.input_indices):
            raise ValueError("Phase-0 gradient vector does not cover its input.")
        if any(not math.isfinite(value) for value in self.signed_gradients):
            raise ValueError("Phase-0 gradients must be finite.")


def rank_candidates_by_absolute_gradient(
    *,
    available_indices: Sequence[int] | set[int] | frozenset[int],
    gradients: Sequence[float],
    shortlist_size: int,
) -> GlobalSingletonGradientPhase0Shortlist:
    """Rank one current generator population by standard ADAPT ``|g|``.

    Ties are resolved by numerical-pool index.  No threshold, metric,
    uncertainty, cost, family, lane, or insertion-position term participates.
    """

    cap = int(shortlist_size)
    if cap < 1:
        raise ValueError("Phase-0 shortlist size must be positive.")
    input_indices = tuple(sorted(int(value) for value in available_indices))
    if not input_indices:
        raise ValueError("Phase-0 requires an available singleton population.")
    if len(set(input_indices)) != len(input_indices):
        raise ValueError("Phase-0 available indices must be unique.")
    gradient_values = tuple(float(value) for value in gradients)
    if any(index < 0 or index >= len(gradient_values) for index in input_indices):
        raise ValueError("Phase-0 available index lies outside the gradient vector.")
    signed = tuple(gradient_values[index] for index in input_indices)
    if any(not math.isfinite(value) for value in signed):
        raise ValueError("Phase-0 gradients must be finite.")
    ranked = tuple(
        index
        for _negative_magnitude, index in sorted(
            (-abs(gradient_values[index]), index) for index in input_indices
        )
    )
    effective_size = min(cap, len(ranked))
    return GlobalSingletonGradientPhase0Shortlist(
        input_indices=input_indices,
        ranked_indices=ranked,
        retained_indices=ranked[:effective_size],
        shortlist_size=effective_size,
        signed_gradients=signed,
    )


def rank_global_singletons_by_absolute_gradient(
    *,
    available_indices: Sequence[int] | set[int] | frozenset[int],
    gradients: Sequence[float],
    shortlist_size: int,
) -> GlobalSingletonGradientPhase0Shortlist:
    """Global-singleton spelling of the representation-neutral ranker."""

    return rank_candidates_by_absolute_gradient(
        available_indices=available_indices,
        gradients=gradients,
        shortlist_size=shortlist_size,
    )


def build_absolute_gradient_phase0_receipt(
    *,
    shortlist: GlobalSingletonGradientPhase0Shortlist,
    pool_labels: Sequence[str],
    requested_shortlist_size: int,
    estimator_occurrences: Sequence[Mapping[str, Any]],
    schema: str,
    policy: str,
    population_scope: str,
    consumer_scope: str,
) -> dict[str, Any]:
    """Close one Phase-0 ranking against its exact gradient-only ledger slice."""

    occurrences = [dict(row) for row in estimator_occurrences]
    requested_cap = int(requested_shortlist_size)
    if requested_cap < 1 or int(shortlist.shortlist_size) != min(
        requested_cap,
        len(shortlist.input_indices),
    ):
        raise ValueError(
            "Phase-0 requested cap does not match the retained ranked prefix."
        )
    if len(occurrences) != len(shortlist.input_indices):
        raise RuntimeError(
            "Phase-0 estimator occurrences do not match the input population."
        )
    if any(
        row.get("component") != "N_grad"
        or row.get("consumer_scope")
        != str(consumer_scope)
        or not isinstance(row.get("sequence"), int)
        or not str(row.get("primitive_id", ""))
        for row in occurrences
    ):
        raise RuntimeError(
            "Phase-0 performed work outside its standard-gradient surface."
        )
    event_keys = tuple(
        (int(row["sequence"]), str(row["primitive_id"]))
        for row in occurrences
    )
    if len(set(event_keys)) != len(event_keys):
        raise RuntimeError("Phase-0 estimator identities must be unique.")
    if any(index >= len(pool_labels) for index in shortlist.input_indices):
        raise ValueError("Phase-0 pool labels do not cover the input population.")

    gradient_by_index = dict(
        zip(shortlist.input_indices, shortlist.signed_gradients, strict=True)
    )
    rank_by_index = {
        index: rank
        for rank, index in enumerate(shortlist.ranked_indices, start=1)
    }
    retained = set(shortlist.retained_indices)
    ranking_rows = [
        {
            "pool_index": int(index),
            "pool_label": str(pool_labels[index]),
            "gradient_signed": float(gradient_by_index[index]),
            "gradient_abs": float(abs(gradient_by_index[index])),
            "rank": int(rank_by_index[index]),
            "retained": bool(index in retained),
        }
        for index in shortlist.ranked_indices
    ]
    event_ids = [
        f"estimator:{int(row['sequence'])}:{str(row['primitive_id'])}"
        for row in occurrences
    ]
    components = {
        "N_H_outer": 0,
        "N_H_refit": 0,
        "N_grad": len(occurrences),
        "N_metric": 0,
    }
    payload: dict[str, Any] = {
        "schema": str(schema),
        "policy": str(policy),
        "population_scope": str(population_scope),
        "score": "absolute_coordinate_energy_gradient_v1",
        "ranking_order": "descending_absolute_gradient_then_pool_index_v1",
        "metric_policy": "off",
        "compile_cost_policy": "off",
        "measurement_cost_policy": "off",
        "lane_policy": "single_global_population_v1",
        "insertion_position_scope": "append_endpoint_generator_scout_v1",
        "requested_shortlist_size": requested_cap,
        "effective_shortlist_size": int(shortlist.shortlist_size),
        "input_candidate_count": len(shortlist.input_indices),
        "retained_candidate_count": len(shortlist.retained_indices),
        "input_pool_indices": list(shortlist.input_indices),
        "retained_pool_indices": list(shortlist.retained_indices),
        "ranking": ranking_rows,
        "estimator_event_ids": event_ids,
        "estimator_accounting": {
            "unit": "executed_logical_scalar_estimator_invocation",
            "components": components,
            **components,
            "S_alg": int(sum(components.values())),
            "zero_metric_measurements": True,
        },
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def build_global_singleton_gradient_phase0_receipt(
    *,
    shortlist: GlobalSingletonGradientPhase0Shortlist,
    pool_labels: Sequence[str],
    requested_shortlist_size: int,
    estimator_occurrences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the named global-singleton Phase-0 receipt."""

    return build_absolute_gradient_phase0_receipt(
        shortlist=shortlist,
        pool_labels=pool_labels,
        requested_shortlist_size=requested_shortlist_size,
        estimator_occurrences=estimator_occurrences,
        schema=GLOBAL_SINGLETON_GRADIENT_PHASE0_RECEIPT_SCHEMA,
        policy=GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY,
        population_scope="current_available_global_guarded_singletons_v1",
        consumer_scope=GLOBAL_SINGLETON_GRADIENT_PHASE0_CONSUMER_SCOPE,
    )


__all__ = [
    "GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY",
    "GLOBAL_SINGLETON_GRADIENT_PHASE0_CONSUMER_SCOPE",
    "GLOBAL_SINGLETON_GRADIENT_PHASE0_RECEIPT_SCHEMA",
    "MACRO_GRADIENT_PHASE0_CONSUMER_SCOPE",
    "MACRO_GRADIENT_PHASE0_RECEIPT_SCHEMA",
    "GlobalSingletonGradientPhase0Shortlist",
    "build_absolute_gradient_phase0_receipt",
    "build_global_singleton_gradient_phase0_receipt",
    "rank_candidates_by_absolute_gradient",
    "rank_global_singletons_by_absolute_gradient",
]
