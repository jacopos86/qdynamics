"""Adaptive Phase-0 shortlisting for append-endpoint generator scores.

The module owns one deterministic seam: an already measured generator-level
append-gradient population enters, and one competitive shortlist decision
leaves.  Candidate-position construction and insertion policy are deliberately
outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sys
from typing import Any, Mapping, Sequence


ADAPTIVE_APPEND_ENDPOINT_SHORTLIST_POLICY = (
    "append_endpoint_graph_weighted_effective_competition_shortlist_v1"
)
ADAPTIVE_APPEND_ENDPOINT_SHORTLIST_RECEIPT_SCHEMA = (
    "paper_i_append_endpoint_adaptive_phase0_shortlist_receipt_v1"
)
ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_POLICY_V2 = (
    "phase0_active_score_effective_competition_shortlist_v2"
)
ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_RECEIPT_SCHEMA_V2 = (
    "paper_i_phase0_active_score_adaptive_shortlist_receipt_v2"
)


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
class AppendEndpointGeneratorScore:
    """One generator's append-endpoint gradient and positive graph cost."""

    generator_index: int
    append_gradient: float
    graph_cost: float


@dataclass(frozen=True, slots=True)
class AdaptivePhase0ActiveScore:
    """One generator and its already resolved nonnegative Phase-0 score."""

    generator_index: int
    active_score: float


@dataclass(frozen=True, slots=True)
class _AdaptiveActiveScoreRankingRow:
    generator_index: int
    active_score: float
    active_score_log: float | None
    active_score_relative_to_champion: float


@dataclass(frozen=True, slots=True)
class AdaptivePhase0ActiveScoreShortlistDecision:
    """A deterministic adaptive prefix of one explicit Phase-0 ranking."""

    ranked_generator_indices: tuple[int, ...]
    retained_generator_indices: tuple[int, ...]
    effective_shortlist_size: int
    effective_competitor_count: float
    status: str
    frontier_saturated: bool
    cap: int
    positive_score_candidate_count: int
    rounded_effective_competitor_count: int
    active_score_policy: str
    _ranking_rows: tuple[_AdaptiveActiveScoreRankingRow, ...]

    def to_receipt(self) -> dict[str, Any]:
        """Return the canonical v2 active-score shortlist decision receipt."""

        retained = set(self.retained_generator_indices)
        ranking = [
            {
                "generator_index": int(row.generator_index),
                "active_score": float(row.active_score),
                "active_score_log": row.active_score_log,
                "active_score_relative_to_champion": float(
                    row.active_score_relative_to_champion
                ),
                "rank": rank,
                "retained": row.generator_index in retained,
            }
            for rank, row in enumerate(self._ranking_rows, start=1)
        ]
        payload: dict[str, Any] = {
            "schema": ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_RECEIPT_SCHEMA_V2,
            "policy": ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_POLICY_V2,
            "population_scope": "append_endpoint_generators_v1",
            "score": str(self.active_score_policy),
            "ranking_order": "descending_active_score_then_generator_index_v2",
            "adaptive_law": "inverse_simpson_active_score_population_v2",
            "rounding_policy": "floor_n_eff_plus_one_half_v1",
            "retention_policy": (
                "ranked_prefix_with_exact_boundary_tie_closure_subject_to_"
                "hard_cap_v2"
            ),
            "tie_policy": "exact_tie_closure_cap_saturation_v1",
            "position_aware_gradient_surface": False,
            "qiskit_compile_cost_policy": "off",
            "metric_policy": "off",
            "requested_cap": int(self.cap),
            "status": str(self.status),
            "frontier_saturated": bool(self.frontier_saturated),
            "input_candidate_count": len(self._ranking_rows),
            "positive_score_candidate_count": int(
                self.positive_score_candidate_count
            ),
            "effective_competitor_count": float(
                self.effective_competitor_count
            ),
            "rounded_effective_competitor_count": int(
                self.rounded_effective_competitor_count
            ),
            "effective_shortlist_size": int(self.effective_shortlist_size),
            "ranked_generator_indices": list(self.ranked_generator_indices),
            "retained_generator_indices": list(
                self.retained_generator_indices
            ),
            "ranking": ranking,
        }
        payload["sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class _AdaptiveRankingRow:
    generator_index: int
    append_gradient: float
    graph_cost: float
    utility: float
    utility_log: float | None
    utility_relative_to_champion: float


@dataclass(frozen=True, slots=True)
class AdaptiveAppendEndpointShortlistDecision:
    """One deterministic effective-competition shortlist decision."""

    ranked_generator_indices: tuple[int, ...]
    retained_generator_indices: tuple[int, ...]
    effective_shortlist_size: int
    effective_competitor_count: float
    raw_gradient_champion_index: int | None
    weighted_utility_champion_index: int | None
    status: str
    frontier_saturated: bool
    cap: int
    positive_utility_candidate_count: int
    rounded_effective_competitor_count: int
    _ranking_rows: tuple[_AdaptiveRankingRow, ...]

    def to_receipt(self) -> dict[str, Any]:
        """Return a canonical, self-digesting decision receipt."""

        retained = set(self.retained_generator_indices)
        ranking = [
            {
                "generator_index": int(row.generator_index),
                "append_gradient_signed": float(row.append_gradient),
                "append_gradient_abs": float(abs(row.append_gradient)),
                "graph_proxy_cost": float(row.graph_cost),
                "utility": float(row.utility),
                "utility_log": row.utility_log,
                "utility_relative_to_champion": float(
                    row.utility_relative_to_champion
                ),
                "rank": rank,
                "retained": row.generator_index in retained,
            }
            for rank, row in enumerate(self._ranking_rows, start=1)
        ]
        payload: dict[str, Any] = {
            "schema": ADAPTIVE_APPEND_ENDPOINT_SHORTLIST_RECEIPT_SCHEMA,
            "policy": ADAPTIVE_APPEND_ENDPOINT_SHORTLIST_POLICY,
            "population_scope": "append_endpoint_generators_v1",
            "score": "squared_append_gradient_over_graph_proxy_cost_v1",
            "utility_representation": (
                "raw_float_plus_log_and_champion_relative_v1"
            ),
            "ranking_order": "descending_utility_then_generator_index_v1",
            "adaptive_law": "inverse_simpson_effective_population_v1",
            "rounding_policy": "floor_n_eff_plus_one_half_v1",
            "champion_policy": "raw_and_weighted_champions_v1",
            "tie_policy": "exact_tie_closure_cap_saturation_v1",
            "position_aware_gradient_surface": False,
            "graph_proxy_cost_policy": "weighted_v1",
            "qiskit_compile_cost_policy": "off",
            "metric_policy": "off",
            "requested_cap": int(self.cap),
            "status": str(self.status),
            "frontier_saturated": bool(self.frontier_saturated),
            "input_candidate_count": len(self._ranking_rows),
            "positive_utility_candidate_count": int(
                self.positive_utility_candidate_count
            ),
            "effective_competitor_count": float(
                self.effective_competitor_count
            ),
            "rounded_effective_competitor_count": int(
                self.rounded_effective_competitor_count
            ),
            "effective_shortlist_size": int(self.effective_shortlist_size),
            "raw_gradient_champion_index": self.raw_gradient_champion_index,
            "weighted_utility_champion_index": (
                self.weighted_utility_champion_index
            ),
            "ranked_generator_indices": list(self.ranked_generator_indices),
            "retained_generator_indices": list(
                self.retained_generator_indices
            ),
            "ranking": ranking,
        }
        payload["sha256"] = _canonical_sha256(payload)
        return payload


def select_adaptive_phase0_active_score_shortlist(
    population: Sequence[AdaptivePhase0ActiveScore],
    *,
    cap: int,
    active_score_policy: str,
) -> AdaptivePhase0ActiveScoreShortlistDecision:
    """Choose an adaptive ranked prefix from already resolved active scores.

    The active route owns the scientific score.  This selector only derives a
    cardinality from that score population, so enabling adaptive shortlisting
    cannot silently replace standard ``|g|`` or proxy ``|g| / K_graph``
    ranking semantics.
    """

    if isinstance(cap, bool):
        raise ValueError("Adaptive shortlist cap must be an integer, not bool.")
    cap_value = int(cap)
    if cap_value < 1:
        raise ValueError("Adaptive shortlist cap must be positive.")
    score_policy = str(active_score_policy)
    if not score_policy:
        raise ValueError("Adaptive shortlist active-score policy is required.")
    records = tuple(population)
    if not records:
        raise ValueError("Adaptive shortlist population must be non-empty.")
    indices = tuple(int(record.generator_index) for record in records)
    if len(set(indices)) != len(indices):
        raise ValueError("Adaptive shortlist generator indices must be unique.")
    score_by_index = {
        int(record.generator_index): float(record.active_score)
        for record in records
    }
    if any(
        not math.isfinite(score) or score < 0.0
        for score in score_by_index.values()
    ):
        raise ValueError(
            "Adaptive shortlist active scores must be nonnegative and finite."
        )
    log_score_by_index = {
        index: (math.log(score) if score > 0.0 else -math.inf)
        for index, score in score_by_index.items()
    }
    ranked = tuple(
        sorted(indices, key=lambda index: (-score_by_index[index], index))
    )
    positive_count = sum(
        math.isfinite(value) for value in log_score_by_index.values()
    )
    champion_score = max(score_by_index.values()) if positive_count else 0.0
    relative_by_index = {
        index: (
            score_by_index[index] / champion_score
            if score_by_index[index] > 0.0
            else 0.0
        )
        for index in indices
    }
    ranking_rows = tuple(
        _AdaptiveActiveScoreRankingRow(
            generator_index=index,
            active_score=score_by_index[index],
            active_score_log=(
                log_score_by_index[index]
                if math.isfinite(log_score_by_index[index])
                else None
            ),
            active_score_relative_to_champion=relative_by_index[index],
        )
        for index in ranked
    )
    if positive_count == 0:
        return AdaptivePhase0ActiveScoreShortlistDecision(
            ranked_generator_indices=ranked,
            retained_generator_indices=(),
            effective_shortlist_size=0,
            effective_competitor_count=0.0,
            status="stationary",
            frontier_saturated=False,
            cap=cap_value,
            positive_score_candidate_count=0,
            rounded_effective_competitor_count=0,
            active_score_policy=score_policy,
            _ranking_rows=ranking_rows,
        )

    normalized_scores = tuple(
        relative_by_index[index]
        for index in indices
        if math.isfinite(log_score_by_index[index])
    )
    normalized_total = math.fsum(normalized_scores)
    normalized_squared_total = math.fsum(
        score * score for score in normalized_scores
    )
    effective_count = (
        normalized_total * normalized_total / normalized_squared_total
    )
    rounded_effective_count = max(1, math.floor(effective_count + 0.5))
    target_size = min(cap_value, positive_count, rounded_effective_count)
    tie_saturated = False
    if target_size < positive_count:
        boundary_score = score_by_index[ranked[target_size - 1]]
        tied_end = target_size
        while (
            tied_end < positive_count
            and score_by_index[ranked[tied_end]] == boundary_score
        ):
            tied_end += 1
        if tied_end <= cap_value:
            target_size = tied_end
        elif tied_end > target_size:
            target_size = cap_value
            tie_saturated = True
    retained = ranked[:target_size]
    return AdaptivePhase0ActiveScoreShortlistDecision(
        ranked_generator_indices=ranked,
        retained_generator_indices=retained,
        effective_shortlist_size=len(retained),
        effective_competitor_count=float(effective_count),
        status="competitive",
        frontier_saturated=(
            tie_saturated
            or (
                positive_count > cap_value
                and rounded_effective_count >= cap_value
            )
        ),
        cap=cap_value,
        positive_score_candidate_count=positive_count,
        rounded_effective_competitor_count=rounded_effective_count,
        active_score_policy=score_policy,
        _ranking_rows=ranking_rows,
    )


def select_adaptive_append_endpoint_shortlist(
    population: Sequence[AppendEndpointGeneratorScore],
    *,
    cap: int,
) -> AdaptiveAppendEndpointShortlistDecision:
    """Select generators by squared append gradient per graph-proxy cost."""

    if isinstance(cap, bool):
        raise ValueError("Adaptive shortlist cap must be an integer, not bool.")
    cap_value = int(cap)
    if cap_value < 1:
        raise ValueError("Adaptive shortlist cap must be positive.")
    records = tuple(population)
    if not records:
        raise ValueError("Adaptive shortlist population must be non-empty.")
    indices = tuple(int(record.generator_index) for record in records)
    if len(set(indices)) != len(indices):
        raise ValueError("Adaptive shortlist generator indices must be unique.")
    if any(
        not math.isfinite(float(record.append_gradient))
        for record in records
    ):
        raise ValueError("Adaptive shortlist gradients must be finite.")
    if any(
        not math.isfinite(float(record.graph_cost))
        or float(record.graph_cost) <= 0.0
        for record in records
    ):
        raise ValueError("Adaptive shortlist graph costs must be positive and finite.")

    utility_by_index: dict[int, float] = {}
    log_utility_by_index: dict[int, float] = {}
    for record in records:
        gradient_abs = abs(float(record.append_gradient))
        if gradient_abs == 0.0:
            utility_by_index[int(record.generator_index)] = 0.0
            log_utility_by_index[int(record.generator_index)] = -math.inf
            continue
        log_utility = (
            2.0 * math.log(gradient_abs)
            - math.log(float(record.graph_cost))
        )
        if log_utility > math.log(sys.float_info.max):
            raise ValueError(
                "Adaptive shortlist utilities must be finite."
            )
        utility = math.exp(log_utility)
        if not math.isfinite(utility):
            raise ValueError("Adaptive shortlist utilities must be finite.")
        utility_by_index[int(record.generator_index)] = utility
        log_utility_by_index[int(record.generator_index)] = log_utility
    gradient_by_index = {
        int(record.generator_index): float(record.append_gradient)
        for record in records
    }
    ranked = tuple(
        sorted(
            indices,
            key=lambda index: (-log_utility_by_index[index], index),
        )
    )
    record_by_index = {
        int(record.generator_index): record for record in records
    }
    positive_count = sum(
        math.isfinite(log_utility) for log_utility in log_utility_by_index.values()
    )
    max_log_utility = (
        max(log_utility_by_index.values()) if positive_count else -math.inf
    )
    relative_utility_by_index = {
        index: (
            math.exp(log_utility_by_index[index] - max_log_utility)
            if math.isfinite(log_utility_by_index[index])
            else 0.0
        )
        for index in indices
    }
    ranking_rows = tuple(
        _AdaptiveRankingRow(
            generator_index=index,
            append_gradient=gradient_by_index[index],
            graph_cost=float(record_by_index[index].graph_cost),
            utility=utility_by_index[index],
            utility_log=(
                log_utility_by_index[index]
                if math.isfinite(log_utility_by_index[index])
                else None
            ),
            utility_relative_to_champion=(
                relative_utility_by_index[index]
            ),
        )
        for index in ranked
    )
    if positive_count == 0:
        return AdaptiveAppendEndpointShortlistDecision(
            ranked_generator_indices=ranked,
            retained_generator_indices=(),
            effective_shortlist_size=0,
            effective_competitor_count=0.0,
            raw_gradient_champion_index=None,
            weighted_utility_champion_index=None,
            status="stationary",
            frontier_saturated=False,
            cap=cap_value,
            positive_utility_candidate_count=0,
            rounded_effective_competitor_count=0,
            _ranking_rows=ranking_rows,
        )

    normalized_utilities = tuple(
        relative_utility_by_index[index]
        for index in indices
        if math.isfinite(log_utility_by_index[index])
    )
    normalized_total = math.fsum(normalized_utilities)
    normalized_squared_total = math.fsum(
        utility * utility for utility in normalized_utilities
    )
    effective_count = (
        normalized_total * normalized_total / normalized_squared_total
    )
    rounded_effective_count = max(1, math.floor(effective_count + 0.5))
    target_size = min(
        cap_value,
        positive_count,
        rounded_effective_count,
    )
    weighted_champion = ranked[0]
    raw_champion = min(
        indices,
        key=lambda index: (-abs(gradient_by_index[index]), index),
    )
    if raw_champion != weighted_champion and cap_value < 2:
        raise ValueError(
            "Adaptive shortlist cap cannot retain both champions."
        )
    tie_saturated = False
    if target_size < positive_count:
        boundary_log_utility = log_utility_by_index[
            ranked[target_size - 1]
        ]
        tied_end = target_size
        while (
            tied_end < positive_count
            and log_utility_by_index[ranked[tied_end]]
            == boundary_log_utility
        ):
            tied_end += 1
        if tied_end <= cap_value:
            target_size = tied_end
        elif tied_end > target_size:
            target_size = cap_value
            tie_saturated = True

    if raw_champion != weighted_champion and target_size < 2:
        target_size = min(cap_value, positive_count, 2)

    retained = list(ranked[:target_size])
    if raw_champion not in retained and target_size >= 2:
        if len(retained) < cap_value:
            retained.append(raw_champion)
        else:
            retained[-1] = raw_champion
            tie_saturated = True
        retained.sort(key=ranked.index)

    return AdaptiveAppendEndpointShortlistDecision(
        ranked_generator_indices=ranked,
        retained_generator_indices=tuple(retained),
        effective_shortlist_size=len(retained),
        effective_competitor_count=float(effective_count),
        raw_gradient_champion_index=raw_champion,
        weighted_utility_champion_index=weighted_champion,
        status="competitive",
        frontier_saturated=(
            tie_saturated
            or (
                positive_count > cap_value
                and rounded_effective_count >= cap_value
            )
        ),
        cap=cap_value,
        positive_utility_candidate_count=positive_count,
        rounded_effective_competitor_count=rounded_effective_count,
        _ranking_rows=ranking_rows,
    )


__all__ = [
    "ADAPTIVE_APPEND_ENDPOINT_SHORTLIST_POLICY",
    "ADAPTIVE_APPEND_ENDPOINT_SHORTLIST_RECEIPT_SCHEMA",
    "ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_POLICY_V2",
    "ADAPTIVE_PHASE0_ACTIVE_SCORE_SHORTLIST_RECEIPT_SCHEMA_V2",
    "AdaptiveAppendEndpointShortlistDecision",
    "AdaptivePhase0ActiveScore",
    "AdaptivePhase0ActiveScoreShortlistDecision",
    "AppendEndpointGeneratorScore",
    "select_adaptive_phase0_active_score_shortlist",
    "select_adaptive_append_endpoint_shortlist",
]
