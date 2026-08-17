"""Typed adaptive shortlisting for native RA-ADAPT Phases I--III.

The active phase owns candidate feasibility and the scientific score.  This
module consumes that already-computed, higher-is-better score population and
changes cardinality only.  It performs no estimator, metric, compiler, or
Qiskit work.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
)
from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256


ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1 = (
    "phase123_active_score_inverse_simpson_adaptive_shortlist_v1"
)
ADAPTIVE_PHASE123_SHORTLIST_RECEIPT_SCHEMA_V1 = (
    "paper_i_ra_phase123_adaptive_shortlist_receipt_v1"
)
# Phase-III no-positive policies.  The typed-terminal default is serialized
# by OMISSION so every receipt written before this field existed keeps its
# exact byte representation and digest.
ADAPTIVE_PHASE3_TYPED_TERMINAL_NO_POSITIVE_POLICY_V1 = "typed_terminal_v1"
ADAPTIVE_PHASE3_FORCED_ADMISSION_NO_POSITIVE_POLICY_V1 = (
    "forced_admission_exact_horizon_v1"
)
ADAPTIVE_PHASE3_FORCED_ADMISSION_STATUS_V1 = "forced_no_positive_admission"
_PHASE_HARD_CAPS = {
    "phase_i": 24,
    "phase_ii": 12,
    "phase_iii": 12,
}


def adaptive_phase_record_id(
    *,
    generator_id: str,
    pool_index: int,
    insertion_position: int,
) -> str:
    """Return the canonical phase-local generator-position identity."""

    generator = str(generator_id)
    pool = int(pool_index)
    position = int(insertion_position)
    if not generator or pool < 0 or position < 0:
        raise ValueError("Adaptive phase record identity is incomplete.")
    return f"{generator}|pool:{pool}|position:{position}"


@dataclass(frozen=True, slots=True)
class AdaptivePhaseCandidateScore:
    """One already-scored candidate-position identity.

    Invalid or negative score values are retained at this typed boundary so
    the selector can record their deterministic exclusion from the active,
    feasible population.
    """

    record_id: str
    pool_index: int
    insertion_position: int
    active_score: float
    tie_break_score: float

    def __post_init__(self) -> None:
        if not str(self.record_id):
            raise ValueError("Adaptive phase record_id must be non-empty.")
        if isinstance(self.pool_index, bool) or int(self.pool_index) < 0:
            raise ValueError("Adaptive phase pool_index must be nonnegative.")
        if (
            isinstance(self.insertion_position, bool)
            or int(self.insertion_position) < 0
        ):
            raise ValueError(
                "Adaptive phase insertion_position must be nonnegative."
            )

    def to_dict(self) -> dict[str, Any]:
        def _score(value: float) -> float | str:
            numeric = float(value)
            if math.isnan(numeric):
                return "nan"
            if numeric == float("inf"):
                return "+inf"
            if numeric == float("-inf"):
                return "-inf"
            return numeric

        return {
            "record_id": str(self.record_id),
            "pool_index": int(self.pool_index),
            "insertion_position": int(self.insertion_position),
            "active_score": _score(self.active_score),
            "tie_break_score": _score(self.tie_break_score),
        }


@dataclass(frozen=True, slots=True)
class AdaptivePhaseRankReceipt:
    record_id: str
    pool_index: int
    insertion_position: int
    active_score: float
    tie_break_score: float | None
    rank: int
    frontier_eligible: bool
    retained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": str(self.record_id),
            "pool_index": int(self.pool_index),
            "insertion_position": int(self.insertion_position),
            "active_score": float(self.active_score),
            "tie_break_score": (
                None
                if self.tie_break_score is None
                else float(self.tie_break_score)
            ),
            "rank": int(self.rank),
            "frontier_eligible": bool(self.frontier_eligible),
            "retained": bool(self.retained),
        }


@dataclass(frozen=True, slots=True)
class AdaptivePhaseShortlistReceipt:
    schema: str
    policy: str
    phase: str
    score_key: str
    score_direction: str
    tie_break_policy: str
    threshold: float | None
    frontier_ratio: float
    frontier_ratio_role: str
    frontier_policy: str
    hard_cap: int
    input_population_count: int
    input_scores: tuple[AdaptivePhaseCandidateScore, ...]
    active_feasible_record_ids: tuple[str, ...]
    excluded_record_ids: tuple[str, ...]
    positive_score_record_ids: tuple[str, ...]
    frontier_eligible_record_ids: tuple[str, ...]
    effective_population_score_record_ids: tuple[str, ...]
    effective_population_size: float
    rounded_effective_population_size: int
    requested_prefix_count: int
    retained_record_ids: tuple[str, ...]
    retained_count: int
    boundary_tie_score: float | None
    boundary_tie_shell_size: int
    saturated: bool
    saturation_reason: str | None
    status: str
    qiskit_work_performed: bool
    metric_work_performed: bool
    compile_work_in_s_alg: bool
    ranking: tuple[AdaptivePhaseRankReceipt, ...]
    no_positive_policy: str = ADAPTIVE_PHASE3_TYPED_TERMINAL_NO_POSITIVE_POLICY_V1

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": str(self.schema),
            "policy": str(self.policy),
            "phase": str(self.phase),
            "score_key": str(self.score_key),
            "score_direction": str(self.score_direction),
            "tie_break_policy": str(self.tie_break_policy),
            "threshold": (
                None if self.threshold is None else float(self.threshold)
            ),
            "frontier_ratio": float(self.frontier_ratio),
            "frontier_ratio_role": str(self.frontier_ratio_role),
            "frontier_policy": str(self.frontier_policy),
            "hard_cap": int(self.hard_cap),
            "input_population_count": int(self.input_population_count),
            "input_scores": [row.to_dict() for row in self.input_scores],
            "active_feasible_record_ids": list(
                self.active_feasible_record_ids
            ),
            "excluded_record_ids": list(self.excluded_record_ids),
            "positive_score_record_ids": list(
                self.positive_score_record_ids
            ),
            "frontier_eligible_record_ids": list(
                self.frontier_eligible_record_ids
            ),
            "effective_population_score_record_ids": list(
                self.effective_population_score_record_ids
            ),
            "effective_population_size": float(
                self.effective_population_size
            ),
            "rounded_effective_population_size": int(
                self.rounded_effective_population_size
            ),
            "requested_prefix_count": int(self.requested_prefix_count),
            "retained_record_ids": list(self.retained_record_ids),
            "retained_count": int(self.retained_count),
            "boundary_tie_score": (
                None
                if self.boundary_tie_score is None
                else float(self.boundary_tie_score)
            ),
            "boundary_tie_shell_size": int(self.boundary_tie_shell_size),
            "saturated": bool(self.saturated),
            "saturation_reason": self.saturation_reason,
            "status": str(self.status),
            "qiskit_work_performed": bool(self.qiskit_work_performed),
            "metric_work_performed": bool(self.metric_work_performed),
            "compile_work_in_s_alg": bool(self.compile_work_in_s_alg),
            "ranking": [row.to_dict() for row in self.ranking],
        }
        if self.no_positive_policy != (
            ADAPTIVE_PHASE3_TYPED_TERMINAL_NO_POSITIVE_POLICY_V1
        ):
            payload["no_positive_policy"] = str(self.no_positive_policy)
        payload["sha256"] = canonical_sha256(payload)
        return payload

    @property
    def sha256(self) -> str:
        return str(self.to_dict()["sha256"])


@dataclass(frozen=True, slots=True)
class AdaptivePhaseShortlistDecision:
    ranked_record_ids: tuple[str, ...]
    retained_record_ids: tuple[str, ...]
    receipt: AdaptivePhaseShortlistReceipt


@dataclass(frozen=True, slots=True)
class AdaptivePhaseSelectionMappingReceipt:
    """Authenticated projection of one serialized live phase selection.

    This receipt binds the adaptive cardinality calculation to the ordered
    runtime population, the actual shortlist, and (for Phase III) the final
    singleton admission.  It is intentionally derived from, rather than
    embedded in, the runtime mapping so report consumers can recompute it.
    """

    phase: str
    population_record_ids: tuple[str, ...]
    shortlist_record_ids: tuple[str, ...]
    adaptive_shortlist: AdaptivePhaseShortlistReceipt
    final_admission_record_id: str | None
    mapping_sha256: str

    @property
    def population_count(self) -> int:
        return len(self.population_record_ids)

    @property
    def adaptive_retained_count(self) -> int:
        return int(self.adaptive_shortlist.retained_count)

    @property
    def final_singleton_count(self) -> int:
        return len(self.shortlist_record_ids)


def _rank_key(
    record: AdaptivePhaseCandidateScore,
) -> tuple[float, float, int, int, str]:
    tie_break = float(record.tie_break_score)
    return (
        -float(record.active_score),
        -tie_break if math.isfinite(tie_break) else math.inf,
        int(record.pool_index),
        int(record.insertion_position),
        str(record.record_id),
    )


def _frontier_prefix(
    ranked_positive: Sequence[AdaptivePhaseCandidateScore],
    *,
    frontier_ratio: float,
    score_epsilon: float = 1.0e-12,
) -> tuple[AdaptivePhaseCandidateScore, ...]:
    if not ranked_positive:
        return ()
    ratio = float(frontier_ratio)
    if not 0.0 < ratio < 1.0:
        return tuple(ranked_positive)
    take = len(ranked_positive)
    for index in range(len(ranked_positive) - 1):
        current = float(ranked_positive[index].active_score)
        following = float(ranked_positive[index + 1].active_score)
        adjacent_ratio = float(
            (following + float(score_epsilon))
            / (current + float(score_epsilon))
        )
        if adjacent_ratio <= ratio:
            take = index + 1
            break
    return tuple(ranked_positive[:take])


def select_adaptive_phase_shortlist(
    population: Sequence[AdaptivePhaseCandidateScore],
    *,
    phase: str,
    score_key: str,
    hard_cap: int,
    threshold: float,
    frontier_ratio: float,
    no_positive_policy: str = (
        ADAPTIVE_PHASE3_TYPED_TERMINAL_NO_POSITIVE_POLICY_V1
    ),
) -> AdaptivePhaseShortlistDecision:
    """Select one deterministic prefix using the exact active phase score."""

    phase_key = str(phase)
    if phase_key not in _PHASE_HARD_CAPS:
        raise ValueError("Adaptive Phase-shortlist phase is unknown.")
    policy_key = str(no_positive_policy)
    if policy_key not in {
        ADAPTIVE_PHASE3_TYPED_TERMINAL_NO_POSITIVE_POLICY_V1,
        ADAPTIVE_PHASE3_FORCED_ADMISSION_NO_POSITIVE_POLICY_V1,
    }:
        raise ValueError("Adaptive Phase-shortlist no-positive policy is unknown.")
    forced_policy = bool(
        phase_key == "phase_iii"
        and policy_key
        == ADAPTIVE_PHASE3_FORCED_ADMISSION_NO_POSITIVE_POLICY_V1
    )
    if (
        policy_key != ADAPTIVE_PHASE3_TYPED_TERMINAL_NO_POSITIVE_POLICY_V1
        and phase_key != "phase_iii"
    ):
        raise ValueError(
            "Forced no-positive admission is a Phase-III-only policy."
        )
    if isinstance(hard_cap, bool) or int(hard_cap) != _PHASE_HARD_CAPS[phase_key]:
        raise ValueError(
            f"Adaptive {phase_key} hard cap must be "
            f"{_PHASE_HARD_CAPS[phase_key]}."
        )
    if not str(score_key):
        raise ValueError("Adaptive Phase-shortlist score key is required.")
    threshold_value = float(threshold)
    threshold_receipt = (
        threshold_value if math.isfinite(threshold_value) else None
    )
    frontier = float(frontier_ratio)
    if not math.isfinite(frontier) or not 0.0 <= frontier <= 1.0:
        raise ValueError("Adaptive Phase-shortlist frontier ratio is invalid.")
    records = tuple(population)
    if not records:
        raise ValueError("Adaptive Phase-shortlist population must be non-empty.")
    record_ids = tuple(str(record.record_id) for record in records)
    if len(set(record_ids)) != len(record_ids):
        raise ValueError(
            "Adaptive Phase-shortlist record identities must be unique."
        )

    active_feasible = tuple(
        record
        for record in records
        if math.isfinite(float(record.active_score))
        and (forced_policy or float(record.active_score) >= 0.0)
        and (
            not math.isfinite(threshold_value)
            or float(record.active_score) >= threshold_value
            or forced_policy
        )
    )
    excluded_ids = tuple(
        str(record.record_id)
        for record in records
        if record not in active_feasible
    )
    ranked = tuple(sorted(active_feasible, key=_rank_key))
    positive = tuple(
        record for record in ranked if float(record.active_score) > 0.0
    )
    frontier_eligible = _frontier_prefix(
        positive,
        frontier_ratio=frontier,
    )

    effective_size = 0.0
    rounded_effective_size = 0
    requested_prefix = 0
    retained: tuple[AdaptivePhaseCandidateScore, ...] = ()
    boundary_score: float | None = None
    boundary_shell_size = 0
    saturated = False
    saturation_reason: str | None = None
    if frontier_eligible:
        champion = float(frontier_eligible[0].active_score)
        normalized = tuple(
            float(record.active_score) / champion
            for record in frontier_eligible
        )
        normalized_sum = math.fsum(normalized)
        normalized_square_sum = math.fsum(
            value * value for value in normalized
        )
        effective_size = float(
            normalized_sum * normalized_sum / normalized_square_sum
        )
        rounded_effective_size = max(
            1,
            int(math.floor(effective_size + 0.5)),
        )
        requested_prefix = min(
            int(hard_cap),
            len(frontier_eligible),
            rounded_effective_size,
        )
        target = int(requested_prefix)
        boundary_score = float(frontier_eligible[target - 1].active_score)
        shell_start = target - 1
        while (
            shell_start > 0
            and float(frontier_eligible[shell_start - 1].active_score)
            == boundary_score
        ):
            shell_start -= 1
        shell_end = target
        while (
            shell_end < len(frontier_eligible)
            and float(frontier_eligible[shell_end].active_score)
            == boundary_score
        ):
            shell_end += 1
        boundary_shell_size = shell_end - shell_start
        if shell_end > target:
            if shell_end <= int(hard_cap):
                target = shell_end
            else:
                target = int(hard_cap)
                saturated = True
                saturation_reason = (
                    "exact_boundary_tie_shell_exceeds_hard_cap"
                )
        elif (
            len(frontier_eligible) > int(hard_cap)
            and rounded_effective_size >= int(hard_cap)
        ):
            saturated = True
            saturation_reason = "hard_cap_reached_before_population_exhausted"
        retained = tuple(frontier_eligible[:target])

    forced_admission = False
    if forced_policy and not retained and ranked:
        # No candidate carries a positive score: force-admit the single
        # deterministic argmax of the signed score (zeros rank above
        # negatives; ties break by the standard rank key) instead of
        # declaring the typed no-positive terminal.
        retained = (ranked[0],)
        forced_admission = True

    retained_ids = tuple(str(record.record_id) for record in retained)
    eligible_ids = {
        str(record.record_id) for record in frontier_eligible
    }
    retained_id_set = set(retained_ids)
    ranking = tuple(
        AdaptivePhaseRankReceipt(
            record_id=str(record.record_id),
            pool_index=int(record.pool_index),
            insertion_position=int(record.insertion_position),
            active_score=float(record.active_score),
            tie_break_score=(
                float(record.tie_break_score)
                if math.isfinite(float(record.tie_break_score))
                else None
            ),
            rank=rank,
            frontier_eligible=str(record.record_id) in eligible_ids,
            retained=str(record.record_id) in retained_id_set,
        )
        for rank, record in enumerate(ranked, start=1)
    )
    receipt = AdaptivePhaseShortlistReceipt(
        schema=ADAPTIVE_PHASE123_SHORTLIST_RECEIPT_SCHEMA_V1,
        policy=ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1,
        phase=phase_key,
        score_key=str(score_key),
        score_direction="higher_is_better",
        tie_break_policy=(
            "descending_tie_break_then_pool_position_record_identity_v1"
        ),
        threshold=threshold_receipt,
        frontier_ratio=frontier,
        frontier_ratio_role="eligibility_only",
        frontier_policy="first_adjacent_score_ratio_drop_prefix_v1",
        hard_cap=int(hard_cap),
        input_population_count=len(records),
        input_scores=records,
        active_feasible_record_ids=tuple(
            str(record.record_id) for record in ranked
        ),
        excluded_record_ids=excluded_ids,
        positive_score_record_ids=tuple(
            str(record.record_id) for record in positive
        ),
        frontier_eligible_record_ids=tuple(
            str(record.record_id) for record in frontier_eligible
        ),
        effective_population_score_record_ids=tuple(
            str(record.record_id) for record in frontier_eligible
        ),
        effective_population_size=effective_size,
        rounded_effective_population_size=rounded_effective_size,
        requested_prefix_count=requested_prefix,
        retained_record_ids=retained_ids,
        retained_count=len(retained_ids),
        boundary_tie_score=boundary_score,
        boundary_tie_shell_size=boundary_shell_size,
        saturated=saturated,
        saturation_reason=saturation_reason,
        status=(
            ADAPTIVE_PHASE3_FORCED_ADMISSION_STATUS_V1
            if forced_admission
            else ("competitive" if retained else "no_positive_population")
        ),
        qiskit_work_performed=False,
        metric_work_performed=False,
        compile_work_in_s_alg=False,
        ranking=ranking,
        no_positive_policy=policy_key,
    )
    return AdaptivePhaseShortlistDecision(
        ranked_record_ids=tuple(
            str(record.record_id) for record in ranked
        ),
        retained_record_ids=retained_ids,
        receipt=receipt,
    )


def validate_adaptive_phase_shortlist_receipt(
    receipt: AdaptivePhaseShortlistReceipt,
    population: Sequence[AdaptivePhaseCandidateScore] | None = None,
) -> AdaptivePhaseShortlistReceipt:
    """Deeply recompute and authenticate one typed shortlist receipt."""

    try:
        if not isinstance(receipt, AdaptivePhaseShortlistReceipt):
            raise TypeError("receipt has the wrong type")
        source_population = (
            receipt.input_scores if population is None else tuple(population)
        )
        if tuple(source_population) != receipt.input_scores:
            raise ValueError("receipt input population drifted")
        expected = select_adaptive_phase_shortlist(
            source_population,
            phase=receipt.phase,
            score_key=receipt.score_key,
            hard_cap=receipt.hard_cap,
            threshold=(
                float("-inf")
                if receipt.threshold is None
                else float(receipt.threshold)
            ),
            frontier_ratio=receipt.frontier_ratio,
            no_positive_policy=receipt.no_positive_policy,
        ).receipt
        if receipt != expected or receipt.to_dict() != expected.to_dict():
            raise ValueError("receipt content drifted")
        return receipt
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid adaptive Phase-shortlist receipt.") from exc


def adaptive_phase_shortlist_receipt_from_mapping(
    raw_receipt: Mapping[str, Any],
) -> AdaptivePhaseShortlistReceipt:
    """Rebuild and deeply authenticate one serialized shortlist receipt."""

    try:
        if not isinstance(raw_receipt, Mapping):
            raise TypeError("receipt must be a mapping")
        payload = dict(raw_receipt)
        raw_scores = payload.get("input_scores")
        if not isinstance(raw_scores, list) or not raw_scores:
            raise ValueError("receipt input scores are missing")

        def _decode(value: Any) -> float:
            if value == "nan":
                return float("nan")
            if value == "+inf":
                return float("inf")
            if value == "-inf":
                return float("-inf")
            return float(value)

        scores = tuple(
            AdaptivePhaseCandidateScore(
                record_id=str(row["record_id"]),
                pool_index=int(row["pool_index"]),
                insertion_position=int(row["insertion_position"]),
                active_score=_decode(row["active_score"]),
                tie_break_score=_decode(row["tie_break_score"]),
            )
            for row in raw_scores
            if isinstance(row, Mapping)
        )
        if len(scores) != len(raw_scores):
            raise ValueError("receipt input score row is malformed")
        rebuilt = select_adaptive_phase_shortlist(
            scores,
            phase=str(payload["phase"]),
            score_key=str(payload["score_key"]),
            hard_cap=int(payload["hard_cap"]),
            threshold=(
                float("-inf")
                if payload.get("threshold") is None
                else float(payload["threshold"])
            ),
            frontier_ratio=float(payload["frontier_ratio"]),
            no_positive_policy=str(
                payload.get(
                    "no_positive_policy",
                    ADAPTIVE_PHASE3_TYPED_TERMINAL_NO_POSITIVE_POLICY_V1,
                )
            ),
        ).receipt
        if rebuilt.to_dict() != payload:
            raise ValueError("serialized receipt failed recomputation")
        return rebuilt
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid adaptive Phase-shortlist receipt mapping.") from exc


def adaptive_phase_selection_receipt_from_mapping(
    raw_phase: Mapping[str, Any],
    *,
    expected_phase: str | None = None,
    expected_score_key: str | None = None,
    expected_hard_cap: int | None = None,
    expected_frontier_ratio: float | None = 0.9,
) -> AdaptivePhaseSelectionMappingReceipt:
    """Authenticate a serialized adaptive phase against its live evidence.

    The nested adaptive receipt is necessary but not sufficient: a detached
    receipt can be internally valid while describing a different population.
    This public seam therefore authenticates ordered population identities and
    scores, the actual retained shortlist, and the Phase-III winner together.
    """

    try:
        if not isinstance(raw_phase, Mapping):
            raise TypeError("phase selection must be a mapping")
        phase_payload = dict(raw_phase)
        phase = str(phase_payload["phase"])
        if expected_phase is not None and phase != str(expected_phase):
            raise ValueError("phase identity drifted")
        adaptive_payload = phase_payload["adaptive_shortlist"]
        if not isinstance(adaptive_payload, Mapping):
            raise TypeError("adaptive receipt must be a mapping")
        adaptive = adaptive_phase_shortlist_receipt_from_mapping(
            adaptive_payload
        )
        if adaptive.phase != phase:
            raise ValueError("adaptive receipt phase drifted")
        if (
            expected_score_key is not None
            and adaptive.score_key != str(expected_score_key)
        ):
            raise ValueError("adaptive score key drifted")
        if (
            expected_hard_cap is not None
            and adaptive.hard_cap != int(expected_hard_cap)
        ):
            raise ValueError("adaptive hard cap drifted")
        if (
            expected_frontier_ratio is not None
            and adaptive.frontier_ratio != float(expected_frontier_ratio)
        ):
            raise ValueError("adaptive frontier ratio drifted")

        raw_records = phase_payload["records"]
        raw_shortlist = phase_payload["shortlist_records"]
        raw_live_scores = phase_payload["adaptive_population_scores"]
        if (
            not isinstance(raw_records, list)
            or not raw_records
            or any(not isinstance(row, Mapping) for row in raw_records)
            or not isinstance(raw_shortlist, list)
            or any(not isinstance(row, Mapping) for row in raw_shortlist)
            or not isinstance(raw_live_scores, list)
            or any(not isinstance(row, Mapping) for row in raw_live_scores)
        ):
            raise TypeError("live phase evidence is malformed")

        def _record_ids(
            rows: Sequence[Mapping[str, Any]],
        ) -> tuple[str, ...]:
            identities: list[str] = []
            for row in rows:
                generator_id = str(row["generator_id"])
                pool_index = int(row["pool_index"])
                insertion_position = int(row["insertion_position"])
                claimed = str(row["adaptive_record_id"])
                canonical = adaptive_phase_record_id(
                    generator_id=generator_id,
                    pool_index=pool_index,
                    insertion_position=insertion_position,
                )
                if claimed != canonical:
                    raise ValueError("adaptive record identity drifted")
                identities.append(claimed)
            return tuple(identities)

        records = [dict(row) for row in raw_records]
        shortlist_records = [dict(row) for row in raw_shortlist]
        live_scores = [dict(row) for row in raw_live_scores]
        population_ids = _record_ids(records)
        shortlist_ids = _record_ids(shortlist_records)
        if len(set(population_ids)) != len(population_ids):
            raise ValueError("adaptive population identities repeat")
        if len(set(shortlist_ids)) != len(shortlist_ids):
            raise ValueError("adaptive shortlist identities repeat")
        if int(phase_payload["population_count"]) != len(records):
            raise ValueError("adaptive population count drifted")
        if int(phase_payload["shortlist_count"]) != len(shortlist_records):
            raise ValueError("adaptive shortlist count drifted")
        if phase_payload["ordered_population_sha256"] != canonical_sha256(
            records
        ):
            raise ValueError("adaptive population order digest drifted")

        receipt_score_rows = [
            score.to_dict() for score in adaptive.input_scores
        ]
        score_ids = tuple(str(row["record_id"]) for row in live_scores)
        if (
            adaptive.input_population_count != len(records)
            or live_scores != receipt_score_rows
            or score_ids != population_ids
            or phase_payload[
                "ordered_adaptive_population_scores_sha256"
            ]
            != canonical_sha256(live_scores)
        ):
            raise ValueError("adaptive live scores detached from population")
        for score_row, record_row in zip(
            live_scores,
            records,
            strict=True,
        ):
            if (
                int(score_row["pool_index"])
                != int(record_row["pool_index"])
                or int(score_row["insertion_position"])
                != int(record_row["insertion_position"])
            ):
                raise ValueError("adaptive live score coordinates drifted")

        population_by_id = {
            str(row["adaptive_record_id"]): row for row in records
        }
        if any(
            population_by_id.get(str(row["adaptive_record_id"])) != row
            for row in shortlist_records
        ):
            raise ValueError("adaptive shortlist detached from population")

        final_admission_raw = phase_payload.get(
            "final_admission_record_id"
        )
        final_admission = (
            None
            if final_admission_raw is None
            else str(final_admission_raw)
        )
        if phase in {"phase_i", "phase_ii"}:
            if (
                adaptive.status != "competitive"
                or adaptive.retained_count < 1
                or not adaptive.retained_record_ids
                or not shortlist_ids
                or shortlist_ids != adaptive.retained_record_ids
                or final_admission is not None
            ):
                raise ValueError("adaptive retained shortlist drifted")
        elif phase == "phase_iii":
            terminal_outcome = phase_payload.get("terminal_outcome")
            if adaptive.status == "no_positive_population":
                if (
                    adaptive.retained_record_ids
                    or adaptive.retained_count != 0
                    or shortlist_ids
                    or final_admission is not None
                    or terminal_outcome
                    != ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
                ):
                    raise ValueError(
                        "adaptive Phase-III terminal evidence drifted"
                    )
            elif not (
                adaptive.retained_record_ids
                and len(shortlist_ids) == 1
                and shortlist_ids[0] == adaptive.retained_record_ids[0]
                and final_admission == shortlist_ids[0]
                and terminal_outcome is None
            ):
                raise ValueError("adaptive Phase-III winner drifted")
        else:
            raise ValueError("adaptive phase identity is unknown")

        return AdaptivePhaseSelectionMappingReceipt(
            phase=phase,
            population_record_ids=population_ids,
            shortlist_record_ids=shortlist_ids,
            adaptive_shortlist=adaptive,
            final_admission_record_id=final_admission,
            mapping_sha256=canonical_sha256(phase_payload),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Invalid adaptive Phase-selection mapping."
        ) from exc


__all__ = [
    "ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1",
    "ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1",
    "ADAPTIVE_PHASE123_SHORTLIST_RECEIPT_SCHEMA_V1",
    "AdaptivePhaseCandidateScore",
    "AdaptivePhaseRankReceipt",
    "AdaptivePhaseSelectionMappingReceipt",
    "AdaptivePhaseShortlistDecision",
    "AdaptivePhaseShortlistReceipt",
    "adaptive_phase_record_id",
    "adaptive_phase_selection_receipt_from_mapping",
    "adaptive_phase_shortlist_receipt_from_mapping",
    "select_adaptive_phase_shortlist",
    "validate_adaptive_phase_shortlist_receipt",
]
