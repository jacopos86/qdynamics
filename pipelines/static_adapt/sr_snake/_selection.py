"""Private SR-SNAKE singleton-selection boundary.

The types in this module deliberately contain identities and immutable
receipts, not the live numerical objects used to acquire them.  The legacy
numerical kernel remains responsible for exact estimator execution while this
module owns the controller-level selection transaction and its invariants.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1,
    SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1,
    canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256 as _canonical_insertion_v1_selection_contract_sha256,
    canonical_sr_snake_insertion_commutation_plateau_v2_contract_sha256 as _canonical_insertion_v2_selection_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256 as _canonical_commutation_reduced_insertion_selection_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256 as _canonical_default_selection_contract_sha256,
    canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract_sha256 as _canonical_prune_selection_contract_sha256,
)


_SHORTLIST_UNIT_CANDIDATE_POSITION = "candidate_position_record"
_SHORTLIST_UNIT_MACRO_OPERATOR = "macro_operator_identity"


def _uses_default_singleton_selection(
    *,
    route_profile: str | None,
    route_profile_sha256: str | None,
    beam_enabled: bool,
) -> bool:
    """Return whether the characterized Issue-10 selection seam owns a run."""

    profile_digest = (
        str(route_profile or ""),
        str(route_profile_sha256 or ""),
    )
    supported = {
        (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
            _canonical_default_selection_contract_sha256(),
        ),
        (
            SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1,
            _canonical_insertion_v1_selection_contract_sha256(),
        ),
        (
            SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2,
            _canonical_insertion_v2_selection_contract_sha256(),
        ),
        (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1,
            _canonical_commutation_reduced_insertion_selection_contract_sha256(),
        ),
        (
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1,
            _canonical_prune_selection_contract_sha256(),
        ),
    }
    return bool(profile_digest in supported and not bool(beam_enabled))


@dataclass(frozen=True, slots=True)
class _SRControllerState:
    controller_round: int
    accepted_operator_ids: tuple[str, ...]
    accepted_insertion_positions: tuple[int, ...]
    logical_parameter_ids: tuple[str, ...]
    logical_parameter_values: tuple[float, ...]
    runtime_parameter_ids: tuple[str, ...]
    runtime_parameter_values: tuple[float, ...]
    accepted_energy: float
    accepted_state_fingerprint: str
    available_generator_ids: tuple[str, ...]
    selection_counts: tuple[tuple[str, int], ...]
    phase_live: tuple[bool, bool, bool]
    trust_state_identity: str
    optimizer_memory_identity: str
    estimator_prefix_identity: str
    admissible_domain_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.controller_round < 0:
            raise ValueError("controller_round must be non-negative")
        if len(self.accepted_operator_ids) != len(
            self.accepted_insertion_positions
        ):
            raise ValueError(
                "accepted operator and insertion-position identities disagree"
            )
        if len(self.logical_parameter_ids) != len(
            self.logical_parameter_values
        ):
            raise ValueError("logical parameter identities and values disagree")
        if len(self.runtime_parameter_ids) != len(
            self.runtime_parameter_values
        ):
            raise ValueError("runtime parameter identities and values disagree")
        if not self.accepted_state_fingerprint:
            raise ValueError("accepted_state_fingerprint must be non-empty")
        if len(set(self.available_generator_ids)) != len(
            self.available_generator_ids
        ):
            duplicate_ids = sorted(
                identity
                for identity in set(self.available_generator_ids)
                if self.available_generator_ids.count(identity) > 1
            )
            raise ValueError(
                "available generator identities must be unique; "
                f"duplicates={duplicate_ids!r}"
            )
        if len(set(self.admissible_domain_record_ids)) != len(
            self.admissible_domain_record_ids
        ):
            raise ValueError(
                "admissible domain-record identities must be unique"
            )
        if len(self.phase_live) != 3:
            raise ValueError("phase_live must identify Phase I, II, and III")


@dataclass(frozen=True, slots=True)
class _CandidatePositionRecord:
    domain_record_id: str
    generator_id: str
    parent_generator_id: str | None
    pool_index: int
    pool_label: str
    insertion_position: int
    symmetry_identity: str
    lineage_identity: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.domain_record_id:
            raise ValueError("domain_record_id must be non-empty")
        if not self.generator_id:
            raise ValueError("generator_id must be non-empty")
        if self.pool_index < 0:
            raise ValueError("pool_index must be non-negative")
        if not self.pool_label:
            raise ValueError("pool_label must be non-empty")
        if self.insertion_position < 0:
            raise ValueError("insertion_position must be non-negative")
        if not self.symmetry_identity:
            raise ValueError("symmetry_identity must be non-empty")
        if not self.lineage_identity:
            raise ValueError("lineage_identity must be non-empty")
        if self.lineage_identity[-1] != self.generator_id:
            raise ValueError(
                "lineage_identity must terminate at the record generator"
            )
        if self.parent_generator_id is None:
            if self.lineage_identity != (self.generator_id,):
                raise ValueError(
                    "root candidate lineage must contain only its generator"
                )
        elif (
            len(self.lineage_identity) < 2
            or self.lineage_identity[-2] != self.parent_generator_id
        ):
            raise ValueError(
                "candidate parent identity must immediately precede its generator"
            )


def _build_candidate_domain(
    records: Sequence[_CandidatePositionRecord],
) -> tuple[_CandidatePositionRecord, ...]:
    """Freeze an already-admissible generator-position domain in source order."""

    domain = tuple(records)
    if not domain:
        raise ValueError("singleton selection requires a non-empty domain")
    seen_ids: set[str] = set()
    for record in domain:
        if record.parent_generator_id is not None:
            raise ValueError(
                "the admissible domain must contain root generator-position records"
            )
        if record.domain_record_id in seen_ids:
            raise ValueError(
                "candidate domain record identities must be unique: "
                f"{record.domain_record_id}"
            )
        seen_ids.add(record.domain_record_id)
    return domain


@dataclass(frozen=True, slots=True)
class _ShortlistRankReceipt:
    """Recorded score rank for one shortlisted candidate-position identity."""

    record_key: tuple[str, str]
    shortlist_rank: int
    primary_score: float
    tie_break_score: float
    pool_index: int
    insertion_position: int
    shortlist_unit: str = _SHORTLIST_UNIT_CANDIDATE_POSITION
    shortlist_identity: str | None = None
    identity_rank: int | None = None
    identity_position_rank: int | None = None
    identity_position_count: int | None = None

    def __post_init__(self) -> None:
        if len(self.record_key) != 2 or any(
            not value for value in self.record_key
        ):
            raise ValueError("shortlist rank record identity must be complete")
        if self.shortlist_rank <= 0:
            raise ValueError("shortlist rank must be one-based and positive")
        if math.isnan(self.primary_score) or math.isnan(self.tie_break_score):
            raise ValueError("shortlist rank scores must not be NaN")
        if self.pool_index < 0:
            raise ValueError("shortlist rank pool index must be non-negative")
        if self.insertion_position < 0:
            raise ValueError(
                "shortlist rank insertion position must be non-negative"
            )
        if self.shortlist_unit not in {
            _SHORTLIST_UNIT_CANDIDATE_POSITION,
            _SHORTLIST_UNIT_MACRO_OPERATOR,
        }:
            raise ValueError("shortlist rank has an unknown selection unit")
        identity_fields = (
            self.shortlist_identity,
            self.identity_rank,
            self.identity_position_rank,
            self.identity_position_count,
        )
        if self.shortlist_unit == _SHORTLIST_UNIT_CANDIDATE_POSITION:
            if any(value is not None for value in identity_fields):
                raise ValueError(
                    "candidate-position rank must not claim identity ranks"
                )
        else:
            if (
                not isinstance(self.shortlist_identity, str)
                or not self.shortlist_identity
                or isinstance(self.identity_rank, bool)
                or not isinstance(self.identity_rank, int)
                or self.identity_rank < 1
                or isinstance(self.identity_position_rank, bool)
                or not isinstance(self.identity_position_rank, int)
                or self.identity_position_rank < 1
                or isinstance(self.identity_position_count, bool)
                or not isinstance(self.identity_position_count, int)
                or self.identity_position_count < 1
                or self.identity_position_rank > self.identity_position_count
            ):
                raise ValueError(
                    "macro-identity rank requires complete positive identity "
                    "and position ranks"
                )


@dataclass(frozen=True, slots=True)
class _PhaseSelectionReceipt:
    phase: str
    population: tuple[_CandidatePositionRecord, ...]
    shortlist: tuple[_CandidatePositionRecord, ...]
    shortlist_ranking: tuple[_ShortlistRankReceipt, ...]
    estimator_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.phase not in {"phase_i", "phase_ii", "phase_iii"}:
            raise ValueError(f"unknown selection phase: {self.phase}")
        if not self.population:
            raise ValueError(f"{self.phase} population must be non-empty")
        if not self.shortlist:
            raise ValueError(f"{self.phase} shortlist must be non-empty")
        if len(self.shortlist_ranking) != len(self.shortlist):
            raise ValueError(
                f"{self.phase} shortlist rank receipt count must match "
                "the shortlist"
            )
        if any(not event_id for event_id in self.estimator_event_ids):
            raise ValueError("estimator event identities must be non-empty")


@dataclass(frozen=True, slots=True)
class _ResponseReceipt:
    identity: str
    coordinate_ids: tuple[str, ...]
    supported_rank: int
    supported_dimension: int

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("response identity must be non-empty")
        if not self.coordinate_ids:
            raise ValueError("response coordinate identities must be non-empty")
        if len(set(self.coordinate_ids)) != len(self.coordinate_ids):
            raise ValueError("response coordinate identities must be unique")
        if self.supported_dimension != len(self.coordinate_ids):
            raise ValueError(
                "supported response dimension must match coordinate identities"
            )
        if not 0 < self.supported_rank <= self.supported_dimension:
            raise ValueError("supported response rank is out of range")


@dataclass(frozen=True, slots=True)
class _TrustSolveReceipt:
    identity: str
    solver_identity: str
    response_identity: str
    supported_rank: int
    proposed_coordinate_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("trust identity must be non-empty")
        if not self.solver_identity:
            raise ValueError("trust solver identity must be non-empty")
        if not self.response_identity:
            raise ValueError("trust response identity must be non-empty")
        if self.supported_rank <= 0:
            raise ValueError("trust supported rank must be positive")
        if not self.proposed_coordinate_values:
            raise ValueError("trust proposal coordinates must be non-empty")


@dataclass(frozen=True, slots=True)
class _PredictiveCostReceipt:
    identity: str
    policy_identity: str
    value: float

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("predictive cost identity must be non-empty")
        if not self.policy_identity:
            raise ValueError("predictive cost policy identity must be non-empty")
        if not math.isfinite(self.value) or self.value < 0.0:
            raise ValueError(
                "predictive candidate cost must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class _EstimatorEventIdentity:
    sequence_index: int
    occurrence_id: str
    reuse_identity: str | None

    def __post_init__(self) -> None:
        if self.sequence_index < 0:
            raise ValueError("estimator sequence index must be non-negative")
        if not self.occurrence_id:
            raise ValueError("estimator occurrence identity must be non-empty")


@dataclass(frozen=True, slots=True)
class _SelectionEvaluation:
    phase_i: _PhaseSelectionReceipt
    phase_ii: _PhaseSelectionReceipt
    phase_iii: _PhaseSelectionReceipt
    selected: _CandidatePositionRecord
    response: _ResponseReceipt
    trust: _TrustSolveReceipt
    predictive_cost: _PredictiveCostReceipt
    estimator_events: tuple[_EstimatorEventIdentity, ...]


@dataclass(frozen=True, slots=True)
class _GreedyBatchProposalReceipt:
    """Joint reduced-plane receipt for one ordered greedy admission."""

    identity: str
    maximum_size: int
    search_window_size: int | None
    selected_record_ids: tuple[str, ...]
    score: float
    modeled_energy_decrease: float
    predictive_cost_excess: float
    denominator: float
    geometry_identity: str
    evaluated_subset_count: int

    def __post_init__(self) -> None:
        if not self.identity or not self.geometry_identity:
            raise ValueError("greedy proposal identities must be non-empty")
        if not 1 <= self.maximum_size <= 5:
            raise ValueError("greedy proposal maximum_size must lie in 1..5")
        if (
            self.search_window_size is not None
            and self.search_window_size <= 0
        ):
            raise ValueError(
                "greedy proposal search_window_size must be positive or None"
            )
        if not self.selected_record_ids:
            raise ValueError("greedy proposal must select at least one record")
        if len(self.selected_record_ids) > self.maximum_size:
            raise ValueError(
                "greedy proposal selected more records than maximum_size"
            )
        if (
            any(not value for value in self.selected_record_ids)
            or len(set(self.selected_record_ids))
            != len(self.selected_record_ids)
        ):
            raise ValueError(
                "greedy proposal record identities must be non-empty and unique"
            )
        numeric_values = (
            self.score,
            self.modeled_energy_decrease,
            self.predictive_cost_excess,
            self.denominator,
        )
        if any(not math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("greedy proposal values must be finite")
        if (
            self.score < 0.0
            or self.modeled_energy_decrease < 0.0
            or self.predictive_cost_excess < 0.0
            or self.denominator <= 0.0
        ):
            raise ValueError(
                "greedy proposal score, decrease, cost, and denominator "
                "must be non-negative with a positive denominator"
            )
        if not math.isclose(
            float(self.denominator),
            1.0 + float(self.predictive_cost_excess),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "greedy proposal denominator must equal one plus cost excess"
            )
        if not math.isclose(
            float(self.score),
            float(self.modeled_energy_decrease) / float(self.denominator),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "greedy proposal score must equal decrease over denominator"
            )
        if self.evaluated_subset_count < 1:
            raise ValueError(
                "greedy proposal must record at least one evaluated subset"
            )


@dataclass(frozen=True, slots=True)
class _GreedyBatchSelectionEvaluation:
    phase_i: _PhaseSelectionReceipt
    phase_ii: _PhaseSelectionReceipt
    phase_iii: _PhaseSelectionReceipt
    selected: tuple[_CandidatePositionRecord, ...]
    response: _ResponseReceipt
    trust: _TrustSolveReceipt
    predictive_cost: _PredictiveCostReceipt
    proposal: _GreedyBatchProposalReceipt
    estimator_events: tuple[_EstimatorEventIdentity, ...]


def _normalized_subset_counts(
    values: tuple[tuple[int, int], ...],
    *,
    name: str,
    maximum_size: int,
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(size), int(count)) for size, count in values)
    if normalized != tuple(sorted(normalized)):
        raise ValueError(f"{name} must be ordered by cardinality")
    sizes = tuple(size for size, _count in normalized)
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"{name} cardinalities must be unique")
    if any(
        size < 1 or size > maximum_size or count < 0
        for size, count in normalized
    ):
        raise ValueError(
            f"{name} must contain non-negative counts for sizes 1..maximum_size"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class _CombinatorialBatchProposalReceipt:
    """Exhaustive reduced-plane receipt for one combinatorial admission."""

    identity: str
    maximum_size: int
    search_window_size: int | None
    ranked_population_count: int
    ranked_window_count: int
    selected_record_ids: tuple[str, ...]
    score: float
    modeled_energy_decrease: float
    predictive_cost_excess: float
    denominator: float
    geometry_identity: str
    evaluated_subset_count: int
    subset_counts_considered: tuple[tuple[int, int], ...]
    subset_counts_evaluated: tuple[tuple[int, int], ...]
    subset_counts_feasible: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.identity or not self.geometry_identity:
            raise ValueError(
                "combinatorial proposal identities must be non-empty"
            )
        if not 1 <= self.maximum_size <= 5:
            raise ValueError(
                "combinatorial proposal maximum_size must lie in 1..5"
            )
        if (
            self.search_window_size is not None
            and self.search_window_size <= 0
        ):
            raise ValueError(
                "combinatorial proposal search_window_size must be positive "
                "or None"
            )
        if self.ranked_population_count < 1:
            raise ValueError(
                "combinatorial proposal ranked population must be non-empty"
            )
        expected_window_count = (
            self.ranked_population_count
            if self.search_window_size is None
            else min(
                self.ranked_population_count,
                self.search_window_size,
            )
        )
        if self.ranked_window_count != expected_window_count:
            raise ValueError(
                "combinatorial proposal ranked window does not match its "
                "resolved search policy"
            )
        if not self.selected_record_ids:
            raise ValueError(
                "combinatorial proposal must select at least one record"
            )
        if len(self.selected_record_ids) > min(
            self.maximum_size,
            self.ranked_window_count,
        ):
            raise ValueError(
                "combinatorial proposal selected more records than its "
                "effective cardinality cap"
            )
        if (
            any(not value for value in self.selected_record_ids)
            or len(set(self.selected_record_ids))
            != len(self.selected_record_ids)
        ):
            raise ValueError(
                "combinatorial proposal record identities must be non-empty "
                "and unique"
            )
        numeric_values = (
            self.score,
            self.modeled_energy_decrease,
            self.predictive_cost_excess,
            self.denominator,
        )
        if any(not math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("combinatorial proposal values must be finite")
        if (
            self.score < 0.0
            or self.modeled_energy_decrease < 0.0
            or self.predictive_cost_excess < 0.0
            or self.denominator <= 0.0
        ):
            raise ValueError(
                "combinatorial proposal score, decrease, cost, and "
                "denominator must be non-negative with a positive denominator"
            )
        if not math.isclose(
            float(self.denominator),
            1.0 + float(self.predictive_cost_excess),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "combinatorial proposal denominator must equal one plus cost "
                "excess"
            )
        if not math.isclose(
            float(self.score),
            float(self.modeled_energy_decrease) / float(self.denominator),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "combinatorial proposal score must equal decrease over "
                "denominator"
            )
        considered = _normalized_subset_counts(
            self.subset_counts_considered,
            name="subset_counts_considered",
            maximum_size=self.maximum_size,
        )
        evaluated = _normalized_subset_counts(
            self.subset_counts_evaluated,
            name="subset_counts_evaluated",
            maximum_size=self.maximum_size,
        )
        feasible = _normalized_subset_counts(
            self.subset_counts_feasible,
            name="subset_counts_feasible",
            maximum_size=self.maximum_size,
        )
        expected_considered = tuple(
            (size, math.comb(self.ranked_window_count, size))
            for size in range(
                1,
                min(self.maximum_size, self.ranked_window_count) + 1,
            )
        )
        if considered != expected_considered:
            raise ValueError(
                "combinatorial considered subset counts must exhaust every "
                "cardinality in the fixed ranked window"
            )
        all_sizes = sorted(
            {
                *(size for size, _count in considered),
                *(size for size, _count in evaluated),
                *(size for size, _count in feasible),
            }
        )
        considered_by_size = dict(considered)
        evaluated_by_size = dict(evaluated)
        feasible_by_size = dict(feasible)
        if any(
            feasible_by_size.get(size, 0)
            > evaluated_by_size.get(size, 0)
            or evaluated_by_size.get(size, 0)
            > considered_by_size.get(size, 0)
            for size in all_sizes
        ):
            raise ValueError(
                "combinatorial subset counts must satisfy feasible <= "
                "evaluated <= considered"
            )
        if self.evaluated_subset_count != sum(
            evaluated_by_size.values()
        ):
            raise ValueError(
                "combinatorial evaluated subset total does not reconcile"
            )
        if self.evaluated_subset_count < 1:
            raise ValueError(
                "combinatorial proposal must evaluate at least one subset"
            )


@dataclass(frozen=True, slots=True)
class _CombinatorialBatchSelectionEvaluation:
    phase_i: _PhaseSelectionReceipt
    phase_ii: _PhaseSelectionReceipt
    phase_iii: _PhaseSelectionReceipt
    selected: tuple[_CandidatePositionRecord, ...]
    response: _ResponseReceipt
    trust: _TrustSolveReceipt
    predictive_cost: _PredictiveCostReceipt
    proposal: _CombinatorialBatchProposalReceipt
    estimator_events: tuple[_EstimatorEventIdentity, ...]


class _SelectionKernel(Protocol):
    """Numerical transaction owned by one active admission workspace."""

    def accepted_state_snapshot(self) -> object:
        """Return the live accepted-state identity at the callback boundary."""

        ...

    def evaluate(
        self,
        domain: tuple[_CandidatePositionRecord, ...],
    ) -> (
        _SelectionEvaluation
        | _GreedyBatchSelectionEvaluation
        | _CombinatorialBatchSelectionEvaluation
    ): ...


@dataclass(frozen=True, slots=True)
class _SelectionWorkspace:
    admissible_records: tuple[_CandidatePositionRecord, ...]
    kernel: _SelectionKernel


@dataclass(frozen=True, slots=True)
class _SingletonAdmissionDecision:
    controller_round: int
    controller_state_fingerprint: str
    selected: _CandidatePositionRecord
    phase_i: _PhaseSelectionReceipt
    phase_ii: _PhaseSelectionReceipt
    phase_iii: _PhaseSelectionReceipt
    response: _ResponseReceipt
    trust: _TrustSolveReceipt
    predictive_cost: _PredictiveCostReceipt
    estimator_events: tuple[_EstimatorEventIdentity, ...]


@dataclass(frozen=True, slots=True)
class _GreedyBatchAdmissionDecision:
    controller_round: int
    controller_state_fingerprint: str
    selected: tuple[_CandidatePositionRecord, ...]
    phase_i: _PhaseSelectionReceipt
    phase_ii: _PhaseSelectionReceipt
    phase_iii: _PhaseSelectionReceipt
    response: _ResponseReceipt
    trust: _TrustSolveReceipt
    predictive_cost: _PredictiveCostReceipt
    proposal: _GreedyBatchProposalReceipt
    estimator_events: tuple[_EstimatorEventIdentity, ...]


@dataclass(frozen=True, slots=True)
class _CombinatorialBatchAdmissionDecision:
    controller_round: int
    controller_state_fingerprint: str
    selected: tuple[_CandidatePositionRecord, ...]
    phase_i: _PhaseSelectionReceipt
    phase_ii: _PhaseSelectionReceipt
    phase_iii: _PhaseSelectionReceipt
    response: _ResponseReceipt
    trust: _TrustSolveReceipt
    predictive_cost: _PredictiveCostReceipt
    proposal: _CombinatorialBatchProposalReceipt
    estimator_events: tuple[_EstimatorEventIdentity, ...]


def _record_key(record: _CandidatePositionRecord) -> tuple[str, str]:
    return record.domain_record_id, record.generator_id


def _assert_ranked_shortlist_membership(
    *,
    label: str,
    population: tuple[_CandidatePositionRecord, ...],
    shortlist: tuple[_CandidatePositionRecord, ...],
    ranking: tuple[_ShortlistRankReceipt, ...],
) -> None:
    population_keys = tuple(_record_key(record) for record in population)
    shortlist_keys = tuple(_record_key(record) for record in shortlist)
    if len(set(population_keys)) != len(population_keys):
        raise ValueError(f"{label} population identities must be unique")
    if len(set(shortlist_keys)) != len(shortlist_keys):
        raise ValueError(f"{label} shortlist identities must be unique")

    population_by_key = dict(zip(population_keys, population, strict=True))
    for record, key in zip(shortlist, shortlist_keys, strict=True):
        population_record = population_by_key.get(key)
        if population_record is None:
            raise ValueError(
                f"{label} shortlist record is not in the population"
            )
        if record != population_record:
            raise ValueError(
                f"{label} shortlist changed its population lineage receipt"
            )

    expected_ranks = tuple(range(1, len(shortlist) + 1))
    recorded_ranks = tuple(item.shortlist_rank for item in ranking)
    if recorded_ranks != expected_ranks:
        raise ValueError(
            f"{label} shortlist ranks must be contiguous and one-based"
        )
    for record, rank_receipt in zip(shortlist, ranking, strict=True):
        if rank_receipt.record_key != _record_key(record):
            raise ValueError(
                f"{label} shortlist rank identifies a different record"
            )
        if rank_receipt.pool_index != record.pool_index:
            raise ValueError(
                f"{label} shortlist rank changed its source pool index"
            )
        if rank_receipt.insertion_position != record.insertion_position:
            raise ValueError(
                f"{label} shortlist rank changed its insertion position"
            )

    shortlist_units = {item.shortlist_unit for item in ranking}
    if len(shortlist_units) != 1:
        raise ValueError(f"{label} shortlist mixes selection units")
    shortlist_unit = next(iter(shortlist_units))
    if shortlist_unit == _SHORTLIST_UNIT_CANDIDATE_POSITION:
        score_rank_keys = tuple(
            (
                -float(item.primary_score),
                -float(item.tie_break_score),
                int(item.pool_index),
                int(item.insertion_position),
            )
            for item in ranking
        )
        if score_rank_keys != tuple(sorted(score_rank_keys)):
            raise ValueError(
                f"{label} shortlist is not in deterministic score rank order"
            )
        return

    grouped: list[list[_ShortlistRankReceipt]] = []
    for item in ranking:
        if (
            not grouped
            or grouped[-1][0].shortlist_identity
            != item.shortlist_identity
        ):
            grouped.append([item])
        else:
            grouped[-1].append(item)
    expected_identity_ranks = tuple(range(1, len(grouped) + 1))
    observed_identity_ranks = tuple(
        int(group[0].identity_rank or 0) for group in grouped
    )
    if observed_identity_ranks != expected_identity_ranks:
        raise ValueError(
            f"{label} macro identities are not contiguous and one-based"
        )
    seen_identities: set[str] = set()
    representative_score_keys: list[tuple[float, float]] = []
    for group in grouped:
        identity = str(group[0].shortlist_identity)
        if identity in seen_identities:
            raise ValueError(
                f"{label} macro identity occupies multiple rank blocks"
            )
        seen_identities.add(identity)
        expected_position_ranks = tuple(range(1, len(group) + 1))
        observed_position_ranks = tuple(
            int(item.identity_position_rank or 0) for item in group
        )
        if observed_position_ranks != expected_position_ranks or any(
            item.identity_rank != group[0].identity_rank
            or item.identity_position_count != len(group)
            or item.shortlist_identity != identity
            for item in group
        ):
            raise ValueError(
                f"{label} macro-identity position ranks are incomplete"
            )
        within_identity_score_keys = tuple(
            (
                -float(item.primary_score),
                -float(item.tie_break_score),
                int(item.insertion_position),
            )
            for item in group
        )
        if within_identity_score_keys != tuple(
            sorted(within_identity_score_keys)
        ):
            raise ValueError(
                f"{label} macro-identity positions are not score ranked"
            )
        representative_score_keys.append(
            (
                -float(group[0].primary_score),
                -float(group[0].tie_break_score),
            )
        )
    if tuple(representative_score_keys) != tuple(
        sorted(representative_score_keys)
    ):
        raise ValueError(
            f"{label} macro representatives are not score ranked"
        )


def _assert_phase_lineage(
    *,
    receipt: _PhaseSelectionReceipt,
    domain_by_id: dict[str, _CandidatePositionRecord],
) -> None:
    for record in receipt.population:
        root = domain_by_id.get(record.domain_record_id)
        if root is None:
            raise ValueError(
                f"{receipt.phase} record escaped the admissible domain: "
                f"{record.domain_record_id}"
            )
        if record.pool_index != root.pool_index:
            raise ValueError(
                f"{receipt.phase} record changed its source pool index"
            )
        if record.insertion_position != root.insertion_position:
            raise ValueError(
                f"{receipt.phase} record changed its insertion position"
            )
        if record.lineage_identity[0] != root.generator_id:
            raise ValueError(
                f"{receipt.phase} record lost its root-generator lineage"
            )
    _assert_ranked_shortlist_membership(
        label=receipt.phase,
        population=receipt.population,
        shortlist=receipt.shortlist,
        ranking=receipt.shortlist_ranking,
    )


def _assert_progression(
    *,
    earlier: _PhaseSelectionReceipt,
    later: _PhaseSelectionReceipt,
) -> None:
    admitted_roots = {
        record.domain_record_id for record in earlier.shortlist
    }
    if any(
        record.domain_record_id not in admitted_roots
        for record in later.population
    ):
        raise ValueError(
            f"{later.phase} population did not descend from "
            f"{earlier.phase} shortlist"
        )


def _select_singleton(
    state: _SRControllerState,
    workspace: _SelectionWorkspace,
) -> _SingletonAdmissionDecision:
    """Run and validate one immutable singleton selection transaction."""

    state_before = tuple(
        getattr(state, field_name)
        for field_name in state.__dataclass_fields__
    )
    domain = _build_candidate_domain(workspace.admissible_records)
    expected_domain_ids = (
        tuple(state.admissible_domain_record_ids)
        if state.admissible_domain_record_ids
        else tuple(state.available_generator_ids)
    )
    observed_domain_ids = (
        tuple(record.domain_record_id for record in domain)
        if state.admissible_domain_record_ids
        else tuple(record.generator_id for record in domain)
    )
    if observed_domain_ids != expected_domain_ids:
        raise ValueError(
            "candidate domain does not match the controller domain"
        )

    accepted_state_before = workspace.kernel.accepted_state_snapshot()
    evaluation = workspace.kernel.evaluate(domain)
    accepted_state_after = workspace.kernel.accepted_state_snapshot()
    if accepted_state_after != accepted_state_before:
        raise RuntimeError(
            "selection mutated live accepted operators, parameters, or state"
        )
    state_after = tuple(
        getattr(state, field_name)
        for field_name in state.__dataclass_fields__
    )
    if state_after != state_before:
        raise RuntimeError("selection mutated the accepted controller state")

    receipts = (
        evaluation.phase_i,
        evaluation.phase_ii,
        evaluation.phase_iii,
    )
    if tuple(receipt.phase for receipt in receipts) != (
        "phase_i",
        "phase_ii",
        "phase_iii",
    ):
        raise ValueError("selection must return Phase I, II, and III in order")
    phase_i_domain_ids = tuple(
        record.domain_record_id for record in evaluation.phase_i.population
    )
    if (
        len(phase_i_domain_ids) != len(domain)
        or set(phase_i_domain_ids) != {
            record.domain_record_id for record in domain
        }
    ):
        raise ValueError(
            "Phase-I population must cover the admissible domain exactly once"
        )

    domain_by_id = {record.domain_record_id: record for record in domain}
    for receipt in receipts:
        _assert_phase_lineage(
            receipt=receipt,
            domain_by_id=domain_by_id,
        )
    _assert_progression(
        earlier=evaluation.phase_i,
        later=evaluation.phase_ii,
    )
    _assert_progression(
        earlier=evaluation.phase_ii,
        later=evaluation.phase_iii,
    )
    if len(evaluation.phase_iii.shortlist) != 1:
        raise ValueError("default singleton selection must retain one winner")
    if evaluation.selected != evaluation.phase_iii.shortlist[0]:
        raise ValueError("selected record must be the Phase-III singleton")

    if evaluation.trust.response_identity != evaluation.response.identity:
        raise ValueError("trust solve does not identify its response receipt")
    if evaluation.trust.supported_rank != evaluation.response.supported_rank:
        raise ValueError("trust and response supported ranks disagree")
    if len(evaluation.trust.proposed_coordinate_values) != (
        evaluation.response.supported_dimension
    ):
        raise ValueError(
            "trust proposal does not cover the response coordinate chart"
        )

    sequence_indices = tuple(
        event.sequence_index for event in evaluation.estimator_events
    )
    if sequence_indices and sequence_indices != tuple(
        range(sequence_indices[0], sequence_indices[0] + len(sequence_indices))
    ):
        raise ValueError("estimator event order must be contiguous")
    occurrence_ids = tuple(
        event.occurrence_id for event in evaluation.estimator_events
    )
    if len(set(occurrence_ids)) != len(occurrence_ids):
        raise ValueError("estimator occurrence identities must be unique")
    phase_event_ids = tuple(
        event_id
        for receipt in receipts
        for event_id in receipt.estimator_event_ids
    )
    if phase_event_ids != occurrence_ids:
        raise ValueError(
            "phase estimator identities must equal the ordered selection delta"
        )

    return _SingletonAdmissionDecision(
        controller_round=state.controller_round,
        controller_state_fingerprint=state.accepted_state_fingerprint,
        selected=evaluation.selected,
        phase_i=evaluation.phase_i,
        phase_ii=evaluation.phase_ii,
        phase_iii=evaluation.phase_iii,
        response=evaluation.response,
        trust=evaluation.trust,
        predictive_cost=evaluation.predictive_cost,
        estimator_events=evaluation.estimator_events,
    )


def _select_greedy_batch(
    state: _SRControllerState,
    workspace: _SelectionWorkspace,
    *,
    maximum_size: int,
    search_window_size: int | None,
) -> _GreedyBatchAdmissionDecision:
    """Run and validate one immutable ordered greedy-batch selection."""

    if not 1 <= int(maximum_size) <= 5:
        raise ValueError("maximum_size must lie in the supported range 1..5")
    if search_window_size is not None and int(search_window_size) < 1:
        raise ValueError("search_window_size must be positive or None")
    state_before = tuple(
        getattr(state, field_name)
        for field_name in state.__dataclass_fields__
    )
    domain = _build_candidate_domain(workspace.admissible_records)
    expected_domain_ids = (
        tuple(state.admissible_domain_record_ids)
        if state.admissible_domain_record_ids
        else tuple(state.available_generator_ids)
    )
    observed_domain_ids = (
        tuple(record.domain_record_id for record in domain)
        if state.admissible_domain_record_ids
        else tuple(record.generator_id for record in domain)
    )
    if observed_domain_ids != expected_domain_ids:
        raise ValueError(
            "candidate domain does not match the controller domain"
        )

    accepted_state_before = workspace.kernel.accepted_state_snapshot()
    evaluation = workspace.kernel.evaluate(domain)
    if not isinstance(evaluation, _GreedyBatchSelectionEvaluation):
        raise TypeError(
            "greedy selection kernel returned the wrong evaluation type"
        )
    accepted_state_after = workspace.kernel.accepted_state_snapshot()
    if accepted_state_after != accepted_state_before:
        raise RuntimeError(
            "selection mutated live accepted operators, parameters, or state"
        )
    state_after = tuple(
        getattr(state, field_name)
        for field_name in state.__dataclass_fields__
    )
    if state_after != state_before:
        raise RuntimeError("selection mutated the accepted controller state")

    receipts = (
        evaluation.phase_i,
        evaluation.phase_ii,
        evaluation.phase_iii,
    )
    if tuple(receipt.phase for receipt in receipts) != (
        "phase_i",
        "phase_ii",
        "phase_iii",
    ):
        raise ValueError("selection must return Phase I, II, and III in order")
    phase_i_domain_ids = tuple(
        record.domain_record_id for record in evaluation.phase_i.population
    )
    if (
        len(phase_i_domain_ids) != len(domain)
        or set(phase_i_domain_ids)
        != {record.domain_record_id for record in domain}
    ):
        raise ValueError(
            "Phase-I population must cover the admissible domain exactly once"
        )
    domain_by_id = {record.domain_record_id: record for record in domain}
    for receipt in receipts:
        _assert_phase_lineage(
            receipt=receipt,
            domain_by_id=domain_by_id,
        )
    _assert_progression(
        earlier=evaluation.phase_i,
        later=evaluation.phase_ii,
    )
    _assert_progression(
        earlier=evaluation.phase_ii,
        later=evaluation.phase_iii,
    )

    selected = tuple(evaluation.selected)
    if not selected:
        raise ValueError("greedy selection must retain at least one winner")
    if len(selected) > int(maximum_size):
        raise ValueError("greedy selection exceeded maximum_size")
    if selected != evaluation.phase_iii.shortlist:
        raise ValueError(
            "selected batch must equal the ordered Phase-III shortlist"
        )
    if len({record.domain_record_id for record in selected}) != len(selected):
        raise ValueError(
            "selected batch domain-record identities must be unique"
        )
    if len({record.generator_id for record in selected}) != len(selected):
        raise ValueError(
            "selected batch generator identities must be globally unique"
        )
    proposal = evaluation.proposal
    if proposal.maximum_size != int(maximum_size):
        raise ValueError(
            "greedy proposal maximum_size differs from the active policy"
        )
    if proposal.search_window_size != search_window_size:
        raise ValueError(
            "greedy proposal search window differs from the active policy"
        )
    if proposal.selected_record_ids != tuple(
        record.domain_record_id for record in selected
    ):
        raise ValueError(
            "greedy proposal identifies a different ordered batch"
        )

    if evaluation.trust.response_identity != evaluation.response.identity:
        raise ValueError("trust solve does not identify its response receipt")
    if evaluation.trust.supported_rank != evaluation.response.supported_rank:
        raise ValueError("trust and response supported ranks disagree")
    if len(evaluation.trust.proposed_coordinate_values) != (
        evaluation.response.supported_dimension
    ):
        raise ValueError(
            "trust proposal does not cover the response coordinate chart"
        )

    sequence_indices = tuple(
        event.sequence_index for event in evaluation.estimator_events
    )
    if sequence_indices and sequence_indices != tuple(
        range(sequence_indices[0], sequence_indices[0] + len(sequence_indices))
    ):
        raise ValueError("estimator event order must be contiguous")
    occurrence_ids = tuple(
        event.occurrence_id for event in evaluation.estimator_events
    )
    if len(set(occurrence_ids)) != len(occurrence_ids):
        raise ValueError("estimator occurrence identities must be unique")
    phase_event_ids = tuple(
        event_id
        for receipt in receipts
        for event_id in receipt.estimator_event_ids
    )
    if phase_event_ids != occurrence_ids:
        raise ValueError(
            "phase estimator identities must equal the ordered selection delta"
        )

    return _GreedyBatchAdmissionDecision(
        controller_round=state.controller_round,
        controller_state_fingerprint=state.accepted_state_fingerprint,
        selected=selected,
        phase_i=evaluation.phase_i,
        phase_ii=evaluation.phase_ii,
        phase_iii=evaluation.phase_iii,
        response=evaluation.response,
        trust=evaluation.trust,
        predictive_cost=evaluation.predictive_cost,
        proposal=proposal,
        estimator_events=evaluation.estimator_events,
    )


def _select_combinatorial_batch(
    state: _SRControllerState,
    workspace: _SelectionWorkspace,
    *,
    maximum_size: int,
    search_window_size: int | None,
) -> _CombinatorialBatchAdmissionDecision:
    """Run and validate one immutable exhaustive-subset admission."""

    if not 1 <= int(maximum_size) <= 5:
        raise ValueError("maximum_size must lie in the supported range 1..5")
    if search_window_size is not None and int(search_window_size) < 1:
        raise ValueError("search_window_size must be positive or None")
    state_before = tuple(
        getattr(state, field_name)
        for field_name in state.__dataclass_fields__
    )
    domain = _build_candidate_domain(workspace.admissible_records)
    expected_domain_ids = (
        tuple(state.admissible_domain_record_ids)
        if state.admissible_domain_record_ids
        else tuple(state.available_generator_ids)
    )
    observed_domain_ids = (
        tuple(record.domain_record_id for record in domain)
        if state.admissible_domain_record_ids
        else tuple(record.generator_id for record in domain)
    )
    if observed_domain_ids != expected_domain_ids:
        raise ValueError(
            "candidate domain does not match the controller domain"
        )

    accepted_state_before = workspace.kernel.accepted_state_snapshot()
    evaluation = workspace.kernel.evaluate(domain)
    if not isinstance(evaluation, _CombinatorialBatchSelectionEvaluation):
        raise TypeError(
            "combinatorial selection kernel returned the wrong evaluation type"
        )
    accepted_state_after = workspace.kernel.accepted_state_snapshot()
    if accepted_state_after != accepted_state_before:
        raise RuntimeError(
            "selection mutated live accepted operators, parameters, or state"
        )
    state_after = tuple(
        getattr(state, field_name)
        for field_name in state.__dataclass_fields__
    )
    if state_after != state_before:
        raise RuntimeError("selection mutated the accepted controller state")

    receipts = (
        evaluation.phase_i,
        evaluation.phase_ii,
        evaluation.phase_iii,
    )
    if tuple(receipt.phase for receipt in receipts) != (
        "phase_i",
        "phase_ii",
        "phase_iii",
    ):
        raise ValueError("selection must return Phase I, II, and III in order")
    phase_i_domain_ids = tuple(
        record.domain_record_id for record in evaluation.phase_i.population
    )
    if (
        len(phase_i_domain_ids) != len(domain)
        or set(phase_i_domain_ids)
        != {record.domain_record_id for record in domain}
    ):
        raise ValueError(
            "Phase-I population must cover the admissible domain exactly once"
        )
    domain_by_id = {record.domain_record_id: record for record in domain}
    for receipt in receipts:
        _assert_phase_lineage(
            receipt=receipt,
            domain_by_id=domain_by_id,
        )
    _assert_progression(
        earlier=evaluation.phase_i,
        later=evaluation.phase_ii,
    )
    _assert_progression(
        earlier=evaluation.phase_ii,
        later=evaluation.phase_iii,
    )

    selected = tuple(evaluation.selected)
    if not selected:
        raise ValueError(
            "combinatorial selection must retain at least one winner"
        )
    if len(selected) > int(maximum_size):
        raise ValueError("combinatorial selection exceeded maximum_size")
    if selected != evaluation.phase_iii.shortlist:
        raise ValueError(
            "selected batch must equal the ordered Phase-III shortlist"
        )
    if len({record.domain_record_id for record in selected}) != len(selected):
        raise ValueError(
            "selected batch domain-record identities must be unique"
        )
    if len({record.generator_id for record in selected}) != len(selected):
        raise ValueError(
            "selected batch generator identities must be globally unique"
        )
    proposal = evaluation.proposal
    if proposal.maximum_size != int(maximum_size):
        raise ValueError(
            "combinatorial proposal maximum_size differs from the active "
            "policy"
        )
    if proposal.search_window_size != search_window_size:
        raise ValueError(
            "combinatorial proposal search window differs from the active "
            "policy"
        )
    if proposal.selected_record_ids != tuple(
        record.domain_record_id for record in selected
    ):
        raise ValueError(
            "combinatorial proposal identifies a different ordered batch"
        )

    if evaluation.trust.response_identity != evaluation.response.identity:
        raise ValueError("trust solve does not identify its response receipt")
    if evaluation.trust.supported_rank != evaluation.response.supported_rank:
        raise ValueError("trust and response supported ranks disagree")
    if len(evaluation.trust.proposed_coordinate_values) != (
        evaluation.response.supported_dimension
    ):
        raise ValueError(
            "trust proposal does not cover the response coordinate chart"
        )

    sequence_indices = tuple(
        event.sequence_index for event in evaluation.estimator_events
    )
    if sequence_indices and sequence_indices != tuple(
        range(sequence_indices[0], sequence_indices[0] + len(sequence_indices))
    ):
        raise ValueError("estimator event order must be contiguous")
    occurrence_ids = tuple(
        event.occurrence_id for event in evaluation.estimator_events
    )
    if len(set(occurrence_ids)) != len(occurrence_ids):
        raise ValueError("estimator occurrence identities must be unique")
    phase_event_ids = tuple(
        event_id
        for receipt in receipts
        for event_id in receipt.estimator_event_ids
    )
    if phase_event_ids != occurrence_ids:
        raise ValueError(
            "phase estimator identities must equal the ordered selection delta"
        )

    return _CombinatorialBatchAdmissionDecision(
        controller_round=state.controller_round,
        controller_state_fingerprint=state.accepted_state_fingerprint,
        selected=selected,
        phase_i=evaluation.phase_i,
        phase_ii=evaluation.phase_ii,
        phase_iii=evaluation.phase_iii,
        response=evaluation.response,
        trust=evaluation.trust,
        predictive_cost=evaluation.predictive_cost,
        proposal=proposal,
        estimator_events=evaluation.estimator_events,
    )
