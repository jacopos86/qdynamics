"""Typed macro-to-child funnel orchestration for Paper-I Route A."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from pipelines.static_adapt.route_a_child_padding import (
    RouteAChildPaddingConfig,
    filter_route_a_child_padding_records,
)
from pipelines.static_adapt.route_a_shortlists import (
    CHILD_IDENTITY_POLICIES,
    CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
    CHILD_IDENTITY_POLICY_PARENT_QUALIFIED_LEGACY_V1,
    ROUTE_A_SHORTLIST_UNIT_PAULI_CHILD,
    child_identity_for_policy,
    deduplicate_child_position_records,
    expand_selected_identities,
    identity_population,
    macro_operator_identity,
)
from pipelines.static_adapt.route_a_schur_selector import (
    RouteASchurSelectorConfig,
)
from pipelines.static_adapt.joint_step_warm_start import (
    RouteAJointStepWarmStartConfig,
)


ROUTE_A_FUNNEL_DIRECT_CHILD_PHASE3_V1 = "direct_child_phase3_v1"
ROUTE_A_FUNNEL_HIERARCHICAL_CHILD_123_V1 = "hierarchical_child_123_v1"
ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1 = "child_12_joint_schur_v1"
ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2 = "child_12_joint_response_v2"
ROUTE_A_FUNNEL_CHILD_12_MODES = frozenset(
    {
        ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
        ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2,
    }
)
ROUTE_A_FUNNEL_MODES = frozenset(
    {
        ROUTE_A_FUNNEL_DIRECT_CHILD_PHASE3_V1,
        ROUTE_A_FUNNEL_HIERARCHICAL_CHILD_123_V1,
        ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1,
        ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2,
    }
)

ROUTE_A_PHASE2_SELECTOR_RAW_GAIN_TIMES_NOVELTY_LEGACY_V1 = (
    "raw_gain_times_novelty_legacy_v1"
)
ROUTE_A_PHASE2_SELECTOR_JOINT_RESPONSE_SINGLETON_V1 = (
    "joint_response_singleton_v1"
)
ROUTE_A_PHASE2_SELECTOR_MODES = frozenset(
    {
        ROUTE_A_PHASE2_SELECTOR_RAW_GAIN_TIMES_NOVELTY_LEGACY_V1,
        ROUTE_A_PHASE2_SELECTOR_JOINT_RESPONSE_SINGLETON_V1,
    }
)

ROUTE_A_PHASE0_DISABLED = "disabled"
ROUTE_A_PHASE0_LEGACY_MACRO_PRESCREEN_V1 = "legacy_macro_prescreen_v1"
ROUTE_A_PHASE0_POLICIES = frozenset(
    {
        ROUTE_A_PHASE0_DISABLED,
        ROUTE_A_PHASE0_LEGACY_MACRO_PRESCREEN_V1,
    }
)

ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1 = "child_only_v1"
ROUTE_A_PHASE3_POPULATION_PARENT_PLUS_CHILD_ABLATION_V1 = (
    "parent_plus_child_ablation_v1"
)
ROUTE_A_PHASE3_POPULATION_MODES = frozenset(
    {
        ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1,
        ROUTE_A_PHASE3_POPULATION_PARENT_PLUS_CHILD_ABLATION_V1,
    }
)

ROUTE_A_BATCHING_OFF = "off"
ROUTE_A_BATCHING_GREEDY_REDUCED_PLANE = "greedy_reduced_plane"
ROUTE_A_BATCHING_COMBINATORIAL_REDUCED_PLANE = "combinatorial_reduced_plane"
ROUTE_A_BATCHING_MODES = frozenset(
    {
        ROUTE_A_BATCHING_OFF,
        ROUTE_A_BATCHING_GREEDY_REDUCED_PLANE,
        ROUTE_A_BATCHING_COMBINATORIAL_REDUCED_PLANE,
    }
)


@dataclass(frozen=True)
class RouteAFunnelConfig:
    mode: str = ROUTE_A_FUNNEL_DIRECT_CHILD_PHASE3_V1
    population_mode: str = ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1
    child_identity_policy: str = CHILD_IDENTITY_POLICY_PARENT_QUALIFIED_LEGACY_V1
    macro_phase0_cap: int = 87
    macro_phase1_cap: int = 24
    macro_phase2_cap: int = 12
    child_phase1_cap: int = 48
    child_phase2_cap: int = 32
    child_phase3_cap: int = 24
    batching_mode: str = ROUTE_A_BATCHING_COMBINATORIAL_REDUCED_PLANE
    batch_size_cap: int = 3
    phase0_policy: str = ROUTE_A_PHASE0_LEGACY_MACRO_PRESCREEN_V1
    phase2_selector_mode: str | None = None
    schur_selector_config: RouteASchurSelectorConfig = field(
        default_factory=RouteASchurSelectorConfig
    )
    joint_step_warm_start: RouteAJointStepWarmStartConfig = field(
        default_factory=RouteAJointStepWarmStartConfig
    )
    child_padding: RouteAChildPaddingConfig = field(
        default_factory=RouteAChildPaddingConfig
    )

    def __post_init__(self) -> None:
        if str(self.mode) not in ROUTE_A_FUNNEL_MODES:
            raise ValueError(
                f"Route-A funnel mode must be one of {sorted(ROUTE_A_FUNNEL_MODES)}."
            )
        if str(self.population_mode) not in ROUTE_A_PHASE3_POPULATION_MODES:
            raise ValueError(
                "Route-A Phase-III population mode must be one of "
                f"{sorted(ROUTE_A_PHASE3_POPULATION_MODES)}."
            )
        if str(self.phase0_policy) not in ROUTE_A_PHASE0_POLICIES:
            raise ValueError(
                f"phase0_policy must be one of {sorted(ROUTE_A_PHASE0_POLICIES)}."
            )
        if not isinstance(self.schur_selector_config, RouteASchurSelectorConfig):
            raise TypeError(
                "schur_selector_config must be a RouteASchurSelectorConfig."
            )
        if not isinstance(
            self.joint_step_warm_start,
            RouteAJointStepWarmStartConfig,
        ):
            raise TypeError(
                "joint_step_warm_start must be a RouteAJointStepWarmStartConfig."
            )
        if not isinstance(self.child_padding, RouteAChildPaddingConfig):
            raise TypeError("child_padding must be a RouteAChildPaddingConfig.")
        resolved_phase2_selector_mode = (
            ROUTE_A_PHASE2_SELECTOR_JOINT_RESPONSE_SINGLETON_V1
            if str(self.mode) == ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2
            else ROUTE_A_PHASE2_SELECTOR_RAW_GAIN_TIMES_NOVELTY_LEGACY_V1
        )
        if self.phase2_selector_mode is not None:
            resolved_phase2_selector_mode = str(self.phase2_selector_mode)
        if resolved_phase2_selector_mode not in ROUTE_A_PHASE2_SELECTOR_MODES:
            raise ValueError(
                "phase2_selector_mode must be one of "
                f"{sorted(ROUTE_A_PHASE2_SELECTOR_MODES)}."
            )
        if (
            str(self.mode) == ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2
            and resolved_phase2_selector_mode
            != ROUTE_A_PHASE2_SELECTOR_JOINT_RESPONSE_SINGLETON_V1
        ):
            raise ValueError(
                "child_12_joint_response_v2 requires "
                "phase2_selector_mode='joint_response_singleton_v1'."
            )
        if (
            str(self.mode) != ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2
            and resolved_phase2_selector_mode
            != ROUTE_A_PHASE2_SELECTOR_RAW_GAIN_TIMES_NOVELTY_LEGACY_V1
        ):
            raise ValueError(
                "joint_response_singleton_v1 is isolated to the experimental "
                "child_12_joint_response_v2 route."
            )
        object.__setattr__(
            self,
            "phase2_selector_mode",
            str(resolved_phase2_selector_mode),
        )
        if str(self.mode) in ROUTE_A_FUNNEL_CHILD_12_MODES:
            if str(self.phase0_policy) != ROUTE_A_PHASE0_DISABLED:
                raise ValueError(
                    "Child-12 joint routes require phase0_policy='disabled'."
                )
            if (
                str(self.population_mode)
                != ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1
            ):
                raise ValueError(
                    "Child-12 joint routes are child-only; parent-plus-child is "
                    "a legacy ablation."
                )
            if (
                str(self.child_identity_policy)
                != CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1
            ):
                raise ValueError(
                    "Child-12 joint routes require globally deduplicated "
                    "Pauli-word identities."
                )
        if str(self.child_identity_policy) not in CHILD_IDENTITY_POLICIES:
            raise ValueError(
                f"child_identity_policy must be one of {sorted(CHILD_IDENTITY_POLICIES)}."
            )
        if str(self.batching_mode) not in ROUTE_A_BATCHING_MODES:
            raise ValueError(
                f"batching_mode must be one of {sorted(ROUTE_A_BATCHING_MODES)}."
            )
        for field_name in (
            "macro_phase0_cap",
            "macro_phase1_cap",
            "macro_phase2_cap",
            "child_phase1_cap",
            "child_phase2_cap",
            "child_phase3_cap",
            "batch_size_cap",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be >= 1.")

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "mode": str(self.mode),
            "population_mode": str(self.population_mode),
            "child_identity_policy": str(self.child_identity_policy),
            "macro_phase0_cap": int(self.macro_phase0_cap),
            "macro_phase1_cap": int(self.macro_phase1_cap),
            "macro_phase2_cap": int(self.macro_phase2_cap),
            "child_phase1_cap": int(self.child_phase1_cap),
            "child_phase2_cap": int(self.child_phase2_cap),
            "child_phase3_cap": int(self.child_phase3_cap),
            "batching_mode": str(self.batching_mode),
            "batch_size_cap": int(self.batch_size_cap),
            "phase0_policy": str(self.phase0_policy),
            "schur_selector": self.schur_selector_config.as_dict(),
            "joint_step_warm_start": self.joint_step_warm_start.as_dict(),
            "child_padding": self.child_padding.as_dict(),
        }
        if str(self.mode) == ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2:
            payload["phase2_selector_mode"] = str(self.phase2_selector_mode)
        return payload


@dataclass(frozen=True)
class RouteAFunnelQueryEvent:
    stage: str
    phase: str
    event_kind: str
    probe_role: str
    records: tuple[dict[str, Any], ...]
    retained_record_count: int
    reused_records: tuple[dict[str, Any], ...] = ()
    reused_probe_role: str | None = None
    reused_record_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": str(self.stage),
            "phase": str(self.phase),
            "event_kind": str(self.event_kind),
            "probe_role": str(self.probe_role),
            "new_operator_probe_record_count": int(len(self.records)),
            "retained_record_count": int(self.retained_record_count),
            "reused_probe_role": (
                None if self.reused_probe_role is None else str(self.reused_probe_role)
            ),
            "reused_record_count": int(self.reused_record_count),
            "reused_measurement_record_count": int(len(self.reused_records)),
            "reuse_incremental_operator_probe_count": 0,
        }


@dataclass(frozen=True)
class RouteAFunnelResult:
    child_population: tuple[dict[str, Any], ...]
    child_phase1_records: tuple[dict[str, Any], ...]
    child_phase2_records: tuple[dict[str, Any], ...]
    child_phase3_records: tuple[dict[str, Any], ...]
    selection_records: tuple[dict[str, Any], ...]
    query_events: tuple[RouteAFunnelQueryEvent, ...]
    telemetry: Mapping[str, Any]
    query_work: Mapping[str, Any]


def _representation(record: Mapping[str, Any]) -> str:
    explicit = str(record.get("route_a_candidate_representation", ""))
    if explicit:
        return explicit
    split_mode = str(
        record.get("runtime_split_mode")
        or getattr(record.get("feature"), "runtime_split_mode", "off")
    )
    return "parent_macro" if split_mode == "off" else "pauli_child"


def _identity_key(
    record: Mapping[str, Any],
    *,
    child_identity_policy: str,
) -> str:
    if _representation(record) == "parent_macro":
        return f"parent|{macro_operator_identity(record)}"
    return "child|" + child_identity_for_policy(
        record,
        policy=child_identity_policy,
    )


def _stage_shortlist(
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    score_key: str,
    tie_break_score_key: str | None,
    cap: int,
    child_identity_policy: str,
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    population = identity_population(
        records,
        identity_key=lambda record: _identity_key(
            record,
            child_identity_policy=child_identity_policy,
        ),
        score_key=score_key,
        tie_break_score_key=tie_break_score_key,
    )
    selected_representatives = list(population.representatives[: int(cap)])
    shortlisted = expand_selected_identities(
        population,
        selected_representatives,
        shortlist_flag=f"{stage}_shortlisted",
        shortlist_unit=ROUTE_A_SHORTLIST_UNIT_PAULI_CHILD,
        feature_updater=feature_updater,
    )
    return shortlisted, {
        "stage": str(stage),
        "score_key": str(score_key),
        "cap": int(cap),
        "cap_unit": ROUTE_A_SHORTLIST_UNIT_PAULI_CHILD,
        "input_record_count": int(len(records)),
        "input_identity_count": int(population.identity_count),
        "survivor_record_count": int(len(shortlisted)),
        "survivor_identity_count": int(len(selected_representatives)),
        "physical_lanes_applied": False,
        "parent_family_quota_applied": False,
        "frontier_threshold_applied": False,
    }


def run_route_a_child_funnel(
    child_records: Sequence[Mapping[str, Any]],
    *,
    config: RouteAFunnelConfig,
    parent_records: Sequence[Mapping[str, Any]] = (),
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    phase1_record_evaluator: (
        Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    ) = None,
    full_record_evaluator: (
        Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    ) = None,
    phase2_record_evaluator: (
        Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    ) = None,
    phase3_record_evaluator: (
        Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    ) = None,
    phase1_population_rescorer: (
        Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]] | None
    ) = None,
    phase2_population_rescorer: (
        Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]] | None
    ) = None,
    phase3_population_rescorer: (
        Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]] | None
    ) = None,
    phase3_pre_evaluation_rescorer: (
        Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]] | None
    ) = None,
    phase2_population_evaluator: (
        Callable[[Sequence[Mapping[str, Any]]], Any] | None
    ) = None,
    candidate_record_filter: (
        Callable[[Mapping[str, Any]], bool] | None
    ) = None,
) -> RouteAFunnelResult:
    """Run direct or hierarchical child stages over one global population."""

    deduplicated_children, dedup_telemetry = deduplicate_child_position_records(
        child_records,
        score_key="phase1_active_score",
        tie_break_score_key="simple_score",
        identity_policy=str(config.child_identity_policy),
    )
    dedup_telemetry = {
        **dict(dedup_telemetry),
        "applied_before_child_phase1": True,
        "applied_before_search_pool_construction": True,
        "applied_before_joint_gram_hessian_measurement": True,
    }
    filtered_children, child_padding_telemetry = (
        filter_route_a_child_padding_records(
            deduplicated_children,
            config=config.child_padding,
        )
    )
    sector_filter_input_count = int(len(filtered_children))
    if candidate_record_filter is not None:
        filtered_children = [
            dict(record)
            for record in filtered_children
            if bool(candidate_record_filter(record))
        ]
    candidate_filter_telemetry = {
        "active": bool(candidate_record_filter is not None),
        "input_record_count": int(sector_filter_input_count),
        "retained_record_count": int(len(filtered_children)),
        "rejected_record_count": int(
            sector_filter_input_count - len(filtered_children)
        ),
        "application_point": "after_padding_before_phase1_evaluation",
    }
    pre_padding_identity_count = len(
        {
            child_identity_for_policy(
                record,
                policy=str(config.child_identity_policy),
            )
            for record in deduplicated_children
        }
    )
    post_padding_identity_count = len(
        {
            child_identity_for_policy(
                record,
                policy=str(config.child_identity_policy),
            )
            for record in filtered_children
        }
    )
    dedup_telemetry.update(
        {
            "pre_padding_filter_record_count": int(
                len(deduplicated_children)
            ),
            "pre_padding_filter_unique_child_identity_count": int(
                pre_padding_identity_count
            ),
            "post_padding_filter_record_count": int(len(filtered_children)),
            "post_padding_filter_unique_child_identity_count": int(
                post_padding_identity_count
            ),
            "padding_rejected_record_count": int(
                child_padding_telemetry.get("rejected_record_count", 0)
            ),
            "padding_rejected_identity_count": int(
                child_padding_telemetry.get("rejected_identity_count", 0)
            ),
        }
    )
    population_records = [
        {**dict(record), "route_a_candidate_representation": "pauli_child"}
        for record in filtered_children
    ]
    if (
        str(config.population_mode)
        == ROUTE_A_PHASE3_POPULATION_PARENT_PLUS_CHILD_ABLATION_V1
    ):
        population_records.extend(
            {
                **dict(record),
                "route_a_candidate_representation": "parent_macro",
            }
            for record in parent_records
        )
    phase1_input_records: list[dict[str, Any]] = []
    for record in population_records:
        if (
            phase1_record_evaluator is None
            or _representation(record) == "parent_macro"
        ):
            phase1_input_records.append(dict(record))
            continue
        phase1_input_records.append(
            dict(phase1_record_evaluator(record))
        )
    phase1_evaluation_telemetry = {
        "deferred_until_after_global_deduplication": bool(
            phase1_record_evaluator is not None
        ),
        "global_deduplication_complete_before_evaluation": True,
        "padding_policy_complete_before_evaluation": True,
        "cooldown_filter_complete_before_evaluation": True,
        "input_record_count": int(len(population_records)),
        "evaluated_record_count": int(
            len(phase1_input_records)
            if phase1_record_evaluator is not None
            else 0
        ),
    }
    if phase1_population_rescorer is not None:
        phase1_input_records = [
            dict(record)
            for record in phase1_population_rescorer(phase1_input_records)
        ]

    stage_telemetry: list[dict[str, Any]] = []
    query_events: list[RouteAFunnelQueryEvent] = []

    def _evaluate_records(
        records: Sequence[Mapping[str, Any]],
        evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if evaluator is None:
            return [dict(record) for record in records]
        evaluated: list[dict[str, Any]] = []
        for record in records:
            if _representation(record) == "parent_macro":
                evaluated.append(dict(record))
                continue
            evaluated.append(dict(evaluator(record)))
        return evaluated

    hierarchical_child_stages = str(config.mode) in {
        ROUTE_A_FUNNEL_HIERARCHICAL_CHILD_123_V1,
        *ROUTE_A_FUNNEL_CHILD_12_MODES,
    }
    phase2_selector_telemetry: dict[str, Any] = {
        "selector_mode_requested": str(config.phase2_selector_mode),
        "selector_mode_effective": str(config.phase2_selector_mode),
        "active": False,
    }
    if hierarchical_child_stages:
        phase1_records, phase1_telemetry = _stage_shortlist(
            phase1_input_records,
            stage="child_phase1",
            score_key="phase1_active_score",
            tie_break_score_key="simple_score",
            cap=int(config.child_phase1_cap),
            child_identity_policy=str(config.child_identity_policy),
            feature_updater=feature_updater,
        )
        stage_telemetry.append(phase1_telemetry)
        query_events.append(
            RouteAFunnelQueryEvent(
                stage="child_phase1",
                phase="phase1",
                event_kind="route_a_child_phase1_gradient",
                probe_role="gradient",
                records=tuple(dict(record) for record in phase1_input_records),
                retained_record_count=int(len(phase1_records)),
            )
        )
        phase2_evaluated_records = _evaluate_records(
            phase1_records,
            phase2_record_evaluator or full_record_evaluator,
        )
        if (
            str(config.phase2_selector_mode)
            == ROUTE_A_PHASE2_SELECTOR_JOINT_RESPONSE_SINGLETON_V1
        ):
            if phase2_population_evaluator is None:
                raise ValueError(
                    "joint_response_singleton_v1 requires a typed Phase-II "
                    "population evaluator."
                )
            if phase2_evaluated_records:
                phase2_evaluation = phase2_population_evaluator(
                    phase2_evaluated_records
                )
                phase2_evaluated_records = [
                    dict(record) for record in phase2_evaluation.records
                ]
                phase2_selector_telemetry = {
                    **dict(phase2_evaluation.telemetry),
                    "active": True,
                }
            else:
                phase2_selector_telemetry = {
                    "schema": "route_a_phase2_empty_population_v1",
                    "active": True,
                    "status": "empty_after_candidate_contract_filter",
                    "input_record_count": 0,
                    "evaluated_record_count": 0,
                    "candidate_pair_measurement_count": 0,
                    "geometry_workspace": {
                        "schema": "batch_full_geometry_workspace_empty_v1",
                        "search_population_count": 0,
                        "matrix_element_query_charge": 0,
                        "query_chargeable_unique_geometry_element_count": 0,
                    },
                }
        if phase2_population_rescorer is not None:
            phase2_evaluated_records = [
                dict(record)
                for record in phase2_population_rescorer(
                    phase2_evaluated_records
                )
            ]
        phase2_records, phase2_telemetry = _stage_shortlist(
            phase2_evaluated_records,
            stage="child_phase2",
            score_key="phase2_raw_score",
            tie_break_score_key="phase1_active_score",
            cap=int(config.child_phase2_cap),
            child_identity_policy=str(config.child_identity_policy),
            feature_updater=feature_updater,
        )
        stage_telemetry.append(phase2_telemetry)
        query_events.append(
            RouteAFunnelQueryEvent(
                stage="child_phase2",
                phase="phase2",
                event_kind="route_a_child_phase2_metric",
                probe_role="metric",
                records=tuple(dict(record) for record in phase2_evaluated_records),
                retained_record_count=int(len(phase2_records)),
                reused_probe_role="gradient",
                reused_record_count=int(len(phase2_evaluated_records)),
            )
        )
        phase3_input = phase2_records
    else:
        phase1_records = list(phase1_input_records)
        phase2_records = _evaluate_records(
            phase1_input_records,
            phase2_record_evaluator or full_record_evaluator,
        )
        if phase2_population_rescorer is not None:
            phase2_records = [
                dict(record)
                for record in phase2_population_rescorer(phase2_records)
            ]
        phase3_input = list(phase2_records)

    if str(config.mode) in ROUTE_A_FUNNEL_CHILD_12_MODES:
        phase3_records: list[dict[str, Any]] = []
        selection_records = list(phase2_records)
    else:
        if phase3_pre_evaluation_rescorer is not None:
            phase3_input = [
                dict(record)
                for record in phase3_pre_evaluation_rescorer(phase3_input)
            ]
        phase3_evaluated_records = _evaluate_records(
            phase3_input,
            phase3_record_evaluator,
        )
        if phase3_population_rescorer is not None:
            phase3_evaluated_records = [
                dict(record)
                for record in phase3_population_rescorer(
                    phase3_evaluated_records
                )
            ]
        phase3_records, phase3_telemetry = _stage_shortlist(
            phase3_evaluated_records,
            stage="child_phase3",
            score_key="full_v2_score",
            tie_break_score_key="phase3_tie_break_score",
            cap=int(config.child_phase3_cap),
            child_identity_policy=str(config.child_identity_policy),
            feature_updater=feature_updater,
        )
        stage_telemetry.append(phase3_telemetry)
        selection_records = list(phase3_records)
        if str(config.mode) == ROUTE_A_FUNNEL_HIERARCHICAL_CHILD_123_V1:
            query_events.append(
                RouteAFunnelQueryEvent(
                    stage="child_phase3",
                    phase="phase3",
                    event_kind="route_a_child_phase3_metric",
                    probe_role="metric",
                    records=(),
                    retained_record_count=int(len(phase3_records)),
                    reused_records=tuple(dict(record) for record in phase3_input),
                    reused_probe_role="phase2_metric_and_phase3_geometry",
                    reused_record_count=int(len(phase3_input)),
                )
            )
        else:
            query_events.extend(
                (
                    RouteAFunnelQueryEvent(
                        stage="child_phase3",
                        phase="phase1",
                        event_kind="route_a_direct_child_phase3_gradient",
                        probe_role="gradient",
                        records=tuple(dict(record) for record in phase3_input),
                        retained_record_count=int(len(phase3_records)),
                    ),
                    RouteAFunnelQueryEvent(
                        stage="child_phase3",
                        phase="phase3",
                        event_kind="route_a_direct_child_phase3_metric",
                        probe_role="metric",
                        records=tuple(dict(record) for record in phase3_input),
                        retained_record_count=int(len(phase3_records)),
                        reused_probe_role="gradient",
                        reused_record_count=int(len(phase3_input)),
                    ),
                )
            )

    query_event_payloads = [event.as_dict() for event in query_events]
    query_work = {
        "schema": "route_a_child_funnel_query_work_v1",
        "events": query_event_payloads,
        "N_grad_probe": int(
            sum(
                len(event.records)
                for event in query_events
                if str(event.probe_role) == "gradient"
            )
        ),
        "N_metric_probe": int(
            sum(
                len(event.records)
                for event in query_events
                if str(event.probe_role) == "metric"
            )
        ),
        "N_grad_record_proxy": int(
            sum(
                len(event.records)
                for event in query_events
                if str(event.probe_role) == "gradient"
            )
        ),
        "N_metric_record_proxy": int(
            sum(
                len(event.records)
                for event in query_events
                if str(event.probe_role) == "metric"
            )
        ),
        "reused_record_count": int(
            sum(int(event.reused_record_count) for event in query_events)
        ),
        "reused_measurement_record_count": int(
            sum(len(event.reused_records) for event in query_events)
        ),
        "reuse_is_zero_incremental_query_work": True,
    }
    telemetry = {
        "schema": "route_a_macro_to_child_funnel_v1",
        "config": config.as_dict(),
        "child_population": dict(dedup_telemetry),
        "child_padding_filter": dict(child_padding_telemetry),
        "candidate_record_filter": dict(candidate_filter_telemetry),
        "child_phase1_evaluation": phase1_evaluation_telemetry,
        **(
            {"phase2_selector": phase2_selector_telemetry}
            if str(config.mode) == ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2
            else {}
        ),
        "parent_record_count": int(len(parent_records)),
        "parent_records_admissible": bool(
            str(config.population_mode)
            == ROUTE_A_PHASE3_POPULATION_PARENT_PLUS_CHILD_ABLATION_V1
        ),
        "macro_phase3_skipped": True,
        "phase0_executed": bool(
            str(config.phase0_policy) != ROUTE_A_PHASE0_DISABLED
        ),
        "child_phase3_skipped": bool(
            str(config.mode) in ROUTE_A_FUNNEL_CHILD_12_MODES
        ),
        "stage_order": [str(row["stage"]) for row in stage_telemetry],
        "stages": stage_telemetry,
        "full_record_evaluator_active": bool(full_record_evaluator is not None),
        "full_record_evaluation_count": int(
            len(phase2_evaluated_records)
            if hierarchical_child_stages
            else len(phase2_records)
        ),
        "final_record_count": int(len(selection_records)),
        "selection_record_source": (
            "child_phase2"
            if str(config.mode) in ROUTE_A_FUNNEL_CHILD_12_MODES
            else "child_phase3"
        ),
    }
    return RouteAFunnelResult(
        child_population=tuple(dict(record) for record in phase1_input_records),
        child_phase1_records=tuple(dict(record) for record in phase1_records),
        child_phase2_records=tuple(dict(record) for record in phase2_records),
        child_phase3_records=tuple(dict(record) for record in phase3_records),
        selection_records=tuple(dict(record) for record in selection_records),
        query_events=tuple(query_events),
        telemetry=telemetry,
        query_work=query_work,
    )


__all__ = [
    "ROUTE_A_BATCHING_COMBINATORIAL_REDUCED_PLANE",
    "ROUTE_A_BATCHING_GREEDY_REDUCED_PLANE",
    "ROUTE_A_BATCHING_MODES",
    "ROUTE_A_BATCHING_OFF",
    "ROUTE_A_FUNNEL_DIRECT_CHILD_PHASE3_V1",
    "ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2",
    "ROUTE_A_FUNNEL_CHILD_12_MODES",
    "ROUTE_A_FUNNEL_CHILD_12_SCHUR_V1",
    "ROUTE_A_FUNNEL_HIERARCHICAL_CHILD_123_V1",
    "ROUTE_A_FUNNEL_MODES",
    "ROUTE_A_PHASE0_DISABLED",
    "ROUTE_A_PHASE0_LEGACY_MACRO_PRESCREEN_V1",
    "ROUTE_A_PHASE0_POLICIES",
    "ROUTE_A_PHASE2_SELECTOR_JOINT_RESPONSE_SINGLETON_V1",
    "ROUTE_A_PHASE2_SELECTOR_MODES",
    "ROUTE_A_PHASE2_SELECTOR_RAW_GAIN_TIMES_NOVELTY_LEGACY_V1",
    "ROUTE_A_PHASE3_POPULATION_CHILD_ONLY_V1",
    "ROUTE_A_PHASE3_POPULATION_MODES",
    "ROUTE_A_PHASE3_POPULATION_PARENT_PLUS_CHILD_ABLATION_V1",
    "RouteAFunnelConfig",
    "RouteAFunnelQueryEvent",
    "RouteAFunnelResult",
    "run_route_a_child_funnel",
]
