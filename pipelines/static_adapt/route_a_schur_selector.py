"""Typed adapter for the canonical Route-A joint Schur selector."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_scoring import (
    BATCH_GEOMETRY_DIAGONAL_HESSIAN_DIAGNOSTIC_V1,
    BATCH_GEOMETRY_FULL_RESIDUAL_GRAM_HESSIAN_V1,
    BATCH_GEOMETRY_MODES,
    BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1,
    BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1,
    BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
    BATCH_JOINT_CONTEXT_MODES,
    BATCH_SEARCH_POPULATION_RANKED_CHILD_PHASE2_V1,
    BATCH_SEARCH_FEASIBILITY_POLICIES,
    BATCH_SEARCH_FEASIBILITY_JOINT_SUBSET_GATE_V1,
    PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE,
    PHASE2_BATCH_GREEDY_REDUCED_PLANE,
    BatchSelectionProposal,
    FullScoreConfig,
    Phase2JointResponseEvaluation,
    combinatorial_reduced_plane_batch_proposals,
    evaluate_phase2_joint_response_singletons,
    greedy_reduced_plane_batch_proposals,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_POLICIES,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
)
from pipelines.static_adapt.route_a_shortlists import (
    CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
    deduplicate_child_position_records,
)


ROUTE_A_SCHUR_GREEDY_REDUCED_PLANE = PHASE2_BATCH_GREEDY_REDUCED_PLANE
ROUTE_A_SCHUR_COMBINATORIAL_REDUCED_PLANE = (
    PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE
)
ROUTE_A_SCHUR_SELECTION_MODES = frozenset(
    {
        ROUTE_A_SCHUR_GREEDY_REDUCED_PLANE,
        ROUTE_A_SCHUR_COMBINATORIAL_REDUCED_PLANE,
    }
)
ROUTE_A_ACTIVE_CONTEXT_EXPLICIT_INDICES_V1 = "explicit_indices_v1"
ROUTE_A_ACTIVE_CONTEXT_TAIL_WINDOW_V1 = "tail_window_v1"
ROUTE_A_ACTIVE_CONTEXT_POLICIES = frozenset(
    {
        ROUTE_A_ACTIVE_CONTEXT_EXPLICIT_INDICES_V1,
        ROUTE_A_ACTIVE_CONTEXT_TAIL_WINDOW_V1,
    }
)

ROUTE_A_TRUST_REGION_FIXED = "fixed"
ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1 = (
    "displacement_calibrated_v1"
)
ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2 = (
    "displacement_calibrated_unbounded_v2"
)
ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1 = (
    "source_metric_inverse_sqrt_no_overlap_v1"
)
ROUTE_A_TRUST_REGION_UPDATE_POLICIES = frozenset(
    {
        ROUTE_A_TRUST_REGION_FIXED,
        ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
        ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
        ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1,
    }
)
ROUTE_A_SELECTOR_EXHAUSTION_STOP = "stop"
ROUTE_A_SELECTOR_EXHAUSTION_EXPAND_ALL_THEN_FORCE_SINGLETON_V1 = (
    "expand_all_then_force_singleton_v1"
)
ROUTE_A_SELECTOR_EXHAUSTION_POLICIES = frozenset(
    {
        ROUTE_A_SELECTOR_EXHAUSTION_STOP,
        ROUTE_A_SELECTOR_EXHAUSTION_EXPAND_ALL_THEN_FORCE_SINGLETON_V1,
    }
)


@dataclass(frozen=True)
class TrustRegionUpdateConfig:
    """Branch-local update policy for the selector's FS trust radius."""

    policy: str = ROUTE_A_TRUST_REGION_FIXED
    radius_min: float = 0.0
    contraction_factor_min: float = 0.5
    expansion_factor_max: float = math.sqrt(2.0)
    displacement_epsilon: float = 1e-12
    direction_cosine_min: float = 0.5
    require_direction_for_expansion: bool = False

    def __post_init__(self) -> None:
        if str(self.policy) not in ROUTE_A_TRUST_REGION_UPDATE_POLICIES:
            raise ValueError(
                "trust-region update policy must be one of "
                f"{sorted(ROUTE_A_TRUST_REGION_UPDATE_POLICIES)}."
            )
        for name in (
            "radius_min",
            "contraction_factor_min",
            "expansion_factor_max",
            "displacement_epsilon",
            "direction_cosine_min",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if float(self.radius_min) < 0.0:
            raise ValueError("radius_min must be nonnegative.")
        if not 0.0 < float(self.contraction_factor_min) <= 1.0:
            raise ValueError("contraction_factor_min must be in (0, 1].")
        if float(self.expansion_factor_max) < 1.0:
            raise ValueError("expansion_factor_max must be >= 1.")
        if float(self.displacement_epsilon) <= 0.0:
            raise ValueError("displacement_epsilon must be positive.")
        if not -1.0 <= float(self.direction_cosine_min) <= 1.0:
            raise ValueError("direction_cosine_min must be in [-1, 1].")

    def as_dict(self) -> dict[str, Any]:
        unbounded = bool(
            str(self.policy)
            in {
                ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
                ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1,
            }
        )
        return {
            "policy": str(self.policy),
            "radius_min": float(self.radius_min),
            "contraction_factor_min": float(self.contraction_factor_min),
            "expansion_factor_max": float(self.expansion_factor_max),
            "scientific_radius_min_effective": (
                0.0 if unbounded else float(self.radius_min)
            ),
            "scientific_radius_max_effective": None,
            "contraction_factor_min_effective": (
                None if unbounded else float(self.contraction_factor_min)
            ),
            "expansion_factor_max_effective": (
                None if unbounded else float(self.expansion_factor_max)
            ),
            "rate_limiter_mode": "none" if unbounded else "clamped_v1",
            "displacement_epsilon": float(self.displacement_epsilon),
            "direction_cosine_min": float(self.direction_cosine_min),
            "require_direction_for_expansion": bool(
                self.require_direction_for_expansion
            ),
        }


@dataclass(frozen=True)
class RouteASchurSelectorConfig:
    """Canonical controls independent of the historical Phase-III surface."""

    mode: str = ROUTE_A_SCHUR_COMBINATORIAL_REDUCED_PLANE
    batch_size_cap: int = 1
    batch_search_pool_size: int = 10
    batch_search_feasibility_policy: str = (
        BATCH_SEARCH_FEASIBILITY_JOINT_SUBSET_GATE_V1
    )
    score_tie_tolerance: float = 1e-12
    rank_relative_tolerance: float = 1e-6
    max_gram_condition_number: float = 1e12
    geometry_mode: str = BATCH_GEOMETRY_FULL_RESIDUAL_GRAM_HESSIAN_V1
    metric_regularization: float = 1e-9
    energy_regularization: float = 1e-9
    joint_linear_solve_policy: str = (
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
    )
    state_consistency_tolerance: float = 1e-8
    max_fubini_study_step: float = 0.25
    joint_batch_context_mode: str = BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1
    active_context_policy: str = ROUTE_A_ACTIVE_CONTEXT_EXPLICIT_INDICES_V1
    active_window_size: int | None = None
    active_context_indices: tuple[int, ...] | None = None
    trust_region_update: TrustRegionUpdateConfig = field(
        default_factory=TrustRegionUpdateConfig
    )
    exhaustion_retry_policy: str = ROUTE_A_SELECTOR_EXHAUSTION_STOP

    def __post_init__(self) -> None:
        if str(self.mode) not in ROUTE_A_SCHUR_SELECTION_MODES:
            raise ValueError(
                f"mode must be one of {sorted(ROUTE_A_SCHUR_SELECTION_MODES)}."
            )
        if int(self.batch_size_cap) < 1:
            raise ValueError("batch_size_cap must be >= 1.")
        if int(self.batch_search_pool_size) < 0:
            raise ValueError("batch_search_pool_size must be >= 0; 0 means all.")
        if (
            str(self.batch_search_feasibility_policy)
            not in BATCH_SEARCH_FEASIBILITY_POLICIES
        ):
            raise ValueError(
                "batch_search_feasibility_policy must be one of "
                f"{sorted(BATCH_SEARCH_FEASIBILITY_POLICIES)}."
            )
        if str(self.geometry_mode) not in BATCH_GEOMETRY_MODES:
            raise ValueError(
                f"geometry_mode must be one of {sorted(BATCH_GEOMETRY_MODES)}."
            )
        if str(self.joint_batch_context_mode) not in BATCH_JOINT_CONTEXT_MODES:
            raise ValueError(
                "joint_batch_context_mode must be one of "
                f"{sorted(BATCH_JOINT_CONTEXT_MODES)}."
            )
        if str(self.joint_linear_solve_policy) not in JOINT_LINEAR_SOLVE_POLICIES:
            raise ValueError(
                "joint_linear_solve_policy must be one of "
                f"{sorted(JOINT_LINEAR_SOLVE_POLICIES)}."
            )
        if str(self.active_context_policy) not in ROUTE_A_ACTIVE_CONTEXT_POLICIES:
            raise ValueError(
                "active_context_policy must be one of "
                f"{sorted(ROUTE_A_ACTIVE_CONTEXT_POLICIES)}."
            )
        if not isinstance(self.trust_region_update, TrustRegionUpdateConfig):
            raise TypeError(
                "trust_region_update must be a TrustRegionUpdateConfig."
            )
        if str(self.exhaustion_retry_policy) not in ROUTE_A_SELECTOR_EXHAUSTION_POLICIES:
            raise ValueError(
                "exhaustion_retry_policy must be one of "
                f"{sorted(ROUTE_A_SELECTOR_EXHAUSTION_POLICIES)}."
            )
        if str(self.joint_batch_context_mode) == BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1:
            if (
                str(self.active_context_policy)
                == ROUTE_A_ACTIVE_CONTEXT_EXPLICIT_INDICES_V1
                and self.active_context_indices is None
            ):
                raise ValueError(
                    "active_window_v1 with explicit_indices_v1 requires "
                    "active_context_indices."
                )
            if (
                str(self.active_context_policy)
                == ROUTE_A_ACTIVE_CONTEXT_TAIL_WINDOW_V1
                and (self.active_window_size is None or int(self.active_window_size) < 1)
            ):
                raise ValueError(
                    "active_window_v1 with tail_window_v1 requires "
                    "active_window_size >= 1."
                )
        if self.active_window_size is not None and int(self.active_window_size) < 1:
            raise ValueError("active_window_size must be >= 1 when provided.")
        if self.active_context_indices is not None:
            resolved_indices = tuple(int(index) for index in self.active_context_indices)
            if len(set(resolved_indices)) != len(resolved_indices):
                raise ValueError("active_context_indices contains duplicates.")
            if any(index < 0 for index in resolved_indices):
                raise ValueError("active_context_indices must be nonnegative.")
            object.__setattr__(self, "active_context_indices", resolved_indices)
        for name in (
            "score_tie_tolerance",
            "rank_relative_tolerance",
            "max_gram_condition_number",
            "metric_regularization",
            "energy_regularization",
            "state_consistency_tolerance",
            "max_fubini_study_step",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if float(self.max_fubini_study_step) <= 0.0:
            raise ValueError("max_fubini_study_step must be positive.")
        if float(self.max_gram_condition_number) < 1.0:
            raise ValueError("max_gram_condition_number must be >= 1.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "batch_size_cap": int(self.batch_size_cap),
            "batch_search_pool_size": int(self.batch_search_pool_size),
            "batch_search_pool_size_semantics": "zero_means_all",
            "batch_search_feasibility_policy": str(
                self.batch_search_feasibility_policy
            ),
            "score_tie_tolerance": float(self.score_tie_tolerance),
            "rank_relative_tolerance": float(self.rank_relative_tolerance),
            "max_gram_condition_number": float(
                self.max_gram_condition_number
            ),
            "geometry_mode": str(self.geometry_mode),
            "metric_regularization": float(self.metric_regularization),
            "energy_regularization": float(self.energy_regularization),
            "joint_linear_solve_policy": str(self.joint_linear_solve_policy),
            "state_consistency_tolerance": float(
                self.state_consistency_tolerance
            ),
            "max_fubini_study_step": float(self.max_fubini_study_step),
            "joint_batch_context_mode": str(self.joint_batch_context_mode),
            "active_context_policy": str(self.active_context_policy),
            "active_window_size": (
                None
                if self.active_window_size is None
                else int(self.active_window_size)
            ),
            "active_context_indices": (
                None
                if self.active_context_indices is None
                else [int(index) for index in self.active_context_indices]
            ),
            "trust_region_update": self.trust_region_update.as_dict(),
            "exhaustion_retry_policy": str(self.exhaustion_retry_policy),
        }


def route_a_schur_score_config(
    base: FullScoreConfig,
    *,
    config: RouteASchurSelectorConfig,
) -> FullScoreConfig:
    """Project typed Route-A controls onto the existing joint scorer."""

    return replace(
        base,
        batch_selection_mode=str(config.mode),
        batch_target_size=int(config.batch_size_cap),
        batch_size_cap=int(config.batch_size_cap),
        batch_search_pool_size=int(config.batch_search_pool_size),
        batch_search_population_mode=BATCH_SEARCH_POPULATION_RANKED_CHILD_PHASE2_V1,
        batch_search_feasibility_policy=str(
            config.batch_search_feasibility_policy
        ),
        batch_score_tie_tolerance=float(config.score_tie_tolerance),
        batch_rank_rel_tol=float(config.rank_relative_tolerance),
        batch_max_gram_condition_number=float(
            config.max_gram_condition_number
        ),
        batch_geometry_mode=str(config.geometry_mode),
        batch_metric_regularization=float(config.metric_regularization),
        batch_energy_regularization=float(config.energy_regularization),
        batch_joint_linear_solve_policy=str(config.joint_linear_solve_policy),
        batch_state_consistency_tolerance=float(
            config.state_consistency_tolerance
        ),
        batch_joint_context_mode=str(config.joint_batch_context_mode),
        batch_active_context_indices=config.active_context_indices,
        rho=float(config.max_fubini_study_step),
    )


def resolve_route_a_schur_context(
    config: RouteASchurSelectorConfig,
    *,
    active_ansatz_depth: int,
) -> RouteASchurSelectorConfig:
    """Resolve a dynamic active window once for the current selector round."""

    if str(config.joint_batch_context_mode) != BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1:
        return config
    if str(config.active_context_policy) == ROUTE_A_ACTIVE_CONTEXT_TAIL_WINDOW_V1:
        window_size = int(config.active_window_size or 0)
        start = int(max(0, int(active_ansatz_depth) - int(window_size)))
        resolved = tuple(range(start, int(active_ansatz_depth)))
    else:
        resolved = tuple(int(index) for index in config.active_context_indices or ())
    if any(index >= int(active_ansatz_depth) for index in resolved):
        raise ValueError(
            "Resolved active-window index is an out-of-range ansatz index for "
            "the current depth."
        )
    return replace(config, active_context_indices=resolved)


@dataclass
class RouteAJointResponseEvaluator:
    """Typed Phase-II adapter over the shared singleton joint solver."""

    config: RouteASchurSelectorConfig
    score_config: FullScoreConfig
    selected_ops: Sequence[Any]
    theta: np.ndarray
    psi_ref: np.ndarray
    psi_state: np.ndarray
    h_compiled: Any
    pauli_action_cache: dict[str, Any] | None = None
    scope: str = "phase2"

    def __call__(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> Phase2JointResponseEvaluation:
        resolved_config = resolve_route_a_schur_context(
            self.config,
            active_ansatz_depth=int(len(self.selected_ops)),
        )
        resolved_score_config = route_a_schur_score_config(
            self.score_config,
            config=resolved_config,
        )
        evaluation = evaluate_phase2_joint_response_singletons(
            records,
            cfg=resolved_score_config,
            selected_ops=self.selected_ops,
            theta=np.asarray(self.theta, dtype=float),
            psi_ref=np.asarray(self.psi_ref, dtype=complex),
            psi_state=np.asarray(self.psi_state, dtype=complex),
            h_compiled=self.h_compiled,
            pauli_action_cache=self.pauli_action_cache,
            scope=str(self.scope),
        )
        return Phase2JointResponseEvaluation(
            records=tuple(dict(record) for record in evaluation.records),
            telemetry={
                **dict(evaluation.telemetry),
                "selector_config": self.config.as_dict(),
                "resolved_context_config": resolved_config.as_dict(),
                "active_context_indices_effective": [
                    int(index)
                    for index in evaluation.workspace.active_indices
                ],
            },
            workspace=evaluation.workspace,
        )


def select_route_a_schur_proposals(
    records: Sequence[Mapping[str, Any]],
    *,
    config: RouteASchurSelectorConfig,
    score_config: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: Any,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, Any] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    max_proposals: int = 1,
) -> tuple[list[BatchSelectionProposal], dict[str, Any]]:
    """Run the preserved greedy/combinatorial joint reduced-plane algorithm."""

    canonical_records, identity_telemetry = deduplicate_child_position_records(
        records,
        score_key="phase2_raw_score",
        tie_break_score_key="phase1_active_score",
        identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
    )

    resolved_config = resolve_route_a_schur_context(
        config,
        active_ansatz_depth=int(len(selected_ops)),
    )
    resolved_score_config = route_a_schur_score_config(
        score_config,
        config=resolved_config,
    )
    proposal_builder = (
        greedy_reduced_plane_batch_proposals
        if str(resolved_config.mode) == ROUTE_A_SCHUR_GREEDY_REDUCED_PLANE
        else combinatorial_reduced_plane_batch_proposals
    )
    proposals, summary = proposal_builder(
        canonical_records,
        cfg=resolved_score_config,
        selected_ops=selected_ops,
        theta=np.asarray(theta, dtype=float),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        psi_state=np.asarray(psi_state, dtype=complex),
        h_compiled=h_compiled,
        novelty_oracle=novelty_oracle,
        curvature_oracle=curvature_oracle,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        tie_break_score_key="phase2_raw_score",
        max_proposals=int(max(1, max_proposals)),
    )
    initial_summary = dict(summary)
    exhaustion_retry = {
        "schema": "route_a_joint_selector_exhaustion_retry_v1",
        "policy": str(config.exhaustion_retry_policy),
        "triggered": bool(not proposals),
        "expanded_to_all_child_phase2_survivors": False,
        "initial_batch_search_pool_size": int(config.batch_search_pool_size),
        "retry_batch_search_pool_size": None,
        "canonical_child_phase2_survivor_count": int(len(canonical_records)),
        "recovered_proposal_count": 0,
        "forced_singleton_required": False,
        "initial_summary": initial_summary,
    }
    if (
        not proposals
        and str(config.exhaustion_retry_policy)
        == ROUTE_A_SELECTOR_EXHAUSTION_EXPAND_ALL_THEN_FORCE_SINGLETON_V1
    ):
        retry_config = replace(config, batch_search_pool_size=0)
        retry_resolved_config = resolve_route_a_schur_context(
            retry_config,
            active_ansatz_depth=int(len(selected_ops)),
        )
        retry_score_config = route_a_schur_score_config(
            score_config,
            config=retry_resolved_config,
        )
        proposals, retry_summary = proposal_builder(
            canonical_records,
            cfg=retry_score_config,
            selected_ops=selected_ops,
            theta=np.asarray(theta, dtype=float),
            psi_ref=np.asarray(psi_ref, dtype=complex),
            psi_state=np.asarray(psi_state, dtype=complex),
            h_compiled=h_compiled,
            novelty_oracle=novelty_oracle,
            curvature_oracle=curvature_oracle,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
            tie_break_score_key="phase2_raw_score",
            max_proposals=int(max(1, max_proposals)),
        )
        summary = dict(retry_summary)
        exhaustion_retry.update(
            {
                "expanded_to_all_child_phase2_survivors": True,
                "retry_batch_search_pool_size": 0,
                "recovered_proposal_count": int(len(proposals)),
                "forced_singleton_required": bool(not proposals),
                "retry_summary": dict(retry_summary),
            }
        )
    summary_out = {
        "schema": "route_a_joint_schur_selector_v1",
        **dict(summary),
        "config": config.as_dict(),
        "resolved_context_config": resolved_config.as_dict(),
        "active_context_selection_policy": str(
            resolved_config.active_context_policy
        ),
        "active_context_indices_effective": (
            None
            if resolved_config.active_context_indices is None
            else [int(index) for index in resolved_config.active_context_indices]
        ),
        "child_phase2_survivor_count_input": int(len(records)),
        "child_phase2_survivor_count": int(len(canonical_records)),
        "global_child_identity_safety_dedup": dict(identity_telemetry),
        "global_dedup_applied_before_search_pool": True,
        "global_dedup_applied_before_joint_geometry_workspace": True,
        "child_phase2_measurement_reuse": True,
        "reused_child_phase2_record_count": int(len(canonical_records)),
        "canonical_selection_stage": "post_child_phase2_joint_selector",
        "measurement_reuse_source_stages": ["child_phase1", "child_phase2"],
        "final_selection_authority": "joint_schur_score",
        "score_formula": "DeltaE_Schur(B)/(1+K(B))",
        "exhaustion_retry": exhaustion_retry,
    }
    return proposals, summary_out


__all__ = [
    "ROUTE_A_ACTIVE_CONTEXT_EXPLICIT_INDICES_V1",
    "ROUTE_A_ACTIVE_CONTEXT_POLICIES",
    "ROUTE_A_ACTIVE_CONTEXT_TAIL_WINDOW_V1",
    "ROUTE_A_SCHUR_COMBINATORIAL_REDUCED_PLANE",
    "ROUTE_A_SCHUR_GREEDY_REDUCED_PLANE",
    "ROUTE_A_SCHUR_SELECTION_MODES",
    "ROUTE_A_SELECTOR_EXHAUSTION_EXPAND_ALL_THEN_FORCE_SINGLETON_V1",
    "ROUTE_A_SELECTOR_EXHAUSTION_POLICIES",
    "ROUTE_A_SELECTOR_EXHAUSTION_STOP",
    "ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1",
    "ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2",
    "ROUTE_A_TRUST_REGION_FIXED",
    "ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1",
    "ROUTE_A_TRUST_REGION_UPDATE_POLICIES",
    "BATCH_GEOMETRY_DIAGONAL_HESSIAN_DIAGNOSTIC_V1",
    "BATCH_GEOMETRY_FULL_RESIDUAL_GRAM_HESSIAN_V1",
    "BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1",
    "BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1",
    "BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1",
    "JOINT_LINEAR_SOLVE_POLICIES",
    "JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1",
    "RouteAJointResponseEvaluator",
    "RouteASchurSelectorConfig",
    "TrustRegionUpdateConfig",
    "route_a_schur_score_config",
    "resolve_route_a_schur_context",
    "select_route_a_schur_proposals",
]
