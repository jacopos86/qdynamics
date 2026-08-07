"""Resolved route profiles for Formal-Manifold SNAKE compositions.

The outer FM controller and its mutable manifold state remain distinct from
the candidate selector named here.  This module is deliberately declarative:
it owns no optimizer, selector, checkpoint, or circuit state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1,
    SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
    SR_ESCAPE_DISABLED,
    SR_POWELL_COORDINATE_CHART_AUTO,
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_REQUEST_CHOICES,
)


FORMAL_MANIFOLD_ROUTE_PROFILE_SCHEMA = "formal_manifold_route_profile_v1"
FORMAL_MANIFOLD_ROUTE_PROFILE_OFF = "off"
FM_ROUTE_PROFILE_SR_PHASE2_PHASE3_WHITENED_ADAPTIVE_TRUST_NO_N2_V1 = (
    "sr_phase2_phase3_whitened_adaptive_trust_no_n2_v1"
)
FM_ROUTE_PROFILE_SR_SOURCE_LOCKED_SUPPORTED_WHITENED_ADAPTIVE_TRUST_V1 = (
    "sr_source_locked_supported_whitened_adaptive_trust_v1"
)
FORMAL_MANIFOLD_ROUTE_PROFILE_CHOICES = (
    FORMAL_MANIFOLD_ROUTE_PROFILE_OFF,
    FM_ROUTE_PROFILE_SR_PHASE2_PHASE3_WHITENED_ADAPTIVE_TRUST_NO_N2_V1,
    FM_ROUTE_PROFILE_SR_SOURCE_LOCKED_SUPPORTED_WHITENED_ADAPTIVE_TRUST_V1,
)

FORMAL_MANIFOLD_ROUTE_FAMILY = "formal_manifold_snake"
FORMAL_MANIFOLD_REOPTIMIZATION_ROUTE = "formal_manifold_warm_start_v1"
FORMAL_MANIFOLD_SR_CANDIDATE_SELECTOR_FAMILY = "singleton_response_snake"
FORMAL_MANIFOLD_SR_CANDIDATE_SELECTOR_PROFILE = "sr_singleton_controller_v1"
FORMAL_MANIFOLD_SR_SOURCE_LOCKED_CANDIDATE_SELECTOR_PROFILE = (
    "supported_whitened_adaptive_trust_v1"
)
FORMAL_MANIFOLD_ADAPTIVE_TRUST_POLICY = (
    "displacement_calibrated_unbounded_v2"
)

NOVELTY_MULTIPLIER_POLICY_LEGACY_ABLATION_MODE_V1 = (
    "legacy_ablation_mode_v1"
)
NOVELTY_MULTIPLIER_POLICY_INACTIVE_ORDINARY_ROUTE_V1 = (
    "inactive_ordinary_route_v1"
)
NOVELTY_MULTIPLIER_POLICY_CHOICES = (
    NOVELTY_MULTIPLIER_POLICY_LEGACY_ABLATION_MODE_V1,
    NOVELTY_MULTIPLIER_POLICY_INACTIVE_ORDINARY_ROUTE_V1,
)


@dataclass(frozen=True)
class FormalManifoldRouteProfile:
    """Fully resolved, serialization-safe FM route-profile contract."""

    schema: str
    route_family: str
    route_profile: str
    adapt_reoptimization_route: str
    candidate_selector_family: str
    candidate_selector_profile: str
    historical_singleton_coordinate_solve_policy: str
    historical_singleton_coordinate_solve_scope: str
    historical_singleton_trust_region_update_policy: str
    sr_powell_coordinate_chart_policy: str
    sr_escape_mode: str
    phase2_novelty_mode: str
    phase3_novelty_ablation_mode: str
    phase2_novelty_multiplier_policy: str
    phase3_novelty_multiplier_policy: str
    phase2_gram_novelty_policy: str
    phase3_gram_novelty_policy: str
    phase0_pilot_enabled: bool
    phase2_enable_batching: bool
    phase3_enable_batching: bool
    structural_rollback_enabled: bool
    route_a_funnel_active: bool
    phase3_runtime_split_mode: str
    phase3_runtime_split_selection_mode: str
    phase3_runtime_split_subset_sizes: tuple[int, ...]
    phase3_runtime_split_child_set_symmetry_policy: str
    phase3_runtime_split_child_padding_policy: str
    candidate_response_model: str
    admission_cardinality: int
    prune_policy: str
    measured_n2_retained: bool
    additional_n3_multiplier_applied: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def normalize_formal_manifold_route_profile(value: Any) -> str:
    """Normalize one FM route-profile request and reject unknown profiles."""

    profile = str(
        FORMAL_MANIFOLD_ROUTE_PROFILE_OFF
        if value is None or value == ""
        else value
    ).strip().lower()
    if profile not in FORMAL_MANIFOLD_ROUTE_PROFILE_CHOICES:
        raise ValueError(
            "formal_manifold_route_profile must be one of "
            f"{list(FORMAL_MANIFOLD_ROUTE_PROFILE_CHOICES)}; got {value!r}."
        )
    return profile


def resolve_formal_manifold_route_profile(
    value: Any,
    *,
    requested_powell_coordinate_chart_policy: str = (
        SR_POWELL_COORDINATE_CHART_AUTO
    ),
) -> FormalManifoldRouteProfile | None:
    """Resolve the opt-in FM/SR composition without changing standalone SR.

    This FM profile intentionally uses the expanded-runtime projected-logical
    Powell chart.  Standalone SR's Phase-II+III resolver continues to select
    its existing reduced-logical chart and is not modified by this registry.
    """

    profile = normalize_formal_manifold_route_profile(value)
    if profile == FORMAL_MANIFOLD_ROUTE_PROFILE_OFF:
        return None

    requested_powell = str(
        requested_powell_coordinate_chart_policy
    ).strip().lower()
    if requested_powell not in SR_POWELL_COORDINATE_CHART_REQUEST_CHOICES:
        raise ValueError(
            "requested_powell_coordinate_chart_policy must be one of "
            f"{list(SR_POWELL_COORDINATE_CHART_REQUEST_CHOICES)}."
        )
    if requested_powell not in {
        SR_POWELL_COORDINATE_CHART_AUTO,
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    }:
        raise ValueError(
            f"{profile} requires "
            f"{SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1}."
        )

    source_locked_profile = (
        profile
        == FM_ROUTE_PROFILE_SR_SOURCE_LOCKED_SUPPORTED_WHITENED_ADAPTIVE_TRUST_V1
    )
    return FormalManifoldRouteProfile(
        schema=FORMAL_MANIFOLD_ROUTE_PROFILE_SCHEMA,
        route_family=FORMAL_MANIFOLD_ROUTE_FAMILY,
        route_profile=profile,
        adapt_reoptimization_route=FORMAL_MANIFOLD_REOPTIMIZATION_ROUTE,
        candidate_selector_family=FORMAL_MANIFOLD_SR_CANDIDATE_SELECTOR_FAMILY,
        candidate_selector_profile=(
            FORMAL_MANIFOLD_SR_SOURCE_LOCKED_CANDIDATE_SELECTOR_PROFILE
            if source_locked_profile
            else FORMAL_MANIFOLD_SR_CANDIDATE_SELECTOR_PROFILE
        ),
        historical_singleton_coordinate_solve_policy=(
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
        ),
        historical_singleton_coordinate_solve_scope=(
            SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1
            if source_locked_profile
            else SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
        ),
        historical_singleton_trust_region_update_policy=(
            FORMAL_MANIFOLD_ADAPTIVE_TRUST_POLICY
        ),
        sr_powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
        sr_escape_mode=SR_ESCAPE_DISABLED,
        phase2_novelty_mode="collective_span_v1",
        # The source-locked profile preserves canonical SR novelty scoring.
        # The no-N2 profile keeps the raw observations while explicitly
        # removing their multipliers from both scores.
        phase3_novelty_ablation_mode=(
            "off" if source_locked_profile else "no_phase2"
        ),
        phase2_novelty_multiplier_policy=(
            NOVELTY_MULTIPLIER_POLICY_LEGACY_ABLATION_MODE_V1
            if source_locked_profile
            else NOVELTY_MULTIPLIER_POLICY_INACTIVE_ORDINARY_ROUTE_V1
        ),
        phase3_novelty_multiplier_policy=(
            NOVELTY_MULTIPLIER_POLICY_LEGACY_ABLATION_MODE_V1
            if source_locked_profile
            else NOVELTY_MULTIPLIER_POLICY_INACTIVE_ORDINARY_ROUTE_V1
        ),
        phase2_gram_novelty_policy="ordinary_multiplier_v1",
        phase3_gram_novelty_policy="ordinary_multiplier_v1",
        phase0_pilot_enabled=False,
        phase2_enable_batching=False,
        phase3_enable_batching=False,
        structural_rollback_enabled=False,
        route_a_funnel_active=False,
        phase3_runtime_split_mode="shortlist_pauli_children_v1",
        phase3_runtime_split_selection_mode="archival_child_set_forward_v1",
        phase3_runtime_split_subset_sizes=(1,),
        phase3_runtime_split_child_set_symmetry_policy="hard_guard",
        phase3_runtime_split_child_padding_policy=(
            "exact_projected_grouped_v1"
        ),
        candidate_response_model="full_active_plus_singleton_v1",
        admission_cardinality=1,
        prune_policy="recoverability_ladder_v1",
        measured_n2_retained=True,
        additional_n3_multiplier_applied=source_locked_profile,
    )


__all__ = [
    "FM_ROUTE_PROFILE_SR_PHASE2_PHASE3_WHITENED_ADAPTIVE_TRUST_NO_N2_V1",
    "FM_ROUTE_PROFILE_SR_SOURCE_LOCKED_SUPPORTED_WHITENED_ADAPTIVE_TRUST_V1",
    "FORMAL_MANIFOLD_REOPTIMIZATION_ROUTE",
    "FORMAL_MANIFOLD_ADAPTIVE_TRUST_POLICY",
    "FORMAL_MANIFOLD_ROUTE_FAMILY",
    "FORMAL_MANIFOLD_ROUTE_PROFILE_CHOICES",
    "FORMAL_MANIFOLD_ROUTE_PROFILE_OFF",
    "FORMAL_MANIFOLD_ROUTE_PROFILE_SCHEMA",
    "FORMAL_MANIFOLD_SR_CANDIDATE_SELECTOR_FAMILY",
    "FORMAL_MANIFOLD_SR_CANDIDATE_SELECTOR_PROFILE",
    "FORMAL_MANIFOLD_SR_SOURCE_LOCKED_CANDIDATE_SELECTOR_PROFILE",
    "FormalManifoldRouteProfile",
    "NOVELTY_MULTIPLIER_POLICY_CHOICES",
    "NOVELTY_MULTIPLIER_POLICY_INACTIVE_ORDINARY_ROUTE_V1",
    "NOVELTY_MULTIPLIER_POLICY_LEGACY_ABLATION_MODE_V1",
    "normalize_formal_manifold_route_profile",
    "resolve_formal_manifold_route_profile",
]
