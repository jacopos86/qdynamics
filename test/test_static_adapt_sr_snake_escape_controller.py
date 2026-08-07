from __future__ import annotations

import pytest

from pipelines.static_adapt.sr_snake_escape_controller import (
    NonstationaryCertificate,
    OrdinaryCertificate,
    PsdCertificate,
    QuotientRedundantCertificate,
    ReachablePopulationAudit,
    SRControllerDecisionKind,
    SREscapeMode,
    SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1,
    SR_ESCAPE_DISABLED,
    SR_ESCAPE_MODE_CHOICES,
    SR_ESCAPE_SADDLE_ONLY,
    SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM,
    SR_POWELL_COORDINATE_CHART_AUTO,
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    SR_ROUTE_FAMILY,
    SR_ROUTE_PROFILE_CONFORMANCE_REGISTERED,
    SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION,
    SR_ROUTE_PROFILE_DISABLED,
    SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED,
    SR_ROUTE_PROFILE_REDUCED_POWELL,
    SR_ROUTE_PROFILE_SADDLE_ONLY,
    SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM,
    SaddleCertificate,
    StateStationarityCertificate,
    UnresolvedCertificate,
    reachable_population_digest,
    resolve_sr_powell_coordinate_chart_policy,
    resolve_sr_powell_route_instance,
    saddle_acquisition,
    select_sr_escape_path,
    sr_route_profile,
)


def _candidate_coupled_saddle(
    *,
    record_id: str = "mixed",
    novelty_statistic: float = 0.4,
) -> SaddleCertificate:
    # H=[[1,2],[2,1]], g=0, rho=1: q_full=1/2 and q_active=0.
    return SaddleCertificate(
        record_id=record_id,
        stationarity_margin=-1.0e-8,
        minimum_eigenvalue_upper_bound=-1.0,
        full_trust_gain_lower_bound=0.5,
        active_trust_gain_lower_bound=0.0,
        active_trust_gain_upper_bound=0.0,
        quotient_participation_lower_bound=0.5,
        phase3_cost=3.0,
        novelty_statistic=novelty_statistic,
    )


def _audit(
    *certificates: object,
    state_stationarity: StateStationarityCertificate | None = None,
) -> ReachablePopulationAudit:
    return ReachablePopulationAudit(
        reachable_record_ids=tuple(
            certificate.record_id for certificate in certificates
        ),
        certificates=tuple(certificates),
        state_stationarity=state_stationarity,
    )


def _state_stationarity(*record_ids: str) -> StateStationarityCertificate:
    return StateStationarityCertificate(
        state_fingerprint="working-state",
        reachable_population_digest=reachable_population_digest(
            tuple(record_ids)
        ),
        comparison_epoch="comparison-epoch",
        support_provenance_digest="support-provenance",
        trust_provenance_digest="trust-provenance",
        trust_radius=0.25,
        stationarity_margin=-1.0e-8,
    )


def test_public_modes_use_stable_manifest_values() -> None:
    assert SREscapeMode.DISABLED.value == SR_ESCAPE_DISABLED
    assert SREscapeMode.SADDLE_ONLY.value == SR_ESCAPE_SADDLE_ONLY
    assert (
        SREscapeMode.SADDLE_PLUS_MODELED_MINIMUM.value
        == SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM
    )
    assert SR_ESCAPE_MODE_CHOICES == (
        "disabled",
        "saddle_only",
        "saddle_plus_modeled_minimum",
    )
    assert SR_ROUTE_FAMILY == "singleton_response_snake"
    assert resolve_sr_powell_coordinate_chart_policy(
        SR_ESCAPE_DISABLED,
        requested_policy=SR_POWELL_COORDINATE_CHART_AUTO,
    ) == SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    assert sr_route_profile(SR_ESCAPE_DISABLED) == SR_ROUTE_PROFILE_DISABLED
    assert (
        sr_route_profile(
            SR_ESCAPE_DISABLED,
            powell_coordinate_chart_policy=(
                SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
            ),
        )
        == SR_ROUTE_PROFILE_REDUCED_POWELL
    )
    assert (
        sr_route_profile(SR_ESCAPE_SADDLE_ONLY)
        == SR_ROUTE_PROFILE_SADDLE_ONLY
    )
    assert (
        sr_route_profile(SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM)
        == SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM
    )
    assert (
        sr_route_profile(
            SR_ESCAPE_DISABLED,
            coordinate_solve_scope=(
                SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
            ),
        )
        == SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED
    )
    assert resolve_sr_powell_coordinate_chart_policy(
        SR_ESCAPE_DISABLED,
        coordinate_solve_scope=(
            SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
        ),
        requested_policy=SR_POWELL_COORDINATE_CHART_AUTO,
    ) == SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    assert (
        sr_route_profile(
            SR_ESCAPE_DISABLED,
            coordinate_solve_scope=(
                SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
            ),
            powell_coordinate_chart_policy=(
                SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
            ),
        )
        == SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED
    )
    registered_resolution = resolve_sr_powell_route_instance(
        SR_ESCAPE_DISABLED,
        coordinate_solve_scope=(
            SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
        ),
        requested_policy=SR_POWELL_COORDINATE_CHART_AUTO,
    )
    assert registered_resolution["powell_coordinate_chart_policy"] == (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )
    assert registered_resolution["route_profile_conformance"] == (
        SR_ROUTE_PROFILE_CONFORMANCE_REGISTERED
    )
    ablation_resolution = resolve_sr_powell_route_instance(
        SR_ESCAPE_DISABLED,
        coordinate_solve_scope=(
            SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
        ),
        requested_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
    )
    assert ablation_resolution["route_profile"] == (
        SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED
    )
    assert ablation_resolution["powell_coordinate_chart_policy"] == (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )
    assert ablation_resolution["request_was_auto"] is False
    assert ablation_resolution["inferred_unpromoted_ablation"] is False
    assert ablation_resolution["route_profile_conformance"] == (
        SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
    )
    with pytest.raises(ValueError, match="non-escape"):
        sr_route_profile(
            SR_ESCAPE_SADDLE_ONLY,
            coordinate_solve_scope=(
                SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
            ),
        )


def test_resolved_ordinary_gain_has_strict_precedence_over_saddle() -> None:
    decision = select_sr_escape_path(
        mode=SREscapeMode.SADDLE_PLUS_MODELED_MINIMUM,
        ordinary=OrdinaryCertificate(
            record_id="ordinary-selected",
            gain_lower_bound=1.0e-3,
        ),
        audit=_audit(_candidate_coupled_saddle()),
    )

    assert decision.kind is SRControllerDecisionKind.ORDINARY
    assert decision.record_id == "ordinary-selected"
    assert decision.consumes_singleton is True
    assert decision.actionable is True
    assert decision.stage_b_eligible is False


def test_candidate_coupled_stationary_saddle_receives_marginal_credit() -> None:
    certificate = _candidate_coupled_saddle()
    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_ONLY,
        ordinary=None,
        audit=_audit(certificate),
    )

    assert certificate.marginal_gain_lower_bound == pytest.approx(0.5)
    assert decision.kind is SRControllerDecisionKind.SADDLE_SINGLETON
    assert decision.record_id == "mixed"
    assert decision.consumes_singleton is True
    assert decision.acquisition == pytest.approx(0.5 / (1.0 + 3.0))


def test_nonstationary_active_correction_precedes_saddle_classification() -> None:
    nonstationary = NonstationaryCertificate(
        record_id="active-refit",
        stationarity_margin=1.0e-3,
        active_trust_gain_lower_bound=2.0e-4,
        active_trust_gain_upper_bound=2.1e-4,
    )
    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_ONLY,
        ordinary=None,
        audit=_audit(nonstationary, _candidate_coupled_saddle()),
    )

    assert (
        decision.kind
        is SRControllerDecisionKind.ACTIVE_STATIONARITY_CORRECTION
    )
    assert decision.certificate_record_id == "active-refit"
    assert decision.record_id is None
    assert decision.actionable is True
    assert decision.consumes_singleton is False


def test_nonstationary_active_correction_precedes_unresolved_bystander() -> None:
    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_ONLY,
        ordinary=None,
        audit=_audit(
            NonstationaryCertificate(
                record_id="active-refit",
                stationarity_margin=1.0e-3,
                active_trust_gain_lower_bound=2.0e-4,
                active_trust_gain_upper_bound=2.1e-4,
            ),
            UnresolvedCertificate(
                record_id="unresolved-bystander",
                reason="supported_hessian_inertia_unresolved",
            ),
        ),
    )

    assert (
        decision.kind
        is SRControllerDecisionKind.ACTIVE_STATIONARITY_CORRECTION
    )
    assert decision.certificate_record_id == "active-refit"
    assert decision.actionable is True


def test_credited_saddle_precedes_unresolved_bystander() -> None:
    saddle = _candidate_coupled_saddle(record_id="credited-saddle")
    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_ONLY,
        ordinary=None,
        audit=_audit(
            saddle,
            UnresolvedCertificate(
                record_id="unresolved-bystander",
                reason="supported_hessian_inertia_unresolved",
            ),
        ),
    )

    assert decision.kind is SRControllerDecisionKind.SADDLE_SINGLETON
    assert decision.record_id == "credited-saddle"
    assert decision.certificate_record_id == "credited-saddle"
    assert decision.actionable is True
    assert decision.consumes_singleton is True


def test_active_only_negative_mode_consumes_no_singleton_transition() -> None:
    # H=diag(-1,1), g=0, rho=1 with the first coordinate active: the full and
    # active optima have the same gain, so Delta q_r=0.
    active_only = SaddleCertificate(
        record_id="active-only",
        stationarity_margin=-1.0e-8,
        minimum_eigenvalue_upper_bound=-1.0,
        full_trust_gain_lower_bound=0.5,
        active_trust_gain_lower_bound=0.5,
        active_trust_gain_upper_bound=0.5,
        quotient_participation_lower_bound=0.0,
        phase3_cost=1.0,
        novelty_statistic=1.0,
    )

    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_ONLY,
        ordinary=None,
        audit=_audit(active_only),
    )

    assert active_only.marginal_gain_lower_bound == pytest.approx(0.0)
    assert decision.kind is SRControllerDecisionKind.ACTIVE_ONLY_CORRECTION
    assert decision.record_id is None
    assert decision.certificate_record_id == "active-only"
    assert decision.consumes_singleton is False
    assert decision.actionable is True


def test_active_only_saddle_precedes_unresolved_bystander() -> None:
    active_only = SaddleCertificate(
        record_id="active-only",
        stationarity_margin=-1.0e-8,
        minimum_eigenvalue_upper_bound=-1.0,
        full_trust_gain_lower_bound=0.5,
        active_trust_gain_lower_bound=0.5,
        active_trust_gain_upper_bound=0.5,
        quotient_participation_lower_bound=0.0,
        phase3_cost=1.0,
    )
    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_ONLY,
        ordinary=None,
        audit=_audit(
            active_only,
            UnresolvedCertificate(
                record_id="unresolved-bystander",
                reason="supported_hessian_inertia_unresolved",
            ),
        ),
    )

    assert decision.kind is SRControllerDecisionKind.ACTIVE_ONLY_CORRECTION
    assert decision.record_id is None
    assert decision.certificate_record_id == "active-only"
    assert decision.actionable is True
    assert decision.consumes_singleton is False


def test_unresolved_record_blocks_modeled_minimum_classification() -> None:
    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM,
        ordinary=None,
        audit=_audit(
            UnresolvedCertificate(
                record_id="unresolved",
                reason="supported_hessian_inertia_unresolved",
            )
        ),
    )

    assert decision.kind is SRControllerDecisionKind.UNRESOLVED
    assert decision.stage_b_eligible is False
    assert decision.actionable is False


def test_psd_plus_unresolved_remains_unresolved() -> None:
    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM,
        ordinary=None,
        audit=_audit(
            PsdCertificate(
                record_id="psd",
                stationarity_margin=-1.0e-8,
                minimum_eigenvalue_lower_bound=0.25,
            ),
            UnresolvedCertificate(
                record_id="unresolved",
                reason="supported_hessian_inertia_unresolved",
            ),
        ),
    )

    assert decision.kind is SRControllerDecisionKind.UNRESOLVED
    assert decision.reason == (
        "reachable_population_contains_unresolved_certificate"
    )
    assert decision.stage_b_eligible is False
    assert decision.actionable is False


def test_complete_psd_redundancy_audit_exposes_stage_b_only_in_combined_mode() -> None:
    audit = _audit(
        PsdCertificate(
            record_id="psd",
            stationarity_margin=-1.0e-8,
            minimum_eigenvalue_lower_bound=0.25,
        ),
        QuotientRedundantCertificate(
            record_id="redundant",
            quotient_norm_upper_bound=1.0e-12,
            support_resolution_floor=1.0e-12,
        ),
        state_stationarity=_state_stationarity("psd", "redundant"),
    )

    saddle_only = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_ONLY,
        ordinary=None,
        audit=audit,
    )
    combined = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM,
        ordinary=None,
        audit=audit,
    )

    assert audit.complete is True
    assert audit.all_psd_or_redundant is True
    assert saddle_only.kind is SRControllerDecisionKind.NO_ACTION
    assert saddle_only.stage_b_eligible is False
    assert combined.kind is SRControllerDecisionKind.MODELED_MINIMUM_ELIGIBLE
    assert combined.stage_b_eligible is True
    # This Stage-A seam never performs the Stage-B state transition.
    assert combined.actionable is False
    assert combined.consumes_singleton is False


def test_redundant_population_cannot_bypass_state_stationarity() -> None:
    audit = _audit(
        QuotientRedundantCertificate(
            record_id="redundant",
            quotient_norm_upper_bound=1.0e-12,
            support_resolution_floor=1.0e-12,
        )
    )

    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM,
        ordinary=None,
        audit=audit,
    )

    assert audit.records_all_psd_or_redundant is True
    assert audit.all_psd_or_redundant is False
    assert decision.kind is SRControllerDecisionKind.UNRESOLVED
    assert decision.reason == (
        "state_stationarity_certificate_missing_or_population_stale"
    )
    assert decision.stage_b_eligible is False


def test_stale_state_stationarity_population_digest_fails_closed() -> None:
    audit = _audit(
        PsdCertificate(
            record_id="psd",
            stationarity_margin=-1.0e-8,
            minimum_eigenvalue_lower_bound=0.25,
        ),
        state_stationarity=_state_stationarity("different-record"),
    )

    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM,
        ordinary=None,
        audit=audit,
    )

    assert audit.state_stationarity_certified is False
    assert decision.kind is SRControllerDecisionKind.UNRESOLVED
    assert decision.stage_b_eligible is False


def test_saddle_acquisition_never_multiplies_by_phase3_novelty() -> None:
    near_zero_novelty = _candidate_coupled_saddle(
        record_id="near-zero-novelty",
        novelty_statistic=1.0e-12,
    )
    unit_novelty = _candidate_coupled_saddle(
        record_id="unit-novelty",
        novelty_statistic=1.0,
    )

    expected = 0.5 / (1.0 + 3.0)
    assert saddle_acquisition(near_zero_novelty) == pytest.approx(expected)
    assert saddle_acquisition(unit_novelty) == pytest.approx(expected)


def test_incomplete_reachable_population_fails_closed() -> None:
    audit = ReachablePopulationAudit(
        reachable_record_ids=("measured", "missing"),
        certificates=(_candidate_coupled_saddle(record_id="measured"),),
    )

    decision = select_sr_escape_path(
        mode=SR_ESCAPE_SADDLE_ONLY,
        ordinary=None,
        audit=audit,
    )

    assert audit.complete is False
    assert decision.kind is SRControllerDecisionKind.UNRESOLVED
    assert decision.reason == "reachable_population_audit_incomplete"
