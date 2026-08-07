from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import hashlib
import json
import math

import pytest

from pipelines.static_adapt.sr_snake_escape_controller import (
    PsdCertificate,
    QuotientRedundantCertificate,
    ReachablePopulationAudit,
    StateStationarityCertificate,
    UnresolvedCertificate,
    reachable_population_digest,
)
from pipelines.static_adapt.sr_snake_modeled_minimum import (
    ACTION_INDEX_SCHEMA,
    BranchStateSnapshot,
    CertificateState,
    ConstrainedWorkingState,
    ControllerMode,
    DisposablePowellProbe,
    EndpointDistanceEvidence,
    EnergyInterval,
    EligibilityStateToken,
    ExposedFamilyEligibility,
    FSExclusionEvidence,
    FrozenServiceItem,
    LogEntitlement,
    ModeledMinimumCheckpoint,
    ModeledMinimumRuntimeState,
    PathActionEvidence,
    PathActionKey,
    PathOrientation,
    ResolutionKind,
    RunEnergyUnit,
    RuntimeHistoryEvent,
    ServiceTag,
    StabilizedTrustPathEvidence,
    UniformBarrierEvidence,
    activate_service_item,
    assess_exposed_family_psd,
    assess_fs_exclusion,
    assess_path_action,
    calkin_wilf_index,
    calkin_wilf_rational,
    canonical_action_index,
    canonical_action_mass,
    canonical_action_receipt_digest,
    compute_barrier_distance_utility,
    decide_disposable_powell_promotion,
    execute_exploratory_transaction,
    inverse_action_index,
    relabel_refinement_as_move,
    serve_fair_service,
)


def _audit(*certificates: object) -> ReachablePopulationAudit:
    record_ids = tuple(certificate.record_id for certificate in certificates)
    return ReachablePopulationAudit(
        reachable_record_ids=record_ids,
        certificates=tuple(certificates),
        state_stationarity=StateStationarityCertificate(
            state_fingerprint="I",
            reachable_population_digest=reachable_population_digest(record_ids),
            comparison_epoch="path-epoch",
            support_provenance_digest="support-provenance",
            trust_provenance_digest="trust-provenance",
            trust_radius=0.25,
            stationarity_margin=-1.0e-8,
        ),
    )


def _eligible():
    return assess_exposed_family_psd(
        _audit(
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
        )
    )


def _energy_unit(*, value: float = 1.0, unit_id: str = "hartree"):
    return RunEnergyUnit(run_id="run-1", unit_id=unit_id, value=value)


def _key(
    *,
    record_order: int = 1,
    orientation: PathOrientation = PathOrientation.NEGATIVE,
    radius_index: int = 1,
    path_index: int = 1,
) -> PathActionKey:
    records = _eligible().reachable_record_ids
    return PathActionKey(
        record_id=records[record_order - 1],
        record_order=record_order,
        record_count=len(records),
        orientation=orientation,
        radius_index=radius_index,
        path_index=path_index,
    )


def _fs_evidence(
    *,
    subject_state_id: str,
    distance: float,
    path_id: str,
    origin_state_id: str,
    component_id: str = "component-A",
    current_exclusion_radius: float = 0.0,
    path_distance_lower_bound: float | None = None,
    incumbent_state_id: str = "I",
    comparison_epoch: str = "path-epoch",
    simultaneous: bool = True,
    action_receipt_digest: str = "standalone-fs-receipt",
) -> FSExclusionEvidence:
    return FSExclusionEvidence(
        witness_id=f"fs:{path_id}",
        action_receipt_digest=action_receipt_digest,
        path_id=path_id,
        component_id=component_id,
        comparison_epoch=comparison_epoch,
        path_origin_state_id=origin_state_id,
        path_endpoint_state_id=subject_state_id,
        incumbent_state_id=incumbent_state_id,
        subject_state_id=subject_state_id,
        overlap_amplitude_estimate=math.cos(distance),
        overlap_error_bound=0.0,
        current_exclusion_radius=current_exclusion_radius,
        path_distance_lower_bound=(
            current_exclusion_radius
            if path_distance_lower_bound is None
            else path_distance_lower_bound
        ),
        overlap_status=CertificateState.PASSED,
        path_status=CertificateState.PASSED,
        component_status=CertificateState.PASSED,
        simultaneous=simultaneous,
    )


def _path_evidence(
    *,
    key: PathActionKey | None = None,
    barrier: float | None = 0.5,
    width: float | None = 0.01,
    distance: float | None = 0.4,
    barrier_status: CertificateState = CertificateState.PASSED,
    trust_status: CertificateState = CertificateState.PASSED,
    simultaneous: bool = True,
    eligibility_token_digest: str | None = None,
    energy_unit: RunEnergyUnit | None = None,
) -> PathActionEvidence:
    resolved_key = key or _key()
    eligibility = _eligible()
    token = eligibility.state_token
    unit = energy_unit or _energy_unit()
    action_receipt = canonical_action_receipt_digest(
        resolved_key,
        token.digest,
    )
    path_id = f"path:{resolved_key.action_index}"
    ratio = resolved_key.radius.numerator / resolved_key.radius.denominator
    scheduled_radius = token.trust_radius * ratio
    incumbent = EnergyInterval(
        state_id="I",
        energy_estimate=0.0,
        energy_error_bound=0.01,
        comparison_epoch=token.comparison_epoch,
    )
    seed = EnergyInterval(
        state_id="Z",
        energy_estimate=0.2,
        energy_error_bound=0.01,
        comparison_epoch=token.comparison_epoch,
    )
    return PathActionEvidence(
        key=resolved_key,
        phase3_cost=2.0,
        eligibility_token_digest=(
            token.digest
            if eligibility_token_digest is None
            else eligibility_token_digest
        ),
        energy_unit_digest=unit.digest,
        endpoint_seed_energy=seed,
        numerical_status=CertificateState.PASSED,
        map_status=CertificateState.PASSED,
        symmetry_status=CertificateState.PASSED,
        padding_status=CertificateState.PASSED,
        trust_path_evidence=StabilizedTrustPathEvidence(
            witness_id=f"trust:{path_id}",
            action_receipt_digest=action_receipt,
            path_id=path_id,
            comparison_epoch=token.comparison_epoch,
            origin_state_id=token.working_state_fingerprint,
            endpoint_state_id=seed.state_id,
            trust_provenance_digest=token.trust_provenance_digest,
            reference_trust_radius=token.trust_radius,
            scheduled_trust_radius=scheduled_radius,
            schedule_error_bound=0.0,
            certified_trust_arclength=scheduled_radius,
            arclength_error_bound=0.0,
            status=trust_status,
            simultaneous=simultaneous,
        ),
        barrier_evidence=UniformBarrierEvidence(
            witness_id=f"barrier:{path_id}",
            action_receipt_digest=action_receipt,
            enclosure_id=f"uniform-enclosure:{path_id}",
            path_id=path_id,
            origin_state_id=token.working_state_fingerprint,
            comparison_epoch=token.comparison_epoch,
            incumbent_energy=incumbent,
            barrier_upper_bound=barrier,
            comparison_energy_width=width,
            incumbent_referenced=True,
            status=barrier_status,
            simultaneous=simultaneous,
        ),
        endpoint_distance_evidence=EndpointDistanceEvidence(
            witness_id=f"distance:{path_id}",
            action_receipt_digest=action_receipt,
            path_id=path_id,
            endpoint_state_id=seed.state_id,
            comparison_epoch=token.comparison_epoch,
            active_manifold_digest=token.support_provenance_digest,
            trust_radius=scheduled_radius,
            distance_lower_bound=distance,
            status=CertificateState.PASSED,
            simultaneous=simultaneous,
        ),
        exclusion_evidence=_fs_evidence(
            subject_state_id=seed.state_id,
            distance=0.4,
            path_id=path_id,
            origin_state_id=token.working_state_fingerprint,
            comparison_epoch=token.comparison_epoch,
            simultaneous=simultaneous,
            action_receipt_digest=action_receipt,
        ),
    )


def _assessment(**kwargs):
    unit = kwargs.pop("energy_unit", _energy_unit())
    return assess_path_action(
        evidence=_path_evidence(energy_unit=unit, **kwargs),
        eligibility=_eligible(),
        energy_unit=unit,
    )


def _certified_action():
    assessment = _assessment()
    assert assessment.kind is ResolutionKind.CERTIFIED
    assert assessment.certified_action is not None
    return assessment.certified_action


def test_complete_state_bound_psd_redundancy_audit_is_required() -> None:
    eligible = _eligible()
    unresolved = assess_exposed_family_psd(
        _audit(
            PsdCertificate(
                record_id="psd",
                stationarity_margin=0.0,
                minimum_eigenvalue_lower_bound=0.0,
            ),
            UnresolvedCertificate(
                record_id="unresolved",
                reason="supported_hessian_inertia_unresolved",
            ),
        )
    )
    record_ids = ("psd",)
    missing_state = assess_exposed_family_psd(
        ReachablePopulationAudit(
            reachable_record_ids=record_ids,
            certificates=(
                PsdCertificate(
                    record_id="psd",
                    stationarity_margin=0.0,
                    minimum_eigenvalue_lower_bound=0.0,
                ),
            ),
            state_stationarity=None,
        )
    )

    assert eligible.eligible is True
    assert eligible.state_token.working_state_fingerprint == "I"
    assert eligible.state_token.reachable_record_ids == ("psd", "redundant")
    assert unresolved.eligible is False
    assert missing_state.eligible is False
    assert "state_stationarity" in missing_state.reason


def test_direct_eligibility_rejects_bogus_population_digest() -> None:
    token_fields = dict(
        working_state_fingerprint="I",
        reachable_record_ids=("psd",),
        reachable_population_digest="bogus",
        comparison_epoch="path-epoch",
        support_provenance_digest="support-provenance",
        trust_provenance_digest="trust-provenance",
        trust_radius=0.25,
        stationarity_margin=0.0,
    )

    with pytest.raises(ValueError, match="does not match ordered record IDs"):
        EligibilityStateToken(**token_fields)
    with pytest.raises(ValueError, match="does not match ordered record IDs"):
        ExposedFamilyEligibility(
            eligible=True,
            reason="forged",
            psd_record_ids=("psd",),
            **token_fields,
        )


def test_calkin_wilf_forward_inverse_matches_canonical_prefix() -> None:
    expected = (
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 2),
        (2, 3),
        (3, 1),
        (1, 4),
        (4, 3),
        (3, 5),
        (5, 2),
        (2, 5),
        (5, 3),
        (3, 4),
        (4, 1),
    )

    for index, fraction in enumerate(expected, start=1):
        rational = calkin_wilf_rational(index)
        assert (rational.numerator, rational.denominator) == fraction
        assert calkin_wilf_index(*fraction) == index
        assert rational.index == index
    for index in range(1, 513):
        rational = calkin_wilf_rational(index)
        assert calkin_wilf_index(
            rational.numerator,
            rational.denominator,
        ) == index


def test_canonical_action_bijection_round_trips_without_caller_index() -> None:
    records = _eligible().reachable_record_ids
    seen: set[int] = set()
    for record_order in (1, 2):
        for orientation in PathOrientation:
            for radius_index in range(1, 8):
                for path_index in range(1, 6):
                    index = canonical_action_index(
                        record_order=record_order,
                        record_count=len(records),
                        orientation=orientation,
                        radius_index=radius_index,
                        path_index=path_index,
                    )
                    assert index not in seen
                    seen.add(index)
                    inverse = inverse_action_index(index, records)
                    assert inverse.record_id == records[record_order - 1]
                    assert inverse.record_order == record_order
                    assert inverse.orientation is orientation
                    assert inverse.radius_index == radius_index
                    assert inverse.path_index == path_index
                    assert inverse.action_index == index

    with pytest.raises(TypeError):
        PathActionKey(
            action_index=1,
            record_id="psd",
            record_order=1,
            record_count=2,
            orientation=PathOrientation.NEGATIVE,
            radius_index=1,
            path_index=1,
        )


def test_symbolic_mass_and_log_virtual_finish_survive_huge_action_index() -> None:
    huge_key = _key(path_index=10**200)
    mass = canonical_action_mass(huge_key.action_index)
    assert math.isfinite(mass.log_value)
    assert mass.as_float() == 0.0
    item = FrozenServiceItem(
        tag=ServiceTag.REFINEMENT,
        action_key=huge_key,
        frozen_entitlement=LogEntitlement.from_coefficient(
            mass.scheduling_coefficient,
            symbolic_expression=mass.symbolic_expression,
        ),
        service_count=10**300,
        service_epoch="epoch",
        eligibility_token_digest=_eligible().state_token.digest,
        energy_unit_digest=_energy_unit().digest,
        activation_reason="huge-index-refinement",
    )

    assert math.isfinite(item.virtual_finish_log)
    decision = serve_fair_service((item,))
    assert decision.selected == (item,)
    assert decision.updated_population[0].service_count == 10**300 + 1


def test_exact_virtual_finish_resolves_float_log_near_tie() -> None:
    records = _eligible().reachable_record_ids
    lower_index = 10**20
    lower_count = 10**19 + 1
    higher_index = lower_index + 1
    higher_count = 10**19

    def item(action_index: int, service_count: int) -> FrozenServiceItem:
        coordinates = inverse_action_index(action_index, records)
        key = PathActionKey(
            record_id=coordinates.record_id,
            record_order=coordinates.record_order,
            record_count=coordinates.record_count,
            orientation=coordinates.orientation,
            radius_index=coordinates.radius_index,
            path_index=coordinates.path_index,
        )
        mass = canonical_action_mass(action_index)
        return FrozenServiceItem(
            tag=ServiceTag.REFINEMENT,
            action_key=key,
            frozen_entitlement=LogEntitlement.from_coefficient(
                mass.scheduling_coefficient,
                symbolic_expression=mass.symbolic_expression,
            ),
            service_count=service_count,
            service_epoch="near-tie",
            eligibility_token_digest=_eligible().state_token.digest,
            energy_unit_digest=_energy_unit().digest,
            activation_reason="near-tie",
        )

    lower = item(lower_index, lower_count)
    higher = item(higher_index, higher_count)
    assert lower.virtual_finish_log == higher.virtual_finish_log
    assert higher.scaled_virtual_finish < lower.scaled_virtual_finish
    decision = serve_fair_service((lower, higher))
    assert decision.selected[0].action_key.action_index == higher_index


@pytest.mark.parametrize("distance", [math.nextafter(0.0, 1.0), 1.0e-200])
def test_positive_tiny_distance_never_underflows_in_utility_control(
    distance: float,
) -> None:
    utility = compute_barrier_distance_utility(
        endpoint_distance_lower_bound=distance,
        barrier_upper_bound=1.0,
        comparison_energy_width=0.0,
        energy_unit=_energy_unit(),
        action_index=1,
        phase3_cost=0.0,
    )

    assert utility.exact_distance_lower_bound_squared > 0
    assert utility.live_entitlement.scheduling_coefficient > 0
    assert math.isfinite(utility.log_raw_utility)
    assert math.isfinite(utility.live_entitlement.log_value)


def test_arbitrary_size_integer_serialization_avoids_decimal_digit_limit() -> None:
    huge_key = _key(path_index=10**5000)
    restored_key = PathActionKey.from_dict(json.loads(huge_key.to_json()))
    assert restored_key == huge_key
    mass = canonical_action_mass(huge_key.action_index)
    assert "0x" in mass.to_json()

    item = FrozenServiceItem(
        tag=ServiceTag.REFINEMENT,
        action_key=huge_key,
        frozen_entitlement=LogEntitlement.from_coefficient(
            mass.scheduling_coefficient,
            symbolic_expression=mass.symbolic_expression,
        ),
        service_count=10**5000,
        service_epoch="huge",
        eligibility_token_digest=_eligible().state_token.digest,
        energy_unit_digest=_energy_unit().digest,
        activation_reason="huge",
    )
    restored_item = FrozenServiceItem.from_dict(json.loads(item.to_json()))
    assert restored_item == item
    runtime = ModeledMinimumRuntimeState(
        eligibility_token=_eligible().state_token,
        energy_unit=_energy_unit(),
        branch=_initial_branch(),
        ordinary_trust_radius=0.25,
        mode=ControllerMode.EXPLORE,
        service_epoch="huge",
        service_population=(item,),
        history=(),
        next_event_index=10**5000,
    )
    checkpoint = ModeledMinimumCheckpoint.create(runtime)
    assert ModeledMinimumCheckpoint.from_json(checkpoint.to_json()) == checkpoint


def test_low_and_high_barriers_are_suppressive() -> None:
    distance = 0.25
    unit = _energy_unit()
    low = compute_barrier_distance_utility(
        endpoint_distance_lower_bound=distance,
        barrier_upper_bound=0.5 * distance**2,
        comparison_energy_width=0.0,
        energy_unit=unit,
        action_index=1,
        phase3_cost=0.0,
    )
    high = compute_barrier_distance_utility(
        endpoint_distance_lower_bound=distance,
        barrier_upper_bound=0.5e6 * distance**2,
        comparison_energy_width=0.0,
        energy_unit=unit,
        action_index=1,
        phase3_cost=0.0,
    )

    assert low.raw_utility == pytest.approx(2.0)
    assert high.raw_utility == pytest.approx(2.0e-6)
    assert low.live_entitlement.log_value > high.live_entitlement.log_value
    assert low.live_weight > high.live_weight > 0.0


def test_zero_barrier_is_maximal_and_energy_rescaling_is_invariant() -> None:
    zero = compute_barrier_distance_utility(
        endpoint_distance_lower_bound=0.2,
        barrier_upper_bound=0.0,
        comparison_energy_width=0.0,
        energy_unit=_energy_unit(value=3.0),
        action_index=2,
        phase3_cost=1.0,
    )
    base = compute_barrier_distance_utility(
        endpoint_distance_lower_bound=0.3,
        barrier_upper_bound=0.08,
        comparison_energy_width=0.02,
        energy_unit=_energy_unit(value=0.5),
        action_index=3,
        phase3_cost=4.0,
    )
    scale = 1000.0
    rescaled = compute_barrier_distance_utility(
        endpoint_distance_lower_bound=0.3,
        barrier_upper_bound=scale * 0.08,
        comparison_energy_width=scale * 0.02,
        energy_unit=_energy_unit(value=scale * 0.5, unit_id="millihartree"),
        action_index=3,
        phase3_cost=4.0,
    )

    assert zero.raw_utility == math.inf
    assert zero.compactified_utility == 1.0
    assert rescaled.log_compactified_utility == pytest.approx(
        base.log_compactified_utility
    )
    assert rescaled.live_entitlement.log_value == pytest.approx(
        base.live_entitlement.log_value
    )


def test_certified_action_binds_full_state_path_barrier_and_energy_unit() -> None:
    action = _certified_action()
    token = _eligible().state_token

    assert action.eligibility_token == token
    assert action.energy_unit == _energy_unit()
    assert action.trust_path_certificate.certified_trust_arclength == 0.25
    assert action.trust_path_certificate.scheduled_trust_radius == 0.25
    assert action.barrier_certificate.incumbent_energy.state_id == "I"
    assert action.barrier_certificate.enclosure_id.startswith("uniform-enclosure")
    assert action.endpoint_distance_certificate.distance_lower_bound == 0.4
    assert action.exclusion_certificate.path_id == (
        action.trust_path_certificate.path_id
    )


def test_unresolved_barrier_enters_positive_log_refinement_service() -> None:
    assessment = _assessment(
        barrier=None,
        barrier_status=CertificateState.UNRESOLVED,
    )
    item = activate_service_item(assessment, service_epoch="state-1")

    assert assessment.kind is ResolutionKind.REFINEMENT
    assert item.tag is ServiceTag.REFINEMENT
    assert math.isfinite(item.frozen_entitlement.log_value)
    assert item.frozen_entitlement.as_float() > 0.0


def test_stale_token_arclength_and_barrier_provenance_fail_closed() -> None:
    stale = _assessment(eligibility_token_digest="stale")
    evidence = _path_evidence()
    trust = replace(
        evidence.trust_path_evidence,
        certified_trust_arclength=0.5,
    )
    bad_arclength = assess_path_action(
        evidence=replace(evidence, trust_path_evidence=trust),
        eligibility=_eligible(),
        energy_unit=_energy_unit(),
    )
    barrier = replace(evidence.barrier_evidence, incumbent_referenced=False)
    bad_barrier = assess_path_action(
        evidence=replace(evidence, barrier_evidence=barrier),
        eligibility=_eligible(),
        energy_unit=_energy_unit(),
    )

    assert stale.kind is ResolutionKind.INVALID
    assert stale.reason == "path_action_eligibility_token_mismatch"
    assert bad_arclength.kind is ResolutionKind.INVALID
    assert bad_arclength.reason == "trust_arclength_radius_mismatch"
    assert bad_barrier.kind is ResolutionKind.INVALID
    assert "incumbent_referenced" in bad_barrier.reason


def test_energy_unit_and_finite_record_order_bindings_fail_closed() -> None:
    evidence = _path_evidence()
    wrong_unit = assess_path_action(
        evidence=evidence,
        eligibility=_eligible(),
        energy_unit=_energy_unit(value=2.0, unit_id="other-unit"),
    )
    wrong_key = PathActionKey(
        record_id="redundant",
        record_order=1,
        record_count=2,
        orientation=PathOrientation.NEGATIVE,
        radius_index=1,
        path_index=1,
    )
    wrong_order = assess_path_action(
        evidence=_path_evidence(key=wrong_key),
        eligibility=_eligible(),
        energy_unit=_energy_unit(),
    )

    assert wrong_unit.kind is ResolutionKind.INVALID
    assert wrong_unit.reason == "path_action_energy_unit_token_mismatch"
    assert wrong_order.kind is ResolutionKind.INVALID
    assert wrong_order.reason == "path_action_record_order_mismatch"


def test_path_witness_binding_and_exact_fs_collapse_fail_closed() -> None:
    evidence = _path_evidence()
    mismatched_fs = replace(evidence.exclusion_evidence, path_id="other-path")
    mismatched = assess_path_action(
        evidence=replace(evidence, exclusion_evidence=mismatched_fs),
        eligibility=_eligible(),
        energy_unit=_energy_unit(),
    )
    collapsed = assess_fs_exclusion(
        _fs_evidence(
            subject_state_id="same",
            distance=0.0,
            path_id="collapsed",
            origin_state_id="I",
        )
    )

    assert mismatched.kind is ResolutionKind.INVALID
    assert mismatched.reason == "path_witness_id_binding_mismatch"
    assert collapsed.kind is ResolutionKind.INVALID
    assert collapsed.reason == "fs_endpoint_separation_nonpositive"


@pytest.mark.parametrize(
    "replacement_key",
    [
        _key(path_index=2),
        _key(orientation=PathOrientation.POSITIVE),
        _key(record_order=2),
    ],
)
def test_every_receipt_is_bound_to_full_canonical_action(
    replacement_key: PathActionKey,
) -> None:
    evidence = replace(_path_evidence(), key=replacement_key)
    assessment = assess_path_action(
        evidence=evidence,
        eligibility=_eligible(),
        energy_unit=_energy_unit(),
    )

    assert assessment.kind is ResolutionKind.INVALID
    assert assessment.reason == "trust_path_action_receipt_mismatch"


@pytest.mark.parametrize(
    ("receipt_name", "field_name"),
    [
        ("trust_path", "trust_path_evidence"),
        ("uniform_barrier", "barrier_evidence"),
        ("endpoint_distance", "endpoint_distance_evidence"),
        ("fs_exclusion", "exclusion_evidence"),
    ],
)
def test_each_individual_receipt_digest_fails_closed(
    receipt_name: str,
    field_name: str,
) -> None:
    evidence = _path_evidence()
    receipt = replace(
        getattr(evidence, field_name),
        action_receipt_digest="wrong-action-receipt",
    )
    assessment = assess_path_action(
        evidence=replace(evidence, **{field_name: receipt}),
        eligibility=_eligible(),
        energy_unit=_energy_unit(),
    )

    assert assessment.kind is ResolutionKind.INVALID
    assert assessment.reason == f"{receipt_name}_action_receipt_mismatch"


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (
            lambda evidence: replace(
                evidence,
                barrier_evidence=replace(
                    evidence.barrier_evidence,
                    barrier_upper_bound=float("nan"),
                ),
            ),
            "uniform_barrier_nonfinite_data",
        ),
        (
            lambda evidence: replace(
                evidence,
                endpoint_distance_evidence=replace(
                    evidence.endpoint_distance_evidence,
                    distance_lower_bound=0.0,
                ),
            ),
            "path_action_endpoint_separation_nonpositive_or_invalid",
        ),
        (
            lambda evidence: replace(
                evidence,
                trust_path_evidence=replace(
                    evidence.trust_path_evidence,
                    simultaneous=False,
                ),
            ),
            "trust_path_failed_or_not_simultaneous",
        ),
    ],
)
def test_invalid_action_data_never_enters_move_service(mutator, reason) -> None:
    assessment = assess_path_action(
        evidence=mutator(_path_evidence()),
        eligibility=_eligible(),
        energy_unit=_energy_unit(),
    )

    assert assessment.kind is ResolutionKind.INVALID
    assert assessment.reason == reason
    with pytest.raises(ValueError, match="cannot enter"):
        activate_service_item(assessment, service_epoch="state-1")


def test_move_and_refinement_share_activation_frozen_fair_queue() -> None:
    token_digest = _eligible().state_token.digest
    unit_digest = _energy_unit().digest
    move = FrozenServiceItem(
        tag=ServiceTag.MOVE,
        action_key=_key(path_index=1),
        frozen_entitlement=LogEntitlement.from_coefficient(
            Fraction(1, 2),
            symbolic_expression="1/2",
        ),
        service_count=0,
        service_epoch="epoch",
        eligibility_token_digest=token_digest,
        energy_unit_digest=unit_digest,
        activation_reason="move",
    )
    refinement = FrozenServiceItem(
        tag=ServiceTag.REFINEMENT,
        action_key=_key(path_index=2),
        frozen_entitlement=LogEntitlement.from_coefficient(
            Fraction(1, 4),
            symbolic_expression="1/4",
        ),
        service_count=0,
        service_epoch="epoch",
        eligibility_token_digest=token_digest,
        energy_unit_digest=unit_digest,
        activation_reason="ref",
    )
    population = (move, refinement)
    selected_tags: list[ServiceTag] = []
    for _ in range(12):
        decision = serve_fair_service(population)
        selected_tags.append(decision.selected[0].tag)
        population = decision.updated_population

    assert selected_tags.count(ServiceTag.MOVE) == 8
    assert selected_tags.count(ServiceTag.REFINEMENT) == 4
    assert tuple(item.service_count for item in population) == (8, 4)


def test_refinement_relabel_transfers_clock_and_token() -> None:
    unresolved = _assessment(
        barrier=None,
        barrier_status=CertificateState.UNRESOLVED,
    )
    ref = activate_service_item(unresolved, service_epoch="epoch")
    ref = replace(ref, service_count=7)
    move = relabel_refinement_as_move(ref, _certified_action())

    assert move.tag is ServiceTag.MOVE
    assert move.service_count == 7
    assert move.eligibility_token_digest == ref.eligibility_token_digest


def _initial_branch(action=None) -> BranchStateSnapshot:
    resolved_action = action or _certified_action()
    incumbent = resolved_action.barrier_certificate.incumbent_energy
    return BranchStateSnapshot(
        incumbent=incumbent,
        working=incumbent,
        exclusion_radius=0.0,
    )


def _constrained_working(action, *, energy: float = 0.1):
    proposed_chi = action.exclusion_certificate.endpoint_distance_lower_bound
    state = EnergyInterval(
        state_id="W",
        energy_estimate=energy,
        energy_error_bound=0.01,
        comparison_epoch=action.comparison_epoch,
    )
    exclusion = assess_fs_exclusion(
        _fs_evidence(
            subject_state_id=state.state_id,
            distance=0.5,
            path_id="refit:path",
            origin_state_id=action.endpoint_seed_energy.state_id,
            component_id=action.exclusion_certificate.component_id,
            current_exclusion_radius=proposed_chi,
            path_distance_lower_bound=proposed_chi,
            comparison_epoch=action.comparison_epoch,
            action_receipt_digest=canonical_action_receipt_digest(
                action.key,
                action.eligibility_token.digest,
            ),
        )
    )
    assert exclusion.certificate is not None
    return ConstrainedWorkingState(
        seed=action.endpoint_seed_energy,
        state=state,
        action_path_id=action.trust_path_certificate.path_id,
        component_id=action.exclusion_certificate.component_id,
        refit_witness_id=exclusion.certificate.witness_id,
        refit_completed=True,
        seed_retained=True,
        simultaneous=True,
        exclusion_certificate=exclusion.certificate,
    )


def test_constrained_refit_rejects_outcome_worse_than_feasible_seed() -> None:
    action = _certified_action()

    with pytest.raises(ValueError, match="worse than feasible seed"):
        _constrained_working(action, energy=0.5)


def test_failed_powell_preserves_uphill_working_state() -> None:
    action = _certified_action()
    branch = _initial_branch(action)
    working = _constrained_working(action)
    transaction = execute_exploratory_transaction(
        branch=branch,
        action=action,
        constrained_working=working,
        probe=DisposablePowellProbe(
            completed=False,
            simultaneous=False,
            comparison_epoch=None,
            one_sided_error_bound=None,
        ),
    )

    assert transaction.accepted is True
    assert transaction.promoted is False
    assert transaction.next_state.incumbent.state_id == "I"
    assert transaction.next_state.working.state_id == "W"
    assert transaction.next_state.exclusion_radius > 0.0


def test_constrained_refit_exclusion_receipt_is_action_bound() -> None:
    action = _certified_action()
    branch = _initial_branch(action)
    working = _constrained_working(action)
    wrong_exclusion = replace(
        working.exclusion_certificate,
        action_receipt_digest="wrong-action-receipt",
    )
    tampered_working = replace(
        working,
        exclusion_certificate=wrong_exclusion,
    )
    transaction = execute_exploratory_transaction(
        branch=branch,
        action=action,
        constrained_working=tampered_working,
        probe=DisposablePowellProbe(
            completed=False,
            simultaneous=False,
            comparison_epoch=None,
            one_sided_error_bound=None,
        ),
    )

    assert transaction.accepted is False
    assert transaction.reason == "working_exclusion_action_receipt_mismatch"


def test_simultaneous_powell_margin_promotes_and_resets_working() -> None:
    action = _certified_action()
    branch = _initial_branch(action)
    working = _constrained_working(action)
    better = EnergyInterval(
        state_id="Y",
        energy_estimate=-0.10,
        energy_error_bound=0.01,
        comparison_epoch=action.comparison_epoch,
    )
    probe = DisposablePowellProbe(
        completed=True,
        simultaneous=True,
        comparison_epoch=action.comparison_epoch,
        one_sided_error_bound=0.005,
        outcomes=(better,),
    )
    promotion = decide_disposable_powell_promotion(
        incumbent=branch.incumbent,
        constrained_working=working.state,
        probe=probe,
    )
    transaction = execute_exploratory_transaction(
        branch=branch,
        action=action,
        constrained_working=working,
        probe=probe,
    )

    assert promotion.promote is True
    assert promotion.promotion_margin_lower_bound == pytest.approx(0.075)
    assert transaction.promoted is True
    assert transaction.next_state.incumbent.state_id == "Y"
    assert transaction.next_state.working.state_id == "Y"
    assert transaction.next_state.exclusion_radius == 0.0


def test_runtime_checkpoint_round_trip_and_tamper_rejection() -> None:
    assessment = _assessment()
    action = assessment.certified_action
    assert action is not None
    queue_item = activate_service_item(assessment, service_epoch="service-1")
    runtime = ModeledMinimumRuntimeState(
        eligibility_token=action.eligibility_token,
        energy_unit=action.energy_unit,
        branch=_initial_branch(action),
        ordinary_trust_radius=0.25,
        mode=ControllerMode.EXPLORE,
        service_epoch="service-1",
        service_population=(queue_item,),
        history=(
            RuntimeHistoryEvent(
                event_index=0,
                event_kind="modeled_minimum_eligible",
                working_state_fingerprint="I",
                details_digest="history-details",
            ),
        ),
        next_event_index=1,
    )
    checkpoint = ModeledMinimumCheckpoint.create(runtime)
    restored = ModeledMinimumCheckpoint.from_json(checkpoint.to_json())

    assert restored == checkpoint
    assert checkpoint.checkpoint_scope == "pure_core_scheduler_only"
    assert checkpoint.runtime_resume_complete is False
    assert checkpoint.integration_ready is False
    assert runtime.runtime_resume_complete is False
    assert runtime.integration_ready is False
    assert checkpoint.action_index_schema == ACTION_INDEX_SCHEMA
    assert runtime.action_index_schema == ACTION_INDEX_SCHEMA
    tampered = json.loads(checkpoint.to_json())
    tampered["runtime"]["ordinary_trust_radius"] = 0.5
    with pytest.raises(ValueError, match="digest mismatch"):
        ModeledMinimumCheckpoint.from_json(
            json.dumps(tampered, sort_keys=True, separators=(",", ":"))
        )

    derived_tamper = json.loads(checkpoint.to_json())
    derived_tamper["runtime"]["service_population"][0]["action_key"][
        "action_index"
    ] = "0x3e7"
    runtime_payload = derived_tamper["runtime"]
    canonical = json.dumps(
        runtime_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    derived_tamper["content_digest"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="runtime payload is invalid"):
        ModeledMinimumCheckpoint.from_json(
            json.dumps(derived_tamper, sort_keys=True, separators=(",", ":"))
        )

    schema_tamper = json.loads(checkpoint.to_json())
    schema_tamper["action_index_schema"] = "tampered"
    with pytest.raises(ValueError, match="action-index schema"):
        ModeledMinimumCheckpoint.from_json(
            json.dumps(schema_tamper, sort_keys=True, separators=(",", ":"))
        )

    runtime_schema_tamper = json.loads(checkpoint.to_json())
    runtime_schema_tamper["runtime"]["action_index_schema"] = "tampered"
    runtime_payload = runtime_schema_tamper["runtime"]
    canonical = json.dumps(
        runtime_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    runtime_schema_tamper["content_digest"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="runtime payload is invalid"):
        ModeledMinimumCheckpoint.from_json(
            json.dumps(
                runtime_schema_tamper,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    key_schema_tamper = json.loads(checkpoint.to_json())
    key_schema_tamper["runtime"]["service_population"][0]["action_key"][
        "action_index_schema"
    ] = "tampered"
    runtime_payload = key_schema_tamper["runtime"]
    canonical = json.dumps(
        runtime_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    key_schema_tamper["content_digest"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="runtime payload is invalid"):
        ModeledMinimumCheckpoint.from_json(
            json.dumps(
                key_schema_tamper,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def test_public_outputs_are_frozen_and_deterministic() -> None:
    action = _certified_action()

    assert action.to_json() == action.to_json()
    assert action.key.to_json() == action.key.to_json()
    with pytest.raises(FrozenInstanceError):
        action.endpoint_seed_energy = replace(
            action.endpoint_seed_energy,
            energy_estimate=10.0,
        )
