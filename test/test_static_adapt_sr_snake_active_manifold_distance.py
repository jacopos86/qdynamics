from __future__ import annotations

from dataclasses import replace
import math

import pytest

from pipelines.static_adapt.sr_snake_active_manifold_distance import (
    COMPUTATIONAL_BASIS_REFERENCE_KIND,
    ActiveManifoldDistanceBindings,
    ActiveManifoldDistanceRequest,
    ActiveManifoldDistanceResult,
    ActiveManifoldDistanceStatus,
    PrimitivePauliSupport,
    canonical_active_execution_mode_digest,
    canonical_active_layout_digest,
    canonical_active_radius_digest,
    canonical_active_support_digest,
    certify_active_manifold_distance,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    reachable_population_digest,
)
from pipelines.static_adapt.sr_snake_modeled_minimum import (
    EligibilityStateToken,
    PathActionKey,
    PathOrientation,
    canonical_action_receipt_digest,
)


def _token(*, qubit_count: int = 2) -> EligibilityStateToken:
    records = ("record-a", "record-b")
    return EligibilityStateToken(
        working_state_fingerprint=f"working:{qubit_count}",
        reachable_record_ids=records,
        reachable_population_digest=reachable_population_digest(records),
        comparison_epoch="comparison:7",
        support_provenance_digest="support-provenance:7",
        trust_provenance_digest="trust-provenance:7",
        trust_radius=0.25,
        stationarity_margin=-1.0e-9,
    )


def _key() -> PathActionKey:
    return PathActionKey(
        record_id="record-a",
        record_order=1,
        record_count=2,
        orientation=PathOrientation.POSITIVE,
        radius_index=3,
        path_index=5,
    )


def _bindings(
    support: tuple[PrimitivePauliSupport, ...],
    *,
    qubit_count: int = 2,
) -> ActiveManifoldDistanceBindings:
    token = _token(qubit_count=qubit_count)
    key = _key()
    return ActiveManifoldDistanceBindings(
        eligibility_token=token,
        working_state_fingerprint=token.working_state_fingerprint,
        reference_state_fingerprint=f"reference:{qubit_count}:00",
        endpoint_state_fingerprint=f"endpoint:{qubit_count}:phase-invariant",
        comparison_epoch=token.comparison_epoch,
        branch_epoch="branch:3",
        active_support_digest=canonical_active_support_digest(
            support,
            qubit_count=qubit_count,
        ),
        layout_digest=canonical_active_layout_digest(qubit_count=qubit_count),
        execution_mode_digest=canonical_active_execution_mode_digest(),
        support_provenance_digest=token.support_provenance_digest,
        trust_provenance_digest=token.trust_provenance_digest,
        radius_digest=canonical_active_radius_digest(
            action_key=key,
            eligibility_token=token,
        ),
        action_receipt_digest=canonical_action_receipt_digest(key, token.digest),
        path_digest="path:canonical:11",
        sector_digest="sector:number-parity:2",
        padding_digest="padding:none",
        action_key=key,
    )


def _request(
    support: tuple[PrimitivePauliSupport, ...],
    endpoint: tuple[complex, ...],
    *,
    reference_bitstring: str = "00",
    reference_kind: str = COMPUTATIONAL_BASIS_REFERENCE_KIND,
    endpoint_error: float = 0.0,
    max_gf2_rank: int = 4,
    max_orbit_size: int = 16,
    bindings: ActiveManifoldDistanceBindings | None = None,
) -> ActiveManifoldDistanceRequest:
    return ActiveManifoldDistanceRequest(
        bindings=bindings or _bindings(support),
        qubit_count=2,
        reference_kind=reference_kind,
        reference_bitstring=reference_bitstring,
        primitive_support=support,
        endpoint_amplitudes=endpoint,
        endpoint_l2_error_bound=endpoint_error,
        max_gf2_rank=max_gf2_rank,
        max_orbit_size=max_orbit_size,
    )


def test_z_only_support_certifies_orthogonal_basis_endpoint() -> None:
    support = (PrimitivePauliSupport("ze"), PrimitivePauliSupport("ez"))
    result = certify_active_manifold_distance(
        _request(support, (0.0, 0.0, 0.0, 1.0))
    )

    assert result.status is ActiveManifoldDistanceStatus.CERTIFIED_POSITIVE
    assert result.gf2_basis_masks == ()
    assert result.affine_orbit_indices == (0,)
    assert result.projection_norm_squared_exact == 0
    assert result.distance_lower_bound == pytest.approx(math.pi / 2.0)
    assert result.distance_lower_bound_squared == pytest.approx((math.pi / 2.0) ** 2)


def test_x_y_flip_masks_define_the_same_gf2_closure_lane() -> None:
    support = (
        PrimitivePauliSupport("xe"),
        PrimitivePauliSupport("ye"),
        PrimitivePauliSupport("ze"),
    )
    result = certify_active_manifold_distance(
        _request(support, (0.0, 1.0, 0.0, 0.0))
    )

    assert result.status is ActiveManifoldDistanceStatus.CERTIFIED_POSITIVE
    assert result.gf2_basis_masks == (2,)
    assert result.affine_orbit_indices == (0, 2)
    assert result.distance_lower_bound == pytest.approx(math.pi / 2.0)


def test_full_flip_closure_is_certified_zero_not_positive() -> None:
    support = (PrimitivePauliSupport("xe"), PrimitivePauliSupport("ex"))
    result = certify_active_manifold_distance(
        _request(support, (0.0, 0.0, 0.0, 1.0))
    )

    assert result.status is ActiveManifoldDistanceStatus.CERTIFIED_ZERO
    assert result.reason == "active_flip_span_is_full_hilbert_space"
    assert result.gf2_rank == 2
    assert result.affine_orbit_indices == (0, 1, 2, 3)
    assert result.distance_lower_bound == 0.0
    assert not result.positive


def test_exact_projection_uses_norm_not_probability_inside_acos() -> None:
    support = (PrimitivePauliSupport("ex"),)
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    endpoint = (inverse_sqrt_two, 0.0, inverse_sqrt_two, 0.0)
    result = certify_active_manifold_distance(
        _request(support, endpoint, endpoint_error=5.0e-16)
    )

    assert result.status is ActiveManifoldDistanceStatus.CERTIFIED_POSITIVE
    assert float(result.projection_norm_squared_exact) == pytest.approx(0.5)
    assert result.projection_norm_upper_bound == pytest.approx(inverse_sqrt_two)
    assert result.distance_lower_bound == pytest.approx(math.pi / 4.0)


def test_global_phase_does_not_change_distance_or_receipt() -> None:
    support = (PrimitivePauliSupport("ze"),)
    real = certify_active_manifold_distance(
        _request(support, (0.0, 0.0, 1.0, 0.0))
    )
    phased = certify_active_manifold_distance(
        _request(support, (0.0j, 0.0j, 1.0j, 0.0j))
    )

    assert phased.distance_lower_bound == real.distance_lower_bound
    assert phased.projection_norm_squared_exact == real.projection_norm_squared_exact
    assert phased.receipt_digest == real.receipt_digest


def test_support_is_invariant_to_order_duplicates_and_nonzero_rescaling() -> None:
    first = (
        PrimitivePauliSupport("xe", 1.0),
        PrimitivePauliSupport("ex", -3.0j),
    )
    second = (
        PrimitivePauliSupport("ex", 1.0e-30),
        PrimitivePauliSupport("xe", -12.0),
        PrimitivePauliSupport("xe", 7.0j),
    )
    endpoint = (1.0, 0.0, 0.0, 0.0)
    left = certify_active_manifold_distance(
        _request(first, endpoint, bindings=_bindings(first))
    )
    right = certify_active_manifold_distance(
        _request(second, endpoint, bindings=_bindings(second))
    )

    assert canonical_active_support_digest(first, qubit_count=2) == (
        canonical_active_support_digest(second, qubit_count=2)
    )
    assert left.canonical_pauli_words == right.canonical_pauli_words
    assert left.gf2_basis_masks == right.gf2_basis_masks
    assert left.affine_orbit_indices == right.affine_orbit_indices
    assert left.receipt_digest == right.receipt_digest


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        (
            "working_state_fingerprint",
            "stale-working-state",
            "working_state_fingerprint_eligibility_mismatch",
        ),
        (
            "comparison_epoch",
            "stale-comparison",
            "comparison_epoch_eligibility_mismatch",
        ),
        (
            "action_receipt_digest",
            "tampered-action-receipt",
            "canonical_action_receipt_mismatch",
        ),
        ("path_digest", "changed-but-bound-path", None),
    ],
)
def test_stale_bindings_fail_closed_and_opaque_path_changes_receipt(
    field: str,
    replacement: str,
    reason: str | None,
) -> None:
    support = (PrimitivePauliSupport("ze"),)
    base = _bindings(support)
    changed = replace(base, **{field: replacement})
    result = certify_active_manifold_distance(
        _request(support, (0.0, 0.0, 1.0, 0.0), bindings=changed)
    )
    if reason is not None:
        assert result.status is ActiveManifoldDistanceStatus.UNRESOLVED
        assert result.reason == reason
    else:
        original = certify_active_manifold_distance(
            _request(support, (0.0, 0.0, 1.0, 0.0), bindings=base)
        )
        assert result.status is original.status
        assert result.receipt_digest != original.receipt_digest


def test_serialized_receipt_rejects_tampering() -> None:
    support = (PrimitivePauliSupport("ze"),)
    result = certify_active_manifold_distance(
        _request(support, (0.0, 0.0, 1.0, 0.0))
    )
    restored = ActiveManifoldDistanceResult.from_dict(result.to_dict())
    assert restored == result

    tampered = result.to_dict()
    tampered["distance_lower_bound"] = 0.01
    with pytest.raises(ValueError, match="stale or tampered"):
        ActiveManifoldDistanceResult.from_dict(tampered)

    nested_tamper = result.to_dict()
    nested_tamper["bindings"]["endpoint_state_fingerprint"] = "other-endpoint"
    with pytest.raises(ValueError, match="stale or tampered"):
        ActiveManifoldDistanceResult.from_dict(nested_tamper)


def test_general_reference_is_explicitly_unresolved() -> None:
    support = (PrimitivePauliSupport("ze"),)
    result = certify_active_manifold_distance(
        _request(
            support,
            (0.0, 0.0, 1.0, 0.0),
            reference_kind="general_statevector",
        )
    )
    assert result.status is ActiveManifoldDistanceStatus.UNRESOLVED
    assert result.reason == "reference_kind_not_supported"
    assert result.distance_lower_bound is None


def test_rank_and_orbit_resource_caps_are_explicitly_unresolved() -> None:
    support = (PrimitivePauliSupport("xe"), PrimitivePauliSupport("ex"))
    endpoint = (1.0, 0.0, 0.0, 0.0)
    rank = certify_active_manifold_distance(
        _request(support, endpoint, max_gf2_rank=1)
    )
    orbit = certify_active_manifold_distance(
        _request(support, endpoint, max_gf2_rank=2, max_orbit_size=2)
    )

    assert rank.status is ActiveManifoldDistanceStatus.UNRESOLVED
    assert rank.reason == "gf2_rank_resource_cap_exceeded"
    assert orbit.status is ActiveManifoldDistanceStatus.UNRESOLVED
    assert orbit.reason == "affine_orbit_resource_cap_exceeded"
    assert rank.distance_lower_bound is None
    assert orbit.distance_lower_bound is None


def test_endpoint_error_that_saturates_projection_is_unresolved() -> None:
    support = (PrimitivePauliSupport("ex"),)
    result = certify_active_manifold_distance(
        _request(
            support,
            (1.0, 0.0, 0.0, 0.0),
            endpoint_error=1.0e-6,
        )
    )

    assert result.status is ActiveManifoldDistanceStatus.UNRESOLVED
    assert result.reason == "endpoint_error_precludes_positive_distance"
    assert result.projection_norm_upper_bound == 1.0
    assert result.distance_lower_bound is None


def test_invalid_norm_and_unsupported_primitive_never_certify_positive() -> None:
    support = (PrimitivePauliSupport("ze"),)
    invalid_norm = certify_active_manifold_distance(
        _request(support, (0.0, 0.0, 0.5, 0.0))
    )
    invalid_primitive = (PrimitivePauliSupport("Ze"),)
    unsupported = certify_active_manifold_distance(
        _request(
            invalid_primitive,
            (0.0, 0.0, 1.0, 0.0),
            bindings=replace(
                _bindings(support),
                active_support_digest="unsupported-support",
            ),
        )
    )

    assert invalid_norm.status is ActiveManifoldDistanceStatus.UNRESOLVED
    assert invalid_norm.reason == "endpoint_normalization_outside_declared_error"
    assert unsupported.status is ActiveManifoldDistanceStatus.UNRESOLVED
    assert unsupported.reason.startswith("primitive_pauli_support_unresolved:")

