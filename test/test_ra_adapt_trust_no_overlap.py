from __future__ import annotations

import numpy as np
import pytest

import pipelines.static_adapt.route_a_trust_region as trust_region_module
import pipelines.static_adapt.ra_adapt.support as support_module
from pipelines.static_adapt.estimator_call_ledger import (
    EstimatorCallKey,
    EstimatorCallLedger,
)
from pipelines.static_adapt.ra_adapt.trust import (
    build_source_gram_no_overlap_trust_transaction,
)


def _selector_summary() -> dict[str, object]:
    support = support_module.factor_retained_support(
        np.diag([1.0, 4.0]),
        rank_relative_tolerance=1.0e-12,
        metric_regularization=0.0,
        source_provenance_id="phase3_projected_source_gram",
    )
    return {
        "joint_linear_solve_policy_effective": (
            "supported_metric_projected_generalized_trust_v1"
        ),
        "G_AA_raw": [[1.0]],
        "G_AB_raw": [[0.0]],
        "G_BB_raw": [[4.0]],
        "raw_metric_eigenvalues": [1.0, 4.0],
        "metric_retained_mask": [True, True],
        "metric_support_threshold": 1.0e-12,
        "supported_metric_projection_provenance_id": (
            support.receipt.factorization_provenance_id
        ),
        "retained_support_receipt": support.receipt.as_dict(),
        "trust_radius_sq": 0.25**2,
        "trust_radius_binding_tolerance_sq": 1.0e-12,
        "joint_step": [0.25, 0.0],
        "joint_fubini_study_displacement_sq": 0.25**2,
    }


@pytest.mark.parametrize("adapter_id", ["macro", "single_pauli_word"])
def test_common_trust_transaction_never_acquires_endpoint_overlap(
    monkeypatch: pytest.MonkeyPatch,
    adapter_id: str,
) -> None:
    ledger = EstimatorCallLedger()
    overlap_key = EstimatorCallKey(
        projective_state_fingerprint="test:state",
        hamiltonian_fingerprint="test:hamiltonian",
        backend_fingerprint="test:exact_backend",
        precision_contract="test:float64_exact",
        primitive_kind="state_overlap",
        observable_or_formula_identity=(
            "projective_endpoint_overlap_magnitude_v1"
        ),
    )

    def _record_forbidden_overlap(
        *_args: object,
        **_kwargs: object,
    ) -> float:
        ledger.record_call(
            overlap_key,
            component="N_metric",
            consumer_scope=f"ra_adapt_{adapter_id}_trust_endpoint",
        )
        return 0.0

    monkeypatch.setattr(
        trust_region_module,
        "exact_fubini_study_distance",
        _record_forbidden_overlap,
    )

    receipt = build_source_gram_no_overlap_trust_transaction(
        _selector_summary(),
        realized_joint_step=[0.5, 0.0],
        radius_before=0.25,
        adapter_id=adapter_id,
    )
    payload = receipt.as_dict()

    assert payload["adapter_id"] == adapter_id
    assert payload["predicted_source_metric_displacement"] == pytest.approx(
        0.25
    )
    assert payload["realized_source_metric_displacement"] == pytest.approx(0.5)
    assert payload["endpoint_overlap_required"] is False
    assert payload["endpoint_overlap_query_charge"] == 0
    assert payload["incremental_quantum_query_charge"] == 0
    assert payload["supported_metric_whitening_active"] is False
    assert payload["supported_metric_inverse_sqrt_constructed"] is False
    assert payload["transaction_complete"] is True
    ledger_payload = ledger.to_payload()
    assert ledger_payload["entries"] == []
    assert ledger_payload["occurrences"] == []
    assert ledger_payload["occurrence_summary"][
        "component_occurrence_counts"
    ]["N_metric"] == 0


def test_common_trust_transaction_validates_the_selector_support_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    original = getattr(
        support_module,
        "validate_retained_support_receipt",
        None,
    )

    def _recording_validation(payload: object) -> object:
        calls.append(payload)
        if original is None:
            return payload
        return original(payload)

    monkeypatch.setattr(
        support_module,
        "validate_retained_support_receipt",
        _recording_validation,
        raising=False,
    )
    receipt = build_source_gram_no_overlap_trust_transaction(
        _selector_summary(),
        realized_joint_step=[0.5, 0.0],
        radius_before=0.25,
        adapter_id="macro",
    )

    assert receipt.supported_rank == 2
    # The representation-neutral owner validates at its facade boundary and
    # the delegated production transaction validates the same receipt again.
    assert len(calls) >= 1
    assert all(call == calls[0] for call in calls)


def test_common_trust_wrapper_fails_closed_on_nonzero_endpoint_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = trust_region_module._sr_projected_source_metric_trust_transaction

    def _charged_transaction(*args: object, **kwargs: object) -> dict[str, object]:
        payload = original(*args, **kwargs)
        assert payload is not None
        payload["endpoint_overlap_query_charge"] = 1
        return payload

    monkeypatch.setattr(
        trust_region_module,
        "_sr_projected_source_metric_trust_transaction",
        _charged_transaction,
    )

    with pytest.raises(
        ValueError,
        match="endpoint-overlap query charge must be zero",
    ):
        build_source_gram_no_overlap_trust_transaction(
            _selector_summary(),
            realized_joint_step=[0.5, 0.0],
            radius_before=0.25,
            adapter_id="macro",
        )
