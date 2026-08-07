from __future__ import annotations

import numpy as np

from pipelines.static_adapt.estimator_call_ledger import (
    EstimatorCallKey,
    EstimatorCallLedger,
    projective_state_fingerprint,
)


def _key(state: np.ndarray, *, observable: str = "H") -> EstimatorCallKey:
    return EstimatorCallKey(
        projective_state_fingerprint=projective_state_fingerprint(state),
        hamiltonian_fingerprint="ham:test",
        backend_fingerprint="backend:test",
        precision_contract="complex128:test",
        primitive_kind="hamiltonian_expectation",
        observable_or_formula_identity=observable,
    )


def test_same_state_unique_charge_preserves_every_execution_occurrence() -> None:
    ledger = EstimatorCallLedger()
    state = np.asarray([1.0, 0.0], dtype=complex)

    first = ledger.record_call(
        _key(state),
        component="N_H_outer",
        consumer_scope="outer_state_refresh",
    )
    second = ledger.record_call(
        _key(state),
        component="N_H_refit",
        consumer_scope="beam_local_reopt",
        branch_id="7",
    )

    assert first.charged is True
    assert second.charged is False
    assert ledger.summary()["S_unique"] == 1
    occurrence = ledger.occurrence_summary()
    assert occurrence["S_alg"] == 2
    assert occurrence["total_call_occurrences"] == 2
    assert occurrence["unique_primitive_count"] == 1
    assert occurrence["same_identity_reuse_occurrence_count"] == 1
    assert occurrence["N_H_outer"] == 1
    assert occurrence["N_H_refit"] == 1
    assert occurrence["occurrence_sequences"] == [1, 2]


def test_different_state_charges_new_unique_primitive_and_branch_views_are_disjoint() -> None:
    ledger = EstimatorCallLedger()
    state_zero = np.asarray([1.0, 0.0], dtype=complex)
    state_one = np.asarray([0.0, 1.0], dtype=complex)
    ledger.record_call(
        _key(state_zero),
        component="N_H_outer",
        consumer_scope="initial_state",
    )
    ledger.record_call(
        _key(state_zero),
        component="N_H_refit",
        consumer_scope="optimizer",
        branch_id="winner",
    )
    third = ledger.record_call(
        _key(state_one),
        component="N_H_refit",
        consumer_scope="optimizer",
        branch_id="discarded",
    )

    assert third.charged is True
    assert ledger.summary()["S_unique"] == 2
    assert ledger.occurrence_summary()["S_alg"] == 3
    shared = ledger.occurrence_summary(branch_ids=[], include_unbranched=True)
    winner = ledger.occurrence_summary(
        branch_ids=["winner"], include_unbranched=False
    )
    discarded = ledger.occurrence_summary(
        branch_ids=["discarded"], include_unbranched=False
    )
    assert shared["total_call_occurrences"] == 1
    assert winner["total_call_occurrences"] == 1
    assert discarded["total_call_occurrences"] == 1


def test_occurrence_ledger_round_trip_retains_order_and_reuse() -> None:
    ledger = EstimatorCallLedger()
    state = np.asarray([1.0, 1.0j], dtype=complex)
    for branch_id in (None, "1", "2"):
        ledger.record_call(
            _key(state),
            component="N_H_refit",
            consumer_scope="powell_objective",
            branch_id=branch_id,
        )

    payload = ledger.to_payload()
    restored = EstimatorCallLedger.from_payload(payload)

    assert restored.to_payload() == payload
    assert [row["sequence"] for row in payload["occurrences"]] == [1, 2, 3]
    assert payload["occurrence_summary"]["total_call_occurrences"] == 3
    assert payload["occurrence_summary"]["same_identity_reuse_occurrence_count"] == 2
