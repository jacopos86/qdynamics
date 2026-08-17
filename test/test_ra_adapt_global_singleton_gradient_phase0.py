from __future__ import annotations

import pytest

from pipelines.static_adapt.ra_adapt.adapters import (
    GLOBAL_SINGLETON_GRADIENT_PHASE0_ADAPTER_ID,
    GlobalSingletonGradientPhase0CandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    RAAdaptRequest,
    ra_adapt_request_from_mapping,
)
from pipelines.static_adapt.ra_adapt.phase0 import (
    GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY,
    GLOBAL_SINGLETON_GRADIENT_PHASE0_CONSUMER_SCOPE,
    build_global_singleton_gradient_phase0_receipt,
    rank_global_singletons_by_absolute_gradient,
)


def _gradient_occurrences(count: int) -> list[dict[str, object]]:
    return [
        {
            "sequence": index,
            "primitive_id": f"gradient:{index}",
            "component": "N_grad",
            "consumer_scope": (
                GLOBAL_SINGLETON_GRADIENT_PHASE0_CONSUMER_SCOPE
            ),
        }
        for index in range(count)
    ]


def test_global_singleton_gradient_phase0_adapter_round_trips_exactly() -> None:
    adapter = GlobalSingletonGradientPhase0CandidateAdapter()
    request = RAAdaptRequest(adapter=adapter)

    restored = ra_adapt_request_from_mapping(request.to_dict())

    assert isinstance(
        restored.adapter,
        GlobalSingletonGradientPhase0CandidateAdapter,
    )
    assert restored.to_dict() == request.to_dict()
    assert adapter.adapter_id == GLOBAL_SINGLETON_GRADIENT_PHASE0_ADAPTER_ID
    assert adapter.phase0_shortlist_policy_id == (
        GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY
    )


def test_global_singleton_gradient_phase0_is_absolute_gradient_only() -> None:
    shortlist = rank_global_singletons_by_absolute_gradient(
        available_indices={4, 0, 3, 1},
        gradients=[0.5, -1.2, 99.0, 1.2, -0.75],
        shortlist_size=3,
    )

    assert shortlist.input_indices == (0, 1, 3, 4)
    assert shortlist.ranked_indices == (1, 3, 4, 0)
    assert shortlist.retained_indices == (1, 3, 4)
    assert shortlist.shortlist_size == 3


def test_global_singleton_gradient_phase0_receipt_closes_n_grad_only() -> None:
    shortlist = rank_global_singletons_by_absolute_gradient(
        available_indices={0, 1, 2},
        gradients=[-0.25, 0.75, -0.5],
        shortlist_size=24,
    )

    receipt = build_global_singleton_gradient_phase0_receipt(
        shortlist=shortlist,
        pool_labels=["g0", "g1", "g2"],
        requested_shortlist_size=24,
        estimator_occurrences=_gradient_occurrences(3),
    )

    assert receipt["retained_pool_indices"] == [1, 2, 0]
    assert receipt["requested_shortlist_size"] == 24
    assert receipt["effective_shortlist_size"] == 3
    assert receipt["metric_policy"] == "off"
    assert receipt["compile_cost_policy"] == "off"
    assert receipt["estimator_accounting"]["components"] == {
        "N_H_outer": 0,
        "N_H_refit": 0,
        "N_grad": 3,
        "N_metric": 0,
    }
    assert receipt["estimator_accounting"]["S_alg"] == 3


def test_global_singleton_gradient_phase0_rejects_metric_work() -> None:
    shortlist = rank_global_singletons_by_absolute_gradient(
        available_indices={0, 1},
        gradients=[0.25, -0.5],
        shortlist_size=1,
    )
    occurrences = _gradient_occurrences(2)
    occurrences[1]["component"] = "N_metric"

    with pytest.raises(
        RuntimeError,
        match="outside its standard-gradient surface",
    ):
        build_global_singleton_gradient_phase0_receipt(
            shortlist=shortlist,
            pool_labels=["g0", "g1"],
            requested_shortlist_size=1,
            estimator_occurrences=occurrences,
        )


def test_global_singleton_gradient_phase0_rejects_cap_drift() -> None:
    shortlist = rank_global_singletons_by_absolute_gradient(
        available_indices={0, 1, 2},
        gradients=[0.25, -0.5, 0.75],
        shortlist_size=2,
    )

    with pytest.raises(ValueError, match="requested cap"):
        build_global_singleton_gradient_phase0_receipt(
            shortlist=shortlist,
            pool_labels=["g0", "g1", "g2"],
            requested_shortlist_size=1,
            estimator_occurrences=_gradient_occurrences(3),
        )


def test_global_singleton_gradient_phase0_rejects_duplicate_events() -> None:
    shortlist = rank_global_singletons_by_absolute_gradient(
        available_indices={0, 1},
        gradients=[0.25, -0.5],
        shortlist_size=1,
    )
    occurrences = _gradient_occurrences(2)
    occurrences[1]["sequence"] = occurrences[0]["sequence"]
    occurrences[1]["primitive_id"] = occurrences[0]["primitive_id"]

    with pytest.raises(RuntimeError, match="identities must be unique"):
        build_global_singleton_gradient_phase0_receipt(
            shortlist=shortlist,
            pool_labels=["g0", "g1"],
            requested_shortlist_size=1,
            estimator_occurrences=occurrences,
        )
