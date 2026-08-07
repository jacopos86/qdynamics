from __future__ import annotations

import pytest

from pipelines.static_adapt.deferred_gram_fallback import (
    DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1,
    deferred_gram_fallback_enabled,
    selected_admission_deferred_gram_fallback_receipt,
    summarize_deferred_gram_fallback,
)


def _route_contract(*, enabled: bool = True) -> dict:
    policy = "fallback_only_v1" if enabled else "off"
    return {
        "execution_settings": {
            "phase2_gram_novelty_policy": policy,
            "phase3_gram_novelty_policy": policy,
        }
    }


def test_route_contract_requires_both_deferred_policies() -> None:
    assert deferred_gram_fallback_enabled(_route_contract())
    assert not deferred_gram_fallback_enabled(
        {
            "execution_settings": {
                "phase2_gram_novelty_policy": "fallback_only_v1",
                "phase3_gram_novelty_policy": "off",
            }
        }
    )


def test_selected_receipt_uses_new_identity_and_load_bearing_fields() -> None:
    receipt = selected_admission_deferred_gram_fallback_receipt(
        [
            {
                "route_a_geometry_expansion_mode": "expand_retained_domain",
                "route_a_geometry_expansion_reason": (
                    "all_whitened_energy_models_infeasible"
                ),
                "route_a_geometry_expansion_query_charge": 0,
            }
        ],
        enabled=True,
        controller_round=3,
    )
    assert receipt == {
        "schema": DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1,
        "scope": "accepted_controller_round",
        "enabled": True,
        "fired": True,
        "rounds": [3],
        "charge": 0,
        "mode": "expand_retained_domain",
        "reason": "all_whitened_energy_models_infeasible",
    }


def test_run_summary_closes_only_new_receipts() -> None:
    unfired = selected_admission_deferred_gram_fallback_receipt(
        [{}],
        enabled=True,
        controller_round=1,
    )
    fired = selected_admission_deferred_gram_fallback_receipt(
        [
            {
                "route_a_geometry_expansion_mode": "expand_retained_domain",
                "route_a_geometry_expansion_reason": "all_models_infeasible",
                "route_a_geometry_expansion_query_charge": 2,
            }
        ],
        enabled=True,
        controller_round=2,
    )
    summary = summarize_deferred_gram_fallback(
        [
            {
                "depth": 1,
                "selected_op": "A",
                DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1: unfired,
            },
            {
                "depth": 2,
                "selected_op": "B",
                DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1: fired,
            },
        ],
        enabled=True,
    )
    assert summary["schema"] == (
        DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1
    )
    assert summary["enabled"] is True
    assert summary["fired"] is True
    assert summary["rounds"] == [2]
    assert summary["charge"] == 2
    assert summary["selected_operators"] == ["B"]


def test_historical_prefix_is_not_reinterpreted() -> None:
    receipt = selected_admission_deferred_gram_fallback_receipt(
        [{}],
        enabled=True,
        controller_round=2,
    )
    summary = summarize_deferred_gram_fallback(
        [
            {
                "depth": 1,
                "all_energy_models_infeasible_novelty_fallback_fired": True,
            },
            {
                "depth": 2,
                DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1: receipt,
            },
        ],
        enabled=True,
        allow_missing_prefix_rounds=1,
    )
    assert summary["fired"] is False
    assert summary["rounds"] == []
    assert summary["charge"] == 0
    assert summary["historical_prefix_rounds_without_new_receipt"] == 1


def test_fired_receipt_fails_closed_when_disabled() -> None:
    with pytest.raises(RuntimeError, match="disabled"):
        selected_admission_deferred_gram_fallback_receipt(
            [
                {
                    "route_a_geometry_expansion_mode": "expand",
                    "route_a_geometry_expansion_reason": "all_models_infeasible",
                }
            ],
            enabled=False,
            controller_round=1,
        )
