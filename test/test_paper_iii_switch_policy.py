"""Focused tests for the declared online switch policy."""

from __future__ import annotations

import pytest

from pipelines.qse_spectra.switch_policy import (
    CANONICAL_METHOD_POLICY,
    TRIGGER_POOL_EXHAUSTED,
    TRIGGER_WINDOW_RESIDUAL_STALL,
    MethodPolicy,
    SwitchPolicyError,
    assert_online_only_receipt,
    resolve_switch_point,
    window_residual_stall_policy,
)
from pipelines.qse_spectra.static_campaign import (
    DEFAULT_TARGET_ROOTS,
    METHOD_RECORD_FIRST,
    StaticCampaignError,
    StaticQSECellKey,
    cell_key_from_payload,
    hh_physics_key,
)


def _round(n: int, residual: float, *, selected_before: int, selected: bool = True) -> dict:
    return {
        "round": n,
        "selected_count_before": selected_before,
        "retained_frame_size": selected_before,
        "max_target_residual_norm": residual,
        "window_top": 1.0,
        "window_margin": 0.0,
        "window_pressure": True,
        "window_closed": False,
        "novel_candidate_count": 3,
        "max_candidate_metric_novelty": 0.4,
        "max_candidate_residual_capture": 0.2,
        "selected_original_index": (selected_before if selected else None),
        "selected_score": 0.5,
        "selected_metric_novelty": 0.3,
        "selected_residual_capture": 0.1,
    }


def _receipt(residuals: list[float], *, stop_reason: str = "budget_reached") -> dict:
    return {
        "stop_reason": stop_reason,
        "final_max_target_residual_norm": residuals[-1] if residuals else None,
        "residual_converged_pending": False,
        "residual_stop_threshold": 1.0e-3,
        "round_trace": tuple(
            _round(i + 1, r, selected_before=i) for i, r in enumerate(residuals)
        ),
    }


# --- policy typing ----------------------------------------------------------


def test_canonical_policy_is_exhaustion_only_with_no_parameters() -> None:
    assert CANONICAL_METHOD_POLICY.switch_trigger == TRIGGER_POOL_EXHAUSTED
    assert CANONICAL_METHOD_POLICY.parameters == ()
    assert CANONICAL_METHOD_POLICY.is_canonical is True
    assert MethodPolicy().digest() == CANONICAL_METHOD_POLICY.digest()


def test_stall_policy_requires_exactly_its_parameters() -> None:
    policy = window_residual_stall_policy(stall_rounds=2, improvement_ratio=0.5)
    assert policy.switch_trigger == TRIGGER_WINDOW_RESIDUAL_STALL
    assert policy.is_canonical is False
    assert policy.digest() != CANONICAL_METHOD_POLICY.digest()

    with pytest.raises(SwitchPolicyError, match="requires exactly the parameters"):
        MethodPolicy(switch_trigger=TRIGGER_WINDOW_RESIDUAL_STALL)
    with pytest.raises(SwitchPolicyError, match="requires exactly the parameters"):
        MethodPolicy(
            switch_trigger=TRIGGER_POOL_EXHAUSTED, parameters=(("stall_rounds", 2.0),)
        )
    with pytest.raises(SwitchPolicyError, match="improvement_ratio"):
        window_residual_stall_policy(improvement_ratio=1.5)
    with pytest.raises(SwitchPolicyError, match="unknown switch trigger"):
        MethodPolicy(switch_trigger="vibes")


def test_policy_round_trips_through_its_payload() -> None:
    policy = window_residual_stall_policy(stall_rounds=3, improvement_ratio=0.25)
    assert MethodPolicy.from_mapping(policy.payload()) == policy


# --- identity integration ---------------------------------------------------


def _cell(method: str, policy: MethodPolicy) -> StaticQSECellKey:
    return StaticQSECellKey(
        physics=hh_physics_key("weak_weak", 0.25, 0.353553390593, 1),
        ground_state_artifact_id="artifact",
        method_id=method,
        target_roots=DEFAULT_TARGET_ROOTS,
        dimension_cap=64,
        pool_policy_digest="pool" * 6,
        spectral_request_digest="spec" * 6,
        resource_policy_id="graph_span_two_qubit_only_v1_plus_pencil_estimator_v1",
        method_policy=policy,
    )


def test_switch_policy_is_part_of_the_cell_identity() -> None:
    canonical = _cell(METHOD_RECORD_FIRST, CANONICAL_METHOD_POLICY)
    ablation = _cell("early_switch", window_residual_stall_policy())
    assert canonical.digest() != ablation.digest()
    assert canonical.payload()["method_policy_digest"] == CANONICAL_METHOD_POLICY.digest()
    assert cell_key_from_payload(ablation.payload()).digest() == ablation.digest()
    assert cell_key_from_payload(ablation.payload()).method_policy == ablation.method_policy


def test_public_methods_may_not_carry_a_non_canonical_policy() -> None:
    with pytest.raises(StaticCampaignError, match="must run the canonical"):
        _cell(METHOD_RECORD_FIRST, window_residual_stall_policy())


def test_two_stall_parameterizations_are_distinct_cells() -> None:
    a = _cell("early_switch", window_residual_stall_policy(stall_rounds=2))
    b = _cell("early_switch", window_residual_stall_policy(stall_rounds=3))
    assert a.digest() != b.digest()


# --- online-only boundary ---------------------------------------------------


def test_non_online_receipt_keys_are_refused() -> None:
    receipt = _receipt([1.0, 0.4])
    receipt["reference_excitations"] = [0.1, 0.2]
    with pytest.raises(SwitchPolicyError, match="non-online receipt keys"):
        assert_online_only_receipt(receipt)
    with pytest.raises(SwitchPolicyError, match="non-online receipt keys"):
        resolve_switch_point(
            window_residual_stall_policy(), receipt, record_count=8
        )


def test_non_online_round_keys_are_refused() -> None:
    receipt = _receipt([1.0, 0.4])
    rows = list(receipt["round_trace"])
    rows[0] = {**rows[0], "max_root_abs_error": 1.0e-6}
    receipt["round_trace"] = tuple(rows)
    with pytest.raises(SwitchPolicyError, match="non-online keys"):
        assert_online_only_receipt(receipt)


# --- canonical trigger ------------------------------------------------------


def test_canonical_trigger_fires_only_on_pool_exhaustion() -> None:
    exhausted = resolve_switch_point(
        CANONICAL_METHOD_POLICY,
        _receipt([1.0, 0.5], stop_reason="pool_exhausted"),
        record_count=2,
    )
    assert exhausted["fired"] is True
    assert exhausted["handoff_k"] is None
    assert exhausted["reason"] == "record_pool_exhausted"

    converged = resolve_switch_point(
        CANONICAL_METHOD_POLICY,
        _receipt([1.0, 0.5], stop_reason="residual_converged"),
        record_count=2,
    )
    assert converged["fired"] is False
    assert converged["handoff_k"] is None


def test_canonical_trigger_ignores_a_stalled_residual() -> None:
    # Flat residuals would trip the stall rule; the canonical policy must not
    # switch on them, because exhaustion-only remains canonical.
    decision = resolve_switch_point(
        CANONICAL_METHOD_POLICY, _receipt([1.0, 1.0, 1.0, 1.0]), record_count=4
    )
    assert decision["fired"] is False
    assert decision["handoff_k"] is None


# --- stall trigger ----------------------------------------------------------


def test_stall_trigger_fires_after_the_declared_consecutive_rounds() -> None:
    # rounds: 1.0 (progress), 0.4 (progress), 0.39 (stall 1), 0.38 (stall 2)
    decision = resolve_switch_point(
        window_residual_stall_policy(stall_rounds=2, improvement_ratio=0.5),
        _receipt([1.0, 0.4, 0.39, 0.38]),
        record_count=10,
    )
    assert decision["fired"] is True
    assert decision["handoff_k"] == 4
    assert decision["reason"] == "window_residual_stall"
    assert decision["evidence"]["consecutive_stalled_rounds"] == 2
    assert decision["evidence"]["stall_round"] == 4
    assert decision["inputs_are_online_only"] is True
    assert decision["uses_exact_reference_for_decision"] is False


def test_steady_progress_never_trips_the_stall_trigger() -> None:
    decision = resolve_switch_point(
        window_residual_stall_policy(stall_rounds=2, improvement_ratio=0.5),
        _receipt([1.0, 0.4, 0.15, 0.05]),
        record_count=10,
    )
    assert decision["fired"] is False
    assert decision["reason"] == "no_stall_detected"
    assert decision["handoff_k"] is None


def test_stall_trigger_falls_back_to_exhaustion_when_no_stall_occurs() -> None:
    decision = resolve_switch_point(
        window_residual_stall_policy(),
        _receipt([1.0, 0.4, 0.1], stop_reason="pool_exhausted"),
        record_count=3,
    )
    assert decision["fired"] is True
    assert decision["handoff_k"] is None
    assert decision["reason"] == "record_pool_exhausted"


def test_handoff_k_is_clamped_into_the_acquisition_order() -> None:
    decision = resolve_switch_point(
        window_residual_stall_policy(stall_rounds=1, improvement_ratio=0.5),
        _receipt([1.0, 1.0, 1.0]),
        record_count=1,
    )
    assert decision["fired"] is True
    assert decision["handoff_k"] == 1


def test_a_stricter_progress_demand_switches_earlier() -> None:
    # improvement_ratio is the factor a round must beat the running best
    # by, so a SMALLER ratio demands more progress and stalls sooner.
    residuals = [1.0, 0.6, 0.3, 0.29]
    lenient = resolve_switch_point(
        window_residual_stall_policy(stall_rounds=1, improvement_ratio=0.9),
        _receipt(residuals),
        record_count=10,
    )
    demanding = resolve_switch_point(
        window_residual_stall_policy(stall_rounds=1, improvement_ratio=0.25),
        _receipt(residuals),
        record_count=10,
    )
    assert demanding["handoff_k"] < lenient["handoff_k"]
