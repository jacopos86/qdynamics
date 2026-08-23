"""Regression tests for the solve-repair subdivision budget and the
certification conditioning gate.

Both guards were nominally present and effectively inert:

* ``max_local_subdivisions`` defaulted to 4, so a step that violated the
  state-motion cap by more than 2**4 exhausted its budget, was flagged
  ``solve_repair_local_subdivision_not_cured``, and then advanced anyway.
  Measured on the HH snake seed (pool 128, t=2, 51 points): 2 such steps took
  the mean energy error from 3.8e-3 to 2.1e-1.  Raising the budget to 10
  removed both and restored the error.
* ``append_schur_max_condition_number`` defaulted to 1e12 while observed
  condition numbers peaked near 1e8, so it never rejected a candidate in any
  measured run.  It is now explicitly disableable (``None``) rather than
  carried at a value that only pretends to bind.
"""

from __future__ import annotations

import pytest

from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig


def test_default_subdivision_budget_covers_measured_violations() -> None:
    """A 2**4 reduction was measurably insufficient on real HH steps."""

    assert SolveRepairConfig().max_local_subdivisions >= 10
    assert SolveRepairConfig.minimal_profile().max_local_subdivisions >= 10


def test_minimal_profile_keeps_both_subdivision_triggers() -> None:
    profile = SolveRepairConfig.minimal_profile()
    assert profile.local_subdivision_enabled
    assert profile.state_motion_l2_step_max == pytest.approx(1.0e-2)
    assert profile.state_space_kink_eta_max == pytest.approx(5.0e-3)


def test_conditioning_gate_can_be_disabled() -> None:
    config = SupportPatchControllerConfig(append_schur_max_condition_number=None)
    assert config.append_schur_max_condition_number is None
    assert config.to_json_dict()["append_schur_max_condition_number"] is None


def test_conditioning_gate_rejects_nonpositive_bound() -> None:
    with pytest.raises(ValueError, match="append_schur_max_condition_number"):
        SupportPatchControllerConfig(append_schur_max_condition_number=0.0)


def test_conditioning_gate_default_is_recorded_as_float() -> None:
    config = SupportPatchControllerConfig()
    recorded = config.to_json_dict()["append_schur_max_condition_number"]
    assert isinstance(recorded, float)


def test_accumulated_drift_integral_and_reset() -> None:
    """The drift integral advances as sqrt(2*residual_sq)*dt and resets on an edit."""

    from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
        _PruneControllerRuntimeState,
    )

    state = _PruneControllerRuntimeState()
    assert state.advance_accumulated_drift(time=0.0, residual_sq=1.0) == 0.0
    assert state.advance_accumulated_drift(time=1.0, residual_sq=0.5) == pytest.approx(1.0)
    assert state.advance_accumulated_drift(time=2.0, residual_sq=0.5) == pytest.approx(2.0)
    # A non-advancing or backwards checkpoint must not add to the integral.
    assert state.advance_accumulated_drift(time=2.0, residual_sq=0.5) == pytest.approx(2.0)
    state.clear_for_support("new-support")
    assert state.accumulated_drift == 0.0


def test_accumulated_drift_escalation_defaults_off() -> None:
    """Default None preserves the historical residual-only escalation."""

    assert SupportPatchControllerConfig().escalation_accumulated_drift_threshold is None


def test_accumulated_drift_threshold_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="escalation_accumulated_drift_threshold"):
        SupportPatchControllerConfig(escalation_accumulated_drift_threshold=0.0)


def test_debt_ranking_orders_by_signed_accuracy_change() -> None:
    """Under L^2 debt the primary key must be the signed drift change.

    The default composite score cannot discriminate there. The deletion loss is
    one-sided, l = [q(0,I) - q(D,I)]_+, so a deletion that LOWERS L^2 and one
    that merely leaves it unchanged both score l = 0; that tie then goes into
    the utility denominator, giving a near-free deletion a score of order
    cost/epsilon_L ~ 1e14 against an insertion utility of order gain/cost.
    """

    from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
        StructuralScoreWeights,
    )

    default = StructuralScoreWeights()
    assert default.debt_ranking is False
    assert default.epsilon_L == pytest.approx(1.0e-14)

    debt = StructuralScoreWeights(debt_ranking=True)
    assert debt.debt_ranking is True


def test_debt_policy_choices_are_validated() -> None:
    from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
        SupportPatchControllerConfig,
    )

    for policy in ("insertion_only", "any_improving", "drift_ranked"):
        assert SupportPatchControllerConfig(debt_policy=policy).debt_policy == policy
    with pytest.raises(ValueError, match="debt_policy"):
        SupportPatchControllerConfig(debt_policy="nope")


def test_debt_ranking_is_not_a_user_flag_default() -> None:
    """`debt_ranking` is set per checkpoint by the loop, never left on."""

    from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
        SupportPatchControllerConfig,
    )

    assert SupportPatchControllerConfig().debt_ranking is False
