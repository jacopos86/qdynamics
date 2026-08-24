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


def _candidate(delta, ins_u=0.0, del_u=0.0, *, debt, tol=1.0e-12, order=(0,)):
    """Build a scored candidate the way exchange_structural does."""

    import numpy as np

    from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
        StructuralCandidate,
    )

    if debt:
        rank_primary = float(round(float(delta) / tol))
        rank_secondary = float(np.tanh(ins_u + del_u))
    else:
        rank_primary = float(ins_u + del_u + delta)
        rank_secondary = 0.0
    return StructuralCandidate(
        removed_runtime_indices=(),
        inserted_selection=(),
        plan=None,
        family="test",
        q=0.0,
        insertion_gain=0.0,
        deletion_loss=0.0,
        delta=float(delta),
        insertion_utility=float(ins_u),
        deletion_utility=float(del_u),
        score=float(ins_u + del_u + delta),
        rank_primary=rank_primary,
        rank_secondary=rank_secondary,
    )


def _rank(candidates):
    return sorted(
        candidates,
        key=lambda c: (
            -float(getattr(c, "rank_primary", c.score)),
            -float(getattr(c, "rank_secondary", 0.0)),
            c.order_key,
        ),
    )


def test_larger_delta_beats_an_arbitrarily_large_deletion_utility() -> None:
    """The divergent deletion utility must not outrank a better delta.

    A near-zero-loss deletion scores ~cost/epsilon_L ~ 1e14 in the composite
    score. Under debt ranking it must still lose to any candidate with a
    strictly larger signed drift change.
    """

    improving_insert = _candidate(1.0e-3, ins_u=1.0e-6, debt=True)
    cheap_deletion = _candidate(-1.0e-4, del_u=1.0e14, debt=True)
    assert _rank([cheap_deletion, improving_insert])[0] is improving_insert


def test_equal_delta_is_ordered_by_the_secondary_utility() -> None:
    lo = _candidate(5.0e-4, ins_u=1.0, debt=True, order=(1,))
    hi = _candidate(5.0e-4, del_u=1.0e9, debt=True, order=(2,))
    ranked = _rank([lo, hi])
    assert ranked[0] is hi and ranked[1] is lo


def test_delta_differences_below_tolerance_count_as_tied() -> None:
    """`delta_rank_tolerance` makes the tie explicit rather than accidental."""

    tol = 1.0e-12
    a = _candidate(1.0e-3, ins_u=0.0, debt=True, tol=tol)
    b = _candidate(1.0e-3 + tol / 4.0, del_u=1.0e12, debt=True, tol=tol)
    assert a.rank_primary == b.rank_primary
    assert _rank([a, b])[0] is b          # utility breaks the declared tie


def test_below_debt_the_composite_score_still_orders() -> None:
    cheap_deletion = _candidate(-1.0e-4, del_u=1.0e14, debt=False)
    improving_insert = _candidate(1.0e-3, ins_u=1.0e-6, debt=False)
    assert _rank([improving_insert, cheap_deletion])[0] is cheap_deletion


def test_a_non_improving_deletion_cannot_head_the_debt_ranking() -> None:
    """Negative delta must sort below every improving candidate."""

    improving = [_candidate(d, debt=True) for d in (1e-6, 1e-5, 1e-4)]
    harmful = _candidate(-1.0e-3, del_u=1.0e14, debt=True)
    ranked = _rank([harmful, *improving])
    assert ranked[-1] is harmful
    assert ranked[0].delta == pytest.approx(1e-4)


def test_debt_ranking_weights_carry_the_declared_tolerance() -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
        StructuralScoreWeights,
    )

    default = StructuralScoreWeights()
    assert default.debt_ranking is False
    assert default.epsilon_L == pytest.approx(1.0e-14)
    assert StructuralScoreWeights(debt_ranking=True).delta_rank_tolerance > 0.0


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


def test_state_and_parameter_guards_compose() -> None:
    """Both bounds must be expressible at once, not one instead of the other.

    The state bound passes a step whenever the state barely moves, which is
    exactly when an ill-conditioned solve returns a large spurious theta_dot.
    Measured on the fast weak drive: under the state bound alone the three
    largest step-to-step jumps carried 81% of all error growth; under a
    parameter bound alone, 6%.
    """

    from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig

    config = SolveRepairConfig.minimal_profile(parameter_step_max=5.0e-3)
    assert config.state_motion_l2_step_max == pytest.approx(1.0e-2)
    assert config.parameter_step_max == pytest.approx(5.0e-3)
    recorded = config.to_json_dict()
    assert recorded["state_motion_l2_step_max"] == pytest.approx(1.0e-2)
    assert recorded["parameter_step_max"] == pytest.approx(5.0e-3)


def test_parameter_guard_defaults_off_so_existing_runs_are_unchanged() -> None:
    from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig

    assert SolveRepairConfig().parameter_step_max is None
    assert SolveRepairConfig.minimal_profile().parameter_step_max is None


def test_parameter_guard_opens_a_subdivision_lane() -> None:
    """An exceeded parameter bound must schedule subdivision, like state motion."""

    from pipelines.time_dynamics.ap_mclachlan.fixed_step import (
        SolveGuardReport,
        SolveRepairConfig,
        _repair_response_schedule,
    )

    config = SolveRepairConfig.minimal_profile(parameter_step_max=5.0e-3)
    report = SolveGuardReport(
        repair_dt=0.04, g_empty=False, g_kappa=False, g_delta=False,
        g_theta=True, g_rho=False, g_kink=False, retained_support_empty=False,
        state_motion_l2_step=1.0e-4,     # state bound satisfied
        parameter_step=4.0e-2,           # parameter bound exceeded 8x
        state_space_kink_eta=None, rho_real=None, rho_expr=None, rho_num=None,
        projected_velocity_l2=None, realized_residual_sq=None,
        best_case_residual_sq=None, guard_reason="parameter_step_above_max",
    )
    schedule = _repair_response_schedule(report, repair_config=config)
    assert "theta" in schedule.active_lanes
    assert schedule.local_subdivision_breadth >= 1
