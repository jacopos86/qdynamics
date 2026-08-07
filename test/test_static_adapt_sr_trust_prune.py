from __future__ import annotations

import numpy as np
import pytest

from pipelines.scaffold.hh_continuation_pruning import (
    AFFINE_DELETION_FS_TRUST_TRIAL_RECEIPT_V1,
    AffineDeletionFSTrustState,
    AffineDeletionFSTrustUpdateConfig,
    initialize_affine_deletion_fs_trust_state,
    rank_prune_candidates,
    recoverability_prune_ladder,
    solve_full_logical_affine_deletion_fs_trust,
    update_affine_deletion_fs_trust_state,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS,
)


def test_recoverability_prune_ladder_preserves_legacy_four_arg_callback() -> None:
    calls: list[int] = []

    def _legacy_eval(
        idx_remove: int,
        theta_cur: np.ndarray,
        labels_cur: list[str],
        active_indices: list[int],
    ) -> tuple[float, np.ndarray]:
        calls.append(int(idx_remove))
        return -1.0, np.delete(theta_cur, idx_remove)

    theta_out, labels_out, decisions, energy_out, _rows = recoverability_prune_ladder(
        theta=np.asarray([0.2, 0.1], dtype=float),
        labels=["keep", "drop"],
        candidate_indices=[1],
        rung_windows_by_index={1: [("full_survivor_refit", [0])]},
        eval_with_removal_window=_legacy_eval,
        energy_before=-1.0,
        max_regression=1.0e-9,
        max_trial_evaluations=1,
    )

    assert calls == [1]
    assert labels_out == ["keep"]
    np.testing.assert_allclose(theta_out, [0.2])
    assert energy_out == pytest.approx(-1.0)
    assert decisions[0].accepted is True


def test_recoverability_prune_ladder_internal_typeerror_never_retries() -> None:
    calls: list[str] = []

    def _five_arg_eval(
        idx_remove: int,
        theta_cur: np.ndarray,
        labels_cur: list[str],
        active_indices: list[int],
        rung_kind: str,
    ) -> tuple[float, np.ndarray]:
        calls.append(str(rung_kind))
        raise TypeError("internal objective failure")

    with pytest.raises(TypeError, match="internal objective failure"):
        recoverability_prune_ladder(
            theta=np.asarray([0.2, 0.1], dtype=float),
            labels=["keep", "drop"],
            candidate_indices=[1],
            rung_windows_by_index={1: [("full_survivor_refit", [0])]},
            eval_with_removal_window=_five_arg_eval,
            energy_before=-1.0,
            max_regression=1.0e-9,
            max_trial_evaluations=1,
        )

    assert calls == ["full_survivor_refit"]


def _complete_receipt(
    *,
    trial_id: str = "trial-7",
    prediction_trial_id: str | None = None,
    realization_trial_id: str | None = None,
    predicted_energy_change: float = 0.1,
    realized_energy_change: float = 0.2,
    energy_comparison_width: float = 1.0e-10,
) -> dict[str, object]:
    return {
        "schema": AFFINE_DELETION_FS_TRUST_TRIAL_RECEIPT_V1,
        "trial_id": str(trial_id),
        "prediction_trial_id": str(
            trial_id if prediction_trial_id is None else prediction_trial_id
        ),
        "realization_trial_id": str(
            trial_id if realization_trial_id is None else realization_trial_id
        ),
        "prediction_complete": True,
        "realization_complete": True,
        "energy_receipt_complete": True,
        "predicted_energy_change": float(predicted_energy_change),
        "realized_energy_change": float(realized_energy_change),
        "energy_comparison_width": float(energy_comparison_width),
    }


def test_affine_deletion_solves_full_logical_interior_model() -> None:
    result = solve_full_logical_affine_deletion_fs_trust(
        theta=np.asarray([0.2, 0.0]),
        gradient=np.asarray([0.0, 0.3]),
        hessian=np.eye(2),
        metric=np.eye(2),
        deletion_index=0,
        trust_radius=1.0,
    )

    assert result.feasible is True
    np.testing.assert_allclose(result.joint_step, [-0.2, -0.3], atol=1.0e-12)
    assert result.joint_step[0] == pytest.approx(-0.2)
    assert result.fubini_study_displacement_sq == pytest.approx(0.13)
    assert result.predicted_energy_change == pytest.approx(-0.025)
    assert result.telemetry["pre_support_coordinate_count"] == 2
    assert result.telemetry["metric_supported_rank"] == 2
    assert result.telemetry[
        "all_logical_coordinates_entered_before_support_reduction"
    ] is True
    assert result.telemetry["classical_quantum_query_charge"] == 0


def test_affine_deletion_respects_full_fs_boundary_including_deleted_coordinate() -> None:
    result = solve_full_logical_affine_deletion_fs_trust(
        theta=np.asarray([0.2, 0.0]),
        gradient=np.asarray([0.0, -2.0]),
        hessian=np.eye(2),
        metric=np.eye(2),
        deletion_index=0,
        trust_radius=0.5,
    )

    assert result.feasible is True
    assert result.joint_step[0] == pytest.approx(-0.2)
    assert result.joint_step[1] == pytest.approx(
        np.sqrt(0.5**2 - 0.2**2), abs=2.0e-9
    )
    assert result.fubini_study_displacement_sq == pytest.approx(
        0.5**2, abs=1.0e-10
    )
    assert result.telemetry["trust_radius_binding"] is True


def test_affine_deletion_fails_when_fixed_coordinate_exceeds_fs_radius() -> None:
    result = solve_full_logical_affine_deletion_fs_trust(
        theta=np.asarray([0.6, 0.0]),
        gradient=np.zeros(2),
        hessian=np.eye(2),
        metric=np.eye(2),
        deletion_index=0,
        trust_radius=0.5,
    )

    assert result.feasible is False
    assert result.reason == "affine_deletion_outside_trust_radius"
    assert result.telemetry[
        "minimum_affine_fubini_study_displacement_sq"
    ] == pytest.approx(0.36)


def test_affine_deletion_removes_only_genuine_metric_null_modes_after_entry() -> None:
    result = solve_full_logical_affine_deletion_fs_trust(
        theta=np.asarray([0.2, 4.0]),
        gradient=np.zeros(2),
        hessian=np.eye(2),
        metric=np.diag([1.0, 0.0]),
        deletion_index=0,
        trust_radius=0.5,
    )

    assert result.feasible is True
    np.testing.assert_allclose(result.joint_step, [-0.2, 0.0], atol=1.0e-12)
    assert result.telemetry["pre_support_coordinate_count"] == 2
    assert result.telemetry["metric_supported_rank"] == 1
    assert result.telemetry["metric_retained_mask"] == [False, True]


def test_affine_deletion_fails_closed_when_deleted_coordinate_is_metric_null() -> None:
    result = solve_full_logical_affine_deletion_fs_trust(
        theta=np.asarray([0.0, 0.2]),
        gradient=np.zeros(2),
        hessian=np.eye(2),
        metric=np.diag([1.0, 0.0]),
        deletion_index=1,
        trust_radius=0.5,
    )

    assert result.feasible is False
    assert result.reason == "deletion_coordinate_not_in_supported_metric_range"


def test_affine_deletion_reuses_global_hard_case_solver() -> None:
    result = solve_full_logical_affine_deletion_fs_trust(
        theta=np.zeros(3),
        gradient=np.zeros(3),
        hessian=np.diag([0.0, -1.0, 1.0]),
        metric=np.eye(3),
        deletion_index=0,
        trust_radius=0.5,
    )

    assert result.feasible is True
    assert result.joint_step[0] == pytest.approx(0.0)
    assert result.fubini_study_displacement_sq == pytest.approx(0.25)
    assert result.predicted_energy_change == pytest.approx(-0.125)
    assert result.telemetry["reduced_trust_solve"]["hard_case_detected"] is True


@pytest.mark.parametrize(
    ("measured_energy", "expected_accepted"),
    [
        (-1.001, True),
        (-0.98, False),
    ],
)
def test_v4_live_prune_nominates_and_measures_exactly_one_trial(
    measured_energy: float,
    expected_accepted: bool,
) -> None:
    """Integrate the v4 model, nomination, exact-refit, and trust contracts.

    Two full-logical affine-deletion models are feasible, but the route cap
    nominates only one.  Both cases use the same model prediction; changing
    only the measured delete-and-refit energy flips acceptance, proving that
    the response model nominates but never authorizes deletion.
    """

    route = CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS
    assert route["phase1_prune_mode"] == "live"
    assert route["adapt_final_full_refit"] == "false"
    assert route["phase1_prune_max_candidates"] == 1
    assert route["phase1_prune_local_window_size"] == 0

    theta = np.asarray([0.1, 0.2, 0.6], dtype=float)
    labels = ["candidate-a", "candidate-b", "outside-radius"]
    gradient = np.zeros(3, dtype=float)
    hessian = np.eye(3, dtype=float)
    metric = np.eye(3, dtype=float)
    trust_state = initialize_affine_deletion_fs_trust_state(radius=0.3)

    model_rows: dict[int, dict[str, object]] = {}
    feasible_indices: list[int] = []
    for index in range(len(labels)):
        result = solve_full_logical_affine_deletion_fs_trust(
            theta=theta,
            gradient=gradient,
            hessian=hessian,
            metric=metric,
            deletion_index=index,
            trust_radius=trust_state.radius,
            metric_damping=trust_state.metric_damping,
        )
        assert result.telemetry["pre_support_coordinate_count"] == len(labels)
        assert result.telemetry[
            "all_logical_coordinates_entered_before_support_reduction"
        ] is True
        if result.feasible:
            feasible_indices.append(index)
            model_rows[index] = {
                "score": float(result.predicted_energy_change),
                "predicted_energy_change": float(
                    result.predicted_energy_change
                ),
                "affine_deletion_model": result.as_dict(),
                "used_for_acceptance": False,
            }

    assert feasible_indices == [0, 1]
    nominated = rank_prune_candidates(
        theta=theta,
        labels=labels,
        marginal_proxy_benefit=[0.0, 0.0, 0.0],
        max_candidates=int(route["phase1_prune_max_candidates"]),
        min_candidates=1,
        fraction_candidates=1.0,
        selector_burden=[0.0, 0.0, 0.0],
        admission_steps=[0, 0, 0],
        cooldown_remaining=[0, 0, 0],
        current_step=4,
        protect_steps=0,
        policy="recoverability_ladder_v1",
        surrogate_scores=model_rows,
        surrogate_score_threshold=0.01,
        surrogate_candidate_cap=1,
        surrogate_score_primary_only=True,
    )
    assert nominated == [0]
    selected_model = model_rows[nominated[0]]
    assert selected_model["used_for_acceptance"] is False

    exact_refit_calls: list[dict[str, object]] = []

    def _measured_delete_and_refit(
        idx_remove: int,
        theta_cur: np.ndarray,
        labels_cur: list[str],
        active_indices: list[int],
        rung_kind: str,
    ) -> tuple[float, np.ndarray]:
        exact_refit_calls.append(
            {
                "idx_remove": int(idx_remove),
                "labels": list(labels_cur),
                "active_indices": list(active_indices),
                "rung_kind": str(rung_kind),
            }
        )
        return float(measured_energy), np.delete(theta_cur, idx_remove)

    energy_before = -1.0
    (
        theta_after,
        labels_after,
        decisions,
        energy_after,
        ladder_rows,
    ) = recoverability_prune_ladder(
        theta=theta,
        labels=labels,
        candidate_indices=nominated,
        rung_windows_by_index={
            nominated[0]: [("full_survivor_refit", [0, 1])]
        },
        eval_with_removal_window=_measured_delete_and_refit,
        energy_before=energy_before,
        max_regression=0.01,
        retained_reference_energy=-0.9,
        admitted_gain=0.1,
        retained_gain_ratio=0.5,
        retained_gain_activation=1.0e-12,
        curvature_guard_mode="off",
        curvature_guard_context={
            "compression_mode": False,
            "terminal_full": False,
        },
        max_trial_evaluations=1,
    )

    assert len(exact_refit_calls) == 1
    assert exact_refit_calls[0] == {
        "idx_remove": 0,
        "labels": labels,
        "active_indices": [0, 1],
        "rung_kind": "full_survivor_refit",
    }
    assert len(decisions) == len(ladder_rows) == 1
    decision = decisions[0]
    assert decision.accepted is expected_accepted
    assert decision.energy_after == pytest.approx(measured_energy)
    assert energy_after == pytest.approx(
        measured_energy if expected_accepted else energy_before
    )
    assert labels_after == (
        ["candidate-b", "outside-radius"]
        if expected_accepted
        else labels
    )
    assert theta_after.size == (2 if expected_accepted else 3)

    trial_id = "sr-v4-prune:round=4:index=0:label=candidate-a"
    trial_receipt = {
        "schema": AFFINE_DELETION_FS_TRUST_TRIAL_RECEIPT_V1,
        "trial_id": trial_id,
        "prediction_trial_id": trial_id,
        "realization_trial_id": trial_id,
        "prediction_complete": True,
        "realization_complete": True,
        "energy_receipt_complete": True,
        "predicted_energy_change": float(
            selected_model["predicted_energy_change"]
        ),
        "realized_energy_change": float(
            decision.energy_after - decision.energy_before
        ),
        "energy_comparison_width": 0.01,
        "measured_delete_refit_is_acceptance_authority": True,
        "endpoint_overlap_measured": False,
        "added_endpoint_overlap_query_count": 0,
    }
    state_after, update = update_affine_deletion_fs_trust_state(
        trust_state,
        contract_radius=not expected_accepted,
        trial_receipt=trial_receipt,
    )

    assert state_after.update_count == 1
    assert state_after.radius <= trust_state.radius
    assert state_after.metric_damping >= trust_state.metric_damping
    assert update["radius_never_increased"] is True
    assert update["metric_damping_never_decreased"] is True
    assert update["classical_quantum_query_charge"] == 0
    if expected_accepted:
        assert state_after.radius == pytest.approx(0.3)
        assert state_after.metric_damping == 0.0
    else:
        assert state_after.radius == pytest.approx(0.15)
        assert state_after.metric_damping == pytest.approx(1.0e-6)


def test_metric_damping_changes_model_without_query_charge() -> None:
    common = {
        "theta": np.asarray([0.2, 0.0]),
        "gradient": np.asarray([0.0, -2.0]),
        "hessian": np.zeros((2, 2)),
        "metric": np.eye(2),
        "deletion_index": 0,
        "trust_radius": 10.0,
    }
    undamped = solve_full_logical_affine_deletion_fs_trust(
        **common,
        metric_damping=0.0,
    )
    damped = solve_full_logical_affine_deletion_fs_trust(
        **common,
        metric_damping=2.0,
    )

    assert undamped.feasible is True
    assert damped.feasible is True
    assert abs(undamped.joint_step[1]) > abs(damped.joint_step[1])
    assert damped.joint_step[1] == pytest.approx(1.0, abs=1.0e-9)
    assert damped.telemetry["classical_quantum_query_charge"] == 0


def test_new_trust_state_starts_undamped_and_radius_never_expands() -> None:
    state = initialize_affine_deletion_fs_trust_state(radius=0.4)
    assert state.metric_damping == 0.0

    held, held_payload = update_affine_deletion_fs_trust_state(
        state,
        contract_radius=False,
    )
    assert held.radius == pytest.approx(0.4)
    assert held.metric_damping == 0.0
    assert held_payload["radius_never_increased"] is True

    contracted, payload = update_affine_deletion_fs_trust_state(
        held,
        contract_radius=True,
        config=AffineDeletionFSTrustUpdateConfig(
            radius_contraction_factor=0.5,
            radius_floor=0.3,
        ),
    )
    assert contracted.radius == pytest.approx(0.3)
    assert contracted.radius <= held.radius
    assert payload["radius_policy"] == "contraction_only_v1"


@pytest.mark.parametrize(
    "receipt, expected_status",
    [
        (None, "missing_receipt"),
        (
            {
                **_complete_receipt(),
                "prediction_complete": False,
            },
            "incomplete_receipt",
        ),
        (
            _complete_receipt(realization_trial_id="different-trial"),
            "trial_identity_mismatch",
        ),
    ],
)
def test_metric_damping_holds_without_complete_same_trial_receipt(
    receipt: dict[str, object] | None,
    expected_status: str,
) -> None:
    state = AffineDeletionFSTrustState(
        radius=0.4,
        metric_damping=0.0,
    )
    updated, payload = update_affine_deletion_fs_trust_state(
        state,
        contract_radius=False,
        trial_receipt=receipt,
    )

    assert updated.metric_damping == 0.0
    assert payload["damping_receipt_status"] == expected_status
    assert payload["complete_same_trial_underprediction"] is False


def test_metric_damping_only_rises_for_material_same_trial_underprediction() -> None:
    config = AffineDeletionFSTrustUpdateConfig(
        damping_initial_increment=2.0e-5,
        damping_growth_factor=3.0,
    )
    state = initialize_affine_deletion_fs_trust_state(radius=0.4)
    increased, payload = update_affine_deletion_fs_trust_state(
        state,
        contract_radius=False,
        trial_receipt=_complete_receipt(
            predicted_energy_change=0.1,
            realized_energy_change=0.2,
            energy_comparison_width=1.0e-4,
        ),
        config=config,
    )

    assert increased.metric_damping == pytest.approx(2.0e-5)
    assert payload["complete_same_trial_underprediction"] is True
    assert payload["metric_damping_action"] == (
        "complete_same_trial_underprediction_increase"
    )

    increased_again, _ = update_affine_deletion_fs_trust_state(
        increased,
        contract_radius=False,
        trial_receipt=_complete_receipt(
            trial_id="trial-8",
            predicted_energy_change=0.1,
            realized_energy_change=0.2,
        ),
        config=config,
    )
    assert increased_again.metric_damping == pytest.approx(6.0e-5)

    held, held_payload = update_affine_deletion_fs_trust_state(
        increased_again,
        contract_radius=False,
        trial_receipt=_complete_receipt(
            trial_id="trial-9",
            predicted_energy_change=0.2,
            realized_energy_change=0.2 + 5.0e-5,
            energy_comparison_width=1.0e-4,
        ),
        config=config,
    )
    assert held.metric_damping == pytest.approx(increased_again.metric_damping)
    assert held_payload["complete_same_trial_underprediction"] is False
    assert held_payload["metric_damping_never_decreased"] is True
