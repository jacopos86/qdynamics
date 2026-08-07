from __future__ import annotations

import numpy as np
import pytest

from pipelines.static_adapt.joint_step_warm_start import (
    ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1,
    RouteAJointStepWarmStartConfig,
    certify_exact_physical_transition,
    guard_exact_joint_step_seed,
    guard_exact_joint_step_sign_candidates,
    propose_exact_joint_step_seed,
    retain_seed_preserving_optimizer_outcome,
)
from src.quantum.ansatz_parameterization import (
    GeneratorParameterBlock,
    AnsatzParameterLayout,
    RotationTermSpec,
)


def _layout(labels: list[str]) -> AnsatzParameterLayout:
    return AnsatzParameterLayout(
        mode="per_pauli_term_v1",
        term_order="sorted",
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        blocks=tuple(
            GeneratorParameterBlock(
                candidate_label=label,
                logical_index=index,
                runtime_start=index,
                terms=(RotationTermSpec(pauli_exyz="x", coeff_real=1.0, nq=1),),
            )
            for index, label in enumerate(labels)
        ),
    )


def test_config_validates_exact_guarded_mode() -> None:
    config = RouteAJointStepWarmStartConfig(
        mode=ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1,
    )

    assert config.enabled is True
    assert config.as_dict()["mode"] == ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1


def _physical_transition_certificate(
    *,
    guard_status: str = "accepted",
    fallback_to_incumbent: bool = False,
    declared_labels: list[str] | None = None,
    realized_labels: list[str] | None = None,
    operator_semantic_order_match: bool | None = None,
    operator_identity_order_match: bool = True,
    declared_state: np.ndarray | None = None,
    realized_state: np.ndarray | None = None,
    mapped_optimizer_x: np.ndarray | None = None,
    transition_runtime_parameters: np.ndarray | None = None,
) -> dict[str, object]:
    expected = (
        np.asarray([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
        if declared_state is None
        else np.asarray(declared_state, dtype=complex)
    )
    realized = (
        np.exp(0.37j) * expected
        if realized_state is None
        else np.asarray(realized_state, dtype=complex)
    )
    declared_order = ["a", "x"] if declared_labels is None else declared_labels
    realized_order = ["a", "x"] if realized_labels is None else realized_labels
    semantic_order_match = (
        declared_order == realized_order
        if operator_semantic_order_match is None
        else bool(operator_semantic_order_match)
    )
    return certify_exact_physical_transition(
        guard_payload={
            "status": str(guard_status),
            "fallback_to_incumbent": bool(fallback_to_incumbent),
            "mapped_seed_proposal_energy": -1.2,
        },
        mapped_optimizer_x=(
            np.asarray([0.1, -0.2])
            if mapped_optimizer_x is None
            else np.asarray(mapped_optimizer_x, dtype=float)
        ),
        transition_optimizer_x=np.asarray([0.1, -0.2]),
        transition_runtime_parameters=(
            np.asarray([0.1, -0.2])
            if transition_runtime_parameters is None
            else np.asarray(transition_runtime_parameters, dtype=float)
        ),
        transition_energy=-1.2,
        transition_state_source="mapped_downhill_seed",
        declared_operator_labels=declared_order,
        realized_operator_labels=realized_order,
        operator_semantic_order_match=bool(semantic_order_match),
        operator_identity_order_match=bool(operator_identity_order_match),
        declared_logical_parameter_count=2,
        realized_logical_parameter_count=2,
        declared_runtime_parameter_count=2,
        realized_runtime_parameter_count=2,
        declared_order_state=expected,
        realized_order_state=realized,
        state_consistency_tolerance=1.0e-8,
        state_consistency_tolerance_source="test",
        declared_state_fingerprint="declared",
        realized_state_fingerprint="realized",
    )


def test_physical_transition_certificate_accepts_phase_aligned_exact_replay() -> None:
    payload = _physical_transition_certificate()

    assert payload["physical_transition_certified"] is True
    assert payload["exact_guard_accepted"] is True
    assert payload["finite_map_status"] == "certified"
    assert payload["resulting_circuit_evaluated"] is True
    assert payload["declared_operator_order_preserved"] is True
    assert payload["state_consistency_certified"] is True
    assert float(payload["phase_aligned_state_distance"]) == pytest.approx(
        0.0, abs=1.0e-14
    )


def test_physical_transition_certificate_stays_false_before_guard_acceptance() -> None:
    payload = _physical_transition_certificate(
        guard_status="rejected",
        fallback_to_incumbent=True,
    )

    assert payload["physical_transition_certified"] is False
    assert payload["failed_checks"] == ["exact_guard_accepted"]


@pytest.mark.parametrize(
    ("kwargs", "failed_check"),
    [
        (
            {
                "realized_labels": ["x", "a"],
                "operator_identity_order_match": False,
            },
            "declared_operator_order_preserved",
        ),
        (
            {"operator_semantic_order_match": False},
            "declared_operator_order_preserved",
        ),
        (
            {
                "transition_runtime_parameters": np.asarray(
                    [0.1, np.nan], dtype=float
                )
            },
            "finite_map_certified",
        ),
        (
            {"realized_state": np.asarray([1.0, np.nan], dtype=complex)},
            "state_consistency_certified",
        ),
    ],
)
def test_physical_transition_certificate_rejects_invalid_map_order_or_state(
    kwargs: dict[str, object],
    failed_check: str,
) -> None:
    payload = _physical_transition_certificate(**kwargs)

    assert payload["physical_transition_certified"] is False
    assert failed_check in payload["failed_checks"]


def test_exact_joint_step_maps_active_and_batch_coordinates_through_insertions() -> None:
    # Original [a,b,c]; insert x at 1 and y at 3 -> [a,x,b,c,y].
    layout = _layout(["a", "x", "b", "c", "y"])
    proposal, telemetry = propose_exact_joint_step_seed(
        canonical_x0=np.asarray([10.0, 0.0, 20.0, 30.0, 0.0]),
        post_layout=layout,
        reopt_runtime_active_indices=[0, 1, 2, 3, 4],
        pre_parameter_count=3,
        positions_in_commit_order=[1, 3],
        selected_records=[
            {"candidate_label": "x", "position_id": 1},
            {"candidate_label": "y", "position_id": 3},
        ],
        selector_summary={
            "selected_labels": ["x", "y"],
            "active_parameter_relaxation": [0.5, -0.25],
            "batch_coordinate_step": [0.1, -0.2],
            "geometry_workspace": {"active_indices": [0, 2]},
        },
    )

    assert proposal is not None
    assert proposal.x0 == pytest.approx([10.5, 0.1, 20.0, 29.75, -0.2])
    assert telemetry["active_post_logical_indices"] == [0, 3]
    assert telemetry["candidate_post_logical_indices"] == [1, 4]
    assert telemetry["legacy_diagonal_geometry_used"] is False


def test_exact_joint_step_rejects_record_reordering_and_nonzero_candidate_seed() -> None:
    layout = _layout(["a", "x"])
    common = {
        "post_layout": layout,
        "reopt_runtime_active_indices": [0, 1],
        "pre_parameter_count": 1,
        "positions_in_commit_order": [1],
        "selected_records": [{"candidate_label": "x", "position_id": 1}],
    }
    reordered, reordered_telemetry = propose_exact_joint_step_seed(
        canonical_x0=np.asarray([1.0, 0.0]),
        selector_summary={
            "selected_labels": ["wrong"],
            "active_parameter_relaxation": [0.1],
            "batch_coordinate_step": [0.2],
            "geometry_workspace": {"active_indices": [0]},
        },
        **common,
    )
    nonzero, nonzero_telemetry = propose_exact_joint_step_seed(
        canonical_x0=np.asarray([1.0, 0.3]),
        selector_summary={
            "selected_labels": ["x"],
            "active_parameter_relaxation": [0.1],
            "batch_coordinate_step": [0.2],
            "geometry_workspace": {"active_indices": [0]},
        },
        **common,
    )

    assert reordered is None
    assert reordered_telemetry["reason"] == "selected_record_order_mismatch"
    assert nonzero is None
    assert nonzero_telemetry["reason"] == "candidate_not_initialized_at_zero"


def test_logical_shared_joint_step_reports_lifted_runtime_norm() -> None:
    layout = AnsatzParameterLayout(
        mode="per_pauli_term_v1",
        term_order="sorted",
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        blocks=(
            GeneratorParameterBlock(
                candidate_label="a",
                logical_index=0,
                runtime_start=0,
                terms=(
                    RotationTermSpec(pauli_exyz="x", coeff_real=1.0, nq=1),
                    RotationTermSpec(pauli_exyz="z", coeff_real=1.0, nq=1),
                ),
            ),
            GeneratorParameterBlock(
                candidate_label="x",
                logical_index=1,
                runtime_start=2,
                terms=(
                    RotationTermSpec(pauli_exyz="x", coeff_real=1.0, nq=1),
                    RotationTermSpec(pauli_exyz="y", coeff_real=1.0, nq=1),
                    RotationTermSpec(pauli_exyz="z", coeff_real=1.0, nq=1),
                ),
            ),
        ),
    )
    proposal, telemetry = propose_exact_joint_step_seed(
        canonical_x0=np.asarray([1.0, 0.0]),
        post_layout=layout,
        reopt_runtime_active_indices=[0, 1],
        pre_parameter_count=1,
        positions_in_commit_order=[1],
        selected_records=[{"candidate_label": "x", "position_id": 1}],
        selector_summary={
            "selected_labels": ["x"],
            "active_parameter_relaxation": [0.1],
            "batch_coordinate_step": [0.2],
            "geometry_workspace": {"active_indices": [0]},
        },
        optimizer_coordinate_mode="logical_shared",
    )

    assert proposal is not None
    assert telemetry["optimizer_coordinate_delta_l2"] == pytest.approx(
        np.sqrt(0.1**2 + 0.2**2)
    )
    assert telemetry["runtime_delta_l2"] == pytest.approx(
        np.sqrt(2 * 0.1**2 + 3 * 0.2**2)
    )


@pytest.mark.parametrize(
    ("proposal_center", "expected_status", "expected_x"),
    [(1.2, "accepted", [1.0, 0.2]), (0.0, "rejected", [1.0, 0.0])],
)
def test_objective_guard_accepts_only_an_improving_exact_joint_seed(
    proposal_center: float,
    expected_status: str,
    expected_x: list[float],
) -> None:
    config = RouteAJointStepWarmStartConfig(
        mode=ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1,
    )
    x0, payload, nfev = guard_exact_joint_step_seed(
        objective=lambda x: float((x[0] - 1.0) ** 2 + (x[1] - proposal_center) ** 2),
        canonical_x0=np.asarray([1.0, 0.0]),
        config=config,
        post_layout=_layout(["a", "x"]),
        reopt_runtime_active_indices=[0, 1],
        pre_parameter_count=1,
        positions_in_commit_order=[1],
        selected_records=[{"candidate_label": "x", "position_id": 1}],
        selector_summary={
            "selected_labels": ["x"],
            "active_parameter_relaxation": [0.0],
            "batch_coordinate_step": [0.2],
            "applied_predicted_reduction": 0.5,
            "geometry_workspace": {"active_indices": [0]},
        },
    )

    assert x0 == pytest.approx(expected_x)
    assert payload["schema"] == "route_a_joint_step_warm_start_v1"
    assert payload["status"] == expected_status
    assert payload["fallback_to_incumbent"] is (expected_status != "accepted")
    assert payload["physical_transition_certified"] is False
    assert nfev == 2
    incumbent_energy = float(proposal_center**2)
    proposal_energy = float((0.2 - proposal_center) ** 2)
    exact_gain = float(incumbent_energy - proposal_energy)
    assert payload["selector_applied_predicted_reduction"] == pytest.approx(0.5)
    assert payload["mapped_seed_incumbent_energy"] == pytest.approx(incumbent_energy)
    assert payload["mapped_seed_proposal_energy"] == pytest.approx(proposal_energy)
    assert payload["mapped_seed_exact_gain"] == pytest.approx(exact_gain)
    assert payload["comparison_event_schema"] == (
        "route_a_exact_joint_step_seed_guard_v1"
    )
    assert payload["numerical_energy_comparison_width"] == pytest.approx(
        payload["guard"]["guard_tolerance"]
    )
    assert payload["optimizer_reproducibility_allowance"] == 0.0
    assert payload["aggregate_simultaneous_comparison_width"] == pytest.approx(
        payload["guard"]["guard_tolerance"]
    )
    if exact_gain > 0.0:
        assert payload["prediction_to_exact_seed_ratio"] == pytest.approx(
            0.5 / exact_gain
        )
    else:
        assert payload["prediction_to_exact_seed_ratio"] is None


def _hard_case_sign_guard_kwargs() -> dict[str, object]:
    return {
        "config": RouteAJointStepWarmStartConfig(
            mode=ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1,
        ),
        "post_layout": _layout(["a", "x"]),
        "reopt_runtime_active_indices": [0, 1],
        "pre_parameter_count": 1,
        "positions_in_commit_order": [1],
        "selected_records": [{"candidate_label": "x", "position_id": 1}],
    }


def _hard_case_selector_summary() -> dict[str, object]:
    return {
        "selected_labels": ["x"],
        "active_parameter_relaxation": [0.0],
        "batch_coordinate_step": [0.4],
        "geometry_workspace": {"active_indices": [0]},
        "hard_case_sign_candidates_joint": [[0.0, 0.4], [0.0, -0.4]],
        "hard_case_sign_candidate_predicted_reductions": [0.06, 0.08],
        "hard_case_selected_sign": 1,
    }


def test_hard_case_sign_guard_evaluates_both_and_selects_lowest_exact_energy() -> None:
    x0, payload, nfev = guard_exact_joint_step_sign_candidates(
        objective=lambda x: float((x[1] + 0.3) ** 2),
        canonical_x0=np.asarray([1.0, 0.0]),
        selector_summary=_hard_case_selector_summary(),
        **_hard_case_sign_guard_kwargs(),
    )

    assert x0 == pytest.approx([1.0, -0.4])
    assert payload["status"] == "accepted"
    assert payload["selected_sign"] == -1
    assert payload["selected_sign_accepted"] is True
    assert payload["fallback_to_incumbent"] is False
    assert payload["physical_transition_certified"] is False
    assert payload["mapped_seed_exact_gain"] == pytest.approx(0.08)
    assert payload["prediction_to_exact_seed_ratio"] == pytest.approx(1.0)
    assert nfev == 3
    sign_evaluations = {
        int(item["sign"]): item for item in payload["sign_evaluations"]
    }
    assert sign_evaluations[1]["energy"] == pytest.approx(0.49)
    assert sign_evaluations[-1]["energy"] == pytest.approx(0.01)
    assert sign_evaluations[1]["mapped_optimizer_delta"] == pytest.approx(
        [0.0, 0.4]
    )
    assert sign_evaluations[-1]["mapped_optimizer_delta"] == pytest.approx(
        [0.0, -0.4]
    )


def test_hard_case_sign_guard_fails_closed_when_one_sign_cannot_map() -> None:
    selector_summary = _hard_case_selector_summary()
    selector_summary["hard_case_sign_candidates_joint"] = [
        [0.0, 0.4],
        [0.0],
    ]
    objective_calls = 0

    def objective(x: np.ndarray) -> float:
        nonlocal objective_calls
        objective_calls += 1
        return float((x[1] - 0.4) ** 2)

    x0, payload, nfev = guard_exact_joint_step_sign_candidates(
        objective=objective,
        canonical_x0=np.asarray([1.0, 0.0]),
        selector_summary=selector_summary,
        **_hard_case_sign_guard_kwargs(),
    )

    assert x0 == pytest.approx([1.0, 0.0])
    assert payload["status"] == "unavailable"
    assert payload["reason"] == "atomic_sign_pair_mapping_failed"
    assert payload["fallback_to_incumbent"] is True
    assert [item["status"] for item in payload["sign_evaluations"]] == [
        "mapped",
        "unavailable",
    ]
    assert objective_calls == 0
    assert nfev == 0


def test_hard_case_sign_guard_keeps_incumbent_when_both_signs_are_non_descending(
) -> None:
    x0, payload, nfev = guard_exact_joint_step_sign_candidates(
        objective=lambda x: float(x[1] ** 2),
        canonical_x0=np.asarray([1.0, 0.0]),
        selector_summary=_hard_case_selector_summary(),
        **_hard_case_sign_guard_kwargs(),
    )

    assert x0 == pytest.approx([1.0, 0.0])
    assert payload["status"] == "rejected"
    assert payload["selected_sign"] == 1
    assert payload["selected_sign_accepted"] is False
    assert payload["fallback_to_incumbent"] is True
    assert payload["mapped_seed_exact_gain"] == pytest.approx(-0.16)
    assert payload["prediction_to_exact_seed_ratio"] is None
    assert payload["finite_mapped_sign_count"] == 2
    assert payload["all_mapped_signs_finite"] is True
    assert payload["all_mapped_signs_non_downhill"] is True
    assert payload["sr_saddle_transaction_outcome"] == (
        "radius_contract_refinement_no_state_mutation"
    )
    assert payload["trust_action"] == "contract_with_numerical_backtracking"
    assert payload["numerical_energy_comparison_width"] == pytest.approx(
        payload["guard"]["guard_tolerance"]
    )
    assert payload["optimizer_reproducibility_allowance"] == 0.0
    assert payload["aggregate_simultaneous_comparison_width"] == pytest.approx(
        payload["guard"]["guard_tolerance"]
    )
    assert nfev == 3


def test_hard_case_sign_guard_holds_when_non_descent_overlaps_comparison_width(
) -> None:
    x0, payload, nfev = guard_exact_joint_step_sign_candidates(
        objective=lambda _x: 0.0,
        canonical_x0=np.asarray([1.0, 0.0]),
        selector_summary=_hard_case_selector_summary(),
        **_hard_case_sign_guard_kwargs(),
    )

    assert x0 == pytest.approx([1.0, 0.0])
    assert payload["status"] == "rejected"
    assert payload["all_mapped_signs_point_estimate_non_downhill"] is True
    assert payload["all_mapped_signs_non_downhill"] is False
    assert payload["sign_pair_energy_comparison_resolved"] is False
    assert payload["sign_pair_energy_comparison_unresolved"] is True
    assert payload["transaction_failure_kind"] == "comparison_unresolved"
    assert payload["sr_saddle_transaction_outcome"] == (
        "no_state_refinement_trust_hold"
    )
    assert payload["trust_action"] == "hold_for_comparison_refinement"
    assert all(
        item["mapped_seed_exact_gain_upper_bound"] > 0.0
        for item in payload["sign_evaluations"]
    )
    assert nfev == 3


def test_seed_preserving_optimizer_outcome_retains_better_mapped_seed() -> None:
    x, energy, payload = retain_seed_preserving_optimizer_outcome(
        mapped_seed_x0=np.asarray([1.0, -0.4]),
        mapped_seed_energy=-1.2,
        optimizer_x=np.asarray([1.1, -0.2]),
        optimizer_energy=-1.1,
        incumbent_energy=-1.0,
    )

    assert x == pytest.approx([1.0, -0.4])
    assert energy == pytest.approx(-1.2)
    assert payload["post_refit_safe_source"] == "mapped_downhill_seed"
    assert payload["mapped_seed_retained"] is True
    assert payload["optimizer_result_discarded"] is True


def test_seed_preserving_optimizer_outcome_accepts_materially_better_powell() -> None:
    x, energy, payload = retain_seed_preserving_optimizer_outcome(
        mapped_seed_x0=np.asarray([1.0, -0.4]),
        mapped_seed_energy=-1.2,
        optimizer_x=np.asarray([1.2, -0.5]),
        optimizer_energy=-1.3,
        incumbent_energy=-1.0,
    )

    assert x == pytest.approx([1.2, -0.5])
    assert energy == pytest.approx(-1.3)
    assert payload["post_refit_safe_source"] == "optimizer_result"
    assert payload["mapped_seed_retained"] is False
    assert payload["optimizer_result_discarded"] is False


def test_seed_preserving_optimizer_outcome_discards_nonfinite_powell() -> None:
    x, energy, payload = retain_seed_preserving_optimizer_outcome(
        mapped_seed_x0=np.asarray([1.0, -0.4]),
        mapped_seed_energy=-1.2,
        optimizer_x=np.asarray([np.nan, -0.5]),
        optimizer_energy=-1.3,
        incumbent_energy=-1.0,
    )

    assert x == pytest.approx([1.0, -0.4])
    assert energy == pytest.approx(-1.2)
    assert payload["optimizer_point_finite"] is False
    assert payload["post_refit_safe_source"] == "mapped_downhill_seed"


def test_seed_preserving_optimizer_outcome_retains_seed_on_tolerance_overlap() -> None:
    x, energy, payload = retain_seed_preserving_optimizer_outcome(
        mapped_seed_x0=np.asarray([1.0, -0.4]),
        mapped_seed_energy=-1.2,
        optimizer_x=np.asarray([1.2, -0.5]),
        optimizer_energy=-1.2 - 5.0e-13,
        incumbent_energy=-1.0,
        guard_abs_tol=1.0e-12,
        guard_rel_tol=0.0,
    )

    assert x == pytest.approx([1.0, -0.4])
    assert energy == pytest.approx(-1.2)
    assert payload["optimizer_materially_lower"] is False
    assert payload["mapped_seed_retained"] is True


def test_hard_case_sign_guard_is_independent_of_supplied_candidate_order() -> None:
    selector_summary = _hard_case_selector_summary()
    common = {
        "objective": lambda x: float((x[1] + 0.3) ** 2),
        "canonical_x0": np.asarray([1.0, 0.0]),
        "selector_summary": selector_summary,
        **_hard_case_sign_guard_kwargs(),
    }
    forward_x0, forward_payload, forward_nfev = (
        guard_exact_joint_step_sign_candidates(
            joint_steps=[[0.0, 0.4], [0.0, -0.4]],
            **common,
        )
    )
    reverse_x0, reverse_payload, reverse_nfev = (
        guard_exact_joint_step_sign_candidates(
            joint_steps=[[0.0, -0.4], [0.0, 0.4]],
            **common,
        )
    )

    assert forward_x0 == pytest.approx(reverse_x0)
    assert forward_payload["selected_sign"] == reverse_payload["selected_sign"] == -1
    assert forward_payload["mapped_seed_exact_gain"] == pytest.approx(
        reverse_payload["mapped_seed_exact_gain"]
    )
    assert forward_nfev == reverse_nfev == 3
