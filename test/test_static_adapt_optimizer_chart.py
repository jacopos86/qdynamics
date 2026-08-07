import numpy as np
import pytest

from pipelines.static_adapt.engine_support import (
    SelectedOptimizerChart,
    _guard_sr_active_only_step,
    _retain_sr_active_only_refit_outcome,
    make_selected_optimizer_chart,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def _two_generator_layout():
    terms = [
        AnsatzTerm(
            label="two_factor",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(2, ps="xy", pc=0.5),
                    PauliTerm(2, ps="yx", pc=-0.5),
                ],
            ),
        ),
        AnsatzTerm(
            label="one_factor",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="ze", pc=1.0)],
            ),
        ),
    ]
    return build_parameter_layout(
        terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )


def test_logical_shared_powell_chart_has_one_coordinate_per_active_generator():
    layout = _two_generator_layout()
    runtime_theta = np.asarray([0.2, 0.2, -0.4], dtype=float)
    seen: list[np.ndarray] = []

    def objective(theta: np.ndarray) -> float:
        seen.append(np.asarray(theta, dtype=float).copy())
        return float(np.dot(theta, theta))

    chart = make_selected_optimizer_chart(
        full_theta=runtime_theta,
        layout=layout,
        active_logical_indices=[1, 0],
        objective=objective,
        parameterization_mode="logical_shared",
        effective_optimizer_key="powell",
        powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
    )

    assert isinstance(chart, SelectedOptimizerChart)
    assert chart.coordinate_mode == "logical_shared"
    assert chart.active_logical_indices == (1, 0)
    assert chart.active_runtime_indices == (2, 0, 1)
    assert chart.active_optimizer_indices == (1, 0)
    assert chart.reduced_positions_by_logical == {1: (0,), 0: (1,)}
    assert chart.x0.tolist() == [-0.4, 0.2]

    trial = np.asarray([0.7, -0.1], dtype=float)
    expected_runtime = np.asarray([-0.1, -0.1, 0.7], dtype=float)
    assert np.array_equal(chart.lift_to_runtime(trial), expected_runtime)
    assert chart.objective(trial) == pytest.approx(float(np.dot(expected_runtime, expected_runtime)))
    assert np.array_equal(seen[-1], expected_runtime)


def test_historical_expanded_powell_chart_projects_runtime_blocks_at_boundaries():
    layout = _two_generator_layout()
    runtime_theta = np.asarray([0.2, 0.3, -0.4], dtype=float)
    seen: list[np.ndarray] = []

    def objective(theta: np.ndarray) -> float:
        seen.append(np.asarray(theta, dtype=float).copy())
        return float(np.dot(theta, theta))

    chart = make_selected_optimizer_chart(
        full_theta=runtime_theta,
        layout=layout,
        active_logical_indices=[0],
        objective=objective,
        parameterization_mode="logical_shared",
        effective_optimizer_key="POWELL",
        powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
    )

    assert chart.coordinate_mode == "runtime"
    assert chart.active_logical_indices == (0,)
    assert chart.active_runtime_indices == (0, 1)
    assert chart.active_optimizer_indices == (0, 1)
    assert chart.reduced_positions_by_logical == {0: (0, 1)}
    assert chart.x0.tolist() == [0.2, 0.3]

    trial = np.asarray([0.6, -0.3], dtype=float)
    expected_runtime = np.asarray([0.15, 0.15, -0.4], dtype=float)
    assert chart.lift_to_runtime(trial).tolist() == pytest.approx(
        expected_runtime.tolist()
    )
    assert chart.objective(trial) == pytest.approx(
        float(np.dot(expected_runtime, expected_runtime))
    )
    assert seen[-1].tolist() == pytest.approx(expected_runtime.tolist())


def test_logical_shared_non_powell_keeps_runtime_chart_but_lifts_uniform_blocks():
    layout = _two_generator_layout()
    runtime_theta = np.asarray([0.2, 0.2, -0.4], dtype=float)

    chart = make_selected_optimizer_chart(
        full_theta=runtime_theta,
        layout=layout,
        active_logical_indices=[0],
        objective=lambda theta: float(np.sum(theta)),
        parameterization_mode="logical_shared",
        effective_optimizer_key="ROTOSOLVE",
        powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
    )

    assert chart.coordinate_mode == "runtime"
    assert chart.active_logical_indices == (0,)
    assert chart.active_runtime_indices == (0, 1)
    assert chart.active_optimizer_indices == (0, 1)
    assert chart.reduced_positions_by_logical == {0: (0, 1)}
    assert chart.x0.tolist() == [0.2, 0.2]
    assert chart.lift_to_runtime(np.asarray([0.6, -0.3])).tolist() == [0.15, 0.15, -0.4]


def test_per_pauli_powell_keeps_unprojected_runtime_chart():
    layout = _two_generator_layout()
    runtime_theta = np.asarray([0.2, 0.2, -0.4], dtype=float)

    chart = make_selected_optimizer_chart(
        full_theta=runtime_theta,
        layout=layout,
        active_logical_indices=[0],
        objective=lambda theta: float(np.sum(theta)),
        parameterization_mode="per_pauli_term",
        effective_optimizer_key="POWELL",
        powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
    )

    assert chart.coordinate_mode == "runtime"
    assert chart.lift_to_runtime(np.asarray([0.6, -0.3])).tolist() == [0.6, -0.3, -0.4]


@pytest.mark.parametrize("optimizer_key", ["POWELL", "SPSA", "ROTOSOLVE"])
def test_logical_shared_chart_rejects_nonuniform_runtime_block_instead_of_averaging(
    optimizer_key: str,
):
    layout = _two_generator_layout()

    with pytest.raises(ValueError, match="refusing to average nonuniform block"):
        make_selected_optimizer_chart(
            full_theta=np.asarray([0.2, 0.3, -0.4], dtype=float),
            layout=layout,
            active_logical_indices=[0, 1],
            objective=lambda theta: float(np.sum(theta)),
            parameterization_mode="logical_shared",
            effective_optimizer_key=optimizer_key,
            powell_coordinate_chart_policy=(
                SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
            ),
        )


def test_optimizer_chart_rejects_duplicate_or_out_of_range_logical_indices():
    layout = _two_generator_layout()
    kwargs = {
        "full_theta": np.asarray([0.2, 0.2, -0.4], dtype=float),
        "layout": layout,
        "objective": lambda theta: float(np.sum(theta)),
        "parameterization_mode": "logical_shared",
        "effective_optimizer_key": "POWELL",
        "powell_coordinate_chart_policy": (
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
    }

    with pytest.raises(ValueError, match="must be unique"):
        make_selected_optimizer_chart(active_logical_indices=[0, 0], **kwargs)
    with pytest.raises(ValueError, match="must be in range"):
        make_selected_optimizer_chart(active_logical_indices=[2], **kwargs)


def test_optimizer_chart_rejects_unknown_powell_policy():
    layout = _two_generator_layout()
    with pytest.raises(ValueError, match="powell_coordinate_chart_policy"):
        make_selected_optimizer_chart(
            full_theta=np.asarray([0.2, 0.2, -0.4], dtype=float),
            layout=layout,
            active_logical_indices=[0],
            objective=lambda theta: float(np.sum(theta)),
            parameterization_mode="logical_shared",
            effective_optimizer_key="POWELL",
            powell_coordinate_chart_policy="unknown_chart",
        )


def test_sr_active_only_step_maps_by_active_logical_order_and_exactly_guards():
    layout = _two_generator_layout()
    runtime_theta = np.asarray([0.2, 0.2, -0.4], dtype=float)
    chart = make_selected_optimizer_chart(
        full_theta=runtime_theta,
        layout=layout,
        active_logical_indices=[1, 0],
        objective=lambda theta: float(np.dot(theta, theta)),
        parameterization_mode="logical_shared",
        effective_optimizer_key="POWELL",
        powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
    )

    seed, payload, nfev = _guard_sr_active_only_step(
        chart=chart,
        active_parameter_step=[0.2, -0.1],
    )

    assert nfev == 2
    assert seed.tolist() == pytest.approx([-0.2, 0.1])
    assert payload["status"] == "accepted"
    assert payload["active_logical_indices"] == [1, 0]
    assert payload["active_reduced_position_groups"] == [[0], [1]]
    assert payload["mapped_seed_materially_downhill"] is True
    assert payload["mapped_seed_energy"] == pytest.approx(0.06)
    assert payload["incumbent_energy"] == pytest.approx(0.24)


def test_sr_active_only_atomic_pair_selects_downhill_reflection() -> None:
    layout = _two_generator_layout()
    chart = make_selected_optimizer_chart(
        full_theta=np.asarray([0.2, 0.2, -0.4], dtype=float),
        layout=layout,
        active_logical_indices=[1, 0],
        objective=lambda theta: float(np.dot(theta, theta)),
        parameterization_mode="logical_shared",
        effective_optimizer_key="POWELL",
        powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
    )

    seed, payload, nfev = _guard_sr_active_only_step(
        chart=chart,
        active_parameter_step=[-0.2, 0.0],
        retained_joint_step_candidates=[[-0.2, 0.0], [0.2, 0.0]],
        retained_candidate_predicted_reductions=[0.1, 0.1],
        retained_candidate_roles=["plus_v", "minus_v"],
    )

    assert nfev == 3
    assert payload["status"] == "accepted"
    assert payload["selected_candidate_index"] == 1
    assert payload["selected_candidate_role"] == "minus_v"
    assert seed.tolist() == pytest.approx([-0.2, 0.2])
    evaluations = payload["candidate_evaluations"]
    assert evaluations[0]["materially_downhill"] is False
    assert evaluations[1]["materially_downhill"] is True


def _one_coordinate_chart(
    *,
    objective,
    coordinate_scale: float = 1.0,
) -> SelectedOptimizerChart:
    scale = float(coordinate_scale)
    return SelectedOptimizerChart(
        objective=lambda x: float(
            objective(scale * float(np.asarray(x, dtype=float)[0]))
        ),
        x0=np.asarray([0.0], dtype=float),
        lift_to_runtime=lambda x: np.asarray(x, dtype=float),
        coordinate_mode="logical_shared",
        active_logical_indices=(0,),
        active_runtime_indices=(0,),
        active_optimizer_indices=(0,),
        reduced_positions_by_logical={0: (0,)},
    )


def _one_coordinate_state_service(*, coordinate_scale: float = 1.0):
    scale = float(coordinate_scale)

    def _state(x: np.ndarray) -> np.ndarray:
        q = scale * float(np.asarray(x, dtype=float)[0])
        return np.asarray([np.cos(q), np.sin(q)], dtype=complex)

    return _state


def test_sr_active_only_backtracks_to_largest_downhill_in_radius_dyadic_fraction():
    chart = _one_coordinate_chart(
        objective=lambda q: float((q - 0.125) ** 2),
    )

    seed, payload, nfev = _guard_sr_active_only_step(
        chart=chart,
        active_parameter_step=[1.0],
        retained_candidate_predicted_reductions=[0.0],
        nonlinear_backtracking_enabled=True,
        state_at_optimizer_x=_one_coordinate_state_service(),
        incumbent_state=np.asarray([1.0, 0.0], dtype=complex),
        max_exact_fubini_study_distance=0.2,
        quadratic_model_gradient=[1.0],
        quadratic_model_hessian=[[2.0]],
        backtracking_max_halvings=8,
    )

    assert nfev == 5
    assert payload["status"] == "accepted"
    assert payload["backtracking_halvings"] == 3
    assert payload["applied_joint_step_scale"] == pytest.approx(0.125)
    assert seed.tolist() == pytest.approx([0.125])
    assert payload["mapped_seed_exact_endpoint_distance"] == pytest.approx(
        0.125
    )
    expected_prediction = 0.125 - 0.125**2
    assert payload["mapped_seed_predicted_reduction"] == pytest.approx(
        expected_prediction
    )
    assert payload["mapped_seed_predicted_reduction"] != pytest.approx(
        0.125 * 0.0
    )
    attempts = payload["backtracking_attempts"]
    assert [row["joint_step_scale"] for row in attempts] == pytest.approx(
        [1.0, 0.5, 0.25, 0.125]
    )
    assert all(row["status"] == "rejected" for row in attempts[:3])


def test_sr_active_only_exact_fs_gate_rejects_downhill_out_of_radius_scales():
    chart = _one_coordinate_chart(objective=lambda q: float(-q))

    seed, payload, _nfev = _guard_sr_active_only_step(
        chart=chart,
        active_parameter_step=[1.0],
        retained_candidate_predicted_reductions=[1.0],
        nonlinear_backtracking_enabled=True,
        state_at_optimizer_x=_one_coordinate_state_service(),
        incumbent_state=np.asarray([1.0, 0.0], dtype=complex),
        max_exact_fubini_study_distance=0.3,
        quadratic_model_gradient=[1.0],
        quadratic_model_hessian=[[0.0]],
        backtracking_max_halvings=4,
    )

    assert payload["status"] == "accepted"
    assert payload["backtracking_halvings"] == 2
    assert seed.tolist() == pytest.approx([0.25])
    assert payload["mapped_seed_exact_endpoint_distance"] == pytest.approx(
        0.25
    )
    attempts = payload["backtracking_attempts"]
    assert [row["reason"] for row in attempts[:2]] == [
        "all_materially_downhill_candidates_exceed_exact_endpoint_distance_budget",
        "all_materially_downhill_candidates_exceed_exact_endpoint_distance_budget",
    ]
    assert all(
        row["candidate_evaluations"][0]["materially_downhill"] is True
        and row["candidate_evaluations"][0][
            "endpoint_distance_within_budget"
        ]
        is False
        for row in attempts[:2]
    )


def test_sr_active_only_backtracking_exhaustion_is_typed_no_state():
    chart = _one_coordinate_chart(objective=lambda q: float(q * q))

    seed, payload, nfev = _guard_sr_active_only_step(
        chart=chart,
        active_parameter_step=[1.0],
        retained_candidate_predicted_reductions=[1.0],
        nonlinear_backtracking_enabled=True,
        state_at_optimizer_x=_one_coordinate_state_service(),
        incumbent_state=np.asarray([1.0, 0.0], dtype=complex),
        max_exact_fubini_study_distance=2.0,
        quadratic_model_gradient=[1.0],
        quadratic_model_hessian=[[0.0]],
        backtracking_max_halvings=3,
    )

    assert nfev == 5
    assert seed.tolist() == pytest.approx([0.0])
    assert payload["status"] == "rejected"
    assert payload["reason"] == "active_only_nonlinear_backtracking_exhausted"
    assert payload["transaction_failure_kind"] == (
        "finite_nonlinear_model_disagreement"
    )
    assert payload["trust_action"] == "contract_branch_radius"
    assert payload["all_backtracking_candidates_finite"] is True
    assert payload["nonlinear_backtracking_exhausted"] is True
    assert payload["no_state_transition"] is True


def test_sr_active_only_backtracking_empty_candidate_set_fails_closed():
    chart = _one_coordinate_chart(objective=lambda q: float(-q))

    seed, payload, nfev = _guard_sr_active_only_step(
        chart=chart,
        active_parameter_step=[1.0],
        retained_joint_step_candidates=[],
        retained_candidate_predicted_reductions=[],
        retained_candidate_roles=[],
        nonlinear_backtracking_enabled=True,
        state_at_optimizer_x=_one_coordinate_state_service(),
        incumbent_state=np.asarray([1.0, 0.0], dtype=complex),
        max_exact_fubini_study_distance=0.3,
        quadratic_model_gradient=[1.0],
        quadratic_model_hessian=[[0.0]],
        backtracking_max_halvings=3,
    )

    assert nfev == 0
    assert seed.tolist() == pytest.approx([0.0])
    assert payload["status"] == "unavailable"
    assert payload["reason"] == (
        "active_only_backtracking_missing_retained_candidates"
    )
    assert payload["transaction_failure_kind"] == "mapping_contract"
    assert payload["nonlinear_backtracking_exhausted"] is False
    assert payload["all_backtracking_candidates_finite"] is False
    assert payload["trust_action"] == "hold"


def test_sr_active_only_backtracking_is_coordinate_rescaling_invariant():
    objective = lambda q: float((q - 0.125) ** 2)
    physical = _guard_sr_active_only_step(
        chart=_one_coordinate_chart(objective=objective),
        active_parameter_step=[1.0],
        nonlinear_backtracking_enabled=True,
        state_at_optimizer_x=_one_coordinate_state_service(),
        incumbent_state=np.asarray([1.0, 0.0], dtype=complex),
        max_exact_fubini_study_distance=0.2,
        quadratic_model_gradient=[1.0],
        quadratic_model_hessian=[[2.0]],
        backtracking_max_halvings=8,
    )
    coordinate_scale = 1.0e3
    rescaled = _guard_sr_active_only_step(
        chart=_one_coordinate_chart(
            objective=objective,
            coordinate_scale=coordinate_scale,
        ),
        active_parameter_step=[1.0 / coordinate_scale],
        nonlinear_backtracking_enabled=True,
        state_at_optimizer_x=_one_coordinate_state_service(
            coordinate_scale=coordinate_scale
        ),
        incumbent_state=np.asarray([1.0, 0.0], dtype=complex),
        max_exact_fubini_study_distance=0.2,
        quadratic_model_gradient=[coordinate_scale],
        quadratic_model_hessian=[[2.0 * coordinate_scale**2]],
        backtracking_max_halvings=8,
    )

    physical_seed, physical_payload, _ = physical
    rescaled_seed, rescaled_payload, _ = rescaled
    assert coordinate_scale * float(rescaled_seed[0]) == pytest.approx(
        float(physical_seed[0])
    )
    assert rescaled_payload["backtracking_halvings"] == (
        physical_payload["backtracking_halvings"]
    )
    assert rescaled_payload["mapped_seed_exact_gain"] == pytest.approx(
        physical_payload["mapped_seed_exact_gain"]
    )
    assert rescaled_payload[
        "mapped_seed_exact_endpoint_distance"
    ] == pytest.approx(
        physical_payload["mapped_seed_exact_endpoint_distance"]
    )
    assert rescaled_payload["mapped_seed_predicted_reduction"] == pytest.approx(
        physical_payload["mapped_seed_predicted_reduction"]
    )


def test_sr_active_only_safe_refit_never_keeps_a_worse_optimizer_result():
    selected_x, selected_energy, payload = _retain_sr_active_only_refit_outcome(
        incumbent_x=np.asarray([0.0]),
        incumbent_energy=-1.0,
        guarded_seed_x=np.asarray([0.2]),
        guarded_seed_energy=-1.1,
        optimizer_x=np.asarray([0.9]),
        optimizer_energy=-0.8,
    )

    assert selected_x.tolist() == pytest.approx([0.2])
    assert selected_energy == pytest.approx(-1.1)
    assert payload["selected_source"] == "mapped_active_restriction_seed"
    assert payload["optimizer_result_discarded"] is True
    assert payload["nonworsening_certified"] is True
