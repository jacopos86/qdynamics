from __future__ import annotations

import json

import numpy as np
import pytest

from pipelines.scaffold.hh_continuation_stage_control import (
    StageControllerConfig,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    EXACT_ORDERED_INSERTION_CHART,
)
from pipelines.static_adapt.ra_adapt.insertion_geometry import (
    InsertionGeometryRequest,
    enumerate_actual_insertion_positions,
    enumerate_candidate_position_plans,
    evaluate_exact_insertion_first_order,
    evaluate_exact_insertion_joint_geometry,
    prepare_exact_insertion_first_order_context,
    prepare_exact_insertion_joint_context,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _term(
    label: str,
    *terms: tuple[str, float],
) -> AnsatzTerm:
    return AnsatzTerm(
        label=label,
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps=word, pc=coefficient)
                for word, coefficient in terms
            ],
        ),
    )


def _normalized_random_state(*, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = np.asarray(
        rng.normal(size=4) + 1.0j * rng.normal(size=4),
        dtype=complex,
    )
    return state / np.linalg.norm(state)


def _case(
    representation: str,
) -> tuple[
    list[AnsatzTerm],
    AnsatzTerm,
    np.ndarray,
    np.ndarray,
    object,
]:
    selected = [
        _term("selected-macro", ("xe", 0.61), ("ze", -0.27)),
        _term("selected-singleton", ("ey", 0.49)),
    ]
    candidate = (
        _term("candidate-macro", ("xy", 0.52), ("zx", -0.31))
        if representation == CANDIDATE_REPRESENTATION_MACRO
        else _term("candidate-singleton", ("zx", -0.31))
    )
    theta = np.asarray([0.23, -0.19], dtype=float)
    psi_ref = _normalized_random_state(seed=20260727)
    h_compiled = compile_polynomial_action(
        PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps="ze", pc=0.73),
                PauliTerm(2, ps="ex", pc=-0.41),
                PauliTerm(2, ps="yy", pc=0.29),
                PauliTerm(2, ps="xz", pc=-0.18),
            ],
        ),
        pauli_action_cache={},
    )
    return selected, candidate, theta, psi_ref, h_compiled


def _finite_difference_gradient(
    energy,
    point: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    out = np.zeros(point.size, dtype=float)
    for index in range(point.size):
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        out[index] = (energy(plus) - energy(minus)) / (2.0 * step)
    return out


def _finite_difference_hessian(
    energy,
    point: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    dimension = int(point.size)
    out = np.zeros((dimension, dimension), dtype=float)
    energy_zero = float(energy(point))
    for row in range(dimension):
        plus = point.copy()
        minus = point.copy()
        plus[row] += step
        minus[row] -= step
        out[row, row] = (
            energy(plus) - 2.0 * energy_zero + energy(minus)
        ) / (step * step)
        for col in range(row + 1, dimension):
            plus_plus = point.copy()
            plus_minus = point.copy()
            minus_plus = point.copy()
            minus_minus = point.copy()
            plus_plus[[row, col]] += step
            plus_minus[row] += step
            plus_minus[col] -= step
            minus_plus[row] -= step
            minus_plus[col] += step
            minus_minus[[row, col]] -= step
            value = (
                energy(plus_plus)
                - energy(plus_minus)
                - energy(minus_plus)
                + energy(minus_minus)
            ) / (4.0 * step * step)
            out[row, col] = value
            out[col, row] = value
    return out


@pytest.mark.parametrize(
    "representation",
    [
        CANDIDATE_REPRESENTATION_MACRO,
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    ],
)
@pytest.mark.parametrize(
    "position_kind,insertion_position",
    [("interior", 1), ("append", 2)],
)
def test_actual_position_gradient_and_hessian_match_finite_differences(
    representation: str,
    position_kind: str,
    insertion_position: int,
) -> None:
    selected, candidate, theta, psi_ref, h_compiled = _case(representation)
    selected_executor = CompiledAnsatzExecutor(selected)
    psi_state = selected_executor.prepare_state(theta, psi_ref)
    hpsi_state = apply_compiled_polynomial(psi_state, h_compiled)
    request = InsertionGeometryRequest(
        candidate_term=candidate,
        insertion_position=insertion_position,
        candidate_representation=representation,
    )

    first_context = prepare_exact_insertion_first_order_context(
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        hpsi_state=hpsi_state,
    )
    first_order = evaluate_exact_insertion_first_order(
        context=first_context,
        request=request,
    )
    joint_context = prepare_exact_insertion_joint_context(
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        active_indices=range(len(selected)),
        h_compiled=h_compiled,
    )
    joint = evaluate_exact_insertion_joint_geometry(
        context=joint_context,
        request=request,
        h_compiled=h_compiled,
    )

    combined_terms = [
        *selected[:insertion_position],
        candidate,
        *selected[insertion_position:],
    ]
    combined_executor = CompiledAnsatzExecutor(combined_terms)
    source_plus_candidate = np.concatenate((theta, [0.0]))

    def energy(source_chart: np.ndarray) -> float:
        ordered_theta = np.insert(
            np.asarray(source_chart[:-1], dtype=float),
            insertion_position,
            float(source_chart[-1]),
        )
        state = combined_executor.prepare_state(ordered_theta, psi_ref)
        value, _ = energy_via_one_apply(state, h_compiled)
        return float(value)

    gradient_fd = _finite_difference_gradient(
        energy,
        source_plus_candidate,
        step=2.0e-7,
    )
    hessian_fd = _finite_difference_hessian(
        energy,
        source_plus_candidate,
        step=1.0e-4,
    )

    payload = joint.payload
    gradient_descent = np.asarray(
        [*payload["g_A"], payload["descent_gradient"]],
        dtype=float,
    )
    hessian = np.block(
        [
            [
                np.asarray(payload["H_AA"], dtype=float),
                np.asarray(payload["H_AB"], dtype=float)[:, None],
            ],
            [
                np.asarray(payload["H_AB"], dtype=float)[None, :],
                np.asarray([[payload["H_BB"]]], dtype=float),
            ],
        ]
    )

    assert request.coordinate_chart == EXACT_ORDERED_INSERTION_CHART
    assert first_order.coordinate_chart == EXACT_ORDERED_INSERTION_CHART
    assert joint.coordinate_chart == EXACT_ORDERED_INSERTION_CHART
    assert first_order.insertion_position == insertion_position
    assert joint.insertion_position == insertion_position
    assert first_order.payload["candidate_position_id"] == insertion_position
    assert joint.payload["candidate_position_id"] == insertion_position
    assert first_order.payload["energy_gradient"] == pytest.approx(
        gradient_fd[-1],
        abs=5.0e-9,
    )
    assert first_order.payload["energy_gradient"] == pytest.approx(
        -float(payload["descent_gradient"]),
        abs=2.0e-12,
    )
    assert first_order.payload["fubini_study_metric"] == pytest.approx(
        float(payload["G_BB"]),
        abs=2.0e-12,
    )
    np.testing.assert_allclose(
        gradient_descent,
        -gradient_fd,
        atol=7.0e-9,
        rtol=7.0e-9,
    )
    np.testing.assert_allclose(
        hessian,
        hessian_fd,
        atol=8.0e-7,
        rtol=8.0e-7,
    )
    assert position_kind == (
        "append" if insertion_position == len(selected) else "interior"
    )
    json.dumps(first_order.as_dict(), allow_nan=False, sort_keys=True)
    json.dumps(joint.as_dict(), allow_nan=False, sort_keys=True)


def test_position_enumeration_owns_full_domain_and_candidate_reduction() -> None:
    selected, candidate, theta, _psi_ref, _h_compiled = _case(
        CANDIDATE_REPRESENTATION_MACRO
    )
    domain = enumerate_actual_insertion_positions(
        insertion_mode="full_commutation_reduced",
        append_eval={},
        append_position=len(selected),
        n_params=theta.size,
        active_window_indices=range(theta.size),
        stage_name="core",
        drop_plateau_hits=0,
        max_grad=0.3,
        eps_grad=1.0e-8,
        finite_angle_fallback=False,
        repeated_family_flat=False,
        stage_controller_config=StageControllerConfig(),
    )
    plans = enumerate_candidate_position_plans(
        pool=[candidate],
        candidate_indices=[0],
        selected_ops=selected,
        domain=domain,
    )

    assert domain.positions == (0, 1, 2)
    assert domain.append_position == 2
    assert domain.coordinate_chart == EXACT_ORDERED_INSERTION_CHART
    assert plans[0]["coordinate_chart"] == EXACT_ORDERED_INSERTION_CHART
    assert plans[0]["candidate_pool_index"] == 0
    assert plans[0]["append_position"] == 2
    assert 2 in plans[0]["requested_positions"]
    assert set(plans[0]["representative_positions"]).issubset(
        set(domain.positions)
    )
    json.dumps(domain.as_dict(), allow_nan=False, sort_keys=True)
    json.dumps(plans, allow_nan=False, sort_keys=True)


def test_geometry_request_rejects_chart_or_position_drift() -> None:
    _selected, candidate, _theta, _psi_ref, _h_compiled = _case(
        CANDIDATE_REPRESENTATION_SINGLE_PAULI
    )
    with pytest.raises(ValueError, match="requires"):
        InsertionGeometryRequest(
            candidate_term=candidate,
            insertion_position=0,
            candidate_representation=(
                CANDIDATE_REPRESENTATION_SINGLE_PAULI
            ),
            coordinate_chart="append_candidate_after_current_ansatz_v1",
        )
    with pytest.raises(ValueError, match="nonnegative"):
        InsertionGeometryRequest(
            candidate_term=candidate,
            insertion_position=-1,
            candidate_representation=(
                CANDIDATE_REPRESENTATION_SINGLE_PAULI
            ),
        )
