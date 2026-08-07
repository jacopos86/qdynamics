from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.integrators import INTEGRATOR_EULER, INTEGRATOR_RK4
from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
from pipelines.time_dynamics.ap_mclachlan.trajectory import run_fixed_mclachlan_trajectory
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(label: str, coeff: float = 1.0) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(1, ps=str(label), pc=float(coeff)))
    poly._reduce()
    return poly


def _state_and_hamiltonian() -> tuple[object, TimeDependentHamiltonian]:
    selected = (AnsatzTerm(label="seed_x", polynomial=_poly("x")),)
    layout = build_parameter_layout(selected)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("x")),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        psi_initial=np.array([1.0, 0.0], dtype=complex),
        base_layout=layout,
        theta_runtime=np.array([0.0], dtype=float),
        theta_logical=np.array([0.0], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=selected,
        provenance={"artifact_json": "toy.json"},
    )
    return (
        state_from_scaffold_runtime_input(runtime_input),
        TimeDependentHamiltonian(static_poly=runtime_input.h_poly),
    )


def test_fixed_trajectory_propagates_exact_x_manifold_with_euler() -> None:
    state, hamiltonian = _state_and_hamiltonian()

    trajectory = run_fixed_mclachlan_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=(0.0, 0.1, 0.2),
        integrator_method=INTEGRATOR_EULER,
    )

    assert len(trajectory.points) == 3
    np.testing.assert_allclose(trajectory.final_theta_runtime, np.array([0.2]), atol=1.0e-12)
    assert trajectory.points[0].fixed_step.theta_dot[0] == pytest.approx(1.0)
    assert trajectory.points[0].integration_to_next is not None
    assert trajectory.points[0].integration_to_next.rhs_evaluation_count == 1
    assert trajectory.to_json_dict()["metadata"]["uses_reference_for_decision"] is False
    assert trajectory.to_json_dict()["metadata"]["uses_exact_reference_for_decision"] is False
    assert trajectory.to_json_dict()["metadata"]["uses_future_exact_forecast_for_decision"] is False
    assert trajectory.to_json_dict()["metadata"]["uses_statevector_as_ideal_observable_estimator"] is True


def test_fixed_trajectory_uses_rk4_rhs_evaluations() -> None:
    state, hamiltonian = _state_and_hamiltonian()

    trajectory = run_fixed_mclachlan_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=(0.0, 0.25),
        integrator_method=INTEGRATOR_RK4,
    )

    np.testing.assert_allclose(trajectory.final_theta_runtime, np.array([0.25]), atol=1.0e-12)
    assert trajectory.points[0].integration_to_next is not None
    assert trajectory.points[0].integration_to_next.rhs_evaluation_count == 4


def test_fixed_trajectory_rejects_decreasing_time_grid() -> None:
    state, hamiltonian = _state_and_hamiltonian()

    with pytest.raises(ValueError, match="monotonically"):
        run_fixed_mclachlan_trajectory(
            state=state,
            hamiltonian=hamiltonian,
            times=(0.0, -0.1),
        )
