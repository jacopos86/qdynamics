"""The SIAM pilot is executable by the unmodified AP-McLachlan trajectory core.

The point of these tests is that SIAM reaches the controller without a
Hubbard-Holstein seed artifact and without touching HH code paths: the AP core
takes a state and a Hamiltonian, and this module supplies both.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
    run_append_mclachlan_trajectory,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
from pipelines.time_dynamics.ap_mclachlan.support_atoms import candidate_append_atoms
from pipelines.time_dynamics.siam_runtime import (
    SIAM_FAMILY_KEY,
    build_siam_runtime,
    evaluate_screening_observables,
    pool_ansatz_terms,
)
from pipelines.time_dynamics.siam_screening import FIXTURE_NB2, SIAMScreeningConfig
from src.quantum.siam_model import (
    DOWN,
    UP,
    SIAMLayout,
    sector_indices,
    total_number_operator,
)
from src.quantum.siam_reference import exact_reference
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


def _patch_config() -> SupportPatchControllerConfig:
    return SupportPatchControllerConfig(
        insertion_gate_mode="mclachlan_l2",
        insertion_l2_cut=1.0e-3,
        append_schur_max_condition_number=None,
        max_joint_patch_evaluations=50000,
        max_certification_attempts_per_level=12,
    )


def test_runtime_starts_with_no_variational_coordinates() -> None:
    runtime = build_siam_runtime(FIXTURE_NB2)
    assert runtime.state.runtime_parameter_count == 0
    assert np.allclose(runtime.state.psi_initial, runtime.state.psi_ref)


def test_pool_reaches_the_state_as_ansatz_terms() -> None:
    runtime = build_siam_runtime(FIXTURE_NB2)
    assert len(runtime.candidate_pool_terms) == len(pool_ansatz_terms(2)) == 14
    assert runtime.state.can_structural_edit
    assert runtime.state.candidate_pool_source.source_kind == "resolved_pool"
    assert runtime.state.candidate_pool_source.candidate_pool_complete


def test_every_runtime_candidate_preserves_the_declared_spin_sector() -> None:
    """The controller's selectable atoms, not only their parents, conserve spin."""

    runtime = build_siam_runtime(FIXTURE_NB2)
    layout = SIAMLayout(2)
    numbers = {
        spin: np.asarray(
            hamiltonian_matrix(total_number_operator(layout, spin)), dtype=complex
        )
        for spin in (UP, DOWN)
    }
    atoms = candidate_append_atoms(runtime.state)
    assert atoms
    for atom in atoms:
        matrix = np.asarray(hamiltonian_matrix(atom.term.polynomial), dtype=complex)
        for spin, number in numbers.items():
            commutator = matrix @ number - number @ matrix
            assert np.allclose(commutator, 0.0, atol=1.0e-10), (
                f"runtime atom {atom.atom_label!r} breaks N_{spin}"
            )


def test_problem_context_is_not_disguised_as_hubbard_holstein() -> None:
    """Claiming family 'hh' would hand SIAM an observable plan built for HH."""

    runtime = build_siam_runtime(FIXTURE_NB2)
    assert runtime.state.resolved_problem.family_key == SIAM_FAMILY_KEY
    assert runtime.state.resolved_problem.family_key != "hh"


def test_hamiltonian_is_time_independent_after_the_quench() -> None:
    runtime = build_siam_runtime(FIXTURE_NB2)
    assert runtime.hamiltonian.drive_enabled is False
    assert np.allclose(
        runtime.hamiltonian.matrix_at(0.0), runtime.hamiltonian.matrix_at(4.2)
    )


def test_pilot_propagates_through_the_unmodified_controller() -> None:
    """End-to-end: the AP route grows SIAM support and tracks the exact sector."""

    config = SIAMScreeningConfig(
        n_bath=2, u=1.5, t_final=1.0, num_times=26, label="siam_runtime_test"
    )
    runtime = build_siam_runtime(config)
    trajectory = run_append_mclachlan_trajectory(
        state=runtime.state,
        hamiltonian=runtime.hamiltonian,
        times=runtime.times,
        integrator_method="rk4",
        support_patch_config=_patch_config(),
        solve_repair_config=SolveRepairConfig.minimal_profile(),
    )
    points = trajectory.points
    assert len(points) == 26
    # the route must actually grow support from the SIAM pool
    assert runtime.state.runtime_parameter_count == 0
    assert points[-1].runtime_parameter_count > runtime.state.runtime_parameter_count

    reference = exact_reference(2, u=1.5, times=runtime.times)
    errors = [
        abs(
            evaluate_screening_observables(runtime, point.geometry.psi)["m_d"]
            - reference.at("m_d")[index]
        )
        for index, point in enumerate(points)
    ]
    assert float(np.mean(errors)) < 5.0e-2, "screening observable drifted from exact"
    target_indices = np.asarray(sector_indices(2), dtype=int)
    sector_weights = [
        float(
            np.sum(
                np.abs(np.asarray(point.geometry.psi)[target_indices]) ** 2
            )
        )
        for point in points
    ]
    assert min(sector_weights) == pytest.approx(1.0, abs=1.0e-10)


def test_observables_at_the_prepared_state_are_the_decoupled_values() -> None:
    runtime = build_siam_runtime(FIXTURE_NB2)
    values = evaluate_screening_observables(runtime, runtime.state.psi_initial)
    assert values["m_d"] == pytest.approx(1.0, abs=1e-12)
    assert values["D_d"] == pytest.approx(0.0, abs=1e-12)
    assert values["C_1"] == pytest.approx(0.0, abs=1e-12)
