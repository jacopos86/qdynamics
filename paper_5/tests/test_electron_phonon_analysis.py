from __future__ import annotations

import numpy as np

from paper5.stability import (
    DimerParameters,
    exact_ground_closed_scalar_coordinates,
)
from paper5.stability.electron_phonon_analysis import (
    analyze_pauli_repair_ablation,
    analyze_matched_case,
    integrate_closed_rk4,
)


def test_decoupled_matched_case_has_zero_model_error() -> None:
    parameters = DimerParameters(
        lambda_ep=0.0,
        gamma=0.5,
        drive_amplitude=0.0,
    )
    case = analyze_matched_case(
        parameters,
        final_time=0.1,
        time_step=0.05,
        phonon_cutoff=2,
        activation_margin=0.0,
        exact_relative_tolerance=1e-10,
        exact_absolute_tolerance=1e-12,
        exact_maximum_step=0.05,
        include_exact_defect=True,
        controller_stride=1,
    )

    assert case.exact_coordinates.shape == (3, 31)
    assert np.max(np.abs(case.raw.coordinates - case.exact_coordinates)) < 1e-10
    assert (
        np.max(np.abs(case.corrected.coordinates - case.exact_coordinates))
        < 1e-10
    )
    np.testing.assert_allclose(
        case.corrected.correction_coordinates,
        0.0,
        atol=1e-28,
    )
    assert case.metrics["exact_derivative_defect"]["block_defect_norms"]["C"][
        "maximum"
    ] < 1e-10


def test_corrected_fixed_step_trajectory_records_joint_diagnostics() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=4,
    )
    trajectory = integrate_closed_rk4(
        parameters,
        initial,
        final_time=0.02,
        time_step=0.01,
        corrected=True,
        activation_margin=1e-5,
        cone_tolerance=1e-8,
    )

    assert trajectory.coordinates.shape == (3, 31)
    assert trajectory.correction_coordinates.shape == (3, 31)
    assert trajectory.raw_barrier_minima.shape == (3, 3)
    assert trajectory.joint_mode_weights.shape == (3, 7)
    np.testing.assert_allclose(
        np.sum(trajectory.joint_mode_weights, axis=1),
        1.0,
        atol=2e-12,
    )
    assert np.max(np.linalg.norm(trajectory.correction_coordinates, axis=1)) > 0.0
    assert np.min(trajectory.corrected_barrier_minima) > -1e-8
    assert trajectory.integration_rhs_evaluations == 8


def test_decoupled_pauli_ablation_has_four_identical_lanes() -> None:
    case = analyze_pauli_repair_ablation(
        DimerParameters(
            lambda_ep=0.0,
            gamma=0.5,
            drive_amplitude=0.0,
        ),
        final_time=0.1,
        time_step=0.05,
        phonon_cutoff=2,
        activation_margin=0.0,
        exact_relative_tolerance=1e-10,
        exact_absolute_tolerance=1e-12,
        exact_maximum_step=0.05,
    )

    for trajectory in (
        case.raw,
        case.pauli_repaired,
        case.controller,
        case.pauli_repaired_controller,
    ):
        np.testing.assert_allclose(
            trajectory.coordinates,
            case.exact_coordinates,
            atol=1e-10,
        )
    assert (
        case.metrics["exact_sample_C_derivative_defect"]["pauli_repaired"]
        ["residual_subtracted_time_rms_l2"]
        < 1e-10
    )
