from __future__ import annotations

import numpy as np

from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.mixed_enriched_propagation import (
    admit_mixed_guided_packets,
    archive_gram_admission_signals,
    mixed_enriched_euler_step,
    mixed_enriched_midpoint_step,
    normalized_packet_state,
    project_mixed_enriched_velocity,
)
from paper5.stability.multi_coherent import pack_multi_coherent_parameters


def _symmetric_parameters() -> np.ndarray:
    coefficients = np.asarray(
        [[0.55], [0.32], [0.32], [0.45]],
        dtype=complex,
    )
    displacements = np.asarray(
        [[0.10 + 0.05j], [0.18 - 0.07j], [0.18 - 0.07j], [-0.08j]],
        dtype=complex,
    )
    return pack_multi_coherent_parameters(coefficients, displacements)


def test_enriched_projection_and_retraction_have_declared_local_velocity() -> None:
    packet_parameters = _symmetric_parameters()
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    drive = GaussianSineDrive(amplitude=1.0, delays=(0.0, 8.0))
    relative_dimension = 15
    projection = project_mixed_enriched_velocity(
        0.4,
        packet_parameters,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive,
    )
    assert projection.enriched_relative_residual <= (
        projection.native_relative_residual + 1e-10
    )

    step_size = 2e-6
    step = mixed_enriched_euler_step(
        0.4,
        packet_parameters,
        step_size,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive,
        retraction_relative_tolerance=1e-12,
    )
    endpoint = normalized_packet_state(
        step.parameters,
        relative_dimension=relative_dimension,
    )
    overlap = np.vdot(projection.state, endpoint)
    endpoint *= np.exp(-1j * np.angle(overlap))
    realized_velocity = (endpoint - projection.state) / step_size
    realized_velocity -= projection.state * np.vdot(
        projection.state,
        realized_velocity,
    )
    relative_error = np.linalg.norm(
        realized_velocity - projection.projected_velocity
    ) / np.linalg.norm(projection.projected_velocity)

    assert step.retraction_state_error < 2e-11
    assert step.retraction_fidelity > 1.0 - 1e-12
    assert relative_error < 2e-3


def test_mixed_enriched_midpoint_has_second_order_self_convergence() -> None:
    initial = _symmetric_parameters()
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    drive = GaussianSineDrive(amplitude=1.0, delays=(0.0, 8.0))
    relative_dimension = 15

    def propagate(time_step: float) -> np.ndarray:
        values = initial.copy()
        step_count = int(round(0.02 / time_step))
        for step_index in range(step_count):
            values = mixed_enriched_midpoint_step(
                step_index * time_step,
                values,
                time_step,
                parameters,
                relative_dimension=relative_dimension,
                drive_protocol=drive,
                retraction_relative_tolerance=1e-11,
            ).parameters
        return normalized_packet_state(
            values,
            relative_dimension=relative_dimension,
        )

    coarse = propagate(0.01)
    medium = propagate(0.005)
    fine = propagate(0.0025)

    def aligned_error(candidate: np.ndarray) -> float:
        aligned = candidate * np.exp(-1j * np.angle(np.vdot(fine, candidate)))
        return float(np.linalg.norm(aligned - fine))

    assert aligned_error(coarse) > 3.0 * aligned_error(medium)


def test_mixed_guided_packet_admission_preserves_state_and_adds_capacity() -> None:
    initial = _symmetric_parameters()
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    drive = GaussianSineDrive(amplitude=1.0, delays=(0.0, 8.0))
    admission = admit_mixed_guided_packets(
        0.0,
        initial,
        parameters,
        relative_dimension=15,
        drive_protocol=drive,
        fit_maximum_iterations=8,
        fit_population_size=4,
        fit_seed=123,
    )

    assert admission.previous_packet_count == 1
    assert admission.packet_count == 2
    assert admission.state_discontinuity < 1e-14
    assert admission.mixed_gain_norm_before > 0.0
    assert admission.fitted_centers[1] == admission.fitted_centers[2]


def test_archive_gram_admission_signals_score_current_state_only() -> None:
    initial = _symmetric_parameters()
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    drive = GaussianSineDrive(amplitude=1.0, delays=(0.0, 8.0))
    keyword_arguments = {
        "relative_dimension": 15,
        "coordinate_scales": np.ones(31),
        "drive_protocol": drive,
    }
    before = archive_gram_admission_signals(
        0.4,
        initial,
        parameters,
        **keyword_arguments,
    )
    admission = admit_mixed_guided_packets(
        0.4,
        initial,
        parameters,
        relative_dimension=15,
        drive_protocol=drive,
        fit_maximum_iterations=8,
        fit_population_size=4,
        fit_seed=123,
    )
    after = archive_gram_admission_signals(
        0.4,
        admission.parameters,
        parameters,
        **keyword_arguments,
    )

    assert before.native_hilbert_residual_squared > 0.0
    assert before.joint_gram_rate_defect_squared > 0.0
    assert before.mixed_observable_impact_squared > 0.0
    assert after.native_hilbert_residual_squared < (
        before.native_hilbert_residual_squared
    )
    assert after.joint_gram_rate_defect_squared < (
        before.joint_gram_rate_defect_squared
    )
    assert after.mixed_observable_impact_squared < (
        before.mixed_observable_impact_squared
    )
    assert after.native_geometric_rank > before.native_geometric_rank
    assert before.minimum_joint_gram_eigenvalue > -1e-12
