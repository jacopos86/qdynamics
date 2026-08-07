from __future__ import annotations

import numpy as np
import pytest

from paper5.stability.archive_auxiliary_memory import (
    ArchiveAuxiliaryFrame,
    build_archive_auxiliary_frame,
    propagate_archive_auxiliary_rk4,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
    _ground_state,
)
from paper5.stability.initial_conditions import (
    exact_ground_closed_scalar_coordinates,
)
from paper5.stability.krylov_memory_closure import (
    build_krylov_closure_construction,
    orthonormal_to_closed_coordinates,
)
from paper5.stability.matrix_reference import (
    matrix_state_to_closed_scalar_coordinates,
    pauli_repaired_closed_scalar_rhs,
)


class _FixedDriveParameters:
    def __init__(self, parameters: DimerParameters, drive_value: float) -> None:
        self._parameters = parameters
        self._drive_value = drive_value

    def __getattr__(self, name: str) -> object:
        return getattr(self._parameters, name)

    def drive_difference(self, time: float) -> float:
        del time
        return self._drive_value


def _archive_field(parameters: DimerParameters):
    def field(state: np.ndarray, drive_value: float) -> np.ndarray:
        return pauli_repaired_closed_scalar_rhs(
            0.0,
            state,
            _FixedDriveParameters(parameters, drive_value),  # type: ignore[arg-type]
        )

    return field


@pytest.fixture(scope="module")
def frame() -> ArchiveAuxiliaryFrame:
    construction = build_krylov_closure_construction(
        DimerParameters(lambda_ep=1.5, gamma=0.5),
        phonon_cutoff=2,
        shell_count=1,
        rank_tolerance=1e-11,
    )
    return build_archive_auxiliary_frame(construction, order=1)


def test_component_blocks_are_skew_and_reciprocal(
    frame: ArchiveAuxiliaryFrame,
) -> None:
    for blocks in (frame.static_blocks, frame.drive_blocks):
        assert blocks.resolved_skew_residual < 3e-13
        assert blocks.hidden_skew_residual < 3e-13
        assert blocks.reciprocity_residual < 3e-13

    prefix = frame.prefix(7)
    assert prefix.hidden_dimension == 7
    np.testing.assert_allclose(
        prefix.static_blocks.resolved_hidden,
        frame.static_blocks.resolved_hidden[:, :7],
    )
    assert prefix.static_blocks.reciprocity_residual < 3e-13


def test_orthogonal_projection_preserves_reciprocity_and_initialization(
    frame: ArchiveAuxiliaryFrame,
) -> None:
    rng = np.random.default_rng(2026080501)
    proposal = rng.normal(size=(frame.hidden_dimension, 6))
    basis, _ = np.linalg.qr(proposal, mode="reduced")
    projected = frame.orthogonal_projection(basis)

    assert projected.hidden_dimension == 6
    assert not projected.has_operator_frame
    for blocks in (projected.static_blocks, projected.drive_blocks):
        assert blocks.hidden_skew_residual < 3e-13
        assert blocks.reciprocity_residual < 3e-13

    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    model = _build_exact_dimer_model(parameters, phonon_cutoff=2)
    _, state_vector = _ground_state(model, eigensolver_tolerance=1e-12)
    closed = matrix_state_to_closed_scalar_coordinates(
        _contract_matrix_state(model, state_vector)
    )
    parent_hidden = frame.contract_hidden_state(state_vector)
    initial = projected.initialize_memory_from_hidden(
        closed,
        basis.T @ parent_hidden,
        _archive_field(parameters),
        drive_value=0.0,
    )

    np.testing.assert_allclose(
        initial.section.hidden_section + initial.memory_coordinates,
        basis.T @ parent_hidden,
        atol=3e-14,
        rtol=3e-14,
    )
    with pytest.raises(RuntimeError, match="parent frame"):
        projected.contract_hidden_state(state_vector)


def test_archive_velocity_has_a_full_rank_raw_lift(
    frame: ArchiveAuxiliaryFrame,
) -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    time = 0.73
    state = exact_ground_closed_scalar_coordinates(parameters, phonon_cutoff=2)
    velocity = pauli_repaired_closed_scalar_rhs(time, state, parameters)
    certificate = frame.section(
        state,
        velocity,
        drive_value=parameters.drive_difference(time),
    )

    assert certificate.raw_lift_rank == 29
    assert certificate.raw_lift_relative_residual < 2e-13
    assert certificate.coupling_rank <= frame.hidden_dimension
    assert certificate.section_relative_residual < 2e-13


def test_first_reciprocal_range_contains_archive_field_on_sampled_chart(
    frame: ArchiveAuxiliaryFrame,
) -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    rng = np.random.default_rng(2026080402)
    for _ in range(12):
        retained = rng.normal(scale=0.15, size=29)
        state = orthonormal_to_closed_coordinates(frame.raw_basis, retained)
        time = float(rng.uniform(0.0, 3.0))
        certificate = frame.section(
            state,
            pauli_repaired_closed_scalar_rhs(time, state, parameters),
            drive_value=parameters.drive_difference(time),
        )

        assert certificate.raw_lift_relative_residual < 2e-13
        assert certificate.centered_section_relative_residual < 2e-13


def test_minimum_norm_section_reconstructs_an_in_range_source(
    frame: ArchiveAuxiliaryFrame,
) -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    state = exact_ground_closed_scalar_coordinates(parameters, phonon_cutoff=2)
    baseline = frame.section(
        state,
        pauli_repaired_closed_scalar_rhs(0.0, state, parameters),
        drive_value=0.0,
    )
    jacobian = baseline.centering_jacobian
    blocks = frame.blocks(drive_value=0.31)
    hidden = np.linspace(-0.1, 0.1, frame.hidden_dimension)
    synthetic_velocity = jacobian @ (
        blocks.resolved_resolved @ baseline.retained_coordinates
        + blocks.resolved_hidden @ hidden
    )
    certificate = frame.section(
        state,
        synthetic_velocity,
        drive_value=0.31,
    )

    assert certificate.section_relative_residual < 2e-13
    assert certificate.centered_section_relative_residual < 2e-13
    np.testing.assert_allclose(
        blocks.resolved_hidden @ certificate.hidden_section,
        blocks.resolved_hidden @ hidden,
        atol=3e-13,
        rtol=3e-13,
    )


def test_reciprocal_flow_preserves_projected_hs_norm(
    frame: ArchiveAuxiliaryFrame,
) -> None:
    rng = np.random.default_rng(2026080401)
    retained = rng.normal(size=29)
    hidden = rng.normal(size=frame.hidden_dimension)

    assert abs(
        frame.lossless_exchange_rate(
            retained,
            hidden,
            drive_value=0.47,
        )
    ) < 2e-12


def test_archive_relative_memory_is_preparation_initialized_and_autonomous(
    frame: ArchiveAuxiliaryFrame,
) -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    model = _build_exact_dimer_model(parameters, phonon_cutoff=2)
    _, state_vector = _ground_state(model, eigensolver_tolerance=1e-12)
    closed = matrix_state_to_closed_scalar_coordinates(
        _contract_matrix_state(model, state_vector)
    )
    field = _archive_field(parameters)
    initial = frame.initialize_memory(
        closed,
        state_vector,
        field,
        drive_value=0.0,
    )

    np.testing.assert_allclose(
        initial.section.hidden_section + initial.memory_coordinates,
        initial.physical_hidden_coordinates,
        atol=2e-14,
        rtol=2e-14,
    )
    backbone = frame.autonomous_velocity(
        initial.retained_coordinates,
        np.zeros(frame.hidden_dimension),
        field,
        drive_value=0.23,
        drive_rate=-0.17,
    )
    np.testing.assert_allclose(
        backbone.centered_velocity,
        field(backbone.closed_coordinates, 0.23),
        atol=3e-12,
        rtol=3e-12,
    )

    velocity = frame.autonomous_velocity(
        initial.retained_coordinates,
        initial.memory_coordinates,
        field,
        drive_value=0.23,
        drive_rate=-0.17,
    )
    assert np.all(np.isfinite(velocity.memory_velocity))
    assert abs(
        velocity.projected_norm_rate
        - velocity.projected_norm_rate_from_incompatibility
    ) < 3e-12

    trajectory = propagate_archive_auxiliary_rk4(
        frame,
        initial,
        field,
        lambda time: 0.0,
        lambda time: 0.0,
        final_time=0.02,
        time_step=0.01,
        sample_step=0.01,
    )
    assert trajectory.closed_coordinates.shape == (3, 31)
    assert trajectory.memory_coordinates.shape == (3, frame.hidden_dimension)
    assert np.max(trajectory.centered_section_relative_residuals) < 2e-13
    assert np.max(trajectory.projected_norm_identity_residuals) < 5e-12
