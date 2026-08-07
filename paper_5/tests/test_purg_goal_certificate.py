from __future__ import annotations

from functools import lru_cache

import numpy as np

from paper5.stability.exact_reference import (
    exact_holstein_wavefunction_trajectory_for_diagnostics,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.krylov_memory_closure import raw_velocity_to_closed_velocity
from paper5.stability.purg import (
    PurgConstructionSettings,
    build_purg_construction,
    build_purg_operator_bounds,
)
from paper5.stability.purg_goal_certificate import (
    build_dual_leakage_envelope,
    correction_residual_projection_form,
    gershgorin_hermitian_enclosure,
    numerical_dual_remainder_radius,
    estimate_centered_derivative_intervals,
    explicit_dual_remainder_radius,
    projected_correction_residual,
    projected_correction_velocity,
    propagate_forward_remainder,
    propagate_explicit_reduced_dual,
    propagate_purg_error_correction,
    quadratic_goal_interval,
    reduced_adjoint_goal_interval,
)


def _orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(np.asarray(matrix, dtype=complex), mode="reduced")
    return basis


def test_projected_correction_residual_has_the_registered_signs() -> None:
    rng = np.random.default_rng(2026080311)
    raw = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    hamiltonian = 0.5 * (raw + raw.conj().T)
    basis = _orthonormal_columns(
        rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    )
    source = rng.normal(size=5) + 1j * rng.normal(size=5)
    correction = rng.normal(size=3) + 1j * rng.normal(size=3)
    velocity = projected_correction_velocity(
        hamiltonian,
        source,
        basis,
        correction,
    )

    direct = projected_correction_residual(
        hamiltonian,
        source,
        basis,
        correction,
        velocity,
    )
    projected = correction_residual_projection_form(
        hamiltonian,
        source,
        basis,
        correction,
    )
    np.testing.assert_allclose(direct, projected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(basis.conj().T @ direct, 0.0, atol=2e-14)


def test_quadratic_goal_interval_contains_constructed_exact_error() -> None:
    rng = np.random.default_rng(2026080312)
    raw = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    operator = 0.5 * (raw + raw.conj().T)
    represented = rng.normal(size=7) + 1j * rng.normal(size=7)
    represented /= np.linalg.norm(represented)
    exact = rng.normal(size=7) + 1j * rng.normal(size=7)
    exact /= np.linalg.norm(exact)
    unresolved = 0.03 * (
        rng.normal(size=7) + 1j * rng.normal(size=7)
    )
    correction = exact - represented - unresolved
    lower, upper = np.linalg.eigvalsh(operator)[[0, -1]]

    interval = quadratic_goal_interval(
        represented_state=represented,
        lifted_correction=correction,
        unresolved_radius=float(np.linalg.norm(unresolved)),
        operator=operator,
        spectral_lower=float(lower),
        spectral_upper=float(upper),
    )
    exact_error = float(
        np.vdot(exact, operator @ exact).real
        - np.vdot(represented, operator @ represented).real
    )
    assert interval.lower - 2e-14 <= exact_error <= interval.upper + 2e-14


def test_gershgorin_encloses_hermitian_spectrum() -> None:
    rng = np.random.default_rng(2026080313)
    raw = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    operator = 0.5 * (raw + raw.conj().T)
    lower, upper = gershgorin_hermitian_enclosure(operator)
    eigenvalues = np.linalg.eigvalsh(operator)
    assert lower <= eigenvalues[0]
    assert upper >= eigenvalues[-1]


def test_reduced_adjoint_interval_contains_constructed_goal_error() -> None:
    rng = np.random.default_rng(2026080314)
    raw = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    operator = 0.5 * (raw + raw.conj().T)
    represented = rng.normal(size=6) + 1j * rng.normal(size=6)
    represented /= np.linalg.norm(represented)
    exact = rng.normal(size=6) + 1j * rng.normal(size=6)
    exact /= np.linalg.norm(exact)
    unresolved = 0.02 * (
        rng.normal(size=6) + 1j * rng.normal(size=6)
    )
    correction = exact - represented - unresolved
    lower, upper = np.linalg.eigvalsh(operator)[[0, -1]]
    dual_basis = np.eye(6, dtype=complex)
    interval = reduced_adjoint_goal_interval(
        represented_state=represented,
        lifted_correction=correction,
        unresolved_radius=float(np.linalg.norm(unresolved)),
        operator=operator,
        spectral_lower=float(lower),
        spectral_upper=float(upper),
        dual_basis=dual_basis,
        forward_remainder_state=unresolved,
        dual_remainder_radius=0.0,
    )
    exact_error = float(
        np.vdot(exact, operator @ exact).real
        - np.vdot(represented, operator @ represented).real
    )
    assert interval.lower - 2e-14 <= exact_error <= interval.upper + 2e-14
    assert interval.dwr_radius <= interval.cheap.radius


@lru_cache(maxsize=1)
def _small_projection():
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5, drive_amplitude=1.0)
    construction = build_purg_construction(
        parameters,
        phonon_cutoff=3,
        settings=PurgConstructionSettings(
            caps=(32,),
            final_time=0.02,
            construction_step=0.01,
        ),
    )
    projection = construction.record(32).projection
    assert projection is not None
    return parameters, construction, projection


def test_full_certificate_space_remainder_bounds_short_exact_path() -> None:
    parameters, construction, projection = _small_projection()
    full_basis = np.eye(projection.full_dimension, dtype=complex)
    result = propagate_purg_error_correction(
        projection,
        parameters,
        full_basis,
        final_time=0.02,
        step=0.01,
        quadrature_absolute_tolerance=1e-11,
    )
    exact = exact_holstein_wavefunction_trajectory_for_diagnostics(
        parameters,
        sample_times=result.times,
        phonon_cutoff=3,
        relative_tolerance=1e-12,
        absolute_tolerance=1e-14,
        maximum_step=0.001,
    )
    phase = np.vdot(exact.state_vectors[:, 0], construction.ground_state)
    phase /= abs(phase)
    for index in range(result.times.size):
        represented = projection.lift(result.primal_states[index])
        corrected = represented + full_basis @ result.correction_states[index]
        actual_remainder = np.linalg.norm(
            phase * exact.state_vectors[:, index] - corrected
        )
        assert actual_remainder <= result.unresolved_error_bound[index] + 2e-9

    assert result.maximum_residual_identity_error < 2e-12
    assert result.quadrature_error_estimate < 1e-9
    np.testing.assert_allclose(
        np.linalg.norm(result.correction_residuals, axis=1),
        result.correction_residual_norms,
        atol=2e-13,
        rtol=2e-13,
    )

    forward = propagate_forward_remainder(
        projection,
        parameters,
        full_basis,
        result,
        full_basis,
        quadrature_absolute_tolerance=1e-11,
    )
    envelope = build_dual_leakage_envelope(
        projection,
        parameters,
        full_basis,
        result.times,
        quadrature_absolute_tolerance=1e-11,
    )
    operator = projection.full_raw_observables[0]
    represented = projection.lift(result.primal_states[-1])
    lifted_correction = full_basis @ result.correction_states[-1]
    lower, upper = gershgorin_hermitian_enclosure(operator)
    mu = 0.5 * (lower + upper)
    goal_action = operator @ (represented + lifted_correction) - mu * (
        represented + lifted_correction
    )
    radius, terminal_defect = numerical_dual_remainder_radius(
        goal_action=goal_action,
        terminal_index=result.times.size - 1,
        dual_basis=full_basis,
        correction=result,
        forward_remainder=forward,
        envelope=envelope,
    )
    assert terminal_defect < 2e-14
    assert radius >= 0.0
    assert envelope.static_leakage_norm < 2e-13
    assert envelope.drive_leakage_norm < 2e-13

    explicit = propagate_explicit_reduced_dual(
        projection,
        parameters,
        full_basis,
        goal_action,
        result.times,
        terminal_index=result.times.size - 1,
        quadrature_absolute_tolerance=1e-11,
    )
    explicit_radius = explicit_dual_remainder_radius(
        explicit,
        result,
        forward,
    )
    assert explicit.terminal_projection_defect < 2e-14
    assert explicit_radius >= 0.0
    assert explicit.quadrature_error_estimate < 1e-9

    estimate = estimate_centered_derivative_intervals(
        projection,
        parameters,
        full_basis,
        result,
        full_basis,
        forward,
        envelope,
        build_purg_operator_bounds(projection),
    )
    for index, time in enumerate(result.times):
        exact_state = phase * exact.state_vectors[:, index]
        drive = parameters.drive_difference(float(time))
        hamiltonian = (
            projection.static_hamiltonian
            + drive * projection.drive_hamiltonian
        )
        exact_velocity = -1j * (hamiltonian @ exact_state)
        raw = np.asarray(
            [
                np.vdot(exact_state, operator @ exact_state).real
                for operator in projection.full_raw_observables
            ]
        )
        raw_velocity = np.asarray(
            [
                2.0 * np.vdot(exact_velocity, operator @ exact_state).real
                for operator in projection.full_raw_observables
            ]
        )
        exact_centered_velocity = raw_velocity_to_closed_velocity(
            raw,
            raw_velocity,
        )
        modeled_centered_velocity = projection.model.centered_velocity(
            result.primal_states[index],
            drive_value=drive,
        )
        error = exact_centered_velocity - modeled_centered_velocity
        assert np.all(error >= estimate.lower[index] - 2e-8)
        assert np.all(error <= estimate.upper[index] + 2e-8)
