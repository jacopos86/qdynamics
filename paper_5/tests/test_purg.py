from __future__ import annotations

from functools import lru_cache

import numpy as np

from paper5.stability import DimerParameters
from paper5.stability.exact_reference import (
    exact_holstein_wavefunction_trajectory_for_diagnostics,
)
from paper5.stability.purg import (
    PurgConstructionSettings,
    build_purg_construction,
    build_purg_operator_bounds,
    certify_purg_projection,
    propagate_purg_midpoint,
    purg_gate_a_diagnostics,
)
from paper5.stability.purg_analysis import (
    run_purg_construction_gate,
    write_purg_construction_gate_artifact,
)


@lru_cache(maxsize=1)
def _small_construction():
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
    record = construction.record(32)
    assert record.projection is not None
    return parameters, construction, record.projection


def test_purg_basis_is_deterministic_and_contains_preparation_directions() -> None:
    parameters, first, projection = _small_construction()
    second = build_purg_construction(
        parameters,
        phonon_cutoff=3,
        settings=first.settings,
    )
    second_projection = second.record(32).projection
    assert second_projection is not None

    assert first.initial_rank <= 32
    assert projection.model.dimension == 32
    np.testing.assert_allclose(
        projection.basis,
        second_projection.basis,
        atol=2e-11,
        rtol=2e-11,
    )
    diagnostics = purg_gate_a_diagnostics(first)
    record = first.record(32)
    assert record.deflated_columns == 3
    assert record.truncated_columns == 2
    cap = diagnostics["caps"]["32"]
    assert cap["orthogonality_residual"] < 1e-12
    assert cap["initial_state_containment_residual"] < 1e-11
    assert cap["initial_drive_direction_containment_residual"] < 1e-11
    assert diagnostics["ground_residual"] < 1e-11
    assert diagnostics["maximum_shifted_solve_relative_residual"] < 1e-11
    assert diagnostics["coordinate_map_round_trip_residual"] < 1e-12
    assert diagnostics["coordinate_jacobian_directional_residual"] < 1e-12
    assert diagnostics["decoupled_force"]["passed"]
    assert diagnostics["online_dependency_audit"]["passed"]
    assert max(projection.compression_hermitian_leakage.values()) < 1e-12


def test_purg_initial_outputs_and_derivatives_are_contracted_from_one_ket() -> None:
    parameters, construction, projection = _small_construction()
    model = projection.model
    state = model.initial_state

    full_raw = np.asarray(
        [
            np.vdot(
                construction.ground_state,
                operator @ construction.ground_state,
            ).real
            for operator in construction.raw_basis.observables
        ]
    )
    np.testing.assert_allclose(
        model.raw_coordinates(state),
        full_raw,
        atol=2e-11,
        rtol=2e-11,
    )

    drive_value = parameters.drive_difference(0.0)
    full_velocity = -1j * (
        projection.static_hamiltonian @ construction.ground_state
        + drive_value
        * (projection.drive_hamiltonian @ construction.ground_state)
    )
    full_raw_velocity = np.asarray(
        [
            2.0
            * np.vdot(
                full_velocity,
                operator @ construction.ground_state,
            ).real
            for operator in construction.raw_basis.observables
        ]
    )
    np.testing.assert_allclose(
        model.raw_velocity(state, drive_value=drive_value),
        full_raw_velocity,
        atol=3e-11,
        rtol=3e-11,
    )


def test_purg_analytic_derivative_norm_and_work_identities() -> None:
    _, _, projection = _small_construction()
    model = projection.model
    rng = np.random.default_rng(2026080308)
    state = rng.normal(size=model.dimension) + 1j * rng.normal(
        size=model.dimension
    )
    state /= np.linalg.norm(state)
    drive_value = 0.37
    velocity = model.rhs(state, drive_value=drive_value)

    assert abs(2.0 * np.vdot(state, velocity).real) < 2e-13
    hamiltonian = model.hamiltonian(drive_value)
    assert abs(2.0 * np.vdot(velocity, hamiltonian @ state).real) < 2e-12

    step = 1e-7
    finite_difference = (
        model.raw_coordinates(state + step * velocity)
        - model.raw_coordinates(state - step * velocity)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        model.raw_velocity(state, drive_value=drive_value),
        finite_difference,
        atol=2e-8,
        rtol=2e-8,
    )


def test_purg_residual_gram_matches_explicit_full_space_residual() -> None:
    _, _, projection = _small_construction()
    model = projection.model
    rng = np.random.default_rng(2026080309)
    state = rng.normal(size=model.dimension) + 1j * rng.normal(
        size=model.dimension
    )
    state /= np.linalg.norm(state)

    for drive_value in (0.0, 0.41):
        explicit = np.linalg.norm(
            projection.projection_residual(state, drive_value=drive_value)
        )
        compressed = projection.projection_residual_norm(
            state,
            drive_value=drive_value,
        )
        np.testing.assert_allclose(compressed, explicit, atol=3e-13, rtol=3e-13)


def test_purg_midpoint_is_unitary_and_duhamel_certificate_bounds_error() -> None:
    parameters, construction, projection = _small_construction()
    midpoint = propagate_purg_midpoint(
        projection.model,
        parameters,
        final_time=0.02,
        step=0.01,
    )
    assert midpoint.norm_defect < 2e-13

    operator_bounds = build_purg_operator_bounds(projection)
    certificate = certify_purg_projection(
        projection,
        parameters,
        operator_bounds,
        final_time=0.02,
        step=0.01,
        quadrature_absolute_tolerance=1e-12,
    )
    exact = exact_holstein_wavefunction_trajectory_for_diagnostics(
        parameters,
        sample_times=np.array([0.0, 0.02]),
        phonon_cutoff=3,
        relative_tolerance=1e-12,
        absolute_tolerance=1e-14,
        maximum_step=0.001,
    )
    lifted = projection.lift(certificate.states[-1])
    phase = np.vdot(exact.state_vectors[:, 0], construction.ground_state)
    phase /= abs(phase)
    actual_error = np.linalg.norm(phase * exact.state_vectors[:, -1] - lifted)

    assert certificate.continuous_norm_defect < 2e-13
    assert certificate.quadrature_error_estimate <= 1e-12
    assert actual_error <= certificate.state_error_bound[-1] + 2e-10
    assert np.all(certificate.raw_derivative_absolute_bounds >= 0.0)
    assert np.all(certificate.centered_derivative_absolute_bounds >= 0.0)


def test_online_purg_model_does_not_retain_full_space_or_exact_trajectory() -> None:
    _, _, projection = _small_construction()
    model_fields = set(projection.model.__dataclass_fields__)

    assert model_fields == {
        "phonon_cutoff",
        "cap_label",
        "static_hamiltonian",
        "drive_hamiltonian",
        "raw_observables",
        "initial_state",
    }
    assert projection.model.static_hamiltonian.shape == (
        projection.model.dimension,
        projection.model.dimension,
    )
    assert projection.basis.shape[0] > projection.model.dimension


def test_purg_construction_gate_emits_machine_readable_artifact(tmp_path) -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5, drive_amplitude=1.0)
    result = run_purg_construction_gate(
        parameters,
        phonon_cutoff=3,
        settings=PurgConstructionSettings(
            caps=(32,),
            final_time=0.02,
            construction_step=0.01,
        ),
        sample_step=0.01,
    )
    output = tmp_path / "purg_gate"
    write_purg_construction_gate_artifact(result, output)

    assert result.gate_a["passed"]
    assert 32 in result.gate_b
    assert not result.passed  # No mandatory next-rank audit was supplied.
    assert (output / "summary.json").is_file()
    assert (output / "arrays.npz").is_file()
    assert (output / "manifest.json").is_file()

    with np.load(output / "arrays.npz") as arrays:
        assert "raw_to_centered_jacobian_at_zero" in arrays
        assert "raw_to_centered_hessian" in arrays
        assert "cap_32_basis" in arrays
        assert "cap_32_static_hamiltonian" in arrays
        assert "cap_32_drive_hamiltonian" in arrays
        assert "cap_32_raw_observables" in arrays
        assert "cap_32_initial_state" in arrays
        assert "cap_32_static_residual_gram" in arrays
        assert "cap_32_cross_residual_gram" in arrays
        assert "cap_32_drive_residual_gram" in arrays
