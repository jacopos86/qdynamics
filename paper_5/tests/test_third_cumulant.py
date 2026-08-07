from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper5.stability import (
    DimerParameters,
    MomentKey,
    THIRD_CUMULANT_MOMENT_KEYS,
    THIRD_CUMULANT_STATE_NAMES,
    pack_third_cumulant_state,
    third_cumulant_energy,
    third_cumulant_matrix_derivative,
    third_cumulant_rhs,
    third_cumulant_to_matrix_state,
    unpack_third_cumulant_state,
)
from paper5.stability.matrix_reference import (
    matrix_total_energy,
    pack_matrix_state,
)
from paper5.stability.third_cumulant import (
    IDENTITY,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    _closed_moment,
)
from paper5.stability.third_cumulant_analysis import run_analysis


def _sample_coordinates(seed: int = 7) -> np.ndarray:
    generator = np.random.default_rng(seed)
    moments = {
        key: float(generator.normal(scale=0.08))
        for key in THIRD_CUMULANT_MOMENT_KEYS
    }
    moments[MomentKey(IDENTITY, IDENTITY, 2, 0)] = 0.7
    moments[MomentKey(IDENTITY, IDENTITY, 0, 2)] = 0.8
    return pack_third_cumulant_state(-0.4 + 0.12j, moments)


def test_third_cumulant_basis_has_declared_degree_counts() -> None:
    assert len(THIRD_CUMULANT_STATE_NAMES) == 47
    assert len(THIRD_CUMULANT_MOMENT_KEYS) == 45
    assert sum(key.degree == 1 for key in THIRD_CUMULANT_MOMENT_KEYS) == 5
    assert sum(key.degree == 2 for key in THIRD_CUMULANT_MOMENT_KEYS) == 15
    assert sum(key.degree == 3 for key in THIRD_CUMULANT_MOMENT_KEYS) == 25

    coordinates = _sample_coordinates()
    center, moments = unpack_third_cumulant_state(coordinates)
    repacked = pack_third_cumulant_state(center, moments)
    np.testing.assert_array_equal(repacked, coordinates)


def test_zero_fourth_cumulant_reconstructs_gaussian_x_fourth_moment() -> None:
    variance = 0.37
    retained = {
        MomentKey(IDENTITY, IDENTITY, 1, 0): 0.0,
        MomentKey(IDENTITY, IDENTITY, 2, 0): variance,
        MomentKey(IDENTITY, IDENTITY, 3, 0): 0.0,
    }
    reconstructed = _closed_moment(
        MomentKey(IDENTITY, IDENTITY, 4, 0), retained
    )
    assert abs(reconstructed - 3.0 * variance**2) < 1e-14


def test_first_moment_equations_match_transformed_hamiltonian() -> None:
    coordinates = _sample_coordinates()
    _, moments = unpack_third_cumulant_state(coordinates)
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    time = 0.4
    _, derivative = unpack_third_cumulant_state(
        third_cumulant_rhs(time, coordinates, parameters)
    )

    spin_x = MomentKey(IDENTITY, PAULI_X, 0, 0)
    spin_y = MomentKey(IDENTITY, PAULI_Y, 0, 0)
    spin_z = MomentKey(IDENTITY, PAULI_Z, 0, 0)
    position = MomentKey(IDENTITY, IDENTITY, 1, 0)
    momentum = MomentKey(IDENTITY, IDENTITY, 0, 1)
    spin_x_position = MomentKey(IDENTITY, PAULI_X, 1, 0)
    spin_y_position = MomentKey(IDENTITY, PAULI_Y, 1, 0)
    drive = parameters.drive_difference(time)
    coupling = parameters.coupling

    assert abs(
        derivative[position] - parameters.omega_ph * moments[momentum]
    ) < 1e-14
    assert abs(
        derivative[momentum]
        + parameters.omega_ph * moments[position]
        + 2.0 * coupling * moments[spin_z]
    ) < 1e-14
    assert abs(
        derivative[spin_x]
        + drive * moments[spin_y]
        + 2.0 * coupling * moments[spin_y_position]
    ) < 1e-14
    assert abs(
        derivative[spin_y]
        - 2.0 * parameters.hopping * moments[spin_z]
        - drive * moments[spin_x]
        - 2.0 * coupling * moments[spin_x_position]
    ) < 1e-14
    assert abs(
        derivative[spin_z]
        + 2.0 * parameters.hopping * moments[spin_y]
    ) < 1e-14


def test_matrix_contraction_differential_and_energy_are_consistent() -> None:
    coordinates = _sample_coordinates()
    generator = np.random.default_rng(12)
    direction = generator.normal(scale=0.1, size=coordinates.size)
    step = 1e-7

    finite_difference = (
        pack_matrix_state(
            third_cumulant_to_matrix_state(coordinates + step * direction)
        )
        - pack_matrix_state(
            third_cumulant_to_matrix_state(coordinates - step * direction)
        )
    ) / (2.0 * step)
    analytic = pack_matrix_state(
        third_cumulant_matrix_derivative(coordinates, direction)
    )
    np.testing.assert_allclose(analytic, finite_difference, atol=4e-10)

    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=0.0,
    )
    matrix_energy = matrix_total_energy(
        third_cumulant_to_matrix_state(coordinates), parameters
    )
    assert abs(
        third_cumulant_energy(0.0, coordinates, parameters) - matrix_energy
    ) < 1e-14


def test_undriven_third_cumulant_rhs_conserves_energy_instantaneously() -> None:
    coordinates = _sample_coordinates()
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=0.0,
    )
    velocity = third_cumulant_rhs(0.0, coordinates, parameters)
    step = 1e-7
    energy_flux = (
        third_cumulant_energy(
            0.0, coordinates + step * velocity, parameters
        )
        - third_cumulant_energy(
            0.0, coordinates - step * velocity, parameters
        )
    ) / (2.0 * step)
    assert abs(energy_flux) < 2e-9


def test_third_cumulant_analysis_writes_retrievable_gate(
    tmp_path: Path,
) -> None:
    summary = run_analysis(
        tmp_path,
        parameters=DimerParameters(
            lambda_ep=0.0,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        final_time=0.1,
        sample_step=0.1,
        phonon_cutoff=3,
        convergence_cutoffs=(3,),
        convergence_time=0.1,
        maximum_step=0.02,
    )

    assert summary["validation_gate"]["lower_equations_passed"]
    assert summary["validation_gate"]["terminal_closure_passed"]
    assert (tmp_path / "third_cumulant_derivative_gate.npz").is_file()
    assert (tmp_path / "third_cumulant_derivative_gate.png").is_file()
    manifest = json.loads(
        (tmp_path / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert "summary.json" in manifest["artifact_hashes"]
