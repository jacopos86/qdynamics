from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper5.stability import (
    DimerParameters,
    FOURTH_CUMULANT_HIERARCHY,
    FOURTH_CUMULANT_MOMENT_KEYS,
    FOURTH_CUMULANT_STATE_NAMES,
    MomentKey,
    THIRD_CUMULANT_HIERARCHY,
    fourth_cumulant_energy,
    fourth_cumulant_matrix_derivative,
    fourth_cumulant_rhs,
    fourth_cumulant_to_matrix_state,
    pack_fourth_cumulant_state,
    pack_third_cumulant_state,
    third_cumulant_rhs,
    unpack_fourth_cumulant_state,
)
from paper5.stability.matrix_reference import (
    matrix_total_energy,
    pack_matrix_state,
)
from paper5.stability.hierarchy_analysis import run_hierarchy_analysis
from paper5.stability.moment_hierarchy import IDENTITY


def _sample_fourth_coordinates(seed: int = 17) -> np.ndarray:
    generator = np.random.default_rng(seed)
    moments = {
        key: float(generator.normal(scale=0.06))
        for key in FOURTH_CUMULANT_MOMENT_KEYS
    }
    moments[MomentKey(IDENTITY, IDENTITY, 2, 0)] = 0.72
    moments[MomentKey(IDENTITY, IDENTITY, 0, 2)] = 0.81
    return pack_fourth_cumulant_state(-0.35 + 0.09j, moments)


def test_fourth_order_hierarchy_has_82_coordinates() -> None:
    assert len(FOURTH_CUMULANT_STATE_NAMES) == 82
    assert len(FOURTH_CUMULANT_MOMENT_KEYS) == 80
    assert sum(key.degree == 1 for key in FOURTH_CUMULANT_MOMENT_KEYS) == 5
    assert sum(key.degree == 2 for key in FOURTH_CUMULANT_MOMENT_KEYS) == 15
    assert sum(key.degree == 3 for key in FOURTH_CUMULANT_MOMENT_KEYS) == 25
    assert sum(key.degree == 4 for key in FOURTH_CUMULANT_MOMENT_KEYS) == 35

    coordinates = _sample_fourth_coordinates()
    center, moments = unpack_fourth_cumulant_state(coordinates)
    np.testing.assert_array_equal(
        pack_fourth_cumulant_state(center, moments),
        coordinates,
    )


def test_zero_fifth_cumulant_reconstructs_gaussian_x_fifth_moment() -> None:
    mean = 0.23
    variance = 0.41
    moments = {
        MomentKey(IDENTITY, IDENTITY, 1, 0): mean,
        MomentKey(IDENTITY, IDENTITY, 2, 0): mean**2 + variance,
        MomentKey(IDENTITY, IDENTITY, 3, 0): (
            mean**3 + 3.0 * mean * variance
        ),
        MomentKey(IDENTITY, IDENTITY, 4, 0): (
            mean**4
            + 6.0 * mean**2 * variance
            + 3.0 * variance**2
        ),
    }
    reconstructed = FOURTH_CUMULANT_HIERARCHY.closed_moment(
        MomentKey(IDENTITY, IDENTITY, 5, 0),
        moments,
    )
    expected = mean**5 + 10.0 * mean**3 * variance + 15.0 * mean * variance**2
    assert abs(reconstructed - expected) < 2e-14


def test_fourth_hierarchy_contains_third_hierarchy_without_equation_drift() -> None:
    generator = np.random.default_rng(22)
    third_moments = {
        key: float(generator.normal(scale=0.05))
        for key in THIRD_CUMULANT_HIERARCHY.moment_keys
    }
    third_moments[MomentKey(IDENTITY, IDENTITY, 2, 0)] = 0.7
    third_moments[MomentKey(IDENTITY, IDENTITY, 0, 2)] = 0.8
    center = -0.4 + 0.1j
    third_state = pack_third_cumulant_state(center, third_moments)

    fourth_moments = dict(third_moments)
    for key in FOURTH_CUMULANT_HIERARCHY.moment_keys:
        if key.degree == 4:
            fourth_moments[key] = THIRD_CUMULANT_HIERARCHY.closed_moment(
                key,
                third_moments,
            )
    fourth_state = pack_fourth_cumulant_state(center, fourth_moments)
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)

    third_derivative = third_cumulant_rhs(0.37, third_state, parameters)
    fourth_derivative = fourth_cumulant_rhs(0.37, fourth_state, parameters)
    np.testing.assert_allclose(
        fourth_derivative[: third_derivative.size],
        third_derivative,
        atol=2e-14,
    )


def test_fourth_matrix_map_differential_and_energy_are_consistent() -> None:
    coordinates = _sample_fourth_coordinates()
    generator = np.random.default_rng(31)
    direction = generator.normal(scale=0.08, size=coordinates.size)
    step = 1e-7
    finite_difference = (
        pack_matrix_state(
            fourth_cumulant_to_matrix_state(coordinates + step * direction)
        )
        - pack_matrix_state(
            fourth_cumulant_to_matrix_state(coordinates - step * direction)
        )
    ) / (2.0 * step)
    analytic = pack_matrix_state(
        fourth_cumulant_matrix_derivative(coordinates, direction)
    )
    np.testing.assert_allclose(analytic, finite_difference, atol=4e-10)

    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=0.0,
    )
    assert abs(
        fourth_cumulant_energy(0.0, coordinates, parameters)
        - matrix_total_energy(
            fourth_cumulant_to_matrix_state(coordinates), parameters
        )
    ) < 2e-14
    velocity = fourth_cumulant_rhs(0.0, coordinates, parameters)
    energy_flux = (
        fourth_cumulant_energy(
            0.0,
            coordinates + step * velocity,
            parameters,
        )
        - fourth_cumulant_energy(
            0.0,
            coordinates - step * velocity,
            parameters,
        )
    ) / (2.0 * step)
    assert abs(energy_flux) < 2e-9


def test_generic_hierarchy_analysis_writes_retrievable_gate(
    tmp_path: Path,
) -> None:
    summary = run_hierarchy_analysis(
        tmp_path,
        hierarchy=FOURTH_CUMULANT_HIERARCHY,
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
    assert (tmp_path / "order_4_cumulant_derivative_gate.npz").is_file()
    assert (tmp_path / "order_4_cumulant_derivative_gate.png").is_file()
    manifest = json.loads(
        (tmp_path / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert "summary.json" in manifest["artifact_hashes"]
