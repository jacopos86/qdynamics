from __future__ import annotations

import numpy as np

from paper5.stability.latent_source_closure import (
    estimate_time_derivative,
    fit_latent_source_basis,
    fit_latent_source_evolution,
    predict_latent_source_evolution,
    project_missing_source,
    reconstruct_missing_source,
    select_latent_source_evolution,
    fit_second_order_latent_source_evolution,
    fit_stable_second_order_latent_source_evolution,
    latent_homogeneous_eigenvalues,
    predict_second_order_latent_source_evolution,
    predict_stable_second_order_latent_source_evolution,
)


def test_five_point_derivative_is_exact_for_quartic_samples() -> None:
    times = np.linspace(-1.0, 1.0, 41)
    values = np.column_stack(
        (
            times**4 - 2.0 * times**2 + 3.0 * times,
            -0.5 * times**3 + 2.0,
        )
    )
    expected = np.column_stack(
        (
            4.0 * times**3 - 4.0 * times + 3.0,
            -1.5 * times**2,
        )
    )

    actual = estimate_time_derivative(times, values)

    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


def test_source_basis_trained_on_one_member_reconstructs_shared_modes() -> None:
    rng = np.random.default_rng(260804)
    physical_basis, _ = np.linalg.qr(rng.normal(size=(14, 3)))
    times = np.linspace(0.0, 6.0, 61)
    source = np.empty((3, times.size, 14), dtype=float)
    offset = rng.normal(scale=0.1, size=14)
    for member in range(3):
        coefficients = np.column_stack(
            (
                np.sin(times) + 0.02 * member,
                np.cos(2.0 * times) - 0.01 * member,
                np.sin(0.5 * times + 0.1 * member),
            )
        )
        source[member] = offset + coefficients @ physical_basis.T
    training_mask = times <= 3.0

    model = fit_latent_source_basis(
        source,
        np.ones(14),
        rank=3,
        training_member=0,
        training_mask=training_mask,
    )
    latent = project_missing_source(source, model)
    reconstructed = reconstruct_missing_source(latent, model)

    np.testing.assert_allclose(reconstructed, source, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(
        model.basis @ model.basis.T,
        np.eye(3),
        atol=2e-14,
        rtol=2e-14,
    )


def test_affine_latent_evolution_generalizes_to_a_held_out_member() -> None:
    times = np.linspace(0.0, 8.0, 161)
    coordinates = np.zeros((2, times.size, 31), dtype=float)
    latent = np.empty((2, times.size, 2), dtype=float)
    drive = np.empty((2, times.size), dtype=float)
    for member in range(2):
        coordinates[member, :, 0] = np.sin(times + 0.1 * member)
        coordinates[member, :, 3] = np.cos(0.7 * times - 0.2 * member)
        latent[member, :, 0] = np.sin(1.3 * times) + 0.05 * member
        latent[member, :, 1] = np.cos(0.9 * times) - 0.03 * member
        drive[member] = np.sin(0.4 * times + 0.15 * member)
    derivative = np.empty_like(latent)
    derivative[..., 0] = (
        0.4
        + 0.7 * coordinates[..., 0]
        - 0.2 * coordinates[..., 3]
        + 0.5 * latent[..., 0]
        - 0.3 * latent[..., 1]
        + 0.6 * drive
    )
    derivative[..., 1] = (
        -0.1
        - 0.4 * coordinates[..., 0]
        + 0.8 * coordinates[..., 3]
        + 0.2 * latent[..., 0]
        + 0.9 * latent[..., 1]
        - 0.25 * drive
    )

    model = fit_latent_source_evolution(
        coordinates[0],
        latent[0],
        derivative[0],
        drive[0],
        np.ones(31),
        feature_family="state_latent_affine",
        ridge_penalty=0.0,
    )
    predicted = predict_latent_source_evolution(
        model,
        coordinates[1],
        latent[1],
        drive[1],
    )

    np.testing.assert_allclose(predicted, derivative[1], atol=2e-12, rtol=2e-12)


def test_candidate_selection_rejects_latent_only_law_when_state_is_required() -> None:
    times = np.linspace(0.0, 12.0, 241)
    coordinates = np.zeros((times.size, 31), dtype=float)
    coordinates[:, 0] = np.sin(times)
    coordinates[:, 3] = np.cos(0.3 * times)
    latent = np.column_stack((np.sin(1.7 * times), np.cos(1.1 * times)))
    drive = np.sin(0.4 * times)
    derivative = np.column_stack(
        (
            0.8 * coordinates[:, 0] - 0.2 * latent[:, 1],
            -0.6 * coordinates[:, 3] + 0.3 * latent[:, 0],
        )
    )
    training = times <= 7.0
    validation = times >= 8.0

    selection = select_latent_source_evolution(
        coordinates,
        latent,
        derivative,
        drive,
        np.ones(31),
        training_mask=training,
        validation_mask=validation,
        feature_families=("latent_affine", "state_latent_affine"),
        ridge_penalties=(0.0,),
    )

    assert selection.model.feature_family == "state_latent_affine"
    assert selection.validation_normalized_rms < 1e-11
    latent_only = next(
        score
        for score in selection.candidates
        if score.feature_family == "latent_affine"
    )
    assert latent_only.validation_normalized_rms > 0.1


def test_second_order_model_enforces_kinematics_and_learns_acceleration() -> None:
    times = np.linspace(0.0, 10.0, 301)
    coordinates = np.zeros((times.size, 31), dtype=float)
    coordinates[:, 0] = np.sin(0.4 * times)
    source = np.column_stack((np.sin(times), np.cos(1.3 * times)))
    rates = np.column_stack((np.cos(times), -1.3 * np.sin(1.3 * times)))
    drive = np.sin(0.2 * times)
    acceleration = np.column_stack(
        (
            -source[:, 0] + 0.2 * coordinates[:, 0],
            -1.3**2 * source[:, 1] - 0.1 * rates[:, 1] + 0.3 * drive,
        )
    )

    model = fit_second_order_latent_source_evolution(
        coordinates,
        source,
        rates,
        acceleration,
        drive,
        np.ones(31),
        feature_family="state_latent_affine",
        ridge_penalty=0.0,
    )
    predicted = predict_second_order_latent_source_evolution(
        model,
        coordinates,
        source,
        rates,
        drive,
    )

    np.testing.assert_array_equal(predicted[:, :2], rates)
    np.testing.assert_allclose(
        predicted[:, 2:],
        acceleration,
        atol=2e-12,
        rtol=2e-12,
    )


def test_stable_second_order_fit_shifts_all_homogeneous_modes_left() -> None:
    times = np.linspace(0.0, 20.0, 501)
    coordinates = np.zeros((times.size, 31), dtype=float)
    coordinates[:, 0] = np.sin(0.3 * times)
    source = np.column_stack((np.sin(times), np.cos(1.4 * times)))
    rates = np.column_stack((np.cos(times), -1.4 * np.sin(1.4 * times)))
    acceleration = np.column_stack(
        (
            -source[:, 0] + 0.1 * rates[:, 0] + coordinates[:, 0],
            -1.4**2 * source[:, 1] + 0.08 * rates[:, 1],
        )
    )
    drive = np.zeros(times.size)

    model = fit_stable_second_order_latent_source_evolution(
        coordinates,
        source,
        rates,
        acceleration,
        drive,
        np.ones(31),
        ridge_penalty=1e-8,
        stability_margin=0.02,
    )
    predicted = predict_stable_second_order_latent_source_evolution(
        model,
        coordinates,
        source,
        rates,
        drive,
    )

    assert np.max(np.real(latent_homogeneous_eigenvalues(model))) <= -0.02 + 1e-10
    np.testing.assert_array_equal(predicted[:, :2], rates)
    assert np.all(np.isfinite(predicted))
