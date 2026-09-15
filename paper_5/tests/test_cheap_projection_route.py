"""Tests for the cheap physicality repair with no conic solve."""

from __future__ import annotations

import numpy as np
import pytest

from paper5.stability.adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    uncentered_joint_moment_matrix,
)
from paper5.stability.apcm_moment_projection import state_lower_moments
from paper5.stability.apcm_positive_extension import (
    SymmetryReducedPositiveExtension,
)
from paper5.stability.cheap_projection_route import (
    AffineGramCoordinates,
    CarriedWitnessAffineGram,
    CheapProjectionSettings,
    _clip_to_floor,
    _hermitian_features,
    _hermitian_from_features,
    integrate_cheap_projection_ssprk3,
    prepare_cheap_projection_route,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.projected_apcm import unpack_projected_apcm_state


@pytest.fixture(scope="module")
def coordinates() -> AffineGramCoordinates:
    return AffineGramCoordinates(
        SymmetryReducedPositiveExtension(
            active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
        )
    )


@pytest.fixture(scope="module")
def prepared():
    settings = CheapProjectionSettings(
        policy="dykstra", target="both", dykstra_iterations=30
    )
    route, state, frontier, energy = prepare_cheap_projection_route(
        DimerParameters(), phonon_cutoff=16, settings=settings
    )
    return route, state, frontier, energy


def test_hermitian_features_round_trip_and_isometry() -> None:
    generator = np.random.default_rng(7)
    matrix = generator.normal(size=(6, 6)) + 1j * generator.normal(size=(6, 6))
    hermitian = 0.5 * (matrix + matrix.conjugate().T)
    features = _hermitian_features(hermitian)
    assert features.shape == (36,)
    assert np.allclose(_hermitian_from_features(features, 6), hermitian)
    assert np.isclose(
        np.linalg.norm(features), np.linalg.norm(hermitian, ord="fro")
    )


def test_clip_to_floor_raises_every_eigenvalue() -> None:
    generator = np.random.default_rng(11)
    matrix = generator.normal(size=(5, 5)) + 1j * generator.normal(size=(5, 5))
    hermitian = 0.5 * (matrix + matrix.conjugate().T)
    clipped = _clip_to_floor(hermitian, 1e-3)
    assert np.linalg.eigvalsh(clipped)[0] >= 1e-3 - 1e-12
    assert np.allclose(clipped, clipped.conjugate().T)


def test_extension_gram_is_affine_in_the_propagated_coordinates(
    coordinates: AffineGramCoordinates,
) -> None:
    generator = np.random.default_rng(3)
    state = generator.normal(scale=0.05, size=coordinates.state_count)
    frontier = generator.normal(scale=0.05, size=coordinates.frontier_count)
    lower = state_lower_moments(
        state[:29], state[29:], coordinates.active_keys
    )
    direct = coordinates.extension.scaled_matrix(
        coordinates.extension.matrix(
            lower, coordinates.frontier_mapping(frontier)
        )
    )
    assert np.allclose(
        coordinates.scaled_extension_matrix(state, frontier),
        direct,
        atol=1e-13,
        rtol=0.0,
    )


def test_retained_gram_is_affine_in_the_raw_coordinates(
    coordinates: AffineGramCoordinates,
) -> None:
    generator = np.random.default_rng(5)
    state = generator.normal(scale=0.05, size=coordinates.state_count)
    direct = uncentered_joint_moment_matrix(state[:29])
    assert np.allclose(
        coordinates.unshifted_retained_matrix(state),
        direct,
        atol=1e-13,
        rtol=0.0,
    )


def test_dykstra_reaches_the_cone_where_one_clip_does_not(
    coordinates: AffineGramCoordinates, prepared
) -> None:
    _, state, frontier, _ = prepared
    generator = np.random.default_rng(0)
    perturbed = state + 1e-3 * generator.normal(size=state.size)
    clip = coordinates.repair(
        perturbed,
        frontier,
        CheapProjectionSettings(policy="eigenvalue_clip", target="both"),
    )
    dykstra = coordinates.repair(
        perturbed,
        frontier,
        CheapProjectionSettings(
            policy="dykstra", target="both", dykstra_iterations=200
        ),
    )
    assert clip.applied and dykstra.applied
    assert clip.extension_minimum_after > clip.extension_minimum_before
    assert clip.extension_minimum_after < -1e-8
    assert dykstra.extension_minimum_after > -1e-12
    assert dykstra.retained_minimum_after > -1e-12


def test_higham_policy_uses_the_zero_floor() -> None:
    settings = CheapProjectionSettings(
        policy="higham_nearest_psd", eigenvalue_floor=0.5
    )
    assert settings.applied_floor == 0.0
    assert settings.applied_retained_floor == 0.0


def test_policy_none_moves_nothing(
    coordinates: AffineGramCoordinates, prepared
) -> None:
    _, state, frontier, _ = prepared
    result = coordinates.repair(
        state, frontier, CheapProjectionSettings(policy="none")
    )
    assert not result.applied
    assert np.array_equal(result.state, state)
    assert np.array_equal(result.frontier_standardized, frontier)


def test_frontier_only_holds_the_propagated_state_fixed(
    coordinates: AffineGramCoordinates, prepared
) -> None:
    _, state, frontier, _ = prepared
    generator = np.random.default_rng(1)
    perturbed = state + 1e-3 * generator.normal(size=state.size)
    result = coordinates.repair(
        perturbed,
        frontier,
        CheapProjectionSettings(
            policy="dykstra",
            target="extension",
            dykstra_iterations=20,
            repair_coordinates="frontier_only",
        ),
    )
    assert result.state_displacement == 0.0
    assert np.array_equal(result.state, perturbed)
    assert result.frontier_displacement > 0.0


def test_frontier_only_cannot_target_the_retained_gram() -> None:
    with pytest.raises(ValueError, match="frontier_only"):
        CheapProjectionSettings(
            target="retained", repair_coordinates="frontier_only"
        )


def test_settings_reject_unknown_names() -> None:
    with pytest.raises(ValueError, match="policy"):
        CheapProjectionSettings(policy="nearest")
    with pytest.raises(ValueError, match="target"):
        CheapProjectionSettings(target="unified")
    with pytest.raises(ValueError, match="cadence"):
        CheapProjectionSettings(cadence="atom")


def test_carried_lift_matches_the_state_lower_moments(prepared) -> None:
    route, state, frontier, _ = prepared
    completion = route.carried_completion(state, frontier)
    raw, hidden = unpack_projected_apcm_state(
        state, active_keys=route.coordinates.active_keys
    )
    lower = state_lower_moments(raw, hidden, route.coordinates.active_keys)
    for key in route.model.extension.lower_keys:
        assert completion.lower_moments[key] == float(lower[key])
    assert completion.success


def test_unguarded_velocity_ignores_the_tangent_cone_projection(
    prepared,
) -> None:
    """The velocity is the unconstrained predictor, not the projected one."""

    route, state, frontier, _ = prepared
    completion = route.carried_completion(state, frontier)
    targets = route.model.targets(0.0, state, completion=completion)
    velocity, _ = route.unguarded_velocity(0.0, state, frontier)
    assert np.array_equal(
        velocity,
        np.concatenate(
            (targets.augmented_retained_velocity, targets.auxiliary_velocity)
        ),
    )


def test_short_rollout_completes_and_keeps_both_grams_at_the_floor(
    prepared,
) -> None:
    route, state, frontier, _ = prepared
    trajectory = integrate_cheap_projection_ssprk3(
        route,
        state,
        frontier,
        final_time=0.005,
        time_step=0.0025,
    )
    assert trajectory.success
    assert trajectory.completed_steps == 2
    assert trajectory.velocity_evaluations == 6
    assert float(np.min(trajectory.scaled_extension_minimum_eigenvalues)) > -1e-9
    assert float(np.min(trajectory.retained_minimum_eigenvalues)) > -1e-9


def test_step_cadence_repairs_once_per_step(prepared) -> None:
    route, state, frontier, _ = prepared
    stepwise = route.__class__(
        route.model,
        CheapProjectionSettings(
            policy="dykstra",
            target="both",
            cadence="step",
            dykstra_iterations=10,
        ),
        coordinates=route.coordinates,
    )
    trajectory = integrate_cheap_projection_ssprk3(
        stepwise, state, frontier, final_time=0.0075, time_step=0.0025
    )
    assert trajectory.success
    assert trajectory.repairs_applied + trajectory.repairs_skipped == 3


def test_rollout_rejects_a_step_that_does_not_divide_the_interval(
    prepared,
) -> None:
    route, state, frontier, _ = prepared
    with pytest.raises(ValueError, match="integer multiple"):
        integrate_cheap_projection_ssprk3(
            route, state, frontier, final_time=0.004, time_step=0.0025
        )


def test_unified_62_row_gram_is_affine_in_the_carried_coordinates() -> None:
    from paper5.stability.apcm_carried_witness import CarriedWitnessGeometry

    geometry = CarriedWitnessGeometry()
    gram = CarriedWitnessAffineGram(geometry)
    assert gram.rows == 62
    assert gram.state_count == geometry.state_count
    generator = np.random.default_rng(13)
    probe = generator.normal(scale=0.05, size=gram.state_count)
    direct = geometry.scaled_unified_matrix(*geometry.unpack_state(probe))
    assert np.allclose(
        gram.scaled_unified_matrix(probe), direct, atol=1e-13, rtol=0.0
    )
    perturbed = probe.copy()
    repaired = gram.repair(
        perturbed,
        CheapProjectionSettings(
            policy="dykstra", target="extension", dykstra_iterations=200
        ),
    )
    assert repaired.applied
    assert repaired.extension_minimum_after > repaired.extension_minimum_before
