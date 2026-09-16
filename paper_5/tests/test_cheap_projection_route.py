"""Tests for the cheap physicality repair with no conic solve."""

from __future__ import annotations

from dataclasses import replace

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
    CheapProjectionRoute,
    CheapProjectionSettings,
    InternalEnergyFunctional,
    _clip_to_floor,
    _hermitian_features,
    _hermitian_from_features,
    integrate_cheap_projection_ssprk3,
    prepare_cheap_projection_route,
)
from paper5.stability.matrix_reference import (
    closed_scalar_to_matrix_state,
    matrix_state_to_closed_scalar_coordinates,
    matrix_total_energy,
)
from paper5.stability.adaptive_positive_moment import (
    raw_moment_coordinates_to_matrix_state,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.projected_apcm import (
    build_completion_ladder,
    unpack_projected_apcm_state,
)


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


def test_lift_reselection_interval_zero_carries_the_preparation_lift(
    prepared,
) -> None:
    """The route's original form: one completion at t=0 and none after."""

    route, state, frontier, _ = prepared
    trajectory = integrate_cheap_projection_ssprk3(
        route,
        state,
        frontier,
        final_time=0.005,
        time_step=0.0025,
        lift_reselection_interval=0,
    )
    assert trajectory.success
    assert trajectory.lift_reselections == 0
    assert trajectory.lift_reselection_failures == 0
    assert trajectory.lift_reselection_seconds == 0.0
    assert trajectory.maximum_lift_reselection_displacement == 0.0


def test_lift_reselection_every_step_solves_once_per_step(prepared) -> None:
    """The hybrid at N=1: one declared completion per accepted step."""

    route, state, frontier, _ = prepared
    trajectory = integrate_cheap_projection_ssprk3(
        route,
        state,
        frontier,
        final_time=0.005,
        time_step=0.0025,
        lift_reselection_interval=1,
        record_stages=True,
    )
    assert trajectory.success
    assert trajectory.lift_reselections == 2
    assert trajectory.lift_reselection_failures == 0
    assert trajectory.lift_reselection_seconds > 0.0
    assert (
        trajectory.minimum_reselected_scaled_extension_eigenvalue > -1e-9
    )
    reselections = [
        record
        for record in trajectory.records
        if record["stage"] == "lift_reselection"
    ]
    assert len(reselections) == 2
    assert all(record["applied"] for record in reselections)


def test_lift_reselection_at_a_cadence_solves_on_that_cadence(
    prepared,
) -> None:
    route, state, frontier, _ = prepared
    trajectory = integrate_cheap_projection_ssprk3(
        route,
        state,
        frontier,
        final_time=0.01,
        time_step=0.0025,
        lift_reselection_interval=4,
    )
    assert trajectory.success
    assert trajectory.completed_steps == 4
    assert trajectory.lift_reselections == 1


def test_lift_reselection_without_a_repair_moves_the_lift_only_on_cadence(
    prepared,
) -> None:
    """The attribution ablation: re-selection only, no repair in between.

    With ``policy="none"`` nothing is repaired at any stage, so the carried
    lift can change only where the declared proximal log-det completion is
    re-solved, and every repair opportunity is counted as skipped.
    """

    route, state, frontier, _ = prepared
    ablation = CheapProjectionRoute(
        route.model,
        CheapProjectionSettings(policy="none", target="both"),
        coordinates=route.coordinates,
    )
    trajectory = integrate_cheap_projection_ssprk3(
        ablation,
        state,
        frontier,
        final_time=0.01,
        time_step=0.0025,
        lift_reselection_interval=2,
    )
    assert trajectory.success
    assert trajectory.completed_steps == 4
    assert trajectory.repairs_applied == 0
    assert trajectory.repairs_skipped == 12
    assert trajectory.maximum_state_displacement == 0.0
    assert trajectory.maximum_frontier_displacement == 0.0
    assert trajectory.lift_reselections == 2
    assert trajectory.lift_reselection_failures == 0
    lifts = trajectory.frontier_standardized
    assert np.array_equal(lifts[1], lifts[0])
    assert not np.array_equal(lifts[2], lifts[1])
    assert np.array_equal(lifts[3], lifts[2])
    assert not np.array_equal(lifts[4], lifts[3])


def test_reselect_lift_returns_the_declared_completion_frontier(
    prepared,
) -> None:
    """The re-selected lift is the completion's own frontier, standardized."""

    route, state, frontier, _ = prepared
    perturbed = 0.5 * np.asarray(frontier, dtype=float)
    selected, completion, seconds = route.reselect_lift(
        state, perturbed, warm_start=True
    )
    assert completion.success
    assert seconds > 0.0
    assert np.allclose(
        selected,
        route.coordinates.standardize_frontier(completion.frontier_moments),
    )
    assert completion.scaled_minimum_eigenvalue > -1e-9
    assert not np.allclose(selected, perturbed)


def test_negative_lift_reselection_interval_is_rejected(prepared) -> None:
    route, state, frontier, _ = prepared
    with pytest.raises(ValueError, match="lift_reselection_interval"):
        integrate_cheap_projection_ssprk3(
            route,
            state,
            frontier,
            final_time=0.0025,
            time_step=0.0025,
            lift_reselection_interval=-1,
        )


def test_internal_energy_is_affine_in_the_propagated_coordinates() -> None:
    """Eq. (22) on the coordinates the scorer uses is one linear functional.

    The gradient vanishes on the 15 entrance coordinates and on the six
    anomalous raw coordinates, and equals ``omega_ph`` on each diagonal raw
    normal coordinate, which is the statement that a correction confined to
    the phonon normal and anomalous moments changes the energy by
    ``omega_ph`` times the trace of the normal correction.
    """

    parameters = DimerParameters(gamma=0.5, lambda_ep=1.5, drive_amplitude=1.0)
    functional = InternalEnergyFunctional(parameters, state_count=44)
    generator = np.random.default_rng(23)
    for _ in range(3):
        probe = generator.normal(scale=0.3, size=44)
        state = raw_moment_coordinates_to_matrix_state(probe[:29])
        closed = matrix_state_to_closed_scalar_coordinates(state)
        direct = matrix_total_energy(
            closed_scalar_to_matrix_state(closed), parameters
        )
        assert np.isclose(functional.value(probe), direct, atol=1e-12, rtol=0.0)
    assert np.all(functional.gradient[29:] == 0.0)
    assert np.all(functional.gradient[11:17] == 0.0)
    assert functional.gradient[7] == pytest.approx(parameters.omega_ph)
    assert functional.gradient[8] == pytest.approx(parameters.omega_ph)


def test_energy_neutral_repair_lands_on_the_declared_reference(
    prepared,
) -> None:
    """The third set of the alternation is met exactly, the cone is not lost."""

    route, state, frontier, _ = prepared
    coordinates = route.coordinates
    assert coordinates.energy is not None
    reference = coordinates.energy.value(state)
    generator = np.random.default_rng(31)
    perturbed = state + 1e-3 * generator.normal(size=state.size)
    free = coordinates.repair(
        perturbed,
        frontier,
        CheapProjectionSettings(
            policy="dykstra", target="both", dykstra_iterations=30
        ),
    )
    held = coordinates.repair(
        perturbed,
        frontier,
        CheapProjectionSettings(
            policy="dykstra",
            target="both",
            dykstra_iterations=30,
            energy_reference_mode="post_pulse",
        ),
        energy_reference=reference,
    )
    assert free.applied and held.applied
    assert held.energy_hyperplane_applied
    assert not free.energy_hyperplane_applied
    assert abs(held.internal_energy_after - reference) < 1e-10
    assert abs(free.internal_energy_after - reference) > 1e-6
    assert held.extension_minimum_after > held.extension_minimum_before
    assert held.state_displacement > free.state_displacement


def test_energy_reference_none_leaves_the_repair_unchanged(prepared) -> None:
    """The named option is off by default and changes nothing when it is."""

    route, state, frontier, _ = prepared
    coordinates = route.coordinates
    generator = np.random.default_rng(37)
    perturbed = state + 1e-3 * generator.normal(size=state.size)
    settings = CheapProjectionSettings(
        policy="dykstra", target="both", dykstra_iterations=30
    )
    assert not settings.holds_internal_energy
    without = coordinates.repair(perturbed, frontier, settings)
    ignored = coordinates.repair(
        perturbed,
        frontier,
        settings,
        energy_reference=coordinates.energy.value(state),
    )
    assert np.array_equal(without.state, ignored.state)
    assert np.array_equal(
        without.frontier_standardized, ignored.frontier_standardized
    )
    assert not ignored.energy_hyperplane_applied


def test_energy_reference_mode_needs_dykstra_and_the_state_coordinates(
) -> None:
    with pytest.raises(ValueError, match="dykstra"):
        CheapProjectionSettings(
            policy="higham_nearest_psd", energy_reference_mode="post_pulse"
        )
    with pytest.raises(ValueError, match="frontier_only"):
        CheapProjectionSettings(
            policy="dykstra",
            repair_coordinates="frontier_only",
            energy_reference_mode="drive_power",
        )
    with pytest.raises(ValueError, match="energy_reference_mode"):
        CheapProjectionSettings(
            policy="dykstra", energy_reference_mode="conserved"
        )


def test_post_pulse_reference_is_off_before_the_declared_time(
    prepared,
) -> None:
    """The constraint is off while the drive runs and holds after it ends."""

    route, state, frontier, _ = prepared
    held = CheapProjectionRoute(
        route.model,
        CheapProjectionSettings(
            policy="dykstra",
            target="both",
            dykstra_iterations=30,
            energy_reference_mode="post_pulse",
        ),
        coordinates=route.coordinates,
    )
    trajectory = integrate_cheap_projection_ssprk3(
        held,
        state,
        frontier,
        final_time=0.01,
        time_step=0.0025,
        energy_reference_time=0.005,
    )
    assert trajectory.success
    references = trajectory.internal_energy_reference
    assert np.all(np.isnan(references[:2]))
    assert not np.any(np.isnan(references[2:]))
    assert trajectory.energy_reference_value == pytest.approx(
        float(trajectory.internal_energy[2]), abs=1e-12
    )
    assert trajectory.maximum_absolute_energy_deviation < 1e-12
    assert trajectory.energy_constrained_repairs > 0
    assert trajectory.energy_hyperplane_refusals == 0


def test_drive_power_reference_follows_the_drive(prepared) -> None:
    """The power is available without the exact state: f(0) is exactly zero.

    The Hamiltonian's only time dependence is the drive difference, and that
    difference vanishes at ``t = 0``, so the velocity evaluated at ``t = 0``
    is the drive-free velocity of the same coordinates and the difference of
    the two contracted with the energy gradient is the power the drive
    delivers.
    """

    route, state, frontier, _ = prepared
    parameters = route.model.parameters
    assert parameters.drive_difference(0.0) == 0.0
    assert abs(parameters.drive_difference(1.0)) > 0.5
    tracked = CheapProjectionRoute(
        route.model,
        CheapProjectionSettings(
            policy="dykstra",
            target="both",
            dykstra_iterations=30,
            energy_reference_mode="drive_power",
        ),
        coordinates=route.coordinates,
    )
    trajectory = integrate_cheap_projection_ssprk3(
        tracked,
        state,
        frontier,
        initial_time=1.0,
        final_time=1.01,
        time_step=0.0025,
    )
    assert trajectory.success
    references = trajectory.internal_energy_reference
    assert not np.any(np.isnan(references))
    assert abs(references[-1] - references[0]) > 1e-6
    assert trajectory.maximum_absolute_energy_deviation < 1e-10
    assert trajectory.energy_constrained_repairs > 0


def test_completion_retry_ladder_is_the_records_declared_ladder(
    prepared,
) -> None:
    """The ladder is the record's own, one declared setting per entry.

    The shifts and the weight are the ones the projective route declares,
    ``1e-8`` and ``1e-7`` on the log-determinant shift and ``1e-1`` on the
    log-determinant weight, and the ladder this route builds is the one
    ``build_completion_ladder`` returns for them.
    """

    route, _, _, _ = prepared
    assert route.completion_ladder == ()
    laddered = CheapProjectionRoute(
        route.model,
        route.settings,
        coordinates=route.coordinates,
        completion_retry_shifts=(1e-8, 1e-7),
        completion_retry_weights=(1e-1,),
    )
    declared = build_completion_ladder(
        route.model.extension, shifts=(1e-8, 1e-7), weights=(1e-1,)
    )
    assert len(laddered.completion_ladder) == len(declared) == 3
    base = route.model.extension.settings
    shifts = [entry.settings.logdet_shift for entry in declared]
    weights = [entry.settings.logdet_weight for entry in declared]
    assert shifts == [1e-8, 1e-7, base.logdet_shift]
    assert weights == [base.logdet_weight, base.logdet_weight, 1e-1]
    for entry in declared:
        assert entry.active_keys == route.model.extension.active_keys


def _refusing_completion(declared):
    """The declared completion with its certificate withheld."""

    def refusing(lower, *, warm_frontier=None):
        return replace(
            declared(lower, warm_frontier=warm_frontier),
            success=False,
            message="test_forced_refusal",
        )

    return refusing


def test_a_refused_lift_reselection_walks_the_declared_ladder(
    prepared, monkeypatch
) -> None:
    """A refusal is retried, and the first certified retry is the lift.

    The declared completion is forced to refuse; the ladder entries are
    separate dictionaries and are left alone, so the re-selection returns
    the retried completion's own frontier and the refusal never reaches the
    caller.
    """

    route, state, frontier, _ = prepared
    laddered = CheapProjectionRoute(
        route.model,
        route.settings,
        coordinates=route.coordinates,
        completion_retry_shifts=(1e-8,),
        completion_retry_weights=(1e-1,),
    )
    monkeypatch.setattr(
        route.model.extension,
        "complete",
        _refusing_completion(route.model.extension.complete),
    )
    selected, completion, seconds = laddered.reselect_lift(state, frontier)
    assert completion.success
    assert seconds > 0.0
    assert laddered.completion_retry_counters["attempted"] == 1
    assert laddered.completion_retry_counters["accepted"] == 1
    assert laddered.completion_retry_counters["exhausted"] == 0
    assert laddered.completion_retry_counters["solves"] == 1
    assert laddered.completion_retry_seconds > 0.0
    assert np.allclose(
        selected,
        route.coordinates.standardize_frontier(completion.frontier_moments),
    )


def test_an_exhausted_ladder_keeps_the_carried_lift(
    prepared, monkeypatch
) -> None:
    """Every entry refusing is counted, and nothing stops."""

    route, state, frontier, _ = prepared
    laddered = CheapProjectionRoute(
        route.model,
        route.settings,
        coordinates=route.coordinates,
        completion_retry_shifts=(1e-8,),
    )
    monkeypatch.setattr(
        route.model.extension,
        "complete",
        _refusing_completion(route.model.extension.complete),
    )
    for entry in laddered.completion_ladder:
        monkeypatch.setattr(
            entry, "complete", _refusing_completion(entry.complete)
        )
    selected, completion, _ = laddered.reselect_lift(state, frontier)
    assert not completion.success
    assert laddered.completion_retry_counters["attempted"] == 1
    assert laddered.completion_retry_counters["accepted"] == 0
    assert laddered.completion_retry_counters["exhausted"] == 1
    assert np.array_equal(selected, np.asarray(frontier, dtype=float))


def test_energy_tolerance_skips_a_repair_the_cones_do_not_need(
    prepared,
) -> None:
    """A declared tolerance holds the hyperplane only where drift has grown.

    At the preparation both cones are met to the declared
    ``repair_tolerance``, so the only reason to repair is the energy.  A
    drift inside the declared ``energy_tolerance`` is left alone and counted
    as skipped; the same drift at tolerance zero is repaired and lands on
    the reference.
    """

    route, state, frontier, _ = prepared
    coordinates = route.coordinates
    reference = coordinates.energy.value(state) + 1e-6
    tolerant = CheapProjectionSettings(
        policy="dykstra",
        target="both",
        dykstra_iterations=30,
        repair_tolerance=1e-6,
        energy_reference_mode="post_pulse",
        energy_tolerance=1e-4,
    )
    exact = replace(tolerant, energy_tolerance=0.0)
    skipped = coordinates.repair(
        state, frontier, tolerant, energy_reference=reference
    )
    applied = coordinates.repair(
        state, frontier, exact, energy_reference=reference
    )
    assert not skipped.applied
    assert skipped.state_displacement == 0.0
    assert applied.applied
    assert abs(applied.internal_energy_after - reference) < 1e-10
    assert abs(skipped.internal_energy_after - reference) == pytest.approx(
        1e-6, rel=1e-6
    )


def test_a_supplied_reference_is_the_one_a_resumed_segment_holds(
    prepared,
) -> None:
    """The declared reference is carried into a segment, not re-declared.

    A segment resumed from a saved state would otherwise declare the
    reference at its own first sample, which with a nonzero
    ``energy_tolerance`` is the drifted value rather than the one the first
    segment selected.
    """

    route, state, frontier, _ = prepared
    held = CheapProjectionRoute(
        route.model,
        CheapProjectionSettings(
            policy="dykstra",
            target="both",
            dykstra_iterations=30,
            energy_reference_mode="post_pulse",
        ),
        coordinates=route.coordinates,
    )
    declared = route.coordinates.energy.value(state) - 1e-3
    supplied = integrate_cheap_projection_ssprk3(
        held,
        state,
        frontier,
        final_time=0.005,
        time_step=0.0025,
        energy_reference_time=0.0,
        energy_reference_value=declared,
    )
    own = integrate_cheap_projection_ssprk3(
        held,
        state,
        frontier,
        final_time=0.005,
        time_step=0.0025,
        energy_reference_time=0.0,
    )
    assert supplied.success and own.success
    assert supplied.energy_reference_supplied
    assert not own.energy_reference_supplied
    assert supplied.energy_reference_value == declared
    assert abs(own.energy_reference_value - declared) == pytest.approx(
        1e-3, rel=1e-6
    )
    assert abs(supplied.internal_energy[-1] - declared) < 1e-10
    with pytest.raises(ValueError, match="energy_reference_mode"):
        integrate_cheap_projection_ssprk3(
            route,
            state,
            frontier,
            final_time=0.0025,
            time_step=0.0025,
            energy_reference_value=declared,
        )


def test_two_chained_segments_reproduce_one_rollout(prepared) -> None:
    """A rollout stopped on its budget is continued from its saved state.

    The segment carries the last accepted state, the frontier carried with
    it, its time and the declared internal-energy reference, which is what
    the driver's ``--resume-from`` reads out of ``trajectory.npz`` and
    ``summary.json``.
    """

    route, state, frontier, _ = prepared
    held = CheapProjectionRoute(
        route.model,
        CheapProjectionSettings(
            policy="dykstra",
            target="both",
            dykstra_iterations=30,
            energy_reference_mode="post_pulse",
        ),
        coordinates=route.coordinates,
    )
    whole = integrate_cheap_projection_ssprk3(
        held,
        state,
        frontier,
        final_time=0.01,
        time_step=0.0025,
        energy_reference_time=0.0025,
        lift_reselection_interval=2,
    )
    first = integrate_cheap_projection_ssprk3(
        held,
        state,
        frontier,
        final_time=0.005,
        time_step=0.0025,
        energy_reference_time=0.0025,
        lift_reselection_interval=2,
    )
    second = integrate_cheap_projection_ssprk3(
        held,
        first.states[-1],
        first.frontier_standardized[-1],
        initial_time=float(first.times[-1]),
        final_time=0.01,
        time_step=0.0025,
        energy_reference_time=0.0025,
        energy_reference_value=first.energy_reference_value,
        lift_reselection_interval=2,
    )
    assert whole.success and first.success and second.success
    assert whole.completed_steps == 4
    assert first.completed_steps == second.completed_steps == 2
    assert second.energy_reference_value == first.energy_reference_value
    assert np.allclose(second.states[-1], whole.states[-1], atol=1e-12)
    assert np.allclose(
        second.frontier_standardized[-1],
        whole.frontier_standardized[-1],
        atol=1e-12,
    )
    assert (
        first.lift_reselections + second.lift_reselections
        == whole.lift_reselections
    )
