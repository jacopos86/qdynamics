from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from paper5.stability.exact_reference import (
    _normal_ordered_weyl_coefficients,
    exact_holstein_joint_moment_initial_state,
)

from paper5.stability.adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    HIDDEN_RELATIVE_MOMENT_KEYS,
    RAW_MOMENT_COORDINATE_NAMES,
    kpd_correlation_velocity_correction,
    matrix_derivative_to_raw_moment_velocity,
    matrix_state_to_raw_moment_coordinates,
)
from paper5.stability.apcm_moment_projection import (
    APCMStageRetraction,
    SymmetryReducedAPCMGeometry,
    state_lower_moments,
)
from paper5.stability.apcm_positive_extension import (
    APCMExtensionSettings,
    SymmetryReducedPositiveExtension,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import MatrixDimerState
from paper5.stability.moment_hierarchy import (
    IDENTITY,
    MomentKey,
    THIRD_ORDER_HIERARCHY,
)
from paper5.stability.projected_apcm import (
    ENTRANCE_PROJECTED_APCM_STATE_NAMES,
    PROJECTED_APCM_STATE_NAMES,
    FixedDictionaryProjectedAPCM,
    ProjectedAPCMFailure,
    ProjectedAPCMSettings,
    integrate_projected_apcm_ssprk3,
    pack_projected_apcm_state,
    prepare_projected_apcm_initial_state,
    projected_apcm_partial_trajectory,
    projected_apcm_stage_failure,
    unpack_projected_apcm_state,
)

import pytest
def _thermal_moment(key: MomentKey) -> float:
    if key.spin_up != IDENTITY or key.spin_down != IDENTITY:
        return 0.0
    if key.x_power % 2 or key.p_power % 2:
        return 0.0

    def odd_double_factorial(power: int) -> int:
        return int(np.prod(np.arange(1, power, 2))) if power else 1

    return float(
        odd_double_factorial(key.x_power)
        * odd_double_factorial(key.p_power)
    )


def test_canonical_embedding_normal_orders_x_squared() -> None:
    coefficients = {
        (creator, annihilator): value
        for creator, annihilator, value in (
            _normal_ordered_weyl_coefficients(2, 0)
        )
    }

    assert set(coefficients) == {(0, 0), (0, 2), (1, 1), (2, 0)}
    np.testing.assert_allclose(
        [coefficients[key] for key in sorted(coefficients)],
        [0.5, 0.5, 1.0, 0.5],
        atol=2e-15,
        rtol=0.0,
    )


def _thermal_state_and_completion():  # noqa: ANN202
    moments = {
        key: _thermal_moment(key)
        for key in THIRD_ORDER_HIERARCHY.moment_keys
    }
    matrix_state = THIRD_ORDER_HIERARCHY.to_matrix_state(
        THIRD_ORDER_HIERARCHY.pack(0.0j, moments)
    )
    raw = matrix_state_to_raw_moment_coordinates(matrix_state)
    hidden = {
        key: moments[key] for key in HIDDEN_RELATIVE_MOMENT_KEYS
    }
    state = pack_projected_apcm_state(raw, hidden)
    extension = SymmetryReducedPositiveExtension(
        APCMExtensionSettings(
            logdet_weight=1e-3,
            conic_tolerance=1e-7,
            maximum_iterations=400,
        )
    )
    model = FixedDictionaryProjectedAPCM(
        DimerParameters(lambda_ep=1.5),
        extension=extension,
    )
    completion = model.select_completion(
        state,
        warm_frontier={
            key: _thermal_moment(key) for key in extension.frontier_keys
        },
    )
    return state, model, completion


def test_projected_targets_add_the_entrance_only_to_c() -> None:
    state, model, completion = _thermal_state_and_completion()

    targets = model.targets(0.3, state, completion=completion)

    for block in (
        "electron_density",
        "coherent_phonon",
        "phonon_density",
        "anomalous_phonon_density",
    ):
        np.testing.assert_allclose(
            getattr(targets.augmented_matrix_velocity, block),
            getattr(targets.archive_matrix_velocity, block),
            atol=1e-13,
            rtol=0.0,
        )
    correlation_increment = (
        targets.augmented_matrix_velocity.electron_phonon_correlation
        - targets.archive_matrix_velocity.electron_phonon_correlation
    )
    np.testing.assert_allclose(
        correlation_increment,
        targets.applied_correlation_velocity_increment,
        atol=1e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.trace(correlation_increment, axis1=1, axis2=2),
        0.0,
        atol=1e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        model.unprojected_velocity_with_frontier(
            0.3,
            state,
            completion.frontier_moments,
        ),
        np.concatenate(
            (
                targets.augmented_retained_velocity,
                targets.auxiliary_velocity,
            )
        ),
        atol=2e-12,
        rtol=0.0,
    )


def test_disabling_entrance_recovers_the_archive_target() -> None:
    state, original, completion = _thermal_state_and_completion()
    model = FixedDictionaryProjectedAPCM(
        original.parameters,
        extension=original.extension,
        geometry=original.geometry,
        settings=ProjectedAPCMSettings(
            include_k=False,
            include_pauli=False,
            include_opposite_spin=False,
        ),
    )

    targets = model.targets(0.3, state, completion=completion)

    np.testing.assert_allclose(
        targets.augmented_retained_velocity,
        targets.archive_retained_velocity,
        atol=1e-13,
        rtol=0.0,
    )


def test_default_model_uses_only_the_reduced_tu_entrance_dictionary() -> None:
    model = FixedDictionaryProjectedAPCM(DimerParameters(lambda_ep=1.5))
    assert model.active_keys == ENTRANCE_RELATIVE_MOMENT_KEYS
    assert model.state_names == ENTRANCE_PROJECTED_APCM_STATE_NAMES
    assert model.extension.dimension == 24
    assert model.geometry.direct_retraction_estimate_bytes < 16 * 1024**2


def test_reduced_tu_entrance_reproduces_the_full_kpd_source() -> None:
    parameters = DimerParameters(lambda_ep=1.5)
    exact = exact_holstein_joint_moment_initial_state(
        parameters,
        hierarchy=THIRD_ORDER_HIERARCHY,
        phonon_cutoff=8,
        canonical_embedding=True,
    )
    raw = matrix_state_to_raw_moment_coordinates(exact.matrix_state)
    _, full_moments = THIRD_ORDER_HIERARCHY.unpack(
        exact.hierarchy_coordinates
    )
    entrance = np.asarray(
        [full_moments[key] for key in ENTRANCE_RELATIVE_MOMENT_KEYS],
        dtype=float,
    )
    reduced_moments = state_lower_moments(
        raw,
        entrance,
        ENTRANCE_RELATIVE_MOMENT_KEYS,
    )

    np.testing.assert_allclose(
        kpd_correlation_velocity_correction(
            exact.matrix_state,
            parameters,
            reduced_moments,
        ),
        kpd_correlation_velocity_correction(
            exact.matrix_state,
            parameters,
            full_moments,
        ),
        atol=3e-13,
        rtol=0.0,
    )


def test_frozen_hamiltonian_gradient_and_kpd_work_flux() -> None:
    parameters = DimerParameters(lambda_ep=1.5)
    exact = exact_holstein_joint_moment_initial_state(
        parameters,
        hierarchy=THIRD_ORDER_HIERARCHY,
        phonon_cutoff=8,
        canonical_embedding=True,
    )
    raw = matrix_state_to_raw_moment_coordinates(exact.matrix_state)
    _, moments = THIRD_ORDER_HIERARCHY.unpack(exact.hierarchy_coordinates)
    entrance = np.asarray(
        [moments[key] for key in ENTRANCE_RELATIVE_MOMENT_KEYS],
        dtype=float,
    )
    reduced_moments = state_lower_moments(
        raw,
        entrance,
        ENTRANCE_RELATIVE_MOMENT_KEYS,
    )
    extension = SymmetryReducedPositiveExtension(
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    geometry = SymmetryReducedAPCMGeometry(extension)
    time = 0.37
    gradient = geometry.frozen_hamiltonian_gradient(time, parameters)

    def raw_energy(coordinates: np.ndarray) -> float:
        return float(
            -2.0 * parameters.hopping * coordinates[0]
            + parameters.drive_difference(time) * coordinates[2]
            + 2.0
            * parameters.coupling
            * (
                coordinates[3]
                + coordinates[5]
                + coordinates[21]
                - coordinates[27]
            )
            + parameters.omega_ph * (coordinates[7] + coordinates[8])
        )

    finite_difference = np.empty_like(raw)
    step = 1e-7
    for index in range(raw.size):
        offset = np.zeros_like(raw)
        offset[index] = step
        finite_difference[index] = (
            raw_energy(raw + offset) - raw_energy(raw - offset)
        ) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, atol=5e-9, rtol=0.0)

    kpd = kpd_correlation_velocity_correction(
        exact.matrix_state,
        parameters,
        reduced_moments,
    )
    zero = np.zeros((2, 2), dtype=complex)
    kpd_matrix_velocity = MatrixDimerState(
        electron_density=zero,
        coherent_phonon=np.zeros(2, dtype=complex),
        phonon_density=zero,
        anomalous_phonon_density=zero,
        electron_phonon_correlation=kpd,
    )
    kpd_raw_velocity = matrix_derivative_to_raw_moment_velocity(
        exact.matrix_state,
        kpd_matrix_velocity,
    )
    assert abs(float(gradient @ kpd_raw_velocity)) < 3e-12


def test_entrance_state_pack_round_trip_has_44_real_coordinates() -> None:
    raw = np.arange(len(RAW_MOMENT_COORDINATE_NAMES), dtype=float)
    hidden = {
        key: float(index)
        for index, key in enumerate(ENTRANCE_RELATIVE_MOMENT_KEYS)
    }
    packed = pack_projected_apcm_state(
        raw,
        hidden,
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS,
    )
    reconstructed_raw, reconstructed_hidden = unpack_projected_apcm_state(
        packed,
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS,
    )
    assert packed.shape == (44,)
    np.testing.assert_array_equal(reconstructed_raw, raw)
    np.testing.assert_array_equal(
        reconstructed_hidden,
        np.arange(len(ENTRANCE_RELATIVE_MOMENT_KEYS), dtype=float),
    )


def test_ssprk3_contains_every_trial_before_the_next_rhs() -> None:
    events: list[str] = []
    completion = SimpleNamespace(
        frontier_moments={},
        scaled_minimum_eigenvalue=1.0,
    )
    retained_metric = np.eye(len(RAW_MOMENT_COORDINATE_NAMES))
    auxiliary_metric = np.eye(len(HIDDEN_RELATIVE_MOMENT_KEYS))

    class FakeModel:
        active_keys = HIDDEN_RELATIVE_MOMENT_KEYS
        state_names = PROJECTED_APCM_STATE_NAMES

        def select_completion(self, state):  # noqa: ANN001, ANN202
            return completion

        def evaluate(self, time, state, *, completion):  # noqa: ANN001, ANN202
            events.append("evaluate")
            projection = SimpleNamespace(
                retained_metric=retained_metric,
                auxiliary_metric=auxiliary_metric,
                retained_correction_norm=0.0,
                auxiliary_correction_norm=0.0,
            )
            targets = SimpleNamespace(completion=completion)
            return SimpleNamespace(
                derivative=np.zeros(len(PROJECTED_APCM_STATE_NAMES)),
                projection=projection,
                targets=targets,
            )

        def contain_stage(  # noqa: ANN202
            self,
            trial_state,
            *,
            warm_frontier,
            retained_metric,
            auxiliary_metric,
        ):
            events.append("contain")
            return APCMStageRetraction(
                raw_coordinates=np.asarray(
                    trial_state[: len(RAW_MOMENT_COORDINATE_NAMES)]
                ),
                hidden_values=np.asarray(
                    trial_state[len(RAW_MOMENT_COORDINATE_NAMES) :]
                ),
                completion=completion,
                retained_correction_norm=0.0,
                auxiliary_correction_norm=0.0,
                iterations=0,
                applied=False,
                status="test",
            )

    integrate_projected_apcm_ssprk3(
        FakeModel(),
        np.zeros(len(PROJECTED_APCM_STATE_NAMES)),
        initial_completion=completion,
        final_time=0.01,
        time_step=0.01,
    )

    assert events == [
        "evaluate",
        "contain",
        "evaluate",
        "contain",
        "evaluate",
        "contain",
    ]


def test_ssprk3_retries_a_declared_stage_failure_with_two_half_steps() -> None:
    events: list[str] = []
    fail_first_containment = True
    completion = SimpleNamespace(
        frontier_moments={},
        scaled_minimum_eigenvalue=1.0,
    )
    retained_metric = np.eye(len(RAW_MOMENT_COORDINATE_NAMES))
    auxiliary_metric = np.eye(len(HIDDEN_RELATIVE_MOMENT_KEYS))

    class FakeModel:
        active_keys = HIDDEN_RELATIVE_MOMENT_KEYS
        state_names = PROJECTED_APCM_STATE_NAMES

        def select_completion(self, state):  # noqa: ANN001, ANN202
            return completion

        def evaluate(self, time, state, *, completion):  # noqa: ANN001, ANN202
            events.append(f"evaluate:{time:.3f}")
            return SimpleNamespace(
                derivative=np.zeros(len(PROJECTED_APCM_STATE_NAMES)),
                projection=SimpleNamespace(
                    retained_metric=retained_metric,
                    auxiliary_metric=auxiliary_metric,
                    retained_correction_norm=0.0,
                    auxiliary_correction_norm=0.0,
                ),
                targets=SimpleNamespace(completion=completion),
            )

        def contain_stage(  # noqa: ANN202
            self,
            trial_state,
            *,
            warm_frontier,
            retained_metric,
            auxiliary_metric,
        ):
            nonlocal fail_first_containment
            events.append("contain")
            if fail_first_containment:
                fail_first_containment = False
                raise ProjectedAPCMFailure("declared test face failure")
            return APCMStageRetraction(
                raw_coordinates=np.asarray(
                    trial_state[: len(RAW_MOMENT_COORDINATE_NAMES)]
                ),
                hidden_values=np.asarray(
                    trial_state[len(RAW_MOMENT_COORDINATE_NAMES) :]
                ),
                completion=completion,
                retained_correction_norm=0.0,
                auxiliary_correction_norm=0.0,
                iterations=0,
                applied=False,
                status="test",
            )

    trajectory = integrate_projected_apcm_ssprk3(
        FakeModel(),
        np.zeros(len(PROJECTED_APCM_STATE_NAMES)),
        initial_completion=completion,
        final_time=0.01,
        time_step=0.01,
        maximum_subdivisions=1,
    )

    assert trajectory.accepted_substeps.tolist() == [0, 2]
    assert trajectory.rhs_evaluations == 7
    assert events[0:2] == ["evaluate:0.000", "contain"]
    assert events[2:] == [
        "evaluate:0.000",
        "contain",
        "evaluate:0.005",
        "contain",
        "evaluate:0.003",
        "contain",
        "evaluate:0.005",
        "contain",
        "evaluate:0.010",
        "contain",
        "evaluate:0.007",
        "contain",
    ]



def _stub_rollout_model(fail_on_contain: int | None = None):
    """A model whose stages are free, so only the rollout logic is exercised."""

    completion = SimpleNamespace(
        frontier_moments={},
        scaled_minimum_eigenvalue=1.0,
    )
    retained_metric = np.eye(len(RAW_MOMENT_COORDINATE_NAMES))
    auxiliary_metric = np.eye(len(HIDDEN_RELATIVE_MOMENT_KEYS))
    calls = {"contain": 0}

    class StubModel:
        active_keys = HIDDEN_RELATIVE_MOMENT_KEYS
        state_names = PROJECTED_APCM_STATE_NAMES

        def select_completion(self, state):  # noqa: ANN001, ANN202
            return completion

        def evaluate(self, time, state, *, completion=completion):  # noqa: ANN001, ANN202
            return SimpleNamespace(
                derivative=np.ones(len(PROJECTED_APCM_STATE_NAMES)),
                projection=SimpleNamespace(
                    retained_metric=retained_metric,
                    auxiliary_metric=auxiliary_metric,
                    retained_correction_norm=0.0,
                    auxiliary_correction_norm=0.0,
                ),
                targets=SimpleNamespace(completion=completion),
            )

        def contain_stage(  # noqa: ANN202
            self,
            trial_state,
            *,
            warm_frontier,
            retained_metric,
            auxiliary_metric,
        ):
            calls["contain"] += 1
            if fail_on_contain is not None and calls["contain"] == fail_on_contain:
                raise ProjectedAPCMFailure("forced test stage failure")
            return APCMStageRetraction(
                raw_coordinates=np.asarray(
                    trial_state[: len(RAW_MOMENT_COORDINATE_NAMES)]
                ),
                hidden_values=np.asarray(
                    trial_state[len(RAW_MOMENT_COORDINATE_NAMES) :]
                ),
                completion=completion,
                retained_correction_norm=0.0,
                auxiliary_correction_norm=0.0,
                iterations=0,
                applied=False,
                status="test",
            )

    return StubModel(), completion


def test_a_failed_rollout_carries_the_steps_it_accepted() -> None:
    """A forced failure in the third step keeps the first two steps."""

    model, completion = _stub_rollout_model(fail_on_contain=7)
    with pytest.raises(ProjectedAPCMFailure) as raised:
        integrate_projected_apcm_ssprk3(
            model,
            np.zeros(len(PROJECTED_APCM_STATE_NAMES)),
            initial_completion=completion,
            final_time=0.02,
            time_step=0.005,
            maximum_subdivisions=0,
        )
    partial = projected_apcm_partial_trajectory(raised.value)
    assert partial is not None
    assert partial.completed_steps == 2
    assert not partial.success
    assert partial.message.startswith("stage_failure after 2 of 4")
    assert partial.times.tolist() == [0.0, 0.005, 0.01]
    assert partial.states.shape == (3, len(PROJECTED_APCM_STATE_NAMES))
    assert np.allclose(partial.states[1], 0.005)


def test_the_failing_stage_carries_its_own_lower_moments() -> None:
    """A declared completion failure records where it was asked to lift."""

    parameters = DimerParameters(lambda_ep=1.5)
    model = FixedDictionaryProjectedAPCM(parameters)
    initialization = prepare_projected_apcm_initial_state(
        parameters, phonon_cutoff=6
    )
    state = initialization.state
    evaluation = model.evaluate(
        0.0, state, completion=initialization.completion
    )
    trial = state + 0.0025 * evaluation.derivative
    complete = model.extension.complete

    def refuse(lower, *, warm_frontier=None):  # noqa: ANN001, ANN202
        raise ProjectedAPCMFailure("forced completion refusal")

    model.extension.complete = refuse
    try:
        with pytest.raises(ProjectedAPCMFailure) as raised:
            model.contain_stage(
                trial,
                warm_frontier=evaluation.targets.completion.frontier_moments,
                retained_metric=evaluation.projection.retained_metric,
                auxiliary_metric=evaluation.projection.auxiliary_metric,
            )
    finally:
        model.extension.complete = complete
    failure = projected_apcm_stage_failure(raised.value)
    assert failure is not None
    assert failure.stage == "contain"
    assert tuple(failure.lower_moments) == model.extension.lower_keys
    assert np.array_equal(failure.state, trial)
    assert failure.warm_frontier_moments is not None


def test_initial_time_default_reproduces_the_recorded_sample_times_bitwise() -> None:
    """``initial_time`` at its default changes no sample time."""

    model, completion = _stub_rollout_model()
    recorded = integrate_projected_apcm_ssprk3(
        model,
        np.zeros(len(PROJECTED_APCM_STATE_NAMES)),
        initial_completion=completion,
        final_time=0.02,
        time_step=0.005,
    )
    model, completion = _stub_rollout_model()
    declared = integrate_projected_apcm_ssprk3(
        model,
        np.zeros(len(PROJECTED_APCM_STATE_NAMES)),
        initial_completion=completion,
        final_time=0.02,
        time_step=0.005,
        initial_time=0.0,
    )
    np.testing.assert_array_equal(recorded.times, declared.times)
    np.testing.assert_array_equal(recorded.states, declared.states)

    model, completion = _stub_rollout_model()
    continued = integrate_projected_apcm_ssprk3(
        model,
        np.zeros(len(PROJECTED_APCM_STATE_NAMES)),
        initial_completion=completion,
        final_time=4.02,
        time_step=0.005,
        initial_time=4.0,
    )
    assert continued.times[0] == 4.0
    assert continued.times[-1] == 4.02
    assert continued.completed_steps == 4


def test_a_tolerated_failure_still_reports_the_failing_stage() -> None:
    """``tolerate_stage_failure`` keeps the stage record on the trajectory."""

    model, completion = _stub_rollout_model(fail_on_contain=7)
    trajectory = integrate_projected_apcm_ssprk3(
        model,
        np.zeros(len(PROJECTED_APCM_STATE_NAMES)),
        initial_completion=completion,
        final_time=0.02,
        time_step=0.005,
        maximum_subdivisions=0,
        tolerate_stage_failure=True,
    )
    assert not trajectory.success
    assert trajectory.completed_steps == 2
    assert trajectory.message.startswith("stage_failure after 2 of 4")
    assert trajectory.stage_failure is None


def test_a_completed_rollout_reports_no_failing_stage() -> None:
    model, completion = _stub_rollout_model()
    trajectory = integrate_projected_apcm_ssprk3(
        model,
        np.zeros(len(PROJECTED_APCM_STATE_NAMES)),
        initial_completion=completion,
        final_time=0.02,
        time_step=0.005,
    )
    assert trajectory.success
    assert trajectory.stage_failure is None


def test_a_tolerated_failure_keeps_the_declared_models_stage_record() -> None:
    """The real model's stage record survives ``tolerate_stage_failure``."""

    parameters = DimerParameters(lambda_ep=1.5)
    model = FixedDictionaryProjectedAPCM(parameters)
    initialization = prepare_projected_apcm_initial_state(
        parameters, phonon_cutoff=6
    )
    complete = model.extension.complete
    calls = {"n": 0}

    def refuse_the_fourth(lower, *, warm_frontier=None):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 4:
            raise ProjectedAPCMFailure("forced completion refusal")
        return complete(lower, warm_frontier=warm_frontier)

    model.extension.complete = refuse_the_fourth
    try:
        trajectory = integrate_projected_apcm_ssprk3(
            model,
            initialization.state,
            initial_completion=initialization.completion,
            final_time=0.005,
            time_step=0.0025,
            maximum_subdivisions=0,
            tolerate_stage_failure=True,
        )
    finally:
        model.extension.complete = complete
    assert not trajectory.success
    assert "forced completion refusal" in trajectory.message
    assert trajectory.stage_failure is not None
    assert trajectory.stage_failure.stage == "contain"
    assert (
        tuple(trajectory.stage_failure.lower_moments)
        == model.extension.lower_keys
    )
