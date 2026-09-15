from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from paper5.stability.apcm_carried_witness import (
    CWRMFSettings,
    CarriedWitnessModel,
    cwrmf_settings_for_profile,
    integrate_cwrmf_ssprk2,
)
from paper5.stability.apcm_carried_witness_analysis import _accuracy_metrics
from paper5.stability.apcm_carried_witness_mode_audit import (
    negative_mode_coupling,
)
from paper5.stability.adaptive_positive_moment import (
    matrix_derivative_to_raw_moment_velocity,
    raw_moment_coordinates_to_matrix_state,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import matrix_dimer_rhs


@pytest.fixture(scope="module")
def prepared_model():
    model = CarriedWitnessModel(
        DimerParameters(
            lambda_ep=1.5,
            gamma=0.5,
            drive_amplitude=1.0,
        )
    )
    return model, model.prepare(phonon_cutoff=16)


def test_carried_witness_geometry_is_literal_62_row_lift(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    geometry = model.geometry

    assert geometry.retained_count == 44
    assert geometry.frontier_count == 238
    assert geometry.center_cross_count == 102
    assert geometry.completion_count == 340
    assert geometry.state_count == 384
    assert geometry.scaled_unified_matrix(
        *geometry.unpack_state(preparation.state)
    ).shape == (62, 62)
    assert geometry.readable_completion_indices.size == 231


def test_default_critical_bundle_has_no_artificial_cap() -> None:
    settings = CWRMFSettings()

    assert settings.maximum_critical_modes is None
    assert settings.critical_mode_limit == 61
    assert settings.bundle_solver_tolerance == 1e-12


def test_balanced_profile_declares_relaxed_full_gram_floor() -> None:
    settings = cwrmf_settings_for_profile("balanced")

    assert settings.psd_inflation == 1e-8
    assert settings.solver_tolerance == 1e-8
    assert settings.bundle_solver_tolerance == 1e-8
    assert settings.enable_full_cone_fallback
    assert settings.spectral_entry_threshold < settings.psd_inflation
    assert settings.maximum_critical_modes is None
    assert settings.critical_mode_limit == 61


def test_bundle_rank_hint_can_be_restored_for_continuation() -> None:
    model = CarriedWitnessModel(DimerParameters())

    assert model.bundle_rank_hint == 0
    model.restore_bundle_rank_hint(31)
    assert model.bundle_rank_hint == 31

    with pytest.raises(ValueError, match="between zero and 61"):
        model.restore_bundle_rank_hint(62)


def test_canonical_preparation_satisfies_shifted_guard_and_restriction(
    prepared_model,
) -> None:
    _, preparation = prepared_model

    assert preparation.hierarchy_degree == 9
    assert preparation.minimum_shifted_lower_bound > 0.0
    assert preparation.restriction_residual < 3e-14
    assert preparation.factorization_residual < 3e-14


def test_completion_coordinate_map_is_affine(prepared_model) -> None:
    model, preparation = prepared_model
    geometry = model.geometry
    retained, completion = geometry.unpack_state(preparation.state)
    rng = np.random.default_rng(20260806)
    increment = 1e-7 * rng.normal(size=geometry.completion_count)

    observed = geometry.scaled_unified_matrix(
        retained, completion + increment
    ) - geometry.scaled_unified_matrix(retained, completion)
    expected = np.einsum(
        "j,jab->ab",
        increment,
        geometry.scaled_completion_coefficients(),
        optimize=True,
    )

    np.testing.assert_allclose(observed, expected, atol=2e-14, rtol=2e-12)


def test_desired_completion_velocity_uses_only_compiler_readable_rows(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    geometry = model.geometry
    retained, completion = geometry.unpack_state(preparation.state)
    velocity = model.desired_completion_velocity(0.0, retained, completion)
    unreadable = np.setdiff1d(
        np.arange(geometry.completion_count),
        geometry.readable_completion_indices,
    )

    assert np.all(np.isfinite(velocity))
    np.testing.assert_array_equal(velocity[unreadable], 0.0)


def test_interior_radial_atom_keeps_augmented_retained_velocity_exact(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    geometry = model.geometry
    retained, completion = geometry.unpack_state(preparation.state)
    expected_retained_velocity = model.retained_velocity(
        0.0, retained, completion
    )

    result = model.radial_atom(0.0, preparation.state, 1e-8)

    assert result.success
    assert result.message == "unconstrained predictor"
    assert result.archive_intervention == 0.0
    np.testing.assert_allclose(
        result.archive_velocity,
        expected_retained_velocity,
        atol=0.0,
        rtol=0.0,
    )
    endpoint_retained, _ = geometry.unpack_state(result.endpoint)
    np.testing.assert_allclose(
        endpoint_retained,
        retained + 1e-8 * expected_retained_velocity,
        atol=2e-16,
        rtol=2e-15,
    )
    assert result.minimum_shifted_lower_bound > 0.0


def test_no_guard_atom_propagates_desired_higher_moment_rates_exactly(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    original_settings = model.settings
    model.settings = replace(original_settings, enforce_gram_guard=False)
    try:
        retained, completion = model.geometry.unpack_state(preparation.state)
        expected_retained = model.retained_velocity(0.0, retained, completion)
        expected_completion = model.desired_completion_velocity(
            0.0, retained, completion
        )
        result = model.radial_atom(0.0, preparation.state, 1e-4)
    finally:
        model.settings = original_settings

    assert result.success
    assert result.message == "unconstrained higher-moment predictor"
    assert result.correction_iterations == 0
    assert result.completion_correction_norm == 0.0
    np.testing.assert_allclose(
        result.archive_velocity, expected_retained, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        result.completion_velocity,
        expected_completion,
        atol=0.0,
        rtol=0.0,
    )


def test_negative_mode_audit_reports_hidden_to_retained_coupling(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    retained, completion = model.geometry.unpack_state(preparation.state)
    perturbed = completion.copy()
    perturbed[0] += 1e-3
    state = model.geometry.pack_state(retained, perturbed)

    result = negative_mode_coupling(model, 0.0, state)

    assert result["minimum_eigenvalue"] < -1e-8
    assert result["negative_mode_count"] > 0
    assert (
        result[
            "maximum_negative_mode_predicted_retained_velocity_relative_change"
        ]
        >= 0.0
    )
    assert (
        result["maximum_negative_mode_predicted_c_velocity_relative_change"]
        >= 0.0
    )


def test_retained_repair_changes_only_the_correlation_rate(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    retained, completion = model.geometry.unpack_state(preparation.state)
    raw, _ = model.geometry.split_retained(retained)
    matrix_state = raw_moment_coordinates_to_matrix_state(raw)
    archive = matrix_derivative_to_raw_moment_velocity(
        matrix_state,
        matrix_dimer_rhs(0.0, matrix_state, model.parameters),
    )
    augmented = model.retained_velocity(0.0, retained, completion)[: raw.size]

    np.testing.assert_allclose(augmented[:17], archive[:17], atol=1e-14)
    assert np.linalg.norm(augmented[17:] - archive[17:]) > 1e-6


def test_trajectory_records_spectrum_selected_critical_modes(
    prepared_model,
) -> None:
    model, preparation = prepared_model

    trajectory = integrate_cwrmf_ssprk2(
        model,
        preparation.state,
        final_time=1e-8,
        time_step=1e-8,
    )

    assert trajectory.success
    assert trajectory.critical_modes.shape == trajectory.times.shape
    assert np.all(trajectory.critical_modes >= 0)


def test_event_guard_policy_validation_and_profile_defaults() -> None:
    settings = CWRMFSettings()

    assert settings.guard_policy == "always"
    assert cwrmf_settings_for_profile("strict").guard_policy == "always"
    assert cwrmf_settings_for_profile("balanced").guard_policy == "always"

    event = replace(settings, guard_policy="event_triggered")
    assert event.event_gram_floor < event.event_gram_hard_floor

    with pytest.raises(ValueError, match="unknown CWRMF guard policy"):
        replace(settings, guard_policy="sometimes")
    with pytest.raises(ValueError, match="requires enforce_gram_guard"):
        replace(
            settings,
            guard_policy="event_triggered",
            enforce_gram_guard=False,
        )
    with pytest.raises(ValueError, match="positive and ordered"):
        replace(settings, event_gram_floor=1e-2)
    with pytest.raises(ValueError, match="event_coupling_threshold"):
        replace(settings, event_coupling_threshold=0.0)


def test_event_guard_fast_path_accepts_psd_endpoint_without_solver(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    original_settings = model.settings
    model.settings = replace(original_settings, guard_policy="event_triggered")
    try:
        retained, completion = model.geometry.unpack_state(preparation.state)
        expected_completion = model.desired_completion_velocity(
            0.0, retained, completion
        )
        result = model.radial_atom(0.0, preparation.state, 1e-8)
    finally:
        model.settings = original_settings

    assert result.success
    assert result.message == "event_guard_fast_path"
    assert result.correction_iterations == 0
    assert result.completion_correction_norm == 0.0
    np.testing.assert_allclose(
        result.completion_velocity,
        expected_completion,
        atol=0.0,
        rtol=0.0,
    )


def test_event_guard_tolerates_declared_benign_negativity(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    retained, completion = model.geometry.unpack_state(preparation.state)
    perturbed = completion.copy()
    perturbed[0] += 1e-5
    state = model.geometry.pack_state(retained, perturbed)
    minimum = float(
        np.linalg.eigvalsh(
            model.geometry.scaled_unified_matrix(retained, perturbed)
        )[0]
    )
    assert -1e-3 < minimum < -1e-8

    original_settings = model.settings
    model.settings = replace(
        original_settings,
        guard_policy="event_triggered",
        event_coupling_threshold=1e6,
    )
    try:
        result = model.radial_atom(0.0, state, 1e-8)
    finally:
        model.settings = original_settings

    assert result.success
    assert result.message.startswith("event_guard_tolerated:")
    assert result.completion_correction_norm == 0.0


def test_event_completion_retraction_restores_cone(prepared_model) -> None:
    model, preparation = prepared_model
    retained, completion = model.geometry.unpack_state(preparation.state)
    perturbed = completion.copy()
    perturbed[0] += 1e-5
    assert (
        float(
            np.linalg.eigvalsh(
                model.geometry.scaled_unified_matrix(retained, perturbed)
            )[0]
        )
        < -1e-8
    )

    retraction = model._event_completion_retraction(retained, perturbed)

    assert retraction is not None
    repaired, iterations = retraction
    assert iterations >= 1
    assert (
        float(
            np.linalg.eigvalsh(
                model.geometry.scaled_unified_matrix(retained, repaired)
            )[0]
        )
        >= -model.settings.event_gram_floor
    )
    assert np.linalg.norm(repaired - perturbed) < 1e-3


def test_event_guard_trigger_retracts_endpoint_completion(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    retained, completion = model.geometry.unpack_state(preparation.state)
    perturbed = completion.copy()
    perturbed[0] += 1e-5
    state = model.geometry.pack_state(retained, perturbed)

    original_settings = model.settings
    model.settings = replace(
        original_settings,
        guard_policy="event_triggered",
        event_coupling_threshold=1e-30,
    )
    try:
        result = model.radial_atom(0.0, state, 1e-8)
    finally:
        model.settings = original_settings

    assert result.success
    assert result.message.startswith("event_guard_retraction:")
    assert result.correction_iterations >= 1
    assert result.completion_correction_norm > 0.0
    assert result.minimum_unshifted_eigenvalue >= (
        -model.settings.event_gram_floor
    )
    endpoint_retained, _ = model.geometry.unpack_state(result.endpoint)
    np.testing.assert_allclose(
        endpoint_retained,
        retained + 1e-8 * result.archive_velocity,
        atol=2e-16,
        rtol=2e-15,
    )


def test_event_guard_falls_back_to_conic_correction_when_retraction_fails(
    prepared_model, monkeypatch
) -> None:
    model, preparation = prepared_model
    retained, completion = model.geometry.unpack_state(preparation.state)
    perturbed = completion.copy()
    perturbed[0] += 1e-5
    state = model.geometry.pack_state(retained, perturbed)

    solver_calls: list[str] = []

    def stub_solver(*args, **kwargs):
        solver_calls.append("relaxed_readable")
        desired = model.desired_completion_velocity(0.0, retained, perturbed)
        return False, desired, "stubbed", float("inf"), 1.0, 0

    monkeypatch.setattr(model, "_solve_relaxed_readable", stub_solver)
    monkeypatch.setattr(
        model, "_event_completion_retraction", lambda *args: None
    )
    original_settings = model.settings
    model.settings = replace(
        original_settings,
        guard_policy="event_triggered",
        event_coupling_threshold=1e-30,
    )
    try:
        result = model.radial_atom(0.0, state, 1e-8)
    finally:
        model.settings = original_settings

    assert solver_calls == ["relaxed_readable"]
    assert not result.success
    assert result.message.startswith("readable_relaxation:stubbed")


def test_event_guard_measure_is_shared_with_offline_audit() -> None:
    from paper5.stability import apcm_carried_witness as model_module
    from paper5.stability import apcm_carried_witness_mode_audit as audit_module

    assert audit_module.negative_mode_coupling is (
        model_module.negative_mode_coupling
    )


def test_event_guard_trajectory_completes_smoke_step(prepared_model) -> None:
    model, preparation = prepared_model
    original_settings = model.settings
    model.settings = replace(original_settings, guard_policy="event_triggered")
    try:
        trajectory = integrate_cwrmf_ssprk2(
            model,
            preparation.state,
            final_time=1e-8,
            time_step=1e-8,
        )
    finally:
        model.settings = original_settings

    assert trajectory.success
    assert trajectory.completed_steps == 1


def test_analysis_parser_rejects_conflicting_guard_flags(tmp_path) -> None:
    from paper5.stability.apcm_carried_witness_analysis import _parser

    parser = _parser()
    arguments = parser.parse_args(
        ["--output-directory", str(tmp_path), "--event-triggered-guard"]
    )
    assert arguments.event_triggered_guard
    assert not arguments.no_gram_guard

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--output-directory",
                str(tmp_path),
                "--event-triggered-guard",
                "--no-gram-guard",
            ]
        )


def test_accuracy_metrics_separate_fixed_offset_from_dynamic_error() -> None:
    parameters = DimerParameters()
    exact = np.zeros((3, 31), dtype=float)
    approximate = np.full((3, 31), 2e-5, dtype=float)

    metrics = _accuracy_metrics(parameters, exact, approximate)

    assert metrics["all_coordinate_scalar_rms_error"] == pytest.approx(2e-5)
    assert metrics["dynamic_all_coordinate_scalar_rms_error"] == 0.0
    assert metrics["initial_coordinate_offset_l2"] == pytest.approx(
        np.sqrt(31) * 2e-5
    )
    for block in metrics["blockwise"].values():
        assert block["dynamic_scalar_rms_error"] == 0.0


def test_tier_timing_is_off_by_default_and_names_every_tier(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    assert model.tier_timing is None

    timed = CarriedWitnessModel(model.parameters, geometry=model.geometry)
    timed.prepare(phonon_cutoff=16)
    timed.enable_tier_timing()
    atom = timed.radial_atom(0.0, preparation.state, 1e-8)
    report = timed.tier_timing

    assert atom.success
    assert report is not None
    assert set(report["seconds"]) == {
        "gram_eigendecomposition",
        "coefficient_assembly_and_svec",
        "clarabel_solve",
        "retraction",
        "unguarded_velocity",
    }
    assert report["calls"]["unguarded_velocity"] == 1
    assert report["calls"]["gram_eigendecomposition"] >= 1
    assert report["seconds"]["unguarded_velocity"] > 0.0
    # The model the other tests share must stay uninstrumented.
    assert model.tier_timing is None


def test_compiled_commutator_rows_reproduce_the_literal_moment_rate(
    prepared_model,
) -> None:
    from paper5.stability.apcm_carried_witness import _moment_derivative
    from paper5.stability.moment_hierarchy import (
        _commutator,
        _hamiltonian_terms,
    )

    model, preparation = prepared_model
    retained, completion = model.geometry.unpack_state(preparation.state)
    moments = model.geometry.moment_mapping(retained, completion)
    time = 0.037

    def literal(key) -> float:
        derivative = 0.0j
        for coefficient, word in _hamiltonian_terms(time, model.parameters):
            for generated, commutator_coefficient in _commutator(
                word, key
            ).items():
                value = 1.0 if generated.degree == 0 else moments[generated]
                derivative += (
                    1j * coefficient * commutator_coefficient * float(value)
                )
        return float(derivative.real)

    checked = 0
    for key in model.geometry.outer_extension.frontier_keys:
        if any(
            generated not in moments
            for _, word in _hamiltonian_terms(time, model.parameters)
            for generated in _commutator(word, key)
            if generated.degree > 0
        ):
            continue
        assert _moment_derivative(
            key, time, model.parameters, moments
        ) == literal(key)
        checked += 1
    assert checked > 0


def test_default_integration_takes_no_subdivision_and_still_terminates(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    calls: list[float] = []
    original = model.radial_atom

    def failing_atom(time, state, step_size, **keywords):
        calls.append(step_size)
        return replace(
            original(time, state, step_size, **keywords),
            success=False,
            message="forced_failure",
        )

    model.radial_atom = failing_atom  # type: ignore[method-assign]
    try:
        trajectory = integrate_cwrmf_ssprk2(
            model,
            preparation.state,
            final_time=2e-8,
            time_step=1e-8,
        )
    finally:
        del model.radial_atom

    assert not trajectory.success
    assert trajectory.completed_steps == 0
    assert trajectory.message.startswith("first atom at t=0.000000")
    assert "subdivision" not in trajectory.message
    assert calls == [1e-8]

    with pytest.raises(ValueError):
        integrate_cwrmf_ssprk2(
            model,
            preparation.state,
            final_time=2e-8,
            time_step=1e-8,
            maximum_step_subdivisions=-1,
        )


def test_bounded_step_halving_recovers_from_a_failed_atom(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    original = model.radial_atom
    sizes: list[float] = []

    def halving_atom(time, state, step_size, **keywords):
        sizes.append(step_size)
        result = original(time, state, step_size, **keywords)
        if step_size > 0.75e-8:
            return replace(
                result, success=False, message="forced_failure_at_full_step"
            )
        return result

    model.radial_atom = halving_atom  # type: ignore[method-assign]
    try:
        stopped = integrate_cwrmf_ssprk2(
            model,
            preparation.state,
            final_time=1e-8,
            time_step=1e-8,
        )
        recovered = integrate_cwrmf_ssprk2(
            model,
            preparation.state,
            final_time=1e-8,
            time_step=1e-8,
            maximum_step_subdivisions=1,
        )
    finally:
        del model.radial_atom

    assert not stopped.success
    assert stopped.completed_steps == 0

    assert recovered.success
    assert recovered.completed_steps == 1
    assert recovered.times.size == 2
    # One failed full step, then two accepted SSPRK2 steps of half the size.
    assert 0.5e-8 in sizes
    assert recovered.atom_evaluations == 5
    assert np.all(np.isfinite(recovered.states))


def test_batched_mode_gradients_are_off_by_default_and_agree_to_roundoff(
    prepared_model,
) -> None:
    from paper5.stability.apcm_carried_witness import (
        _mode_eigenvalue_gradients,
    )

    model, preparation = prepared_model
    assert CWRMFSettings().batched_mode_gradients is False
    assert cwrmf_settings_for_profile("strict").batched_mode_gradients is False
    assert (
        cwrmf_settings_for_profile("balanced").batched_mode_gradients is False
    )

    geometry = model.geometry
    coefficients = geometry.scaled_completion_coefficients()
    retained, completion = geometry.unpack_state(preparation.state)
    _, vectors = np.linalg.eigh(
        geometry.scaled_unified_matrix(retained, completion)
    )
    working = vectors[:, :12]

    per_mode = _mode_eigenvalue_gradients(
        coefficients, working, batched=False
    )
    batched = _mode_eigenvalue_gradients(coefficients, working, batched=True)

    assert per_mode.shape == (12, geometry.completion_count)
    for row in range(12):
        vector = working[:, row]
        assert np.array_equal(
            per_mode[row],
            np.einsum(
                "a,jab,b->j",
                vector.conjugate(),
                coefficients,
                vector,
                optimize=True,
            ).real,
        )
    # One contraction instead of twelve is a different BLAS blocking, so the
    # agreement is to roundoff, not bitwise.
    assert np.max(np.abs(per_mode - batched)) <= 1e-13 * max(
        float(np.max(np.abs(per_mode))), 1e-300
    )


def test_step_scaled_coefficient_cache_returns_the_literal_product(
    prepared_model,
) -> None:
    model, _ = prepared_model
    compiled = model.geometry.scaled_completion_coefficients()

    first = model._step_scaled_completion_coefficients(0.0025)
    second = model._step_scaled_completion_coefficients(0.0025)
    assert first is second
    assert np.array_equal(first, 0.0025 * compiled)
    assert first.shape == (model.geometry.completion_count, 62, 62)

    halved = model._step_scaled_completion_coefficients(0.00125)
    assert halved is not first
    assert np.array_equal(halved, 0.00125 * compiled)
    # The compiled tensor itself is untouched by the scaling.
    assert np.array_equal(
        compiled, model.geometry.scaled_completion_coefficients()
    )


def test_vectorized_svec_packs_the_literal_upper_triangle_order() -> None:
    from paper5.stability.apcm_positive_extension import _clarabel_svec_upper

    def literal(matrix: np.ndarray) -> np.ndarray:
        value = np.asarray(matrix, dtype=float)
        packed: list[float] = []
        root_two = np.sqrt(2.0)
        for column in range(value.shape[1]):
            for row in range(column + 1):
                scale = 1.0 if row == column else root_two
                packed.append(scale * float(value[row, column]))
        return np.asarray(packed, dtype=float)

    generator = np.random.default_rng(20260914)
    for size in (1, 2, 5, 62, 80, 124):
        sample = generator.standard_normal((size, size))
        symmetric = 0.5 * (sample + sample.T)
        assert np.array_equal(
            literal(symmetric), _clarabel_svec_upper(symmetric)
        )
        assert np.array_equal(literal(sample), _clarabel_svec_upper(sample))

    with pytest.raises(ValueError):
        _clarabel_svec_upper(np.zeros((3, 4)))


def test_center_cross_evaluator_matches_the_one_shot_reader(
    prepared_model,
) -> None:
    model, preparation = prepared_model
    geometry = model.geometry
    retained, completion = geometry.unpack_state(preparation.state)
    reader = geometry.center_cross_evaluator(retained, completion)

    checked = 0
    for row in range(9 + geometry.extra_count):
        key = geometry._relative_row_key(row)
        for center_index in range(2):
            assert reader(center_index, key) == geometry.center_cross_value(
                center_index, key, retained, completion
            )
            checked += 1
    assert checked == 2 * (9 + geometry.extra_count)


def test_solver_setting_profile_defaults_to_the_pinned_single_thread_set(
) -> None:
    from paper5.stability.apcm_carried_witness import (
        CWRMF_SOLVER_SETTING_PROFILES,
        DEFAULT_SOLVER_SETTING_PROFILE,
    )

    assert DEFAULT_SOLVER_SETTING_PROFILE == "pinned_single_thread"
    assert CWRMF_SOLVER_SETTING_PROFILES[DEFAULT_SOLVER_SETTING_PROFILE] == {}
    for profile in ("strict", "balanced"):
        assert (
            cwrmf_settings_for_profile(profile).solver_setting_profile
            == DEFAULT_SOLVER_SETTING_PROFILE
        )
    assert (
        CWRMFSettings().solver_setting_profile
        == DEFAULT_SOLVER_SETTING_PROFILE
    )
    with pytest.raises(ValueError, match="unknown CWRMF solver setting"):
        replace(CWRMFSettings(), solver_setting_profile="faster_please")


def test_default_solver_setting_profile_touches_no_clarabel_field() -> None:
    import clarabel

    from paper5.stability.apcm_carried_witness import (
        CWRMF_SOLVER_SETTING_PROFILES,
    )
    from paper5.stability.cwrmf_solver_instances import (
        RECORDED_SETTING_FIELDS,
    )
    from paper5.stability.hubbard_dimer import DimerParameters

    model = CarriedWitnessModel(DimerParameters())
    settings = clarabel.DefaultSettings()
    settings.max_threads = 1
    before = {
        name: getattr(settings, name) for name in RECORDED_SETTING_FIELDS
    }

    model._apply_solver_setting_profile(settings)

    after = {
        name: getattr(settings, name) for name in RECORDED_SETTING_FIELDS
    }
    assert before == after
    # Every declared profile names only fields Clarabel carries.
    for overrides in CWRMF_SOLVER_SETTING_PROFILES.values():
        for field_name in overrides:
            assert hasattr(settings, field_name), field_name


def test_named_solver_setting_profiles_overlay_only_their_own_fields() -> None:
    import clarabel

    from paper5.stability.apcm_carried_witness import (
        CWRMF_SOLVER_SETTING_PROFILES,
    )
    from paper5.stability.hubbard_dimer import DimerParameters

    assert set(CWRMF_SOLVER_SETTING_PROFILES) == {
        "pinned_single_thread",
        "presolve_off",
        "max_threads_8",
    }
    assert CWRMF_SOLVER_SETTING_PROFILES["presolve_off"] == {
        "presolve_enable": False
    }
    assert CWRMF_SOLVER_SETTING_PROFILES["max_threads_8"] == {
        "max_threads": 8
    }

    for name, overrides in CWRMF_SOLVER_SETTING_PROFILES.items():
        model = CarriedWitnessModel(
            DimerParameters(),
            settings=replace(CWRMFSettings(), solver_setting_profile=name),
        )
        settings = clarabel.DefaultSettings()
        settings.max_threads = 1
        settings.presolve_enable = True
        model._apply_solver_setting_profile(settings)
        assert settings.max_threads == overrides.get("max_threads", 1)
        assert settings.presolve_enable == overrides.get(
            "presolve_enable", True
        )
