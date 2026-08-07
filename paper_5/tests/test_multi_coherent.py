from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import paper5.stability.multi_coherent_long_horizon as long_horizon
import pytest

from paper5.stability import (
    DimerParameters,
    GaussianSineDrive,
    MultiCoherentHoldoutSettings,
    closed_coordinate_distance,
    closed_coordinate_error_scores,
    certify_bounded_score,
    development_coordinate_scales,
    energy_work_residual,
    electron_relative_product_to_local_state,
    contract_exact_holdout_closed_coordinates,
    evaluate_multi_coherent_holdout_scores,
    pointwise_resolution_uncertainty,
    score_resolution_uncertainty,
    sensitivity_amplification,
    audit_blind_multi_coherent_summary,
    build_holdout_initial_conditions,
    freeze_blind_multi_coherent_inputs,
    load_frozen_multi_coherent_inputs,
    load_frozen_multi_coherent_model_batch,
    propagate_exact_holdout_dop853,
    propagate_exact_holdout_midpoint,
    run_frozen_multi_coherent_model_trajectory,
    seal_frozen_multi_coherent_model_batch,
    seal_frozen_multi_coherent_model_cost,
    score_frozen_multi_coherent_holdout,
    fit_two_coherent_electron_relative_state,
    multi_coherent_capacity,
    multi_coherent_rhs,
    multi_coherent_state,
    multi_coherent_state_and_tangent,
    normalized_multi_coherent_state_and_horizontal_tangent,
    normalized_diagonal_kick_generator,
    pack_multi_coherent_parameters,
    project_schrodinger_velocity,
    relative_holstein_hamiltonian,
    relative_state_moment_coordinates,
    relative_state_moment_derivative,
    relative_state_closed_coordinates,
    retract_multi_coherent_parameters,
    spawn_residual_coherent_packets,
    symmetric_projected_generator_kick,
    unpack_multi_coherent_parameters,
    moment_hierarchy,
    MomentKey,
)
from paper5.stability.multi_coherent_analysis import (
    run_multi_coherent_analysis,
)
from paper5.stability.multi_coherent_propagation import (
    run_multi_coherent_propagation,
)
from paper5.stability.multi_coherent_long_horizon import (
    run_segmented_multi_coherent_horizon,
)


def _normalized_parameters() -> np.ndarray:
    coefficients = np.array(
        [
            [0.31 + 0.17j, -0.11 + 0.21j],
            [0.23 - 0.08j, 0.19 + 0.13j],
            [-0.27 + 0.12j, 0.16 - 0.20j],
            [0.14 + 0.29j, -0.18 - 0.09j],
        ]
    )
    shared_displacements = np.array([0.45 + 0.12j, -0.72 + 0.31j])
    displacements = np.tile(shared_displacements, (4, 1))
    parameters = pack_multi_coherent_parameters(coefficients, displacements)
    state = multi_coherent_state(parameters, relative_dimension=20)
    coefficients /= np.linalg.norm(state)
    return pack_multi_coherent_parameters(coefficients, displacements)


def _decoupled_vacuum_parameters() -> np.ndarray:
    coefficients = np.zeros((4, 2), dtype=complex)
    coefficients[:, 0] = 0.5
    displacements = np.zeros((4, 2), dtype=complex)
    displacements[:, 1] = (0.1, -0.1, 0.1j, -0.1j)
    return pack_multi_coherent_parameters(coefficients, displacements)


def test_multi_coherent_pack_round_trip_and_tangent_finite_difference() -> None:
    parameters = _normalized_parameters()
    coefficients, displacements = unpack_multi_coherent_parameters(parameters)
    repacked = pack_multi_coherent_parameters(coefficients, displacements)
    np.testing.assert_array_equal(repacked, parameters)

    state, tangent = multi_coherent_state_and_tangent(
        parameters,
        relative_dimension=12,
    )
    for column in (0, 3, 9, 18, 31):
        step = 1e-6
        offset = np.zeros_like(parameters)
        offset[column] = step
        finite_difference = (
            multi_coherent_state(
                parameters + offset,
                relative_dimension=12,
            )
            - multi_coherent_state(
                parameters - offset,
                relative_dimension=12,
            )
        ) / (2.0 * step)
        np.testing.assert_allclose(
            tangent[:, column],
            finite_difference,
            atol=2e-9,
            rtol=2e-8,
        )
    assert state.shape == (48,)


def test_extreme_displacement_remains_finite_after_packet_normalization() -> None:
    parameters = _normalized_parameters()
    coefficients, displacements = unpack_multi_coherent_parameters(parameters)
    displacements[0, 0] = 50.0
    parameters = pack_multi_coherent_parameters(coefficients, displacements)

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        state, tangent = normalized_multi_coherent_state_and_horizontal_tangent(
            parameters,
            relative_dimension=12,
        )

    assert np.all(np.isfinite(state))
    assert np.all(np.isfinite(tangent))
    np.testing.assert_allclose(np.vdot(state, state), 1.0, atol=2e-15)


def test_center_relative_vacuum_embeds_exactly_in_the_local_cutoff() -> None:
    cutoff = 2
    normal_dimension = 2 * cutoff + 1
    electron_relative = np.zeros(4 * normal_dimension, dtype=complex)
    electron_relative[0] = 1.0
    center = np.zeros(normal_dimension, dtype=complex)
    center[0] = 1.0

    embedded = electron_relative_product_to_local_state(
        electron_relative,
        center,
        phonon_cutoff=cutoff,
    )

    expected = np.zeros(4 * (cutoff + 1) ** 2, dtype=complex)
    expected[0] = 1.0
    np.testing.assert_allclose(embedded.state, expected, atol=1e-15)
    np.testing.assert_allclose(embedded.retained_norm, 1.0, atol=1e-15)


def test_independent_exact_holdout_propagators_agree_on_a_small_control() -> None:
    cutoff = 2
    settings = MultiCoherentHoldoutSettings(
        final_time=0.1,
        score_interval=(0.025, 0.05),
        sensitivity_interval=(0.05, 0.1),
        phonon_cutoff=cutoff,
        initial_packets_per_electronic_branch=2,
        segment_length=0.05,
        output_sample_step=0.025,
        exact_midpoint_step=0.0025,
        exact_dop853_maximum_step=0.0025,
    )
    model = DimerParameters(lambda_ep=0.0, gamma=0.5, drive_amplitude=1.0)
    relative_dimension = 2 * cutoff + 1
    relative = multi_coherent_state(
        _decoupled_vacuum_parameters(),
        relative_dimension=relative_dimension,
    )
    center = np.zeros(relative_dimension, dtype=complex)
    center[0] = 1.0
    initial = electron_relative_product_to_local_state(
        relative,
        center,
        phonon_cutoff=cutoff,
    ).state
    initial_states = np.stack((initial, initial, initial))
    times = np.arange(0.0, 0.1000001, 0.025)
    drive = settings.drive_protocol(model)

    dop853 = propagate_exact_holdout_dop853(
        model,
        initial_states,
        times,
        drive_protocol=drive,
        phonon_cutoff=cutoff,
        relative_tolerance=1e-11,
        absolute_tolerance=1e-13,
        maximum_step=0.0025,
    )
    midpoint = propagate_exact_holdout_midpoint(
        model,
        initial_states,
        times,
        drive_protocol=drive,
        phonon_cutoff=cutoff,
        integration_step=0.0025,
        exponential_action_tolerance=1e-13,
    )

    overlaps = np.sum(
        dop853.state_vectors.conj() * midpoint.state_vectors,
        axis=2,
    )
    assert np.max(1.0 - np.abs(overlaps) ** 2) < 1e-9
    assert dop853.maximum_norm_drift < 1e-10
    assert midpoint.maximum_norm_drift < 1e-10
    dop853_closed = contract_exact_holdout_closed_coordinates(
        dop853,
        model,
        phonon_cutoff=cutoff,
    )
    midpoint_closed = contract_exact_holdout_closed_coordinates(
        midpoint,
        model,
        phonon_cutoff=cutoff,
    )
    np.testing.assert_allclose(
        dop853_closed,
        midpoint_closed,
        atol=2e-6,
        rtol=0.0,
    )


def test_holdout_score_evaluation_separates_resolution_and_scientific_gates() -> None:
    settings = MultiCoherentHoldoutSettings(
        final_time=0.1,
        score_interval=(0.025, 0.05),
        sensitivity_interval=(0.05, 0.1),
        output_sample_step=0.025,
    )
    times = np.arange(0.0, 0.1000001, 0.025)
    closed = np.zeros((3, times.size, 31))
    closed[1, :, 3] = 1e-4
    closed[2, :, 3] = -1e-4
    kets = np.zeros((3, times.size, 4), dtype=complex)
    kets[:, :, 0] = 1.0
    work = np.zeros((3, times.size))
    model_closed = {"coarse": closed.copy(), "fine": closed.copy()}
    model_kets = {"coarse": kets.copy(), "fine": kets.copy()}
    model_work = {"coarse": work.copy(), "fine": work.copy()}
    exact_closed = {"dop853": closed.copy(), "midpoint": closed.copy()}
    exact_kets = {"dop853": kets.copy(), "midpoint": kets.copy()}

    passed = evaluate_multi_coherent_holdout_scores(
        times,
        model_closed=model_closed,
        model_kets=model_kets,
        model_normalized_work_residual=model_work,
        exact_closed=exact_closed,
        exact_kets=exact_kets,
        coordinate_scales=np.ones(31),
        initial_distance=closed_coordinate_distance(
            closed[1, 0], closed[2, 0], np.ones(31)
        ),
        settings=settings,
    )

    assert passed.reference_valid
    assert passed.numerically_resolved
    assert passed.scientific_passed

    model_closed["coarse"][:, :, 3] += 0.2
    model_closed["fine"][:, :, 3] += 0.2
    failed = evaluate_multi_coherent_holdout_scores(
        times,
        model_closed=model_closed,
        model_kets=model_kets,
        model_normalized_work_residual=model_work,
        exact_closed=exact_closed,
        exact_kets=exact_kets,
        coordinate_scales=np.ones(31),
        initial_distance=closed_coordinate_distance(
            closed[1, 0], closed[2, 0], np.ones(31)
        ),
        settings=settings,
    )
    assert failed.reference_valid
    assert failed.numerically_resolved
    assert not failed.scientific_passed


def test_normalized_horizontal_tangent_matches_projective_finite_difference() -> None:
    parameters = _normalized_parameters()
    state, tangent = normalized_multi_coherent_state_and_horizontal_tangent(
        parameters,
        relative_dimension=12,
    )

    np.testing.assert_allclose(np.vdot(state, state), 1.0, atol=2e-15)
    np.testing.assert_allclose(
        state.conj() @ tangent,
        0.0,
        atol=2e-14,
    )

    direction = np.random.default_rng(260804).normal(size=parameters.size)
    direction /= np.linalg.norm(direction)
    step = 2e-6
    forward = multi_coherent_state(
        parameters + step * direction,
        relative_dimension=12,
    )
    backward = multi_coherent_state(
        parameters - step * direction,
        relative_dimension=12,
    )
    forward /= np.linalg.norm(forward)
    backward /= np.linalg.norm(backward)
    finite_difference = (forward - backward) / (2.0 * step)
    finite_difference -= state * np.vdot(state, finite_difference)

    np.testing.assert_allclose(
        tangent @ direction,
        finite_difference,
        atol=4e-9,
        rtol=4e-8,
    )


def test_relative_holstein_hamiltonian_is_hermitian() -> None:
    hamiltonian = relative_holstein_hamiltonian(
        0.37,
        DimerParameters(lambda_ep=1.5, gamma=0.5),
        relative_dimension=9,
    )

    np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T)


def test_delayed_drive_is_additive_and_has_the_declared_derivative() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    single = GaussianSineDrive.from_parameters(parameters)
    double = GaussianSineDrive.from_parameters(parameters, delays=(0.0, 8.0))

    for time in (0.0, 0.37, 2.1, 7.9):
        assert single.difference(time) == parameters.drive_difference(time)
        assert double.difference(time) == single.difference(time)

    time = 8.73
    np.testing.assert_allclose(
        double.difference(time),
        single.difference(time) + single.difference(time - 8.0),
        atol=2e-15,
    )
    step = 1e-6
    finite_difference = (
        double.difference(time + step) - double.difference(time - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        double.derivative(time),
        finite_difference,
        atol=2e-10,
        rtol=2e-9,
    )

    driven = relative_holstein_hamiltonian(
        time,
        parameters,
        relative_dimension=9,
        drive_protocol=double,
    )
    np.testing.assert_allclose(driven, driven.conj().T)


def test_capacity_counts_branch_packets_and_raw_coordinates_explicitly() -> None:
    capacity = multi_coherent_capacity(_normalized_parameters())

    assert capacity.packets_per_electronic_branch == 2
    assert capacity.total_branch_packets == 8
    assert capacity.raw_coordinate_count == 32


def test_holdout_settings_translate_the_relative_mode_capacity_explicitly() -> None:
    settings = MultiCoherentHoldoutSettings()
    model = DimerParameters(lambda_ep=1.5, gamma=0.5, drive_amplitude=1.0)

    assert settings.initial_packets_per_electronic_branch == 4
    assert settings.maximum_packets_per_electronic_branch == 6
    assert settings.maximum_total_branch_packets == 24
    assert settings.maximum_raw_coordinate_count == 96
    assert settings.maximum_geometric_tangent_rank == 96
    assert settings.geometric_gram_relative_threshold == 1e-10
    assert settings.relative_damping == 3e-4
    assert settings.drive_protocol(model).delays == (0.0, 8.0)


def test_blind_summary_audit_enforces_capacity_rank_and_reference_separation() -> None:
    settings = MultiCoherentHoldoutSettings()
    summary = {
        "status": "complete",
        "parameters": {
            "target_final_time": 100.0,
            "phonon_cutoff": 16,
            "packet_count": 4,
            "maximum_packet_count": 6,
            "tangent_regularization": "tikhonov",
            "relative_damping": 3e-4,
            "drive_protocol": {
                "amplitude": 1.0,
                "pulse_width": 1.0,
                "delays": [0.0, 8.0],
            },
        },
        "initialization": {
            "exact_reference_used_after_t0_by_model_rhs": False,
        },
        "capacity": {
            "online_exact_reference_used": False,
            "final_total_branch_packets": 24,
            "final_raw_coordinate_count": 96,
        },
        "tangent_diagnostics": {
            "geometric_gram_relative_threshold": 1e-10,
            "geometric_ranks": [62, 94],
        },
        "physicality": {
            "maximum_norm_drift": 1e-15,
            "minimum_electron_density_eigenvalue": 0.02,
            "minimum_relative_uncertainty_margin": 0.3,
        },
        "work_balance": {"maximum_normalized_residual": 4e-4},
        "offline_exact_comparison": None,
    }

    assert audit_blind_multi_coherent_summary(summary, settings).passed

    invalid = json.loads(json.dumps(summary))
    invalid["capacity"]["final_total_branch_packets"] = 25
    invalid["tangent_diagnostics"]["geometric_ranks"] = [97]
    invalid["offline_exact_comparison"] = {"minimum_state_fidelity": 1.0}
    audit = audit_blind_multi_coherent_summary(invalid, settings)
    assert not audit.passed
    assert set(audit.failures) == {
        "exact_reference_was_opened",
        "geometric_tangent_rank_cap_exceeded",
        "total_branch_packet_cap_exceeded",
    }


def test_holdout_initial_pair_is_symmetric_and_model_representable() -> None:
    settings = MultiCoherentHoldoutSettings(
        initial_packets_per_electronic_branch=2,
    )
    model = DimerParameters(lambda_ep=1.5, gamma=0.5, drive_amplitude=1.0)

    initial = build_holdout_initial_conditions(
        _normalized_parameters(),
        np.ones(31),
        settings=settings,
        parameters=model,
    )

    assert initial.initial_distance >= 1e-6
    assert initial.central_parameters.shape == _normalized_parameters().shape
    assert initial.plus_parameters.shape == initial.central_parameters.shape
    assert initial.minus_parameters.shape == initial.central_parameters.shape
    assert initial.closed_coordinates.shape == (3, 31)
    for parameters in (
        initial.central_parameters,
        initial.plus_parameters,
        initial.minus_parameters,
    ):
        state = multi_coherent_state(parameters, relative_dimension=33)
        np.testing.assert_allclose(np.vdot(state, state), 1.0, atol=2e-14)


def test_blind_inputs_freeze_before_any_exact_holdout_output(tmp_path: Path) -> None:
    directory = tmp_path / "prepared"
    settings = MultiCoherentHoldoutSettings(
        initial_packets_per_electronic_branch=2,
    )
    model = DimerParameters(lambda_ep=1.5, gamma=0.5, drive_amplitude=1.0)
    times = np.array([0.0, 20.0])
    development = np.zeros((2, 31))

    manifest = freeze_blind_multi_coherent_inputs(
        directory,
        initial_parameters=_normalized_parameters(),
        development_times=times,
        development_closed_coordinates=development,
        settings=settings,
        parameters=model,
        input_hashes={"development": "a" * 64},
    )

    assert manifest["status"] == "model_inputs_frozen_reference_unopened"
    assert manifest["capacity"]["maximum_total_branch_packets"] == 24
    assert manifest["capacity"]["maximum_raw_coordinate_count"] == 96
    assert "holdout_exact" not in json.dumps(manifest)
    loaded = load_frozen_multi_coherent_inputs(directory)
    assert loaded.manifest_sha256 == manifest["manifest_sha256"]
    assert loaded.initial_conditions.closed_coordinates.shape == (3, 31)
    assert loaded.coordinate_scales.shape == (31,)

    with pytest.raises(FileExistsError):
        freeze_blind_multi_coherent_inputs(
            directory,
            initial_parameters=_normalized_parameters(),
            development_times=times,
            development_closed_coordinates=development,
            settings=settings,
            parameters=model,
        )


def test_frozen_model_trajectory_runs_without_a_reference_or_gate(
    tmp_path: Path,
) -> None:
    prepared_directory = tmp_path / "prepared"
    batch_directory = tmp_path / "batch"
    batch_directory.mkdir()
    run_directory = batch_directory / "coarse_central"
    settings = MultiCoherentHoldoutSettings(
        final_time=0.1,
        score_interval=(0.025, 0.05),
        sensitivity_interval=(0.05, 0.1),
        phonon_cutoff=2,
        initial_packets_per_electronic_branch=2,
        segment_length=0.05,
        output_sample_step=0.025,
        coarse_maximum_step=0.02,
        fine_maximum_step=0.01,
    )
    model = DimerParameters(lambda_ep=0.0, gamma=0.5, drive_amplitude=1.0)
    freeze_blind_multi_coherent_inputs(
        prepared_directory,
        initial_parameters=_decoupled_vacuum_parameters(),
        development_times=np.array([0.0, 0.1]),
        development_closed_coordinates=np.zeros((2, 31)),
        settings=settings,
        parameters=model,
    )

    summary = run_frozen_multi_coherent_model_trajectory(
        prepared_directory,
        run_directory,
        member="central",
        resolution="coarse",
    )

    assert summary["status"] == "complete"
    assert summary["offline_exact_comparison"] is None
    assert summary["parameters"]["output_sample_step"] == 0.025
    audit = json.loads(
        (run_directory / "blind_model_audit.json").read_text(encoding="utf-8")
    )
    assert audit["passed"]

    for resolution in ("coarse", "fine"):
        for member in ("central", "plus", "minus"):
            member_directory = batch_directory / f"{resolution}_{member}"
            if member_directory == run_directory:
                continue
            run_frozen_multi_coherent_model_trajectory(
                prepared_directory,
                member_directory,
                member=member,
                resolution=resolution,
            )
    for repeat in range(1, 4):
        run_frozen_multi_coherent_model_trajectory(
            prepared_directory,
            batch_directory / f"cost_model_repeat_{repeat}",
            member="central",
            resolution="fine",
        )
    first_cost_summary_path = (
        batch_directory / "cost_model_repeat_1" / "summary.json"
    )
    first_cost_summary_text = first_cost_summary_path.read_text(
        encoding="utf-8"
    )
    invalid_cost_summary = json.loads(first_cost_summary_text)
    invalid_cost_summary["parameters"]["maximum_step"] = (
        settings.coarse_maximum_step
    )
    first_cost_summary_path.write_text(
        json.dumps(invalid_cost_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fine numerical-setting mismatch"):
        seal_frozen_multi_coherent_model_cost(
            prepared_directory,
            batch_directory,
        )
    first_cost_summary_path.write_text(
        first_cost_summary_text,
        encoding="utf-8",
    )
    model_cost = seal_frozen_multi_coherent_model_cost(
        prepared_directory,
        batch_directory,
    )
    assert len(model_cost["wall_seconds"]) == 3
    model_cost_path = batch_directory / "model_cost_manifest.json"
    model_cost_text = model_cost_path.read_text(encoding="utf-8")
    invalid_model_cost = json.loads(model_cost_text)
    invalid_model_cost["median_wall_seconds"] *= 2.0
    model_cost_path.write_text(
        json.dumps(invalid_model_cost, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model cost manifest digest"):
        seal_frozen_multi_coherent_model_batch(
            prepared_directory,
            batch_directory,
        )
    model_cost_path.write_text(model_cost_text, encoding="utf-8")
    batch_manifest = seal_frozen_multi_coherent_model_batch(
        prepared_directory,
        batch_directory,
    )
    assert batch_manifest["status"] == (
        "model_outputs_frozen_reference_unopened"
    )
    assert len(batch_manifest["runs"]) == 6
    loaded_batch = load_frozen_multi_coherent_model_batch(
        prepared_directory,
        batch_directory,
    )
    assert loaded_batch.manifest_sha256 == batch_manifest["manifest_sha256"]
    score = score_frozen_multi_coherent_holdout(
        prepared_directory,
        batch_directory,
        tmp_path / "score",
    )
    assert score["status"] in {"scientific_pass", "scientific_failure"}
    assert score["cost_evaluation"]["evaluated"]
    assert score["cost_evaluation"]["repeat_count"] == 3
    with pytest.raises(RuntimeError, match="already been consumed"):
        score_frozen_multi_coherent_holdout(
            prepared_directory,
            batch_directory,
            tmp_path / "score_again",
        )


def test_closed_coordinate_scores_use_physical_blocks_and_fixed_scales() -> None:
    times = np.array([0.0, 1.0])
    reference = np.zeros((2, 31))
    model = reference.copy()
    model[:, 0] = 0.2
    model[:, 3] = 2.0
    scales = np.ones(31)
    scales[3] = 2.0

    scores = closed_coordinate_error_scores(
        times,
        model,
        reference,
        scales,
        interval=(0.0, 1.0),
    )

    np.testing.assert_allclose(scores.electron_trace_distance_maximum, 0.1)
    np.testing.assert_allclose(scores.block_rms["B"], 0.5)
    np.testing.assert_allclose(scores.block_maximum["B"], 1.0)
    assert scores.block_rms["N"] == 0.0
    expected_distance = np.sqrt((0.1**2 + 0.5**2) / 5.0)
    np.testing.assert_allclose(
        closed_coordinate_distance(model[0], reference[0], scales),
        expected_distance,
    )


def test_development_scales_use_observed_excursion_and_kinematic_floor() -> None:
    reference = np.zeros((3, 31))
    reference[:, 3] = (0.0, 0.2, -0.3)

    scales = development_coordinate_scales(
        reference,
        phonon_cutoff=16,
    )

    assert scales.shape == (31,)
    np.testing.assert_allclose(scales[:3], 1.0)
    np.testing.assert_allclose(scales[3], 0.5)
    np.testing.assert_allclose(scales[4:7], 0.008)
    np.testing.assert_allclose(scales[7:9], 0.064)
    np.testing.assert_allclose(scales[9:11], 0.064 * np.sqrt(2.0))
    np.testing.assert_allclose(scales[17:19], 0.016)
    np.testing.assert_allclose(scales[19:21], 0.016 / np.sqrt(2.0))


def test_energy_work_residual_integrates_declared_external_power() -> None:
    times = np.array([0.0, 0.5, 1.0, 1.5])
    power = np.array([2.0, 2.0, 2.0, 2.0])
    energies = 7.0 + 2.0 * times

    residual = energy_work_residual(times, energies, power)

    np.testing.assert_allclose(residual, 0.0, atol=2e-15)


def test_sensitivity_amplification_compares_matched_coordinate_pairs() -> None:
    scales = np.ones(31)
    model_minus = np.zeros((2, 31))
    model_plus = np.zeros((2, 31))
    exact_minus = np.zeros((2, 31))
    exact_plus = np.zeros((2, 31))
    model_plus[:, 3] = (0.2, 0.4)
    exact_plus[:, 3] = (0.2, 0.3)

    result = sensitivity_amplification(
        model_plus,
        model_minus,
        exact_plus,
        exact_minus,
        scales,
    )

    assert result.initial_distance > 0.0
    np.testing.assert_allclose(result.model, (1.0, 2.0))
    np.testing.assert_allclose(result.exact, (1.0, 1.5))


def test_resolution_scores_apply_the_frozen_robustness_rule() -> None:
    score_values = np.array([0.038, 0.040, 0.041])

    np.testing.assert_allclose(
        score_resolution_uncertainty(score_values),
        0.003,
    )
    certificate = certify_bounded_score(
        authoritative=0.040,
        cross_combination_scores=score_values,
        ceiling=0.050,
    )

    np.testing.assert_allclose(certificate.uncertainty, 0.003)
    np.testing.assert_allclose(certificate.robust_upper_bound, 0.043)
    np.testing.assert_allclose(certificate.resolution_limit, 0.005)
    assert certificate.numerically_resolved
    assert certificate.passes

    unresolved = certify_bounded_score(
        authoritative=0.040,
        cross_combination_scores=np.array([0.030, 0.041]),
        ceiling=0.050,
    )
    assert not unresolved.numerically_resolved
    assert not unresolved.passes

    pointwise = pointwise_resolution_uncertainty(
        np.array(
            [
                [1.00, 2.00, 3.00],
                [1.01, 1.98, 3.04],
                [0.99, 2.01, 2.97],
            ]
        )
    )
    np.testing.assert_allclose(pointwise, (0.02, 0.03, 0.07))


def test_multi_coherent_rhs_uses_the_declared_drive_protocol() -> None:
    parameters = _normalized_parameters()
    model = DimerParameters(lambda_ep=1.5, gamma=0.5, drive_amplitude=1.0)
    single = GaussianSineDrive.from_parameters(model)
    double = GaussianSineDrive.from_parameters(model, delays=(0.0, 8.0))

    before_single = multi_coherent_rhs(
        1.2,
        parameters,
        model,
        relative_dimension=12,
        drive_protocol=single,
    )
    before_double = multi_coherent_rhs(
        1.2,
        parameters,
        model,
        relative_dimension=12,
        drive_protocol=double,
    )
    np.testing.assert_allclose(before_double, before_single, atol=2e-13)

    after_single = multi_coherent_rhs(
        8.7,
        parameters,
        model,
        relative_dimension=12,
        drive_protocol=single,
    )
    after_double = multi_coherent_rhs(
        8.7,
        parameters,
        model,
        relative_dimension=12,
        drive_protocol=double,
    )
    assert np.linalg.norm(after_double - after_single) > 1e-3


def test_two_coherent_global_fit_recovers_a_two_packet_branch() -> None:
    parameters = _normalized_parameters()
    target = multi_coherent_state(parameters, relative_dimension=16)
    target_blocks = target.reshape(4, 16)
    target_blocks[1:] = 0.0
    target = target_blocks.reshape(-1)
    target /= np.linalg.norm(target)

    fit = fit_two_coherent_electron_relative_state(
        target,
        maximum_iterations=50,
        population_size=7,
        seed=4,
    )

    assert fit.fidelity > 0.999999
    assert fit.block_fidelities[0] > 0.999999


def test_decoupled_common_packets_have_an_exact_tangent_velocity() -> None:
    relative_dimension = 20
    parameters = _normalized_parameters()
    model = DimerParameters(
        lambda_ep=0.0,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    hamiltonian = relative_holstein_hamiltonian(
        0.41,
        model,
        relative_dimension=relative_dimension,
    )

    projection = project_schrodinger_velocity(
        parameters,
        hamiltonian,
        relative_dimension=relative_dimension,
    )

    assert projection.relative_residual < 2e-8
    assert projection.tangent_rank == parameters.size - 2
    assert projection.geometric_tangent_rank == parameters.size - 2
    assert projection.geometric_gram_relative_threshold == 1e-10


def test_schrodinger_projection_uses_the_horizontal_tangent() -> None:
    parameters = _normalized_parameters()
    coefficients, displacements = unpack_multi_coherent_parameters(parameters)
    parameters = pack_multi_coherent_parameters(
        2.7 * coefficients,
        displacements,
    )
    model = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    hamiltonian = relative_holstein_hamiltonian(
        0.63,
        model,
        relative_dimension=12,
    )
    state, horizontal = (
        normalized_multi_coherent_state_and_horizontal_tangent(
            parameters,
            relative_dimension=12,
        )
    )

    projection = project_schrodinger_velocity(
        parameters,
        hamiltonian,
        relative_dimension=12,
        regularization="truncated_svd",
        relative_singular_value_cutoff=1e-10,
    )

    np.testing.assert_allclose(
        projection.projected_velocity,
        horizontal @ projection.parameter_velocity,
        atol=2e-13,
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        np.vdot(state, projection.projected_velocity),
        0.0,
        atol=2e-13,
    )


def test_parameter_retraction_preserves_the_projective_state() -> None:
    parameters = _normalized_parameters()
    coefficients, displacements = unpack_multi_coherent_parameters(parameters)
    parameters = pack_multi_coherent_parameters(
        2.3j * coefficients,
        displacements,
    )
    before = multi_coherent_state(parameters, relative_dimension=12)
    before /= np.linalg.norm(before)

    retracted = retract_multi_coherent_parameters(
        parameters,
        relative_dimension=12,
    )
    after = multi_coherent_state(retracted, relative_dimension=12)
    retracted_coefficients, _ = unpack_multi_coherent_parameters(retracted)
    anchor = retracted_coefficients.reshape(-1)[
        np.argmax(np.abs(retracted_coefficients))
    ]

    np.testing.assert_allclose(np.vdot(after, after), 1.0, atol=2e-14)
    np.testing.assert_allclose(abs(np.vdot(before, after)), 1.0, atol=2e-14)
    np.testing.assert_allclose(anchor.imag, 0.0, atol=2e-14)
    assert anchor.real >= 0.0
    np.testing.assert_array_equal(
        retract_multi_coherent_parameters(
            retracted,
            relative_dimension=12,
        ),
        retracted,
    )


def test_symmetric_generator_kick_is_model_representable_and_normalized() -> None:
    parameters = _normalized_parameters()
    relative_dimension = 12
    generator = np.kron(
        np.diag([1.0, -1.0, 0.5, -0.5]),
        np.eye(relative_dimension),
    )

    kick = symmetric_projected_generator_kick(
        parameters,
        generator,
        relative_dimension=relative_dimension,
        step=1e-4,
        relative_singular_value_cutoff=1e-10,
    )

    plus = multi_coherent_state(
        kick.plus_parameters,
        relative_dimension=relative_dimension,
    )
    minus = multi_coherent_state(
        kick.minus_parameters,
        relative_dimension=relative_dimension,
    )
    np.testing.assert_allclose(np.vdot(plus, plus), 1.0, atol=2e-13)
    np.testing.assert_allclose(np.vdot(minus, minus), 1.0, atol=2e-13)
    assert kick.projected_direction_norm > 0.0
    assert np.linalg.norm(plus - minus) > 1e-4


def test_diagonal_kick_generator_is_centered_and_variance_normalized() -> None:
    parameters = _normalized_parameters()
    relative_dimension = 12
    state, _ = normalized_multi_coherent_state_and_horizontal_tangent(
        parameters,
        relative_dimension=relative_dimension,
    )

    generator = normalized_diagonal_kick_generator(
        parameters,
        relative_dimension=relative_dimension,
    )

    mean = np.vdot(state, generator @ state)
    variance = np.vdot(state, generator @ generator @ state)
    np.testing.assert_allclose(mean, 0.0, atol=2e-13)
    np.testing.assert_allclose(variance, 1.0, atol=2e-13)


def test_residual_packet_spawn_is_deterministic_and_state_continuous() -> None:
    relative_dimension = 20
    parameters = _normalized_parameters()
    model = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    hamiltonian = relative_holstein_hamiltonian(
        0.41,
        model,
        relative_dimension=relative_dimension,
    )
    projection = project_schrodinger_velocity(
        parameters,
        hamiltonian,
        relative_dimension=relative_dimension,
        regularization="tikhonov",
        relative_damping=3e-3,
    )
    residual = projection.target_velocity - projection.projected_velocity

    first = spawn_residual_coherent_packets(
        parameters,
        residual,
        relative_dimension=relative_dimension,
        maximum_iterations=8,
        population_size=4,
        seed=23,
    )
    second = spawn_residual_coherent_packets(
        parameters,
        residual,
        relative_dimension=relative_dimension,
        maximum_iterations=8,
        population_size=4,
        seed=23,
    )

    np.testing.assert_array_equal(first.parameters, second.parameters)
    assert first.previous_packet_count == 2
    assert first.packet_count == 3
    assert first.state_discontinuity == 0.0
    assert first.norm_change == 0.0
    before = multi_coherent_state(
        parameters,
        relative_dimension=relative_dimension,
    )
    after = multi_coherent_state(
        first.parameters,
        relative_dimension=relative_dimension,
    )
    np.testing.assert_array_equal(after, before)
    coefficients, _ = unpack_multi_coherent_parameters(first.parameters)
    np.testing.assert_array_equal(coefficients[:, -1], 0.0)
    residual_norms = np.linalg.norm(
        residual.reshape(4, relative_dimension),
        axis=1,
    )
    assert first.parent_electronic_index == int(np.argmax(residual_norms))


def test_relative_state_moment_contraction_and_derivative() -> None:
    hierarchy = moment_hierarchy(2)
    vacuum = np.zeros(4 * 10, dtype=complex)
    vacuum[0] = 1.0
    coordinates = relative_state_moment_coordinates(
        vacuum,
        hierarchy,
        center_amplitude=0.0j,
    )
    assert abs(hierarchy.moment_value(coordinates, MomentKey("I", "I", 2, 0)) - 0.5) < 1e-14
    assert abs(hierarchy.moment_value(coordinates, MomentKey("I", "I", 0, 2)) - 0.5) < 1e-14
    assert abs(hierarchy.moment_value(coordinates, MomentKey("I", "I", 1, 1))) < 1e-14

    parameters = _normalized_parameters()
    state, tangent = multi_coherent_state_and_tangent(
        parameters,
        relative_dimension=10,
    )
    generator = np.random.default_rng(12)
    parameter_velocity = generator.normal(size=parameters.size)
    state_derivative = tangent @ parameter_velocity
    contracted = relative_state_moment_derivative(
        state,
        state_derivative,
        hierarchy,
    )
    step = 1e-6
    forward = relative_state_moment_coordinates(
        multi_coherent_state(
            parameters + step * parameter_velocity,
            relative_dimension=10,
        ),
        hierarchy,
        center_amplitude=0.0j,
    )
    backward = relative_state_moment_coordinates(
        multi_coherent_state(
            parameters - step * parameter_velocity,
            relative_dimension=10,
        ),
        hierarchy,
        center_amplitude=0.0j,
    )
    np.testing.assert_allclose(
        contracted,
        (forward - backward) / (2.0 * step),
        atol=2e-8,
        rtol=2e-7,
    )


def test_relative_product_state_contracts_to_the_declared_31_coordinates() -> None:
    hierarchy = moment_hierarchy(2)
    state = np.zeros(4 * 7, dtype=complex)
    state[0] = 1.0

    coordinates = relative_state_closed_coordinates(
        state,
        hierarchy,
        center_amplitude=0.0j,
    )

    expected = np.zeros(31)
    expected[0] = 1.0
    np.testing.assert_allclose(coordinates, expected, atol=2e-14)


def test_multi_coherent_analysis_writes_decoupled_gate(tmp_path: Path) -> None:
    summary = run_multi_coherent_analysis(
        tmp_path,
        parameters=DimerParameters(
            lambda_ep=0.0,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        sample_times=(0.0, 0.1),
        packet_counts=(1,),
        selected_packet_count=1,
        phonon_cutoff=2,
        convergence_cutoffs=(2,),
        convergence_times=(0.0, 0.1),
        maximum_step=0.02,
        fit_maximum_iterations=10,
        fit_population_size=4,
    )

    assert summary["validation_gate"]["all_gates_passed"]
    prefix = "multi_coherent_velocity_gate"
    assert (tmp_path / f"{prefix}.npz").is_file()
    assert (tmp_path / f"{prefix}.png").is_file()
    manifest = json.loads(
        (tmp_path / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert "summary.json" in manifest["artifact_hashes"]


def test_multi_coherent_decoupled_autonomous_propagation(tmp_path: Path) -> None:
    summary = run_multi_coherent_propagation(
        tmp_path,
        parameters=DimerParameters(
            lambda_ep=0.0,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        final_time=0.1,
        sample_step=0.05,
        maximum_step=0.02,
        phonon_cutoff=2,
        packet_count=1,
        fit_maximum_iterations=10,
        fit_population_size=4,
    )

    assert summary["validation_gate"]["passed"]
    assert summary["comparison"]["final_state_fidelity"] > 1.0 - 1e-9
    assert not summary["initialization"][
        "exact_reference_used_after_t0_by_autonomous_rhs"
    ]
    prefix = "multi_coherent_autonomous_trajectory"
    assert (tmp_path / f"{prefix}.npz").is_file()
    assert (tmp_path / f"{prefix}.png").is_file()


def test_segmented_horizon_retains_checkpoints(tmp_path: Path) -> None:
    gate_directory = tmp_path / "gate"
    run_multi_coherent_analysis(
        gate_directory,
        parameters=DimerParameters(
            lambda_ep=0.0,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        sample_times=(0.0, 0.1),
        packet_counts=(1,),
        selected_packet_count=1,
        phonon_cutoff=2,
        convergence_cutoffs=(2,),
        convergence_times=(0.0, 0.1),
        maximum_step=0.02,
        fit_maximum_iterations=10,
        fit_population_size=4,
    )
    run_directory = tmp_path / "long"

    summary = run_segmented_multi_coherent_horizon(
        run_directory,
        gate_directory=gate_directory,
        parameters=DimerParameters(
            lambda_ep=0.0,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        final_time=0.1,
        segment_length=0.05,
        output_sample_step=0.025,
        segment_timeout_seconds=5.0,
        maximum_step=0.02,
        phonon_cutoff=2,
        packet_count=1,
    )

    assert summary["status"] == "complete"
    assert summary["progress"]["last_completed_time"] == 0.1
    assert summary["offline_exact_comparison"]["final_state_fidelity"] > (
        1.0 - 1e-9
    )
    assert summary["offline_exact_comparison"][
        "maximum_closed_coordinate_relative_error"
    ] < 1e-6
    with np.load(run_directory / "segmented_horizon.npz") as arrays:
        np.testing.assert_allclose(
            arrays["times"],
            (0.0, 0.025, 0.05, 0.075, 0.1),
        )
        assert arrays["exact_closed_coordinates"].shape == (5, 31)
    assert summary["parameters"]["output_sample_step"] == 0.025
    assert summary["parameters"]["relative_tolerance"] == 1e-7
    assert summary["parameters"]["absolute_tolerance"] == 1e-9
    assert (run_directory / "checkpoint.npz").is_file()
    progress = json.loads(
        (run_directory / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "complete"


def test_segmented_horizon_records_state_continuous_capacity_spawn(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate_directory = tmp_path / "gate"
    run_multi_coherent_analysis(
        gate_directory,
        parameters=DimerParameters(
            lambda_ep=0.0,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        sample_times=(0.0, 0.1),
        packet_counts=(1,),
        selected_packet_count=1,
        phonon_cutoff=2,
        convergence_cutoffs=(2,),
        convergence_times=(0.0, 0.1),
        maximum_step=0.02,
        fit_maximum_iterations=10,
        fit_population_size=4,
    )
    run_directory = tmp_path / "adaptive"

    original_projection = long_horizon.project_schrodinger_velocity

    def controlled_projection(parameters, *args, **kwargs):
        projection = original_projection(parameters, *args, **kwargs)
        packet_count = np.asarray(parameters).size // 16
        return replace(
            projection,
            absolute_residual=1.0 / packet_count,
            relative_residual=1.0 / packet_count,
        )

    monkeypatch.setattr(
        long_horizon,
        "project_schrodinger_velocity",
        controlled_projection,
    )

    def reject_exact_online(*args, **kwargs):
        del args, kwargs
        raise AssertionError("exact reference entered autonomous propagation")

    monkeypatch.setattr(
        long_horizon,
        "exact_holstein_wavefunction_trajectory_for_diagnostics",
        reject_exact_online,
    )
    monkeypatch.setattr(
        long_horizon,
        "exact_holstein_moment_hierarchy_trajectory",
        reject_exact_online,
    )

    model = DimerParameters(
        lambda_ep=0.0,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    drive_protocol = GaussianSineDrive.from_parameters(
        model,
        delays=(0.0, 8.0),
    )
    with np.load(gate_directory / "multi_coherent_velocity_gate.npz") as arrays:
        initial_override = np.asarray(
            arrays["k1_fitted_parameters"][0],
            dtype=float,
        )
    summary = run_segmented_multi_coherent_horizon(
        run_directory,
        gate_directory=None,
        parameters=model,
        final_time=0.1,
        segment_length=0.05,
        segment_timeout_seconds=5.0,
        maximum_step=0.02,
        phonon_cutoff=2,
        packet_count=1,
        adaptive_capacity=True,
        maximum_packet_count=7,
        spawn_relative_residual_threshold=0.0,
        spawn_absolute_residual_threshold=0.0,
        spawn_fit_maximum_iterations=3,
        spawn_fit_population_size=3,
        compare_exact=False,
        drive_protocol=drive_protocol,
        initial_parameters_override=initial_override,
    )

    capacity = summary["capacity"]
    assert capacity["mode"] == "adaptive_residual_spawn"
    assert capacity["spawn_count"] == 1
    assert capacity["final_packet_count"] == 2
    assert capacity["maximum_packet_count"] == 7
    assert capacity["initial_packets_per_electronic_branch"] == 1
    assert capacity["final_packets_per_electronic_branch"] == 2
    assert capacity["final_total_branch_packets"] == 8
    assert capacity["final_raw_coordinate_count"] == 32
    assert not capacity["online_exact_reference_used"]
    assert summary["offline_exact_comparison"] is None
    assert summary["resource_usage"]["wall_seconds"] > 0.0
    assert summary["resource_usage"]["maximum_resident_set_bytes"] > 0
    assert summary["parameters"]["drive_protocol"] == {
        "amplitude": 1.0,
        "delays": [0.0, 8.0],
        "pulse_width": 1.0,
    }
    assert summary["initialization"]["source"] == "explicit_model_chart"
    assert len(summary["initialization"]["parameter_sha256"]) == 64
    spawn = capacity["spawns"][0]
    assert spawn["state_discontinuity"] == 0.0
    assert spawn["norm_change"] == 0.0
    with np.load(run_directory / "checkpoint.npz") as arrays:
        np.testing.assert_array_equal(
            arrays["packet_count_trajectory"],
            np.array([1, 2, 2]),
        )
        assert arrays["parameter_trajectory"].shape == (3, 32)
        assert np.all(np.isnan(arrays["parameter_trajectory"][0, 16:]))
    with np.load(run_directory / "segmented_horizon.npz") as arrays:
        assert arrays["closed_coordinates"].shape == (3, 31)
        assert arrays["energy"].shape == (3,)
        assert arrays["external_power"].shape == (3,)
        assert arrays["energy_work_residual"].shape == (3,)
    assert summary["work_balance"]["maximum_absolute_residual"] < 1e-5
