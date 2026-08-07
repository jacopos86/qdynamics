from __future__ import annotations

from math import factorial
import json
from pathlib import Path

import numpy as np

from paper5.stability import (
    ConditionalRelativeState,
    analyze_conditional_packet,
    conditional_relative_state,
    electron_relative_state,
    fit_single_gaussian_packet,
    fit_two_coherent_packets,
    local_to_normal_mode_transform,
    DimerParameters,
)
from paper5.stability.conditional_packet_analysis import (
    run_conditional_packet_analysis,
)


def _coherent_state(alpha: complex, dimension: int) -> np.ndarray:
    factorials = np.sqrt(
        np.asarray([factorial(index) for index in range(dimension)])
    )
    state = (
        np.exp(-0.5 * abs(alpha) ** 2)
        * alpha ** np.arange(dimension)
        / factorials
    )
    return state / np.linalg.norm(state)


def test_local_to_normal_mode_transform_is_isometric_with_known_signs() -> None:
    cutoff = 2
    local_dimension = cutoff + 1
    normal_dimension = 2 * cutoff + 1
    transform = local_to_normal_mode_transform(cutoff)

    np.testing.assert_allclose(
        (transform.getH() @ transform).toarray(),
        np.eye(local_dimension**2),
        atol=2e-14,
    )

    local_mode_0 = np.zeros(local_dimension**2)
    local_mode_0[local_dimension] = 1.0
    expected_mode_0 = np.zeros(normal_dimension**2)
    expected_mode_0[normal_dimension] = 1.0 / np.sqrt(2.0)
    expected_mode_0[1] = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(transform @ local_mode_0, expected_mode_0)

    local_mode_1 = np.zeros(local_dimension**2)
    local_mode_1[1] = 1.0
    expected_mode_1 = np.zeros(normal_dimension**2)
    expected_mode_1[normal_dimension] = 1.0 / np.sqrt(2.0)
    expected_mode_1[1] = -1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(transform @ local_mode_1, expected_mode_1)


def test_decoupled_product_vacuum_is_one_exact_conditional_packet() -> None:
    cutoff = 2
    local_dimension = cutoff + 1
    state = np.zeros(4 * local_dimension**2, dtype=complex)
    for electronic_index in range(4):
        state[electronic_index * local_dimension**2] = 0.5

    conditional = conditional_relative_state(
        state,
        electronic_index=1,
        phonon_cutoff=cutoff,
    )
    metrics = analyze_conditional_packet(
        conditional,
        single_gaussian_random_starts=0,
        single_gaussian_maximum_iterations=30,
        two_packet_maximum_iterations=5,
        two_packet_population_size=4,
        husimi_grid_points=21,
    )

    assert abs(conditional.probability - 0.25) < 1e-14
    assert abs(metrics.center_relative_factorization - 1.0) < 1e-14
    assert abs(metrics.relative_purity - 1.0) < 1e-14
    assert metrics.gaussian_non_gaussianity < 1e-14
    assert metrics.husimi_peak_count == 1
    assert metrics.single_gaussian_fit.fidelity > 1.0 - 1e-13
    assert metrics.two_coherent_fit.fidelity > 1.0 - 1e-10

    global_state = electron_relative_state(
        state,
        phonon_cutoff=cutoff,
    )
    expected = np.zeros(4 * (2 * cutoff + 1), dtype=complex)
    expected[:: 2 * cutoff + 1] = 0.5
    assert global_state.center_factorization > 1.0 - 1e-14
    assert abs(np.vdot(expected, global_state.state)) ** 2 > 1.0 - 1e-14


def test_single_gaussian_fit_recovers_a_displaced_vacuum() -> None:
    state = _coherent_state(0.8 - 0.45j, 16)

    fit = fit_single_gaussian_packet(
        state,
        random_starts=0,
        maximum_iterations=100,
    )

    assert fit.fidelity > 1.0 - 1e-10


def test_two_coherent_fit_detects_a_non_gaussian_cat_state() -> None:
    dimension = 18
    packet_0 = _coherent_state(1.35 + 0.15j, dimension)
    packet_1 = _coherent_state(-1.05 + 0.35j, dimension)
    state = packet_0 + (0.55 - 0.7j) * packet_1
    state /= np.linalg.norm(state)

    single = fit_single_gaussian_packet(
        state,
        random_starts=1,
        maximum_iterations=180,
        seed=2,
    )
    two_packet = fit_two_coherent_packets(
        state,
        maximum_iterations=50,
        population_size=7,
        seed=2,
    )

    assert two_packet.fidelity > 0.999999
    assert two_packet.fidelity - single.fidelity > 0.03


def test_conditional_packet_rejects_incompatible_state_shape() -> None:
    with np.testing.assert_raises_regex(ValueError, "expected state shape"):
        conditional_relative_state(
            np.zeros(7),
            electronic_index=0,
            phonon_cutoff=1,
        )


def test_packet_metrics_accept_an_explicit_conditional_state() -> None:
    state = _coherent_state(0.3j, 8)
    density = np.outer(state, state.conj())
    conditional = ConditionalRelativeState(
        electronic_index=3,
        probability=0.2,
        density_matrix=density,
        dominant_state=state,
        center_relative_factorization=1.0,
    )

    metrics = analyze_conditional_packet(
        conditional,
        single_gaussian_random_starts=0,
        single_gaussian_maximum_iterations=50,
        two_packet_maximum_iterations=5,
        two_packet_population_size=4,
        husimi_grid_points=21,
    )

    assert metrics.electronic_index == 3
    assert metrics.probability == 0.2


def test_conditional_packet_analysis_writes_retrievable_control_gate(
    tmp_path: Path,
) -> None:
    summary = run_conditional_packet_analysis(
        tmp_path,
        parameters=DimerParameters(
            lambda_ep=0.0,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        sample_times=(0.0, 0.1),
        phonon_cutoff=2,
        convergence_cutoffs=(2,),
        convergence_times=(0.0, 0.1),
        maximum_step=0.02,
        single_gaussian_random_starts=0,
        single_gaussian_maximum_iterations=30,
        two_packet_maximum_iterations=5,
        two_packet_population_size=4,
        husimi_grid_points=21,
    )

    aggregate = summary["aggregate_metrics"]
    assert aggregate["minimum_center_relative_factorization"] > 1.0 - 1e-12
    assert aggregate["worst_single_gaussian_infidelity"] < 1e-10
    assert summary["validation_gate"]["two_coherent_compression_passed"]
    prefix = "conditional_relative_packet_gate"
    assert (tmp_path / f"{prefix}.npz").is_file()
    assert (tmp_path / f"{prefix}.png").is_file()
    manifest = json.loads(
        (tmp_path / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert "summary.json" in manifest["artifact_hashes"]
