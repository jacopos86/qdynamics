from __future__ import annotations

import numpy as np

from paper5.stability.exact_reference import (
    exact_holstein_moment_hierarchy_trajectory,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.adaptive_positive_moment import (
    HIDDEN_RELATIVE_MOMENT_KEYS,
)
from paper5.stability.moment_hierarchy import FOURTH_ORDER_HIERARCHY
from paper5.stability.positive_moment_completion import (
    PositiveFourthMomentCompletion,
    PositiveMomentCompletionSettings,
    _spin_exchange_coefficient_blocks,
    _spin_exchange_transform,
    pauli_weyl_moment_matrix,
)


def _exact_initial_moments():
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )
    exact = exact_holstein_moment_hierarchy_trajectory(
        parameters,
        hierarchy=FOURTH_ORDER_HIERARCHY,
        sample_times=np.asarray([0.0, 0.01]),
        phonon_cutoff=16,
        maximum_step=0.01,
    )
    _, moments = FOURTH_ORDER_HIERARCHY.unpack(exact.coordinates[0])
    return moments


def test_exact_fourth_moments_form_a_positive_word_gram_matrix() -> None:
    moments = _exact_initial_moments()
    matrix = pauli_weyl_moment_matrix(moments)
    np.testing.assert_allclose(matrix, matrix.conjugate().T, atol=2e-12)
    # Canonical CCR and the local cutoff-16 algebra differ by a top-projector
    # term; the discrepancy is reported rather than imported online.
    assert np.linalg.eigvalsh(matrix)[0] > -2e-4


def test_positive_completion_returns_psd_fourth_moments() -> None:
    exact = _exact_initial_moments()
    lower = {key: value for key, value in exact.items() if key.degree <= 3}
    completion = PositiveFourthMomentCompletion(
        PositiveMomentCompletionSettings(
            logdet_weight=1e-3,
            solver_tolerance=1e-7,
            maximum_iterations=2_000,
        )
    ).complete(lower)
    assert completion.success, completion.message
    assert completion.minimum_moment_matrix_eigenvalue >= -1e-6
    matrix = pauli_weyl_moment_matrix(completion.moments)
    assert np.linalg.eigvalsh(matrix)[0] >= -1e-6


def test_zero_cumulant_prior_is_reported_without_cone_repair() -> None:
    exact = _exact_initial_moments()
    lower = {key: value for key, value in exact.items() if key.degree <= 3}
    selector = PositiveFourthMomentCompletion()
    result = selector.prior_result(lower)
    assert result.success
    assert result.iterations == 0
    assert result.scaled_prior_distance == 0.0
    np.testing.assert_allclose(
        [result.frontier_moments[key] for key in selector.frontier_keys],
        [
            result.prior_frontier_moments[key]
            for key in selector.frontier_keys
        ],
    )


def test_spin_exchange_blocks_exactly_reconstruct_the_full_cone() -> None:
    moments = _exact_initial_moments()
    transform, split = _spin_exchange_transform()
    blocks = _spin_exchange_coefficient_blocks()
    assembled = [
        np.zeros((split, split), dtype=complex),
        np.zeros((transform.shape[0] - split,) * 2, dtype=complex),
    ]
    for key, pair in blocks.items():
        value = 1.0 if key.degree == 0 else moments[key]
        for index in range(2):
            assembled[index] += value * pair[index]
    transformed = transform.conjugate().T @ (
        pauli_weyl_moment_matrix(moments) @ transform
    )
    np.testing.assert_allclose(
        transformed[:split, :split], assembled[0], atol=2e-13, rtol=0.0
    )
    np.testing.assert_allclose(
        transformed[split:, split:], assembled[1], atol=2e-13, rtol=0.0
    )
    np.testing.assert_allclose(
        transformed[:split, split:], 0.0, atol=2e-13, rtol=0.0
    )


def test_spin_exchange_block_retraction_is_separately_certified() -> None:
    exact = _exact_initial_moments()
    lower = {key: value for key, value in exact.items() if key.degree <= 3}
    common = {
        "phonon_envelope": 16.0,
        "logdet_shift": 1e-5,
        "solver_tolerance": 1e-7,
        "maximum_iterations": 2_000,
    }
    serial = PositiveFourthMomentCompletion(
        PositiveMomentCompletionSettings(
            **common,
            cone_representation="full",
        )
    ).retract_lower_moments(
        lower,
        adjustable_keys=HIDDEN_RELATIVE_MOMENT_KEYS,
    )
    blocked = PositiveFourthMomentCompletion(
        PositiveMomentCompletionSettings(
            **common,
            cone_representation="spin_exchange_blocks",
            clarabel_max_threads=4,
        )
    ).retract_lower_moments(
        lower,
        adjustable_keys=HIDDEN_RELATIVE_MOMENT_KEYS,
    )

    assert serial.success, serial.message
    assert blocked.success, blocked.message
    assert serial.minimum_moment_matrix_eigenvalue >= -1e-6
    assert blocked.minimum_moment_matrix_eigenvalue >= -1e-6
