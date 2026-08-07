from __future__ import annotations

import numpy as np

from paper5.stability.adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    HIDDEN_RELATIVE_MOMENT_KEYS,
    matrix_derivative_to_raw_moment_velocity,
    matrix_state_to_raw_moment_coordinates,
    raw_moment_velocity_to_matrix_derivative,
)
from paper5.stability.apcm_moment_projection import (
    SymmetryReducedAPCMGeometry,
    state_lower_moments,
)
from paper5.stability.apcm_positive_extension import (
    APCMExtensionSettings,
    SymmetryReducedPositiveExtension,
    _OperatorKey,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import MatrixDimerState
from paper5.stability.moment_hierarchy import (
    IDENTITY,
    MomentKey,
    THIRD_ORDER_HIERARCHY,
    _commutator,
    _hamiltonian_terms,
)


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


def _random_matrix_derivative(
    rng: np.random.Generator,
) -> MatrixDimerState:
    electron_vector = rng.normal(size=3)
    pauli = np.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ],
        dtype=complex,
    )
    electron = 0.5 * np.tensordot(electron_vector, pauli, axes=(0, 0))
    coherent = rng.normal(size=2) + 1j * rng.normal(size=2)
    normal_seed = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    normal = 0.5 * (normal_seed + normal_seed.conjugate().T)
    anomalous_seed = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    anomalous = 0.5 * (anomalous_seed + anomalous_seed.T)
    correlation = np.empty((2, 2, 2), dtype=complex)
    for mode in range(2):
        coefficients = rng.normal(size=3) + 1j * rng.normal(size=3)
        correlation[mode] = 0.5 * np.tensordot(
            coefficients,
            pauli,
            axes=(0, 0),
        )
    return MatrixDimerState(
        electron_density=electron,
        coherent_phonon=coherent,
        phonon_density=normal,
        anomalous_phonon_density=anomalous,
        electron_phonon_correlation=correlation,
    )


def test_raw_reconstruction_differential_is_bidirectional() -> None:
    rng = np.random.default_rng(8102)
    moments = {
        key: _thermal_moment(key)
        for key in THIRD_ORDER_HIERARCHY.moment_keys
    }
    state = THIRD_ORDER_HIERARCHY.to_matrix_state(
        THIRD_ORDER_HIERARCHY.pack(0.0j, moments)
    )

    for _ in range(20):
        derivative = _random_matrix_derivative(rng)
        velocity = matrix_derivative_to_raw_moment_velocity(
            state,
            derivative,
        )
        reconstructed = raw_moment_velocity_to_matrix_derivative(
            state,
            velocity,
        )
        for name in (
            "electron_density",
            "coherent_phonon",
            "phonon_density",
            "anomalous_phonon_density",
            "electron_phonon_correlation",
        ):
            np.testing.assert_allclose(
                getattr(reconstructed, name),
                getattr(derivative, name),
                atol=5e-13,
                rtol=0.0,
            )


def test_relative_face_projection_preserves_a_viable_zero_velocity() -> None:
    moments = {
        key: _thermal_moment(key)
        for key in THIRD_ORDER_HIERARCHY.moment_keys
    }
    state = THIRD_ORDER_HIERARCHY.to_matrix_state(
        THIRD_ORDER_HIERARCHY.pack(0.0j, moments)
    )
    raw = matrix_state_to_raw_moment_coordinates(state)
    hidden = np.asarray(
        [moments[key] for key in HIDDEN_RELATIVE_MOMENT_KEYS],
        dtype=float,
    )
    extension = SymmetryReducedPositiveExtension(
        APCMExtensionSettings(
            logdet_weight=1e-3,
            conic_tolerance=1e-7,
        )
    )
    lower = state_lower_moments(raw, hidden)
    warm = {
        key: _thermal_moment(key) for key in extension.frontier_keys
    }
    completion = extension.complete(lower, warm_frontier=warm)
    assert completion.success, completion.message
    assert completion.facial_reduction_certified
    geometry = SymmetryReducedAPCMGeometry(extension)
    retained_target = np.zeros(raw.size, dtype=float)
    auxiliary_target = np.zeros(hidden.size, dtype=float)

    projected = geometry.project_velocity(
        0.0,
        raw,
        hidden,
        retained_target,
        auxiliary_target,
        completion,
        DimerParameters(lambda_ep=1.5),
    )

    np.testing.assert_allclose(
        projected.retained_velocity,
        retained_target,
        atol=2e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        projected.auxiliary_velocity,
        auxiliary_target,
        atol=2e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(projected.frontier_velocity, 0.0)


def test_hidden_liouvillian_frontier_contains_every_generated_moment() -> None:
    extension = SymmetryReducedPositiveExtension()
    generated = {
        key
        for observable in HIDDEN_RELATIVE_MOMENT_KEYS
        for _, hamiltonian_word in _hamiltonian_terms(
            0.37,
            DimerParameters(lambda_ep=1.5),
        )
        for key in _commutator(hamiltonian_word, observable)
    }

    assert generated.issubset(
        set(extension.lower_keys).union(extension.frontier_keys)
    )
    assert len({key for key in generated if key.degree == 4}) == 16
    assert set(extension.rhs_frontier_keys) == generated.intersection(
        extension.frontier_keys
    )
    assert set(extension.rhs_frontier_keys).isdisjoint(
        extension.auxiliary_frontier_keys
    )
    assert set(extension.rhs_frontier_keys).union(
        extension.auxiliary_frontier_keys
    ) == set(extension.frontier_keys)


def test_nested_diagnostic_appends_every_candidate_descendant_as_a_halfword() -> None:
    parameters = DimerParameters(lambda_ep=1.5)
    extension = SymmetryReducedPositiveExtension(
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    candidates = extension.rhs_frontier_keys
    descendants = tuple(
        sorted(
            {
                key
                for candidate in candidates
                for _, hamiltonian_word in _hamiltonian_terms(0.37, parameters)
                for key in _commutator(hamiltonian_word, candidate)
                if key.degree > 0
            },
            key=lambda key: (
                key.degree,
                key.spin_up,
                key.spin_down,
                key.x_power,
                key.p_power,
            ),
        )
    )
    diagnostic = SymmetryReducedPositiveExtension(
        active_keys=tuple((*extension.active_keys, *candidates)),
        additional_halfword_keys=descendants,
    )
    diagnostic_words = set(diagnostic.words)

    assert descendants
    assert all(
        _OperatorKey(
            key.spin_up,
            key.spin_down,
            key.x_power,
            key.p_power,
        )
        in diagnostic_words
        for key in descendants
    )
    assert diagnostic.dimension > len(extension.words) + len(candidates)
