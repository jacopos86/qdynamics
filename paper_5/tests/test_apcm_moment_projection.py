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
    APCMProjectionError,
    APCMProjectionSettings,
    SymmetryReducedAPCMGeometry,
    _kernel_basis,
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


def _ambiguous_face_matrix(tolerance: float) -> np.ndarray:
    """One scaled matrix whose smallest eigenvalue sits in the refinement band.

    The declared refinement counts eigenvalues at or below 0.5, 1 and 2 null
    tolerances; an eigenvalue at exactly one tolerance is counted by two of
    the three, which is the refusal the committed path raises.
    """

    return np.diag(
        np.asarray([tolerance, 1.0, 2.0, 3.0], dtype=float)
    ).astype(complex)


def test_the_default_face_policy_refuses_an_unstable_numerical_face() -> None:
    tolerance = 1e-8
    matrix = _ambiguous_face_matrix(tolerance)
    try:
        _kernel_basis(matrix, tolerance, name="extended operator moment matrix")
    except APCMProjectionError as error:
        assert "numerical face is unstable" in str(error)
    else:  # pragma: no cover - the committed path must refuse
        raise AssertionError("the default face policy must refuse")


def test_a_stable_face_is_identified_identically_under_every_policy() -> None:
    """A face that survives the refinement does not depend on the policy."""

    tolerance = 1e-8
    matrix = np.diag(np.asarray([0.0, 1.0, 2.0, 3.0])).astype(complex)
    recorded, rank, source = _kernel_basis(
        matrix, tolerance, name="extended operator moment matrix"
    )
    assert rank == 1 and source == "identified"
    for policy in ("continuation", "inflated"):
        basis, other_rank, other_source = _kernel_basis(
            matrix,
            tolerance,
            name="extended operator moment matrix",
            policy=policy,
            inflation=1e-7 if policy == "inflated" else 0.0,
        )
        if policy == "inflated":
            assert other_rank == 0
        else:
            assert other_rank == rank
            np.testing.assert_array_equal(basis, recorded)
        assert other_source == "identified"


def test_continuation_reuses_a_consistent_face_and_refuses_an_inconsistent_one() -> None:
    tolerance = 1e-8
    matrix = _ambiguous_face_matrix(tolerance)
    carried = (1, np.asarray([[1.0], [0.0], [0.0], [0.0]], dtype=complex))
    basis, rank, source = _kernel_basis(
        matrix,
        tolerance,
        name="extended operator moment matrix",
        policy="continuation",
        carried=carried,
    )
    assert rank == 1 and source == "continued"
    np.testing.assert_allclose(np.abs(basis[0, 0]), 1.0)

    stale = (1, np.asarray([[0.0], [0.0], [0.0], [1.0]], dtype=complex))
    try:
        _kernel_basis(
            matrix,
            tolerance,
            name="extended operator moment matrix",
            policy="continuation",
            carried=stale,
        )
    except APCMProjectionError as error:
        assert "numerical face is unstable" in str(error)
    else:  # pragma: no cover - an inconsistent face must not be carried
        raise AssertionError("an inconsistent carried face must be refused")


def test_the_inflated_cone_exposes_no_face_inside_its_declared_margin() -> None:
    tolerance = 1e-8
    matrix = _ambiguous_face_matrix(tolerance)
    basis, rank, source = _kernel_basis(
        matrix,
        tolerance,
        name="extended operator moment matrix",
        policy="inflated",
        inflation=100.0 * 1e-9,
    )
    assert rank == 0 and source == "identified"
    assert basis.shape == (4, 0)


def test_face_policy_counters_start_at_zero_and_count_identifications() -> None:
    extension = SymmetryReducedPositiveExtension()
    geometry = SymmetryReducedAPCMGeometry(
        extension, APCMProjectionSettings(face_policy="continuation")
    )
    assert geometry.face_policy_counters == {
        "identified": 0,
        "continued": 0,
        "refused": 0,
    }
    geometry._identify_face(
        np.diag(np.asarray([1.0, 2.0])).astype(complex),
        1e-8,
        "extended operator moment matrix",
    )
    assert geometry.face_policy_counters["identified"] == 1
    assert "extended operator moment matrix" in geometry._carried_faces


def test_the_default_face_policy_carries_nothing() -> None:
    extension = SymmetryReducedPositiveExtension()
    geometry = SymmetryReducedAPCMGeometry(extension)
    assert geometry.settings.face_policy == "refuse"
    geometry._identify_face(
        np.diag(np.asarray([1.0, 2.0])).astype(complex),
        1e-8,
        "extended operator moment matrix",
    )
    assert geometry._carried_faces == {}


def test_projection_settings_reject_an_unknown_face_policy() -> None:
    try:
        APCMProjectionSettings(face_policy="carry")
    except ValueError as error:
        assert "face_policy" in str(error)
    else:  # pragma: no cover - the policy set is declared
        raise AssertionError("an unknown face policy must be refused")
