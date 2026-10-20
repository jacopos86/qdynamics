"""Bitwise neutrality of the joint-Gram moment assembly in matrix_reference.

Every assertion here compares the current assembly against the assembly it
replaced, transcribed verbatim below as ``recorded_*``:

* the Pauli means and the nine Pauli-pair traces were ``np.trace`` of an
  explicit matrix product, evaluated in a Python double loop;
* the two correlation matrices were contracted against the Pauli basis one
  ``np.trace`` at a time;
* the boson block, the joint moment matrix and the joint moment derivative
  were assembled with ``np.block``.

The comparison is ``np.array_equal`` and not a tolerance, because the
replacement is only admissible if it leaves the floating-point path alone: the
energy-neutral matrix barrier and the carried-witness flow are both pinned to
byte-identical reproduction of their recorded runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from paper5.stability.matrix_reference import (
    MatrixDimerState,
    _boson_moment_derivative_from_blocks,
    _correlation_pauli_block,
    _electron_fluctuation_moment_derivative,
    _electron_fluctuation_moment_matrix,
    _ELECTRON_FLUCTUATION_BASIS,
    _monomial_diagonal_terms,
    _pauli_product_traces,
    _pauli_trace_means,
    _pauli_traces,
    _validate_matrix_state,
    boson_moment_matrix,
    electron_phonon_moment_derivative,
    electron_phonon_moment_matrix,
)

DRAWS = 2000


def recorded_pauli_traces(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.trace(matrix @ operator)
            for operator in _ELECTRON_FLUCTUATION_BASIS
        ],
        dtype=complex,
    )


def recorded_pauli_product_traces(matrix: np.ndarray) -> np.ndarray:
    traces = np.empty((3, 3), dtype=complex)
    for row, first in enumerate(_ELECTRON_FLUCTUATION_BASIS):
        for column, second in enumerate(_ELECTRON_FLUCTUATION_BASIS):
            traces[row, column] = np.trace(matrix @ first @ second)
    return traces


def recorded_electron_fluctuation_moment_matrix(
    electron_density: np.ndarray,
) -> np.ndarray:
    electron = np.asarray(electron_density, dtype=complex)
    means = np.asarray(
        [
            np.trace(electron @ operator).real
            for operator in _ELECTRON_FLUCTUATION_BASIS
        ],
        dtype=float,
    )
    moment = np.empty((3, 3), dtype=complex)
    for row, first in enumerate(_ELECTRON_FLUCTUATION_BASIS):
        for column, second in enumerate(_ELECTRON_FLUCTUATION_BASIS):
            moment[row, column] = (
                np.trace(electron @ first @ second) - means[row] * means[column]
            )
    return 0.5 * (moment + moment.conjugate().T)


def recorded_electron_fluctuation_moment_derivative(
    electron_density: np.ndarray,
    electron_derivative: np.ndarray,
) -> np.ndarray:
    electron = np.asarray(electron_density, dtype=complex)
    derivative = np.asarray(electron_derivative, dtype=complex)
    means = np.asarray(
        [
            np.trace(electron @ operator).real
            for operator in _ELECTRON_FLUCTUATION_BASIS
        ],
        dtype=float,
    )
    mean_derivatives = np.asarray(
        [
            np.trace(derivative @ operator).real
            for operator in _ELECTRON_FLUCTUATION_BASIS
        ],
        dtype=float,
    )
    moment_derivative = np.empty((3, 3), dtype=complex)
    for row, first in enumerate(_ELECTRON_FLUCTUATION_BASIS):
        for column, second in enumerate(_ELECTRON_FLUCTUATION_BASIS):
            moment_derivative[row, column] = (
                np.trace(derivative @ first @ second)
                - mean_derivatives[row] * means[column]
                - means[row] * mean_derivatives[column]
            )
    return 0.5 * (moment_derivative + moment_derivative.conjugate().T)


def recorded_correlation_pauli_block(correlation: np.ndarray) -> np.ndarray:
    values = np.empty((2, 3), dtype=complex)
    for phonon_index in range(2):
        for operator_index, operator in enumerate(_ELECTRON_FLUCTUATION_BASIS):
            values[phonon_index, operator_index] = np.trace(
                correlation[phonon_index] @ operator
            )
    return values


def recorded_boson_moment_matrix(state: MatrixDimerState) -> np.ndarray:
    phonon = np.asarray(state.phonon_density, dtype=complex)
    anomalous = np.asarray(state.anomalous_phonon_density, dtype=complex)
    return np.block(
        [
            [phonon.T, anomalous.conjugate()],
            [anomalous, np.eye(2, dtype=complex) + phonon],
        ]
    )


def recorded_boson_moment_derivative_from_blocks(
    normal_derivative: np.ndarray,
    anomalous_derivative: np.ndarray,
) -> np.ndarray:
    return np.block(
        [
            [normal_derivative.T, anomalous_derivative.conjugate()],
            [anomalous_derivative, normal_derivative],
        ]
    )


def recorded_electron_phonon_moment_matrix(
    state: MatrixDimerState,
) -> np.ndarray:
    boson = recorded_boson_moment_matrix(state)
    electron = recorded_electron_fluctuation_moment_matrix(
        state.electron_density
    )
    correlation = recorded_correlation_pauli_block(
        state.electron_phonon_correlation
    )
    cross = np.vstack([correlation.conjugate(), correlation])
    moment = np.block([[boson, cross], [cross.conjugate().T, electron]])
    return 0.5 * (moment + moment.conjugate().T)


def recorded_electron_phonon_moment_derivative(
    state: MatrixDimerState,
    derivative: MatrixDimerState,
) -> np.ndarray:
    boson_derivative = recorded_boson_moment_derivative_from_blocks(
        derivative.phonon_density,
        derivative.anomalous_phonon_density,
    )
    electron_derivative = recorded_electron_fluctuation_moment_derivative(
        state.electron_density,
        derivative.electron_density,
    )
    correlation_derivative = recorded_correlation_pauli_block(
        derivative.electron_phonon_correlation
    )
    cross_derivative = np.vstack(
        [correlation_derivative.conjugate(), correlation_derivative]
    )
    moment_derivative = np.block(
        [
            [boson_derivative, cross_derivative],
            [cross_derivative.conjugate().T, electron_derivative],
        ]
    )
    return 0.5 * (moment_derivative + moment_derivative.conjugate().T)


def random_matrix_state(rng: np.random.Generator) -> MatrixDimerState:
    def complex_block(shape: tuple[int, ...]) -> np.ndarray:
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    electron = complex_block((2, 2))
    electron = 0.5 * (electron + electron.conjugate().T)
    phonon = complex_block((2, 2))
    phonon = 0.5 * (phonon + phonon.conjugate().T)
    return MatrixDimerState(
        electron_density=electron,
        coherent_phonon=complex_block((2,)),
        phonon_density=phonon,
        anomalous_phonon_density=complex_block((2, 2)),
        electron_phonon_correlation=complex_block((2, 2, 2)),
    )


def test_monomial_diagonal_terms_reproduce_the_pauli_basis() -> None:
    for operator in _ELECTRON_FLUCTUATION_BASIS:
        terms = _monomial_diagonal_terms(operator)
        assert len(terms) == 2
        for index, (row, coefficient) in enumerate(terms):
            assert operator[row, index] == coefficient
            assert abs(coefficient) == 1.0


def test_monomial_diagonal_terms_reject_a_non_monomial_operator() -> None:
    dense = np.ones((2, 2), dtype=complex)
    with pytest.raises(ValueError, match="exactly one nonzero"):
        _monomial_diagonal_terms(dense)


def test_pauli_traces_are_bitwise_the_recorded_traces() -> None:
    rng = np.random.default_rng(20260915)
    for _ in range(DRAWS):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        assert np.array_equal(
            _pauli_traces(matrix), recorded_pauli_traces(matrix)
        )


def test_pauli_product_traces_are_bitwise_the_recorded_traces() -> None:
    rng = np.random.default_rng(20260916)
    for _ in range(DRAWS):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        assert np.array_equal(
            _pauli_product_traces(matrix),
            recorded_pauli_product_traces(matrix),
        )


def test_electron_fluctuation_moments_are_bitwise_the_recorded_assembly() -> (
    None
):
    rng = np.random.default_rng(20260917)
    for _ in range(DRAWS):
        electron = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        electron = 0.5 * (electron + electron.conjugate().T)
        derivative = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        assert np.array_equal(
            _electron_fluctuation_moment_matrix(electron),
            recorded_electron_fluctuation_moment_matrix(electron),
        )
        assert np.array_equal(
            _electron_fluctuation_moment_derivative(electron, derivative),
            recorded_electron_fluctuation_moment_derivative(
                electron, derivative
            ),
        )


def test_correlation_pauli_block_is_bitwise_the_recorded_assembly() -> None:
    rng = np.random.default_rng(20260918)
    for _ in range(DRAWS):
        correlation = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(
            size=(2, 2, 2)
        )
        assert np.array_equal(
            _correlation_pauli_block(correlation),
            recorded_correlation_pauli_block(correlation),
        )


def test_block_assemblies_are_bitwise_the_recorded_np_block_assembly() -> None:
    rng = np.random.default_rng(20260919)
    for _ in range(DRAWS // 4):
        state = random_matrix_state(rng)
        derivative = random_matrix_state(rng)
        assert np.array_equal(
            boson_moment_matrix(state), recorded_boson_moment_matrix(state)
        )
        assert np.array_equal(
            _boson_moment_derivative_from_blocks(
                derivative.phonon_density,
                derivative.anomalous_phonon_density,
            ),
            recorded_boson_moment_derivative_from_blocks(
                derivative.phonon_density,
                derivative.anomalous_phonon_density,
            ),
        )
        assert np.array_equal(
            electron_phonon_moment_matrix(state),
            recorded_electron_phonon_moment_matrix(state),
        )
        assert np.array_equal(
            electron_phonon_moment_derivative(state, derivative),
            recorded_electron_phonon_moment_derivative(state, derivative),
        )


def test_pauli_mean_cache_returns_the_uncached_values_and_is_read_only() -> (
    None
):
    rng = np.random.default_rng(20260920)
    electron = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    electron = 0.5 * (electron + electron.conjugate().T)
    first = _pauli_trace_means(electron)
    second = _pauli_trace_means(electron.copy())
    assert np.array_equal(first, np.asarray(recorded_pauli_traces(electron).real))
    assert first is second
    assert not first.flags.writeable
    # A distinct operand does not read the cached entry.
    other = electron + 1.0
    assert np.array_equal(
        _pauli_trace_means(other), np.asarray(recorded_pauli_traces(other).real)
    )


def test_pauli_mean_cache_stays_bounded() -> None:
    rng = np.random.default_rng(20260921)
    seen = set()
    for _ in range(64):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        seen.add(_pauli_trace_means(matrix).tobytes())
    from paper5.stability import matrix_reference as module

    assert len(module._PAULI_MEAN_CACHE) <= module._PAULI_MEAN_CACHE_LIMIT
    assert len(seen) == 64


def test_matrix_state_validation_message_is_unchanged() -> None:
    rng = np.random.default_rng(20260922)
    state = random_matrix_state(rng)
    _validate_matrix_state(state)
    wrong = MatrixDimerState(
        electron_density=np.zeros((3, 3), dtype=complex),
        coherent_phonon=state.coherent_phonon,
        phonon_density=state.phonon_density,
        anomalous_phonon_density=state.anomalous_phonon_density,
        electron_phonon_correlation=state.electron_phonon_correlation,
    )
    with pytest.raises(
        ValueError,
        match=r"electron_density must have shape \(2, 2\), got \(3, 3\)",
    ):
        _validate_matrix_state(wrong)
    # A field that is not yet an array still validates by its converted shape.
    from_lists = MatrixDimerState(
        electron_density=[[0.0, 0.0], [0.0, 0.0]],
        coherent_phonon=state.coherent_phonon,
        phonon_density=state.phonon_density,
        anomalous_phonon_density=state.anomalous_phonon_density,
        electron_phonon_correlation=state.electron_phonon_correlation,
    )
    _validate_matrix_state(from_lists)
