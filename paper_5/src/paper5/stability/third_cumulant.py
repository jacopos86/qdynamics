"""Compatibility interface for the 47-coordinate third-order hierarchy."""

from __future__ import annotations

from typing import Mapping

from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import MatrixDimerState
from .moment_hierarchy import (
    IDENTITY,
    PAULI_LABELS,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    THIRD_ORDER_HIERARCHY,
    MomentKey,
    _commutator,
    _operator_product,
    _pauli_product,
    _set_partitions,
    _weyl_product,
)

THIRD_CUMULANT_HIERARCHY = THIRD_ORDER_HIERARCHY
THIRD_CUMULANT_MOMENT_KEYS = THIRD_CUMULANT_HIERARCHY.moment_keys
THIRD_CUMULANT_STATE_NAMES = THIRD_CUMULANT_HIERARCHY.state_names


def pack_third_cumulant_state(
    center_amplitude: complex,
    moments: Mapping[MomentKey, float],
) -> FloatArray:
    return THIRD_CUMULANT_HIERARCHY.pack(center_amplitude, moments)


def unpack_third_cumulant_state(
    state: FloatArray,
) -> tuple[complex, dict[MomentKey, float]]:
    return THIRD_CUMULANT_HIERARCHY.unpack(state)


def moment_value(state: FloatArray, key: MomentKey) -> float:
    return THIRD_CUMULANT_HIERARCHY.moment_value(state, key)


def _closed_moment(
    key: MomentKey,
    moments: Mapping[MomentKey, float],
) -> float:
    """Backward-compatible access to the zero-fourth-cumulant rule."""

    return THIRD_CUMULANT_HIERARCHY.closed_moment(key, moments)


def third_cumulant_rhs(
    time: float,
    state: FloatArray,
    parameters: DimerParameters,
) -> FloatArray:
    return THIRD_CUMULANT_HIERARCHY.rhs(time, state, parameters)


def third_cumulant_to_matrix_state(state: FloatArray) -> MatrixDimerState:
    return THIRD_CUMULANT_HIERARCHY.to_matrix_state(state)


def third_cumulant_matrix_derivative(
    state: FloatArray,
    derivative: FloatArray,
) -> MatrixDimerState:
    return THIRD_CUMULANT_HIERARCHY.matrix_derivative(state, derivative)


def third_cumulant_energy(
    time: float,
    state: FloatArray,
    parameters: DimerParameters,
) -> float:
    return THIRD_CUMULANT_HIERARCHY.energy(time, state, parameters)


__all__ = [
    "IDENTITY",
    "MomentKey",
    "PAULI_LABELS",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "THIRD_CUMULANT_HIERARCHY",
    "THIRD_CUMULANT_MOMENT_KEYS",
    "THIRD_CUMULANT_STATE_NAMES",
    "moment_value",
    "pack_third_cumulant_state",
    "third_cumulant_energy",
    "third_cumulant_matrix_derivative",
    "third_cumulant_rhs",
    "third_cumulant_to_matrix_state",
    "unpack_third_cumulant_state",
]
