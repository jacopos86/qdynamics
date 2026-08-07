"""Public interface for the 82-coordinate fourth-order moment hierarchy."""

from __future__ import annotations

from typing import Mapping

from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import MatrixDimerState
from .moment_hierarchy import (
    FOURTH_ORDER_HIERARCHY,
    MomentKey,
    TerminalMomentClosure,
)

FOURTH_CUMULANT_HIERARCHY = FOURTH_ORDER_HIERARCHY
FOURTH_CUMULANT_MOMENT_KEYS = FOURTH_CUMULANT_HIERARCHY.moment_keys
FOURTH_CUMULANT_STATE_NAMES = FOURTH_CUMULANT_HIERARCHY.state_names


def pack_fourth_cumulant_state(
    center_amplitude: complex,
    moments: Mapping[MomentKey, float],
) -> FloatArray:
    return FOURTH_CUMULANT_HIERARCHY.pack(center_amplitude, moments)


def unpack_fourth_cumulant_state(
    state: FloatArray,
) -> tuple[complex, dict[MomentKey, float]]:
    return FOURTH_CUMULANT_HIERARCHY.unpack(state)


def fourth_cumulant_moment_value(
    state: FloatArray,
    key: MomentKey,
) -> float:
    return FOURTH_CUMULANT_HIERARCHY.moment_value(state, key)


def fourth_cumulant_rhs(
    time: float,
    state: FloatArray,
    parameters: DimerParameters,
    *,
    closure: TerminalMomentClosure | None = None,
) -> FloatArray:
    """Evaluate the autonomous hierarchy with a declared terminal closure."""

    return FOURTH_CUMULANT_HIERARCHY.rhs(
        time,
        state,
        parameters,
        closure=closure,
    )


def fourth_cumulant_to_matrix_state(state: FloatArray) -> MatrixDimerState:
    return FOURTH_CUMULANT_HIERARCHY.to_matrix_state(state)


def fourth_cumulant_matrix_derivative(
    state: FloatArray,
    derivative: FloatArray,
) -> MatrixDimerState:
    return FOURTH_CUMULANT_HIERARCHY.matrix_derivative(state, derivative)


def fourth_cumulant_energy(
    time: float,
    state: FloatArray,
    parameters: DimerParameters,
) -> float:
    return FOURTH_CUMULANT_HIERARCHY.energy(time, state, parameters)


__all__ = [
    "FOURTH_CUMULANT_HIERARCHY",
    "FOURTH_CUMULANT_MOMENT_KEYS",
    "FOURTH_CUMULANT_STATE_NAMES",
    "fourth_cumulant_energy",
    "fourth_cumulant_matrix_derivative",
    "fourth_cumulant_moment_value",
    "fourth_cumulant_rhs",
    "fourth_cumulant_to_matrix_state",
    "pack_fourth_cumulant_state",
    "unpack_fourth_cumulant_state",
]
