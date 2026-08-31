"""Exact-reference error definitions used by staged qualification."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.time_dynamics.diagnostics.physical_errors import (
    combine_resumed_rows,
    physical_errors,
)


def _row(time: float, *, state: np.ndarray | None = None):
    row = {
        "time": time,
        "energy_expectation": 1.0 + time,
        "doublon": 0.2 + time,
        "site_occupations": [0.4 + time, 0.6 - time],
    }
    if state is not None:
        row["psi_real"] = [float(value) for value in state.real]
        row["psi_imag"] = [float(value) for value in state.imag]
    return row


def _exact(time: float, *, state: np.ndarray):
    return {
        "time": time,
        "energy": 0.9 + time,
        "doublon": 0.15 + time,
        "site_occupations": [0.35 + time, 0.65 - time],
        "statevector": state,
    }


def test_componentwise_errors_and_infidelity_are_computed_on_one_grid() -> None:
    exact_state = np.asarray([1.0, 0.0], dtype=complex)
    tilted = np.asarray([np.sqrt(0.75), 0.5], dtype=complex)
    metrics = physical_errors(
        (_row(0.0, state=exact_state), _row(1.0, state=tilted)),
        (_exact(0.0, state=exact_state), _exact(1.0, state=exact_state)),
    )
    assert metrics.mean_abs_energy_error == pytest.approx(0.1)
    assert metrics.mean_abs_doublon_error == pytest.approx(0.05)
    assert metrics.mean_max_abs_site_occupation_error == pytest.approx(0.05)
    assert metrics.mean_infidelity == pytest.approx(0.125)


def test_infidelity_is_optional_only_when_no_states_were_recorded() -> None:
    exact_state = np.asarray([1.0, 0.0], dtype=complex)
    metrics = physical_errors(
        (_row(0.0), _row(1.0)),
        (_exact(0.0, state=exact_state), _exact(1.0, state=exact_state)),
    )
    assert metrics.mean_infidelity is None
    partial = (_row(0.0, state=exact_state), _row(1.0))
    with pytest.raises(ValueError, match="only some"):
        physical_errors(
            partial,
            (_exact(0.0, state=exact_state), _exact(1.0, state=exact_state)),
        )


def test_resume_rows_keep_the_shared_boundary_once() -> None:
    head = ({"time": 0.0}, {"time": 5.0})
    tail = ({"time": 5.0}, {"time": 10.0}, {"time": 20.0})
    combined = combine_resumed_rows(head, tail)
    assert [row["time"] for row in combined] == [0.0, 5.0, 10.0, 20.0]


def test_grid_mismatch_fails_loudly() -> None:
    state = np.asarray([1.0, 0.0], dtype=complex)
    with pytest.raises(ValueError, match="time mismatch"):
        physical_errors((_row(0.0, state=state),), (_exact(0.1, state=state),))
