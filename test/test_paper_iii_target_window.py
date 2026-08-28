"""Focused tests for declared target-window readings."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.qse_spectra.target_window import (
    READING_ALL_ROOTS,
    READING_WEIGHT_CARRYING,
    TargetWindow,
    TargetWindowError,
    all_roots_window,
    assert_shared_window,
    resolve_from_trace_windowed,
    resolve_target_window,
    row_window_error,
)


def _states(dim: int = 8, count: int = 7):
    """Orthonormal basis vectors standing in for exact eigenstates."""

    return [np.eye(1, dim, i, dtype=complex).reshape(-1) for i in range(count)]


def _observable_connecting(dim: int, bright: list[int]) -> np.ndarray:
    """An operator connecting state 0 to exactly the listed target roots."""

    matrix = np.zeros((dim, dim), dtype=complex)
    for r in bright:
        matrix[r, 0] = 1.0
        matrix[0, r] = 1.0
    return matrix


def _row(per_root, *, n2q, max_error=None):
    return {
        "n2q": float(n2q),
        "per_root_abs_error": None if per_root is None else list(per_root),
        "max_root_abs_error": (
            max_error if max_error is not None
            else (None if per_root is None else max(per_root))
        ),
    }


# --- declaration ------------------------------------------------------------


def test_all_roots_selects_the_whole_target_set() -> None:
    window = all_roots_window(6)
    assert window.reading == READING_ALL_ROOTS
    assert window.root_indices == (0, 1, 2, 3, 4, 5)
    assert window.dark_root_indices == ()
    assert window.payload()["depends_on_convergence"] is False


def test_weight_carrying_selects_only_coupled_roots() -> None:
    states = _states()
    # roots 1 and 3 (0-based target indices 0 and 2) carry weight
    observables = {"probe": _observable_connecting(8, [1, 3])}
    window = resolve_target_window(
        reading=READING_WEIGHT_CARRYING,
        exact_states=states,
        observables=observables,
        target_root_count=6,
    )
    assert window.root_indices == (0, 2)
    assert window.dark_root_indices == (1, 3, 4, 5)
    assert window.observable_names == ("probe",)
    assert window.payload()["selection_basis"].startswith("declared_observable_set")


def test_two_observables_union_their_visibility() -> None:
    states = _states()
    window = resolve_target_window(
        reading=READING_WEIGHT_CARRYING,
        exact_states=states,
        observables={
            "a": _observable_connecting(8, [1]),
            "b": _observable_connecting(8, [4]),
        },
        target_root_count=6,
    )
    assert window.root_indices == (0, 3)
    assert window.observable_names == ("a", "b")


def test_weight_carrying_requires_a_declared_observable_set() -> None:
    with pytest.raises(TargetWindowError, match="declared observable set"):
        resolve_target_window(
            reading=READING_WEIGHT_CARRYING,
            exact_states=_states(),
            observables={},
            target_root_count=6,
        )


def test_observables_blind_to_the_whole_window_fail_closed() -> None:
    states = _states()
    blind = np.zeros((8, 8), dtype=complex)
    blind[7, 7] = 1.0
    with pytest.raises(TargetWindowError, match="couple to no target root"):
        resolve_target_window(
            reading=READING_WEIGHT_CARRYING,
            exact_states=states,
            observables={"blind": blind},
            target_root_count=6,
        )


def test_an_empty_window_is_never_a_valid_target() -> None:
    with pytest.raises(TargetWindowError, match="selects no roots"):
        TargetWindow(reading=READING_ALL_ROOTS, root_indices=(), target_root_count=6)


# --- the anti-cherry-picking guarantees -------------------------------------


def test_window_is_independent_of_any_method_or_trace() -> None:
    """Visibility must not be derivable from convergence behaviour."""

    states = _states()
    observables = {"probe": _observable_connecting(8, [1, 3])}
    window = resolve_target_window(
        reading=READING_WEIGHT_CARRYING, exact_states=states,
        observables=observables, target_root_count=6)
    # Same declaration, recomputed with no reference to any run, is identical.
    again = resolve_target_window(
        reading=READING_WEIGHT_CARRYING, exact_states=states,
        observables=observables, target_root_count=6)
    assert window.digest() == again.digest()
    payload = window.payload()
    assert payload["depends_on_method"] is False
    assert payload["depends_on_convergence"] is False


def test_methods_scored_against_different_windows_are_refused() -> None:
    ours = all_roots_window(6)
    theirs = TargetWindow(
        reading=READING_WEIGHT_CARRYING, root_indices=(0, 1),
        target_root_count=6, observable_names=("probe",))
    assert_shared_window({"a": ours, "b": ours})  # same window is fine
    with pytest.raises(TargetWindowError, match="different target windows"):
        assert_shared_window({"ours": ours, "theirs": theirs})


# --- reading a trace --------------------------------------------------------


def test_row_error_maxes_only_over_selected_roots() -> None:
    window = TargetWindow(
        reading=READING_WEIGHT_CARRYING, root_indices=(0, 2),
        target_root_count=4, observable_names=("probe",))
    row = _row([1e-9, 2e-1, 3e-9, 5e-1], n2q=10)
    assert row_window_error(row, window) == pytest.approx(3e-9)
    assert row_window_error(row, all_roots_window(4)) == pytest.approx(5e-1)


def test_row_without_per_root_data_is_unscorable_under_a_subset() -> None:
    window = TargetWindow(
        reading=READING_WEIGHT_CARRYING, root_indices=(0,),
        target_root_count=4, observable_names=("probe",))
    assert row_window_error(_row(None, n2q=10, max_error=1e-3), window) is None
    # ... but the all-roots reading still works from max_root_abs_error alone
    assert row_window_error(_row(None, n2q=10, max_error=1e-3),
                            all_roots_window(4)) == pytest.approx(1e-3)


def test_per_root_length_mismatch_fails_closed() -> None:
    window = TargetWindow(
        reading=READING_WEIGHT_CARRYING, root_indices=(0,),
        target_root_count=6, observable_names=("probe",))
    with pytest.raises(TargetWindowError, match="per-root errors"):
        row_window_error(_row([1e-9, 1e-9], n2q=10), window)


def test_windowed_resolution_finds_the_cheaper_honest_crossing() -> None:
    """The dark-root case: a step is spectroscopically converged earlier."""

    rows = [
        _row([1e-1, 1e-1, 1e-1, 1e-1], n2q=10),
        _row([1e-9, 2e-1, 1e-9, 3e-1], n2q=24),   # bright roots exact, dark ones not
        _row([1e-9, 1e-9, 1e-9, 1e-9], n2q=32),
    ]
    bright = TargetWindow(
        reading=READING_WEIGHT_CARRYING, root_indices=(0, 2),
        target_root_count=4, observable_names=("probe",))

    all_reading = resolve_from_trace_windowed(
        rows, 1e-6, window=all_roots_window(4), extendable=False)
    bright_reading = resolve_from_trace_windowed(
        rows, 1e-6, window=bright, extendable=False)

    assert all_reading["status"] == "REACHED"
    assert all_reading["crossing"]["n2q"] == 32.0
    assert bright_reading["status"] == "REACHED"
    assert bright_reading["crossing"]["n2q"] == 24.0
    assert bright_reading["window_digest"] == bright.digest()


def test_unreached_window_reports_its_terminal_error() -> None:
    rows = [_row([1e-1, 1e-1], n2q=10), _row([1e-3, 5e-1], n2q=20)]
    window = TargetWindow(
        reading=READING_WEIGHT_CARRYING, root_indices=(0,),
        target_root_count=2, observable_names=("probe",))
    out = resolve_from_trace_windowed(rows, 1e-6, window=window, extendable=True)
    assert out["status"] == "NOT_REACHED_WITHIN_POOL"
    assert out["terminal_window_error"] == pytest.approx(1e-3)
