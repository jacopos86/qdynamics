#!/usr/bin/env python3
"""Declared readings of the target window for post-run cost resolution.

`C*(eps_E)` is read from a completed growth trace as the cheapest step whose
window error clears `eps_E`. Which roots that error maxes over is a
**declaration about what the paper measures**, not a free parameter:

``all_roots``
    Every root in the target set. Conservative: it charges a method for
    resolving states that no declared observable couples to, so they can
    never appear in a spectrum. Historical default; keep for continuity.

``weight_carrying``
    Only roots the declared observable set actually connects to the reference
    state. This is the reading a spectroscopy claim needs, because a dark
    root contributes nothing to any spectral function.

Two guarantees make ``weight_carrying`` a declaration rather than
cherry-picking, and both are enforced here:

1. **The subset is a property of the problem, never of a method.**
   :func:`resolve_target_window` computes visibility once from the exact
   reference states and the declared observables, and
   :func:`assert_shared_window` refuses to score methods against different
   subsets. A method cannot earn a cheaper reading by converging differently.
2. **Visibility is fixed by the observable set, never by convergence.**
   Nothing here reads a trace, a method id, or an error value when deciding
   which roots count. Selecting roots because they converged first would be
   cherry-picking; selecting them because no instrument can see the others is
   the definition of the measured window.

Post-run reporting only. Exact reference data is admissible here for the same
reason it is admissible in `C*` itself, and for no other: it never reaches an
online decision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

TARGET_WINDOW_SCHEMA_VERSION = "paper_iii_target_window_v1"

READING_ALL_ROOTS = "all_roots"
READING_WEIGHT_CARRYING = "weight_carrying"
READINGS = (READING_ALL_ROOTS, READING_WEIGHT_CARRYING)

# A root counts as visible when its summed spectral weight over the declared
# observables exceeds this fraction of the window's total weight. Well below
# any physical weight and far above numerical zero (~1e-30 in practice).
DEFAULT_VISIBILITY_THRESHOLD = 1.0e-8


class TargetWindowError(ValueError):
    """Raised on an invalid window declaration or an unshared subset."""


@dataclass(frozen=True)
class TargetWindow:
    """One declared reading of the target window."""

    reading: str
    root_indices: tuple[int, ...]
    target_root_count: int
    observable_names: tuple[str, ...] = ()
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD
    root_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if str(self.reading) not in READINGS:
            raise TargetWindowError(
                f"unknown window reading {self.reading!r}; declared: {list(READINGS)}."
            )
        if not self.root_indices:
            raise TargetWindowError(
                f"reading {self.reading!r} selects no roots; a window with no "
                "measurable state is not a valid accuracy target."
            )
        if any(not 0 <= int(i) < int(self.target_root_count) for i in self.root_indices):
            raise TargetWindowError("window root index outside the target set.")
        if self.reading == READING_ALL_ROOTS and len(self.root_indices) != int(
            self.target_root_count
        ):
            raise TargetWindowError("all_roots must select the whole target set.")
        if self.reading == READING_WEIGHT_CARRYING and not self.observable_names:
            raise TargetWindowError(
                "weight_carrying requires the declared observable names that "
                "define visibility."
            )

    @property
    def dark_root_indices(self) -> tuple[int, ...]:
        selected = set(int(i) for i in self.root_indices)
        return tuple(i for i in range(int(self.target_root_count)) if i not in selected)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": TARGET_WINDOW_SCHEMA_VERSION,
            "reading": str(self.reading),
            "root_indices": [int(i) for i in self.root_indices],
            "dark_root_indices": [int(i) for i in self.dark_root_indices],
            "target_root_count": int(self.target_root_count),
            "observable_names": list(self.observable_names),
            "visibility_threshold": float(self.visibility_threshold),
            "root_weights": [float(w) for w in self.root_weights],
            "selection_basis": (
                "declared_observable_set_and_exact_reference_states"
                if self.reading == READING_WEIGHT_CARRYING
                else "entire_declared_target_set"
            ),
            "depends_on_method": False,
            "depends_on_convergence": False,
        }

    def digest(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def all_roots_window(target_root_count: int) -> TargetWindow:
    return TargetWindow(
        reading=READING_ALL_ROOTS,
        root_indices=tuple(range(int(target_root_count))),
        target_root_count=int(target_root_count),
    )


def resolve_target_window(
    *,
    reading: str,
    exact_states: Sequence[np.ndarray],
    observables: Mapping[str, np.ndarray],
    target_root_count: int,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> TargetWindow:
    """Declare a window reading from the problem alone.

    ``exact_states[0]`` is the reference state and ``exact_states[1:]`` the
    target roots. ``observables`` maps a declared name to its dense matrix.
    No method, trace, or error value is consulted.
    """

    if str(reading) == READING_ALL_ROOTS:
        return all_roots_window(int(target_root_count))
    if str(reading) != READING_WEIGHT_CARRYING:
        raise TargetWindowError(f"unknown window reading {reading!r}.")
    if not observables:
        raise TargetWindowError(
            "weight_carrying needs a declared observable set; without one, "
            "visibility is undefined and the reading would be arbitrary."
        )
    states = list(exact_states)
    if len(states) < int(target_root_count) + 1:
        raise TargetWindowError(
            f"need {int(target_root_count) + 1} exact states (reference plus "
            f"targets); got {len(states)}."
        )
    psi0 = np.asarray(states[0], dtype=complex).reshape(-1)
    weights: list[float] = []
    for r in range(1, int(target_root_count) + 1):
        psi_r = np.asarray(states[r], dtype=complex).reshape(-1)
        total = 0.0
        for matrix in observables.values():
            amplitude = complex(np.vdot(psi_r, np.asarray(matrix, dtype=complex) @ psi0))
            total += float(abs(amplitude) ** 2)
        weights.append(total)
    grand_total = float(sum(weights))
    if grand_total <= 0.0:
        raise TargetWindowError(
            "the declared observables couple to no target root; the window "
            "would be empty. Declare observables that probe this window."
        )
    cutoff = float(visibility_threshold) * grand_total
    indices = tuple(r for r, w in enumerate(weights) if w > cutoff)
    if not indices:
        raise TargetWindowError("no target root exceeds the visibility threshold.")
    return TargetWindow(
        reading=READING_WEIGHT_CARRYING,
        root_indices=indices,
        target_root_count=int(target_root_count),
        observable_names=tuple(sorted(observables)),
        visibility_threshold=float(visibility_threshold),
        root_weights=tuple(float(w) for w in weights),
    )


def assert_shared_window(windows: Mapping[str, TargetWindow]) -> None:
    """Refuse to report methods scored against different windows."""

    digests = {name: window.digest() for name, window in windows.items()}
    distinct = set(digests.values())
    if len(distinct) > 1:
        raise TargetWindowError(
            "methods were scored against different target windows, which would "
            f"make the comparison meaningless: {digests}"
        )


def row_window_error(row: Mapping[str, Any], window: TargetWindow) -> float | None:
    """The window error of one trace row under a declared reading."""

    if str(window.reading) == READING_ALL_ROOTS:
        value = row.get("max_root_abs_error")
        return None if value is None else float(value)
    per_root = row.get("per_root_abs_error")
    if per_root is None:
        return None
    if len(per_root) != int(window.target_root_count):
        raise TargetWindowError(
            f"row reports {len(per_root)} per-root errors; window declares "
            f"{int(window.target_root_count)} target roots."
        )
    return max(float(per_root[int(i)]) for i in window.root_indices)


def resolve_from_trace_windowed(
    rows: Sequence[Mapping[str, Any]],
    eps_e: float,
    *,
    window: TargetWindow,
    extendable: bool,
) -> dict[str, Any]:
    """`C*(eps_E)` under a declared window reading, cheapest crossing by N2q."""

    from pipelines.exact_bench.paper_iii_matched_accuracy_campaign import (
        STATUS_NOT_IN_POOL,
        STATUS_REACHED,
        STATUS_UNATTAINABLE,
    )

    scored: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        error = row_window_error(row, window)
        if error is not None:
            scored.append((error, row))
    reaching = [(e, r) for e, r in scored if e <= float(eps_e)]
    if reaching:
        error, best = min(reaching, key=lambda pair: float(pair[1]["n2q"]))
        return {
            "status": STATUS_REACHED,
            "window_reading": str(window.reading),
            "window_digest": window.digest(),
            "window_error_at_crossing": float(error),
            "crossing": dict(best),
        }
    terminal = min(scored, key=lambda pair: pair[0]) if scored else None
    return {
        "status": STATUS_NOT_IN_POOL if extendable else STATUS_UNATTAINABLE,
        "window_reading": str(window.reading),
        "window_digest": window.digest(),
        "terminal": None if terminal is None else dict(terminal[1]),
        "terminal_window_error": None if terminal is None else float(terminal[0]),
    }


__all__ = [
    "DEFAULT_VISIBILITY_THRESHOLD",
    "READINGS",
    "READING_ALL_ROOTS",
    "READING_WEIGHT_CARRYING",
    "TARGET_WINDOW_SCHEMA_VERSION",
    "TargetWindow",
    "TargetWindowError",
    "all_roots_window",
    "assert_shared_window",
    "resolve_from_trace_windowed",
    "resolve_target_window",
    "row_window_error",
]
