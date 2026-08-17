"""Low-overhead phase timing and bounded profiling receipts for AP-McLachlan.

This module owns measurement only.  It must never influence a scientific
decision: no value produced here feeds candidate scoring, support-patch
selection, integrator acceptance, or solve repair.  Every recording path is a
pure side effect of code that already ran.

Design constraints that follow from the AP-McLachlan route:

* **Zero cost when inactive.**  ``phase()`` returns a shared no-op context
  manager when no profiler is installed, so instrumented call sites cost one
  module-global lookup and a trivial ``with`` block in ordinary runs.
* **Thread safe without locks on the hot path.**  Support-patch candidate
  scoring runs inside an ordered thread pool.  Each thread accumulates into its
  own thread-local table; tables are merged only when a receipt is produced.
  Merging is done in sorted phase-name order so the receipt is deterministic
  regardless of thread completion order.
* **Bounded memory.**  A ``t=50`` trajectory produces millions of phase
  entries.  Nothing retains per-call samples: each phase keeps exact
  ``count``/``total``/``min``/``max``, a fixed log-spaced histogram for quantile
  estimation, and the ``worst_call_count`` slowest calls.  Receipt size is
  therefore independent of trajectory length.

Timings are recorded as both *inclusive* (wall time between enter and exit) and
*exclusive* (inclusive minus the inclusive time of directly nested phases), so a
receipt containing nested phases remains interpretable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import math
import threading
import time
from typing import Any, Iterator, Mapping, Sequence


PROFILE_RECEIPT_SCHEMA_V1 = "ap_mclachlan_profile_receipt_v1"

# Histogram covers 1 ns .. 1000 s.  Anything outside lands in the clamp bins,
# which keeps quantile estimates honest without unbounded storage.
_HISTOGRAM_MIN_LOG10 = -9.0
_HISTOGRAM_MAX_LOG10 = 3.0
_HISTOGRAM_BINS_PER_DECADE = 20
_HISTOGRAM_BIN_COUNT = int(
    (_HISTOGRAM_MAX_LOG10 - _HISTOGRAM_MIN_LOG10) * _HISTOGRAM_BINS_PER_DECADE
)

DEFAULT_WORST_CALL_COUNT = 5


# ---------------------------------------------------------------------------
# Phase names used by the active route.
#
# Keep these stable: a profiling receipt is compared across optimization passes,
# and renaming a phase silently breaks before/after comparison.
# ---------------------------------------------------------------------------

PHASE_CHECKPOINT = "checkpoint"
PHASE_GEOMETRY_EVAL = "geometry.evaluate"
PHASE_FIXED_STEP_SOLVE = "solve.fixed_step"
PHASE_APPEND_SELECT = "patch.append.select"
PHASE_APPEND_GEOMETRY_CACHE = "patch.append.geometry_cache"
PHASE_APPEND_SCORE_BATCH = "patch.append.score_batch"
PHASE_APPEND_SCORE_ONE = "patch.append.score_one"
PHASE_PARENT_SCOUT = "patch.append.parent_scout"
PHASE_PRUNE_SELECT = "patch.prune.select"
PHASE_PRUNE_SCORE_BATCH = "patch.prune.score_batch"
PHASE_PRUNE_SCORE_ONE = "patch.prune.score_one"
PHASE_PRUNE_SAFETY = "patch.prune.deletion_safety"
PHASE_PRUNE_SMOOTHNESS = "patch.prune.smoothness"
PHASE_EXCHANGE_SCORE_ONE = "patch.exchange.score_one"
PHASE_EXCHANGE_FINALIST = "patch.exchange.finalist"
PHASE_UNIFIED_SELECT = "patch.unified.select"
PHASE_MATERIALIZE_PATCH = "patch.materialize"
PHASE_INTEGRATE = "integrator.interval"
PHASE_INTEGRATE_STAGE = "integrator.stage"
PHASE_OBSERVABLES = "observables.evaluate"
PHASE_SERIALIZE = "serialize.report"


# ---------------------------------------------------------------------------
# No-op path
# ---------------------------------------------------------------------------


class _NullPhase:
    """Shared do-nothing context manager for the inactive-profiler path."""

    __slots__ = ()

    def __enter__(self) -> "_NullPhase":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False


_NULL_PHASE = _NullPhase()

#: Public shared no-op span, for call sites that conditionally time a block.
NULL_PHASE = _NULL_PHASE


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------


def _histogram_index(seconds: float) -> int:
    if seconds <= 0.0:
        return 0
    log10_seconds = math.log10(seconds)
    scaled = (log10_seconds - _HISTOGRAM_MIN_LOG10) * _HISTOGRAM_BINS_PER_DECADE
    if scaled <= 0.0:
        return 0
    index = int(scaled)
    if index >= _HISTOGRAM_BIN_COUNT:
        return _HISTOGRAM_BIN_COUNT - 1
    return index


def _histogram_bin_upper_seconds(index: int) -> float:
    log10_seconds = (
        _HISTOGRAM_MIN_LOG10
        + float(int(index) + 1) / float(_HISTOGRAM_BINS_PER_DECADE)
    )
    return float(10.0**log10_seconds)


@dataclass
class _PhaseAccumulator:
    """Bounded per-phase statistics for one thread."""

    count: int = 0
    inclusive_total: float = 0.0
    exclusive_total: float = 0.0
    inclusive_min: float = math.inf
    inclusive_max: float = 0.0
    histogram: list[int] = field(
        default_factory=lambda: [0] * _HISTOGRAM_BIN_COUNT
    )
    worst: list[tuple[float, str | None]] = field(default_factory=list)

    def record(
        self,
        *,
        inclusive: float,
        exclusive: float,
        note: str | None,
        worst_call_count: int,
    ) -> None:
        self.count += 1
        self.inclusive_total += inclusive
        self.exclusive_total += exclusive
        if inclusive < self.inclusive_min:
            self.inclusive_min = inclusive
        if inclusive > self.inclusive_max:
            self.inclusive_max = inclusive
        self.histogram[_histogram_index(inclusive)] += 1
        if worst_call_count <= 0:
            return
        if len(self.worst) < worst_call_count:
            self.worst.append((float(inclusive), note))
            self.worst.sort(key=lambda item: -item[0])
        elif inclusive > self.worst[-1][0]:
            self.worst[-1] = (float(inclusive), note)
            self.worst.sort(key=lambda item: -item[0])

    def merge_from(self, other: "_PhaseAccumulator", *, worst_call_count: int) -> None:
        self.count += other.count
        self.inclusive_total += other.inclusive_total
        self.exclusive_total += other.exclusive_total
        self.inclusive_min = min(self.inclusive_min, other.inclusive_min)
        self.inclusive_max = max(self.inclusive_max, other.inclusive_max)
        for index, value in enumerate(other.histogram):
            if value:
                self.histogram[index] += value
        if worst_call_count > 0 and other.worst:
            combined = self.worst + list(other.worst)
            combined.sort(key=lambda item: -item[0])
            self.worst = combined[:worst_call_count]

    def quantile_seconds(self, quantile: float) -> float | None:
        """Histogram-estimated quantile, upper-bounded by the observed maximum."""

        if self.count <= 0:
            return None
        target = float(quantile) * float(self.count)
        seen = 0
        for index, value in enumerate(self.histogram):
            if not value:
                continue
            seen += value
            if float(seen) >= target:
                return float(min(_histogram_bin_upper_seconds(index), self.inclusive_max))
        return float(self.inclusive_max)


class _ThreadState(threading.local):
    def __init__(self) -> None:  # noqa: D107 - threading.local initializer
        self.phases: dict[str, _PhaseAccumulator] = {}
        self.counters: dict[str, int] = {}
        # Inclusive time accumulated by phases nested directly inside the phase
        # currently on top of the stack.
        self.nested_inclusive: list[float] = []


class PhaseProfiler:
    """Collects bounded phase timings across the threads of one trajectory."""

    def __init__(
        self,
        *,
        worst_call_count: int = DEFAULT_WORST_CALL_COUNT,
        label: str = "ap_mclachlan",
    ) -> None:
        if int(worst_call_count) < 0:
            raise ValueError("worst_call_count must be non-negative.")
        self.label = str(label)
        self.worst_call_count = int(worst_call_count)
        self._thread_state = _ThreadState()
        self._registry_lock = threading.Lock()
        # Every thread's state object, kept so a receipt can merge them.  Only
        # the registry append is locked; the hot path never takes this lock.
        self._registered: list[tuple[dict[str, _PhaseAccumulator], dict[str, int]]] = []
        self._registered_ids: set[int] = set()
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        self._wall_elapsed: float | None = None
        self._cpu_elapsed: float | None = None

    # -- registration ----------------------------------------------------

    def _local_tables(self) -> tuple[dict[str, _PhaseAccumulator], dict[str, int]]:
        state = self._thread_state
        phases = state.phases
        counters = state.counters
        key = id(phases)
        if key not in self._registered_ids:
            with self._registry_lock:
                if key not in self._registered_ids:
                    self._registered_ids.add(key)
                    self._registered.append((phases, counters))
        return phases, counters

    # -- recording -------------------------------------------------------

    @contextmanager
    def phase(self, name: str, *, note: str | None = None) -> Iterator[None]:
        phases, _counters = self._local_tables()
        state = self._thread_state
        nested = state.nested_inclusive
        nested.append(0.0)
        start = time.perf_counter()
        try:
            yield
        finally:
            inclusive = time.perf_counter() - start
            child_inclusive = nested.pop()
            if nested:
                nested[-1] += inclusive
            accumulator = phases.get(name)
            if accumulator is None:
                accumulator = _PhaseAccumulator()
                phases[name] = accumulator
            accumulator.record(
                inclusive=inclusive,
                exclusive=max(0.0, inclusive - child_inclusive),
                note=note,
                worst_call_count=self.worst_call_count,
            )

    def add_counter(self, name: str, value: int = 1) -> None:
        _phases, counters = self._local_tables()
        counters[str(name)] = int(counters.get(str(name), 0)) + int(value)

    def attribute_nested(self, seconds: float) -> None:
        """Charge ``seconds`` of child work to the phase open on this thread.

        Nested-phase bookkeeping is thread-local, so work dispatched to a worker
        pool is invisible to the phase that dispatched it and would otherwise be
        counted as that phase's own exclusive time. A caller that fans work out
        to other threads reports the total back here once the fan-in completes.
        """

        nested = self._thread_state.nested_inclusive
        if nested:
            nested[-1] += float(seconds)

    def finish(self) -> None:
        """Freeze the wall/CPU totals for this profiling session."""

        if self._wall_elapsed is None:
            self._wall_elapsed = time.perf_counter() - self._wall_start
            self._cpu_elapsed = time.process_time() - self._cpu_start

    # -- reporting -------------------------------------------------------

    def _merged(self) -> tuple[dict[str, _PhaseAccumulator], dict[str, int]]:
        with self._registry_lock:
            registered = list(self._registered)
        merged_phases: dict[str, _PhaseAccumulator] = {}
        merged_counters: dict[str, int] = {}
        for phases, counters in registered:
            # Iterate in sorted order so the merge is independent of the order
            # in which worker threads happened to register or finish.
            for name in sorted(phases):
                target = merged_phases.get(name)
                if target is None:
                    target = _PhaseAccumulator()
                    merged_phases[name] = target
                target.merge_from(phases[name], worst_call_count=self.worst_call_count)
            for name in sorted(counters):
                merged_counters[name] = int(
                    merged_counters.get(name, 0)
                ) + int(counters[name])
        return merged_phases, merged_counters

    def receipt(self) -> dict[str, Any]:
        """Return a compact, deterministic, bounded profiling receipt."""

        self.finish()
        merged_phases, merged_counters = self._merged()
        phase_rows = []
        for name in sorted(merged_phases):
            accumulator = merged_phases[name]
            if accumulator.count <= 0:
                continue
            phase_rows.append(
                {
                    "phase": name,
                    "count": int(accumulator.count),
                    "inclusive_seconds": float(accumulator.inclusive_total),
                    "exclusive_seconds": float(accumulator.exclusive_total),
                    "mean_inclusive_seconds": float(
                        accumulator.inclusive_total / accumulator.count
                    ),
                    "min_inclusive_seconds": (
                        None
                        if accumulator.inclusive_min is math.inf
                        else float(accumulator.inclusive_min)
                    ),
                    "max_inclusive_seconds": float(accumulator.inclusive_max),
                    "p50_inclusive_seconds": accumulator.quantile_seconds(0.50),
                    "p90_inclusive_seconds": accumulator.quantile_seconds(0.90),
                    "p99_inclusive_seconds": accumulator.quantile_seconds(0.99),
                    "worst_calls": [
                        {
                            "inclusive_seconds": float(seconds),
                            "note": None if note is None else str(note),
                        }
                        for seconds, note in accumulator.worst
                    ],
                }
            )
        phase_rows.sort(key=lambda row: (-float(row["exclusive_seconds"]), str(row["phase"])))
        return {
            "schema": PROFILE_RECEIPT_SCHEMA_V1,
            "label": str(self.label),
            "session_wall_seconds": float(self._wall_elapsed or 0.0),
            "session_cpu_seconds": float(self._cpu_elapsed or 0.0),
            "quantile_method": (
                "log_histogram_upper_bin_edge_clamped_to_observed_max; "
                f"{_HISTOGRAM_BINS_PER_DECADE} bins per decade over "
                f"1e{_HISTOGRAM_MIN_LOG10:.0f}..1e{_HISTOGRAM_MAX_LOG10:.0f} s"
            ),
            "timing_convention": "inclusive=enter-to-exit; exclusive=inclusive minus nested phases",
            "worst_call_count": int(self.worst_call_count),
            "phases": phase_rows,
            "counters": {name: int(merged_counters[name]) for name in sorted(merged_counters)},
        }


# ---------------------------------------------------------------------------
# Module-level activation
# ---------------------------------------------------------------------------

_ACTIVE_PROFILER: PhaseProfiler | None = None


def active_profiler() -> PhaseProfiler | None:
    """Return the installed profiler, or ``None`` in ordinary runs."""

    return _ACTIVE_PROFILER


def phase(name: str, *, note: str | None = None) -> Any:
    """Time ``name`` when profiling is active; otherwise do nothing.

    The inactive path returns a shared no-op context manager, so instrumented
    call sites stay cheap enough to leave in the production route.
    """

    profiler = _ACTIVE_PROFILER
    if profiler is None:
        return _NULL_PHASE
    return profiler.phase(name, note=note)


def count(name: str, value: int = 1) -> None:
    """Increment a profiling counter when profiling is active."""

    profiler = _ACTIVE_PROFILER
    if profiler is not None:
        profiler.add_counter(name, value)


def attribute_nested(seconds: float) -> None:
    """Charge cross-thread child work to the open phase, when profiling."""

    profiler = _ACTIVE_PROFILER
    if profiler is not None:
        profiler.attribute_nested(seconds)


def timed(name: str) -> Any:
    """Decorator form of :func:`phase` for whole-function spans.

    The wrapper adds one global lookup per call when profiling is inactive, so
    it belongs on checkpoint- or candidate-level functions, not on per-Pauli
    inner helpers.
    """

    def decorate(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = _ACTIVE_PROFILER
            if profiler is None:
                return func(*args, **kwargs)
            with profiler.phase(name):
                return func(*args, **kwargs)

        return wrapper

    return decorate


@contextmanager
def profiling_session(
    *,
    worst_call_count: int = DEFAULT_WORST_CALL_COUNT,
    label: str = "ap_mclachlan",
) -> Iterator[PhaseProfiler]:
    """Install a profiler for the duration of the block.

    Nesting is rejected rather than silently shadowed: two overlapping sessions
    would split one trajectory's timings across two receipts.
    """

    global _ACTIVE_PROFILER
    if _ACTIVE_PROFILER is not None:
        raise RuntimeError("a profiling session is already active.")
    profiler = PhaseProfiler(worst_call_count=worst_call_count, label=label)
    _ACTIVE_PROFILER = profiler
    try:
        yield profiler
    finally:
        profiler.finish()
        _ACTIVE_PROFILER = None


def receipt_summary_rows(
    receipt: Mapping[str, Any],
    *,
    limit: int = 12,
) -> Sequence[Mapping[str, Any]]:
    """Return the ``limit`` phases with the largest exclusive time."""

    phases = list(receipt.get("phases", ()))
    return tuple(phases[: max(0, int(limit))])


__all__ = [
    "DEFAULT_WORST_CALL_COUNT",
    "NULL_PHASE",
    "PROFILE_RECEIPT_SCHEMA_V1",
    "attribute_nested",
    "PHASE_APPEND_GEOMETRY_CACHE",
    "PHASE_APPEND_SCORE_BATCH",
    "PHASE_APPEND_SCORE_ONE",
    "PHASE_APPEND_SELECT",
    "PHASE_CHECKPOINT",
    "PHASE_EXCHANGE_FINALIST",
    "PHASE_EXCHANGE_SCORE_ONE",
    "PHASE_FIXED_STEP_SOLVE",
    "PHASE_GEOMETRY_EVAL",
    "PHASE_INTEGRATE",
    "PHASE_INTEGRATE_STAGE",
    "PHASE_MATERIALIZE_PATCH",
    "PHASE_OBSERVABLES",
    "PHASE_PARENT_SCOUT",
    "PHASE_PRUNE_SAFETY",
    "PHASE_PRUNE_SCORE_BATCH",
    "PHASE_PRUNE_SCORE_ONE",
    "PHASE_PRUNE_SELECT",
    "PHASE_PRUNE_SMOOTHNESS",
    "PHASE_SERIALIZE",
    "PHASE_UNIFIED_SELECT",
    "PhaseProfiler",
    "active_profiler",
    "count",
    "phase",
    "profiling_session",
    "receipt_summary_rows",
    "timed",
]
