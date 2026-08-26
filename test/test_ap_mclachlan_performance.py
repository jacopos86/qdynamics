from __future__ import annotations

import json
import threading
import time

import pytest

from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
    _ordered_parallel_map,
    _support_patch_scoring_worker_count,
)
from pipelines.time_dynamics.ap_mclachlan.performance import (
    attribute_nested,
    NULL_PHASE,
    PROFILE_RECEIPT_SCHEMA_V1,
    PhaseProfiler,
    active_profiler,
    count,
    phase,
    profiling_session,
    timed,
)


def _support_config(workers: int) -> SupportPatchControllerConfig:
    return SupportPatchControllerConfig(
        support_patch_scoring_workers=int(workers),
    )


# ---------------------------------------------------------------------------
# Ordered parallel map: worker count must not change the algorithm.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 2, 3, 8])
def test_ordered_parallel_map_preserves_canonical_order(workers: int) -> None:
    """Tasks that finish in reverse order must still reduce in task order."""

    task_count = 12
    tasks = tuple(range(task_count))

    def score_one(task: int) -> int:
        # Later tasks finish first, so completion order is the reverse of the
        # canonical task order.
        time.sleep(0.002 * (task_count - int(task)))
        return int(task) * 10

    result = _ordered_parallel_map(
        tasks,
        support_config=_support_config(workers),
        score_one=score_one,
    )
    assert result == tuple(task * 10 for task in tasks)


def test_ordered_parallel_map_matches_serial_reference_across_worker_counts() -> None:
    """Worker count 1 is the reference execution of the same code path."""

    tasks = tuple(range(25))

    def score_one(task: int) -> str:
        return f"score-{int(task):03d}"

    reference = _ordered_parallel_map(
        tasks, support_config=_support_config(1), score_one=score_one
    )
    for workers in (2, 3, 8, 64):
        assert (
            _ordered_parallel_map(
                tasks, support_config=_support_config(workers), score_one=score_one
            )
            == reference
        )


def test_ordered_parallel_map_preserves_candidate_identity_on_failure() -> None:
    """An exception must surface as a true runtime error, not a silent gap."""

    def score_one(task: int) -> int:
        if int(task) == 7:
            raise ValueError("candidate 7 failed")
        return int(task)

    with pytest.raises(ValueError, match="candidate 7 failed"):
        _ordered_parallel_map(
            tuple(range(10)),
            support_config=_support_config(4),
            score_one=score_one,
        )


def test_worker_count_is_bounded_by_task_count_and_positive() -> None:
    assert _support_patch_scoring_worker_count(_support_config(8), task_count=3) == 3
    assert _support_patch_scoring_worker_count(_support_config(8), task_count=1) == 1
    assert _support_patch_scoring_worker_count(_support_config(1), task_count=9) == 1
    with pytest.raises(ValueError, match="must be positive"):
        _support_patch_scoring_worker_count(
            SupportPatchControllerConfig(
                support_patch_scoring_workers=0,
            ),
            task_count=4,
        )


def test_ordered_parallel_map_timing_wrapper_does_not_change_results() -> None:
    tasks = tuple(range(9))

    def score_one(task: int) -> int:
        return int(task) ** 2

    plain = _ordered_parallel_map(
        tasks, support_config=_support_config(4), score_one=score_one
    )
    with profiling_session(label="wrapper") as profiler:
        timed_result = _ordered_parallel_map(
            tasks,
            support_config=_support_config(4),
            score_one=score_one,
            batch_phase="test.batch",
            task_phase="test.task",
        )
    assert timed_result == plain
    receipt = profiler.receipt()
    phases = {row["phase"]: row for row in receipt["phases"]}
    assert phases["test.task"]["count"] == len(tasks)
    assert phases["test.batch"]["count"] == 1
    assert receipt["counters"]["test.batch.tasks"] == len(tasks)


# ---------------------------------------------------------------------------
# Profiler behaviour
# ---------------------------------------------------------------------------


def test_phase_is_noop_when_no_session_is_active() -> None:
    assert active_profiler() is None
    assert phase("anything") is NULL_PHASE
    with phase("anything"):
        pass
    count("ignored", 5)  # must not raise
    assert active_profiler() is None


def test_profiling_session_installs_and_removes_the_profiler() -> None:
    with profiling_session(label="session") as profiler:
        assert active_profiler() is profiler
        with phase("work"):
            pass
    assert active_profiler() is None
    receipt = profiler.receipt()
    assert receipt["schema"] == PROFILE_RECEIPT_SCHEMA_V1
    assert receipt["label"] == "session"


def test_profiling_session_rejects_nesting() -> None:
    with profiling_session(label="outer"):
        with pytest.raises(RuntimeError, match="already active"):
            with profiling_session(label="inner"):
                pass
    assert active_profiler() is None


def test_profiling_session_is_removed_after_an_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with profiling_session(label="failing"):
            raise RuntimeError("boom")
    assert active_profiler() is None


def test_phase_records_even_when_the_body_raises() -> None:
    with profiling_session(label="raises") as profiler:
        with pytest.raises(ValueError):
            with phase("explodes"):
                raise ValueError("inner")
    phases = {row["phase"]: row for row in profiler.receipt()["phases"]}
    assert phases["explodes"]["count"] == 1


def test_nested_phases_report_exclusive_time_without_double_counting() -> None:
    with profiling_session(label="nested") as profiler:
        with phase("outer"):
            with phase("inner"):
                time.sleep(0.02)
    phases = {row["phase"]: row for row in profiler.receipt()["phases"]}
    outer, inner = phases["outer"], phases["inner"]
    assert inner["inclusive_seconds"] >= 0.015
    assert outer["inclusive_seconds"] >= inner["inclusive_seconds"]
    # The outer span did almost nothing of its own.
    assert outer["exclusive_seconds"] < inner["inclusive_seconds"]


def test_profiler_merges_thread_local_tables_across_workers() -> None:
    """Scoring runs in a thread pool; every thread's timings must be counted."""

    profiler = PhaseProfiler(label="threads")
    thread_count = 6
    per_thread = 20
    barrier = threading.Barrier(thread_count)

    def worker() -> None:
        barrier.wait()
        for _ in range(per_thread):
            with profiler.phase("threaded"):
                pass
            profiler.add_counter("threaded.calls")

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    receipt = profiler.receipt()
    phases = {row["phase"]: row for row in receipt["phases"]}
    assert phases["threaded"]["count"] == thread_count * per_thread
    assert receipt["counters"]["threaded.calls"] == thread_count * per_thread


def test_receipt_is_deterministic_across_repeated_calls() -> None:
    profiler = PhaseProfiler(label="stable")
    for _ in range(50):
        with profiler.phase("a"):
            pass
        with profiler.phase("b"):
            pass
    first = profiler.receipt()
    second = profiler.receipt()
    assert [row["phase"] for row in first["phases"]] == [
        row["phase"] for row in second["phases"]
    ]
    assert {row["phase"]: row["count"] for row in first["phases"]} == {
        row["phase"]: row["count"] for row in second["phases"]
    }


def test_receipt_size_is_bounded_for_long_trajectories() -> None:
    """A t=50 run records millions of spans; the receipt must not grow with them."""

    profiler = PhaseProfiler(label="bounded")
    for _ in range(200_000):
        with profiler.phase("hot"):
            pass
    receipt = profiler.receipt()
    phases = {row["phase"]: row for row in receipt["phases"]}
    assert phases["hot"]["count"] == 200_000  # exact count, not a sample
    serialized = json.dumps(receipt)
    assert len(serialized) < 8192, f"receipt grew to {len(serialized)} bytes"
    assert len(phases["hot"]["worst_calls"]) <= receipt["worst_call_count"]


def test_receipt_quantiles_are_ordered_and_bounded_by_observed_extremes() -> None:
    profiler = PhaseProfiler(label="quantiles")
    for index in range(200):
        with profiler.phase("varied"):
            time.sleep(0.0005 if index % 20 else 0.005)
    row = {item["phase"]: item for item in profiler.receipt()["phases"]}["varied"]
    assert row["min_inclusive_seconds"] <= row["p50_inclusive_seconds"]
    assert row["p50_inclusive_seconds"] <= row["p90_inclusive_seconds"]
    assert row["p90_inclusive_seconds"] <= row["p99_inclusive_seconds"]
    assert row["p99_inclusive_seconds"] <= row["max_inclusive_seconds"]


def test_timed_decorator_preserves_return_value_and_metadata() -> None:
    @timed("decorated")
    def add(left: int, right: int = 3) -> int:
        """Docstring is preserved."""

        return left + right

    assert add.__name__ == "add"
    assert add.__doc__ == "Docstring is preserved."
    assert add(1, right=4) == 5  # inactive path
    with profiling_session(label="decorator") as profiler:
        assert add(2) == 5
    phases = {row["phase"]: row for row in profiler.receipt()["phases"]}
    assert phases["decorated"]["count"] == 1


def test_receipt_is_json_serializable_with_no_numpy_scalars() -> None:
    with profiling_session(label="json") as profiler:
        with phase("work", note="candidate-3"):
            pass
        count("things", 2)
    receipt = profiler.receipt()
    round_tripped = json.loads(json.dumps(receipt))
    assert round_tripped["counters"]["things"] == 2
    worst = round_tripped["phases"][0]["worst_calls"]
    assert worst and worst[0]["note"] == "candidate-3"


def test_parallel_batch_exclusive_time_excludes_worker_thread_children() -> None:
    """Cross-thread children must be charged to the dispatching phase.

    Worker threads keep their own nested-phase stacks, so without explicit
    attribution the batch phase reports the whole parallel section as its own
    exclusive time.
    """

    tasks = tuple(range(8))

    def score_one(task: int) -> int:
        time.sleep(0.01)
        return int(task)

    with profiling_session(label="cross_thread") as profiler:
        _ordered_parallel_map(
            tasks,
            support_config=_support_config(4),
            score_one=score_one,
            batch_phase="test.parallel_batch",
            task_phase="test.parallel_task",
        )
    phases = {row["phase"]: row for row in profiler.receipt()["phases"]}
    batch = phases["test.parallel_batch"]
    task = phases["test.parallel_task"]
    assert task["count"] == len(tasks)
    # Nearly all the batch's inclusive time is its children's work.
    assert batch["exclusive_seconds"] < 0.5 * batch["inclusive_seconds"]
    assert task["inclusive_seconds"] >= 0.07


def test_serial_batch_exclusive_time_is_not_charged_twice() -> None:
    tasks = tuple(range(5))

    def score_one(task: int) -> int:
        time.sleep(0.01)
        return int(task)

    with profiling_session(label="serial_batch") as profiler:
        _ordered_parallel_map(
            tasks,
            support_config=_support_config(1),
            score_one=score_one,
            batch_phase="test.serial_batch",
            task_phase="test.serial_task",
        )
    phases = {row["phase"]: row for row in profiler.receipt()["phases"]}
    batch = phases["test.serial_batch"]
    assert batch["exclusive_seconds"] >= 0.0
    assert batch["exclusive_seconds"] < 0.5 * batch["inclusive_seconds"]


def test_attribute_nested_is_a_noop_without_an_open_phase() -> None:
    with profiling_session(label="no_open_phase"):
        attribute_nested(1.5)  # must not raise
    attribute_nested(1.5)  # inactive profiler must not raise


def test_support_identity_hash_memoization_is_exact_and_invalidates() -> None:
    """The memoized digest must equal the uncached one and follow the support."""

    import hashlib
    import json as _json

    from pipelines.time_dynamics.ap_mclachlan import adaptive_trajectory as apt

    def uncached(mode, labels):
        text = _json.dumps(
            {
                "parameterization_mode": str(mode),
                "runtime_coordinate_labels": [str(x) for x in labels],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    class _State:
        def __init__(self, mode, labels):
            self.parameterization_mode = mode
            self.runtime_coordinate_labels = tuple(labels)

    base = tuple(f"term_{i}::r0::x" for i in range(91))
    for mode, labels in (
        ("per_pauli_term", base),
        ("per_pauli_term", base[:-1]),          # after a delete
        ("per_pauli_term", base + ("new::r0::z",)),  # after an append
        ("logical_shared", base),               # different parameterization
        ("per_pauli_term", ()),                 # empty support
    ):
        assert apt._support_identity_hash(_State(mode, labels)) == uncached(mode, labels)

    # Distinct supports must not collide, and repeat calls must be stable.
    digests = {
        apt._support_identity_hash(_State("per_pauli_term", base[:n])) for n in range(0, 91)
    }
    assert len(digests) == 91
    state = _State("per_pauli_term", base)
    assert apt._support_identity_hash(state) == apt._support_identity_hash(state)


def test_support_identity_hash_distinguishes_reordered_supports() -> None:
    """Ordering is part of the support identity, so a swap must change it."""

    from pipelines.time_dynamics.ap_mclachlan import adaptive_trajectory as apt

    class _State:
        def __init__(self, labels):
            self.parameterization_mode = "per_pauli_term"
            self.runtime_coordinate_labels = tuple(labels)

    a = ("x::r0", "y::r0", "z::r0")
    b = ("y::r0", "x::r0", "z::r0")
    assert apt._support_identity_hash(_State(a)) != apt._support_identity_hash(_State(b))
