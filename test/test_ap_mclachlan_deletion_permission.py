"""Measurement-free permission contract for Paper-II deletions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.deletion_permission import (
    DELETION_PERMISSION_ANGLE_RAY,
    DELETION_PERMISSION_SCHUR_LOSS,
    DeletionPermissionEvaluator,
)


def _evaluator(
    *,
    K: np.ndarray,
    f: np.ndarray,
    theta: tuple[float, ...],
    ray_max: float = 2.0e-3,
    schur_max: float = 1.0e-3,
) -> DeletionPermissionEvaluator:
    return DeletionPermissionEvaluator(
        gram=np.asarray(K, dtype=float),
        force=np.asarray(f, dtype=float),
        norm_b_sq=1.0,
        theta_runtime=np.asarray(theta, dtype=float),
        rotation_coefficients=np.ones(len(theta), dtype=float),
        ray_distance_max=float(ray_max),
        normalized_schur_loss_max=float(schur_max),
        epsilon_norm=1.0e-12,
    )


def test_small_angle_redundant_coordinate_is_permitted_without_overlap() -> None:
    evaluator = _evaluator(
        K=np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        f=np.asarray([1.0, 1.0]),
        theta=(1.0e-3, 0.0),
    )

    decision = evaluator.assess((0,))

    assert decision.permitted
    assert decision.reasons == ()
    assert decision.ray_distance_upper_bound == pytest.approx(math.sin(1.0e-3))
    assert decision.normalized_schur_loss == pytest.approx(0.0, abs=1.0e-12)


def test_angle_bound_rejects_state_moving_deletion_before_materialization() -> None:
    evaluator = _evaluator(
        K=np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        f=np.asarray([1.0, 1.0]),
        theta=(1.0e-2, 0.0),
    )

    decision = evaluator.assess((0,))

    assert not decision.permitted
    assert decision.reasons == (DELETION_PERMISSION_ANGLE_RAY,)
    assert decision.normalized_schur_loss is None


def test_schur_loss_rejects_forced_novel_direction_even_at_zero_angle() -> None:
    evaluator = _evaluator(
        K=np.eye(2),
        f=np.asarray([0.0, 1.0]),
        theta=(0.0, 0.0),
        schur_max=0.25,
    )

    decision = evaluator.assess((1,))

    assert not decision.permitted
    assert decision.reasons == (DELETION_PERMISSION_SCHUR_LOSS,)
    assert decision.ray_distance_upper_bound == 0.0
    assert decision.normalized_schur_loss == pytest.approx(1.0)


def test_multi_deletion_uses_sum_of_effective_rotation_angles() -> None:
    evaluator = _evaluator(
        K=np.eye(2),
        f=np.zeros(2),
        theta=(1.1e-3, 1.1e-3),
    )

    decision = evaluator.assess((0, 1))

    assert not decision.permitted
    assert decision.reasons == (DELETION_PERMISSION_ANGLE_RAY,)
    assert decision.ray_distance_upper_bound == pytest.approx(math.sin(2.2e-3))


def test_decisions_are_memoized_and_summary_counts_unique_deletion_sets() -> None:
    evaluator = _evaluator(
        K=np.eye(2),
        f=np.asarray([0.0, 1.0]),
        theta=(1.0e-2, 0.0),
        schur_max=0.25,
    )

    first = evaluator.assess((0,))
    assert evaluator.assess((0,)) is first
    evaluator.assess((1,))

    summary = evaluator.summary()
    assert summary["evaluated_deletion_set_count"] == 2
    assert summary["permitted_deletion_set_count"] == 0
    assert summary["rejected_deletion_set_count"] == 2
    assert summary["schur_evaluated_deletion_set_count"] == 1
    assert summary["schur_skipped_by_angle_count"] == 1
    assert summary["rejection_reason_counts"] == {
        DELETION_PERMISSION_ANGLE_RAY: 1,
        DELETION_PERMISSION_SCHUR_LOSS: 1,
    }


def test_concurrent_assess_does_not_inflate_summary_counts() -> None:
    # The evaluator memoizes on the canonical deletion set; concurrent
    # duplicate assessments must produce identical decisions and must not
    # inflate the unique-set summary counts.
    import threading
    from concurrent.futures import ThreadPoolExecutor

    def build():
        return _evaluator(
            K=np.asarray([[1.0, 1.0], [1.0, 1.0]]),
            f=np.asarray([0.0, 1.0]),
            theta=(1.0e-2, 0.0),
            schur_max=0.25,
        )

    serial = build()
    sets = ((0,), (1,), (0, 1), ())
    serial_decisions = {s: serial.assess(s) for s in sets}

    concurrent = build()
    barrier = threading.Barrier(4)

    def assess_after_barrier(deleted):
        barrier.wait()
        return concurrent.assess(deleted)

    tasks = [s for s in sets for _repeat in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(assess_after_barrier, tasks))

    for deleted, decision in zip(tasks, results):
        canonical = tuple(sorted(set(deleted)))
        assert decision.to_json_dict() == serial_decisions[canonical].to_json_dict()
    assert concurrent.summary() == serial.summary()
    assert concurrent.summary()["evaluated_deletion_set_count"] == len(sets)
    for s in sets:
        assert concurrent.assess(s) is concurrent.assess(s)
