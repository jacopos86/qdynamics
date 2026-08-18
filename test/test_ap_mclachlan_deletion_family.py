"""Deletion rungs and the joint-work guard."""

from __future__ import annotations

import pytest

from pipelines.time_dynamics.ap_mclachlan.deletion_family import (
    JointWorkGuard,
    deletion_rung,
    iter_deletion_rungs,
    rung_size,
)


def test_every_hard_feasible_subset_is_enumerated_per_rung() -> None:
    deletable = (2, 5, 7)
    rung = deletion_rung(
        deletable, cardinality=2, total_support=10, min_surviving_support=1
    )
    assert rung == ((2, 5), (2, 7), (5, 7))
    assert len(rung) == rung_size(3, 2)


def test_rung_zero_is_the_stay_branch() -> None:
    assert deletion_rung(
        (1, 2), cardinality=0, total_support=5, min_surviving_support=1
    ) == ((),)


def test_min_surviving_support_truncates_rungs_and_all_later_ones() -> None:
    # 4 total coordinates, 3 deletable, must keep >= 2: d can be 0,1,2 only.
    rungs = dict(
        iter_deletion_rungs((0, 1, 3), total_support=4, min_surviving_support=2)
    )
    assert set(rungs) == {0, 1, 2}
    assert len(rungs[1]) == 3 and len(rungs[2]) == 3
    # d=3 would leave 1 survivor -> generator stops before yielding it.


def test_no_separate_deletion_cardinality_cap_exists() -> None:
    import pipelines.time_dynamics.ap_mclachlan.deletion_family as mod

    assert not any("max_deletion_batch" in name for name in dir(mod))
    # With a permissive surviving-support bound every cardinality up to
    # |J_del| appears.
    rungs = dict(
        iter_deletion_rungs((0, 1, 2), total_support=9, min_surviving_support=0)
    )
    assert set(rungs) == {0, 1, 2, 3}
    assert rungs[3] == ((0, 1, 2),)


def test_guard_none_admits_everything() -> None:
    guard = JointWorkGuard(max_joint_patch_evaluations=None)
    assert guard.admit("d=0", 10**9)
    assert guard.admit("d=1", 10**9)
    assert guard.scored_count == 2 * 10**9
    assert not guard.exhausted


def test_guard_admits_complete_families_and_freezes_on_rejection() -> None:
    guard = JointWorkGuard(max_joint_patch_evaluations=100)
    assert guard.admit("singleton_d0", 40)
    assert guard.admit("singleton_d1", 60)  # exactly at the cap
    assert not guard.admit("singleton_d2", 1)  # would exceed -> rejected
    assert guard.exhausted
    assert guard.rejected_family == "singleton_d2"
    # Frozen after rejection: even a fitting family is refused, and completed
    # results are preserved.
    assert not guard.admit("tiny", 0)
    assert guard.scored_count == 100
    assert guard.admitted_families == ["singleton_d0", "singleton_d1"]


def test_guard_never_partially_admits_a_family() -> None:
    guard = JointWorkGuard(max_joint_patch_evaluations=5)
    assert not guard.admit("big", 6)
    assert guard.scored_count == 0  # nothing scored from the rejected family


def test_guard_serialization_and_validation() -> None:
    guard = JointWorkGuard(max_joint_patch_evaluations=7)
    guard.admit("a", 3)
    payload = guard.to_json_dict()
    assert payload == {
        "max_joint_patch_evaluations": 7,
        "scored_count": 3,
        "admitted_families": ["a"],
        "rejected_family": None,
    }
    with pytest.raises(ValueError):
        JointWorkGuard(max_joint_patch_evaluations=-1)
