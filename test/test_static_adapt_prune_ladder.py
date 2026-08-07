from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold.hh_continuation_pruning import recoverability_prune_ladder
from pipelines.static_adapt.commutation_metadata import (
    EXACTNESS_EXACT,
    AlgebraicPairMetadata,
)
from pipelines.static_adapt.prune_ladder import (
    RECOVERY_CURVATURE_COMPENSATED,
    RECOVERY_FAILED,
    RECOVERY_FLAT_REDUNDANT,
    RUNG_TERMINAL_REFIT,
    build_nested_prune_rungs,
    build_prune_ladder_plan,
    build_typed_compensator_pools,
    classify_recoverability_from_ladder_rows,
    original_to_post_delete_index,
    rung_windows_for_recoverability,
)


def _pair(lhs: str, rhs: str, *, support_overlap: bool, commutes: bool) -> AlgebraicPairMetadata:
    return AlgebraicPairMetadata(
        lhs_key=lhs,
        rhs_key=rhs,
        lhs_label=lhs,
        rhs_label=rhs,
        support_overlap=bool(support_overlap),
        commutes=bool(commutes),
        exactness=EXACTNESS_EXACT,
        relation=(
            "flat_comm"
            if support_overlap and commutes
            else "curv_noncomm"
            if support_overlap and not commutes
            else "disj_comm"
        ),
    )


def _pair_metadata() -> dict[tuple[str, str], AlgebraicPairMetadata]:
    return {
        ("drop", "comm_overlap"): _pair("drop", "comm_overlap", support_overlap=True, commutes=True),
        ("drop", "disjoint_comm"): _pair("drop", "disjoint_comm", support_overlap=False, commutes=True),
        ("drop", "noncomm_overlap"): _pair("drop", "noncomm_overlap", support_overlap=True, commutes=False),
        ("drop", "corr"): _pair("drop", "corr", support_overlap=False, commutes=True),
        ("drop", "age"): _pair("drop", "age", support_overlap=False, commutes=True),
    }


def test_typed_compensator_pools_exclude_disjoint_commuting_from_redundancy_pool() -> None:
    labels = ["drop", "comm_overlap", "disjoint_comm", "noncomm_overlap", "corr", "age"]
    pools = build_typed_compensator_pools(
        removal_index=0,
        labels=labels,
        pair_metadata=_pair_metadata(),
        correlated_indices=[4],
        age_indices=[5],
    )

    assert pools.removal_label == "drop"
    assert pools.survivor_original_indices == (1, 2, 3, 4, 5)
    assert pools.comm_indices == (0,)
    assert pools.nc_indices == (2,)
    assert pools.corr_indices == (3,)
    assert pools.age_indices == (4,)
    assert pools.term_indices == (0, 1, 2, 3, 4)
    assert pools.relation_summary["exact_overlap_commuting"] == 1
    assert pools.relation_summary["exact_overlap_noncommuting"] == 1
    assert pools.relation_summary["exact_disjoint_commuting"] == 3


def test_original_to_post_delete_index_mapping() -> None:
    assert original_to_post_delete_index(0, 2) == 0
    assert original_to_post_delete_index(2, 2) is None
    assert original_to_post_delete_index(4, 2) == 3


def test_nested_prune_rungs_are_monotone_and_post_delete_indexed() -> None:
    labels = ["drop", "comm_overlap", "disjoint_comm", "noncomm_overlap", "corr", "age"]
    pools = build_typed_compensator_pools(
        removal_index=0,
        labels=labels,
        pair_metadata=_pair_metadata(),
        correlated_indices=[4],
        age_indices=[5],
    )
    rungs = build_nested_prune_rungs(pools, terminal_full=True)
    rung_sets = [set(rung.active_logical_indices) for rung in rungs]

    assert rungs[0].active_logical_indices == ()
    assert rungs[1].active_logical_indices == (0,)
    assert rungs[2].active_logical_indices == (0, 3)
    assert rungs[3].active_logical_indices == (0, 2, 3)
    assert rungs[4].active_logical_indices == (0, 1, 2, 3, 4)
    assert "" not in build_nested_prune_rungs(pools, terminal_full=False)[4].opened_pool_kinds
    assert all(rung_sets[idx].issubset(rung_sets[idx + 1]) for idx in range(len(rung_sets) - 1))
    assert all(max(rung.active_logical_indices, default=-1) <= len(labels) - 2 for rung in rungs)


def test_recovery_classification_is_telemetry_only() -> None:
    flat = classify_recoverability_from_ladder_rows(
        [
            {"rung_index": 0, "rung_kind": "frozen_delete", "accepted": False},
            {
                "rung_index": 2,
                "rung_kind": "comm_corr_refit",
                "accepted": True,
                "acceptance_source": "remove_refit_energy_safety",
            },
        ]
    )
    curv = classify_recoverability_from_ladder_rows(
        [{"rung_index": 3, "rung_kind": "comm_corr_nc_refit", "accepted": True}]
    )
    failed = classify_recoverability_from_ladder_rows([{"rung_index": 4, "accepted": False}])

    assert flat["recovery_class"] == RECOVERY_FLAT_REDUNDANT
    assert flat["acceptance_source"] == "remove_refit_energy_safety"
    assert curv["recovery_class"] == RECOVERY_CURVATURE_COMPENSATED
    assert failed["recovery_class"] == RECOVERY_FAILED


def test_rung_windows_feed_existing_remove_refit_acceptance_boundary() -> None:
    labels = ["drop", "comm_overlap", "disjoint_comm", "noncomm_overlap", "corr", "age"]
    plan = build_prune_ladder_plan(
        removal_index=0,
        labels=labels,
        pair_metadata=_pair_metadata(),
        correlated_indices=[4],
        age_indices=[5],
        terminal_full=True,
    )
    calls: list[str] = []

    def _eval(idx_remove, theta_cur, labels_cur, active_indices, rung_kind):
        calls.append(str(rung_kind))
        theta_new = np.delete(theta_cur, idx_remove)
        if str(rung_kind) == RUNG_TERMINAL_REFIT:
            return 1.0, theta_new * 0.0
        return 1.5, theta_new

    theta_out, labels_out, decisions, energy_out, rows = recoverability_prune_ladder(
        theta=np.array([0.2, 0.1, 0.1, -0.1, 0.05, 0.03], dtype=float),
        labels=labels,
        candidate_indices=[0],
        rung_windows_by_index={0: rung_windows_for_recoverability(plan)},
        eval_with_removal_window=_eval,
        energy_before=1.0,
        max_regression=1e-8,
    )

    assert calls[-1] == RUNG_TERMINAL_REFIT
    assert labels_out == labels[1:]
    assert energy_out == 1.0
    assert np.allclose(theta_out, np.zeros(5))
    assert [decision.accepted for decision in decisions][-1] is True
    assert rows[-1]["acceptance_source"] == "remove_refit_energy_safety"
    assert rows[-1]["surrogate_used_for_acceptance"] is False
    assert classify_recoverability_from_ladder_rows(rows)["recovery_class"] == RECOVERY_CURVATURE_COMPENSATED
