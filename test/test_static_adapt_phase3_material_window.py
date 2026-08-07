from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.phase3_material_window import (
    Phase3MaterialWindowPolicy,
    build_phase3_material_window,
)


def _policy(
    *,
    gram_threshold: float = 0.5,
    hessian_threshold: float = 0.5,
    gram_tail: float = 1.0,
    hessian_tail: float = 1.0,
) -> Phase3MaterialWindowPolicy:
    return Phase3MaterialWindowPolicy(
        gram_entry_threshold=gram_threshold,
        hessian_entry_threshold=hessian_threshold,
        gram_omitted_l2_tolerance=gram_tail,
        hessian_omitted_l2_tolerance=hessian_tail,
    )


def test_independent_gram_and_hessian_masks_form_deterministic_union() -> None:
    receipt = build_phase3_material_window(
        active_indices=[7, 2, 9],
        gram_diagonal=[1.0, 1.0, 1.0],
        candidate_gram_cross=[0.8, 0.1, 0.1],
        candidate_gram_self=1.0,
        candidate_hessian_cross=[0.1, 0.8, 0.1],
        candidate_hessian_self=0.0,
        policy=_policy(),
    )

    assert receipt.initial_gram_mask == (True, False, False)
    assert receipt.initial_hessian_mask == (False, True, False)
    assert receipt.initial_union_mask == (True, True, False)
    assert receipt.final_retained_mask == (True, True, False)
    assert receipt.retained_indices == (7, 2)
    assert receipt.omitted_indices == (9,)
    assert receipt.closure_satisfied is True
    assert len(receipt.receipt_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        receipt.closure_reason = "changed"  # type: ignore[misc]


def test_greedy_closure_ties_break_by_active_index() -> None:
    receipt = build_phase3_material_window(
        active_indices=[5, 3, 8],
        gram_diagonal=[1.0, 1.0, 1.0],
        candidate_gram_cross=[0.4, 0.4, 0.1],
        candidate_gram_self=1.0,
        candidate_hessian_cross=[0.4, 0.4, 0.1],
        candidate_hessian_self=1.0,
        policy=_policy(
            gram_threshold=0.9,
            hessian_threshold=0.9,
            gram_tail=0.75,
            hessian_tail=0.75,
        ),
    )

    assert receipt.initial_union_mask == (False, False, False)
    assert receipt.closure_added_indices == (3,)
    assert receipt.final_retained_mask == (False, True, False)
    assert receipt.final_gram_omitted_l2_ratio == pytest.approx(
        (0.4**2 + 0.1**2) ** 0.5 / (0.4**2 + 0.4**2 + 0.1**2) ** 0.5
    )


def test_closure_expands_past_threshold_union_until_both_tails_close() -> None:
    receipt = build_phase3_material_window(
        active_indices=[0, 1, 2],
        gram_diagonal=[1.0, 1.0, 1.0],
        candidate_gram_cross=[0.9, 0.3, 0.2],
        candidate_gram_self=1.0,
        candidate_hessian_cross=[0.1, 0.8, 0.2],
        candidate_hessian_self=1.0,
        policy=_policy(
            gram_threshold=0.85,
            hessian_threshold=0.75,
            gram_tail=0.2,
            hessian_tail=0.2,
        ),
    )

    assert receipt.initial_gram_mask == (True, False, False)
    assert receipt.initial_hessian_mask == (False, True, False)
    assert receipt.initial_union_mask == (True, True, False)
    assert receipt.closure_added_indices == (2,)
    assert receipt.retained_indices == (0, 1, 2)
    assert receipt.final_gram_omitted_l2_ratio == 0.0
    assert receipt.final_hessian_omitted_l2_ratio == 0.0
    assert receipt.closure_reason == "satisfied_after_greedy_expansion"


def test_candidate_only_window_is_closed_and_rank_gain_one_is_ordinary() -> None:
    receipt = build_phase3_material_window(
        active_indices=[],
        gram_diagonal=[],
        candidate_gram_cross=[],
        candidate_gram_self=1.0,
        candidate_hessian_cross=[],
        candidate_hessian_self=2.0,
        policy=_policy(),
        prior_active_nullity=0,
        prior_joint_nullity=0,
    )
    finalized = receipt.finalize_with_support_ranks(
        active_supported_rank=0,
        joint_supported_rank=1,
    )

    assert receipt.closure_reason == "candidate_only"
    assert receipt.retained_indices == ()
    assert finalized.measured_rank_gain == 1
    assert finalized.support_nullity_drift is False
    assert finalized.requires_full_geometry_refresh is False


def test_nonfinite_input_fails_closed_and_requests_full_refresh() -> None:
    receipt = build_phase3_material_window(
        active_indices=[0],
        gram_diagonal=[1.0],
        candidate_gram_cross=[float("nan")],
        candidate_gram_self=1.0,
        candidate_hessian_cross=[0.2],
        candidate_hessian_self=1.0,
        policy=_policy(),
    )
    finalized = receipt.finalize_with_support_ranks(
        active_supported_rank=1,
        joint_supported_rank=2,
    )

    assert receipt.inputs_finite is False
    assert receipt.gram_normalized_scores == (None,)
    assert receipt.closure_satisfied is False
    assert finalized.requires_full_geometry_refresh is True
    assert "nonfinite_input" in finalized.refresh_reasons
    assert "closure_failed" in finalized.refresh_reasons
    assert finalized.to_dict()["gram_normalized_scores"] == [None]


def test_rank_gain_and_nullity_drift_request_refresh() -> None:
    receipt = build_phase3_material_window(
        active_indices=[0, 1],
        gram_diagonal=[1.0, 1.0],
        candidate_gram_cross=[0.8, 0.1],
        candidate_gram_self=1.0,
        candidate_hessian_cross=[0.8, 0.1],
        candidate_hessian_self=1.0,
        policy=_policy(gram_threshold=0.0, hessian_threshold=0.0),
        prior_active_nullity=0,
        prior_joint_nullity=0,
    )

    drifted = receipt.finalize_with_support_ranks(
        active_supported_rank=1,
        joint_supported_rank=1,
    )
    invalid_gain = receipt.finalize_with_support_ranks(
        active_supported_rank=0,
        joint_supported_rank=2,
    )

    assert drifted.measured_rank_gain == 0
    assert drifted.measured_active_nullity == 1
    assert drifted.measured_joint_nullity == 2
    assert drifted.support_nullity_drift is True
    assert "active_support_nullity_drift" in drifted.refresh_reasons
    assert "joint_support_nullity_drift" in drifted.refresh_reasons
    assert invalid_gain.measured_rank_gain == 2
    assert "invalid_rank_gain" in invalid_gain.refresh_reasons


def test_dimension_growth_alone_does_not_trigger_drift() -> None:
    receipt = build_phase3_material_window(
        active_indices=[10, 20],
        gram_diagonal=[1.0, 1.0],
        candidate_gram_cross=[0.6, 0.2],
        candidate_gram_self=1.0,
        candidate_hessian_cross=[0.6, 0.2],
        candidate_hessian_self=1.0,
        policy=_policy(gram_threshold=0.0, hessian_threshold=0.0),
        prior_active_nullity=0,
        prior_joint_nullity=0,
    )
    finalized = receipt.finalize_with_support_ranks(
        active_supported_rank=2,
        joint_supported_rank=3,
    )

    assert finalized.measured_rank_gain == 1
    assert finalized.measured_active_nullity == 0
    assert finalized.measured_joint_nullity == 0
    assert finalized.requires_full_geometry_refresh is False
    assert finalized.refresh_reasons == ()


def test_support_ranks_are_interpreted_on_retained_workspace() -> None:
    receipt = build_phase3_material_window(
        active_indices=[10, 20, 30],
        gram_diagonal=[1.0, 1.0, 1.0],
        candidate_gram_cross=[0.8, 0.01, 0.01],
        candidate_gram_self=1.0,
        candidate_hessian_cross=[0.8, 0.01, 0.01],
        candidate_hessian_self=1.0,
        policy=_policy(),
        prior_active_nullity=0,
        prior_joint_nullity=0,
    )
    assert receipt.retained_indices == (10,)

    finalized = receipt.finalize_with_support_ranks(
        active_supported_rank=1,
        joint_supported_rank=2,
    )

    assert finalized.measured_active_nullity == 0
    assert finalized.measured_joint_nullity == 0
    assert finalized.requires_full_geometry_refresh is False


def test_external_block_closure_failure_forces_full_refresh() -> None:
    receipt = build_phase3_material_window(
        active_indices=[0, 1],
        gram_diagonal=[1.0, 1.0],
        candidate_gram_cross=[1.0, 0.0],
        candidate_gram_self=1.0,
        candidate_hessian_cross=[1.0, 0.0],
        candidate_hessian_self=1.0,
        policy=_policy(),
    )

    finalized = receipt.finalize_with_support_ranks(
        active_supported_rank=len(receipt.retained_indices),
        joint_supported_rank=len(receipt.retained_indices) + 1,
        additional_refresh_reasons=("gram_cross_block_closure_failed",),
    )

    assert finalized.requires_full_geometry_refresh is True
    assert finalized.refresh_reasons == (
        "gram_cross_block_closure_failed",
    )
