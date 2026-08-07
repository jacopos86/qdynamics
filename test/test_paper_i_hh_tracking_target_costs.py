from __future__ import annotations

import io

import pytest

from pipelines.reporting.build_paper_i_hh_tracking_target_costs import (
    TARGET_ABS_ERROR,
    _iter_named_json_array,
    select_target_prefix,
)
from pipelines.reporting.build_paper_i_hh_tracking_plateau_costs import (
    _append_redundant_post_refit_verification_count,
    _signed_checkpoint_estimator_work,
)


def test_snake_target_selection_uses_first_inclusive_crossing() -> None:
    payload = {
        "adapt_vqe": {
            "history": [
                {"delta_abs_current": 1.0e-2, "outer_iteration": 1},
                {"delta_abs_current": TARGET_ABS_ERROR, "outer_iteration": 2},
                {"delta_abs_current": 1.0e-8, "outer_iteration": 3},
            ]
        }
    }
    selected = select_target_prefix(payload, method="snake")
    assert selected is not None
    assert selected["k_target"] == 2
    assert selected["outer_iteration"] == 2
    assert selected["error"] == TARGET_ABS_ERROR


def test_comparator_target_selection_does_not_interpolate() -> None:
    payload = {
        "status": "completed",
        "result": {
            "adapt_history": [
                {"abs_delta_e_same_cutoff_after": 3.0e-4},
                {"abs_delta_e_same_cutoff_after": 1.5e-4},
            ]
        },
    }
    selected = select_target_prefix(payload, method="comparator")
    assert selected is not None
    assert selected["k_target"] == 2
    assert selected["error"] == 1.5e-4


def test_target_selection_returns_none_when_unreached() -> None:
    payload = {
        "adapt_vqe": {
            "history": [
                {"delta_abs_current": 4.0e-4},
                {"delta_abs_current": 3.0e-4},
            ]
        }
    }
    assert select_target_prefix(payload, method="snake") is None


def test_target_selection_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        select_target_prefix(
            {"adapt_vqe": {"history": [{"delta_abs_current": 1.0e-3}]}},
            method="snake",
            target_abs_error=0.0,
        )


def test_signed_checkpoint_estimator_work_uses_closed_runtime_prefix() -> None:
    checkpoint = {
        "estimator_ledger_receipt": {
            "schema": "paper_i_active_prefix_estimator_ledger_receipt_v1",
            "status": "complete",
            "outer_iteration": 3,
            "checkpoint_kind": "post_admission_prune",
            "checkpoint_sequence": 3,
            "canonical_same_state_deduplication_active": True,
            "cumulative_raw_occurrences": {
                "total": 28,
                "components": {
                    "N_H_outer": 4,
                    "N_H_refit": 8,
                    "N_grad": 6,
                    "N_metric": 10,
                },
            },
            "cumulative_unique_primitives": {
                "S_alg": 17,
                "components": {
                    "N_H_outer": 1,
                    "N_H_refit": 4,
                    "N_grad": 5,
                    "N_metric": 7,
                },
            },
        }
    }
    work = _signed_checkpoint_estimator_work(checkpoint, outer_iteration=3)
    assert work is not None
    assert work["raw_total"] == 28
    assert sum(work["raw_components"].values()) == work["raw_total"]


def test_signed_checkpoint_estimator_work_fails_on_nonclosing_receipt() -> None:
    checkpoint = {
        "estimator_ledger_receipt": {
            "schema": "paper_i_active_prefix_estimator_ledger_receipt_v1",
            "status": "complete",
            "outer_iteration": 3,
            "canonical_same_state_deduplication_active": True,
            "cumulative_raw_occurrences": {
                "total": 29,
                "components": {
                    "N_H_outer": 4,
                    "N_H_refit": 8,
                    "N_grad": 6,
                    "N_metric": 10,
                },
            },
            "cumulative_unique_primitives": {
                "S_alg": 18,
                "components": {
                    "N_H_outer": 1,
                    "N_H_refit": 4,
                    "N_grad": 5,
                    "N_metric": 7,
                },
            },
        }
    }
    with pytest.raises(ValueError, match="does not close"):
        _signed_checkpoint_estimator_work(checkpoint, outer_iteration=3)


def test_bounded_memory_array_reader_handles_nested_strings_and_arrays() -> None:
    raw = (
        b'{"prefix":0,"adapt_history":['
        b'{"error":0.1,"text":"brace } and quote \\\"","nested":[1,{"x":2}]},'
        b'{"error":0.0001}],"tail":true}'
    )
    rows = list(_iter_named_json_array(io.BytesIO(raw), "adapt_history"))
    assert [row["error"] for row in rows] == [0.1, 0.0001]
    assert rows[0]["nested"][1]["x"] == 2


def test_append_verifier_count_is_derived_from_legacy_scopes() -> None:
    assert (
        _append_redundant_post_refit_verification_count(
            {
                "occurrence_count_by_consumer_scope": {
                    "round:0:adapt_refit_powell_objective:objective": 7,
                    "round:0:adapt_refit_powell_objective:"
                    "post_optimizer_exact_verification": 1,
                    "round:1:adapt_refit_powell_objective:objective": 11,
                    "round:1:adapt_refit_powell_objective:"
                    "post_optimizer_exact_verification": 1,
                }
            },
            accepted_prefix_length=2,
        )
        == 2
    )


def test_append_verifier_count_accepts_post_fix_zero_pattern() -> None:
    assert (
        _append_redundant_post_refit_verification_count(
            {
                "occurrence_count_by_consumer_scope": {
                    "round:0:adapt_refit_powell_objective:objective": 7,
                    "round:0:adapt_refit_powell_objective:"
                    "required_endpoint_exact_verification": 1,
                    "round:1:adapt_refit_powell_objective:objective": 11,
                }
            },
            accepted_prefix_length=2,
        )
        == 0
    )


def test_append_verifier_count_rejects_mixed_prefix() -> None:
    with pytest.raises(ValueError, match="mixed or incomplete"):
        _append_redundant_post_refit_verification_count(
            {
                "occurrence_count_by_consumer_scope": {
                    "round:0:adapt_refit_powell_objective:"
                    "post_optimizer_exact_verification": 1,
                }
            },
            accepted_prefix_length=2,
        )
