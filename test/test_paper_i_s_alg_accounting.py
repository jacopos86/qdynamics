from __future__ import annotations

import pytest

from pipelines.exact_bench.paper_i_s_alg_accounting import (
    PROJECTED_PHASE_ORDER,
    SNAKE_REPRESENTATION_INTACT_MACRO,
    SNAKE_REPRESENTATION_PROJECTED_SINGLETON,
    append_clean_prefix_work,
    runtime_prefix_work,
    snake_clean_prefix_work,
)


def _controller(
    *, r1: int, r2: int, r3: int, phase2_retained: int | None = None
) -> dict:
    retained = r2 if phase2_retained is None else phase2_retained
    by_phase = {
        "phase1": {
            "method_input_candidate_count_total": r1,
            "actual_operator_probe_count_total": r1,
            "method_shortlist_candidate_count_total": r2,
        },
        "phase2": {
            "method_input_candidate_count_total": r2,
            "actual_evaluated_candidate_count_total": r2,
            "events_count": 1,
        },
        "phase3": {
            "method_input_candidate_count_total": r3,
            "candidate_count_total": r3,
            "actual_evaluated_candidate_count_total": r3,
            "events_count": 1,
        },
    }
    phase2_scope = {
        "method_input_candidate_count_total": r2,
        "actual_evaluated_candidate_count_total": r2,
        "candidate_count_total": r2,
        "pre_shortlist_count_total": r2,
        "records_evaluated": r2,
        "shortlist_size_total": retained,
        "retained_count_total": retained,
    }
    phase3_scope = {
        "method_input_candidate_count_total": r3,
        "actual_evaluated_candidate_count_total": r3,
        "candidate_count_total": r3,
        "pre_shortlist_count_total": r3,
        "records_evaluated": r3,
    }
    return {
        "by_phase": by_phase,
        "by_scope": {
            "static_adapt|phase=phase2|event=phase2_rerank_records|depth=1": (
                phase2_scope
            ),
            (
                "static_adapt|phase=phase3|"
                "event=phase3_reduced_geometry_rerank|depth=1"
            ): phase3_scope,
        },
    }


def _macro_row(
    *, n: int, r1: int, r2: int, r3: int, nfev_opt: int
) -> dict:
    return {
        "phase3_active_logical_coordinate_count": n,
        "nfev_opt": nfev_opt,
        "controller_measurement_work_proxy": _controller(
            r1=r1, r2=r2, r3=r3
        ),
    }


def _split_record(child_count: int) -> dict:
    return {
        "runtime_split_mode": "shortlist_pauli_children_v1",
        "runtime_split_chosen_representation": "child_set",
        "runtime_split_child_count": child_count,
        "runtime_split_child_indices": [0],
        "runtime_split_child_labels": ["child_0"],
        "runtime_split_child_generator_ids": ["child_0"],
    }


def _unsplit_record() -> dict:
    return {
        "runtime_split_mode": "off",
        "runtime_split_chosen_representation": "parent",
    }


def _projected_row(
    *,
    n: int,
    r1: int,
    split_child_counts: list[int],
    unsplit_count: int,
    nfev_opt: int,
    phase3_child_count: int | None = None,
) -> dict:
    records = [
        *[_split_record(count) for count in split_child_counts],
        *[_unsplit_record() for _ in range(unsplit_count)],
    ]
    r2 = len(records)
    child_population = sum(split_child_counts) + unsplit_count
    r3 = (
        child_population
        if phase3_child_count is None
        else phase3_child_count
    )
    row = {
        "phase3_active_logical_coordinate_count": n,
        "nfev_opt": nfev_opt,
        "controller_measurement_work_proxy": _controller(
            r1=r1,
            r2=r2,
            r3=r3,
            phase2_retained=len(records),
        ),
        "scored_surface_size": len(records),
        "scored_surface_records": records,
    }
    return row


def _snake_receipt(*, k: int, h_refit: int) -> dict:
    components = {
        "N_H_outer": k + 1,
        "N_H_refit": h_refit,
        "N_grad": 999,
        "N_metric": 999,
    }
    return {
        "cumulative_raw_occurrences": {
            "components": components,
            "total": sum(components.values()),
        }
    }


def test_runtime_prefix_counts_occurrences_and_excludes_unique_diagnostic():
    receipt = {
        "schema": "paper_i_active_prefix_estimator_ledger_receipt_v2",
        "status": "complete",
        "outer_iteration": 3,
        "cumulative_raw_occurrences": {
            "components": {
                "N_H_outer": 3,
                "N_H_refit": 19,
                "N_grad": 11,
                "N_metric": 23,
            },
            "total": 56,
        },
        "cumulative_unique_primitives": {
            "components": {
                "N_H_outer": 1,
                "N_H_refit": 12,
                "N_grad": 9,
                "N_metric": 17,
            },
            "S_unique": 39,
        },
    }
    work = runtime_prefix_work(
        method="SNAKE",
        representation="intact_macro",
        accepted_prefix_length=3,
        estimator_ledger_receipt=receipt,
    )

    assert work["S_alg"] == 56
    assert work["components"] == {
        "N_H_outer": 3,
        "N_H_refit": 19,
        "N_grad": 11,
        "N_metric": 23,
    }
    assert work["normalization"]["unique_primitive_diagnostic_excluded"] is True


def test_runtime_prefix_removes_one_audited_legacy_initial_outer_refresh():
    receipt = {
        "schema": "paper_i_active_prefix_estimator_ledger_receipt_v1",
        "status": "complete",
        "outer_iteration": 2,
        "cumulative_raw_occurrences": {
            "components": {
                "N_H_outer": 3,
                "N_H_refit": 7,
                "N_grad": 5,
                "N_metric": 11,
            },
            "total": 26,
        },
    }
    work = runtime_prefix_work(
        method="SNAKE",
        representation="intact_macro",
        accepted_prefix_length=2,
        estimator_ledger_receipt=receipt,
    )

    assert work["components"]["N_H_outer"] == 2
    assert work["S_alg"] == 25
    assert (
        work["normalization"][
            "redundant_initial_outer_refresh_count_removed"
        ]
        == 1
    )


def test_snake_projected_counts_children_after_phase1_parent_shortlist():
    history = [
        _projected_row(
            n=0,
            r1=102,
            split_child_counts=[6, 5],
            unsplit_count=1,
            nfev_opt=18,
            phase3_child_count=4,
        ),
        _projected_row(
            n=1,
            r1=102,
            split_child_counts=[5, 4],
            unsplit_count=1,
            nfev_opt=19,
            phase3_child_count=4,
        ),
    ]
    receipt = _snake_receipt(k=2, h_refit=37)
    singleton = snake_clean_prefix_work(
        history=history,
        accepted_prefix_length=2,
        representation=SNAKE_REPRESENTATION_PROJECTED_SINGLETON,
        estimator_ledger_receipt=receipt,
    )

    assert singleton["components"] == {
        "N_H_outer": 2,
        "N_H_refit": 37,
        "N_grad": 227,
        "N_metric": 258,
    }
    assert [row["R3"] for row in singleton["round_cardinalities"]] == [4, 4]
    assert singleton["round_cardinalities"][0][
        "phase1_retained_parent_count"
    ] == 3
    assert singleton["round_cardinalities"][0]["R2"] == 12
    assert PROJECTED_PHASE_ORDER == (
        "phase1_parent_shortlist_then_split_then_phase2_children_then_phase3"
    )


def test_snake_intact_macro_uses_phase3_evaluated_input_not_surface_size():
    history = [
        _macro_row(n=0, r1=102, r2=24, r3=12, nfev_opt=18),
        _macro_row(n=1, r1=102, r2=24, r3=10, nfev_opt=19),
    ]
    macro = snake_clean_prefix_work(
        history=history,
        accepted_prefix_length=2,
        representation=SNAKE_REPRESENTATION_INTACT_MACRO,
        estimator_ledger_receipt=_snake_receipt(k=2, h_refit=37),
    )
    assert macro["components"] == {
        "N_H_outer": 2,
        "N_H_refit": 37,
        "N_grad": 205,
        "N_metric": 274,
    }


def test_snake_multi_event_phase2_uses_exact_candidate_handoff_fields():
    row = _macro_row(n=0, r1=102, r2=24, r3=12, nfev_opt=11)
    phase2 = row["controller_measurement_work_proxy"]["by_phase"]["phase2"]
    phase2.update(
        {
            "events_count": 3,
            "pre_shortlist_count_total": 24,
            "common_expanded_candidate_count_total": 24,
            "method_input_candidate_count_total": 48,
            "actual_evaluated_candidate_count_total": 60,
        }
    )
    work = snake_clean_prefix_work(
        history=[row],
        accepted_prefix_length=1,
        representation=SNAKE_REPRESENTATION_INTACT_MACRO,
        estimator_ledger_receipt=_snake_receipt(k=1, h_refit=11),
    )
    assert work["round_cardinalities"] == [
        {
            "history_index": 0,
            "n_active": 0,
            "R1": 102,
            "R2": 24,
            "phase2_acquisition_event_count": 3,
            "R3": 12,
            "R3_evaluated_candidate_count": 12,
            "N_H_refit": 11,
        }
    ]
    assert work["components"] == {
        "N_H_outer": 1,
        "N_H_refit": 11,
        "N_grad": 102,
        "N_metric": 126,
    }


def test_snake_phase_cardinality_mismatch_fails_closed():
    row = _macro_row(n=0, r1=102, r2=24, r3=12, nfev_opt=11)
    phase3_scope = next(
        value
        for key, value in row["controller_measurement_work_proxy"][
            "by_scope"
        ].items()
        if "|phase=phase3|" in key
    )
    phase3_scope["actual_evaluated_candidate_count_total"] = 11
    with pytest.raises(ValueError, match="does not close"):
        snake_clean_prefix_work(
            history=[row],
            accepted_prefix_length=1,
            representation=SNAKE_REPRESENTATION_INTACT_MACRO,
            estimator_ledger_receipt=_snake_receipt(k=1, h_refit=11),
        )


def test_snake_projected_parent_surface_mismatch_fails_closed():
    row = _projected_row(
        n=0,
        r1=8,
        split_child_counts=[3, 2],
        unsplit_count=1,
        nfev_opt=4,
        phase3_child_count=3,
    )
    row["scored_surface_size"] -= 1
    with pytest.raises(ValueError, match="parent shortlist does not close"):
        snake_clean_prefix_work(
            history=[row],
            accepted_prefix_length=1,
            representation=SNAKE_REPRESENTATION_PROJECTED_SINGLETON,
            estimator_ledger_receipt=_snake_receipt(k=1, h_refit=4),
        )


def test_snake_projected_signed_surface_needs_no_parent_before_split_receipt():
    row = _projected_row(
        n=0,
        r1=102,
        split_child_counts=[8],
        unsplit_count=0,
        nfev_opt=11,
        phase3_child_count=4,
    )
    work = snake_clean_prefix_work(
        history=[row],
        accepted_prefix_length=1,
        representation=SNAKE_REPRESENTATION_PROJECTED_SINGLETON,
        estimator_ledger_receipt=_snake_receipt(k=1, h_refit=11),
    )
    assert work["round_cardinalities"][0]["R2"] == 8
    assert work["round_cardinalities"][0]["R3"] == 4


def test_snake_accepts_already_clean_outer_h_prefix():
    row = _macro_row(n=0, r1=4, r2=2, r3=1, nfev_opt=3)
    receipt = _snake_receipt(k=1, h_refit=3)
    receipt["cumulative_raw_occurrences"]["components"]["N_H_outer"] = 1
    receipt["cumulative_raw_occurrences"]["total"] -= 1
    work = snake_clean_prefix_work(
        history=[row],
        accepted_prefix_length=1,
        representation=SNAKE_REPRESENTATION_INTACT_MACRO,
        estimator_ledger_receipt=receipt,
    )
    assert work["normalization"][
        "redundant_initial_outer_refresh_count_removed"
    ] == 0


def test_snake_runtime_receipt_cannot_understate_clean_component_work():
    row = _macro_row(n=0, r1=4, r2=2, r3=1, nfev_opt=3)
    receipt = _snake_receipt(k=1, h_refit=3)
    receipt["cumulative_raw_occurrences"]["components"]["N_grad"] = 0
    receipt["cumulative_raw_occurrences"]["total"] = sum(
        receipt["cumulative_raw_occurrences"]["components"].values()
    )
    with pytest.raises(ValueError, match="cannot support the clean N_grad"):
        snake_clean_prefix_work(
            history=[row],
            accepted_prefix_length=1,
            representation=SNAKE_REPRESENTATION_INTACT_MACRO,
            estimator_ledger_receipt=receipt,
        )


def test_locked_weak_weak_projected_k29_child_route_recount():
    child_counts = [
        36, 54, 40, 36, 36, 42, 24, 24, 68, 52,
        52, 52, 52, 52, 50, 52, 76, 76, 74, 52,
        22, 68, 74, 74, 70, 68, 74, 68, 66,
    ]
    unsplit_counts = [
        1, 1, 3, 3, 3, 2, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        3, 1, 1, 1, 1, 2, 1, 2, 2,
    ]
    history = []
    for n, (child_total, unsplit) in enumerate(
        zip(child_counts, unsplit_counts, strict=True)
    ):
        split_parent_count = 8 - unsplit
        quotient, remainder = divmod(child_total, split_parent_count)
        per_parent = [
            quotient + (1 if index < remainder else 0)
            for index in range(split_parent_count)
        ]
        history.append(
            _projected_row(
                n=n,
                r1=102,
                split_child_counts=per_parent,
                unsplit_count=unsplit,
                nfev_opt=(18_608 if n == 0 else 0),
                phase3_child_count=4,
            )
        )

    receipt = _snake_receipt(k=29, h_refit=18_608)
    receipt["cumulative_raw_occurrences"]["components"]["N_grad"] = 10_000
    receipt["cumulative_raw_occurrences"]["components"]["N_metric"] = 100_000
    receipt["cumulative_raw_occurrences"]["total"] = sum(
        receipt["cumulative_raw_occurrences"]["components"].values()
    )
    work = snake_clean_prefix_work(
        history=history,
        accepted_prefix_length=29,
        representation=SNAKE_REPRESENTATION_PROJECTED_SINGLETON,
        estimator_ledger_receipt=receipt,
    )
    assert sum(
        row["phase1_retained_parent_count"]
        for row in work["round_cardinalities"]
    ) == 232
    assert sum(row["R2"] for row in work["round_cardinalities"]) == 1_625
    assert sum(row["R3"] for row in work["round_cardinalities"]) == 116
    assert work["components"] == {
        "N_H_outer": 29,
        "N_H_refit": 18_608,
        "N_grad": 4_989,
        "N_metric": 17_576,
    }
    assert work["S_alg"] == 41_202


def test_append_removes_only_legacy_post_refit_verifiers():
    work = append_clean_prefix_work(
        accepted_prefix_length=2,
        cumulative_occurrence_summary={
            "component_occurrence_counts": {
                "N_H_outer": 2,
                "N_H_refit": 19,
                "N_grad": 204,
                "N_metric": 0,
            },
            "total_call_occurrences": 225,
        },
        redundant_post_refit_verification_count=2,
        representation="intact_macro",
    )
    assert work["components"] == {
        "N_H_outer": 2,
        "N_H_refit": 17,
        "N_grad": 204,
        "N_metric": 0,
    }
    assert work["S_alg"] == 223
