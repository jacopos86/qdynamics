from __future__ import annotations

import copy

import pytest

from chtc.phase3_optuna.run_paper_i_hh_sr_query_neutral_prune_matched_weak_weak_20260723 import (
    CANDIDATE_ROUTE_ALIAS,
    EXPECTED_FILTERED_POOL_SIZE,
    RUNTIME_RECEIPT_REPAIR_NEW,
    RUNTIME_RECEIPT_REPAIR_OLD,
    SCOPE_REPAIR_NEW,
    SCOPE_REPAIR_OLD,
    apply_scope_repair,
    build_candidate_argv,
    validate_matched_pool_surface,
    _load_candidate_validator,
)


def _result(*, pool_size: int = EXPECTED_FILTERED_POOL_SIZE) -> dict:
    return {
        "adapt_vqe": {
            "pool_size": pool_size,
            "generator_pool_sector_contract": {
                "generator_count": pool_size,
                "filter": {
                    "applied": True,
                    "removed_count": 21,
                    "removed_labels": ["a", "b"],
                },
            },
            "shared_pauli_pool_ordered_label_hash": "labels",
            "shared_pauli_pool_ordered_pool_hash": "pool",
        }
    }


def test_matched_pool_surface_rejects_live_tree_123_generator_drift():
    parent = _result()
    candidate = _result(pool_size=123)
    candidate["adapt_vqe"]["generator_pool_sector_contract"]["filter"] = {
        "applied": False,
        "removed_count": 0,
        "removed_labels": [],
    }

    with pytest.raises(ValueError, match="pool-surface drift"):
        validate_matched_pool_surface(
            parent_result=parent,
            candidate_result=candidate,
        )


def test_matched_pool_surface_accepts_exact_parent_surface():
    receipt = validate_matched_pool_surface(
        parent_result=_result(),
        candidate_result=copy.deepcopy(_result()),
    )
    assert receipt["status"] == "pass"
    assert receipt["filtered_pool_size"] == EXPECTED_FILTERED_POOL_SIZE


def test_candidate_command_changes_only_route_and_operational_fields(tmp_path):
    parent = [
        "python3",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--sr-route-profile",
        "parent",
        "--adapt-max-depth",
        "50",
        "--adapt-segment-id",
        "parent-segment",
        "--adapt-segment-target-controller-round",
        "50",
        "--adapt-segment-target-depth",
        "50",
        "--adapt-segment-max-new-admissions",
        "50",
        "--adapt-current-json",
        "old-current",
        "--adapt-estimator-call-ledger-json",
        "old-ledger",
        "--output-json",
        "old-result",
    ]
    candidate = build_candidate_argv(
        parent,
        output_dir=tmp_path,
        max_rounds=7,
    )

    scientific_parent = list(parent)
    scientific_candidate = list(candidate)
    operational_options = {
        "--sr-route-profile",
        "--adapt-max-depth",
        "--adapt-segment-id",
        "--adapt-segment-target-controller-round",
        "--adapt-segment-target-depth",
        "--adapt-segment-max-new-admissions",
        "--adapt-current-json",
        "--adapt-estimator-call-ledger-json",
        "--output-json",
        "--phase1-prune-max-candidates",
        "--phase1-prune-local-window-size",
    }
    for values in (scientific_parent, scientific_candidate):
        for option in operational_options:
            if option in values:
                index = values.index(option)
                values[index + 1] = "<normalized>"
            else:
                values.extend((option, "<normalized>"))
        values.sort()
    assert scientific_candidate == scientific_parent
    route_index = candidate.index("--sr-route-profile")
    assert candidate[route_index + 1] == CANDIDATE_ROUTE_ALIAS


def test_scope_repair_is_exact_and_does_not_add_live_pool_policy(tmp_path):
    source = tmp_path / "pipelines/static_adapt"
    source.mkdir(parents=True)
    adapt = source / "adapt_pipeline.py"
    adapt.write_text(
        f"before\n{SCOPE_REPAIR_OLD}middle\n"
        f"{RUNTIME_RECEIPT_REPAIR_OLD}after\n",
        encoding="utf-8",
    )

    receipt = apply_scope_repair(tmp_path)

    repaired = adapt.read_text(encoding="utf-8")
    assert receipt["status"] == "pass"
    assert SCOPE_REPAIR_NEW in repaired
    assert RUNTIME_RECEIPT_REPAIR_NEW in repaired
    assert "def _resolve_parent_sector_filter_policy(" not in repaired


def test_optimizer_nfev_excludes_reporting_only_final_energy():
    validator = _load_candidate_validator()

    expected = validator._expected_query_neutral_optimizer_nfev(
        refit_occurrence_total=20_964,
        outer_occurrence_total=27,
        occurrence_scopes={
            "energy:initial_state": 1,
            "outer_state_refresh": 25,
            "final_state_verification": 1,
        },
    )

    assert expected == 20_965
