from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "position_aware_phase0_canary_20260816.py"
)
RUNNER_PATH = ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_paper_i_position_aware_phase0_sw_always_k15_20260816.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("position_phase0_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_position_gradient_ranking_is_record_level_and_deterministic() -> None:
    module = load_module()
    rows = [
        {
            "domain_record_id": "g0@0",
            "pool_index": 0,
            "insertion_position": 0,
            "gradient_signed": 0.5,
        },
        {
            "domain_record_id": "g0@1",
            "pool_index": 0,
            "insertion_position": 1,
            "gradient_signed": -2.0,
        },
        {
            "domain_record_id": "g1@0",
            "pool_index": 1,
            "insertion_position": 0,
            "gradient_signed": 2.0,
        },
    ]
    ranked = module.rank_position_gradient_rows(rows, shortlist_size=2)
    assert [row["domain_record_id"] for row in ranked] == ["g0@1", "g1@0"]
    assert ranked[0]["gradient_abs"] == 2.0
    assert ranked[1]["gradient_abs"] == 2.0


def test_position_plan_filter_retains_only_selected_commutation_classes() -> None:
    module = load_module()
    plans = {
        3: {
            "schema": "commutation_reduced_insertion_positions_v1",
            "requested_positions": [0, 1, 2, 3],
            "representative_positions": [0, 2, 3],
            "representative_by_position": {0: 0, 1: 0, 2: 2, 3: 3},
            "members_by_representative": {0: [0, 1], 2: [2], 3: [3]},
            "commuting_crossings": [True, False, False],
            "collapsed_position_count": 1,
        }
    }
    filtered = module.filtered_position_plans(plans, {3: [2, 0]})
    assert filtered[3]["representative_positions"] == [0, 2]
    assert filtered[3]["requested_positions"] == [0, 1, 2]
    assert filtered[3]["members_by_representative"] == {0: [0, 1], 2: [2]}
    assert filtered[3]["collapsed_position_count"] == 1


def test_position_plan_filter_rejects_nonrepresentative_position() -> None:
    module = load_module()
    plans = {
        3: {
            "schema": "commutation_reduced_insertion_positions_v1",
            "representative_positions": [0],
            "members_by_representative": {0: [0, 1]},
            "commuting_crossings": [True],
        }
    }
    with pytest.raises(ValueError, match="escaped"):
        module.filtered_position_plans(plans, {3: [1]})


def test_runner_is_fixed_to_one_strong_weak_always_open_k15_diagnostic() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'TARGET_HORIZON = 15' in source
    assert '"strong_weak_u8"' in source
    assert '"always_commutation_reduced"' in source
    assert '"nph": 3' in source
    assert '"run_class": "diagnostic"' in source
    assert '"paper_adoption_authorized": False' in source
    assert '"paper_evidence_adoption_authorized": False' in source
    assert '"submission_authorized": False' in source
    assert "while True" not in source

