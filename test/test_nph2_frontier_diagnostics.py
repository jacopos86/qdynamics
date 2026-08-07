import json
from pathlib import Path

import pytest

from pipelines.exact_bench.nph2_frontier_diagnostics import summarize_frontier_diagnostics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_frontier_diagnostics_parser_prefers_runtime_round_fields(tmp_path: Path):
    adapt_path = tmp_path / "result" / "hk" / "json" / "result.json"
    generic_path = tmp_path / "result" / "generic_static_single.json"
    _write_json(
        adapt_path,
        {
            "adapt_vqe": {
                "stop_reason": "empty",
                "pool_size": 14,
                "ansatz_depth": 3,
                "abs_delta_e": 0.125,
                "continuation": {
                    "beam_search": {
                        "rounds": [
                            {
                                "depth": 1,
                                "raw_candidate_record_count": 42,
                                "phase2_raw_candidate_record_count": 9,
                                "phase1_shortlist_size": 12,
                                "phase2_shortlist_size": 6,
                                "phase3_shortlist_size": 4,
                                "proposal_family_count": 4,
                                "proposals_selected_count": 4,
                                "frontier_input_count": 1,
                                "frontier_kept_count": 4,
                                "terminal_kept_count": 1,
                                "best_available_gradient": 2.5,
                                "best_available_full_v2_score": 0.75,
                            },
                            {
                                "depth": 2,
                                "raw_candidate_record_count": 18,
                                "phase2_raw_candidate_record_count": 5,
                                "phase1_shortlist_size": 5,
                                "phase2_shortlist_size": 2,
                                "phase3_shortlist_size": 1,
                                "proposal_family_count": 0,
                                "proposals_selected_count": 0,
                                "frontier_input_count": 4,
                                "frontier_kept_count": 0,
                                "terminal_kept_count": 4,
                                "stop_reason": "empty",
                                "best_available_gradient": 1.0e-4,
                                "best_available_full_v2_score": 0.0,
                                "parent_stop_reason_counts": {"empty": 4},
                            },
                        ]
                    }
                },
            }
        },
    )
    _write_json(
        generic_path,
        {
            "schema": "generic_static_benchmark_phase3_single_v1",
            "family": "harmonic_kerr_chain",
            "case_id": "harmonic_kerr_chain_L2_nph2",
            "algorithm_id": "static_family_native_adapt_phase3",
            "result": {
                "result_json": str(adapt_path),
                "pool_size": 14,
                "ansatz_depth": 3,
                "stop_reason": "empty",
                "abs_delta_e_same_cutoff": 0.0,
                "abs_delta_e_reference": 0.2,
                "boson_illegal_probability_max": 0.0,
            },
        },
    )

    payload = summarize_frontier_diagnostics(generic_path)

    assert payload["summary"]["collapse_depth"] == 2
    assert payload["summary"]["pool_size"] == 14
    assert payload["summary"]["abs_delta_e_same_cutoff"] == pytest.approx(0.0)
    assert payload["per_depth"][1]["raw_candidate_record_count"] == 18
    assert payload["per_depth"][1]["phase1_shortlist_size"] == 5
    assert payload["per_depth"][1]["phase3_shortlist_size"] == 1
    assert payload["per_depth"][1]["stop_reason"] == "empty"
    assert payload["per_depth"][1]["best_available_full_v2_score"] == pytest.approx(0.0)
    assert "frontier=4->0" in payload["summary"]["frontier_summary"]


def test_frontier_diagnostics_parser_falls_back_to_old_history_and_finalists(tmp_path: Path):
    adapt_path = tmp_path / "json" / "result.json"
    _write_json(
        adapt_path,
        {
            "adapt_vqe": {
                "stop_reason": "empty",
                "pool_size": 61,
                "ansatz_depth": 9,
                "abs_delta_e": 0.176,
                "history": [
                    {
                        "depth": 1,
                        "scored_surface_size": 7,
                        "shortlist_size": 7,
                        "retained_shortlist_size": 3,
                        "max_grad": 2.0,
                        "full_v2_score": 0.25,
                        "phase2_raw_score": 0.1,
                        "delta_abs_drop_from_prev": 0.5,
                    }
                ],
                "continuation": {
                    "beam_search": {
                        "winner_stop_reason": "empty",
                        "rounds": [
                            {
                                "depth": 1,
                                "proposal_family_count": 4,
                                "proposals_selected_count": 4,
                                "frontier_input_count": 1,
                                "frontier_kept_count": 4,
                                "terminal_kept_count": 1,
                            },
                            {
                                "depth": 2,
                                "proposal_family_count": 0,
                                "proposals_selected_count": 0,
                                "frontier_input_count": 4,
                                "frontier_kept_count": 0,
                                "terminal_kept_count": 4,
                            },
                        ],
                        "finalist_summaries": [
                            {
                                "depth_local": 1,
                                "stop_reason": "empty",
                                "scored_surface_count": 13,
                                "retained_shortlist_count": 1,
                            }
                        ],
                    }
                },
            }
        },
    )

    payload = summarize_frontier_diagnostics(adapt_path)

    assert payload["summary"]["collapse_depth"] == 2
    assert payload["per_depth"][0]["phase2_raw_candidate_record_count"] == 7
    assert payload["per_depth"][0]["phase3_shortlist_size"] == 3
    assert payload["per_depth"][0]["best_available_gradient"] == pytest.approx(2.0)
    assert payload["per_depth"][1]["phase2_raw_candidate_record_count"] == 13
    assert payload["per_depth"][1]["phase3_shortlist_size"] == 1
    assert payload["per_depth"][1]["stop_reason"] == "empty"
