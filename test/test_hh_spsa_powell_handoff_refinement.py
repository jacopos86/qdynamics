from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.hh_spsa_powell_handoff_refinement import (
    CaseOutcome,
    _best_compile_row,
    _build_cases,
    _lane_payload,
    _study_has_feasible_points,
)


def test_study_has_feasible_points_detects_count_or_observation() -> None:
    assert _study_has_feasible_points({"studies": [{"feasible_count": 1}]}) is True
    assert _study_has_feasible_points({"studies": [{"feasible_count": 0, "observations": [{"feasible": True}]}]}) is True
    assert _study_has_feasible_points({"studies": [{"feasible_count": 0, "observations": [{"feasible": False}]}]}) is False


def test_build_cases_default_matrix_size_and_order() -> None:
    cases = _build_cases(
        seed_names=["current_98", "current_118", "legacy_75", "legacy_81"],
        policies=["auto", "tile_adapt"],
    )
    assert len(cases) == 8
    assert [case.lane for case in cases[:4]] == ["current", "current", "current", "current"]
    assert [case.trial_name for case in cases[:4]] == ["trial_0000", "trial_0001", "trial_0002", "trial_0003"]
    assert [case.trial_name for case in cases[4:]] == ["trial_0000", "trial_0001", "trial_0002", "trial_0003"]


def test_best_compile_row_prefers_2q_then_depth() -> None:
    row = _best_compile_row(
        {
            "rows": [
                {"compiled_count_2q": 120, "compiled_depth": 300, "compiled_size": 500},
                {"compiled_count_2q": 98, "compiled_depth": 330, "compiled_size": 510},
                {"compiled_count_2q": 98, "compiled_depth": 320, "compiled_size": 540},
            ]
        }
    )
    assert row == {"compiled_count_2q": 98, "compiled_depth": 320, "compiled_size": 540}


def test_lane_payload_counts_feasible() -> None:
    payload = _lane_payload(
        "current",
        6.2e-5,
        outcomes=[
            CaseOutcome(
                lane="current",
                case_name="a",
                case_dir="current/trial_0000",
                params={},
                abs_delta_e=1.0e-5,
                compiled_count_2q=100,
                compiled_depth=200,
                logical_operator_count=10,
                runtime_parameter_count=20,
                feasible=True,
                constraints=[0.0, 0.0],
                result_json="result.json",
                compile_json="compile.json",
                returncode=0,
                compile_returncode=0,
                pipeline_elapsed_s=1.0,
                compile_elapsed_s=1.0,
                total_elapsed_s=2.0,
                invalid_reasons=[],
                seed_name="current_98",
                seed_notes="x",
            ),
            CaseOutcome(
                lane="legacy",
                case_name="b",
                case_dir="legacy/trial_0000",
                params={},
                abs_delta_e=1.0e-5,
                compiled_count_2q=90,
                compiled_depth=180,
                logical_operator_count=9,
                runtime_parameter_count=18,
                feasible=True,
                constraints=[0.0, 0.0],
                result_json="result.json",
                compile_json="compile.json",
                returncode=0,
                compile_returncode=0,
                pipeline_elapsed_s=1.0,
                compile_elapsed_s=1.0,
                total_elapsed_s=2.0,
                invalid_reasons=[],
                seed_name="legacy_75",
                seed_notes="y",
            ),
            CaseOutcome(
                lane="current",
                case_name="c",
                case_dir="current/trial_0001",
                params={},
                abs_delta_e=2.0e-4,
                compiled_count_2q=110,
                compiled_depth=220,
                logical_operator_count=11,
                runtime_parameter_count=22,
                feasible=False,
                constraints=[1.0, 0.0],
                result_json="result.json",
                compile_json="compile.json",
                returncode=0,
                compile_returncode=0,
                pipeline_elapsed_s=1.0,
                compile_elapsed_s=1.0,
                total_elapsed_s=2.0,
                invalid_reasons=["energy_band_failed"],
                seed_name="current_118",
                seed_notes="z",
            ),
        ],
    )
    assert payload["completed_trial_count"] == 2
    assert payload["feasible_count"] == 1
