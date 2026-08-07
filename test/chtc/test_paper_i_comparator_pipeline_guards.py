from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from chtc.phase3_optuna import run_paper_i_hh_spsa_budget_ladder_cell as cell_runner
from pipelines.exact_bench import generic_static_benchmark as benchmark


@pytest.mark.parametrize(
    ("payload_status", "row_status", "expected"),
    [
        ("completed", "ok", 0),
        ("completed_quality_nonpassing", "quality_nonpassing", 3),
        ("failed", "failed", 3),
    ],
)
def test_run_single_cli_propagates_payload_failure_status(
    monkeypatch,
    tmp_path: Path,
    payload_status: str,
    row_status: str,
    expected: int,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "run_single",
        lambda **kwargs: {
            "schema": "test",
            "status": payload_status,
            "result": {"status": row_status},
            "rows": [{"status": row_status}],
        },
    )

    rc = benchmark.main(
        [
            "--run-single",
            "--family",
            "hubbard",
            "--algorithm-id",
            "static_full_meta_append_adapt_vqe",
            "--case-id",
            "hubbard_L2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == expected


def test_generic_comparator_result_summary_reads_nested_result(tmp_path: Path) -> None:
    result_path = tmp_path / "result" / "generic_static_single.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "result": {
                    "status": "ok",
                    "energy": -1.25,
                    "exact_energy": -1.3,
                    "abs_delta_e": 0.05,
                    "adapt_depth_reached": 7,
                    "adapt_stop_reason": "max_adapt_iterations",
                    "adapt_history": [{"depth_after": 7}],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = cell_runner.result_summary_from_artifacts({"method_key": "append"}, tmp_path)

    assert summary["status"] == "ok"
    assert summary["payload_status"] == "completed"
    assert summary["energy"] == -1.25
    assert summary["exact_gs_energy"] == -1.3
    assert summary["ansatz_depth"] == 7
    assert summary["stop_reason"] == "max_adapt_iterations"
    assert summary["winner_branch_history_step_count"] == 1


def test_append_geo_continuation_record_fails_closed_before_launch(tmp_path: Path) -> None:
    row = {
        "method_key": "geo",
        "batch_id": "paper_i_hh_comparator_continuations_local_20260709_v1",
        "schedule_source_note": "Geo comparator continuation from depth 30",
    }

    with pytest.raises(RuntimeError, match="does not load the declared source operators"):
        cell_runner.run_append_geo(row, tmp_path)


def test_paper_i_cell_default_inner_optimizer_is_powell() -> None:
    assert cell_runner.row_inner_optimizer({}) == "POWELL"
    env = cell_runner.append_geo_env(
        {
            "suite_profile": "paper_i_three_model_main_20260525_v1",
            "budget": "200",
            "same_cutoff_exact_gs_energy": "-1.0",
            "exact_reference_energy": "-1.0",
            "exact_reference_n_ph_max": "4",
        },
        Path("/tmp/paper_i_comparator_test"),
    )
    assert env["GENERIC_STATIC_TABLE_ADAPT_OPTIMIZER_KIND"] == "powell"
    assert env["GENERIC_STATIC_TABLE_PHASE3_ADAPT_ALLOW_REPEATS"] == "true"
    assert env["GENERIC_STATIC_TABLE_PROGRESS_STDOUT"] == "1"


def test_comparator_subprocess_progress_is_teed_to_parent_and_files(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    env = dict(os.environ)
    env["GENERIC_STATIC_TABLE_PROGRESS_STDOUT"] = "1"
    command = [
        sys.executable,
        "-c",
        "import sys; print('progress-out', flush=True); print('progress-err', file=sys.stderr, flush=True)",
    ]

    _cmd, overlay, returncode = cell_runner.run_subprocess(command, env, tmp_path)

    captured = capsys.readouterr()
    assert returncode == 0
    assert overlay["GENERIC_STATIC_TABLE_PROGRESS_STDOUT"] == "1"
    assert "progress-out" in captured.out
    assert "progress-err" in captured.err
    assert (tmp_path / "stdout.log").read_text(encoding="utf-8") == "progress-out\n"
    assert (tmp_path / "stderr.log").read_text(encoding="utf-8") == "progress-err\n"
