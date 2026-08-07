from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "chtc" / "paper_iv_h2o_static_snake_resume_20260705"
TASK = RUN_DIR / "run_h2o_sr_novelty_off_metric_prune_tolerance_guardfix_task.sh"
WRAPPER = RUN_DIR / "run_h2o_sr_novelty_off_metric_prune_tolerance_guardfix_apptainer.sh"
SUBMIT = RUN_DIR / "submit_h2o_sr_novelty_off_metric_prune_tolerance_guardfix.sub"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_option(text: str, option: str, value: str) -> None:
    assert f"  {option}\n  {value}\n" in text


def test_task_locks_requested_sr_snake_route() -> None:
    task = _text(TASK)

    assert "--adapt-resume-scaffold-json" not in task
    assert '"starting_depth": 0' in task
    assert '"route_family": "singleton_response_snake"' in task
    assert '"route_profile": "supported_whitened_adaptive_trust_v1"' in task
    assert '"source_cluster": 8787626' in task
    assert '"scientific_settings_changed": False' in task
    assert "  --phase0-no-pilot\n" in task
    assert "  --phase2-no-batching\n" in task
    assert "  --phase3-no-batching\n" in task
    _assert_option(task, "--phase3-novelty-ablation-mode", "all")
    _assert_option(
        task,
        "--historical-singleton-coordinate-solve-policy",
        "supported_metric_whitened_eigh_v1",
    )
    _assert_option(
        task,
        "--historical-singleton-coordinate-solve-scope",
        "phase3_only_v1",
    )
    _assert_option(
        task,
        "--historical-singleton-trust-region-update-policy",
        "displacement_calibrated_unbounded_v2",
    )
    _assert_option(task, "--sr-escape-mode", "disabled")


def test_task_locks_metric_pruning_and_corrected_fixture() -> None:
    task = _text(TASK)

    assert "h2o_linear_fd_sparse_fixture_nph1_ref2_reencoded_v2.json" in task
    assert (
        'EXPECTED_FIXTURE_SHA256="'
        "570690bd126787305b340bd2f7493499c0f3101e3e2820c2d355c55c16afa594"
        '"'
    ) in task
    assert "  --phase1-prune-enabled\n" in task
    _assert_option(task, "--phase1-prune-policy", "recoverability_ladder_v1")
    _assert_option(task, "--phase1-prune-mode", "both")
    _assert_option(
        task,
        "--phase1-prune-schur-nomination-route",
        "metric_regularized_v1",
    )
    _assert_option(task, "--phase1-prune-metric-schur-mu", "0.01")
    _assert_option(
        task,
        "--phase1-prune-metric-schur-solve-mode",
        "stationary_gw_zero_v1",
    )
    _assert_option(
        task,
        "--phase1-prune-metric-schur-cost-weighting",
        "ansatz_entry_denominator_v1",
    )


def test_submit_is_fresh_depth15_sr_job_with_expected_resources() -> None:
    submit = _text(SUBMIT)
    wrapper = _text(WRAPPER)

    assert "resume_depth11" not in submit
    assert "checkpoint" not in submit.lower()
    assert "H2O_ADAPT_MAX_DEPTH=15" in submit
    assert "H2O_ADAPT_SEGMENT_MAX_NEW_ADMISSIONS=15" in submit
    assert "request_cpus = 8" in submit
    assert "request_memory = 32GB" in submit
    assert "+MaxRuntime = 604800" in submit
    assert "run_h2o_sr_novelty_off_metric_prune_tolerance_guardfix_task.sh" in submit
    assert "h2o_sr_snake_novelty_off_metric_prune_tolerance_guardfix_code_20260714.tgz" in submit
    assert "run_h2o_sr_novelty_off_metric_prune_tolerance_guardfix_task.sh" in wrapper
    assert "REPLACE_AFTER_ARCHIVE_BUILD" not in wrapper
    assert (
        'EXPECTED_CODE_ARCHIVE_SHA256="'
        "4d63b7ede5ec87aab9fb3f8e6ccc654e66c9295139add4385068a5cb128fce65"
        '"'
    ) in wrapper
