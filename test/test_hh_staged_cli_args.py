from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_staged_noiseless import parse_args


def test_checkpoint_controller_mode_defaults_off() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.checkpoint_controller_mode) == "off"


def test_staged_parser_keeps_phase1_v1_compatibility_default() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.adapt_continuation_mode) == "phase1_v1"
    assert str(args.replay_continuation_mode) == "auto"
    assert str(args.phase3_oracle_gradient_mode) == "off"
    assert str(args.phase3_oracle_execution_surface) == "auto"
    assert str(args.phase3_oracle_raw_transport) == "auto"


def test_staged_parser_accepts_phase3_raw_oracle_args() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--adapt-continuation-mode",
            "phase3_v1",
            "--phase3-oracle-gradient-mode",
            "backend_scheduled",
            "--phase3-oracle-use-fake-backend",
            "--phase3-oracle-backend-name",
            "FakeNighthawk",
            "--phase3-oracle-execution-surface",
            "raw_measurement_v1",
            "--phase3-oracle-raw-transport",
            "sampler_v2",
            "--phase3-oracle-raw-store-memory",
        ]
    )
    assert str(args.phase3_oracle_gradient_mode) == "backend_scheduled"
    assert bool(args.phase3_oracle_use_fake_backend) is True
    assert str(args.phase3_oracle_execution_surface) == "raw_measurement_v1"
    assert str(args.phase3_oracle_raw_transport) == "sampler_v2"
    assert bool(args.phase3_oracle_raw_store_memory) is True


def test_checkpoint_controller_mode_accepts_exact_v1() -> None:
    args = parse_args(["--L", "2", "--skip-pdf", "--checkpoint-controller-mode", "exact_v1"])
    assert str(args.checkpoint_controller_mode) == "exact_v1"


def test_checkpoint_controller_mode_accepts_oracle_v1() -> None:
    args = parse_args(["--L", "2", "--skip-pdf", "--checkpoint-controller-mode", "oracle_v1"])
    assert str(args.checkpoint_controller_mode) == "oracle_v1"


def test_checkpoint_controller_oracle_selection_policy_defaults_legacy_path() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.checkpoint_controller_oracle_selection_policy) == "measured_gain_commit_veto"


def test_checkpoint_controller_oracle_selection_policy_accepts_oracle_energy_rerank() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-oracle-selection-policy",
            "measured_topk_oracle_energy",
        ]
    )
    assert str(args.checkpoint_controller_oracle_selection_policy) == "measured_topk_oracle_energy"


def test_checkpoint_controller_candidate_step_scales_defaults_full_step() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.checkpoint_controller_candidate_step_scales) == "1.0"


def test_checkpoint_controller_candidate_step_scales_accepts_csv() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-candidate-step-scales",
            "0.25,0.5,1.0",
        ]
    )
    assert str(args.checkpoint_controller_candidate_step_scales) == "0.25,0.5,1.0"


def test_checkpoint_controller_exact_forecast_guardrail_defaults_off() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.checkpoint_controller_exact_forecast_guardrail_mode) == "off"
    assert float(args.checkpoint_controller_exact_forecast_fidelity_loss_tol) == 0.0
    assert float(args.checkpoint_controller_exact_forecast_abs_energy_error_increase_tol) == 0.0


def test_checkpoint_controller_exact_forecast_guardrail_accepts_dual_metric() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-guardrail-mode",
            "dual_metric_v1",
            "--checkpoint-controller-exact-forecast-fidelity-loss-tol",
            "0.01",
            "--checkpoint-controller-exact-forecast-abs-energy-error-increase-tol",
            "0.02",
        ]
    )
    assert str(args.checkpoint_controller_exact_forecast_guardrail_mode) == "dual_metric_v1"
    assert float(args.checkpoint_controller_exact_forecast_fidelity_loss_tol) == 0.01
    assert float(args.checkpoint_controller_exact_forecast_abs_energy_error_increase_tol) == 0.02
