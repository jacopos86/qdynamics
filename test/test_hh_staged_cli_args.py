from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_staged_noiseless import parse_args


def test_checkpoint_controller_mode_defaults_off() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.checkpoint_controller_mode) == "off"


def test_staged_parser_defaults_to_phase3_v1() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.adapt_continuation_mode) == "phase3_v1"
    assert str(args.replay_continuation_mode) == "auto"
    assert str(args.phase3_oracle_gradient_mode) == "off"
    assert str(args.phase3_oracle_execution_surface) == "auto"
    assert str(args.phase3_oracle_raw_transport) == "auto"


def test_staged_parser_rejects_archival_phase3_runtime_split_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--phase3-runtime-split-mode",
                "shortlist_pauli_children_v1",
            ]
        )


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


def test_staged_parser_accepts_analytic_noise_args() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--adapt-analytic-noise-std",
            "0.125",
            "--adapt-analytic-noise-seed",
            "17",
            "--checkpoint-controller-analytic-noise-std",
            "0.25",
            "--checkpoint-controller-analytic-noise-seed",
            "33",
        ]
    )
    assert float(args.adapt_analytic_noise_std) == pytest.approx(0.125)
    assert int(args.adapt_analytic_noise_seed) == 17
    assert float(args.checkpoint_controller_analytic_noise_std) == pytest.approx(0.25)
    assert int(args.checkpoint_controller_analytic_noise_seed) == 33


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


def test_checkpoint_controller_exact_forecast_baseline_step_refine_rounds_default_off() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert int(args.checkpoint_controller_exact_forecast_baseline_step_refine_rounds) == 0


def test_checkpoint_controller_exact_forecast_baseline_step_refine_rounds_accepts_value() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-baseline-step-refine-rounds",
            "2",
        ]
    )
    assert int(args.checkpoint_controller_exact_forecast_baseline_step_refine_rounds) == 2


def test_checkpoint_controller_exact_forecast_baseline_blend_weights_default_off() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.checkpoint_controller_exact_forecast_baseline_proposal_mode) == "norm_locked_blend_v1"
    assert str(args.checkpoint_controller_exact_forecast_baseline_blend_weights) == ""


def test_checkpoint_controller_exact_forecast_baseline_proposal_mode_accepts_value() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-baseline-proposal-mode",
            "anticipatory_drive_basis_v1",
        ]
    )
    assert str(args.checkpoint_controller_exact_forecast_baseline_proposal_mode) == "anticipatory_drive_basis_v1"


def test_checkpoint_controller_exact_forecast_baseline_blend_weights_accepts_csv() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-baseline-blend-weights",
            "0,0.25,0.5,1.0",
        ]
    )
    assert str(args.checkpoint_controller_exact_forecast_baseline_blend_weights) == "0,0.25,0.5,1.0"


def test_checkpoint_controller_exact_forecast_baseline_blend_weights_accepts_signed_csv() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-baseline-blend-weights=-0.5,0,0.5",
        ]
    )
    assert str(args.checkpoint_controller_exact_forecast_baseline_blend_weights) == "-0.5,0,0.5"


def test_checkpoint_controller_exact_forecast_baseline_gain_scales_default_off() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.checkpoint_controller_exact_forecast_baseline_gain_scales) == ""


def test_checkpoint_controller_exact_forecast_baseline_gain_scales_accepts_csv() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-baseline-gain-scales",
            "1.0,1.1,1.25",
        ]
    )
    assert str(args.checkpoint_controller_exact_forecast_baseline_gain_scales) == "1.0,1.1,1.25"


def test_checkpoint_controller_exact_forecast_tangent_secant_flags_accept_values() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-include-tangent-secant-proposal",
            "--checkpoint-controller-exact-forecast-tangent-secant-trust-radius",
            "0.75",
            "--checkpoint-controller-exact-forecast-tangent-secant-signed-energy-lead-limit",
            "1.5",
        ]
    )
    assert bool(args.checkpoint_controller_exact_forecast_include_tangent_secant_proposal) is True
    assert float(args.checkpoint_controller_exact_forecast_tangent_secant_trust_radius) == pytest.approx(0.75)
    assert float(args.checkpoint_controller_exact_forecast_tangent_secant_signed_energy_lead_limit) == pytest.approx(1.5)


def test_checkpoint_controller_exact_forecast_horizon_defaults_to_single_step() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert int(args.checkpoint_controller_exact_forecast_horizon_steps) == 1
    assert str(args.checkpoint_controller_exact_forecast_horizon_weights) == ""


def test_checkpoint_controller_exact_forecast_horizon_accepts_steps_and_weights() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-horizon-steps",
            "3",
            "--checkpoint-controller-exact-forecast-horizon-weights",
            "3,2,1",
        ]
    )
    assert int(args.checkpoint_controller_exact_forecast_horizon_steps) == 3
    assert str(args.checkpoint_controller_exact_forecast_horizon_weights) == "3,2,1"


def test_checkpoint_controller_exact_forecast_energy_shape_weights_default_off() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert float(args.checkpoint_controller_exact_forecast_energy_slope_weight) == 0.0
    assert float(args.checkpoint_controller_exact_forecast_energy_excursion_under_weight) == 0.0
    assert float(args.checkpoint_controller_exact_forecast_energy_excursion_over_weight) == 0.0
    assert float(args.checkpoint_controller_exact_forecast_energy_excursion_rel_tolerance) == 0.0
    assert float(args.checkpoint_controller_exact_forecast_energy_curvature_weight) == 0.0


def test_checkpoint_controller_exact_forecast_energy_shape_weights_accept_values() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-energy-slope-weight",
            "500",
            "--checkpoint-controller-exact-forecast-energy-curvature-weight",
            "25",
        ]
    )
    assert float(args.checkpoint_controller_exact_forecast_energy_slope_weight) == 500.0
    assert float(args.checkpoint_controller_exact_forecast_energy_curvature_weight) == 25.0


def test_checkpoint_controller_exact_forecast_energy_excursion_under_weight_accepts_value() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-energy-excursion-under-weight",
            "300",
        ]
    )
    assert float(args.checkpoint_controller_exact_forecast_energy_excursion_under_weight) == 300.0


def test_checkpoint_controller_exact_forecast_energy_excursion_band_accepts_values() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--checkpoint-controller-exact-forecast-energy-excursion-over-weight",
            "180",
            "--checkpoint-controller-exact-forecast-energy-excursion-rel-tolerance",
            "0.05",
        ]
    )
    assert float(args.checkpoint_controller_exact_forecast_energy_excursion_over_weight) == 180.0
    assert float(args.checkpoint_controller_exact_forecast_energy_excursion_rel_tolerance) == 0.05


def test_spectral_target_surface_defaults_to_auto_density_report() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.spectral_target_observable) == "auto"
    assert str(args.spectral_target_pair) == ""
    assert str(args.spectral_detrend) == "constant"
    assert str(args.spectral_window) == "hann"
    assert int(args.spectral_max_harmonic) == 3


def test_spectral_target_surface_accepts_explicit_values() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--spectral-target-observable",
            "density_difference",
            "--spectral-target-pair",
            "0,1",
            "--spectral-detrend",
            "linear",
            "--spectral-window",
            "none",
            "--spectral-max-harmonic",
            "5",
        ]
    )
    assert str(args.spectral_target_observable) == "density_difference"
    assert str(args.spectral_target_pair) == "0,1"
    assert str(args.spectral_detrend) == "linear"
    assert str(args.spectral_window) == "none"
    assert int(args.spectral_max_harmonic) == 5


def test_checkpoint_controller_exact_forecast_guardrail_defaults_off() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.checkpoint_controller_exact_forecast_guardrail_mode) == "off"
    assert float(args.checkpoint_controller_exact_forecast_fidelity_loss_tol) == 0.0
    assert float(args.checkpoint_controller_exact_forecast_abs_energy_error_increase_tol) == 0.0


def test_checkpoint_controller_exact_forecast_guardrail_accepts_dual_metric() -> None:
    args = parse_args(
        [
            "--L", "2", "--skip-pdf",
            "--checkpoint-controller-exact-forecast-guardrail-mode", "dual_metric_v1",
            "--checkpoint-controller-exact-forecast-fidelity-loss-tol", "0.01",
            "--checkpoint-controller-exact-forecast-abs-energy-error-increase-tol", "0.02",
        ]
    )
    assert str(args.checkpoint_controller_exact_forecast_guardrail_mode) == "dual_metric_v1"
    assert float(args.checkpoint_controller_exact_forecast_fidelity_loss_tol) == 0.01
    assert float(args.checkpoint_controller_exact_forecast_abs_energy_error_increase_tol) == 0.02


def test_checkpoint_controller_confirm_score_mode_defaults_to_compressed_whitened() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.checkpoint_controller_confirm_score_mode) == "compressed_whitened_v1"


def test_checkpoint_controller_confirm_score_mode_accepts_exact_gain_ratio() -> None:
    args = parse_args(
        [
            "--L", "2", "--skip-pdf",
            "--checkpoint-controller-confirm-score-mode", "exact_gain_ratio",
            "--checkpoint-controller-confirm-compress-fraction", "0.25",
            "--checkpoint-controller-confirm-compress-min-modes", "1",
            "--checkpoint-controller-confirm-compress-max-modes", "3",
        ]
    )
    assert str(args.checkpoint_controller_confirm_score_mode) == "exact_gain_ratio"
    assert float(args.checkpoint_controller_confirm_compress_fraction) == 0.25
    assert int(args.checkpoint_controller_confirm_compress_min_modes) == 1
    assert int(args.checkpoint_controller_confirm_compress_max_modes) == 3


def test_checkpoint_controller_prune_mode_accepts_exact_local_v1() -> None:
    args = parse_args(
        [
            "--L", "2", "--skip-pdf",
            "--checkpoint-controller-prune-mode", "exact_local_v1",
            "--checkpoint-controller-prune-miss-threshold", "0.01",
            "--checkpoint-controller-prune-protection-steps", "3",
            "--checkpoint-controller-prune-stagnation-window", "4",
            "--checkpoint-controller-prune-stagnation-alpha", "0.25",
            "--checkpoint-controller-prune-stale-score-threshold", "0.8",
            "--checkpoint-controller-prune-loss-threshold", "0.03",
            "--checkpoint-controller-prune-max-candidates", "2",
            "--checkpoint-controller-prune-cooldown-steps", "5",
            "--checkpoint-controller-prune-safe-miss-increase-tol", "0.02",
            "--checkpoint-controller-prune-state-jump-l2-tol", "0.1",
            "--checkpoint-controller-prune-theta-block-tol", "0.05",
        ]
    )
    assert str(args.checkpoint_controller_prune_mode) == "exact_local_v1"
    assert float(args.checkpoint_controller_prune_miss_threshold) == 0.01
    assert int(args.checkpoint_controller_prune_protection_steps) == 3
    assert int(args.checkpoint_controller_prune_stagnation_window) == 4
    assert float(args.checkpoint_controller_prune_stagnation_alpha) == 0.25
    assert float(args.checkpoint_controller_prune_stale_score_threshold) == 0.8
    assert float(args.checkpoint_controller_prune_loss_threshold) == 0.03
    assert int(args.checkpoint_controller_prune_max_candidates) == 2
    assert int(args.checkpoint_controller_prune_cooldown_steps) == 5
    assert float(args.checkpoint_controller_prune_safe_miss_increase_tol) == 0.02
    assert float(args.checkpoint_controller_prune_state_jump_l2_tol) == 0.1
    assert float(args.checkpoint_controller_prune_theta_block_tol) == 0.05
