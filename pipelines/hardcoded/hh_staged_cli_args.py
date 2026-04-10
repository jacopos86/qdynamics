#!/usr/bin/env python3
"""Shared argparse helpers for staged HH workflow CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_staged_hh_base_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Physics
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=2.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=1.0, dest="g_ep")
    p.add_argument("--n-ph-max", type=int, default=1, dest="n_ph_max")
    p.add_argument("--boson-encoding", choices=["binary"], default="binary")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--sector-n-up", type=int, default=None)
    p.add_argument("--sector-n-dn", type=int, default=None)

    # Warm-start VQE
    p.add_argument("--warm-reps", type=int, default=None)
    p.add_argument("--warm-restarts", type=int, default=None)
    p.add_argument("--warm-maxiter", type=int, default=None)
    p.add_argument(
        "--warm-method",
        choices=["SPSA", "COBYLA", "SLSQP", "L-BFGS-B", "Powell", "Nelder-Mead"],
        default="SPSA",
    )
    p.add_argument("--warm-seed", type=int, default=7)
    p.add_argument("--warm-progress-every-s", type=float, default=60.0)

    # Shared VQE backend / SPSA knobs
    p.add_argument(
        "--vqe-energy-backend",
        choices=["legacy", "one_apply_compiled"],
        default="one_apply_compiled",
    )
    p.add_argument("--vqe-spsa-a", type=float, default=0.2)
    p.add_argument("--vqe-spsa-c", type=float, default=0.1)
    p.add_argument("--vqe-spsa-alpha", type=float, default=0.602)
    p.add_argument("--vqe-spsa-gamma", type=float, default=0.101)
    p.add_argument("--vqe-spsa-A", type=float, default=10.0)
    p.add_argument("--vqe-spsa-avg-last", type=int, default=0)
    p.add_argument("--vqe-spsa-eval-repeats", type=int, default=1)
    p.add_argument("--vqe-spsa-eval-agg", choices=["mean", "median"], default="mean")

    # ADAPT
    p.add_argument("--adapt-pool", type=str, default=None)
    p.add_argument(
        "--adapt-continuation-mode",
        choices=["legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        default="phase3_v1",
        help=(
            "Default staged continuation mode. The manuscript-aligned default is the full all-phase shortlist path phase3_v1; "
            "override explicitly to legacy/phase1_v1/phase2_v1 when needed."
        ),
    )
    p.add_argument("--adapt-max-depth", type=int, default=None)
    p.add_argument("--adapt-maxiter", type=int, default=None)
    p.add_argument("--adapt-eps-grad", type=float, default=None)
    p.add_argument("--adapt-eps-energy", type=float, default=None)
    p.add_argument("--adapt-seed", type=int, default=11)
    p.add_argument("--adapt-inner-optimizer", choices=["COBYLA", "POWELL", "SPSA"], default="SPSA")
    p.set_defaults(adapt_allow_repeats=True)
    p.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.set_defaults(adapt_finite_angle_fallback=True)
    p.add_argument(
        "--adapt-finite-angle-fallback",
        dest="adapt_finite_angle_fallback",
        action="store_true",
    )
    p.add_argument(
        "--adapt-no-finite-angle-fallback",
        dest="adapt_finite_angle_fallback",
        action="store_false",
    )
    p.add_argument("--adapt-finite-angle", type=float, default=0.1)
    p.add_argument("--adapt-finite-angle-min-improvement", type=float, default=1e-12)
    p.add_argument("--adapt-disable-hh-seed", action="store_true")
    p.add_argument(
        "--adapt-reopt-policy",
        choices=["append_only", "full", "windowed"],
        default="append_only",
    )
    p.add_argument("--adapt-window-size", type=int, default=3)
    p.add_argument("--adapt-window-topk", type=int, default=0)
    p.add_argument("--adapt-full-refit-every", type=int, default=0)
    p.set_defaults(adapt_final_full_refit=True)
    p.add_argument("--adapt-final-full-refit", dest="adapt_final_full_refit", action="store_true")
    p.add_argument("--adapt-no-final-full-refit", dest="adapt_final_full_refit", action="store_false")
    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-split-paulis", action="store_true")
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", choices=["none", "fro", "maxcoeff"], default="none")
    p.add_argument("--adapt-spsa-a", type=float, default=0.2)
    p.add_argument("--adapt-spsa-c", type=float, default=0.1)
    p.add_argument("--adapt-spsa-alpha", type=float, default=0.602)
    p.add_argument("--adapt-spsa-gamma", type=float, default=0.101)
    p.add_argument("--adapt-spsa-A", type=float, default=10.0)
    p.add_argument("--adapt-spsa-avg-last", type=int, default=0)
    p.add_argument("--adapt-spsa-eval-repeats", type=int, default=1)
    p.add_argument("--adapt-spsa-eval-agg", choices=["mean", "median"], default="mean")
    p.add_argument("--adapt-spsa-callback-every", type=int, default=5)
    p.add_argument("--adapt-spsa-progress-every-s", type=float, default=60.0)
    p.add_argument(
        "--adapt-analytic-noise-std",
        type=float,
        default=0.0,
        help="Std-dev of run-local Gaussian noise injected into exact ADAPT search-time energy/gradient evaluations (0 = disabled).",
    )
    p.add_argument(
        "--adapt-analytic-noise-seed",
        type=int,
        default=None,
        help="Optional RNG seed for run-local ADAPT analytic Gaussian noise draws.",
    )
    p.add_argument("--phase1-lambda-F", type=float, default=1.0)
    p.add_argument("--phase1-lambda-compile", type=float, default=0.05)
    p.add_argument("--phase1-lambda-measure", type=float, default=0.02)
    p.add_argument("--phase1-lambda-leak", type=float, default=0.0)
    p.add_argument("--phase1-score-z-alpha", type=float, default=0.0)
    p.add_argument("--phase1-probe-max-positions", type=int, default=6)
    p.add_argument("--phase1-plateau-patience", type=int, default=2)
    p.add_argument("--phase1-trough-margin-ratio", type=float, default=1.0)
    p.set_defaults(phase1_prune_enabled=True)
    p.add_argument("--phase1-prune-enabled", dest="phase1_prune_enabled", action="store_true")
    p.add_argument("--phase1-no-prune", dest="phase1_prune_enabled", action="store_false")
    p.add_argument("--phase1-prune-fraction", type=float, default=0.25)
    p.add_argument("--phase1-prune-max-candidates", type=int, default=6)
    p.add_argument("--phase1-prune-max-regression", type=float, default=1e-8)
    p.add_argument("--phase3-motif-source-json", type=Path, default=None)
    p.add_argument(
        "--phase3-symmetry-mitigation-mode",
        choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"],
        default="off",
    )
    p.set_defaults(phase3_enable_rescue=False)
    p.add_argument("--phase3-enable-rescue", dest="phase3_enable_rescue", action="store_true")
    p.add_argument("--phase3-no-rescue", dest="phase3_enable_rescue", action="store_false")
    p.add_argument("--phase3-lifetime-cost-mode", choices=["off", "phase3_v1"], default="phase3_v1")
    p.add_argument(
        "--phase3-runtime-split-mode",
        choices=["off"],
        default="off",
        help=(
            "Phase-3 runtime split mode for the staged public HH surface. "
            "The manuscript-facing canonical path keeps this fixed to 'off'; "
            "shortlist Pauli-child splitting is retained only as an internal archival/testing implementation."
        ),
    )
    p.add_argument(
        "--phase3-oracle-gradient-mode",
        choices=["off", "ideal", "shots", "aer_noise", "backend_scheduled", "runtime"],
        default="off",
        help=(
            "Opt-in direct HH phase3_v1 oracle-gradient scouting for the staged workflow. "
            "Inner re-optimization stays exact in v1."
        ),
    )
    p.add_argument("--phase3-oracle-shots", type=int, default=2048)
    p.add_argument("--phase3-oracle-repeats", type=int, default=1)
    p.add_argument("--phase3-oracle-aggregate", choices=["mean"], default="mean")
    p.add_argument("--phase3-oracle-backend-name", type=str, default=None)
    p.add_argument("--phase3-oracle-use-fake-backend", action="store_true")
    p.add_argument("--phase3-oracle-seed", type=int, default=7)
    p.add_argument(
        "--phase3-oracle-gradient-step",
        type=float,
        default=None,
        help="Finite-difference step for phase3 oracle scouting. Defaults to --adapt-finite-angle when omitted.",
    )
    p.add_argument(
        "--phase3-oracle-mitigation",
        choices=["none", "readout"],
        default="none",
    )
    p.add_argument(
        "--phase3-oracle-local-readout-strategy",
        choices=["mthree"],
        default=None,
    )
    p.add_argument(
        "--phase3-oracle-execution-surface",
        choices=["auto", "expectation_v1", "raw_measurement_v1"],
        default="auto",
        help=(
            "Execution surface for staged phase3 oracle scouting. "
            "'auto' selects raw-shot only for runtime with mitigation=none."
        ),
    )
    p.add_argument(
        "--phase3-oracle-raw-transport",
        choices=["auto", "sampler_v2"],
        default="auto",
        help=(
            "Raw transport preference when phase3 oracle execution resolves to raw_measurement_v1 "
            "on the runtime sampler path."
        ),
    )
    p.add_argument("--phase3-oracle-raw-store-memory", action="store_true")
    p.add_argument("--phase3-oracle-raw-artifact-path", type=str, default=None)
    p.add_argument("--phase3-oracle-seed-transpiler", type=int, default=None)
    p.add_argument("--phase3-oracle-transpile-optimization-level", type=int, default=1)

    # Spectra-facing optimization report surface
    p.add_argument(
        "--spectral-target-observable",
        choices=["auto", "density_difference", "staggered"],
        default="auto",
        help=(
            "Primary spectra-facing observable to summarize in staged optimization reports. "
            "'auto' uses d(t)=n0-n1 for L=2 and staggered density otherwise."
        ),
    )
    p.add_argument(
        "--spectral-target-pair",
        type=str,
        default="",
        help=(
            "Optional site pair i,j for density-difference reporting. "
            "Leave blank to use the auto/default pair for the chosen observable."
        ),
    )
    p.add_argument(
        "--spectral-detrend",
        choices=["constant", "linear"],
        default="constant",
        help="Detrending mode for the spectra-facing optimization summary.",
    )
    p.add_argument(
        "--spectral-window",
        choices=["hann", "none"],
        default="hann",
        help="Window used for drive-line and harmonic fits in the spectra-facing optimization summary.",
    )
    p.add_argument(
        "--spectral-max-harmonic",
        type=int,
        default=3,
        help="Maximum drive harmonic reported in the spectra-facing optimization summary.",
    )

    # Final matched-family replay
    p.add_argument("--final-reps", type=int, default=None)
    p.add_argument("--final-restarts", type=int, default=None)
    p.add_argument("--final-maxiter", type=int, default=None)
    p.add_argument(
        "--final-method",
        choices=["SPSA", "COBYLA", "SLSQP", "L-BFGS-B", "Powell", "Nelder-Mead"],
        default="SPSA",
    )
    p.add_argument("--final-seed", type=int, default=19)
    p.add_argument("--final-progress-every-s", type=float, default=60.0)
    p.add_argument("--legacy-paop-key", type=str, default="paop_lf_std")
    p.add_argument(
        "--replay-seed-policy",
        choices=["auto", "scaffold_plus_zero", "residual_only", "tile_adapt"],
        default="auto",
    )
    p.add_argument(
        "--replay-continuation-mode",
        choices=["auto", "legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        default="auto",
        help=(
            "Replay continuation mode. 'auto' inherits the staged ADAPT continuation mode, which now defaults to the manuscript-aligned all-phase path."
        ),
    )
    p.add_argument("--replay-wallclock-cap-s", type=int, default=43200)
    p.add_argument("--replay-freeze-fraction", type=float, default=0.2)
    p.add_argument("--replay-unfreeze-fraction", type=float, default=0.3)
    p.add_argument("--replay-full-fraction", type=float, default=0.5)
    p.add_argument("--replay-qn-spsa-refresh-every", type=int, default=5)
    p.add_argument("--replay-qn-spsa-refresh-mode", choices=["diag_rms_grad"], default="diag_rms_grad")

    # Post-replay adaptive realtime checkpoint controller (exact/noiseless v1)
    p.add_argument(
        "--checkpoint-controller-mode",
        choices=["off", "exact_v1", "oracle_v1"],
        default="off",
        help=(
            "Opt-in post-replay HH adaptive realtime checkpoint controller. "
            "exact_v1 is exact/noiseless, including the driven fixed-scaffold path; "
            "oracle_v1 is the fresh-stage noisy/oracle scoring slice."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-oracle-selection-policy",
        choices=["measured_gain_commit_veto", "measured_topk_oracle_energy"],
        default="measured_gain_commit_veto",
        help=(
            "Selection policy inside oracle_v1 after measured geometry succeeds. "
            "Default preserves measured-gain selection plus noisy commit veto; "
            "measured_topk_oracle_energy reranks the top measured candidates by noisy horizon-1 energy."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-candidate-step-scales",
        type=str,
        default="1.0",
        help=(
            "Comma-separated rollout scales for controller exact/oracle forecasts. "
            "In exact_v1 these also seed the driven stay-step line search; "
            "use values below 1.0 to test damped steps before accepting full motion."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-step-refine-rounds",
        type=int,
        default=0,
        help=(
            "Optional midpoint-refinement rounds around the best exact_v1 stay-step scale. "
            "0 keeps the coarse seed ladder only; 1-3 progressively densify the local step search."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-proposal-mode",
        choices=["norm_locked_blend_v1", "anticipatory_drive_basis_v1"],
        default="norm_locked_blend_v1",
        help=(
            "exact_v1 stay-direction proposal family. "
            "norm_locked_blend_v1 preserves the current norm-locked residual-blend search; "
            "anticipatory_drive_basis_v1 adds independently normalized drive/look-ahead proposals."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-blend-weights",
        type=str,
        default="",
        help=(
            "Optional comma-separated exact_v1 stay-direction residual-blend weights. "
            "0 keeps the pure McLachlan baseline, positive values add the restricted drive-only residual direction, "
            "and negative values subtract that residual direction. Values must lie in [-1, 1]."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-gain-scales",
        type=str,
        default="",
        help=(
            "Optional comma-separated exact_v1 stay-direction gain scales applied after blend selection. "
            "Use values slightly above 1.0 to let the controller raise the chosen blended motion amplitude "
            "without changing the blend ladder itself. Leave blank for the default gain 1.0 only."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-include-tangent-secant-proposal",
        action="store_true",
        help=(
            "Augment exact_v1 stay proposals with a projected exact one-step tangent-secant direction. "
            "This proposal is added alongside the existing blend/anticipatory families rather than replacing them."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-tangent-secant-trust-radius",
        type=float,
        default=0.0,
        help=(
            "Optional independent metric-norm clip for the exact_v1 tangent-secant stay proposal. "
            "0 disables clipping; positive values cap the secant proposal without tying it to the baseline norm."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-tangent-secant-signed-energy-lead-limit",
        type=float,
        default=0.0,
        help=(
            "Optional soft gate for the exact_v1 tangent-secant stay proposal. "
            "Positive values taper secant startup motion once the controller is already leading the next exact "
            "energy excursion in the same signed direction by more than this multiple of |ΔE_exact,next|. "
            "0 disables the taper."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-horizon-steps",
        type=int,
        default=1,
        help=(
            "Number of exact-v1 future checkpoints scored when choosing stay/candidate step scales. "
            "Use 1 for current next-step behavior, or 2-3 to pay extra attention to upcoming bends."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-horizon-weights",
        type=str,
        default="",
        help=(
            "Optional comma-separated weights for the exact-v1 horizon score, ordered nearest-to-farthest future step. "
            "Leave blank for uniform weights across the active horizon."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-energy-slope-weight",
        type=float,
        default=0.0,
        help=(
            "Optional exact-v1 horizon shape term for energy-slope mismatch. "
            "This only matters when the horizon has at least 2 future checkpoints."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-energy-curvature-weight",
        type=float,
        default=0.0,
        help=(
            "Optional exact-v1 horizon shape term for energy-curvature mismatch. "
            "This matters once the scorer has enough points to form a bend, including the h=2 path with the current-point anchor."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-energy-excursion-under-weight",
        type=float,
        default=0.0,
        help=(
            "Optional exact-v1 horizon amplitude term that penalizes under-response in signed energy excursion "
            "relative to the current-point anchor. Use this to lift a shape-matched controller trace that stays too low."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-energy-excursion-over-weight",
        type=float,
        default=0.0,
        help=(
            "Optional exact-v1 horizon amplitude term that penalizes overshoot in signed energy excursion "
            "relative to the current-point anchor. Use this with the under-weight to target near-exact amplitude "
            "instead of always biasing upward from below."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-energy-excursion-rel-tolerance",
        type=float,
        default=0.0,
        help=(
            "Optional relative deadband around the exact signed energy excursion before the exact-v1 under/over "
            "excursion penalties activate. For example, 0.05 allows about +/-5 percent excursion mismatch before scoring it."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-guardrail-mode",
        choices=["off", "dual_metric_v1"],
        default="off",
        help=(
            "Optional oracle_v1 calibration guardrail that uses the controller's exact next-step reference "
            "to veto proposed appends without adding measurement work."
        ),
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-fidelity-loss-tol",
        type=float,
        default=0.0,
        help="Forecast veto tolerance on next-step fidelity loss relative to stay.",
    )
    p.add_argument(
        "--checkpoint-controller-exact-forecast-abs-energy-error-increase-tol",
        type=float,
        default=0.0,
        help="Forecast veto tolerance on next-step absolute total-energy-error increase relative to stay.",
    )
    p.add_argument(
        "--checkpoint-controller-confirm-score-mode",
        choices=["exact_gain_ratio", "compressed_whitened_v1"],
        default="compressed_whitened_v1",
        help="Confirm-stage ranking surface. compressed_whitened_v1 follows the manuscript's whitened compressed Schur ladder while exact thresholds still gate commit.",
    )
    p.add_argument("--checkpoint-controller-confirm-compress-fraction", type=float, default=0.5)
    p.add_argument("--checkpoint-controller-confirm-compress-min-modes", type=int, default=1)
    p.add_argument("--checkpoint-controller-confirm-compress-max-modes", type=int, default=8)
    p.add_argument(
        "--checkpoint-controller-prune-mode",
        choices=["off", "exact_local_v1"],
        default="off",
        help="Optional manuscript-style prune lane using cached frozen-ablation ranking plus exact local accept/reject.",
    )
    p.add_argument("--checkpoint-controller-prune-miss-threshold", type=float, default=0.02)
    p.add_argument("--checkpoint-controller-prune-protection-steps", type=int, default=2)
    p.add_argument("--checkpoint-controller-prune-stagnation-window", type=int, default=3)
    p.add_argument("--checkpoint-controller-prune-stagnation-alpha", type=float, default=0.5)
    p.add_argument("--checkpoint-controller-prune-stale-score-threshold", type=float, default=0.75)
    p.add_argument("--checkpoint-controller-prune-loss-threshold", type=float, default=0.01)
    p.add_argument("--checkpoint-controller-prune-max-candidates", type=int, default=2)
    p.add_argument("--checkpoint-controller-prune-cooldown-steps", type=int, default=2)
    p.add_argument("--checkpoint-controller-prune-safe-miss-increase-tol", type=float, default=0.01)
    p.add_argument("--checkpoint-controller-prune-state-jump-l2-tol", type=float, default=0.05)
    p.add_argument("--checkpoint-controller-prune-theta-block-tol", type=float, default=0.05)
    p.add_argument("--checkpoint-controller-miss-threshold", type=float, default=0.05)
    p.add_argument("--checkpoint-controller-gain-ratio-threshold", type=float, default=0.02)
    p.add_argument("--checkpoint-controller-append-margin-abs", type=float, default=1e-6)
    p.add_argument("--checkpoint-controller-shortlist-size", type=int, default=4)
    p.add_argument("--checkpoint-controller-shortlist-fraction", type=float, default=0.15)
    p.add_argument("--checkpoint-controller-active-window-size", type=int, default=3)
    p.add_argument("--checkpoint-controller-max-probe-positions", type=int, default=4)
    p.add_argument("--checkpoint-controller-regularization-lambda", type=float, default=1e-8)
    p.add_argument("--checkpoint-controller-candidate-regularization-lambda", type=float, default=1e-8)
    p.add_argument("--checkpoint-controller-pinv-rcond", type=float, default=1e-10)
    p.add_argument("--checkpoint-controller-compile-penalty-weight", type=float, default=0.05)
    p.add_argument("--checkpoint-controller-measurement-penalty-weight", type=float, default=0.02)
    p.add_argument("--checkpoint-controller-directional-penalty-weight", type=float, default=0.01)
    p.add_argument("--checkpoint-controller-motion-calm-direction-cosine-threshold", type=float, default=0.98)
    p.add_argument("--checkpoint-controller-motion-calm-rate-change-ratio-threshold", type=float, default=0.15)
    p.add_argument("--checkpoint-controller-motion-direction-reversal-cosine-threshold", type=float, default=-0.05)
    p.add_argument("--checkpoint-controller-motion-curvature-flip-cosine-threshold", type=float, default=-0.10)
    p.add_argument("--checkpoint-controller-motion-acceleration-l2-threshold", type=float, default=0.05)
    p.add_argument("--checkpoint-controller-motion-kink-rate-change-ratio-threshold", type=float, default=0.50)
    p.add_argument("--checkpoint-controller-motion-calm-shortlist-scale", type=float, default=0.5)
    p.add_argument("--checkpoint-controller-motion-kink-shortlist-bonus", type=int, default=2)
    p.add_argument("--checkpoint-controller-motion-calm-oracle-budget-scale", type=float, default=0.5)
    p.add_argument("--checkpoint-controller-motion-kink-oracle-budget-scale", type=float, default=2.0)
    p.add_argument("--checkpoint-controller-position-jump-tie-margin-abs", type=float, default=1e-6)
    p.add_argument("--checkpoint-controller-reconstruction-tol", type=float, default=1e-10)
    p.add_argument(
        "--checkpoint-controller-grouping-mode",
        choices=["qwc_basis_cover_reuse"],
        default="qwc_basis_cover_reuse",
    )
    p.add_argument("--checkpoint-controller-scout-shots", type=int, default=None)
    p.add_argument("--checkpoint-controller-scout-repeats", type=int, default=None)
    p.add_argument("--checkpoint-controller-confirm-shots", type=int, default=None)
    p.add_argument("--checkpoint-controller-confirm-repeats", type=int, default=None)
    p.add_argument("--checkpoint-controller-commit-shots", type=int, default=None)
    p.add_argument("--checkpoint-controller-commit-repeats", type=int, default=None)
    p.add_argument(
        "--checkpoint-controller-analytic-noise-std",
        type=float,
        default=0.0,
        help="Coarse native-units std-dev of run-local Gaussian noise injected into exact controller geometry G and f (0 = disabled).",
    )
    p.add_argument(
        "--checkpoint-controller-analytic-noise-seed",
        type=int,
        default=None,
        help="Optional RNG seed for run-local checkpoint-controller analytic Gaussian noise draws.",
    )

    # Dynamics
    p.add_argument("--noiseless-methods", type=str, default="suzuki2,cfqm4")
    p.add_argument("--t-final", type=float, default=None)
    p.add_argument("--num-times", type=int, default=None)
    p.add_argument("--trotter-steps", type=int, default=None)
    p.add_argument("--exact-steps-multiplier", type=int, default=None)
    p.add_argument("--fidelity-subspace-energy-tol", type=float, default=1e-9)
    p.add_argument(
        "--cfqm-stage-exp",
        choices=["expm_multiply_sparse", "dense_expm", "pauli_suzuki2"],
        default="expm_multiply_sparse",
    )
    p.add_argument("--cfqm-coeff-drop-abs-tol", type=float, default=0.0)
    p.add_argument("--cfqm-normalize", action="store_true")

    # Drive remains opt-in for this wrapper.
    p.add_argument("--enable-drive", action="store_true")
    p.add_argument("--drive-A", type=float, default=0.6)
    p.add_argument("--drive-omega", type=float, default=2.0)
    p.add_argument("--drive-tbar", type=float, default=2.5)
    p.add_argument("--drive-phi", type=float, default=0.0)
    p.add_argument("--drive-pattern", choices=["staggered", "dimer_bias", "custom"], default="staggered")
    p.add_argument("--drive-custom-s", type=str, default=None)
    p.add_argument("--drive-include-identity", action="store_true")
    p.add_argument("--drive-time-sampling", choices=["midpoint", "left", "right"], default="midpoint")
    p.add_argument("--drive-t0", type=float, default=0.0)

    # Gates and artifacts
    p.add_argument("--ecut-1", type=float, default=None, help="Diagnostic warm->ADAPT handoff gate; recorded, not hard-fail.")
    p.add_argument("--ecut-2", type=float, default=None, help="Diagnostic final replay gate; recorded, not hard-fail.")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--skip-pdf", action="store_true")
    p.add_argument(
        "--smoke-test-intentionally-weak",
        action="store_true",
        help="# SMOKE TEST - intentionally weak settings",
    )
    return p


def add_staged_hh_noise_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--noise-modes", type=str, default="ideal,shots,aer_noise")
    p.add_argument(
        "--audit-noise-modes",
        type=str,
        default=None,
        help="Audit-only noise modes for imported prepared-state/full-circuit audits; defaults to ideal,backend_scheduled in import mode.",
    )
    p.add_argument("--noisy-methods", type=str, default="cfqm4,suzuki2")
    p.add_argument("--benchmark-active-coeff-tol", type=float, default=1e-12)
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--oracle-repeats", type=int, default=4)
    p.add_argument("--oracle-aggregate", choices=["mean", "median"], default="mean")
    p.add_argument("--mitigation", choices=["none", "readout", "zne", "dd"], default="none")
    p.add_argument(
        "--local-readout-strategy",
        choices=["mthree"],
        default="mthree",
        help="Local readout mitigation backend for compiled fake-backend audits.",
    )
    p.set_defaults(local_gate_twirling=False)
    p.add_argument(
        "--local-gate-twirling",
        dest="local_gate_twirling",
        action="store_true",
        help=(
            "Opt-in local 2Q-only Pauli twirling for compiled fake-backend backend_scheduled audits. "
            "This is a local heuristic analogue of Runtime gate twirling, not a full TREX implementation."
        ),
    )
    p.add_argument(
        "--no-local-gate-twirling",
        dest="local_gate_twirling",
        action="store_false",
    )
    p.add_argument(
        "--symmetry-mitigation-mode",
        choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"],
        default="off",
    )
    p.add_argument("--zne-scales", type=str, default=None)
    p.add_argument(
        "--dd-sequence",
        type=str,
        default=None,
        help=(
            "DD sequence selector. Runtime routes use this for Runtime DD where supported. "
            "On local fixed-scaffold noisy replay, only XpXm is supported and it is treated as an additive "
            "saved-theta DD probe rather than optimizer-loop mitigation."
        ),
    )
    p.add_argument("--noise-seed", type=int, default=7)
    p.add_argument(
        "--backend-name",
        type=str,
        default=None,
        help=(
            "Fake/runtime backend name. Imported lean routes default to FakeGuadalupeV2; "
            "fixed-scaffold local noisy replay defaults to FakeMarrakesh; fixed-scaffold Nighthawk compile-control "
            "and attribution routes still default to FakeNighthawk when omitted; fixed-scaffold compile-control scouts "
            "now stay pinned to an explicit local fake backend so requested-vs-observed transpile settings stay auditable."
        ),
    )
    p.add_argument("--use-fake-backend", action="store_true")
    p.set_defaults(allow_aer_fallback=True)
    p.add_argument("--allow-aer-fallback", dest="allow_aer_fallback", action="store_true")
    p.add_argument("--no-allow-aer-fallback", dest="allow_aer_fallback", action="store_false")
    p.set_defaults(omp_shm_workaround=True)
    p.add_argument("--omp-shm-workaround", dest="omp_shm_workaround", action="store_true")
    p.add_argument("--no-omp-shm-workaround", dest="omp_shm_workaround", action="store_false")
    p.add_argument("--noisy-mode-timeout-s", type=int, default=1200)
    p.add_argument("--checkpoint-controller-timeout-s", type=int, default=None)
    p.add_argument("--checkpoint-controller-progress-every-s", type=float, default=5.0)
    p.add_argument(
        "--checkpoint-controller-noise-mode",
        choices=["inherit", "ideal", "shots", "aer_noise", "runtime", "backend_scheduled"],
        default="inherit",
        help=(
            "Noise/oracle execution mode for --checkpoint-controller-mode oracle_v1. "
            "inherit uses the single configured noise mode, or backend_scheduled for local fake-backend runs."
        ),
    )
    p.add_argument("--include-final-audit", action="store_true")
    p.add_argument(
        "--include-full-circuit-audit",
        action="store_true",
        help=(
            "Enable the deeper imported-circuit audit lane: ansatz-input-state audit plus full imported-circuit audit. "
            "If no import JSON is given, defaults to the latest lean pareto_lean_l2 rerun artifact."
        ),
    )
    p.add_argument(
        "--include-fixed-lean-noisy-replay",
        action="store_true",
        help=(
            "Enable imported fixed-lean conventional replay on the local fake-backend noisy path. "
            "Locks the imported lean pareto_lean_l2 scaffold, forces reps=1 for this route only, "
            "and optimizes only continuous parameters under backend_scheduled + readout/mthree."
        ),
    )
    p.add_argument(
        "--include-fixed-lean-noise-attribution",
        action="store_true",
        help=(
            "Enable imported fixed-lean gate-vs-readout attribution on the local fake-backend path. "
            "Runs the same locked imported pareto_lean_l2 circuit under readout_only, "
            "gate_stateprep_only, and full backend_scheduled slices with one shared transpile; "
            "raw diagnostics only (mitigation none, symmetry off)."
        ),
    )
    p.add_argument(
        "--include-fixed-lean-compile-control-scout",
        action="store_true",
        help=(
            "Enable imported fixed-lean energy-only local compile-control scouting. "
            "Keeps the locked pareto_lean_l2 circuit fixed, evaluates only the Hamiltonian energy, "
            "and compares a small transpile seed / optimization-level grid under backend_scheduled + readout/mthree."
        ),
    )
    p.add_argument(
        "--include-fixed-scaffold-noisy-replay",
        action="store_true",
        help=(
            "Enable imported fixed-scaffold conventional replay on the pinned local fake-backend noisy path. "
            "Defaults to the exported Marrakesh/Heron gate-pruned 6-term scaffold when no import JSON is given, "
            "initializes from the saved imported theta_runtime, runs SPSA/Powell under backend_scheduled + "
            "readout/mthree, and treats --dd-sequence XpXm as an additive saved-theta DD probe rather than "
            "an optimizer-loop mitigation mode."
        ),
    )
    p.add_argument(
        "--include-fixed-scaffold-compile-control-scout",
        action="store_true",
        help=(
            "Enable imported fixed-scaffold energy-only local compile-control scouting. "
            "Defaults to the exported Marrakesh/Heron gate-pruned 6-term runtime candidate when no import JSON is given, "
            "keeps the locked imported circuit fixed, evaluates only the Hamiltonian energy, and compares a small "
            "transpile seed / optimization-level grid under backend_scheduled + readout/mthree while recording "
            "requested-vs-observed compile metadata for each candidate. "
            "Requires an explicit --backend-name so the scout stays pinned to the intended local fake backend."
        ),
    )
    p.add_argument(
        "--include-fixed-scaffold-saved-theta-mitigation-matrix",
        action="store_true",
        help=(
            "Enable imported fixed-scaffold fixed-theta local mitigation-matrix evaluation on an imported "
            "locked fixed scaffold. Keeps the imported theta_runtime fixed, stays local-only on a fake "
            "backend, defaults to readout+mthree as the base, and evaluates the compile-preset x ZNE x "
            "suppression-stack matrix without running optimizer-loop replay."
        ),
    )
    p.add_argument(
        "--fixed-scaffold-matrix-zne-scales",
        type=str,
        default="1.0,3.0,5.0",
        help=(
            "Noise-scale factors for the imported fixed-scaffold saved-theta mitigation matrix local ZNE helper. "
            "Used only with --include-fixed-scaffold-saved-theta-mitigation-matrix."
        ),
    )
    p.add_argument(
        "--fixed-scaffold-matrix-compile-presets",
        type=str,
        default=None,
        help=(
            "Optional comma-separated compile-preset override for the imported fixed-scaffold saved-theta "
            "mitigation matrix. Format: label:optimization_level:seed_transpiler."
        ),
    )
    p.add_argument(
        "--fixed-scaffold-matrix-selected-cells",
        type=str,
        default=None,
        help=(
            "Optional comma-separated explicit cell whitelist for the imported fixed-scaffold saved-theta "
            "mitigation matrix. Labels use the form preset__zne_on|off__twirl|dd|twirl_dd."
        ),
    )
    p.add_argument(
        "--fixed-scaffold-matrix-base-mitigation-mode",
        choices=["readout", "none"],
        default="readout",
        help=(
            "Base mitigation mode for the imported fixed-scaffold saved-theta mitigation matrix. "
            "readout keeps the current readout+mthree default; none disables readout so on/off ablations "
            "can reuse the same fixed ZNE/twirl/DD lanes."
        ),
    )
    p.add_argument(
        "--include-fixed-scaffold-noise-attribution",
        action="store_true",
        help=(
            "Enable imported fixed-scaffold gate-vs-readout attribution on the local fake-backend path. "
            "Defaults to the exported Nighthawk gate-pruned 7-term scaffold when no import JSON is given, "
            "runs the same locked imported circuit under readout_only, gate_stateprep_only, and full "
            "backend_scheduled slices with one shared transpile; raw diagnostics only (mitigation none, symmetry off)."
        ),
    )
    p.add_argument(
        "--include-fixed-scaffold-runtime-energy-only-baseline",
        action="store_true",
        help=(
            "Enable imported fixed-scaffold Runtime energy-only baseline on the real Runtime path. "
            "Defaults to the Marrakesh/Heron gate-pruned 6-term runtime candidate when no import JSON is given, "
            "uses EstimatorV2 with explicit runtime profiles plus required session batching, and writes a standalone "
            "energy-only runtime sidecar for downstream follow-up tooling."
        ),
    )
    p.add_argument(
        "--include-fixed-scaffold-runtime-raw-baseline",
        action="store_true",
        help=(
            "Enable imported fixed-scaffold raw-shot baseline. Defaults to the Marrakesh/Heron gate-pruned 6-term "
            "runtime candidate when no import JSON is given. On the real Runtime path it uses the sampler-backed raw "
            "measurement surface with suppression-only sampler profiles; on the local fake-backend path it keeps "
            "acquisition mitigation/symmetry off and supports offline diagonal-only readout-then-symmetry "
            "postprocessing from the all-Z diagnostic sidecar."
        ),
    )
    p.add_argument(
        "--fixed-scaffold-runtime-raw-transport",
        choices=["auto", "sampler_v2"],
        default="auto",
        help=(
            "Raw transport preference for the imported fixed-scaffold raw-shot baseline. 'auto' keeps the public "
            "contract portable: sampler_v2 on the real Runtime path and the local backend-run style path on fake backends."
        ),
    )
    p.add_argument(
        "--fixed-scaffold-runtime-raw-store-memory",
        action="store_true",
        help="Keep fixed-scaffold Runtime raw-shot measurement records in memory during reduction.",
    )
    p.add_argument(
        "--fixed-scaffold-runtime-raw-artifact-path",
        type=str,
        default=None,
        help="Optional NDJSON(.gz) path for imported fixed-scaffold Runtime raw-shot records.",
    )
    p.add_argument(
        "--fixed-scaffold-runtime-raw-profile",
        choices=[
            "legacy_runtime_v0",
            "raw_sampler_twirled_v1",
            "raw_sampler_dd_probe_v1",
        ],
        default="legacy_runtime_v0",
        help=(
            "Sampler-safe Runtime profile for the imported fixed-scaffold raw-shot baseline. "
            "These profiles are suppression-only (twirling/DD only): readout mitigation and ZNE remain "
            "Estimator-side follow-up audits, not Sampler acquisition features."
        ),
    )
    p.add_argument(
        "--fixed-scaffold-runtime-profile",
        choices=[
            "legacy_runtime_v0",
            "main_twirled_readout_v1",
            "dd_probe_twirled_readout_v1",
            "final_audit_zne_twirled_readout_v1",
        ],
        default="main_twirled_readout_v1",
        help="Explicit Runtime Estimator profile for imported fixed-scaffold real-runtime routes.",
    )
    p.add_argument(
        "--fixed-scaffold-runtime-session-policy",
        choices=["prefer_session", "require_session", "backend_only"],
        default="require_session",
        help="Runtime batching policy for imported fixed-scaffold real-runtime routes.",
    )
    p.add_argument(
        "--fixed-scaffold-runtime-transpile-optimization-level",
        type=int,
        default=1,
        help=(
            "Transpile optimization level for imported fixed-scaffold runtime/raw-baseline routes. "
            "Also seeds the fixed-scaffold compile-control scout baseline and imported full-circuit audit compile request metadata."
        ),
    )
    p.add_argument(
        "--fixed-scaffold-runtime-seed-transpiler",
        type=int,
        default=0,
        help=(
            "Transpile seed for imported fixed-scaffold runtime/raw-baseline routes. "
            "Also seeds the fixed-scaffold compile-control scout baseline and imported full-circuit audit compile request metadata."
        ),
    )
    p.add_argument(
        "--include-fixed-scaffold-runtime-dd-probe",
        action="store_true",
        help="Add a DD probe phase on the best fixed-scaffold Runtime point using the dd_probe_twirled_readout_v1 profile.",
    )
    p.add_argument(
        "--include-fixed-scaffold-runtime-final-zne-audit",
        action="store_true",
        help="Add a final ZNE audit phase on the best fixed-scaffold Runtime point using the final_audit_zne_twirled_readout_v1 profile.",
    )
    p.add_argument(
        "--fixed-final-state-json",
        type=Path,
        default=None,
        help=(
            "Import source for prepared-state/full-circuit audits, fixed-lean noisy replay, "
            "fixed-lean noise attribution, fixed-lean compile-control scout, fixed-scaffold noisy replay, "
            "fixed-scaffold compile-control scout, and fixed-scaffold noise attribution. "
            "May be a direct ADAPT artifact, handoff bundle, or staged workflow payload."
        ),
    )
    return p


def build_staged_hh_parser(*, description: str, include_noise: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_staged_hh_base_args(parser)
    if include_noise:
        add_staged_hh_noise_args(parser)
    return parser
