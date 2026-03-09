#!/usr/bin/env python3
"""Thin CLI wrapper for the staged HH noiseless workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_staged_workflow import (
    resolve_staged_hh_config,
    run_staged_hh_noiseless,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Staged HH noiseless workflow: HF -> hh_hva_ptw warm-start -> staged ADAPT -> "
            "matched-family conventional replay -> Suzuki/CFQM vs exact."
        )
    )

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
        default="phase1_v1",
    )
    p.add_argument("--adapt-max-depth", type=int, default=None)
    p.add_argument("--adapt-maxiter", type=int, default=None)
    p.add_argument("--adapt-eps-grad", type=float, default=None)
    p.add_argument("--adapt-eps-energy", type=float, default=None)
    p.add_argument("--adapt-seed", type=int, default=11)
    p.add_argument("--adapt-inner-optimizer", choices=["COBYLA", "SPSA"], default="SPSA")
    p.set_defaults(adapt_allow_repeats=True)
    p.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.set_defaults(adapt_finite_angle_fallback=True)
    p.add_argument("--adapt-finite-angle-fallback", dest="adapt_finite_angle_fallback", action="store_true")
    p.add_argument("--adapt-no-finite-angle-fallback", dest="adapt_finite_angle_fallback", action="store_false")
    p.add_argument("--adapt-finite-angle", type=float, default=0.1)
    p.add_argument("--adapt-finite-angle-min-improvement", type=float, default=1e-12)
    p.add_argument("--adapt-disable-hh-seed", action="store_true")
    p.add_argument("--adapt-reopt-policy", choices=["append_only", "full", "windowed"], default="append_only")
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
        choices=["off", "shortlist_pauli_children_v1"],
        default="off",
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
    )
    p.add_argument("--replay-wallclock-cap-s", type=int, default=43200)
    p.add_argument("--replay-freeze-fraction", type=float, default=0.2)
    p.add_argument("--replay-unfreeze-fraction", type=float, default=0.3)
    p.add_argument("--replay-full-fraction", type=float, default=0.5)
    p.add_argument("--replay-qn-spsa-refresh-every", type=int, default=5)
    p.add_argument("--replay-qn-spsa-refresh-mode", choices=["diag_rms_grad"], default="diag_rms_grad")

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
    p.add_argument("--smoke-test-intentionally-weak", action="store_true", help="# SMOKE TEST - intentionally weak settings")

    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = resolve_staged_hh_config(args)
    payload = run_staged_hh_noiseless(cfg)
    print(f"workflow_json={payload['artifacts']['workflow']['output_json']}")
    if not bool(cfg.artifacts.skip_pdf):
        print(f"workflow_pdf={payload['artifacts']['workflow']['output_pdf']}")
    print(f"adapt_handoff_json={payload['artifacts']['intermediate']['adapt_handoff_json']}")
    print(f"replay_json={payload['artifacts']['intermediate']['replay_output_json']}")


if __name__ == "__main__":
    main()
