#!/usr/bin/env python3
"""Canonical Paper-I Hubbard-Holstein SNAKE/Optuna speed-path launcher.

This wrapper exists to keep the long HH Optuna command reproducible. It does
not implement a new optimizer or pool path; it injects the Paper-I HH speed-path
flags and delegates to ``pipelines.exact_bench.hh_cost_energy_optuna``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench import hh_cost_energy_optuna  # noqa: E402

DEFAULT_EXACT_MANIFEST = REPO_ROOT / "MATH/paper_facing/paper_I_static_scaffold/hh_ed_reference_manifest_20260614.json"
DEFAULT_GEO_TARGET_MANIFEST = REPO_ROOT / "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_geo_targets_20260615.json"
DEFAULT_GEO_GRAPH_PROXY_TARGET_MANIFEST = REPO_ROOT / "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_geo_graph_proxy_targets_20260617.json"
DEFAULT_SNAKE_GRAPH_PROXY_TARGET_MANIFEST = REPO_ROOT / "output/pdf/paper_i_hh_snake_graph_proxy_targets_from_overlay_review_20260616.json"
DEFAULT_CANDIDATE_PRIOR_MANIFEST = REPO_ROOT / "output/pdf/paper_i_hh_shot_focus_candidate_priors_from_overlay_20260616.json"
DEFAULT_CLASS_FILTER = REPO_ROOT / "agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "raw_outputs/local_smokes"
DEFAULT_PARALLEL_EVALUATIONS = max(1, min(8, os.cpu_count() or 1))
FULL_POLICY_OUTPUT_SLUG = "paper_i_hh_snake_fullpolicy_20260622_v1"
FULL_POLICY_PROFILE = hh_cost_energy_optuna._HH_ROUTEA_FULL_POLICY_PROFILE

_REGIMES: dict[str, dict[str, Any]] = {
    "weak-weak": {"u": 0.25, "lambda": 0.25, "n_ph_work": 2, "n_ph_ref": 2},
    "intermediate-weak": {"u": 1.25, "lambda": 0.25, "n_ph_work": 2, "n_ph_ref": 2},
    "weak-strong": {"u": 0.25, "lambda": 1.25, "n_ph_work": 4, "n_ph_ref": 4},
    "intermediate-strong": {"u": 1.25, "lambda": 1.25, "n_ph_work": 4, "n_ph_ref": 4},
    "strong-weak-u8": {"u": 8.0, "lambda": 0.25, "n_ph_work": 2, "n_ph_ref": 2},
    "strong-strong-u8": {"u": 8.0, "lambda": 1.25, "n_ph_work": 4, "n_ph_ref": 4},
}

_ALIASES = {
    "u8-strong-weak": "strong-weak-u8",
    "u8-strong-strong": "strong-strong-u8",
    "strong_weak_u8": "strong-weak-u8",
    "strong_strong_u8": "strong-strong-u8",
}
_AMBIGUOUS_LEGACY_REGIME_LABELS = frozenset({"strong-weak", "strong_weak", "strong-strong", "strong_strong"})


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _canonical_regime(raw: str) -> str:
    key = str(raw).strip().lower().replace("_", "-")
    raw_key = str(raw).strip().lower()
    if raw_key in _AMBIGUOUS_LEGACY_REGIME_LABELS or key in {
        str(label).replace("_", "-") for label in _AMBIGUOUS_LEGACY_REGIME_LABELS
    }:
        raise argparse.ArgumentTypeError(
            f"Ambiguous legacy HH regime {raw!r}; use intermediate-weak/intermediate-strong "
            "or strong-weak-u8/strong-strong-u8 explicitly."
        )
    if key in _REGIMES:
        return key
    alias_key = raw_key
    if alias_key in _ALIASES:
        return _ALIASES[alias_key]
    if key in _ALIASES:
        return _ALIASES[key]
    valid = ", ".join(sorted([*_REGIMES.keys(), *_ALIASES.keys()]))
    raise argparse.ArgumentTypeError(f"Unknown HH regime {raw!r}; valid values/aliases: {valid}")


def _float_slug(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def _has_custom_hh_point(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None) is not None
        for name in ("point_label", "hh_u", "hh_lambda", "n_ph_work", "hh_t", "hh_omega0")
    )


def _custom_point_label(args: argparse.Namespace) -> str:
    raw_label = getattr(args, "point_label", None)
    if raw_label not in {None, ""}:
        label = str(raw_label).strip().replace(" ", "_")
        if not label:
            raise ValueError("--point-label cannot be blank.")
        return label
    return f"custom_u{_float_slug(float(args.hh_u))}_lam{_float_slug(float(args.hh_lambda))}"


def _effective_hh_point(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if not _has_custom_hh_point(args):
        regime = dict(_REGIMES[str(args.regime)])
        if args.n_ph_ref is not None and int(args.n_ph_ref) != int(regime["n_ph_work"]):
            raise ValueError(
                "Paper-I HH comparisons require --n-ph-ref to equal the regime working cutoff."
            )
        regime.update({"t": 1.0, "omega0": 1.0, "custom": False})
        return str(args.regime), regime
    required = {
        "hh_u": args.hh_u,
        "hh_lambda": args.hh_lambda,
        "n_ph_work": args.n_ph_work,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Custom HH point requires --hh-u, --hh-lambda, and --n-ph-work; "
            f"missing {', '.join(missing)}."
        )
    n_ph_work = int(args.n_ph_work)
    n_ph_ref = n_ph_work if args.n_ph_ref is None else int(args.n_ph_ref)
    if n_ph_ref != n_ph_work:
        raise ValueError(
            "Paper-I HH comparisons require --n-ph-ref to equal --n-ph-work; "
            "use a separate explicitly requested cutoff-sensitivity workflow for unequal cutoffs."
        )
    return _custom_point_label(args), {
        "u": float(args.hh_u),
        "lambda": float(args.hh_lambda),
        "t": 1.0 if args.hh_t is None else float(args.hh_t),
        "omega0": 1.0 if args.hh_omega0 is None else float(args.hh_omega0),
        "n_ph_work": n_ph_work,
        "n_ph_ref": n_ph_ref,
        "custom": True,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Launch the Paper-I HH SNAKE/Optuna speed path with full_meta_minus_hva by default "
            "(or HVA-enabled full_meta when requested), Route A, exact-reference reuse, disk pool cache, "
            "runtime split, and symmetry-off defaults."
        )
    )
    p.add_argument("--regime", type=_canonical_regime, default="strong-strong-u8")
    p.add_argument("--point-label", type=str, default=None, help="Custom HH point label for off-canonical training/evaluation cases.")
    p.add_argument("--hh-t", type=float, default=None, help="Custom HH hopping t; defaults to 1.0 for custom points.")
    p.add_argument("--hh-u", type=float, default=None, help="Custom HH Hubbard U for off-canonical points.")
    p.add_argument("--hh-omega0", type=float, default=None, help="Custom HH phonon frequency; defaults to 1.0 for custom points.")
    p.add_argument("--hh-lambda", dest="hh_lambda", type=float, default=None, help="Custom Paper-I lambda_ep=2*g_ep^2/(t*omega0).")
    p.add_argument("--n-ph-work", type=int, default=None, help="Custom HH working phonon cutoff.")
    p.add_argument(
        "--n-ph-ref",
        type=int,
        default=None,
        help="Legacy alias for the exact cutoff; when supplied it must equal --n-ph-work.",
    )
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--n-startup-trials", type=int, default=None)
    p.add_argument("--lanes", type=str, default="canonical")
    p.add_argument("--epsilon-bands", type=str, default="1e9")
    p.add_argument(
        "--optuna-storage",
        type=str,
        default=None,
        help=(
            "Persistent Optuna storage URL or local SQLite path passed to the delegated study. "
            "Use with --load-if-exists to resume local studies."
        ),
    )
    p.add_argument(
        "--study-name-prefix",
        type=str,
        default=None,
        help="Stable persistent-study prefix; lane and epsilon suffix are appended by the delegated runner.",
    )
    p.add_argument("--load-if-exists", action="store_true", help="Resume an existing persistent Optuna study.")
    p.add_argument(
        "--objective-mode",
        choices=[
            "energy",
            "cost_feasible",
            "graph_cost_speed_feasible",
            "geo_dominance_first",
            "geo_energy_then_graph_cost",
            "geo_energy_then_graph_shot_cost",
            "geo_energy_then_shot_graph_cost",
            "shot_then_energy_graph_cost",
            "geo_energy_gate_then_shot_energy_graph_cost",
            "prune_zero_then_energy_shot_graph_cost",
        ],
        default="energy",
    )
    p.add_argument("--with-compile", action="store_true", help="Enable compiled-cost extraction. Required for --objective-mode cost_feasible.")
    p.add_argument(
        "--preserve-cost-surface-with-skip-compile",
        action="store_true",
        help=(
            "When skipping Qiskit compile under --objective-mode energy, keep cost-policy knobs "
            "active instead of forcing --energy-only-surface. Used by global policy tuning."
        ),
    )
    p.add_argument(
        "--search-inner-optimizer",
        choices=sorted(hh_cost_energy_optuna._SEARCH_INNER_OPTIMIZER_CHOICES),
        default=hh_cost_energy_optuna._SEARCH_INNER_OPTIMIZER,
        help="Inner optimizer for delegated trial subprocesses; POWELL is for deterministic SNAKE-policy tuning.",
    )
    p.add_argument(
        "--speed-surface-profile",
        choices=["standard", "staged_graph", "staged_shot", "shortlist_refine", "energy_discovery", FULL_POLICY_PROFILE],
        default="staged_graph",
        help=(
            "Default staged_graph starts heavy, staged_shot biases toward S_alg reduction, and "
            "shortlist_refine anchors non-shortlist settings to current per-regime candidate priors "
            "while sampling explicit shortlist/window/threshold knobs with maturity shots off. "
            "hh_routea_full_policy_v1 enables the candidate-only full Route-A SNAKE policy surface."
        ),
    )
    p.add_argument(
        "--phase2-w-shot-profile-space",
        choices=["default", "legacy_with_zero"],
        default="default",
        help="Pass-through categorical menu for phase2_w_shot_profile; use legacy_with_zero for older v5 studies.",
    )
    p.add_argument("--max-depth", type=int, default=None, help="Forced local ADAPT depth cap. Ignored by --no-depth-cap.")
    p.add_argument(
        "--benchmark-target-abs-delta-e",
        type=float,
        default=None,
        help="Forward a static |Delta E| target to each delegated ADAPT trial for local seed-admission runs.",
    )
    p.add_argument(
        "--force-run-to-depth",
        action="store_true",
        help="Prevent the delegated ADAPT drop-plateau policy from stopping before --max-depth.",
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help=(
            "Forced per-step optimizer-iteration cap. Ignored by --no-depth-cap. "
            "When omitted for hh_routea_full_policy_v1, Optuna samples SPSA budget."
        ),
    )
    p.add_argument(
        "--final-refit-maxiter",
        type=int,
        default=None,
        help="Forced final full-ansatz refit optimizer-iteration cap. Defaults to --maxiter; 0 also reuses --maxiter.",
    )
    p.add_argument(
        "--allow-unequal-final-refit-budget",
        action="store_true",
        help="Diagnostic-only escape hatch permitting --final-refit-maxiter to differ from --maxiter.",
    )
    p.add_argument("--full-refit-every", type=int, default=None, help="Force --adapt-full-refit-every in every trial.")
    p.add_argument(
        "--final-full-refit",
        choices=["true", "false"],
        default="true",
        help="Force --adapt-final-full-refit in every trial.",
    )
    p.set_defaults(allow_repeats=None)
    p.add_argument("--allow-repeats", dest="allow_repeats", action="store_true", help="Force --adapt-allow-repeats in every trial.")
    p.add_argument("--no-repeats", dest="allow_repeats", action="store_false", help="Force --adapt-no-repeats in every trial.")
    p.add_argument("--phase0-pilot-max-records", type=int, default=None, help="Force --phase0-pilot-max-records in every trial.")
    p.add_argument("--phase1-shortlist-size", type=int, default=None, help="Force --phase1-shortlist-size in every trial.")
    p.add_argument("--phase2-shortlist-fraction", type=float, default=None, help="Force --phase2-shortlist-fraction in every trial.")
    p.add_argument("--phase2-shortlist-size", type=int, default=None, help="Force --phase2-shortlist-size in every trial.")
    p.add_argument(
        "--force-phase1-prune-recovery-trust-radius",
        type=float,
        default=None,
        help=(
            "Force bounded Schur-prune nomination compensation radius in every trial. "
            "Use 0 to preserve unconstrained Schur nomination."
        ),
    )
    p.add_argument("--no-depth-cap", action="store_true", help="Do not force --adapt-max-depth/--adapt-maxiter; use the sampled/base settings.")
    p.add_argument("--adapt-resume-scaffold-json", type=Path, default=None, help="Resume every trial from a structural ADAPT scaffold/current JSON.")
    p.add_argument("--adapt-resume-mode", choices=["scaffold_v1"], default="scaffold_v1")
    p.add_argument("--adapt-segment-id", type=str, default=None)
    p.add_argument("--adapt-segment-target-depth", type=int, default=None)
    p.add_argument("--adapt-segment-max-new-admissions", type=int, default=None)
    p.add_argument("--adapt-segment-wallclock-cap-s", type=float, default=None)
    p.add_argument("--adapt-resume-compile-smoke", choices=["required", "auto", "off"], default=None)
    p.add_argument("--adapt-resume-smoke-backend", type=str, default=None)
    p.add_argument("--gradient-workers", type=int, default=0, help="Forced --adapt-parallel-gradient-workers; 0 enables CPU-aware auto sizing.")
    p.add_argument("--beam-parent-workers", type=int, default=0, help="Forced --adapt-beam-parent-workers; 0 enables CPU-aware auto sizing.")
    p.add_argument(
        "--spsa-parallel-evaluations",
        type=int,
        default=DEFAULT_PARALLEL_EVALUATIONS,
        help="Forced --adapt-spsa-parallel-evaluations; use 1 for serial SPSA.",
    )
    p.add_argument(
        "--phase2-w-shot",
        type=float,
        default=None,
        help=(
            "Force a fixed Phase-II shot-cost weight in every trial. By default this is unset, "
            "so staged graph+shot studies let Optuna sample phase2_w_shot_profile. "
            "Pass 0.0 explicitly only for a shot-neutral ablation/timing study."
        ),
    )
    p.add_argument("--do-not-force-phase2-w-shot", action="store_true")
    p.add_argument("--geo-target-abs-delta-e", type=float, default=None, help="Geo-dominance objective energy target to beat.")
    p.add_argument("--geo-target-iteration", type=int, default=None, help="Geo-dominance objective accepted-ADAPT iteration count to beat strictly. Ignored by geo_energy_then_graph_cost/geo_energy_then_graph_shot_cost.")
    p.add_argument("--geo-target-graph-count-2q", type=float, default=None, help="Optional Geo graph-proxy two-qubit count target. Must be same-surface graph proxy, not Paper-I Qiskit/table compiled cost.")
    p.add_argument("--geo-target-graph-depth", type=float, default=None, help="Optional Geo graph-proxy two-qubit depth target. Must be same-surface graph proxy, not Paper-I Qiskit/table compiled cost.")
    p.add_argument("--geo-target-s-alg", type=float, default=None, help="Optional Geo Paper-I normalized algorithmic shot/work target; defaults from geo_S_alg in --geo-target-manifest when available.")
    p.add_argument("--geo-target-manifest", type=Path, default=DEFAULT_GEO_TARGET_MANIFEST)
    p.add_argument(
        "--graph-proxy-target-manifest",
        type=Path,
        default=DEFAULT_SNAKE_GRAPH_PROXY_TARGET_MANIFEST,
        help=(
            "Optional same-surface graph-proxy target manifest. For shot-focused local studies this "
            "defaults to the current SNAKE review-candidate graph-proxy bests."
        ),
    )
    p.add_argument(
        "--geo-graph-proxy-target-manifest",
        type=Path,
        default=DEFAULT_GEO_GRAPH_PROXY_TARGET_MANIFEST,
        help=(
            "Optional same-surface Geo graph-proxy target manifest. The legacy --geo-target-manifest "
            "is used for energy/S_alg only; its Qiskit/table cost cells are not used as graph-proxy targets."
        ),
    )
    p.add_argument(
        "--enqueue-params-json",
        type=Path,
        default=None,
        help="Optional current-regime Optuna parameter-prior manifest passed to hh_cost_energy_optuna.",
    )
    p.add_argument(
        "--warm-start-audit-json",
        type=Path,
        default=None,
        help="Optional warm-start audit JSON generated by paper_i_hh_snake_fullpolicy_warm_start.py.",
    )
    p.add_argument(
        "--search-space-manifest-json",
        type=Path,
        default=None,
        help="Optional full-policy search-space manifest JSON generated before local/CHTC launch.",
    )
    p.add_argument(
        "--spsa-profile",
        choices=sorted(hh_cost_energy_optuna._SPSA_PROFILE_OPTIONS),
        default=None,
        help="Restrict the delegated Optuna search to one predefined SPSA profile.",
    )
    p.add_argument(
        "--runtime-split-mode",
        choices=["off", "shortlist_pauli_children_v1"],
        default="off",
        help=(
            "Archival conditional Phase-3 split mode. Keep off for same-pool child comparisons; "
            "use --child-pool-expansion-mode for global SNAKE child-set pool runs."
        ),
    )
    p.add_argument(
        "--child-pool-expansion-mode",
        choices=["off", "global_pauli_child_sets_v1", "pauli_child_sets_v1"],
        default="global_pauli_child_sets_v1",
        help="Global pre-Phase-1 SNAKE child-set pool expansion used for child-pool comparisons.",
    )
    p.add_argument(
        "--child-pool-expansion-symmetry-policy",
        choices=["off", "hard_guard"],
        default="off",
        help="Symmetry policy for global SNAKE child-set pool expansion.",
    )
    p.add_argument(
        "--child-pool-expansion-max-subset-size",
        type=int,
        default=3,
        help="Maximum subset size for global SNAKE child-set pool expansion.",
    )
    p.add_argument("--symmetry-mode", choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"], default="off")
    p.add_argument("--exact-manifest", type=Path, default=DEFAULT_EXACT_MANIFEST)
    p.add_argument(
        "--no-exact-manifest",
        action="store_true",
        help="Do not pass --adapt-exact-gs-reference-json; the delegated ADAPT run computes the same-cutoff exact reference.",
    )
    p.add_argument("--class-filter-json", type=Path, default=DEFAULT_CLASS_FILTER)
    p.add_argument(
        "--enable-hva-generators",
        action="store_true",
        help=(
            "Use full_meta without the HH full_meta_minus_hva class filter, so HVA-layer generators "
            "are present in the adaptive pool. This intentionally changes the evidence/search surface."
        ),
    )
    p.add_argument(
        "--hva-aggressive-screening",
        action="store_true",
        help=(
            "Requires --enable-hva-generators. Unless explicit shortlist overrides are supplied, force "
            "phase0/phase1/phase2 screening to 64/16/0.20/8 so HVA generators are screened early."
        ),
    )
    p.add_argument("--restrict-base-presets", type=str, default="resolved_default")
    p.add_argument("--use-default-warm-starts", action="store_true", help="By default local speed runs disable default warm-start observations.")
    p.add_argument("--pool-cache-mode", choices=["disk", "memory", "off"], default="disk")
    p.add_argument(
        "--pool-cache-scope",
        choices=["paper_i_holstein_sector", "exact"],
        default="paper_i_holstein_sector",
        help=(
            "Default reuses HH structural pools per Paper-I Holstein sector across Hubbard U values; "
            "the cache key includes class-filter state, so HVA-enabled and minus-HVA surfaces stay distinct."
        ),
    )
    p.add_argument("--pool-cache-dir", type=Path, default=None)
    p.add_argument("--candidate-record-cache-mode", choices=["disk", "memory", "off"], default="disk")
    p.add_argument("--candidate-record-cache-dir", type=Path, default=None)
    p.add_argument("--print-command-only", action="store_true", help="Print the delegated hh_cost_energy_optuna argv and exit.")
    return p


def _require_file(path: Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _configure_cache(args: argparse.Namespace) -> None:
    if str(args.pool_cache_mode) == "off":
        os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
    elif str(args.pool_cache_mode) == "memory":
        os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "memory"
    else:
        os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "disk"
    if args.pool_cache_dir is not None:
        os.environ["STATIC_ADAPT_HH_POOL_CACHE_DIR"] = str(Path(args.pool_cache_dir))
    os.environ["STATIC_ADAPT_HH_POOL_CACHE_SCOPE"] = str(args.pool_cache_scope)
    if str(args.candidate_record_cache_mode) == "off":
        os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
    elif str(args.candidate_record_cache_mode) == "memory":
        os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "memory"
    else:
        os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "disk"
    if args.candidate_record_cache_dir is not None:
        os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR"] = str(Path(args.candidate_record_cache_dir))


def _effective_screening_values(args: argparse.Namespace) -> tuple[int | None, int | None, float | None, int | None]:
    if bool(args.hva_aggressive_screening) and not bool(args.enable_hva_generators):
        raise ValueError("--hva-aggressive-screening requires --enable-hva-generators.")
    phase0 = args.phase0_pilot_max_records
    phase1 = args.phase1_shortlist_size
    phase2_fraction = args.phase2_shortlist_fraction
    phase2_size = args.phase2_shortlist_size
    if bool(args.hva_aggressive_screening):
        if phase0 is None:
            phase0 = 64
        if phase1 is None:
            phase1 = 16
        if phase2_fraction is None:
            phase2_fraction = 0.20
        if phase2_size is None:
            phase2_size = 8
    return phase0, phase1, phase2_fraction, phase2_size


def _uses_full_policy_surface(args: argparse.Namespace) -> bool:
    return str(args.speed_surface_profile) == FULL_POLICY_PROFILE


def _effective_n_trials(args: argparse.Namespace) -> int:
    if args.n_trials is not None:
        return int(args.n_trials)
    return 48 if _uses_full_policy_surface(args) else 1


def _effective_n_startup_trials(args: argparse.Namespace) -> int:
    if args.n_startup_trials is not None:
        return int(args.n_startup_trials)
    return 12 if _uses_full_policy_surface(args) else 1


def _effective_max_depth(args: argparse.Namespace) -> int:
    if args.max_depth is not None:
        return int(args.max_depth)
    return 30 if _uses_full_policy_surface(args) else 13


def _effective_maxiter(args: argparse.Namespace) -> int:
    if args.maxiter is not None:
        return int(args.maxiter)
    return 200 if _uses_full_policy_surface(args) else 800


def _samples_spsa_budget(args: argparse.Namespace) -> bool:
    return _uses_full_policy_surface(args) and args.maxiter is None and not bool(args.no_depth_cap)


def _effective_final_refit_maxiter(args: argparse.Namespace) -> int:
    maxiter = _effective_maxiter(args)
    if int(maxiter) <= 0:
        raise ValueError("--maxiter must be positive.")
    raw = args.final_refit_maxiter
    if _samples_spsa_budget(args) and raw is not None:
        raise ValueError("--final-refit-maxiter requires --maxiter when hh_routea_full_policy_v1 samples SPSA budget.")
    if raw is None or int(raw) == 0:
        return int(maxiter)
    parsed = int(raw)
    if parsed < 0:
        raise ValueError("--final-refit-maxiter must be nonnegative.")
    return parsed


def _validate_optimizer_budget_contract(args: argparse.Namespace) -> int:
    effective = _effective_final_refit_maxiter(args)
    if bool(args.no_depth_cap) or _samples_spsa_budget(args):
        return effective
    if effective != int(_effective_maxiter(args)) and not bool(args.allow_unequal_final_refit_budget):
        raise ValueError(
            "optimizer_fairness_violation: Paper-I HH speed runs require --final-refit-maxiter "
            "to equal --maxiter unless --allow-unequal-final-refit-budget is set for an explicit diagnostic."
        )
    return effective


def _geo_graph_targets(args: argparse.Namespace) -> tuple[float | None, float | None]:
    target_n2q = args.geo_target_graph_count_2q
    target_depth = args.geo_target_graph_depth
    if target_n2q is not None and target_depth is not None:
        return float(target_n2q), float(target_depth)
    point_label, _ = _effective_hh_point(args)
    manifest_path = Path(args.graph_proxy_target_manifest or args.geo_graph_proxy_target_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if manifest_path.exists():
        try:
            row = (json.loads(manifest_path.read_text()).get("regimes") or {}).get(str(point_label)) or {}
            if str(row.get("cost_surface", row.get("surface", ""))) not in {"graph_proxy", "marrakesh_graph_span_v1"}:
                return (None if target_n2q is None else float(target_n2q), None if target_depth is None else float(target_depth))
            if target_n2q is None:
                for key in (
                    "graph_proxy_N2q",
                    "snake_graph_proxy_N2q",
                    "geo_graph_proxy_N2q",
                    "geo_graph_count_2q",
                    "graph_count_2q",
                    "N2Q_proxy",
                    "geo_N2q",
                ):
                    if row.get(key) is not None:
                        target_n2q = float(row[key])
                        break
            if target_depth is None:
                for key in (
                    "graph_proxy_D2q",
                    "snake_graph_proxy_D2q",
                    "geo_graph_proxy_D2q",
                    "geo_graph_depth",
                    "graph_depth",
                    "D2Q_proxy",
                    "geo_D2q",
                ):
                    if row.get(key) is not None:
                        target_depth = float(row[key])
                        break
        except Exception:
            pass
    return (None if target_n2q is None else float(target_n2q), None if target_depth is None else float(target_depth))


def _geo_s_alg_target(args: argparse.Namespace) -> float | None:
    target_s_alg = args.geo_target_s_alg
    if target_s_alg is not None:
        return float(target_s_alg)
    point_label, _ = _effective_hh_point(args)
    manifest_path = Path(args.geo_target_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if manifest_path.exists():
        try:
            row = (json.loads(manifest_path.read_text()).get("regimes") or {}).get(str(point_label)) or {}
            if row.get("geo_S_alg") is not None:
                target_s_alg = float(row["geo_S_alg"])
        except Exception:
            pass
    return None if target_s_alg is None else float(target_s_alg)


def _effective_study_name_prefix(
    raw_prefix: str | None,
    *,
    objective_mode: str,
    graph_target_n2q: float | None,
    graph_target_depth: float | None,
    target_s_alg: float | None,
) -> str | None:
    if raw_prefix in {None, ""}:
        return None
    prefix = str(raw_prefix)
    if str(objective_mode) == "shot_then_energy_graph_cost":
        if target_s_alg is not None:
            for suffix in ("_graphshot_v2", "_graphdom_v3", "_graphdomshot_v4", "_graphdomshot_salg_v5", "_shotdom_salg_v6", "_shotfirst_salg_v7"):
                if prefix.endswith(suffix):
                    return f"{prefix[:-len(suffix)]}_shotfirst_salg_v7"
            if "shotfirst_salg_v7" not in prefix:
                return f"{prefix}_shotfirst_salg_v7"
        return prefix
    if str(objective_mode) == "geo_energy_gate_then_shot_energy_graph_cost":
        if target_s_alg is not None:
            for suffix in (
                "_graphshot_v2",
                "_graphdom_v3",
                "_graphdomshot_v4",
                "_graphdomshot_salg_v5",
                "_shotdom_salg_v6",
                "_shotfirst_salg_v7",
                "_egate_shotfirst_salg_v8",
            ):
                if prefix.endswith(suffix):
                    return f"{prefix[:-len(suffix)]}_egate_shotfirst_salg_v8"
            if "egate_shotfirst_salg_v8" not in prefix:
                return f"{prefix}_egate_shotfirst_salg_v8"
        return prefix
    if str(objective_mode) == "geo_energy_then_shot_graph_cost":
        if target_s_alg is not None:
            for suffix in ("_graphshot_v2", "_graphdom_v3", "_graphdomshot_v4", "_graphdomshot_salg_v5", "_shotdom_salg_v6"):
                if prefix.endswith(suffix):
                    return f"{prefix[:-len(suffix)]}_shotdom_salg_v6"
            if "shotdom_salg_v6" not in prefix:
                return f"{prefix}_shotdom_salg_v6"
        return prefix
    if str(objective_mode) in {"geo_energy_then_graph_cost", "geo_energy_then_graph_shot_cost"} and graph_target_n2q is not None and graph_target_depth is not None:
        if target_s_alg is not None:
            for suffix in ("_graphshot_v2", "_graphdom_v3", "_graphdomshot_v4", "_graphdomshot_salg_v5"):
                if prefix.endswith(suffix):
                    return f"{prefix[:-len(suffix)]}_graphdomshot_salg_v5"
            if "graphdomshot_salg_v5" not in prefix:
                return f"{prefix}_graphdomshot_salg_v5"
        if prefix.endswith("_graphshot_v2"):
            return f"{prefix[:-len('_graphshot_v2')]}_graphdom_v3"
        if "graphdom_v3" not in prefix:
            return f"{prefix}_graphdom_v3"
    return prefix


def _delegated_argv(args: argparse.Namespace) -> list[str]:
    point_label, regime = _effective_hh_point(args)
    final_refit_maxiter = _validate_optimizer_budget_contract(args)
    n_trials = _effective_n_trials(args)
    n_startup_trials = _effective_n_startup_trials(args)
    max_depth = _effective_max_depth(args)
    maxiter = _effective_maxiter(args)
    exact_manifest = None if bool(args.no_exact_manifest) else _require_file(Path(args.exact_manifest), "HH exact-reference manifest")
    class_filter = None if bool(args.enable_hva_generators) else _require_file(Path(args.class_filter_json), "HH full_meta_minus_hva class filter")
    phase0, phase1, phase2_fraction, phase2_size = _effective_screening_values(args)
    default_tag = (
        f"{FULL_POLICY_OUTPUT_SLUG}__{str(point_label).replace('-', '_')}"
        if _uses_full_policy_surface(args)
        else f"paper_i_hh_speed_{str(point_label).replace('-', '_')}_{_timestamp_slug()}"
    )
    tag = str(args.tag or default_tag)
    output_dir = Path(args.output_dir) if args.output_dir is not None else DEFAULT_OUTPUT_ROOT / tag
    if str(args.objective_mode) == "cost_feasible" and not bool(args.with_compile):
        raise ValueError("--objective-mode cost_feasible requires --with-compile.")
    if str(args.objective_mode) == "geo_dominance_first" and (
        args.geo_target_abs_delta_e is None or args.geo_target_iteration is None
    ):
        raise ValueError("--objective-mode geo_dominance_first requires --geo-target-abs-delta-e and --geo-target-iteration.")
    if str(args.objective_mode) in {
        "geo_energy_then_graph_cost",
        "geo_energy_then_graph_shot_cost",
        "geo_energy_then_shot_graph_cost",
        "geo_energy_gate_then_shot_energy_graph_cost",
        "prune_zero_then_energy_shot_graph_cost",
    } and args.geo_target_abs_delta_e is None:
        raise ValueError(f"--objective-mode {args.objective_mode} requires --geo-target-abs-delta-e.")
    geo_graph_n2q, geo_graph_depth = _geo_graph_targets(args)
    geo_s_alg = _geo_s_alg_target(args)
    if str(args.objective_mode) in {
        "geo_energy_then_shot_graph_cost",
        "shot_then_energy_graph_cost",
        "geo_energy_gate_then_shot_energy_graph_cost",
        "prune_zero_then_energy_shot_graph_cost",
    } and geo_s_alg is None:
        raise ValueError(f"--objective-mode {args.objective_mode} requires --geo-target-s-alg or geo_S_alg in --geo-target-manifest.")
    study_name_prefix = _effective_study_name_prefix(
        args.study_name_prefix,
        objective_mode=str(args.objective_mode),
        graph_target_n2q=geo_graph_n2q,
        graph_target_depth=geo_graph_depth,
        target_s_alg=geo_s_alg,
    )

    delegated = [
        "--tag",
        tag,
        "--output-dir",
        str(output_dir),
        "--python-bin",
        str(args.python_bin),
        "--lanes",
        str(args.lanes),
        "--epsilon-bands",
        str(args.epsilon_bands),
        "--n-trials",
        str(int(n_trials)),
        "--n-startup-trials",
        str(int(n_startup_trials)),
        "--objective-mode",
        str(args.objective_mode),
        "--search-inner-optimizer",
        str(args.search_inner_optimizer),
        "--speed-surface-profile",
        str(args.speed_surface_profile),
        "--phase2-w-shot-profile-space",
        str(args.phase2_w_shot_profile_space),
        "--restrict-base-presets",
        str(args.restrict_base_presets),
        "--hh-L",
        "2",
        "--hh-t",
        str(float(regime["t"])),
        "--hh-u",
        str(float(regime["u"])),
        "--hh-omega0",
        str(float(regime["omega0"])),
        "--hh-lambda",
        str(float(regime["lambda"])),
        "--n-ph-work",
        str(int(regime["n_ph_work"])),
        "--n-ph-ref",
        str(int(regime["n_ph_ref"])),
        "--force-adapt-pool",
        "full_meta",
        "--force-static-route-id",
        "route_a",
        "--force-static-meta-feature-profile",
        "paper_i_production_v1",
        "--force-route-a-paper-i-production",
        "--force-phase3-symmetry-mitigation-mode",
        str(args.symmetry_mode),
        "--force-adapt-parallel-gradient-workers",
        str(int(args.gradient_workers)),
        "--force-adapt-beam-parent-workers",
        str(int(args.beam_parent_workers)),
        "--force-runtime-split-mode",
        str(args.runtime_split_mode),
        "--force-adapt-child-pool-expansion-mode",
        str(args.child_pool_expansion_mode),
        "--force-adapt-child-pool-expansion-symmetry-policy",
        str(args.child_pool_expansion_symmetry_policy),
        "--force-adapt-child-pool-expansion-max-subset-size",
        str(int(args.child_pool_expansion_max_subset_size)),
    ]
    if str(args.search_inner_optimizer).upper() == "SPSA":
        delegated.extend([
            "--force-adapt-spsa-parallel-evaluations",
            str(int(args.spsa_parallel_evaluations)),
        ])
    if exact_manifest is not None:
        delegated.extend(["--adapt-exact-gs-reference-json", str(exact_manifest)])
    if class_filter is not None:
        delegated.extend(["--force-adapt-pool-class-filter-json", str(class_filter)])
    if args.benchmark_target_abs_delta_e is not None:
        delegated.extend([
            "--force-adapt-benchmark-target-abs-delta-e",
            str(float(args.benchmark_target_abs_delta_e)),
        ])
    if args.optuna_storage not in {None, ""}:
        delegated.extend(["--optuna-storage", str(args.optuna_storage)])
    if study_name_prefix not in {None, ""}:
        delegated.extend(["--study-name-prefix", str(study_name_prefix)])
    if bool(args.load_if_exists):
        delegated.append("--load-if-exists")
    enqueue_params_json = Path(args.enqueue_params_json) if args.enqueue_params_json is not None else None
    if enqueue_params_json is not None:
        delegated.extend(["--enqueue-params-json", str(enqueue_params_json)])
        delegated.extend(["--enqueue-params-regime", str(point_label)])
    if str(args.objective_mode) == "geo_dominance_first":
        delegated.extend([
            "--dominance-target-abs-delta-e",
            str(float(args.geo_target_abs_delta_e)),
            "--dominance-target-iteration",
            str(int(args.geo_target_iteration)),
        ])
    elif str(args.objective_mode) in {
        "geo_energy_then_graph_cost",
        "geo_energy_then_graph_shot_cost",
        "geo_energy_then_shot_graph_cost",
        "shot_then_energy_graph_cost",
        "geo_energy_gate_then_shot_energy_graph_cost",
        "prune_zero_then_energy_shot_graph_cost",
    }:
        delegated.extend([
            "--dominance-target-abs-delta-e",
            str(float(args.geo_target_abs_delta_e)),
        ])
        if geo_graph_n2q is not None:
            delegated.extend(["--dominance-target-graph-count-2q", str(float(geo_graph_n2q))])
        if geo_graph_depth is not None:
            delegated.extend(["--dominance-target-graph-depth", str(float(geo_graph_depth))])
        if geo_s_alg is not None:
            delegated.extend(["--dominance-target-s-alg", str(float(geo_s_alg))])
    if not bool(args.with_compile):
        delegated.append("--skip-compile")
        if str(args.objective_mode) == "energy" and not bool(args.preserve_cost_surface_with_skip_compile):
            delegated.append("--energy-only-surface")
    if not bool(args.use_default_warm_starts):
        delegated.append("--no-default-warm-starts")
    if not bool(args.no_depth_cap):
        delegated.extend([
            "--force-adapt-max-depth",
            str(int(max_depth)),
        ])
        if not _samples_spsa_budget(args):
            delegated.extend([
                "--force-adapt-maxiter",
                str(int(maxiter)),
                "--force-adapt-final-refit-maxiter",
                str(int(final_refit_maxiter)),
            ])
        if bool(args.force_run_to_depth):
            delegated.extend([
                "--force-adapt-drop-min-depth",
                str(int(max_depth) + 1),
                "--force-adapt-drop-patience",
                "1000000",
            ])
    if args.full_refit_every is not None:
        delegated.extend(["--force-adapt-full-refit-every", str(int(args.full_refit_every))])
    if args.final_full_refit not in {None, ""}:
        delegated.extend(["--force-adapt-final-full-refit", str(args.final_full_refit)])
    if args.allow_repeats is True:
        delegated.append("--force-adapt-allow-repeats")
    elif args.allow_repeats is False:
        delegated.append("--force-adapt-no-repeats")
    if phase0 is not None:
        delegated.extend(["--force-phase0-pilot-max-records", str(int(phase0))])
    if phase1 is not None:
        delegated.extend(["--force-phase1-shortlist-size", str(int(phase1))])
    if phase2_fraction is not None:
        delegated.extend(["--force-phase2-shortlist-fraction", str(float(phase2_fraction))])
    if phase2_size is not None:
        delegated.extend(["--force-phase2-shortlist-size", str(int(phase2_size))])
    if args.force_phase1_prune_recovery_trust_radius is not None:
        delegated.extend([
            "--force-phase1-prune-recovery-trust-radius",
            str(float(args.force_phase1_prune_recovery_trust_radius)),
        ])
    if args.adapt_resume_scaffold_json is not None:
        resume_scaffold = _require_file(Path(args.adapt_resume_scaffold_json), "ADAPT resume scaffold JSON")
        delegated.extend(["--force-adapt-resume-scaffold-json", str(resume_scaffold)])
        delegated.extend(["--force-adapt-resume-mode", str(args.adapt_resume_mode)])
    if args.adapt_segment_id not in {None, ""}:
        delegated.extend(["--force-adapt-segment-id", str(args.adapt_segment_id)])
    if args.adapt_segment_target_depth is not None:
        delegated.extend(["--force-adapt-segment-target-depth", str(int(args.adapt_segment_target_depth))])
    if args.adapt_segment_max_new_admissions is not None:
        delegated.extend(["--force-adapt-segment-max-new-admissions", str(int(args.adapt_segment_max_new_admissions))])
    if args.adapt_segment_wallclock_cap_s is not None:
        delegated.extend(["--force-adapt-segment-wallclock-cap-s", str(float(args.adapt_segment_wallclock_cap_s))])
    if args.adapt_resume_compile_smoke not in {None, ""}:
        delegated.extend(["--force-adapt-resume-compile-smoke", str(args.adapt_resume_compile_smoke)])
    if args.adapt_resume_smoke_backend not in {None, ""}:
        delegated.extend(["--force-adapt-resume-smoke-backend", str(args.adapt_resume_smoke_backend)])
    if (not bool(args.do_not_force_phase2_w_shot)) and args.phase2_w_shot is not None:
        delegated.extend(["--force-phase2-w-shot", str(float(args.phase2_w_shot))])
    if args.spsa_profile not in {None, ""}:
        delegated.extend(["--force-spsa-profile", str(args.spsa_profile)])
    return delegated


def _command_payload(argv: Sequence[str], args: argparse.Namespace) -> Mapping[str, Any]:
    phase0, phase1, phase2_fraction, phase2_size = _effective_screening_values(args)
    final_refit_maxiter = _validate_optimizer_budget_contract(args)
    n_trials = _effective_n_trials(args)
    n_startup_trials = _effective_n_startup_trials(args)
    max_depth = _effective_max_depth(args)
    maxiter = _effective_maxiter(args)
    point_label, hh_point = _effective_hh_point(args)
    return {
        "wrapper": str(Path(__file__).resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "regime": str(args.regime),
        "point_label": str(point_label),
        "custom_hh_point": bool(hh_point.get("custom", False)),
        "hh_point": {
            "L": 2,
            "t": float(hh_point["t"]),
            "u": float(hh_point["u"]),
            "omega0": float(hh_point["omega0"]),
            "lambda": float(hh_point["lambda"]),
            "n_ph_work": int(hh_point["n_ph_work"]),
            "n_ph_ref": int(hh_point["n_ph_ref"]),
        },
        "pool_cache_mode": str(args.pool_cache_mode),
        "pool_cache_dir": None if args.pool_cache_dir is None else str(Path(args.pool_cache_dir)),
        "pool_cache_scope": str(args.pool_cache_scope),
        "candidate_record_cache_mode": str(args.candidate_record_cache_mode),
        "candidate_record_cache_dir": (
            None if args.candidate_record_cache_dir is None else str(Path(args.candidate_record_cache_dir))
        ),
        "objective_mode": str(args.objective_mode),
        "search_inner_optimizer": str(args.search_inner_optimizer),
        "preserve_cost_surface_with_skip_compile": bool(args.preserve_cost_surface_with_skip_compile),
        "no_exact_manifest": bool(args.no_exact_manifest),
        "exact_manifest": None if bool(args.no_exact_manifest) else str(Path(args.exact_manifest)),
        "optuna_storage": args.optuna_storage,
        "study_name_prefix": args.study_name_prefix,
        "effective_study_name_prefix": _effective_study_name_prefix(
            args.study_name_prefix,
            objective_mode=str(args.objective_mode),
            graph_target_n2q=_geo_graph_targets(args)[0],
            graph_target_depth=_geo_graph_targets(args)[1],
            target_s_alg=_geo_s_alg_target(args),
        ),
        "load_if_exists": bool(args.load_if_exists),
        "n_trials": int(n_trials),
        "n_startup_trials": int(n_startup_trials),
        "speed_surface_profile": str(args.speed_surface_profile),
        "samples_spsa_budget": bool(_samples_spsa_budget(args)),
        "phase2_w_shot_profile_space": str(args.phase2_w_shot_profile_space),
        "geo_target_abs_delta_e": args.geo_target_abs_delta_e,
        "geo_target_iteration": args.geo_target_iteration,
        "geo_target_graph_count_2q": args.geo_target_graph_count_2q,
        "geo_target_graph_depth": args.geo_target_graph_depth,
        "geo_target_s_alg": args.geo_target_s_alg,
        "effective_geo_target_s_alg": _geo_s_alg_target(args),
        "geo_target_manifest": str(Path(args.geo_target_manifest)),
        "graph_proxy_target_manifest": str(Path(args.graph_proxy_target_manifest)),
        "geo_graph_proxy_target_manifest": str(Path(args.geo_graph_proxy_target_manifest)),
        "effective_graph_proxy_target_count_2q": _geo_graph_targets(args)[0],
        "effective_graph_proxy_target_depth": _geo_graph_targets(args)[1],
        "enqueue_params_json": (None if args.enqueue_params_json is None else str(Path(args.enqueue_params_json))),
        "warm_start_audit_json": (
            None if args.warm_start_audit_json is None else str(Path(args.warm_start_audit_json))
        ),
        "search_space_manifest_json": (
            None if args.search_space_manifest_json is None else str(Path(args.search_space_manifest_json))
        ),
        "full_policy_output_slug": FULL_POLICY_OUTPUT_SLUG if _uses_full_policy_surface(args) else None,
        "fixed_identity_locks": {
            "paper": "Paper-I",
            "table": "HH Table III",
            "method": "SNAKE",
            "route_id": "route_a",
            "pool_policy": "full_meta_minus_hva" if not bool(args.enable_hva_generators) else "full_meta_hva_enabled",
            "static_meta_feature_profile": "paper_i_production_v1",
            "phase3_batch_selection_mode": "reduced_plane",
            "phase3_batch_prefilter_mode": "off",
            "phase1_prune_policy": "recoverability_ladder_v1",
            "cutoff_contract": {
                "n_ph_work": int(hh_point["n_ph_work"]),
                "n_ph_ref": int(hh_point["n_ph_ref"]),
            },
        },
        "generated_artifact_paths": {
            "output_dir": str(Path(args.output_dir)) if args.output_dir is not None else None,
            "warm_start_audit_json": (
                None if args.warm_start_audit_json is None else str(Path(args.warm_start_audit_json))
            ),
            "enqueue_params_json": (
                None if args.enqueue_params_json is None else str(Path(args.enqueue_params_json))
            ),
            "search_space_manifest_json": (
                None if args.search_space_manifest_json is None else str(Path(args.search_space_manifest_json))
            ),
        },
        "max_depth": int(max_depth),
        "maxiter": int(maxiter),
        "final_refit_maxiter": int(final_refit_maxiter),
        "optimizer_fairness_policy": "paper_i_hh_equal_adapt_and_final_refit_maxiter_v1",
        "allow_unequal_final_refit_budget": bool(args.allow_unequal_final_refit_budget),
        "full_refit_every": args.full_refit_every,
        "final_full_refit": args.final_full_refit,
        "allow_repeats": args.allow_repeats,
        "pool_policy": "full_meta_hva_enabled" if bool(args.enable_hva_generators) else "full_meta_minus_hva",
        "enable_hva_generators": bool(args.enable_hva_generators),
        "hva_aggressive_screening": bool(args.hva_aggressive_screening),
        "class_filter_json": None if bool(args.enable_hva_generators) else str(Path(args.class_filter_json)),
        "effective_phase0_pilot_max_records": phase0,
        "effective_phase1_shortlist_size": phase1,
        "effective_phase2_shortlist_fraction": phase2_fraction,
        "effective_phase2_shortlist_size": phase2_size,
        "phase0_pilot_max_records": args.phase0_pilot_max_records,
        "phase1_shortlist_size": args.phase1_shortlist_size,
        "phase2_shortlist_fraction": args.phase2_shortlist_fraction,
        "phase2_shortlist_size": args.phase2_shortlist_size,
        "force_phase1_prune_recovery_trust_radius": args.force_phase1_prune_recovery_trust_radius,
        "adapt_resume_scaffold_json": (
            None if args.adapt_resume_scaffold_json is None else str(Path(args.adapt_resume_scaffold_json))
        ),
        "adapt_resume_mode": args.adapt_resume_mode,
        "adapt_segment_id": args.adapt_segment_id,
        "adapt_segment_target_depth": args.adapt_segment_target_depth,
        "adapt_segment_max_new_admissions": args.adapt_segment_max_new_admissions,
        "adapt_segment_wallclock_cap_s": args.adapt_segment_wallclock_cap_s,
        "adapt_resume_compile_smoke": args.adapt_resume_compile_smoke,
        "adapt_resume_smoke_backend": args.adapt_resume_smoke_backend,
        "gradient_workers": int(args.gradient_workers),
        "beam_parent_workers": int(args.beam_parent_workers),
        "spsa_parallel_evaluations": int(args.spsa_parallel_evaluations),
        "spsa_parallel_evaluations_effective": (
            int(args.spsa_parallel_evaluations)
            if str(args.search_inner_optimizer).upper() == "SPSA"
            else "not_applicable_for_non_spsa_inner_optimizer"
        ),
        "spsa_profile": args.spsa_profile,
        "delegated_module": "pipelines.exact_bench.hh_cost_energy_optuna",
        "delegated_argv": list(argv),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    child_mode = str(args.child_pool_expansion_mode or "off").strip().lower()
    split_mode = str(args.runtime_split_mode or "off").strip().lower()
    if child_mode not in {"", "off", "none", "false", "0", "disabled"} and split_mode != "off":
        raise ValueError(
            "--child-pool-expansion-mode is a global pre-Phase-1 pool expansion and "
            "cannot be combined with --runtime-split-mode != off."
        )
    _configure_cache(args)
    delegated = _delegated_argv(args)
    payload = _command_payload(delegated, args)
    if bool(args.print_command_only):
        print(json.dumps(payload, indent=2))
        return
    hh_cost_energy_optuna.main(delegated)


if __name__ == "__main__":
    main()
