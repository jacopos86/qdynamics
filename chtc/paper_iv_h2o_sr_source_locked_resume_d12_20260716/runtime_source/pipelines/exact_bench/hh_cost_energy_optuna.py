#!/usr/bin/env python3
"""Constrained Optuna pilot for HH cost-vs-energy scaffold search.

Phase-1 design:
- run separate lane studies (`canonical`, `global`) plus an optional compatibility-only `legacy` lane
- optimize compiled 2Q cost under an energy-band constraint
- seed studies from known artifact-backed presets and optional warm-start artifacts
- evaluate all trial winners on a fixed compile backend (default: FakeMarrakesh)

This is intentionally a harness-level search surface. It does not modify core ADAPT logic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import MISSING, asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.heron_pareto_sweep import pareto_front
from pipelines.exact_bench.snake_table_i_measurement_work import snake_deterministic_shot_proxy_from_payload
from pipelines.static_adapt.prune_risk_dataset import prune_telemetry_counts


_PIPELINE_NAME = "hh_cost_energy_optuna_v1"
_DEFAULT_LANES = ("canonical", "global")
_DEFAULT_EPSILON_BANDS = (1.1e-4, 6.2e-5)
_DEFAULT_COMPILE_BACKEND = "FakeMarrakesh"
_DEFAULT_COMPILE_OPT_LEVEL = 1
_DEFAULT_COMPILE_SEED = 7
_MARRAKESH_GRAPH_SPAN_MODE = "marrakesh_graph_span_v1"
_HH_ROUTEA_FULL_POLICY_PROFILE = "hh_routea_full_policy_v1"
_PAPER_I_SHOTS_PER_PAULI_TERM_PROXY = 1024
# Resource objective used after the Geo-ADAPT energy gate.  The graph-span 2Q
# estimate remains dominant, but Paper-I deterministic shots are a first-class
# cost signal rather than an effectively invisible tie-breaker.
_RESOURCE_OBJECTIVE_GRAPH_2Q_WEIGHT = 1.0e9
_RESOURCE_OBJECTIVE_GRAPH_DEPTH_WEIGHT = 1.0e6
_RESOURCE_OBJECTIVE_GRAPH_1Q_WEIGHT = 1.0e3
_RESOURCE_OBJECTIVE_GRAPH_THETA_WEIGHT = 1.0
_RESOURCE_OBJECTIVE_SHOT_WEIGHT = 1.0
_RESOURCE_OBJECTIVE_ELAPSED_MS_WEIGHT = 1.0e-3
_RESOURCE_OBJECTIVE_MISSING_SHOTS = 1.0e12
_LARGE_OBJECTIVE = int(10**18)

_LEGACY81_RESULT_JSON = (
    REPO_ROOT
    / "artifacts/agent_runs/20260409_hh_l2_hist81_legacy_current_compare_d16_v3/legacy_20260322/json/result.json"
)
_LEGACY81_COMMAND = _LEGACY81_RESULT_JSON.parents[1] / "logs/command.sh"
_PUBLIC_ANCHOR_COMMAND = (
    REPO_ROOT
    / "artifacts/agent_runs/20260405_hh_l2_u4_g05_phase3_public_spsa_baseline_rerun/logs/command.sh"
)
_RAW_EXACT_COMPILE_ONLY_COMMAND = (
    REPO_ROOT
    / "artifacts/agent_runs/20260411_hh_l2_phase3_burden_sweep_v1/cases/raw_exact_compile_only/logs/command.sh"
)
_FULLHORSE_MOTIF_COMMAND = (
    REPO_ROOT
    / "artifacts/agent_runs/20260410_hh_l2_current_fullhorse_recovery_v1/cases/fullhorse_spliton_norepeats_motif/logs/command.sh"
)
_BRIDGE_DIAG_COMMAND = (
    REPO_ROOT
    / "artifacts/agent_runs/20260409_hh_l2_children_repeat_bridge_diag_v1/cases/d10_children_off_repeats_off_hist/logs/command.sh"
)
_BRIDGE_FOCUS_98_COMMAND = (
    REPO_ROOT
    / "artifacts/agent_runs/20260414_hh_l2_bridge_diag_focus_optuna_v1/global/eps_6.200em05/trial_0024/logs/command.sh"
)
_BRIDGE_FOCUS_118_COMMAND = (
    REPO_ROOT
    / "artifacts/agent_runs/20260414_hh_l2_bridge_diag_focus_optuna_v1/global/eps_6.200em05/trial_0005/logs/command.sh"
)
_LEGACY_FOCUS_75_COMMAND = (
    REPO_ROOT
    / "artifacts/agent_runs/20260414_hh_l2_legacy_focus_optuna_v1/legacy/eps_6.200em05/trial_0003/logs/command.sh"
)

_SEARCH_INNER_OPTIMIZER = "SPSA"
_SEARCH_INNER_OPTIMIZER_CHOICES = frozenset({"SPSA", "POWELL", "ROTOSOLVE"})
_SEARCH_ADAPT_MAXITER_FLOOR = 800
_SEARCH_SPSA_A = 0.1
_SEARCH_SPSA_C = 0.02
_SEARCH_SPSA_ALPHA = 0.602
_SEARCH_SPSA_GAMMA = 0.101
_SEARCH_SPSA_A_SHIFT = 5.0
_SEARCH_SPSA_AVG_LAST = 0
_SEARCH_SPSA_EVAL_REPEATS = 1
_SEARCH_SPSA_EVAL_AGG = "mean"
_SEARCH_SPSA_CALLBACK_EVERY = 5
_SEARCH_SPSA_PROGRESS_EVERY_S = 30.0

_CANONICAL_ADAPT_MODULE = "pipelines.static_adapt.adapt_pipeline"
_CANONICAL_ADAPT_SCRIPT = "pipelines/static_adapt/adapt_pipeline.py"
_COMPAT_ADAPT_MODULE = "pipelines.hardcoded.adapt_pipeline"
_COMPAT_ADAPT_SCRIPT = "pipelines/hardcoded/adapt_pipeline.py"

_MATURITY_SHORTLIST_FLAGS = frozenset({
    "--phase1-shortlist-size",
    "--phase2-shortlist-fraction",
    "--phase2-shortlist-size",
    "--phase1-maturity-cap-min",
    "--phase1-maturity-cap-max",
    "--phase2-maturity-cap-min",
    "--phase2-maturity-cap-max",
    "--phase3-maturity-cap-min",
    "--phase3-maturity-cap-max",
})
_PHASE0_PILOT_PROFILE_FLAGS = frozenset({
    "--phase0-pilot-max-records",
})
_PHASE1_SHORTLIST_SIZE_FLAGS = frozenset({
    "--phase1-shortlist-size",
})
_PHASE2_SHORTLIST_FRACTION_FLAGS = frozenset({
    "--phase2-shortlist-fraction",
})
_PHASE2_SHORTLIST_SIZE_FLAGS = frozenset({
    "--phase2-shortlist-size",
})
_MATURITY_SHOT_FLAGS = frozenset({
    "--phase-maturity-shot-min",
    "--phase-maturity-shot-max",
    "--phase1-maturity-shot-cap",
    "--phase2-maturity-shot-cap",
    "--phase3-maturity-shot-cap",
})
_PHASE_LIVE_FLAGS = frozenset({
    "--phase-live-hysteresis-enabled",
    "--phase-live-hysteresis-disabled",
    "--phase2-null-nrem-high-threshold",
    "--phase2-live-nrem-low-threshold",
    "--phase3-null-nrem-high-threshold",
    "--phase3-live-nrem-low-threshold",
    "--phase2-hysteresis-steps",
    "--phase3-hysteresis-steps",
})
_PRUNE_WITNESS_FLAGS = frozenset({
    "--phase1-prune-amplitude-witness-required",
    "--phase1-prune-amplitude-witness-optional",
    "--phase1-prune-collapse-peak-abs-min",
    "--phase1-prune-collapse-current-abs-max",
    "--phase1-prune-collapse-ratio",
    "--phase1-prune-collapse-min-abs-drop",
    "--phase1-prune-collapse-min-observations",
})
_PRUNE_PREFILTER_FLAGS = frozenset({
    "--phase1-prune-prefilter-policy",
    "--phase1-prune-prefilter-json",
    "--phase1-prune-risk-threshold",
    "--phase1-prune-prefilter-max-candidates",
})
_ADAPT_WINDOW_FLAGS = frozenset({
    "--adapt-window-size",
    "--adapt-window-topk",
    "--phase3-geometry-window-size",
})
_ADAPT_HISTORY_WINDOW_FLAGS = frozenset({
    "--adapt-window-size",
    "--adapt-window-topk",
})
_GEOMETRY_WINDOW_FLAGS = frozenset({
    "--phase3-geometry-window-size",
})
_BACKEND_COST_WEIGHT_FLAGS = frozenset({
    "--phase3-backend-w-2q",
    "--phase3-backend-w-depth",
    "--phase3-backend-w-size",
})
_ML_CANDIDATE_PROFILE_FLAGS = frozenset({
    "--phase1-prune-fraction",
    "--phase2-batch-near-degenerate-ratio",
    "--phase3-batch-near-degenerate-ratio",
    "--phase2-batch-rank-rel-tol",
    "--phase3-batch-rank-rel-tol",
    "--phase2-batch-additivity-tol",
    "--phase3-batch-additivity-tol",
})
_PHASE1_PRUNE_FRACTION_FLAGS = frozenset({
    "--phase1-prune-fraction",
})
_BATCH_NEAR_DEGENERATE_FLAGS = frozenset({
    "--phase2-batch-near-degenerate-ratio",
    "--phase3-batch-near-degenerate-ratio",
})
_BATCH_RANK_TOL_FLAGS = frozenset({
    "--phase2-batch-rank-rel-tol",
    "--phase3-batch-rank-rel-tol",
})
_BATCH_ADDITIVITY_TOL_FLAGS = frozenset({
    "--phase2-batch-additivity-tol",
    "--phase3-batch-additivity-tol",
})
_SPSA_PROFILE_FLAGS = frozenset({
    "--adapt-spsa-a",
    "--adapt-spsa-c",
    "--adapt-spsa-alpha",
    "--adapt-spsa-gamma",
    "--adapt-spsa-A",
    "--adapt-spsa-avg-last",
    "--adapt-spsa-eval-repeats",
    "--adapt-spsa-eval-agg",
    "--adapt-spsa-callback-every",
    "--adapt-spsa-progress-every-s",
})
_FULL_POLICY_FLAGS = frozenset({
    "--phase0-pilot-max-records",
    "--phase1-shortlist-size",
    "--phase2-shortlist-fraction",
    "--phase2-shortlist-size",
    "--adapt-window-size",
    "--adapt-window-topk",
    "--phase3-geometry-window-size",
    "--phase2-w-shot",
    "--phase2-rho",
    "--phase2-batch-target-size",
    "--phase3-batch-target-size",
    "--phase2-batch-size-cap",
    "--phase3-batch-size-cap",
    "--phase2-batch-near-degenerate-ratio",
    "--phase3-batch-near-degenerate-ratio",
    "--phase2-batch-rank-rel-tol",
    "--phase3-batch-rank-rel-tol",
    "--phase2-batch-additivity-tol",
    "--phase3-batch-additivity-tol",
    "--phase3-batch-order-selection-mode",
    "--phase3-batch-order-max-permutations",
    "--phase2-frontier-ratio",
    "--phase3-frontier-ratio",
    "--phase3-tie-beam-score-ratio",
    "--phase3-tie-beam-abs-tol",
    "--phase3-tie-beam-max-branches",
    "--phase1-prune-mode",
    "--phase1-prune-fraction",
    "--phase1-prune-min-candidates",
    "--phase1-prune-max-candidates",
    "--phase1-prune-max-regression",
    "--phase1-prune-tolerance-mode",
    "--phase1-prune-tolerance-shot-coeff",
    "--phase1-prune-tolerance-screen-coeff",
    "--phase1-prune-tolerance-chem",
    "--phase1-prune-tolerance-rel-coeff",
    "--phase1-prune-retained-gain-ratio",
    "--phase1-prune-protect-steps",
    "--phase1-prune-stale-age",
    "--phase1-prune-stagnation-threshold",
    "--phase1-prune-small-theta-abs",
    "--phase1-prune-small-theta-relative",
    "--phase1-prune-cooldown-steps",
    "--phase1-prune-local-window-size",
    "--phase1-prune-recovery-trust-radius",
    "--phase1-prune-old-fraction",
    "--phase1-prune-checkpoint-period",
    "--phase1-prune-live-min-depth",
    "--phase1-prune-maturity-threshold",
    "--phase1-prune-snr-threshold",
    "--phase1-prune-collapse-peak-abs-min",
    "--phase1-prune-collapse-current-abs-max",
    "--phase1-prune-collapse-ratio",
    "--phase1-prune-collapse-min-abs-drop",
    "--phase1-prune-collapse-min-observations",
    "--phase3-backend-w-2q",
    "--phase3-backend-w-depth",
    "--phase3-backend-w-size",
    "--adapt-maxiter",
    "--adapt-final-refit-maxiter",
    *_SPSA_PROFILE_FLAGS,
})

_MATURITY_SHORTLIST_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "cap_ramp_narrow": (
        ("--phase1-shortlist-size", 32),
        ("--phase2-shortlist-fraction", 0.5),
        ("--phase2-shortlist-size", 24),
        ("--phase1-maturity-cap-min", 12),
        ("--phase1-maturity-cap-max", 32),
        ("--phase2-maturity-cap-min", 8),
        ("--phase2-maturity-cap-max", 24),
        ("--phase3-maturity-cap-min", 4),
        ("--phase3-maturity-cap-max", 16),
    ),
    "cap_ramp_medium": (
        ("--phase1-shortlist-size", 64),
        ("--phase2-shortlist-fraction", 0.75),
        ("--phase2-shortlist-size", 48),
        ("--phase1-maturity-cap-min", 24),
        ("--phase1-maturity-cap-max", 64),
        ("--phase2-maturity-cap-min", 12),
        ("--phase2-maturity-cap-max", 48),
        ("--phase3-maturity-cap-min", 8),
        ("--phase3-maturity-cap-max", 32),
    ),
    "heavy_full": (
        ("--phase1-shortlist-size", 192),
        ("--phase2-shortlist-fraction", 1.0),
        ("--phase2-shortlist-size", 192),
        ("--phase1-maturity-cap-min", 96),
        ("--phase1-maturity-cap-max", 192),
        ("--phase2-maturity-cap-min", 64),
        ("--phase2-maturity-cap-max", 192),
        ("--phase3-maturity-cap-min", 32),
        ("--phase3-maturity-cap-max", 128),
    ),
    "cap_ramp_tight": (
        ("--phase1-shortlist-size", 24),
        ("--phase2-shortlist-fraction", 0.35),
        ("--phase2-shortlist-size", 16),
        ("--phase1-maturity-cap-min", 8),
        ("--phase1-maturity-cap-max", 24),
        ("--phase2-maturity-cap-min", 6),
        ("--phase2-maturity-cap-max", 16),
        ("--phase3-maturity-cap-min", 4),
        ("--phase3-maturity-cap-max", 12),
    ),
    "cap_ramp_micro": (
        ("--phase1-shortlist-size", 12),
        ("--phase2-shortlist-fraction", 0.20),
        ("--phase2-shortlist-size", 8),
        ("--phase1-maturity-cap-min", 4),
        ("--phase1-maturity-cap-max", 12),
        ("--phase2-maturity-cap-min", 3),
        ("--phase2-maturity-cap-max", 8),
        ("--phase3-maturity-cap-min", 2),
        ("--phase3-maturity-cap-max", 6),
    ),
}
_PHASE0_PILOT_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "pilot_64": (("--phase0-pilot-max-records", 64),),
    "pilot_32": (("--phase0-pilot-max-records", 32),),
    "pilot_24": (("--phase0-pilot-max-records", 24),),
    "pilot_16": (("--phase0-pilot-max-records", 16),),
    "pilot_10": (("--phase0-pilot-max-records", 10),),
}
_PHASE0_PILOT_RECORDS_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "pilot_10": (("--phase0-pilot-max-records", 10),),
    "pilot_16": (("--phase0-pilot-max-records", 16),),
    "pilot_24": (("--phase0-pilot-max-records", 24),),
    "pilot_32": (("--phase0-pilot-max-records", 32),),
    "pilot_48": (("--phase0-pilot-max-records", 48),),
    "pilot_64": (("--phase0-pilot-max-records", 64),),
}
_PHASE1_SHORTLIST_SIZE_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "p1_8": (("--phase1-shortlist-size", 8),),
    "p1_10": (("--phase1-shortlist-size", 10),),
    "p1_12": (("--phase1-shortlist-size", 12),),
    "p1_16": (("--phase1-shortlist-size", 16),),
    "p1_20": (("--phase1-shortlist-size", 20),),
    "p1_24": (("--phase1-shortlist-size", 24),),
    "p1_32": (("--phase1-shortlist-size", 32),),
    "p1_48": (("--phase1-shortlist-size", 48),),
    "p1_64": (("--phase1-shortlist-size", 64),),
}
_PHASE2_SHORTLIST_FRACTION_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "p2frac_0p10": (("--phase2-shortlist-fraction", 0.10),),
    "p2frac_0p15": (("--phase2-shortlist-fraction", 0.15),),
    "p2frac_0p20": (("--phase2-shortlist-fraction", 0.20),),
    "p2frac_0p25": (("--phase2-shortlist-fraction", 0.25),),
    "p2frac_0p35": (("--phase2-shortlist-fraction", 0.35),),
    "p2frac_0p50": (("--phase2-shortlist-fraction", 0.50),),
    "p2frac_0p75": (("--phase2-shortlist-fraction", 0.75),),
    "p2frac_1p00": (("--phase2-shortlist-fraction", 1.00),),
}
_PHASE2_SHORTLIST_SIZE_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "p2_4": (("--phase2-shortlist-size", 4),),
    "p2_6": (("--phase2-shortlist-size", 6),),
    "p2_8": (("--phase2-shortlist-size", 8),),
    "p2_10": (("--phase2-shortlist-size", 10),),
    "p2_12": (("--phase2-shortlist-size", 12),),
    "p2_16": (("--phase2-shortlist-size", 16),),
    "p2_24": (("--phase2-shortlist-size", 24),),
    "p2_32": (("--phase2-shortlist-size", 32),),
    "p2_48": (("--phase2-shortlist-size", 48),),
}
_MATURITY_SHOT_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "late_x2": (
        ("--phase-maturity-shot-min", 1),
        ("--phase-maturity-shot-max", 2),
        ("--phase1-maturity-shot-cap", 2),
        ("--phase2-maturity-shot-cap", 2),
        ("--phase3-maturity-shot-cap", 2),
    ),
    "late_x4_phase23": (
        ("--phase-maturity-shot-min", 1),
        ("--phase-maturity-shot-max", 4),
        ("--phase1-maturity-shot-cap", 2),
        ("--phase2-maturity-shot-cap", 4),
        ("--phase3-maturity-shot-cap", 4),
    ),
    "heavy_x8_phase23": (
        ("--phase-maturity-shot-min", 2),
        ("--phase-maturity-shot-max", 8),
        ("--phase1-maturity-shot-cap", 4),
        ("--phase2-maturity-shot-cap", 8),
        ("--phase3-maturity-shot-cap", 8),
    ),
    "cheap_x1": (
        ("--phase-maturity-shot-min", 1),
        ("--phase-maturity-shot-max", 1),
        ("--phase1-maturity-shot-cap", 1),
        ("--phase2-maturity-shot-cap", 1),
        ("--phase3-maturity-shot-cap", 1),
    ),
}
_ADAPT_WINDOW_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "full_heavy": (
        ("--adapt-window-size", 999999),
        ("--adapt-window-topk", 999999),
        ("--phase3-geometry-window-size", 0),
    ),
    "medium_16": (
        ("--adapt-window-size", 16),
        ("--adapt-window-topk", 16),
        ("--phase3-geometry-window-size", 16),
    ),
    "tight_8": (
        ("--adapt-window-size", 8),
        ("--adapt-window-topk", 8),
        ("--phase3-geometry-window-size", 8),
    ),
    "cheap_4": (
        ("--adapt-window-size", 4),
        ("--adapt-window-topk", 4),
        ("--phase3-geometry-window-size", 4),
    ),
}
_ADAPT_HISTORY_WINDOW_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "hist_4": (("--adapt-window-size", 4), ("--adapt-window-topk", 4)),
    "hist_8": (("--adapt-window-size", 8), ("--adapt-window-topk", 8)),
    "hist_12": (("--adapt-window-size", 12), ("--adapt-window-topk", 12)),
    "hist_16": (("--adapt-window-size", 16), ("--adapt-window-topk", 16)),
    "hist_24": (("--adapt-window-size", 24), ("--adapt-window-topk", 24)),
    "hist_full": (("--adapt-window-size", 999999), ("--adapt-window-topk", 999999)),
}
_GEOMETRY_WINDOW_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "geom_4": (("--phase3-geometry-window-size", 4),),
    "geom_8": (("--phase3-geometry-window-size", 8),),
    "geom_12": (("--phase3-geometry-window-size", 12),),
    "geom_16": (("--phase3-geometry-window-size", 16),),
    "geom_24": (("--phase3-geometry-window-size", 24),),
    "geom_32": (("--phase3-geometry-window-size", 32),),
    "geom_full": (("--phase3-geometry-window-size", 0),),
}
_PHASE2_W_SHOT_PROFILE_OPTIONS: dict[str, float | None] = {
    "base": None,
    "shot_0p15": 0.15,
    "shot_0p30": 0.30,
    "shot_0p50": 0.50,
    "shot_0p75": 0.75,
    "shot_0p08": 0.08,
    "shot_0p04": 0.04,
    "shot_0p02": 0.02,
    # Keep graph+shot Optuna searches cost-aware by default.  A fixed zero
    # shot weight remains available only through the explicit wrapper-level
    # ``--phase2-w-shot 0.0`` ablation/timing override.
}
_PHASE2_RHO_PROFILE_OPTIONS: dict[str, float | None] = {
    "base": None,
    "rho_0p25": 0.25,
    "rho_0p5": 0.5,
    "rho_0p75": 0.75,
}
_BACKEND_COST_WEIGHT_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "marrakesh_balanced": (
        ("--phase3-backend-w-2q", 1.0),
        ("--phase3-backend-w-depth", 0.1),
        ("--phase3-backend-w-size", 0.01),
    ),
    "marrakesh_2q_heavy": (
        ("--phase3-backend-w-2q", 1.5),
        ("--phase3-backend-w-depth", 0.05),
        ("--phase3-backend-w-size", 0.0),
    ),
    "marrakesh_depth_heavy": (
        ("--phase3-backend-w-2q", 1.0),
        ("--phase3-backend-w-depth", 0.25),
        ("--phase3-backend-w-size", 0.01),
    ),
}
_PRUNE_WITNESS_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "relaxed_required": (
        ("--phase1-prune-amplitude-witness-required", None),
        ("--phase1-prune-collapse-peak-abs-min", "5e-4"),
        ("--phase1-prune-collapse-current-abs-max", "2e-3"),
        ("--phase1-prune-collapse-ratio", 0.5),
        ("--phase1-prune-collapse-min-abs-drop", "5e-4"),
        ("--phase1-prune-collapse-min-observations", 2),
    ),
    "strict_required": (
        ("--phase1-prune-amplitude-witness-required", None),
        ("--phase1-prune-collapse-peak-abs-min", "2e-3"),
        ("--phase1-prune-collapse-current-abs-max", "5e-4"),
        ("--phase1-prune-collapse-ratio", 0.2),
        ("--phase1-prune-collapse-min-abs-drop", "2e-3"),
        ("--phase1-prune-collapse-min-observations", 4),
    ),
}
_PRUNE_PREFILTER_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "strict_motif": (
        ("--phase1-prune-prefilter-policy", "motif_risk_v1"),
        ("--phase1-prune-risk-threshold", 0.0),
        ("--phase1-prune-prefilter-max-candidates", 1),
    ),
    "strict_motif_cap2": (
        ("--phase1-prune-prefilter-policy", "motif_risk_v1"),
        ("--phase1-prune-risk-threshold", 0.0),
        ("--phase1-prune-prefilter-max-candidates", 2),
    ),
    "strict_motif_cap3": (
        ("--phase1-prune-prefilter-policy", "motif_risk_v1"),
        ("--phase1-prune-risk-threshold", 0.0),
        ("--phase1-prune-prefilter-max-candidates", 3),
    ),
}
_SPSA_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    # Current local search default, kept as a first-class sampled option.
    "current": (
        ("--adapt-spsa-a", _SEARCH_SPSA_A),
        ("--adapt-spsa-c", _SEARCH_SPSA_C),
        ("--adapt-spsa-alpha", _SEARCH_SPSA_ALPHA),
        ("--adapt-spsa-gamma", _SEARCH_SPSA_GAMMA),
        ("--adapt-spsa-A", _SEARCH_SPSA_A_SHIFT),
        ("--adapt-spsa-avg-last", _SEARCH_SPSA_AVG_LAST),
        ("--adapt-spsa-eval-repeats", _SEARCH_SPSA_EVAL_REPEATS),
        ("--adapt-spsa-eval-agg", _SEARCH_SPSA_EVAL_AGG),
        ("--adapt-spsa-callback-every", _SEARCH_SPSA_CALLBACK_EVERY),
        ("--adapt-spsa-progress-every-s", _SEARCH_SPSA_PROGRESS_EVERY_S),
    ),
    # Observed in Paper-I HH SNAKE strong/intermediate-weak and strong/intermediate-strong source rows.
    "paper_i_strong_like": (
        ("--adapt-spsa-a", 0.08029189973696281),
        ("--adapt-spsa-c", 0.011978122654460147),
        ("--adapt-spsa-alpha", 0.6102106236980185),
        ("--adapt-spsa-gamma", 0.10164107284550583),
        ("--adapt-spsa-A", 37.00879952613117),
        ("--adapt-spsa-avg-last", 0),
        ("--adapt-spsa-eval-repeats", 1),
        ("--adapt-spsa-eval-agg", "mean"),
        ("--adapt-spsa-callback-every", 5),
        ("--adapt-spsa-progress-every-s", 60.0),
    ),
    # Observed in Paper-I HH SNAKE weak-strong source rows.
    "paper_i_weak_strong_like": (
        ("--adapt-spsa-a", 0.009245844022426742),
        ("--adapt-spsa-c", 0.03519712840066804),
        ("--adapt-spsa-alpha", 0.6407619010007662),
        ("--adapt-spsa-gamma", 0.0910651683048572),
        ("--adapt-spsa-A", 71.83989228215009),
        ("--adapt-spsa-avg-last", 0),
        ("--adapt-spsa-eval-repeats", 1),
        ("--adapt-spsa-eval-agg", "mean"),
        ("--adapt-spsa-callback-every", 5),
        ("--adapt-spsa-progress-every-s", 60.0),
    ),
}
_ML_CANDIDATE_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    # p10 of positive-row candidate ranges from
    # output/paper_i_hh_snake_routea_interpretable_ml_analysis_20260614/analysis_summary.json
    "ml_p10": (
        ("--phase1-prune-fraction", 0.1930961457788297),
        ("--phase2-batch-near-degenerate-ratio", 0.914354284671342),
        ("--phase3-batch-near-degenerate-ratio", 0.914354284671342),
        ("--phase2-batch-rank-rel-tol", 7.703203666118798e-07),
        ("--phase3-batch-rank-rel-tol", 7.703203666118798e-07),
        ("--phase2-batch-additivity-tol", 0.010276490515218235),
        ("--phase3-batch-additivity-tol", 0.010276490515218235),
    ),
    # Medians of the same positive-row distributions.
    "ml_median": (
        ("--phase1-prune-fraction", 0.33922934316592934),
        ("--phase2-batch-near-degenerate-ratio", 0.98),
        ("--phase3-batch-near-degenerate-ratio", 0.98),
        ("--phase2-batch-rank-rel-tol", 1.909930091607197e-05),
        ("--phase3-batch-rank-rel-tol", 1.909930091607197e-05),
        ("--phase2-batch-additivity-tol", 0.09993123296803053),
        ("--phase3-batch-additivity-tol", 0.09993123296803053),
    ),
    # p90 of positive-row candidate ranges.
    "ml_p90": (
        ("--phase1-prune-fraction", 0.4101910583864897),
        ("--phase2-batch-near-degenerate-ratio", 0.9982411735035968),
        ("--phase3-batch-near-degenerate-ratio", 0.9982411735035968),
        ("--phase2-batch-rank-rel-tol", 0.00013662376421438911),
        ("--phase3-batch-rank-rel-tol", 0.00013662376421438911),
        ("--phase2-batch-additivity-tol", 0.6663130343903237),
        ("--phase3-batch-additivity-tol", 0.6663130343903237),
    ),
}
_PHASE1_PRUNE_FRACTION_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "prune_frac_0p10": (("--phase1-prune-fraction", 0.10),),
    "prune_frac_0p15": (("--phase1-prune-fraction", 0.15),),
    "prune_frac_0p20": (("--phase1-prune-fraction", 0.20),),
    "prune_frac_0p25": (("--phase1-prune-fraction", 0.25),),
    "prune_frac_0p34": (("--phase1-prune-fraction", 0.33922934316592934),),
    "prune_frac_0p41": (("--phase1-prune-fraction", 0.4101910583864897),),
    "prune_frac_0p50": (("--phase1-prune-fraction", 0.50),),
}
_BATCH_NEAR_DEGENERATE_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "near_0p90": (("--phase2-batch-near-degenerate-ratio", 0.90), ("--phase3-batch-near-degenerate-ratio", 0.90)),
    "near_0p95": (("--phase2-batch-near-degenerate-ratio", 0.95), ("--phase3-batch-near-degenerate-ratio", 0.95)),
    "near_0p98": (("--phase2-batch-near-degenerate-ratio", 0.98), ("--phase3-batch-near-degenerate-ratio", 0.98)),
    "near_0p995": (("--phase2-batch-near-degenerate-ratio", 0.995), ("--phase3-batch-near-degenerate-ratio", 0.995)),
    "near_0p998": (("--phase2-batch-near-degenerate-ratio", 0.9982411735035968), ("--phase3-batch-near-degenerate-ratio", 0.9982411735035968)),
}
_BATCH_RANK_TOL_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "ranktol_1e_6": (("--phase2-batch-rank-rel-tol", 1e-6), ("--phase3-batch-rank-rel-tol", 1e-6)),
    "ranktol_7e_7": (("--phase2-batch-rank-rel-tol", 7.703203666118798e-7), ("--phase3-batch-rank-rel-tol", 7.703203666118798e-7)),
    "ranktol_2e_5": (("--phase2-batch-rank-rel-tol", 1.909930091607197e-5), ("--phase3-batch-rank-rel-tol", 1.909930091607197e-5)),
    "ranktol_1e_4": (("--phase2-batch-rank-rel-tol", 1e-4), ("--phase3-batch-rank-rel-tol", 1e-4)),
    "ranktol_1p4e_4": (("--phase2-batch-rank-rel-tol", 1.3662376421438911e-4), ("--phase3-batch-rank-rel-tol", 1.3662376421438911e-4)),
}
_BATCH_ADDITIVITY_TOL_PROFILE_OPTIONS: dict[str, tuple[tuple[str, str | int | float | None], ...]] = {
    "base": (),
    "addtol_0p01": (("--phase2-batch-additivity-tol", 0.01), ("--phase3-batch-additivity-tol", 0.01)),
    "addtol_0p03": (("--phase2-batch-additivity-tol", 0.03), ("--phase3-batch-additivity-tol", 0.03)),
    "addtol_0p10": (("--phase2-batch-additivity-tol", 0.10), ("--phase3-batch-additivity-tol", 0.10)),
    "addtol_0p30": (("--phase2-batch-additivity-tol", 0.30), ("--phase3-batch-additivity-tol", 0.30)),
    "addtol_0p66": (("--phase2-batch-additivity-tol", 0.6663130343903237), ("--phase3-batch-additivity-tol", 0.6663130343903237)),
}
_FULL_POLICY_PARAM_OPTIONS: dict[str, list[str]] = {
    # Explicit HH Route-A policy grid.  Ranges mirror the validated knobs in
    # phase3_policy_optuna; values are categorical here so persistent Optuna
    # studies remain stable and external enqueue manifests can be normalized.
    "full_phase0_pilot_max_records": ["base", "10", "16", "24", "32", "48", "64"],
    "full_phase1_shortlist_size": ["base", "8", "10", "12", "16", "20", "24", "32", "48", "64"],
    "full_phase2_shortlist_fraction": [
        "base",
        "0.10",
        "0.15",
        "0.20",
        "0.25",
        "0.35",
        "0.50",
        "0.75",
        "1.00",
    ],
    "full_phase2_shortlist_size": ["base", "4", "6", "8", "10", "12", "16", "24", "32", "48"],
    "full_adapt_window_size": ["base", "4", "8", "12", "16", "24", "32", "999999"],
    "full_phase3_geometry_window_size": ["base", "0", "4", "8", "12", "16", "24", "32"],
    "full_phase2_w_shot": ["base", "0.02", "0.04", "0.08", "0.15", "0.30", "0.50", "0.75"],
    "full_phase2_rho": ["base", "0.25", "0.50", "0.75"],
    "full_phase2_batch_target_size": ["base", "2", "3", "4", "6", "8", "12", "16"],
    "full_phase2_batch_size_cap": ["base", "4", "6", "8", "12", "16", "24", "32"],
    "full_phase3_batch_target_size": ["base", "2", "3", "4", "6", "8", "12", "16"],
    "full_phase3_batch_size_cap": ["base", "4", "6", "8", "12", "16", "24", "32"],
    "full_batch_near_degenerate_ratio": ["base", "0.90", "0.95", "0.98", "0.995", "0.9982411735035968"],
    "full_batch_rank_rel_tol": ["base", "1e-9", "7.703203666118798e-7", "1e-6", "1.909930091607197e-5", "1e-4", "0.00013662376421438911", "1e-3"],
    "full_batch_additivity_tol": ["base", "0.001", "0.01", "0.03", "0.10", "0.30", "0.6663130343903237", "1.0"],
    "full_phase3_batch_order_selection_mode": ["base", "finite_step_v1", "score_sorted"],
    "full_phase3_batch_order_max_permutations": ["base", "12", "24", "48", "96"],
    "full_phase2_frontier_ratio": ["base", "0.50", "0.75", "0.90", "1.00"],
    "full_phase3_frontier_ratio": ["base", "0.50", "0.75", "0.90", "1.00"],
    "full_phase3_tie_beam_score_ratio": ["base", "1.00", "1.02", "1.05", "1.10"],
    "full_phase3_tie_beam_abs_tol": ["base", "0.0", "1e-8", "1e-6", "1e-4", "1e-3"],
    "full_phase3_tie_beam_max_branches": ["base", "1", "2", "3", "5"],
    "full_phase1_prune_mode": ["base", "live", "final", "both"],
    "full_phase1_prune_fraction": ["base", "0.05", "0.10", "0.15", "0.1930961457788297", "0.25", "0.33922934316592934", "0.4101910583864897", "0.50"],
    "full_phase1_prune_min_candidates": ["base", "1", "2", "3"],
    "full_phase1_prune_max_candidates": ["base", "2", "4", "6", "8", "10"],
    "full_phase1_prune_max_regression": ["base", "1e-10", "1e-9", "1e-8", "1e-7", "1e-6"],
    "full_phase1_prune_tolerance_mode": ["base", "auto", "fixed", "adaptive_v1"],
    "full_phase1_prune_tolerance_shot_coeff": ["base", "0.0", "0.5", "1.0", "2.0", "5.0"],
    "full_phase1_prune_tolerance_screen_coeff": ["base", "0.0001", "0.001", "0.01", "0.05", "0.10"],
    "full_phase1_prune_tolerance_chem": ["base", "0.0", "1e-8", "1e-6", "1e-4"],
    "full_phase1_prune_tolerance_rel_coeff": ["base", "0.0", "0.05", "0.10", "0.25"],
    "full_phase1_prune_retained_gain_ratio": ["base", "0.10", "0.25", "0.50", "0.60"],
    "full_phase1_prune_protect_steps": ["base", "1", "2", "3", "4", "5"],
    "full_phase1_prune_stale_age": ["base", "1", "2", "3", "4", "6"],
    "full_phase1_prune_stagnation_threshold": ["base", "0.0", "1e-6", "1e-5", "1e-4"],
    "full_phase1_prune_small_theta_abs": ["base", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"],
    "full_phase1_prune_small_theta_relative": ["base", "0.0", "0.25", "0.50", "0.75", "1.0"],
    "full_phase1_prune_cooldown_steps": ["base", "0", "1", "2", "4", "8", "16"],
    "full_phase1_prune_local_window_size": ["base", "1", "2", "4", "8", "16", "32"],
    "full_phase1_prune_recovery_trust_radius": ["base", "0.0", "0.10", "0.25", "0.50", "1.0"],
    "full_phase1_prune_old_fraction": ["base", "0.0", "0.25", "0.50", "0.75", "1.0"],
    "full_phase1_prune_checkpoint_period": ["base", "2", "3", "4", "6"],
    "full_phase1_prune_live_min_depth": ["base", "0", "4", "8", "12", "16"],
    "full_phase1_prune_maturity_threshold": ["base", "0.35", "0.50", "0.65", "0.80"],
    "full_phase1_prune_snr_threshold": ["base", "0.0", "0.5", "1.0", "2.0", "3.0"],
    "full_phase1_prune_collapse_peak_abs_min": ["base", "1e-5", "1e-4", "1e-3", "1e-2"],
    "full_phase1_prune_collapse_current_abs_max": ["base", "1e-6", "1e-4", "1e-3", "1e-2"],
    "full_phase1_prune_collapse_ratio": ["base", "0.05", "0.25", "0.50", "0.75", "0.95"],
    "full_phase1_prune_collapse_min_abs_drop": ["base", "1e-6", "1e-4", "1e-3", "1e-2"],
    "full_phase1_prune_collapse_min_observations": ["base", "2", "3", "4", "6"],
    "full_phase3_backend_w_2q": ["base", "0.5", "1.0", "1.5", "2.0"],
    "full_phase3_backend_w_depth": ["base", "0.0", "0.05", "0.1", "0.25"],
    "full_phase3_backend_w_size": ["base", "0.0", "0.01", "0.05"],
    "full_spsa_maxiter": ["base", "100", "200", "400", "800", "1200", "2000"],
    "full_spsa_a": ["base", "0.009245844022426742", "0.02", "0.05", "0.08029189973696281", "0.1", "0.2"],
    "full_spsa_c": ["base", "0.011978122654460147", "0.02", "0.03519712840066804", "0.05", "0.1"],
    "full_spsa_alpha": ["base", "0.602", "0.6102106236980185", "0.6407619010007662"],
    "full_spsa_gamma": ["base", "0.0910651683048572", "0.101", "0.10164107284550583"],
    "full_spsa_A": ["base", "5.0", "10.0", "37.00879952613117", "71.83989228215009"],
    "full_spsa_avg_last": ["base", "0", "5", "10"],
    "full_spsa_eval_repeats": ["base", "1", "2", "4"],
    "full_spsa_callback_every": ["base", "5", "10", "20"],
}


@dataclass(frozen=True)
class BasePreset:
    name: str
    lane_tags: tuple[str, ...]
    launcher_tokens: tuple[str, ...]
    pipeline_args: tuple[str, ...]
    env_overrides: tuple[tuple[str, str], ...] = ()
    source_artifact_dir: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class TrialParams:
    base_preset: str
    adapt_max_depth: int
    selector_geometry_mode: str = "base"
    runtime_split_mode: str = "base"
    batching_mode: str = "base"
    repeats_mode: str = "base"
    selection_cost_mode: str = "base"
    motif_mode: str = "base"
    phase1_prune_mode: str = "base"
    phase0_pilot_profile: str = "base"
    phase0_pilot_records_profile: str = "base"
    maturity_shortlist_profile: str = "base"
    phase1_shortlist_size_profile: str = "base"
    phase2_shortlist_fraction_profile: str = "base"
    phase2_shortlist_size_profile: str = "base"
    maturity_shot_profile: str = "base"
    phase_live_profile: str = "base"
    prune_witness_profile: str = "base"
    prune_prefilter_profile: str = "base"
    adapt_window_profile: str = "base"
    adapt_history_window_profile: str = "base"
    geometry_window_profile: str = "base"
    backend_cost_weight_profile: str = "base"
    phase2_w_shot_profile: str = "base"
    phase2_rho_profile: str = "base"
    spsa_profile: str = "current"
    ml_candidate_profile: str = "base"
    phase1_prune_fraction_profile: str = "base"
    batch_near_degenerate_profile: str = "base"
    batch_rank_tol_profile: str = "base"
    batch_additivity_tol_profile: str = "base"
    full_phase0_pilot_max_records: str = "base"
    full_phase1_shortlist_size: str = "base"
    full_phase2_shortlist_fraction: str = "base"
    full_phase2_shortlist_size: str = "base"
    full_adapt_window_size: str = "base"
    full_phase3_geometry_window_size: str = "base"
    full_phase2_w_shot: str = "base"
    full_phase2_rho: str = "base"
    full_phase2_batch_target_size: str = "base"
    full_phase2_batch_size_cap: str = "base"
    full_phase3_batch_target_size: str = "base"
    full_phase3_batch_size_cap: str = "base"
    full_batch_near_degenerate_ratio: str = "base"
    full_batch_rank_rel_tol: str = "base"
    full_batch_additivity_tol: str = "base"
    full_phase3_batch_order_selection_mode: str = "base"
    full_phase3_batch_order_max_permutations: str = "base"
    full_phase2_frontier_ratio: str = "base"
    full_phase3_frontier_ratio: str = "base"
    full_phase3_tie_beam_score_ratio: str = "base"
    full_phase3_tie_beam_abs_tol: str = "base"
    full_phase3_tie_beam_max_branches: str = "base"
    full_phase1_prune_mode: str = "base"
    full_phase1_prune_fraction: str = "base"
    full_phase1_prune_min_candidates: str = "base"
    full_phase1_prune_max_candidates: str = "base"
    full_phase1_prune_max_regression: str = "base"
    full_phase1_prune_tolerance_mode: str = "base"
    full_phase1_prune_tolerance_shot_coeff: str = "base"
    full_phase1_prune_tolerance_screen_coeff: str = "base"
    full_phase1_prune_tolerance_chem: str = "base"
    full_phase1_prune_tolerance_rel_coeff: str = "base"
    full_phase1_prune_retained_gain_ratio: str = "base"
    full_phase1_prune_protect_steps: str = "base"
    full_phase1_prune_stale_age: str = "base"
    full_phase1_prune_stagnation_threshold: str = "base"
    full_phase1_prune_small_theta_abs: str = "base"
    full_phase1_prune_small_theta_relative: str = "base"
    full_phase1_prune_cooldown_steps: str = "base"
    full_phase1_prune_local_window_size: str = "base"
    full_phase1_prune_recovery_trust_radius: str = "base"
    full_phase1_prune_old_fraction: str = "base"
    full_phase1_prune_checkpoint_period: str = "base"
    full_phase1_prune_live_min_depth: str = "base"
    full_phase1_prune_maturity_threshold: str = "base"
    full_phase1_prune_snr_threshold: str = "base"
    full_phase1_prune_collapse_peak_abs_min: str = "base"
    full_phase1_prune_collapse_current_abs_max: str = "base"
    full_phase1_prune_collapse_ratio: str = "base"
    full_phase1_prune_collapse_min_abs_drop: str = "base"
    full_phase1_prune_collapse_min_observations: str = "base"
    full_phase3_backend_w_2q: str = "base"
    full_phase3_backend_w_depth: str = "base"
    full_phase3_backend_w_size: str = "base"
    full_spsa_maxiter: str = "base"
    full_spsa_a: str = "base"
    full_spsa_c: str = "base"
    full_spsa_alpha: str = "base"
    full_spsa_gamma: str = "base"
    full_spsa_A: str = "base"
    full_spsa_avg_last: str = "base"
    full_spsa_eval_repeats: str = "base"
    full_spsa_callback_every: str = "base"


_SHORTLIST_REFINE_FREE_PARAM_NAMES = frozenset(
    {
        # Keep the depth cap controlled by the wrapper/regime launch surface.
        "adapt_max_depth",
        # For the S_alg-focused refinement lane, keep Paper-I maturity-shot
        # extensions off even when the prior candidate used them.
        "maturity_shot_profile",
        "prune_prefilter_profile",
        # Narrowed lane degrees of freedom: shortlist breadth, geometry window,
        # and score/selection thresholds.  All other knobs are anchored to the
        # per-regime candidate prior rows when an enqueue manifest is supplied.
        "phase0_pilot_records_profile",
        "phase1_shortlist_size_profile",
        "phase2_shortlist_fraction_profile",
        "phase2_shortlist_size_profile",
        "adapt_history_window_profile",
        "geometry_window_profile",
        "phase1_prune_fraction_profile",
        "batch_near_degenerate_profile",
        "batch_rank_tol_profile",
        "batch_additivity_tol_profile",
        "phase2_w_shot_profile",
        "phase2_rho_profile",
    }
)


@dataclass(frozen=True)
class TrialObservation:
    lane: str
    epsilon_abs_delta_e: float
    params: dict[str, Any]
    objective_lexicographic: int
    abs_delta_e: float | None
    compiled_count_2q: int | None
    compiled_depth: int | None
    logical_operator_count: int | None
    runtime_parameter_count: int | None
    feasible: bool
    constraints: list[float]
    graph_count_2q: float | None = None
    graph_depth: float | None = None
    graph_count_1q: float | None = None
    graph_theta_count: float | None = None
    measurement_work_shots: float | None = None
    paper_i_table_shots_total: float | None = None
    paper_i_table_s_alg: float | None = None
    paper_i_table_shots_status: str | None = None
    paper_i_table_shots_scope: str | None = None
    paper_i_table_shots_history_position: int | None = None
    measurement_work_records: float | None = None
    resource_cost_source: str | None = None
    prune_candidate_count: int | None = None
    prune_accepted_count: int | None = None
    prune_rejected_delete_attempt_count: int | None = None
    prune_no_accept_restore_pass_count: int | None = None
    prune_accepted_then_guard_rolled_back_count: int | None = None
    prune_actual_rollback_count: int | None = None
    prune_prefilter_blocked_count: int | None = None
    prune_prefilter_allowed_count: int | None = None
    adapt_iteration_count: int | None = None
    dominance_target_abs_delta_e: float | None = None
    dominance_target_iteration: int | None = None
    dominance_target_graph_count_2q: float | None = None
    dominance_target_graph_depth: float | None = None
    dominance_target_s_alg: float | None = None
    dominance_prefix_abs_delta_e: float | None = None
    dominance_prefix_iteration: int | None = None
    dominance_first_crossing_iteration: int | None = None
    dominance_energy_violation: float | None = None
    dominance_iteration_violation: float | None = None
    dominance_graph_count_2q_violation: float | None = None
    dominance_graph_depth_violation: float | None = None
    dominance_s_alg_violation: float | None = None
    case_dir: str | None = None
    result_json: str | None = None
    compile_json: str | None = None
    returncode: int | None = None
    compile_returncode: int | None = None
    pipeline_elapsed_s: float | None = None
    compile_elapsed_s: float | None = None
    total_elapsed_s: float | None = None
    dropped_args: list[str] = field(default_factory=list)
    base_preset: str | None = None
    family_path_signature: list[str] = field(default_factory=list)
    selected_op_signature: list[str] = field(default_factory=list)
    source_artifact_dir: str | None = None
    warm_start: bool = False
    hamiltonian_overrides: dict[str, Any] = field(default_factory=dict)
    invalid_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HhHamiltonianOverrides:
    """Optional HH/cutoff override layer for replaying route parameters at a new physics point."""

    L: int | None = None
    t: float | None = None
    u: float | None = None
    omega0: float | None = None
    lambda_value: float | None = None
    g_ep: float | None = None
    n_ph_work: int | None = None
    n_ph_ref: int | None = None
    adapt_pool: str | None = None

    @property
    def active(self) -> bool:
        return any(value is not None for value in asdict(self).values())


@dataclass(frozen=True)
class StudySpec:
    lane: str
    epsilon_abs_delta_e: float
    n_trials: int
    n_startup_trials: int
    extra_warm_start_preset_names: tuple[str, ...] = ()
    enqueue_preset_names: tuple[str, ...] = ()
    restricted_base_preset_names: tuple[str, ...] = ()


def _preset_is_legacy(preset: BasePreset | None) -> bool:
    if preset is None:
        return False
    return ("legacy" in preset.lane_tags) or any("legacy_20260322" in tok for tok in preset.launcher_tokens)


def _normalize_launcher_tokens(
    name: str,
    lane_tags: Sequence[str],
    launcher_tokens: Sequence[str],
) -> tuple[str, ...]:
    if ("legacy" in {str(x) for x in lane_tags}) or str(name).startswith("legacy_"):
        return tuple(str(x) for x in launcher_tokens)
    normalized: list[str] = []
    for token in launcher_tokens:
        tok = str(token)
        if tok == _COMPAT_ADAPT_MODULE:
            normalized.append(_CANONICAL_ADAPT_MODULE)
        elif tok == _COMPAT_ADAPT_SCRIPT:
            normalized.append(_CANONICAL_ADAPT_SCRIPT)
        elif tok.endswith(_COMPAT_ADAPT_SCRIPT):
            normalized.append(tok[: -len(_COMPAT_ADAPT_SCRIPT)] + _CANONICAL_ADAPT_SCRIPT)
        else:
            normalized.append(tok)
    return tuple(normalized)


def _import_optuna() -> Any:
    try:
        import optuna  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Optuna is not installed. Install it before running this harness, for example via `python -m pip install optuna`."
        ) from exc
    return optuna


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unnamed"


def _float_slug(value: float) -> str:
    return _safe_slug(f"{float(value):.3e}".replace("+", "").replace("-", "m"))


def _strip_redirections(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(tokens):
        tok = str(tokens[idx])
        if tok in {">", "1>", "2>", ">>", "1>>", "2>>"}:
            break
        out.append(tok)
        idx += 1
    return out


_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def _shell_expand_text(text: str, variables: Mapping[str, str]) -> str:
    env = {**{str(k): str(v) for k, v in variables.items()}, "PWD": str(REPO_ROOT)}

    def _replace(match: re.Match[str]) -> str:
        braced = match.group(1)
        plain = match.group(2)
        name = str(braced or plain or "")
        return str(env.get(name, match.group(0)))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", _replace, str(text))


def _is_python_token(token: str) -> bool:
    raw = Path(str(token)).name
    return bool(re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", raw))


def _extract_command_line_from_script(command_sh: Path) -> tuple[dict[str, str], list[str], list[str]]:
    text = command_sh.read_text(encoding="utf-8")
    logical_lines: list[str] = []
    current: list[str] = []
    for raw in text.splitlines():
        line = str(raw).rstrip()
        if not line:
            if current:
                logical_lines.append(" ".join(current).strip())
                current = []
            continue
        if line.endswith("\\"):
            current.append(line[:-1].strip())
            continue
        if current:
            current.append(line.strip())
            logical_lines.append(" ".join(current).strip())
            current = []
        else:
            logical_lines.append(line.strip())
    if current:
        logical_lines.append(" ".join(current).strip())

    shell_vars: dict[str, str] = {"PWD": str(REPO_ROOT)}
    candidate_lines = []
    for line in logical_lines:
        if not line or line.startswith("#") or line.startswith("set ") or line.startswith("cd "):
            continue
        assign_line = str(line)
        if assign_line.startswith("export "):
            assign_line = assign_line[len("export "):].strip()
        assign_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", assign_line)
        if assign_match and "python" not in assign_line:
            key = str(assign_match.group(1))
            raw_value = _shell_expand_text(str(assign_match.group(2)), shell_vars)
            try:
                parsed = shlex.split(raw_value)
                value = parsed[0] if parsed else ""
            except Exception:
                value = raw_value.strip('"').strip("'")
            shell_vars[key] = str(value)
            continue
        if "python" not in line or "--" not in line:
            continue
        candidate_lines.append(_shell_expand_text(line, shell_vars))
    if not candidate_lines:
        raise ValueError(f"No pipeline command found in {command_sh}")
    cmd_tokens = _strip_redirections(shlex.split(candidate_lines[-1]))
    if len(cmd_tokens) < 2:
        raise ValueError(f"Could not parse python launch command from {command_sh}")
    env_overrides: dict[str, str] = {}
    idx = 0
    while idx < len(cmd_tokens) and _ENV_ASSIGN_RE.match(str(cmd_tokens[idx])):
        key, value = str(cmd_tokens[idx]).split("=", 1)
        env_overrides[str(key)] = str(value)
        idx += 1
    if idx < len(cmd_tokens) and str(cmd_tokens[idx]) == "/usr/bin/env":
        idx += 1
        while idx < len(cmd_tokens) and _ENV_ASSIGN_RE.match(str(cmd_tokens[idx])):
            key, value = str(cmd_tokens[idx]).split("=", 1)
            env_overrides[str(key)] = str(value)
            idx += 1
    if idx >= len(cmd_tokens) or not _is_python_token(str(cmd_tokens[idx])):
        raise ValueError(f"Could not locate python launcher in {command_sh}")
    idx += 1
    first_flag_idx = next((pos for pos in range(idx, len(cmd_tokens)) if str(cmd_tokens[pos]).startswith("--")), None)
    if first_flag_idx is None:
        raise ValueError(f"Could not locate pipeline args in {command_sh}")
    launcher_tokens = list(cmd_tokens[idx:first_flag_idx])
    pipeline_args = list(cmd_tokens[first_flag_idx:])
    return env_overrides, launcher_tokens, pipeline_args


def _remove_option(args: Sequence[str], flag: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(args):
        tok = str(args[idx])
        if tok == flag:
            idx += 1
            if idx < len(args) and not str(args[idx]).startswith("--"):
                idx += 1
            continue
        if tok.startswith(flag + "="):
            idx += 1
            continue
        out.append(tok)
        idx += 1
    return out


def _set_option(args: Sequence[str], flag: str, value: str | int | float | None) -> list[str]:
    updated = _remove_option(args, flag)
    if value is None:
        return updated
    return [*updated, str(flag), str(value)]


def _get_option_value(args: Sequence[str], flag: str) -> str | None:
    idx = 0
    while idx < len(args):
        tok = str(args[idx])
        if tok == flag:
            if idx + 1 < len(args) and not str(args[idx + 1]).startswith("--"):
                return str(args[idx + 1])
            return None
        if tok.startswith(flag + "="):
            return str(tok.split("=", 1)[1])
        idx += 1
    return None


def _format_cli_float(value: float) -> str:
    return f"{float(value):.12g}"


def _hh_lambda_to_g_ep(lambda_value: float, *, t: float = 1.0, omega0: float = 1.0) -> float:
    """Convert the Paper-I HH convention lambda=2 g^2/(t omega0) to g."""

    lambda_f = float(lambda_value)
    t_f = float(t)
    omega_f = float(omega0)
    if lambda_f < 0.0:
        raise ValueError("HH lambda must be nonnegative.")
    if t_f <= 0.0 or omega_f <= 0.0:
        raise ValueError("HH lambda conversion requires positive t and omega0.")
    return float(math.sqrt(lambda_f * t_f * omega_f / 2.0))


def _resolved_hh_g_ep(overrides: HhHamiltonianOverrides | None) -> float | None:
    if overrides is None:
        return None
    if overrides.g_ep is not None:
        return float(overrides.g_ep)
    if overrides.lambda_value is None:
        return None
    return _hh_lambda_to_g_ep(
        float(overrides.lambda_value),
        t=float(overrides.t if overrides.t is not None else 1.0),
        omega0=float(overrides.omega0 if overrides.omega0 is not None else 1.0),
    )


def _validate_hh_hamiltonian_overrides(overrides: HhHamiltonianOverrides) -> HhHamiltonianOverrides:
    if not overrides.active:
        return overrides
    if overrides.n_ph_work is not None and int(overrides.n_ph_work) < 0:
        raise ValueError("--n-ph-work must be >= 0.")
    if overrides.n_ph_ref is not None and int(overrides.n_ph_ref) < 0:
        raise ValueError("--n-ph-ref must be >= 0.")
    if (
        overrides.n_ph_work is not None
        and overrides.n_ph_ref is not None
        and int(overrides.n_ph_ref) < int(overrides.n_ph_work)
    ):
        raise ValueError("--n-ph-ref should be >= --n-ph-work for Paper-I cutoff-pair diagnostics.")
    if overrides.lambda_value is not None:
        derived = _hh_lambda_to_g_ep(
            float(overrides.lambda_value),
            t=float(overrides.t if overrides.t is not None else 1.0),
            omega0=float(overrides.omega0 if overrides.omega0 is not None else 1.0),
        )
        if overrides.g_ep is not None and not math.isclose(
            float(overrides.g_ep),
            derived,
            rel_tol=1.0e-6,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "--hh-g-ep is inconsistent with --hh-lambda under lambda=2 g^2/(t omega0). "
                f"Expected approximately {derived:.12g}."
            )
    return overrides


def _hh_hamiltonian_override_payload(overrides: HhHamiltonianOverrides | None) -> dict[str, Any]:
    if overrides is None or not overrides.active:
        return {}
    payload: dict[str, Any] = {
        "active": True,
        "problem": "hh",
        "lambda_convention": "lambda_ep = 2*g_ep^2/(t*omega0)",
    }
    for key in ("L", "t", "u", "omega0", "lambda_value", "n_ph_work", "n_ph_ref", "adapt_pool"):
        value = getattr(overrides, key)
        if value is not None:
            payload[key] = value
    g_ep = _resolved_hh_g_ep(overrides)
    if g_ep is not None:
        payload["g_ep"] = float(g_ep)
        payload["g_ep_source"] = "explicit" if overrides.g_ep is not None else "derived_from_lambda"
    if overrides.n_ph_work is not None:
        payload["working_cutoff_flag"] = "--n-ph-max"
    if overrides.n_ph_ref is not None:
        payload["reference_cutoff_role"] = "metadata_only_for_posthoc_reference_sidecar"
    return payload


def _apply_hh_hamiltonian_overrides(
    pipeline_args: Sequence[str],
    overrides: HhHamiltonianOverrides | None,
) -> list[str]:
    if overrides is None or not overrides.active:
        return list(str(x) for x in pipeline_args)
    _validate_hh_hamiltonian_overrides(overrides)
    args = list(str(x) for x in pipeline_args)
    args = _set_option(args, "--problem", "hh")
    if overrides.L is not None:
        args = _set_option(args, "--L", int(overrides.L))
    if overrides.t is not None:
        args = _set_option(args, "--t", _format_cli_float(float(overrides.t)))
    if overrides.u is not None:
        args = _set_option(args, "--u", _format_cli_float(float(overrides.u)))
    if overrides.omega0 is not None:
        args = _set_option(args, "--omega0", _format_cli_float(float(overrides.omega0)))
    g_ep = _resolved_hh_g_ep(overrides)
    if g_ep is not None:
        args = _set_option(args, "--g-ep", _format_cli_float(float(g_ep)))
    if overrides.n_ph_work is not None:
        args = _set_option(args, "--n-ph-max", int(overrides.n_ph_work))
    if overrides.adapt_pool is not None:
        args = _set_option(args, "--adapt-pool", str(overrides.adapt_pool))
    return args


def _set_toggle_pair(args: Sequence[str], positive_flag: str, negative_flag: str, enabled: bool) -> list[str]:
    updated = _remove_option(_remove_option(args, positive_flag), negative_flag)
    return [*updated, str(positive_flag if enabled else negative_flag)]


def _remove_options(args: Sequence[str], flags: Iterable[str]) -> list[str]:
    out = list(str(x) for x in args)
    for flag in sorted({str(x) for x in flags}):
        out = _remove_option(out, flag)
    return out


def _apply_option_profile(
    args: Sequence[str],
    options: Sequence[tuple[str, str | int | float | None]],
) -> list[str]:
    out = list(str(x) for x in args)
    for flag, value in options:
        if value is None:
            out = _remove_option(out, str(flag))
            out = [*out, str(flag)]
        else:
            out = _set_option(out, str(flag), value)
    return out


def _profile_options(
    profile_name: str,
    mapping: Mapping[str, tuple[tuple[str, str | int | float | None], ...]],
    *,
    field_name: str,
) -> tuple[tuple[str, str | int | float | None], ...]:
    key = str(profile_name)
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"Unknown {field_name}: {key!r}") from exc


@lru_cache(maxsize=None)
def _supported_long_options(
    python_bin: str,
    launcher_tokens: tuple[str, ...],
    env_overrides: tuple[tuple[str, str], ...] = (),
) -> set[str]:
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in env_overrides})
    result = subprocess.run(
        [str(python_bin), *list(launcher_tokens), "--help"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    return set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", help_text))


def _filter_args_for_entrypoint(
    python_bin: str,
    launcher_tokens: Sequence[str],
    pipeline_args: Sequence[str],
    env_overrides: Sequence[tuple[str, str]] = (),
) -> tuple[list[str], list[str]]:
    supported = _supported_long_options(
        str(python_bin),
        tuple(str(x) for x in launcher_tokens),
        tuple((str(k), str(v)) for k, v in env_overrides),
    )
    filtered: list[str] = []
    dropped: list[str] = []
    idx = 0
    while idx < len(pipeline_args):
        token = str(pipeline_args[idx])
        if token.startswith("--"):
            option_name = token.split("=", 1)[0]
            if option_name not in supported:
                dropped.append(token)
                if "=" not in token and idx + 1 < len(pipeline_args) and not str(pipeline_args[idx + 1]).startswith("--"):
                    dropped.append(str(pipeline_args[idx + 1]))
                    idx += 2
                    continue
                idx += 1
                continue
            filtered.append(token)
            if "=" not in token and idx + 1 < len(pipeline_args) and not str(pipeline_args[idx + 1]).startswith("--"):
                filtered.append(str(pipeline_args[idx + 1]))
                idx += 2
                continue
        else:
            filtered.append(token)
        idx += 1
    return filtered, dropped


def _manual_resolved_default_preset() -> BasePreset:
    return BasePreset(
        name="resolved_default",
        lane_tags=("canonical", "global"),
        launcher_tokens=("-u", "-m", _CANONICAL_ADAPT_MODULE),
        pipeline_args=(
            "--L", "2", "--problem", "hh", "--t", "1.0", "--u", "4.0", "--dv", "0.0",
            "--omega0", "1.0", "--g-ep", "0.5", "--n-ph-max", "1", "--boson-encoding", "binary",
            "--ordering", "blocked", "--boundary", "open", "--term-order", "sorted",
            "--adapt-continuation-mode", "phase3_v1", "--adapt-max-depth", "160", "--adapt-eps-grad", "5e-7",
            "--adapt-eps-energy", "1e-9", "--adapt-seed", "7", "--adapt-inner-optimizer", "SPSA",
            "--adapt-spsa-a", "0.1", "--adapt-spsa-c", "0.02", "--adapt-spsa-A", "5.0",
            "--adapt-spsa-callback-every", "5", "--adapt-spsa-progress-every-s", "30", "--adapt-maxiter", "3200",
            "--adapt-state-backend", "compiled", "--adapt-reopt-policy", "windowed", "--adapt-window-size", "999999",
            "--adapt-window-topk", "999999", "--adapt-full-refit-every", "8", "--adapt-final-full-refit", "true",
            "--adapt-beam-live-branches", "3", "--adapt-beam-children-per-parent", "2", "--adapt-beam-terminated-keep", "3",
            "--phase1-prune-enabled", "--phase1-prune-fraction", "0.25", "--phase1-prune-max-candidates", "6",
            "--phase1-prune-max-regression", "1e-8", "--phase1-probe-max-positions", "999999",
            "--phase1-trough-margin-ratio", "1.0", "--phase1-shortlist-size", "64", "--phase2-shortlist-fraction", "1.0",
            "--phase2-shortlist-size", "64", "--phase2-lambda-H", "1e-6", "--phase2-rho", "0.25",
            "--phase2-gamma-N", "1.0", "--phase2-w-depth", "0.2", "--phase2-w-group", "0.15",
            "--phase2-w-shot", "0.15", "--phase2-w-optdim", "0.1", "--phase2-w-reuse", "0.1",
            "--phase2-w-lifetime", "0.05", "--phase2-frontier-ratio", "0.9", "--phase3-frontier-ratio", "0.9",
            "--phase2-batch-target-size", "8", "--phase2-batch-size-cap", "16", "--phase2-batch-near-degenerate-ratio", "0.9",
            "--phase3-symmetry-mitigation-mode", "verify_only", "--phase3-enable-rescue",
            "--phase3-backend-cost-mode", "proxy", "--phase3-runtime-split-mode", "off",
            "--phase3-lifetime-cost-mode", "off", "--adapt-drop-floor", "5e-4", "--adapt-drop-patience", "3",
            "--adapt-drop-min-depth", "12", "--phase2-no-batching",
        ),
        notes="Current direct route with omitted adapt-pool so runtime resolves the canonical narrow-core default.",
    )


def _artifact_backed_preset(name: str, command_sh: Path, lane_tags: Sequence[str], notes: str | None = None) -> BasePreset:
    env_overrides, launcher_tokens, pipeline_args = _extract_command_line_from_script(command_sh)
    return BasePreset(
        name=name,
        lane_tags=tuple(str(x) for x in lane_tags),
        launcher_tokens=_normalize_launcher_tokens(name, lane_tags, launcher_tokens),
        pipeline_args=tuple(pipeline_args),
        env_overrides=tuple(sorted((str(k), str(v)) for k, v in env_overrides.items())),
        source_artifact_dir=str(command_sh.parents[1]),
        notes=notes,
    )


@lru_cache(maxsize=1)
def _base_preset_library() -> dict[str, BasePreset]:
    presets: dict[str, BasePreset] = {
        "resolved_default": _manual_resolved_default_preset(),
    }
    artifact_specs = [
        ("public_anchor", _PUBLIC_ANCHOR_COMMAND, ("canonical", "global"), "Current public direct-route anchor."),
        ("legacy_20260322", _LEGACY81_COMMAND, ("global", "legacy"), "Frozen historical legacy oracle."),
        ("legacy_focus_75", _LEGACY_FOCUS_75_COMMAND, ("legacy",), "Best fresh Powell legacy-compat line so far."),
        ("raw_exact_compile_only", _RAW_EXACT_COMPILE_ONLY_COMMAND, ("global",), "Best current-route comparable replay."),
        ("fullhorse_motif", _FULLHORSE_MOTIF_COMMAND, ("global",), "Best current fullhorse-style replay."),
        ("bridge_diag", _BRIDGE_DIAG_COMMAND, ("global",), "Best current diagnostic native recovery line."),
        ("bridge_focus_98", _BRIDGE_FOCUS_98_COMMAND, ("global",), "Best fresh Powell current-code bridge-focused line so far."),
        ("bridge_focus_118", _BRIDGE_FOCUS_118_COMMAND, ("global",), "Runner-up fresh Powell current-code bridge-focused line so far."),
    ]
    for name, command_sh, lanes, notes in artifact_specs:
        if command_sh.exists():
            presets[name] = _artifact_backed_preset(name, command_sh, lanes, notes)
    return presets


@lru_cache(maxsize=None)
def _available_presets_for_lane(lane: str) -> tuple[str, ...]:
    presets = _base_preset_library()
    available = [
        name
        for name, preset in presets.items()
        if str(lane) in set(preset.lane_tags) and (str(lane) == "legacy" or str(name) != "legacy_20260322")
    ]
    available.sort()
    return tuple(available)


def _searchable_presets_for_lane(
    lane: str,
    restricted_base_preset_names: Sequence[str] = (),
) -> tuple[str, ...]:
    available = list(_available_presets_for_lane(lane))
    if not restricted_base_preset_names:
        return tuple(available)
    allowed = {str(name) for name in restricted_base_preset_names}
    filtered = [name for name in available if name in allowed]
    if not filtered:
        raise ValueError(f"No searchable presets remain for lane {lane} after applying restrictions {list(allowed)}.")
    return tuple(filtered)


def _full_policy_base_param_space() -> dict[str, list[str]]:
    return {str(name): ["base"] for name in _FULL_POLICY_PARAM_OPTIONS}


def _full_policy_param_space() -> dict[str, list[str]]:
    return {str(name): list(values) for name, values in _FULL_POLICY_PARAM_OPTIONS.items()}


def _preset_param_space(
    base_preset: str,
    *,
    energy_only_surface: bool = False,
    speed_surface_profile: str = "standard",
    force_spsa_profile: str | None = None,
    phase2_w_shot_profile_space: str = "default",
    enable_prune_prefilter_profile_space: bool = False,
) -> dict[str, list[Any]]:
    presets = _base_preset_library()
    if base_preset not in presets:
        raise KeyError(base_preset)
    preset = presets[base_preset]
    is_legacy = _preset_is_legacy(preset)
    if is_legacy:
        # Compatibility-only legacy lane: keep this search surface to knobs confirmed on
        # the frozen legacy entrypoint, so Optuna does not waste budget on modern-only flags.
        space = {
            "adapt_max_depth": [12, 16, 20, 24, 32],
            "selector_geometry_mode": ["base"],
            "runtime_split_mode": ["base", "off", "shortlist_pauli_children_v1"],
            "batching_mode": ["base", "on", "off"],
            "repeats_mode": ["base", "allow", "disable"],
            "selection_cost_mode": ["base", "proxy", "transpile_single_v1"],
            "motif_mode": ["base", "off"],
            "phase1_prune_mode": ["base", "off", "live"],
            "phase0_pilot_profile": ["base"],
            "phase0_pilot_records_profile": ["base"],
            "maturity_shortlist_profile": ["base"],
            "phase1_shortlist_size_profile": ["base"],
            "phase2_shortlist_fraction_profile": ["base"],
            "phase2_shortlist_size_profile": ["base"],
            "maturity_shot_profile": ["base"],
            "phase_live_profile": ["base"],
            "prune_witness_profile": ["base"],
            "prune_prefilter_profile": ["base"],
            "adapt_window_profile": ["base"],
            "adapt_history_window_profile": ["base"],
            "geometry_window_profile": ["base"],
            "backend_cost_weight_profile": ["base"],
            "phase2_w_shot_profile": ["base"],
            "phase2_rho_profile": ["base"],
            "spsa_profile": ["current", "paper_i_strong_like", "paper_i_weak_strong_like"],
            "ml_candidate_profile": ["base", "ml_p10", "ml_median", "ml_p90"],
            "phase1_prune_fraction_profile": ["base"],
            "batch_near_degenerate_profile": ["base"],
            "batch_rank_tol_profile": ["base"],
            "batch_additivity_tol_profile": ["base"],
        }
        space.update(_full_policy_base_param_space())
        if energy_only_surface:
            space["selection_cost_mode"] = ["base"]
            space["maturity_shot_profile"] = ["base"]
        if force_spsa_profile not in {None, ""}:
            value = str(force_spsa_profile)
            if value not in _SPSA_PROFILE_OPTIONS:
                raise ValueError(f"Unknown force_spsa_profile: {value!r}")
            space["spsa_profile"] = [value]
        return space
    space = {
        "adapt_max_depth": [10, 16, 24, 40, 80, 160],
        "selector_geometry_mode": ["base", "reduced", "raw_exact", "proxy_reduced"],
        # Current direct-route ADAPT fixes phase3 runtime splitting to `off`.
        # Keep the legacy shortlist value only on the frozen legacy entrypoint above.
        "runtime_split_mode": ["off"],
        "batching_mode": ["base", "on", "off"],
        "repeats_mode": ["base", "allow", "disable"],
        "selection_cost_mode": ["base", "proxy", "transpile_single_v1", _MARRAKESH_GRAPH_SPAN_MODE],
        "motif_mode": ["base", "off", "legacy81"],
        "phase1_prune_mode": ["base", "off", "live"],
        "phase0_pilot_profile": ["base", "pilot_64", "pilot_32"],
        "phase0_pilot_records_profile": ["base"],
        "maturity_shortlist_profile": ["base", "cap_ramp_narrow", "cap_ramp_medium"],
        "phase1_shortlist_size_profile": ["base"],
        "phase2_shortlist_fraction_profile": ["base"],
        "phase2_shortlist_size_profile": ["base"],
        "maturity_shot_profile": ["base", "late_x2", "late_x4_phase23"],
        "phase_live_profile": ["base", "keep_all", "phase3_early_null"],
        "prune_witness_profile": ["base", "relaxed_required", "strict_required"],
        "prune_prefilter_profile": ["base"],
        "adapt_window_profile": ["base", "full_heavy", "medium_16", "tight_8"],
        "adapt_history_window_profile": ["base"],
        "geometry_window_profile": ["base"],
        "backend_cost_weight_profile": ["base", "marrakesh_balanced", "marrakesh_2q_heavy", "marrakesh_depth_heavy"],
        "phase2_w_shot_profile": ["base", "shot_0p15", "shot_0p08", "shot_0p04", "shot_0p02"],
        "phase2_rho_profile": ["base", "rho_0p25", "rho_0p5", "rho_0p75"],
        "spsa_profile": ["current", "paper_i_strong_like", "paper_i_weak_strong_like"],
        "ml_candidate_profile": ["base", "ml_p10", "ml_median", "ml_p90"],
        "phase1_prune_fraction_profile": ["base"],
        "batch_near_degenerate_profile": ["base"],
        "batch_rank_tol_profile": ["base"],
        "batch_additivity_tol_profile": ["base"],
    }
    space.update(_full_policy_base_param_space())
    if bool(enable_prune_prefilter_profile_space):
        space["prune_prefilter_profile"] = ["base", "strict_motif", "strict_motif_cap2"]
    if str(speed_surface_profile) == "staged_graph":
        space["selection_cost_mode"] = [_MARRAKESH_GRAPH_SPAN_MODE]
        space["motif_mode"] = ["off"]
        space["phase1_prune_mode"] = ["live"]
        space["runtime_split_mode"] = ["shortlist_pauli_children_v1"]
        space["batching_mode"] = ["on"]
        space["phase0_pilot_profile"] = ["base", "pilot_64", "pilot_32", "pilot_24"]
        space["maturity_shortlist_profile"] = ["heavy_full", "cap_ramp_medium", "cap_ramp_narrow", "cap_ramp_tight"]
        space["maturity_shot_profile"] = ["heavy_x8_phase23", "late_x4_phase23", "late_x2", "cheap_x1"]
        space["adapt_window_profile"] = ["full_heavy", "medium_16", "tight_8", "cheap_4"]
        space["backend_cost_weight_profile"] = ["marrakesh_balanced", "marrakesh_2q_heavy", "marrakesh_depth_heavy"]
        space["phase2_w_shot_profile"] = ["shot_0p15", "shot_0p08", "shot_0p04", "shot_0p02"]
        if str(phase2_w_shot_profile_space) == "legacy_with_zero":
            space["phase2_w_shot_profile"] = ["shot_0p15", "shot_0p08", "shot_0p04", "shot_0p02", "shot_0p00"]
        space["phase2_rho_profile"] = ["rho_0p25", "rho_0p5", "rho_0p75"]
        space["spsa_profile"] = ["current", "paper_i_strong_like", "paper_i_weak_strong_like"]
        space["ml_candidate_profile"] = ["ml_p10", "ml_median", "ml_p90"]
        if bool(enable_prune_prefilter_profile_space):
            space["prune_prefilter_profile"] = ["base", "strict_motif", "strict_motif_cap2"]
    if str(speed_surface_profile) == "staged_shot":
        space["selection_cost_mode"] = [_MARRAKESH_GRAPH_SPAN_MODE]
        space["motif_mode"] = ["off"]
        space["phase1_prune_mode"] = ["live"]
        space["runtime_split_mode"] = ["shortlist_pauli_children_v1"]
        space["batching_mode"] = ["on"]
        space["phase0_pilot_profile"] = ["base", "pilot_64", "pilot_32", "pilot_24", "pilot_16", "pilot_10"]
        space["maturity_shortlist_profile"] = ["cap_ramp_micro", "cap_ramp_tight", "cap_ramp_narrow", "cap_ramp_medium"]
        space["maturity_shot_profile"] = ["cheap_x1", "late_x2", "late_x4_phase23"]
        space["phase_live_profile"] = ["phase3_early_null", "base"]
        space["prune_witness_profile"] = ["base", "relaxed_required", "strict_required"]
        space["adapt_window_profile"] = ["cheap_4", "tight_8", "medium_16"]
        space["backend_cost_weight_profile"] = ["marrakesh_balanced", "marrakesh_2q_heavy"]
        space["phase2_w_shot_profile"] = ["shot_0p75", "shot_0p50", "shot_0p30", "shot_0p15", "shot_0p08", "shot_0p04", "shot_0p02"]
        space["phase2_rho_profile"] = ["rho_0p25", "rho_0p5", "rho_0p75"]
        space["spsa_profile"] = ["current", "paper_i_strong_like", "paper_i_weak_strong_like"]
        space["ml_candidate_profile"] = ["ml_p10", "ml_median", "ml_p90"]
        if bool(enable_prune_prefilter_profile_space):
            space["prune_prefilter_profile"] = ["base", "strict_motif", "strict_motif_cap2", "strict_motif_cap3"]
    if str(speed_surface_profile) == "shortlist_refine":
        space["adapt_max_depth"] = [13]
        space["selector_geometry_mode"] = ["base"]
        space["selection_cost_mode"] = [_MARRAKESH_GRAPH_SPAN_MODE]
        space["motif_mode"] = ["off"]
        space["phase1_prune_mode"] = ["live"]
        space["runtime_split_mode"] = ["shortlist_pauli_children_v1"]
        space["batching_mode"] = ["on"]
        space["repeats_mode"] = ["base"]
        space["phase0_pilot_profile"] = ["base"]
        space["phase0_pilot_records_profile"] = [
            "base",
            "pilot_10",
            "pilot_16",
            "pilot_24",
            "pilot_32",
            "pilot_48",
            "pilot_64",
        ]
        space["maturity_shortlist_profile"] = [
            "base",
            "cap_ramp_micro",
            "cap_ramp_tight",
            "cap_ramp_narrow",
            "cap_ramp_medium",
        ]
        space["phase1_shortlist_size_profile"] = [
            "base",
            "p1_8",
            "p1_10",
            "p1_12",
            "p1_16",
            "p1_20",
            "p1_24",
            "p1_32",
            "p1_48",
            "p1_64",
        ]
        space["phase2_shortlist_fraction_profile"] = [
            "base",
            "p2frac_0p10",
            "p2frac_0p15",
            "p2frac_0p20",
            "p2frac_0p25",
            "p2frac_0p35",
            "p2frac_0p50",
            "p2frac_0p75",
            "p2frac_1p00",
        ]
        space["phase2_shortlist_size_profile"] = [
            "base",
            "p2_4",
            "p2_6",
            "p2_8",
            "p2_10",
            "p2_12",
            "p2_16",
            "p2_24",
            "p2_32",
            "p2_48",
        ]
        space["maturity_shot_profile"] = ["base"]
        space["phase_live_profile"] = ["phase3_early_null", "base"]
        space["prune_witness_profile"] = ["base", "relaxed_required", "strict_required"]
        space["prune_prefilter_profile"] = (
            ["base", "strict_motif", "strict_motif_cap2", "strict_motif_cap3"]
            if bool(enable_prune_prefilter_profile_space)
            else ["base"]
        )
        space["adapt_window_profile"] = ["base"]
        space["adapt_history_window_profile"] = [
            "base",
            "hist_4",
            "hist_8",
            "hist_12",
            "hist_16",
            "hist_24",
            "hist_full",
        ]
        space["geometry_window_profile"] = [
            "base",
            "geom_4",
            "geom_8",
            "geom_12",
            "geom_16",
            "geom_24",
            "geom_32",
            "geom_full",
        ]
        space["backend_cost_weight_profile"] = ["marrakesh_2q_heavy"]
        space["phase2_w_shot_profile"] = [
            "shot_0p75",
            "shot_0p50",
            "shot_0p30",
            "shot_0p15",
            "shot_0p08",
            "shot_0p04",
            "shot_0p02",
        ]
        space["phase2_rho_profile"] = ["rho_0p25", "rho_0p5", "rho_0p75"]
        space["spsa_profile"] = ["current"]
        space["ml_candidate_profile"] = ["base", "ml_p10", "ml_median", "ml_p90"]
        space["phase1_prune_fraction_profile"] = [
            "base",
            "prune_frac_0p10",
            "prune_frac_0p15",
            "prune_frac_0p20",
            "prune_frac_0p25",
            "prune_frac_0p34",
            "prune_frac_0p41",
            "prune_frac_0p50",
        ]
        space["batch_near_degenerate_profile"] = [
            "base",
            "near_0p90",
            "near_0p95",
            "near_0p98",
            "near_0p995",
            "near_0p998",
        ]
        space["batch_rank_tol_profile"] = [
            "base",
            "ranktol_1e_6",
            "ranktol_7e_7",
            "ranktol_2e_5",
            "ranktol_1e_4",
            "ranktol_1p4e_4",
        ]
        space["batch_additivity_tol_profile"] = [
            "base",
            "addtol_0p01",
            "addtol_0p03",
            "addtol_0p10",
            "addtol_0p30",
            "addtol_0p66",
        ]
    if str(speed_surface_profile) == "energy_discovery":
        space["selection_cost_mode"] = [_MARRAKESH_GRAPH_SPAN_MODE]
        space["motif_mode"] = ["off"]
        space["phase1_prune_mode"] = ["live"]
        space["runtime_split_mode"] = ["shortlist_pauli_children_v1"]
        space["batching_mode"] = ["on"]
        space["phase0_pilot_profile"] = ["base", "pilot_64"]
        space["maturity_shortlist_profile"] = ["heavy_full", "cap_ramp_medium"]
        space["maturity_shot_profile"] = ["heavy_x8_phase23", "late_x4_phase23"]
        space["phase_live_profile"] = ["keep_all", "base"]
        space["prune_witness_profile"] = ["base", "relaxed_required"]
        space["adapt_window_profile"] = ["full_heavy", "medium_16"]
        space["backend_cost_weight_profile"] = ["base"]
        space["phase2_w_shot_profile"] = ["base"]
        space["phase2_rho_profile"] = ["base"]
        space["spsa_profile"] = ["current", "paper_i_strong_like", "paper_i_weak_strong_like"]
        space["ml_candidate_profile"] = ["base", "ml_median", "ml_p90"]
        if bool(enable_prune_prefilter_profile_space):
            space["prune_prefilter_profile"] = ["base", "strict_motif", "strict_motif_cap2"]
    if str(speed_surface_profile) == _HH_ROUTEA_FULL_POLICY_PROFILE:
        space["adapt_max_depth"] = [30]
        space["selector_geometry_mode"] = ["base"]
        space["selection_cost_mode"] = [_MARRAKESH_GRAPH_SPAN_MODE]
        space["motif_mode"] = ["off"]
        space["phase1_prune_mode"] = ["live"]
        space["runtime_split_mode"] = ["shortlist_pauli_children_v1"]
        space["batching_mode"] = ["on"]
        space["repeats_mode"] = ["base"]
        space["phase0_pilot_profile"] = ["base"]
        space["phase0_pilot_records_profile"] = ["base"]
        space["maturity_shortlist_profile"] = ["base"]
        space["phase1_shortlist_size_profile"] = ["base"]
        space["phase2_shortlist_fraction_profile"] = ["base"]
        space["phase2_shortlist_size_profile"] = ["base"]
        space["maturity_shot_profile"] = ["base"]
        space["phase_live_profile"] = ["base"]
        space["prune_witness_profile"] = ["base"]
        space["prune_prefilter_profile"] = ["base"]
        space["adapt_window_profile"] = ["base"]
        space["adapt_history_window_profile"] = ["base"]
        space["geometry_window_profile"] = ["base"]
        space["backend_cost_weight_profile"] = ["base"]
        space["phase2_w_shot_profile"] = ["base"]
        space["phase2_rho_profile"] = ["base"]
        space["spsa_profile"] = ["current"]
        space["ml_candidate_profile"] = ["base"]
        space["phase1_prune_fraction_profile"] = ["base"]
        space["batch_near_degenerate_profile"] = ["base"]
        space["batch_rank_tol_profile"] = ["base"]
        space["batch_additivity_tol_profile"] = ["base"]
        space.update(_full_policy_param_space())
    if energy_only_surface:
        space["selection_cost_mode"] = ["base"]
        space["maturity_shot_profile"] = ["base"]
        space["adapt_window_profile"] = ["base"]
        space["backend_cost_weight_profile"] = ["base"]
        space["phase2_w_shot_profile"] = ["base"]
        space["phase2_rho_profile"] = ["base"]
        space["adapt_history_window_profile"] = ["base"]
        space["geometry_window_profile"] = ["base"]
        space["prune_prefilter_profile"] = ["base"]
        space["spsa_profile"] = ["current", "paper_i_strong_like", "paper_i_weak_strong_like"]
        space["ml_candidate_profile"] = ["ml_p10", "ml_median", "ml_p90"]
    if force_spsa_profile not in {None, ""}:
        value = str(force_spsa_profile)
        if value not in _SPSA_PROFILE_OPTIONS:
            raise ValueError(f"Unknown force_spsa_profile: {value!r}")
        space["spsa_profile"] = [value]
    return space


def _lane_union_param_space(
    lane: str,
    restricted_base_preset_names: Sequence[str] = (),
    *,
    energy_only_surface: bool = False,
    speed_surface_profile: str = "standard",
    force_spsa_profile: str | None = None,
    phase2_w_shot_profile_space: str = "default",
    anchor_param_values: Mapping[str, Sequence[Any]] | None = None,
    enable_prune_prefilter_profile_space: bool = False,
) -> dict[str, list[Any]]:
    union: dict[str, list[Any]] = {}
    for preset_name in _searchable_presets_for_lane(lane, restricted_base_preset_names):
        for key, values in _preset_param_space(
            str(preset_name),
            energy_only_surface=energy_only_surface,
            speed_surface_profile=str(speed_surface_profile),
            force_spsa_profile=force_spsa_profile,
            phase2_w_shot_profile_space=str(phase2_w_shot_profile_space),
            enable_prune_prefilter_profile_space=bool(enable_prune_prefilter_profile_space),
        ).items():
            bucket = union.setdefault(str(key), [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)
    return _apply_anchor_param_values_to_space(union, anchor_param_values)


def _suggest_trial_params(
    trial: Any,
    lane: str,
    restricted_base_preset_names: Sequence[str] = (),
    *,
    energy_only_surface: bool = False,
    speed_surface_profile: str = "standard",
    force_spsa_profile: str | None = None,
    phase2_w_shot_profile_space: str = "default",
    anchor_param_values: Mapping[str, Sequence[Any]] | None = None,
    enable_prune_prefilter_profile_space: bool = False,
) -> TrialParams:
    base_preset = trial.suggest_categorical(
        "base_preset",
        _searchable_presets_for_lane_with_anchor(lane, restricted_base_preset_names, anchor_param_values),
    )
    space = _lane_union_param_space(
        lane,
        restricted_base_preset_names,
        energy_only_surface=energy_only_surface,
        speed_surface_profile=str(speed_surface_profile),
        force_spsa_profile=force_spsa_profile,
        phase2_w_shot_profile_space=str(phase2_w_shot_profile_space),
        anchor_param_values=anchor_param_values,
        enable_prune_prefilter_profile_space=bool(enable_prune_prefilter_profile_space),
    )
    trial_values: dict[str, Any] = {}
    for name in _TRIAL_PARAM_NAMES:
        if name == "base_preset":
            continue
        value = trial.suggest_categorical(name, list(space[name]))
        trial_values[name] = int(value) if name == "adapt_max_depth" else str(value)
    return TrialParams(base_preset=str(base_preset), **trial_values)


def _build_distributions(
    optuna: Any,
    lane: str,
    params: Mapping[str, Any],
    restricted_base_preset_names: Sequence[str] = (),
    *,
    energy_only_surface: bool = False,
    speed_surface_profile: str = "standard",
    force_spsa_profile: str | None = None,
    phase2_w_shot_profile_space: str = "default",
    anchor_param_values: Mapping[str, Sequence[Any]] | None = None,
    enable_prune_prefilter_profile_space: bool = False,
) -> dict[str, Any]:
    del params
    space = _lane_union_param_space(
        lane,
        restricted_base_preset_names,
        energy_only_surface=energy_only_surface,
        speed_surface_profile=str(speed_surface_profile),
        force_spsa_profile=force_spsa_profile,
        phase2_w_shot_profile_space=str(phase2_w_shot_profile_space),
        anchor_param_values=anchor_param_values,
        enable_prune_prefilter_profile_space=bool(enable_prune_prefilter_profile_space),
    )
    distributions = {
        "base_preset": optuna.distributions.CategoricalDistribution(
            _searchable_presets_for_lane_with_anchor(lane, restricted_base_preset_names, anchor_param_values)
        ),
    }
    for name in _TRIAL_PARAM_NAMES:
        if name == "base_preset":
            continue
        distributions[name] = optuna.distributions.CategoricalDistribution(list(space[name]))
    return distributions


_TRIAL_PARAM_NAMES = tuple(TrialParams.__dataclass_fields__.keys())


def _coerce_trial_param_value(name: str, value: Any) -> Any:
    if str(name) == "adapt_max_depth":
        return int(value)
    return str(value)


def _unique_trial_param_values(name: str, values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for raw_value in values:
        if raw_value is None:
            continue
        try:
            value = _coerce_trial_param_value(str(name), raw_value)
        except Exception:
            continue
        if value not in out:
            out.append(value)
    return out


def _apply_anchor_param_values_to_space(
    space: Mapping[str, Sequence[Any]],
    anchor_param_values: Mapping[str, Sequence[Any]] | None,
) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {str(key): list(values) for key, values in space.items()}
    if not anchor_param_values:
        return out
    for name, values in anchor_param_values.items():
        if str(name) in {"base_preset"}:
            continue
        if str(name) not in out:
            continue
        anchored_values = _unique_trial_param_values(str(name), values)
        if anchored_values:
            out[str(name)] = anchored_values
    return out


def _searchable_presets_for_lane_with_anchor(
    lane: str,
    restricted_base_preset_names: Sequence[str] = (),
    anchor_param_values: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[str, ...]:
    choices = list(_searchable_presets_for_lane(lane, restricted_base_preset_names))
    if not anchor_param_values or "base_preset" not in anchor_param_values:
        return tuple(choices)
    anchored = _unique_trial_param_values("base_preset", anchor_param_values.get("base_preset", ()))
    if not anchored:
        return tuple(choices)
    missing = [str(value) for value in anchored if str(value) not in choices]
    if missing:
        raise ValueError(
            f"Anchored base presets are not searchable in lane {lane}: {missing}. "
            "Adjust --restrict-base-presets or the enqueue prior manifest."
        )
    return tuple(str(value) for value in anchored)


def _shortlist_refine_anchor_param_values(
    enqueue_param_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Any]]:
    if not enqueue_param_rows:
        return {}
    anchor_values: dict[str, list[Any]] = {}
    for name in _TRIAL_PARAM_NAMES:
        if name in _SHORTLIST_REFINE_FREE_PARAM_NAMES:
            continue
        field_info = TrialParams.__dataclass_fields__[name]
        default_value = None if field_info.default is MISSING else field_info.default
        values = []
        for row in enqueue_param_rows:
            if not isinstance(row, Mapping):
                continue
            values.append(row.get(name, default_value))
        unique_values = _unique_trial_param_values(str(name), values)
        if unique_values:
            anchor_values[str(name)] = unique_values
    return anchor_values


def _normalise_enqueue_params_for_space(
    raw_params: Mapping[str, Any],
    *,
    lane: str,
    restricted_base_preset_names: Sequence[str],
    energy_only_surface: bool,
    speed_surface_profile: str,
    force_spsa_profile: str | None,
    phase2_w_shot_profile_space: str,
    anchor_param_values: Mapping[str, Sequence[Any]] | None = None,
    enable_prune_prefilter_profile_space: bool = False,
) -> dict[str, Any]:
    space = _lane_union_param_space(
        lane,
        restricted_base_preset_names,
        energy_only_surface=bool(energy_only_surface),
        speed_surface_profile=str(speed_surface_profile),
        force_spsa_profile=force_spsa_profile,
        phase2_w_shot_profile_space=str(phase2_w_shot_profile_space),
        anchor_param_values=anchor_param_values,
        enable_prune_prefilter_profile_space=bool(enable_prune_prefilter_profile_space),
    )
    choices_by_name: dict[str, list[Any]] = {
        "base_preset": list(
            _searchable_presets_for_lane_with_anchor(lane, restricted_base_preset_names, anchor_param_values)
        ),
    }
    for name in _TRIAL_PARAM_NAMES:
        if name != "base_preset":
            choices_by_name[name] = list(space.get(name, []))

    out: dict[str, Any] = {}
    for name in _TRIAL_PARAM_NAMES:
        choices = choices_by_name.get(name) or []
        if not choices:
            continue
        default_value = TrialParams.__dataclass_fields__[name].default
        value = raw_params.get(name, default_value)
        if value not in choices:
            value = default_value if default_value in choices else choices[0]
        out[name] = value
    return out


def _enqueue_param_rows_from_json(path: Path | None, *, regime: str | None) -> list[Mapping[str, Any]]:
    if path is None:
        return []
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not manifest_path.exists():
        return []
    payload = _load_json(manifest_path)
    raw_rows: Any = None
    if isinstance(payload.get("regimes"), Mapping):
        entry = (payload.get("regimes") or {}).get(str(regime)) or {}
        if isinstance(entry, Mapping):
            raw_rows = entry.get("enqueue_params") or entry.get("params")
        elif isinstance(entry, list):
            raw_rows = entry
    if raw_rows is None:
        raw_rows = payload.get("enqueue_params") or payload.get("params")
    if isinstance(raw_rows, Mapping):
        raw_rows = [raw_rows]
    if not isinstance(raw_rows, list):
        return []
    rows: list[Mapping[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        params = row.get("params") if isinstance(row.get("params"), Mapping) else row
        if isinstance(params, Mapping):
            rows.append(params)
    return rows


def _full_policy_raw(params: TrialParams, name: str) -> str | None:
    raw = getattr(params, name)
    value = str(raw).strip()
    if value == "" or value.lower() in {"base", "none", "null"}:
        return None
    return value


def _full_policy_float(params: TrialParams, name: str) -> float | None:
    raw = _full_policy_raw(params, name)
    return None if raw is None else float(raw)


def _full_policy_int(params: TrialParams, name: str) -> int | None:
    raw = _full_policy_raw(params, name)
    return None if raw is None else int(float(raw))


def _set_full_policy_float(
    args: Sequence[str],
    params: TrialParams,
    name: str,
    flags: Sequence[str],
) -> list[str]:
    value = _full_policy_float(params, name)
    if value is None:
        return list(args)
    out = list(args)
    for flag in flags:
        out = _set_option(out, flag, _format_cli_float(float(value)))
    return out


def _set_full_policy_int(
    args: Sequence[str],
    params: TrialParams,
    name: str,
    flags: Sequence[str],
) -> list[str]:
    value = _full_policy_int(params, name)
    if value is None:
        return list(args)
    out = list(args)
    for flag in flags:
        out = _set_option(out, flag, int(value))
    return out


def _apply_full_policy_overrides(params: TrialParams, pipeline_args: Sequence[str]) -> list[str]:
    args = list(pipeline_args)
    args = _set_full_policy_int(args, params, "full_phase0_pilot_max_records", ("--phase0-pilot-max-records",))
    args = _set_full_policy_int(args, params, "full_phase1_shortlist_size", ("--phase1-shortlist-size",))
    args = _set_full_policy_float(args, params, "full_phase2_shortlist_fraction", ("--phase2-shortlist-fraction",))
    args = _set_full_policy_int(args, params, "full_phase2_shortlist_size", ("--phase2-shortlist-size",))

    window_size = _full_policy_int(params, "full_adapt_window_size")
    if window_size is not None:
        args = _set_option(args, "--adapt-window-size", int(window_size))
        args = _set_option(args, "--adapt-window-topk", int(window_size))
    args = _set_full_policy_int(args, params, "full_phase3_geometry_window_size", ("--phase3-geometry-window-size",))
    args = _set_full_policy_float(args, params, "full_phase2_w_shot", ("--phase2-w-shot",))
    args = _set_full_policy_float(args, params, "full_phase2_rho", ("--phase2-rho",))

    phase2_batch_target_size = _full_policy_int(params, "full_phase2_batch_target_size")
    phase3_batch_target_size = _full_policy_int(params, "full_phase3_batch_target_size")
    if phase2_batch_target_size is not None:
        args = _set_option(args, "--phase2-batch-target-size", int(phase2_batch_target_size))
        if phase3_batch_target_size is None:
            args = _set_option(args, "--phase3-batch-target-size", int(phase2_batch_target_size))
    if phase3_batch_target_size is not None:
        args = _set_option(args, "--phase3-batch-target-size", int(phase3_batch_target_size))

    phase2_batch_size_cap = _full_policy_int(params, "full_phase2_batch_size_cap")
    phase3_batch_size_cap = _full_policy_int(params, "full_phase3_batch_size_cap")
    if phase2_batch_size_cap is not None:
        args = _set_option(args, "--phase2-batch-size-cap", int(phase2_batch_size_cap))
        if phase3_batch_size_cap is None:
            args = _set_option(args, "--phase3-batch-size-cap", int(phase2_batch_size_cap))
    if phase3_batch_size_cap is not None:
        args = _set_option(args, "--phase3-batch-size-cap", int(phase3_batch_size_cap))

    args = _set_full_policy_float(
        args,
        params,
        "full_batch_near_degenerate_ratio",
        ("--phase2-batch-near-degenerate-ratio", "--phase3-batch-near-degenerate-ratio"),
    )
    args = _set_full_policy_float(
        args,
        params,
        "full_batch_rank_rel_tol",
        ("--phase2-batch-rank-rel-tol", "--phase3-batch-rank-rel-tol"),
    )
    args = _set_full_policy_float(
        args,
        params,
        "full_batch_additivity_tol",
        ("--phase2-batch-additivity-tol", "--phase3-batch-additivity-tol"),
    )
    phase3_order_selection_mode = _full_policy_raw(params, "full_phase3_batch_order_selection_mode")
    if phase3_order_selection_mode is not None:
        args = _set_option(args, "--phase3-batch-order-selection-mode", str(phase3_order_selection_mode))
    args = _set_full_policy_int(
        args,
        params,
        "full_phase3_batch_order_max_permutations",
        ("--phase3-batch-order-max-permutations",),
    )

    args = _set_full_policy_float(args, params, "full_phase2_frontier_ratio", ("--phase2-frontier-ratio",))
    args = _set_full_policy_float(args, params, "full_phase3_frontier_ratio", ("--phase3-frontier-ratio",))
    args = _set_full_policy_float(args, params, "full_phase3_tie_beam_score_ratio", ("--phase3-tie-beam-score-ratio",))
    args = _set_full_policy_float(args, params, "full_phase3_tie_beam_abs_tol", ("--phase3-tie-beam-abs-tol",))
    args = _set_full_policy_int(args, params, "full_phase3_tie_beam_max_branches", ("--phase3-tie-beam-max-branches",))

    full_phase1_prune_mode = _full_policy_raw(params, "full_phase1_prune_mode")
    if full_phase1_prune_mode is not None:
        args = _set_option(args, "--phase1-prune-mode", str(full_phase1_prune_mode))
    args = _set_full_policy_float(args, params, "full_phase1_prune_fraction", ("--phase1-prune-fraction",))
    args = _set_full_policy_int(args, params, "full_phase1_prune_min_candidates", ("--phase1-prune-min-candidates",))
    args = _set_full_policy_int(args, params, "full_phase1_prune_max_candidates", ("--phase1-prune-max-candidates",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_max_regression", ("--phase1-prune-max-regression",))
    tolerance_mode = _full_policy_raw(params, "full_phase1_prune_tolerance_mode")
    if tolerance_mode is not None:
        args = _set_option(args, "--phase1-prune-tolerance-mode", str(tolerance_mode))
    args = _set_full_policy_float(args, params, "full_phase1_prune_tolerance_shot_coeff", ("--phase1-prune-tolerance-shot-coeff",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_tolerance_screen_coeff", ("--phase1-prune-tolerance-screen-coeff",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_tolerance_chem", ("--phase1-prune-tolerance-chem",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_tolerance_rel_coeff", ("--phase1-prune-tolerance-rel-coeff",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_retained_gain_ratio", ("--phase1-prune-retained-gain-ratio",))
    args = _set_full_policy_int(args, params, "full_phase1_prune_protect_steps", ("--phase1-prune-protect-steps",))
    args = _set_full_policy_int(args, params, "full_phase1_prune_stale_age", ("--phase1-prune-stale-age",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_stagnation_threshold", ("--phase1-prune-stagnation-threshold",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_small_theta_abs", ("--phase1-prune-small-theta-abs",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_small_theta_relative", ("--phase1-prune-small-theta-relative",))
    args = _set_full_policy_int(args, params, "full_phase1_prune_cooldown_steps", ("--phase1-prune-cooldown-steps",))
    args = _set_full_policy_int(args, params, "full_phase1_prune_local_window_size", ("--phase1-prune-local-window-size",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_recovery_trust_radius", ("--phase1-prune-recovery-trust-radius",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_old_fraction", ("--phase1-prune-old-fraction",))
    args = _set_full_policy_int(args, params, "full_phase1_prune_checkpoint_period", ("--phase1-prune-checkpoint-period",))
    args = _set_full_policy_int(args, params, "full_phase1_prune_live_min_depth", ("--phase1-prune-live-min-depth",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_maturity_threshold", ("--phase1-prune-maturity-threshold",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_snr_threshold", ("--phase1-prune-snr-threshold",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_collapse_peak_abs_min", ("--phase1-prune-collapse-peak-abs-min",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_collapse_current_abs_max", ("--phase1-prune-collapse-current-abs-max",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_collapse_ratio", ("--phase1-prune-collapse-ratio",))
    args = _set_full_policy_float(args, params, "full_phase1_prune_collapse_min_abs_drop", ("--phase1-prune-collapse-min-abs-drop",))
    args = _set_full_policy_int(args, params, "full_phase1_prune_collapse_min_observations", ("--phase1-prune-collapse-min-observations",))

    args = _set_full_policy_float(args, params, "full_phase3_backend_w_2q", ("--phase3-backend-w-2q",))
    args = _set_full_policy_float(args, params, "full_phase3_backend_w_depth", ("--phase3-backend-w-depth",))
    args = _set_full_policy_float(args, params, "full_phase3_backend_w_size", ("--phase3-backend-w-size",))

    spsa_maxiter = _full_policy_int(params, "full_spsa_maxiter")
    if spsa_maxiter is not None:
        args = _set_option(args, "--adapt-maxiter", int(spsa_maxiter))
        args = _set_option(args, "--adapt-final-refit-maxiter", int(spsa_maxiter))
    args = _set_full_policy_float(args, params, "full_spsa_a", ("--adapt-spsa-a",))
    args = _set_full_policy_float(args, params, "full_spsa_c", ("--adapt-spsa-c",))
    args = _set_full_policy_float(args, params, "full_spsa_alpha", ("--adapt-spsa-alpha",))
    args = _set_full_policy_float(args, params, "full_spsa_gamma", ("--adapt-spsa-gamma",))
    args = _set_full_policy_float(args, params, "full_spsa_A", ("--adapt-spsa-A",))
    args = _set_full_policy_int(args, params, "full_spsa_avg_last", ("--adapt-spsa-avg-last",))
    args = _set_full_policy_int(args, params, "full_spsa_eval_repeats", ("--adapt-spsa-eval-repeats",))
    args = _set_full_policy_int(args, params, "full_spsa_callback_every", ("--adapt-spsa-callback-every",))
    return args


def _apply_full_policy_post_route_lock_overrides(params: TrialParams, pipeline_args: Sequence[str]) -> list[str]:
    """Re-apply sampled controls that the fixed Route-A lock seeds first."""
    args = list(pipeline_args)
    full_phase1_prune_mode = _full_policy_raw(params, "full_phase1_prune_mode")
    if full_phase1_prune_mode is not None:
        args = _set_option(args, "--phase1-prune-mode", str(full_phase1_prune_mode))
    phase3_order_selection_mode = _full_policy_raw(params, "full_phase3_batch_order_selection_mode")
    if phase3_order_selection_mode is not None:
        args = _set_option(args, "--phase3-batch-order-selection-mode", str(phase3_order_selection_mode))
    return args


def _normalize_search_inner_optimizer(raw: str | None) -> str:
    value = str(raw or _SEARCH_INNER_OPTIMIZER).strip().upper()
    if value not in _SEARCH_INNER_OPTIMIZER_CHOICES:
        choices = ", ".join(sorted(_SEARCH_INNER_OPTIMIZER_CHOICES))
        raise ValueError(f"search_inner_optimizer must be one of {{{choices}}}; got {raw!r}.")
    return value


def _apply_trial_overrides(
    params: TrialParams,
    pipeline_args: Sequence[str],
    *,
    search_inner_optimizer: str = _SEARCH_INNER_OPTIMIZER,
) -> list[str]:
    args = list(str(x) for x in pipeline_args)
    args = _set_option(args, "--adapt-max-depth", int(params.adapt_max_depth))
    preset = _base_preset_library().get(str(params.base_preset))
    is_legacy = _preset_is_legacy(preset)

    if params.selector_geometry_mode != "base":
        args = _set_option(args, "--phase3-selector-geometry-mode", params.selector_geometry_mode)
    if is_legacy:
        if params.runtime_split_mode != "base":
            args = _set_option(args, "--phase3-runtime-split-mode", params.runtime_split_mode)
    else:
        split_mode = "off" if params.runtime_split_mode == "base" else str(params.runtime_split_mode)
        args = _set_option(args, "--phase3-runtime-split-mode", split_mode)
        args = _remove_option(args, "--allow-archival-phase3-runtime-split")
        if split_mode != "off":
            args = [*args, "--allow-archival-phase3-runtime-split"]
    if params.batching_mode == "on":
        args = _set_toggle_pair(args, "--phase2-enable-batching", "--phase2-no-batching", True)
    elif params.batching_mode == "off":
        args = _set_toggle_pair(args, "--phase2-enable-batching", "--phase2-no-batching", False)
    if params.repeats_mode == "disable":
        args = [*_remove_option(args, "--adapt-no-repeats"), "--adapt-no-repeats"]
    elif params.repeats_mode == "allow":
        args = _remove_option(args, "--adapt-no-repeats")
    if params.selection_cost_mode == "proxy":
        args = _set_option(args, "--phase3-backend-cost-mode", "proxy")
        args = _remove_option(args, "--phase3-backend-name")
        args = _remove_option(args, "--phase3-backend-transpile-seed")
        args = _remove_option(args, "--phase3-backend-optimization-level")
    elif params.selection_cost_mode == "transpile_single_v1":
        args = _set_option(args, "--phase3-backend-cost-mode", "transpile_single_v1")
        args = _set_option(args, "--phase3-backend-name", "FakeNighthawk")
        args = _set_option(args, "--phase3-backend-transpile-seed", 7)
        args = _set_option(args, "--phase3-backend-optimization-level", 1)
    elif params.selection_cost_mode == _MARRAKESH_GRAPH_SPAN_MODE:
        args = _set_option(args, "--phase3-backend-cost-mode", _MARRAKESH_GRAPH_SPAN_MODE)
        args = _set_option(args, "--phase3-backend-name", "FakeMarrakesh")
        args = _remove_option(args, "--phase3-backend-shortlist")
        args = _set_option(args, "--phase3-backend-transpile-seed", 7)
        args = _set_option(args, "--phase3-backend-optimization-level", 1)
    if params.motif_mode == "off":
        args = _remove_option(args, "--phase3-motif-source-json")
    elif params.motif_mode == "legacy81" and _LEGACY81_RESULT_JSON.exists():
        args = _set_option(args, "--phase3-motif-source-json", str(_LEGACY81_RESULT_JSON))
    if params.phase1_prune_mode == "off":
        args = _set_toggle_pair(args, "--phase1-prune-enabled", "--phase1-no-prune", False)
    elif params.phase1_prune_mode == "live":
        args = _set_toggle_pair(args, "--phase1-prune-enabled", "--phase1-no-prune", True)
        if not is_legacy:
            args = _set_option(args, "--phase1-prune-mode", "live")

    if not is_legacy:
        phase0_pilot_options = _profile_options(
            params.phase0_pilot_profile,
            _PHASE0_PILOT_PROFILE_OPTIONS,
            field_name="phase0_pilot_profile",
        )
        phase0_pilot_records_options = _profile_options(
            params.phase0_pilot_records_profile,
            _PHASE0_PILOT_RECORDS_PROFILE_OPTIONS,
            field_name="phase0_pilot_records_profile",
        )
        maturity_shortlist_options = _profile_options(
            params.maturity_shortlist_profile,
            _MATURITY_SHORTLIST_PROFILE_OPTIONS,
            field_name="maturity_shortlist_profile",
        )
        phase1_shortlist_size_options = _profile_options(
            params.phase1_shortlist_size_profile,
            _PHASE1_SHORTLIST_SIZE_PROFILE_OPTIONS,
            field_name="phase1_shortlist_size_profile",
        )
        phase2_shortlist_fraction_options = _profile_options(
            params.phase2_shortlist_fraction_profile,
            _PHASE2_SHORTLIST_FRACTION_PROFILE_OPTIONS,
            field_name="phase2_shortlist_fraction_profile",
        )
        phase2_shortlist_size_options = _profile_options(
            params.phase2_shortlist_size_profile,
            _PHASE2_SHORTLIST_SIZE_PROFILE_OPTIONS,
            field_name="phase2_shortlist_size_profile",
        )
        maturity_shot_options = _profile_options(
            params.maturity_shot_profile,
            _MATURITY_SHOT_PROFILE_OPTIONS,
            field_name="maturity_shot_profile",
        )
        prune_witness_options = _profile_options(
            params.prune_witness_profile,
            _PRUNE_WITNESS_PROFILE_OPTIONS,
            field_name="prune_witness_profile",
        )
        prune_prefilter_options = _profile_options(
            params.prune_prefilter_profile,
            _PRUNE_PREFILTER_PROFILE_OPTIONS,
            field_name="prune_prefilter_profile",
        )
        adapt_window_options = _profile_options(
            params.adapt_window_profile,
            _ADAPT_WINDOW_PROFILE_OPTIONS,
            field_name="adapt_window_profile",
        )
        adapt_history_window_options = _profile_options(
            params.adapt_history_window_profile,
            _ADAPT_HISTORY_WINDOW_PROFILE_OPTIONS,
            field_name="adapt_history_window_profile",
        )
        geometry_window_options = _profile_options(
            params.geometry_window_profile,
            _GEOMETRY_WINDOW_PROFILE_OPTIONS,
            field_name="geometry_window_profile",
        )
        backend_cost_weight_options = _profile_options(
            params.backend_cost_weight_profile,
            _BACKEND_COST_WEIGHT_PROFILE_OPTIONS,
            field_name="backend_cost_weight_profile",
        )
        ml_candidate_options = _profile_options(
            params.ml_candidate_profile,
            _ML_CANDIDATE_PROFILE_OPTIONS,
            field_name="ml_candidate_profile",
        )
        spsa_options = _profile_options(
            params.spsa_profile,
            _SPSA_PROFILE_OPTIONS,
            field_name="spsa_profile",
        )
        phase1_prune_fraction_options = _profile_options(
            params.phase1_prune_fraction_profile,
            _PHASE1_PRUNE_FRACTION_PROFILE_OPTIONS,
            field_name="phase1_prune_fraction_profile",
        )
        batch_near_degenerate_options = _profile_options(
            params.batch_near_degenerate_profile,
            _BATCH_NEAR_DEGENERATE_PROFILE_OPTIONS,
            field_name="batch_near_degenerate_profile",
        )
        batch_rank_tol_options = _profile_options(
            params.batch_rank_tol_profile,
            _BATCH_RANK_TOL_PROFILE_OPTIONS,
            field_name="batch_rank_tol_profile",
        )
        batch_additivity_tol_options = _profile_options(
            params.batch_additivity_tol_profile,
            _BATCH_ADDITIVITY_TOL_PROFILE_OPTIONS,
            field_name="batch_additivity_tol_profile",
        )
        phase_live_profile = str(params.phase_live_profile)
        if phase_live_profile not in {"base", "keep_all", "phase3_early_null"}:
            raise ValueError(f"Unknown phase_live_profile: {phase_live_profile!r}")

        if str(params.phase0_pilot_profile) != "base":
            args = _remove_options(args, _PHASE0_PILOT_PROFILE_FLAGS)
        args = _apply_option_profile(args, phase0_pilot_options)
        if str(params.phase0_pilot_records_profile) != "base":
            args = _remove_options(args, _PHASE0_PILOT_PROFILE_FLAGS)
        args = _apply_option_profile(args, phase0_pilot_records_options)

        if str(params.maturity_shortlist_profile) != "base":
            args = _remove_options(args, _MATURITY_SHORTLIST_FLAGS)
        args = _apply_option_profile(args, maturity_shortlist_options)
        if str(params.phase1_shortlist_size_profile) != "base":
            args = _remove_options(args, _PHASE1_SHORTLIST_SIZE_FLAGS)
        args = _apply_option_profile(args, phase1_shortlist_size_options)
        if str(params.phase2_shortlist_fraction_profile) != "base":
            args = _remove_options(args, _PHASE2_SHORTLIST_FRACTION_FLAGS)
        args = _apply_option_profile(args, phase2_shortlist_fraction_options)
        if str(params.phase2_shortlist_size_profile) != "base":
            args = _remove_options(args, _PHASE2_SHORTLIST_SIZE_FLAGS)
        args = _apply_option_profile(args, phase2_shortlist_size_options)

        if str(params.maturity_shot_profile) != "base":
            args = _remove_options(args, _MATURITY_SHOT_FLAGS)
        args = _apply_option_profile(args, maturity_shot_options)

        if phase_live_profile != "base":
            args = _remove_options(args, _PHASE_LIVE_FLAGS)
        if phase_live_profile == "keep_all":
            args = _set_toggle_pair(
                args,
                "--phase-live-hysteresis-enabled",
                "--phase-live-hysteresis-disabled",
                False,
            )
        elif phase_live_profile == "phase3_early_null":
            args = _set_toggle_pair(
                args,
                "--phase-live-hysteresis-enabled",
                "--phase-live-hysteresis-disabled",
                True,
            )
            args = _set_option(args, "--phase2-null-nrem-high-threshold", 0.0)
            args = _set_option(args, "--phase2-live-nrem-low-threshold", 0.25)
            args = _set_option(args, "--phase3-null-nrem-high-threshold", 0.75)
            args = _set_option(args, "--phase3-live-nrem-low-threshold", 1.25)
            args = _set_option(args, "--phase2-hysteresis-steps", 2)
            args = _set_option(args, "--phase3-hysteresis-steps", 1)

        if str(params.prune_witness_profile) != "base":
            args = _remove_options(args, _PRUNE_WITNESS_FLAGS)
        args = _apply_option_profile(args, prune_witness_options)
        if str(params.prune_prefilter_profile) != "base":
            args = _remove_options(args, _PRUNE_PREFILTER_FLAGS)
        args = _apply_option_profile(args, prune_prefilter_options)

        if str(params.adapt_window_profile) != "base":
            args = _remove_options(args, _ADAPT_WINDOW_FLAGS)
        args = _apply_option_profile(args, adapt_window_options)
        if str(params.adapt_history_window_profile) != "base":
            args = _remove_options(args, _ADAPT_HISTORY_WINDOW_FLAGS)
        args = _apply_option_profile(args, adapt_history_window_options)
        if str(params.geometry_window_profile) != "base":
            args = _remove_options(args, _GEOMETRY_WINDOW_FLAGS)
        args = _apply_option_profile(args, geometry_window_options)

        if str(params.backend_cost_weight_profile) != "base":
            args = _remove_options(args, _BACKEND_COST_WEIGHT_FLAGS)
        args = _apply_option_profile(args, backend_cost_weight_options)

        if str(params.ml_candidate_profile) != "base":
            args = _remove_options(args, _ML_CANDIDATE_PROFILE_FLAGS)
        args = _apply_option_profile(args, ml_candidate_options)
        if str(params.phase1_prune_fraction_profile) != "base":
            args = _remove_options(args, _PHASE1_PRUNE_FRACTION_FLAGS)
        args = _apply_option_profile(args, phase1_prune_fraction_options)
        if str(params.batch_near_degenerate_profile) != "base":
            args = _remove_options(args, _BATCH_NEAR_DEGENERATE_FLAGS)
        args = _apply_option_profile(args, batch_near_degenerate_options)
        if str(params.batch_rank_tol_profile) != "base":
            args = _remove_options(args, _BATCH_RANK_TOL_FLAGS)
        args = _apply_option_profile(args, batch_rank_tol_options)
        if str(params.batch_additivity_tol_profile) != "base":
            args = _remove_options(args, _BATCH_ADDITIVITY_TOL_FLAGS)
        args = _apply_option_profile(args, batch_additivity_tol_options)

        phase2_w_shot = _PHASE2_W_SHOT_PROFILE_OPTIONS.get(str(params.phase2_w_shot_profile))
        if phase2_w_shot is not None:
            args = _set_option(args, "--phase2-w-shot", _format_cli_float(float(phase2_w_shot)))
        phase2_rho = _PHASE2_RHO_PROFILE_OPTIONS.get(str(params.phase2_rho_profile))
        if phase2_rho is not None:
            args = _set_option(args, "--phase2-rho", _format_cli_float(float(phase2_rho)))
    else:
        ml_candidate_options = _profile_options(
            params.ml_candidate_profile,
            _ML_CANDIDATE_PROFILE_OPTIONS,
            field_name="ml_candidate_profile",
        )
        spsa_options = _profile_options(
            params.spsa_profile,
            _SPSA_PROFILE_OPTIONS,
            field_name="spsa_profile",
        )
        if str(params.ml_candidate_profile) != "base":
            args = _remove_options(args, _ML_CANDIDATE_PROFILE_FLAGS)
        args = _apply_option_profile(args, ml_candidate_options)

    existing_maxiter_raw = _get_option_value(args, "--adapt-maxiter")
    try:
        existing_maxiter = int(existing_maxiter_raw) if existing_maxiter_raw is not None else 0
    except Exception:
        existing_maxiter = 0
    search_inner_optimizer = _normalize_search_inner_optimizer(search_inner_optimizer)
    args = _set_option(args, "--adapt-inner-optimizer", search_inner_optimizer)
    args = _set_option(args, "--adapt-maxiter", max(int(_SEARCH_ADAPT_MAXITER_FLOOR), int(existing_maxiter)))
    args = _remove_options(args, _SPSA_PROFILE_FLAGS)
    if search_inner_optimizer == "SPSA":
        args = _apply_option_profile(args, spsa_options)
    args = _apply_full_policy_overrides(params, args)
    return args


def _trial_case_dir(output_dir: Path, lane: str, epsilon_abs_delta_e: float, trial_index: int, warm_start: bool = False) -> Path:
    bucket = "warm_start" if warm_start else f"trial_{int(trial_index):04d}"
    return output_dir / str(lane) / f"eps_{_float_slug(float(epsilon_abs_delta_e))}" / bucket


def _effective_trial_params(params: TrialParams, dropped_args: Sequence[str]) -> dict[str, Any]:
    effective = dict(asdict(params))
    dropped = {str(x) for x in dropped_args}
    if "--phase3-selector-geometry-mode" in dropped:
        effective["selector_geometry_mode"] = "inactive"
    if "--phase3-runtime-split-mode" in dropped:
        effective["runtime_split_mode"] = "inactive"
    if ("--phase2-enable-batching" in dropped) or ("--phase2-no-batching" in dropped):
        effective["batching_mode"] = "inactive"
    if "--adapt-no-repeats" in dropped:
        effective["repeats_mode"] = "inactive"
    if dropped.intersection({
        "--phase3-backend-cost-mode",
        "--phase3-backend-name",
        "--phase3-backend-transpile-seed",
        "--phase3-backend-optimization-level",
    }):
        effective["selection_cost_mode"] = "inactive"
    if "--phase3-motif-source-json" in dropped:
        effective["motif_mode"] = "inactive"
    if dropped.intersection({"--phase1-no-prune", "--phase1-prune-enabled", "--phase1-prune-mode"}):
        effective["phase1_prune_mode"] = "inactive"
    if dropped.intersection(_PHASE0_PILOT_PROFILE_FLAGS):
        effective["phase0_pilot_profile"] = "inactive"
        effective["phase0_pilot_records_profile"] = "inactive"
    if dropped.intersection(_MATURITY_SHORTLIST_FLAGS):
        effective["maturity_shortlist_profile"] = "inactive"
    if dropped.intersection(_PHASE1_SHORTLIST_SIZE_FLAGS):
        effective["phase1_shortlist_size_profile"] = "inactive"
    if dropped.intersection(_PHASE2_SHORTLIST_FRACTION_FLAGS):
        effective["phase2_shortlist_fraction_profile"] = "inactive"
    if dropped.intersection(_PHASE2_SHORTLIST_SIZE_FLAGS):
        effective["phase2_shortlist_size_profile"] = "inactive"
    if dropped.intersection(_MATURITY_SHOT_FLAGS):
        effective["maturity_shot_profile"] = "inactive"
    if dropped.intersection(_PHASE_LIVE_FLAGS):
        effective["phase_live_profile"] = "inactive"
    if dropped.intersection(_PRUNE_WITNESS_FLAGS):
        effective["prune_witness_profile"] = "inactive"
    if dropped.intersection(_PRUNE_PREFILTER_FLAGS):
        effective["prune_prefilter_profile"] = "inactive"
    if dropped.intersection(_ADAPT_WINDOW_FLAGS):
        effective["adapt_window_profile"] = "inactive"
    if dropped.intersection(_ADAPT_HISTORY_WINDOW_FLAGS):
        effective["adapt_history_window_profile"] = "inactive"
    if dropped.intersection(_GEOMETRY_WINDOW_FLAGS):
        effective["geometry_window_profile"] = "inactive"
    if dropped.intersection(_BACKEND_COST_WEIGHT_FLAGS):
        effective["backend_cost_weight_profile"] = "inactive"
    if dropped.intersection(_PHASE1_PRUNE_FRACTION_FLAGS):
        effective["phase1_prune_fraction_profile"] = "inactive"
    if dropped.intersection(_BATCH_NEAR_DEGENERATE_FLAGS):
        effective["batch_near_degenerate_profile"] = "inactive"
    if dropped.intersection(_BATCH_RANK_TOL_FLAGS):
        effective["batch_rank_tol_profile"] = "inactive"
    if dropped.intersection(_BATCH_ADDITIVITY_TOL_FLAGS):
        effective["batch_additivity_tol_profile"] = "inactive"
    if dropped.intersection(_FULL_POLICY_FLAGS):
        for name in _FULL_POLICY_PARAM_OPTIONS:
            if effective.get(str(name)) != "base":
                effective[str(name)] = "inactive"
    return effective


def _apply_route_a_paper_i_production_lock(args: Sequence[str]) -> list[str]:
    """Force the Paper-I Route-A production contract after Optuna trial sampling."""
    out = list(str(x) for x in args)
    out = _set_option(out, "--static-route-id", "route_a")
    out = _set_option(out, "--static-meta-feature-profile", "paper_i_production_v1")
    out = _set_option(out, "--adapt-continuation-mode", "phase3_v1")
    out = _set_option(out, "--phase2-novelty-mode", "collective_span_v1")
    out = _set_option(out, "--hardware-resolution-mode", "ideal")
    out = _set_option(out, "--phase3-selector-policy", "algebraic_nested_v1")
    out = _set_option(out, "--phase3-selector-geometry-mode", "reduced")
    out = _set_option(out, "--phase3-novelty-ablation-mode", "off")
    out = _set_option(out, "--phase3-window-relaxation-mode", "reduced")
    out = _set_toggle_pair(out, "--phase2-enable-batching", "--phase2-no-batching", True)
    out = _set_toggle_pair(out, "--phase3-enable-batching", "--phase3-no-batching", True)
    out = _set_option(out, "--phase3-batch-selection-mode", "reduced_plane")
    out = _set_option(out, "--phase3-batch-prefilter-mode", "off")
    out = _set_option(out, "--phase3-batch-order-selection-mode", "finite_step_v1")
    out = _set_toggle_pair(out, "--phase1-prune-enabled", "--phase1-no-prune", True)
    out = _set_option(out, "--phase1-prune-policy", "recoverability_ladder_v1")
    out = _set_option(out, "--phase1-prune-mode", "both")
    out = _set_toggle_pair(
        out,
        "--phase1-prune-amplitude-witness-required",
        "--phase1-prune-amplitude-witness-optional",
        False,
    )
    return out


def _build_trial_command(
    *,
    python_bin: str,
    params: TrialParams,
    case_dir: Path,
    hamiltonian_overrides: HhHamiltonianOverrides | None = None,
    phase2_w_shot_override: float | None = None,
    runtime_split_mode_override: str | None = None,
    child_pool_expansion_mode_override: str | None = None,
    child_pool_expansion_symmetry_policy_override: str | None = None,
    child_pool_expansion_max_subset_size_override: int | None = None,
    exact_gs_override: float | None = None,
    exact_gs_reference_json: Path | None = None,
    force_adapt_benchmark_target_abs_delta_e: float | None = None,
    force_adapt_max_depth: int | None = None,
    force_adapt_maxiter: int | None = None,
    force_adapt_final_refit_maxiter: int | None = None,
    force_adapt_drop_floor: float | None = None,
    force_adapt_drop_patience: int | None = None,
    force_adapt_drop_min_depth: int | None = None,
    force_adapt_full_refit_every: int | None = None,
    force_adapt_final_full_refit: str | None = None,
    force_adapt_allow_repeats: bool | None = None,
    force_phase0_pilot_max_records: int | None = None,
    force_phase1_shortlist_size: int | None = None,
    force_phase2_shortlist_fraction: float | None = None,
    force_phase2_shortlist_size: int | None = None,
    force_adapt_parallel_gradient_workers: int | None = None,
    force_adapt_beam_parent_workers: int | None = None,
    force_adapt_spsa_parallel_evaluations: int | None = None,
    force_adapt_pool_class_filter_json: Path | None = None,
    force_phase1_prune_prefilter_json: Path | None = None,
    force_adapt_resume_scaffold_json: Path | None = None,
    force_adapt_resume_mode: str | None = None,
    force_adapt_segment_id: str | None = None,
    force_adapt_segment_target_depth: int | None = None,
    force_adapt_segment_max_new_admissions: int | None = None,
    force_adapt_segment_wallclock_cap_s: float | None = None,
    force_adapt_resume_compile_smoke: str | None = None,
    force_adapt_resume_smoke_backend: str | None = None,
    force_static_route_id: str | None = None,
    force_static_meta_feature_profile: str | None = None,
    force_phase3_symmetry_mitigation_mode: str | None = None,
    force_route_a_paper_i_production: bool = False,
    force_phase1_prune_full_window: bool = False,
    force_phase1_prune_recovery_trust_radius: float | None = None,
    force_phase1_prune_schur_nomination_route: str | None = None,
    force_phase1_prune_metric_schur_mu: float | None = None,
    force_phase1_prune_metric_schur_solve_mode: str | None = None,
    force_phase1_prune_metric_schur_cost_weighting: str | None = None,
    force_skip_trajectory: bool | None = None,
    search_inner_optimizer: str = _SEARCH_INNER_OPTIMIZER,
) -> tuple[list[str], list[str], tuple[tuple[str, str], ...], dict[str, Any]]:
    search_inner_optimizer = _normalize_search_inner_optimizer(search_inner_optimizer)
    presets = _base_preset_library()
    preset = presets[str(params.base_preset)]
    pipeline_args = _apply_trial_overrides(
        params,
        preset.pipeline_args,
        search_inner_optimizer=search_inner_optimizer,
    )
    pipeline_args = _apply_hh_hamiltonian_overrides(pipeline_args, hamiltonian_overrides)
    if phase2_w_shot_override is not None:
        pipeline_args = _set_option(pipeline_args, "--phase2-w-shot", _format_cli_float(float(phase2_w_shot_override)))
    if runtime_split_mode_override is not None:
        split_mode = str(runtime_split_mode_override).strip()
        pipeline_args = _set_option(pipeline_args, "--phase3-runtime-split-mode", split_mode)
        pipeline_args = _remove_option(pipeline_args, "--allow-archival-phase3-runtime-split")
        if split_mode and split_mode != "off":
            pipeline_args = [*pipeline_args, "--allow-archival-phase3-runtime-split"]
    child_pool_expansion_enabled = False
    if child_pool_expansion_mode_override is not None:
        child_mode = str(child_pool_expansion_mode_override).strip()
        pipeline_args = _set_option(pipeline_args, "--adapt-child-pool-expansion-mode", child_mode)
        if child_mode not in {"", "off", "none", "false", "0", "disabled"}:
            child_pool_expansion_enabled = True
            pipeline_args = _set_option(pipeline_args, "--phase3-runtime-split-mode", "off")
            pipeline_args = _remove_option(pipeline_args, "--allow-archival-phase3-runtime-split")
    if child_pool_expansion_enabled and child_pool_expansion_symmetry_policy_override is None:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-child-pool-expansion-symmetry-policy",
            "hard_guard",
        )
    if child_pool_expansion_symmetry_policy_override is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-child-pool-expansion-symmetry-policy",
            str(child_pool_expansion_symmetry_policy_override).strip(),
        )
    if child_pool_expansion_max_subset_size_override is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-child-pool-expansion-max-subset-size",
            int(child_pool_expansion_max_subset_size_override),
        )
    if exact_gs_override is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-exact-gs-override", _format_cli_float(float(exact_gs_override)))
    if exact_gs_reference_json is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-exact-gs-reference-json", str(Path(exact_gs_reference_json)))
    if force_adapt_benchmark_target_abs_delta_e is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-benchmark-target-abs-delta-e",
            _format_cli_float(float(force_adapt_benchmark_target_abs_delta_e)),
        )
    if force_adapt_max_depth is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-max-depth", int(force_adapt_max_depth))
    if force_adapt_maxiter is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-maxiter", int(force_adapt_maxiter))
    if force_adapt_final_refit_maxiter is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-final-refit-maxiter",
            int(force_adapt_final_refit_maxiter),
        )
    if force_adapt_drop_floor is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-drop-floor", _format_cli_float(float(force_adapt_drop_floor)))
    if force_adapt_drop_patience is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-drop-patience", int(force_adapt_drop_patience))
    if force_adapt_drop_min_depth is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-drop-min-depth", int(force_adapt_drop_min_depth))
    if force_adapt_full_refit_every is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-full-refit-every", int(force_adapt_full_refit_every))
    if force_adapt_final_full_refit not in {None, ""}:
        value = str(force_adapt_final_full_refit).strip().lower()
        if value not in {"true", "false"}:
            raise ValueError("--force-adapt-final-full-refit must be true or false.")
        pipeline_args = _set_option(pipeline_args, "--adapt-final-full-refit", value)
    if force_adapt_allow_repeats is not None:
        pipeline_args = _set_toggle_pair(
            pipeline_args,
            "--adapt-allow-repeats",
            "--adapt-no-repeats",
            bool(force_adapt_allow_repeats),
        )
    if force_phase0_pilot_max_records is not None:
        pipeline_args = _set_option(pipeline_args, "--phase0-pilot-max-records", int(force_phase0_pilot_max_records))
    if force_phase1_shortlist_size is not None:
        pipeline_args = _set_option(pipeline_args, "--phase1-shortlist-size", int(force_phase1_shortlist_size))
    if force_phase2_shortlist_fraction is not None:
        pipeline_args = _set_option(pipeline_args, "--phase2-shortlist-fraction", _format_cli_float(float(force_phase2_shortlist_fraction)))
    if force_phase2_shortlist_size is not None:
        pipeline_args = _set_option(pipeline_args, "--phase2-shortlist-size", int(force_phase2_shortlist_size))
    if force_adapt_parallel_gradient_workers is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-parallel-gradient-workers",
            int(force_adapt_parallel_gradient_workers),
        )
    if force_adapt_beam_parent_workers is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-beam-parent-workers",
            int(force_adapt_beam_parent_workers),
        )
    if force_adapt_spsa_parallel_evaluations is not None and search_inner_optimizer == "SPSA":
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-spsa-parallel-evaluations",
            int(force_adapt_spsa_parallel_evaluations),
        )
    elif search_inner_optimizer != "SPSA":
        pipeline_args = _remove_option(pipeline_args, "--adapt-spsa-parallel-evaluations")
    if force_adapt_pool_class_filter_json is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-pool-class-filter-json",
            str(Path(force_adapt_pool_class_filter_json)),
        )
    if force_phase1_prune_prefilter_json is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--phase1-prune-prefilter-json",
            str(Path(force_phase1_prune_prefilter_json)),
        )
    if force_adapt_resume_scaffold_json is not None:
        pipeline_args = _remove_option(pipeline_args, "--adapt-ref-json")
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-resume-scaffold-json",
            str(Path(force_adapt_resume_scaffold_json)),
        )
    if force_adapt_resume_mode not in {None, ""}:
        pipeline_args = _set_option(pipeline_args, "--adapt-resume-mode", str(force_adapt_resume_mode))
    if force_adapt_segment_id not in {None, ""}:
        pipeline_args = _set_option(pipeline_args, "--adapt-segment-id", str(force_adapt_segment_id))
    if force_adapt_segment_target_depth is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-segment-target-depth", int(force_adapt_segment_target_depth))
    if force_adapt_segment_max_new_admissions is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-segment-max-new-admissions",
            int(force_adapt_segment_max_new_admissions),
        )
    if force_adapt_segment_wallclock_cap_s is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-segment-wallclock-cap-s",
            _format_cli_float(float(force_adapt_segment_wallclock_cap_s)),
        )
    if force_adapt_resume_compile_smoke not in {None, ""}:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-resume-compile-smoke",
            str(force_adapt_resume_compile_smoke),
        )
    if force_adapt_resume_smoke_backend not in {None, ""}:
        pipeline_args = _set_option(
            pipeline_args,
            "--adapt-resume-smoke-backend",
            str(force_adapt_resume_smoke_backend),
        )
    if force_static_route_id not in {None, ""}:
        pipeline_args = _set_option(pipeline_args, "--static-route-id", str(force_static_route_id))
    if force_static_meta_feature_profile not in {None, ""}:
        pipeline_args = _set_option(
            pipeline_args,
            "--static-meta-feature-profile",
            str(force_static_meta_feature_profile),
        )
    if force_phase3_symmetry_mitigation_mode not in {None, ""}:
        pipeline_args = _set_option(
            pipeline_args,
            "--phase3-symmetry-mitigation-mode",
            str(force_phase3_symmetry_mitigation_mode),
        )
    if bool(force_route_a_paper_i_production):
        pipeline_args = _apply_route_a_paper_i_production_lock(pipeline_args)
        pipeline_args = _apply_full_policy_post_route_lock_overrides(params, pipeline_args)
        if force_phase3_symmetry_mitigation_mode not in {None, ""}:
            pipeline_args = _set_option(
                pipeline_args,
                "--phase3-symmetry-mitigation-mode",
                str(force_phase3_symmetry_mitigation_mode),
            )
    if bool(force_phase1_prune_full_window):
        pipeline_args = _set_option(pipeline_args, "--phase1-prune-local-window-size", 0)
    if force_phase1_prune_recovery_trust_radius is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--phase1-prune-recovery-trust-radius",
            _format_cli_float(float(force_phase1_prune_recovery_trust_radius)),
        )
    if force_phase1_prune_schur_nomination_route not in {None, ""}:
        pipeline_args = _set_option(
            pipeline_args,
            "--phase1-prune-schur-nomination-route",
            str(force_phase1_prune_schur_nomination_route),
        )
    if force_phase1_prune_metric_schur_mu is not None:
        pipeline_args = _set_option(
            pipeline_args,
            "--phase1-prune-metric-schur-mu",
            _format_cli_float(float(force_phase1_prune_metric_schur_mu)),
        )
    if force_phase1_prune_metric_schur_solve_mode not in {None, ""}:
        pipeline_args = _set_option(
            pipeline_args,
            "--phase1-prune-metric-schur-solve-mode",
            str(force_phase1_prune_metric_schur_solve_mode),
        )
    if force_phase1_prune_metric_schur_cost_weighting not in {None, ""}:
        pipeline_args = _set_option(
            pipeline_args,
            "--phase1-prune-metric-schur-cost-weighting",
            str(force_phase1_prune_metric_schur_cost_weighting),
        )
    pipeline_args = _remove_option(pipeline_args, "--output-json")
    pipeline_args = _remove_option(pipeline_args, "--output-pdf")
    pipeline_args = _remove_option(pipeline_args, "--skip-pdf")
    pipeline_args = _remove_option(pipeline_args, "--skip-trajectory")
    pipeline_args = _remove_option(pipeline_args, "--adapt-current-json")
    pipeline_args = _remove_option(pipeline_args, "--adapt-current-json-every-depth")
    pipeline_args = _remove_option(pipeline_args, "--adapt-current-json-keep-history-tail")
    pipeline_args = [
        *pipeline_args,
        "--output-json", str(case_dir / "json" / "result.json"),
        "--adapt-current-json", str(case_dir / "current.json"),
        "--adapt-current-json-every-depth", "1",
        "--adapt-current-json-keep-history-tail", "100",
        "--skip-pdf",
    ]
    if force_skip_trajectory is not False:
        pipeline_args.append("--skip-trajectory")
    filtered_args, dropped_args = _filter_args_for_entrypoint(
        python_bin,
        preset.launcher_tokens,
        pipeline_args,
        preset.env_overrides,
    )
    effective_params = _effective_trial_params(params, dropped_args)
    effective_params["search_inner_optimizer"] = search_inner_optimizer
    if force_skip_trajectory is not None:
        effective_params["force_skip_trajectory"] = bool(force_skip_trajectory)
    override_payload = _hh_hamiltonian_override_payload(hamiltonian_overrides)
    if override_payload:
        effective_params["hamiltonian_overrides"] = dict(override_payload)
    if runtime_split_mode_override is not None:
        effective_params["runtime_split_mode_override"] = str(runtime_split_mode_override)
    if child_pool_expansion_mode_override is not None:
        effective_params["child_pool_expansion_mode_override"] = str(child_pool_expansion_mode_override)
    if child_pool_expansion_symmetry_policy_override is not None:
        effective_params["child_pool_expansion_symmetry_policy_override"] = str(
            child_pool_expansion_symmetry_policy_override
        )
    elif child_pool_expansion_enabled:
        effective_params["child_pool_expansion_symmetry_policy_override"] = "hard_guard"
    if child_pool_expansion_max_subset_size_override is not None:
        effective_params["child_pool_expansion_max_subset_size_override"] = int(
            child_pool_expansion_max_subset_size_override
        )
    if phase2_w_shot_override is not None:
        effective_params["phase2_w_shot_override"] = float(phase2_w_shot_override)
    if force_adapt_spsa_parallel_evaluations is not None and search_inner_optimizer == "SPSA":
        effective_params["force_adapt_spsa_parallel_evaluations"] = int(force_adapt_spsa_parallel_evaluations)
    elif force_adapt_spsa_parallel_evaluations is not None:
        effective_params["force_adapt_spsa_parallel_evaluations"] = "ignored_for_non_spsa_inner_optimizer"
    if exact_gs_override is not None:
        effective_params["adapt_exact_gs_override"] = float(exact_gs_override)
    if exact_gs_reference_json is not None:
        effective_params["adapt_exact_gs_reference_json"] = str(Path(exact_gs_reference_json))
    if force_adapt_benchmark_target_abs_delta_e is not None:
        effective_params["force_adapt_benchmark_target_abs_delta_e"] = float(force_adapt_benchmark_target_abs_delta_e)
    if force_adapt_max_depth is not None:
        effective_params["force_adapt_max_depth"] = int(force_adapt_max_depth)
    if force_adapt_maxiter is not None:
        effective_params["force_adapt_maxiter"] = int(force_adapt_maxiter)
    if force_adapt_final_refit_maxiter is not None:
        effective_params["force_adapt_final_refit_maxiter"] = int(force_adapt_final_refit_maxiter)
    if force_adapt_drop_floor is not None:
        effective_params["force_adapt_drop_floor"] = float(force_adapt_drop_floor)
    if force_adapt_drop_patience is not None:
        effective_params["force_adapt_drop_patience"] = int(force_adapt_drop_patience)
    if force_adapt_drop_min_depth is not None:
        effective_params["force_adapt_drop_min_depth"] = int(force_adapt_drop_min_depth)
    if force_adapt_full_refit_every is not None:
        effective_params["force_adapt_full_refit_every"] = int(force_adapt_full_refit_every)
    if force_adapt_final_full_refit not in {None, ""}:
        effective_params["force_adapt_final_full_refit"] = str(force_adapt_final_full_refit).strip().lower()
    if force_adapt_allow_repeats is not None:
        effective_params["force_adapt_allow_repeats"] = bool(force_adapt_allow_repeats)
    if force_phase0_pilot_max_records is not None:
        effective_params["force_phase0_pilot_max_records"] = int(force_phase0_pilot_max_records)
    if force_phase1_shortlist_size is not None:
        effective_params["force_phase1_shortlist_size"] = int(force_phase1_shortlist_size)
    if force_phase2_shortlist_fraction is not None:
        effective_params["force_phase2_shortlist_fraction"] = float(force_phase2_shortlist_fraction)
    if force_phase2_shortlist_size is not None:
        effective_params["force_phase2_shortlist_size"] = int(force_phase2_shortlist_size)
    if force_adapt_parallel_gradient_workers is not None:
        effective_params["force_adapt_parallel_gradient_workers"] = int(force_adapt_parallel_gradient_workers)
    if force_adapt_beam_parent_workers is not None:
        effective_params["force_adapt_beam_parent_workers"] = int(force_adapt_beam_parent_workers)
    if force_adapt_pool_class_filter_json is not None:
        effective_params["force_adapt_pool_class_filter_json"] = str(Path(force_adapt_pool_class_filter_json))
    if force_phase1_prune_prefilter_json is not None:
        effective_params["force_phase1_prune_prefilter_json"] = str(Path(force_phase1_prune_prefilter_json))
    if bool(force_phase1_prune_full_window):
        effective_params["force_phase1_prune_full_window"] = True
    if force_phase1_prune_recovery_trust_radius is not None:
        effective_params["force_phase1_prune_recovery_trust_radius"] = float(
            force_phase1_prune_recovery_trust_radius
        )
    if force_adapt_resume_scaffold_json is not None:
        effective_params["force_adapt_resume_scaffold_json"] = str(Path(force_adapt_resume_scaffold_json))
    if force_adapt_resume_mode not in {None, ""}:
        effective_params["force_adapt_resume_mode"] = str(force_adapt_resume_mode)
    if force_adapt_segment_id not in {None, ""}:
        effective_params["force_adapt_segment_id"] = str(force_adapt_segment_id)
    if force_adapt_segment_target_depth is not None:
        effective_params["force_adapt_segment_target_depth"] = int(force_adapt_segment_target_depth)
    if force_adapt_segment_max_new_admissions is not None:
        effective_params["force_adapt_segment_max_new_admissions"] = int(force_adapt_segment_max_new_admissions)
    if force_adapt_segment_wallclock_cap_s is not None:
        effective_params["force_adapt_segment_wallclock_cap_s"] = float(force_adapt_segment_wallclock_cap_s)
    if force_adapt_resume_compile_smoke not in {None, ""}:
        effective_params["force_adapt_resume_compile_smoke"] = str(force_adapt_resume_compile_smoke)
    if force_adapt_resume_smoke_backend not in {None, ""}:
        effective_params["force_adapt_resume_smoke_backend"] = str(force_adapt_resume_smoke_backend)
    if force_static_route_id not in {None, ""}:
        effective_params["force_static_route_id"] = str(force_static_route_id)
    if force_static_meta_feature_profile not in {None, ""}:
        effective_params["force_static_meta_feature_profile"] = str(force_static_meta_feature_profile)
    if force_phase3_symmetry_mitigation_mode not in {None, ""}:
        effective_params["force_phase3_symmetry_mitigation_mode"] = str(force_phase3_symmetry_mitigation_mode)
    if bool(force_route_a_paper_i_production):
        effective_params["force_route_a_paper_i_production"] = True
    return (
        [str(python_bin), *list(preset.launcher_tokens), *filtered_args],
        dropped_args,
        tuple(preset.env_overrides),
        effective_params,
    )


def _write_command_log(path: Path, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join([str(x) for x in command]) + "\n",
        encoding="utf-8",
    )


def _completed_result_artifact_ready(path: Path) -> bool:
    """Return true when an adapt_pipeline result artifact is complete enough to ingest."""
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, Mapping):
        return False
    adapt_vqe = payload.get("adapt_vqe", {})
    return isinstance(adapt_vqe, Mapping) and bool(adapt_vqe)


def _run_subprocess_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    env_overrides: Mapping[str, str] | None = None,
    live_progress_callback: Callable[[], None] | None = None,
    live_progress_interval_s: float = 15.0,
    completed_result_path: Path | None = None,
    completed_result_stable_s: float = 180.0,
    completed_result_terminate_grace_s: float = 20.0,
) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
        env = dict(os.environ)
        if env_overrides:
            env.update({str(k): str(v) for k, v in env_overrides.items()})
        proc = subprocess.Popen(
            [str(x) for x in command],
            cwd=str(cwd),
            text=True,
            stdout=stdout_fh,
            stderr=stderr_fh,
            env=env,
        )
        callback_interval = float(max(1.0, live_progress_interval_s))
        last_callback_s = 0.0
        while True:
            now_s = time.perf_counter()
            if live_progress_callback is not None and now_s - last_callback_s >= callback_interval:
                try:
                    live_progress_callback()
                except Exception as exc:
                    stderr_fh.write(
                        f"\n[hh_cost_energy_optuna] live_progress_callback_failed: {type(exc).__name__}: {exc}\n"
                    )
                    stderr_fh.flush()
                last_callback_s = now_s
            returncode = proc.poll()
            if returncode is not None:
                break
            if completed_result_path is not None:
                result_path = Path(completed_result_path)
                try:
                    result_stat = result_path.stat()
                    result_age_s = time.time() - float(result_stat.st_mtime)
                except OSError:
                    result_age_s = -1.0
                if result_age_s >= float(max(1.0, completed_result_stable_s)) and _completed_result_artifact_ready(result_path):
                    stderr_fh.write(
                        "\n[hh_cost_energy_optuna] subprocess_result_artifact_complete; "
                        f"terminating lingering child after {result_age_s:.1f}s stable result: {result_path}\n"
                    )
                    stderr_fh.flush()
                    proc.terminate()
                    try:
                        proc.wait(timeout=float(max(1.0, completed_result_terminate_grace_s)))
                    except subprocess.TimeoutExpired:
                        stderr_fh.write(
                            "[hh_cost_energy_optuna] lingering child ignored terminate; sending kill\n"
                        )
                        stderr_fh.flush()
                        proc.kill()
                        proc.wait()
                    returncode = 0
                    break
            time.sleep(min(1.0, callback_interval))
        if live_progress_callback is not None:
            try:
                live_progress_callback()
            except Exception as exc:
                stderr_fh.write(
                    f"\n[hh_cost_energy_optuna] live_progress_callback_failed: {type(exc).__name__}: {exc}\n"
                )
                stderr_fh.flush()
    elapsed_s = float(time.perf_counter() - started)
    return int(returncode), elapsed_s


def _extract_abs_delta_e(payload: Mapping[str, Any]) -> float | None:
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    if isinstance(adapt_vqe, Mapping) and adapt_vqe.get("abs_delta_e") is not None:
        try:
            value = float(adapt_vqe.get("abs_delta_e"))
        except Exception:
            return None
        return value if math.isfinite(value) else None
    if isinstance(adapt_vqe, Mapping):
        energy = adapt_vqe.get("energy")
        exact = adapt_vqe.get("exact_gs_energy")
        if energy is not None and exact is not None:
            try:
                value = abs(float(energy) - float(exact))
            except Exception:
                return None
            return value if math.isfinite(value) else None
    return None


def _extract_history_signature(payload: Mapping[str, Any], limit: int = 6) -> tuple[list[str], list[str]]:
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    history = adapt_vqe.get("history", []) if isinstance(adapt_vqe, Mapping) else []
    families: list[str] = []
    ops: list[str] = []
    if not isinstance(history, Sequence):
        return families, ops
    for row in history[: int(max(0, limit))]:
        if not isinstance(row, Mapping):
            continue
        op = row.get("selected_op")
        family = row.get("candidate_family")
        if family in {None, ""} and op not in {None, ""}:
            family = str(op).split(":", 1)[0]
        if family not in {None, ""}:
            families.append(str(family))
        if op not in {None, ""}:
            ops.append(str(op))
    return families, ops


def _extract_adapt_iteration_metrics(
    payload: Mapping[str, Any],
    *,
    target_abs_delta_e: float | None = None,
    target_iteration: int | None = None,
) -> tuple[int | None, int | None, float | None, int | None]:
    """Return final accepted-iteration count and first prefix crossing.

    The Paper-I HH convergence tables use accepted ADAPT iteration/prefix depth,
    not optimized-ansatz parameter count.  The stable per-prefix field is
    ``adapt_vqe.history[*].depth_cumulative`` with ``depth``/row index fallback.
    """
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    history = adapt_vqe.get("history", []) if isinstance(adapt_vqe, Mapping) else []
    final_iteration: int | None = None
    first_crossing: int | None = None
    best_prefix_delta: float | None = None
    best_prefix_iteration: int | None = None
    target = _maybe_float(target_abs_delta_e)
    allowed_iteration: int | None = None
    if target_iteration is not None:
        try:
            allowed_iteration = max(0, int(target_iteration) - 1)
        except Exception:
            allowed_iteration = None
    if isinstance(history, Sequence):
        for idx, row in enumerate(history):
            if not isinstance(row, Mapping):
                continue
            depth_raw = row.get("depth_cumulative", row.get("depth", idx + 1))
            try:
                depth = int(depth_raw)
            except Exception:
                depth = int(idx + 1)
            final_iteration = max(int(final_iteration or 0), int(depth))
            row_delta = _maybe_float(row.get("delta_abs_current"))
            if row_delta is None:
                row_delta = _maybe_float(row.get("exact_abs_delta_e_from_final_state"))
            if row_delta is not None and (allowed_iteration is None or int(depth) <= int(allowed_iteration)):
                if best_prefix_delta is None or float(row_delta) < float(best_prefix_delta):
                    best_prefix_delta = float(row_delta)
                    best_prefix_iteration = int(depth)
            if target is not None and first_crossing is None:
                if row_delta is not None and float(row_delta) <= float(target):
                    first_crossing = int(depth)
    if final_iteration is None and isinstance(adapt_vqe, Mapping):
        # Fallback only. Prefer prefix depth over ansatz_depth because batching can
        # make ansatz parameter count differ from accepted ADAPT iteration count.
        for key in ("depth_cumulative", "adapt_iteration_count", "adapt_depth"):
            value = _maybe_float(adapt_vqe.get(key))
            if value is not None:
                final_iteration = int(round(float(value)))
                break
    if target is not None and first_crossing is None:
        abs_delta_e = _extract_abs_delta_e(payload)
        if abs_delta_e is not None and final_iteration is not None and float(abs_delta_e) <= float(target):
            first_crossing = int(final_iteration)
    if best_prefix_delta is None and final_iteration is not None:
        abs_delta_e = _extract_abs_delta_e(payload)
        if abs_delta_e is not None and (allowed_iteration is None or int(final_iteration) <= int(allowed_iteration)):
            best_prefix_delta = float(abs_delta_e)
            best_prefix_iteration = int(final_iteration)
    return final_iteration, first_crossing, best_prefix_delta, best_prefix_iteration


def _adapt_history_position_for_prefix_iteration(
    payload: Mapping[str, Any],
    prefix_iteration: int | None,
) -> int | None:
    """Map an accepted-prefix depth to a 1-based history slice position.

    Batched SNAKE rows may report ``depth_cumulative`` values that are not
    identical to the number of accepted history rows.  Prefix-scoped Paper-I
    shot/work reconstruction needs the history row position, not the display
    depth.  Return the first history row whose cumulative depth reaches the
    requested prefix depth.
    """
    if prefix_iteration is None:
        return None
    try:
        target = int(prefix_iteration)
    except Exception:
        return None
    if target < 1:
        return None
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    history = adapt_vqe.get("history", []) if isinstance(adapt_vqe, Mapping) else []
    if not isinstance(history, Sequence):
        return None
    for idx, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            continue
        depth_raw = row.get("depth_cumulative", row.get("depth", idx))
        try:
            depth = int(depth_raw)
        except Exception:
            depth = int(idx)
        if depth >= target:
            return int(idx)
    if target <= len(history):
        return int(target)
    return None


def _extract_compile_metrics(payload: Mapping[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
    selected_backend = payload.get("selected_backend", {}) if isinstance(payload, Mapping) else {}
    logical_circuit = payload.get("logical_circuit", {}) if isinstance(payload, Mapping) else {}
    def _maybe_int(raw: Any) -> int | None:
        try:
            value = int(raw)
        except Exception:
            return None
        return value
    return (
        _maybe_int(selected_backend.get("compiled_count_2q")),
        _maybe_int(selected_backend.get("compiled_depth")),
        _maybe_int(logical_circuit.get("logical_parameter_count")),
        _maybe_int(logical_circuit.get("runtime_parameter_count")),
    )


def _maybe_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _extract_graph_cost_metrics(
    payload: Mapping[str, Any],
    *,
    paper_i_shot_scope: str = "terminal",
    paper_i_shot_history_position: int | None = None,
) -> dict[str, Any]:
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    summary = adapt_vqe.get("backend_compile_cost_summary", {}) if isinstance(adapt_vqe, Mapping) else {}
    if not isinstance(summary, Mapping):
        summary = {}
    work = adapt_vqe.get("controller_measurement_work_summary", {}) if isinstance(adapt_vqe, Mapping) else {}
    if not isinstance(work, Mapping):
        work = {}
    paper_i_shots_total: float | None = None
    paper_i_s_alg: float | None = None
    paper_i_shots_status: str | None = None
    if isinstance(payload, Mapping) and payload:
        try:
            paper_i_fields, paper_i_audit = snake_deterministic_shot_proxy_from_payload(
                payload,
                scope=str(paper_i_shot_scope or "terminal"),
                history_position=paper_i_shot_history_position,
                shots_per_pauli_term_proxy=_PAPER_I_SHOTS_PER_PAULI_TERM_PROXY,
            )
            paper_i_shots_total = _maybe_float(paper_i_fields.get("shots_total"))
            paper_i_s_alg = _maybe_float(paper_i_audit.get("S_alg"))
            if paper_i_s_alg is None:
                nested_audit = paper_i_fields.get("snake_deterministic_shot_proxy")
                if isinstance(nested_audit, Mapping):
                    paper_i_s_alg = _maybe_float(nested_audit.get("S_alg"))
            paper_i_shots_status = str(paper_i_audit.get("status") or "unknown")
        except Exception as exc:
            paper_i_shots_status = f"error:{type(exc).__name__}"
    source = summary.get("mode") or adapt_vqe.get("compile_cost_mode") if isinstance(adapt_vqe, Mapping) else None
    return {
        "graph_count_2q": _maybe_float(summary.get("absolute_c_hat_2q")),
        "graph_depth": _maybe_float(summary.get("absolute_c_hat_d")),
        "graph_count_1q": _maybe_float(summary.get("absolute_c_hat_1q")),
        "graph_theta_count": _maybe_float(summary.get("absolute_theta_count")),
        "measurement_work_shots": _maybe_float(work.get("shots_total") or work.get("total_shots_new") or work.get("reuse_count_cost")),
        "paper_i_table_shots_total": paper_i_shots_total,
        "paper_i_table_s_alg": paper_i_s_alg,
        "paper_i_table_shots_status": paper_i_shots_status,
        "paper_i_table_shots_scope": str(paper_i_shot_scope or "terminal"),
        "paper_i_table_shots_history_position": (
            None if paper_i_shot_history_position is None else int(paper_i_shot_history_position)
        ),
        "measurement_work_records": _maybe_float(work.get("records_evaluated")),
        "resource_cost_source": None if source in {None, ""} else str(source),
    }


def _objective_lexicographic(
    compiled_count_2q: int | None,
    compiled_depth: int | None,
    logical_operator_count: int | None,
    runtime_parameter_count: int | None,
) -> int:
    if compiled_count_2q is None or compiled_depth is None:
        return _LARGE_OBJECTIVE
    logical_val = int(logical_operator_count or 0)
    runtime_val = int(runtime_parameter_count or 0)
    return (
        int(compiled_count_2q) * 10**12
        + int(compiled_depth) * 10**6
        + int(logical_val) * 10**3
        + int(runtime_val)
    )


def _paper_i_shot_cost_scalar(obs: TrialObservation) -> float:
    """Return the Paper-I normalized shot/work proxy used by Optuna.

    Prefer ``S_alg`` because it is the normalized algorithmic work quantity used
    by the Paper-I SNAKE shot-proxy support.  The deterministic ``shots_total``
    field is a scaled reporting proxy; use it only when ``S_alg`` is unavailable.
    Fall back to controller measurement-work telemetry only when the table proxy
    is unavailable, and penalize missing shot telemetry instead of silently
    treating it as free.
    """
    if obs.paper_i_table_s_alg is not None:
        shots_source = obs.paper_i_table_s_alg
    elif obs.paper_i_table_shots_total is not None:
        shots_source = obs.paper_i_table_shots_total
    else:
        shots_source = obs.measurement_work_shots
    shots = _maybe_float(shots_source)
    if shots is None:
        return float(_RESOURCE_OBJECTIVE_MISSING_SHOTS)
    return float(max(0.0, shots))


def _graph_hardware_objective_scalar(obs: TrialObservation) -> float:
    graph_count_2q = _maybe_float(obs.graph_count_2q)
    graph_depth = _maybe_float(obs.graph_depth)
    if graph_count_2q is None or graph_depth is None:
        return float(_LARGE_OBJECTIVE)
    graph_count_1q = _maybe_float(obs.graph_count_1q) or 0.0
    graph_theta_count = _maybe_float(obs.graph_theta_count) or 0.0
    return float(
        max(0.0, graph_count_2q) * _RESOURCE_OBJECTIVE_GRAPH_2Q_WEIGHT
        + max(0.0, graph_depth) * _RESOURCE_OBJECTIVE_GRAPH_DEPTH_WEIGHT
        + max(0.0, graph_count_1q) * _RESOURCE_OBJECTIVE_GRAPH_1Q_WEIGHT
        + max(0.0, graph_theta_count) * _RESOURCE_OBJECTIVE_GRAPH_THETA_WEIGHT
    )


def _graph_speed_objective(obs: TrialObservation) -> float:
    hardware = _graph_hardware_objective_scalar(obs)
    if hardware >= float(_LARGE_OBJECTIVE):
        return float(_LARGE_OBJECTIVE)
    shots = min(_paper_i_shot_cost_scalar(obs), float(_RESOURCE_OBJECTIVE_MISSING_SHOTS))
    elapsed_ms = 0.0 if obs.total_elapsed_s is None else max(0.0, float(obs.total_elapsed_s) * 1000.0)
    # After the energy gate, optimize the same resource family the Paper-I HH
    # table cares about: Marrakesh graph-span hardware cost plus deterministic
    # Paper-I shot/work proxy.  Wallclock remains only a weak tie-breaker.
    return float(
        hardware
        + shots * _RESOURCE_OBJECTIVE_SHOT_WEIGHT
        + min(elapsed_ms, 1.0e9) * _RESOURCE_OBJECTIVE_ELAPSED_MS_WEIGHT
    )


def _geo_dominance_first_objective(obs: TrialObservation) -> float:
    """Lexicographic-like objective for beating a Geo-ADAPT comparator first.

    Priority:
      1. beat Geo-ADAPT final/prefix energy error;
      2. cross that energy level at a strictly lower accepted ADAPT iteration;
      3. after both are satisfied, minimize graph/circuit cost, shots, and time.
    """
    target_e = _maybe_float(obs.dominance_target_abs_delta_e)
    target_k = obs.dominance_target_iteration
    abs_delta_e = _maybe_float(obs.abs_delta_e)
    if target_e is None or target_k is None or int(target_k) <= 0:
        return float(_LARGE_OBJECTIVE)
    prefix_delta = _maybe_float(obs.dominance_prefix_abs_delta_e)
    energy_signal = prefix_delta if prefix_delta is not None else abs_delta_e
    if energy_signal is None:
        return float(_LARGE_OBJECTIVE)
    allowed_k = max(0, int(target_k) - 1)
    crossing_k = obs.dominance_first_crossing_iteration
    energy_gap = max(0.0, float(energy_signal) - float(target_e))
    if energy_gap > 0.0:
        # Missing energy dominance is the top-level failure. Keep the magnitude
        # visible to TPE while optimizing the early prefix, not the final depth.
        return float(3.0e15 + min(energy_gap, 1.0e3) * 1.0e18)
    if crossing_k is None:
        return float(2.9e15)
    iter_gap = max(0, int(crossing_k) - int(allowed_k))
    if iter_gap > 0:
        return float(2.0e15 + float(iter_gap) * 1.0e12 + float(energy_signal) * 1.0e15)
    # Dominance achieved. Cost/shots/time now break ties, with smaller crossing
    # depth and smaller residual retained as secondary signals.
    graph_speed = _graph_speed_objective(obs)
    return float(float(crossing_k) * 1.0e12 + float(energy_signal) * 1.0e15 + min(graph_speed, 1.0e12))


def _graph_dominance_violations(obs: TrialObservation) -> tuple[float | None, float | None]:
    target_n2q = _maybe_float(obs.dominance_target_graph_count_2q)
    target_depth = _maybe_float(obs.dominance_target_graph_depth)
    graph_n2q = _maybe_float(obs.graph_count_2q)
    graph_depth = _maybe_float(obs.graph_depth)
    n2q_violation = None if target_n2q is None or graph_n2q is None else max(0.0, float(graph_n2q) - float(target_n2q))
    depth_violation = None if target_depth is None or graph_depth is None else max(0.0, float(graph_depth) - float(target_depth))
    return n2q_violation, depth_violation


def _shot_dominance_violation(obs: TrialObservation) -> float | None:
    target_s_alg = _maybe_float(obs.dominance_target_s_alg)
    if target_s_alg is None:
        return None
    s_alg = _maybe_float(obs.paper_i_table_s_alg)
    if s_alg is None:
        return 1.0
    return float(max(0.0, float(s_alg) - float(target_s_alg)))


def _geo_energy_then_graph_cost_objective(obs: TrialObservation) -> float:
    """Energy-gated objective for SNAKE-vs-Geo HH Pareto search.

    Priority:
      1. reach at least the Geo-ADAPT plateau/final energy error;
      2. after that gate is satisfied, minimize graph hardware cost, shot work,
         and wallclock. Iteration is telemetry only for this objective.

    This mode now uses the Paper-I graph+shot objective.  Use the explicit
    ``geo_energy_then_graph_shot_cost`` alias for new studies so Optuna storage
    does not mix old graph-only values with graph+shot values.
    """
    target_e = _maybe_float(obs.dominance_target_abs_delta_e)
    abs_delta_e = _maybe_float(obs.abs_delta_e)
    energy_gate_penalty = 1.0e30
    if target_e is None or abs_delta_e is None:
        return float(energy_gate_penalty)
    energy_gap = max(0.0, float(abs_delta_e) - float(target_e))
    if energy_gap > 0.0:
        return float(energy_gate_penalty + min(energy_gap, 1.0e3) * 1.0e33)
    graph_n2q_violation, graph_depth_violation = _graph_dominance_violations(obs)
    if graph_n2q_violation is not None and graph_depth_violation is not None:
        graph_gap = float(graph_n2q_violation) * _RESOURCE_OBJECTIVE_GRAPH_2Q_WEIGHT + float(graph_depth_violation) * _RESOURCE_OBJECTIVE_GRAPH_DEPTH_WEIGHT
        if graph_gap > 0.0:
            return float(5.0e20 + min(graph_gap, 1.0e18))
    shot_violation = _shot_dominance_violation(obs)
    if shot_violation is not None and float(shot_violation) > 0.0:
        return float(4.0e20 + min(float(shot_violation), 1.0e12) * 1.0e12)
    graph_speed = _graph_speed_objective(obs)
    # Once energy is good enough, cost/shot dominates. Keep residual energy as a
    # small tie-break so TPE still prefers deeper energy improvements at similar
    # cost, but never over a materially cheaper energy-feasible circuit.
    return float(graph_speed + float(abs_delta_e) * 1.0e9)


def _geo_energy_then_shot_graph_cost_objective(obs: TrialObservation) -> float:
    """Energy-gated HH objective that prioritizes Paper-I S_alg reduction.

    Priority:
      1. reach the Geo-ADAPT energy target;
      2. beat or match Geo S_alg when that target is available;
      3. then beat graph proxy targets;
      4. finally minimize the shot/work scalar before graph cost.

    This is intentionally a separate objective mode so historical
    graph-dominant studies keep interpretable objective values.
    """
    target_e = _maybe_float(obs.dominance_target_abs_delta_e)
    abs_delta_e = _maybe_float(obs.abs_delta_e)
    energy_gate_penalty = 1.0e30
    if target_e is None or abs_delta_e is None:
        return float(energy_gate_penalty)
    energy_gap = max(0.0, float(abs_delta_e) - float(target_e))
    if energy_gap > 0.0:
        return float(energy_gate_penalty + min(energy_gap, 1.0e3) * 1.0e33)

    shot_violation = _shot_dominance_violation(obs)
    if shot_violation is not None and float(shot_violation) > 0.0:
        return float(6.0e20 + min(float(shot_violation), 1.0e12) * 1.0e12)

    graph_n2q_violation, graph_depth_violation = _graph_dominance_violations(obs)
    if graph_n2q_violation is not None and graph_depth_violation is not None:
        graph_gap = (
            float(graph_n2q_violation) * _RESOURCE_OBJECTIVE_GRAPH_2Q_WEIGHT
            + float(graph_depth_violation) * _RESOURCE_OBJECTIVE_GRAPH_DEPTH_WEIGHT
        )
        if graph_gap > 0.0:
            return float(5.0e20 + min(graph_gap, 1.0e18))

    shot_scalar = min(_paper_i_shot_cost_scalar(obs), float(_RESOURCE_OBJECTIVE_MISSING_SHOTS))
    hardware = _graph_hardware_objective_scalar(obs)
    elapsed_ms = 0.0 if obs.total_elapsed_s is None else max(0.0, float(obs.total_elapsed_s) * 1000.0)
    return float(
        shot_scalar * 1.0e9
        + min(hardware, 1.0e18)
        + min(elapsed_ms, 1.0e9) * _RESOURCE_OBJECTIVE_ELAPSED_MS_WEIGHT
        + float(abs_delta_e) * 1.0e9
    )


def _shot_then_energy_graph_cost_objective(obs: TrialObservation) -> float:
    """Shot-first HH objective: S_alg, then energy, then graph proxy.

    This mode is intentionally distinct from the Geo-energy-gated objectives so
    persistent Optuna studies do not mix incompatible objective values. It is
    for S_alg minimization sweeps where energy quality remains the first
    tie-breaker after the shot/work scalar.
    """
    shot_scalar = min(_paper_i_shot_cost_scalar(obs), float(_RESOURCE_OBJECTIVE_MISSING_SHOTS))
    abs_delta_e = _maybe_float(obs.abs_delta_e)
    target_e = _maybe_float(obs.dominance_target_abs_delta_e)
    if abs_delta_e is None:
        energy_scalar = 2.0e9
    elif target_e is not None and float(abs_delta_e) > float(target_e):
        energy_gap = float(abs_delta_e) - float(target_e)
        energy_scalar = 1.0e9 + min(max(0.0, energy_gap) * 1.0e9, 1.0e9)
    else:
        energy_scalar = min(max(0.0, float(abs_delta_e)) * 1.0e9, 1.0e9)
    graph_scalar = min(_graph_hardware_objective_scalar(obs), 1.0e18)
    elapsed_ms = 0.0 if obs.total_elapsed_s is None else max(0.0, float(obs.total_elapsed_s) * 1000.0)
    return float(
        shot_scalar * 1.0e24
        + energy_scalar * 1.0e9
        + graph_scalar
        + min(elapsed_ms, 1.0e9) * _RESOURCE_OBJECTIVE_ELAPSED_MS_WEIGHT
    )


def _geo_energy_gate_then_shot_energy_graph_cost_objective(obs: TrialObservation) -> float:
    """Energy-gated shot objective: Geo energy gate, then S_alg, Delta E, graph.

    Any trial whose final same-cutoff Delta E is worse than the Geo-ADAPT target
    is ranked behind every energy-feasible trial.  Within the feasible set,
    S_alg is the dominant target; residual energy and graph proxy are only
    tie-breakers.
    """
    target_e = _maybe_float(obs.dominance_target_abs_delta_e)
    abs_delta_e = _maybe_float(obs.abs_delta_e)
    energy_gate_penalty = 1.0e30
    shot_scalar = min(_paper_i_shot_cost_scalar(obs), float(_RESOURCE_OBJECTIVE_MISSING_SHOTS))
    graph_scalar = min(_graph_hardware_objective_scalar(obs), 1.0e14)
    elapsed_ms = 0.0 if obs.total_elapsed_s is None else max(0.0, float(obs.total_elapsed_s) * 1000.0)
    if target_e is None or abs_delta_e is None:
        return float(energy_gate_penalty + shot_scalar * 1.0e6 + graph_scalar * 1.0e-6)
    energy_gap = max(0.0, float(abs_delta_e) - float(target_e))
    if energy_gap > 0.0:
        return float(
            energy_gate_penalty
            + min(energy_gap, 1.0e3) * 1.0e33
            + shot_scalar * 1.0e6
            + graph_scalar * 1.0e-6
            + min(elapsed_ms, 1.0e9) * _RESOURCE_OBJECTIVE_ELAPSED_MS_WEIGHT
        )
    return float(
        shot_scalar * 1.0e12
        + min(max(0.0, float(abs_delta_e)), 1.0) * 1.0e14
        + graph_scalar * 1.0e-3
        + min(elapsed_ms, 1.0e9) * _RESOURCE_OBJECTIVE_ELAPSED_MS_WEIGHT
    )


def _prune_zero_violation_count(obs: TrialObservation) -> int:
    if obs.prune_actual_rollback_count is not None:
        return int(max(0, int(obs.prune_actual_rollback_count or 0)))
    return int(
        int(obs.prune_no_accept_restore_pass_count or 0)
        + int(obs.prune_accepted_then_guard_rolled_back_count or 0)
    )


def _prune_zero_then_energy_shot_graph_cost_objective(obs: TrialObservation) -> float:
    prune_violation = _prune_zero_violation_count(obs)
    if prune_violation > 0:
        return float(1.0e40 + min(int(prune_violation), 10**9) * 1.0e30)
    return _geo_energy_gate_then_shot_energy_graph_cost_objective(obs)


_SEARCH_CONTROL_FLAGS = {
    "--adapt-max-depth",
    "--adapt-benchmark-target-abs-delta-e",
    "--adapt-inner-optimizer",
    "--adapt-maxiter",
    "--adapt-parallel-gradient-workers",
    "--adapt-beam-parent-workers",
    "--adapt-spsa-a",
    "--adapt-spsa-c",
    "--adapt-spsa-alpha",
    "--adapt-spsa-gamma",
    "--adapt-spsa-A",
    "--adapt-spsa-avg-last",
    "--adapt-spsa-eval-repeats",
    "--adapt-spsa-eval-agg",
    "--adapt-spsa-callback-every",
    "--adapt-spsa-progress-every-s",
    "--adapt-continuation-mode",
    "--static-route-id",
    "--static-meta-feature-profile",
    "--phase2-novelty-mode",
    "--hardware-resolution-mode",
    "--phase3-selector-policy",
    "--phase3-selector-geometry-mode",
    "--phase3-novelty-ablation-mode",
    "--phase3-window-relaxation-mode",
    "--phase3-runtime-split-mode",
    "--allow-archival-phase3-runtime-split",
    "--phase2-enable-batching",
    "--phase2-no-batching",
    "--phase3-enable-batching",
    "--phase3-no-batching",
    "--phase3-batch-selection-mode",
    "--phase3-batch-prefilter-mode",
    "--adapt-no-repeats",
    "--phase3-backend-cost-mode",
    "--phase3-backend-name",
    "--phase3-backend-transpile-seed",
    "--phase3-backend-optimization-level",
    "--phase2-w-shot",
    "--phase3-motif-source-json",
    "--phase1-no-prune",
    "--phase1-prune-enabled",
    "--phase1-prune-policy",
    "--phase1-prune-mode",
    "--phase1-prune-amplitude-witness-required",
    "--phase1-prune-amplitude-witness-optional",
    *_MATURITY_SHORTLIST_FLAGS,
    *_MATURITY_SHOT_FLAGS,
    *_PHASE_LIVE_FLAGS,
    *_PRUNE_PREFILTER_FLAGS,
    *_PRUNE_WITNESS_FLAGS,
    *_ADAPT_WINDOW_FLAGS,
    *_BACKEND_COST_WEIGHT_FLAGS,
    *_ML_CANDIDATE_PROFILE_FLAGS,
    *_SPSA_PROFILE_FLAGS,
}


def _compute_constraints(
    *,
    abs_delta_e: float | None,
    epsilon_abs_delta_e: float,
    returncode: int | None,
    compile_returncode: int | None,
    compiled_count_2q: int | None,
    dropped_search_flags: Sequence[str] = (),
    require_compile: bool = True,
    prune_rejected_delete_attempt_count: int | None = None,
    prune_no_accept_restore_pass_count: int | None = None,
    prune_accepted_then_guard_rolled_back_count: int | None = None,
    prune_actual_rollback_count: int | None = None,
    require_zero_prune_rejections: bool = False,
) -> list[float]:
    energy_violation = float(max(0.0, float(abs_delta_e) - float(epsilon_abs_delta_e))) if abs_delta_e is not None else float(1.0)
    run_violation = 0.0 if int(returncode or 0) == 0 else 1.0
    compile_violation = (
        0.0
        if not bool(require_compile)
        else (0.0 if int(compile_returncode or 0) == 0 and compiled_count_2q is not None else 1.0)
    )
    dropped_violation = 1.0 if any(str(flag) in _SEARCH_CONTROL_FLAGS for flag in dropped_search_flags) else 0.0
    constraints = [float(energy_violation), float(run_violation + compile_violation + dropped_violation)]
    if bool(require_zero_prune_rejections):
        prune_violation = (
            int(prune_actual_rollback_count)
            if prune_actual_rollback_count is not None
            else (
                int(prune_no_accept_restore_pass_count or 0)
                + int(prune_accepted_then_guard_rolled_back_count or 0)
            )
        )
        constraints.append(float(max(0, prune_violation)))
    return constraints


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable copy of lightweight monitoring payload data."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        return _json_safe(asdict(value))
    except Exception:
        return str(value)


def _history_row_delta(row: Mapping[str, Any]) -> float | None:
    for key in ("delta_abs_current", "exact_abs_delta_e_from_final_state", "abs_delta_e", "energy_error_abs"):
        value = _maybe_float(row.get(key))
        if value is not None:
            return float(value)
    return None


def _history_row_iteration(row: Mapping[str, Any], fallback_index: int) -> int:
    for key in ("depth_cumulative", "adapt_iteration_count", "adapt_depth", "depth"):
        value = _maybe_float(row.get(key))
        if value is not None:
            return int(round(float(value)))
    return int(fallback_index + 1)


def _live_current_progress_payload(
    *,
    current_json: Path,
    case_dir: Path,
    output_dir: Path,
    trial_index: int,
    lane: str,
    epsilon_abs_delta_e: float,
    effective_params: Mapping[str, Any],
    dominance_target_abs_delta_e: float | None,
    dominance_target_iteration: int | None,
    dominance_target_graph_count_2q: float | None,
    dominance_target_graph_depth: float | None,
    dominance_target_s_alg: float | None,
) -> dict[str, Any] | None:
    if not current_json.exists():
        return None
    try:
        payload = json.loads(current_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(adapt_vqe, Mapping):
        return None
    beam = adapt_vqe.get("beam_replay_telemetry", {})
    beam = beam if isinstance(beam, Mapping) else {}
    current_round = beam.get("current_round", {})
    current_round = current_round if isinstance(current_round, Mapping) else {}
    live_iteration_raw = current_round.get("depth")
    if live_iteration_raw is None:
        live_iteration_raw = adapt_vqe.get("ansatz_depth") or adapt_vqe.get("history_count")
    live_iteration = None if live_iteration_raw is None else int(round(float(live_iteration_raw)))
    live_delta = _maybe_float(adapt_vqe.get("benchmark_target_abs_delta_e_current"))
    if live_delta is None:
        live_delta = _maybe_float(adapt_vqe.get("abs_delta_e"))
    history_tail = adapt_vqe.get("history_tail")
    if not isinstance(history_tail, Sequence):
        checkpoint_branch = beam.get("checkpoint_branch", {})
        if isinstance(checkpoint_branch, Mapping):
            history_tail = checkpoint_branch.get("history_tail")
    rows: list[Mapping[str, Any]] = [row for row in history_tail if isinstance(row, Mapping)] if isinstance(history_tail, Sequence) else []
    best_delta = live_delta
    best_iteration = live_iteration
    for idx, row in enumerate(rows):
        delta = _history_row_delta(row)
        if delta is None:
            continue
        iteration = _history_row_iteration(row, idx)
        if best_delta is None or float(delta) < float(best_delta):
            best_delta = float(delta)
            best_iteration = int(iteration)
    energy_violation = None
    if dominance_target_abs_delta_e is not None and best_delta is not None:
        energy_violation = float(max(0.0, float(best_delta) - float(dominance_target_abs_delta_e)))
    stat = current_json.stat()
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "paper_i_hh_optuna_live_current_best_v1",
        "source": "adapt_current_json_while_trial_running",
        "trial_number": int(trial_index),
        "trial_state": "running",
        "lane": str(lane),
        "epsilon_abs_delta_e": float(epsilon_abs_delta_e),
        "case_dir": str(case_dir),
        "output_dir": str(output_dir),
        "current_json": str(current_json),
        "result_json": str(case_dir / "json" / "result.json"),
        "current_json_mtime_ns": int(stat.st_mtime_ns),
        "live_iteration": live_iteration,
        "live_abs_delta_e": live_delta,
        "best_abs_delta_e_so_far": best_delta,
        "best_iteration_so_far": best_iteration,
        "adapt_iteration_count": adapt_vqe.get("ansatz_depth") or adapt_vqe.get("history_count"),
        "energy": adapt_vqe.get("energy"),
        "exact_gs_energy": adapt_vqe.get("exact_gs_energy"),
        "dominance_target_abs_delta_e": None if dominance_target_abs_delta_e is None else float(dominance_target_abs_delta_e),
        "dominance_target_iteration": None if dominance_target_iteration is None else int(dominance_target_iteration),
        "dominance_target_graph_count_2q": None if dominance_target_graph_count_2q is None else float(dominance_target_graph_count_2q),
        "dominance_target_graph_depth": None if dominance_target_graph_depth is None else float(dominance_target_graph_depth),
        "dominance_target_s_alg": None if dominance_target_s_alg is None else float(dominance_target_s_alg),
        "dominance_energy_violation": energy_violation,
        "selected_operator_count": len(adapt_vqe.get("operators", [])) if isinstance(adapt_vqe.get("operators"), Sequence) else None,
        "operator_labels_tail": list(adapt_vqe.get("operator_labels", [])[-8:]) if isinstance(adapt_vqe.get("operator_labels"), Sequence) else [],
            "params": _json_safe(dict(effective_params)),
    }


def _write_live_current_progress(
    *,
    current_json: Path,
    case_dir: Path,
    output_dir: Path,
    trial_index: int,
    lane: str,
    epsilon_abs_delta_e: float,
    effective_params: Mapping[str, Any],
    dominance_target_abs_delta_e: float | None,
    dominance_target_iteration: int | None,
    dominance_target_graph_count_2q: float | None,
    dominance_target_graph_depth: float | None,
    dominance_target_s_alg: float | None,
) -> int | None:
    payload = _live_current_progress_payload(
        current_json=current_json,
        case_dir=case_dir,
        output_dir=output_dir,
        trial_index=trial_index,
        lane=lane,
        epsilon_abs_delta_e=epsilon_abs_delta_e,
        effective_params=effective_params,
        dominance_target_abs_delta_e=dominance_target_abs_delta_e,
        dominance_target_iteration=dominance_target_iteration,
        dominance_target_graph_count_2q=dominance_target_graph_count_2q,
        dominance_target_graph_depth=dominance_target_graph_depth,
        dominance_target_s_alg=dominance_target_s_alg,
    )
    if payload is None:
        return None
    for path in (
        case_dir / "live_current_best.json",
        output_dir / "live_current_best.json",
        output_dir / "current_best.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    event_path = output_dir / "live_current_events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return int(payload["current_json_mtime_ns"])


def _evaluate_trial(
    *,
    python_bin: str,
    params: TrialParams,
    lane: str,
    epsilon_abs_delta_e: float,
    output_dir: Path,
    trial_index: int,
    compile_backend: str,
    compile_opt_level: int,
    compile_seed: int,
    warm_start: bool = False,
    hamiltonian_overrides: HhHamiltonianOverrides | None = None,
    compile_enabled: bool = True,
    phase2_w_shot_override: float | None = None,
    runtime_split_mode_override: str | None = None,
    child_pool_expansion_mode_override: str | None = None,
    child_pool_expansion_symmetry_policy_override: str | None = None,
    child_pool_expansion_max_subset_size_override: int | None = None,
    exact_gs_override: float | None = None,
    exact_gs_reference_json: Path | None = None,
    force_adapt_benchmark_target_abs_delta_e: float | None = None,
    force_adapt_max_depth: int | None = None,
    force_adapt_maxiter: int | None = None,
    force_adapt_final_refit_maxiter: int | None = None,
    force_adapt_drop_floor: float | None = None,
    force_adapt_drop_patience: int | None = None,
    force_adapt_drop_min_depth: int | None = None,
    force_adapt_full_refit_every: int | None = None,
    force_adapt_final_full_refit: str | None = None,
    force_adapt_allow_repeats: bool | None = None,
    force_phase0_pilot_max_records: int | None = None,
    force_phase1_shortlist_size: int | None = None,
    force_phase2_shortlist_fraction: float | None = None,
    force_phase2_shortlist_size: int | None = None,
    force_adapt_parallel_gradient_workers: int | None = None,
    force_adapt_beam_parent_workers: int | None = None,
    force_adapt_spsa_parallel_evaluations: int | None = None,
    force_adapt_pool_class_filter_json: Path | None = None,
    force_phase1_prune_prefilter_json: Path | None = None,
    force_adapt_resume_scaffold_json: Path | None = None,
    force_adapt_resume_mode: str | None = None,
    force_adapt_segment_id: str | None = None,
    force_adapt_segment_target_depth: int | None = None,
    force_adapt_segment_max_new_admissions: int | None = None,
    force_adapt_segment_wallclock_cap_s: float | None = None,
    force_adapt_resume_compile_smoke: str | None = None,
    force_adapt_resume_smoke_backend: str | None = None,
    force_static_route_id: str | None = None,
    force_static_meta_feature_profile: str | None = None,
    force_phase3_symmetry_mitigation_mode: str | None = None,
    force_route_a_paper_i_production: bool = False,
    force_phase1_prune_full_window: bool = False,
    force_phase1_prune_recovery_trust_radius: float | None = None,
    force_phase1_prune_schur_nomination_route: str | None = None,
    force_phase1_prune_metric_schur_mu: float | None = None,
    force_phase1_prune_metric_schur_solve_mode: str | None = None,
    force_phase1_prune_metric_schur_cost_weighting: str | None = None,
    force_skip_trajectory: bool | None = None,
    search_inner_optimizer: str = _SEARCH_INNER_OPTIMIZER,
    require_graph_cost: bool = True,
    require_zero_prune_rejections: bool = False,
    dominance_target_abs_delta_e: float | None = None,
    dominance_target_iteration: int | None = None,
    dominance_target_graph_count_2q: float | None = None,
    dominance_target_graph_depth: float | None = None,
    dominance_target_s_alg: float | None = None,
) -> TrialObservation:
    case_dir = _trial_case_dir(output_dir, lane, epsilon_abs_delta_e, trial_index, warm_start=warm_start)
    if case_dir.exists() and not bool(warm_start):
        try:
            shutil.rmtree(case_dir)
        except FileNotFoundError:
            # FAT/external volumes can report stale directory entries while
            # cleaning partial retry artifacts.  Missing children are safe to
            # ignore because the case directory is recreated immediately below.
            shutil.rmtree(case_dir, ignore_errors=True)
    (case_dir / "logs").mkdir(parents=True, exist_ok=True)
    (case_dir / "json").mkdir(parents=True, exist_ok=True)
    command, dropped_args, env_overrides, effective_params = _build_trial_command(
        python_bin=str(python_bin),
        params=params,
        case_dir=case_dir,
        hamiltonian_overrides=hamiltonian_overrides,
        phase2_w_shot_override=phase2_w_shot_override,
        runtime_split_mode_override=runtime_split_mode_override,
        child_pool_expansion_mode_override=child_pool_expansion_mode_override,
        child_pool_expansion_symmetry_policy_override=child_pool_expansion_symmetry_policy_override,
        child_pool_expansion_max_subset_size_override=child_pool_expansion_max_subset_size_override,
        exact_gs_override=exact_gs_override,
        exact_gs_reference_json=exact_gs_reference_json,
        force_adapt_benchmark_target_abs_delta_e=force_adapt_benchmark_target_abs_delta_e,
        force_adapt_max_depth=force_adapt_max_depth,
        force_adapt_maxiter=force_adapt_maxiter,
        force_adapt_final_refit_maxiter=force_adapt_final_refit_maxiter,
        force_adapt_drop_floor=force_adapt_drop_floor,
        force_adapt_drop_patience=force_adapt_drop_patience,
        force_adapt_drop_min_depth=force_adapt_drop_min_depth,
        force_adapt_full_refit_every=force_adapt_full_refit_every,
        force_adapt_final_full_refit=force_adapt_final_full_refit,
        force_adapt_allow_repeats=force_adapt_allow_repeats,
        force_phase0_pilot_max_records=force_phase0_pilot_max_records,
        force_phase1_shortlist_size=force_phase1_shortlist_size,
        force_phase2_shortlist_fraction=force_phase2_shortlist_fraction,
        force_phase2_shortlist_size=force_phase2_shortlist_size,
        force_adapt_parallel_gradient_workers=force_adapt_parallel_gradient_workers,
        force_adapt_beam_parent_workers=force_adapt_beam_parent_workers,
        force_adapt_spsa_parallel_evaluations=force_adapt_spsa_parallel_evaluations,
        force_adapt_pool_class_filter_json=force_adapt_pool_class_filter_json,
        force_phase1_prune_prefilter_json=force_phase1_prune_prefilter_json,
        force_adapt_resume_scaffold_json=force_adapt_resume_scaffold_json,
        force_adapt_resume_mode=force_adapt_resume_mode,
        force_adapt_segment_id=force_adapt_segment_id,
        force_adapt_segment_target_depth=force_adapt_segment_target_depth,
        force_adapt_segment_max_new_admissions=force_adapt_segment_max_new_admissions,
        force_adapt_segment_wallclock_cap_s=force_adapt_segment_wallclock_cap_s,
        force_adapt_resume_compile_smoke=force_adapt_resume_compile_smoke,
        force_adapt_resume_smoke_backend=force_adapt_resume_smoke_backend,
        force_static_route_id=force_static_route_id,
        force_static_meta_feature_profile=force_static_meta_feature_profile,
        force_phase3_symmetry_mitigation_mode=force_phase3_symmetry_mitigation_mode,
        force_route_a_paper_i_production=force_route_a_paper_i_production,
        force_phase1_prune_full_window=force_phase1_prune_full_window,
        force_phase1_prune_recovery_trust_radius=force_phase1_prune_recovery_trust_radius,
        force_phase1_prune_schur_nomination_route=force_phase1_prune_schur_nomination_route,
        force_phase1_prune_metric_schur_mu=force_phase1_prune_metric_schur_mu,
        force_phase1_prune_metric_schur_solve_mode=force_phase1_prune_metric_schur_solve_mode,
        force_phase1_prune_metric_schur_cost_weighting=force_phase1_prune_metric_schur_cost_weighting,
        force_skip_trajectory=force_skip_trajectory,
        search_inner_optimizer=search_inner_optimizer,
    )
    _write_command_log(case_dir / "logs" / "command.sh", command)
    if dropped_args:
        (case_dir / "logs" / "dropped_args.json").write_text(json.dumps(dropped_args, indent=2), encoding="utf-8")
    last_live_current_mtime_ns: int | None = None

    def _emit_live_progress_if_changed() -> None:
        nonlocal last_live_current_mtime_ns
        current_json = case_dir / "current.json"
        if not current_json.exists():
            return
        current_mtime_ns = int(current_json.stat().st_mtime_ns)
        if last_live_current_mtime_ns == current_mtime_ns:
            return
        written_mtime_ns = _write_live_current_progress(
            current_json=current_json,
            case_dir=case_dir,
            output_dir=output_dir,
            trial_index=trial_index,
            lane=lane,
            epsilon_abs_delta_e=epsilon_abs_delta_e,
            effective_params=effective_params,
            dominance_target_abs_delta_e=dominance_target_abs_delta_e,
            dominance_target_iteration=dominance_target_iteration,
            dominance_target_graph_count_2q=dominance_target_graph_count_2q,
            dominance_target_graph_depth=dominance_target_graph_depth,
            dominance_target_s_alg=dominance_target_s_alg,
        )
        if written_mtime_ns is not None:
            last_live_current_mtime_ns = int(written_mtime_ns)

    result_json = case_dir / "json" / "result.json"
    returncode, pipeline_elapsed_s = _run_subprocess_logged(
        command,
        cwd=REPO_ROOT,
        stdout_path=case_dir / "logs" / "stdout.log",
        stderr_path=case_dir / "logs" / "stderr.log",
        env_overrides=dict(env_overrides),
        live_progress_callback=_emit_live_progress_if_changed,
        live_progress_interval_s=15.0,
        completed_result_path=result_json,
    )

    compile_json = _compile_scout_output_path_for_artifact_dir(case_dir, compile_backend) if bool(compile_enabled) else None
    compile_command: list[str] | None = None
    compile_returncode: int | None = None
    if bool(compile_enabled) and result_json.exists() and compile_json is not None:
        compile_command = [
            str(python_bin), "-u", "-m", "pipelines.hardcoded.adapt_circuit_cost",
            "--artifact-json", str(result_json),
            "--backend-name", str(compile_backend),
            "--optimization-level", str(int(compile_opt_level)),
            "--seed-transpiler", str(int(compile_seed)),
            "--output-json", str(compile_json),
        ]
        _write_command_log(case_dir / "logs" / "compile_command.sh", compile_command)
        compile_returncode, compile_elapsed_s = _run_subprocess_logged(
            compile_command,
            cwd=REPO_ROOT,
            stdout_path=case_dir / "logs" / "compile_stdout.log",
            stderr_path=case_dir / "logs" / "compile_stderr.log",
        )
    else:
        compile_elapsed_s = None

    result_payload = _load_json(result_json) if result_json.exists() else {}
    compile_payload = _load_json(compile_json) if compile_json is not None and compile_json.exists() else {}
    abs_delta_e = _extract_abs_delta_e(result_payload)
    (
        adapt_iteration_count,
        dominance_first_crossing_iteration,
        dominance_prefix_abs_delta_e,
        dominance_prefix_iteration,
    ) = _extract_adapt_iteration_metrics(
        result_payload,
        target_abs_delta_e=dominance_target_abs_delta_e,
        target_iteration=dominance_target_iteration,
    )
    # The active Paper-I HH search objective is energy-gated cost dominance:
    # iteration is telemetry, while hardware/S_alg should be evaluated at the
    # best/reporting prefix when one is identifiable.  Do not default to the
    # first energy crossing here; that would optimize a different first-crossing
    # objective and can falsely penalize runs whose Paper-I shot proxy is only
    # reconstructible at the reporting/terminal prefix.
    shot_prefix_iteration = dominance_prefix_iteration or dominance_first_crossing_iteration
    shot_history_position = _adapt_history_position_for_prefix_iteration(result_payload, shot_prefix_iteration)
    if shot_prefix_iteration is not None and (
        adapt_iteration_count is None or int(shot_prefix_iteration) < int(adapt_iteration_count)
    ):
        shot_scope = "display_prefix"
    else:
        # If the selected Paper-I prefix is the terminal accepted prefix, the
        # terminal deterministic S_alg is the same reporting scope and preserves
        # final-refit accounting without requiring prefix reconstruction.
        shot_scope = "terminal"
        shot_history_position = None
    graph_metrics = _extract_graph_cost_metrics(
        result_payload,
        paper_i_shot_scope=shot_scope,
        paper_i_shot_history_position=shot_history_position,
    )
    prune_metrics = prune_telemetry_counts(result_payload)
    compiled_count_2q, compiled_depth, logical_operator_count, runtime_parameter_count = _extract_compile_metrics(compile_payload)
    if compiled_count_2q is None and graph_metrics.get("graph_count_2q") is not None:
        compiled_count_2q = int(round(float(graph_metrics["graph_count_2q"])))
    if compiled_depth is None and graph_metrics.get("graph_depth") is not None:
        compiled_depth = int(round(float(graph_metrics["graph_depth"])))
    if logical_operator_count is None and graph_metrics.get("graph_theta_count") is not None:
        logical_operator_count = int(round(float(graph_metrics["graph_theta_count"])))
    if runtime_parameter_count is None and graph_metrics.get("graph_theta_count") is not None:
        runtime_parameter_count = int(round(float(graph_metrics["graph_theta_count"])))
    family_signature, op_signature = _extract_history_signature(result_payload)
    constraints = _compute_constraints(
        abs_delta_e=abs_delta_e,
        epsilon_abs_delta_e=float(epsilon_abs_delta_e),
        returncode=returncode,
        compile_returncode=compile_returncode,
        compiled_count_2q=compiled_count_2q,
        dropped_search_flags=dropped_args,
        require_compile=bool(compile_enabled),
        prune_rejected_delete_attempt_count=prune_metrics.get("prune_rejected_delete_attempt_count"),
        prune_no_accept_restore_pass_count=prune_metrics.get("prune_no_accept_restore_pass_count"),
        prune_accepted_then_guard_rolled_back_count=prune_metrics.get("prune_accepted_then_guard_rolled_back_count"),
        prune_actual_rollback_count=prune_metrics.get("prune_actual_rollback_count"),
        require_zero_prune_rejections=bool(require_zero_prune_rejections),
    )
    if (not bool(compile_enabled)) and bool(require_graph_cost) and compiled_count_2q is None:
        constraints[1] = float(constraints[1] + 1.0)
    dominance_energy_violation = None
    dominance_iteration_violation = None
    dominance_graph_count_2q_violation = None
    dominance_graph_depth_violation = None
    dominance_s_alg_violation = None
    if dominance_target_abs_delta_e is not None:
        dominance_energy_value = (
            dominance_prefix_abs_delta_e
            if dominance_target_iteration is not None
            else abs_delta_e
        )
        if dominance_energy_value is None:
            dominance_energy_violation = 1.0
        else:
            dominance_energy_violation = float(max(0.0, float(dominance_energy_value) - float(dominance_target_abs_delta_e)))
        constraints[0] = float(dominance_energy_violation)
    if dominance_target_iteration is not None:
        allowed_k = max(0, int(dominance_target_iteration) - 1)
        if dominance_first_crossing_iteration is None:
            dominance_iteration_violation = 1.0
        else:
            dominance_iteration_violation = float(max(0, int(dominance_first_crossing_iteration) - int(allowed_k)))
        constraints.append(float(dominance_iteration_violation))
    if dominance_target_graph_count_2q is not None:
        if graph_metrics.get("graph_count_2q") is None:
            dominance_graph_count_2q_violation = 1.0
        else:
            dominance_graph_count_2q_violation = float(max(0.0, float(graph_metrics["graph_count_2q"]) - float(dominance_target_graph_count_2q)))
        constraints.append(float(dominance_graph_count_2q_violation))
    if dominance_target_graph_depth is not None:
        if graph_metrics.get("graph_depth") is None:
            dominance_graph_depth_violation = 1.0
        else:
            dominance_graph_depth_violation = float(max(0.0, float(graph_metrics["graph_depth"]) - float(dominance_target_graph_depth)))
        constraints.append(float(dominance_graph_depth_violation))
    if dominance_target_s_alg is not None:
        if graph_metrics.get("paper_i_table_s_alg") is None:
            dominance_s_alg_violation = 1.0
        else:
            dominance_s_alg_violation = float(max(0.0, float(graph_metrics["paper_i_table_s_alg"]) - float(dominance_target_s_alg)))
        constraints.append(float(dominance_s_alg_violation))
    feasible = bool(all(float(x) <= 0.0 for x in constraints))
    invalid_reasons: list[str] = []
    if abs_delta_e is None:
        invalid_reasons.append("missing_abs_delta_e")
    elif float(abs_delta_e) > float(epsilon_abs_delta_e):
        invalid_reasons.append("energy_band_failed")
    if int(returncode) != 0:
        invalid_reasons.append("pipeline_failed")
    if bool(compile_enabled) and (int(compile_returncode or 1) != 0 or compiled_count_2q is None):
        invalid_reasons.append("compile_failed")
    if (not bool(compile_enabled)) and bool(require_graph_cost) and compiled_count_2q is None:
        invalid_reasons.append("graph_cost_missing")
    if dominance_energy_violation is not None and float(dominance_energy_violation) > 0.0:
        invalid_reasons.append("geo_energy_dominance_failed")
    if dominance_iteration_violation is not None and float(dominance_iteration_violation) > 0.0:
        invalid_reasons.append("geo_iteration_dominance_failed")
    if dominance_graph_count_2q_violation is not None and float(dominance_graph_count_2q_violation) > 0.0:
        invalid_reasons.append("geo_graph_count_2q_dominance_failed")
    if dominance_graph_depth_violation is not None and float(dominance_graph_depth_violation) > 0.0:
        invalid_reasons.append("geo_graph_depth_dominance_failed")
    if dominance_s_alg_violation is not None and float(dominance_s_alg_violation) > 0.0:
        invalid_reasons.append("geo_s_alg_dominance_failed")
    dropped_control_flags = [str(flag) for flag in dropped_args if str(flag) in _SEARCH_CONTROL_FLAGS]
    if dropped_control_flags:
        invalid_reasons.append("unsupported_search_flags_dropped")
    prune_zero_violation = int(prune_metrics.get("prune_actual_rollback_count", 0) or 0)
    if bool(require_zero_prune_rejections) and prune_zero_violation > 0:
        invalid_reasons.append("prune_actual_rollback_nonzero")
    return TrialObservation(
        lane=str(lane),
        epsilon_abs_delta_e=float(epsilon_abs_delta_e),
        params=dict(effective_params),
        objective_lexicographic=_objective_lexicographic(compiled_count_2q, compiled_depth, logical_operator_count, runtime_parameter_count),
        abs_delta_e=abs_delta_e,
        compiled_count_2q=compiled_count_2q,
        compiled_depth=compiled_depth,
        logical_operator_count=logical_operator_count,
        runtime_parameter_count=runtime_parameter_count,
        feasible=bool(feasible),
        constraints=list(constraints),
        graph_count_2q=graph_metrics.get("graph_count_2q"),
        graph_depth=graph_metrics.get("graph_depth"),
        graph_count_1q=graph_metrics.get("graph_count_1q"),
        graph_theta_count=graph_metrics.get("graph_theta_count"),
        measurement_work_shots=graph_metrics.get("measurement_work_shots"),
        paper_i_table_shots_total=graph_metrics.get("paper_i_table_shots_total"),
        paper_i_table_s_alg=graph_metrics.get("paper_i_table_s_alg"),
        paper_i_table_shots_status=graph_metrics.get("paper_i_table_shots_status"),
        paper_i_table_shots_scope=graph_metrics.get("paper_i_table_shots_scope"),
        paper_i_table_shots_history_position=graph_metrics.get("paper_i_table_shots_history_position"),
        measurement_work_records=graph_metrics.get("measurement_work_records"),
        resource_cost_source=(
            graph_metrics.get("resource_cost_source")
            if graph_metrics.get("resource_cost_source") is not None
            else ("compile_scout" if bool(compile_enabled) else None)
        ),
        prune_candidate_count=prune_metrics.get("prune_candidate_count"),
        prune_accepted_count=prune_metrics.get("prune_accepted_count"),
        prune_rejected_delete_attempt_count=prune_metrics.get("prune_rejected_delete_attempt_count"),
        prune_no_accept_restore_pass_count=prune_metrics.get("prune_no_accept_restore_pass_count"),
        prune_accepted_then_guard_rolled_back_count=prune_metrics.get("prune_accepted_then_guard_rolled_back_count"),
        prune_actual_rollback_count=prune_metrics.get("prune_actual_rollback_count"),
        prune_prefilter_blocked_count=prune_metrics.get("prune_prefilter_blocked_count"),
        prune_prefilter_allowed_count=prune_metrics.get("prune_prefilter_allowed_count"),
        adapt_iteration_count=adapt_iteration_count,
        dominance_target_abs_delta_e=(None if dominance_target_abs_delta_e is None else float(dominance_target_abs_delta_e)),
        dominance_target_iteration=(None if dominance_target_iteration is None else int(dominance_target_iteration)),
        dominance_target_graph_count_2q=(None if dominance_target_graph_count_2q is None else float(dominance_target_graph_count_2q)),
        dominance_target_graph_depth=(None if dominance_target_graph_depth is None else float(dominance_target_graph_depth)),
        dominance_target_s_alg=(None if dominance_target_s_alg is None else float(dominance_target_s_alg)),
        dominance_prefix_abs_delta_e=dominance_prefix_abs_delta_e,
        dominance_prefix_iteration=dominance_prefix_iteration,
        dominance_first_crossing_iteration=dominance_first_crossing_iteration,
        dominance_energy_violation=dominance_energy_violation,
        dominance_iteration_violation=dominance_iteration_violation,
        dominance_graph_count_2q_violation=dominance_graph_count_2q_violation,
        dominance_graph_depth_violation=dominance_graph_depth_violation,
        dominance_s_alg_violation=dominance_s_alg_violation,
        case_dir=str(case_dir),
        result_json=(str(result_json) if result_json.exists() else None),
        compile_json=(str(compile_json) if compile_json is not None and compile_json.exists() else None),
        returncode=int(returncode),
        compile_returncode=(None if compile_returncode is None else int(compile_returncode)),
        pipeline_elapsed_s=float(pipeline_elapsed_s),
        compile_elapsed_s=(None if compile_elapsed_s is None else float(compile_elapsed_s)),
        total_elapsed_s=float(pipeline_elapsed_s + float(compile_elapsed_s or 0.0)),
        dropped_args=list(dropped_args),
        base_preset=str(params.base_preset),
        family_path_signature=list(family_signature),
        selected_op_signature=list(op_signature),
        source_artifact_dir=_base_preset_library()[str(params.base_preset)].source_artifact_dir,
        warm_start=bool(warm_start),
        hamiltonian_overrides=_hh_hamiltonian_override_payload(hamiltonian_overrides),
        invalid_reasons=list(invalid_reasons),
    )


def _staged_graph_speed_enqueue_params(
    lane: str,
    restricted_base_preset_names: Sequence[str] = (),
    *,
    force_spsa_profile: str | None = None,
) -> list[dict[str, Any]]:
    base_preset = _searchable_presets_for_lane(str(lane), restricted_base_preset_names)[0]
    common = {
        "base_preset": str(base_preset),
        "adapt_max_depth": 16,
        "selector_geometry_mode": "reduced",
        "runtime_split_mode": "shortlist_pauli_children_v1",
        "batching_mode": "on",
        "repeats_mode": "base",
        "selection_cost_mode": _MARRAKESH_GRAPH_SPAN_MODE,
        "motif_mode": "off",
        "phase1_prune_mode": "live",
        "phase0_pilot_profile": "base",
        "phase_live_profile": "base",
        "prune_witness_profile": "base",
    }
    staged = [
        ("heavy_full", "heavy_x8_phase23", "full_heavy", "marrakesh_balanced", "current", "ml_median", "rho_0p25"),
        ("cap_ramp_medium", "late_x4_phase23", "medium_16", "marrakesh_balanced", "paper_i_strong_like", "ml_p90", "rho_0p5"),
        ("cap_ramp_narrow", "late_x2", "tight_8", "marrakesh_2q_heavy", "paper_i_weak_strong_like", "ml_p10", "rho_0p75"),
        ("cap_ramp_tight", "cheap_x1", "cheap_4", "marrakesh_depth_heavy", "current", "ml_median", "rho_0p25"),
    ]
    out: list[dict[str, Any]] = []
    for shortlist, shots, window, weights, spsa_profile, ml_candidate_profile, phase2_rho_profile in staged:
        effective_spsa_profile = str(force_spsa_profile or spsa_profile)
        if effective_spsa_profile not in _SPSA_PROFILE_OPTIONS:
            raise ValueError(f"Unknown force_spsa_profile: {effective_spsa_profile!r}")
        row = dict(common)
        row.update(
            {
                "maturity_shortlist_profile": str(shortlist),
                "maturity_shot_profile": str(shots),
                "adapt_window_profile": str(window),
                "backend_cost_weight_profile": str(weights),
                "spsa_profile": effective_spsa_profile,
                "ml_candidate_profile": str(ml_candidate_profile),
                "phase2_rho_profile": str(phase2_rho_profile),
                "phase2_w_shot_profile": {
                    "heavy_full": "shot_0p15",
                    "cap_ramp_medium": "shot_0p08",
                    "cap_ramp_narrow": "shot_0p04",
                    "cap_ramp_tight": "shot_0p02",
                }[str(shortlist)],
            }
        )
        out.append(row)
    return out


def _staged_shot_speed_enqueue_params(
    lane: str,
    restricted_base_preset_names: Sequence[str] = (),
    *,
    force_spsa_profile: str | None = None,
) -> list[dict[str, Any]]:
    base_preset = _searchable_presets_for_lane(str(lane), restricted_base_preset_names)[0]
    common = {
        "base_preset": str(base_preset),
        "adapt_max_depth": 16,
        "selector_geometry_mode": "reduced",
        "runtime_split_mode": "shortlist_pauli_children_v1",
        "batching_mode": "on",
        "repeats_mode": "base",
        "selection_cost_mode": _MARRAKESH_GRAPH_SPAN_MODE,
        "motif_mode": "off",
        "phase1_prune_mode": "live",
        "phase0_pilot_profile": "base",
        "phase_live_profile": "phase3_early_null",
        "prune_witness_profile": "base",
    }
    staged = [
        ("pilot_10", "cap_ramp_micro", "cheap_x1", "cheap_4", "marrakesh_2q_heavy", "current", "ml_p10", "rho_0p25", "shot_0p75"),
        ("pilot_16", "cap_ramp_tight", "cheap_x1", "cheap_4", "marrakesh_balanced", "current", "ml_p10", "rho_0p25", "shot_0p75"),
        ("pilot_24", "cap_ramp_narrow", "cheap_x1", "tight_8", "marrakesh_2q_heavy", "paper_i_weak_strong_like", "ml_p10", "rho_0p5", "shot_0p75"),
        ("pilot_32", "cap_ramp_medium", "late_x2", "medium_16", "marrakesh_balanced", "paper_i_strong_like", "ml_median", "rho_0p75", "shot_0p50"),
        ("base", "cap_ramp_tight", "late_x2", "cheap_4", "marrakesh_2q_heavy", "current", "ml_p90", "rho_0p5", "shot_0p30"),
    ]
    out: list[dict[str, Any]] = []
    for pilot, shortlist, shots, window, weights, spsa_profile, ml_candidate_profile, phase2_rho_profile, phase2_w_shot_profile in staged:
        effective_spsa_profile = str(force_spsa_profile or spsa_profile)
        if effective_spsa_profile not in _SPSA_PROFILE_OPTIONS:
            raise ValueError(f"Unknown force_spsa_profile: {effective_spsa_profile!r}")
        row = dict(common)
        row.update(
            {
                "phase0_pilot_profile": str(pilot),
                "maturity_shortlist_profile": str(shortlist),
                "maturity_shot_profile": str(shots),
                "adapt_window_profile": str(window),
                "backend_cost_weight_profile": str(weights),
                "spsa_profile": effective_spsa_profile,
                "ml_candidate_profile": str(ml_candidate_profile),
                "phase2_rho_profile": str(phase2_rho_profile),
                "phase2_w_shot_profile": str(phase2_w_shot_profile),
            }
        )
        out.append(row)
    return out


def _warm_start_params_from_preset_name(name: str) -> TrialParams:
    presets = _base_preset_library()
    if name not in presets:
        raise KeyError(name)
    preset = presets[name]
    is_legacy = _preset_is_legacy(preset)
    args = list(preset.pipeline_args)
    max_depth = 20
    for idx, tok in enumerate(args):
        if tok == "--adapt-max-depth" and idx + 1 < len(args):
            max_depth = int(args[idx + 1])
            break
    selector_mode = "base"
    if "--phase3-selector-geometry-mode" in args:
        pos = args.index("--phase3-selector-geometry-mode")
        if pos + 1 < len(args):
            selector_mode = str(args[pos + 1])
    runtime_split_mode = "base"
    if "--phase3-runtime-split-mode" in args:
        pos = args.index("--phase3-runtime-split-mode")
        if pos + 1 < len(args):
            runtime_split_mode = str(args[pos + 1])
    if not is_legacy:
        runtime_split_mode = "off"
    batching_mode = "base"
    if "--phase2-no-batching" in args:
        batching_mode = "off"
    elif "--phase2-enable-batching" in args:
        batching_mode = "on"
    repeats_mode = "disable" if "--adapt-no-repeats" in args else "base"
    selection_cost_mode = "base"
    if "--phase3-backend-cost-mode" in args:
        pos = args.index("--phase3-backend-cost-mode")
        if pos + 1 < len(args):
            selection_cost_mode = str(args[pos + 1])
    motif_mode = "legacy81" if "--phase3-motif-source-json" in args else "base"
    phase1_prune_mode = "base"
    if "--phase1-no-prune" in args:
        phase1_prune_mode = "off"
    elif "--phase1-prune-mode" in args:
        pos = args.index("--phase1-prune-mode")
        if pos + 1 < len(args):
            phase1_prune_mode = str(args[pos + 1])
        else:
            phase1_prune_mode = "live"
    elif "--phase1-prune-enabled" in args:
        phase1_prune_mode = "live"
    phase0_pilot_profile = "base"
    if "--phase0-pilot-max-records" in args:
        pos = args.index("--phase0-pilot-max-records")
        if pos + 1 < len(args):
            raw_cap = str(args[pos + 1])
            phase0_pilot_profile = {
                "64": "pilot_64",
                "32": "pilot_32",
                "24": "pilot_24",
                "16": "pilot_16",
                "10": "pilot_10",
            }.get(raw_cap, "base")
    return TrialParams(
        base_preset=str(name),
        adapt_max_depth=int(max_depth),
        selector_geometry_mode=str(selector_mode),
        runtime_split_mode=str(runtime_split_mode),
        batching_mode=str(batching_mode),
        repeats_mode=str(repeats_mode),
        selection_cost_mode=str(selection_cost_mode),
        motif_mode=str(motif_mode),
        phase1_prune_mode=str(phase1_prune_mode),
        phase0_pilot_profile=str(phase0_pilot_profile),
    )


def _artifact_dir_for_preset(name: str) -> Path | None:
    source = _base_preset_library().get(str(name), BasePreset("", (), (), ())).source_artifact_dir
    return None if source in {None, ""} else Path(str(source))


def _compile_backend_slug_candidates(compile_backend: str) -> tuple[str, ...]:
    raw = str(compile_backend).strip()
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", re.sub(r"[^A-Za-z0-9]+", "_", raw)).lower().strip("_")
    compact = _safe_slug(raw).lower()
    out: list[str] = []
    for value in (snake, compact):
        if value and value not in out:
            out.append(value)
    return tuple(out)


def _compile_scout_output_path_for_artifact_dir(artifact_dir: Path, compile_backend: str) -> Path:
    primary = _compile_backend_slug_candidates(compile_backend)[0]
    return artifact_dir / "json" / f"compile_scout_{primary}.json"


def _compile_scout_path_for_artifact_dir(artifact_dir: Path, compile_backend: str) -> Path | None:
    for slug in _compile_backend_slug_candidates(compile_backend):
        candidate = artifact_dir / "json" / f"compile_scout_{slug}.json"
        if candidate.exists():
            return candidate
    return None


def _load_observation_from_artifact_dir(
    *,
    lane: str,
    epsilon_abs_delta_e: float,
    preset_name: str,
    artifact_dir: Path,
    compile_backend: str,
) -> TrialObservation | None:
    result_json = artifact_dir / "json" / "result.json"
    compile_json = _compile_scout_path_for_artifact_dir(artifact_dir, compile_backend)
    if not result_json.exists() or compile_json is None or not compile_json.exists():
        return None
    result_payload = _load_json(result_json)
    compile_payload = _load_json(compile_json)
    abs_delta_e = _extract_abs_delta_e(result_payload)
    prune_metrics = prune_telemetry_counts(result_payload)
    compiled_count_2q, compiled_depth, logical_operator_count, runtime_parameter_count = _extract_compile_metrics(compile_payload)
    family_signature, op_signature = _extract_history_signature(result_payload)
    constraints = _compute_constraints(
        abs_delta_e=abs_delta_e,
        epsilon_abs_delta_e=float(epsilon_abs_delta_e),
        returncode=0,
        compile_returncode=0,
        compiled_count_2q=compiled_count_2q,
    )
    feasible = bool(all(float(x) <= 0.0 for x in constraints))
    return TrialObservation(
        lane=str(lane),
        epsilon_abs_delta_e=float(epsilon_abs_delta_e),
        params=dict(asdict(_warm_start_params_from_preset_name(preset_name))),
        objective_lexicographic=_objective_lexicographic(compiled_count_2q, compiled_depth, logical_operator_count, runtime_parameter_count),
        abs_delta_e=abs_delta_e,
        compiled_count_2q=compiled_count_2q,
        compiled_depth=compiled_depth,
        logical_operator_count=logical_operator_count,
        runtime_parameter_count=runtime_parameter_count,
        feasible=bool(feasible),
        constraints=list(constraints),
        prune_candidate_count=prune_metrics.get("prune_candidate_count"),
        prune_accepted_count=prune_metrics.get("prune_accepted_count"),
        prune_rejected_delete_attempt_count=prune_metrics.get("prune_rejected_delete_attempt_count"),
        prune_no_accept_restore_pass_count=prune_metrics.get("prune_no_accept_restore_pass_count"),
        prune_accepted_then_guard_rolled_back_count=prune_metrics.get("prune_accepted_then_guard_rolled_back_count"),
        prune_actual_rollback_count=prune_metrics.get("prune_actual_rollback_count"),
        prune_prefilter_blocked_count=prune_metrics.get("prune_prefilter_blocked_count"),
        prune_prefilter_allowed_count=prune_metrics.get("prune_prefilter_allowed_count"),
        case_dir=str(artifact_dir),
        result_json=str(result_json),
        compile_json=str(compile_json),
        returncode=0,
        compile_returncode=0,
        base_preset=str(preset_name),
        family_path_signature=list(family_signature),
        selected_op_signature=list(op_signature),
        source_artifact_dir=str(artifact_dir),
        warm_start=True,
        invalid_reasons=([] if feasible else ["warm_start_not_feasible"]),
    )


def _default_warm_start_observations(
    lane: str,
    epsilon_abs_delta_e: float,
    compile_backend: str,
    extra_preset_names: Sequence[str] = (),
    restricted_base_preset_names: Sequence[str] = (),
) -> list[TrialObservation]:
    searchable = set(_searchable_presets_for_lane(lane, restricted_base_preset_names))
    preset_names = list(
        dict.fromkeys(
            [
                *{
                    "canonical": ["public_anchor"],
                    "global": ["bridge_focus_98", "bridge_focus_118", "raw_exact_compile_only", "fullhorse_motif", "bridge_diag", "public_anchor"],
                    "legacy": ["legacy_focus_75", "legacy_20260322"],
                }.get(str(lane), []),
                *(str(x) for x in extra_preset_names if str(x) in searchable),
            ]
        )
    )
    out: list[TrialObservation] = []
    for name in preset_names:
        artifact_dir = _artifact_dir_for_preset(name)
        if artifact_dir is None:
            continue
        obs = _load_observation_from_artifact_dir(
            lane=str(lane),
            epsilon_abs_delta_e=float(epsilon_abs_delta_e),
            preset_name=str(name),
            artifact_dir=artifact_dir,
            compile_backend=str(compile_backend),
        )
        if obs is not None:
            out.append(obs)
    return out


def _default_reference_observations(
    lane: str,
    epsilon_abs_delta_e: float,
    compile_backend: str,
    extra_preset_names: Sequence[str] = (),
    restricted_base_preset_names: Sequence[str] = (),
) -> list[TrialObservation]:
    searchable = set(_searchable_presets_for_lane(lane, restricted_base_preset_names))
    preset_names = list(
        dict.fromkeys(
            [
                *({"global": ["legacy_focus_75", "legacy_20260322"], "canonical": ["legacy_focus_75"]}.get(str(lane), [])),
                *(str(x) for x in extra_preset_names if str(x) not in searchable),
            ]
        )
    )
    out: list[TrialObservation] = []
    for name in preset_names:
        artifact_dir = _artifact_dir_for_preset(name)
        if artifact_dir is None:
            continue
        obs = _load_observation_from_artifact_dir(
            lane=str(lane),
            epsilon_abs_delta_e=float(epsilon_abs_delta_e),
            preset_name=str(name),
            artifact_dir=artifact_dir,
            compile_backend=str(compile_backend),
        )
        if obs is not None:
            out.append(obs)
    return out


def _best_feasible_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    feasible_rows = [dict(row) for row in rows if row.get("best_compiled_count_2q") is not None]
    if not feasible_rows:
        return None
    return min(
        feasible_rows,
        key=lambda row: (
            int(row.get("best_compiled_count_2q", 10**9)),
            int(row.get("best_compiled_depth", 10**9)),
            int(row.get("logical_operator_count", 10**9)),
            int(row.get("runtime_parameter_count", 10**9)),
            float(row.get("delta_e_abs", float("inf"))),
        ),
    )


def _observation_to_user_attrs(obs: TrialObservation) -> dict[str, Any]:
    payload = asdict(obs)
    payload["constraints"] = list(obs.constraints)
    return payload


def _objective_value_for_mode(obs: TrialObservation, objective_mode: str) -> float:
    mode = str(objective_mode)
    if mode == "energy":
        return float(obs.abs_delta_e) if obs.abs_delta_e is not None else float(_LARGE_OBJECTIVE)
    if mode == "cost_feasible":
        return float(obs.objective_lexicographic)
    if mode == "graph_cost_speed_feasible":
        return _graph_speed_objective(obs)
    if mode == "geo_dominance_first":
        return _geo_dominance_first_objective(obs)
    if mode in {"geo_energy_then_graph_cost", "geo_energy_then_graph_shot_cost"}:
        return _geo_energy_then_graph_cost_objective(obs)
    if mode == "geo_energy_then_shot_graph_cost":
        return _geo_energy_then_shot_graph_cost_objective(obs)
    if mode == "shot_then_energy_graph_cost":
        return _shot_then_energy_graph_cost_objective(obs)
    if mode == "geo_energy_gate_then_shot_energy_graph_cost":
        return _geo_energy_gate_then_shot_energy_graph_cost_objective(obs)
    if mode == "prune_zero_then_energy_shot_graph_cost":
        return _prune_zero_then_energy_shot_graph_cost_objective(obs)
    raise ValueError(f"Unknown objective mode: {objective_mode!r}")


def _write_progress(output_dir: Path, payload: Mapping[str, Any]) -> None:
    progress_path = Path(output_dir) / "progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _trial_event_payload(
    *,
    study_name: str,
    trial_number: int,
    objective_mode: str,
    objective_value: float,
    obs: TrialObservation,
) -> dict[str, Any]:
    payload = _observation_to_user_attrs(obs)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "paper_i_hh_optuna_trial_event_v1",
        "study_name": str(study_name),
        "trial_number": int(trial_number),
        "objective_mode": str(objective_mode),
        "objective_value": float(objective_value),
        "abs_delta_e": obs.abs_delta_e,
        "adapt_iteration_count": obs.adapt_iteration_count,
        "dominance_target_abs_delta_e": obs.dominance_target_abs_delta_e,
        "dominance_target_graph_count_2q": obs.dominance_target_graph_count_2q,
        "dominance_target_graph_depth": obs.dominance_target_graph_depth,
        "dominance_target_s_alg": obs.dominance_target_s_alg,
        "dominance_energy_violation": obs.dominance_energy_violation,
        "dominance_graph_count_2q_violation": obs.dominance_graph_count_2q_violation,
        "dominance_graph_depth_violation": obs.dominance_graph_depth_violation,
        "dominance_s_alg_violation": obs.dominance_s_alg_violation,
        "feasible": bool(obs.feasible),
        "graph_count_2q": obs.graph_count_2q,
        "graph_depth": obs.graph_depth,
        "graph_count_1q": obs.graph_count_1q,
        "graph_theta_count": obs.graph_theta_count,
        "measurement_work_shots": obs.measurement_work_shots,
        "paper_i_table_shots_total": obs.paper_i_table_shots_total,
        "paper_i_table_s_alg": obs.paper_i_table_s_alg,
        "paper_i_table_shots_status": obs.paper_i_table_shots_status,
        "measurement_work_records": obs.measurement_work_records,
        "graph_hardware_objective_scalar": _graph_hardware_objective_scalar(obs),
        "paper_i_shot_cost_scalar": _paper_i_shot_cost_scalar(obs),
        "graph_plus_shot_objective_scalar": _graph_speed_objective(obs),
        "result_json": obs.result_json,
        "case_dir": obs.case_dir,
        "invalid_reasons": list(obs.invalid_reasons),
        "observation": payload,
    }


def _append_trial_event(output_dir: Path, payload: Mapping[str, Any]) -> None:
    path = Path(output_dir) / "trial_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _graph_cost_scalar(obs: TrialObservation) -> float:
    graph_count_2q = _maybe_float(obs.graph_count_2q)
    graph_depth = _maybe_float(obs.graph_depth)
    if graph_count_2q is not None and graph_depth is not None:
        graph_count_1q = _maybe_float(obs.graph_count_1q) or 0.0
        graph_theta_count = _maybe_float(obs.graph_theta_count) or 0.0
        return float(graph_count_2q * 10**12 + graph_depth * 10**6 + graph_count_1q * 10**3 + graph_theta_count)
    return float(_LARGE_OBJECTIVE)


def _prefix_delta_from_history_row(row: Mapping[str, Any]) -> float | None:
    for key in (
        "delta_abs_current",
        "exact_abs_delta_e_from_final_state",
        "abs_delta_e",
        "energy_error_abs",
    ):
        value = _maybe_float(row.get(key))
        if value is not None:
            return float(value)
    return None


def _prefix_iteration_from_history_row(row: Mapping[str, Any], fallback_index: int) -> int:
    for key in ("depth_cumulative", "adapt_iteration_count", "adapt_depth", "depth"):
        value = _maybe_float(row.get(key))
        if value is not None:
            return int(round(float(value)))
    return int(fallback_index + 1)


def _load_trial_prefix_rows(obs: TrialObservation, *, objective_mode: str) -> list[dict[str, Any]]:
    result_json = None if obs.result_json in {None, ""} else Path(str(obs.result_json))
    if result_json is None or not result_json.exists():
        return []
    try:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    history = adapt_vqe.get("history", []) if isinstance(adapt_vqe, Mapping) else []
    if not isinstance(history, Sequence):
        return []
    trial_number = _trial_number_from_observation(obs)
    rows: list[dict[str, Any]] = []
    best_delta: float | None = None
    best_iteration: int | None = None
    for idx, history_row in enumerate(history):
        if not isinstance(history_row, Mapping):
            continue
        iteration = _prefix_iteration_from_history_row(history_row, idx)
        delta = _prefix_delta_from_history_row(history_row)
        if delta is None:
            continue
        if best_delta is None or float(delta) < float(best_delta):
            best_delta = float(delta)
            best_iteration = int(iteration)
        rows.append(
            {
                "trial_number": trial_number,
                "case_dir": obs.case_dir,
                "result_json": obs.result_json,
                "source": "adapt_vqe.history",
                "iteration": int(iteration),
                "abs_delta_e": float(delta),
                "best_abs_delta_e_so_far": None if best_delta is None else float(best_delta),
                "best_iteration_so_far": best_iteration,
                "objective_value": _objective_value_for_mode(obs, objective_mode),
                "graph_cost_scalar": _graph_cost_scalar(obs),
                "paper_i_table_shots_total": obs.paper_i_table_shots_total,
                "paper_i_table_shots_status": obs.paper_i_table_shots_status,
            }
        )
    terminal_delta = _maybe_float(obs.abs_delta_e)
    terminal_iteration = obs.adapt_iteration_count
    if terminal_delta is not None and terminal_iteration is not None:
        try:
            iteration = int(terminal_iteration)
            delta = float(terminal_delta)
            if best_delta is None or delta < float(best_delta):
                best_delta = delta
                best_iteration = iteration
            rows.append(
                {
                    "trial_number": trial_number,
                    "case_dir": obs.case_dir,
                    "result_json": obs.result_json,
                    "source": "trial_observation_terminal",
                    "iteration": iteration,
                    "abs_delta_e": delta,
                    "best_abs_delta_e_so_far": float(best_delta),
                    "best_iteration_so_far": best_iteration,
                    "objective_value": _objective_value_for_mode(obs, objective_mode),
                    "graph_cost_scalar": _graph_cost_scalar(obs),
                    "paper_i_table_shots_total": obs.paper_i_table_shots_total,
                    "paper_i_table_shots_status": obs.paper_i_table_shots_status,
                }
            )
        except Exception:
            pass
    return rows


def _trial_number_from_observation(obs: TrialObservation) -> int | None:
    params = obs.params if isinstance(obs.params, Mapping) else {}
    raw_trial_number = params.get("trial_number") or params.get("number")
    if raw_trial_number is not None:
        try:
            return int(raw_trial_number)
        except Exception:
            pass
    for raw_path in (obs.case_dir, obs.result_json):
        if raw_path in {None, ""}:
            continue
        match = re.search(r"trial_(\d+)", str(raw_path))
        if match is not None:
            try:
                return int(match.group(1))
            except Exception:
                pass
    return None


def _prefix_best_payload(observations: Sequence[TrialObservation], *, objective_mode: str) -> dict[str, Any]:
    per_trial: list[dict[str, Any]] = []
    global_best_by_iteration: dict[int, dict[str, Any]] = {}
    global_incumbent: dict[str, Any] | None = None
    for obs in observations:
        prefix_rows = _load_trial_prefix_rows(obs, objective_mode=objective_mode)
        if not prefix_rows:
            continue
        trial_best = min(prefix_rows, key=lambda row: float(row["abs_delta_e"]))
        per_trial.append(dict(trial_best))
        for prefix_row in prefix_rows:
            iteration = int(prefix_row["iteration"])
            previous = global_best_by_iteration.get(iteration)
            if previous is None or float(prefix_row["abs_delta_e"]) < float(previous["abs_delta_e"]):
                global_best_by_iteration[iteration] = dict(prefix_row)
            if global_incumbent is None or float(prefix_row["abs_delta_e"]) < float(global_incumbent["abs_delta_e"]):
                global_incumbent = dict(prefix_row)
    return {
        "schema": "paper_i_hh_optuna_prefix_best_v1",
        "source": "completed_trial_result_json_history",
        "per_trial_prefix_best": per_trial,
        "global_best_by_iteration": [
            global_best_by_iteration[k] for k in sorted(global_best_by_iteration)
        ],
        "global_current_best_prefix": global_incumbent,
    }


def _write_current_best(output_dir: Path, *, study_name: str, objective_mode: str, observations: Sequence[TrialObservation]) -> None:
    rows = [obs for obs in observations if not bool(obs.warm_start) and int(obs.returncode or 0) == 0]
    def _row(obs: TrialObservation) -> dict[str, Any]:
        return {
            "trial_number": _trial_number_from_observation(obs),
            "objective_value": _objective_value_for_mode(obs, objective_mode),
            "graph_cost_scalar": _graph_cost_scalar(obs),
            "graph_hardware_objective_scalar": _graph_hardware_objective_scalar(obs),
            "paper_i_shot_cost_scalar": _paper_i_shot_cost_scalar(obs),
            "graph_plus_shot_objective_scalar": _graph_speed_objective(obs),
            **_observation_to_user_attrs(obs),
        }
    best_by_objective = min(observations, key=lambda obs: _objective_value_for_mode(obs, objective_mode), default=None)
    best_by_delta_e = min([obs for obs in observations if obs.abs_delta_e is not None], key=lambda obs: float(obs.abs_delta_e), default=None)
    energy_feasible = [obs for obs in observations if obs.dominance_target_abs_delta_e is not None and obs.abs_delta_e is not None and float(obs.abs_delta_e) <= float(obs.dominance_target_abs_delta_e)]
    best_energy_feasible_by_graph_cost = min(energy_feasible, key=_graph_speed_objective, default=None)
    graph_dominant = []
    graph_shot_dominant = []
    for obs in energy_feasible:
        graph_n2q_violation, graph_depth_violation = _graph_dominance_violations(obs)
        if graph_n2q_violation is not None and graph_depth_violation is not None and graph_n2q_violation <= 0.0 and graph_depth_violation <= 0.0:
            graph_dominant.append(obs)
            shot_violation = _shot_dominance_violation(obs)
            if shot_violation is None or float(shot_violation) <= 0.0:
                graph_shot_dominant.append(obs)
    best_geo_dominant_by_graph_plus_shot = min(graph_shot_dominant, key=_graph_speed_objective, default=None)
    prefix_best = _prefix_best_payload(rows, objective_mode=objective_mode)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "paper_i_hh_optuna_current_best_v2",
        "study_name": str(study_name),
        "objective_mode": str(objective_mode),
        "completed_trial_count": len(rows),
        "energy_feasible_count": len(energy_feasible),
        "geo_graph_dominant_count": len(graph_dominant),
        "geo_graph_shot_dominant_count": len(graph_shot_dominant),
        "best_by_objective": None if best_by_objective is None else _row(best_by_objective),
        "best_by_delta_e": None if best_by_delta_e is None else _row(best_by_delta_e),
        "best_energy_feasible_by_graph_cost": None if best_energy_feasible_by_graph_cost is None else _row(best_energy_feasible_by_graph_cost),
        "best_energy_feasible_by_graph_plus_shot": None if best_energy_feasible_by_graph_cost is None else _row(best_energy_feasible_by_graph_cost),
        "best_geo_dominant_by_graph_plus_shot": None if best_geo_dominant_by_graph_plus_shot is None else _row(best_geo_dominant_by_graph_plus_shot),
        "prefix_best": prefix_best,
    }
    path = Path(output_dir) / "current_best.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_optuna_storage(raw: str | None) -> str | None:
    if raw in {None, ""}:
        return None
    storage = str(raw)
    if "://" in storage:
        return storage
    path = Path(storage)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _resolved_study_name(*, lane: str, epsilon_abs_delta_e: float, study_name_prefix: str | None = None) -> str:
    suffix = f"{lane}_eps_{_float_slug(float(epsilon_abs_delta_e))}"
    prefix = "" if study_name_prefix in {None, ""} else str(study_name_prefix).strip()
    return suffix if prefix == "" else f"{prefix}_{suffix}"


def _study_has_params(study: Any, params: Mapping[str, Any]) -> bool:
    target = {str(k): v for k, v in dict(params).items()}
    for frozen in study.get_trials(deepcopy=False):
        if {str(k): v for k, v in dict(getattr(frozen, "params", {}) or {}).items()} == target:
            return True
    return False


def _stored_observations_from_study(optuna: Any, study: Any) -> list[TrialObservation]:
    observations: list[TrialObservation] = []
    field_names = set(TrialObservation.__dataclass_fields__.keys())
    required = {
        "lane",
        "epsilon_abs_delta_e",
        "params",
        "objective_lexicographic",
        "abs_delta_e",
        "compiled_count_2q",
        "compiled_depth",
        "logical_operator_count",
        "runtime_parameter_count",
        "feasible",
        "constraints",
    }
    for frozen in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        attrs = dict(getattr(frozen, "user_attrs", {}) or {})
        if not required.issubset(attrs):
            continue
        payload = {name: attrs[name] for name in field_names if name in attrs}
        try:
            observations.append(TrialObservation(**payload))
        except Exception:
            continue
    return observations


def _study_progress_snapshot(*, studies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": _PIPELINE_NAME,
        "studies": [dict(x) for x in studies],
    }


def _feasible_rows(observations: Sequence[TrialObservation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obs in observations:
        if not bool(obs.feasible):
            continue
        if obs.abs_delta_e is None or obs.compiled_count_2q is None:
            continue
        rows.append(
            {
                "base_preset": obs.base_preset,
                "params": dict(obs.params),
                "delta_e_abs": float(obs.abs_delta_e),
                "best_compiled_count_2q": int(obs.compiled_count_2q),
                "best_compiled_depth": int(obs.compiled_depth or 0),
                "logical_operator_count": int(obs.logical_operator_count or 0),
                "runtime_parameter_count": int(obs.runtime_parameter_count or 0),
                "graph_count_2q": obs.graph_count_2q,
                "graph_depth": obs.graph_depth,
                "graph_count_1q": obs.graph_count_1q,
                "graph_theta_count": obs.graph_theta_count,
                "measurement_work_shots": obs.measurement_work_shots,
                "paper_i_table_shots_total": obs.paper_i_table_shots_total,
                "paper_i_table_s_alg": obs.paper_i_table_s_alg,
                "measurement_work_records": obs.measurement_work_records,
                "resource_cost_source": obs.resource_cost_source,
                "case_dir": obs.case_dir,
                "warm_start": bool(obs.warm_start),
            }
        )
    return rows


def _run_single_study(
    *,
    optuna: Any,
    python_bin: str,
    study_spec: StudySpec,
    output_dir: Path,
    compile_backend: str,
    compile_opt_level: int,
    compile_seed: int,
    use_default_warm_starts: bool,
    hamiltonian_overrides: HhHamiltonianOverrides | None = None,
    objective_mode: str = "cost_feasible",
    compile_enabled: bool = True,
    energy_only_surface: bool = False,
    speed_surface_profile: str = "standard",
    phase2_w_shot_profile_space: str = "default",
    enable_prune_prefilter_profile_space: bool = False,
    dominance_target_abs_delta_e: float | None = None,
    dominance_target_iteration: int | None = None,
    dominance_target_graph_count_2q: float | None = None,
    dominance_target_graph_depth: float | None = None,
    dominance_target_s_alg: float | None = None,
    phase2_w_shot_override: float | None = None,
    runtime_split_mode_override: str | None = None,
    child_pool_expansion_mode_override: str | None = None,
    child_pool_expansion_symmetry_policy_override: str | None = None,
    child_pool_expansion_max_subset_size_override: int | None = None,
    exact_gs_override: float | None = None,
    exact_gs_reference_json: Path | None = None,
    force_adapt_benchmark_target_abs_delta_e: float | None = None,
    force_adapt_max_depth: int | None = None,
    force_adapt_maxiter: int | None = None,
    force_adapt_final_refit_maxiter: int | None = None,
    force_adapt_drop_floor: float | None = None,
    force_adapt_drop_patience: int | None = None,
    force_adapt_drop_min_depth: int | None = None,
    force_adapt_full_refit_every: int | None = None,
    force_adapt_final_full_refit: str | None = None,
    force_adapt_allow_repeats: bool | None = None,
    force_phase0_pilot_max_records: int | None = None,
    force_phase1_shortlist_size: int | None = None,
    force_phase2_shortlist_fraction: float | None = None,
    force_phase2_shortlist_size: int | None = None,
    force_adapt_parallel_gradient_workers: int | None = None,
    force_adapt_beam_parent_workers: int | None = None,
    force_adapt_spsa_parallel_evaluations: int | None = None,
    force_adapt_pool_class_filter_json: Path | None = None,
    force_phase1_prune_prefilter_json: Path | None = None,
    force_spsa_profile: str | None = None,
    force_adapt_resume_scaffold_json: Path | None = None,
    force_adapt_resume_mode: str | None = None,
    force_adapt_segment_id: str | None = None,
    force_adapt_segment_target_depth: int | None = None,
    force_adapt_segment_max_new_admissions: int | None = None,
    force_adapt_segment_wallclock_cap_s: float | None = None,
    force_adapt_resume_compile_smoke: str | None = None,
    force_adapt_resume_smoke_backend: str | None = None,
    force_static_route_id: str | None = None,
    force_static_meta_feature_profile: str | None = None,
    force_phase3_symmetry_mitigation_mode: str | None = None,
    force_route_a_paper_i_production: bool = False,
    force_phase1_prune_full_window: bool = False,
    force_phase1_prune_recovery_trust_radius: float | None = None,
    force_phase1_prune_schur_nomination_route: str | None = None,
    force_phase1_prune_metric_schur_mu: float | None = None,
    force_phase1_prune_metric_schur_solve_mode: str | None = None,
    force_phase1_prune_metric_schur_cost_weighting: str | None = None,
    search_inner_optimizer: str = _SEARCH_INNER_OPTIMIZER,
    enqueue_param_rows: Sequence[Mapping[str, Any]] = (),
    optuna_storage: str | None = None,
    study_name_prefix: str | None = None,
    load_if_exists: bool = False,
) -> dict[str, Any]:
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        constant_liar=True,
        constraints_func=lambda frozen_trial: list(frozen_trial.user_attrs.get("constraints", [1.0, 1.0])),
        n_startup_trials=int(max(1, study_spec.n_startup_trials)),
    )
    study_name = _resolved_study_name(
        lane=str(study_spec.lane),
        epsilon_abs_delta_e=float(study_spec.epsilon_abs_delta_e),
        study_name_prefix=study_name_prefix,
    )
    normalized_storage = _normalize_optuna_storage(optuna_storage)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
        storage=normalized_storage,
        load_if_exists=bool(load_if_exists),
    )
    observations: list[TrialObservation] = _stored_observations_from_study(optuna, study)
    override_payload = _hh_hamiltonian_override_payload(hamiltonian_overrides)
    artifact_observation_policy = (
        "disabled_for_hamiltonian_override"
        if override_payload
        else "artifact_backed_observations_allowed"
    )
    reference_observations = [] if override_payload else _default_reference_observations(
        lane=study_spec.lane,
        epsilon_abs_delta_e=study_spec.epsilon_abs_delta_e,
        compile_backend=compile_backend,
        extra_preset_names=study_spec.extra_warm_start_preset_names,
        restricted_base_preset_names=study_spec.restricted_base_preset_names,
    )
    shortlist_refine_anchor_param_values = (
        _shortlist_refine_anchor_param_values(enqueue_param_rows)
        if str(speed_surface_profile) == "shortlist_refine"
        else {}
    )
    search_inner_optimizer = _normalize_search_inner_optimizer(search_inner_optimizer)
    effective_force_spsa_profile = force_spsa_profile if search_inner_optimizer == "SPSA" else "current"
    if use_default_warm_starts and not override_payload:
        for obs in _default_warm_start_observations(
            lane=study_spec.lane,
            epsilon_abs_delta_e=study_spec.epsilon_abs_delta_e,
            compile_backend=compile_backend,
            extra_preset_names=study_spec.extra_warm_start_preset_names,
            restricted_base_preset_names=study_spec.restricted_base_preset_names,
        ):
            try:
                if _study_has_params(study, obs.params):
                    continue
                study.add_trial(
                    optuna.create_trial(
                        params=dict(obs.params),
                        distributions=_build_distributions(
                            optuna,
                            study_spec.lane,
                            obs.params,
                            study_spec.restricted_base_preset_names,
                            energy_only_surface=bool(energy_only_surface),
                            speed_surface_profile=str(speed_surface_profile),
                            force_spsa_profile=effective_force_spsa_profile,
                            phase2_w_shot_profile_space=str(phase2_w_shot_profile_space),
                            anchor_param_values=shortlist_refine_anchor_param_values,
                            enable_prune_prefilter_profile_space=bool(enable_prune_prefilter_profile_space),
                        ),
                        value=_objective_value_for_mode(obs, objective_mode),
                        user_attrs=_observation_to_user_attrs(obs),
                    )
                )
                observations.append(obs)
            except Exception:
                continue

    if str(speed_surface_profile) == "staged_graph" and bool(energy_only_surface):
        spsa_profiles = [str(effective_force_spsa_profile)] if effective_force_spsa_profile not in {None, ""} else list(_SPSA_PROFILE_OPTIONS)
        for spsa_profile in spsa_profiles:
            for ml_candidate_profile in ("ml_p10", "ml_median", "ml_p90"):
                try:
                    study.enqueue_trial(
                        {
                            "spsa_profile": str(spsa_profile),
                            "ml_candidate_profile": str(ml_candidate_profile),
                        },
                        skip_if_exists=True,
                    )
                except Exception:
                    continue
    elif str(speed_surface_profile) == "staged_graph":
        for staged_params in _staged_graph_speed_enqueue_params(
            study_spec.lane,
            study_spec.restricted_base_preset_names,
            force_spsa_profile=effective_force_spsa_profile,
        ):
            try:
                study.enqueue_trial(dict(staged_params), skip_if_exists=True)
            except Exception:
                continue
    elif str(speed_surface_profile) == "staged_shot":
        for staged_params in _staged_shot_speed_enqueue_params(
            study_spec.lane,
            study_spec.restricted_base_preset_names,
            force_spsa_profile=effective_force_spsa_profile,
        ):
            try:
                study.enqueue_trial(dict(staged_params), skip_if_exists=True)
            except Exception:
                continue

    enqueued_external_param_count = 0
    for raw_params in enqueue_param_rows:
        try:
            staged_params = _normalise_enqueue_params_for_space(
                raw_params,
                lane=study_spec.lane,
                restricted_base_preset_names=study_spec.restricted_base_preset_names,
                energy_only_surface=bool(energy_only_surface),
                speed_surface_profile=str(speed_surface_profile),
                force_spsa_profile=effective_force_spsa_profile,
                phase2_w_shot_profile_space=str(phase2_w_shot_profile_space),
                anchor_param_values=shortlist_refine_anchor_param_values,
                enable_prune_prefilter_profile_space=bool(enable_prune_prefilter_profile_space),
            )
            study.enqueue_trial(staged_params, skip_if_exists=True)
            enqueued_external_param_count += 1
        except Exception:
            continue

    for preset_name in study_spec.enqueue_preset_names:
        if str(preset_name) not in _base_preset_library():
            continue
        try:
            study.enqueue_trial(dict(asdict(_warm_start_params_from_preset_name(str(preset_name)))), skip_if_exists=True)
        except Exception:
            continue

    for step_idx in range(int(max(0, study_spec.n_trials))):
        trial = study.ask()
        params = _suggest_trial_params(
            trial,
            study_spec.lane,
            study_spec.restricted_base_preset_names,
            energy_only_surface=bool(energy_only_surface),
            speed_surface_profile=str(speed_surface_profile),
            force_spsa_profile=effective_force_spsa_profile,
            phase2_w_shot_profile_space=str(phase2_w_shot_profile_space),
            anchor_param_values=shortlist_refine_anchor_param_values,
            enable_prune_prefilter_profile_space=bool(enable_prune_prefilter_profile_space),
        )
        obs = _evaluate_trial(
            python_bin=str(python_bin),
            params=params,
            lane=study_spec.lane,
            epsilon_abs_delta_e=study_spec.epsilon_abs_delta_e,
            output_dir=output_dir,
            trial_index=int(trial.number),
            compile_backend=compile_backend,
            compile_opt_level=int(compile_opt_level),
            compile_seed=int(compile_seed),
            hamiltonian_overrides=hamiltonian_overrides,
            compile_enabled=bool(compile_enabled),
            phase2_w_shot_override=phase2_w_shot_override,
            runtime_split_mode_override=runtime_split_mode_override,
            child_pool_expansion_mode_override=child_pool_expansion_mode_override,
            child_pool_expansion_symmetry_policy_override=child_pool_expansion_symmetry_policy_override,
            child_pool_expansion_max_subset_size_override=child_pool_expansion_max_subset_size_override,
            exact_gs_override=exact_gs_override,
            exact_gs_reference_json=exact_gs_reference_json,
            force_adapt_benchmark_target_abs_delta_e=force_adapt_benchmark_target_abs_delta_e,
            force_adapt_max_depth=force_adapt_max_depth,
            force_adapt_maxiter=force_adapt_maxiter,
            force_adapt_final_refit_maxiter=force_adapt_final_refit_maxiter,
            force_adapt_drop_floor=force_adapt_drop_floor,
            force_adapt_drop_patience=force_adapt_drop_patience,
            force_adapt_drop_min_depth=force_adapt_drop_min_depth,
            force_adapt_full_refit_every=force_adapt_full_refit_every,
            force_adapt_final_full_refit=force_adapt_final_full_refit,
            force_adapt_allow_repeats=force_adapt_allow_repeats,
            force_phase0_pilot_max_records=force_phase0_pilot_max_records,
            force_phase1_shortlist_size=force_phase1_shortlist_size,
            force_phase2_shortlist_fraction=force_phase2_shortlist_fraction,
            force_phase2_shortlist_size=force_phase2_shortlist_size,
            force_adapt_parallel_gradient_workers=force_adapt_parallel_gradient_workers,
            force_adapt_beam_parent_workers=force_adapt_beam_parent_workers,
            force_adapt_spsa_parallel_evaluations=force_adapt_spsa_parallel_evaluations,
            force_adapt_pool_class_filter_json=force_adapt_pool_class_filter_json,
            force_phase1_prune_prefilter_json=force_phase1_prune_prefilter_json,
            force_adapt_resume_scaffold_json=force_adapt_resume_scaffold_json,
            force_adapt_resume_mode=force_adapt_resume_mode,
            force_adapt_segment_id=force_adapt_segment_id,
            force_adapt_segment_target_depth=force_adapt_segment_target_depth,
            force_adapt_segment_max_new_admissions=force_adapt_segment_max_new_admissions,
            force_adapt_segment_wallclock_cap_s=force_adapt_segment_wallclock_cap_s,
            force_adapt_resume_compile_smoke=force_adapt_resume_compile_smoke,
            force_adapt_resume_smoke_backend=force_adapt_resume_smoke_backend,
            force_static_route_id=force_static_route_id,
            force_static_meta_feature_profile=force_static_meta_feature_profile,
            force_phase3_symmetry_mitigation_mode=force_phase3_symmetry_mitigation_mode,
            force_route_a_paper_i_production=force_route_a_paper_i_production,
            force_phase1_prune_full_window=bool(force_phase1_prune_full_window),
            force_phase1_prune_recovery_trust_radius=force_phase1_prune_recovery_trust_radius,
            force_phase1_prune_schur_nomination_route=force_phase1_prune_schur_nomination_route,
            force_phase1_prune_metric_schur_mu=force_phase1_prune_metric_schur_mu,
            force_phase1_prune_metric_schur_solve_mode=force_phase1_prune_metric_schur_solve_mode,
            force_phase1_prune_metric_schur_cost_weighting=force_phase1_prune_metric_schur_cost_weighting,
            search_inner_optimizer=search_inner_optimizer,
            require_graph_cost=(
                str(objective_mode)
                in {
                    "cost_feasible",
                    "graph_cost_speed_feasible",
                    "geo_energy_then_graph_cost",
                    "geo_energy_then_graph_shot_cost",
                    "geo_energy_then_shot_graph_cost",
                    "prune_zero_then_energy_shot_graph_cost",
                }
            ),
            require_zero_prune_rejections=bool(
                str(objective_mode) == "prune_zero_then_energy_shot_graph_cost"
            ),
            dominance_target_abs_delta_e=dominance_target_abs_delta_e,
            dominance_target_iteration=dominance_target_iteration,
            dominance_target_graph_count_2q=dominance_target_graph_count_2q,
            dominance_target_graph_depth=dominance_target_graph_depth,
            dominance_target_s_alg=dominance_target_s_alg,
        )
        for key, value in _observation_to_user_attrs(obs).items():
            trial.set_user_attr(str(key), value)
        objective_value = _objective_value_for_mode(obs, objective_mode)
        study.tell(trial, objective_value)
        observations.append(obs)
        _append_trial_event(
            output_dir,
            _trial_event_payload(
                study_name=study_name,
                trial_number=int(trial.number),
                objective_mode=str(objective_mode),
                objective_value=float(objective_value),
                obs=obs,
            ),
        )
        _write_current_best(output_dir, study_name=study_name, objective_mode=str(objective_mode), observations=observations)
        _write_progress(
            output_dir,
            _study_progress_snapshot(
                studies=[
                    {
                        "study_name": study_name,
                        "lane": study_spec.lane,
                        "epsilon_abs_delta_e": float(study_spec.epsilon_abs_delta_e),
                        "warm_start_count": int(sum(1 for x in observations if x.warm_start)),
                        "completed_trial_count": int(sum(1 for x in observations if not x.warm_start)),
                        "feasible_count": int(sum(1 for x in observations if x.feasible)),
                        "last_completed_trial": int(step_idx),
                        "objective_mode": str(objective_mode),
                        "compile_enabled": bool(compile_enabled),
                        "energy_only_surface": bool(energy_only_surface),
                        "speed_surface_profile": str(speed_surface_profile),
                        "runtime_split_mode_override": runtime_split_mode_override,
                        "child_pool_expansion_mode_override": child_pool_expansion_mode_override,
                    }
                ]
            ),
        )
        energy_rows = [x for x in observations if x.abs_delta_e is not None]
        best_energy = min(energy_rows, key=lambda x: float(x.abs_delta_e)) if energy_rows else None
        feasible_energy_rows = [x for x in energy_rows if bool(x.feasible)]
        best_feasible_energy = (
            min(feasible_energy_rows, key=lambda x: float(x.abs_delta_e))
            if feasible_energy_rows
            else None
        )
        print(
            "OPTUNA_TRIAL_PROGRESS "
            + json.dumps(
                {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "study_name": study_name,
                    "lane": study_spec.lane,
                    "trial_number": int(trial.number),
                    "completed_trial_count": int(sum(1 for x in observations if not x.warm_start)),
                    "n_trials_requested": int(study_spec.n_trials),
                    "abs_delta_e": obs.abs_delta_e,
                    "feasible": bool(obs.feasible),
                    "invalid_reasons": list(obs.invalid_reasons),
                    "graph_count_2q": obs.graph_count_2q,
                    "graph_depth": obs.graph_depth,
                    "measurement_work_shots": obs.measurement_work_shots,
                    "paper_i_table_shots_total": obs.paper_i_table_shots_total,
                    "paper_i_table_s_alg": obs.paper_i_table_s_alg,
                    "measurement_work_records": obs.measurement_work_records,
                    "resource_cost_source": obs.resource_cost_source,
                    "total_elapsed_s": obs.total_elapsed_s,
                    "best_abs_delta_e_so_far": (
                        None if best_energy is None else float(best_energy.abs_delta_e)
                    ),
                    "best_abs_delta_e_trial_case_dir": (
                        None if best_energy is None else str(best_energy.case_dir)
                    ),
                    "best_feasible_abs_delta_e_so_far": (
                        None if best_feasible_energy is None else float(best_feasible_energy.abs_delta_e)
                    ),
                    "best_feasible_trial_case_dir": (
                        None if best_feasible_energy is None else str(best_feasible_energy.case_dir)
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    study_feasible_rows = _feasible_rows(observations)
    reference_feasible_rows = _feasible_rows(reference_observations)
    feasible_frontier = pareto_front(
        [*study_feasible_rows, *reference_feasible_rows],
        x_key="delta_e_abs",
        y_key="best_compiled_count_2q",
    )
    payload = {
        "study_name": study_name,
        "optuna_storage": normalized_storage,
        "study_name_prefix": study_name_prefix,
        "load_if_exists": bool(load_if_exists),
        "lane": study_spec.lane,
        "epsilon_abs_delta_e": float(study_spec.epsilon_abs_delta_e),
        "objective_mode": str(objective_mode),
        "compile_enabled": bool(compile_enabled),
        "energy_only_surface": bool(energy_only_surface),
        "search_inner_optimizer": _normalize_search_inner_optimizer(search_inner_optimizer),
        "speed_surface_profile": str(speed_surface_profile),
        "shortlist_refine_anchor_enabled": bool(shortlist_refine_anchor_param_values),
        "shortlist_refine_anchor_param_values": dict(shortlist_refine_anchor_param_values),
        "shortlist_refine_free_param_names": sorted(_SHORTLIST_REFINE_FREE_PARAM_NAMES),
        "phase2_w_shot_override": phase2_w_shot_override,
        "runtime_split_mode_override": runtime_split_mode_override,
        "dominance_target_graph_count_2q": dominance_target_graph_count_2q,
        "dominance_target_graph_depth": dominance_target_graph_depth,
        "dominance_target_s_alg": dominance_target_s_alg,
        "force_adapt_benchmark_target_abs_delta_e": (
            None
            if force_adapt_benchmark_target_abs_delta_e is None
            else float(force_adapt_benchmark_target_abs_delta_e)
        ),
        "force_adapt_max_depth": (None if force_adapt_max_depth is None else int(force_adapt_max_depth)),
        "force_adapt_maxiter": (None if force_adapt_maxiter is None else int(force_adapt_maxiter)),
        "force_adapt_final_refit_maxiter": (
            None if force_adapt_final_refit_maxiter is None else int(force_adapt_final_refit_maxiter)
        ),
        "force_adapt_full_refit_every": (
            None if force_adapt_full_refit_every is None else int(force_adapt_full_refit_every)
        ),
        "force_adapt_final_full_refit": force_adapt_final_full_refit,
        "force_adapt_allow_repeats": force_adapt_allow_repeats,
        "force_phase0_pilot_max_records": (
            None if force_phase0_pilot_max_records is None else int(force_phase0_pilot_max_records)
        ),
        "force_phase1_shortlist_size": (
            None if force_phase1_shortlist_size is None else int(force_phase1_shortlist_size)
        ),
        "force_phase2_shortlist_fraction": (
            None if force_phase2_shortlist_fraction is None else float(force_phase2_shortlist_fraction)
        ),
        "force_phase2_shortlist_size": (
            None if force_phase2_shortlist_size is None else int(force_phase2_shortlist_size)
        ),
        "force_adapt_parallel_gradient_workers": (
            None if force_adapt_parallel_gradient_workers is None else int(force_adapt_parallel_gradient_workers)
        ),
        "force_adapt_beam_parent_workers": (
            None if force_adapt_beam_parent_workers is None else int(force_adapt_beam_parent_workers)
        ),
        "force_adapt_pool_class_filter_json": (
            None if force_adapt_pool_class_filter_json is None else str(Path(force_adapt_pool_class_filter_json))
        ),
        "force_static_route_id": force_static_route_id,
        "force_static_meta_feature_profile": force_static_meta_feature_profile,
        "force_phase3_symmetry_mitigation_mode": force_phase3_symmetry_mitigation_mode,
        "force_route_a_paper_i_production": bool(force_route_a_paper_i_production),
        "adapt_exact_gs_override": (None if exact_gs_override is None else float(exact_gs_override)),
        "adapt_exact_gs_reference_json": (None if exact_gs_reference_json is None else str(Path(exact_gs_reference_json))),
        "n_trials_requested": int(study_spec.n_trials),
        "n_startup_trials": int(study_spec.n_startup_trials),
        "external_enqueue_param_count": int(enqueued_external_param_count),
        "restricted_base_preset_names": list(study_spec.restricted_base_preset_names),
        "hamiltonian_overrides": dict(override_payload),
        "artifact_observation_policy": artifact_observation_policy,
        "warm_start_count": int(sum(1 for x in observations if x.warm_start)),
        "completed_trial_count": int(sum(1 for x in observations if not x.warm_start)),
        "reference_count": int(len(reference_observations)),
        "feasible_count": int(sum(1 for x in observations if x.feasible)),
        "study_feasible_frontier": pareto_front(study_feasible_rows, x_key="delta_e_abs", y_key="best_compiled_count_2q"),
        "feasible_frontier": feasible_frontier,
        "observations": [asdict(x) for x in observations],
        "reference_observations": [asdict(x) for x in reference_observations],
        "best_energy_observation": (
            asdict(min([x for x in observations if x.abs_delta_e is not None], key=lambda x: float(x.abs_delta_e)))
            if any(x.abs_delta_e is not None for x in observations)
            else None
        ),
        "best_feasible": _best_feasible_row([*study_feasible_rows, *reference_feasible_rows]),
    }
    _write_current_best(output_dir, study_name=study_name, objective_mode=str(objective_mode), observations=observations)
    study_dir = output_dir / study_spec.lane / f"eps_{_float_slug(study_spec.epsilon_abs_delta_e)}"
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "study_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Constrained Optuna pilot for HH cost-vs-energy scaffold search."
    )
    p.add_argument("--tag", type=str, default=f"hh_l2_cost_energy_optuna_{_timestamp_slug()}")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--lanes", type=str, default=",".join(_DEFAULT_LANES), help="Comma-separated study lanes (canonical,global,legacy).")
    p.add_argument("--epsilon-bands", type=str, default=",".join(str(x) for x in _DEFAULT_EPSILON_BANDS), help="Comma-separated abs-delta-E feasibility bands.")
    p.add_argument("--n-trials", type=int, default=12)
    p.add_argument("--n-startup-trials", type=int, default=6)
    p.add_argument(
        "--optuna-storage",
        type=str,
        default=None,
        help=(
            "Persistent Optuna storage URL or local SQLite path. Plain paths are "
            "resolved relative to the repo and converted to sqlite:/// URLs."
        ),
    )
    p.add_argument(
        "--study-name-prefix",
        type=str,
        default=None,
        help=(
            "Optional stable prefix for persistent studies. The lane and epsilon "
            "suffix are appended automatically."
        ),
    )
    p.add_argument(
        "--load-if-exists",
        action="store_true",
        help="Resume an existing persistent study instead of failing when the study name already exists.",
    )
    p.add_argument("--compile-backend", type=str, default=_DEFAULT_COMPILE_BACKEND)
    p.add_argument("--compile-opt-level", type=int, default=_DEFAULT_COMPILE_OPT_LEVEL)
    p.add_argument("--compile-seed", type=int, default=_DEFAULT_COMPILE_SEED)
    p.add_argument(
        "--objective-mode",
        choices=[
            "cost_feasible",
            "energy",
            "graph_cost_speed_feasible",
            "geo_dominance_first",
            "geo_energy_then_graph_cost",
            "geo_energy_then_graph_shot_cost",
            "geo_energy_then_shot_graph_cost",
            "shot_then_energy_graph_cost",
            "geo_energy_gate_then_shot_energy_graph_cost",
            "prune_zero_then_energy_shot_graph_cost",
        ],
        default="cost_feasible",
        help=(
            "Optuna objective: cost_feasible minimizes compiled cost under the energy band; "
            "graph_cost_speed_feasible uses cached Marrakesh graph-span cost then measurement/runtime work; "
            "geo_dominance_first first beats a Geo-ADAPT energy/iteration target, then uses resource work; "
            "geo_energy_then_graph_cost/geo_energy_then_graph_shot_cost gates on Geo-ADAPT energy only, then minimizes graph hardware/shot work; "
            "geo_energy_then_shot_graph_cost gates on Geo-ADAPT energy, then prioritizes S_alg before graph proxy; "
            "shot_then_energy_graph_cost prioritizes S_alg first, then energy, then graph proxy; "
            "geo_energy_gate_then_shot_energy_graph_cost gates on Geo-ADAPT energy, then prioritizes S_alg, residual Delta E, and graph proxy; "
            "prune_zero_then_energy_shot_graph_cost first requires zero prune rollback/delete rejections, then uses the energy-gated S_alg/graph objective; "
            "energy minimizes abs delta-E directly."
        ),
    )
    p.add_argument("--skip-compile", action="store_true", help="Skip per-trial compile/resource extraction; intended for --objective-mode energy.")
    p.add_argument("--energy-only-surface", action="store_true", help="Freeze cost/shot categorical knobs that only make sense for cost-oriented studies.")
    p.add_argument(
        "--search-inner-optimizer",
        choices=sorted(_SEARCH_INNER_OPTIMIZER_CHOICES),
        default=_SEARCH_INNER_OPTIMIZER,
        help=(
            "Inner optimizer used by fresh Optuna trial subprocesses. Defaults to SPSA for "
            "backward compatibility; POWELL is intended for deterministic SNAKE-policy tuning."
        ),
    )
    p.add_argument(
        "--speed-surface-profile",
        choices=["standard", "staged_graph", "staged_shot", "shortlist_refine", "energy_discovery", _HH_ROUTEA_FULL_POLICY_PROFILE],
        default="standard",
        help=(
            "Optuna search surface; staged_graph starts with heavy graph-cost trials, staged_shot "
            "biases toward S_alg, and shortlist_refine anchors non-shortlist settings to any "
            "per-regime enqueue-prior rows while sampling explicit shortlist/window/threshold knobs "
            "with maturity_shot_profile forced to base. hh_routea_full_policy_v1 samples explicit "
            "Route-A SNAKE prune, batch, SPSA, shortlist, and scoring controls while preserving "
            "the Paper-I production identity locks."
        ),
    )
    p.add_argument(
        "--phase2-w-shot-profile-space",
        choices=["default", "legacy_with_zero"],
        default="default",
        help="Categorical menu for phase2_w_shot_profile; legacy_with_zero preserves older v5 Optuna study compatibility.",
    )
    p.add_argument("--dominance-target-abs-delta-e", type=float, default=None, help="Geo-dominance objective energy target to beat.")
    p.add_argument("--dominance-target-iteration", type=int, default=None, help="Geo-dominance objective accepted-ADAPT iteration count to beat strictly.")
    p.add_argument("--dominance-target-graph-count-2q", type=float, default=None, help="Geo graph-span two-qubit count target to beat or match after energy dominance.")
    p.add_argument("--dominance-target-graph-depth", type=float, default=None, help="Geo graph-span two-qubit depth target to beat or match after energy dominance.")
    p.add_argument("--dominance-target-s-alg", type=float, default=None, help="Geo Paper-I normalized algorithmic shot/work target S_alg to beat or match after energy and graph dominance.")
    p.add_argument("--force-phase2-w-shot", type=float, default=None, help="Override --phase2-w-shot in every evaluated trial, e.g. 0.0 for shot-neutral energy search.")
    p.add_argument(
        "--force-runtime-split-mode",
        choices=["off", "shortlist_pauli_children_v1"],
        default=None,
        help="Force --phase3-runtime-split-mode for every trial; non-off values add the explicit archival diagnostic override flag.",
    )
    p.add_argument(
        "--force-adapt-child-pool-expansion-mode",
        choices=["off", "global_pauli_child_sets_v1", "pauli_child_sets_v1"],
        default=None,
        help=(
            "Force SNAKE global child-set pool expansion before Phase 1 for every trial. "
            "Non-off values also force --phase3-runtime-split-mode off."
        ),
    )
    p.add_argument(
        "--force-adapt-child-pool-expansion-symmetry-policy",
        choices=["off", "hard_guard"],
        default=None,
        help="Force the global child-set pool symmetry policy for every trial.",
    )
    p.add_argument(
        "--force-adapt-child-pool-expansion-max-subset-size",
        type=int,
        default=None,
        help="Force the global child-set pool max subset size for every trial.",
    )
    p.add_argument("--adapt-exact-gs-override", type=float, default=None, help="Precomputed working-cutoff exact ground-state energy passed to every trial subprocess.")
    p.add_argument("--adapt-exact-gs-reference-json", type=Path, default=None, help="Precomputed exact-reference manifest passed to every trial subprocess for strict per-run lookup.")
    p.add_argument("--hh-L", dest="hh_L", type=int, default=None, help="Override HH site count for replayed trials.")
    p.add_argument("--hh-t", type=float, default=None, help="Override HH hopping t for replayed trials.")
    p.add_argument("--hh-u", type=float, default=None, help="Override HH Hubbard U for replayed trials.")
    p.add_argument("--hh-omega0", type=float, default=None, help="Override HH phonon frequency omega0 for replayed trials.")
    p.add_argument("--hh-lambda", dest="hh_lambda", type=float, default=None, help="Paper-I HH lambda_ep=2*g_ep^2/(t*omega0); derives --g-ep when --hh-g-ep is omitted.")
    p.add_argument("--hh-g-ep", type=float, default=None, help="Override HH electron-phonon coupling g directly.")
    p.add_argument("--n-ph-work", type=int, default=None, help="Override working HH cutoff; passed to the ADAPT CLI as --n-ph-max.")
    p.add_argument("--n-ph-ref", type=int, default=None, help="Reference cutoff to record for posthoc same-suite sidecars; not passed to the ADAPT CLI.")
    p.add_argument("--force-adapt-pool", type=str, default=None, help="Force the ADAPT pool passed to the underlying static ADAPT CLI, e.g. full_meta.")
    p.add_argument("--force-adapt-pool-class-filter-json", type=Path, default=None, help="Force the HH full_meta class filter JSON, e.g. the full_meta_minus_hva filter.")
    p.add_argument(
        "--force-adapt-benchmark-target-abs-delta-e",
        type=float,
        default=None,
        help="Force --adapt-benchmark-target-abs-delta-e in every trial so local admission runs can stop at a static energy target.",
    )
    p.add_argument("--force-adapt-max-depth", type=int, default=None, help="Force --adapt-max-depth in every trial; use for local smokes/timing.")
    p.add_argument("--force-adapt-maxiter", type=int, default=None, help="Force --adapt-maxiter in every trial; use for local smokes/timing.")
    p.add_argument(
        "--force-adapt-final-refit-maxiter",
        type=int,
        default=None,
        help="Force --adapt-final-refit-maxiter in every trial; 0 reuses --adapt-maxiter.",
    )
    p.add_argument("--force-adapt-drop-floor", type=float, default=None, help="Force --adapt-drop-floor in every trial.")
    p.add_argument("--force-adapt-drop-patience", type=int, default=None, help="Force --adapt-drop-patience in every trial.")
    p.add_argument("--force-adapt-drop-min-depth", type=int, default=None, help="Force --adapt-drop-min-depth in every trial.")
    p.add_argument("--force-adapt-full-refit-every", type=int, default=None, help="Force --adapt-full-refit-every in every trial.")
    p.add_argument(
        "--force-adapt-final-full-refit",
        choices=["true", "false"],
        default=None,
        help="Force --adapt-final-full-refit in every trial.",
    )
    p.set_defaults(force_adapt_allow_repeats=None)
    p.add_argument(
        "--force-adapt-allow-repeats",
        dest="force_adapt_allow_repeats",
        action="store_true",
        help="Force --adapt-allow-repeats in every trial.",
    )
    p.add_argument(
        "--force-adapt-no-repeats",
        dest="force_adapt_allow_repeats",
        action="store_false",
        help="Force --adapt-no-repeats in every trial.",
    )
    p.add_argument("--force-phase0-pilot-max-records", type=int, default=None, help="Force --phase0-pilot-max-records in every trial.")
    p.add_argument("--force-phase1-shortlist-size", type=int, default=None, help="Force --phase1-shortlist-size in every trial.")
    p.add_argument("--force-phase2-shortlist-fraction", type=float, default=None, help="Force --phase2-shortlist-fraction in every trial.")
    p.add_argument("--force-phase2-shortlist-size", type=int, default=None, help="Force --phase2-shortlist-size in every trial.")
    p.add_argument("--force-adapt-parallel-gradient-workers", type=int, default=None, help="Force --adapt-parallel-gradient-workers in every trial; 0 enables CPU-aware auto sizing.")
    p.add_argument("--force-adapt-beam-parent-workers", type=int, default=None, help="Force --adapt-beam-parent-workers in every trial; 0 enables CPU-aware auto sizing.")
    p.add_argument("--force-adapt-spsa-parallel-evaluations", type=int, default=None, help="Force --adapt-spsa-parallel-evaluations in every trial; values must be >= 1.")
    p.add_argument(
        "--enable-prune-prefilter-profile-space",
        action="store_true",
        help="Allow Optuna to sample motif-risk prune prefilter profiles. Requires --force-phase1-prune-prefilter-json.",
    )
    p.add_argument(
        "--force-phase1-prune-prefilter-json",
        type=Path,
        default=None,
        help="Force --phase1-prune-prefilter-json in every trial for sampled motif-risk prune profiles.",
    )
    p.add_argument(
        "--force-phase1-prune-recovery-trust-radius",
        type=float,
        default=None,
        help="Force bounded Schur prune nomination with --phase1-prune-recovery-trust-radius in every trial.",
    )
    p.add_argument(
        "--force-phase1-prune-full-window",
        action="store_true",
        help="Force --phase1-prune-local-window-size 0 in every trial.",
    )
    p.add_argument(
        "--force-phase1-prune-schur-nomination-route",
        choices=["hessian_coupling_v1", "metric_regularized_v1"],
        default=None,
        help="Force the prune Schur nomination route in every trial.",
    )
    p.add_argument(
        "--force-phase1-prune-metric-schur-mu",
        type=float,
        default=None,
        help="Force --phase1-prune-metric-schur-mu in every trial.",
    )
    p.add_argument(
        "--force-phase1-prune-metric-schur-solve-mode",
        choices=["stationary_gw_zero_v1", "gradient_corrected_v1"],
        default=None,
        help="Force the metric-prune Schur solve mode in every trial.",
    )
    p.add_argument(
        "--force-phase1-prune-metric-schur-cost-weighting",
        choices=["ansatz_entry_denominator_v1", "off"],
        default=None,
        help="Force the metric-prune Schur cost weighting in every trial.",
    )
    p.add_argument("--force-adapt-resume-scaffold-json", type=Path, default=None, help="Force --adapt-resume-scaffold-json in every trial and remove any conflicting --adapt-ref-json.")
    p.add_argument("--force-adapt-resume-mode", choices=["scaffold_v1"], default=None, help="Force --adapt-resume-mode in every trial.")
    p.add_argument("--force-adapt-segment-id", type=str, default=None, help="Force --adapt-segment-id in every trial.")
    p.add_argument("--force-adapt-segment-target-depth", type=int, default=None, help="Force --adapt-segment-target-depth in every trial.")
    p.add_argument("--force-adapt-segment-max-new-admissions", type=int, default=None, help="Force --adapt-segment-max-new-admissions in every trial.")
    p.add_argument("--force-adapt-segment-wallclock-cap-s", type=float, default=None, help="Force --adapt-segment-wallclock-cap-s in every trial.")
    p.add_argument(
        "--force-adapt-resume-compile-smoke",
        choices=["required", "auto", "off"],
        default=None,
        help="Force --adapt-resume-compile-smoke in every trial.",
    )
    p.add_argument("--force-adapt-resume-smoke-backend", type=str, default=None, help="Force --adapt-resume-smoke-backend in every trial.")
    p.add_argument("--force-spsa-profile", choices=sorted(_SPSA_PROFILE_OPTIONS), default=None, help="Restrict the Optuna search to one predefined SPSA profile.")
    p.add_argument("--force-static-route-id", type=str, default=None, help="Force --static-route-id in every trial, e.g. route_a.")
    p.add_argument("--force-static-meta-feature-profile", type=str, default=None, help="Force --static-meta-feature-profile in every trial, e.g. paper_i_production_v1.")
    p.add_argument("--force-route-a-paper-i-production", action="store_true", help="Lock the Paper-I Route-A production contract after Optuna sampling so sampled trial knobs cannot disable algebraic selector, batching, or recoverability prune.")
    p.add_argument(
        "--force-phase3-symmetry-mitigation-mode",
        choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"],
        default=None,
        help="Force --phase3-symmetry-mitigation-mode in every trial; use off for the HH speed path unless testing mitigation.",
    )
    p.add_argument("--restrict-base-presets", type=str, default="", help="Comma-separated searchable base presets for a focused study.")
    p.add_argument("--extra-warm-start-presets", type=str, default="", help="Comma-separated preset names to add as completed warm starts when artifact data exists.")
    p.add_argument("--enqueue-presets", type=str, default="", help="Comma-separated preset names to queue for fresh reruns under the current harness.")
    p.add_argument(
        "--enqueue-params-json",
        type=Path,
        default=None,
        help="Optional JSON manifest containing external Optuna parameter rows to enqueue before fresh trials.",
    )
    p.add_argument(
        "--enqueue-params-regime",
        type=str,
        default=None,
        help="Regime key used to select rows from --enqueue-params-json when it has a regimes object.",
    )
    p.add_argument("--no-default-warm-starts", action="store_true")
    return p


def _parse_csv(values: str | None) -> list[str]:
    if values in {None, ""}:
        return []
    return [str(x).strip() for x in str(values).split(",") if str(x).strip() != ""]


def _parse_float_csv(values: str | None) -> list[float]:
    out: list[float] = []
    for raw in _parse_csv(values):
        out.append(float(raw))
    return out


def _hh_hamiltonian_overrides_from_namespace(args: argparse.Namespace) -> HhHamiltonianOverrides:
    return _validate_hh_hamiltonian_overrides(
        HhHamiltonianOverrides(
            L=getattr(args, "hh_L", None),
            t=getattr(args, "hh_t", None),
            u=getattr(args, "hh_u", None),
            omega0=getattr(args, "hh_omega0", None),
            lambda_value=getattr(args, "hh_lambda", None),
            g_ep=getattr(args, "hh_g_ep", None),
            n_ph_work=getattr(args, "n_ph_work", None),
            n_ph_ref=getattr(args, "n_ph_ref", None),
            adapt_pool=getattr(args, "force_adapt_pool", None),
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    optuna = _import_optuna()
    hamiltonian_overrides = _hh_hamiltonian_overrides_from_namespace(args)
    objective_mode = str(args.objective_mode)
    compile_enabled = not bool(args.skip_compile)
    if objective_mode == "cost_feasible" and not compile_enabled:
        raise ValueError(
            "--skip-compile is only valid with --objective-mode energy, graph_cost_speed_feasible, "
            "geo_dominance_first, geo_energy_then_graph_cost, geo_energy_then_graph_shot_cost, "
            "geo_energy_then_shot_graph_cost, shot_then_energy_graph_cost, "
            "geo_energy_gate_then_shot_energy_graph_cost, "
            "or prune_zero_then_energy_shot_graph_cost."
        )
    if bool(args.enable_prune_prefilter_profile_space) and args.force_phase1_prune_prefilter_json is None:
        raise ValueError("--enable-prune-prefilter-profile-space requires --force-phase1-prune-prefilter-json.")
    child_pool_force = str(args.force_adapt_child_pool_expansion_mode or "off").strip().lower()
    split_force = str(args.force_runtime_split_mode or "off").strip().lower()
    if child_pool_force not in {"", "off", "none", "false", "0", "disabled"} and split_force != "off":
        raise ValueError(
            "--force-adapt-child-pool-expansion-mode is a global pre-Phase-1 pool expansion and "
            "cannot be combined with --force-runtime-split-mode != off."
        )
    if objective_mode == "geo_dominance_first" and (
        args.dominance_target_abs_delta_e is None or args.dominance_target_iteration is None
    ):
        raise ValueError("--objective-mode geo_dominance_first requires --dominance-target-abs-delta-e and --dominance-target-iteration.")
    if objective_mode in {
        "geo_energy_then_graph_cost",
        "geo_energy_then_graph_shot_cost",
        "geo_energy_then_shot_graph_cost",
        "geo_energy_gate_then_shot_energy_graph_cost",
        "prune_zero_then_energy_shot_graph_cost",
    } and args.dominance_target_abs_delta_e is None:
        raise ValueError(f"--objective-mode {objective_mode} requires --dominance-target-abs-delta-e.")
    if objective_mode == "geo_energy_then_shot_graph_cost" and args.dominance_target_s_alg is None:
        raise ValueError("--objective-mode geo_energy_then_shot_graph_cost requires --dominance-target-s-alg.")
    lanes = [lane for lane in _parse_csv(args.lanes) if lane in {"canonical", "global", "legacy"}]
    if not lanes:
        raise ValueError("No valid lanes requested. Use canonical, global, and/or legacy.")
    epsilon_bands = _parse_float_csv(args.epsilon_bands)
    if not epsilon_bands:
        raise ValueError("At least one epsilon band is required.")
    output_dir = Path(args.output_dir) if args.output_dir is not None else REPO_ROOT / "artifacts/agent_runs" / str(args.tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_warm_start_preset_names = tuple(
        name for name in _parse_csv(args.extra_warm_start_presets) if name in _base_preset_library()
    )
    enqueue_preset_names = tuple(
        name for name in _parse_csv(args.enqueue_presets) if name in _base_preset_library()
    )
    restricted_base_preset_names = tuple(
        name for name in _parse_csv(args.restrict_base_presets) if name in _base_preset_library()
    )
    enqueue_param_rows = _enqueue_param_rows_from_json(args.enqueue_params_json, regime=args.enqueue_params_regime)
    for lane in lanes:
        searchable_presets = _searchable_presets_for_lane(lane, restricted_base_preset_names)
        invalid_enqueue = [name for name in enqueue_preset_names if name not in searchable_presets]
        if invalid_enqueue:
            raise ValueError(
                f"Enqueued presets are not searchable in lane {lane}: {invalid_enqueue}. Use extra warm-start presets for reference-only rows."
            )

    study_payloads: list[dict[str, Any]] = []
    for lane in lanes:
        for epsilon_abs_delta_e in epsilon_bands:
            payload = _run_single_study(
                optuna=optuna,
                python_bin=str(args.python_bin),
                study_spec=StudySpec(
                    lane=str(lane),
                    epsilon_abs_delta_e=float(epsilon_abs_delta_e),
                    n_trials=int(max(0, args.n_trials)),
                    n_startup_trials=int(max(1, args.n_startup_trials)),
                    extra_warm_start_preset_names=extra_warm_start_preset_names,
                    enqueue_preset_names=enqueue_preset_names,
                    restricted_base_preset_names=restricted_base_preset_names,
                ),
                output_dir=output_dir,
                compile_backend=str(args.compile_backend),
                compile_opt_level=int(args.compile_opt_level),
                compile_seed=int(args.compile_seed),
                use_default_warm_starts=not bool(args.no_default_warm_starts),
                hamiltonian_overrides=hamiltonian_overrides,
                objective_mode=objective_mode,
                compile_enabled=compile_enabled,
                energy_only_surface=bool(args.energy_only_surface),
                speed_surface_profile=str(args.speed_surface_profile),
                phase2_w_shot_profile_space=str(args.phase2_w_shot_profile_space),
                enable_prune_prefilter_profile_space=bool(args.enable_prune_prefilter_profile_space),
                dominance_target_abs_delta_e=args.dominance_target_abs_delta_e,
                dominance_target_iteration=args.dominance_target_iteration,
                dominance_target_graph_count_2q=args.dominance_target_graph_count_2q,
                dominance_target_graph_depth=args.dominance_target_graph_depth,
                dominance_target_s_alg=args.dominance_target_s_alg,
                phase2_w_shot_override=args.force_phase2_w_shot,
                runtime_split_mode_override=args.force_runtime_split_mode,
                child_pool_expansion_mode_override=args.force_adapt_child_pool_expansion_mode,
                child_pool_expansion_symmetry_policy_override=(
                    args.force_adapt_child_pool_expansion_symmetry_policy
                ),
                child_pool_expansion_max_subset_size_override=(
                    args.force_adapt_child_pool_expansion_max_subset_size
                ),
                exact_gs_override=args.adapt_exact_gs_override,
                exact_gs_reference_json=args.adapt_exact_gs_reference_json,
                force_adapt_benchmark_target_abs_delta_e=(
                    args.force_adapt_benchmark_target_abs_delta_e
                ),
                force_adapt_max_depth=args.force_adapt_max_depth,
                force_adapt_maxiter=args.force_adapt_maxiter,
                force_adapt_final_refit_maxiter=args.force_adapt_final_refit_maxiter,
                force_adapt_drop_floor=args.force_adapt_drop_floor,
                force_adapt_drop_patience=args.force_adapt_drop_patience,
                force_adapt_drop_min_depth=args.force_adapt_drop_min_depth,
                force_adapt_full_refit_every=args.force_adapt_full_refit_every,
                force_adapt_final_full_refit=args.force_adapt_final_full_refit,
                force_adapt_allow_repeats=args.force_adapt_allow_repeats,
                force_phase0_pilot_max_records=args.force_phase0_pilot_max_records,
                force_phase1_shortlist_size=args.force_phase1_shortlist_size,
                force_phase2_shortlist_fraction=args.force_phase2_shortlist_fraction,
                force_phase2_shortlist_size=args.force_phase2_shortlist_size,
                force_adapt_parallel_gradient_workers=args.force_adapt_parallel_gradient_workers,
                force_adapt_beam_parent_workers=args.force_adapt_beam_parent_workers,
                force_adapt_spsa_parallel_evaluations=args.force_adapt_spsa_parallel_evaluations,
                force_phase1_prune_prefilter_json=args.force_phase1_prune_prefilter_json,
                force_adapt_pool_class_filter_json=args.force_adapt_pool_class_filter_json,
                force_spsa_profile=args.force_spsa_profile,
                force_adapt_resume_scaffold_json=args.force_adapt_resume_scaffold_json,
                force_adapt_resume_mode=args.force_adapt_resume_mode,
                force_adapt_segment_id=args.force_adapt_segment_id,
                force_adapt_segment_target_depth=args.force_adapt_segment_target_depth,
                force_adapt_segment_max_new_admissions=args.force_adapt_segment_max_new_admissions,
                force_adapt_segment_wallclock_cap_s=args.force_adapt_segment_wallclock_cap_s,
                force_adapt_resume_compile_smoke=args.force_adapt_resume_compile_smoke,
                force_adapt_resume_smoke_backend=args.force_adapt_resume_smoke_backend,
                force_static_route_id=args.force_static_route_id,
                force_static_meta_feature_profile=args.force_static_meta_feature_profile,
                force_phase3_symmetry_mitigation_mode=args.force_phase3_symmetry_mitigation_mode,
                force_route_a_paper_i_production=bool(args.force_route_a_paper_i_production),
                force_phase1_prune_full_window=bool(
                    args.force_phase1_prune_full_window
                    or str(objective_mode) == "prune_zero_then_energy_shot_graph_cost"
                ),
                force_phase1_prune_recovery_trust_radius=(
                    args.force_phase1_prune_recovery_trust_radius
                ),
                force_phase1_prune_schur_nomination_route=(
                    args.force_phase1_prune_schur_nomination_route
                ),
                force_phase1_prune_metric_schur_mu=args.force_phase1_prune_metric_schur_mu,
                force_phase1_prune_metric_schur_solve_mode=(
                    args.force_phase1_prune_metric_schur_solve_mode
                ),
                force_phase1_prune_metric_schur_cost_weighting=(
                    args.force_phase1_prune_metric_schur_cost_weighting
                ),
                search_inner_optimizer=str(args.search_inner_optimizer),
                enqueue_param_rows=enqueue_param_rows,
                optuna_storage=args.optuna_storage,
                study_name_prefix=args.study_name_prefix,
                load_if_exists=bool(args.load_if_exists),
            )
            study_payloads.append(payload)

    lane_frontiers: dict[str, list[dict[str, Any]]] = {}
    for lane in lanes:
        lane_rows: list[dict[str, Any]] = []
        for payload in study_payloads:
            if str(payload.get("lane")) != str(lane):
                continue
            lane_rows.extend(list(payload.get("feasible_frontier", [])))
        lane_frontiers[str(lane)] = pareto_front(lane_rows, x_key="delta_e_abs", y_key="best_compiled_count_2q") if lane_rows else []

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": _PIPELINE_NAME,
        "tag": str(args.tag),
        "output_dir": str(output_dir),
        "compile_backend": str(args.compile_backend),
        "compile_opt_level": int(args.compile_opt_level),
        "compile_seed": int(args.compile_seed),
        "objective_mode": objective_mode,
        "optuna_storage": _normalize_optuna_storage(args.optuna_storage),
        "study_name_prefix": args.study_name_prefix,
        "load_if_exists": bool(args.load_if_exists),
        "compile_enabled": bool(compile_enabled),
        "energy_only_surface": bool(args.energy_only_surface),
        "search_inner_optimizer": str(args.search_inner_optimizer),
        "speed_surface_profile": str(args.speed_surface_profile),
        "phase2_w_shot_profile_space": str(args.phase2_w_shot_profile_space),
        "dominance_target_abs_delta_e": args.dominance_target_abs_delta_e,
        "dominance_target_iteration": args.dominance_target_iteration,
        "dominance_target_graph_count_2q": args.dominance_target_graph_count_2q,
        "dominance_target_graph_depth": args.dominance_target_graph_depth,
        "phase2_w_shot_override": args.force_phase2_w_shot,
        "runtime_split_mode_override": args.force_runtime_split_mode,
        "child_pool_expansion_mode_override": args.force_adapt_child_pool_expansion_mode,
        "child_pool_expansion_symmetry_policy_override": args.force_adapt_child_pool_expansion_symmetry_policy,
        "child_pool_expansion_max_subset_size_override": args.force_adapt_child_pool_expansion_max_subset_size,
        "force_adapt_benchmark_target_abs_delta_e": args.force_adapt_benchmark_target_abs_delta_e,
        "force_adapt_max_depth": args.force_adapt_max_depth,
        "force_adapt_maxiter": args.force_adapt_maxiter,
        "force_adapt_final_refit_maxiter": args.force_adapt_final_refit_maxiter,
        "force_adapt_full_refit_every": args.force_adapt_full_refit_every,
        "force_adapt_final_full_refit": args.force_adapt_final_full_refit,
        "force_adapt_allow_repeats": args.force_adapt_allow_repeats,
        "force_phase0_pilot_max_records": args.force_phase0_pilot_max_records,
        "force_phase1_shortlist_size": args.force_phase1_shortlist_size,
        "force_phase2_shortlist_fraction": args.force_phase2_shortlist_fraction,
        "force_phase2_shortlist_size": args.force_phase2_shortlist_size,
        "force_adapt_parallel_gradient_workers": args.force_adapt_parallel_gradient_workers,
        "force_adapt_beam_parent_workers": args.force_adapt_beam_parent_workers,
        "force_adapt_spsa_parallel_evaluations": args.force_adapt_spsa_parallel_evaluations,
        "force_spsa_profile": args.force_spsa_profile,
        "force_adapt_pool_class_filter_json": (
            None if args.force_adapt_pool_class_filter_json is None else str(Path(args.force_adapt_pool_class_filter_json))
        ),
        "force_adapt_resume_scaffold_json": (
            None if args.force_adapt_resume_scaffold_json is None else str(Path(args.force_adapt_resume_scaffold_json))
        ),
        "force_adapt_resume_mode": args.force_adapt_resume_mode,
        "force_adapt_segment_id": args.force_adapt_segment_id,
        "force_adapt_segment_target_depth": args.force_adapt_segment_target_depth,
        "force_adapt_segment_max_new_admissions": args.force_adapt_segment_max_new_admissions,
        "force_adapt_segment_wallclock_cap_s": args.force_adapt_segment_wallclock_cap_s,
        "force_adapt_resume_compile_smoke": args.force_adapt_resume_compile_smoke,
        "force_adapt_resume_smoke_backend": args.force_adapt_resume_smoke_backend,
        "force_static_route_id": args.force_static_route_id,
        "force_static_meta_feature_profile": args.force_static_meta_feature_profile,
        "force_phase3_symmetry_mitigation_mode": args.force_phase3_symmetry_mitigation_mode,
        "force_route_a_paper_i_production": bool(args.force_route_a_paper_i_production),
        "force_phase1_prune_full_window": bool(args.force_phase1_prune_full_window),
        "force_phase1_prune_recovery_trust_radius": args.force_phase1_prune_recovery_trust_radius,
        "force_phase1_prune_schur_nomination_route": args.force_phase1_prune_schur_nomination_route,
        "force_phase1_prune_metric_schur_mu": args.force_phase1_prune_metric_schur_mu,
        "force_phase1_prune_metric_schur_solve_mode": args.force_phase1_prune_metric_schur_solve_mode,
        "force_phase1_prune_metric_schur_cost_weighting": args.force_phase1_prune_metric_schur_cost_weighting,
        "enqueue_params_json": (None if args.enqueue_params_json is None else str(Path(args.enqueue_params_json))),
        "enqueue_params_regime": args.enqueue_params_regime,
        "enqueue_param_row_count": int(len(enqueue_param_rows)),
        "hamiltonian_overrides": _hh_hamiltonian_override_payload(hamiltonian_overrides),
        "studies": study_payloads,
        "lane_frontiers": lane_frontiers,
        "preset_library": {name: asdict(preset) for name, preset in _base_preset_library().items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_progress(output_dir, {"generated_utc": datetime.now(timezone.utc).isoformat(), "pipeline": _PIPELINE_NAME, "done": True, "study_count": len(study_payloads)})
    print(f"Wrote summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
