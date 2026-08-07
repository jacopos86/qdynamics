#!/usr/bin/env python3
"""Neutral exact-benchmark contracts and static runner support.

Extracted from the retired calibration harness. This module owns only the
retained generic benchmark interface; it is not a study, sampler, route
registry, or execution-authority surface.
"""
from __future__ import annotations

import json

import math

import os

import signal

import shlex

import subprocess

import sys

import threading

import time

from dataclasses import asdict, dataclass, field, fields, replace

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

from pipelines.exact_bench.noise_oracle_defaults import gate_tuple_to_cli_value, normalize_gate_name_tuple

from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec

from pipelines.static_adapt.builders.hh_pool_presets import HH_MATH_MD_FULL_META_POOL_KEY

from pipelines.static_adapt.builders.problem_registry import get_problem_family_spec

from pipelines.static_adapt.output_artifacts import extract_adapt_energy_metrics

from pipelines.static_adapt.plateau_acquisition import (
    PLATEAU_ACQUISITION_SCORE_CHOICES,
    PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
    PLATEAU_ACQUISITION_MODE_OFF,
    PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
    PLATEAU_TRIAL_OPTIMIZER_CHOICES,
    PLATEAU_TRIAL_OPTIMIZER_INHERIT,
    normalize_plateau_acquisition_config,
)

from pipelines.static_adapt.selector_measurement_proxy import (
    controller_proxy_from_adapt_payload,
    validate_controller_proxy_for_shot_objective,
)

from pipelines.static_adapt.runtime_heartbeat import (
    LiveHeartbeatRecorder,
    normalize_ai_log_progress,
    parse_ai_log_line,
)

from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site

from src.quantum.chemistry.psi4_adapter import load_restricted_closed_shell_problem_from_json

_DEFAULT_MOLECULAR_PROBLEM_JSON = REPO_ROOT / "test_support" / "molecular_problem_h2_sto3g.json"

_DEFAULT_LIH_MOLECULAR_PROBLEM_JSON = REPO_ROOT / "test_support" / "molecular_problem_lih_sto3g.json"

_DEFAULT_H2O_MOLECULAR_PROBLEM_JSON = REPO_ROOT / "src" / "quantum" / "chemistry" / "h2o_sto3g_fast_result.json"

_MOLECULAR_ROLE_CONTROL = "control"

_MOLECULAR_ROLE_TRAIN_PRIMARY = "train_primary"

_MOLECULAR_ROLE_TRANSFER = "transfer"

_MOLECULAR_ROLE_OVERRIDE_TRAIN = "override_train"

_MOLECULAR_ROLE_TAG_PREFIX = "molecular_role:"

_MOLECULAR_DENSE_EIGH_MAX_DIM = 1024

@dataclass(frozen=True)
class _DefaultMolecularBenchmarkAsset:
    label: str | None
    path: Path
    role: str
    tag: str

_DEFAULT_MOLECULAR_BENCHMARK_ASSETS: tuple[_DefaultMolecularBenchmarkAsset, ...] = (
    _DefaultMolecularBenchmarkAsset(
        label=None,
        path=_DEFAULT_MOLECULAR_PROBLEM_JSON,
        role=_MOLECULAR_ROLE_CONTROL,
        tag="h2_sto3g",
    ),
    _DefaultMolecularBenchmarkAsset(
        label="lih_sto3g",
        path=_DEFAULT_LIH_MOLECULAR_PROBLEM_JSON,
        role=_MOLECULAR_ROLE_TRAIN_PRIMARY,
        tag="lih_sto3g",
    ),
    _DefaultMolecularBenchmarkAsset(
        label="h2o_sto3g",
        path=_DEFAULT_H2O_MOLECULAR_PROBLEM_JSON,
        role=_MOLECULAR_ROLE_TRANSFER,
        tag="h2o_sto3g",
    ),
)

_HEAVY_BOSONIC_CUTOFF2_FAMILIES = frozenset({"hh", "bose_hubbard", "harmonic_kerr_chain", "spin_boson"})

_BOSON_ILLEGAL_PROBABILITY_FAIL_THRESHOLD = 1e-4

_SPSA_SCHEDULE_INNER_OPTIMIZERS = frozenset({"SPSA", "QNSPSA"})

_DEFAULT_INNER_OPTIMIZER = "SPSA"

_PHASE3_BATCH_SELECTION_MODE_CHOICES = (
    "reduced_plane",
    "greedy_reduced_plane",
    "combinatorial_reduced_plane",
    "overlap_orthogonal_benchmark",
    "ceo_commuting_benchmark",
)

_PHASE3_BATCH_PREFILTER_MODE_CHOICES = (
    "off",
    "exact_support_disjoint",
    "exact_commuting",
    "exact_support_disjoint_and_commuting",
)

_DEFAULT_PHASE1_PRUNE_POLICY = "recoverability_ladder_v1"

def _default_boson_cutoff_for_family(family: str) -> int:
    """Default local boson/phonon cutoff for benchmark-suite construction."""

    return 2 if str(family).strip().lower() in _HEAVY_BOSONIC_CUTOFF2_FAMILIES else 1

def _resolve_default_molecular_benchmark_assets(
    path: str | Path | None,
) -> tuple[_DefaultMolecularBenchmarkAsset, ...]:
    if path not in {None, ""}:
        return (
            _DefaultMolecularBenchmarkAsset(
                label=None,
                path=Path(str(path)),
                role=_MOLECULAR_ROLE_OVERRIDE_TRAIN,
                tag="molecular_override",
            ),
        )
    return tuple(asset for asset in _DEFAULT_MOLECULAR_BENCHMARK_ASSETS if asset.path.exists())

def _molecular_problem_dimensions(path: Path) -> tuple[int, int]:
    problem = load_restricted_closed_shell_problem_from_json(Path(path))
    n_spatial = int(problem.n_spatial_orbitals)
    n_spin = int(problem.n_spin_orbitals)
    if n_spatial <= 0 or n_spin <= 0:
        raise ValueError(f"Molecular problem JSON {path} is missing positive orbital dimensions.")
    return n_spatial, n_spin

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

def _set_toggle_pair(args: Sequence[str], positive_flag: str, negative_flag: str, enabled: bool) -> list[str]:
    updated = _remove_option(_remove_option(args, positive_flag), negative_flag)
    return [*updated, positive_flag if enabled else negative_flag]

@dataclass(frozen=True)
class ProblemFeatureVector:
    """Transferable problem features used by size-aware policy rules."""

    problem: str
    size_label: str
    L: int
    n_qubits: int
    pool_size_hint: int
    hamiltonian_term_count_hint: int = 0
    spinful: bool = False
    bosonic: bool = False
    molecular: bool = False
    max_graph_degree_hint: int = 2

@dataclass(frozen=True)
class SizeScaledBudget:
    """N = max(min_count, ceil(pool_fraction*|G|), ceil(qubit_slope*nq)), capped by max_count."""

    min_count: int
    max_count: int
    pool_fraction: float
    qubit_slope: float

    def resolve(self, features: ProblemFeatureVector) -> int:
        raw = max(
            int(self.min_count),
            int(math.ceil(float(self.pool_fraction) * max(1, int(features.pool_size_hint)))),
            int(math.ceil(float(self.qubit_slope) * max(1, int(features.n_qubits)))),
        )
        return int(min(max(raw, int(self.min_count)), int(self.max_count)))

@dataclass(frozen=True)
class PoolPolicy:
    """Transferable pool and shortlist budgets for a generic benchmark."""

    pool_key: str = "full_meta"
    family_prior_hopping: float = 1.0
    family_prior_onsite: float = 1.0
    family_prior_density: float = 1.0
    family_prior_quadrature: float = 1.0
    family_prior_assisted: float = 1.0
    family_prior_bridge: float = 1.0
    family_prior_boson: float = 1.0
    family_repeat_penalty: float = 1.0
    phase1_budget: SizeScaledBudget = field(default_factory=lambda: SizeScaledBudget(24, 256, 0.35, 8.0))
    phase2_budget: SizeScaledBudget = field(default_factory=lambda: SizeScaledBudget(12, 128, 0.25, 6.0))
    rescue_expand_factor: float = 2.0

@dataclass(frozen=True)
class StaticScaffoldPolicy:
    """Retained generic static-benchmark execution policy."""

    lambda_compile: float = 0.05
    lambda_measure: float = 0.02
    lambda_leak: float = 0.0
    lambda_2q: float = 0.14
    lambda_d: float = 0.14
    lambda_1q: float = 0.14
    lambda_theta: float = 0.14
    lambda_shot: float = 0.14
    suppress_explicit_hardware_lambdas: bool = False
    compile_cx_weight: float = 1.0
    compile_sq_weight: float = 0.5
    compile_rotation_step_weight: float = 1.0
    compile_position_shift_weight: float = 1.0
    compile_refit_active_weight: float = 1.0
    measure_groups_weight: float = 1.0
    measure_shots_weight: float = 1.0
    measure_reuse_weight: float = 1.0
    opt_dim_cost_scale: float = 1.0
    phase2_w_depth: float = 0.2
    phase2_w_group: float = 0.15
    phase2_w_shot: float = 0.15
    phase2_w_optdim: float = 0.1
    phase2_w_reuse: float = 0.1
    phase2_w_lifetime: float = 0.05
    phase2_leakage_cap: float = 1e6
    phase2_frontier_ratio: float = 1.0
    phase3_frontier_ratio: float = 1.0
    phase3_tie_beam_score_ratio: float = 1.05
    phase3_tie_beam_abs_tol: float = 1e-6
    phase3_tie_beam_max_branches: int = 3
    phase3_tie_beam_max_late_coordinate: float = 1.0
    phase3_tie_beam_min_depth_left: int = 1
    adapt_beam_live_branches: int = 5
    adapt_beam_children_per_parent: int = 4
    adapt_beam_terminated_keep: int = 3
    adapt_beam_lambda: float = 0.0
    adapt_reopt_policy: str = "windowed"
    adapt_window_size: int = 128
    adapt_window_topk: int = 64
    adapt_full_refit_every: int = 1
    adapt_final_full_refit: bool = True
    adapt_final_refit_maxiter: int = 0
    adapt_insertion_mode: str = "adaptive"
    adapt_allow_repeats: bool = False
    adapt_allow_repeats_override: bool | None = None
    phase1_probe_max_positions: int = 6
    phase2_shortlist_fraction: float = 0.25
    phase2_motif_bonus_weight: float | None = None
    phase2_rho: float = 0.25
    phase1_score_mode: str = "trust_region_v1"
    phase1_score_z_alpha: float | None = None
    phase2_score_z_alpha: float | None = None
    phase2_enable_batching: bool = True
    phase2_batch_target_size: int = 8
    phase2_batch_size_cap: int = 16
    phase2_batch_near_degenerate_ratio: float = 0.98
    phase2_batch_rank_rel_tol: float = 1e-6
    phase2_batch_additivity_tol: float = 0.25
    phase1_prune_enabled: bool = True
    phase1_prune_policy: str = _DEFAULT_PHASE1_PRUNE_POLICY
    phase1_prune_mode: str = "both"
    phase1_prune_fraction: float = 0.25
    phase1_prune_min_candidates: int = 1
    phase1_prune_max_candidates: int = 6
    phase1_prune_max_regression: float = 1e-8
    phase1_prune_tolerance_mode: str = "auto"
    phase1_prune_tolerance_shot_coeff: float = 0.0
    phase1_prune_tolerance_screen_coeff: float = 0.01
    phase1_prune_tolerance_chem: float = 0.0
    phase1_prune_tolerance_rel_coeff: float = 0.0
    phase1_prune_retained_gain_ratio: float = 0.25
    phase1_prune_protect_steps: int = 1
    phase1_prune_cooldown_steps: int = 2
    phase1_prune_local_window_size: int = 4
    phase1_prune_recovery_trust_radius: float = 0.0
    phase1_prune_old_fraction: float = 0.25
    phase1_prune_checkpoint_period: int = 3
    phase1_prune_live_min_depth: int = 0
    phase1_prune_maturity_threshold: float = 0.5
    phase1_prune_snr_threshold: float = 1.0
    phase1_maturity_cap_min_fraction: float = 0.35
    phase1_maturity_cap_max_fraction: float = 1.0
    phase2_maturity_cap_min_fraction: float = 0.35
    phase2_maturity_cap_max_fraction: float = 1.0
    phase3_maturity_cap_min_fraction: float = 0.25
    phase3_maturity_cap_max_fraction: float = 0.75
    phase_maturity_shot_min: int = 1
    phase_maturity_shot_max: int = 1
    phase1_maturity_shot_cap: int = 0
    phase2_maturity_shot_cap: int = 0
    phase3_maturity_shot_cap: int = 0
    adapt_max_depth: int = 96
    adapt_maxiter: int = 4000
    adapt_drop_floor: float = 1e-8
    adapt_drop_patience: int = 12
    adapt_drop_min_depth: int = 16
    adapt_eps_grad: float = 1e-9
    adapt_eps_energy: float = 1e-13
    adapt_parallel_gradient_workers: int = 1
    adapt_beam_parent_workers: int = 1
    phase3_plateau_acquisition_mode: str = PLATEAU_ACQUISITION_MODE_OFF
    phase3_plateau_acquisition_score: str = PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1
    phase3_plateau_unlock_margin: float = 1e-8
    phase3_plateau_duplicate_policy: str = PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1
    phase3_plateau_lambda_vol: float = 1e-8
    phase3_plateau_sigma_min: float = 0.0
    phase3_plateau_nu_min: float = 0.0
    phase3_plateau_volume_min: float = 0.0
    phase3_plateau_failed_family_patience: int = 0
    phase3_plateau_trial_optimizer: str = PLATEAU_TRIAL_OPTIMIZER_INHERIT
    phase3_plateau_trial_qngd_maxiter: int = 64
    phase3_batch_selection_mode: str = "reduced_plane"
    phase3_batch_prefilter_mode: str = "off"
    phase3_backend_cost_mode: str = "auto"
    phase3_hardware_cost_normalization_mode: str | None = None
    phase3_source_lock_preferred_sequence: str = ""
    phase3_runtime_split_mode: str = "off"
    allow_archival_phase3_runtime_split: bool = False
    phase3_runtime_split_selection_mode: str = "proxy_child_set_preselection"
    phase3_runtime_split_max_subset_size: int = 3
    shared_pauli_pool_mode: str = "off"
    shared_pauli_pool_symmetry_policy: str = "off"
    shared_pauli_pool_max_subset_size: int = 3
    static_meta_feature_profile: str = "off"
    static_route_id: str = "unspecified"
    hardware_resolution_mode: str = "ideal"
    hardware_resolution_profile_json: str | None = None
    hardware_resolution_profile_name: str | None = None
    phase3_oracle_gradient_mode: str | None = None
    phase3_oracle_backend_name: str | None = None
    phase3_oracle_use_fake_backend: bool | None = None
    phase3_oracle_shots: int | None = None
    phase3_oracle_repeats: int | None = None
    phase3_oracle_aggregate: str | None = None
    phase3_oracle_seed: int | None = None
    phase3_oracle_gradient_step: float | None = None
    phase3_oracle_execution_surface: str | None = None
    phase3_oracle_inner_objective_mode: str | None = None
    phase3_oracle_value_noise_model: str | None = None
    phase3_oracle_value_noise_std: float | None = None
    phase3_oracle_value_noise_seed: int | None = None
    phase3_oracle_value_noise_sigma0_abs: float | None = None
    phase3_oracle_value_noise_n_eff: float | None = None
    phase3_oracle_value_noise_seed_policy: str | None = None
    phase3_oracle_value_noise_base_seed: int | None = None
    phase3_oracle_value_noise_replicate_id: str | None = None
    phase3_oracle_synthetic_depolarizing_1q_error: float | None = None
    phase3_oracle_synthetic_depolarizing_2q_error: float | None = None
    phase3_oracle_synthetic_depolarizing_1q_gates: str | None = None
    phase3_oracle_synthetic_depolarizing_2q_gates: str | None = None
    phase3_oracle_synthetic_coherent_1q_angle_std: float | None = None
    phase3_oracle_synthetic_coherent_2q_angle_std: float | None = None
    phase3_oracle_synthetic_coherent_seed: int | None = None
    phase3_oracle_synthetic_coherent_generator_mode: str | None = None
    phase3_oracle_synthetic_coherent_1q_gates: str | None = None
    phase3_oracle_synthetic_coherent_2q_gates: str | None = None
    adapt_noise_floor_stop_policy: str | None = None
    adapt_noise_floor_snr_threshold: float | None = None
    adapt_noise_floor_n_rem_high_threshold: float | None = None
    adapt_noise_floor_useful_horizon_threshold: float | None = None

@dataclass(frozen=True)
class InnerOptimizerPolicy:
    inner_optimizer: str = _DEFAULT_INNER_OPTIMIZER
    final_optimizer_type: str = _DEFAULT_INNER_OPTIMIZER
    spsa_a: float = 0.1
    spsa_c: float = 0.1
    spsa_A: float = 10.0
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    refit_maxiter: int = 4000
    final_maxiter: int = 4000
    grad_tol: float = 1e-9
    energy_tol: float = 1e-13

@dataclass(frozen=True)
class AlgorithmPolicy:
    pool: PoolPolicy = field(default_factory=PoolPolicy)
    static: StaticScaffoldPolicy = field(default_factory=StaticScaffoldPolicy)
    inner_optimizer: InnerOptimizerPolicy = field(default_factory=InnerOptimizerPolicy)

    @classmethod
    def default(cls) -> "AlgorithmPolicy":
        return cls()

def _normalize_fixed_inner_optimizer(value: str | None = None) -> str:
    """Normalize the optimizer selected by a static benchmark policy."""

    raw = _DEFAULT_INNER_OPTIMIZER if value in {None, ""} else str(value).strip().upper()
    if not raw:
        raise ValueError("Static benchmark inner optimizer must be non-empty.")
    return raw

def _is_spsa_optimizer(value: str | None) -> bool:
    """Return whether the optimizer consumes the ADAPT SPSA-schedule knobs."""

    return str(value or "").strip().upper() in _SPSA_SCHEDULE_INNER_OPTIMIZERS

def _normalize_retained_policy(policy: AlgorithmPolicy) -> AlgorithmPolicy:
    """Normalize only fields retained by the generic static benchmark runner."""

    inner_name = _normalize_fixed_inner_optimizer(policy.inner_optimizer.inner_optimizer)
    inner = replace(
        policy.inner_optimizer,
        inner_optimizer=inner_name,
        final_optimizer_type=str(
            policy.inner_optimizer.final_optimizer_type or inner_name
        ).strip().upper(),
    )
    static = policy.static
    batch_cap = max(1, int(static.phase2_batch_size_cap))
    batch_target = min(max(1, int(static.phase2_batch_target_size)), batch_cap)
    prune_max = max(1, int(static.phase1_prune_max_candidates))
    prune_min = min(max(1, int(static.phase1_prune_min_candidates)), prune_max)
    route_id = str(getattr(static, "static_route_id", "unspecified") or "unspecified").strip().lower()
    if route_id not in {"route_a", "unspecified"}:
        raise ValueError(
            "Static benchmark route identity must be 'route_a' or 'unspecified'; "
            f"received {route_id!r}."
        )
    meta_profile = str(
        getattr(static, "static_meta_feature_profile", "off") or "off"
    ).strip().lower()
    if meta_profile not in {"off", "paper_i_production_v1", "safe_core_v1"}:
        raise ValueError(f"Unknown static benchmark feature profile {meta_profile!r}.")
    static = replace(
        static,
        static_route_id=route_id,
        static_meta_feature_profile=meta_profile,
        phase2_batch_size_cap=batch_cap,
        phase2_batch_target_size=batch_target,
        phase1_prune_min_candidates=prune_min,
        phase1_prune_max_candidates=prune_max,
    )
    if inner == policy.inner_optimizer and static == policy.static:
        return policy
    return replace(policy, static=static, inner_optimizer=inner)

@dataclass(frozen=True)
class HamiltonianBenchmarkSpec:
    benchmark_id: str
    family: str
    features: ProblemFeatureVector
    base_pipeline_args: tuple[str, ...]
    baseline_abs_delta_e: float
    baseline_count_2q: int | None = None
    baseline_depth_2q: int | None = None
    baseline_parameter_count: int | None = None
    baseline_shot_cost_proxy: float | None = None
    baseline_artifact_json: str | None = None
    baseline_source: str | None = None
    exact_reference_n_ph_max: int | None = None
    split: str = "train"
    tags: tuple[str, ...] = ()
    selected_logical_route: str = "standard"
    selected_logical_source_json: str | None = None
    selected_logical_source_kind: str | None = None
    selected_logical_source_record_count: int = 0
    selected_logical_transfer_mode: str = "exact_match_v1"

@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_id: str
    family: str
    success: bool
    abs_delta_e: float | None
    energy: float | None = None
    exact_gs_energy: float | None = None
    same_cutoff_exact_gs_energy: float | None = None
    exact_reference_energy: float | None = None
    exact_reference_n_ph_max: int | None = None
    abs_delta_e_same_cutoff: float | None = None
    abs_delta_e_reference: float | None = None
    cutoff_abs_delta_e: float | None = None
    count_2q: int | None = None
    depth_2q: int | None = None
    circuit_depth: int | None = None
    parameter_count: int | None = None
    runtime_parameter_count: int | None = None
    measurement_groups_proxy: float | None = None
    measurement_shots_proxy: float | None = None
    shot_cost_proxy: float | None = None
    walltime_s: float | None = None
    failure_reason: str | None = None
    stop_reason: str | None = None
    ansatz_depth: int | None = None
    initial_energy: float | None = None
    initial_abs_delta_e: float | None = None
    max_gradient_seen: float | None = None
    quality_gate_reason: str | None = None
    boson_illegal_probability_max: float | None = None
    boson_legal_probability_min: float | None = None
    measurement_proxy_validated: bool = False
    measurement_proxy_validation: Mapping[str, Any] = field(default_factory=dict)
    policy_roundtrip_audit: Mapping[str, Any] | None = None
    policy_roundtrip_audit_json: str | None = None
    physical_target_manifest: Mapping[str, Any] | None = None
    cutoff_diagnostics: Mapping[str, Any] | None = None
    paper_i_first_crossing: Mapping[str, Any] | None = None
    target_hit_classification: Mapping[str, Any] | None = None
    result_json: str | None = None
    compile_json: str | None = None
    policy_json: str | None = None

def _fractional_cap_pair(size: int, min_fraction: float, max_fraction: float) -> tuple[int, int]:
    size_val = max(1, int(size))
    lo = _clamp(float(min_fraction), 0.0, 1.0)
    hi = _clamp(float(max_fraction), 0.0, 1.0)
    if lo > hi:
        lo, hi = hi, lo
    min_cap = max(1, int(math.ceil(size_val * lo)))
    max_cap = max(min_cap, int(math.ceil(size_val * hi)))
    return int(min_cap), int(max_cap)

def _static_hardware_resolution_mode(value: Any) -> str:
    return str("ideal" if value in {None, ""} else value).strip().lower() or "ideal"

def _static_hardware_profile_value(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value).strip()
    return text if text else None

def _validate_static_hardware_resolution_policy(static: StaticScaffoldPolicy) -> tuple[str, str | None, str | None]:
    mode = _static_hardware_resolution_mode(static.hardware_resolution_mode)
    profile_json = _static_hardware_profile_value(static.hardware_resolution_profile_json)
    profile_name = _static_hardware_profile_value(static.hardware_resolution_profile_name)
    if mode not in {"ideal", "profile"}:
        raise ValueError("hardware_resolution_mode policy overlay must be one of {'ideal','profile'}.")
    if mode == "ideal":
        if profile_json is not None or profile_name is not None:
            raise ValueError("hardware_resolution profile JSON/name requires hardware_resolution_mode='profile'.")
        return mode, None, None
    if profile_json is None or profile_name is None:
        raise ValueError(
            "hardware_resolution_mode='profile' requires both "
            "hardware_resolution_profile_json and hardware_resolution_profile_name."
        )
    return mode, profile_json, profile_name

def _validate_static_route_policy(static: StaticScaffoldPolicy, *, hardware_resolution_mode: str) -> str:
    route_id = str(
        getattr(static, "static_route_id", "unspecified") or "unspecified"
    ).strip().lower()
    if route_id not in {"route_a", "unspecified"}:
        raise ValueError(f"Unknown static benchmark route identity {route_id!r}.")
    if route_id == "route_a" and str(hardware_resolution_mode).strip().lower() != "ideal":
        raise ValueError(
            "static_route_id='route_a' requires hardware_resolution_mode='ideal'; "
            "use static_route_id='unspecified' for diagnostic hardware-resolution profile smoke."
        )
    return route_id

_PHASE3_ORACLE_CLI_OPTIONS: tuple[tuple[str, str], ...] = (
    ("phase3_oracle_gradient_mode", "--phase3-oracle-gradient-mode"),
    ("phase3_oracle_shots", "--phase3-oracle-shots"),
    ("phase3_oracle_repeats", "--phase3-oracle-repeats"),
    ("phase3_oracle_aggregate", "--phase3-oracle-aggregate"),
    ("phase3_oracle_backend_name", "--phase3-oracle-backend-name"),
    ("phase3_oracle_seed", "--phase3-oracle-seed"),
    ("phase3_oracle_gradient_step", "--phase3-oracle-gradient-step"),
    ("phase3_oracle_value_noise_model", "--phase3-oracle-value-noise-model"),
    ("phase3_oracle_value_noise_std", "--phase3-oracle-value-noise-std"),
    ("phase3_oracle_value_noise_seed", "--phase3-oracle-value-noise-seed"),
    ("phase3_oracle_value_noise_sigma0_abs", "--phase3-oracle-value-noise-sigma0-abs"),
    ("phase3_oracle_value_noise_n_eff", "--phase3-oracle-value-noise-n-eff"),
    ("phase3_oracle_synthetic_depolarizing_1q_error", "--phase3-oracle-synthetic-depolarizing-1q-error"),
    ("phase3_oracle_synthetic_depolarizing_2q_error", "--phase3-oracle-synthetic-depolarizing-2q-error"),
    ("phase3_oracle_synthetic_depolarizing_1q_gates", "--phase3-oracle-synthetic-depolarizing-1q-gates"),
    ("phase3_oracle_synthetic_depolarizing_2q_gates", "--phase3-oracle-synthetic-depolarizing-2q-gates"),
    ("phase3_oracle_synthetic_coherent_1q_angle_std", "--phase3-oracle-synthetic-coherent-1q-angle-std"),
    ("phase3_oracle_synthetic_coherent_2q_angle_std", "--phase3-oracle-synthetic-coherent-2q-angle-std"),
    ("phase3_oracle_synthetic_coherent_seed", "--phase3-oracle-synthetic-coherent-seed"),
    ("phase3_oracle_synthetic_coherent_generator_mode", "--phase3-oracle-synthetic-coherent-generator-mode"),
    ("phase3_oracle_synthetic_coherent_1q_gates", "--phase3-oracle-synthetic-coherent-1q-gates"),
    ("phase3_oracle_synthetic_coherent_2q_gates", "--phase3-oracle-synthetic-coherent-2q-gates"),
    ("phase3_oracle_execution_surface", "--phase3-oracle-execution-surface"),
    ("phase3_oracle_inner_objective_mode", "--phase3-oracle-inner-objective-mode"),
)

_PHASE3_ORACLE_VALUELESS_OPTIONS: tuple[tuple[str, str], ...] = (
    ("phase3_oracle_use_fake_backend", "--phase3-oracle-use-fake-backend"),
)

_PHASE3_ORACLE_CLI_FLAGS: tuple[str, ...] = tuple(flag for _, flag in _PHASE3_ORACLE_CLI_OPTIONS) + tuple(
    flag for _, flag in _PHASE3_ORACLE_VALUELESS_OPTIONS
)

_PHASE3_ORACLE_GATE_LIST_FIELDS = frozenset(
    {
        "phase3_oracle_synthetic_depolarizing_1q_gates",
        "phase3_oracle_synthetic_depolarizing_2q_gates",
        "phase3_oracle_synthetic_coherent_1q_gates",
        "phase3_oracle_synthetic_coherent_2q_gates",
    }
)

def _canonical_phase3_oracle_gate_cli_value(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and str(value).strip().lower() in {"", "none"}:
        return None
    gates = normalize_gate_name_tuple(value, default=None, field_name=field_name)
    if not gates:
        return None
    return gate_tuple_to_cli_value(gates, field_name=field_name)

def _static_phase3_oracle_overlay_requested(static: StaticScaffoldPolicy) -> bool:
    for field_name, _ in _PHASE3_ORACLE_CLI_OPTIONS:
        if getattr(static, field_name) is not None:
            return True
    for field_name, _ in _PHASE3_ORACLE_VALUELESS_OPTIONS:
        if getattr(static, field_name) is not None:
            return True
    return False

def _append_static_phase3_oracle_cli_args(args: list[str], static: StaticScaffoldPolicy) -> list[str]:
    out = list(args)
    for field_name, flag in _PHASE3_ORACLE_CLI_OPTIONS:
        value = getattr(static, field_name)
        if value is not None:
            if field_name in _PHASE3_ORACLE_GATE_LIST_FIELDS:
                value = _canonical_phase3_oracle_gate_cli_value(value, field_name=field_name)
            out = _set_option(out, flag, value)
    for field_name, flag in _PHASE3_ORACLE_VALUELESS_OPTIONS:
        value = getattr(static, field_name)
        if value is True:
            out = _remove_option(out, flag)
            out.append(flag)
    return out

def policy_to_cli_args(policy: AlgorithmPolicy, spec: HamiltonianBenchmarkSpec) -> list[str]:
    """Return CLI override tokens for canonical ``static_adapt.adapt_pipeline``."""

    policy = _normalize_retained_policy(policy)
    features = spec.features
    pool = policy.pool
    static = policy.static
    inner = policy.inner_optimizer
    family_spec = get_problem_family_spec(spec.family)
    pool_key = str(pool.pool_key)
    if pool_key not in set(family_spec.admissible_pool_keys):
        pool_key = "full_meta" if "full_meta" in set(family_spec.admissible_pool_keys) else str(family_spec.default_pool_key)
    if (
        str(spec.family).strip().lower() == "hh"
        and pool_key in {"full_meta", "math_md_full_meta", HH_MATH_MD_FULL_META_POOL_KEY}
        and HH_MATH_MD_FULL_META_POOL_KEY in set(family_spec.admissible_pool_keys)
    ):
        pool_key = HH_MATH_MD_FULL_META_POOL_KEY
    selected_route = str(getattr(spec, "selected_logical_route", "standard") or "standard").strip().lower().replace("-", "_")
    selected_source = getattr(spec, "selected_logical_source_json", None)
    if selected_route != "standard" and static.static_route_id == "route_a":
        static = replace(static, static_route_id="unspecified")
    if selected_route == "historical_selected" and selected_source not in {None, ""}:
        # Historical-selected means: start from the problem-generic mega pool,
        # then filter by the historical operator-family closure. A narrower
        # base pool can exclude that family before the filter is applied.
        if "full_meta" in set(family_spec.admissible_pool_keys):
            pool_key = "full_meta"
    phase1_shortlist = pool.phase1_budget.resolve(features)
    phase2_shortlist = pool.phase2_budget.resolve(features)
    phase1_cap_min, phase1_cap_max = _fractional_cap_pair(
        phase1_shortlist,
        static.phase1_maturity_cap_min_fraction,
        static.phase1_maturity_cap_max_fraction,
    )
    phase2_cap_min, phase2_cap_max = _fractional_cap_pair(
        phase2_shortlist,
        static.phase2_maturity_cap_min_fraction,
        static.phase2_maturity_cap_max_fraction,
    )
    phase3_cap_min, phase3_cap_max = _fractional_cap_pair(
        phase2_shortlist,
        static.phase3_maturity_cap_min_fraction,
        static.phase3_maturity_cap_max_fraction,
    )
    shot_min = max(1, int(static.phase_maturity_shot_min))
    shot_max = max(shot_min, int(static.phase_maturity_shot_max))

    args: list[str] = []
    args = _set_option(args, "--adapt-pool", pool_key)
    selected_transfer = str(getattr(spec, "selected_logical_transfer_mode", "exact_match_v1") or "exact_match_v1")
    if selected_route == "historical_selected" and selected_source not in {None, ""}:
        args = _set_option(args, "--adapt-selected-logical-source-json", str(selected_source))
        args = _set_option(args, "--adapt-selected-logical-mode", "family_closure_with_full_fallback")
        args = _set_option(args, "--adapt-selected-logical-transfer-mode", selected_transfer)
    else:
        args = _set_option(args, "--adapt-selected-logical-mode", "off")
        args = _set_option(args, "--adapt-selected-logical-transfer-mode", "exact_match_v1")
    args = _set_option(args, "--adapt-continuation-mode", "phase3_v1")
    args = _set_option(args, "--adapt-inner-optimizer", inner.inner_optimizer)
    if _is_spsa_optimizer(inner.inner_optimizer):
        args = _set_option(args, "--adapt-spsa-a", inner.spsa_a)
        args = _set_option(args, "--adapt-spsa-c", inner.spsa_c)
        args = _set_option(args, "--adapt-spsa-A", inner.spsa_A)
        args = _set_option(args, "--adapt-spsa-alpha", inner.spsa_alpha)
        args = _set_option(args, "--adapt-spsa-gamma", inner.spsa_gamma)
    args = _set_option(args, "--adapt-max-depth", static.adapt_max_depth)
    args = _set_option(args, "--adapt-maxiter", static.adapt_maxiter)
    args = _set_option(args, "--adapt-drop-floor", static.adapt_drop_floor)
    args = _set_option(args, "--adapt-drop-patience", static.adapt_drop_patience)
    args = _set_option(args, "--adapt-drop-min-depth", static.adapt_drop_min_depth)
    if static.adapt_noise_floor_stop_policy is not None:
        args = _set_option(args, "--adapt-noise-floor-stop-policy", static.adapt_noise_floor_stop_policy)
    if static.adapt_noise_floor_snr_threshold is not None:
        args = _set_option(args, "--adapt-noise-floor-snr-threshold", static.adapt_noise_floor_snr_threshold)
    if static.adapt_noise_floor_n_rem_high_threshold is not None:
        args = _set_option(
            args,
            "--adapt-noise-floor-n-rem-high-threshold",
            static.adapt_noise_floor_n_rem_high_threshold,
        )
    if static.adapt_noise_floor_useful_horizon_threshold is not None:
        args = _set_option(
            args,
            "--adapt-noise-floor-useful-horizon-threshold",
            static.adapt_noise_floor_useful_horizon_threshold,
        )
    args = _set_option(args, "--adapt-eps-grad", static.adapt_eps_grad)
    args = _set_option(args, "--adapt-eps-energy", static.adapt_eps_energy)
    args = _set_option(args, "--adapt-parallel-gradient-workers", max(0, int(static.adapt_parallel_gradient_workers)))
    args = _set_option(args, "--adapt-beam-parent-workers", max(0, int(static.adapt_beam_parent_workers)))
    args = _set_option(args, "--adapt-reopt-policy", static.adapt_reopt_policy)
    args = _set_option(args, "--adapt-window-size", static.adapt_window_size)
    args = _set_option(args, "--adapt-window-topk", static.adapt_window_topk)
    args = _set_option(args, "--adapt-full-refit-every", static.adapt_full_refit_every)
    args = _set_option(args, "--adapt-final-full-refit", "true" if static.adapt_final_full_refit else "false")
    args = _set_option(args, "--adapt-final-refit-maxiter", static.adapt_final_refit_maxiter)
    args = _set_option(args, "--adapt-insertion-mode", static.adapt_insertion_mode)
    args = _set_option(args, "--adapt-beam-live-branches", static.adapt_beam_live_branches)
    args = _set_option(args, "--adapt-beam-children-per-parent", static.adapt_beam_children_per_parent)
    args = _set_option(args, "--adapt-beam-terminated-keep", static.adapt_beam_terminated_keep)
    args = _set_option(args, "--adapt-beam-lambda", max(0.0, float(static.adapt_beam_lambda)))
    args = _set_toggle_pair(args, "--adapt-allow-repeats", "--adapt-no-repeats", static.adapt_allow_repeats)

    args = _set_option(args, "--phase1-lambda-compile", static.lambda_compile)
    args = _set_option(args, "--phase1-lambda-measure", static.lambda_measure)
    args = _set_option(args, "--phase1-lambda-leak", static.lambda_leak)
    if not bool(static.suppress_explicit_hardware_lambdas):
        args = _set_option(args, "--phase1-lambda-2q", static.lambda_2q)
        args = _set_option(args, "--phase1-lambda-d", static.lambda_d)
        args = _set_option(args, "--phase1-lambda-1q", static.lambda_1q)
        args = _set_option(args, "--phase1-lambda-theta", static.lambda_theta)
        args = _set_option(args, "--phase1-lambda-shot", static.lambda_shot)
    args = _set_option(args, "--phase2-rho", static.phase2_rho)
    args = _set_option(args, "--phase1-score-mode", static.phase1_score_mode)
    if static.phase1_score_z_alpha is not None:
        args = _set_option(args, "--phase1-score-z-alpha", static.phase1_score_z_alpha)
    if static.phase2_score_z_alpha is not None:
        args = _set_option(args, "--phase2-score-z-alpha", static.phase2_score_z_alpha)
    args = _set_option(args, "--phase1-compile-cx-proxy-weight", static.compile_cx_weight)
    args = _set_option(args, "--phase1-compile-sq-proxy-weight", static.compile_sq_weight)
    args = _set_option(args, "--phase1-compile-rotation-step-weight", static.compile_rotation_step_weight)
    args = _set_option(args, "--phase1-compile-position-shift-weight", static.compile_position_shift_weight)
    args = _set_option(args, "--phase1-compile-refit-active-weight", static.compile_refit_active_weight)
    args = _set_option(args, "--phase1-measure-groups-weight", static.measure_groups_weight)
    args = _set_option(args, "--phase1-measure-shots-weight", static.measure_shots_weight)
    args = _set_option(args, "--phase1-measure-reuse-weight", static.measure_reuse_weight)
    args = _set_option(args, "--phase1-opt-dim-cost-scale", static.opt_dim_cost_scale)
    args = _set_option(args, "--phase1-family-repeat-cost-scale", pool.family_repeat_penalty)
    args = _set_option(args, "--phase1-shortlist-size", phase1_shortlist)
    args = _set_option(args, "--phase1-maturity-cap-min", phase1_cap_min)
    args = _set_option(args, "--phase1-maturity-cap-max", phase1_cap_max)
    args = _set_option(args, "--phase2-maturity-cap-min", phase2_cap_min)
    args = _set_option(args, "--phase2-maturity-cap-max", phase2_cap_max)
    args = _set_option(args, "--phase3-maturity-cap-min", phase3_cap_min)
    args = _set_option(args, "--phase3-maturity-cap-max", phase3_cap_max)
    args = _set_option(args, "--phase-maturity-shot-min", shot_min)
    args = _set_option(args, "--phase-maturity-shot-max", shot_max)
    args = _set_option(args, "--phase1-maturity-shot-cap", max(0, int(static.phase1_maturity_shot_cap)))
    args = _set_option(args, "--phase2-maturity-shot-cap", max(0, int(static.phase2_maturity_shot_cap)))
    args = _set_option(args, "--phase3-maturity-shot-cap", max(0, int(static.phase3_maturity_shot_cap)))
    args = _set_option(args, "--phase1-probe-max-positions", static.phase1_probe_max_positions)
    args = _set_toggle_pair(args, "--phase1-prune-enabled", "--phase1-no-prune", static.phase1_prune_enabled)
    args = _set_option(args, "--phase1-prune-policy", static.phase1_prune_policy)
    if static.phase1_prune_enabled:
        args = _set_option(args, "--phase1-prune-mode", static.phase1_prune_mode)
        args = _set_option(args, "--phase1-prune-fraction", static.phase1_prune_fraction)
        args = _set_option(args, "--phase1-prune-min-candidates", static.phase1_prune_min_candidates)
        args = _set_option(args, "--phase1-prune-max-candidates", static.phase1_prune_max_candidates)
        args = _set_option(args, "--phase1-prune-max-regression", static.phase1_prune_max_regression)
        args = _set_option(args, "--phase1-prune-tolerance-mode", static.phase1_prune_tolerance_mode)
        args = _set_option(args, "--phase1-prune-tolerance-shot-coeff", static.phase1_prune_tolerance_shot_coeff)
        args = _set_option(args, "--phase1-prune-tolerance-screen-coeff", static.phase1_prune_tolerance_screen_coeff)
        args = _set_option(args, "--phase1-prune-tolerance-chem", static.phase1_prune_tolerance_chem)
        args = _set_option(args, "--phase1-prune-tolerance-rel-coeff", static.phase1_prune_tolerance_rel_coeff)
        args = _set_option(args, "--phase1-prune-retained-gain-ratio", static.phase1_prune_retained_gain_ratio)
        args = _set_option(args, "--phase1-prune-protect-steps", static.phase1_prune_protect_steps)
        args = _set_option(args, "--phase1-prune-cooldown-steps", static.phase1_prune_cooldown_steps)
        args = _set_option(args, "--phase1-prune-local-window-size", static.phase1_prune_local_window_size)
        args = _set_option(args, "--phase1-prune-recovery-trust-radius", static.phase1_prune_recovery_trust_radius)
        args = _set_option(args, "--phase1-prune-old-fraction", static.phase1_prune_old_fraction)
        args = _set_option(args, "--phase1-prune-checkpoint-period", static.phase1_prune_checkpoint_period)
        args = _set_option(args, "--phase1-prune-live-min-depth", static.phase1_prune_live_min_depth)
        args = _set_option(args, "--phase1-prune-maturity-threshold", static.phase1_prune_maturity_threshold)
        args = _set_option(args, "--phase1-prune-snr-threshold", static.phase1_prune_snr_threshold)
    args = _set_option(args, "--phase2-shortlist-fraction", static.phase2_shortlist_fraction)
    args = _set_option(args, "--phase2-shortlist-size", phase2_shortlist)
    args = _set_option(args, "--phase2-frontier-ratio", static.phase2_frontier_ratio)
    args = _set_option(args, "--phase2-compile-cx-proxy-weight", static.compile_cx_weight)
    args = _set_option(args, "--phase2-compile-sq-proxy-weight", static.compile_sq_weight)
    args = _set_option(args, "--phase2-compile-rotation-step-weight", static.compile_rotation_step_weight)
    args = _set_option(args, "--phase2-compile-position-shift-weight", static.compile_position_shift_weight)
    args = _set_option(args, "--phase2-compile-refit-active-weight", static.compile_refit_active_weight)
    args = _set_option(args, "--phase2-measure-groups-weight", static.measure_groups_weight)
    args = _set_option(args, "--phase2-measure-shots-weight", static.measure_shots_weight)
    args = _set_option(args, "--phase2-measure-reuse-weight", static.measure_reuse_weight)
    args = _set_option(args, "--phase2-opt-dim-cost-scale", static.opt_dim_cost_scale)
    args = _set_option(args, "--phase2-family-repeat-cost-scale", pool.family_repeat_penalty)
    args = _set_option(args, "--phase2-leakage-cap", static.phase2_leakage_cap)
    if not bool(static.suppress_explicit_hardware_lambdas):
        args = _set_option(args, "--phase2-lambda-2q", static.lambda_2q)
        args = _set_option(args, "--phase2-lambda-d", static.lambda_d)
        args = _set_option(args, "--phase2-lambda-1q", static.lambda_1q)
        args = _set_option(args, "--phase2-lambda-theta", static.lambda_theta)
        args = _set_option(args, "--phase2-lambda-shot", static.lambda_shot)
    args = _set_option(args, "--phase2-w-depth", static.phase2_w_depth)
    args = _set_option(args, "--phase2-w-group", static.phase2_w_group)
    args = _set_option(args, "--phase2-w-shot", static.phase2_w_shot)
    args = _set_option(args, "--phase2-w-optdim", static.phase2_w_optdim)
    args = _set_option(args, "--phase2-w-reuse", static.phase2_w_reuse)
    args = _set_option(args, "--phase2-w-lifetime", static.phase2_w_lifetime)
    motif_bonus_weight = (
        0.0
        if static.phase2_motif_bonus_weight is None
        else float(static.phase2_motif_bonus_weight)
    )
    args = _set_option(args, "--phase2-motif-bonus-weight", motif_bonus_weight)
    args = _set_toggle_pair(args, "--phase3-enable-batching", "--phase3-no-batching", static.phase2_enable_batching)
    args = _set_toggle_pair(args, "--phase2-enable-batching", "--phase2-no-batching", static.phase2_enable_batching)
    args = _set_option(args, "--phase2-batch-target-size", static.phase2_batch_target_size)
    args = _set_option(args, "--phase2-batch-size-cap", static.phase2_batch_size_cap)
    args = _set_option(args, "--phase2-batch-near-degenerate-ratio", static.phase2_batch_near_degenerate_ratio)
    args = _set_option(args, "--phase2-batch-rank-rel-tol", static.phase2_batch_rank_rel_tol)
    args = _set_option(args, "--phase2-batch-additivity-tol", static.phase2_batch_additivity_tol)
    args = _set_option(args, "--phase3-batch-selection-mode", static.phase3_batch_selection_mode)
    args = _set_option(args, "--phase3-batch-prefilter-mode", static.phase3_batch_prefilter_mode)

    args = _set_option(args, "--phase3-frontier-ratio", static.phase3_frontier_ratio)
    args = _set_option(args, "--phase3-tie-beam-score-ratio", static.phase3_tie_beam_score_ratio)
    args = _set_option(args, "--phase3-tie-beam-abs-tol", static.phase3_tie_beam_abs_tol)
    args = _set_option(args, "--phase3-tie-beam-max-branches", static.phase3_tie_beam_max_branches)
    args = _set_option(args, "--phase3-tie-beam-max-late-coordinate", static.phase3_tie_beam_max_late_coordinate)
    args = _set_option(args, "--phase3-tie-beam-min-depth-left", static.phase3_tie_beam_min_depth_left)
    plateau_cfg = normalize_plateau_acquisition_config(
        mode=getattr(static, "phase3_plateau_acquisition_mode", PLATEAU_ACQUISITION_MODE_OFF),
        acquisition_score=getattr(
            static,
            "phase3_plateau_acquisition_score",
            PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
        ),
        unlock_margin=getattr(static, "phase3_plateau_unlock_margin", 1e-8),
        duplicate_policy=getattr(
            static,
            "phase3_plateau_duplicate_policy",
            PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
        ),
        lambda_vol=getattr(static, "phase3_plateau_lambda_vol", 1e-8),
        sigma_min=getattr(static, "phase3_plateau_sigma_min", 0.0),
        nu_min=getattr(static, "phase3_plateau_nu_min", 0.0),
        volume_min=getattr(static, "phase3_plateau_volume_min", 0.0),
        failed_family_patience=getattr(static, "phase3_plateau_failed_family_patience", 0),
        trial_optimizer=getattr(
            static,
            "phase3_plateau_trial_optimizer",
            PLATEAU_TRIAL_OPTIMIZER_INHERIT,
        ),
        trial_qngd_maxiter=getattr(static, "phase3_plateau_trial_qngd_maxiter", 64),
    )
    args = _set_option(args, "--phase3-plateau-acquisition-mode", plateau_cfg.mode)
    args = _set_option(args, "--phase3-plateau-acquisition-score", plateau_cfg.acquisition_score)
    args = _set_option(args, "--phase3-plateau-unlock-margin", plateau_cfg.unlock_margin)
    args = _set_option(args, "--phase3-plateau-duplicate-policy", plateau_cfg.duplicate_policy)
    args = _set_option(args, "--phase3-plateau-lambda-vol", plateau_cfg.lambda_vol)
    args = _set_option(args, "--phase3-plateau-sigma-min", plateau_cfg.sigma_min)
    args = _set_option(args, "--phase3-plateau-nu-min", plateau_cfg.nu_min)
    args = _set_option(args, "--phase3-plateau-volume-min", plateau_cfg.volume_min)
    args = _set_option(args, "--phase3-plateau-failed-family-patience", plateau_cfg.failed_family_patience)
    args = _set_option(args, "--phase3-plateau-trial-optimizer", plateau_cfg.trial_optimizer)
    args = _set_option(args, "--phase3-plateau-trial-qngd-maxiter", plateau_cfg.trial_qngd_maxiter)
    args = _set_option(args, "--phase3-backend-cost-mode", static.phase3_backend_cost_mode)
    if static.phase3_hardware_cost_normalization_mode is not None:
        args = _set_option(
            args,
            "--phase3-hardware-cost-normalization-mode",
            static.phase3_hardware_cost_normalization_mode,
        )
    if str(static.phase3_source_lock_preferred_sequence or "").strip():
        args = _set_option(
            args,
            "--phase3-source-lock-preferred-sequence",
            static.phase3_source_lock_preferred_sequence,
        )
    args = _append_static_phase3_oracle_cli_args(args, static)
    hardware_resolution_mode, hardware_profile_json, hardware_profile_name = _validate_static_hardware_resolution_policy(static)
    _validate_static_route_policy(
        static,
        hardware_resolution_mode=hardware_resolution_mode,
    )
    if hardware_resolution_mode == "profile":
        args = _set_option(args, "--hardware-resolution-mode", "profile")
        args = _set_option(args, "--hardware-resolution-profile-json", hardware_profile_json)
        args = _set_option(args, "--hardware-resolution-profile-name", hardware_profile_name)
    args = _set_option(args, "--phase3-lifetime-cost-mode", "off")
    runtime_split_mode = str(getattr(static, "phase3_runtime_split_mode", "off") or "off").strip().lower()
    args = _set_option(args, "--phase3-runtime-split-mode", runtime_split_mode)
    args = _remove_option(args, "--allow-archival-phase3-runtime-split")
    if runtime_split_mode != "off":
        if not bool(getattr(static, "allow_archival_phase3_runtime_split", False)):
            raise ValueError(
                "non-off phase3_runtime_split_mode requires allow_archival_phase3_runtime_split=True; "
                "runtime split remains diagnostic-only."
            )
        args = [*args, "--allow-archival-phase3-runtime-split"]
        args = _set_option(
            args,
            "--phase3-runtime-split-selection-mode",
            str(getattr(static, "phase3_runtime_split_selection_mode", "proxy_child_set_preselection")),
        )
        args = _set_option(
            args,
            "--phase3-runtime-split-max-subset-size",
            int(getattr(static, "phase3_runtime_split_max_subset_size", 3)),
        )
    args = _set_option(
        args,
        "--shared-pauli-pool-mode",
        str(getattr(static, "shared_pauli_pool_mode", "off") or "off"),
    )
    args = _set_option(
        args,
        "--shared-pauli-pool-symmetry-policy",
        str(getattr(static, "shared_pauli_pool_symmetry_policy", "off") or "off"),
    )
    args = _set_option(
        args,
        "--shared-pauli-pool-max-subset-size",
        int(getattr(static, "shared_pauli_pool_max_subset_size", 3)),
    )
    args = _set_toggle_pair(args, "--phase3-enable-rescue", "--phase3-no-rescue", True)
    return args

def apply_policy_to_pipeline_args(
    base_args: Sequence[str],
    policy: AlgorithmPolicy,
    spec: HamiltonianBenchmarkSpec,
) -> list[str]:
    normalized_policy = _normalize_retained_policy(policy)
    args = [str(x) for x in base_args]
    for flag in (
        "--adapt-selected-logical-source-json",
        "--adapt-selected-logical-mode",
        "--adapt-selected-logical-transfer-mode",
    ):
        args = _remove_option(args, flag)
    if _static_phase3_oracle_overlay_requested(normalized_policy.static):
        for flag in _PHASE3_ORACLE_CLI_FLAGS:
            args = _remove_option(args, flag)
    idx = 0
    overrides = policy_to_cli_args(normalized_policy, spec)
    toggles = {
        "--phase1-prune-enabled": ("--phase1-prune-enabled", "--phase1-no-prune", True),
        "--phase1-no-prune": ("--phase1-prune-enabled", "--phase1-no-prune", False),
        "--phase3-enable-batching": ("--phase3-enable-batching", "--phase3-no-batching", True),
        "--phase3-no-batching": ("--phase3-enable-batching", "--phase3-no-batching", False),
        "--phase2-enable-batching": ("--phase2-enable-batching", "--phase2-no-batching", True),
        "--phase2-no-batching": ("--phase2-enable-batching", "--phase2-no-batching", False),
        "--adapt-allow-repeats": ("--adapt-allow-repeats", "--adapt-no-repeats", True),
        "--adapt-no-repeats": ("--adapt-allow-repeats", "--adapt-no-repeats", False),
        "--phase3-enable-rescue": ("--phase3-enable-rescue", "--phase3-no-rescue", True),
        "--phase3-no-rescue": ("--phase3-enable-rescue", "--phase3-no-rescue", False),
    }
    single_flags = {
        "--allow-archival-phase3-runtime-split",
    }
    while idx < len(overrides):
        flag = str(overrides[idx])
        if flag in toggles:
            positive, negative, enabled = toggles[flag]
            args = _set_toggle_pair(args, positive, negative, enabled)
            idx += 1
            continue
        if flag in single_flags:
            args = _remove_option(args, flag)
            args = [*args, flag]
            idx += 1
            continue
        value = overrides[idx + 1] if idx + 1 < len(overrides) else None
        args = _set_option(args, flag, value)
        idx += 2
    return args

def build_static_command(
    *,
    python_bin: str,
    spec: HamiltonianBenchmarkSpec,
    policy: AlgorithmPolicy,
    output_json: Path,
    adapt_current_json: Path | None = None,
    adapt_current_json_every_depth: int = 1,
    adapt_current_json_keep_history_tail: int = 100,
    benchmark_target_abs_delta_e: float | None = None,
    benchmark_target_reference_energy: float | None = None,
) -> list[str]:
    args = apply_policy_to_pipeline_args(spec.base_pipeline_args, policy, spec)
    args = _remove_option(_remove_option(_remove_option(_remove_option(args, "--output-json"), "--output-pdf"), "--skip-pdf"), "--skip-trajectory")
    args = _remove_option(args, "--adapt-current-json")
    args = _remove_option(args, "--adapt-current-json-every-depth")
    args = _remove_option(args, "--adapt-current-json-keep-history-tail")
    args = _remove_option(args, "--adapt-benchmark-target-abs-delta-e")
    args = _remove_option(args, "--adapt-benchmark-target-reference-energy")
    if adapt_current_json is not None:
        args = [
            *args,
            "--adapt-current-json",
            str(adapt_current_json),
            "--adapt-current-json-every-depth",
            str(max(1, int(adapt_current_json_every_depth))),
            "--adapt-current-json-keep-history-tail",
            str(max(0, int(adapt_current_json_keep_history_tail))),
        ]
    if benchmark_target_abs_delta_e is not None and float(benchmark_target_abs_delta_e) > 0.0:
        args = [*args, "--adapt-benchmark-target-abs-delta-e", str(float(benchmark_target_abs_delta_e))]
        if benchmark_target_reference_energy is not None:
            args = [*args, "--adapt-benchmark-target-reference-energy", str(float(benchmark_target_reference_energy))]
    args = [*args, "--output-json", str(output_json), "--skip-pdf", "--skip-trajectory"]
    return [str(python_bin), "-u", "-m", "pipelines.static_adapt.adapt_pipeline", *args]

def build_compile_command(
    *,
    python_bin: str,
    artifact_json: Path,
    compile_json: Path,
    compile_backend: str,
    compile_opt_level: int,
    compile_seed: int,
) -> list[str]:
    return [
        str(python_bin),
        "-u",
        "-m",
        "pipelines.scaffold.adapt_circuit_cost",
        "--artifact-json",
        str(artifact_json),
        "--backend-name",
        str(compile_backend),
        "--optimization-level",
        str(int(compile_opt_level)),
        "--seed-transpiler",
        str(int(compile_seed)),
        "--output-json",
        str(compile_json),
    ]


_RUNTIME_SCHEMA = "static_benchmark_runtime_v1"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_command_log(path: Path, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shlex.join([str(item) for item in command])
        + "\n",
        encoding="utf-8",
    )


def _terminate_process_group(proc: subprocess.Popen[str], *, grace_s: float = 10.0) -> None:
    try:
        os.killpg(int(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=float(grace_s))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(int(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=float(grace_s))


def _run_subprocess_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float | None = None,
    subprocess_label: str = "subprocess",
    benchmark_id: str | None = None,
    heartbeat_path: Path | None = None,
    heartbeat_events_path: Path | None = None,
    heartbeat_metadata: Mapping[str, Any] | None = None,
) -> tuple[int, float]:
    """Run one benchmark child while preserving logs and forwarding ``AI_LOG``.

    Search-trial ownership, pruning, and study lifecycle are intentionally not
    part of this retained neutral runner.
    """

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    recorder = (
        None
        if heartbeat_path in {None, ""}
        else LiveHeartbeatRecorder(
            heartbeat_path=Path(heartbeat_path),
            event_jsonl_path=heartbeat_events_path,
            metadata={
                "subprocess_label": str(subprocess_label),
                "benchmark_id": benchmark_id,
                **dict(heartbeat_metadata or {}),
            },
        )
    )
    child_state: dict[str, Any] = {}
    reader_errors: list[str] = []

    with (
        stdout_path.open("w", encoding="utf-8") as stdout_fh,
        stderr_path.open("w", encoding="utf-8") as stderr_fh,
    ):
        proc = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=stderr_fh,
            env=dict(os.environ),
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if recorder is not None:
            child_state = dict(
                recorder.mark_started(
                    pid=int(proc.pid),
                    command=[str(item) for item in command],
                )
            )

        def _forward_stdout() -> None:
            nonlocal child_state
            if proc.stdout is None:
                return
            try:
                for line in proc.stdout:
                    stdout_fh.write(line)
                    stdout_fh.flush()
                    payload = parse_ai_log_line(line)
                    if payload is None:
                        continue
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    elapsed = float(time.perf_counter() - started)
                    if recorder is not None:
                        child_state = dict(
                            recorder.update_from_ai_log(
                                payload,
                                elapsed_s=elapsed,
                                pid=int(proc.pid),
                            )
                        )
                    else:
                        child_state = dict(
                            normalize_ai_log_progress(
                                payload,
                                elapsed_s=elapsed,
                                pid=int(proc.pid),
                                previous=child_state,
                            )
                        )
            except Exception as exc:  # logging must not invalidate the child
                reader_errors.append(f"stdout_reader:{type(exc).__name__}:{exc}")

        reader = threading.Thread(
            target=_forward_stdout,
            name=f"static-benchmark-{subprocess_label}-stdout",
            daemon=True,
        )
        reader.start()
        returncode: int | None = None
        try:
            timeout = None if timeout_s is None or float(timeout_s) <= 0.0 else float(timeout_s)
            returncode = int(proc.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            returncode = 124
            stderr_fh.write(
                "\n[static_benchmark_runtime] subprocess timeout "
                f"after {time.perf_counter() - started:.3f}s\n"
            )
        except BaseException:
            _terminate_process_group(proc)
            raise
        finally:
            reader.join(timeout=2.0)
            for error in reader_errors:
                stderr_fh.write(f"\n[static_benchmark_runtime] {error}\n")
            if recorder is not None:
                recorder.mark_finished(
                    status="completed" if returncode == 0 else "failed",
                    returncode=returncode,
                    elapsed_s=float(time.perf_counter() - started),
                )
    assert returncode is not None
    return int(returncode), float(time.perf_counter() - started)


def run_static_benchmark(
    spec: HamiltonianBenchmarkSpec,
    policy: AlgorithmPolicy,
    *,
    output_dir: Path,
    python_bin: str = sys.executable,
    compile_backend: str = "FakeMarrakesh",
    compile_opt_level: int = 1,
    compile_seed: int = 7,
    adapt_timeout_s: float | None = None,
    compile_timeout_s: float | None = None,
    benchmark_target_abs_delta_e: float | None = None,
) -> BenchmarkResult:
    """Execute one retained generic static benchmark without study machinery."""

    policy = _normalize_retained_policy(policy)
    case_dir = Path(output_dir) / spec.benchmark_id
    logs_dir = case_dir / "logs"
    json_dir = case_dir / "json"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    policy_json = json_dir / "policy.json"
    result_json = json_dir / "result.json"
    current_json = json_dir / "current.json"
    compile_json = json_dir / "compile_scout_fake_marrakesh.json"
    _write_json(
        policy_json,
        {
            "schema": _RUNTIME_SCHEMA,
            "generated_utc": _now_utc(),
            "benchmark": asdict(spec),
            "policy": asdict(policy),
        },
    )

    reference_energy, reference_nph, reference_failure = _reference_cutoff_energy_for_spec(spec)
    command = build_static_command(
        python_bin=python_bin,
        spec=spec,
        policy=policy,
        output_json=result_json,
        adapt_current_json=current_json,
        benchmark_target_abs_delta_e=benchmark_target_abs_delta_e,
        benchmark_target_reference_energy=reference_energy,
    )
    _write_command_log(logs_dir / "command.sh", command)
    returncode, elapsed = _run_subprocess_logged(
        command,
        cwd=REPO_ROOT,
        stdout_path=logs_dir / "stdout.log",
        stderr_path=logs_dir / "stderr.log",
        timeout_s=adapt_timeout_s,
        subprocess_label="adapt",
        benchmark_id=spec.benchmark_id,
    )
    if returncode != 0 or not result_json.exists():
        reason = "adapt_timeout" if returncode == 124 else f"adapt_returncode:{returncode}"
        return BenchmarkResult(
            benchmark_id=spec.benchmark_id,
            family=spec.family,
            success=False,
            abs_delta_e=None,
            walltime_s=float(elapsed),
            failure_reason=reason,
            policy_json=str(policy_json),
        )

    compile_command = build_compile_command(
        python_bin=python_bin,
        artifact_json=result_json,
        compile_json=compile_json,
        compile_backend=compile_backend,
        compile_opt_level=compile_opt_level,
        compile_seed=compile_seed,
    )
    _write_command_log(logs_dir / "compile_command.sh", compile_command)
    compile_returncode, compile_elapsed = _run_subprocess_logged(
        compile_command,
        cwd=REPO_ROOT,
        stdout_path=logs_dir / "compile_stdout.log",
        stderr_path=logs_dir / "compile_stderr.log",
        timeout_s=compile_timeout_s,
        subprocess_label="compile",
        benchmark_id=spec.benchmark_id,
    )

    result_payload = json.loads(result_json.read_text(encoding="utf-8"))
    metrics = extract_adapt_energy_metrics(result_payload)
    objective_exact = metrics.exact_gs_energy
    objective_delta = metrics.abs_delta_e
    cutoff_delta = None
    reference_delta = None
    if reference_energy is not None and metrics.energy is not None:
        reference_delta = abs(float(metrics.energy) - float(reference_energy))
        objective_delta = reference_delta
        objective_exact = reference_energy
        if metrics.exact_gs_energy is not None:
            cutoff_delta = abs(float(metrics.exact_gs_energy) - float(reference_energy))

    count_2q, depth_2q, circuit_depth, logical_params, runtime_params = _compile_metrics(
        compile_json
    )
    groups, shots, shot_cost = _measurement_cost_proxies(result_payload)
    proxy_validation = _controller_proxy_validation_from_payload(result_payload)
    proxy_valid = bool(proxy_validation.get("valid", False))
    if not proxy_valid:
        shot_cost = None
    quality_ok, quality_reason, quality_meta = classify_static_result_quality(
        result_payload,
        spec,
        abs_delta_e=metrics.abs_delta_e,
    )
    target_ok = True
    if benchmark_target_abs_delta_e is not None and float(benchmark_target_abs_delta_e) > 0.0:
        target_ok = bool(
            objective_delta is not None
            and float(objective_delta) <= float(benchmark_target_abs_delta_e)
        )
    success = bool(
        objective_delta is not None
        and compile_returncode == 0
        and count_2q is not None
        and quality_ok
        and reference_failure is None
        and target_ok
    )
    reasons: list[str] = []
    if compile_returncode != 0:
        reasons.append(
            "compile_timeout"
            if compile_returncode == 124
            else f"compile_returncode:{compile_returncode}"
        )
    if count_2q is None:
        reasons.append("missing_compile_2q_count")
    if not quality_ok and quality_reason is not None:
        reasons.append(f"quality_gate:{quality_reason}")
    if reference_failure is not None:
        reasons.append(reference_failure)
    if not target_ok:
        reasons.append("benchmark_target_non_hit")

    return BenchmarkResult(
        benchmark_id=spec.benchmark_id,
        family=spec.family,
        success=success,
        abs_delta_e=objective_delta,
        energy=metrics.energy,
        exact_gs_energy=objective_exact,
        same_cutoff_exact_gs_energy=metrics.exact_gs_energy,
        exact_reference_energy=reference_energy,
        exact_reference_n_ph_max=reference_nph,
        abs_delta_e_same_cutoff=metrics.abs_delta_e,
        abs_delta_e_reference=reference_delta,
        cutoff_abs_delta_e=cutoff_delta,
        count_2q=count_2q,
        depth_2q=depth_2q,
        circuit_depth=circuit_depth,
        parameter_count=logical_params,
        runtime_parameter_count=runtime_params,
        measurement_groups_proxy=groups,
        measurement_shots_proxy=shots,
        shot_cost_proxy=shot_cost,
        walltime_s=float(elapsed + compile_elapsed),
        failure_reason=None if success else ";".join(reasons),
        stop_reason=quality_meta.get("stop_reason"),
        ansatz_depth=quality_meta.get("ansatz_depth"),
        initial_energy=quality_meta.get("initial_energy"),
        initial_abs_delta_e=quality_meta.get("initial_abs_delta_e"),
        max_gradient_seen=quality_meta.get("max_gradient_seen"),
        quality_gate_reason=quality_reason,
        boson_illegal_probability_max=_as_float_or_none(
            quality_meta.get("boson_illegal_probability_max")
        ),
        boson_legal_probability_min=_as_float_or_none(
            quality_meta.get("boson_legal_probability_min")
        ),
        measurement_proxy_validated=proxy_valid,
        measurement_proxy_validation=dict(proxy_validation),
        result_json=str(result_json),
        compile_json=str(compile_json) if compile_json.exists() else None,
        policy_json=str(policy_json),
    )


def _compile_metrics(path: Path) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    if not path.exists():
        return None, None, None, None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_backend", {}) if isinstance(payload, Mapping) else {}
    logical = payload.get("logical_circuit", {}) if isinstance(payload, Mapping) else {}

    def _maybe_int(value: Any) -> int | None:
        try:
            return None if value is None else int(value)
        except Exception:
            return None

    return (
        _maybe_int(selected.get("compiled_count_2q")),
        _maybe_int(selected.get("compiled_depth_2q")),
        _maybe_int(selected.get("compiled_depth")),
        _maybe_int(logical.get("logical_parameter_count")),
        _maybe_int(logical.get("runtime_parameter_count")),
    )

def _measurement_cost_proxies(result_payload: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    """Return live controller group/nominal-shot work proxies from ADAPT history."""

    proxy = controller_proxy_from_adapt_payload(result_payload)
    if int(proxy.get("history_row_count", 0)) <= 0 and int(proxy.get("events_count", 0)) <= 0:
        return None, None, None
    groups_new = _as_float_or_none(
        proxy.get("controller_group_proxy", proxy.get("total_groups_new", proxy.get("groups_new")))
    )
    shots_new = _as_float_or_none(
        proxy.get("controller_shot_proxy", proxy.get("total_shots_new", proxy.get("shots_new")))
    )
    if groups_new is None and shots_new is None:
        return None, None, None
    shot_cost_proxy = shots_new if shots_new is not None else groups_new
    return groups_new, shots_new, shot_cost_proxy

def _controller_proxy_validation_from_payload(result_payload: Mapping[str, Any]) -> dict[str, Any]:
    proxy = controller_proxy_from_adapt_payload(result_payload)
    return validate_controller_proxy_for_shot_objective(proxy)

def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None

def _as_int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None

def _adapt_vqe_section(result_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result_payload.get("adapt_vqe", {}) if isinstance(result_payload, Mapping) else {}
    return value if isinstance(value, Mapping) else {}

def _adapt_history_rows(adapt_vqe: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = adapt_vqe.get("history", []) if isinstance(adapt_vqe, Mapping) else []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    return tuple(row for row in rows if isinstance(row, Mapping))

def _collect_numeric_fields(
    value: Any,
    key_fragment: str,
    *,
    limit: int = 128,
    ignored_key_fragments: Sequence[str] = (),
) -> list[float]:
    found: list[float] = []
    ignored = tuple(str(fragment).lower() for fragment in ignored_key_fragments)

    def _walk(node: Any) -> None:
        if len(found) >= int(limit):
            return
        if isinstance(node, Mapping):
            for key, child in node.items():
                key_lower = str(key).lower()
                if any(fragment in key_lower for fragment in ignored):
                    continue
                if key_fragment in key_lower and not (key_fragment == "legal_probability" and "illegal_probability" in key_lower):
                    number = _as_float_or_none(child)
                    if number is not None:
                        found.append(number)
                _walk(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for child in node:
                _walk(child)

    _walk(value)
    return found

def _result_quality_metadata(result_payload: Mapping[str, Any], abs_delta_e: float | None) -> dict[str, Any]:
    adapt_vqe = _adapt_vqe_section(result_payload)
    history = _adapt_history_rows(adapt_vqe)
    exact = _as_float_or_none(adapt_vqe.get("exact_gs_energy"))
    if exact is None:
        ground = result_payload.get("ground_state", {}) if isinstance(result_payload, Mapping) else {}
        if isinstance(ground, Mapping):
            exact = _as_float_or_none(ground.get("exact_energy"))
    initial_energy = None
    if history:
        initial_energy = _as_float_or_none(history[0].get("energy_before_opt"))
    if initial_energy is None:
        initial_state = result_payload.get("initial_state", {}) if isinstance(result_payload, Mapping) else {}
        if isinstance(initial_state, Mapping):
            initial_energy = _as_float_or_none(initial_state.get("energy"))
    initial_abs_delta_e = None
    if initial_energy is not None and exact is not None:
        initial_abs_delta_e = abs(float(initial_energy) - float(exact))
    max_gradients = [_as_float_or_none(row.get("max_grad")) for row in history]
    max_gradient_seen = max((abs(x) for x in max_gradients if x is not None), default=None)
    ignored_population_diagnostics = ("pool_legal_subspace_filter",)
    illegal_values = [
        x
        for x in _collect_numeric_fields(
            result_payload,
            "illegal_probability",
            ignored_key_fragments=ignored_population_diagnostics,
        )
        if x is not None
    ]
    legal_values = [
        x
        for x in _collect_numeric_fields(
            result_payload,
            "legal_probability",
            ignored_key_fragments=ignored_population_diagnostics,
        )
        if x is not None
    ]
    # Avoid double-counting keys named illegal_probability as legal_probability.
    legal_values = [x for x in legal_values if 0.0 <= float(x) <= 1.0]
    illegal_max = max([float(x) for x in illegal_values], default=None)
    legal_min = min([float(x) for x in legal_values], default=None)
    if legal_min is not None:
        inferred_illegal = max(0.0, 1.0 - float(legal_min))
        illegal_max = inferred_illegal if illegal_max is None else max(float(illegal_max), inferred_illegal)
    return {
        "stop_reason": None if adapt_vqe.get("stop_reason") is None else str(adapt_vqe.get("stop_reason")),
        "ansatz_depth": _as_int_or_none(adapt_vqe.get("ansatz_depth")),
        "initial_energy": initial_energy,
        "initial_abs_delta_e": initial_abs_delta_e,
        "max_gradient_seen": max_gradient_seen,
        "boson_illegal_probability_max": illegal_max,
        "boson_legal_probability_min": legal_min,
        "abs_delta_e": abs_delta_e,
    }

def classify_static_result_quality(
    result_payload: Mapping[str, Any],
    spec: HamiltonianBenchmarkSpec,
    *,
    abs_delta_e: float | None,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Fail closed on mechanically successful but physically useless ADAPT runs.

    A benchmark must not treat a valid JSON artifact plus successful compile as
    a useful result when ADAPT stopped immediately, made no energy progress, or left
    a large benchmark-normalized error. These are correctness/precondition
    failures, not merely poor hyperparameters.
    """

    meta = _result_quality_metadata(result_payload, abs_delta_e)
    final_error = _as_float_or_none(abs_delta_e)
    if final_error is None:
        return False, "missing_abs_delta_e", meta
    illegal_probability = _as_float_or_none(meta.get("boson_illegal_probability_max"))
    if (
        illegal_probability is not None
        and illegal_probability > _BOSON_ILLEGAL_PROBABILITY_FAIL_THRESHOLD
    ):
        return False, "boson_illegal_population", meta
    baseline = max(float(spec.baseline_abs_delta_e), 1e-12)
    large_error = final_error > max(10.0 * baseline, 1e-6)
    depth = meta.get("ansatz_depth")
    stop_reason = str(meta.get("stop_reason") or "")
    if depth is not None and int(depth) <= 0 and large_error:
        return False, "depth0_large_error", meta
    if "eps_grad" in stop_reason and large_error:
        return False, "eps_grad_large_error", meta
    initial_error = _as_float_or_none(meta.get("initial_abs_delta_e"))
    if initial_error is not None and initial_error > 0.0 and large_error:
        improvement = initial_error - final_error
        tolerance = max(1e-10, 1e-6 * initial_error)
        if improvement <= tolerance:
            return False, "no_energy_improvement", meta
    return True, None, meta

def _reference_cutoff_energy_for_spec(spec: HamiltonianBenchmarkSpec) -> tuple[float | None, int | None, str | None]:
    ref_nph = spec.exact_reference_n_ph_max
    if ref_nph is None:
        return None, None, None
    args = tuple(spec.base_pipeline_args)
    n_ph = _option_int(_parse_cli_option_map(" ".join(args)), "n_ph_max")
    if n_ph is not None and int(ref_nph) <= int(n_ph):
        return None, None, None
    try:
        exact, _key_hash, _key = exact_energy_for_spec(spec, n_ph_max=int(ref_nph))
        return float(exact), int(ref_nph), None
    except Exception as exc:
        return None, int(ref_nph), f"reference_exact_failed:{exc}"

def _clamp(value: float, low: float, high: float) -> float:
    return float(max(float(low), min(float(high), float(value))))

def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(value)

def _parse_cli_option_map(command: str | None) -> dict[str, Any]:
    if not command:
        return {}
    try:
        tokens = shlex.split(str(command).replace("\\\n", " "))
    except Exception:
        return {}
    options: dict[str, Any] = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("--"):
            idx += 1
            continue
        key = token[2:].replace("-", "_")
        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            options[key] = tokens[idx + 1]
            idx += 2
        else:
            options[key] = True
            idx += 1
    return options

def _option_int(options: Mapping[str, Any], key: str) -> int | None:
    try:
        if key not in options:
            return None
        return int(float(options[key]))
    except Exception:
        return None

def _paper_i_history_row_acceptance_status(
    row: Mapping[str, Any],
    *,
    committed_operator_count: int | None,
    initial_operator_count: int | None,
    committed_depth: int | None,
) -> tuple[bool, str]:
    """Return whether a history row represents a committed ADAPT admission."""

    for key in ("accepted_admission", "accepted"):
        if key in row:
            return bool(row.get(key)), f"explicit_{key}"
    for key in ("logical_parameters_added_this_step", "parameters_added_this_step"):
        value = _as_int_or_none(row.get(key))
        if value is not None:
            return value > 0, f"{key}_{'positive' if value > 0 else 'nonpositive'}"
    operator_count = _as_int_or_none(row.get("logical_num_parameters_after_opt"))
    if operator_count is not None:
        if committed_operator_count is not None:
            return operator_count > int(committed_operator_count), "logical_operator_count_increased"
        if initial_operator_count is None:
            return False, "preexisting_initial_operator_count"
        return operator_count > int(initial_operator_count), "logical_operator_count_increased_from_initial"
    depth = _as_int_or_none(row.get("depth"))
    if depth is not None and row.get("energy_after_opt") is not None:
        if int(depth) <= 0:
            return False, "legacy_depth_zero_initial_row"
        if committed_depth is not None and int(depth) <= int(committed_depth):
            return False, "legacy_depth_not_increased"
        return True, "legacy_depth_energy_row"
    return False, "no_admission_evidence"

def default_static_benchmark_suite(
    split: str = "train",
    *,
    molecular_problem_json: str | Path | None = None,
    boson_cutoff: int | None = None,
    boson_cutoffs: Sequence[int] | None = None,
    exact_reference_boson_cutoff: int | None = 4,
    physics_grid_profile: str = "canonical",
) -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Small canonical manifest for L=2/L=3 policy studies.

    These are benchmark descriptors, not executed at import time. Molecular uses
    active-orbital/qubit count rather than lattice L and is expected to be
    filled with a real ``--molecular-problem-json`` by the caller.
    """

    if boson_cutoff is not None and int(boson_cutoff) < 1:
        raise ValueError("boson_cutoff must be >= 1 when provided.")
    if boson_cutoffs is not None and any(int(value) < 1 for value in boson_cutoffs):
        raise ValueError("boson_cutoffs values must be >= 1 when provided.")
    if exact_reference_boson_cutoff is not None and int(exact_reference_boson_cutoff) < 1:
        raise ValueError("exact_reference_boson_cutoff must be >= 1 when provided.")

    def resolved_boson_cutoff(family: str) -> int:
        return int(boson_cutoff) if boson_cutoff is not None else _default_boson_cutoff_for_family(family)

    def resolved_boson_cutoff_values(family: str, bosonic: bool) -> tuple[int, ...]:
        if not bosonic:
            return (resolved_boson_cutoff(family),)
        if boson_cutoffs is not None:
            return tuple(dict.fromkeys(int(value) for value in boson_cutoffs))
        return (resolved_boson_cutoff(family),)

    def resolved_reference_cutoff(family: str) -> int | None:
        if not bool(family in _HEAVY_BOSONIC_CUTOFF2_FAMILIES):
            return None
        return None if exact_reference_boson_cutoff is None else int(exact_reference_boson_cutoff)

    profile = str(physics_grid_profile or "canonical").strip().lower()
    if profile not in {"canonical", "small_robust", "robust", "paper_i_clean"}:
        raise ValueError(f"Unsupported physics_grid_profile: {physics_grid_profile!r}")

    def physics_variants(family: str, L: int) -> tuple[tuple[str, Mapping[str, str], tuple[str, ...]], ...]:
        if profile == "paper_i_clean":
            if int(L) != 2:
                return ()
            if family in {"hubbard", "ionic_hubbard", "ttprime_hubbard"}:
                return (
                    ("_clean_weak", {"u": "2.0"}, ("physics_clean", "weak", "u2")),
                    ("_clean_strong", {"u": "8.0"}, ("physics_clean", "strong", "u8")),
                )
            if family == "extended_hubbard":
                return (
                    ("_clean_weak", {"u": "2.0", "v_nn": "0.5"}, ("physics_clean", "weak", "u2", "v0p5")),
                    ("_clean_strong", {"u": "8.0", "v_nn": "1.5"}, ("physics_clean", "strong", "u8", "v1p5")),
                )
            if family == "spinless_tv":
                return (
                    ("_clean_weak", {"v_nn": "0.5"}, ("physics_clean", "weak", "v0p5")),
                    ("_clean_strong", {"v_nn": "1.5"}, ("physics_clean", "strong", "v1p5")),
                )
            if family == "bose_hubbard":
                return (
                    ("_clean_weak", {"u": "2.0"}, ("physics_clean", "weak", "u2")),
                    ("_clean_strong", {"u": "6.0"}, ("physics_clean", "strong", "u6")),
                )
            if family == "harmonic_kerr_chain":
                return (
                    ("_clean_weak", {"omega0": "1.0"}, ("physics_clean", "weak", "w1")),
                    ("_clean_strong", {"omega0": "0.75"}, ("physics_clean", "strong", "w0p75")),
                )
            if family == "hh":
                return (
                    ("_clean_weak", {"u": "2.0", "g_ep": "0.25", "omega0": "1.0"}, ("physics_clean", "weak", "u2", "g0p25")),
                    ("_clean_strong", {"u": "8.0", "g_ep": "1.0", "omega0": "1.0"}, ("physics_clean", "strong", "u8", "g1")),
                )
            if family == "spin_boson":
                return (
                    ("_clean_weak", {"u": "0.0", "dv": "0.0", "omega0": "1.0", "g_ep": "0.25"}, ("physics_clean", "weak", "g0p25")),
                    ("_clean_strong", {"u": "0.0", "dv": "0.0", "omega0": "1.0", "g_ep": "1.0"}, ("physics_clean", "strong", "g1")),
                )
            return ()
        variants: list[tuple[str, Mapping[str, str], tuple[str, ...]]] = [("", {}, ("physics_canonical",))]
        if profile in {"small_robust", "robust"} and int(L) <= 2:
            if family in {"hubbard", "ionic_hubbard", "extended_hubbard", "ttprime_hubbard"}:
                variants.append(("_u6", {"u": "6.0"}, ("physics_perturbation", "u6")))
            elif family == "spinless_tv":
                variants.append(("_v1p5", {"v_nn": "1.5"}, ("physics_perturbation", "v1p5")))
            elif family in {"hh", "spin_boson"}:
                variants.append(("_g0p7", {"g_ep": "0.7"}, ("physics_perturbation", "g0p7")))
            elif family == "bose_hubbard":
                variants.append(("_u2", {"u": "2.0"}, ("physics_perturbation", "u2")))
            elif family == "harmonic_kerr_chain":
                variants.append(("_w0p75", {"omega0": "0.75"}, ("physics_perturbation", "w0p75")))
        if profile == "robust":
            if family in {"hubbard", "ionic_hubbard", "extended_hubbard", "ttprime_hubbard"}:
                variants.append(("_u2", {"u": "2.0"}, ("physics_perturbation", "u2")))
            elif family == "spinless_tv":
                variants.append(("_v0p5", {"v_nn": "0.5"}, ("physics_perturbation", "v0p5")))
            elif family in {"hh", "spin_boson"}:
                variants.append(("_g0p3", {"g_ep": "0.3"}, ("physics_perturbation", "g0p3")))
            elif family == "bose_hubbard":
                variants.append(("_u6", {"u": "6.0"}, ("physics_perturbation", "u6")))
            elif family == "harmonic_kerr_chain":
                variants.append(("_w1p25", {"omega0": "1.25"}, ("physics_perturbation", "w1p25")))
        return tuple(variants)

    specs: list[HamiltonianBenchmarkSpec] = []
    families = (
        ("hubbard", True, False, False, 2, 4, 64),
        ("hh", True, True, False, 2, 6, 128),
        ("ionic_hubbard", True, False, False, 2, 4, 64),
        ("extended_hubbard", True, False, False, 2, 4, 96),
        ("ttprime_hubbard", True, False, False, 2, 4, 128),
        ("spinless_tv", False, False, False, 2, 2, 48),
        ("bose_hubbard", False, True, False, 2, 4, 96),
        ("harmonic_kerr_chain", False, True, False, 2, 4, 96),
        ("spin_boson", False, True, False, 2, 6, 96),
    )
    for family, spinful, bosonic, molecular, base_l, qubits_per_l2, pool_hint in families:
        if family == "spin_boson" and profile != "paper_i_clean":
            continue
        get_problem_family_spec(family)
        for L in (2, 3):
            cutoff_values = resolved_boson_cutoff_values(family, bosonic)
            for cutoff in cutoff_values:
                for variant_suffix, overrides, variant_tags in physics_variants(family, L):
                    if family == "spin_boson":
                        nq = 2 + int(L) * int(boson_qubits_per_site(int(cutoff), "binary"))
                    else:
                        nq = int(qubits_per_l2 if L == 2 else max(qubits_per_l2 + 2, int(round(qubits_per_l2 * 1.5))))
                    values = {
                        "t": "1.0",
                        "u": "4.0",
                        "dv": "0.0" if family == "hubbard" else "0.25",
                        "omega0": "1.0",
                        "g_ep": "0.5",
                        "v_nn": "1.0" if family in {"extended_hubbard", "spinless_tv"} else "0.0",
                        "t_prime": "0.4" if family == "ttprime_hubbard" else "0.0",
                        "boundary": "periodic" if family == "hubbard" else "open",
                    }
                    values.update({str(key): str(value) for key, value in overrides.items()})
                    args = (
                        "--problem",
                        family,
                        "--L",
                        str(L),
                        "--t",
                        values["t"],
                        "--u",
                        values["u"],
                        "--dv",
                        values["dv"],
                        "--omega0",
                        values["omega0"],
                        "--g-ep",
                        values["g_ep"],
                        "--n-ph-max",
                        str(cutoff),
                        "--boson-encoding",
                        "binary",
                        "--ordering",
                        "blocked",
                        "--boundary",
                        values["boundary"],
                        "--v-nn",
                        values["v_nn"],
                        "--t-prime",
                        values["t_prime"],
                    )
                    cutoff_suffix = (
                        f"_nph{cutoff}"
                        if bosonic and (len(cutoff_values) > 1 or profile == "paper_i_clean")
                        else ""
                    )
                    specs.append(
                        HamiltonianBenchmarkSpec(
                            benchmark_id=f"{family}_L{L}{cutoff_suffix}{variant_suffix}",
                            family=family,
                            features=ProblemFeatureVector(
                                problem=family,
                                size_label=f"L{L}{cutoff_suffix}{variant_suffix}",
                                L=L,
                                n_qubits=nq,
                                pool_size_hint=int(pool_hint * (L / base_l)),
                                spinful=spinful,
                                bosonic=bosonic,
                                molecular=molecular,
                            ),
                            base_pipeline_args=args,
                            baseline_abs_delta_e=1e-4,
                            baseline_count_2q=1000,
                            baseline_depth_2q=3000,
                            baseline_parameter_count=64,
                            baseline_shot_cost_proxy=64,
                            exact_reference_n_ph_max=resolved_reference_cutoff(family),
                            split=split,
                            tags=("static_phase3", "size_sweep", f"nph{cutoff}", f"physics_profile:{profile}", *variant_tags),
                        )
                    )
    # Non-clean spin_boson remains the historical L=1 diagnostic row.  The
    # clean Paper-I profile above uses L=2 explicitly and must not alias back
    # to this legacy case.
    if profile != "paper_i_clean":
        family = "spin_boson"
        get_problem_family_spec(family)
        spin_boson_cutoffs = resolved_boson_cutoff_values(family, True)
        for cutoff in spin_boson_cutoffs:
            cutoff_suffix = f"_nph{cutoff}" if len(spin_boson_cutoffs) > 1 else ""
            for variant_suffix, overrides, variant_tags in physics_variants(family, 1):
                values = {"u": "0.0", "dv": "0.0", "omega0": "1.0", "g_ep": "0.5"}
                values.update({str(key): str(value) for key, value in overrides.items()})
                specs.append(
                    HamiltonianBenchmarkSpec(
                        benchmark_id=f"spin_boson_L1{cutoff_suffix}{variant_suffix}",
                        family=family,
                        features=ProblemFeatureVector(
                            problem=family,
                            size_label=f"L1{cutoff_suffix}{variant_suffix}",
                            L=1,
                            n_qubits=4,
                            pool_size_hint=64,
                            bosonic=True,
                        ),
                        base_pipeline_args=(
                            "--problem",
                            family,
                            "--L",
                            "1",
                            "--t",
                            "1.0",
                            "--u",
                            values["u"],
                            "--dv",
                            values["dv"],
                            "--omega0",
                            values["omega0"],
                            "--g-ep",
                            values["g_ep"],
                            "--n-ph-max",
                            str(cutoff),
                            "--boson-encoding",
                            "binary",
                            "--ordering",
                            "blocked",
                            "--boundary",
                            "open",
                        ),
                        baseline_abs_delta_e=1e-4,
                        baseline_count_2q=1400,
                        baseline_depth_2q=4500,
                        baseline_parameter_count=64,
                        baseline_shot_cost_proxy=64,
                        exact_reference_n_ph_max=resolved_reference_cutoff(family),
                        split=split,
                        tags=("static_phase3", "size_sweep", "spin_boson_l1_only", f"nph{cutoff}", f"physics_profile:{profile}", *variant_tags),
                    )
                )
    if profile == "paper_i_clean":
        family = "molecular_vibronic_h2"
        get_problem_family_spec(family)
        if boson_cutoffs is not None and int(exact_reference_boson_cutoff or 0) == 6:
            h2_cutoff_values = tuple(dict.fromkeys(int(value) for value in boson_cutoffs))
            h2_reference_cutoff = 6
        else:
            h2_cutoff_values = (1,)
            h2_reference_cutoff = 4
        h2_omega_au = "0.022328470326434775"
        for h2_cutoff in h2_cutoff_values:
            h2_n_qubits = 4 + int(boson_qubits_per_site(int(h2_cutoff), "binary"))
            for regime, g_ep, regime_tags in (
                ("weak", "0.25", ("weak", "g0p25")),
                ("strong", "1.0", ("strong", "g1")),
            ):
                variant_suffix = f"_nph{int(h2_cutoff)}_clean_{regime}"
                specs.append(
                    HamiltonianBenchmarkSpec(
                        benchmark_id=f"{family}_L2{variant_suffix}",
                        family=family,
                        features=ProblemFeatureVector(
                            problem=family,
                            size_label=f"L2{variant_suffix}",
                            L=2,
                            n_qubits=h2_n_qubits,
                            pool_size_hint=13,
                            spinful=True,
                            bosonic=True,
                            molecular=True,
                        ),
                        base_pipeline_args=(
                            "--problem",
                            family,
                            "--L",
                            "2",
                            "--t",
                            "1.0",
                            "--u",
                            "0.0",
                            "--dv",
                            "0.0",
                            "--omega0",
                            h2_omega_au,
                            "--g-ep",
                            str(g_ep),
                            "--n-ph-max",
                            str(h2_cutoff),
                            "--boson-encoding",
                            "binary",
                            "--ordering",
                            "blocked",
                            "--boundary",
                            "open",
                            "--v-nn",
                            "0.0",
                            "--t-prime",
                            "0.0",
                        ),
                        baseline_abs_delta_e=1e-4,
                        baseline_count_2q=1000,
                        baseline_depth_2q=3000,
                        baseline_parameter_count=64,
                        baseline_shot_cost_proxy=64,
                        exact_reference_n_ph_max=h2_reference_cutoff,
                        split=split,
                        tags=(
                            "static_phase3",
                            "size_sweep",
                            "molecular_vibronic",
                            f"nph{h2_cutoff}",
                            f"ref{h2_reference_cutoff}",
                            "physics_profile:paper_i_clean",
                            "physics_clean",
                            *regime_tags,
                            f"nph{h2_cutoff}_ref{h2_reference_cutoff}",
                        ),
                    )
                )
    if profile != "paper_i_clean":
        family = "molecular_vibronic_h2o"
        get_problem_family_spec(family)
        if boson_cutoffs is None:
            h2o_cutoff_values = (1,)
        else:
            h2o_cutoff_values = tuple(dict.fromkeys(int(value) for value in boson_cutoffs))
        h2o_reference_cutoff = None if exact_reference_boson_cutoff is None else int(exact_reference_boson_cutoff)
        h2o_omega_au = "0.017"
        for h2o_cutoff in h2o_cutoff_values:
            h2o_n_qubits = 4 + int(boson_qubits_per_site(int(h2o_cutoff), "binary"))
            cutoff_suffix = f"_nph{int(h2o_cutoff)}"
            specs.append(
                HamiltonianBenchmarkSpec(
                    benchmark_id=f"{family}_active2{cutoff_suffix}",
                    family=family,
                    features=ProblemFeatureVector(
                        problem=family,
                        size_label=f"active2{cutoff_suffix}",
                        L=2,
                        n_qubits=h2o_n_qubits,
                        pool_size_hint=16,
                        spinful=True,
                        bosonic=True,
                        molecular=True,
                    ),
                    base_pipeline_args=(
                        "--problem",
                        family,
                        "--L",
                        "2",
                        "--t",
                        "1.0",
                        "--u",
                        "0.0",
                        "--dv",
                        "0.0",
                        "--omega0",
                        h2o_omega_au,
                        "--g-ep",
                        "1.0",
                        "--n-ph-max",
                        str(h2o_cutoff),
                        "--boson-encoding",
                        "binary",
                        "--ordering",
                        "blocked",
                        "--boundary",
                        "open",
                        "--v-nn",
                        "0.0",
                        "--t-prime",
                        "0.0",
                    ),
                    baseline_abs_delta_e=1e-4,
                    baseline_count_2q=1000,
                    baseline_depth_2q=3000,
                    baseline_parameter_count=64,
                    baseline_shot_cost_proxy=64,
                    exact_reference_n_ph_max=h2o_reference_cutoff,
                    split=split,
                    tags=(
                        "static_phase3",
                        "size_sweep",
                        "molecular_vibronic",
                        "h2o",
                        "active_space_prototype",
                        f"nph{h2o_cutoff}",
                        f"physics_profile:{profile}",
                    ),
                )
            )
    for molecular_asset in _resolve_default_molecular_benchmark_assets(molecular_problem_json):
        family = "molecular_restricted_closed_shell"
        get_problem_family_spec(family)
        molecular_label = molecular_asset.label
        molecular_path = molecular_asset.path
        n_spatial, n_spin = _molecular_problem_dimensions(molecular_path)
        label_suffix = "" if molecular_label in {None, ""} else f"_{molecular_label}"
        size_label = f"L{n_spatial}" if molecular_label in {None, ""} else f"{molecular_label}_L{n_spatial}"
        tags = tuple(
            dict.fromkeys(
                (
                    "static_phase3",
                    "size_sweep",
                    "molecular",
                    "closed_shell",
                    str(molecular_asset.tag),
                    f"{_MOLECULAR_ROLE_TAG_PREFIX}{molecular_asset.role}",
                )
            )
        )
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=f"molecular_restricted_closed_shell{label_suffix}_L{n_spatial}",
                family=family,
                features=ProblemFeatureVector(
                    problem=family,
                    size_label=size_label,
                    L=int(n_spatial),
                    n_qubits=int(n_spin),
                    pool_size_hint=max(64, int(24 * n_spin)),
                    spinful=True,
                    bosonic=False,
                    molecular=True,
                ),
                base_pipeline_args=(
                    "--problem",
                    family,
                    "--molecular-problem-json",
                    str(molecular_path),
                    "--L",
                    str(n_spatial),
                    "--ordering",
                    "blocked",
                    "--boundary",
                    "open",
                    "--dense-eigh-max-dim",
                    str(_MOLECULAR_DENSE_EIGH_MAX_DIM),
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=64,
                split=split,
                tags=tags,
            )
        )
    return tuple(specs)
