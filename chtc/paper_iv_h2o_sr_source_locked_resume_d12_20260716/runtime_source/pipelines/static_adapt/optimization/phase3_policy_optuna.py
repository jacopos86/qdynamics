#!/usr/bin/env python3
"""Size-aware Optuna policy surface for phase3 static ADAPT.

This module defines the outer optimization layer: Optuna samples algorithm
policy hyperparameters, then the existing static ADAPT and compile-scout
pipelines perform the physics and circuit evaluations.
"""
from __future__ import annotations

import json
import math
import os
import signal
import shlex
import subprocess
import sys
import time
import argparse
import hashlib
import inspect
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.noise_oracle_defaults import gate_tuple_to_cli_value, normalize_gate_name_tuple
from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec
from pipelines.static_adapt.builders.hh_pool_presets import HH_MATH_MD_FULL_META_POOL_KEY
from pipelines.static_adapt.builders.problem_registry import get_problem_family_spec
from pipelines.static_adapt.builders.problem_setup import (
    _exact_gs_energy_for_problem,
    build_problem_hamiltonian,
)
from pipelines.static_adapt.cli_config import _resolve_value_noise_std_contract
from pipelines.static_adapt.plateau_acquisition import (
    PLATEAU_ACQUISITION_SCORE_CHOICES,
    PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
    PLATEAU_ACQUISITION_MODE_OFF,
    PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
    PLATEAU_TRIAL_OPTIMIZER_CHOICES,
    PLATEAU_TRIAL_OPTIMIZER_INHERIT,
    normalize_plateau_acquisition_config,
)
from pipelines.scaffold.hh_continuation_motifs import load_selected_logical_library_from_payload
from pipelines.static_adapt.output_artifacts import extract_adapt_energy_metrics
from pipelines.static_adapt.runtime_heartbeat import (
    LiveHeartbeatRecorder,
    normalize_ai_log_progress,
    parse_ai_log_line,
)
from pipelines.static_adapt.route_identity import (
    LEGACY_ROUTE_ID_CHOICES,
    ROUTE_ID_A,
    ROUTE_ID_UNSPECIFIED,
    STATIC_META_FEATURE_PROFILE_CHOICES,
    STATIC_META_FEATURE_PROFILE_OFF,
    STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
    STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1,
    normalize_static_meta_feature_profile,
    normalize_static_route_id,
)
from pipelines.static_adapt.lane_routes import (
    PHYSICAL_LANE_SHORTLIST_AGGRESSIVENESS_CHOICES,
    STATIC_LANE_ROUTE_ALGEBRAIC,
    STATIC_LANE_ROUTE_CHOICES,
    normalize_physical_lane_shortlist_aggressiveness,
    normalize_static_lane_route,
)
from pipelines.static_adapt.selector_measurement_proxy import (
    controller_proxy_from_adapt_payload,
    validate_controller_proxy_for_shot_objective,
)
from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site
from src.quantum.chemistry.psi4_adapter import load_restricted_closed_shell_problem_from_json

_PIPELINE_NAME = "phase3_policy_optuna_v1"
_LARGE_OBJECTIVE = float(10**18)
_ZERO_NOISE_ORACLE_INNER_EXACT_GUARD_REASON = "zero_noise_noisy_v1_exact_equivalent_guard_v1"
_VALUE_NOISE_ORACLE_INNER_EXACT_STRUCTURE_REASON = "value_noise_noisy_v1_exact_structure_scalar_guard_v1"
_VALUE_NOISE_ORACLE_INNER_EXACT_STRUCTURE_MODE = "phase3_exact_structure_plus_value_noise_v1"
_DEFAULT_FAMILIES = (
    "hubbard",
    "hh",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "spin_boson",
    "molecular_vibronic_h2",
    "molecular_vibronic_h2o",
    "molecular_restricted_closed_shell",
)
_CANONICAL_LANE_FAMILIES: dict[str, tuple[str, ...]] = {
    "fermionic": ("hubbard", "ionic_hubbard", "extended_hubbard", "ttprime_hubbard", "spinless_tv"),
    "bosonic": ("bose_hubbard", "harmonic_kerr_chain", "spin_boson"),
    "mixed": ("hh", "molecular_vibronic_h2", "molecular_vibronic_h2o"),
    "molecular": ("molecular_restricted_closed_shell",),
}
_POLICY_SEARCH_PROFILES = (
    "default",
    "fermionic_protected_correlation",
    "bosonic_fullmeta_compact",
    "spin_boson_2q_batching_v1",
    "route_a_batching_v1",
    "hh_novelty_surface_v1",
    "snake_u8_no_novelty_v1",
    "snake_u8_flat_novelty_v1",
    "snake_u8_exponent_novelty_v1",
)
_META_FEATURE_PROFILE_OFF = STATIC_META_FEATURE_PROFILE_OFF
_META_FEATURE_PROFILE_SAFE_CORE = STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1
_META_FEATURE_PROFILE_PAPER_I_PRODUCTION = STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1
_META_FEATURE_PROFILES = STATIC_META_FEATURE_PROFILE_CHOICES
_DEFAULT_META_FEATURE_PROFILE = _META_FEATURE_PROFILE_SAFE_CORE
_PAPER_I_PRODUCTION_LEAKAGE_LAMBDA = 0.0
_PAPER_I_PRODUCTION_LEAKAGE_CAP = 1e6
_PHASE2_NOVELTY_MODE_CHOICES = ("collective_span_v1", "legacy_pairwise_v1")
_DEFAULT_PHASE2_NOVELTY_MODE = "collective_span_v1"
_ACTIVE_PHASE2_NOVELTY_MODE = os.environ.get("PHASE3_POLICY_PHASE2_NOVELTY_MODE", _DEFAULT_PHASE2_NOVELTY_MODE).strip().lower()
if _ACTIVE_PHASE2_NOVELTY_MODE not in _PHASE2_NOVELTY_MODE_CHOICES:
    raise ValueError(
        "PHASE3_POLICY_PHASE2_NOVELTY_MODE must be one of "
        f"{_PHASE2_NOVELTY_MODE_CHOICES}; got {_ACTIVE_PHASE2_NOVELTY_MODE!r}."
    )
_REQUIRED_TARGET_PROFILES = ("none", "fermionic_hubbard_core", "paper_i_phys_v1")
_DISCOVERY_OBJECTIVE_MODE_TERMINAL_PROXY = "terminal_proxy"
_DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING = "discovery_first_crossing"
_MULTI_OBJECTIVE_MODE_OFF = "off"
_MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO = "same_cutoff_pareto"
_MULTI_OBJECTIVE_MODES = (_MULTI_OBJECTIVE_MODE_OFF, _MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO)
_SAME_CUTOFF_PARETO_OBJECTIVE_NAMES = (
    "same_cutoff_abs_delta_e",
    "count_2q",
    "depth_2q",
    "circuit_depth",
    "parameter_count",
    "shot_cost_proxy",
)
_BENCHMARK_TARGET_HIT_STOP_REASON = "benchmark_abs_delta_e_target"
_DISCOVERY_OBJECTIVE_MODES = (
    _DISCOVERY_OBJECTIVE_MODE_TERMINAL_PROXY,
    _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING,
)
_CALIBRATION_PROFILE_NPH2_ROUTE_A_HK_HH = "nph2_route_a_hk_hh_v1"
_CALIBRATION_PROFILE_NPH2_REF3_ROUTE_A_BOSONIC_MIXED_WEIGHTED = "nph2_ref3_route_a_bosonic_mixed_weighted_v1"
_CALIBRATION_PROFILE_CHOICES = (
    "off",
    _CALIBRATION_PROFILE_NPH2_ROUTE_A_HK_HH,
    _CALIBRATION_PROFILE_NPH2_REF3_ROUTE_A_BOSONIC_MIXED_WEIGHTED,
)
_NPH2_ROUTE_A_HK_HH_CALIBRATION_BENCHMARK_IDS = (
    "harmonic_kerr_chain_L2_nph2",
    "harmonic_kerr_chain_L2_nph2_w0p75",
    "hh_L2_nph2",
)
_NPH2_REF3_ROUTE_A_BOSONIC_MIXED_WEIGHTED_BENCHMARK_IDS = (
    "bose_hubbard_L2_nph2",
    "bose_hubbard_L2_nph2_u2",
    "harmonic_kerr_chain_L2_nph2",
    "harmonic_kerr_chain_L2_nph2_w0p75",
    "spin_boson_L1_nph2",
    "spin_boson_L1_nph2_g0p7",
    "hh_L2_nph2",
)
_OBJECTIVE_WEIGHT_PRESET_UNIFORM = "uniform"
_OBJECTIVE_WEIGHT_PRESET_NPH2_REF3_BOSONIC_MIXED = "nph2_ref3_bosonic_mixed_v1"
_OBJECTIVE_WEIGHT_PRESETS: dict[str, Mapping[str, float]] = {
    _OBJECTIVE_WEIGHT_PRESET_UNIFORM: {},
    _OBJECTIVE_WEIGHT_PRESET_NPH2_REF3_BOSONIC_MIXED: {
        "harmonic_kerr_chain": 2.0,
        "hh": 1.5,
        "bose_hubbard": 1.0,
        "spin_boson": 0.5,
    },
}
_FERMIONIC_HUBBARD_CORE_REQUIRED_IDS = (
    "hubbard_L2",
    "hubbard_L2_u6",
    "ttprime_hubbard_L2",
    "ttprime_hubbard_L2_u6",
)
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
_ARCHIVAL_INNER_OPTIMIZERS = ("POWELL", "COBYLA", "SPSA", "QNSPSA")
_SPSA_SCHEDULE_INNER_OPTIMIZERS = frozenset({"SPSA", "QNSPSA"})
_ACTIVE_INNER_OPTIMIZER = os.environ.get("PHASE3_POLICY_INNER_OPTIMIZER", "SPSA").strip().upper()
if _ACTIVE_INNER_OPTIMIZER not in _ARCHIVAL_INNER_OPTIMIZERS:
    raise ValueError(
        "PHASE3_POLICY_INNER_OPTIMIZER must be one of "
        f"{_ARCHIVAL_INNER_OPTIMIZERS}; got {_ACTIVE_INNER_OPTIMIZER!r}."
    )
_INNER_OPTIMIZER_POLICY_LABEL = f"fixed_{_ACTIVE_INNER_OPTIMIZER.lower()}_phase3_v1"
_DEFAULT_PHASE3_SELECTOR_POLICY = "algebraic_nested_v1"
_PHASE3_SELECTOR_POLICY_CHOICES = ("algebraic_nested_v1", "hardware_resolvable_v1", "legacy_phase3_v1")
_PHASE3_SELECTOR_GEOMETRY_MODE_CHOICES = ("reduced", "proxy_reduced", "raw_exact")
_PHASE3_NOVELTY_ABLATION_MODE_CHOICES = ("off", "no_phase2", "no_phase3", "all")
_PHASE3_WINDOW_RELAXATION_MODE_CHOICES = ("reduced", "no_relaxation")
_PHASE3_BATCH_SELECTION_MODE_CHOICES = (
    "reduced_plane",
    "greedy_reduced_plane",
    "combinatorial_reduced_plane",
    "tetris_disjoint_benchmark",
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
_PHASE1_PRUNE_POLICY_CHOICES = (_DEFAULT_PHASE1_PRUNE_POLICY,)
_ARCHIVAL_PHASE1_PRUNE_POLICY_CHOICES = (_DEFAULT_PHASE1_PRUNE_POLICY, "legacy_small_angle_v1")
_DEFAULT_ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD = 0.10
_DEFAULT_ALGEBRAIC_PHASE1_LANE_QUOTA_PRESSURE = 0.70
_DEFAULT_ALGEBRAIC_PHASE2_LANE_QUOTA_PRESSURE = 0.70
_ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD_RANGE = (0.0, 0.35)
_ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE = (0.25, 1.0)
PHASE0_OPTUNA_DEFAULTS: dict[str, Any] = {
    "phase0_pilot_enabled": True,
    "phase0_pilot_alpha": 0.1,
    "phase0_pilot_threshold": 0.0,
    "phase0_pilot_max_records": 96,
    "phase0_lane_quota_pressure": 0.7,
    "phase0_algebraic_lane_mode": "weak",
}
PHASE0_PILOT_MAX_RECORDS_CHOICES = (24, 48, 96, 192)
PHASE0_PILOT_THRESHOLD_CHOICES = (0.0, 1e-5, 5e-5, 1e-4, 5e-4)
PHASE0_ALGEBRAIC_LANE_MODE_CHOICES = ("weak", "off")
_FORCE_ACTIVE_INNER_OPTIMIZER = "__active_phase3_policy_inner_optimizer__"
_SPSA_SCHEDULE_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "spsa_a": (1e-3, 1.0),
    "spsa_c": (1e-3, 1.0),
    "spsa_A": (0.0, 100.0),
    "spsa_alpha": (0.5, 0.9),
    "spsa_gamma": (0.05, 0.2),
}
_SPSA_CLI_OPTIONS: dict[str, str] = {
    "spsa_a": "adapt_spsa_a",
    "spsa_c": "adapt_spsa_c",
    "spsa_A": "adapt_spsa_A",
    "spsa_alpha": "adapt_spsa_alpha",
    "spsa_gamma": "adapt_spsa_gamma",
}


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


def _resolve_default_molecular_problem_jsons(path: str | Path | None) -> tuple[tuple[str | None, Path], ...]:
    return tuple(
        (asset.label, asset.path)
        for asset in _resolve_default_molecular_benchmark_assets(path)
    )


def _resolve_default_molecular_problem_json(path: str | Path | None) -> Path | None:
    resolved = _resolve_default_molecular_benchmark_assets(path)
    if not resolved:
        return None
    for asset in resolved:
        if asset.role in {_MOLECULAR_ROLE_TRAIN_PRIMARY, _MOLECULAR_ROLE_OVERRIDE_TRAIN}:
            return asset.path
    return resolved[0].path


def _molecular_role_for_spec(spec: "HamiltonianBenchmarkSpec") -> str:
    for tag in tuple(str(tag) for tag in spec.tags):
        tag_key = tag.strip().lower()
        if tag_key.startswith(_MOLECULAR_ROLE_TAG_PREFIX):
            role = tag_key.split(":", 1)[1]
            if role in {
                _MOLECULAR_ROLE_CONTROL,
                _MOLECULAR_ROLE_TRAIN_PRIMARY,
                _MOLECULAR_ROLE_TRANSFER,
                _MOLECULAR_ROLE_OVERRIDE_TRAIN,
            }:
                return role

    # Backward-compatible classification for legacy specs created before
    # explicit molecular_role:* tags existed.
    legacy_identity = " ".join((str(spec.benchmark_id), *(str(tag) for tag in spec.tags))).lower()
    if "h2o_sto3g" in legacy_identity:
        return _MOLECULAR_ROLE_TRANSFER
    if "lih_sto3g" in legacy_identity:
        return _MOLECULAR_ROLE_TRAIN_PRIMARY
    return _MOLECULAR_ROLE_CONTROL


def _molecular_problem_dimensions(path: Path) -> tuple[int, int]:
    problem = load_restricted_closed_shell_problem_from_json(Path(path))
    n_spatial = int(problem.n_spatial_orbitals)
    n_spin = int(problem.n_spin_orbitals)
    if n_spatial <= 0 or n_spin <= 0:
        raise ValueError(f"Molecular problem JSON {path} is missing positive orbital dimensions.")
    return n_spatial, n_spin


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


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


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        return ()
    max_logit = max(float(x) for x in logits)
    exps = [math.exp(float(x) - max_logit) for x in logits]
    total = sum(exps)
    if total <= 0.0:
        return tuple(1.0 / len(exps) for _ in exps)
    return tuple(float(x / total) for x in exps)


def _cvar(values: Sequence[float], quantile: float) -> float:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return _LARGE_OBJECTIVE
    q = min(max(float(quantile), 0.0), 1.0)
    start = int(math.floor(q * (len(finite) - 1)))
    tail = finite[start:]
    return float(sum(tail) / len(tail))


def _import_optuna() -> Any:
    try:
        import optuna  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        raise RuntimeError("Optuna is required for phase3 policy studies. Install optuna or run the library helpers only.") from exc
    return optuna


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
    """Transferable pool controls; these bias family behavior without exact label masks."""

    pool_key: str = "full_meta"
    family_prior_hopping: float = 1.0
    family_prior_onsite: float = 1.0
    family_prior_density: float = 1.0
    family_prior_quadrature: float = 1.0
    family_prior_assisted: float = 1.0
    family_prior_bridge: float = 1.0
    family_prior_boson: float = 1.0
    family_repeat_penalty: float = 1.0
    novelty_bonus: float = 0.05
    phase1_budget: SizeScaledBudget = field(default_factory=lambda: SizeScaledBudget(24, 256, 0.35, 8.0))
    phase2_budget: SizeScaledBudget = field(default_factory=lambda: SizeScaledBudget(12, 128, 0.25, 6.0))
    rescue_expand_factor: float = 2.0


@dataclass(frozen=True)
class StaticScaffoldPolicy:
    """Static phase3 ADAPT policy knobs exposed to the outer optimizer."""

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
    phase0_pilot_enabled: bool = bool(PHASE0_OPTUNA_DEFAULTS["phase0_pilot_enabled"])
    phase0_pilot_alpha: float = float(PHASE0_OPTUNA_DEFAULTS["phase0_pilot_alpha"])
    phase0_pilot_threshold: float = float(PHASE0_OPTUNA_DEFAULTS["phase0_pilot_threshold"])
    phase0_pilot_max_records: int = int(PHASE0_OPTUNA_DEFAULTS["phase0_pilot_max_records"])
    phase0_lane_quota_pressure: float = float(PHASE0_OPTUNA_DEFAULTS["phase0_lane_quota_pressure"])
    phase0_algebraic_lane_mode: str = str(PHASE0_OPTUNA_DEFAULTS["phase0_algebraic_lane_mode"])
    algebraic_phase2_lane_rel_threshold: float = _DEFAULT_ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD
    algebraic_phase1_lane_quota_pressure: float = _DEFAULT_ALGEBRAIC_PHASE1_LANE_QUOTA_PRESSURE
    algebraic_phase2_lane_quota_pressure: float = _DEFAULT_ALGEBRAIC_PHASE2_LANE_QUOTA_PRESSURE
    phase2_novelty_mode: str = _DEFAULT_PHASE2_NOVELTY_MODE
    phase2_gamma_N: float = 1.0
    phase2_gamma_N_schedule_mode: str = "fixed"
    phase2_gamma_N_schedule_start: float | None = None
    phase2_gamma_N_schedule_end: float | None = None
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
    phase1_prune_stale_age: int = 2
    phase1_prune_stagnation_threshold: float = 0.0
    phase1_prune_small_theta_abs: float = 1e-3
    phase1_prune_small_theta_relative: float = 0.5
    phase1_prune_cooldown_steps: int = 2
    phase1_prune_local_window_size: int = 4
    phase1_prune_recovery_trust_radius: float = 0.0
    phase1_prune_old_fraction: float = 0.25
    phase1_prune_checkpoint_period: int = 3
    phase1_prune_live_min_depth: int = 0
    phase1_prune_maturity_threshold: float = 0.5
    phase1_prune_snr_threshold: float = 1.0
    phase1_prune_amplitude_witness_required: bool = True
    phase1_prune_collapse_peak_abs_min: float = 1e-3
    phase1_prune_collapse_current_abs_max: float = 1e-3
    phase1_prune_collapse_ratio: float = 0.25
    phase1_prune_collapse_min_abs_drop: float = 1e-3
    phase1_prune_collapse_min_observations: int = 3
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
    phase_live_hysteresis_enabled: bool = True
    phase2_null_nrem_high_threshold: float = 0.0
    phase2_live_nrem_low_threshold: float = 0.25
    phase3_null_nrem_high_threshold: float = 0.5
    phase3_live_nrem_low_threshold: float = 1.0
    phase2_hysteresis_steps: int = 2
    phase3_hysteresis_steps: int = 2
    adapt_max_depth: int = 96
    adapt_maxiter: int = 4000
    adapt_drop_floor: float = 1e-8
    adapt_drop_patience: int = 12
    adapt_drop_min_depth: int = 16
    adapt_eps_grad: float = 1e-9
    adapt_eps_energy: float = 1e-13
    adapt_parallel_gradient_workers: int = 1
    adapt_beam_parent_workers: int = 1
    phase3_selector_policy: str = _DEFAULT_PHASE3_SELECTOR_POLICY
    phase3_selector_geometry_mode: str = "reduced"
    phase3_novelty_ablation_mode: str = "off"
    phase3_window_relaxation_mode: str = "reduced"
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
    static_meta_feature_profile: str = _META_FEATURE_PROFILE_OFF
    static_route_id: str = ROUTE_ID_A
    static_lane_route: str = STATIC_LANE_ROUTE_ALGEBRAIC
    physical_lane_shortlist_aggressiveness: int = 3
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
    inner_optimizer: str = _ACTIVE_INNER_OPTIMIZER
    final_optimizer_type: str = _ACTIVE_INNER_OPTIMIZER
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
    """Return the active Phase3 Optuna inner optimizer, or fail closed."""

    raw = _ACTIVE_INNER_OPTIMIZER if value in {None, ""} else str(value).strip().upper()
    if raw != _ACTIVE_INNER_OPTIMIZER:
        raise ValueError(
            "This Phase3 policy Optuna process is fixed to "
            f"{_ACTIVE_INNER_OPTIMIZER}; received fixed inner optimizer {value!r}. "
            "Set PHASE3_POLICY_INNER_OPTIMIZER before process start for comparator runs."
        )
    return _ACTIVE_INNER_OPTIMIZER


def _is_spsa_optimizer(value: str | None) -> bool:
    """Return whether the optimizer consumes the ADAPT SPSA-schedule knobs."""

    return str(value or "").strip().upper() in _SPSA_SCHEDULE_INNER_OPTIMIZERS


def _normalize_meta_feature_profile(value: str | None) -> str:
    return normalize_static_meta_feature_profile(value, default=_DEFAULT_META_FEATURE_PROFILE)


def _is_paper_i_production_profile(meta_feature_profile: str | None) -> bool:
    return _normalize_meta_feature_profile(meta_feature_profile) == _META_FEATURE_PROFILE_PAPER_I_PRODUCTION


def _ensure_paper_i_production_phase2_novelty_mode(meta_feature_profile: str | None) -> None:
    if _is_paper_i_production_profile(meta_feature_profile) and _ACTIVE_PHASE2_NOVELTY_MODE != _DEFAULT_PHASE2_NOVELTY_MODE:
        raise ValueError(
            "paper_i_production_v1 requires Phase-II collective novelty "
            f"{_DEFAULT_PHASE2_NOVELTY_MODE!r}; this process has "
            f"PHASE3_POLICY_PHASE2_NOVELTY_MODE={_ACTIVE_PHASE2_NOVELTY_MODE!r}."
        )


def _sync_feature_params(out: dict[str, Any], *, meta_feature_profile: str) -> dict[str, Any]:
    """Mirror feature-prefixed Optuna choices into runtime policy params."""

    out["meta_feature_profile"] = str(meta_feature_profile)
    if _is_paper_i_production_profile(meta_feature_profile):
        _ensure_paper_i_production_phase2_novelty_mode(meta_feature_profile)
        batch_enabled = out.get("feature_phase3_batching_enabled", out.get("phase2_enable_batching", True))
        out["feature_phase0_pilot_enabled"] = True
        out["phase0_pilot_enabled"] = True
        out["feature_phase0_algebraic_lane_mode"] = "weak"
        out["phase0_algebraic_lane_mode"] = "weak"
        out["feature_phase3_batching_enabled"] = _coerce_bool(batch_enabled)
        out["phase2_enable_batching"] = _coerce_bool(batch_enabled)
        out["feature_phase1_prune_enabled"] = True
        out["phase1_prune_enabled"] = True
        out["feature_phase1_prune_amplitude_witness_required"] = True
        out["phase1_prune_amplitude_witness_required"] = True
        out["feature_phase3_selector_policy"] = _DEFAULT_PHASE3_SELECTOR_POLICY
        out["phase3_selector_policy"] = _DEFAULT_PHASE3_SELECTOR_POLICY
        out["phase3_selector_geometry_mode"] = "reduced"
        out["feature_phase3_novelty_ablation_mode"] = "off"
        out["phase3_novelty_ablation_mode"] = "off"
        out["feature_phase3_window_relaxation_mode"] = "reduced"
        out["phase3_window_relaxation_mode"] = "reduced"
        out["feature_phase3_batch_selection_mode"] = "reduced_plane"
        out["phase3_batch_selection_mode"] = "reduced_plane"
        out["feature_phase3_batch_prefilter_mode"] = "off"
        out["phase3_batch_prefilter_mode"] = "off"
        return out
    phase0_enabled = out.get("feature_phase0_pilot_enabled", out.get("phase0_pilot_enabled", True))
    out["feature_phase0_pilot_enabled"] = _coerce_bool(phase0_enabled)
    out["phase0_pilot_enabled"] = _coerce_bool(phase0_enabled)
    phase0_lane = str(
        out.get(
            "feature_phase0_algebraic_lane_mode",
            out.get("phase0_algebraic_lane_mode", PHASE0_OPTUNA_DEFAULTS["phase0_algebraic_lane_mode"]),
        )
    ).strip().lower()
    if phase0_lane not in set(PHASE0_ALGEBRAIC_LANE_MODE_CHOICES):
        phase0_lane = str(PHASE0_OPTUNA_DEFAULTS["phase0_algebraic_lane_mode"])
    out["feature_phase0_algebraic_lane_mode"] = phase0_lane
    out["phase0_algebraic_lane_mode"] = phase0_lane
    if meta_feature_profile == _META_FEATURE_PROFILE_SAFE_CORE:
        batch_enabled = out.get("feature_phase3_batching_enabled", out.get("phase2_enable_batching", True))
        prune_enabled = out.get("feature_phase1_prune_enabled", out.get("phase1_prune_enabled", True))
        witness_required = out.get(
            "feature_phase1_prune_amplitude_witness_required",
            out.get("phase1_prune_amplitude_witness_required", True),
        )
        out["feature_phase3_batching_enabled"] = _coerce_bool(batch_enabled)
        out["phase2_enable_batching"] = _coerce_bool(batch_enabled)
        out["feature_phase1_prune_enabled"] = _coerce_bool(prune_enabled)
        out["phase1_prune_enabled"] = _coerce_bool(prune_enabled)
        out["feature_phase1_prune_amplitude_witness_required"] = _coerce_bool(witness_required)
        out["phase1_prune_amplitude_witness_required"] = _coerce_bool(witness_required)
        for feature_key, policy_key, choices, default in (
            ("feature_phase3_selector_policy", "phase3_selector_policy", _PHASE3_SELECTOR_POLICY_CHOICES, _DEFAULT_PHASE3_SELECTOR_POLICY),
            ("feature_phase3_novelty_ablation_mode", "phase3_novelty_ablation_mode", _PHASE3_NOVELTY_ABLATION_MODE_CHOICES, "off"),
            ("feature_phase3_window_relaxation_mode", "phase3_window_relaxation_mode", _PHASE3_WINDOW_RELAXATION_MODE_CHOICES, "reduced"),
            ("feature_phase3_batch_selection_mode", "phase3_batch_selection_mode", _PHASE3_BATCH_SELECTION_MODE_CHOICES, "reduced_plane"),
            ("feature_phase3_batch_prefilter_mode", "phase3_batch_prefilter_mode", _PHASE3_BATCH_PREFILTER_MODE_CHOICES, "off"),
        ):
            value = str(out.get(feature_key, out.get(policy_key, default))).strip().lower()
            if value not in set(choices):
                value = str(default)
            out[feature_key] = value
            out[policy_key] = value
    else:
        out["feature_phase3_batching_enabled"] = True
        out["phase2_enable_batching"] = True
        out["feature_phase1_prune_enabled"] = True
        out["phase1_prune_enabled"] = True
        out["feature_phase1_prune_amplitude_witness_required"] = True
        out["phase1_prune_amplitude_witness_required"] = True
        out["feature_phase3_selector_policy"] = _DEFAULT_PHASE3_SELECTOR_POLICY
        out["phase3_selector_policy"] = _DEFAULT_PHASE3_SELECTOR_POLICY
        out["feature_phase3_novelty_ablation_mode"] = "off"
        out["phase3_novelty_ablation_mode"] = "off"
        out["feature_phase3_window_relaxation_mode"] = "reduced"
        out["phase3_window_relaxation_mode"] = "reduced"
        out["feature_phase3_batch_selection_mode"] = "reduced_plane"
        out["phase3_batch_selection_mode"] = "reduced_plane"
        out["feature_phase3_batch_prefilter_mode"] = "off"
        out["phase3_batch_prefilter_mode"] = "off"
    return out


def _normalize_active_policy(
    policy: AlgorithmPolicy,
    *,
    fixed_inner_optimizer: str | None = None,
    meta_feature_profile: str | None = None,
) -> AlgorithmPolicy:
    """Coerce generated policies to this process's fixed inner-optimizer route."""

    fixed = _normalize_fixed_inner_optimizer(fixed_inner_optimizer)
    inner = replace(policy.inner_optimizer, inner_optimizer=fixed, final_optimizer_type=fixed)
    static = policy.static
    meta_profile = normalize_static_meta_feature_profile(
        meta_feature_profile
        if meta_feature_profile is not None
        else getattr(static, "static_meta_feature_profile", _META_FEATURE_PROFILE_OFF),
        default=_META_FEATURE_PROFILE_OFF,
    )
    production_profile = meta_profile == _META_FEATURE_PROFILE_PAPER_I_PRODUCTION
    if production_profile:
        _ensure_paper_i_production_phase2_novelty_mode(meta_profile)
    pool = policy.pool
    declared_route_id = normalize_static_route_id(
        getattr(static, "static_route_id", ROUTE_ID_A),
        default=ROUTE_ID_A,
    )
    static_lane_route = normalize_static_lane_route(
        getattr(static, "static_lane_route", STATIC_LANE_ROUTE_ALGEBRAIC),
        default=STATIC_LANE_ROUTE_ALGEBRAIC,
    )
    physical_lane_shortlist_aggressiveness = normalize_physical_lane_shortlist_aggressiveness(
        getattr(static, "physical_lane_shortlist_aggressiveness", 3),
        default=3,
    )
    production_route_locked = bool(production_profile and declared_route_id == ROUTE_ID_A)
    if production_profile:
        pool = replace(pool, pool_key="full_meta", family_repeat_penalty=0.0, novelty_bonus=0.0)
    batch_cap = max(1, int(static.phase2_batch_size_cap))
    batch_target = min(max(1, int(static.phase2_batch_target_size)), int(batch_cap))
    prune_max_candidates = max(1, int(static.phase1_prune_max_candidates))
    prune_min_candidates = min(
        max(1, int(static.phase1_prune_min_candidates)),
        int(prune_max_candidates),
    )
    def _clamped_static_float(value: Any, *, default: float, low: float, high: float) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(default)
        if not math.isfinite(out):
            out = float(default)
        return float(max(float(low), min(float(high), out)))

    phase3_selector_policy = _DEFAULT_PHASE3_SELECTOR_POLICY
    phase3_selector_geometry_mode = "reduced"
    phase3_novelty_ablation_mode = "off"
    phase3_window_relaxation_mode = "reduced"
    phase3_batch_selection_mode = "reduced_plane"
    phase3_batch_prefilter_mode = "off"
    ablation_overrides_allowed = bool(
        meta_profile == _META_FEATURE_PROFILE_SAFE_CORE
        or (production_profile and not production_route_locked)
    )
    if ablation_overrides_allowed:
        raw_selector_policy = str(getattr(static, "phase3_selector_policy", _DEFAULT_PHASE3_SELECTOR_POLICY)).strip().lower()
        if raw_selector_policy in set(_PHASE3_SELECTOR_POLICY_CHOICES):
            phase3_selector_policy = raw_selector_policy
        raw_geometry = str(getattr(static, "phase3_selector_geometry_mode", "reduced")).strip().lower()
        if raw_geometry in set(_PHASE3_SELECTOR_GEOMETRY_MODE_CHOICES):
            phase3_selector_geometry_mode = raw_geometry
        raw_novelty_ablation = str(getattr(static, "phase3_novelty_ablation_mode", "off")).strip().lower()
        if raw_novelty_ablation in set(_PHASE3_NOVELTY_ABLATION_MODE_CHOICES):
            phase3_novelty_ablation_mode = raw_novelty_ablation
        raw_window = str(getattr(static, "phase3_window_relaxation_mode", "reduced")).strip().lower()
        if raw_window in set(_PHASE3_WINDOW_RELAXATION_MODE_CHOICES):
            phase3_window_relaxation_mode = raw_window
        raw_batch_selection = str(getattr(static, "phase3_batch_selection_mode", "reduced_plane")).strip().lower()
        if raw_batch_selection in set(_PHASE3_BATCH_SELECTION_MODE_CHOICES):
            phase3_batch_selection_mode = raw_batch_selection
        raw_prefilter = str(getattr(static, "phase3_batch_prefilter_mode", "off")).strip().lower()
        if raw_prefilter in set(_PHASE3_BATCH_PREFILTER_MODE_CHOICES):
            phase3_batch_prefilter_mode = raw_prefilter
    phase2_novelty_mode = _ACTIVE_PHASE2_NOVELTY_MODE
    phase2_enable_batching = _coerce_bool(getattr(static, "phase2_enable_batching", True))
    if meta_profile == _META_FEATURE_PROFILE_OFF:
        phase2_enable_batching = True
    phase1_prune_enabled = _coerce_bool(getattr(static, "phase1_prune_enabled", True))
    if meta_profile == _META_FEATURE_PROFILE_OFF or production_route_locked:
        phase1_prune_enabled = True
    phase1_prune_policy = _DEFAULT_PHASE1_PRUNE_POLICY
    phase1_prune_mode = "both"
    phase1_prune_amplitude_witness_required = _coerce_bool(
        getattr(static, "phase1_prune_amplitude_witness_required", True)
    )
    if meta_profile == _META_FEATURE_PROFILE_OFF:
        phase1_prune_amplitude_witness_required = True
    elif production_route_locked:
        phase1_prune_amplitude_witness_required = False
    algebraic_phase2_lane_rel_threshold = _clamped_static_float(
        getattr(static, "algebraic_phase2_lane_rel_threshold", _DEFAULT_ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD),
        default=_DEFAULT_ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD,
        low=_ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD_RANGE[0],
        high=_ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD_RANGE[1],
    )
    algebraic_phase1_lane_quota_pressure = _clamped_static_float(
        getattr(static, "algebraic_phase1_lane_quota_pressure", _DEFAULT_ALGEBRAIC_PHASE1_LANE_QUOTA_PRESSURE),
        default=_DEFAULT_ALGEBRAIC_PHASE1_LANE_QUOTA_PRESSURE,
        low=_ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE[0],
        high=_ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE[1],
    )
    algebraic_phase2_lane_quota_pressure = _clamped_static_float(
        getattr(static, "algebraic_phase2_lane_quota_pressure", _DEFAULT_ALGEBRAIC_PHASE2_LANE_QUOTA_PRESSURE),
        default=_DEFAULT_ALGEBRAIC_PHASE2_LANE_QUOTA_PRESSURE,
        low=_ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE[0],
        high=_ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE[1],
    )
    phase0_pilot_alpha = _clamped_static_float(
        getattr(static, "phase0_pilot_alpha", PHASE0_OPTUNA_DEFAULTS["phase0_pilot_alpha"]),
        default=float(PHASE0_OPTUNA_DEFAULTS["phase0_pilot_alpha"]),
        low=1e-12,
        high=10.0,
    )
    phase0_pilot_threshold = _clamped_static_float(
        getattr(static, "phase0_pilot_threshold", PHASE0_OPTUNA_DEFAULTS["phase0_pilot_threshold"]),
        default=float(PHASE0_OPTUNA_DEFAULTS["phase0_pilot_threshold"]),
        low=0.0,
        high=1.0,
    )
    phase0_lane_quota_pressure = _clamped_static_float(
        getattr(static, "phase0_lane_quota_pressure", PHASE0_OPTUNA_DEFAULTS["phase0_lane_quota_pressure"]),
        default=float(PHASE0_OPTUNA_DEFAULTS["phase0_lane_quota_pressure"]),
        low=0.0,
        high=1.0,
    )
    try:
        phase0_pilot_max_records = int(getattr(static, "phase0_pilot_max_records", PHASE0_OPTUNA_DEFAULTS["phase0_pilot_max_records"]))
    except Exception:
        phase0_pilot_max_records = int(PHASE0_OPTUNA_DEFAULTS["phase0_pilot_max_records"])
    phase0_pilot_max_records = max(0, min(100000, int(phase0_pilot_max_records)))
    phase0_algebraic_lane_mode = str(
        getattr(static, "phase0_algebraic_lane_mode", PHASE0_OPTUNA_DEFAULTS["phase0_algebraic_lane_mode"])
    ).strip().lower()
    if phase0_algebraic_lane_mode not in set(PHASE0_ALGEBRAIC_LANE_MODE_CHOICES):
        phase0_algebraic_lane_mode = str(PHASE0_OPTUNA_DEFAULTS["phase0_algebraic_lane_mode"])
    phase0_pilot_enabled = _coerce_bool(
        getattr(static, "phase0_pilot_enabled", PHASE0_OPTUNA_DEFAULTS["phase0_pilot_enabled"])
    )
    phase_live_hysteresis_enabled = _coerce_bool(getattr(static, "phase_live_hysteresis_enabled", True))
    compile_position_shift_weight = float(getattr(static, "compile_position_shift_weight", 0.0))
    lambda_leak = getattr(static, "lambda_leak", _PAPER_I_PRODUCTION_LEAKAGE_LAMBDA)
    phase2_leakage_cap = getattr(static, "phase2_leakage_cap", _PAPER_I_PRODUCTION_LEAKAGE_CAP)
    adapt_reopt_policy = str(getattr(static, "adapt_reopt_policy", "windowed")).strip().lower()
    adapt_insertion_mode = str(getattr(static, "adapt_insertion_mode", "adaptive")).strip().lower()
    if production_profile:
        phase0_pilot_enabled = True
        phase0_algebraic_lane_mode = "weak"
        phase_live_hysteresis_enabled = False
        compile_position_shift_weight = 0.0
        lambda_leak = _PAPER_I_PRODUCTION_LEAKAGE_LAMBDA
        phase2_leakage_cap = _PAPER_I_PRODUCTION_LEAKAGE_CAP
        if production_route_locked and adapt_reopt_policy == "append_only":
            adapt_reopt_policy = "windowed"
        if production_route_locked and adapt_insertion_mode == "append_only":
            adapt_insertion_mode = "adaptive"
    canonical_static = replace(
        static,
        static_meta_feature_profile=str(meta_profile),
        static_route_id=declared_route_id,
        static_lane_route=str(static_lane_route),
        physical_lane_shortlist_aggressiveness=int(physical_lane_shortlist_aggressiveness),
        phase2_batch_size_cap=int(batch_cap),
        phase2_batch_target_size=int(batch_target),
        phase1_prune_min_candidates=int(prune_min_candidates),
        phase1_prune_max_candidates=int(prune_max_candidates),
        phase2_novelty_mode=str(phase2_novelty_mode),
        phase2_enable_batching=bool(phase2_enable_batching),
        phase3_selector_policy=str(phase3_selector_policy),
        phase3_selector_geometry_mode=str(phase3_selector_geometry_mode),
        phase3_novelty_ablation_mode=str(phase3_novelty_ablation_mode),
        phase3_window_relaxation_mode=str(phase3_window_relaxation_mode),
        phase3_batch_selection_mode=str(phase3_batch_selection_mode),
        phase3_batch_prefilter_mode=str(phase3_batch_prefilter_mode),
        phase1_prune_enabled=bool(phase1_prune_enabled),
        phase1_prune_policy=str(phase1_prune_policy),
        phase1_prune_mode=str(phase1_prune_mode),
        phase1_prune_amplitude_witness_required=bool(phase1_prune_amplitude_witness_required),
        compile_position_shift_weight=float(compile_position_shift_weight),
        lambda_leak=lambda_leak,
        phase2_leakage_cap=phase2_leakage_cap,
        adapt_reopt_policy=str(adapt_reopt_policy),
        adapt_insertion_mode=str(adapt_insertion_mode),
        algebraic_phase2_lane_rel_threshold=float(algebraic_phase2_lane_rel_threshold),
        algebraic_phase1_lane_quota_pressure=float(algebraic_phase1_lane_quota_pressure),
        algebraic_phase2_lane_quota_pressure=float(algebraic_phase2_lane_quota_pressure),
        phase0_pilot_enabled=bool(phase0_pilot_enabled),
        phase0_pilot_alpha=float(phase0_pilot_alpha),
        phase0_pilot_threshold=float(phase0_pilot_threshold),
        phase0_pilot_max_records=int(phase0_pilot_max_records),
        phase0_lane_quota_pressure=float(phase0_lane_quota_pressure),
        phase0_algebraic_lane_mode=str(phase0_algebraic_lane_mode),
        phase_live_hysteresis_enabled=bool(phase_live_hysteresis_enabled),
    )
    if canonical_static != static:
        static = canonical_static
    if inner == policy.inner_optimizer and static == policy.static and pool == policy.pool:
        return policy
    return replace(policy, pool=pool, static=static, inner_optimizer=inner)


def _normalize_policy_search_profile(value: str | None) -> str:
    key = str(value or "default").strip().lower().replace("-", "_")
    if key not in set(_POLICY_SEARCH_PROFILES):
        raise ValueError(f"Unsupported policy search profile: {value!r}")
    return key


def _is_hh_novelty_surface_profile(value: str | None) -> bool:
    return _normalize_policy_search_profile(value) == "hh_novelty_surface_v1"


_SNAKE_U8_NOVELTY_PROFILES = frozenset(
    {
        "snake_u8_no_novelty_v1",
        "snake_u8_flat_novelty_v1",
        "snake_u8_exponent_novelty_v1",
    }
)


def _is_snake_u8_novelty_profile(value: str | None) -> bool:
    return _normalize_policy_search_profile(value) in _SNAKE_U8_NOVELTY_PROFILES


def _apply_policy_search_profile(policy: AlgorithmPolicy, profile: str | None) -> AlgorithmPolicy:
    """Apply coarse, non-label-memorizing search-profile constraints.

    The fermionic protected-correlation profile is deliberately not a motif
    bonus.  It keeps the same Hamiltonian-generic pool semantics but prevents
    the outer optimizer from selecting policies that spend all early budget on
    cheap one-body directions and then terminate before Hubbard-like
    correlation generators have received a fair insertion/refit opportunity.
    """

    profile_key = _normalize_policy_search_profile(profile)
    policy = _normalize_active_policy(policy)
    if profile_key == "default":
        return policy
    if profile_key == "hh_novelty_surface_v1":
        return policy
    if profile_key in _SNAKE_U8_NOVELTY_PROFILES:
        static = replace(
            policy.static,
            phase2_enable_batching=True,
            phase1_prune_enabled=True,
            phase1_prune_policy=_DEFAULT_PHASE1_PRUNE_POLICY,
            phase1_prune_mode="both",
            phase1_prune_amplitude_witness_required=False,
            phase2_motif_bonus_weight=0.0,
            phase3_selector_policy=_DEFAULT_PHASE3_SELECTOR_POLICY,
            phase3_selector_geometry_mode="reduced",
            phase3_window_relaxation_mode="reduced",
            phase3_batch_selection_mode="reduced_plane",
            phase3_batch_prefilter_mode="off",
            phase3_novelty_ablation_mode=("all" if profile_key == "snake_u8_no_novelty_v1" else "off"),
            compile_position_shift_weight=0.0,
            phase_live_hysteresis_enabled=False,
        )
        pool = replace(policy.pool, pool_key="full_meta", family_repeat_penalty=0.0, novelty_bonus=0.0)
        return _normalize_active_policy(replace(policy, pool=pool, static=static))
    if profile_key in {"spin_boson_2q_batching_v1", "route_a_batching_v1"}:
        static = replace(
            policy.static,
            phase2_enable_batching=True,
            phase2_batch_target_size=max(3, int(policy.static.phase2_batch_target_size)),
            phase2_batch_size_cap=max(8, int(policy.static.phase2_batch_size_cap)),
            phase1_prune_enabled=True,
            phase1_prune_policy=_DEFAULT_PHASE1_PRUNE_POLICY,
            phase1_prune_mode="both",
            phase1_prune_amplitude_witness_required=True,
        )
        return _normalize_active_policy(replace(policy, static=static))
    if profile_key == "bosonic_fullmeta_compact":
        production_profile = _is_paper_i_production_profile(policy.static.static_meta_feature_profile)
        pool = replace(
            policy.pool,
            pool_key="full_meta",
            family_repeat_penalty=float(max(float(policy.pool.family_repeat_penalty), 2.0)),
            novelty_bonus=float(max(float(policy.pool.novelty_bonus), 0.05)),
            phase1_budget=SizeScaledBudget(
                min_count=max(64, int(policy.pool.phase1_budget.min_count)),
                max_count=max(192, int(policy.pool.phase1_budget.max_count)),
                pool_fraction=max(0.75, float(policy.pool.phase1_budget.pool_fraction)),
                qubit_slope=max(8.0, float(policy.pool.phase1_budget.qubit_slope)),
            ),
            phase2_budget=SizeScaledBudget(
                min_count=max(32, int(policy.pool.phase2_budget.min_count)),
                max_count=max(96, int(policy.pool.phase2_budget.max_count)),
                pool_fraction=max(0.50, float(policy.pool.phase2_budget.pool_fraction)),
                qubit_slope=max(6.0, float(policy.pool.phase2_budget.qubit_slope)),
            ),
        )
        static = replace(
            policy.static,
            lambda_compile=float(min(float(policy.static.lambda_compile), 0.08)),
            lambda_measure=float(min(float(policy.static.lambda_measure), 0.08)),
            phase2_w_depth=float(min(float(policy.static.phase2_w_depth), 0.25)),
            phase2_w_group=float(min(float(policy.static.phase2_w_group), 0.25)),
            phase2_w_shot=float(min(float(policy.static.phase2_w_shot), 0.25)),
            phase2_w_optdim=float(min(float(policy.static.phase2_w_optdim), 0.20)),
            phase2_frontier_ratio=float(max(float(policy.static.phase2_frontier_ratio), 0.85)),
            phase3_frontier_ratio=float(max(float(policy.static.phase3_frontier_ratio), 0.85)),
            adapt_beam_live_branches=min(max(2, int(policy.static.adapt_beam_live_branches)), 4),
            adapt_beam_children_per_parent=min(max(2, int(policy.static.adapt_beam_children_per_parent)), 3),
            adapt_beam_terminated_keep=min(max(2, int(policy.static.adapt_beam_terminated_keep)), 4),
            adapt_reopt_policy=(
                "windowed"
                if str(policy.static.adapt_reopt_policy).strip().lower() == "append_only"
                else policy.static.adapt_reopt_policy
            ),
            adapt_window_size=max(64, int(policy.static.adapt_window_size)),
            adapt_window_topk=max(24, int(policy.static.adapt_window_topk)),
            adapt_insertion_mode=(
                "adaptive"
                if str(policy.static.adapt_insertion_mode).strip().lower() == "append_only"
                else policy.static.adapt_insertion_mode
            ),
            adapt_allow_repeats=bool(policy.static.adapt_allow_repeats) if production_profile else False,
            phase1_probe_max_positions=max(6, int(policy.static.phase1_probe_max_positions)),
            phase2_shortlist_fraction=max(0.35, float(policy.static.phase2_shortlist_fraction)),
            phase2_enable_batching=bool(policy.static.phase2_enable_batching),
            phase2_batch_target_size=min(max(3, int(policy.static.phase2_batch_target_size)), 6),
            phase2_batch_size_cap=min(max(6, int(policy.static.phase2_batch_size_cap)), 12),
            adapt_max_depth=max(64, int(policy.static.adapt_max_depth)),
            adapt_maxiter=max(2400, int(policy.static.adapt_maxiter)),
            adapt_drop_floor=float(min(float(policy.static.adapt_drop_floor), 1e-7)),
            adapt_drop_patience=max(12, int(policy.static.adapt_drop_patience)),
            adapt_drop_min_depth=max(24, int(policy.static.adapt_drop_min_depth)),
        )
        inner = replace(
            policy.inner_optimizer,
            refit_maxiter=max(2400, int(policy.inner_optimizer.refit_maxiter)),
            final_maxiter=max(2400, int(policy.inner_optimizer.final_maxiter)),
        )
        return _normalize_active_policy(replace(policy, pool=pool, static=static, inner_optimizer=inner))
    if profile_key != "fermionic_protected_correlation":  # defensive; normalized above
        raise ValueError(f"Unsupported policy search profile: {profile!r}")

    pool = replace(
        policy.pool,
        pool_key="full_meta",
        family_repeat_penalty=float(min(float(policy.pool.family_repeat_penalty), 1.0)),
        novelty_bonus=float(min(float(policy.pool.novelty_bonus), 0.05)),
        phase1_budget=SizeScaledBudget(
            min_count=max(64, int(policy.pool.phase1_budget.min_count)),
            max_count=max(256, int(policy.pool.phase1_budget.max_count)),
            pool_fraction=max(0.75, float(policy.pool.phase1_budget.pool_fraction)),
            qubit_slope=max(10.0, float(policy.pool.phase1_budget.qubit_slope)),
        ),
        phase2_budget=SizeScaledBudget(
            min_count=max(32, int(policy.pool.phase2_budget.min_count)),
            max_count=max(128, int(policy.pool.phase2_budget.max_count)),
            pool_fraction=max(0.50, float(policy.pool.phase2_budget.pool_fraction)),
            qubit_slope=max(8.0, float(policy.pool.phase2_budget.qubit_slope)),
        ),
    )
    static = replace(
        policy.static,
        lambda_compile=float(min(float(policy.static.lambda_compile), 0.08)),
        lambda_measure=float(min(float(policy.static.lambda_measure), 0.04)),
        phase2_w_depth=float(min(float(policy.static.phase2_w_depth), 0.30)),
        phase2_w_group=float(min(float(policy.static.phase2_w_group), 0.30)),
        phase2_w_shot=float(min(float(policy.static.phase2_w_shot), 0.30)),
        phase2_w_optdim=float(min(float(policy.static.phase2_w_optdim), 0.20)),
        phase2_frontier_ratio=float(max(float(policy.static.phase2_frontier_ratio), 0.90)),
        phase3_frontier_ratio=float(max(float(policy.static.phase3_frontier_ratio), 0.90)),
        adapt_reopt_policy=(
            "windowed"
            if str(policy.static.adapt_reopt_policy).strip().lower() == "append_only"
            else policy.static.adapt_reopt_policy
        ),
        adapt_insertion_mode=(
            "adaptive"
            if str(policy.static.adapt_insertion_mode).strip().lower() == "append_only"
            else policy.static.adapt_insertion_mode
        ),
        adapt_window_size=max(96, int(policy.static.adapt_window_size)),
        adapt_window_topk=max(32, int(policy.static.adapt_window_topk)),
        phase1_probe_max_positions=max(8, int(policy.static.phase1_probe_max_positions)),
        phase2_shortlist_fraction=max(0.50, float(policy.static.phase2_shortlist_fraction)),
        phase1_prune_protect_steps=max(12, int(policy.static.phase1_prune_protect_steps)),
        phase1_prune_collapse_min_observations=max(4, int(policy.static.phase1_prune_collapse_min_observations)),
        adapt_max_depth=max(96, int(policy.static.adapt_max_depth)),
        adapt_maxiter=max(3200, int(policy.static.adapt_maxiter)),
        adapt_drop_floor=float(min(float(policy.static.adapt_drop_floor), 1e-7)),
        adapt_drop_patience=max(16, int(policy.static.adapt_drop_patience)),
        adapt_drop_min_depth=max(32, int(policy.static.adapt_drop_min_depth)),
    )
    inner = replace(
        policy.inner_optimizer,
        refit_maxiter=max(2400, int(policy.inner_optimizer.refit_maxiter)),
        final_maxiter=max(2400, int(policy.inner_optimizer.final_maxiter)),
    )
    return _normalize_active_policy(replace(policy, pool=pool, static=static, inner_optimizer=inner))


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


@dataclass(frozen=True)
class StaticObjectiveWeights:
    energy: float = 1.0
    count_2q: float = 0.15
    depth_2q: float = 0.05
    circuit_depth: float = 0.05
    parameters: float = 0.03
    shot_cost: float = 0.0
    fail_penalty: float = 100.0
    epsilon_energy: float = 1e-12
    target_abs_delta_e: float | None = None


@dataclass(frozen=True)
class GlobalObjectiveConfig:
    robust_cvar_quantile: float = 0.8
    gamma_robust: float = 0.5
    gamma_family_std: float = 0.1
    gamma_fail: float = 20.0
    weights: StaticObjectiveWeights = field(default_factory=StaticObjectiveWeights)
    required_target_benchmark_ids: tuple[str, ...] = ()
    required_target_abs_delta_e: float | None = None
    required_target_penalty: float = 1000.0
    objective_weight_preset: str = _OBJECTIVE_WEIGHT_PRESET_UNIFORM
    objective_family_weights: Mapping[str, float] = field(default_factory=dict)
    objective_benchmark_weights: Mapping[str, float] = field(default_factory=dict)
    discovery_objective_mode: str = _DISCOVERY_OBJECTIVE_MODE_TERMINAL_PROXY
    multi_objective_mode: str = _MULTI_OBJECTIVE_MODE_OFF
    objective_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _normalize_objective_weight_preset(self.objective_weight_preset)
        _normalize_discovery_objective_mode(self.discovery_objective_mode)
        _normalize_multi_objective_mode(self.multi_objective_mode)
        _validate_objective_weight_map(self.objective_family_weights, label="objective_family_weights")
        _validate_objective_weight_map(self.objective_benchmark_weights, label="objective_benchmark_weights")


def _default_objective_provenance_payload() -> dict[str, Any]:
    return {
        "schema": "phase3_objective_provenance_v1",
        "objective_noise_mode": "exact_noiseless_v1",
        "objective_metric_source": "exact_final_energy_vs_same_or_reference_cutoff",
        "objective_consumes_noisy_energy": False,
        "objective_consumes_exact_final_state_energy": True,
        "phase3_oracle_inner_objective_mode": "exact",
        "phase3_oracle_value_noise_model": "off",
        "phase3_oracle_value_noise": {
            "enabled": False,
            "model": "off",
            "std": 0.0,
            "semantic": "post_expectation_value_noise_not_physical_shots",
            "physical_shots_unchanged": True,
            "fixed_gate_error_reduction_claimed": False,
        },
        "phase3_oracle_execution_surface": None,
    }


def objective_provenance_payload(config: GlobalObjectiveConfig | None = None) -> dict[str, Any]:
    resolved_config = config if config is not None else GlobalObjectiveConfig()
    payload = _default_objective_provenance_payload()
    payload.update(dict(resolved_config.objective_provenance or {}))
    payload["discovery_objective_mode"] = _normalize_discovery_objective_mode(resolved_config.discovery_objective_mode)
    payload["multi_objective_mode"] = _normalize_multi_objective_mode(resolved_config.multi_objective_mode)
    if payload["multi_objective_mode"] == _MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO:
        payload["objective_metric_source"] = "same_cutoff_pareto_vector"
        payload["objective_vector_names"] = list(_SAME_CUTOFF_PARETO_OBJECTIVE_NAMES)
    if payload["discovery_objective_mode"] == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING:
        payload["objective_metric_source"] = "paper_i_first_crossing_resource_subject_to_abs_error_Nph_plus_1"
    return _jsonable(payload)


def _normalize_objective_weight_preset(preset: str | None) -> str:
    key = str(preset or _OBJECTIVE_WEIGHT_PRESET_UNIFORM).strip().lower().replace("-", "_")
    if key not in _OBJECTIVE_WEIGHT_PRESETS:
        raise ValueError(f"Unsupported objective weight preset: {preset!r}")
    return key


def _normalize_discovery_objective_mode(mode: str | None) -> str:
    key = str(mode or _DISCOVERY_OBJECTIVE_MODE_TERMINAL_PROXY).strip().lower().replace("-", "_")
    if key not in set(_DISCOVERY_OBJECTIVE_MODES):
        raise ValueError(f"Unsupported discovery objective mode: {mode!r}")
    return key


def _normalize_multi_objective_mode(mode: str | None) -> str:
    key = str(mode or _MULTI_OBJECTIVE_MODE_OFF).strip().lower().replace("-", "_")
    if key not in set(_MULTI_OBJECTIVE_MODES):
        raise ValueError(f"Unsupported multi-objective mode: {mode!r}")
    return key


def _validate_objective_weight(value: Any, *, label: str) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite weight; got {value!r}") from exc
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"{label} must be a positive finite weight; got {value!r}")
    return float(weight)


def _validate_objective_weight_map(weights: Mapping[str, Any] | None, *, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_key, raw_value in dict(weights or {}).items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"{label} contains an empty key")
        out[key] = _validate_objective_weight(raw_value, label=f"{label}[{key!r}]")
    return out


def _parse_objective_weight_map(raw: str | None, *, label: str) -> dict[str, float]:
    text = str(raw or "").strip()
    if not text:
        return {}
    out: dict[str, float] = {}
    for item in text.split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{label} entries must use key=value format; got {part!r}")
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{label} contains an empty key")
        out[key] = _validate_objective_weight(value.strip(), label=f"{label}[{key!r}]")
    return out


@dataclass(frozen=True)
class WarmStartCandidate:
    """Auditable Optuna warm-start trial parameters."""

    params: Mapping[str, Any]
    source_kind: str
    source_id: str
    benchmark_id: str | None = None
    family: str | None = None
    source_score: float | None = None
    source_payload: Mapping[str, Any] = field(default_factory=dict)
    compatibility_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class WarmStartSkip:
    """Skipped warm-start candidate with a stable reason for summaries."""

    source_kind: str
    source_id: str
    reason: str
    benchmark_id: str | None = None
    family: str | None = None
    detail: str | None = None
    source_payload: Mapping[str, Any] = field(default_factory=dict)


def _fractional_cap_pair(size: int, min_fraction: float, max_fraction: float) -> tuple[int, int]:
    size_val = max(1, int(size))
    lo = _clamp(float(min_fraction), 0.0, 1.0)
    hi = _clamp(float(max_fraction), 0.0, 1.0)
    if lo > hi:
        lo, hi = hi, lo
    min_cap = max(1, int(math.ceil(size_val * lo)))
    max_cap = max(min_cap, int(math.ceil(size_val * hi)))
    return int(min_cap), int(max_cap)


def _ordered_threshold_pair(low: float, high: float) -> tuple[float, float]:
    lo = max(0.0, float(low))
    hi = max(0.0, float(high))
    return (lo, hi) if lo <= hi else (hi, lo)


def _ordered_phase_hysteresis_thresholds(
    phase2_null: float,
    phase2_live: float,
    phase3_null: float,
    phase3_live: float,
) -> tuple[float, float, float, float]:
    """Return thresholds satisfying stage-controller hysteresis invariants.

    The controller requires:

        0 <= phase2_null <= phase2_live < phase3_null <= phase3_live

    Optuna samples these values independently, so the harness must normalize
    them before launching ADAPT.  Otherwise valid trials are wasted on CLI
    configurations rejected before the first ADAPT iteration.
    """

    vals = sorted(max(0.0, float(x)) for x in (phase2_null, phase2_live, phase3_null, phase3_live))
    phase2_null_val, phase2_live_val, phase3_null_val, phase3_live_val = vals
    min_gap = 1e-12
    if phase3_null_val <= phase2_live_val:
        phase3_null_val = phase2_live_val + min_gap
    if phase3_live_val < phase3_null_val:
        phase3_live_val = phase3_null_val
    return (
        float(phase2_null_val),
        float(phase2_live_val),
        float(phase3_null_val),
        float(phase3_live_val),
    )


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
    route_id = normalize_static_route_id(
        getattr(static, "static_route_id", ROUTE_ID_A),
        default=ROUTE_ID_A,
    )
    if route_id == ROUTE_ID_A and str(hardware_resolution_mode).strip().lower() != "ideal":
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

    policy = _normalize_active_policy(policy)
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
    if selected_route != "standard" and normalize_static_route_id(
        getattr(static, "static_route_id", ROUTE_ID_A),
        default=ROUTE_ID_A,
    ) == ROUTE_ID_A:
        static = replace(static, static_route_id=ROUTE_ID_UNSPECIFIED)
    route_a_identity_changed = any(
        (
            str(getattr(static, "phase3_selector_policy", _DEFAULT_PHASE3_SELECTOR_POLICY)).strip().lower() != _DEFAULT_PHASE3_SELECTOR_POLICY,
            str(getattr(static, "phase3_selector_geometry_mode", "reduced")).strip().lower() != "reduced",
            str(getattr(static, "phase3_novelty_ablation_mode", "off")).strip().lower() != "off",
            str(getattr(static, "phase3_window_relaxation_mode", "reduced")).strip().lower() != "reduced",
            str(getattr(static, "phase3_batch_selection_mode", "reduced_plane")).strip().lower() != "reduced_plane",
            str(getattr(static, "phase3_batch_prefilter_mode", "off")).strip().lower() != "off",
        )
    )
    if route_a_identity_changed and normalize_static_route_id(
        getattr(static, "static_route_id", ROUTE_ID_A),
        default=ROUTE_ID_A,
    ) == ROUTE_ID_A:
        static = replace(static, static_route_id=ROUTE_ID_UNSPECIFIED)
    if selected_route == "historical_selected" and selected_source not in {None, ""}:
        # Historical-selected means: start from the problem-generic mega pool,
        # then filter by the historical operator-family closure.  Letting
        # Optuna sample a narrower base pool can silently exclude the selected
        # family before the selected-logical filter gets a chance to apply.
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
    phase2_null, phase2_live, phase3_null, phase3_live = _ordered_phase_hysteresis_thresholds(
        static.phase2_null_nrem_high_threshold,
        static.phase2_live_nrem_low_threshold,
        static.phase3_null_nrem_high_threshold,
        static.phase3_live_nrem_low_threshold,
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
    args = _set_toggle_pair(
        args,
        "--phase-live-hysteresis-enabled",
        "--phase-live-hysteresis-disabled",
        bool(static.phase_live_hysteresis_enabled),
    )
    args = _set_option(args, "--phase2-null-nrem-high-threshold", phase2_null)
    args = _set_option(args, "--phase2-live-nrem-low-threshold", phase2_live)
    args = _set_option(args, "--phase3-null-nrem-high-threshold", phase3_null)
    args = _set_option(args, "--phase3-live-nrem-low-threshold", phase3_live)
    args = _set_option(args, "--phase2-hysteresis-steps", max(1, int(static.phase2_hysteresis_steps)))
    args = _set_option(args, "--phase3-hysteresis-steps", max(1, int(static.phase3_hysteresis_steps)))
    args = _set_option(args, "--phase1-probe-max-positions", static.phase1_probe_max_positions)
    args = _set_toggle_pair(args, "--phase0-pilot-enabled", "--phase0-no-pilot", bool(static.phase0_pilot_enabled))
    args = _set_option(args, "--phase0-pilot-alpha", static.phase0_pilot_alpha)
    args = _set_option(args, "--phase0-pilot-threshold", static.phase0_pilot_threshold)
    args = _set_option(args, "--phase0-pilot-max-records", static.phase0_pilot_max_records)
    args = _set_option(args, "--phase0-lane-quota-pressure", static.phase0_lane_quota_pressure)
    args = _set_option(args, "--phase0-algebraic-lane-mode", static.phase0_algebraic_lane_mode)
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
        args = _set_option(args, "--phase1-prune-stale-age", static.phase1_prune_stale_age)
        args = _set_option(args, "--phase1-prune-stagnation-threshold", static.phase1_prune_stagnation_threshold)
        args = _set_option(args, "--phase1-prune-small-theta-abs", static.phase1_prune_small_theta_abs)
        args = _set_option(args, "--phase1-prune-small-theta-relative", static.phase1_prune_small_theta_relative)
        args = _set_option(args, "--phase1-prune-cooldown-steps", static.phase1_prune_cooldown_steps)
        args = _set_option(args, "--phase1-prune-local-window-size", static.phase1_prune_local_window_size)
        args = _set_option(args, "--phase1-prune-recovery-trust-radius", static.phase1_prune_recovery_trust_radius)
        args = _set_option(args, "--phase1-prune-old-fraction", static.phase1_prune_old_fraction)
        args = _set_option(args, "--phase1-prune-checkpoint-period", static.phase1_prune_checkpoint_period)
        args = _set_option(args, "--phase1-prune-live-min-depth", static.phase1_prune_live_min_depth)
        args = _set_option(args, "--phase1-prune-maturity-threshold", static.phase1_prune_maturity_threshold)
        args = _set_option(args, "--phase1-prune-snr-threshold", static.phase1_prune_snr_threshold)
        args = _set_toggle_pair(
            args,
            "--phase1-prune-amplitude-witness-required",
            "--phase1-prune-amplitude-witness-optional",
            bool(static.phase1_prune_amplitude_witness_required),
        )
        args = _set_option(args, "--phase1-prune-collapse-peak-abs-min", static.phase1_prune_collapse_peak_abs_min)
        args = _set_option(args, "--phase1-prune-collapse-current-abs-max", static.phase1_prune_collapse_current_abs_max)
        args = _set_option(args, "--phase1-prune-collapse-ratio", static.phase1_prune_collapse_ratio)
        args = _set_option(args, "--phase1-prune-collapse-min-abs-drop", static.phase1_prune_collapse_min_abs_drop)
        args = _set_option(args, "--phase1-prune-collapse-min-observations", static.phase1_prune_collapse_min_observations)

    args = _set_option(args, "--phase2-shortlist-fraction", static.phase2_shortlist_fraction)
    args = _set_option(args, "--phase2-shortlist-size", phase2_shortlist)
    args = _set_option(args, "--algebraic-phase2-lane-rel-threshold", static.algebraic_phase2_lane_rel_threshold)
    args = _set_option(args, "--algebraic-phase1-lane-quota-pressure", static.algebraic_phase1_lane_quota_pressure)
    args = _set_option(args, "--algebraic-phase2-lane-quota-pressure", static.algebraic_phase2_lane_quota_pressure)
    args = _set_option(args, "--phase2-novelty-mode", static.phase2_novelty_mode)
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
    args = _set_option(args, "--phase2-gamma-N", static.phase2_gamma_N)
    args = _set_option(args, "--phase2-gamma-N-schedule-mode", static.phase2_gamma_N_schedule_mode)
    args = _set_option(args, "--phase2-gamma-N-schedule-start", static.phase2_gamma_N_schedule_start)
    args = _set_option(args, "--phase2-gamma-N-schedule-end", static.phase2_gamma_N_schedule_end)
    motif_bonus_weight = (
        float(pool.novelty_bonus)
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
    args = _set_option(args, "--phase3-selector-policy", static.phase3_selector_policy)
    args = _set_option(args, "--phase3-selector-geometry-mode", static.phase3_selector_geometry_mode)
    args = _set_option(args, "--phase3-novelty-ablation-mode", static.phase3_novelty_ablation_mode)
    args = _set_option(args, "--phase3-window-relaxation-mode", static.phase3_window_relaxation_mode)
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
    static_route_id = _validate_static_route_policy(static, hardware_resolution_mode=hardware_resolution_mode)
    args = _set_option(args, "--static-route-id", static_route_id)
    args = _remove_option(args, "--allow-legacy-static-route")
    if static_route_id in LEGACY_ROUTE_ID_CHOICES:
        args = [*args, "--allow-legacy-static-route"]
    args = _set_option(args, "--static-meta-feature-profile", static.static_meta_feature_profile)
    args = _set_option(
        args,
        "--static-lane-route",
        normalize_static_lane_route(getattr(static, "static_lane_route", STATIC_LANE_ROUTE_ALGEBRAIC)),
    )
    args = _set_option(
        args,
        "--physical-lane-shortlist-aggressiveness",
        normalize_physical_lane_shortlist_aggressiveness(
            getattr(static, "physical_lane_shortlist_aggressiveness", 3)
        ),
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
    normalized_policy = _normalize_active_policy(policy)
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
        "--phase0-pilot-enabled": ("--phase0-pilot-enabled", "--phase0-no-pilot", True),
        "--phase0-no-pilot": ("--phase0-pilot-enabled", "--phase0-no-pilot", False),
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
        "--phase-live-hysteresis-enabled": (
            "--phase-live-hysteresis-enabled",
            "--phase-live-hysteresis-disabled",
            True,
        ),
        "--phase-live-hysteresis-disabled": (
            "--phase-live-hysteresis-enabled",
            "--phase-live-hysteresis-disabled",
            False,
        ),
        "--phase1-prune-amplitude-witness-required": (
            "--phase1-prune-amplitude-witness-required",
            "--phase1-prune-amplitude-witness-optional",
            True,
        ),
        "--phase1-prune-amplitude-witness-optional": (
            "--phase1-prune-amplitude-witness-required",
            "--phase1-prune-amplitude-witness-optional",
            False,
        ),
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


def _write_command_log(path: Path, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join([str(x) for x in command]) + "\n", encoding="utf-8")


@dataclass
class _ActiveSubprocess:
    pid: int
    label: str
    benchmark_id: str | None
    trial_number: int | None
    command: tuple[str, ...]
    started_utc: str
    process: Any = field(repr=False, compare=False)


class _ProgressReporter:
    """Fail-soft JSONL/current-file reporter for CHTC-visible Optuna progress."""

    def __init__(self, progress_dir: str | Path | None) -> None:
        self.progress_dir = None if progress_dir in {None, ""} else Path(progress_dir)
        self.errors: list[str] = []
        self._lock = threading.RLock()
        if self.progress_dir is not None:
            try:
                self.progress_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover - filesystem dependent
                self.errors.append(f"mkdir:{type(exc).__name__}:{exc}")
                self.progress_dir = None

    @property
    def enabled(self) -> bool:
        return self.progress_dir is not None

    def _safe(self, label: str, func: Callable[[], None]) -> None:
        if self.progress_dir is None:
            return
        try:
            with self._lock:
                func()
        except Exception as exc:  # progress reporting must never fail the study
            self.errors.append(f"{label}:{type(exc).__name__}:{exc}")

    def append_event(self, event: str, **fields: Any) -> None:
        def _write() -> None:
            assert self.progress_dir is not None
            payload = {
                "schema": "phase3_progress_event_v1",
                "timestamp_utc": _now_utc(),
                "event": str(event),
                **_jsonable(fields),
            }
            with (self.progress_dir / "trial_events.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True) + "\n")

        self._safe(f"append_event:{event}", _write)

    @staticmethod
    def _is_incumbent_best_payload(payload: Mapping[str, Any]) -> bool:
        best_trial_number = payload.get("best_trial_number")
        if best_trial_number is None:
            return False
        if "trial_number" not in payload:
            return True
        return payload.get("trial_number") == best_trial_number

    def write_current(self, **fields: Any) -> None:
        def _write() -> None:
            assert self.progress_dir is not None
            payload = {
                "schema": "phase3_progress_current_v1",
                "timestamp_utc": _now_utc(),
                **fields,
            }
            _write_json(self.progress_dir / "current.json", payload)
            if self._is_incumbent_best_payload(payload):
                _write_json(self.progress_dir / "current_best.json", payload)

        self._safe("write_current", _write)

    def write_status_snapshot(self, **fields: Any) -> None:
        def _write() -> None:
            assert self.progress_dir is not None
            _write_json(
                self.progress_dir / "status_snapshot.json",
                {
                    "schema": "phase3_progress_status_snapshot_v1",
                    "timestamp_utc": _now_utc(),
                    "reporter_errors": list(self.errors),
                    **fields,
                },
            )

        self._safe("write_status_snapshot", _write)

    def write_active_processes(self, processes: Sequence[Mapping[str, Any]]) -> None:
        def _write() -> None:
            assert self.progress_dir is not None
            _write_json(
                self.progress_dir / "active_processes.json",
                {
                    "schema": "phase3_active_processes_v1",
                    "timestamp_utc": _now_utc(),
                    "active_processes": list(processes),
                    "active_process_count": len(processes),
                },
            )

        self._safe("write_active_processes", _write)

    def summary_payload(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "progress_dir": None if self.progress_dir is None else str(self.progress_dir), "errors": list(self.errors)}


def _terminate_process_group_for_proc(
    proc: Any,
    *,
    reason: str,
    cleanup_errors: list[str] | None = None,
    grace_s: float = 10.0,
) -> None:
    """Best-effort SIGTERM/SIGKILL for a subprocess group started with a new session."""

    pid = int(getattr(proc, "pid"))
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as exc:
        if cleanup_errors is not None:
            cleanup_errors.append(f"{reason}:sigterm_pid_{pid}:{type(exc).__name__}:{exc}")
    try:
        proc.wait(timeout=float(grace_s))
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception as exc:
        if cleanup_errors is not None:
            cleanup_errors.append(f"{reason}:wait_after_sigterm_pid_{pid}:{type(exc).__name__}:{exc}")

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception as exc:
        if cleanup_errors is not None:
            cleanup_errors.append(f"{reason}:sigkill_pid_{pid}:{type(exc).__name__}:{exc}")
    try:
        proc.wait(timeout=float(grace_s))
    except Exception as exc:
        if cleanup_errors is not None:
            cleanup_errors.append(f"{reason}:wait_after_sigkill_pid_{pid}:{type(exc).__name__}:{exc}")


def _first_finite_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _first_int_like(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(float(value))
        except Exception:
            continue
    return None


def _trial_prune_state_from_current_json(
    path: Path,
    *,
    metric: str = "same_cutoff_abs_delta_e",
) -> dict[str, Any] | None:
    """Read a live ADAPT checkpoint enough to apply a trial-level prune gate."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    adapt = payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        adapt = {}
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        checkpoint = {}
    classification = adapt.get("benchmark_target_classification")
    if not isinstance(classification, Mapping):
        classification = checkpoint.get("target_hit_classification")
    if not isinstance(classification, Mapping):
        classification = {}

    depth = _first_int_like(
        adapt.get("ansatz_depth"),
        checkpoint.get("ansatz_depth"),
        checkpoint.get("depth"),
        adapt.get("depth"),
        payload.get("ansatz_depth"),
        payload.get("depth"),
    )
    same_cutoff_error = _first_finite_float(
        adapt.get("abs_delta_e"),
        adapt.get("same_cutoff_abs_delta_e"),
        payload.get("abs_delta_e"),
        payload.get("same_cutoff_abs_delta_e"),
    )
    target_error = _first_finite_float(
        adapt.get("benchmark_target_abs_delta_e_current"),
        classification.get("target_error"),
        payload.get("benchmark_target_abs_delta_e_current"),
    )

    metric_key = str(metric or "same_cutoff_abs_delta_e").strip().lower()
    if metric_key in {"target_abs_delta_e", "benchmark_target_abs_delta_e", "reference_abs_delta_e"}:
        delta_e = target_error if target_error is not None else same_cutoff_error
    else:
        delta_e = same_cutoff_error if same_cutoff_error is not None else target_error
    if depth is None or delta_e is None:
        return None
    return {
        "depth": int(depth),
        "delta_e": float(delta_e),
        "metric": metric_key,
        "same_cutoff_abs_delta_e": same_cutoff_error,
        "target_abs_delta_e": target_error,
        "source_json": str(path),
    }


class _ManagedOptunaRunLifecycle:
    """Run-scoped owner for local Optuna trials and child process groups."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        optuna_module: Any | None = None,
        progress_reporter: _ProgressReporter | None = None,
    ) -> None:
        self.run_id = str(run_id or uuid.uuid4())
        self.run_pid = int(os.getpid())
        self.started_utc = _now_utc()
        self.study: Any | None = None
        self._optuna = optuna_module
        self.progress_reporter = progress_reporter
        self.active_processes: dict[int, _ActiveSubprocess] = {}
        self.owned_trial_numbers: set[int] = set()
        self.shutdown_requested = threading.Event()
        self.received_signal: int | None = None
        self.cleanup_errors: list[str] = []
        self.reconciled_running_trial_numbers: list[int] = []
        self.failed_owned_running_trial_numbers: list[int] = []
        self._lock = threading.RLock()
        self._previous_signal_handlers: dict[int, Any] = {}
        self._signals_installed = False

    def __enter__(self) -> "_ManagedOptunaRunLifecycle":
        self.install_signal_handlers()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> bool:
        try:
            self.terminate_all_process_groups("lifecycle_exit")
            self.fail_owned_running_trials("lifecycle_exit")
        finally:
            self.restore_signal_handlers()
        return False

    def attach_study(self, study: Any) -> None:
        self.study = study

    def _trial_state(self) -> Any:
        if self._optuna is None:
            self._optuna = _import_optuna()
        return self._optuna.trial.TrialState

    def install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        if self._signals_installed:
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            except Exception as exc:  # pragma: no cover - platform/interpreter dependent
                self.cleanup_errors.append(f"install_signal_{signum}:{type(exc).__name__}:{exc}")
        self._signals_installed = True

    def restore_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, handler in list(self._previous_signal_handlers.items()):
            try:
                signal.signal(signum, handler)
            except Exception as exc:  # pragma: no cover - platform/interpreter dependent
                self.cleanup_errors.append(f"restore_signal_{signum}:{type(exc).__name__}:{exc}")
        self._previous_signal_handlers.clear()
        self._signals_installed = False

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.received_signal = int(signum)
        self.shutdown_requested.set()
        self.terminate_all_process_groups(f"signal_{int(signum)}")
        if int(signum) == int(signal.SIGINT):
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))

    def _running_trial_numbers(self) -> set[int]:
        if self.study is None:
            return set()
        trial_state = self._trial_state()
        try:
            trials = self.study.get_trials(deepcopy=False, states=(trial_state.RUNNING,))
        except TypeError:
            trials = self.study.get_trials(deepcopy=False)
            trials = [trial for trial in trials if trial.state == trial_state.RUNNING]
        except Exception as exc:
            self.cleanup_errors.append(f"get_running_trials:{type(exc).__name__}:{exc}")
            return set()
        return {int(trial.number) for trial in trials}

    def _tell_fail_trial_number(self, trial_number: int, *, reason: str) -> bool:
        if self.study is None:
            return False
        number = int(trial_number)
        if number not in self._running_trial_numbers():
            return False
        trial_state = self._trial_state()
        try:
            self.study.tell(number, state=trial_state.FAIL, skip_if_finished=True)
            return True
        except TypeError:
            try:
                self.study.tell(number, state=trial_state.FAIL)
                return True
            except Exception as exc:
                self.cleanup_errors.append(f"{reason}:tell_fail_trial_{number}:{type(exc).__name__}:{exc}")
                return False
        except Exception as exc:
            self.cleanup_errors.append(f"{reason}:tell_fail_trial_{number}:{type(exc).__name__}:{exc}")
            return False

    def reconcile_existing_running_trials(self) -> list[int]:
        reconciled: list[int] = []
        for number in sorted(self._running_trial_numbers()):
            if self._tell_fail_trial_number(number, reason="startup_reconcile"):
                reconciled.append(int(number))
        self.reconciled_running_trial_numbers.extend(
            number for number in reconciled if number not in self.reconciled_running_trial_numbers
        )
        return reconciled

    def adopt_trial(self, trial: Any) -> None:
        number = int(trial.number)
        with self._lock:
            self.owned_trial_numbers.add(number)
        if hasattr(trial, "set_user_attr"):
            try:
                trial.set_user_attr("phase3_lifecycle_managed_v1", True)
                trial.set_user_attr("phase3_run_id", self.run_id)
                trial.set_user_attr("phase3_run_pid", self.run_pid)
                trial.set_user_attr("phase3_run_started_utc", self.started_utc)
            except Exception as exc:
                self.cleanup_errors.append(f"adopt_trial_{number}:set_user_attr:{type(exc).__name__}:{exc}")

    def finish_trial(self, trial_number: int) -> None:
        with self._lock:
            self.owned_trial_numbers.discard(int(trial_number))

    def fail_trial_number(self, trial_number: int, *, reason: str) -> bool:
        number = int(trial_number)
        failed = self._tell_fail_trial_number(number, reason=reason)
        if failed and number not in self.failed_owned_running_trial_numbers:
            self.failed_owned_running_trial_numbers.append(number)
        return failed

    def fail_owned_running_trials(self, reason: str) -> list[int]:
        with self._lock:
            owned = sorted(self.owned_trial_numbers)
        failed: list[int] = []
        for number in owned:
            if self.fail_trial_number(number, reason=reason):
                failed.append(number)
        return failed

    def register_process(
        self,
        proc: Any,
        *,
        label: str,
        benchmark_id: str | None,
        trial_number: int | None,
        command: Sequence[str],
    ) -> None:
        pid = int(getattr(proc, "pid"))
        record = _ActiveSubprocess(
            pid=pid,
            label=str(label),
            benchmark_id=None if benchmark_id is None else str(benchmark_id),
            trial_number=None if trial_number is None else int(trial_number),
            command=tuple(str(x) for x in command),
            started_utc=_now_utc(),
            process=proc,
        )
        with self._lock:
            self.active_processes[pid] = record
        if self.progress_reporter is not None:
            self.progress_reporter.append_event(
                "subprocess_started",
                pid=pid,
                label=record.label,
                benchmark_id=record.benchmark_id,
                trial_number=record.trial_number,
            )
            self.progress_reporter.write_active_processes(self.active_process_payloads())

    def active_process_payloads(self) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self.active_processes.values())
        return [
            {
                "pid": int(record.pid),
                "label": record.label,
                "benchmark_id": record.benchmark_id,
                "trial_number": record.trial_number,
                "started_utc": record.started_utc,
                "command": list(record.command),
            }
            for record in records
        ]

    def unregister_process(self, pid: int, *, returncode: int | None = None, elapsed_s: float | None = None) -> None:
        with self._lock:
            record = self.active_processes.pop(int(pid), None)
        if record is not None and self.progress_reporter is not None:
            self.progress_reporter.append_event(
                "subprocess_exited",
                pid=int(pid),
                label=record.label,
                benchmark_id=record.benchmark_id,
                trial_number=record.trial_number,
                returncode=returncode,
                elapsed_s=elapsed_s,
            )
            self.progress_reporter.write_active_processes(self.active_process_payloads())

    def terminate_process_group(self, pid: int, reason: str) -> None:
        with self._lock:
            record = self.active_processes.get(int(pid))
        if record is None:
            return
        try:
            _terminate_process_group_for_proc(record.process, reason=reason, cleanup_errors=self.cleanup_errors)
        finally:
            self.unregister_process(int(pid))

    def terminate_all_process_groups(self, reason: str) -> None:
        with self._lock:
            records = list(self.active_processes.values())
        for record in records:
            self.terminate_process_group(record.pid, reason=reason)

    def active_process_count(self) -> int:
        with self._lock:
            return len(self.active_processes)

    def summary_payload(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "run_id": self.run_id,
            "run_pid": self.run_pid,
            "run_started_utc": self.started_utc,
            "received_signal": self.received_signal,
            "shutdown_requested": bool(self.shutdown_requested.is_set()),
            "reconciled_running_trial_numbers": list(self.reconciled_running_trial_numbers),
            "failed_owned_running_trial_numbers": list(self.failed_owned_running_trial_numbers),
            "cleanup_errors": list(self.cleanup_errors),
            "active_process_count_after_cleanup": int(self.active_process_count()),
            "active_processes_after_cleanup": self.active_process_payloads(),
        }


def _run_subprocess_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float | None = None,
    run_lifecycle: _ManagedOptunaRunLifecycle | None = None,
    subprocess_label: str = "subprocess",
    benchmark_id: str | None = None,
    trial_number: int | None = None,
    heartbeat_path: Path | None = None,
    heartbeat_events_path: Path | None = None,
    heartbeat_metadata: Mapping[str, Any] | None = None,
    trial_prune_current_json: Path | None = None,
    trial_prune_depth: int | None = None,
    trial_prune_abs_delta_e: float | None = None,
    trial_prune_metric: str = "same_cutoff_abs_delta_e",
    trial_prune_status_path: Path | None = None,
    trial_prune_poll_s: float = 5.0,
) -> tuple[int, float]:
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
                "trial_number": trial_number,
                **dict(heartbeat_metadata or {}),
            },
        )
    )
    child_heartbeat_state: dict[str, Any] = {}
    stdout_reader_errors: list[str] = []

    def _publish_child_heartbeat(
        state: Mapping[str, Any],
        *,
        ai_log_payload: Mapping[str, Any] | None = None,
    ) -> None:
        if run_lifecycle is None or run_lifecycle.progress_reporter is None:
            return
        try:
            run_lifecycle.progress_reporter.write_current(
                active_child_count=int(run_lifecycle.active_process_count()),
                child_label=str(subprocess_label),
                child_benchmark_id=benchmark_id,
                child_trial_number=trial_number,
                last_child_heartbeat=dict(state),
                last_child_ai_log=None if ai_log_payload is None else dict(ai_log_payload),
            )
            if ai_log_payload is not None:
                progress = state.get("progress")
                if not isinstance(progress, Mapping):
                    progress = {}
                run_lifecycle.progress_reporter.append_event(
                    "child_ai_log",
                    child_label=str(subprocess_label),
                    benchmark_id=benchmark_id,
                    trial_number=trial_number,
                    ai_log_event=state.get("last_ai_log_event"),
                    child_status=state.get("status"),
                    child_elapsed_s=state.get("elapsed_s"),
                    depth=progress.get("depth"),
                    energy=progress.get("energy"),
                    delta_abs_current=progress.get("delta_abs_current"),
                    max_grad=progress.get("max_grad"),
                    stop_reason_so_far=progress.get("stop_reason_so_far"),
                    progress=dict(progress),
                    ai_log_payload=dict(ai_log_payload),
                )
        except Exception as exc:  # progress reporting must never fail the child
            run_lifecycle.progress_reporter.errors.append(
                f"write_child_heartbeat:{type(exc).__name__}:{exc}"
            )

    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
        timeout = None if timeout_s is None or float(timeout_s) <= 0.0 else float(timeout_s)
        proc: subprocess.Popen[str] | None = None
        returncode: int | None = None
        stdout_thread: threading.Thread | None = None

        def _stdout_reader(stream: Any) -> None:
            nonlocal child_heartbeat_state
            try:
                for line in stream:
                    stdout_fh.write(line)
                    stdout_fh.flush()
                    payload = parse_ai_log_line(line)
                    if payload is None:
                        continue
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    elapsed_s = float(time.perf_counter() - started)
                    pid = None if proc is None else int(proc.pid)
                    if recorder is not None:
                        state = recorder.update_from_ai_log(
                            payload,
                            elapsed_s=elapsed_s,
                            pid=pid,
                        )
                    else:
                        state = normalize_ai_log_progress(
                            payload,
                            elapsed_s=elapsed_s,
                            pid=pid,
                            previous=child_heartbeat_state,
                        )
                    child_heartbeat_state = dict(state)
                    _publish_child_heartbeat(state, ai_log_payload=payload)
            except Exception as exc:  # fail-soft: preserve the child run if logging fails
                stdout_reader_errors.append(f"stdout_reader:{type(exc).__name__}:{exc}")

        try:
            if run_lifecycle is not None and run_lifecycle.shutdown_requested.is_set():
                stderr_fh.write(
                    "\n[phase3_policy_optuna] subprocess launch skipped because managed lifecycle shutdown was requested\n"
                )
                returncode = 130
                if recorder is not None:
                    state = recorder.mark_finished(
                        status="skipped",
                        returncode=returncode,
                        elapsed_s=float(time.perf_counter() - started),
                    )
                    _publish_child_heartbeat(state)
                return 130, float(time.perf_counter() - started)
            proc = subprocess.Popen(
                [str(x) for x in command],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
                env=dict(os.environ),
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            if recorder is not None:
                state = recorder.mark_started(pid=int(proc.pid), command=[str(x) for x in command])
                child_heartbeat_state = dict(state)
                _publish_child_heartbeat(state)
            if run_lifecycle is not None:
                run_lifecycle.register_process(
                    proc,
                    label=subprocess_label,
                    benchmark_id=benchmark_id,
                    trial_number=trial_number,
                    command=command,
                )
            if proc.stdout is not None:
                stdout_thread = threading.Thread(
                    target=_stdout_reader,
                    args=(proc.stdout,),
                    name=f"phase3-{subprocess_label}-stdout-forwarder",
                    daemon=True,
                )
                stdout_thread.start()
            deadline = None if timeout is None else float(started + timeout)
            while True:
                wait_s = max(0.25, float(trial_prune_poll_s))
                if deadline is not None:
                    remaining = float(deadline - time.perf_counter())
                    if remaining <= 0.0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    wait_s = min(wait_s, remaining)
                try:
                    returncode = int(proc.wait(timeout=wait_s))
                    break
                except subprocess.TimeoutExpired:
                    if (
                        trial_prune_current_json is None
                        or trial_prune_depth is None
                        or trial_prune_abs_delta_e is None
                    ):
                        continue
                    state_payload = _trial_prune_state_from_current_json(
                        Path(trial_prune_current_json),
                        metric=str(trial_prune_metric),
                    )
                    if not state_payload:
                        continue
                    depth = int(state_payload["depth"])
                    delta_e = float(state_payload["delta_e"])
                    threshold = float(trial_prune_abs_delta_e)
                    if depth <= int(trial_prune_depth) or delta_e <= threshold:
                        continue
                    elapsed = float(time.perf_counter() - started)
                    prune_payload = {
                        "schema": "phase3_trial_prune_gate_v1",
                        "status": "pruned",
                        "reason": "comparator_threshold_not_beaten_after_depth",
                        "benchmark_id": benchmark_id,
                        "trial_number": trial_number,
                        "subprocess_label": str(subprocess_label),
                        "depth": depth,
                        "gate_depth": int(trial_prune_depth),
                        "delta_e": delta_e,
                        "threshold_abs_delta_e": threshold,
                        "metric": str(trial_prune_metric),
                        "elapsed_s": elapsed,
                        "current_json": str(trial_prune_current_json),
                    }
                    if trial_prune_status_path is not None:
                        try:
                            _write_json(Path(trial_prune_status_path), prune_payload)
                        except Exception as exc:
                            stderr_fh.write(
                                "\n[phase3_policy_optuna] failed to write trial prune status "
                                f"{trial_prune_status_path}: {type(exc).__name__}: {exc}\n"
                            )
                    stderr_fh.write(
                        "\n[phase3_policy_optuna] trial prune gate fired: "
                        f"depth={depth} gate_depth={int(trial_prune_depth)} "
                        f"delta_e={delta_e:.12g} threshold={threshold:.12g} "
                        f"metric={trial_prune_metric}\n"
                    )
                    if proc is not None:
                        if run_lifecycle is not None:
                            run_lifecycle.terminate_process_group(proc.pid, "trial_prune_gate")
                        else:
                            _terminate_process_group_for_proc(proc, reason="trial_prune_gate")
                    returncode = 125
                    return 125, elapsed
        except subprocess.TimeoutExpired:
            elapsed = float(time.perf_counter() - started)
            returncode = 124
            stderr_fh.write(
                "\n[phase3_policy_optuna] subprocess timeout "
                f"after {elapsed:.3f}s; configured timeout_s={float(timeout_s):.3f}\n"
            )
            if proc is not None:
                if run_lifecycle is not None:
                    run_lifecycle.terminate_process_group(proc.pid, "subprocess_timeout")
                else:
                    _terminate_process_group_for_proc(proc, reason="subprocess_timeout")
            return 124, elapsed
        except BaseException:
            if proc is not None:
                if run_lifecycle is not None:
                    run_lifecycle.terminate_process_group(proc.pid, "subprocess_exception")
                else:
                    _terminate_process_group_for_proc(proc, reason="subprocess_exception")
            raise
        finally:
            if stdout_thread is not None:
                stdout_thread.join(timeout=2.0)
            for error in stdout_reader_errors:
                stderr_fh.write(f"\n[phase3_policy_optuna] {error}\n")
            elapsed_final = float(time.perf_counter() - started)
            if recorder is not None:
                status = (
                    "completed"
                    if returncode == 0
                    else ("failed" if returncode is not None else "interrupted")
                )
                state = recorder.mark_finished(
                    status=status,
                    returncode=returncode,
                    elapsed_s=elapsed_final,
                )
                child_heartbeat_state = dict(state)
                _publish_child_heartbeat(state)
            if proc is not None and run_lifecycle is not None:
                run_lifecycle.unregister_process(
                    proc.pid,
                    returncode=None if returncode is None else int(returncode),
                    elapsed_s=elapsed_final,
                )
    return int(returncode), float(time.perf_counter() - started)


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


def _stop_reason_from_result_payload(result_payload: Mapping[str, Any] | None) -> str | None:
    if not isinstance(result_payload, Mapping):
        return None
    adapt = _adapt_vqe_section(result_payload)
    for block in (adapt, result_payload):
        if isinstance(block, Mapping) and block.get("stop_reason") not in {None, ""}:
            return str(block.get("stop_reason"))
    return None


def _target_hit_classification_payload(
    *,
    stop_reason: str | None,
    target_error: float | None,
    target_threshold: float | None,
    source: str,
    accepted_crossing_reached: bool = False,
) -> dict[str, Any]:
    stop_key = str(stop_reason or "").strip()
    threshold = _as_float_or_none(target_threshold)
    target_configured = bool(threshold is not None and float(threshold) > 0.0)
    error_value = _as_float_or_none(target_error)
    error_within_threshold = bool(
        target_configured
        and error_value is not None
        and float(error_value) <= float(threshold)
    )
    accepted_target_stop = bool(stop_key == _BENCHMARK_TARGET_HIT_STOP_REASON)
    target_hit_success = bool(
        target_configured
        and accepted_target_stop
        and (error_within_threshold or bool(accepted_crossing_reached))
    )
    if not target_configured:
        status = "target_not_requested"
        non_hit_reason = "benchmark_target_abs_delta_e_not_configured"
    elif target_hit_success:
        status = "target_hit_success"
        non_hit_reason = None
    elif accepted_target_stop:
        status = "inconsistent_target_stop_non_hit"
        non_hit_reason = "benchmark_target_stop_without_in_threshold_error_or_accepted_crossing"
    elif stop_key == "":
        status = "active_or_missing_terminal_non_hit"
        non_hit_reason = "missing_terminal_benchmark_target_stop"
    else:
        status = "non_hit_diagnostic"
        non_hit_reason = f"terminal_stop_reason_not_target_hit:{stop_key}"
    return {
        "schema_version": "static_adapt_target_hit_classification_v1",
        "source": str(source),
        "target_hit_success": bool(target_hit_success),
        "status": str(status),
        "non_hit_reason": non_hit_reason,
        "terminal_stop_reason": (None if stop_key == "" else stop_key),
        "required_stop_reason": _BENCHMARK_TARGET_HIT_STOP_REASON,
        "target_configured": bool(target_configured),
        "target_error": error_value,
        "target_threshold": threshold,
        "target_error_within_threshold": bool(error_within_threshold),
        "target_error_within_threshold_without_target_stop": bool(
            target_configured and error_within_threshold and not accepted_target_stop
        ),
        "accepted_crossing_reached": bool(accepted_crossing_reached),
    }


def _payload_target_hit_classification(
    result_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(result_payload, Mapping):
        return None
    adapt = _adapt_vqe_section(result_payload)
    continuation = adapt.get("continuation", {}) if isinstance(adapt, Mapping) else {}
    candidate_blocks = (
        continuation if isinstance(continuation, Mapping) else {},
        adapt,
        result_payload,
    )
    for block in candidate_blocks:
        if not isinstance(block, Mapping):
            continue
        for key in ("benchmark_target_classification", "target_hit_classification"):
            value = block.get(key)
            if isinstance(value, Mapping) and "target_hit_success" in value:
                return dict(value)
    return None


def target_hit_classification_for_result(
    result: BenchmarkResult | None,
    *,
    target_abs_delta_e: float | None = None,
    first_crossing: Mapping[str, Any] | None = None,
    source: str = "benchmark_result",
) -> dict[str, Any]:
    if result is None:
        return _target_hit_classification_payload(
            stop_reason=None,
            target_error=None,
            target_threshold=target_abs_delta_e,
            source=source,
            accepted_crossing_reached=False,
        )
    first = dict(first_crossing or result.paper_i_first_crossing or {})
    threshold = _as_float_or_none(target_abs_delta_e)
    if threshold is None:
        threshold = _as_float_or_none(first.get("tau_phys"))
    target_error = _as_float_or_none(result.abs_delta_e_reference)
    if target_error is None:
        target_error = _as_float_or_none(result.abs_delta_e)
    return _target_hit_classification_payload(
        stop_reason=result.stop_reason,
        target_error=target_error,
        target_threshold=threshold,
        source=source,
        accepted_crossing_reached=bool(first.get("reached", False)),
    )


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
    """Fail closed on mechanically successful but physically useless ADAPT trials.

    Optuna should not treat a valid JSON artifact plus successful compile as a
    good trial when ADAPT stopped immediately, made no energy progress, or left
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


def _benchmark_target_reference_for_spec(
    spec: HamiltonianBenchmarkSpec,
    *,
    benchmark_target_abs_delta_e: float | None,
) -> tuple[float | None, int | None, str | None, str]:
    if benchmark_target_abs_delta_e is None or float(benchmark_target_abs_delta_e) <= 0.0:
        return None, None, None, "target_stop_disabled"
    reference_energy, reference_nph, reference_failure = _reference_cutoff_energy_for_spec(spec)
    requires_external_reference = False
    raw_nph = _pipeline_arg_value(spec.base_pipeline_args, "--n-ph-max")
    if bool(spec.features.bosonic) and spec.exact_reference_n_ph_max is not None and raw_nph not in {None, ""}:
        try:
            requires_external_reference = int(spec.exact_reference_n_ph_max) > int(float(str(raw_nph)))
        except Exception:
            requires_external_reference = False
    if reference_energy is None:
        source = "adapt_exact_gs" if reference_failure is None else "reference_cutoff_unavailable"
        if requires_external_reference:
            reason = reference_failure or "missing_reference_cutoff_energy"
            raise ValueError(
                "benchmark_target_reference_energy_required_for_phonon_external_target:"
                f"{spec.benchmark_id}:{reason}"
            )
        return None, reference_nph, reference_failure, source
    return float(reference_energy), reference_nph, None, "reference_cutoff_energy"


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


@lru_cache(maxsize=128)
def _reference_cutoff_energy_cached(
    problem: str,
    L: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    v_nn: float,
    t_prime: float,
    ref_nph: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
) -> float:
    h_poly = build_problem_hamiltonian(
        problem_key=str(problem),
        num_sites=int(L),
        t=float(t),
        u=float(u),
        dv=float(dv),
        omega0=float(omega0),
        g_ep=float(g_ep),
        n_ph_max=int(ref_nph),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        v_nn=float(v_nn),
        t_prime=float(t_prime),
    )
    half_fill = max(1, int(L) // 2)
    return float(
        _exact_gs_energy_for_problem(
            h_poly,
            problem=str(problem),
            num_sites=int(L),
            num_particles=(half_fill, half_fill),
            indexing=str(ordering),
            n_ph_max=int(ref_nph),
            boson_encoding=str(boson_encoding),
            t=float(t),
            u=float(u),
            dv=float(dv),
            v_nn=float(v_nn),
            t_prime=float(t_prime),
            omega0=float(omega0),
            g_ep=float(g_ep),
            boundary=str(boundary),
        )
    )


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
    objective_weights: StaticObjectiveWeights = StaticObjectiveWeights(),
    run_lifecycle: _ManagedOptunaRunLifecycle | None = None,
    trial_number: int | None = None,
    trial_prune_depth: int | None = None,
    trial_prune_abs_delta_e: float | None = None,
    trial_prune_metric: str = "same_cutoff_abs_delta_e",
) -> BenchmarkResult:
    policy = _normalize_active_policy(policy)
    case_dir = Path(output_dir) / spec.benchmark_id
    (case_dir / "logs").mkdir(parents=True, exist_ok=True)
    (case_dir / "json").mkdir(parents=True, exist_ok=True)
    policy_json = case_dir / "json" / "policy.json"
    result_json = case_dir / "json" / "result.json"
    adapt_current_json = case_dir / "json" / "current.json"
    compile_json = case_dir / "json" / "compile_scout_fake_marrakesh.json"
    _write_json(policy_json, {"pipeline": _PIPELINE_NAME, "generated_utc": _now_utc(), "benchmark": asdict(spec), "policy": asdict(policy)})
    benchmark_target_reference_energy, benchmark_target_reference_nph, benchmark_target_reference_failure, _benchmark_target_reference_source = _benchmark_target_reference_for_spec(
        spec,
        benchmark_target_abs_delta_e=benchmark_target_abs_delta_e,
    )
    command = build_static_command(
        python_bin=python_bin,
        spec=spec,
        policy=policy,
        output_json=result_json,
        adapt_current_json=adapt_current_json,
        adapt_current_json_every_depth=1,
        adapt_current_json_keep_history_tail=100,
        benchmark_target_abs_delta_e=benchmark_target_abs_delta_e,
        benchmark_target_reference_energy=benchmark_target_reference_energy,
    )
    _write_command_log(case_dir / "logs" / "command.sh", command)
    returncode, elapsed = _run_subprocess_logged(
        command,
        cwd=REPO_ROOT,
        stdout_path=case_dir / "logs" / "stdout.log",
        stderr_path=case_dir / "logs" / "stderr.log",
        timeout_s=adapt_timeout_s,
        run_lifecycle=run_lifecycle,
        subprocess_label="adapt",
        benchmark_id=spec.benchmark_id,
        trial_number=trial_number,
        trial_prune_current_json=adapt_current_json,
        trial_prune_depth=trial_prune_depth,
        trial_prune_abs_delta_e=trial_prune_abs_delta_e,
        trial_prune_metric=trial_prune_metric,
        trial_prune_status_path=case_dir / "json" / "trial_prune_gate.json",
    )
    if returncode != 0 or not result_json.exists():
        if returncode == 125:
            reason = "adapt_trial_pruned_by_comparator_gate"
        else:
            reason = "adapt_timeout" if returncode == 124 else f"adapt_returncode:{returncode}"
        result = BenchmarkResult(
            benchmark_id=spec.benchmark_id,
            family=spec.family,
            success=False,
            abs_delta_e=None,
            walltime_s=elapsed,
            failure_reason=reason,
            policy_json=str(policy_json),
        )
        return _ensure_policy_roundtrip_audit(
            result,
            sampled_params=None,
            policy=policy,
            spec=spec,
            output_dir=output_dir,
            objective_weights=objective_weights,
            write_path=case_dir / "json" / "policy_roundtrip_audit.json",
            emitted_command=command,
        )
    compile_command = build_compile_command(
        python_bin=python_bin,
        artifact_json=result_json,
        compile_json=compile_json,
        compile_backend=compile_backend,
        compile_opt_level=compile_opt_level,
        compile_seed=compile_seed,
    )
    _write_command_log(case_dir / "logs" / "compile_command.sh", compile_command)
    compile_returncode, compile_elapsed = _run_subprocess_logged(
        compile_command,
        cwd=REPO_ROOT,
        stdout_path=case_dir / "logs" / "compile_stdout.log",
        stderr_path=case_dir / "logs" / "compile_stderr.log",
        timeout_s=compile_timeout_s,
        run_lifecycle=run_lifecycle,
        subprocess_label="compile",
        benchmark_id=spec.benchmark_id,
        trial_number=trial_number,
    )
    result_payload = json.loads(result_json.read_text(encoding="utf-8"))
    metrics = extract_adapt_energy_metrics(result_payload)
    if benchmark_target_reference_nph is None and benchmark_target_reference_failure is None:
        reference_energy, reference_nph, reference_failure = _reference_cutoff_energy_for_spec(spec)
    else:
        reference_energy, reference_nph, reference_failure = (
            benchmark_target_reference_energy,
            benchmark_target_reference_nph,
            benchmark_target_reference_failure,
        )
    objective_abs_delta_e = metrics.abs_delta_e
    cutoff_abs_delta_e = None
    abs_delta_e_reference = None
    objective_exact_energy = metrics.exact_gs_energy
    if reference_energy is not None and metrics.energy is not None:
        abs_delta_e_reference = abs(float(metrics.energy) - float(reference_energy))
        objective_abs_delta_e = abs_delta_e_reference
        objective_exact_energy = reference_energy
        if metrics.exact_gs_energy is not None:
            cutoff_abs_delta_e = abs(float(metrics.exact_gs_energy) - float(reference_energy))
    count_2q, depth_2q, circuit_depth, logical_params, runtime_params = _compile_metrics(compile_json)
    measurement_groups_proxy, measurement_shots_proxy, shot_cost_proxy = _measurement_cost_proxies(result_payload)
    measurement_proxy_validation = _controller_proxy_validation_from_payload(result_payload)
    measurement_proxy_validated = bool(measurement_proxy_validation.get("valid", False))
    if not measurement_proxy_validated:
        shot_cost_proxy = None
    quality_ok, quality_reason, quality_meta = classify_static_result_quality(result_payload, spec, abs_delta_e=metrics.abs_delta_e)
    target_required = bool(
        benchmark_target_abs_delta_e is not None and float(benchmark_target_abs_delta_e) > 0.0
    )
    payload_target_hit_classification = _payload_target_hit_classification(result_payload)
    target_hit_preclassification = payload_target_hit_classification or _target_hit_classification_payload(
        stop_reason=quality_meta.get("stop_reason"),
        target_error=objective_abs_delta_e,
        target_threshold=benchmark_target_abs_delta_e,
        source="run_static_benchmark_preclassification",
        accepted_crossing_reached=False,
    )
    target_hit_required_ok = bool(
        (not target_required) or target_hit_preclassification.get("target_hit_success", False)
    )
    success = (
        objective_abs_delta_e is not None
        and compile_returncode == 0
        and count_2q is not None
        and quality_ok
        and reference_failure is None
        and target_hit_required_ok
    )
    failure_reasons: list[str] = []
    if compile_returncode != 0:
        failure_reasons.append("compile_timeout" if compile_returncode == 124 else f"compile_returncode:{compile_returncode}")
    if count_2q is None:
        failure_reasons.append("missing_compile_2q_count")
    if not quality_ok and quality_reason is not None:
        failure_reasons.append(f"quality_gate:{quality_reason}")
    if reference_failure is not None:
        failure_reasons.append(reference_failure)
    if target_required and not target_hit_required_ok:
        failure_reasons.append(
            "benchmark_target_non_hit:"
            + str(target_hit_preclassification.get("non_hit_reason", "target_hit_not_reached"))
        )
    result = BenchmarkResult(
        benchmark_id=spec.benchmark_id,
        family=spec.family,
        success=bool(success),
        abs_delta_e=objective_abs_delta_e,
        energy=metrics.energy,
        exact_gs_energy=objective_exact_energy,
        same_cutoff_exact_gs_energy=metrics.exact_gs_energy,
        exact_reference_energy=reference_energy,
        exact_reference_n_ph_max=reference_nph,
        abs_delta_e_same_cutoff=metrics.abs_delta_e,
        abs_delta_e_reference=abs_delta_e_reference,
        cutoff_abs_delta_e=cutoff_abs_delta_e,
        count_2q=count_2q,
        depth_2q=depth_2q,
        circuit_depth=circuit_depth,
        parameter_count=logical_params,
        runtime_parameter_count=runtime_params,
        measurement_groups_proxy=measurement_groups_proxy,
        measurement_shots_proxy=measurement_shots_proxy,
        shot_cost_proxy=shot_cost_proxy,
        walltime_s=float(elapsed + compile_elapsed),
        failure_reason=None if success else ";".join(failure_reasons),
        stop_reason=quality_meta.get("stop_reason"),
        ansatz_depth=quality_meta.get("ansatz_depth"),
        initial_energy=quality_meta.get("initial_energy"),
        initial_abs_delta_e=quality_meta.get("initial_abs_delta_e"),
        max_gradient_seen=quality_meta.get("max_gradient_seen"),
        quality_gate_reason=quality_reason,
        boson_illegal_probability_max=_as_float_or_none(quality_meta.get("boson_illegal_probability_max")),
        boson_legal_probability_min=_as_float_or_none(quality_meta.get("boson_legal_probability_min")),
        measurement_proxy_validated=bool(measurement_proxy_validated),
        measurement_proxy_validation=dict(measurement_proxy_validation),
        result_json=str(result_json),
        compile_json=str(compile_json) if compile_json.exists() else None,
        policy_json=str(policy_json),
    )
    physical_target_manifest, cutoff_diagnostics, paper_i_first_crossing = _paper_i_artifacts_for_result(
        result,
        spec,
        result_payload=result_payload,
    )
    target_hit_classification = payload_target_hit_classification or target_hit_classification_for_result(
        result,
        target_abs_delta_e=benchmark_target_abs_delta_e,
        first_crossing=paper_i_first_crossing,
        source="run_static_benchmark",
    )
    _write_paper_i_artifacts_into_result_json(
        result_json,
        result_payload=result_payload,
        physical_target_manifest=physical_target_manifest,
        cutoff_diagnostics=cutoff_diagnostics,
        paper_i_first_crossing=paper_i_first_crossing,
        target_hit_classification=target_hit_classification,
    )
    result = replace(
        result,
        physical_target_manifest=physical_target_manifest,
        cutoff_diagnostics=cutoff_diagnostics,
        paper_i_first_crossing=paper_i_first_crossing,
        target_hit_classification=target_hit_classification,
    )
    return _ensure_policy_roundtrip_audit(
        result,
        sampled_params=None,
        policy=policy,
        spec=spec,
        output_dir=output_dir,
        objective_weights=objective_weights,
        result_payload=result_payload,
        write_path=case_dir / "json" / "policy_roundtrip_audit.json",
        emitted_command=command,
    )


def normalized_static_score(
    result: BenchmarkResult,
    spec: HamiltonianBenchmarkSpec,
    weights: StaticObjectiveWeights = StaticObjectiveWeights(),
) -> float:
    eps = float(weights.epsilon_energy)
    baseline_energy = max(float(spec.baseline_abs_delta_e), eps)
    if not result.success or result.abs_delta_e is None:
        score = float(weights.fail_penalty)
        if result.abs_delta_e is not None:
            score += max(0.0, float(weights.energy) * math.log((float(result.abs_delta_e) + eps) / (baseline_energy + eps)))
        illegal_probability = _as_float_or_none(result.boson_illegal_probability_max)
        if illegal_probability is not None:
            score += min(10.0, 10.0 * max(0.0, float(illegal_probability)))
        return float(score)
    if weights.target_abs_delta_e is not None and float(weights.target_abs_delta_e) > 0.0:
        target = max(float(weights.target_abs_delta_e), eps)
        # Soft cost-at-accuracy target: below target, stop rewarding extra
        # accuracy; above target, add a smooth violation penalty while still
        # keeping 2q/depth/parameter/shot-cost terms active.  The final accept
        # decision remains a report/user decision rather than a hard optimizer
        # gate at the nominal target.
        energy_score = max(0.0, math.log((float(result.abs_delta_e) + eps) / (target + eps)))
    else:
        energy_score = math.log((float(result.abs_delta_e) + eps) / (baseline_energy + eps))
    score = float(weights.energy) * energy_score
    if result.count_2q is not None and spec.baseline_count_2q is not None:
        score += float(weights.count_2q) * math.log((float(result.count_2q) + 1.0) / (float(spec.baseline_count_2q) + 1.0))
    if result.depth_2q is not None and spec.baseline_depth_2q is not None:
        score += float(weights.depth_2q) * math.log((float(result.depth_2q) + 1.0) / (float(spec.baseline_depth_2q) + 1.0))
    if result.circuit_depth is not None and spec.baseline_depth_2q is not None:
        score += float(weights.circuit_depth) * math.log((float(result.circuit_depth) + 1.0) / (float(spec.baseline_depth_2q) + 1.0))
    if result.parameter_count is not None and spec.baseline_parameter_count is not None:
        score += float(weights.parameters) * math.log((float(result.parameter_count) + 1.0) / (float(spec.baseline_parameter_count) + 1.0))
    if result.shot_cost_proxy is not None and bool(result.measurement_proxy_validated):
        baseline_shot = float(spec.baseline_shot_cost_proxy) if spec.baseline_shot_cost_proxy is not None else 1.0
        score += float(weights.shot_cost) * math.log((float(result.shot_cost_proxy) + 1.0) / (baseline_shot + 1.0))
    return float(score)


def discovery_first_crossing_score_components(
    result: BenchmarkResult | None,
    spec: HamiltonianBenchmarkSpec,
    config: GlobalObjectiveConfig,
) -> dict[str, Any]:
    """Feasible-first Paper-I score: cross tau_phys, then minimize first-crossing resource."""

    penalty_base = max(float(config.required_target_penalty), float(config.weights.fail_penalty), 1_000_000.0)
    if result is None:
        return {
            "schema": "paper_i_discovery_first_crossing_score_v1",
            "objective_mode": _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING,
            "score": penalty_base,
            "feasible": False,
            "primary_error_status": "missing_result",
            "violation_reason": "missing_result",
            "resource_score_source": "none_infeasible",
        }
    try:
        physical_target, cutoff_diagnostics, first_crossing = _paper_i_artifacts_for_result(result, spec)
    except Exception as exc:
        physical_target = _audited_missing(f"physical_target_failed:{type(exc).__name__}:{exc}", required_for_cthc=False)
        cutoff_diagnostics = _audited_missing(f"cutoff_diagnostics_failed:{type(exc).__name__}:{exc}", required_for_cthc=False)
        first_crossing = _audited_missing(f"first_crossing_failed:{type(exc).__name__}:{exc}", required_for_cthc=False)
    tau = _as_float_or_none(physical_target.get("tau_phys") if isinstance(physical_target, Mapping) else None)
    feasibility_metric = (
        str(physical_target.get("accuracy_gate_metric") or _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC)
        if isinstance(physical_target, Mapping)
        else _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC
    )
    terminal_primary = _as_float_or_none(
        first_crossing.get("terminal_primary_error") if isinstance(first_crossing, Mapping) else None
    )
    if terminal_primary is None:
        terminal_primary = _as_float_or_none(result.abs_delta_e_reference)
    if terminal_primary is None:
        terminal_primary = _as_float_or_none(result.abs_delta_e)
    reached = isinstance(first_crossing, Mapping) and bool(first_crossing.get("reached"))
    target_hit_classification = target_hit_classification_for_result(
        result,
        target_abs_delta_e=tau,
        first_crossing=first_crossing if isinstance(first_crossing, Mapping) else None,
        source="discovery_first_crossing_score",
    )
    if not bool(result.success):
        return {
            "schema": "paper_i_discovery_first_crossing_score_v1",
            "objective_mode": _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING,
            "score": float(penalty_base * 10.0),
            "feasible": False,
            "primary_error_status": "unsuccessful_result",
            "primary_error": terminal_primary,
            "feasibility_threshold": tau,
            "violation_reason": "unsuccessful_result",
            "failure_reason": result.failure_reason,
            "target_hit_classification": _jsonable(target_hit_classification),
            "paper_i_first_crossing": _jsonable(first_crossing),
            "physical_target_manifest": _jsonable(physical_target),
            "cutoff_diagnostics": _jsonable(cutoff_diagnostics),
            "resource_score_source": "none_infeasible",
        }
    if (not reached) or tau is None:
        eps = max(float(config.weights.epsilon_energy), 1e-12)
        if tau is not None and terminal_primary is not None:
            violation_log = max(0.0, math.log((float(terminal_primary) + eps) / (float(tau) + eps)))
            penalty = penalty_base * (1.0 + violation_log)
            if (
                isinstance(first_crossing, Mapping)
                and first_crossing.get("status") == "non_target_terminal"
            ):
                primary_error_status = "non_target_terminal"
            else:
                primary_error_status = "above_threshold" if terminal_primary > tau else "no_history_or_missing_crossing"
        else:
            violation_log = None
            penalty = penalty_base * 10.0
            primary_error_status = "missing_primary_error_or_threshold"
        return {
            "schema": "paper_i_discovery_first_crossing_score_v1",
            "objective_mode": _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING,
            "score": float(penalty),
            "feasible": False,
            "primary_error_status": primary_error_status,
            "primary_error": terminal_primary,
            "feasibility_threshold": tau,
            "violation_log": violation_log,
            "violation_reason": (
                target_hit_classification.get("non_hit_reason")
                if isinstance(first_crossing, Mapping)
                and first_crossing.get("status") == "non_target_terminal"
                else (
                    first_crossing.get("status", "not_reached")
                    if isinstance(first_crossing, Mapping)
                    else "first_crossing_missing"
                )
            ),
            "target_hit_classification": _jsonable(target_hit_classification),
            "paper_i_first_crossing": _jsonable(first_crossing),
            "physical_target_manifest": _jsonable(physical_target),
            "cutoff_diagnostics": _jsonable(cutoff_diagnostics),
            "resource_score_source": "none_infeasible",
        }
    if not bool(target_hit_classification.get("target_hit_success", False)):
        return {
            "schema": "paper_i_discovery_first_crossing_score_v1",
            "objective_mode": _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING,
            "score": float(penalty_base * 10.0),
            "feasible": False,
            "primary_error_status": "non_target_terminal",
            "primary_error": terminal_primary,
            "feasibility_threshold": tau,
            "violation_reason": target_hit_classification.get("non_hit_reason", "target_hit_not_reached"),
            "target_hit_classification": _jsonable(target_hit_classification),
            "paper_i_first_crossing": _jsonable(first_crossing),
            "physical_target_manifest": _jsonable(physical_target),
            "cutoff_diagnostics": _jsonable(cutoff_diagnostics),
            "resource_score_source": "none_infeasible",
        }
    resource_score = _as_float_or_none(first_crossing.get("resource_score"))
    if resource_score is None:
        resource_score = _as_float_or_none(first_crossing.get("operator_count_at_crossing"))
    if resource_score is None:
        resource_score = _as_float_or_none(first_crossing.get("k_tau"))
    if resource_score is None:
        resource_score = _as_float_or_none(first_crossing.get("parameter_count_at_crossing"))
    if resource_score is None:
        return {
            "schema": "paper_i_discovery_first_crossing_score_v1",
            "objective_mode": _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING,
            "score": float(penalty_base * 10.0),
            "feasible": False,
            "primary_error_status": "missing_first_crossing_resource",
            "primary_error": terminal_primary,
            "feasibility_threshold": float(tau),
            "violation_reason": "missing_first_crossing_resource",
            "target_hit_classification": _jsonable(target_hit_classification),
            "paper_i_first_crossing": _jsonable(first_crossing),
            "physical_target_manifest": _jsonable(physical_target),
            "cutoff_diagnostics": _jsonable(cutoff_diagnostics),
            "resource_score_source": "none_infeasible",
        }
    params = _as_float_or_none(first_crossing.get("parameter_count_at_crossing"))
    crossing_primary = _as_float_or_none(first_crossing.get("primary_error_at_crossing"))
    def _bounded_tie(value: float | None) -> float:
        raw = max(0.0, float(value or 0.0))
        return raw / (1.0 + raw)

    stop_reason = str(result.stop_reason or "").strip().lower()
    accepted_count = _as_int_or_none(first_crossing.get("accepted_history_row_count"))
    k_tau_int = _as_int_or_none(first_crossing.get("k_tau"))
    terminal_is_first_hit = bool(
        stop_reason == "benchmark_abs_delta_e_target"
        and accepted_count is not None
        and k_tau_int is not None
        and int(accepted_count) == int(k_tau_int)
    )

    def _relative_log_cost(value: float | None, baseline: float | None = None) -> float | None:
        numeric = _as_float_or_none(value)
        if numeric is None:
            return None
        if baseline is not None and float(baseline) > 0.0:
            return math.log((float(numeric) + 1.0) / (float(baseline) + 1.0))
        return math.log(float(numeric) + 1.0)

    hardware_terms: dict[str, float] = {}
    hardware_score: float | None = None
    if terminal_is_first_hit and result.count_2q is not None:
        hardware_score = 0.0
        term = _relative_log_cost(result.count_2q, spec.baseline_count_2q)
        if term is not None:
            hardware_terms["count_2q"] = float(term)
            hardware_score += float(config.weights.count_2q) * float(term)
        term = _relative_log_cost(result.depth_2q, spec.baseline_depth_2q)
        if term is not None:
            hardware_terms["depth_2q"] = float(term)
            hardware_score += float(config.weights.depth_2q) * float(term)
        term = _relative_log_cost(result.circuit_depth, spec.baseline_depth_2q)
        if term is not None:
            hardware_terms["circuit_depth"] = float(term)
            hardware_score += float(config.weights.circuit_depth) * float(term)
        term = _relative_log_cost(params, spec.baseline_parameter_count)
        if term is not None:
            hardware_terms["parameter_count"] = float(term)
            hardware_score += float(config.weights.parameters) * float(term)
        if result.shot_cost_proxy is not None and bool(result.measurement_proxy_validated):
            term = _relative_log_cost(result.shot_cost_proxy, spec.baseline_shot_cost_proxy)
            if term is not None:
                hardware_terms["shot_cost_proxy"] = float(term)
                hardware_score += float(config.weights.shot_cost) * float(term)
        # Keep first crossing before terminal-cost ties if all weights are zero.
        hardware_score += 1e-9 * _bounded_tie(resource_score)
        hardware_score += 1e-12 * _bounded_tie(crossing_primary)

    if hardware_score is not None and hardware_terms:
        score = float(hardware_score)
        resource_score_source = "terminal_qiskit_compile_at_benchmark_target_stop"
        qiskit_cost_status = "recoverable_terminal_first_hit"
    else:
        # Feasible but no recoverable compiled first-hit cost. Keep it in the
        # feasible set but rank it behind recoverable hardware-cost evidence.
        score = float(100000.0 + resource_score)
        score += 1e-9 * _bounded_tie(params)
        score += 1e-12 * _bounded_tie(crossing_primary)
        resource_score_source = "first_crossing_history_surrogate_qiskit_unavailable"
        qiskit_cost_status = "missing_recoverable_first_hit_qiskit_cost"
    return {
        "schema": "paper_i_discovery_first_crossing_score_v1",
        "objective_mode": _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING,
        "score": float(score),
        "feasible": True,
        "primary_error_status": "crossed_tau_phys",
        "primary_error": terminal_primary,
        "feasibility_metric": feasibility_metric,
        "feasibility_threshold": float(tau),
        "resource_score": float(resource_score),
        "resource_score_source": resource_score_source,
        "qiskit_first_hit_cost_status": qiskit_cost_status,
        "terminal_compile_matches_first_hit": bool(terminal_is_first_hit),
        "target_hit_classification": _jsonable(target_hit_classification),
        "paper_facing_resource_display_allowed": bool(terminal_is_first_hit and hardware_score is not None and hardware_terms),
        "paper_facing_resource_display_reason": (
            "benchmark_target_stop_terminal_compile_is_first_hit"
            if bool(terminal_is_first_hit and hardware_score is not None and hardware_terms)
            else "qiskit_first_hit_cost_not_recoverable"
        ),
        "hardware_score_terms": dict(sorted(hardware_terms.items())),
        "terminal_compiled_resources": {
            "count_2q": result.count_2q,
            "depth_2q": result.depth_2q,
            "circuit_depth": result.circuit_depth,
            "shot_cost_proxy": result.shot_cost_proxy if bool(result.measurement_proxy_validated) else None,
            "measurement_proxy_validated": bool(result.measurement_proxy_validated),
        },
        "first_crossing_resource_score_source": first_crossing.get("resource_score_source", "first_crossing"),
        "parameter_count_tie": params,
        "primary_error_at_crossing_tie": crossing_primary,
        "deterministic_tie_order": [
            "recoverable_qiskit_first_hit_hardware_score",
            "first_crossing_resource_score_surrogate",
            "first_crossing_parameter_count",
            "primary_error_at_crossing",
            "trial_number",
        ],
        "paper_i_first_crossing": _jsonable(first_crossing),
        "physical_target_manifest": _jsonable(physical_target),
        "cutoff_diagnostics": _jsonable(cutoff_diagnostics),
    }


def discovery_first_crossing_score(
    result: BenchmarkResult | None,
    spec: HamiltonianBenchmarkSpec,
    config: GlobalObjectiveConfig,
) -> float:
    return float(discovery_first_crossing_score_components(result, spec, config)["score"])


def _required_target_violation(
    result: BenchmarkResult | None,
    *,
    target_abs_delta_e: float | None,
    penalty_scale: float,
) -> tuple[float, dict[str, Any] | None]:
    if target_abs_delta_e is None or float(target_abs_delta_e) <= 0.0:
        return 0.0, None
    target = max(float(target_abs_delta_e), 1e-12)
    if result is None:
        penalty = float(penalty_scale)
        return penalty, {"reason": "missing_result", "penalty": penalty, "target_abs_delta_e": target}
    if (not bool(result.success)) or result.abs_delta_e is None:
        penalty = float(penalty_scale)
        return penalty, {
            "reason": "unsuccessful_or_missing_energy",
            "penalty": penalty,
            "target_abs_delta_e": target,
            "success": bool(result.success),
            "abs_delta_e": result.abs_delta_e,
            "failure_reason": result.failure_reason,
        }
    delta = float(result.abs_delta_e)
    target_hit_classification = target_hit_classification_for_result(
        result,
        target_abs_delta_e=target,
        source="required_target_violation",
    )
    if not bool(target_hit_classification.get("target_hit_success", False)):
        penalty = float(penalty_scale)
        return penalty, {
            "reason": target_hit_classification.get("non_hit_reason", "target_hit_not_reached"),
            "penalty": penalty,
            "target_abs_delta_e": target,
            "abs_delta_e": delta,
            "success": bool(result.success),
            "failure_reason": result.failure_reason,
            "target_hit_classification": _jsonable(target_hit_classification),
        }
    if delta <= target:
        return 0.0, None
    violation_log = max(0.0, math.log((delta + 1e-12) / (target + 1e-12)))
    penalty = float(penalty_scale) * (1.0 + violation_log)
    return penalty, {
        "reason": "above_required_target",
        "penalty": penalty,
        "target_abs_delta_e": target,
        "abs_delta_e": delta,
        "violation_log": violation_log,
        "success": bool(result.success),
        "failure_reason": result.failure_reason,
    }


def required_target_violations(
    results_by_id: Mapping[str, BenchmarkResult],
    specs: Sequence[HamiltonianBenchmarkSpec],
    config: GlobalObjectiveConfig = GlobalObjectiveConfig(),
) -> dict[str, dict[str, Any]]:
    required = {str(x) for x in config.required_target_benchmark_ids}
    if not required:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.benchmark_id not in required:
            continue
        _, payload = _required_target_violation(
            results_by_id.get(spec.benchmark_id),
            target_abs_delta_e=config.required_target_abs_delta_e,
            penalty_scale=config.required_target_penalty,
        )
        if payload is not None:
            out[spec.benchmark_id] = dict(payload)
    return out


def resolved_objective_weight_rows(
    specs: Sequence[HamiltonianBenchmarkSpec],
    config: GlobalObjectiveConfig = GlobalObjectiveConfig(),
) -> tuple[dict[str, Any], ...]:
    preset_key = _normalize_objective_weight_preset(config.objective_weight_preset)
    preset_family_weights = {
        str(key): _validate_objective_weight(value, label=f"objective_weight_preset[{preset_key!r}][{key!r}]")
        for key, value in _OBJECTIVE_WEIGHT_PRESETS[preset_key].items()
    }
    family_overrides = _validate_objective_weight_map(config.objective_family_weights, label="objective_family_weights")
    benchmark_overrides = _validate_objective_weight_map(
        config.objective_benchmark_weights,
        label="objective_benchmark_weights",
    )
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        benchmark_id = str(spec.benchmark_id)
        family = str(spec.family)
        if benchmark_id in benchmark_overrides:
            weight = benchmark_overrides[benchmark_id]
            source = "benchmark_override"
        elif family in family_overrides:
            weight = family_overrides[family]
            source = "family_override"
        elif family in preset_family_weights:
            weight = preset_family_weights[family]
            source = "preset_family"
        else:
            weight = 1.0
            source = "default"
        rows.append(
            {
                "index": int(idx),
                "benchmark_id": benchmark_id,
                "family": family,
                "weight": float(weight),
                "source": source,
            }
        )
    return tuple(rows)


def objective_weighting_payload(
    specs: Sequence[HamiltonianBenchmarkSpec],
    config: GlobalObjectiveConfig = GlobalObjectiveConfig(),
) -> dict[str, Any]:
    preset_key = _normalize_objective_weight_preset(config.objective_weight_preset)
    rows = resolved_objective_weight_rows(specs, config)
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row["source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "schema": "phase3_objective_weighting_v1",
        "preset": preset_key,
        "default_weight": 1.0,
        "normalization": "weighted_mean_sum_weights",
        "preset_family_weights": {key: float(value) for key, value in sorted(_OBJECTIVE_WEIGHT_PRESETS[preset_key].items())},
        "family_overrides": {
            key: float(value)
            for key, value in sorted(_validate_objective_weight_map(config.objective_family_weights, label="objective_family_weights").items())
        },
        "benchmark_overrides": {
            key: float(value)
            for key, value in sorted(
                _validate_objective_weight_map(
                    config.objective_benchmark_weights,
                    label="objective_benchmark_weights",
                ).items()
            )
        },
        "resolved_weights": [dict(row) for row in rows],
        "total_weight": float(sum(float(row["weight"]) for row in rows)),
        "weight_source_counts": dict(sorted(source_counts.items())),
    }


def aggregate_global_score_components(
    results_by_id: Mapping[str, BenchmarkResult],
    specs: Sequence[HamiltonianBenchmarkSpec],
    config: GlobalObjectiveConfig = GlobalObjectiveConfig(),
) -> dict[str, Any]:
    selected = tuple(specs)
    objective_mode = _normalize_discovery_objective_mode(config.discovery_objective_mode)
    weight_rows = resolved_objective_weight_rows(selected, config)
    scores: list[float] = []
    weighted_terms: list[float] = []
    by_family: dict[str, list[tuple[float, float]]] = {}
    failures = 0
    per_benchmark: list[dict[str, Any]] = []
    required = {str(x) for x in config.required_target_benchmark_ids}
    for spec, weight_row in zip(selected, weight_rows):
        result = results_by_id.get(spec.benchmark_id)
        missing_result = result is None
        unsuccessful_result = False
        failure_reason = None
        discovery_components = None
        if objective_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING:
            discovery_components = discovery_first_crossing_score_components(result, spec, config)
            score = float(discovery_components["score"])
            unsuccessful_result = False if result is None else not bool(result.success)
            failure_reason = None if result is None else result.failure_reason
            if missing_result or unsuccessful_result or not bool(discovery_components.get("feasible", False)):
                failures += 1
        elif result is None:
            failures += 1
            score = float(config.weights.fail_penalty)
        else:
            unsuccessful_result = not bool(result.success)
            failure_reason = result.failure_reason
            if unsuccessful_result:
                failures += 1
            score = normalized_static_score(result, spec, config.weights)
        required_penalty = 0.0
        required_violation = None
        if objective_mode != _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING and spec.benchmark_id in required:
            required_penalty, required_violation = _required_target_violation(
                result,
                target_abs_delta_e=config.required_target_abs_delta_e,
                penalty_scale=config.required_target_penalty,
            )
            if required_penalty > 0.0:
                failures += 1
                score += float(required_penalty)
        weight = float(weight_row["weight"])
        scores.append(float(score))
        weighted_terms.append(float(weight * score))
        by_family.setdefault(spec.family, []).append((float(score), weight))
        per_benchmark_target_hit_classification = (
            None
            if result is None
            else target_hit_classification_for_result(
                result,
                target_abs_delta_e=(
                    config.required_target_abs_delta_e
                    if spec.benchmark_id in required
                    else None
                ),
                source="aggregate_global_score_components",
            )
        )
        per_benchmark.append(
            {
                "benchmark_id": spec.benchmark_id,
                "family": spec.family,
                "score": float(score),
                "weight": weight,
                "weighted_score": float(weight * score),
                "weight_source": weight_row["source"],
                "success": None if result is None else bool(result.success),
                "missing_result": bool(missing_result),
                "unsuccessful_result": bool(unsuccessful_result),
                "failure_reason": failure_reason,
                "objective_mode": objective_mode,
                "target_hit_classification": _jsonable(per_benchmark_target_hit_classification),
                "target_hit_success": (
                    None
                    if per_benchmark_target_hit_classification is None
                    else bool(per_benchmark_target_hit_classification.get("target_hit_success", False))
                ),
                "discovery_first_crossing": _jsonable(discovery_components) if discovery_components is not None else None,
                "required_target_penalty": float(required_penalty),
                "required_target_violation": required_violation,
            }
        )
    if not scores:
        return {
            "schema": "phase3_global_score_components_v1",
            "score": float(_LARGE_OBJECTIVE),
            "reason": "empty_specs",
            "objective_weighting": objective_weighting_payload(selected, config),
            "objective_provenance": objective_provenance_payload(config),
            "discovery_objective_mode": objective_mode,
        }
    total_weight = float(sum(float(row["weight"]) for row in weight_rows))
    weighted_mean_score = float(sum(weighted_terms) / total_weight)
    unweighted_mean_score = float(sum(scores) / len(scores))
    family_rows: list[dict[str, Any]] = []
    family_means: list[float] = []
    for family, values in by_family.items():
        family_weight_sum = float(sum(weight for _score, weight in values))
        family_mean = float(sum(score * weight for score, weight in values) / family_weight_sum)
        family_rows.append(
            {
                "family": family,
                "weighted_mean_score": family_mean,
                "total_weight": family_weight_sum,
                "benchmark_count": len(values),
            }
        )
        family_means.append(family_mean)
    family_std = 0.0
    if len(family_means) > 1:
        fam_mean = sum(family_means) / len(family_means)
        family_std = math.sqrt(sum((x - fam_mean) ** 2 for x in family_means) / len(family_means))
    fail_rate = float(failures / len(selected))
    cvar_score = _cvar(scores, config.robust_cvar_quantile)
    robust_component = float(config.gamma_robust) * float(cvar_score)
    family_std_component = float(config.gamma_family_std) * float(family_std)
    failure_event_penalty = float(config.gamma_fail) * float(fail_rate)
    score = float(weighted_mean_score + robust_component + family_std_component + failure_event_penalty)
    return {
        "schema": "phase3_global_score_components_v1",
        "score": score,
        "discovery_objective_mode": objective_mode,
        "mean_score": weighted_mean_score,
        "weighted_mean_score": weighted_mean_score,
        "unweighted_mean_score": unweighted_mean_score,
        "robust_cvar_score": float(cvar_score),
        "robust_cvar_quantile": float(config.robust_cvar_quantile),
        "robust_component": robust_component,
        "family_std": float(family_std),
        "family_std_component": family_std_component,
        "failures": int(failures),
        "failure_events": int(failures),
        "fail_rate": fail_rate,
        "failure_event_penalty": failure_event_penalty,
        "gamma_robust": float(config.gamma_robust),
        "gamma_family_std": float(config.gamma_family_std),
        "gamma_fail": float(config.gamma_fail),
        "objective_weighting": objective_weighting_payload(selected, config),
        "objective_provenance": objective_provenance_payload(config),
        "per_benchmark_scores": per_benchmark,
        "family_scores": family_rows,
    }


def aggregate_global_score(
    results_by_id: Mapping[str, BenchmarkResult],
    specs: Sequence[HamiltonianBenchmarkSpec],
    config: GlobalObjectiveConfig = GlobalObjectiveConfig(),
) -> float:
    return float(aggregate_global_score_components(results_by_id, specs, config)["score"])


def oracle_gap(agnostic_score: float, oracle_score: float, epsilon: float = 1e-12) -> float:
    return float((float(agnostic_score) - float(oracle_score)) / (float(oracle_score) + float(epsilon)))


def _suggest_logit_weights(trial: Any, prefix: str, names: Sequence[str], *, scale_low: float, scale_high: float) -> dict[str, float]:
    logits = [float(trial.suggest_float(f"{prefix}_logit_{name}", -3.0, 3.0)) for name in names]
    scale = float(trial.suggest_float(f"{prefix}_scale", scale_low, scale_high, log=True))
    return {str(name): float(scale * weight) for name, weight in zip(names, _softmax(logits))}


def sample_policy_from_trial(
    trial: Any,
    *,
    base_policy: AlgorithmPolicy | None = None,
    policy_search_profile: str | None = "default",
    meta_feature_profile: str | None = _DEFAULT_META_FEATURE_PROFILE,
    trial_param_overrides: Mapping[str, Any] | None = None,
) -> AlgorithmPolicy:
    base = base_policy or AlgorithmPolicy.default()
    policy_profile_key = _normalize_policy_search_profile(policy_search_profile)
    novelty_surface_profile = policy_profile_key == "hh_novelty_surface_v1"
    snake_u8_profile = policy_profile_key in _SNAKE_U8_NOVELTY_PROFILES
    meta_profile = _normalize_meta_feature_profile(meta_feature_profile)
    production_profile = meta_profile == _META_FEATURE_PROFILE_PAPER_I_PRODUCTION
    if production_profile:
        _ensure_paper_i_production_phase2_novelty_mode(meta_profile)
    phase2_motif_bonus_weight: float | None = None
    phase2_gamma_N = float(base.static.phase2_gamma_N)
    phase2_gamma_N_schedule_mode = str(base.static.phase2_gamma_N_schedule_mode)
    phase2_gamma_N_schedule_start = base.static.phase2_gamma_N_schedule_start
    phase2_gamma_N_schedule_end = base.static.phase2_gamma_N_schedule_end
    sampled_novelty_bonus: float | None = None
    if novelty_surface_profile:
        sampled_novelty_bonus = float(trial.suggest_float("novelty_bonus", 0.0, 0.5))
        phase2_motif_bonus_weight = float(sampled_novelty_bonus)
        phase2_gamma_N_schedule_mode = str(
            trial.suggest_categorical("phase2_gamma_N_schedule_mode", ["fixed", "depth_linear_v1"])
        )
        if phase2_gamma_N_schedule_mode == "depth_linear_v1":
            phase2_gamma_N = 1.0
            phase2_gamma_N_schedule_start = float(
                trial.suggest_float("phase2_gamma_N_schedule_start", 0.0, 4.0)
            )
            phase2_gamma_N_schedule_end = float(
                trial.suggest_float("phase2_gamma_N_schedule_end", 0.0, 1.5)
            )
        else:
            phase2_gamma_N = float(trial.suggest_float("phase2_gamma_N", 0.0, 3.0))
            phase2_gamma_N_schedule_start = None
            phase2_gamma_N_schedule_end = None
    elif snake_u8_profile:
        sampled_novelty_bonus = 0.0
        phase2_motif_bonus_weight = 0.0
        if policy_profile_key == "snake_u8_no_novelty_v1":
            phase2_gamma_N = 0.0
            phase2_gamma_N_schedule_mode = "fixed"
            phase2_gamma_N_schedule_start = None
            phase2_gamma_N_schedule_end = None
        elif policy_profile_key == "snake_u8_flat_novelty_v1":
            phase2_gamma_N = float(trial.suggest_float("phase2_gamma_N", 0.0, 3.0))
            phase2_gamma_N_schedule_mode = "fixed"
            phase2_gamma_N_schedule_start = None
            phase2_gamma_N_schedule_end = None
        elif policy_profile_key == "snake_u8_exponent_novelty_v1":
            phase2_gamma_N = 1.0
            phase2_gamma_N_schedule_mode = "depth_linear_v1"
            phase2_gamma_N_schedule_start = float(
                trial.suggest_float("phase2_gamma_N_schedule_start", 0.0, 4.0)
            )
            phase2_gamma_N_schedule_end = float(
                trial.suggest_float("phase2_gamma_N_schedule_end", 0.0, 1.5)
            )
    cost = _suggest_logit_weights(trial, "cost", ("depth", "group", "shot", "optdim", "reuse"), scale_low=1e-3, scale_high=10.0)
    hardware_cost = _suggest_logit_weights(
        trial,
        "hardware_cost",
        ("2q", "d", "1q", "theta", "shot"),
        scale_low=1e-4,
        scale_high=10.0,
    )
    burden = _suggest_logit_weights(trial, "burden", ("compile", "measure"), scale_low=1e-4, scale_high=1.0)
    if production_profile:
        compile_cost = _suggest_logit_weights(
            trial,
            "compile_cost",
            ("cx", "sq", "rotation_step", "refit_active"),
            scale_low=1e-2,
            scale_high=20.0,
        )
        compile_cost["position_shift"] = 0.0
    else:
        compile_cost = _suggest_logit_weights(
            trial,
            "compile_cost",
            ("cx", "sq", "rotation_step", "position_shift", "refit_active"),
            scale_low=1e-2,
            scale_high=20.0,
        )
    measure_cost = _suggest_logit_weights(
        trial,
        "measure_cost",
        ("groups", "shots", "reuse"),
        scale_low=1e-2,
        scale_high=20.0,
    )
    phase1_budget = SizeScaledBudget(
        min_count=int(trial.suggest_int("phase1_min_count", 8, 96)),
        max_count=int(trial.suggest_int("phase1_max_count", 96, 100000, log=True)),
        pool_fraction=float(trial.suggest_float("phase1_pool_fraction", 0.05, 1.0)),
        qubit_slope=float(trial.suggest_float("phase1_qubit_slope", 1.0, 16.0)),
    )
    phase2_budget = SizeScaledBudget(
        min_count=int(trial.suggest_int("phase2_min_count", 4, 64)),
        max_count=int(trial.suggest_int("phase2_max_count", 64, 100000, log=True)),
        pool_fraction=float(trial.suggest_float("phase2_pool_fraction", 0.05, 1.0)),
        qubit_slope=float(trial.suggest_float("phase2_qubit_slope", 1.0, 12.0)),
    )
    if production_profile:
        pool = replace(
            base.pool,
            pool_key="full_meta",
            family_repeat_penalty=0.0,
            novelty_bonus=0.0 if sampled_novelty_bonus is None else float(sampled_novelty_bonus),
            phase1_budget=phase1_budget,
            phase2_budget=phase2_budget,
        )
    else:
        pool = replace(
            base.pool,
            pool_key=str(trial.suggest_categorical("pool_key", ["full_meta", "hamiltonian_quadratures"])),
            family_repeat_penalty=float(trial.suggest_float("family_repeat_penalty", 0.0, 5.0)),
            novelty_bonus=float(trial.suggest_float("novelty_bonus", 0.0, 0.5)),
            phase1_budget=phase1_budget,
            phase2_budget=phase2_budget,
        )
    phase2_batch_size_cap = int(trial.suggest_int("phase2_batch_size_cap", 4, 32))
    phase2_batch_target_size = min(
        int(trial.suggest_int("phase2_batch_target_size", 2, 16)),
        int(phase2_batch_size_cap),
    )
    force_profile_batching = policy_profile_key in {
        "spin_boson_2q_batching_v1",
        "route_a_batching_v1",
        *_SNAKE_U8_NOVELTY_PROFILES,
    }
    if production_profile:
        feature_phase0_pilot_enabled = True
        feature_phase0_algebraic_lane_mode = "weak"
        feature_phase3_batching_enabled = True if force_profile_batching else bool(
            trial.suggest_categorical("feature_phase3_batching_enabled", [True, False])
        )
        feature_phase1_prune_enabled = True
        feature_phase1_prune_amplitude_witness_required = True
        feature_phase3_selector_policy = _DEFAULT_PHASE3_SELECTOR_POLICY
        feature_phase3_novelty_ablation_mode = "all" if policy_profile_key == "snake_u8_no_novelty_v1" else "off"
        feature_phase3_window_relaxation_mode = "reduced"
        feature_phase3_batch_selection_mode = "reduced_plane"
        feature_phase3_batch_prefilter_mode = "off"
    else:
        feature_phase0_pilot_enabled = bool(trial.suggest_categorical("feature_phase0_pilot_enabled", [True, False]))
        feature_phase0_algebraic_lane_mode = str(
            trial.suggest_categorical("feature_phase0_algebraic_lane_mode", list(PHASE0_ALGEBRAIC_LANE_MODE_CHOICES))
        )
    if meta_profile == _META_FEATURE_PROFILE_SAFE_CORE:
        feature_phase3_batching_enabled = bool(trial.suggest_categorical("feature_phase3_batching_enabled", [True, False]))
        feature_phase1_prune_enabled = bool(trial.suggest_categorical("feature_phase1_prune_enabled", [True, False]))
        feature_phase1_prune_amplitude_witness_required = bool(
            trial.suggest_categorical("feature_phase1_prune_amplitude_witness_required", [True, False])
        )
        feature_phase3_selector_policy = str(
            trial.suggest_categorical("feature_phase3_selector_policy", list(_PHASE3_SELECTOR_POLICY_CHOICES))
        )
        feature_phase3_novelty_ablation_mode = str(
            trial.suggest_categorical("feature_phase3_novelty_ablation_mode", list(_PHASE3_NOVELTY_ABLATION_MODE_CHOICES))
        )
        feature_phase3_window_relaxation_mode = str(
            trial.suggest_categorical("feature_phase3_window_relaxation_mode", list(_PHASE3_WINDOW_RELAXATION_MODE_CHOICES))
        )
        feature_phase3_batch_selection_mode = str(
            trial.suggest_categorical("feature_phase3_batch_selection_mode", list(_PHASE3_BATCH_SELECTION_MODE_CHOICES))
        )
        feature_phase3_batch_prefilter_mode = str(
            trial.suggest_categorical("feature_phase3_batch_prefilter_mode", list(_PHASE3_BATCH_PREFILTER_MODE_CHOICES))
        )
    elif not production_profile:
        feature_phase3_batching_enabled = True
        feature_phase1_prune_enabled = True
        feature_phase1_prune_amplitude_witness_required = True
        feature_phase3_selector_policy = _DEFAULT_PHASE3_SELECTOR_POLICY
        feature_phase3_novelty_ablation_mode = "off"
        feature_phase3_window_relaxation_mode = "reduced"
        feature_phase3_batch_selection_mode = "reduced_plane"
        feature_phase3_batch_prefilter_mode = "off"
    lambda_leak = (
        _PAPER_I_PRODUCTION_LEAKAGE_LAMBDA
        if production_profile
        else float(trial.suggest_float("phase1_lambda_leak", 0.0, 5.0))
    )
    phase2_leakage_cap = (
        _PAPER_I_PRODUCTION_LEAKAGE_CAP
        if production_profile
        else float(trial.suggest_float("phase2_leakage_cap", 1e-9, 1e6, log=True))
    )
    static = replace(
        base.static,
        static_meta_feature_profile=str(meta_profile),
        lambda_compile=burden["compile"],
        lambda_measure=burden["measure"],
        lambda_leak=lambda_leak,
        lambda_2q=hardware_cost["2q"],
        lambda_d=hardware_cost["d"],
        lambda_1q=hardware_cost["1q"],
        lambda_theta=hardware_cost["theta"],
        lambda_shot=hardware_cost["shot"],
        compile_cx_weight=compile_cost["cx"],
        compile_sq_weight=compile_cost["sq"],
        compile_rotation_step_weight=compile_cost["rotation_step"],
        compile_position_shift_weight=compile_cost["position_shift"],
        compile_refit_active_weight=compile_cost["refit_active"],
        measure_groups_weight=measure_cost["groups"],
        measure_shots_weight=measure_cost["shots"],
        measure_reuse_weight=measure_cost["reuse"],
        phase2_w_depth=cost["depth"],
        phase2_w_group=cost["group"],
        phase2_w_shot=cost["shot"],
        phase2_w_optdim=cost["optdim"],
        phase2_w_reuse=cost["reuse"],
        phase2_leakage_cap=phase2_leakage_cap,
        phase2_frontier_ratio=float(trial.suggest_float("phase2_frontier_ratio", 0.5, 1.0)),
        phase3_frontier_ratio=float(trial.suggest_float("phase3_frontier_ratio", 0.5, 1.0)),
        phase3_tie_beam_score_ratio=float(trial.suggest_float("phase3_tie_beam_score_ratio", 1.0, 1.10)),
        phase3_tie_beam_abs_tol=float(trial.suggest_float("phase3_tie_beam_abs_tol", 1e-8, 1e-3, log=True)),
        phase3_tie_beam_max_branches=int(trial.suggest_int("phase3_tie_beam_max_branches", 1, 5)),
        adapt_beam_live_branches=int(trial.suggest_int("adapt_beam_live_branches", 1, 8)),
        adapt_beam_children_per_parent=int(trial.suggest_int("adapt_beam_children_per_parent", 1, 6)),
        adapt_beam_terminated_keep=int(trial.suggest_int("adapt_beam_terminated_keep", 1, 6)),
        adapt_reopt_policy=str(
            trial.suggest_categorical(
                "adapt_reopt_policy",
                ["full", "windowed"] if production_profile else ["append_only", "full", "windowed"],
            )
        ),
        adapt_window_size=int(trial.suggest_int("adapt_window_size", 8, 192, log=True)),
        adapt_window_topk=int(trial.suggest_int("adapt_window_topk", 0, 96)),
        adapt_insertion_mode=str(
            trial.suggest_categorical(
                "adapt_insertion_mode",
                ["adaptive", "always"] if production_profile else ["adaptive", "always", "append_only"],
            )
        ),
        adapt_allow_repeats=(
            bool(base.static.adapt_allow_repeats_override)
            if base.static.adapt_allow_repeats_override is not None
            else bool(trial.suggest_categorical("adapt_allow_repeats", [False, True]))
        ),
        phase1_probe_max_positions=int(trial.suggest_int("phase1_probe_max_positions", 2, 12)),
        phase2_shortlist_fraction=float(trial.suggest_float("phase2_shortlist_fraction", 0.05, 1.0)),
        phase0_pilot_enabled=bool(feature_phase0_pilot_enabled),
        phase0_pilot_alpha=float(trial.suggest_float("phase0_pilot_alpha", 0.02, 0.5, log=True)),
        phase0_pilot_threshold=float(trial.suggest_categorical("phase0_pilot_threshold", list(PHASE0_PILOT_THRESHOLD_CHOICES))),
        phase0_pilot_max_records=int(trial.suggest_categorical("phase0_pilot_max_records", list(PHASE0_PILOT_MAX_RECORDS_CHOICES))),
        phase0_lane_quota_pressure=float(trial.suggest_float("phase0_lane_quota_pressure", 0.0, 1.0)),
        phase0_algebraic_lane_mode=str(feature_phase0_algebraic_lane_mode),
        algebraic_phase2_lane_rel_threshold=float(
            trial.suggest_float(
                "algebraic_phase2_lane_rel_threshold",
                _ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD_RANGE[0],
                _ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD_RANGE[1],
            )
        ),
        algebraic_phase1_lane_quota_pressure=float(
            trial.suggest_float(
                "algebraic_phase1_lane_quota_pressure",
                _ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE[0],
                _ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE[1],
            )
        ),
        algebraic_phase2_lane_quota_pressure=float(
            trial.suggest_float(
                "algebraic_phase2_lane_quota_pressure",
                _ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE[0],
                _ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE[1],
            )
        ),
        phase2_enable_batching=bool(feature_phase3_batching_enabled),
        phase2_gamma_N=float(phase2_gamma_N),
        phase2_gamma_N_schedule_mode=str(phase2_gamma_N_schedule_mode),
        phase2_gamma_N_schedule_start=phase2_gamma_N_schedule_start,
        phase2_gamma_N_schedule_end=phase2_gamma_N_schedule_end,
        phase2_motif_bonus_weight=phase2_motif_bonus_weight,
        phase3_selector_policy=str(feature_phase3_selector_policy),
        phase3_novelty_ablation_mode=str(feature_phase3_novelty_ablation_mode),
        phase3_window_relaxation_mode=str(feature_phase3_window_relaxation_mode),
        phase3_batch_selection_mode=str(feature_phase3_batch_selection_mode),
        phase3_batch_prefilter_mode=str(feature_phase3_batch_prefilter_mode),
        phase2_batch_target_size=int(phase2_batch_target_size),
        phase2_batch_size_cap=int(phase2_batch_size_cap),
        phase2_batch_near_degenerate_ratio=float(trial.suggest_float("phase2_batch_near_degenerate_ratio", 0.90, 1.0)),
        phase2_batch_rank_rel_tol=float(trial.suggest_float("phase2_batch_rank_rel_tol", 1e-9, 1e-3, log=True)),
        phase2_batch_additivity_tol=float(trial.suggest_float("phase2_batch_additivity_tol", 1e-3, 1.0, log=True)),
        phase1_prune_enabled=bool(feature_phase1_prune_enabled),
        phase1_prune_policy=_DEFAULT_PHASE1_PRUNE_POLICY,
        phase1_prune_mode="both",
        phase1_prune_fraction=float(trial.suggest_float("phase1_prune_fraction", 0.05, 0.50)),
        phase1_prune_min_candidates=int(trial.suggest_int("phase1_prune_min_candidates", 1, 3)),
        phase1_prune_max_candidates=int(trial.suggest_int("phase1_prune_max_candidates", 2, 10)),
        phase1_prune_max_regression=float(trial.suggest_float("phase1_prune_max_regression", 1e-10, 1e-6, log=True)),
        phase1_prune_tolerance_mode=str(
            trial.suggest_categorical("phase1_prune_tolerance_mode", ["auto", "fixed", "adaptive_v1"])
        ),
        phase1_prune_tolerance_shot_coeff=float(trial.suggest_float("phase1_prune_tolerance_shot_coeff", 0.0, 2.0)),
        phase1_prune_tolerance_screen_coeff=float(
            trial.suggest_float("phase1_prune_tolerance_screen_coeff", 1e-4, 5e-2, log=True)
        ),
        phase1_prune_tolerance_chem=float(trial.suggest_float("phase1_prune_tolerance_chem", 0.0, 1e-6)),
        phase1_prune_tolerance_rel_coeff=float(trial.suggest_float("phase1_prune_tolerance_rel_coeff", 0.0, 0.10)),
        phase1_prune_retained_gain_ratio=float(trial.suggest_float("phase1_prune_retained_gain_ratio", 0.10, 0.60)),
        phase1_prune_protect_steps=int(trial.suggest_int("phase1_prune_protect_steps", 1, 4)),
        phase1_prune_stale_age=int(trial.suggest_int("phase1_prune_stale_age", 1, 6)),
        phase1_prune_stagnation_threshold=float(
            trial.suggest_float("phase1_prune_stagnation_threshold", 0.0, 1e-5)
        ),
        phase1_prune_small_theta_abs=float(trial.suggest_float("phase1_prune_small_theta_abs", 1e-5, 1e-1, log=True)),
        phase1_prune_small_theta_relative=float(trial.suggest_float("phase1_prune_small_theta_relative", 0.05, 1.0)),
        phase1_prune_cooldown_steps=int(trial.suggest_int("phase1_prune_cooldown_steps", 0, 8)),
        phase1_prune_local_window_size=int(trial.suggest_int("phase1_prune_local_window_size", 1, 16)),
        phase1_prune_old_fraction=float(trial.suggest_float("phase1_prune_old_fraction", 0.0, 0.75)),
        phase1_prune_checkpoint_period=int(trial.suggest_int("phase1_prune_checkpoint_period", 2, 6)),
        phase1_prune_live_min_depth=int(trial.suggest_int("phase1_prune_live_min_depth", 0, 0)),
        phase1_prune_maturity_threshold=float(trial.suggest_float("phase1_prune_maturity_threshold", 0.35, 0.80)),
        phase1_prune_snr_threshold=float(trial.suggest_float("phase1_prune_snr_threshold", 0.0, 3.0)),
        phase1_prune_amplitude_witness_required=bool(feature_phase1_prune_amplitude_witness_required),
        phase1_prune_collapse_peak_abs_min=float(
            trial.suggest_float("phase1_prune_collapse_peak_abs_min", 1e-5, 1e-1, log=True)
        ),
        phase1_prune_collapse_current_abs_max=float(
            trial.suggest_float("phase1_prune_collapse_current_abs_max", 1e-6, 1e-2, log=True)
        ),
        phase1_prune_collapse_ratio=float(trial.suggest_float("phase1_prune_collapse_ratio", 0.05, 0.95)),
        phase1_prune_collapse_min_abs_drop=float(
            trial.suggest_float("phase1_prune_collapse_min_abs_drop", 1e-6, 1e-1, log=True)
        ),
        phase1_prune_collapse_min_observations=int(
            trial.suggest_int("phase1_prune_collapse_min_observations", 2, 6)
        ),
        phase1_maturity_cap_min_fraction=float(trial.suggest_float("phase1_maturity_cap_min_fraction", 0.05, 1.0)),
        phase1_maturity_cap_max_fraction=float(trial.suggest_float("phase1_maturity_cap_max_fraction", 0.10, 1.0)),
        phase2_maturity_cap_min_fraction=float(trial.suggest_float("phase2_maturity_cap_min_fraction", 0.05, 1.0)),
        phase2_maturity_cap_max_fraction=float(trial.suggest_float("phase2_maturity_cap_max_fraction", 0.10, 1.0)),
        phase3_maturity_cap_min_fraction=float(trial.suggest_float("phase3_maturity_cap_min_fraction", 0.0, 1.0)),
        phase3_maturity_cap_max_fraction=float(trial.suggest_float("phase3_maturity_cap_max_fraction", 0.05, 1.0)),
        phase_maturity_shot_min=int(trial.suggest_int("phase_maturity_shot_min", 1, 2)),
        phase_maturity_shot_max=int(trial.suggest_int("phase_maturity_shot_max", 1, 8)),
        phase1_maturity_shot_cap=int(trial.suggest_int("phase1_maturity_shot_cap", 0, 8)),
        phase2_maturity_shot_cap=int(trial.suggest_int("phase2_maturity_shot_cap", 0, 8)),
        phase3_maturity_shot_cap=int(trial.suggest_int("phase3_maturity_shot_cap", 0, 8)),
        phase_live_hysteresis_enabled=(
            False
            if production_profile
            else bool(trial.suggest_categorical("phase_live_hysteresis_enabled", [True, False]))
        ),
        phase2_null_nrem_high_threshold=float(trial.suggest_float("phase2_null_nrem_high_threshold", 0.0, 0.75)),
        phase2_live_nrem_low_threshold=float(trial.suggest_float("phase2_live_nrem_low_threshold", 0.0, 1.25)),
        phase3_null_nrem_high_threshold=float(trial.suggest_float("phase3_null_nrem_high_threshold", 0.0, 1.25)),
        phase3_live_nrem_low_threshold=float(trial.suggest_float("phase3_live_nrem_low_threshold", 0.0, 1.75)),
        phase2_hysteresis_steps=int(trial.suggest_int("phase2_hysteresis_steps", 1, 4)),
        phase3_hysteresis_steps=int(trial.suggest_int("phase3_hysteresis_steps", 1, 4)),
        adapt_max_depth=int(trial.suggest_int("adapt_max_depth", 12, 128)),
        adapt_maxiter=int(trial.suggest_int("adapt_maxiter", 800, 6000, log=True)),
        adapt_drop_floor=float(trial.suggest_float("adapt_drop_floor", 1e-10, 1e-3, log=True)),
        adapt_drop_patience=int(trial.suggest_int("adapt_drop_patience", 2, 16)),
        adapt_drop_min_depth=int(trial.suggest_int("adapt_drop_min_depth", 4, 32)),
    )
    spsa_kwargs: dict[str, float] = {}
    if _is_spsa_optimizer(_ACTIVE_INNER_OPTIMIZER):
        spsa_kwargs = {
            "spsa_a": float(trial.suggest_float("spsa_a", 1e-3, 1.0, log=True)),
            "spsa_c": float(trial.suggest_float("spsa_c", 1e-3, 1.0, log=True)),
            "spsa_A": float(trial.suggest_float("spsa_A", 0.0, 100.0)),
            "spsa_alpha": float(trial.suggest_float("spsa_alpha", 0.5, 0.9)),
            "spsa_gamma": float(trial.suggest_float("spsa_gamma", 0.05, 0.2)),
        }
    inner = replace(
        base.inner_optimizer,
        inner_optimizer=_ACTIVE_INNER_OPTIMIZER,
        final_optimizer_type=_ACTIVE_INNER_OPTIMIZER,
        **spsa_kwargs,
        refit_maxiter=static.adapt_maxiter,
        final_maxiter=static.adapt_maxiter,
        grad_tol=static.adapt_eps_grad,
        energy_tol=static.adapt_eps_energy,
    )
    policy = _apply_policy_search_profile(
        AlgorithmPolicy(pool=pool, static=static, inner_optimizer=inner),
        policy_search_profile,
    )
    policy = _apply_trial_param_overrides_to_policy(policy, trial_param_overrides)
    if normalize_static_route_id(policy.static.static_route_id, default=ROUTE_ID_A) == ROUTE_ID_A:
        policy = replace(policy, pool=replace(policy.pool, pool_key="full_meta"))
    return _normalize_active_policy(policy, meta_feature_profile=meta_profile)


def _apply_trial_param_search_profile(params: Mapping[str, Any], profile: str | None) -> dict[str, Any]:
    out = dict(params)
    profile_key = _normalize_policy_search_profile(profile)
    if profile_key == "hh_novelty_surface_v1":
        out.update(
            {
                "pool_key": "full_meta",
                "family_repeat_penalty": 0.0,
                "phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
                "feature_phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
                "phase3_novelty_ablation_mode": "off",
                "feature_phase3_novelty_ablation_mode": "off",
                "phase3_window_relaxation_mode": "reduced",
                "feature_phase3_window_relaxation_mode": "reduced",
                "phase3_batch_selection_mode": "reduced_plane",
                "feature_phase3_batch_selection_mode": "reduced_plane",
                "phase3_batch_prefilter_mode": "off",
                "feature_phase3_batch_prefilter_mode": "off",
            }
        )
        out.setdefault("novelty_bonus", 0.05)
        out.setdefault("phase2_gamma_N", 1.0)
        out.setdefault("phase2_gamma_N_schedule_mode", "fixed")
        out.setdefault("phase2_gamma_N_schedule_start", None)
        out.setdefault("phase2_gamma_N_schedule_end", None)
        return out
    if profile_key in _SNAKE_U8_NOVELTY_PROFILES:
        out.update(
            {
                "pool_key": "full_meta",
                "family_repeat_penalty": 0.0,
                "novelty_bonus": 0.0,
                "phase2_motif_bonus_weight": 0.0,
                "compile_position_shift_weight": 0.0,
                "phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
                "feature_phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
                "phase3_selector_geometry_mode": "reduced",
                "phase3_window_relaxation_mode": "reduced",
                "feature_phase3_window_relaxation_mode": "reduced",
                "phase3_batch_selection_mode": "reduced_plane",
                "feature_phase3_batch_selection_mode": "reduced_plane",
                "phase3_batch_prefilter_mode": "off",
                "feature_phase3_batch_prefilter_mode": "off",
                "phase2_enable_batching": True,
                "feature_phase3_batching_enabled": True,
                "phase1_prune_enabled": True,
                "feature_phase1_prune_enabled": True,
                "phase1_prune_amplitude_witness_required": False,
                "feature_phase1_prune_amplitude_witness_required": False,
                "phase_live_hysteresis_enabled": False,
            }
        )
        if profile_key == "snake_u8_no_novelty_v1":
            out.update(
                {
                    "phase2_gamma_N": 0.0,
                    "phase2_gamma_N_schedule_mode": "fixed",
                    "phase2_gamma_N_schedule_start": None,
                    "phase2_gamma_N_schedule_end": None,
                    "phase3_novelty_ablation_mode": "all",
                    "feature_phase3_novelty_ablation_mode": "all",
                }
            )
        elif profile_key == "snake_u8_flat_novelty_v1":
            out.setdefault("phase2_gamma_N", 1.0)
            out.update(
                {
                    "phase2_gamma_N_schedule_mode": "fixed",
                    "phase2_gamma_N_schedule_start": None,
                    "phase2_gamma_N_schedule_end": None,
                    "phase3_novelty_ablation_mode": "off",
                    "feature_phase3_novelty_ablation_mode": "off",
                }
            )
        elif profile_key == "snake_u8_exponent_novelty_v1":
            out.update(
                {
                    "phase2_gamma_N": 1.0,
                    "phase2_gamma_N_schedule_mode": "depth_linear_v1",
                    "phase2_gamma_N_schedule_start": out.get("phase2_gamma_N_schedule_start", 1.0),
                    "phase2_gamma_N_schedule_end": out.get("phase2_gamma_N_schedule_end", 0.25),
                    "phase3_novelty_ablation_mode": "off",
                    "feature_phase3_novelty_ablation_mode": "off",
                }
            )
        return out
    if profile_key == "bosonic_fullmeta_compact":
        production_profile = _is_paper_i_production_profile(out.get("meta_feature_profile"))
        out.update(
            {
                "pool_key": "full_meta",
                "family_repeat_penalty": max(float(out.get("family_repeat_penalty", 1.0)), 2.0),
                "novelty_bonus": max(float(out.get("novelty_bonus", 0.05)), 0.05),
                "phase1_min_count": max(64, int(out.get("phase1_min_count", 64))),
                "phase1_max_count": max(192, int(out.get("phase1_max_count", 192))),
                "phase1_pool_fraction": max(0.75, float(out.get("phase1_pool_fraction", 0.75))),
                "phase2_min_count": max(32, int(out.get("phase2_min_count", 32))),
                "phase2_max_count": max(96, int(out.get("phase2_max_count", 96))),
                "phase2_pool_fraction": max(0.50, float(out.get("phase2_pool_fraction", 0.50))),
                "adapt_beam_live_branches": min(max(2, int(out.get("adapt_beam_live_branches", 3))), 4),
                "adapt_beam_children_per_parent": min(
                    max(2, int(out.get("adapt_beam_children_per_parent", 2))),
                    3,
                ),
                "adapt_beam_terminated_keep": min(max(2, int(out.get("adapt_beam_terminated_keep", 3))), 4),
                "adapt_reopt_policy": "windowed"
                if str(out.get("adapt_reopt_policy", "windowed")).strip().lower() == "append_only"
                else out.get("adapt_reopt_policy", "windowed"),
                "adapt_insertion_mode": "adaptive"
                if str(out.get("adapt_insertion_mode", "adaptive")).strip().lower() == "append_only"
                else out.get("adapt_insertion_mode", "adaptive"),
                "adapt_allow_repeats": _coerce_bool(out.get("adapt_allow_repeats", False))
                if production_profile
                else False,
                "adapt_window_size": max(64, int(out.get("adapt_window_size", 64))),
                "adapt_window_topk": max(24, int(out.get("adapt_window_topk", 24))),
                "phase1_probe_max_positions": max(6, int(out.get("phase1_probe_max_positions", 6))),
                "phase2_shortlist_fraction": max(0.35, float(out.get("phase2_shortlist_fraction", 0.35))),
                "phase2_enable_batching": _coerce_bool(out.get("phase2_enable_batching", True))
                if production_profile
                else True,
                "phase2_batch_target_size": min(max(3, int(out.get("phase2_batch_target_size", 4))), 6),
                "phase2_batch_size_cap": min(max(6, int(out.get("phase2_batch_size_cap", 8))), 12),
                "adapt_max_depth": max(64, int(out.get("adapt_max_depth", 64))),
                "adapt_maxiter": max(2400, int(out.get("adapt_maxiter", 2400))),
                "adapt_drop_floor": min(float(out.get("adapt_drop_floor", 1e-7)), 1e-7),
                "adapt_drop_patience": max(12, int(out.get("adapt_drop_patience", 12))),
                "adapt_drop_min_depth": max(24, int(out.get("adapt_drop_min_depth", 24))),
            }
        )
        return out
    if profile_key in {"spin_boson_2q_batching_v1", "route_a_batching_v1"}:
        out.update(
            {
                "feature_phase3_batching_enabled": True,
                "phase2_enable_batching": True,
                "phase1_prune_enabled": True,
                "feature_phase1_prune_enabled": True,
                "phase1_prune_amplitude_witness_required": True,
                "feature_phase1_prune_amplitude_witness_required": True,
                "phase2_batch_target_size": max(3, int(out.get("phase2_batch_target_size", 3))),
                "phase2_batch_size_cap": max(8, int(out.get("phase2_batch_size_cap", 8))),
            }
        )
        return out
    if profile_key != "fermionic_protected_correlation":
        return out
    out.update(
        {
            "pool_key": "full_meta",
            "family_repeat_penalty": min(float(out.get("family_repeat_penalty", 1.0)), 1.0),
            "novelty_bonus": min(float(out.get("novelty_bonus", 0.05)), 0.05),
            "phase1_min_count": max(64, int(out.get("phase1_min_count", 64))),
            "phase1_max_count": max(256, int(out.get("phase1_max_count", 256))),
            "phase1_pool_fraction": max(0.75, float(out.get("phase1_pool_fraction", 0.75))),
            "phase1_qubit_slope": max(10.0, float(out.get("phase1_qubit_slope", 10.0))),
            "phase2_min_count": max(32, int(out.get("phase2_min_count", 32))),
            "phase2_max_count": max(128, int(out.get("phase2_max_count", 128))),
            "phase2_pool_fraction": max(0.50, float(out.get("phase2_pool_fraction", 0.50))),
            "phase2_qubit_slope": max(8.0, float(out.get("phase2_qubit_slope", 8.0))),
            "adapt_reopt_policy": "windowed"
            if str(out.get("adapt_reopt_policy", "windowed")).strip().lower() == "append_only"
            else out.get("adapt_reopt_policy", "windowed"),
            "adapt_insertion_mode": "adaptive"
            if str(out.get("adapt_insertion_mode", "adaptive")).strip().lower() == "append_only"
            else out.get("adapt_insertion_mode", "adaptive"),
            "adapt_window_size": max(96, int(out.get("adapt_window_size", 96))),
            "adapt_window_topk": max(32, int(out.get("adapt_window_topk", 32))),
            "phase1_probe_max_positions": max(8, int(out.get("phase1_probe_max_positions", 8))),
            "phase2_shortlist_fraction": max(0.50, float(out.get("phase2_shortlist_fraction", 0.50))),
            "phase1_prune_protect_steps": max(12, int(out.get("phase1_prune_protect_steps", 12))),
            "phase1_prune_collapse_min_observations": max(
                4,
                int(out.get("phase1_prune_collapse_min_observations", 4)),
            ),
            "adapt_max_depth": max(96, int(out.get("adapt_max_depth", 96))),
            "adapt_maxiter": max(3200, int(out.get("adapt_maxiter", 3200))),
            "adapt_drop_floor": min(float(out.get("adapt_drop_floor", 1e-7)), 1e-7),
            "adapt_drop_patience": max(16, int(out.get("adapt_drop_patience", 16))),
            "adapt_drop_min_depth": max(32, int(out.get("adapt_drop_min_depth", 32))),
        }
    )
    return out


def _force_canonical_static_trial_route(
    params: Mapping[str, Any],
    *,
    meta_feature_profile: str | None = None,
    policy_search_profile: str | None = "default",
) -> dict[str, Any]:
    """Force the current Optuna route identity after defaults/profile/import transforms."""

    out = dict(params)
    policy_profile = _normalize_policy_search_profile(policy_search_profile)
    novelty_surface_profile = policy_profile == "hh_novelty_surface_v1"
    no_novelty_profile = policy_profile == "snake_u8_no_novelty_v1"
    profile = _normalize_meta_feature_profile(meta_feature_profile or out.get("meta_feature_profile"))
    if profile == _META_FEATURE_PROFILE_PAPER_I_PRODUCTION:
        _ensure_paper_i_production_phase2_novelty_mode(profile)
    out = _sync_feature_params(out, meta_feature_profile=profile)
    forced = {
        "pool_key": "full_meta",
        "phase2_novelty_mode": _ACTIVE_PHASE2_NOVELTY_MODE,
        "phase1_prune_policy": _DEFAULT_PHASE1_PRUNE_POLICY,
        "phase1_prune_mode": "both",
    }
    if profile != _META_FEATURE_PROFILE_SAFE_CORE:
        forced.update(
            {
                "phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
                "feature_phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
                "phase3_selector_geometry_mode": "reduced",
                "phase3_novelty_ablation_mode": "off",
                "feature_phase3_novelty_ablation_mode": "off",
                "phase3_window_relaxation_mode": "reduced",
                "feature_phase3_window_relaxation_mode": "reduced",
                "phase3_batch_selection_mode": "reduced_plane",
                "feature_phase3_batch_selection_mode": "reduced_plane",
                "phase3_batch_prefilter_mode": "off",
                "feature_phase3_batch_prefilter_mode": "off",
            }
        )
    out.update(forced)
    if profile == _META_FEATURE_PROFILE_OFF:
        out.update(
            {
                "phase2_enable_batching": True,
                "feature_phase3_batching_enabled": True,
                "phase1_prune_enabled": True,
                "feature_phase1_prune_enabled": True,
                "phase1_prune_amplitude_witness_required": True,
                "feature_phase1_prune_amplitude_witness_required": True,
            }
        )
    if profile == _META_FEATURE_PROFILE_PAPER_I_PRODUCTION:
        out.update(
            {
                "pool_key": "full_meta",
                "family_repeat_penalty": 0.0,
                "phase0_pilot_enabled": True,
                "feature_phase0_pilot_enabled": True,
                "phase0_algebraic_lane_mode": "weak",
                "feature_phase0_algebraic_lane_mode": "weak",
                "phase1_prune_enabled": True,
                "feature_phase1_prune_enabled": True,
                "phase1_prune_amplitude_witness_required": False,
                "feature_phase1_prune_amplitude_witness_required": False,
                "phase_live_hysteresis_enabled": False,
                "phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
                "feature_phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
                "phase3_novelty_ablation_mode": "off",
                "feature_phase3_novelty_ablation_mode": "off",
                "phase3_window_relaxation_mode": "reduced",
                "feature_phase3_window_relaxation_mode": "reduced",
                "phase3_batch_selection_mode": "reduced_plane",
                "feature_phase3_batch_selection_mode": "reduced_plane",
                "phase3_batch_prefilter_mode": "off",
                "feature_phase3_batch_prefilter_mode": "off",
            }
        )
        if not novelty_surface_profile:
            out["novelty_bonus"] = 0.0
        if policy_profile in _SNAKE_U8_NOVELTY_PROFILES:
            out["phase2_motif_bonus_weight"] = 0.0
            out["compile_position_shift_weight"] = 0.0
            out["feature_phase1_prune_amplitude_witness_required"] = False
            out["phase1_prune_amplitude_witness_required"] = False
            out["feature_phase3_batching_enabled"] = True
            out["phase2_enable_batching"] = True
            out["phase3_batch_prefilter_mode"] = "off"
            out["feature_phase3_batch_prefilter_mode"] = "off"
        if no_novelty_profile:
            out["phase2_gamma_N"] = 0.0
            out["phase2_gamma_N_schedule_mode"] = "fixed"
            out["phase2_gamma_N_schedule_start"] = None
            out["phase2_gamma_N_schedule_end"] = None
            out["phase3_novelty_ablation_mode"] = "all"
            out["feature_phase3_novelty_ablation_mode"] = "all"
        if str(out.get("adapt_reopt_policy", "windowed")).strip().lower() == "append_only":
            out["adapt_reopt_policy"] = "windowed"
        if str(out.get("adapt_insertion_mode", "adaptive")).strip().lower() == "append_only":
            out["adapt_insertion_mode"] = "adaptive"
    return out


def default_trial_params(
    policy_search_profile: str | None = "default",
    meta_feature_profile: str | None = _DEFAULT_META_FEATURE_PROFILE,
) -> dict[str, Any]:
    """Neutral warm-start point for Optuna ``enqueue_trial``.

    The values intentionally encode a broad, high-budget phase3 policy rather
    than a Hamiltonian-specific winner. Per-Hamiltonian oracle studies may add
    more seed trials externally.
    """

    profile = _normalize_meta_feature_profile(meta_feature_profile)
    params = {
        "meta_feature_profile": profile,
        "cost_logit_depth": 0.0,
        "cost_logit_group": 0.0,
        "cost_logit_shot": 0.0,
        "cost_logit_optdim": 0.0,
        "cost_logit_reuse": 0.0,
        "cost_scale": 0.7,
        "hardware_cost_logit_2q": 0.0,
        "hardware_cost_logit_d": 0.0,
        "hardware_cost_logit_1q": 0.0,
        "hardware_cost_logit_theta": 0.0,
        "hardware_cost_logit_shot": 0.0,
        "hardware_cost_scale": 0.7,
        "burden_logit_compile": 0.5,
        "burden_logit_measure": -0.5,
        "burden_scale": 0.07,
        "compile_cost_logit_cx": 0.5,
        "compile_cost_logit_sq": 0.0,
        "compile_cost_logit_rotation_step": 0.0,
        "compile_cost_logit_position_shift": 0.0,
        "compile_cost_logit_refit_active": 0.0,
        "compile_cost_scale": 4.5,
        "measure_cost_logit_groups": 0.0,
        "measure_cost_logit_shots": 0.0,
        "measure_cost_logit_reuse": 0.0,
        "measure_cost_scale": 3.0,
        "phase1_lambda_leak": 0.0,
        "phase1_min_count": 32,
        "phase1_max_count": 256,
        "phase1_pool_fraction": 0.35,
        "phase1_qubit_slope": 8.0,
        "phase2_min_count": 16,
        "phase2_max_count": 128,
        "phase2_pool_fraction": 0.25,
        "phase2_qubit_slope": 6.0,
        "pool_key": "full_meta",
        "family_repeat_penalty": 1.0,
        "novelty_bonus": 0.05,
        "phase2_frontier_ratio": 1.0,
        "phase2_leakage_cap": 1e6,
        "phase3_frontier_ratio": 1.0,
        "phase3_tie_beam_score_ratio": 1.05,
        "phase3_tie_beam_abs_tol": 1e-6,
        "phase3_tie_beam_max_branches": 3,
        "adapt_beam_live_branches": 5,
        "adapt_beam_children_per_parent": 4,
        "adapt_beam_terminated_keep": 3,
        "adapt_beam_lambda": 0.0,
        "adapt_reopt_policy": "windowed",
        "adapt_window_size": 128,
        "adapt_window_topk": 64,
        "adapt_insertion_mode": "adaptive",
        "adapt_allow_repeats": False,
        "phase1_probe_max_positions": 6,
        "phase2_shortlist_fraction": 0.25,
        "feature_phase0_pilot_enabled": True,
        "feature_phase0_algebraic_lane_mode": "weak",
        "feature_phase3_batching_enabled": True,
        "feature_phase1_prune_enabled": True,
        "feature_phase1_prune_amplitude_witness_required": True,
        "feature_phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
        "feature_phase3_novelty_ablation_mode": "off",
        "feature_phase3_window_relaxation_mode": "reduced",
        "feature_phase3_batch_selection_mode": "reduced_plane",
        "feature_phase3_batch_prefilter_mode": "off",
        **dict(PHASE0_OPTUNA_DEFAULTS),
        "algebraic_phase2_lane_rel_threshold": _DEFAULT_ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD,
        "algebraic_phase1_lane_quota_pressure": _DEFAULT_ALGEBRAIC_PHASE1_LANE_QUOTA_PRESSURE,
        "algebraic_phase2_lane_quota_pressure": _DEFAULT_ALGEBRAIC_PHASE2_LANE_QUOTA_PRESSURE,
        "phase2_novelty_mode": _ACTIVE_PHASE2_NOVELTY_MODE,
        "phase2_enable_batching": True,
        "phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
        "phase3_selector_geometry_mode": "reduced",
        "phase3_novelty_ablation_mode": "off",
        "phase3_window_relaxation_mode": "reduced",
        "phase3_batch_selection_mode": "reduced_plane",
        "phase3_batch_prefilter_mode": "off",
        "phase2_batch_target_size": 8,
        "phase2_batch_size_cap": 16,
        "phase2_batch_near_degenerate_ratio": 0.98,
        "phase2_batch_rank_rel_tol": 1e-6,
        "phase2_batch_additivity_tol": 0.25,
        "phase1_prune_enabled": True,
        "phase1_prune_policy": _DEFAULT_PHASE1_PRUNE_POLICY,
        "phase1_prune_mode": "both",
        "phase1_prune_fraction": 0.25,
        "phase1_prune_min_candidates": 1,
        "phase1_prune_max_candidates": 6,
        "phase1_prune_max_regression": 1e-8,
        "phase1_prune_tolerance_mode": "auto",
        "phase1_prune_tolerance_shot_coeff": 0.0,
        "phase1_prune_tolerance_screen_coeff": 0.01,
        "phase1_prune_tolerance_chem": 0.0,
        "phase1_prune_tolerance_rel_coeff": 0.0,
        "phase1_prune_retained_gain_ratio": 0.25,
        "phase1_prune_protect_steps": 1,
        "phase1_prune_stale_age": 2,
        "phase1_prune_stagnation_threshold": 0.0,
        "phase1_prune_small_theta_abs": 1e-3,
        "phase1_prune_small_theta_relative": 0.5,
        "phase1_prune_cooldown_steps": 2,
        "phase1_prune_local_window_size": 4,
        "phase1_prune_recovery_trust_radius": 0.0,
        "phase1_prune_old_fraction": 0.25,
        "phase1_prune_checkpoint_period": 3,
        "phase1_prune_live_min_depth": 0,
        "phase1_prune_maturity_threshold": 0.5,
        "phase1_prune_snr_threshold": 1.0,
        "phase1_prune_amplitude_witness_required": True,
        "phase1_prune_collapse_peak_abs_min": 1e-3,
        "phase1_prune_collapse_current_abs_max": 1e-3,
        "phase1_prune_collapse_ratio": 0.25,
        "phase1_prune_collapse_min_abs_drop": 1e-3,
        "phase1_prune_collapse_min_observations": 3,
        "phase1_maturity_cap_min_fraction": 0.35,
        "phase1_maturity_cap_max_fraction": 1.0,
        "phase2_maturity_cap_min_fraction": 0.35,
        "phase2_maturity_cap_max_fraction": 1.0,
        "phase3_maturity_cap_min_fraction": 0.25,
        "phase3_maturity_cap_max_fraction": 0.75,
        "phase_maturity_shot_min": 1,
        "phase_maturity_shot_max": 1,
        "phase1_maturity_shot_cap": 0,
        "phase2_maturity_shot_cap": 0,
        "phase3_maturity_shot_cap": 0,
        "phase_live_hysteresis_enabled": True,
        "phase2_null_nrem_high_threshold": 0.0,
        "phase2_live_nrem_low_threshold": 0.25,
        "phase3_null_nrem_high_threshold": 0.5,
        "phase3_live_nrem_low_threshold": 1.0,
        "phase2_hysteresis_steps": 2,
        "phase3_hysteresis_steps": 2,
        "adapt_max_depth": 96,
        "adapt_maxiter": 4000,
        "adapt_drop_floor": 1e-8,
        "adapt_drop_patience": 12,
        "adapt_drop_min_depth": 16,
        "inner_optimizer": _ACTIVE_INNER_OPTIMIZER,
    }
    if _is_spsa_optimizer(_ACTIVE_INNER_OPTIMIZER):
        params.update(
            {
                "spsa_a": 0.1,
                "spsa_c": 0.1,
                "spsa_A": 10.0,
                "spsa_alpha": 0.602,
                "spsa_gamma": 0.101,
            }
        )
    return _force_canonical_static_trial_route(
        _apply_trial_param_search_profile(params, policy_search_profile),
        meta_feature_profile=profile,
        policy_search_profile=policy_search_profile,
    )


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


def _option_float(options: Mapping[str, Any], key: str) -> float | None:
    try:
        if key not in options:
            return None
        return float(options[key])
    except Exception:
        return None


def _option_float_first(options: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _option_float(options, key)
        if value is not None:
            return value
    return None


def _option_int(options: Mapping[str, Any], key: str) -> int | None:
    try:
        if key not in options:
            return None
        return int(float(options[key]))
    except Exception:
        return None


def _option_str(options: Mapping[str, Any], key: str) -> str | None:
    value = options.get(key)
    return None if value in {None, True, False, ""} else str(value)


def _set_if_present(params: dict[str, Any], options: Mapping[str, Any], option_key: str, param_key: str, caster: Callable[[Mapping[str, Any], str], Any]) -> None:
    value = caster(options, option_key)
    if value is not None:
        params[param_key] = value


def _encode_weight_family(
    params: dict[str, Any],
    *,
    prefix: str,
    names: Sequence[str],
    values: Sequence[float | None],
    scale_low: float,
    scale_high: float,
) -> None:
    if not any(value is not None for value in values):
        return
    raw = [float(value if value is not None else 0.0) for value in values]
    if not any(abs(value) > 1e-11 for value in raw):
        params[f"{prefix}_scale"] = float(scale_low)
        for name in names:
            params[f"{prefix}_logit_{name}"] = 0.0
        return
    clean = [max(1e-12, value) for value in raw]
    scale = _clamp(sum(clean), scale_low, scale_high)
    logs = [math.log(value) for value in clean]
    mean_log = sum(logs) / len(logs)
    params[f"{prefix}_scale"] = scale
    for name, log_value in zip(names, logs):
        params[f"{prefix}_logit_{name}"] = _clamp(log_value - mean_log, -3.0, 3.0)


def trial_params_from_cli_command(
    command: str | None,
    *,
    force_inner_optimizer: str | None = _FORCE_ACTIVE_INNER_OPTIMIZER,
    meta_feature_profile: str | None = None,
) -> dict[str, Any]:
    """Best-effort conversion from archived ADAPT CLI flags to Optuna params.

    This does not reconstruct every historical detail. It maps the current
    problem-generic policy surface knobs that are sampled by
    ``sample_policy_from_trial`` so old good runs can seed the outer optimizer.
    """

    options = _parse_cli_option_map(command)
    profile = _normalize_meta_feature_profile(
        meta_feature_profile
        or _option_str(options, "static_meta_feature_profile")
        or _option_str(options, "meta_feature_profile")
    )
    params = default_trial_params(meta_feature_profile=profile)
    direct_str = {
        "adapt_pool": "pool_key",
        "adapt_reopt_policy": "adapt_reopt_policy",
        "adapt_insertion_mode": "adapt_insertion_mode",
        "phase3_selector_policy": "phase3_selector_policy",
        "phase3_selector_geometry_mode": "phase3_selector_geometry_mode",
        "phase3_novelty_ablation_mode": "phase3_novelty_ablation_mode",
        "phase3_window_relaxation_mode": "phase3_window_relaxation_mode",
        "phase3_batch_selection_mode": "phase3_batch_selection_mode",
        "phase3_batch_prefilter_mode": "phase3_batch_prefilter_mode",
        "phase2_novelty_mode": "phase2_novelty_mode",
        "phase2_gamma_N_schedule_mode": "phase2_gamma_N_schedule_mode",
        "phase1_prune_policy": "phase1_prune_policy",
        "phase1_prune_tolerance_mode": "phase1_prune_tolerance_mode",
        "phase0_algebraic_lane_mode": "phase0_algebraic_lane_mode",
        "static_meta_feature_profile": "meta_feature_profile",
        "static_lane_route": "static_lane_route",
        "adapt_inner_optimizer": "inner_optimizer",
    }
    for option_key, param_key in direct_str.items():
        _set_if_present(params, options, option_key, param_key, _option_str)
    if "phase0_algebraic_lane_mode" in options:
        params["feature_phase0_algebraic_lane_mode"] = params.get("phase0_algebraic_lane_mode")
    for key in (
        "phase3_selector_policy",
        "phase3_novelty_ablation_mode",
        "phase3_window_relaxation_mode",
        "phase3_batch_selection_mode",
        "phase3_batch_prefilter_mode",
    ):
        if key in options:
            params[f"feature_{key}"] = params.get(key)
    direct_int = {
        "phase3_tie_beam_max_branches": "phase3_tie_beam_max_branches",
        "adapt_beam_live_branches": "adapt_beam_live_branches",
        "adapt_beam_children_per_parent": "adapt_beam_children_per_parent",
        "adapt_beam_terminated_keep": "adapt_beam_terminated_keep",
        "adapt_window_size": "adapt_window_size",
        "adapt_window_topk": "adapt_window_topk",
        "phase1_probe_max_positions": "phase1_probe_max_positions",
        "adapt_max_depth": "adapt_max_depth",
        "adapt_maxiter": "adapt_maxiter",
        "adapt_drop_patience": "adapt_drop_patience",
        "adapt_drop_min_depth": "adapt_drop_min_depth",
        "physical_lane_shortlist_aggressiveness": "physical_lane_shortlist_aggressiveness",
        "phase2_batch_target_size": "phase2_batch_target_size",
        "phase2_batch_size_cap": "phase2_batch_size_cap",
        "phase1_prune_min_candidates": "phase1_prune_min_candidates",
        "phase1_prune_max_candidates": "phase1_prune_max_candidates",
        "phase1_prune_protect_steps": "phase1_prune_protect_steps",
        "phase1_prune_stale_age": "phase1_prune_stale_age",
        "phase1_prune_cooldown_steps": "phase1_prune_cooldown_steps",
        "phase1_prune_local_window_size": "phase1_prune_local_window_size",
        "phase1_prune_recovery_trust_radius": "phase1_prune_recovery_trust_radius",
        "phase1_prune_checkpoint_period": "phase1_prune_checkpoint_period",
        "phase1_prune_live_min_depth": "phase1_prune_live_min_depth",
        "phase1_prune_collapse_min_observations": "phase1_prune_collapse_min_observations",
        "phase_maturity_shot_min": "phase_maturity_shot_min",
        "phase_maturity_shot_max": "phase_maturity_shot_max",
        "phase1_maturity_shot_cap": "phase1_maturity_shot_cap",
        "phase2_maturity_shot_cap": "phase2_maturity_shot_cap",
        "phase3_maturity_shot_cap": "phase3_maturity_shot_cap",
        "phase2_hysteresis_steps": "phase2_hysteresis_steps",
        "phase3_hysteresis_steps": "phase3_hysteresis_steps",
        "phase0_pilot_max_records": "phase0_pilot_max_records",
    }
    for option_key, param_key in direct_int.items():
        _set_if_present(params, options, option_key, param_key, _option_int)
    direct_float = {
        "family_repeat_cost_scale": "family_repeat_penalty",
        "phase1_family_repeat_cost_scale": "family_repeat_penalty",
        "phase2_motif_bonus_weight": "novelty_bonus",
        "phase2_gamma_N": "phase2_gamma_N",
        "phase2_gamma_N_schedule_start": "phase2_gamma_N_schedule_start",
        "phase2_gamma_N_schedule_end": "phase2_gamma_N_schedule_end",
        "phase2_frontier_ratio": "phase2_frontier_ratio",
        "phase3_frontier_ratio": "phase3_frontier_ratio",
        "phase3_tie_beam_score_ratio": "phase3_tie_beam_score_ratio",
        "phase3_tie_beam_abs_tol": "phase3_tie_beam_abs_tol",
        "adapt_beam_lambda": "adapt_beam_lambda",
        "phase1_lambda_leak": "phase1_lambda_leak",
        "phase2_shortlist_fraction": "phase2_shortlist_fraction",
        "algebraic_phase2_lane_rel_threshold": "algebraic_phase2_lane_rel_threshold",
        "algebraic_phase1_lane_quota_pressure": "algebraic_phase1_lane_quota_pressure",
        "algebraic_phase2_lane_quota_pressure": "algebraic_phase2_lane_quota_pressure",
        "phase0_pilot_alpha": "phase0_pilot_alpha",
        "phase0_pilot_threshold": "phase0_pilot_threshold",
        "phase0_lane_quota_pressure": "phase0_lane_quota_pressure",
        "phase2_leakage_cap": "phase2_leakage_cap",
        "phase2_batch_near_degenerate_ratio": "phase2_batch_near_degenerate_ratio",
        "phase2_batch_rank_rel_tol": "phase2_batch_rank_rel_tol",
        "phase2_batch_additivity_tol": "phase2_batch_additivity_tol",
        "phase1_prune_fraction": "phase1_prune_fraction",
        "phase1_prune_max_regression": "phase1_prune_max_regression",
        "phase1_prune_tolerance_shot_coeff": "phase1_prune_tolerance_shot_coeff",
        "phase1_prune_tolerance_screen_coeff": "phase1_prune_tolerance_screen_coeff",
        "phase1_prune_tolerance_chem": "phase1_prune_tolerance_chem",
        "phase1_prune_tolerance_rel_coeff": "phase1_prune_tolerance_rel_coeff",
        "phase1_prune_retained_gain_ratio": "phase1_prune_retained_gain_ratio",
        "phase1_prune_stagnation_threshold": "phase1_prune_stagnation_threshold",
        "phase1_prune_small_theta_abs": "phase1_prune_small_theta_abs",
        "phase1_prune_small_theta_relative": "phase1_prune_small_theta_relative",
        "phase1_prune_old_fraction": "phase1_prune_old_fraction",
        "phase1_prune_maturity_threshold": "phase1_prune_maturity_threshold",
        "phase1_prune_snr_threshold": "phase1_prune_snr_threshold",
        "phase1_prune_collapse_peak_abs_min": "phase1_prune_collapse_peak_abs_min",
        "phase1_prune_collapse_current_abs_max": "phase1_prune_collapse_current_abs_max",
        "phase1_prune_collapse_ratio": "phase1_prune_collapse_ratio",
        "phase1_prune_collapse_min_abs_drop": "phase1_prune_collapse_min_abs_drop",
        "phase2_null_nrem_high_threshold": "phase2_null_nrem_high_threshold",
        "phase2_live_nrem_low_threshold": "phase2_live_nrem_low_threshold",
        "phase3_null_nrem_high_threshold": "phase3_null_nrem_high_threshold",
        "phase3_live_nrem_low_threshold": "phase3_live_nrem_low_threshold",
        "adapt_drop_floor": "adapt_drop_floor",
        "adapt_spsa_a": "spsa_a",
        "adapt_spsa_c": "spsa_c",
        "adapt_spsa_A": "spsa_A",
        "adapt_spsa_alpha": "spsa_alpha",
        "adapt_spsa_gamma": "spsa_gamma",
    }
    for option_key, param_key in direct_float.items():
        _set_if_present(params, options, option_key, param_key, _option_float)
    if "adapt_no_repeats" in options:
        params["adapt_allow_repeats"] = False
    if "adapt_allow_repeats" in options:
        params["adapt_allow_repeats"] = True
    if "phase3_enable_batching" in options or "phase2_enable_batching" in options:
        params["phase2_enable_batching"] = True
        params["feature_phase3_batching_enabled"] = True
    if "phase3_no_batching" in options or "phase2_no_batching" in options:
        params["phase2_enable_batching"] = False
        params["feature_phase3_batching_enabled"] = False
    if "phase1_prune_enabled" in options:
        params["phase1_prune_enabled"] = True
        params["feature_phase1_prune_enabled"] = True
    if "phase1_no_prune" in options or "phase1_prune_disabled" in options:
        params["phase1_prune_enabled"] = False
        params["feature_phase1_prune_enabled"] = False
    if "phase1_prune_amplitude_witness_required" in options:
        params["phase1_prune_amplitude_witness_required"] = True
        params["feature_phase1_prune_amplitude_witness_required"] = True
    if "phase1_prune_amplitude_witness_optional" in options:
        params["phase1_prune_amplitude_witness_required"] = False
        params["feature_phase1_prune_amplitude_witness_required"] = False
    if "phase0_pilot_enabled" in options:
        params["phase0_pilot_enabled"] = True
        params["feature_phase0_pilot_enabled"] = True
    if "phase0_no_pilot" in options:
        params["phase0_pilot_enabled"] = False
        params["feature_phase0_pilot_enabled"] = False
    if "phase_live_hysteresis_enabled" in options:
        params["phase_live_hysteresis_enabled"] = True
    if "phase_live_hysteresis_disabled" in options:
        params["phase_live_hysteresis_enabled"] = False
    _set_if_present(params, options, "phase1_prune_mode", "phase1_prune_mode", _option_str)

    phase1_shortlist = _option_int(options, "phase1_shortlist_size")
    if phase1_shortlist is not None:
        params["phase1_min_count"] = max(8, min(96, phase1_shortlist))
        params["phase1_max_count"] = max(96, min(100000, phase1_shortlist))
        params["phase1_pool_fraction"] = 1.0
    phase2_shortlist = _option_int(options, "phase2_shortlist_size")
    if phase2_shortlist is not None:
        params["phase2_min_count"] = max(4, min(64, phase2_shortlist))
        params["phase2_max_count"] = max(64, min(100000, phase2_shortlist))
        params["phase2_pool_fraction"] = 1.0

    def _set_cap_fraction(option_key: str, param_key: str, denominator: int | None) -> None:
        raw = _option_int(options, option_key)
        if raw is None:
            return
        denom = max(1, int(denominator or raw))
        params[param_key] = _clamp(float(raw) / float(denom), 0.0, 1.0)

    _set_cap_fraction("phase1_maturity_cap_min", "phase1_maturity_cap_min_fraction", phase1_shortlist)
    _set_cap_fraction("phase1_maturity_cap_max", "phase1_maturity_cap_max_fraction", phase1_shortlist)
    _set_cap_fraction("phase2_maturity_cap_min", "phase2_maturity_cap_min_fraction", phase2_shortlist)
    _set_cap_fraction("phase2_maturity_cap_max", "phase2_maturity_cap_max_fraction", phase2_shortlist)
    _set_cap_fraction("phase3_maturity_cap_min", "phase3_maturity_cap_min_fraction", phase2_shortlist)
    _set_cap_fraction("phase3_maturity_cap_max", "phase3_maturity_cap_max_fraction", phase2_shortlist)

    _encode_weight_family(
        params,
        prefix="burden",
        names=("compile", "measure"),
        values=(
            _option_float(options, "phase1_lambda_compile"),
            _option_float(options, "phase1_lambda_measure"),
        ),
        scale_low=1e-4,
        scale_high=1.0,
    )
    _encode_weight_family(
        params,
        prefix="cost",
        names=("depth", "group", "shot", "optdim", "reuse"),
        values=(
            _option_float(options, "phase2_w_depth"),
            _option_float(options, "phase2_w_group"),
            _option_float(options, "phase2_w_shot"),
            _option_float(options, "phase2_w_optdim"),
            _option_float(options, "phase2_w_reuse"),
        ),
        scale_low=1e-3,
        scale_high=10.0,
    )
    _encode_weight_family(
        params,
        prefix="hardware_cost",
        names=("2q", "d", "1q", "theta", "shot"),
        values=(
            _option_float_first(options, "phase1_lambda_2q", "phase2_lambda_2q"),
            _option_float_first(options, "phase1_lambda_d", "phase2_lambda_d"),
            _option_float_first(options, "phase1_lambda_1q", "phase2_lambda_1q"),
            _option_float_first(options, "phase1_lambda_theta", "phase2_lambda_theta"),
            _option_float_first(options, "phase1_lambda_shot", "phase2_lambda_shot"),
        ),
        scale_low=1e-4,
        scale_high=10.0,
    )
    _encode_weight_family(
        params,
        prefix="compile_cost",
        names=("cx", "sq", "rotation_step", "position_shift", "refit_active"),
        values=(
            _option_float(options, "phase1_compile_cx_proxy_weight"),
            _option_float(options, "phase1_compile_sq_proxy_weight"),
            _option_float(options, "phase1_compile_rotation_step_weight"),
            _option_float(options, "phase1_compile_position_shift_weight"),
            _option_float(options, "phase1_compile_refit_active_weight"),
        ),
        scale_low=1e-2,
        scale_high=20.0,
    )
    _encode_weight_family(
        params,
        prefix="measure_cost",
        names=("groups", "shots", "reuse"),
        values=(
            _option_float(options, "phase1_measure_groups_weight"),
            _option_float(options, "phase1_measure_shots_weight"),
            _option_float(options, "phase1_measure_reuse_weight"),
        ),
        scale_low=1e-2,
        scale_high=20.0,
    )
    effective_force = (
        _ACTIVE_INNER_OPTIMIZER
        if force_inner_optimizer == _FORCE_ACTIVE_INNER_OPTIMIZER
        else force_inner_optimizer
    )
    return _sanitize_trial_params(params, force_inner_optimizer=effective_force, meta_feature_profile=profile)


def _sanitize_trial_params(
    params: Mapping[str, Any],
    *,
    force_inner_optimizer: str | None = _FORCE_ACTIVE_INNER_OPTIMIZER,
    meta_feature_profile: str | None = None,
    policy_search_profile: str | None = "default",
) -> dict[str, Any]:
    """Clamp trial params to the current Optuna distributions.

    By default the active Phase3 policy route coerces historical/default
    optimizer params to SPSA. Pass ``force_inner_optimizer=None`` only for
    archival parsing tools/tests that need to preserve old records.
    """

    out = dict(params)
    policy_profile = _normalize_policy_search_profile(policy_search_profile)
    profile = _normalize_meta_feature_profile(meta_feature_profile or out.get("meta_feature_profile"))
    for key, value in PHASE0_OPTUNA_DEFAULTS.items():
        out.setdefault(key, value)
    float_ranges = {
        "cost_scale": (1e-3, 10.0),
        "burden_scale": (1e-4, 1.0),
        "compile_cost_scale": (1e-2, 20.0),
        "measure_cost_scale": (1e-2, 20.0),
        "phase1_lambda_leak": (0.0, 5.0),
        "phase1_pool_fraction": (0.05, 1.0),
        "phase1_qubit_slope": (1.0, 16.0),
        "phase2_pool_fraction": (0.05, 1.0),
        "phase2_qubit_slope": (1.0, 12.0),
        "family_repeat_penalty": (0.0, 5.0),
        "novelty_bonus": (0.0, 0.5),
        "phase2_gamma_N": (0.0, 4.0),
        "phase2_gamma_N_schedule_start": (0.0, 4.0),
        "phase2_gamma_N_schedule_end": (0.0, 1.5),
        "phase2_motif_bonus_weight": (0.0, 0.5),
        "rescue_expand_factor": (1.0, 4.0),
        "phase2_frontier_ratio": (0.5, 1.0),
        "phase2_leakage_cap": (1e-9, 1e6),
        "phase2_batch_near_degenerate_ratio": (0.90, 1.0),
        "phase2_batch_rank_rel_tol": (1e-9, 1e-3),
        "phase2_batch_additivity_tol": (1e-3, 1.0),
        "algebraic_phase2_lane_rel_threshold": _ALGEBRAIC_PHASE2_LANE_REL_THRESHOLD_RANGE,
        "algebraic_phase1_lane_quota_pressure": _ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE,
        "algebraic_phase2_lane_quota_pressure": _ALGEBRAIC_PHASE_LANE_QUOTA_PRESSURE_RANGE,
        "phase0_pilot_alpha": (0.02, 0.5),
        "phase0_pilot_threshold": (0.0, 1.0),
        "phase0_lane_quota_pressure": (0.0, 1.0),
        "phase1_prune_fraction": (0.05, 0.50),
        "phase1_prune_max_regression": (1e-10, 1e-6),
        "phase1_prune_tolerance_shot_coeff": (0.0, 5.0),
        "phase1_prune_tolerance_screen_coeff": (0.0, 0.1),
        "phase1_prune_tolerance_chem": (0.0, 1e-4),
        "phase1_prune_tolerance_rel_coeff": (0.0, 0.25),
        "phase1_prune_retained_gain_ratio": (0.10, 0.60),
        "phase1_prune_stagnation_threshold": (0.0, 1e-4),
        "phase1_prune_small_theta_abs": (1e-5, 1e-1),
        "phase1_prune_small_theta_relative": (0.0, 1.0),
        "phase1_prune_old_fraction": (0.0, 1.0),
        "phase1_prune_maturity_threshold": (0.35, 0.80),
        "phase1_prune_snr_threshold": (0.0, 3.0),
        "phase1_prune_collapse_peak_abs_min": (1e-5, 1e-1),
        "phase1_prune_collapse_current_abs_max": (1e-6, 1e-2),
        "phase1_prune_collapse_ratio": (0.05, 0.95),
        "phase1_prune_collapse_min_abs_drop": (1e-6, 1e-1),
        "phase1_maturity_cap_min_fraction": (0.0, 1.0),
        "phase1_maturity_cap_max_fraction": (0.0, 1.0),
        "phase2_maturity_cap_min_fraction": (0.0, 1.0),
        "phase2_maturity_cap_max_fraction": (0.0, 1.0),
        "phase3_maturity_cap_min_fraction": (0.0, 1.0),
        "phase3_maturity_cap_max_fraction": (0.0, 1.0),
        "phase2_null_nrem_high_threshold": (0.0, 1.25),
        "phase2_live_nrem_low_threshold": (0.0, 1.25),
        "phase3_null_nrem_high_threshold": (0.0, 1.75),
        "phase3_live_nrem_low_threshold": (0.0, 1.75),
        "phase3_frontier_ratio": (0.5, 1.0),
        "phase3_tie_beam_score_ratio": (1.0, 1.10),
        "phase3_tie_beam_abs_tol": (1e-8, 1e-3),
        "adapt_beam_lambda": (0.0, 1e6),
        "phase2_shortlist_fraction": (0.05, 1.0),
        # Historical HH/Powell/SPSA oracle policies used drop floors up to the
        # high 1e-4 range as an intentional compact-scaffold stop.  Clamping
        # warm starts to 1e-5 silently turns those policies into deeper runs,
        # which breaks exact replay and biases cost-at-accuracy searches.
        "adapt_drop_floor": (1e-10, 1e-3),
        "hardware_cost_scale": (1e-4, 10.0),
    }
    float_ranges.update(_SPSA_SCHEDULE_PARAM_RANGES)
    for name in ("depth", "group", "shot", "optdim", "reuse"):
        float_ranges[f"cost_logit_{name}"] = (-3.0, 3.0)
    for name in ("2q", "d", "1q", "theta", "shot"):
        float_ranges[f"hardware_cost_logit_{name}"] = (-3.0, 3.0)
    for name in ("compile", "measure"):
        float_ranges[f"burden_logit_{name}"] = (-3.0, 3.0)
    for name in ("cx", "sq", "rotation_step", "position_shift", "refit_active"):
        float_ranges[f"compile_cost_logit_{name}"] = (-3.0, 3.0)
    for name in ("groups", "shots", "reuse"):
        float_ranges[f"measure_cost_logit_{name}"] = (-3.0, 3.0)
    nullable_float_params = {
        "phase2_gamma_N_schedule_start",
        "phase2_gamma_N_schedule_end",
        "phase2_motif_bonus_weight",
    }
    for key, (low, high) in float_ranges.items():
        if key in out:
            if out[key] is None and key in nullable_float_params:
                continue
            out[key] = _clamp(float(out[key]), low, high)

    int_ranges = {
        "phase1_min_count": (8, 96),
        "phase1_max_count": (96, 100000),
        "phase2_min_count": (4, 64),
        "phase2_max_count": (64, 100000),
        "phase3_tie_beam_max_branches": (1, 5),
        "adapt_beam_live_branches": (1, 8),
        "adapt_beam_children_per_parent": (1, 6),
        "adapt_beam_terminated_keep": (1, 6),
        "adapt_window_size": (8, 192),
        "adapt_window_topk": (0, 96),
        "phase1_probe_max_positions": (2, 12),
        "phase0_pilot_max_records": (0, 100000),
        "adapt_max_depth": (8, 128),
        "adapt_maxiter": (100, 6000),
        "adapt_drop_patience": (1, 16),
        "adapt_drop_min_depth": (1, 32),
        "phase2_batch_target_size": (2, 16),
        "phase2_batch_size_cap": (4, 32),
        "phase1_prune_min_candidates": (1, 3),
        "phase1_prune_max_candidates": (2, 10),
        "phase1_prune_protect_steps": (1, 4),
        "phase1_prune_stale_age": (1, 6),
        "phase1_prune_cooldown_steps": (0, 16),
        "phase1_prune_local_window_size": (1, 32),
        "phase1_prune_recovery_trust_radius": (0.0, 1.0),
        "phase1_prune_checkpoint_period": (2, 6),
        "phase1_prune_live_min_depth": (0, 512),
        "phase1_prune_collapse_min_observations": (2, 6),
        "phase_maturity_shot_min": (1, 2),
        "phase_maturity_shot_max": (1, 8),
        "phase1_maturity_shot_cap": (0, 8),
        "phase2_maturity_shot_cap": (0, 8),
        "phase3_maturity_shot_cap": (0, 8),
        "phase2_hysteresis_steps": (1, 4),
        "phase3_hysteresis_steps": (1, 4),
    }
    for key, (low, high) in int_ranges.items():
        if key in out:
            out[key] = int(max(low, min(high, int(out[key]))))
    if "phase2_batch_target_size" in out and "phase2_batch_size_cap" in out:
        out["phase2_batch_target_size"] = int(
            min(int(out["phase2_batch_target_size"]), int(out["phase2_batch_size_cap"]))
        )
    if "phase1_prune_min_candidates" in out and "phase1_prune_max_candidates" in out:
        out["phase1_prune_min_candidates"] = int(
            min(int(out["phase1_prune_min_candidates"]), int(out["phase1_prune_max_candidates"]))
        )
    if "inner_optimizer" in out:
        out["inner_optimizer"] = str(out["inner_optimizer"]).strip().upper()
    categories = {
        "pool_key": {"full_meta", "hamiltonian_quadratures"},
        "adapt_reopt_policy": {"append_only", "full", "windowed"},
        "adapt_insertion_mode": {"adaptive", "always", "append_only"},
        "phase1_prune_mode": {"live", "final", "both"},
        "phase1_prune_policy": set(_PHASE1_PRUNE_POLICY_CHOICES),
        "phase1_prune_tolerance_mode": {"auto", "fixed", "adaptive_v1"},
        "phase2_novelty_mode": set(_PHASE2_NOVELTY_MODE_CHOICES),
        "phase2_gamma_N_schedule_mode": {"fixed", "depth_linear_v1"},
        "phase3_selector_policy": set(_PHASE3_SELECTOR_POLICY_CHOICES),
        "feature_phase3_selector_policy": set(_PHASE3_SELECTOR_POLICY_CHOICES),
        "phase3_selector_geometry_mode": set(_PHASE3_SELECTOR_GEOMETRY_MODE_CHOICES),
        "phase3_novelty_ablation_mode": set(_PHASE3_NOVELTY_ABLATION_MODE_CHOICES),
        "feature_phase3_novelty_ablation_mode": set(_PHASE3_NOVELTY_ABLATION_MODE_CHOICES),
        "phase3_window_relaxation_mode": set(_PHASE3_WINDOW_RELAXATION_MODE_CHOICES),
        "feature_phase3_window_relaxation_mode": set(_PHASE3_WINDOW_RELAXATION_MODE_CHOICES),
        "phase3_batch_selection_mode": set(_PHASE3_BATCH_SELECTION_MODE_CHOICES),
        "feature_phase3_batch_selection_mode": set(_PHASE3_BATCH_SELECTION_MODE_CHOICES),
        "phase3_batch_prefilter_mode": set(_PHASE3_BATCH_PREFILTER_MODE_CHOICES),
        "feature_phase3_batch_prefilter_mode": set(_PHASE3_BATCH_PREFILTER_MODE_CHOICES),
        "phase0_algebraic_lane_mode": set(PHASE0_ALGEBRAIC_LANE_MODE_CHOICES),
        "feature_phase0_algebraic_lane_mode": set(PHASE0_ALGEBRAIC_LANE_MODE_CHOICES),
        "static_lane_route": set(STATIC_LANE_ROUTE_CHOICES),
        "physical_lane_shortlist_aggressiveness": set(PHYSICAL_LANE_SHORTLIST_AGGRESSIVENESS_CHOICES),
        "meta_feature_profile": set(_META_FEATURE_PROFILES),
        "inner_optimizer": set(_ARCHIVAL_INNER_OPTIMIZERS),
    }
    defaults = default_trial_params(policy_profile, meta_feature_profile=profile)
    if "phase1_prune_policy" not in out:
        out["phase1_prune_policy"] = _DEFAULT_PHASE1_PRUNE_POLICY
    if "phase3_selector_policy" not in out:
        out["phase3_selector_policy"] = _DEFAULT_PHASE3_SELECTOR_POLICY
    for key, choices in categories.items():
        if key in out and out[key] not in choices:
            out[key] = defaults[key]
    for key in (
        "adapt_allow_repeats",
        "phase2_enable_batching",
        "feature_phase3_batching_enabled",
        "phase1_prune_enabled",
        "feature_phase1_prune_enabled",
        "phase1_prune_amplitude_witness_required",
        "feature_phase1_prune_amplitude_witness_required",
        "phase_live_hysteresis_enabled",
        "phase0_pilot_enabled",
        "feature_phase0_pilot_enabled",
    ):
        if key in out:
            out[key] = _coerce_bool(out[key])
    if force_inner_optimizer is not None:
        effective_force = (
            _ACTIVE_INNER_OPTIMIZER
            if force_inner_optimizer == _FORCE_ACTIVE_INNER_OPTIMIZER
            else force_inner_optimizer
        )
        out["inner_optimizer"] = _normalize_fixed_inner_optimizer(effective_force)
    out = _force_canonical_static_trial_route(
        out,
        meta_feature_profile=profile,
        policy_search_profile=policy_profile,
    )
    return out


def load_historical_ledger(path: str | Path | None) -> Mapping[str, Any] | None:
    if path in {None, ""}:
        return None
    ledger_path = Path(path)
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to load historical ledger {ledger_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Historical ledger {ledger_path} is not a JSON object.")
    out = dict(payload)
    out.setdefault("_source_artifact_path", str(ledger_path))
    out.setdefault("_source_sha256", _sha256_file(ledger_path))
    return out


def _pipeline_arg_value(args: Sequence[str], flag: str) -> str | None:
    try:
        idx = tuple(args).index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(args):
        return None
    return str(args[idx + 1])


def _args_option_map(args: Sequence[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    idx = 0
    tokens = [str(x) for x in args]
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


_EFFECTIVE_TRIAL_MANIFEST_SCHEMA = "phase3_effective_trial_manifest_v1"
_EFFECTIVE_BENCHMARK_MANIFEST_SCHEMA = "phase3_effective_benchmark_manifest_v1"
_EFFECTIVE_TRIAL_REPLAY_AUDIT_SCHEMA = "phase3_effective_trial_replay_audit_v1"
_EFFECTIVE_TRIAL_MANIFEST_USER_ATTR_KEY = "effective_trial_manifest"
_PAPER_I_PHYS_V1_PROFILE = "paper_i_phys_v1"
_PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC = "abs_error_Nph_plus_1"
_PAPER_I_PHYS_V1_REFERENCE_CUTOFF_METRIC = "abs_error_reference_cutoff"
_PAPER_I_PHYS_V1_SAME_CUTOFF_METRIC = "abs_error_same_cutoff"
_PAPER_I_CLEAN_TAU_PHYS = 2e-4
_PAPER_I_CLEAN_TAU_TIGHT = _PAPER_I_CLEAN_TAU_PHYS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_effective_trial_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = Path(path)
    _write_json(manifest_path, payload)
    sha256 = _sha256_file(manifest_path)
    return {
        "schema": "phase3_effective_trial_manifest_pointer_v1",
        "path": str(manifest_path),
        "sha256": sha256,
        "lifecycle_status": str(payload.get("lifecycle_status", "unknown")),
        "manifest_schema": str(payload.get("schema", _EFFECTIVE_TRIAL_MANIFEST_SCHEMA)),
    }


def _set_effective_trial_manifest_user_attrs(trial: Any, pointer: Mapping[str, Any]) -> None:
    if not hasattr(trial, "set_user_attr"):
        return
    trial.set_user_attr(_EFFECTIVE_TRIAL_MANIFEST_USER_ATTR_KEY, _jsonable(dict(pointer)))
    trial.set_user_attr("effective_trial_manifest_json", str(pointer.get("path")))
    trial.set_user_attr("effective_trial_manifest_sha256", str(pointer.get("sha256")))
    trial.set_user_attr("effective_trial_manifest_schema", str(pointer.get("manifest_schema")))


def _audited_missing(reason: str, *, required_for_cthc: bool = False, profile: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "audited_missing",
        "reason": str(reason),
        "required_for_cthc": bool(required_for_cthc),
    }
    if profile not in {None, ""}:
        payload["profile"] = str(profile)
    return payload


def resolve_paper_i_phys_v1_target(
    spec: HamiltonianBenchmarkSpec,
    *,
    energy_unit_E0: float = 1.0,
) -> dict[str, Any]:
    """Resolve the clean Paper-I physical target for static rows.

    The clean Table-I target is a fixed total error threshold for the displayed
    L=2, unit-normalized rows.  Phonon-bearing rows use the higher-cutoff
    external/reference error as the primary metric; non-phonon rows use the
    same-Hilbert-space exact error.  The local phonon cutoff is recorded
    separately and never multiplies N_phys.
    """

    family = str(spec.family).strip().lower()
    if bool(spec.features.molecular) and family != "molecular_vibronic_h2":
        raise ValueError(f"paper_i_phys_v1_not_supported_for_molecular_family:{spec.family}")
    L = int(spec.features.L)
    if L < 1:
        raise ValueError(f"paper_i_phys_v1_not_supported_for_size:family={family},L={L}")
    phonon_cutoff_work: int | None = None
    phonon_cutoff_eval_plus_one: int | None = None
    phonon_cutoff_eval_reference: int | None = None
    phonon_reference_policy = "not_applicable"
    accuracy_gate_metric = _PAPER_I_PHYS_V1_SAME_CUTOFF_METRIC
    if bool(spec.features.bosonic):
        if family not in _HEAVY_BOSONIC_CUTOFF2_FAMILIES and family != "molecular_vibronic_h2":
            raise ValueError(f"paper_i_phys_v1_not_supported_for_family:{spec.family}")
        raw_nph = _pipeline_arg_value(spec.base_pipeline_args, "--n-ph-max")
        if raw_nph in {None, ""}:
            raise ValueError(f"paper_i_phys_v1_missing_phonon_cutoff:{spec.benchmark_id}")
        phonon_cutoff_work = int(float(str(raw_nph)))
        if phonon_cutoff_work < 1:
            raise ValueError(f"paper_i_phys_v1_invalid_phonon_cutoff:{spec.benchmark_id}:{phonon_cutoff_work}")
        phonon_cutoff_eval_plus_one = int(phonon_cutoff_work + 1)
        phonon_cutoff_eval_reference = (
            int(spec.exact_reference_n_ph_max)
            if spec.exact_reference_n_ph_max is not None
            else int(phonon_cutoff_eval_plus_one)
        )
        if phonon_cutoff_eval_reference < phonon_cutoff_work:
            raise ValueError(
                "paper_i_phys_v1_reference_cutoff_below_work_cutoff:"
                f"{spec.benchmark_id}:work={phonon_cutoff_work}:ref={phonon_cutoff_eval_reference}"
            )
        phonon_reference_policy = (
            "work_plus_one_reference_cutoff"
            if phonon_cutoff_eval_reference == phonon_cutoff_eval_plus_one
            else "explicit_reference_cutoff"
        )
        accuracy_gate_metric = _PAPER_I_PHYS_V1_REFERENCE_CUTOFF_METRIC
    n_phys = int(L)
    E0 = float(energy_unit_E0)
    tau_phys = float(_PAPER_I_CLEAN_TAU_PHYS * E0)
    tau_tight = float(_PAPER_I_CLEAN_TAU_TIGHT * E0)
    n_phys_policy = (
        "site_normalized_for_hubbard_holstein"
        if family == "hh"
        else "problem_size_L_recorded_not_target_scaled"
    )
    return {
        "schema": "paper_i_physical_target_v1",
        "status": "present",
        "accuracy_policy_name": _PAPER_I_PHYS_V1_PROFILE,
        "tight_accuracy_policy_name": "paper_i_tight_v1",
        "family": family,
        "benchmark_id": str(spec.benchmark_id),
        "N_phys": n_phys,
        "N_phys_policy": n_phys_policy,
        "energy_unit_E0": E0,
        "tau_phys": tau_phys,
        "tau_tight": tau_tight,
        "accuracy_threshold_value": tau_phys,
        "tight_accuracy_threshold_value": tau_tight,
        "accuracy_gate_metric": accuracy_gate_metric,
        "phonon_cutoff_work": phonon_cutoff_work,
        "phonon_cutoff_eval_same": phonon_cutoff_work,
        "phonon_cutoff_eval_plus_one": phonon_cutoff_eval_plus_one,
        "phonon_cutoff_eval_reference": phonon_cutoff_eval_reference,
        "phonon_reference_policy": phonon_reference_policy,
        "cutoff_ladder_acceptance_threshold": tau_tight,
        "local_phonon_branches_per_site": 1 if bool(spec.features.bosonic) else 0,
        "legacy_same_cutoff_thresholds": [1e-6, 1e-8],
        "normalization_notes": (
            "Clean Paper-I uses a total tau_phys=2e-4 for L=2 unit-normalized rows. "
            "Do not scale by N_ph or by site plus local oscillator count. "
            "For phonon-bearing rows, same-cutoff error is diagnostic only."
        ),
    }


def _paper_i_phys_v1_target_or_missing(spec: HamiltonianBenchmarkSpec) -> dict[str, Any]:
    try:
        return resolve_paper_i_phys_v1_target(spec)
    except ValueError as exc:
        return _audited_missing(str(exc), required_for_cthc=False, profile=_PAPER_I_PHYS_V1_PROFILE)


def _runner_keyword(runner: Any, name: str, default: Any = None) -> Any:
    keywords = getattr(runner, "keywords", None)
    if isinstance(keywords, Mapping) and name in keywords:
        return keywords[name]
    return default


def _benchmark_artifact_paths(trial_dir: Path, spec: HamiltonianBenchmarkSpec) -> dict[str, str]:
    case_dir = Path(trial_dir) / spec.benchmark_id
    return {
        "case_dir": str(case_dir),
        "policy_json": str(case_dir / "json" / "policy.json"),
        "result_json": str(case_dir / "json" / "result.json"),
        "compile_json": str(case_dir / "json" / "compile_scout_fake_marrakesh.json"),
        "policy_roundtrip_audit_json": str(case_dir / "json" / "policy_roundtrip_audit.json"),
        "command_sh": str(case_dir / "logs" / "command.sh"),
        "compile_command_sh": str(case_dir / "logs" / "compile_command.sh"),
        "stdout_log": str(case_dir / "logs" / "stdout.log"),
        "stderr_log": str(case_dir / "logs" / "stderr.log"),
        "compile_stdout_log": str(case_dir / "logs" / "compile_stdout.log"),
        "compile_stderr_log": str(case_dir / "logs" / "compile_stderr.log"),
    }


def _artifact_digest_payload(artifact_paths: Mapping[str, Any] | None) -> dict[str, Any]:
    digests: dict[str, Any] = {"schema": "phase3_artifact_digest_manifest_v1"}
    for key, raw_path in dict(artifact_paths or {}).items():
        if raw_path in {None, ""}:
            continue
        path = Path(str(raw_path))
        entry: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            entry["sha256"] = _sha256_file(path)
            entry["size_bytes"] = int(path.stat().st_size)
        digests[str(key)] = entry
    return _jsonable(digests)


def _single_artifact_digest_payload(path_value: Any) -> dict[str, Any]:
    if path_value in {None, ""}:
        return {"path": None, "exists": False, "sha256": None}
    path = Path(str(path_value))
    entry: dict[str, Any] = {"path": str(path), "exists": path.exists(), "sha256": None}
    if path.exists() and path.is_file():
        entry["sha256"] = _sha256_file(path)
        entry["size_bytes"] = int(path.stat().st_size)
    return _jsonable(entry)


def _work_phonon_cutoff_from_spec_payload(spec_payload: Mapping[str, Any] | None) -> int | None:
    if not isinstance(spec_payload, Mapping):
        return None
    raw_args = spec_payload.get("base_pipeline_args", ())
    if not isinstance(raw_args, Sequence) or isinstance(raw_args, (str, bytes)):
        return None
    raw_nph = _pipeline_arg_value(tuple(str(x) for x in raw_args), "--n-ph-max")
    if raw_nph in {None, ""}:
        return None
    try:
        return int(float(str(raw_nph)))
    except Exception:
        return None


def _artifact_result_metric_mismatches(result: BenchmarkResult, result_json_path: str | Path | None) -> list[str]:
    if result_json_path in {None, ""}:
        return ["missing_result_json_path"]
    path = Path(str(result_json_path))
    if not path.exists():
        return ["missing_result_json"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return ["result_json_not_object"]
        metrics = extract_adapt_energy_metrics(payload)
    except Exception as exc:
        return [f"result_json_unreadable:{type(exc).__name__}:{exc}"]
    mismatches: list[str] = []

    def _compare(name: str, left: Any, right: Any) -> None:
        if left is None or right is None:
            return
        try:
            if not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-10):
                mismatches.append(name)
        except Exception:
            if str(left) != str(right):
                mismatches.append(name)

    _compare("energy", metrics.energy, result.energy)
    _compare("same_cutoff_exact_gs_energy", metrics.exact_gs_energy, result.same_cutoff_exact_gs_energy)
    _compare("abs_delta_e_same_cutoff", metrics.abs_delta_e, result.abs_delta_e_same_cutoff)
    return mismatches


def _load_json_mapping(path: str | Path | None) -> dict[str, Any] | None:
    if path in {None, ""}:
        return None
    file_path = Path(str(path))
    if not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _jsonable(dict(payload)) if isinstance(payload, Mapping) else None


def _artifact_result_from_files(
    result: BenchmarkResult,
    spec: HamiltonianBenchmarkSpec,
    artifacts: Mapping[str, Any] | None,
) -> tuple[BenchmarkResult | None, dict[str, Any] | None, list[str]]:
    """Rebuild objective-input fields from result/compile artifacts, not manifest echoes."""

    paths = dict(artifacts or {})
    failures: list[str] = []
    result_path = paths.get("result_json") or result.result_json
    compile_path = paths.get("compile_json") or result.compile_json
    result_payload = _load_json_mapping(result_path)
    if result_payload is None:
        return None, None, ["missing_or_unreadable_result_json"]
    try:
        metrics = extract_adapt_energy_metrics(result_payload)
    except Exception as exc:
        return None, result_payload, [f"result_json_metrics_failed:{type(exc).__name__}:{exc}"]
    reference_energy, reference_nph, reference_failure = _reference_cutoff_energy_for_spec(spec)
    if reference_failure is not None:
        failures.append(reference_failure)
    energy = _as_float_or_none(metrics.energy)
    same_cutoff_exact = _as_float_or_none(metrics.exact_gs_energy)
    same_error = _as_float_or_none(metrics.abs_delta_e)
    reference_error = None
    cutoff_error = None
    if energy is not None and reference_energy is not None:
        reference_error = abs(float(energy) - float(reference_energy))
    if same_cutoff_exact is not None and reference_energy is not None:
        cutoff_error = abs(float(same_cutoff_exact) - float(reference_energy))
    count_2q, depth_2q, circuit_depth, logical_params, runtime_params = (
        _compile_metrics(Path(str(compile_path))) if compile_path not in {None, ""} else (None, None, None, None, None)
    )
    if compile_path in {None, ""}:
        failures.append("missing_compile_json_path")
    elif not Path(str(compile_path)).exists():
        failures.append("missing_compile_json")
    artifact_result = replace(
        result,
        energy=energy,
        exact_gs_energy=same_cutoff_exact,
        same_cutoff_exact_gs_energy=same_cutoff_exact,
        exact_reference_energy=reference_energy,
        exact_reference_n_ph_max=reference_nph,
        abs_delta_e=reference_error if reference_error is not None else same_error,
        abs_delta_e_same_cutoff=same_error,
        abs_delta_e_reference=reference_error,
        cutoff_abs_delta_e=cutoff_error,
        count_2q=count_2q,
        depth_2q=depth_2q,
        circuit_depth=circuit_depth,
        parameter_count=logical_params,
        runtime_parameter_count=runtime_params,
        physical_target_manifest=None,
        cutoff_diagnostics=None,
        paper_i_first_crossing=None,
        result_json=str(result_path) if result_path not in {None, ""} else result.result_json,
        compile_json=str(compile_path) if compile_path not in {None, ""} else result.compile_json,
    )
    return artifact_result, result_payload, failures


def _compare_floatish_payload(
    mismatches: list[str],
    *,
    label: str,
    left: Any,
    right: Any,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-10,
) -> None:
    if left is None and right is None:
        return
    if left is None or right is None:
        mismatches.append(label)
        return
    left_float = _as_float_or_none(left)
    right_float = _as_float_or_none(right)
    if left_float is not None and right_float is not None:
        if not math.isclose(float(left_float), float(right_float), rel_tol=rel_tol, abs_tol=abs_tol):
            mismatches.append(label)
        return
    if str(left) != str(right):
        mismatches.append(label)


def _paper_i_artifact_objective_input_mismatches(
    *,
    child: Mapping[str, Any],
    spec: HamiltonianBenchmarkSpec,
    manifest_result: BenchmarkResult,
    artifact_result: BenchmarkResult,
    result_payload: Mapping[str, Any] | None,
) -> list[str]:
    mismatches: list[str] = []
    target_payload = child.get("physical_target") if isinstance(child.get("physical_target"), Mapping) else None
    physical_target, cutoff, crossing = _paper_i_artifacts_for_result(
        artifact_result,
        spec,
        result_payload=result_payload,
        physical_target=target_payload,
    )
    stored_metrics = child.get("exact_reference_metrics") if isinstance(child.get("exact_reference_metrics"), Mapping) else {}
    recomputed_metrics = _exact_reference_metrics_payload(
        artifact_result,
        physical_target,
        child.get("spec") if isinstance(child.get("spec"), Mapping) else None,
    )
    for key in (
        "energy_exact_Nph",
        "energy_exact_Nph_plus_1",
        "energy_exact_reference_cutoff",
        "abs_error_same_cutoff",
        "abs_error_Nph_plus_1",
        "abs_error_reference_cutoff",
        "cutoff_abs_delta_e",
    ):
        _compare_floatish_payload(mismatches, label=f"exact_reference_metrics.{key}", left=stored_metrics.get(key), right=recomputed_metrics.get(key))
    for key in ("plus_one_status", "reference_cutoff_status", "same_cutoff_and_plus_one_distinct"):
        if stored_metrics.get(key) != recomputed_metrics.get(key):
            mismatches.append(f"exact_reference_metrics.{key}")
    stored_cutoff = child.get("cutoff_diagnostics") if isinstance(child.get("cutoff_diagnostics"), Mapping) else {}
    for key in (
        "status",
        "plus_one_status",
        "reference_cutoff_status",
        "same_cutoff_and_plus_one_distinct",
        "accuracy_gate_metric",
    ):
        if stored_cutoff.get(key) != cutoff.get(key):
            mismatches.append(f"cutoff_diagnostics.{key}")
    for key in (
        "energy_adapt_Nph",
        "energy_exact_Nph",
        "energy_exact_Nph_plus_1",
        "energy_exact_reference_cutoff",
        "abs_error_same_cutoff",
        "abs_error_Nph_plus_1",
        "abs_error_reference_cutoff",
        "cutoff_abs_delta_e",
    ):
        _compare_floatish_payload(mismatches, label=f"cutoff_diagnostics.{key}", left=stored_cutoff.get(key), right=cutoff.get(key))
    stored_crossing = child.get("paper_i_first_crossing") if isinstance(child.get("paper_i_first_crossing"), Mapping) else {}
    for key in ("status", "reached", "accuracy_gate_metric"):
        if stored_crossing.get(key) != crossing.get(key):
            mismatches.append(f"paper_i_first_crossing.{key}")
    for key in (
        "history_position_tau",
        "k_tau",
        "primary_error_at_crossing",
        "same_cutoff_error_at_crossing",
        "resource_score",
        "terminal_primary_error",
        "terminal_same_cutoff_error",
    ):
        _compare_floatish_payload(mismatches, label=f"paper_i_first_crossing.{key}", left=stored_crossing.get(key), right=crossing.get(key))
    for key in ("count_2q", "depth_2q", "circuit_depth", "parameter_count", "runtime_parameter_count"):
        _compare_floatish_payload(
            mismatches,
            label=f"compile_metrics.{key}",
            left=getattr(manifest_result, key),
            right=getattr(artifact_result, key),
        )
    return mismatches


def _paper_i_physical_target_mismatches(
    spec: HamiltonianBenchmarkSpec,
    target_payload: Mapping[str, Any] | None,
    *,
    label: str,
) -> list[str]:
    if not isinstance(target_payload, Mapping) or target_payload.get("status") != "present":
        return [f"{label}:missing"]
    mismatches: list[str] = []
    try:
        canonical = resolve_paper_i_phys_v1_target(spec)
    except Exception as exc:
        return [f"{label}:canonical_resolver_failed:{type(exc).__name__}:{exc}"]
    for key in (
        "N_phys",
        "N_phys_policy",
        "tau_phys",
        "tau_tight",
        "accuracy_gate_metric",
        "phonon_cutoff_work",
        "phonon_cutoff_eval_same",
        "phonon_cutoff_eval_plus_one",
    ):
        _compare_floatish_payload(mismatches, label=f"{label}.{key}", left=target_payload.get(key), right=canonical.get(key))
    return mismatches


def _artifact_hash_mismatches(artifact_hashes: Mapping[str, Any] | None) -> list[str]:
    mismatches: list[str] = []
    for key, payload in dict(artifact_hashes or {}).items():
        if key == "schema" or not isinstance(payload, Mapping):
            continue
        path = payload.get("path")
        stored_sha = payload.get("sha256")
        expected_exists = bool(payload.get("exists"))
        if not path or not expected_exists or not stored_sha:
            continue
        file_path = Path(str(path))
        if not file_path.exists():
            mismatches.append(f"{key}:missing")
            continue
        if _sha256_file(file_path) != str(stored_sha):
            mismatches.append(f"{key}:sha256")
    return mismatches


def _collect_cthc_required_missing(node: Any, path: str = "$", *, limit: int = 200) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(node, Mapping):
        if node.get("status") == "audited_missing" and bool(node.get("required_for_cthc")):
            out.append({"path": path, "reason": node.get("reason"), "profile": node.get("profile")})
            if len(out) >= limit:
                return out
        if bool(node.get("required_for_cthc")) and node.get("cthc_ready") is False:
            out.append({"path": path, "reason": "cthc_ready_false", "status": node.get("status")})
            if len(out) >= limit:
                return out
        for key, value in node.items():
            out.extend(_collect_cthc_required_missing(value, f"{path}.{key}", limit=limit - len(out)))
            if len(out) >= limit:
                return out
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for idx, value in enumerate(node):
            out.extend(_collect_cthc_required_missing(value, f"{path}[{idx}]", limit=limit - len(out)))
            if len(out) >= limit:
                return out
    return out


def _hardware_noise_policy_payload(
    policy: AlgorithmPolicy,
    objective_weights: StaticObjectiveWeights | None,
    objective_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    static = policy.static
    provenance = dict(objective_provenance or {})
    shot_weight = None if objective_weights is None else float(objective_weights.shot_cost)
    return {
        "schema": "phase3_noiseless_inactive_hardware_noise_policy_v1",
        "regime": "noiseless_static_adapt",
        "hardware_resolution_mode": str(getattr(static, "hardware_resolution_mode", "ideal") or "ideal"),
        "hardware_resolution_profile_json": getattr(static, "hardware_resolution_profile_json", None),
        "hardware_resolution_profile_name": getattr(static, "hardware_resolution_profile_name", None),
        "phase3_backend_cost_mode": str(getattr(static, "phase3_backend_cost_mode", "proxy") or "proxy"),
        "phase3_lifetime_cost_mode": "off",
        "phase3_runtime_split_mode": str(getattr(static, "phase3_runtime_split_mode", "off") or "off"),
        "noise_model": "inactive",
        "mitigation": "inactive",
        "readout_mitigation": "not_used",
        "gate_twirling": "not_used",
        "dynamical_decoupling": "not_used",
        "zne": "not_used",
        "objective_consumes_noisy_energy": bool(provenance.get("objective_consumes_noisy_energy", False)),
        "objective_noise_mode": str(provenance.get("objective_noise_mode", "exact_noiseless_v1")),
        "objective_shot_weight": shot_weight,
        "shot_weight_status": "inactive" if shot_weight is None or shot_weight <= 0.0 else "active_measurement_proxy_required",
    }


def _objective_config_payload(config: GlobalObjectiveConfig | None, objective_weights: StaticObjectiveWeights | None) -> dict[str, Any] | None:
    if config is not None:
        return _jsonable(asdict(config))
    if objective_weights is not None:
        return {"weights": _jsonable(asdict(objective_weights))}
    return None


def _build_effective_benchmark_intent(
    *,
    spec: HamiltonianBenchmarkSpec,
    policy: AlgorithmPolicy,
    trial_dir: Path,
    python_bin: str,
    benchmark_target_abs_delta_e: float | None,
) -> dict[str, Any]:
    artifacts = _benchmark_artifact_paths(trial_dir, spec)
    output_json = Path(artifacts["result_json"])
    benchmark_target_reference_energy, benchmark_target_reference_nph, benchmark_target_reference_failure, benchmark_target_reference_source = _benchmark_target_reference_for_spec(
        spec,
        benchmark_target_abs_delta_e=benchmark_target_abs_delta_e,
    )
    command = build_static_command(
        python_bin=python_bin,
        spec=spec,
        policy=policy,
        output_json=output_json,
        benchmark_target_abs_delta_e=benchmark_target_abs_delta_e,
        benchmark_target_reference_energy=benchmark_target_reference_energy,
    )
    requested_route = normalize_static_route_id(policy.static.static_route_id, default=ROUTE_ID_A)
    emitted_route = normalize_static_route_id(
        _pipeline_arg_value(command, "--static-route-id") or requested_route,
        default=ROUTE_ID_UNSPECIFIED,
    )
    selected_route = str(getattr(spec, "selected_logical_route", "standard") or "standard").strip().lower().replace("-", "_")
    return {
        "schema": _EFFECTIVE_BENCHMARK_MANIFEST_SCHEMA,
        "lifecycle_status": "intent",
        "benchmark_id": str(spec.benchmark_id),
        "family": str(spec.family),
        "spec": _jsonable(asdict(spec)),
        "artifact_paths": artifacts,
        "emitted_cli_args": [str(token) for token in command],
        "route_identity": {
            "schema": "phase3_effective_static_route_identity_v1",
            "requested_static_route_id": requested_route,
            "emitted_static_route_id": emitted_route,
            "selected_logical_route": selected_route,
            "meta_feature_profile": str(policy.static.static_meta_feature_profile),
            "meta_feature_policy": _jsonable(_meta_feature_policy_payload(policy)),
            "coercion_status": (
                "forced_route_a_to_unspecified_for_nonstandard_selected_logical_route"
                if requested_route == ROUTE_ID_A and emitted_route == ROUTE_ID_UNSPECIFIED and selected_route != "standard"
                else "none"
            ),
        },
        "benchmark_target_abs_delta_e": None if benchmark_target_abs_delta_e is None else float(benchmark_target_abs_delta_e),
        "benchmark_target_reference_energy": None if benchmark_target_reference_energy is None else float(benchmark_target_reference_energy),
        "benchmark_target_reference_n_ph_max": benchmark_target_reference_nph,
        "benchmark_target_reference_source": str(benchmark_target_reference_source),
        "benchmark_target_reference_failure": benchmark_target_reference_failure,
        "physical_target": _paper_i_phys_v1_target_or_missing(spec),
        "exact_reference_metrics": _audited_missing("pending_benchmark_result", required_for_cthc=True),
        "result": None,
        "policy_roundtrip_audit_summary": _audited_missing("pending_benchmark_result", required_for_cthc=True),
    }


def build_effective_trial_manifest_intent(
    *,
    mode: str,
    trial: Any,
    specs: Sequence[HamiltonianBenchmarkSpec],
    trial_dir: Path,
    policy: AlgorithmPolicy,
    sampled_params: Mapping[str, Any],
    seed: int,
    objective_weights: StaticObjectiveWeights | None = None,
    config: GlobalObjectiveConfig | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    suite_profile: str | None = None,
    benchmarks_per_trial_jobs: int | None = None,
    runner: Any = None,
    warm_start_provenance: Mapping[str, Any] | None = None,
    benchmark_policies: Mapping[str, AlgorithmPolicy] | None = None,
) -> dict[str, Any]:
    selected = tuple(specs)
    benchmark_policy_by_id = {str(key): value for key, value in dict(benchmark_policies or {}).items()}
    effective_weights = config.weights if config is not None else objective_weights
    objective_provenance = objective_provenance_payload(config) if config is not None else objective_provenance_payload(None)
    discovery_mode = _normalize_discovery_objective_mode(objective_provenance.get("discovery_objective_mode"))
    python_bin = str(_runner_keyword(runner, "python_bin", sys.executable))
    benchmark_target_abs_delta_e = _runner_keyword(runner, "benchmark_target_abs_delta_e", None)
    if benchmark_target_abs_delta_e is not None:
        try:
            benchmark_target_abs_delta_e = float(benchmark_target_abs_delta_e)
        except Exception:
            benchmark_target_abs_delta_e = None
    trial_number = int(getattr(trial, "number", seed))
    benchmark_manifests = [
        _build_effective_benchmark_intent(
            spec=spec,
            policy=benchmark_policy_by_id.get(str(spec.benchmark_id), policy),
            trial_dir=trial_dir,
            python_bin=python_bin,
            benchmark_target_abs_delta_e=benchmark_target_abs_delta_e,
        )
        for spec in selected
    ]
    return {
        "schema": _EFFECTIVE_TRIAL_MANIFEST_SCHEMA,
        "lifecycle_status": "intent",
        "generated_utc": _now_utc(),
        "pipeline": _PIPELINE_NAME,
        "mode": str(mode),
        "trial_identity": {
            "trial_number": trial_number,
            "study_name": None if study_name in {None, ""} else str(study_name),
            "storage": None if storage in {None, ""} else str(storage),
            "seed": int(seed),
            "artifact_root": str(Path(trial_dir).parent),
            "trial_dir": str(Path(trial_dir)),
            "run_timestamp_utc": _now_utc(),
        },
        "repo_revision": _audited_missing("repo_revision_not_captured_in_first_slice", required_for_cthc=True),
        "suite_profile": str(suite_profile or "standard").strip().lower().replace("-", "_"),
        "required_target_profile": str(objective_provenance.get("required_target_profile", "none") or "none"),
        "benchmark_count": int(len(selected)),
        "benchmark_suite": [_jsonable(asdict(spec)) for spec in selected],
        "benchmark_manifests": benchmark_manifests,
        "global_suite_child_manifest_sidecars": (
            _audited_missing("single_suite_level_manifest_only_in_first_slice", required_for_cthc=False)
            if str(mode) == "global"
            else {"status": "not_applicable_single_benchmark_oracle"}
        ),
        "sampled_params": _jsonable(dict(sampled_params)),
        "policy": _jsonable(asdict(policy)),
        "policy_summary": _policy_audit_summary(policy),
        "benchmark_effective_policies": _jsonable(
            {key: asdict(value) for key, value in benchmark_policy_by_id.items()}
        ),
        "benchmark_effective_policy_summaries": _jsonable(
            {key: _policy_audit_summary(value) for key, value in benchmark_policy_by_id.items()}
        ),
        "meta_feature_profile": str(policy.static.static_meta_feature_profile),
        "meta_feature_policy": _jsonable(_meta_feature_policy_payload(policy)),
        "fixed_inner_optimizer": _ACTIVE_INNER_OPTIMIZER,
        "inner_optimizer_policy": _INNER_OPTIMIZER_POLICY_LABEL,
        "fixed_phase2_novelty_mode": _ACTIVE_PHASE2_NOVELTY_MODE,
        "static_route_id": normalize_static_route_id(policy.static.static_route_id, default=ROUTE_ID_A),
        "parallelism": {
            "benchmarks_per_trial_jobs": benchmarks_per_trial_jobs,
            "adapt_parallel_gradient_workers": int(policy.static.adapt_parallel_gradient_workers),
            "adapt_beam_parent_workers": int(policy.static.adapt_beam_parent_workers),
            "deterministic_aggregation": "serial_order_for_jobs_1; thread completion collected by benchmark_id for jobs_gt_1",
        },
        "objective": {
            "schema": "phase3_effective_trial_objective_v1",
            "objective_mode": (
                _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING
                if discovery_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING
                else ("legacy_global_aggregate_static_score_v1" if str(mode) == "global" else "legacy_normalized_static_score_v1")
            ),
            "discovery_objective_mode": discovery_mode,
            "feasibility_metric": (
                _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC
                if discovery_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING
                else None
            ),
            "feasibility_threshold": None,
            "resource_score_source": (
                "first_crossing"
                if discovery_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING
                else "terminal_proxy_legacy"
            ),
            "terminal_proxy_behavior": (
                "inactive"
                if discovery_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING
                else "active_legacy_terminal_energy_reference_or_same_cutoff_proxy"
            ),
            "weights": None if effective_weights is None else _jsonable(asdict(effective_weights)),
            "config": _objective_config_payload(config, objective_weights),
            "objective_provenance": _jsonable(objective_provenance),
            "stored_trial_value": None,
            "score_components": None,
            "required_target_violations": None,
        },
        "warm_start_provenance": _jsonable(
            dict(warm_start_provenance)
            if isinstance(warm_start_provenance, Mapping)
            else _trial_warm_start_provenance_payload(
                sampled_params,
                enqueued_records=(),
                warm_start_skips=(),
                enqueue_default=False,
            )
        ),
        "hardware_noise_policy": _hardware_noise_policy_payload(policy, effective_weights, objective_provenance),
        "replay_status": {
            "status": "intent_pending_results",
            "validator": "validate_effective_trial_replay",
            "sidecar_hash_authority": "effective_trial_manifest_sha256 user_attr",
        },
    }


def _exact_reference_metrics_payload(
    result: BenchmarkResult,
    physical_target: Mapping[str, Any] | None,
    spec_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = dict(physical_target or {})
    plus_one = target.get("phonon_cutoff_eval_plus_one") if target.get("status") == "present" else None
    reference_cutoff = target.get("phonon_cutoff_eval_reference") if target.get("status") == "present" else None
    accuracy_gate_metric = str(target.get("accuracy_gate_metric") or _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC)
    work_cutoff = _work_phonon_cutoff_from_spec_payload(spec_payload)
    if plus_one is None and work_cutoff is not None:
        plus_one = int(work_cutoff + 1)
    if reference_cutoff is None:
        reference_cutoff = plus_one
    ref_cutoff_matches_plus_one = plus_one is not None and result.exact_reference_n_ph_max is not None and int(result.exact_reference_n_ph_max) == int(plus_one)
    ref_cutoff_matches_reference = (
        reference_cutoff is not None
        and result.exact_reference_n_ph_max is not None
        and int(result.exact_reference_n_ph_max) == int(reference_cutoff)
    )
    ref_has_energy = _as_float_or_none(result.exact_reference_energy) is not None
    ref_has_error = _as_float_or_none(result.abs_delta_e_reference) is not None
    ref_is_plus_one = bool(ref_cutoff_matches_plus_one and ref_has_energy and ref_has_error)
    ref_is_reference = bool(ref_cutoff_matches_reference and ref_has_energy and ref_has_error)
    if ref_is_plus_one:
        plus_one_status = "present"
    elif ref_cutoff_matches_plus_one:
        plus_one_status = "reference_cutoff_matches_but_energy_or_error_missing"
    else:
        plus_one_status = "audited_missing_or_reference_cutoff_not_plus_one"
    if ref_is_reference:
        reference_status = "present"
    elif ref_cutoff_matches_reference:
        reference_status = "reference_cutoff_matches_but_energy_or_error_missing"
    else:
        reference_status = "audited_missing_or_reference_cutoff_mismatch"
    return {
        "schema": "phase3_exact_reference_metrics_v1",
        "accuracy_gate_metric": accuracy_gate_metric,
        "energy_adapt_Nph": result.energy,
        "energy_exact_Nph": result.same_cutoff_exact_gs_energy,
        "energy_exact_Nph_plus_1": result.exact_reference_energy if ref_is_plus_one else None,
        "energy_exact_reference_cutoff": result.exact_reference_energy if ref_is_reference else None,
        "energy_exact_reference": result.exact_reference_energy,
        "exact_reference_n_ph_max": result.exact_reference_n_ph_max,
        "abs_error_same_cutoff": result.abs_delta_e_same_cutoff,
        "abs_error_Nph_plus_1": result.abs_delta_e_reference if ref_is_plus_one else None,
        "abs_error_reference_cutoff": result.abs_delta_e_reference if ref_is_reference else None,
        "abs_error_reference": result.abs_delta_e_reference,
        "primary_error": (
            result.abs_delta_e_same_cutoff
            if accuracy_gate_metric == _PAPER_I_PHYS_V1_SAME_CUTOFF_METRIC
            else (result.abs_delta_e_reference if ref_is_reference else None)
        ),
        "cutoff_abs_delta_e": result.cutoff_abs_delta_e,
        "same_cutoff_and_plus_one_distinct": bool(ref_is_plus_one),
        "phonon_cutoff_work_inferred": work_cutoff,
        "phonon_cutoff_eval_plus_one_inferred": plus_one,
        "phonon_cutoff_eval_reference": reference_cutoff,
        "plus_one_status": plus_one_status,
        "reference_cutoff_status": reference_status,
    }


def _load_result_payload_for_result(result: BenchmarkResult, result_payload: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    if isinstance(result_payload, Mapping):
        return _jsonable(dict(result_payload))
    if result.result_json in {None, ""}:
        return None
    path = Path(str(result.result_json))
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _jsonable(dict(payload)) if isinstance(payload, Mapping) else None


def _history_rows_from_result_payload(result_payload: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(result_payload, Mapping):
        return ()
    adapt = _adapt_vqe_section(result_payload)
    rows = _adapt_history_rows(adapt)
    if rows:
        return rows
    return _adapt_history_rows(result_payload)


def _paper_i_cutoff_diagnostics_payload(
    result: BenchmarkResult,
    physical_target: Mapping[str, Any] | None,
    exact_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = dict(physical_target or {})
    metrics = dict(exact_metrics or {})
    accuracy_gate_metric = str(target.get("accuracy_gate_metric") or _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC)
    status = (
        "present"
        if (
            target.get("status") == "present"
            and (
                (
                    accuracy_gate_metric == _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC
                    and metrics.get("plus_one_status") == "present"
                    and metrics.get("abs_error_Nph_plus_1") is not None
                )
                or (
                    accuracy_gate_metric == _PAPER_I_PHYS_V1_REFERENCE_CUTOFF_METRIC
                    and metrics.get("reference_cutoff_status") == "present"
                    and metrics.get("abs_error_reference_cutoff") is not None
                )
                or (
                    accuracy_gate_metric == _PAPER_I_PHYS_V1_SAME_CUTOFF_METRIC
                    and metrics.get("abs_error_same_cutoff") is not None
                )
            )
        )
        else "audited_missing"
    )
    return {
        "schema": "paper_i_cutoff_diagnostics_v1",
        "status": status,
        "accuracy_gate_metric": target.get("accuracy_gate_metric", _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC),
        "phonon_cutoff_work": target.get("phonon_cutoff_work"),
        "phonon_cutoff_eval_same": target.get("phonon_cutoff_eval_same"),
        "phonon_cutoff_eval_plus_one": target.get("phonon_cutoff_eval_plus_one"),
        "phonon_cutoff_eval_reference": target.get("phonon_cutoff_eval_reference"),
        "phonon_reference_policy": target.get("phonon_reference_policy"),
        "energy_adapt_Nph": metrics.get("energy_adapt_Nph", result.energy),
        "energy_exact_Nph": metrics.get("energy_exact_Nph", result.same_cutoff_exact_gs_energy),
        "energy_exact_Nph_plus_1": metrics.get("energy_exact_Nph_plus_1"),
        "energy_exact_reference_cutoff": metrics.get("energy_exact_reference_cutoff"),
        "exact_reference_n_ph_max": metrics.get("exact_reference_n_ph_max", result.exact_reference_n_ph_max),
        "abs_error_same_cutoff": metrics.get("abs_error_same_cutoff", result.abs_delta_e_same_cutoff),
        "abs_error_Nph_plus_1": metrics.get("abs_error_Nph_plus_1"),
        "abs_error_reference_cutoff": metrics.get("abs_error_reference_cutoff"),
        "primary_error": metrics.get("primary_error"),
        "abs_error_reference": metrics.get("abs_error_reference", result.abs_delta_e_reference),
        "cutoff_abs_delta_e": metrics.get("cutoff_abs_delta_e", result.cutoff_abs_delta_e),
        "same_cutoff_and_plus_one_distinct": bool(metrics.get("same_cutoff_and_plus_one_distinct", False)),
        "plus_one_status": metrics.get("plus_one_status"),
        "reference_cutoff_status": metrics.get("reference_cutoff_status"),
        "terminal_proxy_label": "terminal_reference_proxy_not_first_crossing",
    }


def _paper_i_history_row_metrics(
    row: Mapping[str, Any],
    *,
    same_cutoff_exact: float | None,
    reference_exact: float,
    accepted_step_index: int,
) -> dict[str, Any] | None:
    energy = _as_float_or_none(row.get("energy_after_opt"))
    if energy is None:
        energy = _as_float_or_none(row.get("energy"))
    if energy is None:
        return None
    same_error = _as_float_or_none(row.get("delta_abs_current"))
    if same_error is None and same_cutoff_exact is not None:
        same_error = abs(float(energy) - float(same_cutoff_exact))
    primary_error = abs(float(energy) - float(reference_exact))
    depth = _as_int_or_none(row.get("depth"))
    operator_count = _as_int_or_none(row.get("logical_num_parameters_after_opt"))
    parameter_count = _as_int_or_none(row.get("num_parameters_after_opt"))
    if parameter_count is None:
        parameter_count = operator_count
    k_tau = int(accepted_step_index)
    resource_score_source = "accepted_step_index_at_crossing"
    resource_score = float(k_tau)
    return {
        "energy": float(energy),
        "primary_error": float(primary_error),
        "same_cutoff_error": same_error,
        "k_tau": None if k_tau is None else int(k_tau),
        "operator_count": None if operator_count is None else int(operator_count),
        "parameter_count": None if parameter_count is None else int(parameter_count),
        "resource_score": resource_score,
        "resource_score_source": resource_score_source,
        "accepted_step_index": int(accepted_step_index),
    }


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


def _paper_i_first_hit(
    row_metrics: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    threshold: float,
    metric: str,
) -> dict[str, Any]:
    for position, metrics in row_metrics:
        value = _as_float_or_none(metrics.get(metric))
        if value is not None and float(value) <= float(threshold):
            return {
                "reached": True,
                "history_position": int(position),
                "threshold": float(threshold),
                "metric": str(metric),
                "value": float(value),
            }
    return {"reached": False, "threshold": float(threshold), "metric": str(metric), "value": None}


def compute_paper_i_first_crossing_from_payload(
    result_payload: Mapping[str, Any] | None,
    *,
    physical_target: Mapping[str, Any] | None,
    reference_energy: float | None,
    same_cutoff_exact_energy: float | None = None,
    terminal_primary_error: float | None = None,
    terminal_same_cutoff_error: float | None = None,
) -> dict[str, Any]:
    target = dict(physical_target or {})
    if target.get("status") != "present":
        return {
            "schema": "paper_i_first_crossing_v1",
            "status": "not_replayable_physical_target_missing",
            "reached": False,
            "reason": target.get("reason", "physical_target_manifest_missing"),
            "accuracy_gate_metric": target.get("accuracy_gate_metric", _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC),
        }
    tau_phys = _as_float_or_none(target.get("tau_phys"))
    tau_tight = _as_float_or_none(target.get("tau_tight"))
    ref_energy = _as_float_or_none(reference_energy)
    if tau_phys is None or ref_energy is None:
        return {
            "schema": "paper_i_first_crossing_v1",
            "status": "not_replayable_physical_target_missing",
            "reached": False,
            "reason": "missing_tau_phys_or_energy_exact_Nph_plus_1",
            "accuracy_gate_metric": target.get("accuracy_gate_metric", _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC),
            "tau_phys": tau_phys,
            "terminal_primary_error": terminal_primary_error,
            "terminal_same_cutoff_error": terminal_same_cutoff_error,
        }
    rows = _history_rows_from_result_payload(result_payload)
    if not rows:
        return {
            "schema": "paper_i_first_crossing_v1",
            "status": "no_history",
            "reached": False,
            "reason": "adapt_history_missing; no terminal same-cutoff fallback used",
            "accuracy_gate_metric": target.get("accuracy_gate_metric", _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC),
            "tau_phys": float(tau_phys),
            "tau_tight": tau_tight,
            "history_row_count": 0,
            "terminal_primary_error": terminal_primary_error,
            "terminal_same_cutoff_error": terminal_same_cutoff_error,
        }
    accepted_metrics: list[tuple[int, dict[str, Any]]] = []
    missing_energy_rows = 0
    skipped_unaccepted_rows = 0
    acceptance_reason_counts: dict[str, int] = {}
    committed_operator_count: int | None = None
    initial_operator_count: int | None = None
    committed_depth: int | None = None
    for idx, row in enumerate(rows, start=1):
        accepted, acceptance_reason = _paper_i_history_row_acceptance_status(
            row,
            committed_operator_count=committed_operator_count,
            initial_operator_count=initial_operator_count,
            committed_depth=committed_depth,
        )
        acceptance_reason_counts[acceptance_reason] = int(acceptance_reason_counts.get(acceptance_reason, 0)) + 1
        row_operator_count = _as_int_or_none(row.get("logical_num_parameters_after_opt"))
        if not accepted:
            if acceptance_reason == "preexisting_initial_operator_count" and row_operator_count is not None and initial_operator_count is None:
                initial_operator_count = int(row_operator_count)
            skipped_unaccepted_rows += 1
            continue
        metrics = _paper_i_history_row_metrics(
            row,
            same_cutoff_exact=same_cutoff_exact_energy,
            reference_exact=float(ref_energy),
            accepted_step_index=len(accepted_metrics) + 1,
        )
        if metrics is None:
            missing_energy_rows += 1
            continue
        accepted_metrics.append((idx, metrics))
        if row_operator_count is not None:
            committed_operator_count = int(row_operator_count)
        row_depth = _as_int_or_none(row.get("depth"))
        if row_depth is not None:
            committed_depth = int(row_depth)
    if not accepted_metrics and len(rows) == 1:
        first_row = rows[0]
        first_depth = _as_int_or_none(first_row.get("depth"))
        if first_depth is not None and int(first_depth) > 0 and first_row.get("energy_after_opt") is not None:
            metrics = _paper_i_history_row_metrics(
                first_row,
                same_cutoff_exact=same_cutoff_exact_energy,
                reference_exact=float(ref_energy),
                accepted_step_index=1,
            )
            if metrics is not None and _as_float_or_none(metrics.get("primary_error")) is not None:
                accepted_metrics.append((1, metrics))
                missing_energy_rows = 0
                skipped_unaccepted_rows = max(0, int(skipped_unaccepted_rows) - 1)
                acceptance_reason_counts["single_positive_depth_fallback"] = 1
    if not accepted_metrics:
        return {
            "schema": "paper_i_first_crossing_v1",
            "status": "audited_missing",
            "reached": False,
            "reason": "adapt_history_has_no_accepted_energy_rows",
            "accuracy_gate_metric": target.get("accuracy_gate_metric", _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC),
            "tau_phys": float(tau_phys),
            "tau_tight": tau_tight,
            "history_row_count": int(len(rows)),
            "missing_energy_rows": int(missing_energy_rows),
            "skipped_unaccepted_rows": int(skipped_unaccepted_rows),
            "acceptance_reason_counts": dict(sorted(acceptance_reason_counts.items())),
            "terminal_primary_error": terminal_primary_error,
            "terminal_same_cutoff_error": terminal_same_cutoff_error,
        }
    crossing_idx = None
    crossing_metrics: Mapping[str, Any] | None = None
    for idx, metrics in accepted_metrics:
        if float(metrics["primary_error"]) <= float(tau_phys):
            crossing_idx = int(idx)
            crossing_metrics = metrics
            break
    reached = crossing_metrics is not None
    terminal_stop_reason = _stop_reason_from_result_payload(result_payload)
    target_hit_classification = _payload_target_hit_classification(result_payload) or _target_hit_classification_payload(
        stop_reason=terminal_stop_reason,
        target_error=terminal_primary_error,
        target_threshold=float(tau_phys),
        source="paper_i_first_crossing",
        accepted_crossing_reached=bool(reached),
    )
    target_hit_success = bool(target_hit_classification.get("target_hit_success", False))
    support_thresholds = [float(x) for x in target.get("legacy_same_cutoff_thresholds", []) if _as_float_or_none(x) is not None]
    support = {
        "tau_tight_primary_hit": (
            _paper_i_first_hit(accepted_metrics, threshold=float(tau_tight), metric="primary_error")
            if tau_tight is not None
            else {"reached": False, "reason": "tau_tight_missing"}
        ),
        "legacy_same_cutoff_hits": [
            _paper_i_first_hit(accepted_metrics, threshold=threshold, metric="same_cutoff_error")
            for threshold in support_thresholds
        ],
        "support_label": "support_diagnostics_not_main_objective_gate",
    }
    payload: dict[str, Any] = {
        "schema": "paper_i_first_crossing_v1",
        "status": "reached" if target_hit_success else ("non_target_terminal" if reached else "not_reached"),
        "reached": bool(target_hit_success),
        "history_crossed_threshold": bool(reached),
        "target_hit_success": bool(target_hit_success),
        "target_hit_classification": dict(target_hit_classification),
        "terminal_stop_reason": terminal_stop_reason,
        "required_stop_reason": _BENCHMARK_TARGET_HIT_STOP_REASON,
        "accuracy_gate_metric": target.get("accuracy_gate_metric", _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC),
        "tau_phys": float(tau_phys),
        "tau_tight": tau_tight,
        "history_row_count": int(len(rows)),
        "accepted_history_row_count": int(len(accepted_metrics)),
        "missing_energy_rows": int(missing_energy_rows),
        "skipped_unaccepted_rows": int(skipped_unaccepted_rows),
        "acceptance_reason_counts": dict(sorted(acceptance_reason_counts.items())),
        "terminal_primary_error": terminal_primary_error,
        "terminal_same_cutoff_error": terminal_same_cutoff_error,
        "support_diagnostics": support,
    }
    if not reached or crossing_metrics is None:
        payload["reason"] = "physical_tau_not_crossed_in_adapt_history"
        return _jsonable(payload)
    if not target_hit_success:
        payload["reason"] = target_hit_classification.get(
            "non_hit_reason", "physical_tau_crossed_but_terminal_stop_was_not_benchmark_target"
        )
    payload.update(
        {
            "k_tau": crossing_metrics.get("k_tau"),
            "history_position_tau": crossing_idx,
            "primary_error_at_crossing": crossing_metrics.get("primary_error"),
            "same_cutoff_error_at_crossing": crossing_metrics.get("same_cutoff_error"),
            "energy_at_crossing": crossing_metrics.get("energy"),
            "operator_count_at_crossing": crossing_metrics.get("operator_count"),
            "parameter_count_at_crossing": crossing_metrics.get("parameter_count"),
            "resource_score": crossing_metrics.get("resource_score"),
            "resource_score_source": crossing_metrics.get("resource_score_source"),
            "resource_proxies_at_crossing": {
                "operator_count": crossing_metrics.get("operator_count"),
                "parameter_count": crossing_metrics.get("parameter_count"),
                "compiled_two_qubit_count": _audited_missing("per_step_compiled_two_qubit_count_not_available", required_for_cthc=False),
                "compiled_depth": _audited_missing("per_step_compiled_depth_not_available", required_for_cthc=False),
            },
        }
    )
    return _jsonable(payload)


def _paper_i_artifacts_for_result(
    result: BenchmarkResult,
    spec: HamiltonianBenchmarkSpec,
    *,
    result_payload: Mapping[str, Any] | None = None,
    physical_target: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target = dict(physical_target or _paper_i_phys_v1_target_or_missing(spec))
    if isinstance(result.physical_target_manifest, Mapping):
        target = dict(result.physical_target_manifest)
    spec_payload = asdict(spec)
    exact_metrics = _exact_reference_metrics_payload(result, target, spec_payload)
    cutoff = dict(result.cutoff_diagnostics) if isinstance(result.cutoff_diagnostics, Mapping) else _paper_i_cutoff_diagnostics_payload(result, target, exact_metrics)
    payload = _load_result_payload_for_result(result, result_payload)
    first = (
        dict(result.paper_i_first_crossing)
        if isinstance(result.paper_i_first_crossing, Mapping)
        else compute_paper_i_first_crossing_from_payload(
            payload,
            physical_target=target,
            reference_energy=(
                exact_metrics.get("energy_exact_Nph")
                if target.get("accuracy_gate_metric") == _PAPER_I_PHYS_V1_SAME_CUTOFF_METRIC
                else exact_metrics.get("energy_exact_reference_cutoff")
            ),
            same_cutoff_exact_energy=exact_metrics.get("energy_exact_Nph"),
            terminal_primary_error=exact_metrics.get("primary_error"),
            terminal_same_cutoff_error=exact_metrics.get("abs_error_same_cutoff"),
        )
    )
    if isinstance(first, dict) and "benchmark_id" not in first:
        first["benchmark_id"] = str(spec.benchmark_id)
    if isinstance(first, dict):
        target_hit_classification = _payload_target_hit_classification(payload) or target_hit_classification_for_result(
            result,
            target_abs_delta_e=_as_float_or_none(first.get("tau_phys")),
            first_crossing=first,
            source="paper_i_artifacts_for_result",
        )
        first["target_hit_classification"] = _jsonable(target_hit_classification)
        first["target_hit_success"] = bool(
            target_hit_classification.get("target_hit_success", False)
        )
        first["terminal_stop_reason"] = result.stop_reason
        first["required_stop_reason"] = _BENCHMARK_TARGET_HIT_STOP_REASON
        if bool(first.get("reached", False)) and not bool(
            target_hit_classification.get("target_hit_success", False)
        ):
            first["history_crossed_threshold"] = True
            first["reached"] = False
            first["status"] = "non_target_terminal"
            first["reason"] = target_hit_classification.get(
                "non_hit_reason", "physical_tau_crossed_but_terminal_stop_was_not_benchmark_target"
            )
    return _jsonable(target), _jsonable(cutoff), _jsonable(first)


def _ensure_paper_i_result_artifacts(
    result: BenchmarkResult,
    spec: HamiltonianBenchmarkSpec,
    *,
    result_payload: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    if (
        isinstance(result.physical_target_manifest, Mapping)
        and isinstance(result.cutoff_diagnostics, Mapping)
        and isinstance(result.paper_i_first_crossing, Mapping)
    ):
        first = dict(result.paper_i_first_crossing)
        if "benchmark_id" in first:
            return result
        first["benchmark_id"] = str(spec.benchmark_id)
        return replace(result, paper_i_first_crossing=_jsonable(first))
    physical_target_manifest, cutoff_diagnostics, paper_i_first_crossing = _paper_i_artifacts_for_result(
        result,
        spec,
        result_payload=result_payload,
    )
    return replace(
        result,
        physical_target_manifest=physical_target_manifest,
        cutoff_diagnostics=cutoff_diagnostics,
        paper_i_first_crossing=paper_i_first_crossing,
    )


def _write_paper_i_artifacts_into_result_json(
    path: str | Path | None,
    *,
    result_payload: Mapping[str, Any] | None,
    physical_target_manifest: Mapping[str, Any],
    cutoff_diagnostics: Mapping[str, Any],
    paper_i_first_crossing: Mapping[str, Any],
    target_hit_classification: Mapping[str, Any] | None = None,
) -> None:
    if path in {None, ""}:
        return
    result_path = Path(str(path))
    if not result_path.exists():
        return
    payload = _load_result_payload_for_result(
        BenchmarkResult(benchmark_id="_artifact", family="_artifact", success=False, abs_delta_e=None, result_json=str(result_path)),
        result_payload,
    )
    if not isinstance(payload, Mapping):
        return
    out = dict(payload)
    out["physical_target_manifest"] = _jsonable(dict(physical_target_manifest))
    out["cutoff_diagnostics"] = _jsonable(dict(cutoff_diagnostics))
    out["paper_i_first_crossing"] = _jsonable(dict(paper_i_first_crossing))
    if isinstance(target_hit_classification, Mapping):
        out["target_hit_classification"] = _jsonable(dict(target_hit_classification))
        out["benchmark_target_classification"] = _jsonable(dict(target_hit_classification))
        out["benchmark_target_hit_success"] = bool(
            target_hit_classification.get("target_hit_success", False)
        )
        out["benchmark_target_non_hit_reason"] = target_hit_classification.get("non_hit_reason")
    _write_json(result_path, out)


def finalize_effective_trial_manifest(
    manifest: Mapping[str, Any],
    *,
    results: Mapping[str, BenchmarkResult] | None = None,
    score: float | None = None,
    global_score_components: Mapping[str, Any] | None = None,
    required_target_violations_payload: Mapping[str, Any] | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    payload = _jsonable(dict(manifest))
    payload["generated_utc"] = _now_utc()
    payload["lifecycle_status"] = "failed" if failure_reason else "finalized"
    if failure_reason:
        payload["failure_reason"] = str(failure_reason)
    result_map = dict(results or {})
    finalized_children: list[dict[str, Any]] = []
    for child in payload.get("benchmark_manifests", []):
        child_payload = dict(child)
        benchmark_id = str(child_payload.get("benchmark_id"))
        result = result_map.get(benchmark_id)
        if result is None:
            child_payload["lifecycle_status"] = "missing_result"
            child_payload["result"] = None
            child_payload["exact_reference_metrics"] = _audited_missing("missing_benchmark_result", required_for_cthc=True)
            child_payload["physical_target_manifest"] = child_payload.get("physical_target")
            child_payload["cutoff_diagnostics"] = _audited_missing("missing_benchmark_result", required_for_cthc=True)
            child_payload["paper_i_first_crossing"] = _audited_missing("missing_benchmark_result", required_for_cthc=True)
            child_payload["target_hit_classification"] = _target_hit_classification_payload(
                stop_reason=None,
                target_error=None,
                target_threshold=None,
                source="effective_trial_manifest_missing_child",
                accepted_crossing_reached=False,
            )
            child_payload["target_hit_success"] = False
            child_payload["target_non_hit_reason"] = "missing_benchmark_result"
        else:
            child_payload["lifecycle_status"] = "complete" if bool(result.success) else "failed"
            target_payload = child_payload.get("physical_target") if isinstance(child_payload.get("physical_target"), Mapping) else None
            child_payload["result"] = _jsonable(asdict(result))
            child_payload["exact_reference_metrics"] = _exact_reference_metrics_payload(
                result,
                target_payload,
                child_payload.get("spec") if isinstance(child_payload.get("spec"), Mapping) else None,
            )
            try:
                spec_for_child = _spec_from_payload(child_payload.get("spec") if isinstance(child_payload.get("spec"), Mapping) else {})
                physical_target_manifest, cutoff_diagnostics, paper_i_first_crossing = _paper_i_artifacts_for_result(
                    result,
                    spec_for_child,
                    physical_target=target_payload,
                )
            except Exception as exc:
                physical_target_manifest = dict(target_payload or _audited_missing(f"physical_target_manifest_failed:{type(exc).__name__}:{exc}", required_for_cthc=False))
                cutoff_diagnostics = _audited_missing(f"cutoff_diagnostics_failed:{type(exc).__name__}:{exc}", required_for_cthc=False)
                paper_i_first_crossing = _audited_missing(f"paper_i_first_crossing_failed:{type(exc).__name__}:{exc}", required_for_cthc=False)
            child_payload["physical_target_manifest"] = _jsonable(physical_target_manifest)
            child_payload["cutoff_diagnostics"] = _jsonable(cutoff_diagnostics)
            child_payload["paper_i_first_crossing"] = _jsonable(paper_i_first_crossing)
            target_hit_classification = (
                dict(result.target_hit_classification)
                if isinstance(result.target_hit_classification, Mapping)
                else target_hit_classification_for_result(
                    result,
                    first_crossing=paper_i_first_crossing,
                    source="effective_trial_manifest_child",
                )
            )
            child_payload["target_hit_classification"] = _jsonable(target_hit_classification)
            child_payload["target_hit_success"] = bool(
                target_hit_classification.get("target_hit_success", False)
            )
            child_payload["target_non_hit_reason"] = target_hit_classification.get("non_hit_reason")
            child_payload["policy_roundtrip_audit_summary"] = _summarize_policy_roundtrip_audits(
                (result.policy_roundtrip_audit,) if isinstance(result.policy_roundtrip_audit, Mapping) else ()
            )
        child_payload["artifact_hashes"] = _artifact_digest_payload(
            child_payload.get("artifact_paths") if isinstance(child_payload.get("artifact_paths"), Mapping) else {}
        )
        finalized_children.append(child_payload)
    payload["benchmark_manifests"] = finalized_children
    objective = dict(payload.get("objective", {}))
    objective["stored_trial_value"] = None if score is None else float(score)
    if global_score_components is not None:
        objective["score_components"] = _jsonable(dict(global_score_components))
    elif score is not None:
        objective["score_components"] = {"score": float(score)}
    if objective.get("discovery_objective_mode") == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING:
        thresholds = sorted(
            {
                float(child["paper_i_first_crossing"]["tau_phys"])
                for child in finalized_children
                if isinstance(child.get("paper_i_first_crossing"), Mapping)
                and _as_float_or_none(child["paper_i_first_crossing"].get("tau_phys")) is not None
            }
        )
        metrics = sorted(
            {
                str(child["paper_i_first_crossing"].get("accuracy_gate_metric"))
                for child in finalized_children
                if isinstance(child.get("paper_i_first_crossing"), Mapping)
                and str(child["paper_i_first_crossing"].get("accuracy_gate_metric") or "").strip()
            }
        )
        objective["feasibility_metric"] = metrics[0] if len(metrics) == 1 else metrics
        objective["feasibility_threshold"] = thresholds[0] if len(thresholds) == 1 else thresholds
        objective["resource_score_source"] = "first_crossing"
    if required_target_violations_payload is not None:
        objective["required_target_violations"] = _jsonable(dict(required_target_violations_payload))
    payload["objective"] = objective
    payload["replay_status"] = {
        "status": "not_validated",
        "validator": "validate_effective_trial_replay",
        "sidecar_hash_authority": "effective_trial_manifest_sha256 user_attr",
    }
    return payload


def _dataclass_kwargs(cls: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field_info.name for field_info in fields(cls)}
    return {key: value for key, value in dict(payload).items() if key in allowed}


def _problem_features_from_payload(payload: Mapping[str, Any]) -> ProblemFeatureVector:
    return ProblemFeatureVector(**_dataclass_kwargs(ProblemFeatureVector, payload))


def _spec_from_payload(payload: Mapping[str, Any]) -> HamiltonianBenchmarkSpec:
    data = dict(payload)
    features_payload = data.get("features")
    if not isinstance(features_payload, Mapping):
        raise ValueError("manifest_spec_missing_features")
    data["features"] = _problem_features_from_payload(features_payload)
    if "base_pipeline_args" in data:
        data["base_pipeline_args"] = tuple(str(x) for x in data["base_pipeline_args"])
    if "tags" in data:
        data["tags"] = tuple(str(x) for x in data["tags"])
    return HamiltonianBenchmarkSpec(**_dataclass_kwargs(HamiltonianBenchmarkSpec, data))


def _policy_from_payload(payload: Mapping[str, Any]) -> AlgorithmPolicy:
    data = dict(payload)
    pool_payload = dict(data.get("pool", {}) if isinstance(data.get("pool"), Mapping) else {})
    for budget_key in ("phase1_budget", "phase2_budget"):
        if isinstance(pool_payload.get(budget_key), Mapping):
            pool_payload[budget_key] = SizeScaledBudget(**_dataclass_kwargs(SizeScaledBudget, pool_payload[budget_key]))
    return AlgorithmPolicy(
        pool=PoolPolicy(**_dataclass_kwargs(PoolPolicy, pool_payload)),
        static=StaticScaffoldPolicy(**_dataclass_kwargs(StaticScaffoldPolicy, data.get("static", {}) if isinstance(data.get("static"), Mapping) else {})),
        inner_optimizer=InnerOptimizerPolicy(**_dataclass_kwargs(InnerOptimizerPolicy, data.get("inner_optimizer", {}) if isinstance(data.get("inner_optimizer"), Mapping) else {})),
    )


def _result_from_payload(payload: Mapping[str, Any]) -> BenchmarkResult:
    return BenchmarkResult(**_dataclass_kwargs(BenchmarkResult, payload))


def _objective_weights_from_payload(payload: Mapping[str, Any] | None) -> StaticObjectiveWeights:
    if not isinstance(payload, Mapping):
        return StaticObjectiveWeights()
    return StaticObjectiveWeights(**_dataclass_kwargs(StaticObjectiveWeights, payload))


def _global_config_from_payload(payload: Mapping[str, Any] | None) -> GlobalObjectiveConfig:
    if not isinstance(payload, Mapping):
        return GlobalObjectiveConfig()
    data = dict(payload)
    data["weights"] = _objective_weights_from_payload(data.get("weights") if isinstance(data.get("weights"), Mapping) else None)
    if "required_target_benchmark_ids" in data:
        data["required_target_benchmark_ids"] = tuple(str(x) for x in data["required_target_benchmark_ids"])
    return GlobalObjectiveConfig(**_dataclass_kwargs(GlobalObjectiveConfig, data))


def _load_effective_trial_manifest(manifest_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest_or_path, Mapping):
        return _jsonable(dict(manifest_or_path))
    path = Path(manifest_or_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Effective trial manifest is not a JSON object: {path}")
    return _jsonable(dict(payload))


def _normalize_command_for_replay(tokens: Sequence[Any]) -> list[str]:
    path_value_flags = {
        "--output-json",
        "--adapt-selected-logical-source-json",
        "--hardware-resolution-profile-json",
    }
    normalized: list[str] = []
    previous = None
    for token in tokens:
        text = str(token)
        if previous in path_value_flags:
            try:
                text = str(Path(text).expanduser().resolve())
            except Exception:
                pass
        normalized.append(text)
        previous = text if text.startswith("--") else None
    return normalized


def _read_saved_command_log(path: str | Path | None) -> list[str] | None:
    if path in {None, ""}:
        return None
    command_path = Path(str(path))
    if not command_path.exists():
        return None
    lines = []
    for raw_line in command_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#!") or line == "set -euo pipefail":
            continue
        lines.append(line)
    if not lines:
        return None
    try:
        return [str(token) for token in shlex.split(" ".join(lines))]
    except Exception:
        return None


def _manifest_row(family: str, passed: bool, *, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    row = {"family": str(family), "passed": bool(passed), "status": "pass" if passed else "fail"}
    if reason not in {None, ""}:
        row["reason"] = str(reason)
    row.update(_jsonable(extra))
    return row


def validate_effective_trial_replay(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    trial_value: float | None = None,
    write_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate first-slice replayability and current-objective recomputation."""

    manifest = _load_effective_trial_manifest(manifest_or_path)
    rows: list[dict[str, Any]] = []
    if manifest.get("schema") != _EFFECTIVE_TRIAL_MANIFEST_SCHEMA:
        rows.append(_manifest_row("identity", False, reason="unsupported_manifest_schema", schema=manifest.get("schema")))
    else:
        rows.append(_manifest_row("identity", True, schema=manifest.get("schema"), mode=manifest.get("mode")))

    policy_payload = manifest.get("policy") if isinstance(manifest.get("policy"), Mapping) else {}
    try:
        policy = _policy_from_payload(policy_payload)
        rows.append(_manifest_row("route", normalize_static_route_id(policy.static.static_route_id, default=ROUTE_ID_A) == manifest.get("static_route_id")))
        rows.append(_manifest_row("SPSA", str(policy.inner_optimizer.inner_optimizer).upper() == "SPSA"))
    except Exception as exc:
        policy = None
        rows.append(_manifest_row("route", False, reason=f"policy_reconstruction_failed:{type(exc).__name__}:{exc}"))
        rows.append(_manifest_row("SPSA", False, reason="policy_reconstruction_failed"))

    command_failures: list[str] = []
    artifact_failures: list[str] = []
    artifact_hash_failures: list[str] = []
    artifact_value_failures: list[str] = []
    physical_target_failures: list[str] = []
    result_map: dict[str, BenchmarkResult] = {}
    spec_map: dict[str, HamiltonianBenchmarkSpec] = {}
    artifact_exact_metrics_map: dict[str, dict[str, Any]] = {}
    policy_mismatch_count = 0
    children = [child for child in manifest.get("benchmark_manifests", []) if isinstance(child, Mapping)]
    objective_payload = manifest.get("objective") if isinstance(manifest.get("objective"), Mapping) else {}
    objective_mode_requested = str(
        objective_payload.get("discovery_objective_mode")
        or objective_payload.get("objective_mode")
        or ""
    )
    physical_target_required = (
        str(manifest.get("required_target_profile", "none") or "none") == _PAPER_I_PHYS_V1_PROFILE
        or objective_mode_requested == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING
    )
    physical_target_seen = False
    finalized_manifest = str(manifest.get("lifecycle_status")) in {"finalized", "failed"}
    for child in children:
        benchmark_id = str(child.get("benchmark_id"))
        spec_payload = child.get("spec")
        try:
            if not isinstance(spec_payload, Mapping):
                raise ValueError("missing_spec")
            spec = _spec_from_payload(spec_payload)
            spec_map[benchmark_id] = spec
            artifacts = child.get("artifact_paths") if isinstance(child.get("artifact_paths"), Mapping) else {}
            if policy is not None:
                saved_command = _read_saved_command_log(artifacts.get("command_sh"))
                manifest_command = [str(token) for token in child.get("emitted_cli_args", [])]
                fallback = Path(str(manifest["trial_identity"]["trial_dir"])) / benchmark_id / "json" / "result.json"
                output_json = Path(str(artifacts.get("result_json", fallback)))
                python_bin = (saved_command or manifest_command or [sys.executable])[0]
                expected = build_static_command(
                    python_bin=python_bin,
                    spec=spec,
                    policy=policy,
                    output_json=output_json,
                    benchmark_target_abs_delta_e=child.get("benchmark_target_abs_delta_e"),
                    benchmark_target_reference_energy=child.get("benchmark_target_reference_energy"),
                )
                if saved_command is None:
                    command_failures.append(f"{benchmark_id}:missing_command_sh")
                elif _normalize_command_for_replay(saved_command) != _normalize_command_for_replay(expected):
                    command_failures.append(benchmark_id)
                route_identity = child.get("route_identity") if isinstance(child.get("route_identity"), Mapping) else {}
                emitted_route = _pipeline_arg_value(manifest_command, "--static-route-id")
                if emitted_route is None and saved_command is not None:
                    emitted_route = _pipeline_arg_value(saved_command, "--static-route-id")
                if route_identity and emitted_route is not None:
                    normalized_emitted = normalize_static_route_id(emitted_route, default=ROUTE_ID_UNSPECIFIED)
                    if route_identity.get("emitted_static_route_id") != normalized_emitted:
                        command_failures.append(f"{benchmark_id}:route_identity_emitted_static_route_id")
            result_payload = child.get("result")
            if isinstance(result_payload, Mapping):
                manifest_result = _result_from_payload(result_payload)
                result_map[benchmark_id] = manifest_result
                if isinstance(artifacts, Mapping):
                    value_mismatches = _artifact_result_metric_mismatches(manifest_result, artifacts.get("result_json"))
                    artifact_value_failures.extend(f"{benchmark_id}:{item}" for item in value_mismatches)
                    artifact_result, artifact_payload, artifact_input_failures = _artifact_result_from_files(
                        manifest_result,
                        spec,
                        artifacts,
                    )
                    if bool(manifest_result.success):
                        artifact_failures.extend(f"{benchmark_id}:{item}" for item in artifact_input_failures)
                    if artifact_result is not None:
                        result_map[benchmark_id] = artifact_result
                        artifact_exact_metrics_map[benchmark_id] = _exact_reference_metrics_payload(
                            artifact_result,
                            child.get("physical_target") if isinstance(child.get("physical_target"), Mapping) else None,
                            spec_payload if isinstance(spec_payload, Mapping) else None,
                        )
                        objective_input_mismatches = _paper_i_artifact_objective_input_mismatches(
                            child=child,
                            spec=spec,
                            manifest_result=manifest_result,
                            artifact_result=artifact_result,
                            result_payload=artifact_payload,
                        )
                        artifact_value_failures.extend(
                            f"{benchmark_id}:objective_input.{item}" for item in objective_input_mismatches
                        )
            hash_mismatches = _artifact_hash_mismatches(child.get("artifact_hashes") if isinstance(child.get("artifact_hashes"), Mapping) else None)
            artifact_hash_failures.extend(f"{benchmark_id}:{item}" for item in hash_mismatches)
            target_payload = child.get("physical_target")
            if isinstance(target_payload, Mapping) and target_payload.get("status") == "present":
                physical_target_seen = True
                if physical_target_required:
                    physical_target_failures.extend(
                        f"{benchmark_id}:{item}"
                        for item in _paper_i_physical_target_mismatches(spec, target_payload, label="physical_target")
                    )
                    manifest_target_payload = child.get("physical_target_manifest")
                    if isinstance(manifest_target_payload, Mapping):
                        physical_target_failures.extend(
                            f"{benchmark_id}:{item}"
                            for item in _paper_i_physical_target_mismatches(spec, manifest_target_payload, label="physical_target_manifest")
                        )
                metrics_payload = artifact_exact_metrics_map.get(benchmark_id)
                if metrics_payload is None:
                    metrics_payload = child.get("exact_reference_metrics")
                accuracy_gate_metric = str(target_payload.get("accuracy_gate_metric") or "")
                if accuracy_gate_metric == _PAPER_I_PHYS_V1_REFERENCE_CUTOFF_METRIC:
                    if not (
                        isinstance(metrics_payload, Mapping)
                        and metrics_payload.get("reference_cutoff_status") == "present"
                        and metrics_payload.get("abs_error_reference_cutoff") is not None
                    ):
                        physical_target_failures.append(f"{benchmark_id}:missing_abs_error_reference_cutoff")
                elif accuracy_gate_metric == _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC:
                    if not (
                        isinstance(metrics_payload, Mapping)
                        and metrics_payload.get("plus_one_status") == "present"
                        and metrics_payload.get("abs_error_Nph_plus_1") is not None
                    ):
                        physical_target_failures.append(f"{benchmark_id}:missing_abs_error_Nph_plus_1")
                if physical_target_required:
                    required_metric_keys = ["energy_exact_Nph", "abs_error_same_cutoff"]
                    if accuracy_gate_metric == _PAPER_I_PHYS_V1_REFERENCE_CUTOFF_METRIC:
                        required_metric_keys.extend(("energy_exact_reference_cutoff", "abs_error_reference_cutoff"))
                    elif accuracy_gate_metric == _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC:
                        required_metric_keys.extend(("energy_exact_Nph_plus_1", "abs_error_Nph_plus_1"))
                    for metric_key in required_metric_keys:
                        if not isinstance(metrics_payload, Mapping) or metrics_payload.get(metric_key) is None:
                            physical_target_failures.append(f"{benchmark_id}:missing_{metric_key}")
                    if (
                        accuracy_gate_metric == _PAPER_I_PHYS_V1_N_PH_PLUS_ONE_METRIC
                        and isinstance(metrics_payload, Mapping)
                        and not bool(metrics_payload.get("same_cutoff_and_plus_one_distinct", False))
                    ):
                        physical_target_failures.append(f"{benchmark_id}:same_cutoff_and_plus_one_not_distinct")
            elif physical_target_required:
                physical_target_failures.append(f"{benchmark_id}:missing_paper_i_phys_v1_target")
            if not isinstance(artifacts, Mapping) or not artifacts.get("result_json") or not artifacts.get("command_sh"):
                artifact_failures.append(f"{benchmark_id}:missing_artifact_paths")
            elif finalized_manifest:
                required_artifact_keys = ["command_sh", "result_json"]
                child_result = result_map.get(benchmark_id)
                if child_result is not None and bool(child_result.success):
                    required_artifact_keys.append("compile_json")
                for artifact_key in tuple(required_artifact_keys):
                    artifact_path = artifacts.get(artifact_key)
                    if artifact_path in {None, ""} or not Path(str(artifact_path)).exists():
                        artifact_failures.append(f"{benchmark_id}:missing_{artifact_key}")
            audit_summary = child.get("policy_roundtrip_audit_summary")
            if isinstance(audit_summary, Mapping):
                policy_mismatch_count += int(dict(audit_summary.get("status_counts", {})).get("mismatch", 0) or 0)
        except Exception as exc:
            command_failures.append(f"{benchmark_id}:{type(exc).__name__}:{exc}")
    rows.append(_manifest_row("CLI", not command_failures, failures=command_failures))
    rows.append(_manifest_row("policy", policy_mismatch_count == 0, mismatch_count=policy_mismatch_count))
    rows.append(_manifest_row("artifacts", not artifact_failures and bool(children), failures=artifact_failures))
    rows.append(_manifest_row("artifact_hashes", not artifact_hash_failures, failures=artifact_hash_failures))
    rows.append(_manifest_row("artifact_values", not artifact_value_failures, failures=artifact_value_failures))
    physical_target_ok = not physical_target_failures and (physical_target_seen or not physical_target_required) and bool(children)
    rows.append(
        _manifest_row(
            "physical_targets",
            physical_target_ok,
            failures=physical_target_failures,
            required=physical_target_required,
            present=physical_target_seen,
        )
    )

    hardware_noise = manifest.get("hardware_noise_policy")
    inactive_reasons: list[str] = []
    inactive_ok = isinstance(hardware_noise, Mapping) and hardware_noise.get("noise_model") == "inactive" and hardware_noise.get("mitigation") == "inactive"
    if not inactive_ok:
        inactive_reasons.append("hardware_noise_or_mitigation_active")
    objective_provenance = objective_payload.get("objective_provenance") if isinstance(objective_payload.get("objective_provenance"), Mapping) else {}
    paper_i_noiseless_required = bool(
        physical_target_required
        or str(manifest.get("suite_profile", "")).startswith("paper_i")
    )
    if paper_i_noiseless_required:
        if isinstance(hardware_noise, Mapping):
            if hardware_noise.get("objective_noise_mode") != "exact_noiseless_v1":
                inactive_reasons.append("hardware_noise_policy_objective_noise_mode_not_exact_noiseless")
            if bool(hardware_noise.get("objective_consumes_noisy_energy", False)):
                inactive_reasons.append("hardware_noise_policy_consumes_noisy_energy")
        if isinstance(objective_provenance, Mapping):
            if objective_provenance.get("objective_noise_mode") != "exact_noiseless_v1":
                inactive_reasons.append("objective_provenance_noise_mode_not_exact_noiseless")
            if bool(objective_provenance.get("objective_consumes_noisy_energy", False)):
                inactive_reasons.append("objective_provenance_consumes_noisy_energy")
            if str(objective_provenance.get("phase3_oracle_inner_objective_mode", "exact")) != "exact":
                inactive_reasons.append("objective_provenance_inner_objective_not_exact")
            if str(objective_provenance.get("phase3_oracle_value_noise_model", "off")) != "off":
                inactive_reasons.append("objective_provenance_value_noise_not_off")
            if objective_provenance.get("phase3_oracle_execution_surface") not in {None, ""}:
                inactive_reasons.append("objective_provenance_execution_surface_set")
    rows.append(
        _manifest_row(
            "inactive_hardware_noise",
            inactive_ok and not inactive_reasons,
            failures=inactive_reasons,
            payload=hardware_noise if isinstance(hardware_noise, Mapping) else None,
            objective_provenance=objective_provenance if isinstance(objective_provenance, Mapping) else None,
        )
    )

    recomputed_value = None
    objective_reason = None
    try:
        objective = manifest.get("objective") if isinstance(manifest.get("objective"), Mapping) else {}
        stored_value = objective.get("stored_trial_value")
        mode = str(manifest.get("mode"))
        objective_mode = str(objective.get("discovery_objective_mode") or objective.get("objective_mode") or "")
        if mode == "oracle":
            if len(children) != 1:
                raise ValueError("oracle_manifest_requires_one_child")
            child = children[0]
            benchmark_id = str(child.get("benchmark_id"))
            spec = spec_map[benchmark_id]
            result = result_map[benchmark_id]
            weights = _objective_weights_from_payload(objective.get("weights") if isinstance(objective.get("weights"), Mapping) else None)
            if objective_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING:
                config = _global_config_from_payload(objective.get("config") if isinstance(objective.get("config"), Mapping) else None)
                if _normalize_discovery_objective_mode(config.discovery_objective_mode) != _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING:
                    config = replace(config, discovery_objective_mode=_DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING, weights=weights)
                recomputed_value = discovery_first_crossing_score(result, spec, config)
            else:
                recomputed_value = normalized_static_score(result, spec, weights)
        elif mode == "global":
            config = _global_config_from_payload(objective.get("config") if isinstance(objective.get("config"), Mapping) else None)
            if objective_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING and _normalize_discovery_objective_mode(config.discovery_objective_mode) != _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING:
                config = replace(config, discovery_objective_mode=_DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING)
            ordered_specs = [spec_map[str(child.get("benchmark_id"))] for child in children]
            recomputed_value = aggregate_global_score_components(result_map, ordered_specs, config)["score"]
        else:
            raise ValueError(f"unsupported_mode:{mode}")
        expected_value = trial_value if trial_value is not None else stored_value
        objective_ok = expected_value is not None and math.isclose(float(recomputed_value), float(expected_value), rel_tol=1e-12, abs_tol=1e-12)
        if not objective_ok:
            objective_reason = "objective_value_mismatch"
    except Exception as exc:
        objective_ok = False
        objective_reason = f"objective_recompute_failed:{type(exc).__name__}:{exc}"
    expected_value_payload = trial_value
    if expected_value_payload is None and isinstance(manifest.get("objective"), Mapping):
        expected_value_payload = manifest["objective"].get("stored_trial_value")
    rows.append(
        _manifest_row(
            "objective",
            objective_ok,
            reason=objective_reason,
            recomputed_value=recomputed_value,
            expected_value=expected_value_payload,
        )
    )

    cthc_blockers = _collect_cthc_required_missing(manifest)
    local_replay_pass = all(bool(row.get("passed")) for row in rows)
    cthc_ready = bool(local_replay_pass and not cthc_blockers)
    status = (
        "pass"
        if cthc_ready
        else ("local_replay_pass_cthc_blocked" if local_replay_pass else "fail")
    )
    audit = {
        "schema": _EFFECTIVE_TRIAL_REPLAY_AUDIT_SCHEMA,
        "generated_utc": _now_utc(),
        "manifest_schema": manifest.get("schema"),
        "manifest_path": str(manifest_or_path) if not isinstance(manifest_or_path, Mapping) else None,
        "status": status,
        "local_replay_pass": bool(local_replay_pass),
        "cthc_ready": cthc_ready,
        "cthc_blockers": cthc_blockers,
        "chtc_ready": cthc_ready,
        "chtc_blockers": cthc_blockers,
        "recomputed_value": recomputed_value,
        "rows": rows + [_manifest_row("cthc_required_fields", not cthc_blockers, failures=cthc_blockers)],
    }
    if write_path is not None:
        _write_json(Path(write_path), audit)
    return _jsonable(audit)


def _option_values_equivalent(left: Any, right: Any) -> bool:
    if left in {None, ""} or right in {None, ""}:
        return True
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    except Exception:
        return str(left).strip() == str(right).strip()


_MISSING = object()


def _compact_mapping(mapping: Mapping[str, Any] | None, *, limit: int = 128) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        return {}
    items = sorted((str(k), v) for k, v in mapping.items())
    out = {key: _jsonable(value) for key, value in items[: int(limit)]}
    if len(items) > int(limit):
        out["_truncated_key_count"] = int(len(items) - int(limit))
    return out


def _path_get(root: Any, path: str) -> Any:
    node = root
    for part in str(path).split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _adapt_runtime_payload(result_payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(result_payload, Mapping):
        return {}
    adapt = _adapt_vqe_section(result_payload)
    if adapt:
        return adapt
    return result_payload


def _runtime_path_get(result_payload: Mapping[str, Any] | None, path: str | None) -> Any:
    if not path:
        return _MISSING
    if not isinstance(result_payload, Mapping):
        return _MISSING
    direct = _path_get(result_payload, path)
    if direct is not _MISSING:
        return direct
    prefix = "adapt_vqe."
    if str(path).startswith(prefix):
        return _path_get(_adapt_runtime_payload(result_payload), str(path)[len(prefix) :])
    return _MISSING


def _strip_controller_summary(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        return {}
    keep = (
        "schema",
        "source",
        "source_kind",
        "controller_proxy_source",
        "legacy_fallback_used",
        "controller_proxy_legacy_fallback_used",
        "events_count",
        "history_row_count",
        "native_row_count",
        "legacy_row_count",
        "skipped_row_count",
        "records_evaluated",
        "records_with_group_keys",
        "total_groups_new",
        "groups_new",
        "total_shots_new",
        "shots_new",
        "controller_group_proxy",
        "controller_shot_proxy",
    )
    return {key: _jsonable(summary.get(key)) for key in keep if key in summary}


def _runtime_payload_summary(result_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result_payload, Mapping):
        return {}
    adapt = _adapt_runtime_payload(result_payload)
    continuation = adapt.get("continuation", {}) if isinstance(adapt, Mapping) else {}
    if not isinstance(continuation, Mapping):
        continuation = {}
    phase2 = continuation.get("phase2", {})
    if not isinstance(phase2, Mapping):
        phase2 = {}
    algebraic_lane_policy = continuation.get("algebraic_lane_policy", {})
    if not isinstance(algebraic_lane_policy, Mapping):
        algebraic_lane_policy = {}
    phase0_pilot = continuation.get("phase0_pilot", {})
    if not isinstance(phase0_pilot, Mapping):
        phase0_pilot = {}
    algebraic_shortlist_controls = algebraic_lane_policy.get("shortlist_controls", {})
    if not isinstance(algebraic_shortlist_controls, Mapping):
        algebraic_shortlist_controls = {}
    active_surface = continuation.get("active_phase3_surface_summary", {})
    if not isinstance(active_surface, Mapping):
        active_surface = {}
    score_config = active_surface.get("score_config", {})
    if not isinstance(score_config, Mapping):
        score_config = {}
    controller = adapt.get("controller_measurement_work_summary")
    if not isinstance(controller, Mapping):
        controller = continuation.get("controller_measurement_work_summary", {})
    scored_rows = continuation.get("phase2_scored_rows", ())
    row_sample: Mapping[str, Any] = {}
    if isinstance(scored_rows, Sequence) and not isinstance(scored_rows, (str, bytes)) and scored_rows:
        first = scored_rows[0]
        row_sample = first if isinstance(first, Mapping) else {}
    phase3_field_presence = {
        key: bool(key in row_sample)
        for key in (
            "phase3_primary_score",
            "phase3_tie_break_score",
            "phase3_auxiliary_score_mode",
            "phase3_canonical_score_formula",
        )
    }
    return {
        "adapt_inner_optimizer": adapt.get("adapt_inner_optimizer") if isinstance(adapt, Mapping) else None,
        "adapt_spsa": _compact_mapping(adapt.get("adapt_spsa") if isinstance(adapt, Mapping) else None, limit=16),
        "continuation_phase2": _compact_mapping(
            {
                key: phase2.get(key)
                for key in (
                    "canonical_score_formula",
                    "primary_selector_score_key",
                    "selector_tie_break_score_key",
                    "secondary_geometry_score_key",
                    "auxiliary_terms_primary_mode",
                    "phase2_novelty_mode",
                    "phase2_raw_score_formula",
                    "phase2_novelty_eps",
                    "batch_target_size_requested",
                    "batch_target_size_effective",
                    "batch_size_cap_requested",
                    "batch_size_cap_effective",
                    "batch_target_size",
                    "batch_size_cap",
                )
                if key in phase2
            },
            limit=32,
        ),
        "active_phase3_surface_score_config": _compact_mapping(score_config, limit=32),
        "algebraic_shortlist_controls": _compact_mapping(algebraic_shortlist_controls, limit=16),
        "phase0_pilot": _compact_mapping(phase0_pilot, limit=24),
        "controller_measurement_work_summary": _strip_controller_summary(controller if isinstance(controller, Mapping) else {}),
        "phase3_scored_row_count": (
            int(len(scored_rows))
            if isinstance(scored_rows, Sequence) and not isinstance(scored_rows, (str, bytes))
            else None
        ),
        "phase3_field_presence": phase3_field_presence,
    }


def _policy_audit_summary(policy: AlgorithmPolicy) -> dict[str, Any]:
    inner = policy.inner_optimizer
    static = policy.static
    summary = {
        "pool_key": str(policy.pool.pool_key),
        "inner_optimizer": str(inner.inner_optimizer),
        "final_optimizer_type": str(inner.final_optimizer_type),
        "phase2_batch_target_size": int(static.phase2_batch_target_size),
        "phase2_batch_size_cap": int(static.phase2_batch_size_cap),
        "phase2_novelty_mode": str(static.phase2_novelty_mode),
        "phase2_enable_batching": bool(static.phase2_enable_batching),
        "phase1_prune_enabled": bool(static.phase1_prune_enabled),
        "phase1_prune_policy": str(static.phase1_prune_policy),
        "phase1_prune_mode": str(static.phase1_prune_mode),
        "phase1_prune_amplitude_witness_required": bool(static.phase1_prune_amplitude_witness_required),
        "algebraic_phase2_lane_rel_threshold": float(static.algebraic_phase2_lane_rel_threshold),
        "algebraic_phase1_lane_quota_pressure": float(static.algebraic_phase1_lane_quota_pressure),
        "algebraic_phase2_lane_quota_pressure": float(static.algebraic_phase2_lane_quota_pressure),
        "phase0_pilot_enabled": bool(static.phase0_pilot_enabled),
        "phase0_pilot_alpha": float(static.phase0_pilot_alpha),
        "phase0_pilot_threshold": float(static.phase0_pilot_threshold),
        "phase0_pilot_max_records": int(static.phase0_pilot_max_records),
        "phase0_lane_quota_pressure": float(static.phase0_lane_quota_pressure),
        "phase0_algebraic_lane_mode": str(static.phase0_algebraic_lane_mode),
        "phase3_selector_geometry_mode": str(static.phase3_selector_geometry_mode),
        "phase2_gamma_N": float(static.phase2_gamma_N),
        "phase2_gamma_N_schedule_mode": str(static.phase2_gamma_N_schedule_mode),
        "phase2_gamma_N_schedule_start": static.phase2_gamma_N_schedule_start,
        "phase2_gamma_N_schedule_end": static.phase2_gamma_N_schedule_end,
        "phase2_motif_bonus_weight": static.phase2_motif_bonus_weight,
        "phase3_novelty_ablation_mode": str(static.phase3_novelty_ablation_mode),
        "phase3_window_relaxation_mode": str(static.phase3_window_relaxation_mode),
        "phase3_backend_cost_mode": str(static.phase3_backend_cost_mode),
        "static_meta_feature_profile": normalize_static_meta_feature_profile(
            getattr(static, "static_meta_feature_profile", _META_FEATURE_PROFILE_OFF),
            default=_META_FEATURE_PROFILE_OFF,
        ),
        "static_route_id": normalize_static_route_id(
            getattr(static, "static_route_id", ROUTE_ID_A),
            default=ROUTE_ID_A,
        ),
        "static_lane_route": normalize_static_lane_route(
            getattr(static, "static_lane_route", STATIC_LANE_ROUTE_ALGEBRAIC),
            default=STATIC_LANE_ROUTE_ALGEBRAIC,
        ),
        "physical_lane_shortlist_aggressiveness": normalize_physical_lane_shortlist_aggressiveness(
            getattr(static, "physical_lane_shortlist_aggressiveness", 3),
            default=3,
        ),
    }
    if _is_spsa_optimizer(inner.inner_optimizer):
        summary["spsa_schedule"] = {
            "spsa_a": float(inner.spsa_a),
            "spsa_c": float(inner.spsa_c),
            "spsa_A": float(inner.spsa_A),
            "spsa_alpha": float(inner.spsa_alpha),
            "spsa_gamma": float(inner.spsa_gamma),
        }
    return summary


def _meta_feature_policy_payload(policy: AlgorithmPolicy) -> dict[str, Any]:
    static = policy.static
    profile = normalize_static_meta_feature_profile(
        getattr(static, "static_meta_feature_profile", _META_FEATURE_PROFILE_OFF),
        default=_META_FEATURE_PROFILE_OFF,
    )
    feature_bundle = {
        "phase0_pilot_enabled": bool(static.phase0_pilot_enabled),
        "phase0_algebraic_lane_mode": str(static.phase0_algebraic_lane_mode),
        "phase3_batching_enabled": bool(static.phase2_enable_batching),
        "phase1_prune_enabled": bool(static.phase1_prune_enabled),
        "phase1_prune_amplitude_witness_required": bool(static.phase1_prune_amplitude_witness_required),
        "adapt_allow_repeats": bool(static.adapt_allow_repeats),
        "adapt_reopt_policy": str(static.adapt_reopt_policy),
        "adapt_insertion_mode": str(static.adapt_insertion_mode),
        "adapt_beam_live_branches": int(static.adapt_beam_live_branches),
        "adapt_beam_children_per_parent": int(static.adapt_beam_children_per_parent),
        "adapt_beam_terminated_keep": int(static.adapt_beam_terminated_keep),
        "adapt_beam_lambda": float(static.adapt_beam_lambda),
    }
    hard_constraints = {
        "pool_key": "full_meta",
        "phase2_novelty_mode": _ACTIVE_PHASE2_NOVELTY_MODE,
        "phase3_selector_policy": _DEFAULT_PHASE3_SELECTOR_POLICY,
        "phase3_selector_geometry_mode": "reduced",
        "hardware_resolution_mode": "ideal",
        "ed_reference_decision_leakage": "forbidden",
    }
    tunable_feature_toggles = []
    if profile == _META_FEATURE_PROFILE_PAPER_I_PRODUCTION:
        hard_constraints.update(
            {
                "candidate_position_route": "r=(m,p)",
                "post_score_position_prior": "forbidden",
                "compile_position_shift_weight": 0.0,
                "family_repeat_penalty": 0.0,
                "motif_bonus_weight": 0.0,
                "phase0_pilot_enabled": True,
                "phase1_prune_enabled": True,
                "phase1_prune_amplitude_witness_required": True,
                "phase_live_hysteresis_enabled": False,
            }
        )
        tunable_feature_toggles = ["phase3_batching_enabled", "adapt_allow_repeats"]
    elif profile == _META_FEATURE_PROFILE_SAFE_CORE:
        tunable_feature_toggles = [
            "phase0_pilot_enabled",
            "phase3_batching_enabled",
            "phase1_prune_enabled",
            "phase1_prune_amplitude_witness_required",
        ]
    disabled = sorted(
        key
        for key, value in feature_bundle.items()
        if isinstance(value, bool) and value is False
    )
    return {
        "schema": "phase3_meta_feature_policy_v1",
        "profile": profile,
        "hard_identity_constraints": hard_constraints,
        "feature_bundle": feature_bundle,
        "tunable_feature_toggles": tunable_feature_toggles,
        "disabled_features": disabled,
        "promotion_label": "paper_i_production_static_snake_v1"
        if profile == _META_FEATURE_PROFILE_PAPER_I_PRODUCTION
        else (
            "canonical_route_a_safe_core_feature_bundle"
            if profile == _META_FEATURE_PROFILE_SAFE_CORE
            else "canonical_route_a_strict_all_on"
        ),
        "canonical_route_a_identity_preserved": True,
    }


def _audit_status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = int(counts.get(status, 0)) + 1
    return dict(sorted(counts.items()))


def _sampled_key_coverage(sampled: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    covered: set[str] = set()
    for row in rows:
        parsed = row.get("parsed_cli_param")
        if parsed not in {None, ""}:
            covered.add(str(parsed))
        knob = row.get("knob")
        if knob not in {None, ""}:
            covered.add(str(knob))
    known = set(default_trial_params())
    known.add("rescue_expand_factor")
    mapped_without_row = sorted(str(key) for key in sampled if str(key) not in covered and str(key) in known)
    unaudited = sorted(str(key) for key in sampled if str(key) not in covered and str(key) not in known)
    return {
        "sampled_key_count": int(len(sampled)),
        "row_covered_sampled_knobs": sorted(str(key) for key in sampled if str(key) in covered),
        "mapped_sampled_knobs_without_rows": mapped_without_row,
        "unaudited_sampled_knobs": unaudited,
    }


def _roundtrip_toggle_audit_row(
    *,
    knob: str,
    sampled_params: Mapping[str, Any],
    policy_value: bool,
    emitted_options: Mapping[str, Any],
    parsed_cli_params: Mapping[str, Any],
    result_payload: Mapping[str, Any] | None,
    positive_flag: str,
    negative_flag: str,
    parsed_param_key: str | None = None,
    runtime_path: str | None = None,
) -> dict[str, Any]:
    positive_key = str(positive_flag).lstrip("-").replace("-", "_")
    negative_key = str(negative_flag).lstrip("-").replace("-", "_")
    emitted_flag = positive_flag if bool(policy_value) else negative_flag
    if positive_key in emitted_options:
        emitted_value: Any = True
    elif negative_key in emitted_options:
        emitted_value = False
    else:
        emitted_value = _MISSING
    parsed_key = str(parsed_param_key or knob)
    parsed_value = parsed_cli_params.get(parsed_key, _MISSING)
    runtime_value = _runtime_path_get(result_payload, runtime_path)
    sampled_value = sampled_params.get(parsed_key, _MISSING)
    final_status = "ok"
    final_reason = None
    if emitted_value is _MISSING:
        final_status = "not_forwarded"
        final_reason = "missing_emitted_cli_flag"
    elif not _option_values_equivalent(emitted_value, bool(policy_value)):
        final_status = "mismatch"
        final_reason = "policy_vs_emitted"
    elif parsed_value is not _MISSING and not _option_values_equivalent(parsed_value, bool(policy_value)):
        final_status = "mismatch"
        final_reason = "policy_vs_parsed_cli"
    elif runtime_value is _MISSING and runtime_path is not None:
        final_status = "not_observable"
        final_reason = "runtime_payload_missing_path"
    elif runtime_value is not _MISSING and not _option_values_equivalent(runtime_value, bool(policy_value)):
        final_status = "mismatch"
        final_reason = "policy_vs_runtime"
    return {
        "knob": str(knob),
        "sampled_value": None if sampled_value is _MISSING else _jsonable(sampled_value),
        "policy_value": bool(policy_value),
        "emitted_flag": emitted_flag,
        "emitted_value": None if emitted_value is _MISSING else _jsonable(emitted_value),
        "parsed_cli_param": parsed_key,
        "parsed_cli_value": None if parsed_value is _MISSING else _jsonable(parsed_value),
        "runtime_path": runtime_path,
        "runtime_value": None if runtime_value is _MISSING else _jsonable(runtime_value),
        "status": final_status,
        "reason": final_reason,
    }


def _roundtrip_audit_row(
    *,
    knob: str,
    sampled_params: Mapping[str, Any],
    policy_value: Any = None,
    emitted_options: Mapping[str, Any],
    parsed_cli_params: Mapping[str, Any],
    result_payload: Mapping[str, Any] | None,
    emitted_flag: str | None = None,
    parsed_param_key: str | None = None,
    runtime_path: str | None = None,
    status: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    option_key = None if emitted_flag is None else str(emitted_flag).lstrip("-").replace("-", "_")
    emitted_value = _MISSING if option_key is None else emitted_options.get(option_key, _MISSING)
    parsed_value = (
        _MISSING
        if parsed_param_key is None
        else parsed_cli_params.get(str(parsed_param_key), _MISSING)
    )
    runtime_value = _runtime_path_get(result_payload, runtime_path)
    sampled_value = sampled_params.get(str(parsed_param_key or knob), _MISSING)

    final_status = status
    final_reason = reason
    if final_status is None:
        final_status = "ok"
        if option_key is not None and emitted_value is _MISSING:
            final_status = "not_forwarded"
            final_reason = final_reason or "missing_emitted_cli_flag"
        elif runtime_path is not None and runtime_value is _MISSING:
            final_status = "not_observable"
            final_reason = final_reason or "runtime_payload_missing_path"
        elif emitted_value is not _MISSING and policy_value is not None and not _option_values_equivalent(emitted_value, policy_value):
            final_status = "mismatch"
            final_reason = final_reason or "policy_vs_emitted"
        elif parsed_value is not _MISSING and policy_value is not None and not _option_values_equivalent(parsed_value, policy_value):
            final_status = "mismatch"
            final_reason = final_reason or "policy_vs_parsed_cli"
        elif runtime_value is not _MISSING and policy_value is not None and not _option_values_equivalent(runtime_value, policy_value):
            final_status = "mismatch"
            final_reason = final_reason or "policy_vs_runtime"

    return {
        "knob": str(knob),
        "sampled_value": None if sampled_value is _MISSING else _jsonable(sampled_value),
        "policy_value": _jsonable(policy_value),
        "emitted_flag": emitted_flag,
        "emitted_value": None if emitted_value is _MISSING else _jsonable(emitted_value),
        "parsed_cli_param": parsed_param_key,
        "parsed_cli_value": None if parsed_value is _MISSING else _jsonable(parsed_value),
        "runtime_path": runtime_path,
        "runtime_value": None if runtime_value is _MISSING else _jsonable(runtime_value),
        "status": str(final_status),
        "reason": final_reason,
    }


def build_policy_roundtrip_audit(
    *,
    sampled_params: Mapping[str, Any] | None,
    policy: AlgorithmPolicy,
    spec: HamiltonianBenchmarkSpec,
    emitted_command: Sequence[str],
    result_payload: Mapping[str, Any] | None,
    objective_weights: StaticObjectiveWeights = StaticObjectiveWeights(),
    measurement_proxy_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact sampled-policy -> CLI -> runtime audit payload."""

    policy = _normalize_active_policy(policy)
    sampled = dict(sampled_params or {})
    command = [str(token) for token in emitted_command]
    emitted_options = _args_option_map(command)
    command_string = " ".join(shlex.quote(token) for token in command)
    parsed_cli_params = trial_params_from_cli_command(command_string)
    runtime_summary = _runtime_payload_summary(result_payload)
    validation = (
        dict(measurement_proxy_validation)
        if isinstance(measurement_proxy_validation, Mapping) and measurement_proxy_validation
        else (
            _controller_proxy_validation_from_payload(result_payload)
            if isinstance(result_payload, Mapping)
            else {"schema": "controller_measurement_proxy_validation_v1", "valid": False, "reason": "missing_result_payload"}
        )
    )

    rows: list[dict[str, Any]] = []
    inner = policy.inner_optimizer
    static = policy.static
    meta_feature_policy = _meta_feature_policy_payload(policy)
    rows.append(
        _roundtrip_audit_row(
            knob="inner_optimizer",
            sampled_params=sampled,
            policy_value=str(inner.inner_optimizer),
            emitted_options=emitted_options,
            parsed_cli_params=parsed_cli_params,
            result_payload=result_payload,
            emitted_flag="--adapt-inner-optimizer",
            parsed_param_key="inner_optimizer",
            runtime_path="adapt_vqe.adapt_inner_optimizer",
        )
    )
    if "inner_optimizer" in sampled and str(sampled.get("inner_optimizer", "")).strip().upper() != str(inner.inner_optimizer):
        rows[-1]["status"] = "forced"
        rows[-1]["reason"] = "fixed_inner_optimizer_route_identity"

    spsa_runtime_keys = {
        "spsa_a": "a",
        "spsa_c": "c",
        "spsa_A": "A",
        "spsa_alpha": "alpha",
        "spsa_gamma": "gamma",
    }
    for key, flag_option in _SPSA_CLI_OPTIONS.items():
        policy_value = getattr(inner, key)
        flag = "--" + flag_option.replace("_", "-")
        if _is_spsa_optimizer(inner.inner_optimizer):
            rows.append(
                _roundtrip_audit_row(
                    knob=key,
                    sampled_params=sampled,
                    policy_value=policy_value,
                    emitted_options=emitted_options,
                    parsed_cli_params=parsed_cli_params,
                    result_payload=result_payload,
                    emitted_flag=flag,
                    parsed_param_key=key,
                    runtime_path=f"adapt_vqe.adapt_spsa.{spsa_runtime_keys[key]}",
                )
            )
        else:
            rows.append(
                _roundtrip_audit_row(
                    knob=key,
                    sampled_params=sampled,
                    policy_value=policy_value,
                    emitted_options=emitted_options,
                    parsed_cli_params=parsed_cli_params,
                    result_payload=result_payload,
                    emitted_flag=flag,
                    parsed_param_key=key,
                    status="inactive",
                    reason="non_spsa_inner_optimizer",
                )
            )

    rows.append(
        _roundtrip_audit_row(
            knob="phase3_selector_policy",
            sampled_params=sampled,
            policy_value=str(static.phase3_selector_policy),
            emitted_options=emitted_options,
            parsed_cli_params=parsed_cli_params,
            result_payload=result_payload,
            emitted_flag="--phase3-selector-policy",
            parsed_param_key="phase3_selector_policy",
            runtime_path="adapt_vqe.continuation.phase3_selector_policy",
        )
    )
    requested_static_route_id = normalize_static_route_id(
        getattr(static, "static_route_id", ROUTE_ID_A),
        default=ROUTE_ID_A,
    )
    selected_route = str(getattr(spec, "selected_logical_route", "standard") or "standard").strip().lower().replace("-", "_")
    emitted_static_route_id = normalize_static_route_id(
        emitted_options.get("static_route_id", requested_static_route_id),
        default=ROUTE_ID_UNSPECIFIED,
    )
    route_forced = (
        requested_static_route_id == ROUTE_ID_A
        and emitted_static_route_id == ROUTE_ID_UNSPECIFIED
        and selected_route != "standard"
    )
    route_row = _roundtrip_audit_row(
        knob="static_route_id",
        sampled_params=sampled,
        policy_value=emitted_static_route_id if route_forced else requested_static_route_id,
        emitted_options=emitted_options,
        parsed_cli_params=parsed_cli_params,
        result_payload=result_payload,
        emitted_flag="--static-route-id",
        parsed_param_key="static_route_id",
        runtime_path="adapt_vqe.continuation.static_route_identity.route_id",
    )
    route_row["requested_policy_value"] = requested_static_route_id
    route_row["selected_logical_route"] = selected_route
    if route_forced and route_row.get("status") != "mismatch":
        route_row["status"] = "forced"
        route_row["reason"] = "route_a_forced_to_unspecified_for_nonstandard_selected_logical_route"
    rows.append(route_row)

    rows.append(
        _roundtrip_audit_row(
            knob="static_meta_feature_profile",
            sampled_params=sampled,
            policy_value=str(static.static_meta_feature_profile),
            emitted_options=emitted_options,
            parsed_cli_params=parsed_cli_params,
            result_payload=result_payload,
            emitted_flag="--static-meta-feature-profile",
            parsed_param_key="meta_feature_profile",
            runtime_path="adapt_vqe.continuation.static_route_identity.meta_feature_profile",
        )
    )

    rows.append(
        _roundtrip_toggle_audit_row(
            knob="phase2_enable_batching",
            sampled_params=sampled,
            policy_value=bool(static.phase2_enable_batching),
            emitted_options=emitted_options,
            parsed_cli_params=parsed_cli_params,
            result_payload=result_payload,
            positive_flag="--phase3-enable-batching",
            negative_flag="--phase3-no-batching",
            parsed_param_key="phase2_enable_batching",
            runtime_path="adapt_vqe.continuation.phase2.batching_enabled",
        )
    )
    normalized_meta_profile = normalize_static_meta_feature_profile(
        getattr(static, "static_meta_feature_profile", _META_FEATURE_PROFILE_OFF),
        default=_META_FEATURE_PROFILE_OFF,
    )
    if normalized_meta_profile == _META_FEATURE_PROFILE_OFF:
        rows[-1]["status"] = "forced"
        rows[-1]["reason"] = "canonical_route_a_requires_phase3_batching_on"

    rows.append(
        _roundtrip_toggle_audit_row(
            knob="phase1_prune_enabled",
            sampled_params=sampled,
            policy_value=bool(static.phase1_prune_enabled),
            emitted_options=emitted_options,
            parsed_cli_params=parsed_cli_params,
            result_payload=result_payload,
            positive_flag="--phase1-prune-enabled",
            negative_flag="--phase1-no-prune",
            parsed_param_key="phase1_prune_enabled",
            runtime_path="adapt_vqe.continuation.phase1_prune.enabled",
        )
    )

    for key, flag, policy_value, runtime_path in (
        (
            "phase1_prune_policy",
            "--phase1-prune-policy",
            str(static.phase1_prune_policy),
            "adapt_vqe.continuation.phase1_prune.prune_policy",
        ),
        (
            "phase1_prune_mode",
            "--phase1-prune-mode",
            str(static.phase1_prune_mode),
            "adapt_vqe.continuation.phase1_prune.prune_mode",
        ),
        (
            "phase1_prune_amplitude_witness_required",
            "--phase1-prune-amplitude-witness-required",
            bool(static.phase1_prune_amplitude_witness_required),
            "adapt_vqe.continuation.phase1_prune.amplitude_witness_config_required",
        ),
    ):
        rows.append(
            _roundtrip_audit_row(
                knob=key,
                sampled_params=sampled,
                policy_value=policy_value,
                emitted_options=emitted_options,
                parsed_cli_params=parsed_cli_params,
                result_payload=result_payload,
                emitted_flag=flag,
                parsed_param_key=key,
                runtime_path=runtime_path,
            )
        )

    for key, flag, policy_value, runtime_path in (
        (
            "algebraic_phase2_lane_rel_threshold",
            "--algebraic-phase2-lane-rel-threshold",
            float(static.algebraic_phase2_lane_rel_threshold),
            "adapt_vqe.continuation.algebraic_lane_policy.shortlist_controls.phase2_lane_rel_threshold",
        ),
        (
            "algebraic_phase1_lane_quota_pressure",
            "--algebraic-phase1-lane-quota-pressure",
            float(static.algebraic_phase1_lane_quota_pressure),
            "adapt_vqe.continuation.algebraic_lane_policy.shortlist_controls.phase1_lane_quota_pressure",
        ),
        (
            "algebraic_phase2_lane_quota_pressure",
            "--algebraic-phase2-lane-quota-pressure",
            float(static.algebraic_phase2_lane_quota_pressure),
            "adapt_vqe.continuation.algebraic_lane_policy.shortlist_controls.phase2_lane_quota_pressure",
        ),
    ):
        rows.append(
            _roundtrip_audit_row(
                knob=key,
                sampled_params=sampled,
                policy_value=policy_value,
                emitted_options=emitted_options,
                parsed_cli_params=parsed_cli_params,
                result_payload=result_payload,
                emitted_flag=flag,
                parsed_param_key=key,
                runtime_path=runtime_path,
            )
        )

    rows.append(
        _roundtrip_toggle_audit_row(
            knob="phase0_pilot_enabled",
            sampled_params=sampled,
            policy_value=bool(static.phase0_pilot_enabled),
            emitted_options=emitted_options,
            parsed_cli_params=parsed_cli_params,
            result_payload=result_payload,
            positive_flag="--phase0-pilot-enabled",
            negative_flag="--phase0-no-pilot",
            parsed_param_key="phase0_pilot_enabled",
            runtime_path="adapt_vqe.continuation.phase0_pilot.enabled",
        )
    )
    for key, flag, policy_value, runtime_path in (
        (
            "phase0_pilot_alpha",
            "--phase0-pilot-alpha",
            float(static.phase0_pilot_alpha),
            "adapt_vqe.continuation.phase0_pilot.alpha0",
        ),
        (
            "phase0_pilot_threshold",
            "--phase0-pilot-threshold",
            float(static.phase0_pilot_threshold),
            "adapt_vqe.continuation.phase0_pilot.threshold",
        ),
        (
            "phase0_pilot_max_records",
            "--phase0-pilot-max-records",
            int(static.phase0_pilot_max_records),
            "adapt_vqe.continuation.phase0_pilot.max_records",
        ),
        (
            "phase0_lane_quota_pressure",
            "--phase0-lane-quota-pressure",
            float(static.phase0_lane_quota_pressure),
            "adapt_vqe.continuation.phase0_pilot.lane_quota_pressure",
        ),
        (
            "phase0_algebraic_lane_mode",
            "--phase0-algebraic-lane-mode",
            str(static.phase0_algebraic_lane_mode),
            "adapt_vqe.continuation.phase0_pilot.algebraic_lane_mode",
        ),
    ):
        rows.append(
            _roundtrip_audit_row(
                knob=key,
                sampled_params=sampled,
                policy_value=policy_value,
                emitted_options=emitted_options,
                parsed_cli_params=parsed_cli_params,
                result_payload=result_payload,
                emitted_flag=flag,
                parsed_param_key=key,
                runtime_path=runtime_path,
            )
        )

    for key, flag, policy_value in (
        (
            "phase1_prune_min_candidates",
            "--phase1-prune-min-candidates",
            int(static.phase1_prune_min_candidates),
        ),
        (
            "phase1_prune_max_candidates",
            "--phase1-prune-max-candidates",
            int(static.phase1_prune_max_candidates),
        ),
        (
            "phase1_prune_stale_age",
            "--phase1-prune-stale-age",
            int(static.phase1_prune_stale_age),
        ),
        (
            "phase1_prune_stagnation_threshold",
            "--phase1-prune-stagnation-threshold",
            float(static.phase1_prune_stagnation_threshold),
        ),
        (
            "phase1_prune_small_theta_abs",
            "--phase1-prune-small-theta-abs",
            float(static.phase1_prune_small_theta_abs),
        ),
        (
            "phase1_prune_small_theta_relative",
            "--phase1-prune-small-theta-relative",
            float(static.phase1_prune_small_theta_relative),
        ),
        (
            "phase1_prune_cooldown_steps",
            "--phase1-prune-cooldown-steps",
            int(static.phase1_prune_cooldown_steps),
        ),
        (
            "phase1_prune_local_window_size",
            "--phase1-prune-local-window-size",
            int(static.phase1_prune_local_window_size),
        ),
        (
            "phase1_prune_recovery_trust_radius",
            "--phase1-prune-recovery-trust-radius",
            float(static.phase1_prune_recovery_trust_radius),
        ),
        (
            "phase1_prune_old_fraction",
            "--phase1-prune-old-fraction",
            float(static.phase1_prune_old_fraction),
        ),
        (
            "phase1_prune_checkpoint_period",
            "--phase1-prune-checkpoint-period",
            int(static.phase1_prune_checkpoint_period),
        ),
        (
            "phase1_prune_live_min_depth",
            "--phase1-prune-live-min-depth",
            int(static.phase1_prune_live_min_depth),
        ),
        (
            "phase1_prune_maturity_threshold",
            "--phase1-prune-maturity-threshold",
            float(static.phase1_prune_maturity_threshold),
        ),
        (
            "phase1_prune_snr_threshold",
            "--phase1-prune-snr-threshold",
            float(static.phase1_prune_snr_threshold),
        ),
        (
            "phase1_prune_tolerance_mode",
            "--phase1-prune-tolerance-mode",
            str(static.phase1_prune_tolerance_mode),
        ),
        (
            "phase1_prune_tolerance_shot_coeff",
            "--phase1-prune-tolerance-shot-coeff",
            float(static.phase1_prune_tolerance_shot_coeff),
        ),
        (
            "phase1_prune_tolerance_screen_coeff",
            "--phase1-prune-tolerance-screen-coeff",
            float(static.phase1_prune_tolerance_screen_coeff),
        ),
        (
            "phase1_prune_tolerance_chem",
            "--phase1-prune-tolerance-chem",
            float(static.phase1_prune_tolerance_chem),
        ),
        (
            "phase1_prune_tolerance_rel_coeff",
            "--phase1-prune-tolerance-rel-coeff",
            float(static.phase1_prune_tolerance_rel_coeff),
        ),
    ):
        rows.append(
            _roundtrip_audit_row(
                knob=key,
                sampled_params=sampled,
                policy_value=policy_value,
                emitted_options=emitted_options,
                parsed_cli_params=parsed_cli_params,
                result_payload=result_payload,
                emitted_flag=flag,
                parsed_param_key=key,
            )
        )

    rows.append(
        _roundtrip_audit_row(
            knob="phase2_novelty_mode",
            sampled_params=sampled,
            policy_value=str(static.phase2_novelty_mode),
            emitted_options=emitted_options,
            parsed_cli_params=parsed_cli_params,
            result_payload=result_payload,
            emitted_flag="--phase2-novelty-mode",
            parsed_param_key="phase2_novelty_mode",
            runtime_path="adapt_vqe.continuation.phase2.phase2_novelty_mode",
        )
    )

    for key, flag, policy_value, runtime_path in (
        (
            "phase2_batch_size_cap",
            "--phase2-batch-size-cap",
            int(static.phase2_batch_size_cap),
            "adapt_vqe.continuation.phase2.batch_size_cap_effective",
        ),
        (
            "phase2_batch_target_size",
            "--phase2-batch-target-size",
            int(static.phase2_batch_target_size),
            "adapt_vqe.continuation.phase2.batch_target_size_effective",
        ),
    ):
        status = None
        reason = None
        sampled_value = sampled.get(key)
        if key == "phase2_batch_target_size" and sampled_value is not None:
            try:
                if int(sampled_value) != int(policy_value):
                    status = "clamped"
                    reason = "phase2_batch_target_size_limited_by_cap"
            except Exception:
                pass
        rows.append(
            _roundtrip_audit_row(
                knob=key,
                sampled_params=sampled,
                policy_value=policy_value,
                emitted_options=emitted_options,
                parsed_cli_params=parsed_cli_params,
                result_payload=result_payload,
                emitted_flag=flag,
                parsed_param_key=key,
                runtime_path=runtime_path,
                status=status,
                reason=reason,
            )
        )

    if "rescue_expand_factor" in sampled:
        rows.append(
            _roundtrip_audit_row(
                knob="rescue_expand_factor",
                sampled_params=sampled,
                policy_value=getattr(policy.pool, "rescue_expand_factor", None),
                emitted_options=emitted_options,
                parsed_cli_params=parsed_cli_params,
                result_payload=result_payload,
                parsed_param_key="rescue_expand_factor",
                status="not_forwarded",
                reason="historical_or_inactive_optuna_knob_not_emitted",
            )
        )

    base_options = _args_option_map(spec.base_pipeline_args)
    selected_mode = emitted_options.get("adapt_selected_logical_mode")
    base_selected_mode = base_options.get("adapt_selected_logical_mode")
    rows.append(
        _roundtrip_audit_row(
            knob="adapt_selected_logical_mode",
            sampled_params=sampled,
            policy_value=selected_mode,
            emitted_options=emitted_options,
            parsed_cli_params=parsed_cli_params,
            result_payload=result_payload,
            emitted_flag="--adapt-selected-logical-mode",
            runtime_path="adapt_vqe.adapt_selected_logical_mode",
            status=(
                "forced"
                if base_selected_mode is not None and str(base_selected_mode) != str(selected_mode)
                else None
            ),
            reason=(
                "policy_overrides_base_selected_logical_mode"
                if base_selected_mode is not None and str(base_selected_mode) != str(selected_mode)
                else None
            ),
        )
    )
    for knob, flag, forced_value in (
        ("phase3_lifetime_cost_mode", "--phase3-lifetime-cost-mode", "off"),
        (
            "phase3_runtime_split_mode",
            "--phase3-runtime-split-mode",
            str(getattr(policy.static, "phase3_runtime_split_mode", "off") or "off"),
        ),
        ("phase3_enable_rescue", "--phase3-enable-rescue", True),
    ):
        rows.append(
            _roundtrip_audit_row(
                knob=knob,
                sampled_params=sampled,
                policy_value=forced_value,
                emitted_options=emitted_options,
                parsed_cli_params=parsed_cli_params,
                result_payload=result_payload,
                emitted_flag=flag,
                status="forced",
                reason="canonical_phase3_optuna_route_forces_cli_value",
            )
        )


    for suffix, attr_name in (
        ("2q", "lambda_2q"),
        ("d", "lambda_d"),
        ("1q", "lambda_1q"),
        ("theta", "lambda_theta"),
        ("shot", "lambda_shot"),
    ):
        rows.append(
            _roundtrip_audit_row(
                knob=f"hardware_cost_lambda_{suffix}",
                sampled_params=sampled,
                policy_value=float(getattr(static, attr_name)),
                emitted_options=emitted_options,
                parsed_cli_params=parsed_cli_params,
                result_payload=result_payload,
                emitted_flag=f"--phase2-lambda-{suffix}",
                runtime_path=f"adapt_vqe.continuation.phase2.lambda_{suffix}",
            )
        )

    shot_weight = float(objective_weights.shot_cost)
    shot_valid = bool(validation.get("valid", False))
    rows.append(
        {
            "knob": "objective_shot_weight",
            "sampled_value": None,
            "policy_value": shot_weight,
            "emitted_flag": None,
            "emitted_value": None,
            "parsed_cli_param": None,
            "parsed_cli_value": None,
            "runtime_path": "adapt_vqe.controller_measurement_work_summary",
            "runtime_value": _jsonable(_strip_controller_summary(
                controller_proxy_from_adapt_payload(result_payload) if isinstance(result_payload, Mapping) else {}
            )),
            "status": "ok" if shot_weight > 0.0 and shot_valid else "inactive",
            "reason": (
                "validated_native_controller_work"
                if shot_weight > 0.0 and shot_valid
                else ("zero_objective_shot_weight" if shot_weight <= 0.0 else str(validation.get("reason", "invalid_measurement_proxy")))
            ),
        }
    )

    for knob, runtime_path in (
        ("canonical_score_formula", "adapt_vqe.continuation.phase2.canonical_score_formula"),
        ("phase2_raw_score_formula", "adapt_vqe.continuation.phase2.phase2_raw_score_formula"),
        ("primary_selector_score_key", "adapt_vqe.continuation.phase2.primary_selector_score_key"),
        ("selector_tie_break_score_key", "adapt_vqe.continuation.phase2.selector_tie_break_score_key"),
        ("secondary_geometry_score_key", "adapt_vqe.continuation.phase2.secondary_geometry_score_key"),
        ("auxiliary_terms_primary_mode", "adapt_vqe.continuation.phase2.auxiliary_terms_primary_mode"),
    ):
        rows.append(
            _roundtrip_audit_row(
                knob=knob,
                sampled_params=sampled,
                emitted_options=emitted_options,
                parsed_cli_params=parsed_cli_params,
                result_payload=result_payload,
                runtime_path=runtime_path,
                status=None,
            )
        )

    initial_coverage = _sampled_key_coverage(sampled, rows)
    for key in initial_coverage.get("unaudited_sampled_knobs", ()):
        rows.append(
            {
                "knob": str(key),
                "sampled_value": _jsonable(sampled.get(str(key))),
                "policy_value": None,
                "emitted_flag": None,
                "emitted_value": None,
                "parsed_cli_param": str(key),
                "parsed_cli_value": None,
                "runtime_path": None,
                "runtime_value": None,
                "status": "not_forwarded",
                "reason": "sampled_key_not_in_policy_roundtrip_allowlist",
            }
        )

    sampled_key_coverage = _sampled_key_coverage(sampled, rows)
    status_counts = _audit_status_counts(rows)
    audit = {
        "schema": "phase3_policy_roundtrip_audit_v1",
        "benchmark_id": str(spec.benchmark_id),
        "family": str(spec.family),
        "fixed_inner_optimizer": _ACTIVE_INNER_OPTIMIZER,
        "inner_optimizer_policy": _INNER_OPTIMIZER_POLICY_LABEL,
        "meta_feature_profile": str(static.static_meta_feature_profile),
        "meta_feature_policy": _jsonable(meta_feature_policy),
        "promotion_label": str(meta_feature_policy.get("promotion_label")),
        "sampled_param_count": int(len(sampled)),
        "sampled_params": _compact_mapping(sampled, limit=160),
        "normalized_policy_summary": _policy_audit_summary(policy),
        "emitted_cli_args": command,
        "emitted_options": _compact_mapping(emitted_options, limit=160),
        "parsed_cli_params": _compact_mapping(parsed_cli_params, limit=160),
        "runtime_payload_summary": runtime_summary,
        "objective_weights": _jsonable(asdict(objective_weights)),
        "measurement_proxy_validation": _jsonable(validation),
        "sampled_key_coverage": sampled_key_coverage,
        "unaudited_sampled_knobs": sampled_key_coverage["unaudited_sampled_knobs"],
        "rows": rows,
        "status_counts": status_counts,
    }
    for status in ("forced", "clamped", "inactive", "not_forwarded", "not_observable", "mismatch"):
        audit[f"{status}_knobs"] = [row["knob"] for row in rows if row.get("status") == status]
    return _jsonable(audit)


def _load_result_payload_for_audit(result: BenchmarkResult) -> Mapping[str, Any] | None:
    if result.result_json in {None, ""}:
        return None
    try:
        path = Path(str(result.result_json))
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, Mapping) else None
    except Exception:
        return None
    return None


def _audit_output_json_path(result: BenchmarkResult, spec: HamiltonianBenchmarkSpec, output_dir: Path) -> Path:
    if result.result_json not in {None, ""}:
        return Path(str(result.result_json))
    return Path(output_dir) / spec.benchmark_id / "json" / "result.json"


def _ensure_policy_roundtrip_audit(
    result: BenchmarkResult,
    *,
    sampled_params: Mapping[str, Any] | None,
    policy: AlgorithmPolicy,
    spec: HamiltonianBenchmarkSpec,
    output_dir: Path,
    objective_weights: StaticObjectiveWeights,
    result_payload: Mapping[str, Any] | None = None,
    write_path: Path | None = None,
    emitted_command: Sequence[str] | None = None,
) -> BenchmarkResult:
    payload = result_payload if isinstance(result_payload, Mapping) else _load_result_payload_for_audit(result)
    prior_command = (
        result.policy_roundtrip_audit.get("emitted_cli_args")
        if isinstance(result.policy_roundtrip_audit, Mapping)
        else None
    )
    if isinstance(emitted_command, Sequence) and not isinstance(emitted_command, (str, bytes)):
        command = [str(token) for token in emitted_command]
    elif isinstance(prior_command, Sequence) and not isinstance(prior_command, (str, bytes)):
        command = [str(token) for token in prior_command]
    else:
        command = build_static_command(
            python_bin=sys.executable,
            spec=spec,
            policy=policy,
            output_json=_audit_output_json_path(result, spec, Path(output_dir)),
        )
    audit = build_policy_roundtrip_audit(
        sampled_params=sampled_params,
        policy=policy,
        spec=spec,
        emitted_command=command,
        result_payload=payload,
        objective_weights=objective_weights,
        measurement_proxy_validation=result.measurement_proxy_validation,
    )
    audit_json_path = None
    target_write_path = write_path
    if target_write_path is None and result.policy_roundtrip_audit_json not in {None, ""}:
        target_write_path = Path(str(result.policy_roundtrip_audit_json))
    if target_write_path is not None:
        _write_json(Path(target_write_path), audit)
        audit_json_path = str(target_write_path)
    return replace(
        result,
        policy_roundtrip_audit=audit,
        policy_roundtrip_audit_json=audit_json_path or result.policy_roundtrip_audit_json,
    )


def _summarize_policy_roundtrip_audits(audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    benchmark_ids: list[str] = []
    for audit in audits:
        if not isinstance(audit, Mapping):
            continue
        bid = audit.get("benchmark_id")
        if bid not in {None, ""}:
            benchmark_ids.append(str(bid))
        for status, count in dict(audit.get("status_counts", {})).items():
            counts[str(status)] = int(counts.get(str(status), 0)) + int(count)
    return {
        "schema": "phase3_policy_roundtrip_audit_summary_v1",
        "audit_count": int(len([audit for audit in audits if isinstance(audit, Mapping)])),
        "benchmark_ids": list(dict.fromkeys(benchmark_ids)),
        "status_counts": dict(sorted(counts.items())),
    }


def _spec_local_n_ph_max(spec: HamiltonianBenchmarkSpec) -> int | None:
    raw = _pipeline_arg_value(spec.base_pipeline_args, "--n-ph-max")
    if raw in {None, ""}:
        return None
    try:
        return int(float(str(raw)))
    except Exception:
        return None


def _row_n_ph_max(row: Mapping[str, Any]) -> int | None:
    raw = row.get("n_ph_max")
    if raw in {None, ""}:
        return None
    try:
        return int(float(str(raw)))
    except Exception:
        return None


def _row_source_id(row: Mapping[str, Any], fallback: str) -> str:
    for key in ("artifact_json", "command_path", "record_id", "case_name", "label"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return str(fallback)


def _historical_row_for_spec(ledger: Mapping[str, Any], spec: HamiltonianBenchmarkSpec) -> Mapping[str, Any] | None:
    if any(str(tag).startswith("physics_perturbation") for tag in spec.tags):
        return None
    nph = _pipeline_arg_value(spec.base_pipeline_args, "--n-ph-max")
    candidate_tables = []
    if nph is not None:
        candidate_tables.append((ledger.get("best_warm_start_by_problem_size_cutoff") or {}, f"{spec.family}|L={spec.features.L}|nph={nph}"))
    candidate_tables.append((ledger.get("best_warm_start_by_problem") or {}, spec.family))
    for table, key in candidate_tables:
        if isinstance(table, Mapping) and key in table and isinstance(table[key], Mapping):
            return table[key]
    return None


def _normalize_selected_logical_route(value: str | None) -> str:
    route = str(value or "standard").strip().lower().replace("-", "_")
    if route not in {"standard", "historical_selected"}:
        raise ValueError("selected logical route must be one of {'standard','historical-selected'}.")
    return route


def _selected_logical_payload_for_spec(
    source_payload: Mapping[str, Any] | None,
    spec: HamiltonianBenchmarkSpec,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if not isinstance(source_payload, Mapping):
        return None, None, None
    nph = _pipeline_arg_value(spec.base_pipeline_args, "--n-ph-max")
    candidate_entries: list[tuple[str, Any]] = []
    if nph is not None:
        key = f"{spec.family}|L={spec.features.L}|nph={nph}"
        table = source_payload.get("best_selected_logical_by_problem_size_cutoff")
        if isinstance(table, Mapping) and key in table:
            candidate_entries.append((f"best_selected_logical_by_problem_size_cutoff:{key}", table.get(key)))
    table_problem = source_payload.get("best_selected_logical_by_problem")
    if isinstance(table_problem, Mapping) and spec.family in table_problem:
        candidate_entries.append((f"best_selected_logical_by_problem:{spec.family}", table_problem.get(spec.family)))
    candidate_entries.append(("direct_source", source_payload))
    for source_key, entry in candidate_entries:
        if not isinstance(entry, Mapping):
            continue
        try:
            selected = load_selected_logical_library_from_payload(entry)
        except Exception:
            selected = None
        if isinstance(selected, Mapping):
            return dict(selected), str(source_key), str(selected.get("source_kind", "unknown"))
    return None, None, None


def materialize_selected_logical_sidecars_for_specs(
    specs: Sequence[HamiltonianBenchmarkSpec],
    *,
    selected_logical_source: Mapping[str, Any] | None,
    output_dir: Path,
    route: str = "standard",
    transfer_mode: str = "exact_match_v1",
) -> tuple[HamiltonianBenchmarkSpec, ...]:
    route_key = _normalize_selected_logical_route(route)
    transfer_key = str(transfer_mode or "exact_match_v1").strip().lower()
    if transfer_key not in {"exact_match_v1", "boundary_v1"}:
        raise ValueError("selected logical transfer mode must be one of {'exact_match_v1','boundary_v1'}.")
    if route_key == "standard":
        return tuple(
            replace(
                spec,
                selected_logical_route="standard",
                selected_logical_source_json=None,
                selected_logical_source_kind=None,
                selected_logical_source_record_count=0,
                selected_logical_transfer_mode="exact_match_v1",
            )
            for spec in specs
        )
    sidecar_dir = Path(output_dir) / "selected_logical_sources"
    out: list[HamiltonianBenchmarkSpec] = []
    for spec in specs:
        selected, source_key, source_kind = _selected_logical_payload_for_spec(selected_logical_source, spec)
        if not isinstance(selected, Mapping):
            out.append(
                replace(
                    spec,
                    selected_logical_route="historical_selected",
                    selected_logical_source_json=None,
                    selected_logical_source_kind=None,
                    selected_logical_source_record_count=0,
                    selected_logical_transfer_mode=str(transfer_key),
                )
            )
            continue
        records = selected.get("records", [])
        record_count = int(len(records)) if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)) else 0
        sidecar_payload = {
            **dict(selected),
            "optuna_source_key": source_key,
            "optuna_benchmark_id": str(spec.benchmark_id),
            "optuna_family": str(spec.family),
        }
        sidecar_path = sidecar_dir / f"{spec.benchmark_id}.selected_logical.json"
        _write_json(sidecar_path, sidecar_payload)
        out.append(
            replace(
                spec,
                selected_logical_route="historical_selected",
                selected_logical_source_json=str(sidecar_path),
                selected_logical_source_kind=str(source_kind or selected.get("source_kind", "unknown")),
                selected_logical_source_record_count=int(record_count),
                selected_logical_transfer_mode=str(transfer_key),
            )
        )
    return tuple(out)


def _selected_logical_route_summary(specs: Sequence[HamiltonianBenchmarkSpec]) -> dict[str, Any]:
    route_counts: dict[str, int] = {}
    source_kind_counts: dict[str, int] = {}
    with_source = 0
    record_total = 0
    for spec in specs:
        route = str(getattr(spec, "selected_logical_route", "standard") or "standard")
        route_counts[route] = int(route_counts.get(route, 0)) + 1
        source = getattr(spec, "selected_logical_source_json", None)
        if source not in {None, ""}:
            with_source += 1
            kind = str(getattr(spec, "selected_logical_source_kind", None) or "unknown")
            source_kind_counts[kind] = int(source_kind_counts.get(kind, 0)) + 1
            record_total += int(getattr(spec, "selected_logical_source_record_count", 0) or 0)
    return {
        "route_counts": dict(sorted(route_counts.items())),
        "spec_count": int(len(specs)),
        "specs_with_source": int(with_source),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "source_record_count_total": int(record_total),
    }


def _trial_params_default_equivalent(params: Mapping[str, Any]) -> bool:
    return _sanitize_trial_params(params) == _sanitize_trial_params(default_trial_params())


def _nondefault_trial_params_with_audit(
    params: Mapping[str, Any],
    *,
    source: str,
    has_command: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    clean = _sanitize_trial_params(params)
    defaults = _sanitize_trial_params(default_trial_params())
    diff_keys = sorted(k for k in set(clean) | set(defaults) if clean.get(k) != defaults.get(k))
    audit: dict[str, Any] = {
        "source": source,
        "has_command": bool(has_command),
        "diff_from_default_count": len(diff_keys),
        "diff_from_default_keys": diff_keys[:32],
        "inner_optimizer": clean.get("inner_optimizer"),
    }
    if not diff_keys:
        audit["reason"] = "duplicates_default_seed"
        return None, audit
    return clean, audit


def _trial_params_from_cli_command_with_audit(command: str | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    audit: dict[str, Any] = {"has_command": bool(command), "source": "cli_command"}
    if not command:
        audit["reason"] = "missing_command"
        return None, audit
    params = trial_params_from_cli_command(command)
    return _nondefault_trial_params_with_audit(params, source="cli_command", has_command=True)


def _trial_params_from_historical_row_with_audit(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    for key in ("trial_params", "params"):
        raw = row.get(key)
        if isinstance(raw, Mapping) and raw:
            return _nondefault_trial_params_with_audit(raw, source=f"historical_ledger.{key}", has_command=bool(row.get("command")))
    return _trial_params_from_cli_command_with_audit(row.get("command"))


def _warm_start_candidate_json(record: WarmStartCandidate) -> dict[str, Any]:
    payload = dict(record.source_payload)
    return {
        "source_kind": record.source_kind,
        "source_id": record.source_id,
        "benchmark_id": record.benchmark_id,
        "family": record.family,
        "source_score": record.source_score,
        "params": dict(record.params),
        "source_artifact_path": payload.get("source_artifact_path") or payload.get("summary_path") or payload.get("ledger_path") or payload.get("artifact_json"),
        "source_sha256": payload.get("source_sha256") or payload.get("summary_sha256") or payload.get("ledger_sha256"),
        "source_payload": payload,
        "compatibility_warnings": list(record.compatibility_warnings),
    }


def _warm_start_skip_json(record: WarmStartSkip) -> dict[str, Any]:
    return {
        "source_kind": record.source_kind,
        "source_id": record.source_id,
        "benchmark_id": record.benchmark_id,
        "family": record.family,
        "reason": record.reason,
        "detail": record.detail,
        "source_payload": dict(record.source_payload),
    }


def _warm_start_counts(
    records: Sequence[WarmStartCandidate],
    skips: Sequence[WarmStartSkip],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = out.setdefault(record.source_kind, {"enqueued": 0, "skipped": 0})
        bucket["enqueued"] += 1
    for record in skips:
        bucket = out.setdefault(record.source_kind, {"enqueued": 0, "skipped": 0})
        bucket["skipped"] += 1
    return out


def _trial_warm_start_provenance_payload(
    trial_params: Mapping[str, Any],
    *,
    enqueued_records: Sequence[WarmStartCandidate] = (),
    warm_start_skips: Sequence[WarmStartSkip] = (),
    enqueue_default: bool = False,
    policy_search_profile: str | None = "default",
    meta_feature_profile: str | None = _DEFAULT_META_FEATURE_PROFILE,
) -> dict[str, Any]:
    params = _sanitize_trial_params(
        trial_params,
        meta_feature_profile=meta_feature_profile,
        policy_search_profile=policy_search_profile,
    )

    def _record_matches_trial(record: WarmStartCandidate) -> bool:
        record_params = _sanitize_trial_params(
            record.params,
            meta_feature_profile=meta_feature_profile,
            policy_search_profile=policy_search_profile,
        )
        if record_params == params:
            return True
        # Some source summaries carry fixed route fields that Optuna no longer
        # samples because the production profile clamps them outside the search
        # surface.  Treat the record as the provenance source when every sampled
        # coordinate matches and the record only has extra fixed coordinates.
        return bool(params) and all(record_params.get(key) == value for key, value in params.items())

    matches = [record for record in enqueued_records if _record_matches_trial(record)]
    default_match = bool(
        enqueue_default
        and params
        == _sanitize_trial_params(
            default_trial_params(policy_search_profile, meta_feature_profile=meta_feature_profile),
            meta_feature_profile=meta_feature_profile,
            policy_search_profile=policy_search_profile,
        )
    )
    candidate_payloads = [_warm_start_candidate_json(record) for record in matches]
    missing_hashes = [
        str(item.get("source_id"))
        for item in candidate_payloads
        if item.get("source_artifact_path") in {None, ""} or item.get("source_sha256") in {None, ""}
    ]
    if matches:
        status = "present"
        enabled = True
        source_type = "matched_enqueued_warm_start"
    elif default_match:
        status = "default_seed"
        enabled = False
        source_type = "default_trial_params"
    else:
        status = "sampled_no_warm_start"
        enabled = False
        source_type = "optuna_sampler_or_unmatched_enqueue"
    cthc_ready = not missing_hashes
    return {
        "schema": "phase3_trial_warm_start_provenance_v1",
        "status": status,
        "required_for_cthc": bool(matches),
        "cthc_ready": bool(cthc_ready),
        "enabled": enabled,
        "source_type": source_type,
        "source_id": candidate_payloads[0].get("source_id") if candidate_payloads else None,
        "source_artifact_path": candidate_payloads[0].get("source_artifact_path") if candidate_payloads else None,
        "source_sha256": candidate_payloads[0].get("source_sha256") if candidate_payloads else None,
        "candidate_count": len(candidate_payloads),
        "candidates": _jsonable(candidate_payloads),
        "skipped_warm_start_count": len(tuple(warm_start_skips)),
        "skipped_warm_start_records": _jsonable([_warm_start_skip_json(record) for record in warm_start_skips]),
        "missing_source_hashes": missing_hashes,
    }


def _dedupe_warm_start_candidates(
    records: Sequence[WarmStartCandidate],
    *,
    enqueue_default: bool,
    policy_search_profile: str | None = "default",
    meta_feature_profile: str | None = _DEFAULT_META_FEATURE_PROFILE,
) -> tuple[tuple[WarmStartCandidate, ...], tuple[WarmStartSkip, ...]]:
    seen: dict[str, WarmStartCandidate] = {}
    skips: list[WarmStartSkip] = []
    defaults = _sanitize_trial_params(
        default_trial_params(policy_search_profile, meta_feature_profile=meta_feature_profile),
        meta_feature_profile=meta_feature_profile,
        policy_search_profile=policy_search_profile,
    )
    for record in records:
        # Historical summaries predate newly introduced policy dimensions.  A
        # warm start must keep old winning coordinates and fill new knobs with
        # deterministic route defaults; otherwise Optuna samples the missing
        # dimensions randomly and the first "warm-start" trial is not a replay.
        params = _sanitize_trial_params(
            {**defaults, **dict(record.params)},
            meta_feature_profile=meta_feature_profile,
            policy_search_profile=policy_search_profile,
        )
        key = json.dumps(params, sort_keys=True)
        if enqueue_default and params == defaults:
            skips.append(
                WarmStartSkip(
                    source_kind=record.source_kind,
                    source_id=record.source_id,
                    benchmark_id=record.benchmark_id,
                    family=record.family,
                    reason="duplicates_default_seed",
                    source_payload=record.source_payload,
                )
            )
            continue
        if key in seen:
            skips.append(
                WarmStartSkip(
                    source_kind=record.source_kind,
                    source_id=record.source_id,
                    benchmark_id=record.benchmark_id,
                    family=record.family,
                    reason="duplicate_params",
                    detail=seen[key].source_id,
                    source_payload=record.source_payload,
                )
            )
            continue
        seen[key] = replace(record, params=params)
    return tuple(seen.values()), tuple(skips)


def _historical_row_compatible_with_spec(
    row: Mapping[str, Any],
    spec: HamiltonianBenchmarkSpec,
    *,
    strict_physics: bool = False,
) -> tuple[bool, str | None, tuple[str, ...]]:
    warnings: list[str] = []
    if not row.get("warm_start_eligible", True):
        return False, str(row.get("warm_start_exclusion_reason") or "warm_start_ineligible"), ()
    row_family = row.get("problem") or row.get("family")
    if row_family not in {None, ""} and str(row_family) != str(spec.family):
        return False, "family_mismatch", ()
    row_l = row.get("L")
    if row_l not in {None, ""}:
        try:
            if int(float(str(row_l))) != int(spec.features.L):
                return False, "L_mismatch", ()
        except Exception:
            pass
    spec_nph = _spec_local_n_ph_max(spec)
    row_nph = _row_n_ph_max(row)
    if spec_nph is not None and row_nph is not None and int(spec_nph) != int(row_nph):
        return False, "local_n_ph_max_mismatch", ()
    pool = row.get("pool") or row.get("adapt_pool")
    if pool not in {None, "", "full_meta"}:
        warnings.append(f"pool_coerced_to_full_meta:{pool}")
    continuation = row.get("continuation") or row.get("continuation_mode") or row.get("adapt_continuation_mode")
    if continuation not in {None, "", "phase3_v1"}:
        return False, "continuation_mismatch", ()
    command_options = _parse_cli_option_map(row.get("command"))
    target_options = _args_option_map(spec.base_pipeline_args)
    for key in ("t", "u", "dv", "omega0", "g_ep", "v_nn", "t_prime", "boundary", "ordering"):
        if key in command_options and key in target_options and not _option_values_equivalent(command_options.get(key), target_options.get(key)):
            if strict_physics:
                return False, f"physics_{key}_mismatch", ()
            warnings.append(f"physics_{key}_mismatch:{command_options.get(key)}->{target_options.get(key)}")
    opt = row.get("inner_optimizer")
    if opt not in {None, "", _ACTIVE_INNER_OPTIMIZER}:
        warnings.append(f"inner_optimizer_coerced_to_spsa:{opt}")
    return True, None, tuple(warnings)


def historical_warm_start_records_for_specs(
    specs: Sequence[HamiltonianBenchmarkSpec],
    ledger: Mapping[str, Any] | None,
) -> tuple[dict[str, tuple[WarmStartCandidate, ...]], dict[str, tuple[WarmStartSkip, ...]]]:
    records_by_benchmark: dict[str, list[WarmStartCandidate]] = {spec.benchmark_id: [] for spec in specs}
    skips_by_benchmark: dict[str, list[WarmStartSkip]] = {spec.benchmark_id: [] for spec in specs}
    if ledger is None:
        return {k: tuple(v) for k, v in records_by_benchmark.items()}, {k: tuple(v) for k, v in skips_by_benchmark.items()}
    ledger_path = ledger.get("_source_artifact_path") if isinstance(ledger, Mapping) else None
    ledger_sha256 = ledger.get("_source_sha256") if isinstance(ledger, Mapping) else None
    for spec in specs:
        row = _historical_row_for_spec(ledger, spec)
        if row is None:
            continue
        source_id = _row_source_id(row, f"{spec.benchmark_id}:historical_ledger")
        ok, reason, warnings = _historical_row_compatible_with_spec(row, spec)
        if not ok:
            skips_by_benchmark[spec.benchmark_id].append(
                WarmStartSkip(
                    source_kind="historical_ledger",
                    source_id=source_id,
                    benchmark_id=spec.benchmark_id,
                    family=spec.family,
                    reason=str(reason or "incompatible"),
                    source_payload={
                        **{k: row.get(k) for k in ("artifact_json", "abs_delta_e", "count_2q", "circuit_depth", "n_ph_max", "pool", "adapt_pool")},
                        "artifact_digest": _single_artifact_digest_payload(row.get("artifact_json")),
                        "ledger_path": ledger_path,
                        "ledger_sha256": ledger_sha256,
                        "source_artifact_path": ledger_path,
                        "source_sha256": ledger_sha256,
                    },
                )
            )
            continue
        params, audit = _trial_params_from_historical_row_with_audit(row)
        if params is None:
            skips_by_benchmark[spec.benchmark_id].append(
                WarmStartSkip(
                    source_kind="historical_ledger",
                    source_id=source_id,
                    benchmark_id=spec.benchmark_id,
                    family=spec.family,
                    reason=str(audit.get("reason") or "no_params"),
                    source_payload={
                        "artifact_json": row.get("artifact_json"),
                        "artifact_digest": _single_artifact_digest_payload(row.get("artifact_json")),
                        "ledger_path": ledger_path,
                        "ledger_sha256": ledger_sha256,
                        "source_artifact_path": ledger_path,
                        "source_sha256": ledger_sha256,
                        "abs_delta_e": row.get("abs_delta_e"),
                        "count_2q": row.get("count_2q"),
                        "circuit_depth": row.get("circuit_depth"),
                        "audit": audit,
                    },
                )
            )
            continue
        try:
            score = float(row.get("abs_delta_e"))
        except Exception:
            score = None
        records_by_benchmark[spec.benchmark_id].append(
            WarmStartCandidate(
                params=params,
                source_kind="historical_ledger",
                source_id=source_id,
                benchmark_id=spec.benchmark_id,
                family=spec.family,
                source_score=score,
                source_payload={
                    "artifact_json": row.get("artifact_json"),
                    "artifact_digest": _single_artifact_digest_payload(row.get("artifact_json")),
                    "ledger_path": ledger_path,
                    "ledger_sha256": ledger_sha256,
                    "source_artifact_path": ledger_path,
                    "source_sha256": ledger_sha256,
                    "abs_delta_e": row.get("abs_delta_e"),
                    "count_2q": row.get("count_2q"),
                    "circuit_depth": row.get("circuit_depth"),
                    "n_ph_max": row.get("n_ph_max"),
                    "exact_reference_n_ph_max": row.get("exact_reference_n_ph_max"),
                    "static_route_id": row.get("static_route_id"),
                    "suite_profile": row.get("suite_profile"),
                    "phase0_aware": row.get("phase0_aware"),
                    "audit": audit,
                },
                compatibility_warnings=warnings,
            )
        )
    return {k: tuple(v) for k, v in records_by_benchmark.items()}, {k: tuple(v) for k, v in skips_by_benchmark.items()}


def apply_historical_ledger_to_specs(
    specs: Sequence[HamiltonianBenchmarkSpec],
    ledger: Mapping[str, Any] | None,
) -> tuple[HamiltonianBenchmarkSpec, ...]:
    if ledger is None:
        return tuple(specs)
    out: list[HamiltonianBenchmarkSpec] = []
    for spec in specs:
        row = _historical_row_for_spec(ledger, spec)
        if row is None:
            out.append(spec)
            continue
        compatible, _reason, _warnings = _historical_row_compatible_with_spec(row, spec, strict_physics=True)
        if not compatible:
            out.append(spec)
            continue
        abs_delta_e = row.get("abs_delta_e")
        try:
            baseline_abs_delta_e = float(abs_delta_e)
        except Exception:
            out.append(spec)
            continue
        spec_nph = _spec_local_n_ph_max(spec)
        row_nph = _row_n_ph_max(row)
        if spec_nph is not None and row_nph is not None and int(row_nph) != int(spec_nph):
            out.append(spec)
            continue
        tags = tuple(spec.tags) + ("historical_ledger_baseline",)
        out.append(
            replace(
                spec,
                baseline_abs_delta_e=max(baseline_abs_delta_e, 1e-12),
                baseline_count_2q=row.get("count_2q") if row.get("count_2q") is not None else spec.baseline_count_2q,
                baseline_depth_2q=row.get("circuit_depth") if row.get("circuit_depth") is not None else spec.baseline_depth_2q,
                baseline_artifact_json=str(row.get("artifact_json")) if row.get("artifact_json") else None,
                baseline_source="historical_ledger",
                tags=tags,
            )
        )
    return tuple(out)


def historical_trial_params_for_specs(
    specs: Sequence[HamiltonianBenchmarkSpec],
    ledger: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    records_by_benchmark, _skips = historical_warm_start_records_for_specs(specs, ledger)
    params_by_key: dict[str, dict[str, Any]] = {}
    for records in records_by_benchmark.values():
        for record in records:
            params = _sanitize_trial_params(record.params)
            key = json.dumps(params, sort_keys=True)
            params_by_key[key] = params
    return tuple(params_by_key.values())


def _oracle_summary_param_records_from_payload(payload: Mapping[str, Any]) -> tuple[tuple[float, dict[str, Any]], ...]:
    """Extract scored Optuna trial-parameter warm starts from summaries."""

    out: list[tuple[float, dict[str, Any]]] = []
    summaries = payload.get("summaries")
    if isinstance(summaries, Mapping):
        for summary in summaries.values():
            if isinstance(summary, Mapping):
                out.extend(_oracle_summary_param_records_from_payload(summary))
        return tuple(out)

    params = payload.get("best_params")
    if isinstance(params, Mapping) and params:
        try:
            value = float(payload.get("best_value"))
        except Exception:
            value = _LARGE_OBJECTIVE
        out.append((value if math.isfinite(value) else _LARGE_OBJECTIVE, _sanitize_trial_params(dict(params))))
    return tuple(out)


def oracle_summary_trial_params(paths: Sequence[str | Path] | None, *, limit: int | None = None) -> tuple[dict[str, Any], ...]:
    """Load best trial params from prior oracle-grid summary files/directories."""

    if not paths:
        return ()
    records_by_key: dict[str, tuple[float, dict[str, Any]]] = {}
    for raw_path in paths:
        if raw_path in {None, ""}:
            continue
        root = Path(raw_path)
        candidates = (
            (root,)
            if root.is_file()
            else tuple(sorted({*root.rglob("summary.json"), *root.rglob("*_summary.json")}))
        )
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except Exception as exc:
                raise ValueError(f"Failed to load oracle summary {path}: {exc}") from exc
            if not isinstance(payload, Mapping):
                continue
            for value, params in _oracle_summary_param_records_from_payload(payload):
                if not params:
                    continue
                key = json.dumps(params, sort_keys=True)
                old = records_by_key.get(key)
                if old is None or float(value) < float(old[0]):
                    records_by_key[key] = (float(value), params)
    ordered = [params for _value, params in sorted(records_by_key.values(), key=lambda item: item[0])]
    if limit is not None and int(limit) >= 0:
        ordered = ordered[: int(limit)]
    return tuple(ordered)


def _summary_benchmark_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    benchmarks = payload.get("benchmarks")
    if isinstance(benchmarks, Sequence) and not isinstance(benchmarks, (str, bytes)) and benchmarks:
        first = benchmarks[0]
        if isinstance(first, Mapping):
            return first
    return {}


def _payload_local_n_ph_max(payload: Mapping[str, Any]) -> int | None:
    args = payload.get("base_pipeline_args")
    if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
        try:
            return _row_n_ph_max({"n_ph_max": _pipeline_arg_value(tuple(str(x) for x in args), "--n-ph-max")})
        except Exception:
            return None
    return None


def _oracle_summary_walk(payload: Mapping[str, Any], *, source_id: str, summary_key: str | None = None) -> tuple[tuple[str | None, Mapping[str, Any]], ...]:
    summaries = payload.get("summaries")
    if isinstance(summaries, Mapping):
        out: list[tuple[str | None, Mapping[str, Any]]] = []
        for key, summary in summaries.items():
            if isinstance(summary, Mapping):
                out.extend(_oracle_summary_walk(summary, source_id=source_id, summary_key=str(key)))
        return tuple(out)
    if isinstance(payload.get("best_params"), Mapping) and payload.get("best_params"):
        return ((summary_key, payload),)
    return ()


def _summary_nested_get(payload: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value = _path_get(payload, path)
        if value is not _MISSING:
            return value
    return None


def _summary_static_route_id(summary: Mapping[str, Any], root_payload: Mapping[str, Any]) -> str | None:
    value = _summary_nested_get(
        summary,
        "static_route_id",
        "objective_provenance.static_route_id",
        "base_policy.static.static_route_id",
    )
    if value in {None, ""}:
        value = _summary_nested_get(
            root_payload,
            "static_route_id",
            "objective_provenance.static_route_id",
            "base_policy.static.static_route_id",
        )
    if value in {None, ""}:
        return None
    return normalize_static_route_id(value, default=ROUTE_ID_UNSPECIFIED)


def _summary_suite_profile(summary: Mapping[str, Any], root_payload: Mapping[str, Any]) -> str | None:
    value = _summary_nested_get(summary, "suite_profile", "metadata.suite_profile")
    if value in {None, ""}:
        value = _summary_nested_get(root_payload, "suite_profile", "metadata.suite_profile")
    return None if value in {None, ""} else str(value).strip().lower().replace("-", "_")


def _summary_phase0_aware(summary: Mapping[str, Any], root_payload: Mapping[str, Any]) -> bool:
    for payload in (summary, root_payload):
        value = _summary_nested_get(payload, "phase0_aware", "phase0.phase0_aware")
        if value not in {None, ""}:
            return _coerce_bool(value)
        best_params = payload.get("best_params")
        if isinstance(best_params, Mapping) and any(str(key).startswith("phase0_") for key in best_params):
            return True
        base_static = _summary_nested_get(payload, "base_policy.static")
        if isinstance(base_static, Mapping) and any(str(key).startswith("phase0_") for key in base_static):
            return True
    return False


def _normalize_reference_nph(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        out = int(float(str(value)))
    except Exception:
        return None
    return None if out <= 0 else int(out)


def oracle_summary_warm_start_records_for_specs(
    paths: Sequence[str | Path] | None,
    specs: Sequence[HamiltonianBenchmarkSpec],
    *,
    limit_per_benchmark: int | None = None,
    required_static_route_id: str | None = None,
    required_suite_profile: str | None = None,
    require_phase0_aware: bool = False,
    require_compatible_warm_starts: bool = False,
) -> tuple[dict[str, tuple[WarmStartCandidate, ...]], dict[str, tuple[WarmStartSkip, ...]]]:
    records_by_benchmark: dict[str, list[WarmStartCandidate]] = {spec.benchmark_id: [] for spec in specs}
    skips_by_benchmark: dict[str, list[WarmStartSkip]] = {spec.benchmark_id: [] for spec in specs}
    required_route = (
        None
        if required_static_route_id in {None, ""}
        else normalize_static_route_id(required_static_route_id, default=ROUTE_ID_UNSPECIFIED)
    )
    required_profile = (
        None
        if required_suite_profile in {None, ""}
        else str(required_suite_profile).strip().lower().replace("-", "_")
    )
    if not paths:
        return {k: tuple(v) for k, v in records_by_benchmark.items()}, {k: tuple(v) for k, v in skips_by_benchmark.items()}
    specs_by_id = {spec.benchmark_id: spec for spec in specs}
    for raw_path in paths:
        if raw_path in {None, ""}:
            continue
        root = Path(raw_path)
        candidates = (
            (root,)
            if root.is_file()
            else tuple(sorted({*root.rglob("summary.json"), *root.rglob("*_summary.json")}))
        )
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except Exception as exc:
                raise ValueError(f"Failed to load oracle summary {path}: {exc}") from exc
            if not isinstance(payload, Mapping):
                continue
            summary_digest = _single_artifact_digest_payload(path)
            for summary_key, summary in _oracle_summary_walk(payload, source_id=str(path)):
                bench = _summary_benchmark_payload(summary)
                bench_id = str(bench.get("benchmark_id") or summary_key or "")
                target = specs_by_id.get(bench_id)
                if target is None:
                    family = bench.get("family")
                    matches = [spec for spec in specs if family in {None, "", spec.family}]
                    if len(matches) == 1:
                        target = matches[0]
                if target is None:
                    for spec in specs:
                        skips_by_benchmark[spec.benchmark_id].append(
                            WarmStartSkip(
                                source_kind="oracle_summary",
                                source_id=str(path),
                                benchmark_id=spec.benchmark_id,
                                family=spec.family,
                                reason="oracle_summary_benchmark_mismatch",
                                detail=bench_id or summary_key,
                            )
                        )
                    continue
                family = bench.get("family")
                if family not in {None, "", target.family}:
                    skips_by_benchmark[target.benchmark_id].append(
                        WarmStartSkip("oracle_summary", str(path), "family_mismatch", target.benchmark_id, target.family, str(family))
                    )
                    continue
                bench_l = bench.get("features", {}).get("L") if isinstance(bench.get("features"), Mapping) else bench.get("L")
                if bench_l not in {None, ""}:
                    try:
                        if int(float(str(bench_l))) != int(target.features.L):
                            skips_by_benchmark[target.benchmark_id].append(
                                WarmStartSkip("oracle_summary", str(path), "L_mismatch", target.benchmark_id, target.family, str(bench_l))
                            )
                            continue
                    except Exception:
                        pass
                source_nph = _payload_local_n_ph_max(bench)
                target_nph = _spec_local_n_ph_max(target)
                if source_nph is not None and target_nph is not None and int(source_nph) != int(target_nph):
                    skips_by_benchmark[target.benchmark_id].append(
                        WarmStartSkip("oracle_summary", str(path), "local_n_ph_max_mismatch", target.benchmark_id, target.family, str(source_nph))
                    )
                    continue
                if required_route is not None:
                    source_route = _summary_static_route_id(summary, payload)
                    if source_route != required_route:
                        skips_by_benchmark[target.benchmark_id].append(
                            WarmStartSkip(
                                "oracle_summary",
                                str(path),
                                "static_route_id_mismatch",
                                target.benchmark_id,
                                target.family,
                                str(source_route),
                            )
                        )
                        continue
                if required_profile is not None:
                    source_profile = _summary_suite_profile(summary, payload)
                    if source_profile != required_profile:
                        skips_by_benchmark[target.benchmark_id].append(
                            WarmStartSkip(
                                "oracle_summary",
                                str(path),
                                "suite_profile_mismatch",
                                target.benchmark_id,
                                target.family,
                                str(source_profile),
                            )
                        )
                        continue
                if require_phase0_aware and not _summary_phase0_aware(summary, payload):
                    skips_by_benchmark[target.benchmark_id].append(
                        WarmStartSkip(
                            "oracle_summary",
                            str(path),
                            "phase0_awareness_missing",
                            target.benchmark_id,
                            target.family,
                            summary_key or bench_id,
                        )
                    )
                    continue
                if require_compatible_warm_starts:
                    source_ref_nph = _normalize_reference_nph(bench.get("exact_reference_n_ph_max"))
                    target_ref_nph = _normalize_reference_nph(target.exact_reference_n_ph_max)
                    if source_ref_nph != target_ref_nph:
                        skips_by_benchmark[target.benchmark_id].append(
                            WarmStartSkip(
                                "oracle_summary",
                                str(path),
                                "exact_reference_n_ph_max_mismatch",
                                target.benchmark_id,
                                target.family,
                                f"{source_ref_nph}!={target_ref_nph}",
                            )
                        )
                        continue
                params = _sanitize_trial_params(dict(summary.get("best_params") or {}))
                if not params:
                    continue
                try:
                    value = float(summary.get("best_value"))
                except Exception:
                    value = None
                records_by_benchmark[target.benchmark_id].append(
                    WarmStartCandidate(
                        params=params,
                        source_kind="oracle_summary",
                        source_id=f"{path}:{summary_key or bench_id or target.benchmark_id}",
                        benchmark_id=target.benchmark_id,
                        family=target.family,
                        source_score=value,
                        source_payload={
                            "summary_path": str(path),
                            "summary_digest": summary_digest,
                            "source_artifact_path": str(path),
                            "source_sha256": summary_digest.get("sha256"),
                            "summary_key": summary_key,
                            "best_value": summary.get("best_value"),
                            "best_trial_number": summary.get("best_trial_number"),
                            "static_route_id": _summary_static_route_id(summary, payload),
                            "suite_profile": _summary_suite_profile(summary, payload),
                            "phase0_aware": _summary_phase0_aware(summary, payload),
                        },
                    )
                )
    if limit_per_benchmark is not None and int(limit_per_benchmark) >= 0:
        for key, records in list(records_by_benchmark.items()):
            records.sort(key=lambda rec: _LARGE_OBJECTIVE if rec.source_score is None else float(rec.source_score))
            records_by_benchmark[key] = records[: int(limit_per_benchmark)]
    return {k: tuple(v) for k, v in records_by_benchmark.items()}, {k: tuple(v) for k, v in skips_by_benchmark.items()}


def _call_benchmark_runner(
    runner: Any,
    spec: HamiltonianBenchmarkSpec,
    policy: AlgorithmPolicy,
    *,
    output_dir: Path,
    run_lifecycle: _ManagedOptunaRunLifecycle | None = None,
    trial_number: int | None = None,
) -> BenchmarkResult:
    kwargs: dict[str, Any] = {"output_dir": output_dir}
    if run_lifecycle is not None or trial_number is not None:
        try:
            signature = inspect.signature(runner)
            accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
            accepts_lifecycle = accepts_kwargs or "run_lifecycle" in signature.parameters
            accepts_trial_number = accepts_kwargs or "trial_number" in signature.parameters
        except (TypeError, ValueError):
            accepts_lifecycle = False
            accepts_trial_number = False
        if run_lifecycle is not None and accepts_lifecycle:
            kwargs["run_lifecycle"] = run_lifecycle
        if trial_number is not None and accepts_trial_number:
            kwargs["trial_number"] = trial_number
    return runner(spec, policy, **kwargs)


def _finite_objective_component(value: Any, *, default: float = _LARGE_OBJECTIVE) -> float:
    parsed = _as_float_or_none(value)
    if parsed is None or not math.isfinite(float(parsed)):
        return float(default)
    return float(max(0.0, float(parsed)))


def same_cutoff_pareto_vector_for_result(result: BenchmarkResult | None) -> tuple[float, ...]:
    """Return the no-target same-cutoff Pareto objective vector for one benchmark result."""

    if result is None or not bool(result.success):
        return tuple(float(_LARGE_OBJECTIVE) for _ in _SAME_CUTOFF_PARETO_OBJECTIVE_NAMES)
    same_cutoff_error = result.abs_delta_e_same_cutoff
    if same_cutoff_error is None:
        same_cutoff_error = result.abs_delta_e
    parameter_count = result.runtime_parameter_count
    if parameter_count is None:
        parameter_count = result.parameter_count
    return (
        _finite_objective_component(same_cutoff_error),
        _finite_objective_component(result.count_2q),
        _finite_objective_component(result.depth_2q),
        _finite_objective_component(result.circuit_depth),
        _finite_objective_component(parameter_count),
        _finite_objective_component(result.shot_cost_proxy, default=0.0),
    )


def same_cutoff_pareto_score_components(result: BenchmarkResult | None) -> dict[str, Any]:
    values = same_cutoff_pareto_vector_for_result(result)
    return {
        "schema": "phase3_same_cutoff_pareto_score_components_v1",
        "multi_objective_mode": _MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO,
        "objective_vector_names": list(_SAME_CUTOFF_PARETO_OBJECTIVE_NAMES),
        "objective_vector": list(values),
        "success": False if result is None else bool(result.success),
    }


def objective_oracle(
    trial: Any,
    spec: HamiltonianBenchmarkSpec,
    *,
    output_dir: Path,
    seed: int = 7,
    runner: Any = run_static_benchmark,
    objective_weights: StaticObjectiveWeights = StaticObjectiveWeights(),
    config: GlobalObjectiveConfig | None = None,
    run_lifecycle: _ManagedOptunaRunLifecycle | None = None,
    policy_search_profile: str | None = "default",
    meta_feature_profile: str | None = _DEFAULT_META_FEATURE_PROFILE,
    base_policy: AlgorithmPolicy | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    suite_profile: str | None = "standard",
    enqueued_records: Sequence[WarmStartCandidate] = (),
    warm_start_skips: Sequence[WarmStartSkip] = (),
    enqueue_default: bool = False,
    trial_param_overrides: Mapping[str, Any] | None = None,
) -> float | tuple[float, ...]:
    effective_config = config
    effective_weights = effective_config.weights if effective_config is not None else objective_weights
    objective_mode = (
        _normalize_discovery_objective_mode(effective_config.discovery_objective_mode)
        if effective_config is not None
        else _DISCOVERY_OBJECTIVE_MODE_TERMINAL_PROXY
    )
    policy = sample_policy_from_trial(
        trial,
        base_policy=base_policy,
        policy_search_profile=policy_search_profile,
        meta_feature_profile=meta_feature_profile,
        trial_param_overrides=trial_param_overrides,
    )
    trial_number = int(getattr(trial, "number", seed))
    sampled_params = dict(getattr(trial, "params", {}) or {})
    warm_start_provenance = _trial_warm_start_provenance_payload(
        sampled_params,
        enqueued_records=enqueued_records,
        warm_start_skips=warm_start_skips,
        enqueue_default=enqueue_default,
        policy_search_profile=policy_search_profile,
        meta_feature_profile=meta_feature_profile,
    )
    trial_dir = Path(output_dir) / f"trial_{trial_number:04d}"
    manifest_path = trial_dir / "effective_trial_manifest.json"
    manifest = build_effective_trial_manifest_intent(
        mode="oracle",
        trial=trial,
        specs=(spec,),
        trial_dir=trial_dir,
        policy=policy,
        sampled_params=sampled_params,
        seed=seed,
        objective_weights=effective_weights,
        config=effective_config,
        study_name=study_name,
        storage=storage,
        suite_profile=suite_profile,
        benchmarks_per_trial_jobs=1,
        runner=runner,
        warm_start_provenance=warm_start_provenance,
    )
    _set_effective_trial_manifest_user_attrs(trial, _write_effective_trial_manifest(manifest_path, manifest))
    try:
        result = _call_benchmark_runner(
            runner,
            spec,
            policy,
            output_dir=trial_dir,
            run_lifecycle=run_lifecycle,
            trial_number=trial_number,
        )
        result = _ensure_policy_roundtrip_audit(
            result,
            sampled_params=sampled_params,
            policy=policy,
            spec=spec,
            output_dir=trial_dir,
            objective_weights=effective_weights,
        )
        result = _ensure_paper_i_result_artifacts(result, spec)
        if (effective_config is not None and _normalize_multi_objective_mode(effective_config.multi_objective_mode) == _MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO):
            score_components = same_cutoff_pareto_score_components(result)
            score = tuple(float(x) for x in score_components["objective_vector"])
        elif objective_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING:
            score_components = discovery_first_crossing_score_components(result, spec, effective_config or GlobalObjectiveConfig())
            score = float(score_components["score"])
        else:
            score = normalized_static_score(result, spec, effective_weights)
            score_components = {"score": float(score), "objective_mode": "legacy_normalized_static_score_v1"}
        manifest = finalize_effective_trial_manifest(
            manifest,
            results={spec.benchmark_id: result},
            score=(score_components.get("score") if isinstance(score_components, Mapping) else None),
            global_score_components=score_components,
        )
        pointer = _write_effective_trial_manifest(manifest_path, manifest)
        _set_effective_trial_manifest_user_attrs(trial, pointer)
    except Exception as exc:
        manifest = finalize_effective_trial_manifest(
            manifest,
            results={},
            score=None,
            failure_reason=f"trial_exception:{type(exc).__name__}:{exc}",
        )
        pointer = _write_effective_trial_manifest(manifest_path, manifest)
        _set_effective_trial_manifest_user_attrs(trial, pointer)
        raise
    if hasattr(trial, "set_user_attr"):
        trial.set_user_attr("policy", asdict(policy))
        trial.set_user_attr("meta_feature_profile", str(policy.static.static_meta_feature_profile))
        trial.set_user_attr("meta_feature_policy", _jsonable(_meta_feature_policy_payload(policy)))
        trial.set_user_attr("promotion_label", str(_meta_feature_policy_payload(policy).get("promotion_label")))
        trial.set_user_attr("fixed_inner_optimizer", _ACTIVE_INNER_OPTIMIZER)
        trial.set_user_attr("inner_optimizer_policy", _INNER_OPTIMIZER_POLICY_LABEL)
        trial.set_user_attr("policy_roundtrip_audit", result.policy_roundtrip_audit)
        trial.set_user_attr(
            "policy_roundtrip_audit_summary",
            _summarize_policy_roundtrip_audits(
                (result.policy_roundtrip_audit,) if isinstance(result.policy_roundtrip_audit, Mapping) else ()
            ),
        )
        trial.set_user_attr("result", asdict(result))
        trial.set_user_attr("discovery_objective_mode", objective_mode)
        trial.set_user_attr("objective_score_components", _jsonable(score_components))
        _set_trial_telemetry_attrs(trial, score_components)
        trial.set_user_attr("physical_target_manifest", _jsonable(result.physical_target_manifest))
        trial.set_user_attr("cutoff_diagnostics", _jsonable(result.cutoff_diagnostics))
        trial.set_user_attr("paper_i_first_crossing", _jsonable(result.paper_i_first_crossing))
    return score if isinstance(score, tuple) else float(score)


def _benchmark_exception_result(spec: HamiltonianBenchmarkSpec, exc: BaseException) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_id=spec.benchmark_id,
        family=spec.family,
        success=False,
        abs_delta_e=None,
        failure_reason=f"benchmark_exception:{type(exc).__name__}:{exc}",
    )


def _run_global_benchmarks_for_policy(
    specs: Sequence[HamiltonianBenchmarkSpec],
    policy: AlgorithmPolicy,
    *,
    trial_dir: Path,
    runner: Any,
    benchmarks_per_trial_jobs: int = 1,
    run_lifecycle: _ManagedOptunaRunLifecycle | None = None,
    trial_number: int | None = None,
    progress_reporter: _ProgressReporter | None = None,
    benchmark_policies: Mapping[str, AlgorithmPolicy] | None = None,
) -> dict[str, BenchmarkResult]:
    selected = tuple(specs)
    benchmark_policy_by_id = {str(key): value for key, value in dict(benchmark_policies or {}).items()}
    if not selected:
        return {}
    jobs = int(benchmarks_per_trial_jobs)
    if jobs <= 0:
        jobs = len(selected)
    jobs = max(1, min(jobs, len(selected)))

    def _run_one(spec: HamiltonianBenchmarkSpec) -> BenchmarkResult:
        started = time.perf_counter()
        if progress_reporter is not None:
            effective_policy = benchmark_policy_by_id.get(str(spec.benchmark_id), policy)
            progress_reporter.append_event(
                "benchmark_started",
                trial_number=trial_number,
                benchmark_id=spec.benchmark_id,
                family=spec.family,
                adapt_max_depth=int(effective_policy.static.adapt_max_depth),
            )
            progress_reporter.write_current(
                state="benchmark_running",
                trial_number=trial_number,
                benchmark_id=spec.benchmark_id,
                family=spec.family,
            )
        try:
            effective_policy = benchmark_policy_by_id.get(str(spec.benchmark_id), policy)
            result = _call_benchmark_runner(
                runner,
                spec,
                effective_policy,
                output_dir=trial_dir,
                run_lifecycle=run_lifecycle,
                trial_number=trial_number,
            )
        except (KeyboardInterrupt, SystemExit):
            if progress_reporter is not None:
                progress_reporter.append_event(
                    "benchmark_interrupted",
                    trial_number=trial_number,
                    benchmark_id=spec.benchmark_id,
                    elapsed_s=float(time.perf_counter() - started),
                )
            raise
        except BaseException as exc:  # pragma: no cover - defensive subprocess harness guard
            result = _benchmark_exception_result(spec, exc)
        if progress_reporter is not None:
            result_progress_payload = {
                "abs_delta_e": result.abs_delta_e,
                "delta_e": result.abs_delta_e,
                "energy": result.energy,
                "count_2q": result.count_2q,
                "depth_2q": result.depth_2q,
                "circuit_depth": result.circuit_depth,
                "parameter_count": result.parameter_count,
                "measurement_shots_proxy": result.measurement_shots_proxy,
                "shot_cost_proxy": result.shot_cost_proxy,
                "stop_reason": result.stop_reason,
            }
            progress_reporter.append_event(
                "benchmark_completed",
                trial_number=trial_number,
                benchmark_id=spec.benchmark_id,
                family=spec.family,
                success=bool(result.success),
                failure_reason=result.failure_reason,
                elapsed_s=float(time.perf_counter() - started),
                **result_progress_payload,
            )
            progress_reporter.write_current(
                state="benchmark_completed",
                trial_number=trial_number,
                benchmark_id=spec.benchmark_id,
                family=spec.family,
                success=bool(result.success),
                failure_reason=result.failure_reason,
                elapsed_s=float(time.perf_counter() - started),
                **result_progress_payload,
            )
        return result

    if jobs == 1:
        return {spec.benchmark_id: _run_one(spec) for spec in selected}

    results: dict[str, BenchmarkResult] = {}
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="phase3-benchmark") as pool:
        future_to_spec = {pool.submit(_run_one, spec): spec for spec in selected}
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                results[spec.benchmark_id] = future.result()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # pragma: no cover - defensive subprocess harness guard
                results[spec.benchmark_id] = _benchmark_exception_result(spec, exc)
    return {spec.benchmark_id: results[spec.benchmark_id] for spec in selected}


def objective_global_agnostic(
    trial: Any,
    train_suite: Sequence[HamiltonianBenchmarkSpec],
    *,
    output_dir: Path,
    seed: int = 7,
    runner: Any = run_static_benchmark,
    config: GlobalObjectiveConfig = GlobalObjectiveConfig(),
    benchmarks_per_trial_jobs: int = 1,
    run_lifecycle: _ManagedOptunaRunLifecycle | None = None,
    policy_search_profile: str | None = "default",
    meta_feature_profile: str | None = _DEFAULT_META_FEATURE_PROFILE,
    progress_reporter: _ProgressReporter | None = None,
    base_policy: AlgorithmPolicy | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    suite_profile: str | None = "standard",
    enqueued_records: Sequence[WarmStartCandidate] = (),
    warm_start_skips: Sequence[WarmStartSkip] = (),
    enqueue_default: bool = False,
    trial_param_overrides: Mapping[str, Any] | None = None,
    benchmark_trial_param_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> float:
    objective_mode = _normalize_discovery_objective_mode(config.discovery_objective_mode)
    policy = sample_policy_from_trial(
        trial,
        base_policy=base_policy,
        policy_search_profile=policy_search_profile,
        meta_feature_profile=meta_feature_profile,
        trial_param_overrides=trial_param_overrides,
    )
    benchmark_policies = _benchmark_effective_policies(policy, train_suite, benchmark_trial_param_overrides)
    trial_number = int(getattr(trial, "number", seed))
    sampled_params = dict(getattr(trial, "params", {}) or {})
    warm_start_provenance = _trial_warm_start_provenance_payload(
        sampled_params,
        enqueued_records=enqueued_records,
        warm_start_skips=warm_start_skips,
        enqueue_default=enqueue_default,
        policy_search_profile=policy_search_profile,
        meta_feature_profile=meta_feature_profile,
    )
    trial_dir = Path(output_dir) / f"trial_{trial_number:04d}"
    manifest_path = trial_dir / "effective_trial_manifest.json"
    manifest = build_effective_trial_manifest_intent(
        mode="global",
        trial=trial,
        specs=train_suite,
        trial_dir=trial_dir,
        policy=policy,
        sampled_params=sampled_params,
        seed=seed,
        config=config,
        study_name=study_name,
        storage=storage,
        suite_profile=suite_profile,
        benchmarks_per_trial_jobs=benchmarks_per_trial_jobs,
        runner=runner,
        warm_start_provenance=warm_start_provenance,
        benchmark_policies=benchmark_policies,
    )
    _set_effective_trial_manifest_user_attrs(trial, _write_effective_trial_manifest(manifest_path, manifest))
    try:
        results = _run_global_benchmarks_for_policy(
            train_suite,
            policy,
            trial_dir=trial_dir,
            runner=runner,
            benchmarks_per_trial_jobs=benchmarks_per_trial_jobs,
            run_lifecycle=run_lifecycle,
            trial_number=trial_number,
            progress_reporter=progress_reporter,
            benchmark_policies=benchmark_policies,
        )
        results = {
            spec.benchmark_id: _ensure_paper_i_result_artifacts(
                _ensure_policy_roundtrip_audit(
                    results[spec.benchmark_id],
                    sampled_params=sampled_params,
                    policy=benchmark_policies.get(str(spec.benchmark_id), policy),
                    spec=spec,
                    output_dir=trial_dir,
                    objective_weights=config.weights,
                ),
                spec,
            )
            for spec in train_suite
            if spec.benchmark_id in results
        }
        global_score_components = aggregate_global_score_components(results, train_suite, config)
        score = float(global_score_components["score"])
        if objective_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING:
            required_violations = {
                str(row.get("benchmark_id")): dict(row.get("discovery_first_crossing") or {})
                for row in global_score_components.get("per_benchmark_scores", [])
                if isinstance(row, Mapping)
                and isinstance(row.get("discovery_first_crossing"), Mapping)
                and not bool(row["discovery_first_crossing"].get("feasible", False))
            }
        else:
            required_violations = required_target_violations(results, train_suite, config)
        manifest = finalize_effective_trial_manifest(
            manifest,
            results=results,
            score=score,
            global_score_components=global_score_components,
            required_target_violations_payload=required_violations,
        )
        pointer = _write_effective_trial_manifest(manifest_path, manifest)
        _set_effective_trial_manifest_user_attrs(trial, pointer)
    except Exception as exc:
        manifest = finalize_effective_trial_manifest(
            manifest,
            results={},
            score=None,
            failure_reason=f"trial_exception:{type(exc).__name__}:{exc}",
        )
        pointer = _write_effective_trial_manifest(manifest_path, manifest)
        _set_effective_trial_manifest_user_attrs(trial, pointer)
        raise
    if hasattr(trial, "set_user_attr"):
        trial.set_user_attr("policy", asdict(policy))
        trial.set_user_attr("benchmark_policies", {key: asdict(value) for key, value in benchmark_policies.items()})
        trial.set_user_attr("meta_feature_profile", str(policy.static.static_meta_feature_profile))
        trial.set_user_attr("meta_feature_policy", _jsonable(_meta_feature_policy_payload(policy)))
        trial.set_user_attr("promotion_label", str(_meta_feature_policy_payload(policy).get("promotion_label")))
        trial.set_user_attr("fixed_inner_optimizer", _ACTIVE_INNER_OPTIMIZER)
        trial.set_user_attr("inner_optimizer_policy", _INNER_OPTIMIZER_POLICY_LABEL)
        trial.set_user_attr("benchmarks_per_trial_jobs", int(benchmarks_per_trial_jobs))
        audits = {
            key: value.policy_roundtrip_audit
            for key, value in results.items()
            if isinstance(value.policy_roundtrip_audit, Mapping)
        }
        trial.set_user_attr("policy_roundtrip_audits", audits)
        trial.set_user_attr("policy_roundtrip_audit_summary", _summarize_policy_roundtrip_audits(tuple(audits.values())))
        trial.set_user_attr("results", {key: asdict(value) for key, value in results.items()})
        trial.set_user_attr("required_target_violations", required_violations)
        trial.set_user_attr("discovery_objective_mode", objective_mode)
        trial.set_user_attr("physical_target_manifest", _jsonable({key: value.physical_target_manifest for key, value in results.items()}))
        trial.set_user_attr("cutoff_diagnostics", _jsonable({key: value.cutoff_diagnostics for key, value in results.items()}))
        trial.set_user_attr("paper_i_first_crossing", _jsonable({key: value.paper_i_first_crossing for key, value in results.items()}))
        trial.set_user_attr("objective_weighting", global_score_components["objective_weighting"])
        trial.set_user_attr("objective_provenance", global_score_components["objective_provenance"])
        trial.set_user_attr("global_score_components", global_score_components)
        trial.set_user_attr("global_score", score)
        _set_trial_telemetry_attrs(trial, global_score_components)
    return float(score)


def filter_static_benchmark_suite(
    specs: Sequence[HamiltonianBenchmarkSpec] | None = None,
    *,
    families: Sequence[str] | None = None,
    sizes: Sequence[int] | None = None,
    split: str | None = None,
    molecular_problem_json: str | Path | None = None,
    boson_cutoff: int | None = None,
    boson_cutoffs: Sequence[int] | None = None,
    exact_reference_boson_cutoff: int | None = 4,
    physics_grid_profile: str = "canonical",
) -> tuple[HamiltonianBenchmarkSpec, ...]:
    suite = (
        tuple(specs)
        if specs is not None
        else default_static_benchmark_suite(
            split=split or "train",
            molecular_problem_json=molecular_problem_json,
            boson_cutoff=boson_cutoff,
            boson_cutoffs=boson_cutoffs,
            exact_reference_boson_cutoff=exact_reference_boson_cutoff,
            physics_grid_profile=physics_grid_profile,
        )
    )
    family_set = {str(x) for x in families} if families is not None else None
    size_set = {int(x) for x in sizes} if sizes is not None else None
    out: list[HamiltonianBenchmarkSpec] = []
    for spec in suite:
        if family_set is not None and spec.family not in family_set:
            continue
        if size_set is not None and int(spec.features.L) not in size_set:
            continue
        if split is not None and spec.split != split:
            continue
        out.append(spec)
    return tuple(out)


def filter_static_benchmark_suite_by_ids(
    specs: Sequence[HamiltonianBenchmarkSpec],
    benchmark_ids: Sequence[str] | None,
) -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Select exact benchmark IDs from an already-materialized suite, preserving requested order."""

    requested = tuple(str(x).strip() for x in (benchmark_ids or ()) if str(x).strip())
    if not requested:
        return tuple(specs)
    by_id = {str(spec.benchmark_id): spec for spec in specs}
    missing = [benchmark_id for benchmark_id in requested if benchmark_id not in by_id]
    if missing:
        available = ", ".join(sorted(by_id))
        raise ValueError(f"Requested benchmark_id(s) not present in selected suite: {missing}; available: {available}")
    return tuple(by_id[benchmark_id] for benchmark_id in dict.fromkeys(requested))


def canonical_lane_families(lane: str | None) -> tuple[str, ...] | None:
    """Return the coarse canonical-policy family set for a named lane.

    Lanes are deliberately coarser than Hamiltonian identities.  They define
    the policy class we are willing to publish as canonical without pretending
    that a single flat vector transfers across fermionic, bosonic, mixed
    electron/phonon, and molecular routes.
    """

    if lane in {None, ""}:
        return None
    key = str(lane).strip().lower().replace("-", "_")
    if key not in _CANONICAL_LANE_FAMILIES:
        raise ValueError(f"Unsupported canonical lane: {lane!r}")
    return tuple(_CANONICAL_LANE_FAMILIES[key])


def required_target_benchmark_ids_for_profile(profile: str | None) -> tuple[str, ...]:
    key = str(profile or "none").strip().lower().replace("-", "_")
    if key not in set(_REQUIRED_TARGET_PROFILES):
        raise ValueError(f"Unsupported required target profile: {profile!r}")
    if key == "fermionic_hubbard_core":
        return tuple(_FERMIONIC_HUBBARD_CORE_REQUIRED_IDS)
    return ()


def _normalize_calibration_profile(profile: str | None) -> str:
    key = str(profile or "off").strip().lower().replace("-", "_")
    if key not in set(_CALIBRATION_PROFILE_CHOICES):
        raise ValueError(f"Unsupported calibration profile: {profile!r}")
    return key


def calibration_static_benchmark_specs(profile: str | None) -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return additive nph2/ref3 Route-A calibration benchmark slices.

    Calibration profiles intentionally select by exact benchmark IDs from the
    train/small_robust, boson_cutoffs=(1, 2), ref-cutoff=3 suite and then fail
    closed if any selected case drifts in working cutoff, reference cutoff, or
    family membership.
    """

    profile_key = _normalize_calibration_profile(profile)
    if profile_key == "off":
        return ()
    if profile_key == _CALIBRATION_PROFILE_NPH2_ROUTE_A_HK_HH:
        expected_ids = _NPH2_ROUTE_A_HK_HH_CALIBRATION_BENCHMARK_IDS
        expected_families = {"harmonic_kerr_chain", "hh"}
    elif profile_key == _CALIBRATION_PROFILE_NPH2_REF3_ROUTE_A_BOSONIC_MIXED_WEIGHTED:
        expected_ids = _NPH2_REF3_ROUTE_A_BOSONIC_MIXED_WEIGHTED_BENCHMARK_IDS
        expected_families = {"bose_hubbard", "harmonic_kerr_chain", "spin_boson", "hh"}
    else:  # pragma: no cover - guarded by _normalize_calibration_profile
        raise ValueError(f"Unsupported calibration profile: {profile!r}")
    suite = default_static_benchmark_suite(
        split="train",
        boson_cutoffs=(1, 2),
        exact_reference_boson_cutoff=3,
        physics_grid_profile="small_robust",
    )
    by_id = {spec.benchmark_id: spec for spec in suite}
    missing = [benchmark_id for benchmark_id in expected_ids if benchmark_id not in by_id]
    if missing:
        raise ValueError(f"Calibration profile {profile_key!r} missing benchmark specs: {missing}")
    selected = tuple(by_id[benchmark_id] for benchmark_id in expected_ids)
    selected_ids = tuple(spec.benchmark_id for spec in selected)
    if selected_ids != tuple(expected_ids):
        raise ValueError(f"Calibration profile {profile_key!r} selected wrong benchmark IDs: {selected_ids}")
    selected_families = {spec.family for spec in selected}
    if selected_families != expected_families:
        raise ValueError(
            f"Calibration profile {profile_key!r} selected wrong families: "
            f"{sorted(selected_families)} != {sorted(expected_families)}"
        )
    for spec in selected:
        n_ph = _pipeline_arg_value(spec.base_pipeline_args, "--n-ph-max")
        if n_ph != "2":
            raise ValueError(f"Calibration profile {profile_key!r} requires working n_ph_max=2 for {spec.benchmark_id}.")
        if spec.exact_reference_n_ph_max != 3:
            raise ValueError(f"Calibration profile {profile_key!r} requires reference n_ph_max=3 for {spec.benchmark_id}.")
        if "physics_profile:small_robust" not in spec.tags:
            raise ValueError(f"Calibration profile {profile_key!r} requires small_robust tags for {spec.benchmark_id}.")
    return selected


def filter_canonical_lane_specs(
    specs: Sequence[HamiltonianBenchmarkSpec],
    *,
    lane: str | None,
    stage: str = "train",
) -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Filter benchmark specs for a canonical lane train/transfer stage.

    Train stages use the smallest wired representative instances: lattice
    routes use L=2; spin_boson is kept at L=1 because that is its current
    registry support; molecular train keeps only explicit train-primary or
    override-train specs.  Transfer stages use L=3 where wired and keep
    spin_boson L=1 as a no-larger-size stress anchor.
    """

    if lane in {None, ""}:
        return tuple(specs)
    lane_key = str(lane).strip().lower().replace("-", "_")
    families = set(canonical_lane_families(lane_key) or ())
    stage_key = str(stage or "train").strip().lower().replace("-", "_")
    if stage_key not in {"train", "transfer", "all"}:
        raise ValueError(f"Unsupported canonical lane stage: {stage!r}")

    out: list[HamiltonianBenchmarkSpec] = []
    for spec in specs:
        if spec.family not in families:
            continue
        L = int(spec.features.L)
        keep = False
        if stage_key == "all":
            keep = True
        elif lane_key == "molecular":
            role = _molecular_role_for_spec(spec)
            keep = (
                role in {_MOLECULAR_ROLE_TRAIN_PRIMARY, _MOLECULAR_ROLE_OVERRIDE_TRAIN}
                if stage_key == "train"
                else role == _MOLECULAR_ROLE_TRANSFER
            )
        elif lane_key == "bosonic" and spec.family == "spin_boson":
            # spin_boson is a bosonic/spin-oscillator case and is currently
            # L=1-only; keep it in bosonic train and transfer/stress stages.
            keep = True
        elif stage_key == "train":
            keep = L == 2
        elif stage_key == "transfer":
            keep = L == 3
        if keep:
            out.append(
                replace(
                    spec,
                    tags=tuple(
                        dict.fromkeys(
                            (
                                *spec.tags,
                                f"canonical_lane:{lane_key}",
                                f"canonical_lane_stage:{stage_key}",
                            )
                        )
                    ),
                )
            )
    return tuple(out)


def _run_phase3_robustness_gate_preflight(
    *,
    output_dir: Path,
    lanes: Sequence[str],
    target_abs_delta_e: float,
    python_bin: str = sys.executable,
    adapt_timeout_s: float | None = None,
    compile_timeout_s: float | None = None,
) -> Mapping[str, Any]:
    """Run the static Hubbard L2 robustness gate before broad studies."""

    from pipelines.static_adapt.optimization.phase3_robustness_gate import (
        run_hubbard_l2_robustness_gate,
    )

    return run_hubbard_l2_robustness_gate(
        output_dir=Path(output_dir),
        lanes=tuple(str(lane) for lane in lanes),
        target_abs_delta_e=float(target_abs_delta_e),
        python_bin=str(python_bin),
        adapt_timeout_s=adapt_timeout_s,
        compile_timeout_s=compile_timeout_s,
    )


def _should_apply_phase3_robustness_gate(
    *,
    mode: str,
    specs: Sequence[HamiltonianBenchmarkSpec],
    gate_mode: str,
    n_trials: int,
) -> bool:
    from pipelines.static_adapt.optimization.phase3_robustness_gate import (
        should_apply_robustness_gate,
    )

    return bool(
        should_apply_robustness_gate(
            mode=str(mode),
            specs=tuple(specs),
            gate_mode=str(gate_mode),
            n_trials=int(n_trials),
        )
    )


def _trial_result_records(trial: Any) -> list[tuple[str, Mapping[str, Any]]]:
    attrs = dict(getattr(trial, "user_attrs", {}) or {})
    rows: list[tuple[str, Mapping[str, Any]]] = []
    results = attrs.get("results")
    if isinstance(results, Mapping):
        for key, value in results.items():
            if isinstance(value, Mapping):
                rows.append((str(key), value))
    single = attrs.get("result")
    if isinstance(single, Mapping):
        rows.append((str(single.get("benchmark_id", "benchmark")), single))
    return rows


def _trial_first_crossing_records(trial: Any) -> list[tuple[str, Mapping[str, Any]]]:
    attrs = dict(getattr(trial, "user_attrs", {}) or {})
    payload = attrs.get("paper_i_first_crossing")
    rows: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(payload, Mapping) and payload.get("schema") == "paper_i_first_crossing_v1":
        rows.append((str(payload.get("benchmark_id", "benchmark")), payload))
    elif isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, Mapping):
                rows.append((str(key), value))
    return rows


def _trial_value_payload(trial: Any) -> float | list[float] | None:
    try:
        values = getattr(trial, "values", None)
    except Exception:
        values = None
    if values is not None:
        try:
            parsed_values = [float(value) for value in values]
        except Exception:
            return None
        return parsed_values[0] if len(parsed_values) == 1 else parsed_values
    try:
        value = getattr(trial, "value", None)
    except Exception:
        value = None
    return None if value is None else float(value)


def _trial_brief(trial: Any) -> dict[str, Any]:
    value_payload = _trial_value_payload(trial)
    return {
        "trial_number": int(getattr(trial, "number", -1)),
        "value": value_payload,
        "values": value_payload if isinstance(value_payload, list) else None,
        "params": dict(getattr(trial, "params", {}) or {}),
    }


def _best_trial_by_result_metric(trials: Sequence[Any], metric: str) -> dict[str, Any] | None:
    best: tuple[float, int, dict[str, Any]] | None = None
    for trial in trials:
        for benchmark_id, result in _trial_result_records(trial):
            value = _as_float_or_none(result.get(metric))
            if value is None:
                continue
            row = {
                **_trial_brief(trial),
                "benchmark_id": benchmark_id,
                "metric": metric,
                "metric_value": float(value),
            }
            key = (float(value), int(getattr(trial, "number", 0)), row)
            if best is None or key[:2] < best[:2]:
                best = key
    return None if best is None else best[2]


def _best_resource_subject_tau(trials: Sequence[Any]) -> dict[str, Any] | None:
    best: tuple[tuple[float, float, float, float, float, int], dict[str, Any]] | None = None
    for trial in trials:
        for benchmark_id, crossing in _trial_first_crossing_records(trial):
            if not bool(crossing.get("reached", False)):
                continue
            resource = _as_float_or_none(crossing.get("resource_score"))
            if resource is None:
                resource = _as_float_or_none(crossing.get("operator_count_at_crossing"))
            if resource is None:
                resource = _as_float_or_none(crossing.get("k_tau"))
            if resource is None:
                continue
            key = (
                float(resource if resource is not None else 0.0),
                float(_as_float_or_none(crossing.get("parameter_count_at_crossing")) or 0.0),
                float(_as_float_or_none(crossing.get("primary_error_at_crossing")) or 0.0),
                int(getattr(trial, "number", 0)),
            )
            row = {
                **_trial_brief(trial),
                "benchmark_id": benchmark_id,
                "paper_i_first_crossing": _jsonable(dict(crossing)),
                "deterministic_tie_order": [
                    "first_crossing_resource_score",
                    "first_crossing_parameter_count",
                    "primary_error_at_crossing",
                    "trial_number",
                ],
            }
            if best is None or key < best[0]:
                best = (key, row)
    return None if best is None else best[1]


def _first_crossing_status_summary(trials: Sequence[Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    reached = 0
    total = 0
    for trial in trials:
        for _benchmark_id, crossing in _trial_first_crossing_records(trial):
            total += 1
            status = str(crossing.get("status", "unknown"))
            counts[status] = int(counts.get(status, 0)) + 1
            reached += int(bool(crossing.get("reached", False)))
    return {"schema": "paper_i_first_crossing_status_summary_v1", "record_count": total, "reached_count": reached, "status_counts": counts}


def _effective_manifest_summary(trials: Sequence[Any]) -> dict[str, Any]:
    pointers: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for trial in trials:
        attrs = dict(getattr(trial, "user_attrs", {}) or {})
        pointer = attrs.get(_EFFECTIVE_TRIAL_MANIFEST_USER_ATTR_KEY)
        if not isinstance(pointer, Mapping):
            continue
        item = {"trial_number": int(getattr(trial, "number", -1)), **_jsonable(dict(pointer))}
        pointers.append(item)
        status = str(pointer.get("lifecycle_status", "unknown"))
        status_counts[status] = int(status_counts.get(status, 0)) + 1
    return {
        "schema": "phase3_objective_replay_manifest_summary_v1",
        "status": "available_not_validated" if pointers else "unavailable",
        "objective_replay_table_path": None,
        "effective_trial_manifest_count": len(pointers),
        "manifest_status_counts": status_counts,
        "manifest_pointers": pointers,
    }


def _best_trial_interpretation(complete_trials: Sequence[Any], best_trial: Any | None) -> dict[str, Any]:
    per_benchmark_resource_best = _best_resource_subject_tau(complete_trials)
    return {
        "schema": "phase3_best_trial_interpretation_v1",
        "active_objective_best": None if best_trial is None else _trial_brief(best_trial),
        "terminal_energy_best": _best_trial_by_result_metric(complete_trials, "energy"),
        "terminal_same_cutoff_error_best": _best_trial_by_result_metric(complete_trials, "abs_delta_e_same_cutoff"),
        "terminal_reference_error_best": _best_trial_by_result_metric(complete_trials, "abs_delta_e_reference"),
        "per_benchmark_resource_subject_tau_phys_best": per_benchmark_resource_best,
        "resource_subject_tau_phys_best": {
            "deprecated_alias_for": "per_benchmark_resource_subject_tau_phys_best",
            "value": per_benchmark_resource_best,
        },
    }


def _terminal_error_rows_for_trial(trial: Any | None) -> list[dict[str, Any]]:
    if trial is None:
        return []
    rows: list[dict[str, Any]] = []
    for benchmark_id, result in _trial_result_records(trial):
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "energy": result.get("energy"),
                "same_cutoff_exact_gs_energy": result.get("same_cutoff_exact_gs_energy"),
                "exact_reference_energy": result.get("exact_reference_energy"),
                "exact_reference_n_ph_max": result.get("exact_reference_n_ph_max"),
                "abs_delta_e_same_cutoff": result.get("abs_delta_e_same_cutoff"),
                "abs_delta_e_reference": result.get("abs_delta_e_reference"),
                "cutoff_abs_delta_e": result.get("cutoff_abs_delta_e"),
            }
        )
    return _jsonable(rows)


def _telemetry_float_text(value: Any) -> str:
    parsed = _as_float_or_none(value)
    if parsed is None or not math.isfinite(float(parsed)):
        return "na"
    return f"{float(parsed):.12g}"


def _telemetry_text(value: Any) -> str:
    if value in {None, ""}:
        return "na"
    return str(value).replace(" ", "_")


def _telemetry_payload_from_score_components(score_components: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(score_components.get("per_benchmark_scores"), Sequence):
        rows = [row for row in score_components.get("per_benchmark_scores", ()) if isinstance(row, Mapping)]
        crossings = [
            row.get("discovery_first_crossing")
            for row in rows
            if isinstance(row.get("discovery_first_crossing"), Mapping)
        ]
        primary_values = [
            _as_float_or_none(crossing.get("primary_error"))
            for crossing in crossings
            if _as_float_or_none(crossing.get("primary_error")) is not None
        ]
        tau_phys_values = {
            float(value)
            for value in (_as_float_or_none(crossing.get("feasibility_threshold")) for crossing in crossings)
            if value is not None
        }
        tau_tight_values = {
            float(value)
            for value in (
                _as_float_or_none(
                    (crossing.get("physical_target_manifest") or {}).get("tau_tight")
                    if isinstance(crossing.get("physical_target_manifest"), Mapping)
                    else None
                )
                for crossing in crossings
            )
            if value is not None
        }
        reached = sum(1 for crossing in crossings if bool(crossing.get("feasible", crossing.get("reached", False))))
        total = len(crossings) if crossings else len(rows)
        return {
            "primary_error": max(primary_values) if primary_values else None,
            "first_crossing": f"{reached}/{total}" if total else "na",
            "tau_phys": next(iter(tau_phys_values)) if len(tau_phys_values) == 1 else ("mixed" if tau_phys_values else None),
            "tau_tight": next(iter(tau_tight_values)) if len(tau_tight_values) == 1 else ("mixed" if tau_tight_values else None),
            "resource_score": score_components.get("score"),
        }
    first = score_components.get("paper_i_first_crossing")
    target = score_components.get("physical_target_manifest")
    if not isinstance(first, Mapping):
        first = {}
    if not isinstance(target, Mapping):
        target = {}
    return {
        "primary_error": score_components.get("primary_error"),
        "first_crossing": "1/1" if bool(score_components.get("feasible")) else "0/1",
        "tau_phys": score_components.get("feasibility_threshold") or target.get("tau_phys"),
        "tau_tight": target.get("tau_tight"),
        "resource_score": score_components.get("resource_score"),
    }


def _set_trial_telemetry_attrs(trial: Any, score_components: Mapping[str, Any]) -> None:
    if not hasattr(trial, "set_user_attr"):
        return
    payload = _telemetry_payload_from_score_components(score_components)
    trial.set_user_attr("telemetry_primary_error", payload.get("primary_error"))
    trial.set_user_attr("telemetry_first_crossing", payload.get("first_crossing"))
    trial.set_user_attr("telemetry_tau_phys", payload.get("tau_phys"))
    trial.set_user_attr("telemetry_tau_tight", payload.get("tau_tight"))
    trial.set_user_attr("telemetry_resource_score", payload.get("resource_score"))


_POOL_TRIAL_PARAM_OVERRIDE_FIELDS = {
    "pool_key",
    "family_repeat_penalty",
    "novelty_bonus",
}
_POOL_BUDGET_TRIAL_PARAM_OVERRIDE_FIELDS = {
    "phase1_min_count",
    "phase1_max_count",
    "phase1_pool_fraction",
    "phase1_qubit_slope",
    "phase2_min_count",
    "phase2_max_count",
    "phase2_pool_fraction",
    "phase2_qubit_slope",
}
_INNER_TRIAL_PARAM_OVERRIDE_FIELDS = {
    "spsa_a",
    "spsa_c",
    "spsa_A",
    "spsa_alpha",
    "spsa_gamma",
}
_STATIC_TRIAL_PARAM_OVERRIDE_FIELDS = {
    "lambda_compile",
    "lambda_measure",
    "lambda_leak",
    "lambda_2q",
    "lambda_d",
    "lambda_1q",
    "lambda_theta",
    "lambda_shot",
    "suppress_explicit_hardware_lambdas",
    "compile_cx_weight",
    "compile_sq_weight",
    "compile_rotation_step_weight",
    "compile_position_shift_weight",
    "compile_refit_active_weight",
    "measure_groups_weight",
    "measure_shots_weight",
    "measure_reuse_weight",
    "opt_dim_cost_scale",
    "phase2_w_depth",
    "phase2_w_group",
    "phase2_w_shot",
    "phase2_w_optdim",
    "phase2_w_reuse",
    "phase2_w_lifetime",
    "phase2_leakage_cap",
    "phase2_frontier_ratio",
    "phase3_frontier_ratio",
    "phase3_tie_beam_score_ratio",
    "phase3_tie_beam_abs_tol",
    "phase3_tie_beam_max_branches",
    "phase3_tie_beam_max_late_coordinate",
    "phase3_tie_beam_min_depth_left",
    "adapt_beam_live_branches",
    "adapt_beam_children_per_parent",
    "adapt_beam_terminated_keep",
    "adapt_beam_lambda",
    "adapt_reopt_policy",
    "adapt_window_size",
    "adapt_window_topk",
    "adapt_full_refit_every",
    "adapt_final_full_refit",
    "adapt_final_refit_maxiter",
    "adapt_insertion_mode",
    "adapt_allow_repeats",
    "adapt_allow_repeats_override",
    "adapt_max_depth",
    "adapt_maxiter",
    "adapt_drop_floor",
    "adapt_drop_patience",
    "adapt_drop_min_depth",
    "adapt_eps_grad",
    "adapt_eps_energy",
    "phase1_probe_max_positions",
    "phase2_shortlist_fraction",
    "phase0_pilot_enabled",
    "phase0_pilot_alpha",
    "phase0_pilot_threshold",
    "phase0_pilot_max_records",
    "phase0_lane_quota_pressure",
    "phase0_algebraic_lane_mode",
    "algebraic_phase2_lane_rel_threshold",
    "algebraic_phase1_lane_quota_pressure",
    "algebraic_phase2_lane_quota_pressure",
    "phase2_novelty_mode",
    "phase2_gamma_N",
    "phase2_gamma_N_schedule_mode",
    "phase2_gamma_N_schedule_start",
    "phase2_gamma_N_schedule_end",
    "phase2_motif_bonus_weight",
    "phase2_rho",
    "phase1_score_mode",
    "phase1_score_z_alpha",
    "phase2_score_z_alpha",
    "phase2_enable_batching",
    "phase2_batch_target_size",
    "phase2_batch_size_cap",
    "phase2_batch_near_degenerate_ratio",
    "phase2_batch_rank_rel_tol",
    "phase2_batch_additivity_tol",
    "phase1_prune_enabled",
    "phase1_prune_policy",
    "phase1_prune_mode",
    "phase1_prune_fraction",
    "phase1_prune_min_candidates",
    "phase1_prune_max_candidates",
    "phase1_prune_max_regression",
    "phase1_prune_tolerance_mode",
    "phase1_prune_tolerance_shot_coeff",
    "phase1_prune_tolerance_screen_coeff",
    "phase1_prune_tolerance_chem",
    "phase1_prune_tolerance_rel_coeff",
    "phase1_prune_retained_gain_ratio",
    "phase1_prune_protect_steps",
    "phase1_prune_stale_age",
    "phase1_prune_stagnation_threshold",
    "phase1_prune_small_theta_abs",
    "phase1_prune_small_theta_relative",
    "phase1_prune_cooldown_steps",
    "phase1_prune_local_window_size",
    "phase1_prune_recovery_trust_radius",
    "phase1_prune_old_fraction",
    "phase1_prune_checkpoint_period",
    "phase1_prune_live_min_depth",
    "phase1_prune_maturity_threshold",
    "phase1_prune_snr_threshold",
    "phase1_prune_amplitude_witness_required",
    "phase1_prune_collapse_peak_abs_min",
    "phase1_prune_collapse_current_abs_max",
    "phase1_prune_collapse_ratio",
    "phase1_prune_collapse_min_abs_drop",
    "phase1_prune_collapse_min_observations",
    "phase1_maturity_cap_min_fraction",
    "phase1_maturity_cap_max_fraction",
    "phase2_maturity_cap_min_fraction",
    "phase2_maturity_cap_max_fraction",
    "phase3_maturity_cap_min_fraction",
    "phase3_maturity_cap_max_fraction",
    "phase_maturity_shot_min",
    "phase_maturity_shot_max",
    "phase1_maturity_shot_cap",
    "phase2_maturity_shot_cap",
    "phase3_maturity_shot_cap",
    "phase_live_hysteresis_enabled",
    "phase2_null_nrem_high_threshold",
    "phase2_live_nrem_low_threshold",
    "phase3_null_nrem_high_threshold",
    "phase3_live_nrem_low_threshold",
    "phase2_hysteresis_steps",
    "phase3_hysteresis_steps",
    "phase3_selector_policy",
    "phase3_selector_geometry_mode",
    "phase3_novelty_ablation_mode",
    "phase3_window_relaxation_mode",
    "phase3_batch_selection_mode",
    "phase3_batch_prefilter_mode",
    "phase3_backend_cost_mode",
    "phase3_hardware_cost_normalization_mode",
    "phase3_source_lock_preferred_sequence",
    "phase3_runtime_split_mode",
    "allow_archival_phase3_runtime_split",
    "phase3_runtime_split_selection_mode",
    "phase3_runtime_split_max_subset_size",
    "shared_pauli_pool_mode",
    "shared_pauli_pool_symmetry_policy",
    "shared_pauli_pool_max_subset_size",
    "static_meta_feature_profile",
    "static_route_id",
    "static_lane_route",
    "physical_lane_shortlist_aggressiveness",
    "adapt_noise_floor_stop_policy",
    "adapt_noise_floor_snr_threshold",
    "adapt_noise_floor_n_rem_high_threshold",
    "adapt_noise_floor_useful_horizon_threshold",
}


def _coerce_like(value: Any, template: Any) -> Any:
    if template is None:
        if value is None:
            return None
        text = str(value).strip()
        if text.lower() in {"", "none", "null"}:
            return None
        try:
            return float(text)
        except Exception:
            return text
    if isinstance(template, bool):
        return _coerce_bool(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(float(str(value)))
    if isinstance(template, float):
        return float(value)
    return str(value)


def _sanitize_trial_param_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    static_defaults = StaticScaffoldPolicy()
    static_field_names = {field.name for field in fields(StaticScaffoldPolicy)}
    pool_defaults = PoolPolicy()
    pool_field_names = {field.name for field in fields(PoolPolicy)}
    inner_defaults = InnerOptimizerPolicy()
    inner_field_names = {field.name for field in fields(InnerOptimizerPolicy)}
    budget_templates = {
        "phase1_min_count": pool_defaults.phase1_budget.min_count,
        "phase1_max_count": pool_defaults.phase1_budget.max_count,
        "phase1_pool_fraction": pool_defaults.phase1_budget.pool_fraction,
        "phase1_qubit_slope": pool_defaults.phase1_budget.qubit_slope,
        "phase2_min_count": pool_defaults.phase2_budget.min_count,
        "phase2_max_count": pool_defaults.phase2_budget.max_count,
        "phase2_pool_fraction": pool_defaults.phase2_budget.pool_fraction,
        "phase2_qubit_slope": pool_defaults.phase2_budget.qubit_slope,
    }
    sanitized: dict[str, Any] = {}
    for key, value in dict(overrides).items():
        key = str(key).strip()
        if key in _STATIC_TRIAL_PARAM_OVERRIDE_FIELDS and key in static_field_names:
            sanitized[key] = _coerce_like(value, getattr(static_defaults, key))
        elif key in _POOL_TRIAL_PARAM_OVERRIDE_FIELDS and key in pool_field_names:
            sanitized[key] = _coerce_like(value, getattr(pool_defaults, key))
        elif key in _POOL_BUDGET_TRIAL_PARAM_OVERRIDE_FIELDS:
            sanitized[key] = _coerce_like(value, budget_templates[key])
        elif key in _INNER_TRIAL_PARAM_OVERRIDE_FIELDS and key in inner_field_names:
            sanitized[key] = _coerce_like(value, getattr(inner_defaults, key))
    return sanitized


def _apply_trial_param_overrides_to_policy(
    policy: AlgorithmPolicy,
    overrides: Mapping[str, Any] | None,
) -> AlgorithmPolicy:
    sanitized = _sanitize_trial_param_overrides(overrides)
    if not sanitized:
        return policy
    static_field_names = {field.name for field in fields(StaticScaffoldPolicy)}
    pool_field_names = {field.name for field in fields(PoolPolicy)}
    inner_field_names = {field.name for field in fields(InnerOptimizerPolicy)}
    static_updates = {
        key: value
        for key, value in sanitized.items()
        if key in _STATIC_TRIAL_PARAM_OVERRIDE_FIELDS and key in static_field_names
    }
    pool_updates = {
        key: value
        for key, value in sanitized.items()
        if key in _POOL_TRIAL_PARAM_OVERRIDE_FIELDS and key in pool_field_names
    }
    inner_updates = {
        key: value
        for key, value in sanitized.items()
        if key in _INNER_TRIAL_PARAM_OVERRIDE_FIELDS and key in inner_field_names
    }
    pool = policy.pool
    phase1_budget_updates = {
        key.removeprefix("phase1_"): value
        for key, value in sanitized.items()
        if key in _POOL_BUDGET_TRIAL_PARAM_OVERRIDE_FIELDS and key.startswith("phase1_")
    }
    phase2_budget_updates = {
        key.removeprefix("phase2_"): value
        for key, value in sanitized.items()
        if key in _POOL_BUDGET_TRIAL_PARAM_OVERRIDE_FIELDS and key.startswith("phase2_")
    }
    if phase1_budget_updates:
        pool = replace(pool, phase1_budget=replace(pool.phase1_budget, **phase1_budget_updates))
    if phase2_budget_updates:
        pool = replace(pool, phase2_budget=replace(pool.phase2_budget, **phase2_budget_updates))
    if pool_updates:
        pool = replace(pool, **pool_updates)
    static = replace(policy.static, **static_updates) if static_updates else policy.static
    inner = replace(policy.inner_optimizer, **inner_updates) if inner_updates else policy.inner_optimizer
    if "adapt_maxiter" in sanitized:
        inner = replace(
            inner,
            refit_maxiter=int(sanitized["adapt_maxiter"]),
            final_maxiter=int(sanitized["adapt_maxiter"]),
        )
    return replace(policy, pool=pool, static=static, inner_optimizer=inner)


def _load_trial_param_overrides(path: str | Path | None) -> dict[str, Any]:
    if path in {None, ""}:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"trial param overrides must be a JSON object: {path}")
    raw = payload.get("trial_param_overrides", payload)
    if not isinstance(raw, Mapping):
        raise ValueError(f"trial_param_overrides must be a JSON object: {path}")
    return _sanitize_trial_param_overrides(raw)


def _load_enqueue_trial_param_records(path: str | Path | None) -> tuple[WarmStartCandidate, ...]:
    if path in {None, ""}:
        return ()
    record_path = Path(path)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"enqueue trial params must be a JSON object: {path}")
    raw_records = payload.get(
        "enqueue_trial_records",
        payload.get("warm_start_records", payload.get("records", payload.get("enqueue_trials", ()))),
    )
    if isinstance(raw_records, Mapping):
        raw_records = tuple(raw_records.values())
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError(f"enqueue trial records must be a JSON array: {path}")
    path_sha256 = _sha256_file(record_path)
    out: list[WarmStartCandidate] = []
    for idx, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"enqueue trial record {idx} must be a JSON object: {path}")
        params = raw.get("params", raw.get("trial_params"))
        if not isinstance(params, Mapping) or not params:
            raise ValueError(f"enqueue trial record {idx} missing params object: {path}")
        source_score = raw.get("source_score")
        try:
            source_score_value = None if source_score in {None, ""} else float(source_score)
        except Exception:
            source_score_value = None
        source_payload = dict(raw.get("source_payload") or {})
        source_payload.setdefault("source_artifact_path", str(record_path))
        source_payload.setdefault("source_sha256", path_sha256)
        out.append(
            WarmStartCandidate(
                params=dict(params),
                source_kind=str(raw.get("source_kind") or payload.get("source_kind") or "explicit_enqueue_json"),
                source_id=str(raw.get("source_id") or f"{record_path}:{idx}"),
                benchmark_id=(
                    None
                    if raw.get("benchmark_id") in {None, ""}
                    else str(raw.get("benchmark_id"))
                ),
                family=None if raw.get("family") in {None, ""} else str(raw.get("family")),
                source_score=source_score_value,
                source_payload=source_payload,
                compatibility_warnings=tuple(str(item) for item in raw.get("compatibility_warnings", ()) or ()),
            )
        )
    return tuple(out)


def _sanitize_benchmark_trial_param_overrides(
    overrides_by_selector: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if not overrides_by_selector:
        return {}
    sanitized: dict[str, dict[str, Any]] = {}
    for selector, overrides in dict(overrides_by_selector).items():
        selector_key = str(selector).strip()
        if not selector_key:
            continue
        if overrides is None:
            continue
        if not isinstance(overrides, Mapping):
            raise ValueError(f"benchmark trial overrides for {selector_key!r} must be a JSON object")
        cleaned = _sanitize_trial_param_overrides(overrides)
        if cleaned:
            sanitized[selector_key] = cleaned
    return sanitized


def _load_benchmark_trial_param_overrides(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path in {None, ""}:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"benchmark trial param overrides must be a JSON object: {path}")
    raw = payload.get("benchmark_trial_param_overrides", payload.get("by_benchmark_id", payload))
    if not isinstance(raw, Mapping):
        raise ValueError(f"benchmark_trial_param_overrides must be a JSON object: {path}")
    return _sanitize_benchmark_trial_param_overrides(raw)


def _trial_param_overrides_for_spec(
    spec: HamiltonianBenchmarkSpec,
    overrides_by_selector: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if not overrides_by_selector:
        return {}
    selectors = (
        "default",
        f"family:{spec.family}",
        f"benchmark_id:{spec.benchmark_id}",
        str(spec.benchmark_id),
    )
    merged: dict[str, Any] = {}
    for selector in selectors:
        overrides = overrides_by_selector.get(selector)
        if isinstance(overrides, Mapping):
            merged.update(_sanitize_trial_param_overrides(overrides))
    return merged


def _apply_benchmark_trial_param_overrides_to_policy(
    policy: AlgorithmPolicy,
    spec: HamiltonianBenchmarkSpec,
    overrides_by_selector: Mapping[str, Mapping[str, Any]] | None,
) -> AlgorithmPolicy:
    overrides = _trial_param_overrides_for_spec(spec, overrides_by_selector)
    if not overrides:
        return policy
    adjusted = _apply_trial_param_overrides_to_policy(policy, overrides)
    return _normalize_active_policy(
        adjusted,
        meta_feature_profile=getattr(policy.static, "static_meta_feature_profile", _DEFAULT_META_FEATURE_PROFILE),
    )


def _benchmark_effective_policies(
    policy: AlgorithmPolicy,
    specs: Sequence[HamiltonianBenchmarkSpec],
    overrides_by_selector: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, AlgorithmPolicy]:
    if not overrides_by_selector:
        return {}
    out: dict[str, AlgorithmPolicy] = {}
    for spec in specs:
        overrides = _trial_param_overrides_for_spec(spec, overrides_by_selector)
        if overrides:
            out[str(spec.benchmark_id)] = _apply_benchmark_trial_param_overrides_to_policy(
                policy,
                spec,
                overrides_by_selector,
            )
    return out


def _format_optuna_best_telemetry_line(*, record_id: str, trial: Any) -> str:
    attrs = dict(getattr(trial, "user_attrs", {}) or {})
    return (
        f"OPTUNA_BEST record_id={_telemetry_text(record_id)} "
        f"trial={int(getattr(trial, 'number', -1))} "
        f"best_value={_telemetry_float_text(getattr(trial, 'value', None))} "
        f"primary_error={_telemetry_float_text(attrs.get('telemetry_primary_error'))} "
        f"first_crossing={_telemetry_text(attrs.get('telemetry_first_crossing'))} "
        f"tau_phys={_telemetry_float_text(attrs.get('telemetry_tau_phys'))} "
        f"tau_tight={_telemetry_float_text(attrs.get('telemetry_tau_tight'))} "
        f"resource_score={_telemetry_float_text(attrs.get('telemetry_resource_score'))}"
    )


def _trial_telemetry_progress_payload(trial: Any, *, prefix: str = "") -> dict[str, Any]:
    attrs = dict(getattr(trial, "user_attrs", {}) or {})
    primary_error = attrs.get("telemetry_primary_error")
    objective_provenance = attrs.get("objective_provenance")
    if not isinstance(objective_provenance, Mapping):
        global_score_components = attrs.get("global_score_components")
        if isinstance(global_score_components, Mapping):
            objective_provenance = global_score_components.get("objective_provenance")
    robustness_value_noise = (
        dict(objective_provenance.get("phase3_oracle_value_noise"))
        if isinstance(objective_provenance, Mapping)
        and isinstance(objective_provenance.get("phase3_oracle_value_noise"), Mapping)
        else None
    )
    return {
        f"{prefix}primary_error": primary_error,
        f"{prefix}delta_e": primary_error,
        f"{prefix}first_crossing": attrs.get("telemetry_first_crossing"),
        f"{prefix}tau_phys": attrs.get("telemetry_tau_phys"),
        f"{prefix}tau_tight": attrs.get("telemetry_tau_tight"),
        f"{prefix}resource_score": attrs.get("telemetry_resource_score"),
        f"{prefix}robustness_value_noise": _jsonable(robustness_value_noise),
    }


def _optuna_best_progress_payload(study: Any) -> dict[str, Any]:
    try:
        best_trial = study.best_trial
    except Exception:
        try:
            best_trials = list(study.best_trials)
        except Exception:
            best_trials = []
        return {
            "best_trial_number": None,
            "best_value": None,
            "best_user_attrs": {},
            "pareto_front_size": len(best_trials),
            "pareto_best_trials": [_trial_brief(trial) for trial in best_trials],
        }
    return {
        "best_trial_number": int(getattr(best_trial, "number", -1)),
        "best_value": _trial_value_payload(best_trial),
        "best_user_attrs": _jsonable(dict(getattr(best_trial, "user_attrs", {}) or {})),
        **_jsonable(_trial_telemetry_progress_payload(best_trial, prefix="best_")),
    }


class _OptunaBestTelemetry:
    def __init__(self, *, record_id: str, min_interval_sec: float = 0.0) -> None:
        self.record_id = str(record_id)
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._last_trial_number: int | None = None
        self._last_log_time = 0.0
        self._lock = threading.Lock()

    def maybe_log(self, study: Any, _trial: Any | None = None) -> None:
        try:
            best_trial = study.best_trial
        except Exception:
            return
        number = int(getattr(best_trial, "number", -1))
        now = time.monotonic()
        with self._lock:
            if self._last_trial_number == number and (now - self._last_log_time) < self.min_interval_sec:
                return
            self._last_trial_number = number
            self._last_log_time = now
        print(_format_optuna_best_telemetry_line(record_id=self.record_id, trial=best_trial), flush=True)


def run_optuna_study(
    *,
    mode: str,
    specs: Sequence[HamiltonianBenchmarkSpec],
    output_dir: Path,
    n_trials: int,
    seed: int = 7,
    n_jobs: int = 1,
    storage: str | None = None,
    study_name: str | None = None,
    enqueue_default: bool = True,
    enqueue_trials: Sequence[Mapping[str, Any]] = (),
    enqueue_records: Sequence[WarmStartCandidate] | None = None,
    warm_start_skips: Sequence[WarmStartSkip] = (),
    runner: Callable[..., BenchmarkResult] = run_static_benchmark,
    config: GlobalObjectiveConfig = GlobalObjectiveConfig(),
    benchmarks_per_trial_jobs: int = 1,
    policy_search_profile: str | None = "default",
    meta_feature_profile: str | None = _DEFAULT_META_FEATURE_PROFILE,
    progress_dir: str | Path | None = None,
    base_policy: AlgorithmPolicy | None = None,
    suite_profile: str | None = "standard",
    telemetry_record_id: str | None = None,
    optuna_best_telemetry: str = "auto",
    optuna_best_telemetry_min_interval_sec: float = 0.0,
    trial_param_overrides: Mapping[str, Any] | None = None,
    benchmark_trial_param_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one executable Optuna study for a single oracle spec or global suite."""

    optuna = _import_optuna()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = tuple(specs)
    progress = _ProgressReporter(progress_dir)
    if not selected:
        raise ValueError("run_optuna_study requires at least one benchmark spec")
    if mode not in {"oracle", "global"}:
        raise ValueError(f"unsupported study mode: {mode!r}")
    if mode == "oracle" and len(selected) != 1:
        raise ValueError("oracle mode accepts exactly one benchmark spec")
    multi_objective_mode = _normalize_multi_objective_mode(config.multi_objective_mode)
    if multi_objective_mode != _MULTI_OBJECTIVE_MODE_OFF and mode != "oracle":
        raise ValueError("multi-objective Phase3 Optuna is currently supported only for oracle/oracle-grid studies")

    if multi_objective_mode == _MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO:
        sampler = optuna.samplers.NSGAIISampler(seed=int(seed))
    else:
        sampler = optuna.samplers.TPESampler(
            seed=int(seed),
            multivariate=True,
            group=True,
            constant_liar=True,
        )
    study_kwargs: dict[str, Any] = {
        "sampler": sampler,
        "storage": storage,
        "study_name": study_name,
        "load_if_exists": bool(storage and study_name),
    }
    if multi_objective_mode == _MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO:
        study_kwargs["directions"] = ["minimize"] * len(_SAME_CUTOFF_PARETO_OBJECTIVE_NAMES)
    else:
        study_kwargs["direction"] = "minimize"
    study = optuna.create_study(
        **study_kwargs,
    )
    policy_search_profile_key = _normalize_policy_search_profile(policy_search_profile)
    meta_feature_profile_key = _normalize_meta_feature_profile(meta_feature_profile)
    sanitized_trial_param_overrides = _sanitize_trial_param_overrides(trial_param_overrides)
    sanitized_benchmark_trial_param_overrides = _sanitize_benchmark_trial_param_overrides(
        benchmark_trial_param_overrides
    )
    suite_profile_key = str(suite_profile or "standard").strip().lower().replace("-", "_")
    normalized_base_policy = _normalize_active_policy(
        base_policy or AlgorithmPolicy.default(),
        meta_feature_profile=meta_feature_profile_key,
    )
    if enqueue_default:
        study.enqueue_trial(default_trial_params(policy_search_profile_key, meta_feature_profile=meta_feature_profile_key))
    if enqueue_records is None:
        raw_records = tuple(
            WarmStartCandidate(
                params=_sanitize_trial_params(
                    params,
                    meta_feature_profile=meta_feature_profile_key,
                    policy_search_profile=policy_search_profile_key,
                ),
                source_kind="legacy_enqueue",
                source_id=f"legacy_enqueue:{idx}",
                benchmark_id=selected[0].benchmark_id if len(selected) == 1 else None,
                family=selected[0].family if len(selected) == 1 else None,
            )
            for idx, params in enumerate(enqueue_trials)
        )
    else:
        raw_records = tuple(enqueue_records)
    enqueued_records, dedupe_skips = _dedupe_warm_start_candidates(
        raw_records,
        enqueue_default=enqueue_default,
        policy_search_profile=policy_search_profile_key,
        meta_feature_profile=meta_feature_profile_key,
    )
    enqueued_historical = [
        _sanitize_trial_params(
            record.params,
            meta_feature_profile=meta_feature_profile_key,
            policy_search_profile=policy_search_profile_key,
        )
        for record in enqueued_records
    ]
    for params in enqueued_historical:
        study.enqueue_trial(params)
    all_warm_start_skips = tuple(warm_start_skips) + tuple(dedupe_skips)
    telemetry_mode = str(optuna_best_telemetry or "auto").strip().lower().replace("-", "_")
    if telemetry_mode not in {"auto", "on", "off"}:
        raise ValueError("optuna_best_telemetry must be one of {'auto','on','off'}.")
    effective_record_id = str(telemetry_record_id or os.environ.get("PHASE3_RECORD_ID") or "").strip()
    telemetry = (
        _OptunaBestTelemetry(
            record_id=effective_record_id,
            min_interval_sec=float(optuna_best_telemetry_min_interval_sec),
        )
        if telemetry_mode == "on" or (telemetry_mode == "auto" and effective_record_id)
        else None
    )

    def _objective(trial: Any, *, run_lifecycle: _ManagedOptunaRunLifecycle | None = None) -> float | tuple[float, ...]:
        if mode == "oracle":
            return objective_oracle(
                trial,
                selected[0],
                output_dir=output_dir,
                seed=seed,
                runner=runner,
                objective_weights=config.weights,
                config=config,
                run_lifecycle=run_lifecycle,
                policy_search_profile=policy_search_profile_key,
                meta_feature_profile=meta_feature_profile_key,
                base_policy=base_policy,
                study_name=study.study_name,
                storage=storage,
                suite_profile=suite_profile_key,
                enqueued_records=enqueued_records,
                warm_start_skips=all_warm_start_skips,
                enqueue_default=enqueue_default,
                trial_param_overrides=sanitized_trial_param_overrides,
            )
        return objective_global_agnostic(
            trial,
            selected,
            output_dir=output_dir,
            seed=seed,
            runner=runner,
            config=config,
            benchmarks_per_trial_jobs=benchmarks_per_trial_jobs,
            run_lifecycle=run_lifecycle,
            policy_search_profile=policy_search_profile_key,
            meta_feature_profile=meta_feature_profile_key,
            progress_reporter=progress,
            base_policy=base_policy,
            study_name=study.study_name,
            storage=storage,
            suite_profile=suite_profile_key,
            enqueued_records=enqueued_records,
            warm_start_skips=all_warm_start_skips,
            enqueue_default=enqueue_default,
            trial_param_overrides=sanitized_trial_param_overrides,
            benchmark_trial_param_overrides=sanitized_benchmark_trial_param_overrides,
        )

    progress.append_event(
        "study_started",
        mode=mode,
        study_name=study.study_name,
        benchmark_count=len(selected),
        n_trials_requested=int(n_trials),
    )
    progress.write_status_snapshot(
        state="study_started",
        mode=mode,
        study_name=study.study_name,
        benchmark_count=len(selected),
        n_trials_requested=int(n_trials),
    )
    progress.write_active_processes([])

    def _build_summary(run_lifecycle: _ManagedOptunaRunLifecycle | None) -> dict[str, Any]:
        complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        best_trial = None
        pareto_best_trials: list[Any] = []
        if complete:
            if multi_objective_mode == _MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO:
                try:
                    pareto_best_trials = list(study.best_trials)
                except Exception as exc:
                    if run_lifecycle is not None:
                        run_lifecycle.cleanup_errors.append(f"best_trials:{type(exc).__name__}:{exc}")
            else:
                try:
                    best_trial = study.best_trial
                except Exception as exc:
                    if run_lifecycle is not None:
                        run_lifecycle.cleanup_errors.append(f"best_trial:{type(exc).__name__}:{exc}")
        roundtrip_audits: list[Mapping[str, Any]] = []
        objective_provenance = _jsonable(objective_provenance_payload(config))
        physical_target_manifest = objective_provenance.get("paper_i_phys_v1_targets") if isinstance(objective_provenance, Mapping) else None
        if physical_target_manifest is None and complete:
            attrs = dict(complete[0].user_attrs)
            physical_target_manifest = attrs.get("physical_target_manifest")
        for trial in study.trials:
            attrs = dict(trial.user_attrs)
            single = attrs.get("policy_roundtrip_audit")
            if isinstance(single, Mapping):
                roundtrip_audits.append(single)
            multi = attrs.get("policy_roundtrip_audits")
            if isinstance(multi, Mapping):
                roundtrip_audits.extend(
                    audit for audit in multi.values() if isinstance(audit, Mapping)
                )
        return {
            "generated_utc": _now_utc(),
            "pipeline": _PIPELINE_NAME,
            "mode": mode,
            "study_name": study.study_name,
            "storage": storage,
            "output_dir": str(output_dir),
            "n_trials_requested": int(n_trials),
            "n_jobs": int(n_jobs),
            "benchmarks_per_trial_jobs": int(benchmarks_per_trial_jobs),
            "seed": int(seed),
            "fixed_inner_optimizer": _ACTIVE_INNER_OPTIMIZER,
            "inner_optimizer_policy": _INNER_OPTIMIZER_POLICY_LABEL,
            "fixed_phase2_novelty_mode": _ACTIVE_PHASE2_NOVELTY_MODE,
            "policy_search_profile": policy_search_profile_key,
            "meta_feature_profile": meta_feature_profile_key,
            "suite_profile": suite_profile_key,
            "trial_param_overrides": _jsonable(sanitized_trial_param_overrides),
            "benchmark_trial_param_overrides": _jsonable(sanitized_benchmark_trial_param_overrides),
            "static_route_id": normalize_static_route_id(normalized_base_policy.static.static_route_id, default=ROUTE_ID_A),
            "phase0_aware": True,
            "meta_feature_policy": _meta_feature_policy_payload(normalized_base_policy),
            "phase0": {
                "schema": "phase0_optuna_policy_summary_v1",
                "phase0_aware": True,
                "phase0_is_route_identity": False,
                "policy_defaults": _jsonable(dict(PHASE0_OPTUNA_DEFAULTS)),
                "base_policy": {
                    "phase0_pilot_enabled": bool(normalized_base_policy.static.phase0_pilot_enabled),
                    "phase0_pilot_alpha": float(normalized_base_policy.static.phase0_pilot_alpha),
                    "phase0_pilot_threshold": float(normalized_base_policy.static.phase0_pilot_threshold),
                    "phase0_pilot_max_records": int(normalized_base_policy.static.phase0_pilot_max_records),
                    "phase0_lane_quota_pressure": float(normalized_base_policy.static.phase0_lane_quota_pressure),
                    "phase0_algebraic_lane_mode": str(normalized_base_policy.static.phase0_algebraic_lane_mode),
                },
                "max_records_semantics": "0_uncapped; Optuna defaults use capped records for CHTC safety",
            },
            "base_policy": _jsonable(asdict(normalized_base_policy)),
            "progress_reporter": _jsonable(progress.summary_payload()),
            "objective_weighting": _jsonable(objective_weighting_payload(selected, config)),
            "objective_provenance": objective_provenance,
            "discovery_objective_mode": _normalize_discovery_objective_mode(config.discovery_objective_mode),
            "multi_objective_mode": multi_objective_mode,
            "objective_vector_names": list(_SAME_CUTOFF_PARETO_OBJECTIVE_NAMES) if multi_objective_mode == _MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO else None,
            "terminal_proxy_behavior": (
                "inactive"
                if _normalize_discovery_objective_mode(config.discovery_objective_mode) == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING
                else "active_legacy_terminal_energy_reference_or_same_cutoff_proxy"
            ),
            "accuracy_policy": {
                "required_target_profile": objective_provenance.get("required_target_profile", "none") if isinstance(objective_provenance, Mapping) else "none",
                "feasibility_metric": (
                    "paper_i_phys_v1_primary_error"
                    if _normalize_discovery_objective_mode(config.discovery_objective_mode) == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING
                    else None
                ),
                "objective_mode": _normalize_discovery_objective_mode(config.discovery_objective_mode),
            },
            "physical_target_manifest": _jsonable(physical_target_manifest),
            "first_crossing_status": _first_crossing_status_summary(complete),
            "objective_replay": _effective_manifest_summary(study.trials),
            "best_trial_interpretation": _best_trial_interpretation(complete, best_trial),
            "best_trial_benchmark_terminal_errors": _terminal_error_rows_for_trial(best_trial),
            "required_target_benchmark_ids": list(config.required_target_benchmark_ids),
            "required_target_abs_delta_e": config.required_target_abs_delta_e,
            "required_target_penalty": float(config.required_target_penalty),
            "enqueue_default": bool(enqueue_default),
            "enqueue_trial_count": len(enqueued_historical),
            "enqueued_trial_params": _jsonable(enqueued_historical),
            "warm_start_count": len(enqueued_records),
            "warm_start_source_counts": _jsonable(_warm_start_counts(enqueued_records, all_warm_start_skips)),
            "selected_logical_pool_route": _jsonable(_selected_logical_route_summary(selected)),
            "enqueued_warm_start_records": _jsonable([_warm_start_candidate_json(record) for record in enqueued_records]),
            "skipped_warm_start_records": _jsonable([_warm_start_skip_json(record) for record in all_warm_start_skips]),
            "benchmarks": [asdict(spec) for spec in selected],
            "best_value": None if best_trial is None else _trial_value_payload(best_trial),
            "best_trial_number": None if best_trial is None else int(best_trial.number),
            "best_params": {} if best_trial is None else dict(best_trial.params),
            "best_user_attrs": {} if best_trial is None else _jsonable(dict(best_trial.user_attrs)),
            "pareto_front_size": len(pareto_best_trials),
            "pareto_best_trials": [_jsonable(_trial_brief(trial)) for trial in pareto_best_trials],
            "policy_roundtrip_audit_summary": _summarize_policy_roundtrip_audits(tuple(roundtrip_audits)),
            "managed_lifecycle": (
                {"enabled": False}
                if run_lifecycle is None
                else _jsonable(run_lifecycle.summary_payload())
            ),
            "trials": [
                {
                    "number": int(trial.number),
                    "state": str(trial.state.name),
                    "value": _trial_value_payload(trial),
                    "values": _trial_value_payload(trial) if isinstance(_trial_value_payload(trial), list) else None,
                    "params": dict(trial.params),
                    "user_attrs": _jsonable(dict(trial.user_attrs)),
                }
                for trial in study.trials
            ],
        }

    if int(max(1, n_jobs)) > 1:
        study.optimize(
            _objective,
            n_trials=int(max(0, n_trials)),
            n_jobs=int(max(1, n_jobs)),
            callbacks=([telemetry.maybe_log] if telemetry is not None and multi_objective_mode == _MULTI_OBJECTIVE_MODE_OFF else None),
        )
        summary = _build_summary(None)
        _write_json(output_dir / "summary.json", summary)
        progress.append_event("study_summary_written", mode=mode, study_name=study.study_name, summary_path=str(output_dir / "summary.json"))
        progress.write_status_snapshot(state="study_summary_written", mode=mode, study_name=study.study_name)
        return summary

    lifecycle = _ManagedOptunaRunLifecycle(optuna_module=optuna, progress_reporter=progress)
    summary: dict[str, Any] | None = None
    with lifecycle:
        lifecycle.attach_study(study)
        try:
            lifecycle.reconcile_existing_running_trials()
            for _ in range(int(max(0, n_trials))):
                if lifecycle.shutdown_requested.is_set():
                    break
                trial = study.ask()
                lifecycle.adopt_trial(trial)
                progress.append_event("trial_started", trial_number=int(trial.number), mode=mode, study_name=study.study_name)
                progress.write_current(state="trial_running", trial_number=int(trial.number), mode=mode, study_name=study.study_name)
                try:
                    value = _objective(trial, run_lifecycle=lifecycle)
                    if lifecycle.shutdown_requested.is_set():
                        lifecycle.fail_trial_number(trial.number, reason="shutdown_requested")
                        progress.append_event("trial_failed", trial_number=int(trial.number), reason="shutdown_requested")
                        break
                    tell_value = tuple(float(x) for x in value) if isinstance(value, tuple) else float(value)
                    study.tell(trial, tell_value)
                    if telemetry is not None and multi_objective_mode == _MULTI_OBJECTIVE_MODE_OFF:
                        telemetry.maybe_log(study, trial)
                    trial_progress_payload = _jsonable(_trial_telemetry_progress_payload(trial))
                    best_progress_payload = _optuna_best_progress_payload(study)
                    value_payload = list(tell_value) if isinstance(tell_value, tuple) else float(tell_value)
                    progress.append_event(
                        "trial_completed",
                        trial_number=int(trial.number),
                        value=value_payload,
                        state="COMPLETE",
                        **trial_progress_payload,
                        **best_progress_payload,
                    )
                    progress.write_current(
                        state="trial_completed",
                        trial_number=int(trial.number),
                        value=value_payload,
                        mode=mode,
                        study_name=study.study_name,
                        **trial_progress_payload,
                        **best_progress_payload,
                    )
                except BaseException as exc:
                    progress.append_event("trial_failed", trial_number=int(trial.number), exception_type=type(exc).__name__, exception=str(exc))
                    lifecycle.terminate_all_process_groups(f"trial_exception:{type(exc).__name__}")
                    lifecycle.fail_trial_number(trial.number, reason=f"trial_exception:{type(exc).__name__}")
                    raise
                finally:
                    lifecycle.finish_trial(trial.number)
        finally:
            lifecycle.terminate_all_process_groups("run_finally")
            lifecycle.fail_owned_running_trials("run_finally")
            summary = _build_summary(lifecycle)
            _write_json(output_dir / "summary.json", summary)
            progress.append_event("study_summary_written", mode=mode, study_name=study.study_name, summary_path=str(output_dir / "summary.json"))
            progress.write_status_snapshot(state="study_summary_written", mode=mode, study_name=study.study_name)
    return summary


def run_oracle_grid(
    *,
    specs: Sequence[HamiltonianBenchmarkSpec],
    output_dir: Path,
    progress_dir: str | Path | None = None,
    n_trials: int,
    seed: int = 7,
    n_jobs: int = 1,
    storage: str | None = None,
    study_prefix: str = "static_oracle",
    enqueue_default: bool = True,
    enqueue_trials_by_benchmark: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    enqueue_records_by_benchmark: Mapping[str, Sequence[WarmStartCandidate]] | None = None,
    warm_start_skips_by_benchmark: Mapping[str, Sequence[WarmStartSkip]] | None = None,
    runner: Callable[..., BenchmarkResult] = run_static_benchmark,
    config: GlobalObjectiveConfig = GlobalObjectiveConfig(),
    policy_search_profile: str | None = "default",
    meta_feature_profile: str | None = _DEFAULT_META_FEATURE_PROFILE,
    base_policy: AlgorithmPolicy | None = None,
    suite_profile: str | None = "standard",
    telemetry_record_id: str | None = None,
    optuna_best_telemetry: str = "auto",
    optuna_best_telemetry_min_interval_sec: float = 0.0,
    trial_param_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one oracle study per benchmark spec.

    Parallelism is delegated to Optuna's trial-level ``n_jobs`` and/or to
    separate process launches per family. This keeps the study database layout
    explicit and agent-readable.
    """

    output_dir = Path(output_dir)
    summaries: dict[str, Any] = {}
    meta_feature_profile_key = _normalize_meta_feature_profile(meta_feature_profile)
    for spec in specs:
        study_name = f"{study_prefix}/{spec.family}/{spec.benchmark_id}"
        summaries[spec.benchmark_id] = run_optuna_study(
            mode="oracle",
            specs=(spec,),
            output_dir=output_dir / spec.benchmark_id,
            n_trials=n_trials,
            seed=seed,
            n_jobs=n_jobs,
            storage=storage,
            study_name=study_name,
            enqueue_default=enqueue_default,
            enqueue_trials=() if enqueue_trials_by_benchmark is None else tuple(enqueue_trials_by_benchmark.get(spec.benchmark_id, ())),
            enqueue_records=None if enqueue_records_by_benchmark is None else tuple(enqueue_records_by_benchmark.get(spec.benchmark_id, ())),
            warm_start_skips=() if warm_start_skips_by_benchmark is None else tuple(warm_start_skips_by_benchmark.get(spec.benchmark_id, ())),
            runner=runner,
            config=config,
            policy_search_profile=policy_search_profile,
            meta_feature_profile=meta_feature_profile_key,
            progress_dir=None if progress_dir is None else Path(progress_dir) / spec.benchmark_id,
            base_policy=base_policy,
            suite_profile=suite_profile,
            telemetry_record_id=(
                f"{telemetry_record_id}/{spec.benchmark_id}" if telemetry_record_id else None
            ),
            optuna_best_telemetry=optuna_best_telemetry,
            optuna_best_telemetry_min_interval_sec=optuna_best_telemetry_min_interval_sec,
            trial_param_overrides=trial_param_overrides,
        )
    summary = {
        "generated_utc": _now_utc(),
        "pipeline": _PIPELINE_NAME,
        "mode": "oracle-grid",
        "output_dir": str(output_dir),
        "fixed_inner_optimizer": _ACTIVE_INNER_OPTIMIZER,
        "inner_optimizer_policy": _INNER_OPTIMIZER_POLICY_LABEL,
        "suite_profile": str(suite_profile or "standard").strip().lower().replace("-", "_"),
        "meta_feature_profile": meta_feature_profile_key,
        "trial_param_overrides": _jsonable(_sanitize_trial_param_overrides(trial_param_overrides)),
        "discovery_objective_mode": _normalize_discovery_objective_mode(config.discovery_objective_mode),
        "multi_objective_mode": _normalize_multi_objective_mode(config.multi_objective_mode),
        "objective_vector_names": list(_SAME_CUTOFF_PARETO_OBJECTIVE_NAMES) if _normalize_multi_objective_mode(config.multi_objective_mode) == _MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO else None,
        "objective_provenance": _jsonable(objective_provenance_payload(config)),
        "static_route_id": normalize_static_route_id(
            _normalize_active_policy(
                base_policy or AlgorithmPolicy.default(),
                meta_feature_profile=meta_feature_profile_key,
            ).static.static_route_id,
            default=ROUTE_ID_A,
        ),
        "meta_feature_policy": _jsonable(
            _meta_feature_policy_payload(
                _normalize_active_policy(
                    base_policy or AlgorithmPolicy.default(),
                    meta_feature_profile=meta_feature_profile_key,
                )
            )
        ),
        "phase0_aware": True,
        "phase0": {
            "schema": "phase0_optuna_policy_summary_v1",
            "phase0_aware": True,
            "phase0_is_route_identity": False,
            "policy_defaults": _jsonable(dict(PHASE0_OPTUNA_DEFAULTS)),
        },
        "benchmark_count": len(specs),
        "n_trials_per_benchmark": int(n_trials),
        "selected_logical_pool_route": _jsonable(_selected_logical_route_summary(specs)),
        "summaries": summaries,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


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


def _phase3_inner_zero_noise_exact_guard_eligible_from_args(
    args: argparse.Namespace,
    *,
    inner_mode: str,
    gradient_mode: str,
    value_noise_model: str,
    value_noise_std: float,
) -> bool:
    if str(inner_mode) != "noisy_v1":
        return False
    execution_surface = str(getattr(args, "phase3_oracle_execution_surface", "auto") or "auto").strip().lower()
    if execution_surface in {"", "auto"}:
        execution_surface = "expectation_v1"
    if execution_surface != "expectation_v1":
        return False
    if str(value_noise_model) != "off" or float(value_noise_std) != 0.0:
        return False
    if getattr(args, "phase3_oracle_value_noise_sigma0_abs", None) not in {None, ""}:
        return False
    if getattr(args, "phase3_oracle_value_noise_n_eff", None) not in {None, ""}:
        return False
    if float(getattr(args, "phase3_oracle_synthetic_depolarizing_1q_error", 0.0) or 0.0) != 0.0:
        return False
    if float(getattr(args, "phase3_oracle_synthetic_depolarizing_2q_error", 0.0) or 0.0) != 0.0:
        return False
    if float(getattr(args, "phase3_oracle_synthetic_coherent_1q_angle_std", 0.0) or 0.0) != 0.0:
        return False
    if float(getattr(args, "phase3_oracle_synthetic_coherent_2q_angle_std", 0.0) or 0.0) != 0.0:
        return False
    if str(getattr(args, "phase3_oracle_mitigation", "none") or "none").strip().lower() != "none":
        return False
    if str(getattr(args, "phase3_oracle_zne_scales", "") or "").strip():
        return False
    if bool(getattr(args, "phase3_oracle_local_gate_twirling", False)):
        return False
    if str(getattr(args, "phase3_oracle_dd_sequence", "") or "").strip().lower() not in {"", "none"}:
        return False
    return str(gradient_mode) in {
        "ideal",
        "aer_density_matrix_synthetic_depolarizing",
        "aer_density_matrix_synthetic_coherent",
    }


def _phase3_inner_value_noise_exact_structure_eligible_from_args(
    args: argparse.Namespace,
    *,
    inner_mode: str,
    gradient_mode: str,
    value_noise_model: str,
    value_noise_std: float,
) -> bool:
    if str(inner_mode) != "noisy_v1":
        return False
    execution_surface = str(getattr(args, "phase3_oracle_execution_surface", "auto") or "auto").strip().lower()
    if execution_surface in {"", "auto"}:
        execution_surface = "expectation_v1"
    if execution_surface != "expectation_v1":
        return False
    if str(value_noise_model) != "gaussian_iid_v1":
        return False
    try:
        std = float(value_noise_std)
    except (TypeError, ValueError):
        return False
    if (not math.isfinite(std)) or std <= 0.0:
        return False
    if float(getattr(args, "phase3_oracle_synthetic_depolarizing_1q_error", 0.0) or 0.0) != 0.0:
        return False
    if float(getattr(args, "phase3_oracle_synthetic_depolarizing_2q_error", 0.0) or 0.0) != 0.0:
        return False
    if float(getattr(args, "phase3_oracle_synthetic_coherent_1q_angle_std", 0.0) or 0.0) != 0.0:
        return False
    if float(getattr(args, "phase3_oracle_synthetic_coherent_2q_angle_std", 0.0) or 0.0) != 0.0:
        return False
    if str(getattr(args, "phase3_oracle_mitigation", "none") or "none").strip().lower() != "none":
        return False
    if str(getattr(args, "phase3_oracle_zne_scales", "") or "").strip():
        return False
    if bool(getattr(args, "phase3_oracle_local_gate_twirling", False)):
        return False
    if str(getattr(args, "phase3_oracle_dd_sequence", "") or "").strip().lower() not in {"", "none"}:
        return False
    return str(gradient_mode) in {
        "ideal",
        "aer_density_matrix_synthetic_depolarizing",
        "aer_density_matrix_synthetic_coherent",
    }


def _phase3_objective_provenance_from_args(args: argparse.Namespace) -> dict[str, Any]:
    inner_mode = str(args.phase3_oracle_inner_objective_mode or "exact").strip().lower() or "exact"
    value_noise_model = str(args.phase3_oracle_value_noise_model or "off").strip().lower() or "off"
    gradient_mode = str(args.phase3_oracle_gradient_mode or "off").strip().lower() or "off"
    value_noise_std, value_noise_contract = _resolve_value_noise_std_contract(
        label="phase3_oracle",
        value_noise_model=value_noise_model,
        value_noise_std=args.phase3_oracle_value_noise_std,
        value_noise_sigma0_abs=args.phase3_oracle_value_noise_sigma0_abs,
        value_noise_n_eff=args.phase3_oracle_value_noise_n_eff,
    )
    value_noise_payload = {
        **dict(value_noise_contract),
        "seed": args.phase3_oracle_value_noise_seed,
        "seed_policy": args.phase3_oracle_value_noise_seed_policy,
        "base_seed": args.phase3_oracle_value_noise_base_seed,
        "replicate_id": args.phase3_oracle_value_noise_replicate_id,
    }
    requested_noisy_inner = inner_mode == "noisy_v1" or value_noise_model != "off"
    zero_noise_exact_guard_eligible = _phase3_inner_zero_noise_exact_guard_eligible_from_args(
        args,
        inner_mode=inner_mode,
        gradient_mode=gradient_mode,
        value_noise_model=value_noise_model,
        value_noise_std=float(value_noise_std),
    )
    value_noise_exact_structure_eligible = (
        (not zero_noise_exact_guard_eligible)
        and _phase3_inner_value_noise_exact_structure_eligible_from_args(
            args,
            inner_mode=inner_mode,
            gradient_mode=gradient_mode,
            value_noise_model=value_noise_model,
            value_noise_std=float(value_noise_std),
        )
    )
    noisy_inner = bool(
        requested_noisy_inner
        and not zero_noise_exact_guard_eligible
        and not value_noise_exact_structure_eligible
    )
    objective_inner_noise_mode = (
        _VALUE_NOISE_ORACLE_INNER_EXACT_STRUCTURE_MODE
        if value_noise_exact_structure_eligible
        else ("phase3_noisy_v1" if noisy_inner else "exact_noiseless_v1")
    )
    objective_inner_guard_reason = (
        _ZERO_NOISE_ORACLE_INNER_EXACT_GUARD_REASON
        if zero_noise_exact_guard_eligible
        else (
            _VALUE_NOISE_ORACLE_INNER_EXACT_STRUCTURE_REASON
            if value_noise_exact_structure_eligible
            else None
        )
    )
    inner_consumes_noisy_energy = bool(noisy_inner or value_noise_exact_structure_eligible)
    payload = {
        "phase3_oracle_gradient_mode": gradient_mode,
        "phase3_oracle_backend_name": args.phase3_oracle_backend_name,
        "phase3_oracle_use_fake_backend": bool(args.phase3_oracle_use_fake_backend)
        if args.phase3_oracle_use_fake_backend is not None
        else False,
        "phase3_oracle_shots": args.phase3_oracle_shots,
        "phase3_oracle_repeats": args.phase3_oracle_repeats,
        "phase3_oracle_aggregate": args.phase3_oracle_aggregate,
        "phase3_oracle_seed": args.phase3_oracle_seed,
        "phase3_oracle_execution_surface": args.phase3_oracle_execution_surface,
        "phase3_oracle_inner_objective_mode": inner_mode,
        "phase3_oracle_value_noise_model": value_noise_model,
        "phase3_oracle_value_noise_std": float(value_noise_std),
        "phase3_oracle_value_noise_seed": args.phase3_oracle_value_noise_seed,
        "phase3_oracle_value_noise_sigma0_abs": args.phase3_oracle_value_noise_sigma0_abs,
        "phase3_oracle_value_noise_n_eff": args.phase3_oracle_value_noise_n_eff,
        "phase3_oracle_value_noise_seed_policy": args.phase3_oracle_value_noise_seed_policy,
        "phase3_oracle_value_noise_base_seed": args.phase3_oracle_value_noise_base_seed,
        "phase3_oracle_value_noise_replicate_id": args.phase3_oracle_value_noise_replicate_id,
        "phase3_oracle_value_noise": value_noise_payload,
        "phase3_oracle_synthetic_depolarizing_1q_error": args.phase3_oracle_synthetic_depolarizing_1q_error,
        "phase3_oracle_synthetic_depolarizing_2q_error": args.phase3_oracle_synthetic_depolarizing_2q_error,
        "phase3_oracle_synthetic_depolarizing_1q_gates": _canonical_phase3_oracle_gate_cli_value(
            args.phase3_oracle_synthetic_depolarizing_1q_gates,
            field_name="phase3_oracle_synthetic_depolarizing_1q_gates",
        ),
        "phase3_oracle_synthetic_depolarizing_2q_gates": _canonical_phase3_oracle_gate_cli_value(
            args.phase3_oracle_synthetic_depolarizing_2q_gates,
            field_name="phase3_oracle_synthetic_depolarizing_2q_gates",
        ),
        "phase3_oracle_synthetic_coherent_1q_angle_std": args.phase3_oracle_synthetic_coherent_1q_angle_std,
        "phase3_oracle_synthetic_coherent_2q_angle_std": args.phase3_oracle_synthetic_coherent_2q_angle_std,
        "phase3_oracle_synthetic_coherent_seed": args.phase3_oracle_synthetic_coherent_seed,
        "phase3_oracle_synthetic_coherent_generator_mode": args.phase3_oracle_synthetic_coherent_generator_mode,
        "phase3_oracle_synthetic_coherent_1q_gates": _canonical_phase3_oracle_gate_cli_value(
            args.phase3_oracle_synthetic_coherent_1q_gates,
            field_name="phase3_oracle_synthetic_coherent_1q_gates",
        ),
        "phase3_oracle_synthetic_coherent_2q_gates": _canonical_phase3_oracle_gate_cli_value(
            args.phase3_oracle_synthetic_coherent_2q_gates,
            field_name="phase3_oracle_synthetic_coherent_2q_gates",
        ),
        "adapt_noise_floor_stop_policy": args.adapt_noise_floor_stop_policy,
        "adapt_noise_floor_snr_threshold": args.adapt_noise_floor_snr_threshold,
        "adapt_noise_floor_n_rem_high_threshold": args.adapt_noise_floor_n_rem_high_threshold,
        "adapt_noise_floor_useful_horizon_threshold": args.adapt_noise_floor_useful_horizon_threshold,
        "phase3_adapt_parallel_gradient_workers": int(args.phase3_adapt_parallel_gradient_workers),
        "phase3_adapt_beam_parent_workers": int(args.phase3_adapt_beam_parent_workers),
        "phase3_adapt_beam_lambda": float(args.phase3_adapt_beam_lambda),
        "static_route_id": normalize_static_route_id(args.static_route_id, default=ROUTE_ID_A),
        "meta_feature_profile": _normalize_meta_feature_profile(args.meta_feature_profile),
        "hardware_resolution_mode": _static_hardware_resolution_mode(args.hardware_resolution_mode),
        "objective_inner_optimization_noise_mode": objective_inner_noise_mode,
        "objective_inner_optimization_noise_mode_requested": (
            "phase3_noisy_v1" if requested_noisy_inner else "exact_noiseless_v1"
        ),
        "objective_inner_optimization_zero_noise_exact_guard_eligible": bool(
            zero_noise_exact_guard_eligible
        ),
        "objective_inner_optimization_value_noise_exact_structure_eligible": bool(
            value_noise_exact_structure_eligible
        ),
        "objective_inner_optimization_runtime_guard_reason": objective_inner_guard_reason,
        "objective_inner_optimization_preserves_exact_structure": bool(
            zero_noise_exact_guard_eligible or value_noise_exact_structure_eligible
        ),
        "objective_inner_optimization_value_noise_scalar_only": bool(
            value_noise_exact_structure_eligible
        ),
        "objective_inner_optimization_oracle_gradient_value_noise_suppressed": bool(
            value_noise_exact_structure_eligible
        ),
        "objective_final_score_noise_mode": "exact_noiseless_v1",
        "objective_final_score_consumes_noisy_energy": False,
        "objective_inner_optimization_consumes_noisy_energy": bool(inner_consumes_noisy_energy),
        "objective_inner_optimization_consumes_noisy_energy_requested": bool(
            requested_noisy_inner
        ),
    }
    if value_noise_exact_structure_eligible:
        payload["objective_noise_mode"] = "phase3_inner_exact_structure_value_noise_final_score_exact_v1"
    elif noisy_inner:
        payload["objective_noise_mode"] = "phase3_inner_noisy_v1_final_score_exact_v1"
    return payload


def _base_policy_from_args(args: argparse.Namespace) -> AlgorithmPolicy:
    workers = int(args.phase3_adapt_parallel_gradient_workers)
    if workers < 0:
        raise ValueError("--phase3-adapt-parallel-gradient-workers must be >= 0 (0=auto)")
    beam_parent_workers = int(args.phase3_adapt_beam_parent_workers)
    if beam_parent_workers < 0:
        raise ValueError("--phase3-adapt-beam-parent-workers must be >= 0 (0=auto)")
    adapt_beam_lambda_arg = float(args.phase3_adapt_beam_lambda)
    if (not math.isfinite(adapt_beam_lambda_arg)) or adapt_beam_lambda_arg < 0.0:
        raise ValueError("--phase3-adapt-beam-lambda must be finite and >= 0")
    default_policy = AlgorithmPolicy.default()
    static_kwargs: dict[str, Any] = {
        "static_route_id": normalize_static_route_id(args.static_route_id, default=ROUTE_ID_A),
        "static_meta_feature_profile": _normalize_meta_feature_profile(args.meta_feature_profile),
        "hardware_resolution_mode": _static_hardware_resolution_mode(args.hardware_resolution_mode),
        "hardware_resolution_profile_json": _static_hardware_profile_value(args.hardware_resolution_profile_json),
        "hardware_resolution_profile_name": _static_hardware_profile_value(args.hardware_resolution_profile_name),
        "adapt_parallel_gradient_workers": workers,
        "adapt_beam_parent_workers": beam_parent_workers,
        "adapt_beam_lambda": float(adapt_beam_lambda_arg),
    }
    for field_name in PHASE0_OPTUNA_DEFAULTS:
        value = getattr(args, field_name, None)
        if value is not None:
            static_kwargs[field_name] = value
    for field_name in (
        "phase3_selector_policy",
        "phase3_selector_geometry_mode",
        "phase3_novelty_ablation_mode",
        "phase3_window_relaxation_mode",
        "phase3_plateau_acquisition_mode",
        "phase3_plateau_acquisition_score",
        "phase3_plateau_unlock_margin",
        "phase3_plateau_duplicate_policy",
        "phase3_plateau_lambda_vol",
        "phase3_plateau_sigma_min",
        "phase3_plateau_nu_min",
        "phase3_plateau_volume_min",
        "phase3_plateau_failed_family_patience",
        "phase3_plateau_trial_optimizer",
        "phase3_plateau_trial_qngd_maxiter",
        "phase3_batch_selection_mode",
        "phase3_batch_prefilter_mode",
        "phase3_hardware_cost_normalization_mode",
        "phase3_runtime_split_mode",
        "phase3_runtime_split_selection_mode",
        "phase3_runtime_split_max_subset_size",
        "shared_pauli_pool_mode",
        "shared_pauli_pool_symmetry_policy",
        "shared_pauli_pool_max_subset_size",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            static_kwargs[field_name] = value
    if getattr(args, "allow_archival_phase3_runtime_split", None) is not None:
        static_kwargs["allow_archival_phase3_runtime_split"] = bool(args.allow_archival_phase3_runtime_split)
    if getattr(args, "phase3_enable_batching", None) is not None:
        static_kwargs["phase2_enable_batching"] = bool(args.phase3_enable_batching)
    if getattr(args, "phase2_rho", None) is not None:
        phase2_rho = float(args.phase2_rho)
        if not math.isfinite(phase2_rho) or phase2_rho <= 0.0:
            raise ValueError("--phase2-rho must be finite and > 0")
        static_kwargs["phase2_rho"] = phase2_rho
    for arg_name, flag in (
        ("phase1_score_z_alpha", "--phase1-score-z-alpha"),
        ("phase2_score_z_alpha", "--phase2-score-z-alpha"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            z_alpha = float(value)
            if not math.isfinite(z_alpha) or z_alpha < 0.0:
                raise ValueError(f"{flag} must be finite and >= 0")
            static_kwargs[arg_name] = z_alpha
    if getattr(args, "phase1_prune_enabled", None) is not None:
        static_kwargs["phase1_prune_enabled"] = bool(args.phase1_prune_enabled)
    if getattr(args, "adapt_allow_repeats", None) is not None:
        static_kwargs["adapt_allow_repeats"] = bool(args.adapt_allow_repeats)
        static_kwargs["adapt_allow_repeats_override"] = bool(args.adapt_allow_repeats)
    for field_name, _ in _PHASE3_ORACLE_CLI_OPTIONS:
        value = getattr(args, field_name)
        if value is not None:
            static_kwargs[field_name] = value
    for field_name in (
        "phase3_oracle_value_noise_seed_policy",
        "phase3_oracle_value_noise_base_seed",
        "phase3_oracle_value_noise_replicate_id",
        "adapt_noise_floor_stop_policy",
        "adapt_noise_floor_snr_threshold",
        "adapt_noise_floor_n_rem_high_threshold",
        "adapt_noise_floor_useful_horizon_threshold",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            static_kwargs[field_name] = value
    if args.phase3_oracle_use_fake_backend is not None:
        static_kwargs["phase3_oracle_use_fake_backend"] = bool(args.phase3_oracle_use_fake_backend)
    return replace(default_policy, static=replace(default_policy.static, **static_kwargs))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Size-aware Optuna outer policy studies for canonical phase3 static ADAPT.")
    p.add_argument("--mode", choices=["global", "oracle-grid"], default="global")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--progress-dir", type=Path, default=None, help="Optional directory for fail-soft JSONL/current progress artifacts.")
    p.add_argument("--n-trials", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=1, help="Optuna trial-level jobs. Keep this at 1 for canonical global local runs.")
    p.add_argument("--benchmarks-per-trial-jobs", type=int, default=1, help="Global-mode benchmark subprocesses to evaluate concurrently inside one Optuna trial. Use <=0 for all selected benchmarks.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///phase3_policy.db")
    p.add_argument("--study-prefix", default="phase3_policy")
    p.add_argument(
        "--canonical-lane",
        choices=sorted(_CANONICAL_LANE_FAMILIES),
        default=None,
        help=(
            "Named coarse policy lane. When set, the suite is restricted to the "
            "fermionic, bosonic, mixed fermion-boson, or molecular canonical-policy family."
        ),
    )
    p.add_argument(
        "--canonical-lane-stage",
        choices=["train", "transfer", "all"],
        default="train",
        help=(
            "For a named canonical lane, select the train role, transfer role, "
            "or all wired sizes; molecular lanes use role tags rather than fixed L=2/L=3 sizes."
        ),
    )
    p.add_argument(
        "--fixed-inner-optimizer",
        choices=[_ACTIVE_INNER_OPTIMIZER],
        default=_ACTIVE_INNER_OPTIMIZER,
        help="Phase3 Optuna route identity for this process; default is SPSA unless PHASE3_POLICY_INNER_OPTIMIZER is set before launch.",
    )
    p.add_argument(
        "--fixed-phase2-novelty-mode",
        choices=[_ACTIVE_PHASE2_NOVELTY_MODE],
        default=_ACTIVE_PHASE2_NOVELTY_MODE,
        help=(
            "Fixed Phase-2 novelty route identity for this process; default is collective_span_v1 unless "
            "PHASE3_POLICY_PHASE2_NOVELTY_MODE is set before launch. Used for A/B bakeoffs."
        ),
    )
    p.add_argument(
        "--static-route-id",
        default=ROUTE_ID_A,
        help="Static SNAKE route identity overlay applied to sampled child ADAPT policies.",
    )
    p.add_argument(
        "--hardware-resolution-mode",
        choices=["ideal", "profile"],
        default="ideal",
        help="Hardware-resolution overlay for sampled child ADAPT policies; Route-A requires ideal.",
    )
    p.add_argument("--hardware-resolution-profile-json", default=None)
    p.add_argument("--hardware-resolution-profile-name", default=None)
    p.add_argument(
        "--phase3-adapt-parallel-gradient-workers",
        type=int,
        default=1,
        help="Forwarded exact-gradient worker count for child ADAPT runs; 0 enables CPU-aware auto sizing."
    )
    p.add_argument(
        "--phase3-adapt-beam-parent-workers",
        type=int,
        default=1,
        help="Forwarded exact/noiseless beam-parent worker count for child ADAPT runs; 0 enables CPU-aware auto sizing."
    )
    p.add_argument(
        "--phase3-adapt-beam-lambda",
        type=float,
        default=0.0,
        help="Forwarded energy-cost beam survival lambda; fixed unless explicitly overridden."
    )
    p.set_defaults(phase0_pilot_enabled=None)
    p.add_argument("--phase0-pilot-enabled", dest="phase0_pilot_enabled", action="store_true")
    p.add_argument("--phase0-no-pilot", dest="phase0_pilot_enabled", action="store_false")
    p.add_argument("--phase0-pilot-alpha", type=float, default=None)
    p.add_argument("--phase0-pilot-threshold", type=float, default=None)
    p.add_argument("--phase0-pilot-max-records", type=int, default=None)
    p.add_argument("--phase0-lane-quota-pressure", type=float, default=None)
    p.add_argument("--phase0-algebraic-lane-mode", choices=list(PHASE0_ALGEBRAIC_LANE_MODE_CHOICES), default=None)
    p.add_argument("--phase3-selector-policy", choices=list(_PHASE3_SELECTOR_POLICY_CHOICES), default=None)
    p.add_argument("--phase3-selector-geometry-mode", choices=list(_PHASE3_SELECTOR_GEOMETRY_MODE_CHOICES), default=None)
    p.add_argument("--phase3-novelty-ablation-mode", choices=list(_PHASE3_NOVELTY_ABLATION_MODE_CHOICES), default=None)
    p.add_argument("--phase3-window-relaxation-mode", choices=list(_PHASE3_WINDOW_RELAXATION_MODE_CHOICES), default=None)
    p.add_argument("--phase3-plateau-acquisition-mode", choices=["off", "novelty_cost_v1"], default=None)
    p.add_argument("--phase3-plateau-acquisition-score", choices=list(PLATEAU_ACQUISITION_SCORE_CHOICES), default=None)
    p.add_argument("--phase3-plateau-unlock-margin", type=float, default=None)
    p.add_argument(
        "--phase3-plateau-duplicate-policy",
        choices=["block_exact_position_v1", "allow_exact_position_replay"],
        default=None,
    )
    p.add_argument("--phase3-plateau-lambda-vol", type=float, default=None)
    p.add_argument("--phase3-plateau-sigma-min", type=float, default=None)
    p.add_argument("--phase3-plateau-nu-min", type=float, default=None)
    p.add_argument("--phase3-plateau-volume-min", type=float, default=None)
    p.add_argument("--phase3-plateau-failed-family-patience", type=int, default=None)
    p.add_argument("--phase3-plateau-trial-optimizer", choices=list(PLATEAU_TRIAL_OPTIMIZER_CHOICES), default=None)
    p.add_argument("--phase3-plateau-trial-qngd-maxiter", type=int, default=None)
    p.add_argument("--phase3-batch-selection-mode", choices=list(_PHASE3_BATCH_SELECTION_MODE_CHOICES), default=None)
    p.add_argument("--phase3-batch-prefilter-mode", choices=list(_PHASE3_BATCH_PREFILTER_MODE_CHOICES), default=None)
    p.add_argument(
        "--phase2-rho",
        type=float,
        default=None,
        help="Forwarded trust-region radius for child static ADAPT scoring.",
    )
    p.add_argument(
        "--phase1-score-z-alpha",
        type=float,
        default=None,
        help="Optional forwarded Phase-1 confidence multiplier for child static ADAPT scoring.",
    )
    p.add_argument(
        "--phase2-score-z-alpha",
        type=float,
        default=None,
        help=(
            "Optional forwarded Phase-2/3 confidence multiplier for child static ADAPT scoring; "
            "child ADAPT defaults to Phase-1 z_alpha when omitted."
        ),
    )
    p.add_argument(
        "--phase3-hardware-cost-normalization-mode",
        choices=["family_robust_v1", "raw_legacy_v1"],
        default=None,
        help="Source-lock compatibility override for the child ADAPT hardware-cost denominator mode.",
    )
    p.add_argument(
        "--phase3-runtime-split-mode",
        choices=["off", "shortlist_pauli_children_v1"],
        default=None,
        help=(
            "Diagnostic-only Phase-3 runtime split override forwarded to child ADAPT runs. "
            "Production defaults keep this off."
        ),
    )
    p.add_argument(
        "--allow-archival-phase3-runtime-split",
        action="store_true",
        default=None,
        help="Required when --phase3-runtime-split-mode is not off; keeps Pauli-child split runs explicitly diagnostic.",
    )
    p.add_argument(
        "--phase3-runtime-split-selection-mode",
        choices=[
            "proxy_child_set_preselection",
            "full_child_set_scoring",
            "parent_family_sum_top2_scoring",
        ],
        default=None,
    )
    p.add_argument("--phase3-runtime-split-max-subset-size", type=int, default=None)
    p.add_argument(
        "--shared-pauli-pool-mode",
        choices=["off", "shared_pauli_child_sets_v1", "pauli_child_sets_v1", "global_pauli_child_sets_v1"],
        default=None,
        help="Canonical shared parent-plus-Pauli-child-set pool mode forwarded to child ADAPT runs.",
    )
    p.add_argument(
        "--shared-pauli-pool-symmetry-policy",
        choices=["off", "hard_guard"],
        default=None,
        help="Symmetry policy for the canonical shared Pauli-child pool.",
    )
    p.add_argument("--shared-pauli-pool-max-subset-size", type=int, default=None)
    p.set_defaults(phase3_enable_batching=None)
    p.add_argument("--phase3-enable-batching", dest="phase3_enable_batching", action="store_true")
    p.add_argument("--phase3-no-batching", dest="phase3_enable_batching", action="store_false")
    p.set_defaults(phase1_prune_enabled=None)
    p.add_argument("--phase1-prune-enabled", dest="phase1_prune_enabled", action="store_true")
    p.add_argument("--phase1-no-prune", dest="phase1_prune_enabled", action="store_false")
    p.set_defaults(adapt_allow_repeats=None)
    p.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.add_argument(
        "--phase3-oracle-gradient-mode",
        choices=[
            "off",
            "ideal",
            "shots",
            "aer_noise",
            "aer_density_matrix",
            "aer_density_matrix_synthetic_depolarizing",
            "aer_density_matrix_synthetic_coherent",
            "backend_scheduled",
            "runtime",
        ],
        default=None,
    )
    p.add_argument("--phase3-oracle-backend-name", default=None)
    p.add_argument("--phase3-oracle-use-fake-backend", action="store_true", default=None)
    p.add_argument("--phase3-oracle-shots", type=int, default=None)
    p.add_argument("--phase3-oracle-repeats", type=int, default=None)
    p.add_argument("--phase3-oracle-aggregate", choices=["mean"], default=None)
    p.add_argument("--phase3-oracle-seed", type=int, default=None)
    p.add_argument("--phase3-oracle-gradient-step", type=float, default=None)
    p.add_argument(
        "--phase3-oracle-execution-surface",
        choices=["auto", "expectation_v1", "raw_measurement_v1"],
        default=None,
    )
    p.add_argument("--phase3-oracle-inner-objective-mode", choices=["exact", "noisy_v1"], default=None)
    p.add_argument("--phase3-oracle-value-noise-model", choices=["off", "gaussian_iid_v1"], default=None)
    p.add_argument("--phase3-oracle-value-noise-std", type=float, default=None)
    p.add_argument("--phase3-oracle-value-noise-seed", type=int, default=None)
    p.add_argument("--phase3-oracle-value-noise-sigma0-abs", type=float, default=None)
    p.add_argument("--phase3-oracle-value-noise-n-eff", type=float, default=None)
    p.add_argument("--phase3-oracle-synthetic-depolarizing-1q-error", type=float, default=None)
    p.add_argument("--phase3-oracle-synthetic-depolarizing-2q-error", type=float, default=None)
    p.add_argument("--phase3-oracle-synthetic-depolarizing-1q-gates", default=None)
    p.add_argument("--phase3-oracle-synthetic-depolarizing-2q-gates", default=None)
    p.add_argument("--phase3-oracle-synthetic-coherent-1q-angle-std", type=float, default=None)
    p.add_argument("--phase3-oracle-synthetic-coherent-2q-angle-std", type=float, default=None)
    p.add_argument("--phase3-oracle-synthetic-coherent-seed", type=int, default=None)
    p.add_argument("--phase3-oracle-synthetic-coherent-generator-mode", default=None)
    p.add_argument("--phase3-oracle-synthetic-coherent-1q-gates", default=None)
    p.add_argument("--phase3-oracle-synthetic-coherent-2q-gates", default=None)
    p.add_argument("--phase3-oracle-value-noise-seed-policy", default=None)
    p.add_argument("--phase3-oracle-value-noise-base-seed", type=int, default=None)
    p.add_argument("--phase3-oracle-value-noise-replicate-id", default=None)
    p.add_argument("--adapt-noise-floor-stop-policy", choices=["off", "noise_floor_agreement_v1"], default=None)
    p.add_argument("--adapt-noise-floor-snr-threshold", type=float, default=None)
    p.add_argument("--adapt-noise-floor-n-rem-high-threshold", type=float, default=None)
    p.add_argument("--adapt-noise-floor-useful-horizon-threshold", type=float, default=None)
    p.add_argument(
        "--policy-search-profile",
        choices=list(_POLICY_SEARCH_PROFILES),
        default="default",
        help=(
            "Optional coarse constraints on the Optuna policy surface. "
            "fermionic_protected_correlation widens early shortlist/insertion/refit opportunity "
            "and delays premature prune/drop pressure without adding motif score bias. "
            "bosonic_fullmeta_compact forces full-meta bosonic generators while capping beam "
            "branching to avoid harmonic-Kerr timeout explosions."
        ),
    )
    p.add_argument(
        "--meta-feature-profile",
        choices=list(_META_FEATURE_PROFILES),
        default=_DEFAULT_META_FEATURE_PROFILE,
        help=(
            "Allowlisted Optuna feature-toggle scope. safe_core_v1 samples "
            "approved feature on/off choices while holding the hard SNAKE "
            "selector, novelty, pool, and no-ED-leakage identity fixed. "
            "paper_i_production_v1 forces the clean Paper-I production "
            "route (candidate-position SNAKE, Phase0/algebraic/prune on, "
            "reduced Phase-III geometry, phase liveness off, zero "
            "position-shift/family-repeat/motif priors) while leaving "
            "batching and repeats as approved Optuna toggles. "
            "spin_boson_2q_batching_v1 and route_a_batching_v1 are narrow "
            "rerun profiles that keep the same production identity while "
            "forcing batching on."
        ),
    )
    p.add_argument(
        "--required-target-profile",
        choices=list(_REQUIRED_TARGET_PROFILES),
        default="none",
        help=(
            "Optional hard feasibility target set. fermionic_hubbard_core requires Hubbard/ttprime "
            "L2 train cases to meet the target before cost can win."
        ),
    )
    p.add_argument(
        "--calibration-profile",
        choices=list(_CALIBRATION_PROFILE_CHOICES),
        default="off",
        help="Optional additive calibration suite; supports HK/HH and weighted seven-case nph2/ref3 Route-A profiles.",
    )
    p.add_argument(
        "--required-target-benchmark-id",
        action="append",
        default=None,
        help="Additional benchmark_id that must meet --required-target-abs-delta-e or receive a large penalty.",
    )
    p.add_argument(
        "--required-target-abs-delta-e",
        type=float,
        default=None,
        help="Hard feasibility target for required benchmark IDs. Defaults to --target-abs-delta-e when omitted.",
    )
    p.add_argument(
        "--required-target-penalty",
        type=float,
        default=1000.0,
        help="Penalty scale added for each required benchmark that misses its hard target.",
    )
    p.add_argument("--suite-profile", default="standard", help="Machine-readable suite/profile label recorded in summaries and oracle warm-start guards.")
    p.add_argument(
        "--benchmark-id",
        action="append",
        default=None,
        help="Exact benchmark_id to include. Repeat to build a precise Table-I/Hamiltonian slice after family/size filters.",
    )
    p.add_argument("--families", nargs="+", default=list(_DEFAULT_FAMILIES))
    p.add_argument("--sizes", nargs="+", type=int, default=None, help="Optional lattice sizes to include. Omit to keep registry-limited cases such as spin_boson_L1.")
    p.add_argument("--boson-cutoff", type=int, default=None, help="Override --n-ph-max for bosonic benchmark specs in this study. Defaults remain family-specific.")
    p.add_argument("--boson-cutoffs", nargs="+", type=int, default=None, help="Create separate bosonic benchmark instances for each listed ADAPT n_ph_max, e.g. 1 2 3. Overrides --boson-cutoff for bosonic families.")
    p.add_argument("--exact-reference-boson-cutoff", type=int, default=4, help="Higher n_ph_max used only for bosonic exact-reference objectives. Use 0 to disable high-cutoff references.")
    p.add_argument(
        "--force-same-cutoff-objective",
        action="store_true",
        help=(
            "After benchmark selection, clear bosonic exact-reference cutoffs so "
            "Optuna scoring and child target checks use the working/same-cutoff ED energy."
        ),
    )
    p.add_argument("--physics-grid-profile", choices=["canonical", "small_robust", "robust", "paper_i_clean"], default="canonical", help="Physics-parameter perturbation suite. small_robust adds one held-in perturbation per family at smallest wired size; paper_i_clean selects the clean weak/strong Paper-I grid.")
    p.add_argument("--molecular-problem-json", type=Path, default=None, help="Closed-shell molecular problem JSON override to include as a single molecular train case. Defaults to the built-in H2 control, LiH train-primary, and H2O transfer suite when present.")
    p.add_argument("--historical-ledger", type=Path, default=None, help="Historical-best ledger JSON used to set per-spec baselines and enqueue old-good policy seeds.")
    p.add_argument(
        "--selected-logical-route",
        choices=["standard", "historical-selected", "historical_selected"],
        default="standard",
        help=(
            "Pool route for canonical/global studies. standard forces ADAPT selected-logical mode off; "
            "historical-selected materializes per-spec selected-logical sidecars and passes the opt-in ADAPT flags."
        ),
    )
    p.add_argument(
        "--selected-logical-source-json",
        type=Path,
        default=None,
        help=(
            "Selected-logical JSON or historical ledger used by --selected-logical-route historical-selected. "
            "If omitted, --historical-ledger is reused when present."
        ),
    )
    p.add_argument(
        "--selected-logical-transfer-mode",
        choices=["exact_match_v1", "boundary_v1"],
        default="exact_match_v1",
        help="Transfer mode forwarded to ADAPT selected-logical pool filtering.",
    )
    p.add_argument("--oracle-summary-root", type=Path, action="append", default=None, help="Prior oracle/global summary file or directory. Best params are enqueued as warm starts.")
    p.add_argument("--oracle-enqueue-limit", type=int, default=8, help="Maximum prior oracle/global best-param vectors to enqueue. Use a negative value for no limit.")
    p.add_argument("--oracle-required-static-route-id", default=None, help="Require oracle warm starts to declare this static route id.")
    p.add_argument("--oracle-required-suite-profile", default=None, help="Require oracle warm starts to declare this suite/profile label.")
    p.add_argument("--oracle-require-phase0-aware", action="store_true", help="Require oracle summaries to advertise Phase0-aware policy metadata.")
    p.add_argument("--oracle-require-compatible-warm-starts", action="store_true", help="Fail before Optuna if any selected benchmark lacks a compatible oracle warm start.")
    p.add_argument(
        "--enqueue-trial-params-json",
        type=Path,
        default=None,
        help="Explicit auditable warm-start records with trial params to enqueue before historical/oracle records.",
    )
    p.add_argument("--trial-timeout-sec", type=float, default=None, help="Wall-clock timeout for each ADAPT subprocess trial. Timed-out trials receive a failure penalty and Optuna continues.")
    p.add_argument(
        "--trial-prune-depth",
        type=int,
        default=None,
        help=(
            "Optional trial-level comparator gate. If the live ADAPT checkpoint "
            "exceeds this accepted depth and the selected error metric is still "
            "above --trial-prune-abs-delta-e, terminate that trial."
        ),
    )
    p.add_argument(
        "--trial-prune-abs-delta-e",
        type=float,
        default=None,
        help="Comparator error threshold used with --trial-prune-depth.",
    )
    p.add_argument(
        "--trial-prune-metric",
        choices=["same_cutoff_abs_delta_e", "target_abs_delta_e", "benchmark_target_abs_delta_e", "reference_abs_delta_e"],
        default="same_cutoff_abs_delta_e",
        help="Live error metric used by the trial-level comparator gate.",
    )
    p.add_argument("--compile-timeout-sec", type=float, default=900.0, help="Wall-clock timeout for each compile-scout subprocess. Use <=0 to disable.")
    p.add_argument("--telemetry-record-id", default=None, help="Record identifier printed in live OPTUNA_BEST stdout telemetry.")
    p.add_argument("--optuna-best-telemetry", choices=["auto", "on", "off"], default="auto", help="Emit live OPTUNA_BEST stdout lines after best-trial changes.")
    p.add_argument("--optuna-best-telemetry-min-interval-sec", type=float, default=0.0, help="Minimum seconds between repeated OPTUNA_BEST lines for the same best trial.")
    p.add_argument("--trial-param-overrides-json", type=Path, default=None, help="JSON object of policy parameter overrides applied to every trial after Optuna sampling.")
    p.add_argument(
        "--benchmark-trial-param-overrides-json",
        type=Path,
        default=None,
        help=(
            "JSON mapping from benchmark_id, benchmark_id:<id>, family:<family>, or default to policy "
            "parameter overrides applied only to that child benchmark after joint Optuna sampling."
        ),
    )
    p.add_argument("--no-enqueue-default", action="store_true")
    p.add_argument("--no-enqueue-historical", action="store_true")
    p.add_argument("--gamma-robust", type=float, default=GlobalObjectiveConfig.gamma_robust)
    p.add_argument("--gamma-family-std", type=float, default=GlobalObjectiveConfig.gamma_family_std)
    p.add_argument("--gamma-fail", type=float, default=GlobalObjectiveConfig.gamma_fail)
    p.add_argument("--cvar-quantile", type=float, default=GlobalObjectiveConfig.robust_cvar_quantile)
    p.add_argument("--objective-profile", choices=["balanced", "cost_at_accuracy", "paper_i_discovery_first_crossing", "same_cutoff_pareto"], default="balanced")
    p.add_argument(
        "--discovery-objective-mode",
        choices=list(_DISCOVERY_OBJECTIVE_MODES),
        default=_DISCOVERY_OBJECTIVE_MODE_TERMINAL_PROXY,
        help=(
            "Opt-in Paper-I objective mode. terminal_proxy preserves legacy terminal score; "
            "discovery_first_crossing minimizes first-crossing resource subject to paper_i_phys_v1."
        ),
    )
    p.add_argument("--target-abs-delta-e", type=float, default=1e-5)
    p.add_argument("--objective-energy-weight", type=float, default=None)
    p.add_argument("--objective-2q-weight", type=float, default=None)
    p.add_argument("--objective-2q-depth-weight", type=float, default=None)
    p.add_argument("--objective-depth-weight", type=float, default=None)
    p.add_argument("--objective-parameter-weight", type=float, default=None)
    p.add_argument("--objective-shot-weight", type=float, default=None)
    p.add_argument(
        "--objective-weight-preset",
        choices=tuple(_OBJECTIVE_WEIGHT_PRESETS),
        default=_OBJECTIVE_WEIGHT_PRESET_UNIFORM,
        help="Benchmark weighting preset for global objective aggregation; uniform preserves legacy equal weighting.",
    )
    p.add_argument(
        "--objective-family-weights",
        default=None,
        help="Optional comma-separated family=weight overrides for global objective aggregation.",
    )
    p.add_argument(
        "--objective-benchmark-weights",
        default=None,
        help="Optional comma-separated benchmark_id=weight overrides for global objective aggregation.",
    )
    p.add_argument(
        "--robustness-gate",
        choices=["auto", "require", "off"],
        default="auto",
        help=(
            "Static Hubbard L2 Phase3 preflight gate. auto runs before broad/global studies "
            "but skips tiny single-case smoke launches; require always runs; off disables it."
        ),
    )
    p.add_argument(
        "--robustness-gate-lanes",
        nargs="+",
        choices=["SPSA", "POWELL"],
        default=["SPSA", "POWELL"],
        help="Inner-optimizer lanes required by the static Hubbard L2 robustness gate.",
    )
    p.add_argument(
        "--robustness-gate-target-abs-delta-e",
        type=float,
        default=1e-5,
        help="Target abs_delta_e threshold enforced by the robustness gate lanes.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _normalize_fixed_inner_optimizer(args.fixed_inner_optimizer)
    if str(args.fixed_phase2_novelty_mode).strip().lower() != _ACTIVE_PHASE2_NOVELTY_MODE:
        raise ValueError("fixed_phase2_novelty_mode does not match PHASE3_POLICY_PHASE2_NOVELTY_MODE")
    base_policy = _base_policy_from_args(args)
    policy_search_profile = _normalize_policy_search_profile(args.policy_search_profile)
    meta_feature_profile = _normalize_meta_feature_profile(args.meta_feature_profile)
    required_ids = set(required_target_benchmark_ids_for_profile(args.required_target_profile))
    required_ids.update(str(x) for x in (args.required_target_benchmark_id or ()) if str(x).strip())
    required_target_abs_delta_e = (
        float(args.required_target_abs_delta_e)
        if args.required_target_abs_delta_e is not None and float(args.required_target_abs_delta_e) > 0.0
        else None
    )
    user_required_target_abs_delta_e = required_target_abs_delta_e is not None
    multi_objective_mode = _MULTI_OBJECTIVE_MODE_SAME_CUTOFF_PARETO if str(args.objective_profile) == "same_cutoff_pareto" else _MULTI_OBJECTIVE_MODE_OFF
    if str(args.objective_profile) == "paper_i_discovery_first_crossing":
        args.objective_profile = "cost_at_accuracy"
    if args.objective_profile == "cost_at_accuracy":
        weights = StaticObjectiveWeights(
            energy=1.0 if args.objective_energy_weight is None else float(args.objective_energy_weight),
            count_2q=4.0 if args.objective_2q_weight is None else float(args.objective_2q_weight),
            depth_2q=0.5 if args.objective_2q_depth_weight is None else float(args.objective_2q_depth_weight),
            circuit_depth=0.05 if args.objective_depth_weight is None else float(args.objective_depth_weight),
            parameters=0.10 if args.objective_parameter_weight is None else float(args.objective_parameter_weight),
            shot_cost=0.05 if args.objective_shot_weight is None else float(args.objective_shot_weight),
            target_abs_delta_e=float(args.target_abs_delta_e),
        )
    else:
        base_weights = StaticObjectiveWeights()
        weights = StaticObjectiveWeights(
            energy=base_weights.energy if args.objective_energy_weight is None else float(args.objective_energy_weight),
            count_2q=base_weights.count_2q if args.objective_2q_weight is None else float(args.objective_2q_weight),
            depth_2q=base_weights.depth_2q if args.objective_2q_depth_weight is None else float(args.objective_2q_depth_weight),
            circuit_depth=base_weights.circuit_depth if args.objective_depth_weight is None else float(args.objective_depth_weight),
            parameters=base_weights.parameters if args.objective_parameter_weight is None else float(args.objective_parameter_weight),
            shot_cost=base_weights.shot_cost if args.objective_shot_weight is None else float(args.objective_shot_weight),
            target_abs_delta_e=None,
        )
    if required_ids and required_target_abs_delta_e is None:
        required_target_abs_delta_e = (
            float(weights.target_abs_delta_e)
            if weights.target_abs_delta_e is not None and float(weights.target_abs_delta_e) > 0.0
            else float(args.target_abs_delta_e)
        )
    objective_family_weights = _parse_objective_weight_map(
        args.objective_family_weights,
        label="objective_family_weights",
    )
    objective_benchmark_weights = _parse_objective_weight_map(
        args.objective_benchmark_weights,
        label="objective_benchmark_weights",
    )
    discovery_objective_mode = _normalize_discovery_objective_mode(args.discovery_objective_mode)
    if discovery_objective_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING and str(args.required_target_profile or "none") != _PAPER_I_PHYS_V1_PROFILE:
        raise ValueError("discovery_first_crossing_requires_required_target_profile=paper_i_phys_v1")
    config = GlobalObjectiveConfig(
        robust_cvar_quantile=float(args.cvar_quantile),
        gamma_robust=float(args.gamma_robust),
        gamma_family_std=float(args.gamma_family_std),
        gamma_fail=float(args.gamma_fail),
        weights=weights,
        required_target_benchmark_ids=tuple(sorted(required_ids)),
        required_target_abs_delta_e=required_target_abs_delta_e,
        required_target_penalty=float(args.required_target_penalty),
        objective_weight_preset=str(args.objective_weight_preset),
        objective_family_weights=objective_family_weights,
        objective_benchmark_weights=objective_benchmark_weights,
        discovery_objective_mode=discovery_objective_mode,
        multi_objective_mode=multi_objective_mode,
        objective_provenance=_phase3_objective_provenance_from_args(args),
    )
    calibration_profile_key = _normalize_calibration_profile(args.calibration_profile)
    if calibration_profile_key != "off":
        specs = calibration_static_benchmark_specs(calibration_profile_key)
    else:
        suite_profile_key = str(args.suite_profile or "").strip().lower().replace("-", "_")
        if suite_profile_key and tuple(args.benchmark_id or ()):
            from pipelines.exact_bench.table_i_canonical_cases import table_i_executable_specs, table_i_suite_profile

            specs = table_i_executable_specs(table_i_suite_profile(suite_profile_key))
            families_filter = set(canonical_lane_families(args.canonical_lane) or tuple(args.families))
            if families_filter:
                specs = tuple(spec for spec in specs if str(spec.family) in families_filter)
        else:
            families = canonical_lane_families(args.canonical_lane) or tuple(args.families)
            specs = filter_static_benchmark_suite(
                families=families,
                sizes=args.sizes,
                molecular_problem_json=args.molecular_problem_json,
                boson_cutoff=None if args.boson_cutoffs else args.boson_cutoff,
                boson_cutoffs=args.boson_cutoffs,
                exact_reference_boson_cutoff=None if int(args.exact_reference_boson_cutoff) <= 0 else int(args.exact_reference_boson_cutoff),
                physics_grid_profile=args.physics_grid_profile,
            )
            specs = filter_canonical_lane_specs(
                specs,
                lane=args.canonical_lane,
                stage=args.canonical_lane_stage,
            )
    specs = filter_static_benchmark_suite_by_ids(specs, args.benchmark_id)
    if bool(args.force_same_cutoff_objective):
        specs = tuple(
            replace(spec, exact_reference_n_ph_max=None)
            if bool(getattr(getattr(spec, "features", None), "bosonic", False))
            else spec
            for spec in specs
        )
    required_target_profile_key = str(args.required_target_profile or "none").strip().lower().replace("-", "_")
    if required_target_profile_key == _PAPER_I_PHYS_V1_PROFILE:
        paper_i_targets = {spec.benchmark_id: resolve_paper_i_phys_v1_target(spec) for spec in specs}
        for spec in specs:
            expected_ref_raw = paper_i_targets[spec.benchmark_id].get("phonon_cutoff_eval_reference")
            if expected_ref_raw is None:
                continue
            expected_ref = int(expected_ref_raw)
            actual_ref = None if spec.exact_reference_n_ph_max is None else int(spec.exact_reference_n_ph_max)
            if actual_ref != expected_ref:
                raise ValueError(
                    "paper_i_phys_v1_requires_exact_reference_n_ph_max=explicit_reference_cutoff; "
                    f"{spec.benchmark_id} has {actual_ref}, expected {expected_ref}"
                )
        required_ids.update(paper_i_targets)
        tau_values = sorted({float(payload["tau_phys"]) for payload in paper_i_targets.values()})
        if len(tau_values) != 1:
            raise ValueError(
                "paper_i_phys_v1_requires_single_tau_in_required_target_scalar; "
                f"got {tau_values}"
            )
        canonical_tau_phys = float(tau_values[0])
        if user_required_target_abs_delta_e and not math.isclose(
            float(required_target_abs_delta_e),
            canonical_tau_phys,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "paper_i_phys_v1_forbids_stale_required_target_abs_delta_e; "
                f"got {required_target_abs_delta_e}, expected {canonical_tau_phys}"
            )
        required_target_abs_delta_e = canonical_tau_phys
        config = replace(
            config,
            required_target_benchmark_ids=tuple(sorted(required_ids)),
            required_target_abs_delta_e=required_target_abs_delta_e,
            objective_provenance={
                **dict(config.objective_provenance or {}),
                "required_target_profile": _PAPER_I_PHYS_V1_PROFILE,
                "paper_i_phys_v1_targets": _jsonable(paper_i_targets),
            },
        )
    if _should_apply_phase3_robustness_gate(
        mode=str(args.mode),
        specs=specs,
        gate_mode=str(args.robustness_gate),
        n_trials=int(args.n_trials),
    ):
        gate_payload = _run_phase3_robustness_gate_preflight(
            output_dir=args.output_dir,
            lanes=tuple(args.robustness_gate_lanes),
            target_abs_delta_e=float(args.robustness_gate_target_abs_delta_e),
            python_bin=sys.executable,
            adapt_timeout_s=(
                None
                if args.trial_timeout_sec is None or float(args.trial_timeout_sec) <= 0.0
                else float(args.trial_timeout_sec)
            ),
            compile_timeout_s=(
                None
                if args.compile_timeout_sec is None or float(args.compile_timeout_sec) <= 0.0
                else float(args.compile_timeout_sec)
            ),
        )
        if not bool(gate_payload.get("ok", False)):
            print(
                "Phase3 robustness gate failed; broad/static Optuna study was not started. "
                f"See {Path(args.output_dir) / 'robustness_gate.json'}",
                file=sys.stderr,
            )
            return 2
    historical_ledger = load_historical_ledger(args.historical_ledger)
    selected_logical_route = _normalize_selected_logical_route(args.selected_logical_route)
    selected_logical_source_path = args.selected_logical_source_json
    if selected_logical_source_path is None and selected_logical_route == "historical_selected":
        selected_logical_source_path = args.historical_ledger
    selected_logical_source_payload = (
        load_historical_ledger(selected_logical_source_path)
        if selected_logical_route == "historical_selected" and selected_logical_source_path is not None
        else None
    )
    specs = materialize_selected_logical_sidecars_for_specs(
        specs,
        selected_logical_source=selected_logical_source_payload,
        output_dir=args.output_dir,
        route=str(selected_logical_route),
        transfer_mode=str(args.selected_logical_transfer_mode),
    )
    specs = apply_historical_ledger_to_specs(specs, historical_ledger)
    explicit_enqueue_records = _load_enqueue_trial_param_records(args.enqueue_trial_params_json)
    selected_benchmark_ids = {str(spec.benchmark_id) for spec in specs}
    unknown_enqueue_benchmark_ids = sorted(
        {
            str(record.benchmark_id)
            for record in explicit_enqueue_records
            if record.benchmark_id not in {None, ""} and str(record.benchmark_id) not in selected_benchmark_ids
        }
    )
    if unknown_enqueue_benchmark_ids:
        raise ValueError(
            "Explicit enqueue records target benchmark_id(s) outside the selected suite: "
            + ", ".join(unknown_enqueue_benchmark_ids)
        )
    explicit_enqueue_records_by_benchmark = {
        spec.benchmark_id: tuple(
            record
            for record in explicit_enqueue_records
            if record.benchmark_id in {None, "", spec.benchmark_id}
        )
        for spec in specs
    }
    oracle_limit = None if int(args.oracle_enqueue_limit) < 0 else int(args.oracle_enqueue_limit)
    historical_records_by_benchmark, historical_skips_by_benchmark = (
        ({spec.benchmark_id: () for spec in specs}, {spec.benchmark_id: () for spec in specs})
        if args.no_enqueue_historical
        else historical_warm_start_records_for_specs(specs, historical_ledger)
    )
    oracle_records_by_benchmark, oracle_skips_by_benchmark = oracle_summary_warm_start_records_for_specs(
        tuple(args.oracle_summary_root or ()),
        specs,
        limit_per_benchmark=oracle_limit,
        required_static_route_id=args.oracle_required_static_route_id,
        required_suite_profile=args.oracle_required_suite_profile,
        require_phase0_aware=bool(args.oracle_require_phase0_aware),
        require_compatible_warm_starts=bool(args.oracle_require_compatible_warm_starts),
    )
    if bool(args.oracle_require_compatible_warm_starts):
        missing_compatible = [
            spec.benchmark_id
            for spec in specs
            if not tuple(oracle_records_by_benchmark.get(spec.benchmark_id, ()))
        ]
        if missing_compatible:
            raise ValueError(
                "Oracle-compatible warm starts are required but missing for benchmark_id(s): "
                + ", ".join(missing_compatible)
            )

    trial_param_overrides = _load_trial_param_overrides(args.trial_param_overrides_json)
    benchmark_trial_param_overrides = _load_benchmark_trial_param_overrides(args.benchmark_trial_param_overrides_json)

    def _merged_records_for(spec: HamiltonianBenchmarkSpec) -> tuple[WarmStartCandidate, ...]:
        return tuple(explicit_enqueue_records_by_benchmark.get(spec.benchmark_id, ())) + tuple(
            historical_records_by_benchmark.get(spec.benchmark_id, ())
        ) + tuple(
            oracle_records_by_benchmark.get(spec.benchmark_id, ())
        )

    def _merged_skips_for(spec: HamiltonianBenchmarkSpec) -> tuple[WarmStartSkip, ...]:
        return tuple(historical_skips_by_benchmark.get(spec.benchmark_id, ())) + tuple(
            oracle_skips_by_benchmark.get(spec.benchmark_id, ())
        )

    benchmark_target_abs_delta_e_for_child = None
    if discovery_objective_mode == _DISCOVERY_OBJECTIVE_MODE_FIRST_CROSSING or required_target_profile_key == _PAPER_I_PHYS_V1_PROFILE:
        if required_target_abs_delta_e is None or float(required_target_abs_delta_e) <= 0.0:
            raise ValueError("paper_i_first_crossing_requires_positive_child_benchmark_target_abs_delta_e")
        benchmark_target_abs_delta_e_for_child = float(required_target_abs_delta_e)
    elif args.objective_profile == "cost_at_accuracy" and float(args.target_abs_delta_e) > 0.0:
        benchmark_target_abs_delta_e_for_child = float(args.target_abs_delta_e)

    runner = partial(
        run_static_benchmark,
        adapt_timeout_s=None if args.trial_timeout_sec is None or float(args.trial_timeout_sec) <= 0.0 else float(args.trial_timeout_sec),
        compile_timeout_s=None if args.compile_timeout_sec is None or float(args.compile_timeout_sec) <= 0.0 else float(args.compile_timeout_sec),
        benchmark_target_abs_delta_e=benchmark_target_abs_delta_e_for_child,
        objective_weights=weights,
        trial_prune_depth=args.trial_prune_depth,
        trial_prune_abs_delta_e=args.trial_prune_abs_delta_e,
        trial_prune_metric=str(args.trial_prune_metric),
    )
    if args.mode == "global":
        run_optuna_study(
            mode="global",
            specs=specs,
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            seed=args.seed,
            n_jobs=args.n_jobs,
            storage=args.storage,
            study_name=f"{args.study_prefix}/global_static_agnostic",
            enqueue_default=not args.no_enqueue_default,
            enqueue_records=tuple(record for spec in specs for record in _merged_records_for(spec)),
            warm_start_skips=tuple(skip for spec in specs for skip in _merged_skips_for(spec)),
            runner=runner,
            config=config,
            benchmarks_per_trial_jobs=args.benchmarks_per_trial_jobs,
            policy_search_profile=policy_search_profile,
            meta_feature_profile=meta_feature_profile,
            progress_dir=args.progress_dir,
            base_policy=base_policy,
            suite_profile=args.suite_profile,
            telemetry_record_id=args.telemetry_record_id,
            optuna_best_telemetry=args.optuna_best_telemetry,
            optuna_best_telemetry_min_interval_sec=args.optuna_best_telemetry_min_interval_sec,
            trial_param_overrides=trial_param_overrides,
            benchmark_trial_param_overrides=benchmark_trial_param_overrides,
        )
    else:
        enqueue_records_by_benchmark = {spec.benchmark_id: _merged_records_for(spec) for spec in specs}
        warm_start_skips_by_benchmark = {spec.benchmark_id: _merged_skips_for(spec) for spec in specs}
        run_oracle_grid(
            specs=specs,
            output_dir=args.output_dir,
            progress_dir=args.progress_dir,
            n_trials=args.n_trials,
            seed=args.seed,
            n_jobs=args.n_jobs,
            storage=args.storage,
            study_prefix=f"{args.study_prefix}/static_oracle",
            enqueue_default=not args.no_enqueue_default,
            enqueue_records_by_benchmark=enqueue_records_by_benchmark,
            warm_start_skips_by_benchmark=warm_start_skips_by_benchmark,
            runner=runner,
            config=config,
            policy_search_profile=policy_search_profile,
            meta_feature_profile=meta_feature_profile,
            base_policy=base_policy,
            suite_profile=args.suite_profile,
            telemetry_record_id=args.telemetry_record_id,
            optuna_best_telemetry=args.optuna_best_telemetry,
            optuna_best_telemetry_min_interval_sec=args.optuna_best_telemetry_min_interval_sec,
            trial_param_overrides=trial_param_overrides,
        )
    return 0


__all__ = [
    "AlgorithmPolicy",
    "BenchmarkResult",
    "GlobalObjectiveConfig",
    "HamiltonianBenchmarkSpec",
    "InnerOptimizerPolicy",
    "PoolPolicy",
    "ProblemFeatureVector",
    "SizeScaledBudget",
    "StaticObjectiveWeights",
    "StaticScaffoldPolicy",
    "PHASE0_OPTUNA_DEFAULTS",
    "WarmStartCandidate",
    "WarmStartSkip",
    "aggregate_global_score",
    "aggregate_global_score_components",
    "compute_paper_i_first_crossing_from_payload",
    "target_hit_classification_for_result",
    "discovery_first_crossing_score",
    "discovery_first_crossing_score_components",
    "objective_weighting_payload",
    "objective_provenance_payload",
    "resolved_objective_weight_rows",
    "apply_historical_ledger_to_specs",
    "apply_policy_to_pipeline_args",
    "build_compile_command",
    "build_policy_roundtrip_audit",
    "build_static_command",
    "classify_static_result_quality",
    "calibration_static_benchmark_specs",
    "default_static_benchmark_suite",
    "normalized_static_score",
    "objective_global_agnostic",
    "objective_oracle",
    "oracle_gap",
    "oracle_summary_trial_params",
    "policy_to_cli_args",
    "run_static_benchmark",
    "default_trial_params",
    "filter_static_benchmark_suite",
    "filter_static_benchmark_suite_by_ids",
    "canonical_lane_families",
    "filter_canonical_lane_specs",
    "required_target_benchmark_ids_for_profile",
    "resolve_paper_i_phys_v1_target",
    "build_effective_trial_manifest_intent",
    "finalize_effective_trial_manifest",
    "validate_effective_trial_replay",
    "required_target_violations",
    "historical_trial_params_for_specs",
    "historical_warm_start_records_for_specs",
    "oracle_summary_warm_start_records_for_specs",
    "load_historical_ledger",
    "materialize_selected_logical_sidecars_for_specs",
    "run_optuna_study",
    "run_oracle_grid",
    "_run_phase3_robustness_gate_preflight",
    "_should_apply_phase3_robustness_gate",
    "build_parser",
    "main",
    "sample_policy_from_trial",
    "trial_params_from_cli_command",
]


if __name__ == "__main__":
    raise SystemExit(main())
