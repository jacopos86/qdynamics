#!/usr/bin/env python3
"""Benchmark-local generic static ADAPT selector variants.

This module supplies Table-I competitor rows that are intentionally isolated to
``pipelines.exact_bench``.  QEB rows use a benchmark-local qubit-excitation
singles/doubles pool; TETRIS and visible ``static_geo_adapt_vqe`` use the
problem-local ``full_meta`` pool.  ``static_geo_qeb_adapt_vqe`` is retained
as a QEB Geo-ADAPT reference row.  ``static_pos_geo_adapt_vqe`` is PosGeo
diagnostic evidence only, not the visible main-table Geo-ADAPT row: it uses the
problem-local full_meta pool, projected Fubini--Study natural-gradient
selection/stopping, position-optimized insertion, with-replacement selection
except immediate repeats, and a local QNGD-style inner optimizer with seeded
energy-only SPSA fallback if QNGD stalls.  None of these rows call the
Phase3/SNAKE static ADAPT controller.  The visible Append and Geo comparator
routes use Powell for the full refit and a fixed iteration horizon.  Exact
target energies are reporting-only unless the optional benchmark target-stop
mode is explicitly enabled, in which case the emitted guardrails disclose the
post-iteration reference-assisted stop.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
import time
from argparse import Namespace
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.benchmark_decision_noise import (
    BenchmarkDecisionNoiseConfig,
    BenchmarkDecisionNoiseRecorder,
    coerce_config as coerce_benchmark_decision_noise_config,
    copy_decision_noise_metadata,
)
from pipelines.exact_bench.benchmark_metrics_proxy import write_proxy_sidecars
from pipelines.exact_bench.deterministic_shot_proxy import (
    DETERMINISTIC_SHOT_PROXY_FORMULA,
    build_deterministic_shot_proxy_fields,
)
from pipelines.exact_bench.comparator_provenance import comparator_source_fields
from pipelines.exact_bench.molecular_vibronic_h2_fixture_override import (
    with_molecular_vibronic_h2_fixture_override,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (
    PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    normalize_paper_i_main_tables_spsa_profile,
)
from pipelines.exact_bench.generic_static_hea_qiskit_vqe import sector_probability
from pipelines.static_adapt.builders.pool_resolution import (
    resolve_pool_plan,
    resolve_requested_pool_filters,
)
from pipelines.static_adapt.builders.problem_registry import (
    ProblemRequest,
    ResolvedProblemContext,
    resolve_problem_context,
)
from pipelines.static_adapt.builders.primitive_pools import build_qeb_pool_specs
from pipelines.exact_bench.table_i_canonical_cases import (
    table_i_canonical_case_ids,
    table_i_canonical_spec_by_case_id,
)
from pipelines.exact_bench.table_i_qiskit_resource_compile import (
    TableICompileUnavailable,
    compile_table_i_ansatz_terms,
)
from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
from pipelines.scaffold.hh_continuation_generators import (
    build_runtime_split_child_sets,
    build_runtime_split_children,
)
from pipelines.static_adapt.builders.shared_pauli_pool_contract import (
    SHARED_PAULI_POOL_MODE_OFF,
    SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF,
    SharedPauliPoolParent,
    build_shared_pauli_child_pool,
    normalize_shared_pauli_pool_mode,
    normalize_shared_pauli_pool_symmetry_policy,
)
from pipelines.static_adapt.optimization.phase3_policy_optuna import (
    HamiltonianBenchmarkSpec,
)
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    GeneratorParameterBlock,
    build_parameter_layout,
    deserialize_layout,
    expand_legacy_logical_theta,
    project_runtime_theta_block_mean,
    serialize_layout,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial,
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.coordinate_descent_optimizer import (
    rotosolve_coordinate_descent,
    rotosolve_stencil_from_executor,
)
from src.quantum.adapt_spsa_refit import (
    GENERIC_STATIC_ADAPT_SPSA_REFIT_ENGINE_ENV,
    LEGACY_SPSA_POLISH_OPTIMIZER_LABEL,
    NATIVE_SPSA_OPTIMIZER_LABEL,
    SPSAEnergyDescentSchedule,
    default_spsa_energy_descent_schedule,
    resolve_adapt_spsa_refit_engine_label,
    spsa_energy_descent_minimize,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qnspsa_optimizer import qnspsa_minimize
from src.quantum.qubitization_module import PauliTerm
from src.quantum.spsa_optimizer import spsa_minimize
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

SCHEMA_VERSION = "generic_static_adapt_variants_v4"
_RUNNER_MODULE = "pipelines.exact_bench.generic_static_adapt_variants"
_NATIVE_SPSA_OPTIMIZER_LABEL = NATIVE_SPSA_OPTIMIZER_LABEL
_LEGACY_SPSA_POLISH_OPTIMIZER_LABEL = LEGACY_SPSA_POLISH_OPTIMIZER_LABEL
_ADAPT_SPSA_REFIT_ENGINE_ENV = GENERIC_STATIC_ADAPT_SPSA_REFIT_ENGINE_ENV
GENERIC_STATIC_ALLOW_UNSUPPORTED_ROTOSOLVE_FALLBACK_ENV = (
    "GENERIC_STATIC_ALLOW_UNSUPPORTED_ROTOSOLVE_FALLBACK"
)
TABLE_I_EVENT_LEDGER_SCHEMA = "table_i_measurement_event_ledger_v1"
STATIC_QUBIT_QEB_ADAPT_VQE = "static_qubit_qeb_adapt_vqe"
STATIC_TETRIS_QUBIT_ADAPT_VQE = "static_tetris_qubit_adapt_vqe"
STATIC_GEO_QUBIT_ADAPT_VQE = "static_geo_qubit_adapt_vqe"
STATIC_GEO_QEB_ADAPT_VQE = "static_geo_qeb_adapt_vqe"
STATIC_GEO_ADAPT_VQE = "static_geo_adapt_vqe"
STATIC_POS_GEO_ADAPT_VQE = "static_pos_geo_adapt_vqe"
STATIC_FULL_META_APPEND_ADAPT_VQE = "static_full_meta_append_adapt_vqe"
GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS: tuple[str, ...] = (
    STATIC_FULL_META_APPEND_ADAPT_VQE,
    STATIC_QUBIT_QEB_ADAPT_VQE,
    STATIC_TETRIS_QUBIT_ADAPT_VQE,
    STATIC_GEO_QUBIT_ADAPT_VQE,
    STATIC_GEO_QEB_ADAPT_VQE,
    STATIC_GEO_ADAPT_VQE,
    STATIC_POS_GEO_ADAPT_VQE,
)

_QUBIT_CAP = 10
_POOL_TERM_CAP = 256
_RESOURCE_QUBIT_CAP_ENV = "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP"
_RESOURCE_POOL_TERM_CAP_ENV = "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP"
_EXACT_FIDELITY_MAX_QUBITS_ENV = "GENERIC_STATIC_TABLE_EXACT_FIDELITY_MAX_QUBITS"
_DEFAULT_MAX_ADAPT_ITERATIONS = 1000
_DEFAULT_OPTIMIZER_MAXITER = 5000
_DEFAULT_GRADIENT_THRESHOLD = 1e-5
_TETRIS_FIXED_HORIZON_NUMERICAL_ZERO_GRADIENT_FLOOR = 1e-14
_DEFAULT_METRIC_FLOOR = 1e-8
_DEFAULT_MAX_TETRIS_BATCH_SIZE = 4
_DEFAULT_SHOTS_PER_PAULI_TERM_PROXY = 1024
_FIXED_HORIZON_NO_TARGET_STOP_POLICY = "fixed_horizon_no_target_v1"
_POWELL_MAXITER_CAP_POLICY_STRICT = "strict_failure_v1"
_POWELL_MAXITER_CAP_POLICY_ACCEPT_FINITE_NONINCREASING = (
    "accept_finite_nonincreasing_v1"
)
_POWELL_MAXITER_CAP_POLICY_CHOICES = frozenset(
    {
        _POWELL_MAXITER_CAP_POLICY_STRICT,
        _POWELL_MAXITER_CAP_POLICY_ACCEPT_FINITE_NONINCREASING,
    }
)
_POWELL_MAXITER_MESSAGE = "maximum number of iterations has been exceeded"
_POWELL_MAXITER_STATUS = 2
_POWELL_CAP_ENERGY_REL_TOL = 1.0e-10
_POWELL_CAP_ENERGY_ABS_TOL = 1.0e-10
_ENERGY_STOP_TARGET_ENV = "GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET"
_FIRST_HIT_THRESHOLDS_ENV = "GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS"
_SAME_CUTOFF_EXACT_GS_ENERGY_ENV = "GENERIC_STATIC_TABLE_SAME_CUTOFF_EXACT_GS_ENERGY"
_EXACT_REFERENCE_ENERGY_ENV = "GENERIC_STATIC_TABLE_EXACT_REFERENCE_ENERGY"
_EXACT_REFERENCE_N_PH_MAX_ENV = "GENERIC_STATIC_TABLE_EXACT_REFERENCE_N_PH_MAX"
_PRIMARY_ENERGY_METRIC_ENV = "GENERIC_STATIC_TABLE_PRIMARY_ENERGY_METRIC"
_SAME_CUTOFF_ERROR_ROLE_ENV = "GENERIC_STATIC_TABLE_SAME_CUTOFF_ERROR_ROLE"
_HH_POOL_CACHE_ENV = "STATIC_ADAPT_HH_POOL_CACHE"
_HH_POOL_CACHE_SCOPE_ENV = "STATIC_ADAPT_HH_POOL_CACHE_SCOPE"
_HH_POOL_CACHE_DIR_ENV = "STATIC_ADAPT_HH_POOL_CACHE_DIR"
_SHOT_PROXY_FORMULA = DETERMINISTIC_SHOT_PROXY_FORMULA
_GEO_SCORE_FORMULA = (
    "score = abs(delta_theta_i), where delta_theta = pinv(S, rcond=max(1e-10, metric_floor)) "
    "@ [-2 Re(<t_i|Hpsi - Epsi>)]"
)
_GEO_STOP_RULE = "fubini_study_natural_gradient_norm"
_GEO_REFERENCE_ALGORITHM = "Sohail--Koike-Akino Geo-ADAPT-VQE"
_GEO_QNGD_MAX_ABS_STEP = 0.25
_GEO_QNGD_MAX_BACKTRACKS = 10
_GEO_SPSA_POLISH_MAXITER = 200
_GEO_SPSA_A0 = 0.05


def _resource_cap_from_env(name: str, default: int | None) -> int | None:
    raw = os.environ.get(str(name), "")
    if raw is None or str(raw).strip() == "":
        return default
    key = str(raw).strip().lower()
    if key in {"0", "none", "off", "false", "unbounded", "unlimited"}:
        return None
    value = int(key)
    if value < 1:
        return None
    return int(value)


def _exact_fidelity_max_qubits_from_env(default: int = 12) -> int:
    raw = os.environ.get(_EXACT_FIDELITY_MAX_QUBITS_ENV, "")
    if raw is None or str(raw).strip() == "":
        return int(default)
    value = int(str(raw).strip())
    if value < 0:
        raise ValueError(f"{_EXACT_FIDELITY_MAX_QUBITS_ENV} must be >= 0; got {value}.")
    return int(value)
_GEO_SPSA_C0 = 0.05
_GEO_SPSA_ALPHA = 0.602
_GEO_SPSA_GAMMA = 0.101
_GEO_SPSA_A_SHIFT = 5.0
_GEO_SPSA_ACCEPT_TOL = 1e-12
_GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF = "off"
_GENERIC_ADAPT_RUNTIME_SPLIT_MODE_PAULI_CHILDREN = "shortlist_pauli_children_v1"
_GENERIC_ADAPT_RUNTIME_SPLIT_MODES = frozenset(
    {
        _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF,
        _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_PAULI_CHILDREN,
    }
)
_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_OFF = "off"
_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_HARD_GUARD = "hard_guard"
_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICIES = frozenset(
    {
        _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_OFF,
        _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_HARD_GUARD,
    }
)
_TETRIS_COMPATIBILITY_RULE = "greedy batches contain candidates with pairwise-disjoint qubit support"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HH_FULL_META_MINUS_HVA_CLASS_FILTER_JSON = (
    _REPO_ROOT / "agent_guidance" / "static-adapt" / "hh_full_meta_minus_hva_class_filter.json"
)
_HH_ACTIVE_FULL_META_MINUS_HVA_ALGORITHM_IDS = frozenset(
    {
        STATIC_FULL_META_APPEND_ADAPT_VQE,
        STATIC_GEO_ADAPT_VQE,
    }
)
_HH_ADAPTIVE_POOL_PROFILE_LEGACY_AUTO = "legacy_auto"
_HH_ADAPTIVE_POOL_PROFILE_FULL_META_MINUS_HVA = "full_meta_minus_hva"
_HH_ADAPTIVE_POOL_PROFILE_FULL_META_UNFILTERED = "full_meta_unfiltered"
_HH_ADAPTIVE_POOL_PROFILES = frozenset(
    {
        _HH_ADAPTIVE_POOL_PROFILE_LEGACY_AUTO,
        _HH_ADAPTIVE_POOL_PROFILE_FULL_META_MINUS_HVA,
        _HH_ADAPTIVE_POOL_PROFILE_FULL_META_UNFILTERED,
    }
)
_GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_FAMILIES = frozenset({"hh", "hubbard"})


def _positive_int(value: int | str | None, *, field: str) -> int:
    try:
        out = int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{field} must be a positive integer; got {value!r}.") from exc
    if out < 1:
        raise ValueError(f"{field} must be a positive integer; got {value!r}.")
    return int(out)


def _blank_to_none(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _positive_float(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{field} must be a positive finite float; got {value!r}.") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{field} must be a positive finite float; got {value!r}.")
    return float(out)


_SpsaPolishSchedule = SPSAEnergyDescentSchedule


def _default_spsa_polish_schedule() -> _SpsaPolishSchedule:
    defaults = default_spsa_energy_descent_schedule()
    return _SpsaPolishSchedule(
        a=float(defaults.a),
        c=float(defaults.c),
        alpha=float(defaults.alpha),
        gamma=float(defaults.gamma),
        big_a=float(defaults.big_a),
    )


def _coerce_spsa_polish_schedule(
    *,
    adapt_spsa_a: float | str | None = None,
    adapt_spsa_c: float | str | None = None,
    adapt_spsa_alpha: float | str | None = None,
    adapt_spsa_gamma: float | str | None = None,
    adapt_spsa_big_a: float | str | None = None,
) -> _SpsaPolishSchedule:
    defaults = _default_spsa_polish_schedule()
    raw = {
        "adapt_spsa_a": _blank_to_none(adapt_spsa_a),
        "adapt_spsa_c": _blank_to_none(adapt_spsa_c),
        "adapt_spsa_alpha": _blank_to_none(adapt_spsa_alpha),
        "adapt_spsa_gamma": _blank_to_none(adapt_spsa_gamma),
        "adapt_spsa_big_a": _blank_to_none(adapt_spsa_big_a),
    }
    return _SpsaPolishSchedule(
        a=defaults.a if raw["adapt_spsa_a"] is None else _positive_float(raw["adapt_spsa_a"], field="adapt_spsa_a"),
        c=defaults.c if raw["adapt_spsa_c"] is None else _positive_float(raw["adapt_spsa_c"], field="adapt_spsa_c"),
        alpha=defaults.alpha
        if raw["adapt_spsa_alpha"] is None
        else _positive_float(raw["adapt_spsa_alpha"], field="adapt_spsa_alpha"),
        gamma=defaults.gamma
        if raw["adapt_spsa_gamma"] is None
        else _positive_float(raw["adapt_spsa_gamma"], field="adapt_spsa_gamma"),
        big_a=defaults.big_a
        if raw["adapt_spsa_big_a"] is None
        else _positive_float(raw["adapt_spsa_big_a"], field="adapt_spsa_big_a"),
    )


GENERIC_STATIC_ADAPT_VARIANT_FAMILIES: tuple[str, ...] = (
    "hh",
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_vibronic_h2",
)

_FULL_META_PAOP_R = 2
_FULL_META_PAOP_SPLIT_PAULIS = False
_FULL_META_PAOP_PRUNE_EPS = 1e-12
_FULL_META_PAOP_NORMALIZATION = "none"

VariantKind = Literal[
    "full_meta_append_only",
    "qeb",
    "tetris",
    "geo_full_meta",
    "geo_qeb",
    "geo_powell_full_meta",
    "geo_spsa_full_meta",
    "pos_geo_full_meta",
]
PoolKind = Literal["qeb", "full_meta"]
OptimizerKind = Literal["bfgs", "geo_qngd", "powell", "rotosolve", "spsa"]
StopRule = Literal["raw_gradient", "geo_natural_gradient_norm"]
RepeatPolicy = Literal[
    "exclude_selected_labels",
    "with_replacement",
    "with_replacement_except_immediate_repeat",
]
PositionPolicy = Literal["append", "best_insert_refit"]
_POS_GEO_POSITION_POLICY_ENV = "GENERIC_STATIC_TABLE_PHASE3_POS_GEO_POSITION_POLICY"


@dataclass(frozen=True)
class _VariantConfig:
    algorithm_id: str
    variant: VariantKind
    display_name: str
    method_kind: str
    ansatz_name: str
    selector_rule: str
    pool_kind: PoolKind
    optimizer_kind: OptimizerKind
    stop_rule: StopRule
    repeat_policy: RepeatPolicy
    position_policy: PositionPolicy = "append"
    faithful_geo_adapt_vqe: bool = False


@dataclass(frozen=True)
class _PoolCandidate:
    label: str
    polynomial: PauliPolynomial
    support: tuple[int, ...]
    pauli_labels_exyz: tuple[str, ...]
    construction: str
    parent_label: str | None = None
    runtime_split_mode: str = _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF
    runtime_split_representation: str = "parent"
    runtime_split_child_indices: tuple[int, ...] = ()
    runtime_split_child_labels: tuple[str, ...] = ()
    runtime_split_symmetry_gate: dict[str, Any] | None = None
    execution_mode: str = "termwise_product"
    generator_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class _CompiledCandidate:
    candidate: _PoolCandidate
    compiled: CompiledPolynomialAction


@dataclass(frozen=True)
class _FullMetaCandidatePoolResult:
    candidates: tuple[_PoolCandidate, ...]
    selected_logical_filter_meta: dict[str, Any] | None
    full_meta_class_filter_meta: dict[str, Any] | None = None
    full_meta_label_filter_meta: dict[str, Any] | None = None
    pool_legal_subspace_filter_meta: dict[str, Any] | None = None
    pool_key: str | None = None
    pool_cache_events: tuple[dict[str, Any], ...] = ()


_VARIANTS: dict[str, _VariantConfig] = {
    STATIC_FULL_META_APPEND_ADAPT_VQE: _VariantConfig(
        algorithm_id=STATIC_FULL_META_APPEND_ADAPT_VQE,
        variant="full_meta_append_only",
        display_name="Append-only ADAPT-VQE (local full_meta)",
        method_kind="full_meta_append_only_adapt",
        ansatz_name="benchmark_local_full_meta_append_only_adapt",
        selector_rule=(
            "single largest absolute ADAPT commutator gradient per iteration "
            "over the problem-local full_meta pool"
        ),
        pool_kind="full_meta",
        optimizer_kind="powell",
        stop_rule="raw_gradient",
        repeat_policy="with_replacement",
    ),
    STATIC_QUBIT_QEB_ADAPT_VQE: _VariantConfig(
        algorithm_id=STATIC_QUBIT_QEB_ADAPT_VQE,
        variant="qeb",
        display_name="Qubit/QEB-ADAPT-VQE",
        method_kind="qubit_excitation_adapt",
        ansatz_name="benchmark_local_pairwise_qubit_excitation_adapt",
        selector_rule="single largest absolute ADAPT commutator gradient per iteration",
        pool_kind="qeb",
        optimizer_kind="bfgs",
        stop_rule="raw_gradient",
        repeat_policy="exclude_selected_labels",
    ),
    STATIC_TETRIS_QUBIT_ADAPT_VQE: _VariantConfig(
        algorithm_id=STATIC_TETRIS_QUBIT_ADAPT_VQE,
        variant="tetris",
        display_name="TETRIS-ADAPT-VQE",
        method_kind="tetris_full_meta_adapt",
        ansatz_name="benchmark_local_tetris_batched_full_meta_adapt",
        selector_rule=(
            "rank full_meta candidates by absolute ADAPT commutator gradient, then greedily batch candidates "
            "with disjoint qubit support"
        ),
        pool_kind="full_meta",
        optimizer_kind="bfgs",
        stop_rule="raw_gradient",
        repeat_policy="exclude_selected_labels",
    ),
    STATIC_GEO_QUBIT_ADAPT_VQE: _VariantConfig(
        algorithm_id=STATIC_GEO_QUBIT_ADAPT_VQE,
        variant="geo_full_meta",
        display_name="legacy geometry diagnostic (removed from Table I)",
        method_kind="metric_aware_full_meta_adapt",
        ansatz_name="benchmark_local_full_meta_projected_metric_adapt",
        selector_rule=(
            "solve the full projected tangent metric over the remaining full_meta pool, "
            "then select the largest absolute natural-gradient step"
        ),
        pool_kind="full_meta",
        optimizer_kind="bfgs",
        stop_rule="raw_gradient",
        repeat_policy="exclude_selected_labels",
    ),
    STATIC_GEO_QEB_ADAPT_VQE: _VariantConfig(
        algorithm_id=STATIC_GEO_QEB_ADAPT_VQE,
        variant="geo_qeb",
        display_name="Geo-ADAPT-VQE (QEB reference)",
        method_kind="geo_adapt_qeb_excitation",
        ansatz_name="benchmark_local_qeb_projected_metric_geo_adapt",
        selector_rule=(
            "solve the projected Fubini-Study tangent metric over the QEB singles/doubles pool, "
            "select the largest absolute natural-gradient step, and stop on natural-gradient norm"
        ),
        pool_kind="qeb",
        optimizer_kind="geo_qngd",
        stop_rule="geo_natural_gradient_norm",
        repeat_policy="with_replacement_except_immediate_repeat",
        faithful_geo_adapt_vqe=True,
    ),
    STATIC_GEO_ADAPT_VQE: _VariantConfig(
        algorithm_id=STATIC_GEO_ADAPT_VQE,
        variant="geo_powell_full_meta",
        display_name="Geo-ADAPT-VQE",
        method_kind="geo_adapt_full_meta_powell",
        ansatz_name="benchmark_local_full_meta_geo_adapt_powell",
        selector_rule=(
            "solve the projected Fubini-Study tangent metric over the full_meta pool, "
            "select the largest absolute natural-gradient step, append the selected "
            "generator, refit the complete ansatz with Powell, and stop on the configured horizon"
        ),
        pool_kind="full_meta",
        optimizer_kind="powell",
        stop_rule="geo_natural_gradient_norm",
        repeat_policy="with_replacement_except_immediate_repeat",
        position_policy="append",
        faithful_geo_adapt_vqe=True,
    ),
    STATIC_POS_GEO_ADAPT_VQE: _VariantConfig(
        algorithm_id=STATIC_POS_GEO_ADAPT_VQE,
        variant="pos_geo_full_meta",
        display_name="Pos-Geo-ADAPT-VQE",
        method_kind="pos_geo_adapt_full_meta",
        ansatz_name="benchmark_local_full_meta_pos_geo_adapt",
        selector_rule=(
            "solve the projected Fubini-Study tangent metric over the full_meta pool, "
            "select the largest absolute natural-gradient step, test all insertion positions by local refit, "
            "and stop on natural-gradient norm"
        ),
        pool_kind="full_meta",
        optimizer_kind="geo_qngd",
        stop_rule="geo_natural_gradient_norm",
        repeat_policy="with_replacement_except_immediate_repeat",
        position_policy="best_insert_refit",
        faithful_geo_adapt_vqe=True,
    ),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            return str(value)
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_config(algorithm_id: str) -> _VariantConfig:
    key = str(algorithm_id).strip()
    if key not in _VARIANTS:
        known = ", ".join(sorted(_VARIANTS))
        raise ValueError(f"Unknown generic static ADAPT variant {algorithm_id!r}. Known: {known}")
    return _VARIANTS[key]


def _apply_environment_position_policy_override(config: _VariantConfig, *, family: str) -> _VariantConfig:
    raw = str(os.environ.get(_POS_GEO_POSITION_POLICY_ENV, "") or "").strip().lower()
    if raw in {"", "default"}:
        return config
    if raw not in {"append", "best_insert_refit"}:
        raise ValueError(
            f"{_POS_GEO_POSITION_POLICY_ENV} must be one of "
            "{'append','best_insert_refit','default'}."
        )
    if config.algorithm_id != STATIC_POS_GEO_ADAPT_VQE:
        return config
    if str(family).strip() != "hh":
        return config
    return replace(config, position_policy=raw)  # type: ignore[arg-type]


def _effective_optimizer_settings_for_config(
    config: _VariantConfig,
    *,
    optimizer_profile: str | None = None,
    optimizer_profile_source: str | None = None,
    adapt_optimizer_kind: str | None = None,
    adapt_spsa_maxiter: int | None = None,
    adapt_spsa_seed: int | None = None,
    adapt_spsa_a: float | str | None = None,
    adapt_spsa_c: float | str | None = None,
    adapt_spsa_alpha: float | str | None = None,
    adapt_spsa_gamma: float | str | None = None,
    adapt_spsa_big_a: float | str | None = None,
    optimizer_maxiter: int,
    seed: int,
    optimizer_overlay_source: str | None = None,
) -> tuple[_VariantConfig, dict[str, Any]]:
    raw_schedule = {
        "adapt_spsa_a": _blank_to_none(adapt_spsa_a),
        "adapt_spsa_c": _blank_to_none(adapt_spsa_c),
        "adapt_spsa_alpha": _blank_to_none(adapt_spsa_alpha),
        "adapt_spsa_gamma": _blank_to_none(adapt_spsa_gamma),
        "adapt_spsa_big_a": _blank_to_none(adapt_spsa_big_a),
    }
    schedule_requested = any(value is not None for value in raw_schedule.values())
    profile = normalize_paper_i_main_tables_spsa_profile(optimizer_profile)
    if profile == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID:
        if config.algorithm_id not in PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS:
            raise ValueError(
                f"optimizer_profile={profile} is only valid for visible Paper-I comparator methods; "
                f"{config.algorithm_id!r} is diagnostic or handled outside generic ADAPT variants."
            )
        requested = str(adapt_optimizer_kind or "spsa").strip().lower()
        optimizer_source = "adapt_optimizer_kind" if adapt_optimizer_kind not in {None, ""} else "optimizer_profile_default"
        if requested != "spsa":
            raise ValueError(f"optimizer_profile={profile} requires adapt_optimizer_kind=spsa; got {adapt_optimizer_kind!r}.")
    elif adapt_optimizer_kind not in {None, ""}:
        requested = str(adapt_optimizer_kind).strip().lower()
        optimizer_source = "adapt_optimizer_kind"
    else:
        requested = str(config.optimizer_kind)
        optimizer_source = "variant_default"
    if requested not in {"bfgs", "geo_qngd", "powell", "rotosolve", "spsa"}:
        raise ValueError(
            "adapt_optimizer_kind must be one of "
            f"{{bfgs, geo_qngd, powell, rotosolve, spsa}}; got {requested!r}."
        )
    if schedule_requested and requested != "spsa":
        names = ", ".join(sorted(field for field, value in raw_schedule.items() if value is not None))
        raise ValueError(f"generic ADAPT SPSA schedule fields {{{names}}} require adapt_optimizer_kind=spsa; got {requested!r}.")
    effective_config = config if requested == config.optimizer_kind else replace(config, optimizer_kind=requested)  # type: ignore[arg-type]
    effective_optimizer_maxiter = _positive_int(
        adapt_spsa_maxiter if requested == "spsa" and adapt_spsa_maxiter is not None else optimizer_maxiter,
        field="adapt_spsa_maxiter" if requested == "spsa" and adapt_spsa_maxiter is not None else "optimizer_maxiter",
    )
    spsa_seed_base = _positive_int(
        adapt_spsa_seed if requested == "spsa" and adapt_spsa_seed is not None else seed,
        field="adapt_spsa_seed" if requested == "spsa" and adapt_spsa_seed is not None else "seed",
    )
    spsa_schedule = (
        _coerce_spsa_polish_schedule(
            adapt_spsa_a=raw_schedule["adapt_spsa_a"],
            adapt_spsa_c=raw_schedule["adapt_spsa_c"],
            adapt_spsa_alpha=raw_schedule["adapt_spsa_alpha"],
            adapt_spsa_gamma=raw_schedule["adapt_spsa_gamma"],
            adapt_spsa_big_a=raw_schedule["adapt_spsa_big_a"],
        )
        if requested == "spsa"
        else None
    )
    return effective_config, {
        "optimizer_kind": requested,
        "optimizer_source": optimizer_source,
        "optimizer_profile": profile,
        "optimizer_profile_source": optimizer_profile_source if profile is not None else None,
        "optimizer_overlay_source": optimizer_overlay_source,
        "optimizer_maxiter": int(effective_optimizer_maxiter),
        "spsa_seed_base": int(spsa_seed_base),
        "adapt_spsa_maxiter": int(effective_optimizer_maxiter) if requested == "spsa" else None,
        "adapt_spsa_seed": int(spsa_seed_base) if requested == "spsa" else None,
        "adapt_spsa_a": None if spsa_schedule is None else float(spsa_schedule.a),
        "adapt_spsa_c": None if spsa_schedule is None else float(spsa_schedule.c),
        "adapt_spsa_alpha": None if spsa_schedule is None else float(spsa_schedule.alpha),
        "adapt_spsa_gamma": None if spsa_schedule is None else float(spsa_schedule.gamma),
        "adapt_spsa_big_a": None if spsa_schedule is None else float(spsa_schedule.big_a),
        "spsa_schedule": spsa_schedule,
    }


def _is_geo_config(config: _VariantConfig) -> bool:
    return "geo" in str(config.variant)


def _uses_raw_gradient_stop(config: _VariantConfig) -> bool:
    return config.stop_rule == "raw_gradient"


def _repeat_enabled_policy_for_config(config: _VariantConfig) -> RepeatPolicy:
    if _is_geo_config(config):
        return "with_replacement_except_immediate_repeat"
    return "with_replacement"


def _geo_outer_selector_matches_source_algorithm(config: _VariantConfig) -> bool:
    """Whether the outer selector/repeat/stop sequence matches Geo-ADAPT.

    This deliberately does not claim end-to-end paper conformance.  The
    benchmark's Hamiltonian-generic full-meta pool and shared optimizer
    overlays are useful comparator controls, but they differ from the source
    paper's excitation-pool/fixed-step-QNGD experiment.
    """

    return bool(
        _is_geo_config(config)
        and config.position_policy == "append"
        and config.stop_rule == "geo_natural_gradient_norm"
        and config.repeat_policy == "with_replacement_except_immediate_repeat"
    )


def _geo_strict_source_conformance(config: _VariantConfig) -> bool:
    # The current QNGD engine adds clipping, backtracking, early termination,
    # and an SPSA fallback.  Even the QEB route is therefore a robustness
    # extension rather than the source paper's fixed-step inner loop.
    return False


def _geo_source_deviations(config: _VariantConfig) -> list[str]:
    if not _is_geo_config(config):
        return []
    deviations = ["moore_penrose_pseudoinverse_selector"]
    if config.pool_kind != "qeb":
        deviations.append("problem_local_full_meta_pool_instead_of_excitation_pool")
    if config.optimizer_kind != "geo_qngd":
        deviations.append(f"{config.optimizer_kind}_inner_optimizer_instead_of_fixed_step_qngd")
    else:
        deviations.append("qngd_robustness_extensions_and_spsa_fallback")
    if config.position_policy != "append":
        deviations.append("position_optimized_insertion_extension")
    return deviations


def _requires_scipy_minimize(config: _VariantConfig) -> bool:
    return config.optimizer_kind in {"bfgs", "powell"}


def _optimizer_failure_reason(config: _VariantConfig) -> str:
    kind = "qngd" if config.optimizer_kind == "geo_qngd" else str(config.optimizer_kind)
    return f"{kind}_optimizer_failed"


def _pool_source_for_config(config: _VariantConfig) -> str:
    if config.pool_kind == "full_meta":
        return "problem_local_full_meta_pool"
    return "benchmark_local_qubit_excitation_singles_doubles_pool"


def _pool_name_for_config(config: _VariantConfig) -> str:
    if config.pool_kind == "full_meta":
        return "full_meta"
    return "qubit_excitation_singles_doubles_pool"


def _pool_construction_for_config(config: _VariantConfig) -> str:
    if config.pool_kind == "full_meta":
        return "problem-local full_meta pool from static_adapt pool builders"
    return "Hermitian single and double qubit-excitation generators expanded into repo exyz Pauli words"


def _taxonomy_role_for_config(config: _VariantConfig) -> str:
    if config.position_policy == "best_insert_refit":
        return "same_pool_pos_geo_comparator"
    if config.pool_kind == "qeb":
        return "operator_class_geo_comparator" if config.faithful_geo_adapt_vqe else "operator_class_comparator"
    return "same_pool_controller_comparator"


def _required_pool_key_for_config(config: _VariantConfig) -> str | None:
    return "full_meta" if config.pool_kind == "full_meta" else None


def _geo_selector_mode_for_config(config: _VariantConfig) -> str | None:
    if not _is_geo_config(config):
        return None
    if config.pool_kind == "qeb":
        return "qeb_pool_projected_natural_gradient"
    if config.position_policy == "best_insert_refit":
        return "full_meta_pool_projected_natural_gradient_position_optimized"
    return "full_pool_projected_natural_gradient"


def _geo_metric_definition_for_config(config: _VariantConfig) -> str | None:
    if not _is_geo_config(config):
        return None
    pool = _pool_name_for_config(config)
    return f"S_ij = Re(<t_i|t_j>) with t_i=(I-|psi><psi|)(-i G_i|psi>) over the scored {pool} pool"


def _geo_inner_optimizer_for_config(config: _VariantConfig) -> str | None:
    if not _is_geo_config(config):
        return None
    if config.optimizer_kind == "geo_qngd":
        return "qngd"
    if config.optimizer_kind == "spsa":
        return "spsa"
    if config.optimizer_kind == "powell":
        return "powell"
    if config.optimizer_kind == "rotosolve":
        return "rotosolve"
    return "bfgs"


def _scipy_or_coordinate_optimizer_method(config: _VariantConfig) -> str:
    if config.optimizer_kind == "powell":
        return "Powell"
    if config.optimizer_kind == "rotosolve":
        return "ROTOSOLVE"
    return "BFGS"


def _blocked_labels_for_config(
    config: _VariantConfig,
    *,
    selected_labels: set[str],
    previous_selected_label: str | None,
) -> set[str]:
    """Return labels removed before the full selector scan.

    With-replacement ADAPT scans never drain the pool.  Geo-ADAPT's adjacent
    duplicate rule is applied *after* the full-pool natural-gradient solve, as
    specified by the source algorithm; removing the previous label here would
    change every coupled component of the inverse-metric score.
    """
    if config.repeat_policy == "exclude_selected_labels":
        return {str(label) for label in selected_labels}
    return set()


def has_scipy_minimize_support() -> bool:
    try:
        from scipy.optimize import minimize as _minimize  # noqa: F401
    except Exception:
        return False
    return True


def _import_scipy_minimize():
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - optional dependency failure varies
        raise ImportError("generic static ADAPT variants require scipy.optimize.minimize") from exc
    return minimize


def default_static_adapt_variant_case_ids(family: str, algorithm_id: str | None = None) -> tuple[str, ...]:
    """Return canonical Table-I cases for a variant row."""
    if algorithm_id is not None:
        _get_config(algorithm_id)
    family_key = str(family).strip()
    if family_key not in GENERIC_STATIC_ADAPT_VARIANT_FAMILIES:
        return ()
    return table_i_canonical_case_ids(family_key)


def _spec_by_case_id(family: str, case_id: str, algorithm_id: str) -> HamiltonianBenchmarkSpec:
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    if family_key not in GENERIC_STATIC_ADAPT_VARIANT_FAMILIES:
        raise ValueError(f"{algorithm_id} is not implemented for family={family_key!r}")
    if case_key not in default_static_adapt_variant_case_ids(family_key, algorithm_id):
        raise ValueError(f"{algorithm_id} is not implemented for {family_key}/{case_key}")
    return with_molecular_vibronic_h2_fixture_override(
        table_i_canonical_spec_by_case_id(family_key, case_key),
        family=family_key,
    )


def _namespace_from_base_args(argv: Sequence[str]) -> Namespace:
    defaults: dict[str, Any] = {
        "problem": "hubbard",
        "L": 2,
        "t": 1.0,
        "u": 4.0,
        "dv": 0.0,
        "omega0": 1.0,
        "g_ep": 0.5,
        "n_ph_max": 1,
        "boson_encoding": "binary",
        "ordering": "blocked",
        "boundary": "periodic",
        "include_zero_point": True,
        "molecular_problem_json": None,
        "molecular_vibronic_h2_fixture_json": None,
        "v_nn": 0.0,
        "t_prime": 0.0,
        "n_fermions": None,
    }
    key_map = {
        "--problem": "problem",
        "--L": "L",
        "--t": "t",
        "--u": "u",
        "--dv": "dv",
        "--omega0": "omega0",
        "--g-ep": "g_ep",
        "--n-ph-max": "n_ph_max",
        "--boson-encoding": "boson_encoding",
        "--ordering": "ordering",
        "--boundary": "boundary",
        "--molecular-problem-json": "molecular_problem_json",
        "--molecular-vibronic-h2-fixture-json": "molecular_vibronic_h2_fixture_json",
        "--v-nn": "v_nn",
        "--t-prime": "t_prime",
        "--n-fermions": "n_fermions",
    }
    int_keys = {"L", "n_ph_max", "n_fermions"}
    float_keys = {"t", "u", "dv", "omega0", "g_ep", "v_nn", "t_prime"}
    values = dict(defaults)
    idx = 0
    argv_tuple = tuple(str(x) for x in argv)
    while idx < len(argv_tuple):
        token = argv_tuple[idx]
        if token == "--include-zero-point":
            values["include_zero_point"] = True
            idx += 1
            continue
        if token == "--no-include-zero-point":
            values["include_zero_point"] = False
            idx += 1
            continue
        if token not in key_map:
            idx += 1
            continue
        if idx + 1 >= len(argv_tuple):
            raise ValueError(f"Missing value for {token}")
        key = key_map[token]
        raw = argv_tuple[idx + 1]
        if key in int_keys and raw not in {"", "None", "none"}:
            values[key] = int(raw)
        elif key in float_keys:
            values[key] = float(raw)
        elif key == "n_fermions" and raw in {"", "None", "none"}:
            values[key] = None
        else:
            values[key] = raw
        idx += 2
    return Namespace(**values)


def _resolve_context_from_spec(spec: HamiltonianBenchmarkSpec) -> ResolvedProblemContext:
    request = ProblemRequest.from_namespace(_namespace_from_base_args(spec.base_pipeline_args))
    return resolve_problem_context(request)


def _safe_exact_energy(context: ResolvedProblemContext) -> float | None:
    try:
        return float(context.exact_target.resolve_energy(ai_log=None))
    except TypeError:
        try:
            return float(context.exact_target.resolve_energy())
        except Exception:
            return None
    except Exception:
        return None


def _spec_metadata(spec: HamiltonianBenchmarkSpec) -> dict[str, Any]:
    features = getattr(spec, "features", None)
    return {
        "benchmark_id": str(spec.benchmark_id),
        "family": str(spec.family),
        "base_pipeline_args": list(spec.base_pipeline_args),
        "split": str(spec.split),
        "tags": list(getattr(spec, "tags", ())),
        "features": asdict(features) if is_dataclass(features) else _json_default(features),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _append_jsonl(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def _source_fields(algorithm_id: str, **overrides: Any) -> dict[str, Any]:
    return comparator_source_fields(str(algorithm_id), runner_module=_RUNNER_MODULE, **overrides)


def _decision_value(
    recorder: BenchmarkDecisionNoiseRecorder | None,
    exact_value: float,
    *,
    surface: str,
    value_kind: str,
    phase: str,
    extra_scope: Mapping[str, Any] | None = None,
) -> float:
    """Return the decision value for a local ADAPT comparison surface.

    The recorder owns all draw accounting.  This helper never touches benchmark
    measurement counters; it only separates noisy decision values from exact
    energies/gradients used for reporting.
    """

    value = float(exact_value)
    if not math.isfinite(value):
        raise ValueError(f"ADAPT decision value must be finite; got {exact_value!r}")
    if recorder is None or not bool(getattr(recorder.config, "enabled", False)):
        return value
    return float(
        recorder.apply(
            value,
            surface=str(surface),
            value_kind=str(value_kind),
            phase=str(phase),
            extra_scope=extra_scope,
        )
    )


def _runtime_seed_settings_from_context(context: Any) -> dict[str, Any]:
    request = getattr(context, "request", None)
    if request is None:
        raise ValueError("runtime seed export requires a resolved problem context with request metadata")
    raw_n_fermions = getattr(request, "n_fermions", None)
    if raw_n_fermions is None or raw_n_fermions == "":
        n_fermions = None
    elif isinstance(raw_n_fermions, Sequence) and not isinstance(raw_n_fermions, (str, bytes, bytearray)):
        n_fermions = [int(x) for x in raw_n_fermions]
    else:
        n_fermions = int(raw_n_fermions)
    return {
        "L": int(getattr(request, "num_sites", 0)),
        "t": float(getattr(request, "t", 1.0)),
        "u": float(getattr(request, "u", 0.0)),
        "dv": float(getattr(request, "dv", 0.0)),
        "omega0": float(getattr(request, "omega0", 1.0)),
        "g_ep": float(getattr(request, "g_ep", 0.0)),
        "n_ph_max": int(getattr(request, "n_ph_max", 0)),
        "boson_encoding": str(getattr(request, "boson_encoding", "binary")),
        "ordering": str(getattr(request, "ordering", "blocked")),
        "boundary": str(getattr(request, "boundary", "open")),
        "problem": str(getattr(request, "problem_key", "hh")),
        "n_fermions": n_fermions,
        "paop_r": _FULL_META_PAOP_R,
        "paop_split_paulis": _FULL_META_PAOP_SPLIT_PAULIS,
        "paop_prune_eps": _FULL_META_PAOP_PRUNE_EPS,
        "paop_normalization": _FULL_META_PAOP_NORMALIZATION,
    }


def _build_runtime_seed_payload(
    *,
    context: Any,
    family: str,
    case_id: str,
    config: _VariantConfig,
    selected: Sequence[Any],
    selected_batches: Sequence[Sequence[Any]],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_final: np.ndarray,
    row: Mapping[str, Any],
    spec: HamiltonianBenchmarkSpec,
    generated_utc: str,
) -> dict[str, Any]:
    settings = _runtime_seed_settings_from_context(context)
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    theta_list = [float(x) for x in theta_arr]
    operator_labels = [str(candidate.label) for candidate in selected]
    parameterization_layout = None
    if all(hasattr(candidate, "polynomial") for candidate in selected):
        parameterization_layout = build_parameter_layout(
            list(selected),
            ignore_identity=True,
            coefficient_tolerance=1e-12,
            sort_terms=True,
        )
    row_parameterization_mode = str(row.get("parameterization_mode") or "").strip().lower()
    if int(theta_arr.size) == int(len(operator_labels)):
        logical_theta_arr = np.asarray(theta_arr, dtype=float)
        parameterization_mode = row_parameterization_mode or "logical_shared"
    elif parameterization_layout is not None and int(theta_arr.size) == int(parameterization_layout.runtime_parameter_count):
        logical_theta_arr = np.asarray(
            project_runtime_theta_block_mean(theta_arr, parameterization_layout),
            dtype=float,
        )
        parameterization_mode = row_parameterization_mode or "per_pauli_term"
    else:
        raise ValueError(
            "runtime seed selected/theta length mismatch: "
            f"{len(operator_labels)} operators vs {len(theta_list)} theta values "
            f"and runtime parameter count {None if parameterization_layout is None else int(parameterization_layout.runtime_parameter_count)}"
        )
    logical_theta_list = [float(x) for x in logical_theta_arr.reshape(-1)]
    parameterization_payload = (
        serialize_layout(parameterization_layout)
        if parameterization_layout is not None
        and str(parameterization_mode).strip().lower().startswith("per_pauli")
        else None
    )
    selected_paulis = [list(getattr(candidate, "pauli_labels_exyz", ())) for candidate in selected]
    selected_supports = [list(getattr(candidate, "support", ())) for candidate in selected]
    selected_execution_modes = [
        str(getattr(candidate, "execution_mode", "termwise_product") or "termwise_product")
        for candidate in selected
    ]
    selected_pauli_terms = [_candidate_pauli_terms_payload(candidate) for candidate in selected]
    generator_semantics_payload = [
        {
            "label": label,
            "execution_mode": mode,
            "pauli_labels_exyz": paulis,
            "pauli_terms": pauli_terms,
        }
        for label, mode, paulis, pauli_terms in zip(
            operator_labels,
            selected_execution_modes,
            selected_paulis,
            selected_pauli_terms,
            strict=False,
        )
    ]
    generator_semantics_sha256 = hashlib.sha256(
        json.dumps(generator_semantics_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    exact_energy = row.get("exact_energy", row.get("exact_gs_energy"))
    return {
        "generated_utc": str(generated_utc),
        "pipeline": "generic_static_adapt_variants_runtime_seed_v1",
        "schema": "paper_ii_static_seed_runtime_payload_v1",
        "family": str(family),
        "case_id": str(case_id),
        "algorithm_id": str(config.algorithm_id),
        "method_id": str(config.algorithm_id),
        "settings": settings,
        "ground_state": {
            "exact_energy": None if exact_energy is None or exact_energy == "" else float(exact_energy),
            "exact_energy_source": str(row.get("primary_reference_source", "generic_static_adapt_variants")),
            "method": "static_variant_reference_metric",
        },
        "adapt_vqe": {
            "algorithm_id": str(config.algorithm_id),
            "method_id": str(config.algorithm_id),
            "method_label": str(row.get("method_label", config.display_name)),
            "ansatz_depth": int(len(operator_labels)),
            "operators": operator_labels,
            "optimal_point": theta_list,
            "logical_optimal_point": logical_theta_list,
            "theta": theta_list,
            "num_parameters": int(len(theta_list)),
            "logical_parameter_count": int(len(logical_theta_list)),
            "parameterization_mode": parameterization_mode,
            "parameterization": parameterization_payload,
            "pool_type": str(row.get("pool_name", _pool_name_for_config(config))),
            "adapt_pool": str(row.get("pool_name", _pool_name_for_config(config))),
            "required_pool_key": str(row.get("required_pool_key", _required_pool_key_for_config(config))),
            "selected_operator_pauli_labels_exyz": selected_paulis,
            "selected_operator_supports": selected_supports,
            "selected_operator_execution_modes": selected_execution_modes,
            "selected_generator_semantics_sha256": generator_semantics_sha256,
            "selected_operator_pauli_terms": selected_pauli_terms,
            "selected_operator_runtime_split_metadata": [
                _pool_runtime_split_metadata(candidate) for candidate in selected
            ],
            "selected_operator_batches": [[str(candidate.label) for candidate in batch] for batch in selected_batches],
            "abs_delta_e": row.get("abs_delta_e"),
            "energy": row.get("energy"),
            "exact_gs_energy": row.get("exact_energy", row.get("exact_gs_energy")),
            "adapt_stop_reason": row.get("adapt_stop_reason"),
            "position_policy": row.get("position_policy"),
            "position_optimized_geo_adapt": row.get("position_optimized_geo_adapt"),
            "static_seed_source_runner": "pipelines.exact_bench.generic_static_adapt_variants",
        },
        "initial_state": build_statevector_manifest(
            psi_state=np.asarray(psi_final, dtype=complex).reshape(-1),
            source="generic_static_adapt_variants.final_prepared_state",
            handoff_state_kind="prepared_state",
            amplitude_cutoff=1e-12,
        ),
        "ansatz_input_state": build_statevector_manifest(
            psi_state=np.asarray(psi_ref, dtype=complex).reshape(-1),
            source="resolved_problem.reference_state",
            handoff_state_kind="reference_state",
            amplitude_cutoff=1e-12,
        ),
        "paper_ii_static_seed_export": {
            "schema": "paper_ii_static_seed_export_v1",
            "static_algorithm_id": str(config.algorithm_id),
            "static_seed_display_label": str(row.get("method_label", config.display_name)),
            "source_static_case_id": str(case_id),
            "source_family": str(family),
            "source_suite_profile": str(getattr(spec, "split", "")),
            "static_abs_delta_e": row.get("abs_delta_e"),
            "static_parameter_count": int(len(theta_list)),
            "runtime_loadability_status": "runtime_seed_sidecar_written_not_dry_loaded",
        },
        "source_static_table_result": {
            "schema": str(row.get("schema", SCHEMA_VERSION)),
            "algorithm_id": str(config.algorithm_id),
            "case_id": str(case_id),
            "result_json": "result.json",
            "generic_static_single_json": "generic_static_single.json",
        },
    }


def _write_artifacts(
    output_dir: Path,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    runtime_seed_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_with_source = dict(payload)
    algorithm_id = str(payload_with_source.get("algorithm_id") or payload_with_source.get("method_id") or "")
    if algorithm_id:
        payload_with_source.setdefault("comparator_source", _source_fields(algorithm_id))
    if runtime_seed_payload is not None:
        _write_json(output_dir / "runtime_seed.json", dict(runtime_seed_payload))
        payload_with_source["runtime_seed_json"] = str(output_dir / "runtime_seed.json")
        payload_with_source["runtime_seed_schema"] = str(runtime_seed_payload.get("schema", ""))
    rows_payload = {"schema": f"{SCHEMA_VERSION}_rows", "rows": list(rows)}
    if isinstance(payload_with_source.get("benchmark_decision_noise"), Mapping):
        rows_payload.update(
            {
                "benchmark_decision_noise_status": payload_with_source.get("benchmark_decision_noise_status"),
                "benchmark_decision_noise": copy_decision_noise_metadata(payload_with_source["benchmark_decision_noise"]),
            }
        )
    _write_json(output_dir / "result.json", payload_with_source)
    _write_json(output_dir / "rows.json", rows_payload)
    _write_json(
        output_dir / "manifest.json",
        {"schema": f"{SCHEMA_VERSION}_manifest", **{k: v for k, v in payload_with_source.items() if k != "schema"}},
    )
    _write_json(output_dir / "generic_static_single.json", payload_with_source)
    write_proxy_sidecars(rows, output_dir, summary_extras={"schema_source": SCHEMA_VERSION})
    return dict(payload_with_source)


def _guardrails(
    config: _VariantConfig,
    *,
    exact_reference_usage: str,
    uses_reference_for_decision: bool = False,
) -> dict[str, Any]:
    return {
        "uses_exact_for_decision": bool(uses_reference_for_decision),
        "uses_reference_for_decision": bool(uses_reference_for_decision),
        "exact_reference_usage": str(exact_reference_usage),
        "phase3_controller_called": False,
        "static_adapt_controller_boundary": "not_called",
        "phase3_emulation": False,
        "runner_boundary": "pipelines.exact_bench.generic_static_adapt_variants_only",
        "pool_source": _pool_source_for_config(config),
        "pool_name": _pool_name_for_config(config),
        "required_pool_key": _required_pool_key_for_config(config),
        "taxonomy_role": _taxonomy_role_for_config(config),
        "selector_variant": config.variant,
        "faithful_geo_adapt_vqe_implementation": bool(_geo_strict_source_conformance(config)),
        "geo_outer_selector_source_faithful": bool(
            _geo_outer_selector_matches_source_algorithm(config)
        ),
        "geo_source_algorithm_deviations": _geo_source_deviations(config),
        "geo_reference_algorithm": _GEO_REFERENCE_ALGORITHM if _is_geo_config(config) else None,
        "geo_stop_rule": _GEO_STOP_RULE if config.stop_rule == "geo_natural_gradient_norm" else "raw_gradient",
        "geo_inner_optimizer": _geo_inner_optimizer_for_config(config),
        "geo_replacement_policy": (
            "score_full_pool_with_replacement; skip_append_after_immediate_repeat_wins"
            if config.repeat_policy == "with_replacement_except_immediate_repeat"
            else "score_full_pool_with_replacement"
            if config.repeat_policy == "with_replacement"
            else "selected_labels_excluded"
        ),
        "repeat_policy": str(config.repeat_policy),
        "position_policy": config.position_policy,
        "position_optimized_geo_adapt": bool(config.position_policy == "best_insert_refit"),
    }


def _base_row(
    *,
    family: str,
    case_id: str,
    config: _VariantConfig,
    status: str,
    started_utc: str,
    finished_utc: str,
) -> dict[str, Any]:
    return {
        "run_id": f"{case_id}::{config.algorithm_id}",
        "schema": SCHEMA_VERSION,
        "family": family,
        "problem": family,
        "hamiltonian_id": case_id,
        "case_id": case_id,
        "algorithm_id": config.algorithm_id,
        "method_id": config.algorithm_id,
        "method_label": config.display_name,
        "method_kind": config.method_kind,
        "ansatz_name": config.ansatz_name,
        "algorithm_origin": "benchmark_local_statevector_adapt_variant",
        "status": status,
        "uses_exact_for_decision": False,
        "uses_reference_for_decision": False,
        "exact_reference_usage": "reporting_only_after_optimization",
        "phase3_controller_called": False,
        "static_adapt_controller_boundary": "not_called",
        "phase3_emulation": False,
        "adapt_append_only": bool(config.position_policy == "append"),
        "selector_variant": config.variant,
        "selector_rule": config.selector_rule,
        "pool_source": _pool_source_for_config(config),
        "pool_name": _pool_name_for_config(config),
        "pool_construction": _pool_construction_for_config(config),
        "required_pool_key": _required_pool_key_for_config(config),
        "taxonomy_role": _taxonomy_role_for_config(config),
        "faithful_geo_adapt_vqe_implementation": bool(_geo_strict_source_conformance(config)),
        "geo_outer_selector_source_faithful": bool(
            _geo_outer_selector_matches_source_algorithm(config)
        ),
        "geo_source_algorithm_deviations": _geo_source_deviations(config),
        "geo_legacy_faithfulness_route_flag": bool(config.faithful_geo_adapt_vqe),
        "geo_reference_algorithm": _GEO_REFERENCE_ALGORITHM if _is_geo_config(config) else None,
        "geo_faithfulness_scope": (
            "position_optimized_geo_adapt_over_problem_local_full_meta_pool"
            if config.position_policy == "best_insert_refit"
            else "append_only_geo_adapt_over_problem_local_full_meta_pool"
            if _geo_outer_selector_matches_source_algorithm(config) and config.pool_kind == "full_meta"
            else "exact_bench_local_qeb_excitation_pool_approximation"
            if _geo_outer_selector_matches_source_algorithm(config)
            else "not_faithful_geo_adapt_vqe_full_meta_metric_selector"
            if _is_geo_config(config)
            else None
        ),
        "geo_pool_faithfulness": (
            "problem-local full_meta pool for Hamiltonian-generic same-pool comparison"
            if config.pool_kind == "full_meta" and _geo_outer_selector_matches_source_algorithm(config)
            else "qubit_excitation_pool; strongest paper-faithfulness for fermionic/molecular cases"
            if _geo_outer_selector_matches_source_algorithm(config)
            else "full_meta pool; Geo-style projected-metric selector, not faithful QEB Geo-ADAPT-VQE"
            if _is_geo_config(config)
            else None
        ),
        "geo_stop_rule": _GEO_STOP_RULE if config.stop_rule == "geo_natural_gradient_norm" else "raw_gradient",
        "geo_natural_step_norm_threshold": None,
        "raw_gradient_used_for_stop": bool(_uses_raw_gradient_stop(config)),
        "raw_gradient_stop_rule": (
            str(config.stop_rule) if _uses_raw_gradient_stop(config) else None
        ),
        "geo_inner_optimizer": _geo_inner_optimizer_for_config(config),
        "optimizer_uses_exact_reference": False,
        "adapt_selection_with_replacement": bool(
            config.repeat_policy in {"with_replacement", "with_replacement_except_immediate_repeat"}
        ),
        "adapt_repeat_policy": str(config.repeat_policy),
        "geo_selection_with_replacement": bool(
            _is_geo_config(config)
            and config.repeat_policy == "with_replacement_except_immediate_repeat"
        ),
        "geo_immediate_repeat_blocked": bool(config.repeat_policy == "with_replacement_except_immediate_repeat"),
        "geo_immediate_repeat_policy_stage": (
            "post_full_pool_selection_skip_append"
            if config.repeat_policy == "with_replacement_except_immediate_repeat"
            else None
        ),
        "geo_replacement_policy": (
            "score_full_pool_with_replacement; skip_append_after_immediate_repeat_wins"
            if config.repeat_policy == "with_replacement_except_immediate_repeat"
            else "score_full_pool_with_replacement"
            if config.repeat_policy == "with_replacement"
            else "selected_labels_excluded"
        ),
        "position_policy": config.position_policy,
        "position_optimized_geo_adapt": bool(config.position_policy == "best_insert_refit"),
        "pauli_ordering": "left-to-right q_(n-1)...q_0; qubit 0 rightmost",
        "internal_pauli_alphabet": "e/x/y/z",
        "shots_total": 0,
        "static_shot_estimate_status": "not_applicable_not_completed",
        "shot_proxy_formula": _SHOT_PROXY_FORMULA,
        "shots_per_pauli_term_proxy": _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
        "hamiltonian_pauli_term_count": 0,
        "energy_eval_count_proxy": 0,
        "gradient_scan_count_proxy": 0,
        "gradient_operator_probe_count_proxy": 0,
        "metric_operator_probe_count_proxy": 0,
        "compiled_circuit_stats_status": "not_applicable_not_completed",
        "compiled_depth_total": None,
        "compiled_count_2q_total": None,
        "compiled_op_counts": None,
        **_source_fields(config.algorithm_id),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }


def _skip_payload(
    *,
    family: str,
    case_id: str,
    config: _VariantConfig,
    output_dir: Path,
    reason: str,
    started_utc: str,
) -> dict[str, Any]:
    finished = _utc_now()
    row = _base_row(
        family=family,
        case_id=case_id,
        config=config,
        status="skipped_optional_dependency",
        started_utc=started_utc,
        finished_utc=finished,
    )
    row.update(
        {
            "reason": reason,
            "exact_reference_usage": "not_resolved_for_dependency_skip",
            "energy": None,
            "exact_energy": None,
            "delta_E_abs": None,
            "num_qubits": None,
            "num_parameters": 0,
            "selected_operator_count": 0,
            "adapt_depth_reached": 0,
            "adapt_stop_reason": "optional_dependency_unavailable",
        }
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": config.algorithm_id,
        "method_id": config.algorithm_id,
        "status": "skipped_optional_dependency",
        "reason": reason,
        "runner": "pipelines.exact_bench.generic_static_adapt_variants.run_generic_static_adapt_variant_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(config, exact_reference_usage="not_resolved_for_dependency_skip"),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    return _write_artifacts(output_dir, payload, [row])


def _resource_guard_payload(
    *,
    family: str,
    case_id: str,
    config: _VariantConfig,
    output_dir: Path,
    spec: HamiltonianBenchmarkSpec,
    started_utc: str,
    reason: str,
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    finished = _utc_now()
    row = _base_row(
        family=family,
        case_id=case_id,
        config=config,
        status="skipped_resource_guard",
        started_utc=started_utc,
        finished_utc=finished,
    )
    row.update(
        {
            "reason": reason,
            "exact_reference_usage": "not_resolved_resource_guard",
            "energy": None,
            "exact_energy": None,
            "delta_E_abs": None,
            "num_qubits": guard.get("num_qubits"),
            "pool_term_count": guard.get("pool_term_count"),
            "num_parameters": 0,
            "selected_operator_count": 0,
            "adapt_depth_reached": 0,
            "adapt_stop_reason": "resource_guard",
            "resource_guard": True,
        }
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": config.algorithm_id,
        "method_id": config.algorithm_id,
        "status": "skipped_resource_guard",
        "reason": reason,
        "runner": "pipelines.exact_bench.generic_static_adapt_variants.run_generic_static_adapt_variant_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(config, exact_reference_usage="not_resolved_resource_guard"),
        "resource_guard": dict(guard),
        "spec": _spec_metadata(spec),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    return _write_artifacts(output_dir, payload, [row])


def _failure_payload(
    *,
    family: str,
    case_id: str,
    config: _VariantConfig,
    output_dir: Path,
    reason: str,
    exception_type: str,
    started_utc: str,
) -> dict[str, Any]:
    finished = _utc_now()
    row = _base_row(
        family=family,
        case_id=case_id,
        config=config,
        status="failed",
        started_utc=started_utc,
        finished_utc=finished,
    )
    row.update(
        {
            "reason": reason,
            "exception_type": exception_type,
            "exact_reference_usage": "reporting_only_after_optimization_or_not_reached",
            "energy": None,
            "exact_energy": None,
            "delta_E_abs": None,
            "runtime_s": None,
            "num_parameters": None,
            "selected_operator_count": None,
            "adapt_depth_reached": None,
            "adapt_stop_reason": "failed",
        }
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": config.algorithm_id,
        "method_id": config.algorithm_id,
        "status": "failed",
        "reason": reason,
        "exception_type": exception_type,
        "runner": "pipelines.exact_bench.generic_static_adapt_variants.run_generic_static_adapt_variant_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(config, exact_reference_usage="reporting_only_after_optimization_or_not_reached"),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    return _write_artifacts(output_dir, payload, [row])


def _label_for_two_qubit_generator(nq: int, qubit_a: int, op_a: str, qubit_b: int, op_b: str) -> str:
    chars = ["e"] * int(nq)
    chars[int(nq) - 1 - int(qubit_a)] = str(op_a)
    chars[int(nq) - 1 - int(qubit_b)] = str(op_b)
    return "".join(chars)


def _ladder_expansion_terms(num_qubits: int, ops: Sequence[tuple[int, str]]) -> dict[str, complex]:
    """Expand a product of qubit ladder operators into repo exyz Pauli words."""
    nq = int(num_qubits)
    terms: dict[tuple[str, ...], complex] = {tuple("e" for _ in range(nq)): 1.0 + 0.0j}
    for qubit, kind in ops:
        if kind == "+":
            factors = (("x", 0.5 + 0.0j), ("y", -0.5j))
        elif kind == "-":
            factors = (("x", 0.5 + 0.0j), ("y", 0.5j))
        else:  # pragma: no cover - internal misuse guard
            raise ValueError(f"unknown ladder kind {kind!r}")
        next_terms: dict[tuple[str, ...], complex] = {}
        for label, coeff in terms.items():
            for op, factor in factors:
                chars = list(label)
                chars[nq - 1 - int(qubit)] = op
                key = tuple(chars)
                next_terms[key] = next_terms.get(key, 0.0 + 0.0j) + coeff * factor
        terms = next_terms
    return {"".join(label): coeff for label, coeff in terms.items() if abs(coeff) > 1e-14}


def _double_qubit_excitation_terms(
    num_qubits: int,
    *,
    sources: tuple[int, int],
    targets: tuple[int, int],
) -> tuple[PauliTerm, ...]:
    """Return Hermitian i(T-T†) double qubit-excitation Pauli expansion."""
    nq = int(num_qubits)
    i, j = tuple(sorted(int(q) for q in sources))
    a, b = tuple(sorted(int(q) for q in targets))
    forward = _ladder_expansion_terms(nq, ((a, "+"), (b, "+"), (j, "-"), (i, "-")))
    backward = _ladder_expansion_terms(nq, ((i, "+"), (j, "+"), (b, "-"), (a, "-")))
    combined: dict[str, complex] = {}
    for label, coeff in forward.items():
        combined[label] = combined.get(label, 0.0 + 0.0j) + 1j * coeff
    for label, coeff in backward.items():
        combined[label] = combined.get(label, 0.0 + 0.0j) - 1j * coeff
    terms: list[PauliTerm] = []
    for label, coeff in sorted(combined.items()):
        if abs(coeff) <= 1e-14:
            continue
        if abs(coeff.imag) > 1e-12:
            raise ValueError(f"double qubit-excitation coefficient is not real for {label}: {coeff}")
        terms.append(PauliTerm(nq, ps=label, pc=float(coeff.real)))
    return tuple(terms)


def build_pairwise_qubit_excitation_pool(num_qubits: int, *, max_terms: int | None = _POOL_TERM_CAP) -> tuple[_PoolCandidate, ...]:
    """Build generic QEB singles+doubles in repo exyz convention.

    Singles use ``G_ij=(X_iY_j-Y_iX_j)/2``.  Doubles use the Hermitian
    generator ``i(T-T†)`` for qubit ladder excitation products.  This remains a
    benchmark-local competitor pool and does not call project Phase3/SNAKE code.
    The public function name is kept for compatibility with existing tests and
    CHTC records.
    """
    return tuple(
        _PoolCandidate(
            label=spec.label,
            polynomial=spec.polynomial,
            support=tuple(spec.support),
            pauli_labels_exyz=tuple(spec.pauli_labels_exyz),
            construction=spec.construction,
        )
        for spec in build_qeb_pool_specs(int(num_qubits), max_terms=max_terms)
    )


def _polynomial_labels_and_support(polynomial: PauliPolynomial) -> tuple[tuple[str, ...], tuple[int, ...]]:
    labels: list[str] = []
    support: set[int] = set()
    for term in list(polynomial.return_polynomial()):
        coeff = complex(term.p_coeff)
        if abs(coeff) <= 1e-12:
            continue
        label = str(term.pw2strng()).lower()
        if not label:
            continue
        labels.append(label)
        nq = int(getattr(term, "N", len(label))) if hasattr(term, "N") else len(label)
        for idx, ch in enumerate(label):
            if ch != "e":
                support.add(int(nq - 1 - idx))
    return tuple(labels), tuple(sorted(support))


def _serialized_terms_exyz(polynomial: PauliPolynomial, *, tol: float = 1e-12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for term in list(polynomial.return_polynomial()):
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        out.append(
            {
                "pauli_exyz": str(term.pw2strng()).lower(),
                "coeff_re": float(coeff.real),
                "coeff_im": float(coeff.imag),
                "nq": int(term.nqubit()),
            }
        )
    return out


def _normalize_generic_adapt_runtime_split_mode(value: str | None) -> str:
    key = str(value or _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF).strip().lower()
    if key in {"", "none", "false", "0", "disabled"}:
        key = _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF
    if key not in _GENERIC_ADAPT_RUNTIME_SPLIT_MODES:
        allowed = ", ".join(sorted(_GENERIC_ADAPT_RUNTIME_SPLIT_MODES))
        raise ValueError(f"generic_adapt_runtime_split_mode must be one of {{{allowed}}}; got {value!r}.")
    return key


def _normalize_generic_adapt_runtime_split_symmetry_policy(value: str | None) -> str:
    key = str(value or _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_OFF).strip().lower().replace("-", "_")
    if key in {"", "none", "false", "0", "disabled"}:
        key = _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_OFF
    if key not in _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICIES:
        allowed = ", ".join(sorted(_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICIES))
        raise ValueError(f"generic_adapt_runtime_split_symmetry_policy must be one of {{{allowed}}}; got {value!r}.")
    return key


def _runtime_split_qpb(context: ResolvedProblemContext) -> int:
    num_sites = int(getattr(context.request, "num_sites", 0) or 0)
    total_qubits = int(getattr(context.layout, "total_qubits", 0) or 0)
    fermion_qubits = int(getattr(context.layout, "fermion_qubits", 2 * max(0, num_sites)) or 0)
    boson_qubits = max(0, total_qubits - fermion_qubits)
    if num_sites <= 0 or boson_qubits <= 0:
        return 1
    return max(1, int(math.ceil(float(boson_qubits) / float(num_sites))))


def _generic_adapt_runtime_split_symmetry_spec(policy: str) -> dict[str, Any]:
    if policy == _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_HARD_GUARD:
        return {
            "policy": _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_HARD_GUARD,
            "hard_guard": True,
            "particle_number_mode": "preserving",
            "spin_sector_mode": "preserving",
            "source": "generic_static_adapt_variants_runtime_split",
        }
    return {
        "policy": _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_OFF,
        "hard_guard": False,
        "particle_number_mode": "off",
        "spin_sector_mode": "off",
        "source": "generic_static_adapt_variants_runtime_split",
    }


def _candidate_parent_generator_metadata(candidate: _PoolCandidate) -> dict[str, Any]:
    meta = dict(candidate.generator_metadata or {})
    meta.setdefault("label", str(candidate.label))
    compile_meta = dict(meta.get("compile_metadata", {})) if isinstance(meta.get("compile_metadata"), Mapping) else {}
    serialized = _serialized_terms_exyz(candidate.polynomial)
    compile_meta.setdefault("serialized_terms_exyz", [dict(item) for item in serialized])
    compile_meta.setdefault("num_polynomial_terms", int(len(serialized)))
    compile_meta.setdefault("signature_size", int(len(serialized)))
    meta["compile_metadata"] = compile_meta
    meta.setdefault("is_macro_generator", bool(len(serialized) > 1))
    return meta


def _runtime_split_candidate_from_child_set(
    *,
    parent: _PoolCandidate,
    child_set: Mapping[str, Any],
    split_mode: str,
) -> _PoolCandidate | None:
    polynomial = child_set.get("candidate_polynomial")
    if not isinstance(polynomial, PauliPolynomial):
        return None
    labels, support = _polynomial_labels_and_support(polynomial)
    if not labels or not support:
        return None
    child_indices = tuple(int(idx) for idx in child_set.get("child_indices", ()) or ())
    child_labels = tuple(str(label) for label in child_set.get("child_labels", ()) or ())
    raw_gate = child_set.get("symmetry_gate")
    symmetry_gate = dict(raw_gate) if isinstance(raw_gate, Mapping) else None
    raw_meta = child_set.get("candidate_generator_metadata")
    generator_metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else None
    execution_mode = str(child_set.get("recommended_execution_mode") or "termwise_product")
    return _PoolCandidate(
        label=str(child_set.get("candidate_label", "")),
        polynomial=polynomial,
        support=tuple(int(q) for q in support),
        pauli_labels_exyz=tuple(str(label).lower() for label in labels),
        construction=f"{parent.construction}::runtime_split_child_set",
        parent_label=str(parent.label),
        runtime_split_mode=str(split_mode),
        runtime_split_representation="child_set",
        runtime_split_child_indices=child_indices,
        runtime_split_child_labels=child_labels,
        runtime_split_symmetry_gate=symmetry_gate,
        execution_mode=execution_mode,
        generator_metadata=generator_metadata,
    )


def _pool_runtime_split_metadata(candidate: _PoolCandidate) -> dict[str, Any] | None:
    mode = str(getattr(candidate, "runtime_split_mode", _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF))
    if mode == _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF:
        return None
    return {
        "mode": mode,
        "representation": str(getattr(candidate, "runtime_split_representation", "")),
        "parent_label": getattr(candidate, "parent_label", None),
        "child_indices": [int(idx) for idx in getattr(candidate, "runtime_split_child_indices", ())],
        "child_labels": [str(label) for label in getattr(candidate, "runtime_split_child_labels", ())],
        "execution_mode": str(getattr(candidate, "execution_mode", "termwise_product")),
        "symmetry_gate": (
            dict(getattr(candidate, "runtime_split_symmetry_gate", None))
            if isinstance(getattr(candidate, "runtime_split_symmetry_gate", None), Mapping)
            else None
        ),
    }


def _expand_pool_with_runtime_split_children(
    *,
    pool: Sequence[_PoolCandidate],
    context: ResolvedProblemContext,
    config: _VariantConfig,
    split_mode: str | None,
    symmetry_policy: str | None,
    max_subset_size: int | str | None,
    max_terms: int | None,
) -> tuple[tuple[_PoolCandidate, ...], dict[str, Any]]:
    mode = _normalize_generic_adapt_runtime_split_mode(split_mode)
    policy = _normalize_generic_adapt_runtime_split_symmetry_policy(symmetry_policy)
    subset_cap_raw = 1 if max_subset_size in {None, ""} else max_subset_size
    subset_cap = _positive_int(subset_cap_raw, field="generic_adapt_runtime_split_max_subset_size")
    base_pool = tuple(pool)
    meta: dict[str, Any] = {
        "schema": "generic_adapt_runtime_split_pool_expansion_v1",
        "enabled": bool(mode != _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF),
        "mode": mode,
        "symmetry_policy": policy,
        "max_subset_size": int(subset_cap),
        "base_pool_term_count": int(len(base_pool)),
        "expanded_pool_term_count": int(len(base_pool)),
        "split_parent_count": 0,
        "child_atom_count": 0,
        "child_set_candidate_count": 0,
        "added_child_set_count": 0,
        "symmetry_checked_child_atom_count": 0,
        "symmetry_rejected_child_atom_count": 0,
        "symmetry_checked_child_set_count": 0,
        "symmetry_gate_enforced": bool(policy == _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_HARD_GUARD),
        "supports_only": "paper_i_hh_or_hubbard_full_meta_append_and_geo",
    }
    if mode == _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF:
        return base_pool, meta
    if config.pool_kind != "full_meta" or config.algorithm_id not in _HH_ACTIVE_FULL_META_MINUS_HVA_ALGORITHM_IDS:
        raise ValueError(
            "generic ADAPT runtime split currently supports only Paper-I HH/Hubbard full_meta append and Geo comparator rows; "
            f"algorithm_id={config.algorithm_id!r}, pool_kind={config.pool_kind!r}."
        )
    family_key = _runtime_split_family_id(context)
    if family_key not in _GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_FAMILIES:
        raise ValueError(
            "generic ADAPT runtime split currently supports only HH/Hubbard full_meta comparator rows; "
            f"context family={getattr(context, 'family_key', None)!r}."
        )

    num_sites = int(context.request.num_sites)
    ordering = str(getattr(context.request, "ordering", "blocked"))
    qpb = _runtime_split_qpb(context)
    symmetry_spec = _generic_adapt_runtime_split_symmetry_spec(policy)
    expanded: list[_PoolCandidate] = list(base_pool)
    child_label_seen = {str(candidate.label) for candidate in expanded}

    for parent in base_pool:
        if len(parent.pauli_labels_exyz) <= 1:
            continue
        parent_meta = _candidate_parent_generator_metadata(parent)
        children = build_runtime_split_children(
            parent_label=str(parent.label),
            polynomial=parent.polynomial,
            family_id=str(family_key),
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(qpb),
            split_mode=mode,
            parent_generator_metadata=parent_meta,
            symmetry_spec=symmetry_spec,
        )
        if not children:
            continue
        meta["split_parent_count"] = int(meta["split_parent_count"]) + 1
        meta["child_atom_count"] = int(meta["child_atom_count"]) + int(len(children))
        for child in children:
            gate = child.get("symmetry_gate")
            if isinstance(gate, Mapping) and bool(gate.get("checked", False)):
                meta["symmetry_checked_child_atom_count"] = int(meta["symmetry_checked_child_atom_count"]) + 1
                if not bool(gate.get("passed", True)):
                    meta["symmetry_rejected_child_atom_count"] = int(meta["symmetry_rejected_child_atom_count"]) + 1
        child_sets = build_runtime_split_child_sets(
            parent_label=str(parent.label),
            family_id=str(family_key),
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(qpb),
            split_mode=mode,
            children=children,
            parent_generator_metadata=parent_meta,
            symmetry_spec=symmetry_spec,
            max_subset_size=int(subset_cap),
        )
        meta["child_set_candidate_count"] = int(meta["child_set_candidate_count"]) + int(len(child_sets))
        for child_set in child_sets:
            candidate = _runtime_split_candidate_from_child_set(
                parent=parent,
                child_set=child_set,
                split_mode=mode,
            )
            if candidate is None:
                continue
            if str(candidate.label) in child_label_seen:
                continue
            gate = candidate.runtime_split_symmetry_gate
            if isinstance(gate, Mapping) and bool(gate.get("checked", False)):
                meta["symmetry_checked_child_set_count"] = int(meta["symmetry_checked_child_set_count"]) + 1
            expanded.append(candidate)
            child_label_seen.add(str(candidate.label))
            meta["added_child_set_count"] = int(meta["added_child_set_count"]) + 1
            if max_terms is not None and len(expanded) > int(max_terms):
                raise ValueError(f"runtime-split full_meta pool exceeds cap: {len(expanded)} > {int(max_terms)}")

    meta["expanded_pool_term_count"] = int(len(expanded))
    meta["expansion_factor"] = (
        float(len(expanded)) / float(len(base_pool))
        if len(base_pool) > 0
        else None
    )
    return tuple(expanded), meta


def _shared_parent_from_pool_candidate(candidate: _PoolCandidate) -> SharedPauliPoolParent:
    return SharedPauliPoolParent(
        label=str(candidate.label),
        polynomial=candidate.polynomial,
        family_id=str(candidate.generator_metadata.get("family_id", "")) if isinstance(candidate.generator_metadata, Mapping) and candidate.generator_metadata.get("family_id") is not None else _runtime_split_family_id_for_candidate(candidate),
        stage_family=str(candidate.construction or "shared"),
        construction=str(candidate.construction or "parent"),
        execution_mode=str(candidate.execution_mode or "termwise_product"),
        symmetry_spec=None,
        generator_metadata=dict(candidate.generator_metadata or {}),
    )


def _runtime_split_family_id_for_candidate(candidate: _PoolCandidate) -> str:
    if isinstance(candidate.generator_metadata, Mapping) and candidate.generator_metadata.get("family_id") is not None:
        return str(candidate.generator_metadata.get("family_id"))
    label = str(candidate.label)
    if "::" in label:
        return label.split("::", 1)[0]
    return "unknown"


def _pool_candidate_from_shared_pauli(candidate: Any) -> _PoolCandidate | None:
    polynomial = getattr(candidate, "polynomial", None)
    if not isinstance(polynomial, PauliPolynomial):
        return None
    labels, support = _polynomial_labels_and_support(polynomial)
    if not labels or not support:
        return None
    representation = str(getattr(candidate, "representation", "parent"))
    mode = (
        str(getattr(candidate, "generator_metadata", {}).get("shared_pauli_pool_contract", {}).get("mode", "off"))
        if isinstance(getattr(candidate, "generator_metadata", None), Mapping)
        else "off"
    )
    return _PoolCandidate(
        label=str(getattr(candidate, "label")),
        polynomial=polynomial,
        support=tuple(int(q) for q in support),
        pauli_labels_exyz=tuple(str(label).lower() for label in labels),
        construction=str(getattr(candidate, "construction", "shared_pauli_pool")),
        parent_label=getattr(candidate, "parent_label", None),
        runtime_split_mode=("off" if representation == "parent" else mode),
        runtime_split_representation=representation,
        runtime_split_child_indices=tuple(int(idx) for idx in getattr(candidate, "child_indices", ()) or ()),
        runtime_split_child_labels=tuple(str(label) for label in getattr(candidate, "child_labels", ()) or ()),
        runtime_split_symmetry_gate=(
            dict(getattr(candidate, "symmetry_gate"))
            if isinstance(getattr(candidate, "symmetry_gate", None), Mapping)
            else None
        ),
        execution_mode=str(getattr(candidate, "execution_mode", "termwise_product")),
        generator_metadata=dict(getattr(candidate, "generator_metadata", {}) or {}),
    )


def _expand_pool_with_shared_pauli_children(
    *,
    pool: Sequence[_PoolCandidate],
    context: ResolvedProblemContext,
    config: _VariantConfig,
    mode: str | None,
    symmetry_policy: str | None,
    max_subset_size: int | str | None,
    max_terms: int | None,
) -> tuple[tuple[_PoolCandidate, ...], dict[str, Any]]:
    mode_key = normalize_shared_pauli_pool_mode(mode)
    policy_key = normalize_shared_pauli_pool_symmetry_policy(symmetry_policy)
    subset_cap_raw = 3 if max_subset_size in {None, ""} else max_subset_size
    subset_cap = _positive_int(subset_cap_raw, field="shared_pauli_pool_max_subset_size")
    base_pool = tuple(pool)
    family_key = _runtime_split_family_id(context)
    if mode_key != SHARED_PAULI_POOL_MODE_OFF:
        if config.pool_kind != "full_meta" or config.algorithm_id not in _HH_ACTIVE_FULL_META_MINUS_HVA_ALGORITHM_IDS:
            raise ValueError(
                "shared_pauli_pool_mode currently supports only Paper-I HH/Hubbard full_meta append and Geo comparator rows; "
                f"algorithm_id={config.algorithm_id!r}, pool_kind={config.pool_kind!r}."
            )
        if family_key not in _GENERIC_ADAPT_RUNTIME_SPLIT_SUPPORTED_FAMILIES:
            raise ValueError(
                "shared_pauli_pool_mode currently supports only HH/Hubbard full_meta comparator rows; "
                f"context family={getattr(context, 'family_key', None)!r}."
            )
    parents = tuple(
        SharedPauliPoolParent(
            label=str(candidate.label),
            polynomial=candidate.polynomial,
            family_id=str(family_key),
            stage_family=str(candidate.construction or "full_meta"),
            construction=str(candidate.construction or "full_meta"),
            execution_mode=str(candidate.execution_mode or "termwise_product"),
            symmetry_spec=None,
            generator_metadata=_candidate_parent_generator_metadata(candidate),
        )
        for candidate in base_pool
    )
    result = build_shared_pauli_child_pool(
        parents=parents,
        mode=mode_key,
        symmetry_policy=policy_key,
        max_subset_size=int(subset_cap),
        problem_key=str(getattr(context.request, "problem_key", family_key)),
        num_sites=int(context.request.num_sites),
        ordering=str(getattr(context.request, "ordering", "blocked")),
        qpb=int(_runtime_split_qpb(context)),
        max_terms=max_terms,
    )
    converted = tuple(
        item
        for item in (_pool_candidate_from_shared_pauli(candidate) for candidate in result.candidates)
        if item is not None
    )
    meta = dict(result.meta)
    meta["applied_to_algorithm_id"] = str(config.algorithm_id)
    meta["applied_to_method_kind"] = str(config.method_kind)
    meta["base_pool_name"] = _pool_name_for_config(config)
    if len(converted) != len(result.candidates):
        meta["conversion_dropped_count"] = int(len(result.candidates) - len(converted))
    return converted, meta


def _selected_logical_mode_from_route(route: str | None) -> str:
    route_key = str(route or "standard").strip().lower().replace("-", "_")
    if route_key in {"", "standard", "off"}:
        return "off"
    if route_key == "historical_selected":
        return "family_closure_with_full_fallback"
    raise ValueError("selected_logical_route must be one of {'standard','historical_selected'}.")


def _repo_relative_or_abs(path: str | Path | None) -> str | None:
    if path in {None, ""}:
        return None
    candidate = Path(str(path))
    try:
        return str(candidate.resolve().relative_to(_REPO_ROOT.resolve()))
    except Exception:
        return str(candidate)


def _is_hh_context(context: ResolvedProblemContext) -> bool:
    family = str(getattr(context, "family_key", "") or getattr(getattr(context, "request", None), "problem_key", ""))
    return family.strip().lower() == "hh"


def _runtime_split_family_id(context: ResolvedProblemContext) -> str:
    family = str(getattr(context, "family_key", "") or getattr(getattr(context, "request", None), "problem_key", ""))
    return family.strip().lower()


def _active_hh_full_meta_minus_hva_class_filter_json(
    *,
    config: _VariantConfig,
    context: ResolvedProblemContext,
) -> Path | None:
    if not _is_hh_context(context):
        return None
    if config.pool_kind != "full_meta":
        return None
    if config.algorithm_id not in _HH_ACTIVE_FULL_META_MINUS_HVA_ALGORITHM_IDS:
        return None
    return _HH_FULL_META_MINUS_HVA_CLASS_FILTER_JSON


def _normalize_hh_adaptive_pool_profile(value: str | None) -> str:
    key = str(value or _HH_ADAPTIVE_POOL_PROFILE_LEGACY_AUTO).strip().lower().replace("-", "_")
    if key in {"", "source", "preserve"}:
        key = _HH_ADAPTIVE_POOL_PROFILE_LEGACY_AUTO
    if key not in _HH_ADAPTIVE_POOL_PROFILES:
        allowed = ", ".join(sorted(_HH_ADAPTIVE_POOL_PROFILES))
        raise ValueError(f"hh_adaptive_pool_profile must be one of {{{allowed}}}; got {value!r}.")
    return key


def _resolve_hh_full_meta_pool_profile(
    *,
    config: _VariantConfig,
    context: ResolvedProblemContext,
    hh_adaptive_pool_profile: str | None = None,
    hh_full_meta_class_filter_json: str | Path | None = None,
) -> tuple[str | None, Path | None]:
    """Return the effective HH full_meta profile and class filter path.

    ``legacy_auto`` preserves the historical generic append/Geo behavior:
    active HH full_meta comparator rows use the minus-HVA filter.  Matrix rows
    that need the unfiltered full_meta contract must opt in explicitly.
    """
    if not _is_hh_context(context) or config.pool_kind != "full_meta":
        return None, None

    profile = _normalize_hh_adaptive_pool_profile(hh_adaptive_pool_profile)
    raw_filter = None if hh_full_meta_class_filter_json in {None, ""} else str(hh_full_meta_class_filter_json).strip()
    if raw_filter:
        filter_key = raw_filter.lower().replace("-", "_")
        if filter_key in {"off", "none", "unfiltered"}:
            if profile == _HH_ADAPTIVE_POOL_PROFILE_FULL_META_MINUS_HVA:
                raise ValueError("hh_full_meta_class_filter_json=off conflicts with full_meta_minus_hva profile.")
            return _HH_ADAPTIVE_POOL_PROFILE_FULL_META_UNFILTERED, None
        if profile == _HH_ADAPTIVE_POOL_PROFILE_FULL_META_UNFILTERED:
            raise ValueError("full_meta_unfiltered profile cannot also set hh_full_meta_class_filter_json.")
        return _HH_ADAPTIVE_POOL_PROFILE_FULL_META_MINUS_HVA, Path(raw_filter)

    if profile == _HH_ADAPTIVE_POOL_PROFILE_FULL_META_UNFILTERED:
        return profile, None
    if profile == _HH_ADAPTIVE_POOL_PROFILE_FULL_META_MINUS_HVA:
        return profile, _HH_FULL_META_MINUS_HVA_CLASS_FILTER_JSON

    path = _active_hh_full_meta_minus_hva_class_filter_json(config=config, context=context)
    return (_HH_ADAPTIVE_POOL_PROFILE_FULL_META_MINUS_HVA if path is not None else None), path


def _pool_cache_events(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for raw in events:
        event = str(raw.get("event", ""))
        if not (
            event.startswith("hardcoded_adapt_pool_cache")
            or event.startswith("hardcoded_adapt_generator_registry_cache")
        ):
            continue
        out.append({str(k): v for k, v in raw.items()})
    return tuple(out)


def _plan_mapping_meta(plan: Any, field: str) -> dict[str, Any] | None:
    value = getattr(plan, str(field), None)
    return dict(value) if isinstance(value, Mapping) else None


def _build_full_meta_candidate_pool_with_meta(
    context: ResolvedProblemContext,
    *,
    max_terms: int | None = _POOL_TERM_CAP,
    hh_full_meta_class_filter_json: str | Path | None = None,
) -> _FullMetaCandidatePoolResult:
    """Build the problem-local full_meta pool as benchmark-local candidates."""
    class_filter_path = None if hh_full_meta_class_filter_json in {None, ""} else Path(str(hh_full_meta_class_filter_json))
    pool_events: list[dict[str, Any]] = []

    def _pool_ai_log(event: str, **fields: Any) -> None:
        pool_events.append({"event": str(event), **fields})

    filter_resolution = resolve_requested_pool_filters(
        problem_key=str(context.family_key),
        num_sites=int(context.request.num_sites),
        n_ph_max=int(getattr(context.request, "n_ph_max", 0) or 0),
        adapt_pool="full_meta",
        adapt_pool_class_filter_json=class_filter_path,
        adapt_pool_label_filter_json=None,
        adapt_selected_logical_source_json=None,
        adapt_selected_logical_mode="off",
        adapt_selected_logical_transfer_mode="exact_match_v1",
    )
    plan = resolve_pool_plan(
        resolved_problem=context,
        continuation_mode="benchmark_static_geo_adapt",
        adapt_pool="full_meta",
        paop_r=_FULL_META_PAOP_R,
        paop_split_paulis=_FULL_META_PAOP_SPLIT_PAULIS,
        paop_prune_eps=_FULL_META_PAOP_PRUNE_EPS,
        paop_normalization=_FULL_META_PAOP_NORMALIZATION,
        phase3_symmetry_mitigation_mode="off",
        filter_resolution=filter_resolution,
        ai_log=_pool_ai_log,
    )
    out: list[_PoolCandidate] = []
    for term in list(plan.pool):
        labels, support = _polynomial_labels_and_support(term.polynomial)
        if not labels or not support:
            continue
        raw_metadata = getattr(term, "generator_metadata", None)
        if not isinstance(raw_metadata, Mapping):
            raw_metadata = getattr(term, "metadata", None)
        out.append(
            _PoolCandidate(
                label=str(term.label),
                polynomial=term.polynomial,
                support=tuple(int(q) for q in support),
                pauli_labels_exyz=tuple(labels),
                construction=f"full_meta::{plan.pool_key}",
                execution_mode=str(
                    getattr(term, "execution_mode", "termwise_product") or "termwise_product"
                ),
                generator_metadata=(dict(raw_metadata) if isinstance(raw_metadata, Mapping) else None),
            )
        )
        if max_terms is not None and len(out) > int(max_terms):
            raise ValueError(f"full_meta pool exceeds cap: {len(out)} > {int(max_terms)}")
    return _FullMetaCandidatePoolResult(
        candidates=tuple(out),
        selected_logical_filter_meta=_plan_mapping_meta(plan, "selected_logical_filter_meta"),
        full_meta_class_filter_meta=_plan_mapping_meta(plan, "full_meta_class_filter_meta"),
        full_meta_label_filter_meta=_plan_mapping_meta(plan, "full_meta_label_filter_meta"),
        pool_legal_subspace_filter_meta=_plan_mapping_meta(plan, "pool_legal_subspace_filter_meta"),
        pool_key=str(plan.pool_key),
        pool_cache_events=_pool_cache_events(pool_events),
    )


def build_full_meta_candidate_pool(
    context: ResolvedProblemContext,
    *,
    max_terms: int | None = _POOL_TERM_CAP,
) -> tuple[_PoolCandidate, ...]:
    return _build_full_meta_candidate_pool_with_meta(context, max_terms=max_terms).candidates


def _build_reduced_full_meta_candidate_pool_with_meta(
    context: ResolvedProblemContext,
    *,
    max_terms: int | None,
    selected_logical_source_json: str | Path | None,
    selected_logical_mode: str,
    selected_logical_transfer_mode: str,
    hh_full_meta_class_filter_json: str | Path | None = None,
) -> _FullMetaCandidatePoolResult:
    source_path = (
        None
        if selected_logical_source_json in {None, ""}
        else Path(str(selected_logical_source_json))
    )
    class_filter_path = None if hh_full_meta_class_filter_json in {None, ""} else Path(str(hh_full_meta_class_filter_json))
    pool_events: list[dict[str, Any]] = []

    def _pool_ai_log(event: str, **fields: Any) -> None:
        pool_events.append({"event": str(event), **fields})

    filter_resolution = resolve_requested_pool_filters(
        problem_key=str(context.family_key),
        num_sites=int(context.request.num_sites),
        n_ph_max=int(getattr(context.request, "n_ph_max", 0) or 0),
        adapt_pool="full_meta",
        adapt_pool_class_filter_json=class_filter_path,
        adapt_pool_label_filter_json=None,
        adapt_selected_logical_source_json=source_path,
        adapt_selected_logical_mode=str(selected_logical_mode or "off"),
        adapt_selected_logical_transfer_mode=str(selected_logical_transfer_mode or "exact_match_v1"),
    )
    plan = resolve_pool_plan(
        resolved_problem=context,
        continuation_mode="benchmark_static_geo_adapt",
        adapt_pool="full_meta",
        paop_r=_FULL_META_PAOP_R,
        paop_split_paulis=_FULL_META_PAOP_SPLIT_PAULIS,
        paop_prune_eps=_FULL_META_PAOP_PRUNE_EPS,
        paop_normalization=_FULL_META_PAOP_NORMALIZATION,
        phase3_symmetry_mitigation_mode="off",
        filter_resolution=filter_resolution,
        ai_log=_pool_ai_log,
    )
    out: list[_PoolCandidate] = []
    for term in list(plan.pool):
        labels, support = _polynomial_labels_and_support(term.polynomial)
        if not labels or not support:
            continue
        raw_metadata = getattr(term, "generator_metadata", None)
        if not isinstance(raw_metadata, Mapping):
            raw_metadata = getattr(term, "metadata", None)
        out.append(
            _PoolCandidate(
                label=str(term.label),
                polynomial=term.polynomial,
                support=tuple(int(q) for q in support),
                pauli_labels_exyz=tuple(labels),
                construction=f"full_meta::{plan.pool_key}",
                execution_mode=str(
                    getattr(term, "execution_mode", "termwise_product") or "termwise_product"
                ),
                generator_metadata=(dict(raw_metadata) if isinstance(raw_metadata, Mapping) else None),
            )
        )
        if max_terms is not None and len(out) > int(max_terms):
            raise ValueError(f"full_meta pool exceeds cap: {len(out)} > {int(max_terms)}")
    return _FullMetaCandidatePoolResult(
        candidates=tuple(out),
        selected_logical_filter_meta=_plan_mapping_meta(plan, "selected_logical_filter_meta"),
        full_meta_class_filter_meta=_plan_mapping_meta(plan, "full_meta_class_filter_meta"),
        full_meta_label_filter_meta=_plan_mapping_meta(plan, "full_meta_label_filter_meta"),
        pool_legal_subspace_filter_meta=_plan_mapping_meta(plan, "pool_legal_subspace_filter_meta"),
        pool_key=str(plan.pool_key),
        pool_cache_events=_pool_cache_events(pool_events),
    )


def _hamiltonian_pauli_term_count(hamiltonian: Any, *, tol: float = 1e-12) -> int:
    labels: set[str] = set()
    for term in list(hamiltonian.return_polynomial()):
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        label = str(term.pw2strng()).lower()
        if not label or label == "e" * len(label):
            continue
        labels.add(label)
    return int(len(labels))


def _resource_guard_for_context(
    context: ResolvedProblemContext,
    pool: Sequence[_PoolCandidate],
    *,
    pool_cap: int | None = _POOL_TERM_CAP,
    qubit_cap: int | None = _QUBIT_CAP,
    pool_name: str = "qubit_excitation_singles_doubles_pool",
) -> dict[str, Any] | None:
    num_qubits = int(context.layout.total_qubits)
    pool_count = int(len(tuple(pool)))
    if qubit_cap is not None and num_qubits > int(qubit_cap):
        return {
            "resource_guard": True,
            "resource_guard_kind": "generic_adapt_variant_qubit_cap",
            "reason": "Generic statevector ADAPT variant canonical case qubit count exceeds cap",
            "num_qubits": num_qubits,
            "qubit_cap": int(qubit_cap),
            "pool_term_count": pool_count,
            "pool_term_cap": None if pool_cap is None else int(pool_cap),
        }
    if pool_count <= 0:
        return {
            "resource_guard": True,
            "resource_guard_kind": "generic_adapt_variant_empty_pool",
            "reason": f"{pool_name} is empty",
            "num_qubits": num_qubits,
            "qubit_cap": None if qubit_cap is None else int(qubit_cap),
            "pool_term_count": pool_count,
            "pool_term_cap": None if pool_cap is None else int(pool_cap),
        }
    if pool_cap is not None and pool_count > int(pool_cap):
        return {
            "resource_guard": True,
            "resource_guard_kind": "generic_adapt_variant_pool_term_cap",
            "reason": f"{pool_name} exceeds cap",
            "num_qubits": num_qubits,
            "qubit_cap": None if qubit_cap is None else int(qubit_cap),
            "pool_term_count": pool_count,
            "pool_term_cap": int(pool_cap),
        }
    return None


def _compile_pool(
    pool: Sequence[_PoolCandidate],
    *,
    pauli_action_cache: dict[str, Any],
) -> tuple[_CompiledCandidate, ...]:
    return tuple(
        _CompiledCandidate(
            candidate=candidate,
            compiled=compile_polynomial_action(
                candidate.polynomial,
                tol=1e-12,
                pauli_action_cache=pauli_action_cache,
            ),
        )
        for candidate in pool
    )


def _prepare_selected_state(
    *,
    selected: Sequence[_PoolCandidate],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    pauli_action_cache: dict[str, Any],
    parameterization_mode: str = "logical_shared",
    parameterization_layout: AnsatzParameterLayout | None = None,
) -> np.ndarray:
    if not selected:
        return np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
    executor = CompiledAnsatzExecutor(
        list(selected),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode="per_pauli_term" if str(parameterization_mode).startswith("per_pauli") else "logical_shared",
        parameterization_layout=parameterization_layout,
    )
    return np.asarray(executor.prepare_state(np.asarray(theta, dtype=float).reshape(-1), psi_ref), dtype=complex).reshape(-1)


def _dense_exact_state_fidelity_for_selected(
    *,
    selected: Sequence[_PoolCandidate],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any],
    exact_energy: float | None,
    parameterization_mode: str = "logical_shared",
    parameterization_layout: AnsatzParameterLayout | None = None,
    max_qubits: int = 12,
) -> dict[str, Any]:
    nq = int(h_compiled.nq)
    if nq > int(max_qubits):
        return {
            "infidelity_exact": None,
            "exact_state_fidelity": None,
            "infidelity_status": "not_available_dense_diagonalization_qubit_cap",
            "exact_state_fidelity_source": "dense_diagonalization_skipped",
            "exact_state_fidelity_qubit_cap": int(max_qubits),
        }
    dim = 1 << nq
    hmat = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        basis = np.zeros(dim, dtype=complex)
        basis[col] = 1.0
        hmat[:, col] = apply_compiled_polynomial(basis, h_compiled)
    hmat = 0.5 * (hmat + hmat.conj().T)
    evals, evecs = np.linalg.eigh(hmat)
    if exact_energy is None or not math.isfinite(float(exact_energy)):
        target_index = 0
    else:
        target_index = int(np.argmin(np.abs(evals - float(exact_energy))))
    psi_exact = np.asarray(evecs[:, target_index], dtype=complex).reshape(-1)
    psi_adapt = _prepare_selected_state(
        selected=selected,
        theta=np.asarray(theta, dtype=float).reshape(-1),
        psi_ref=psi_ref,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode=parameterization_mode,
        parameterization_layout=parameterization_layout,
    )
    psi_adapt = psi_adapt / max(float(np.linalg.norm(psi_adapt)), 1.0e-300)
    fidelity = float(abs(np.vdot(psi_exact, psi_adapt)) ** 2)
    fidelity = min(1.0, max(0.0, fidelity))
    return {
        "infidelity_exact": float(max(0.0, 1.0 - fidelity)),
        "exact_state_fidelity": float(fidelity),
        "infidelity_status": "computed_dense_diagonalization",
        "exact_state_fidelity_source": "dense_diagonalization_selected_eigenvector",
        "exact_state_fidelity_energy": float(evals[target_index]),
        "exact_state_fidelity_target_energy": None if exact_energy is None else float(exact_energy),
        "exact_state_fidelity_energy_abs_delta": None
        if exact_energy is None
        else float(abs(float(evals[target_index]) - float(exact_energy))),
    }


def _optimize_selected(
    *,
    minimize_fn: Callable[..., Any],
    selected: Sequence[_PoolCandidate],
    x0: np.ndarray,
    psi_ref: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any],
    optimizer_maxiter: int,
    optimizer_method: str = "BFGS",
    parameterization_mode: str = "logical_shared",
    parameterization_layout: AnsatzParameterLayout | None = None,
    decision_noise_recorder: BenchmarkDecisionNoiseRecorder | None = None,
    decision_scope: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    method_raw = str(optimizer_method or "BFGS").strip()
    method_key = method_raw.lower()
    uses_rotosolve = False
    if method_key in {"bfgs", "scipy.optimize.minimize:bfgs"}:
        method_name = "BFGS"
        method_surface = "adapt_refit_bfgs_objective"
        method_options: dict[str, Any] = {"maxiter": int(optimizer_maxiter), "gtol": 1e-8}
    elif method_key in {"powell", "scipy.optimize.minimize:powell"}:
        method_name = "Powell"
        method_surface = "adapt_refit_powell_objective"
        method_options = {"maxiter": int(optimizer_maxiter), "xtol": 1e-5, "ftol": 1e-12}
    elif method_key in {"rotosolve", "coordinate_descent", "repo_coordinate_descent:rotosolve_coordinate_descent"}:
        method_name = "ROTOSOLVE"
        method_surface = "adapt_refit_rotosolve_objective"
        method_options = {"maxiter": int(optimizer_maxiter)}
        uses_rotosolve = True
    else:
        raise ValueError(
            f"unsupported optimizer method {optimizer_method!r}; expected BFGS, Powell, or ROTOSOLVE"
        )
    optimizer_label = (
        "repo_coordinate_descent:rotosolve_coordinate_descent"
        if uses_rotosolve
        else f"scipy.optimize.minimize:{method_name}"
    )
    if not selected:
        energy, _ = energy_via_one_apply(np.asarray(psi_ref, dtype=complex).reshape(-1), h_compiled)
        exact_energy = float(energy)
        return (
            np.zeros(0, dtype=float),
            exact_energy,
            {
                "nfev": 1,
                "nit": 0,
                "success": True,
                "message": "empty_ansatz",
                "optimizer": optimizer_label,
                "optimizer_decision_energy": exact_energy,
                "optimizer_exact_energy": exact_energy,
            },
        )

    executor = CompiledAnsatzExecutor(
        list(selected),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode="per_pauli_term" if str(parameterization_mode).startswith("per_pauli") else "logical_shared",
        parameterization_layout=parameterization_layout,
        enable_prefix_state_cache=True,
    )
    if uses_rotosolve:
        stencil = rotosolve_stencil_from_executor(executor)
        if stencil is None:
            allow_unsupported_fallback = os.environ.get(
                GENERIC_STATIC_ALLOW_UNSUPPORTED_ROTOSOLVE_FALLBACK_ENV,
                "",
            ).strip().lower() in {"1", "true", "yes", "on"}
            if not allow_unsupported_fallback:
                raise ValueError(
                    "ROTOSOLVE requires coefficient-aware singleton-Pauli coordinates; "
                    "the selected ansatz contains a logical/macro coordinate with either "
                    "multiple Pauli terms or invalid coefficients. Refusing the historical "
                    "generic_single_frequency_default fallback. Set "
                    f"{GENERIC_STATIC_ALLOW_UNSUPPORTED_ROTOSOLVE_FALLBACK_ENV}=1 only "
                    "for explicitly labelled diagnostic fixed-stencil coordinate-descent runs."
                )
            method_options["period"] = 2.0 * math.pi
            method_options["shift"] = 0.5 * math.pi
            method_options["rotosolve_stencil_source"] = "diagnostic_generic_single_frequency_default"
        else:
            period_values, shift_values = stencil
            method_options["period"] = period_values
            method_options["shift"] = shift_values
            method_options["rotosolve_stencil_source"] = "compiled_executor_single_pauli_coefficients"

    base_scope = dict(decision_scope or {})
    selected_labels = tuple(str(candidate.label) for candidate in selected)

    def exact_energy_for(theta_vec: np.ndarray) -> float:
        psi = np.asarray(
            executor.prepare_state(np.asarray(theta_vec, dtype=float).reshape(-1), psi_ref),
            dtype=complex,
        ).reshape(-1)
        energy_val, _ = energy_via_one_apply(psi, h_compiled)
        return float(energy_val)

    eval_count = 0

    def objective(theta_vec: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        exact_energy = exact_energy_for(theta_vec)
        return _decision_value(
            decision_noise_recorder,
            exact_energy,
            surface=method_surface,
            value_kind="energy",
            phase="optimizer",
            extra_scope={
                **base_scope,
                "eval_index": int(eval_count),
                "selected_operator_count": int(len(selected)),
                "selected_labels": selected_labels,
            },
        )

    theta0 = np.asarray(x0, dtype=float).reshape(-1)
    expected_size = int(executor.num_parameters)
    if int(theta0.size) != expected_size:
        raise ValueError(f"optimizer theta length mismatch: got {theta0.size}, expected {expected_size}")

    if uses_rotosolve:
        result = rotosolve_coordinate_descent(
            objective,
            theta0,
            maxiter=int(optimizer_maxiter),
            period=method_options["period"],
            shift=method_options["shift"],
        )
    else:
        result = minimize_fn(
            objective,
            theta0,
            method=method_name,
            options=method_options,
        )
    theta = np.asarray(getattr(result, "x", theta0), dtype=float).reshape(-1)
    optimizer_decision_energy = getattr(result, "fun", None)
    if optimizer_decision_energy is None:
        optimizer_decision_energy = objective(theta)
    energy = exact_energy_for(theta)
    nfev = getattr(result, "nfev", None)
    nit = getattr(result, "nit", None)
    status = getattr(result, "status", None)

    def _stencil_values_for_info(key: str) -> list[float] | None:
        if not uses_rotosolve or key not in method_options:
            return None
        return [float(x) for x in np.asarray(method_options[key], dtype=float).reshape(-1).tolist()]

    info = {
        "nfev": int(nfev) if nfev is not None else int(eval_count),
        "nit": int(nit) if nit is not None else None,
        "status": int(status) if status is not None else None,
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "optimizer": optimizer_label,
        "accepted_steps": int(getattr(result, "accepted_steps", 0)) if uses_rotosolve else None,
        "rotosolve_stencil_source": method_options.get("rotosolve_stencil_source") if uses_rotosolve else None,
        "rotosolve_period": _stencil_values_for_info("period"),
        "rotosolve_shift": _stencil_values_for_info("shift"),
        "optimizer_decision_energy": float(optimizer_decision_energy),
        "optimizer_reported_energy": float(optimizer_decision_energy),
        "optimizer_exact_energy": float(energy),
        "optimizer_decision_surface": method_surface
        if decision_noise_recorder is not None and bool(getattr(decision_noise_recorder.config, "enabled", False))
        else None,
        "prefix_state_cache_enabled": bool(executor.enable_prefix_state_cache),
        "prefix_state_cache_id": str(executor.PREFIX_STATE_CACHE_ID),
        "grouped_exact_plan_cache_id": str(executor.GROUPED_EXACT_PLAN_CACHE_ID),
        "prefix_state_cache_evaluation_count": int(executor.prefix_state_cache_evaluation_count),
        "prefix_state_cache_hit_count": int(executor.prefix_state_cache_hit_count),
        "prefix_state_cache_reused_operator_count": int(executor.prefix_state_cache_reused_operator_count),
    }
    return theta, float(energy), info


def _spsa_polish(
    *,
    theta0: np.ndarray,
    energy0: float,
    objective: Callable[[np.ndarray], float],
    rng_seed: int,
    maxiter: int,
    max_abs_step: float,
    accept_tol: float,
    schedule: _SpsaPolishSchedule | None = None,
    decision_noise_recorder: BenchmarkDecisionNoiseRecorder | None = None,
    decision_scope: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    spsa_schedule = schedule or _default_spsa_polish_schedule()
    base_scope = dict(decision_scope or {})
    last_decision_energy = [float(energy0)]

    def _decision_value_for_shared(exact_value: float, eval_kind: str, iteration: int) -> float:
        phase_by_kind = {
            "current": "spsa_current",
            "probe_plus": "spsa_probe_plus",
            "probe_minus": "spsa_probe_minus",
            "candidate": "spsa_candidate",
        }
        noise_eval_kind = {
            "current": "current",
            "probe_plus": "plus",
            "probe_minus": "minus",
            "candidate": "candidate",
        }.get(str(eval_kind), str(eval_kind))
        value = _decision_value(
            decision_noise_recorder,
            float(exact_value),
            surface="adapt_refit_spsa_objective",
            value_kind="energy",
            phase=phase_by_kind.get(str(eval_kind), f"spsa_{eval_kind}"),
            extra_scope={
                **base_scope,
                "spsa_iteration": int(iteration),
                "spsa_eval_kind": noise_eval_kind,
            },
        )
        last_decision_energy[0] = float(value)
        return float(value)

    result = spsa_energy_descent_minimize(
        objective,
        np.asarray(theta0, dtype=float).reshape(-1),
        maxiter=int(maxiter),
        seed=int(rng_seed),
        initial_fun=float(energy0),
        schedule=spsa_schedule,
        max_abs_step=float(max_abs_step),
        accept_tol=float(accept_tol),
        bounds=None,
        project="none",
        decision_value=_decision_value_for_shared,
    )
    theta = np.asarray(result.x, dtype=float).reshape(-1)
    current_energy = float(objective(theta))
    memory = result.optimizer_memory if isinstance(result.optimizer_memory, Mapping) else {}
    return (
        theta,
        current_energy,
        {
            "success": bool(result.success),
            "message": str(result.message),
            "seed": int(rng_seed),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "accepted_step_count": int(memory.get("accepted_step_count") or 0),
            "energy_decrease_total": float(memory.get("energy_decrease_total") or 0.0),
            "energy_before": float(energy0),
            "energy_after": current_energy,
            "optimizer_decision_energy": float(result.fun if math.isfinite(float(result.fun)) else last_decision_energy[0]),
            "optimizer_exact_energy": float(current_energy),
            "optimizer_decision_surface": "adapt_refit_spsa_objective"
            if decision_noise_recorder is not None and bool(getattr(decision_noise_recorder.config, "enabled", False))
            else None,
            "spsa_a": float(spsa_schedule.a),
            "spsa_c": float(spsa_schedule.c),
            "spsa_alpha": float(spsa_schedule.alpha),
            "spsa_gamma": float(spsa_schedule.gamma),
            "spsa_A": float(spsa_schedule.big_a),
            "spsa_big_a": float(spsa_schedule.big_a),
        },
    )


def _adapt_spsa_refit_engine_label() -> str:
    return resolve_adapt_spsa_refit_engine_label(
        env_names=(
            GENERIC_STATIC_ADAPT_SPSA_REFIT_ENGINE_ENV,
            "ADAPT_SPSA_REFIT_ENGINE",
        )
    )


def _optimize_selected_spsa(
    *,
    selected: Sequence[_PoolCandidate],
    x0: np.ndarray,
    psi_ref: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any],
    optimizer_maxiter: int,
    spsa_seed: int,
    spsa_schedule: _SpsaPolishSchedule | None = None,
    parameterization_mode: str = "logical_shared",
    parameterization_layout: AnsatzParameterLayout | None = None,
    decision_noise_recorder: BenchmarkDecisionNoiseRecorder | None = None,
    decision_scope: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    optimizer_name = _adapt_spsa_refit_engine_label()
    schedule = spsa_schedule or _default_spsa_polish_schedule()
    if not selected:
        energy, _ = energy_via_one_apply(np.asarray(psi_ref, dtype=complex).reshape(-1), h_compiled)
        exact_energy = float(energy)
        return (
            np.zeros(0, dtype=float),
            exact_energy,
            {
                "nfev": 1,
                "nit": 0,
                "success": True,
                "message": "empty_ansatz",
                "optimizer": optimizer_name,
                "optimizer_decision_energy": exact_energy,
                "optimizer_reported_energy": exact_energy,
                "optimizer_exact_energy": exact_energy,
                "spsa_refit_engine": optimizer_name,
                "spsa_return_policy": "empty_ansatz",
                "spsa_seed": int(spsa_seed),
                "spsa_accepted_step_count": None,
                "spsa_energy_decrease_total": 0.0,
                "spsa_energy_before": exact_energy,
                "spsa_energy_after": exact_energy,
                "spsa_a": float(schedule.a),
                "spsa_c": float(schedule.c),
                "spsa_alpha": float(schedule.alpha),
                "spsa_gamma": float(schedule.gamma),
                "spsa_A": float(schedule.big_a),
                "spsa_big_a": float(schedule.big_a),
            },
        )

    executor = CompiledAnsatzExecutor(
        list(selected),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode="per_pauli_term" if str(parameterization_mode).startswith("per_pauli") else "logical_shared",
        parameterization_layout=parameterization_layout,
    )

    def objective(theta_vec: np.ndarray) -> float:
        psi = np.asarray(
            executor.prepare_state(np.asarray(theta_vec, dtype=float).reshape(-1), psi_ref),
            dtype=complex,
        ).reshape(-1)
        energy_val, _ = energy_via_one_apply(psi, h_compiled)
        return float(energy_val)

    theta0 = np.asarray(x0, dtype=float).reshape(-1)
    expected_size = int(executor.num_parameters)
    if int(theta0.size) != expected_size:
        raise ValueError(f"SPSA theta length mismatch: got {theta0.size}, expected {expected_size}")
    energy_cache: dict[bytes, float] = {}

    def _cache_key(theta_vec: np.ndarray) -> bytes:
        return np.asarray(theta_vec, dtype=float).reshape(-1).tobytes()

    def exact_objective(theta_vec: np.ndarray) -> float:
        key = _cache_key(theta_vec)
        if key not in energy_cache:
            energy_cache[key] = float(objective(theta_vec))
        return float(energy_cache[key])

    base_scope = {
        **dict(decision_scope or {}),
        "optimizer": "spsa",
        "selected_operator_count": int(len(selected)),
        "selected_labels": tuple(str(candidate.label) for candidate in selected),
    }
    decision_eval_counter = [0]

    def decision_objective(theta_vec: np.ndarray) -> float:
        exact_energy = exact_objective(theta_vec)
        if decision_noise_recorder is None or not bool(getattr(decision_noise_recorder.config, "enabled", False)):
            return float(exact_energy)
        decision_eval_counter[0] += 1
        return _decision_value(
            decision_noise_recorder,
            exact_energy,
            surface="adapt_refit_spsa_objective",
            value_kind="energy",
            phase="spsa_minimize_eval",
            extra_scope={**base_scope, "spsa_eval_index": int(decision_eval_counter[0])},
        )

    energy0 = exact_objective(theta0)
    if int(optimizer_maxiter) < 1:
        return (
            theta0,
            float(energy0),
            {
                "nfev": 0,
                "nit": 0,
                "success": False,
                "message": "spsa_minimize_no_budget",
                "optimizer": optimizer_name,
                "optimizer_decision_energy": float(energy0),
                "optimizer_reported_energy": float(energy0),
                "optimizer_exact_energy": float(energy0),
                "spsa_refit_engine": optimizer_name,
                "spsa_return_policy": "best_observed_with_x0_seed_avg_last_0",
                "spsa_seed": int(spsa_seed),
                "spsa_accepted_step_count": None,
                "spsa_energy_decrease_total": 0.0,
                "spsa_energy_before": float(energy0),
                "spsa_energy_after": float(energy0),
                "spsa_a": float(schedule.a),
                "spsa_c": float(schedule.c),
                "spsa_alpha": float(schedule.alpha),
                "spsa_gamma": float(schedule.gamma),
                "spsa_A": float(schedule.big_a),
                "spsa_big_a": float(schedule.big_a),
            },
        )

    if optimizer_name == _LEGACY_SPSA_POLISH_OPTIMIZER_LABEL:
        theta, energy, legacy_info = _spsa_polish(
            theta0=theta0,
            energy0=float(energy0),
            objective=exact_objective,
            rng_seed=int(spsa_seed),
            maxiter=int(optimizer_maxiter),
            max_abs_step=float(_GEO_QNGD_MAX_ABS_STEP),
            accept_tol=float(_GEO_SPSA_ACCEPT_TOL),
            schedule=schedule,
            decision_noise_recorder=decision_noise_recorder,
            decision_scope=base_scope,
        )
        decision_energy = float(legacy_info.get("optimizer_decision_energy", energy))
        return (
            np.asarray(theta, dtype=float).reshape(-1),
            float(energy),
            {
                "nfev": int(legacy_info.get("nfev") or 0),
                "nit": int(legacy_info.get("nit") or 0),
                "success": bool(legacy_info.get("success", False)),
                "message": str(legacy_info.get("message", "")),
                "optimizer": optimizer_name,
                "optimizer_decision_energy": decision_energy,
                "optimizer_reported_energy": decision_energy,
                "optimizer_exact_energy": float(energy),
                "optimizer_decision_surface": legacy_info.get("optimizer_decision_surface"),
                "spsa_refit_engine": optimizer_name,
                "spsa_return_policy": "legacy_energy_descent_stop_after_stationary",
                "spsa_seed": int(spsa_seed),
                "spsa_accepted_step_count": int(legacy_info.get("accepted_step_count") or 0),
                "spsa_energy_decrease_total": float(legacy_info.get("energy_decrease_total") or 0.0),
                "spsa_energy_before": float(energy0),
                "spsa_energy_after": float(energy),
                "spsa_a": float(schedule.a),
                "spsa_c": float(schedule.c),
                "spsa_alpha": float(schedule.alpha),
                "spsa_gamma": float(schedule.gamma),
                "spsa_A": float(schedule.big_a),
                "spsa_big_a": float(schedule.big_a),
            },
        )

    result = spsa_minimize(
        decision_objective,
        theta0,
        maxiter=int(optimizer_maxiter),
        seed=int(spsa_seed),
        a=float(schedule.a),
        c=float(schedule.c),
        alpha=float(schedule.alpha),
        gamma=float(schedule.gamma),
        A=float(schedule.big_a),
        bounds=None,
        project="none",
        eval_repeats=1,
        eval_agg="mean",
        avg_last=0,
    )
    theta = np.asarray(result.x, dtype=float).reshape(-1)
    energy = exact_objective(theta)
    decision_energy = float(result.fun)
    history_tail = [dict(item) for item in list(result.history)[-32:]]
    return (
        theta,
        float(energy),
        {
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "success": bool(result.success),
            "message": str(result.message),
            "optimizer": optimizer_name,
            "optimizer_decision_energy": decision_energy,
            "optimizer_reported_energy": decision_energy,
            "optimizer_exact_energy": float(energy),
            "optimizer_decision_surface": "adapt_refit_spsa_objective"
            if decision_noise_recorder is not None and bool(getattr(decision_noise_recorder.config, "enabled", False))
            else None,
            "spsa_refit_engine": optimizer_name,
            "spsa_return_policy": "best_observed_with_x0_seed_avg_last_0",
            "spsa_optimizer_memory": result.optimizer_memory,
            "spsa_history_tail": history_tail,
            "spsa_seed": int(spsa_seed),
            "spsa_accepted_step_count": None,
            "spsa_energy_decrease_total": max(0.0, float(energy0) - float(energy)),
            "spsa_energy_before": float(energy0),
            "spsa_energy_after": float(energy),
            "spsa_a": float(schedule.a),
            "spsa_c": float(schedule.c),
            "spsa_alpha": float(schedule.alpha),
            "spsa_gamma": float(schedule.gamma),
            "spsa_A": float(schedule.big_a),
            "spsa_big_a": float(schedule.big_a),
        },
    )


def _optimize_selected_qnspsa(
    *,
    selected: Sequence[_PoolCandidate],
    x0: np.ndarray,
    psi_ref: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any],
    optimizer_maxiter: int,
    qnspsa_seed: int,
    spsa_schedule: _SpsaPolishSchedule | None = None,
    parameterization_mode: str = "logical_shared",
    parameterization_layout: AnsatzParameterLayout | None = None,
    decision_noise_recorder: BenchmarkDecisionNoiseRecorder | None = None,
    decision_scope: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    optimizer_name = "exact_bench_qnspsa:energy_fubini_study_metric"
    schedule = spsa_schedule or _default_spsa_polish_schedule()
    if not selected:
        energy, _ = energy_via_one_apply(np.asarray(psi_ref, dtype=complex).reshape(-1), h_compiled)
        exact_energy = float(energy)
        return (
            np.zeros(0, dtype=float),
            exact_energy,
            {
                "nfev": 1,
                "nit": 0,
                "success": True,
                "message": "empty_ansatz",
                "optimizer": optimizer_name,
                "optimizer_decision_energy": exact_energy,
                "optimizer_reported_energy": exact_energy,
                "optimizer_exact_energy": exact_energy,
                "qnspsa_seed": int(qnspsa_seed),
                "qnspsa_objective_nfev": 1,
                "qnspsa_fidelity_nfev": 0,
                "qnspsa_history_tail": [],
            },
        )

    executor = CompiledAnsatzExecutor(
        list(selected),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode="per_pauli_term" if str(parameterization_mode).startswith("per_pauli") else "logical_shared",
        parameterization_layout=parameterization_layout,
    )
    theta0 = np.asarray(x0, dtype=float).reshape(-1)
    expected_size = int(executor.num_parameters)
    if int(theta0.size) != expected_size:
        raise ValueError(f"QN-SPSA theta length mismatch: got {theta0.size}, expected {expected_size}")
    base_scope = dict(decision_scope or {})
    selected_labels = tuple(str(candidate.label) for candidate in selected)

    def state_for(theta_vec: np.ndarray) -> np.ndarray:
        return np.asarray(
            executor.prepare_state(np.asarray(theta_vec, dtype=float).reshape(-1), psi_ref),
            dtype=complex,
        ).reshape(-1)

    def exact_energy_for(theta_vec: np.ndarray) -> float:
        energy_val, _ = energy_via_one_apply(state_for(theta_vec), h_compiled)
        return float(energy_val)

    eval_count = 0

    def objective(theta_vec: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        exact_energy = exact_energy_for(theta_vec)
        return _decision_value(
            decision_noise_recorder,
            exact_energy,
            surface="adapt_refit_qnspsa_objective",
            value_kind="energy",
            phase="optimizer_qnspsa",
            extra_scope={
                **base_scope,
                "eval_index": int(eval_count),
                "selected_operator_count": int(len(selected)),
                "selected_labels": selected_labels,
            },
        )

    def fidelity(theta_left: np.ndarray, theta_right: np.ndarray) -> float:
        psi_left = state_for(theta_left)
        psi_right = state_for(theta_right)
        norm_left = float(np.vdot(psi_left, psi_left).real)
        norm_right = float(np.vdot(psi_right, psi_right).real)
        denom = max(norm_left * norm_right, 1e-300)
        overlap = abs(complex(np.vdot(psi_left, psi_right))) ** 2 / denom
        return float(min(1.0, max(0.0, overlap)))

    result = qnspsa_minimize(
        objective,
        fidelity,
        theta0,
        maxiter=max(0, int(optimizer_maxiter)),
        seed=int(qnspsa_seed),
        a=float(schedule.a),
        c=float(schedule.c),
        alpha=float(schedule.alpha),
        gamma=float(schedule.gamma),
        A=float(schedule.big_a),
        avg_last=0,
        resamplings=1,
        regularization=max(1e-8, float(_DEFAULT_METRIC_FLOOR)),
        psd_floor=max(1e-10, float(_DEFAULT_METRIC_FLOOR) * 1e-2),
    )
    theta = np.asarray(result.x, dtype=float).reshape(-1)
    energy = exact_energy_for(theta)
    return (
        theta,
        float(energy),
        {
            "nfev": int(result.nfev) + 1,
            "nit": int(result.nit),
            "success": bool(result.success),
            "message": str(result.message),
            "optimizer": optimizer_name,
            "optimizer_decision_energy": float(result.fun),
            "optimizer_reported_energy": float(result.fun),
            "optimizer_exact_energy": float(energy),
            "optimizer_decision_surface": "adapt_refit_qnspsa_objective"
            if decision_noise_recorder is not None and bool(getattr(decision_noise_recorder.config, "enabled", False))
            else None,
            "qnspsa_seed": int(qnspsa_seed),
            "qnspsa_objective_nfev": int(result.objective_nfev),
            "qnspsa_fidelity_nfev": int(result.fidelity_nfev),
            "qnspsa_a": float(schedule.a),
            "qnspsa_c": float(schedule.c),
            "qnspsa_alpha": float(schedule.alpha),
            "qnspsa_gamma": float(schedule.gamma),
            "qnspsa_A": float(schedule.big_a),
            "qnspsa_big_a": float(schedule.big_a),
            "qnspsa_history_tail": [dict(item) for item in list(result.history)[-16:]],
        },
    )


def _qngd_no_fallback_telemetry() -> dict[str, Any]:
    return {
        "qngd_fallback_optimizer": None,
        "qngd_spsa_polish_attempted": False,
        "qngd_spsa_polish_success": False,
        "qngd_spsa_polish_message": None,
        "qngd_spsa_polish_seed": None,
        "qngd_spsa_polish_nfev": 0,
        "qngd_spsa_polish_nit": 0,
        "qngd_spsa_polish_accepted_step_count": 0,
        "qngd_spsa_polish_energy_before": None,
        "qngd_spsa_polish_energy_after": None,
        "qngd_spsa_polish_energy_decrease_total": 0.0,
        "qngd_bfgs_polish_attempted": False,
        "qngd_bfgs_polish_success": False,
        "qngd_bfgs_polish_message": None,
        "qngd_bfgs_polish_nfev": 0,
        "qngd_bfgs_polish_nit": None,
        "qngd_bfgs_polish_energy_before": None,
        "qngd_bfgs_polish_energy_after": None,
    }


def _optimize_selected_qngd(
    *,
    selected: Sequence[_PoolCandidate],
    x0: np.ndarray,
    psi_ref: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any],
    optimizer_maxiter: int,
    metric_floor: float,
    spsa_seed: int,
    decision_noise_recorder: BenchmarkDecisionNoiseRecorder | None = None,
    decision_scope: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    optimizer_name = "exact_bench_qngd:logical_shared_metric_backtracking"
    if not selected:
        energy, _ = energy_via_one_apply(np.asarray(psi_ref, dtype=complex).reshape(-1), h_compiled)
        return (
            np.zeros(0, dtype=float),
            float(energy),
            {
                "nfev": 1,
                "nit": 0,
                "success": True,
                "message": "empty_ansatz",
                "optimizer": optimizer_name,
                "qngd_metric_rank_last": 0,
                "qngd_metric_condition_last": None,
                "qngd_step_fs_norm_last": 0.0,
                "qngd_step_l2_norm_last": 0.0,
                "qngd_max_abs_step_last": 0.0,
                "qngd_line_search_backtracks_total": 0,
                "qngd_accepted_step_count": 0,
                "qngd_energy_decrease_total": 0.0,
                "qngd_metric_eval_count": 0,
                "qngd_metric_operator_probe_count_total": 0,
                "qngd_gradient_operator_probe_count_total": 0,
                "optimizer_decision_energy": float(energy),
                "optimizer_exact_energy": float(energy),
                **_qngd_no_fallback_telemetry(),
            },
        )

    base_scope = dict(decision_scope or {})
    selected_labels = tuple(str(candidate.label) for candidate in selected)
    executor = CompiledAnsatzExecutor(
        list(selected),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode="per_pauli_term",
    )
    theta = np.asarray(x0, dtype=float).reshape(-1).copy()
    if int(theta.size) != int(len(selected)):
        raise ValueError(f"QNGD logical theta length mismatch: got {theta.size}, expected {len(selected)}")

    nfev = 0
    nit = 0
    accepted_step_count = 0
    energy_decrease_total = 0.0
    qngd_metric_eval_count = 0
    qngd_metric_probe_count_total = 0
    qngd_gradient_probe_count_total = 0
    backtracks_total = 0
    last_rank = 0
    last_condition: float | None = None
    last_fs_norm = 0.0
    last_l2_norm = 0.0
    last_max_abs_step = 0.0
    message = "qngd_maxiter_reached"
    success = False
    step_threshold = max(1e-10, float(metric_floor))

    def _runtime_theta(theta_logical: np.ndarray) -> np.ndarray:
        return expand_legacy_logical_theta(theta_logical, executor.layout)

    def _energy_for(theta_logical: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        nonlocal nfev
        psi = np.asarray(executor.prepare_state(_runtime_theta(theta_logical), psi_ref), dtype=complex).reshape(-1)
        energy_val, hpsi_val = energy_via_one_apply(psi, h_compiled)
        nfev += 1
        return float(energy_val), psi, np.asarray(hpsi_val, dtype=complex).reshape(-1)

    def _state_and_logical_tangents(theta_logical: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        runtime_theta = _runtime_theta(theta_logical)
        psi_val, runtime_tangents = executor.prepare_state_with_runtime_tangents(runtime_theta, psi_ref)
        psi_arr = np.asarray(psi_val, dtype=complex).reshape(-1)
        logical_tangents: list[np.ndarray] = []
        for block in executor.layout.blocks:
            tangent = np.zeros_like(psi_arr, dtype=complex)
            for runtime_idx in range(int(block.runtime_start), int(block.runtime_stop)):
                tangent += np.asarray(runtime_tangents.get(int(runtime_idx), 0.0), dtype=complex).reshape(-1)
            tangent = tangent - psi_arr * complex(np.vdot(psi_arr, tangent))
            logical_tangents.append(tangent)
        return psi_arr, logical_tangents

    current_energy, _current_psi, _current_hpsi = _energy_for(theta)
    current_decision_energy = _decision_value(
        decision_noise_recorder,
        current_energy,
        surface="adapt_refit_qngd_objective",
        value_kind="energy",
        phase="optimizer_qngd_initial",
        extra_scope={
            **base_scope,
            "qngd_iteration": -1,
            "selected_operator_count": int(len(selected)),
            "selected_labels": selected_labels,
        },
    )
    maxiter = max(0, int(optimizer_maxiter))
    if maxiter <= 0:
        return (
            theta,
            float(current_energy),
            {
                "nfev": int(nfev),
                "nit": 0,
                "success": True,
                "message": "qngd_maxiter_zero",
                "optimizer": optimizer_name,
                "qngd_metric_rank_last": 0,
                "qngd_metric_condition_last": None,
                "qngd_step_fs_norm_last": 0.0,
                "qngd_step_l2_norm_last": 0.0,
                "qngd_max_abs_step_last": 0.0,
                "qngd_line_search_backtracks_total": 0,
                "qngd_accepted_step_count": 0,
                "qngd_energy_decrease_total": 0.0,
                "qngd_metric_eval_count": 0,
                "qngd_metric_operator_probe_count_total": 0,
                "qngd_gradient_operator_probe_count_total": 0,
                "optimizer_decision_energy": float(current_decision_energy),
                "optimizer_exact_energy": float(current_energy),
                "optimizer_decision_surface": "adapt_refit_qngd_objective"
                if decision_noise_recorder is not None and bool(getattr(decision_noise_recorder.config, "enabled", False))
                else None,
                **_qngd_no_fallback_telemetry(),
            },
        )

    for _iteration in range(maxiter):
        psi_current, tangents = _state_and_logical_tangents(theta)
        energy_current, hpsi_current = energy_via_one_apply(psi_current, h_compiled)
        nfev += 1
        current_energy = float(energy_current)
        current_decision_energy = _decision_value(
            decision_noise_recorder,
            current_energy,
            surface="adapt_refit_qngd_objective",
            value_kind="energy",
            phase="optimizer_qngd_current",
            extra_scope={
                **base_scope,
                "qngd_iteration": int(_iteration),
                "selected_operator_count": int(len(selected)),
                "selected_labels": selected_labels,
            },
        )
        residual = np.asarray(hpsi_current, dtype=complex).reshape(-1) - current_energy * psi_current
        diag = _geo_metric_diagnostics(tangents, metric_floor=float(metric_floor))
        metric = np.asarray(diag["metric"], dtype=float)
        qngd_metric_eval_count += 1
        qngd_metric_probe_count_total += int(len(tangents) * (len(tangents) + 1) // 2)
        qngd_gradient_probe_count_total += int(len(tangents))
        regularization = float(diag["regularization"])
        force = np.asarray([-2.0 * float(np.real(np.vdot(tangent, residual))) for tangent in tangents], dtype=float)
        pinv_rcond = float(max(1e-10, float(metric_floor)))
        step = np.linalg.pinv(metric, rcond=pinv_rcond) @ force
        step = np.asarray(step, dtype=float).reshape(-1)
        last_rank = int(diag["rank"])
        last_condition = diag["condition"]
        last_fs_norm = float(math.sqrt(max(0.0, float(step @ metric @ step)))) if step.size else 0.0
        last_l2_norm = float(np.linalg.norm(step)) if step.size else 0.0
        last_max_abs_step = float(np.max(np.abs(step))) if step.size else 0.0
        if last_fs_norm < step_threshold:
            message = "qngd_natural_step_threshold"
            success = True
            break
        if last_max_abs_step > _GEO_QNGD_MAX_ABS_STEP:
            step = step * float(_GEO_QNGD_MAX_ABS_STEP / last_max_abs_step)
            last_max_abs_step = _GEO_QNGD_MAX_ABS_STEP
            last_l2_norm = float(np.linalg.norm(step))
            last_fs_norm = float(math.sqrt(max(0.0, float(step @ metric @ step)))) if step.size else 0.0

        accepted = False
        trial_energy = current_energy
        trial_decision_energy = float(current_decision_energy)
        trial_theta = theta
        alpha = 1.0
        for backtrack in range(_GEO_QNGD_MAX_BACKTRACKS + 1):
            candidate_theta = theta + alpha * step
            candidate_energy, _candidate_psi, _candidate_hpsi = _energy_for(candidate_theta)
            candidate_decision_energy = _decision_value(
                decision_noise_recorder,
                candidate_energy,
                surface="adapt_refit_qngd_objective",
                value_kind="energy",
                phase="optimizer_qngd_line_search",
                extra_scope={
                    **base_scope,
                    "qngd_iteration": int(_iteration),
                    "line_search_backtrack": int(backtrack),
                    "selected_operator_count": int(len(selected)),
                    "selected_labels": selected_labels,
                },
            )
            if candidate_decision_energy <= current_decision_energy - 1e-12:
                trial_theta = candidate_theta
                trial_energy = float(candidate_energy)
                trial_decision_energy = float(candidate_decision_energy)
                accepted = True
                backtracks_total += int(backtrack)
                break
            alpha *= 0.5
        if not accepted:
            backtracks_total += int(_GEO_QNGD_MAX_BACKTRACKS + 1)
            if accepted_step_count > 0:
                message = "qngd_stationary_line_search_exhausted"
                success = True
            else:
                message = "qngd_line_search_failed"
                success = False
            break
        theta = np.asarray(trial_theta, dtype=float).reshape(-1)
        nit += 1
        accepted_step_count += 1
        energy_decrease_total += max(0.0, float(current_energy) - float(trial_energy))
        if abs(current_energy - trial_energy) < 1e-12:
            current_energy = float(trial_energy)
            current_decision_energy = float(trial_decision_energy)
            message = "qngd_energy_decrease_threshold"
            success = True
            break
        current_energy = float(trial_energy)
        current_decision_energy = float(trial_decision_energy)
    else:
        success = False
        message = "qngd_maxiter_reached"

    spsa_attempted = False
    spsa_success = False
    spsa_message: str | None = None
    spsa_nfev = 0
    spsa_nit = 0
    spsa_accepted_step_count = 0
    spsa_energy_before: float | None = None
    spsa_energy_after: float | None = None
    spsa_energy_decrease_total = 0.0
    if not bool(success):
        spsa_executor = CompiledAnsatzExecutor(
            list(selected),
            coefficient_tolerance=1e-12,
            ignore_identity=True,
            sort_terms=True,
            pauli_action_cache=pauli_action_cache,
        )

        def _polish_objective(theta_vec: np.ndarray) -> float:
            psi_val = np.asarray(
                spsa_executor.prepare_state(np.asarray(theta_vec, dtype=float).reshape(-1), psi_ref),
                dtype=complex,
            ).reshape(-1)
            energy_val, _ = energy_via_one_apply(psi_val, h_compiled)
            return float(energy_val)

        spsa_attempted = True
        spsa_energy_before = float(current_energy)
        spsa_theta, spsa_energy, spsa_info = _spsa_polish(
            theta0=np.asarray(theta, dtype=float).reshape(-1),
            energy0=float(current_energy),
            objective=_polish_objective,
            rng_seed=int(spsa_seed),
            maxiter=min(_GEO_SPSA_POLISH_MAXITER, max(0, int(optimizer_maxiter))),
            max_abs_step=float(_GEO_QNGD_MAX_ABS_STEP),
            accept_tol=float(_GEO_SPSA_ACCEPT_TOL),
            decision_noise_recorder=decision_noise_recorder,
            decision_scope={
                **base_scope,
                "fallback": "spsa",
                "selected_operator_count": int(len(selected)),
                "selected_labels": selected_labels,
            },
        )
        spsa_nfev = int(spsa_info.get("nfev") or 0)
        spsa_nit = int(spsa_info.get("nit") or 0)
        spsa_accepted_step_count = int(spsa_info.get("accepted_step_count") or 0)
        spsa_energy_decrease_total = float(spsa_info.get("energy_decrease_total") or 0.0)
        spsa_message = str(spsa_info.get("message", ""))
        spsa_energy_after = float(spsa_energy)
        nfev += int(spsa_nfev)
        nit += int(spsa_nit)
        if bool(spsa_info.get("success", False)) and float(spsa_energy) <= float(current_energy) - _GEO_SPSA_ACCEPT_TOL:
            theta = np.asarray(spsa_theta, dtype=float).reshape(-1)
            current_energy = float(spsa_energy)
            current_decision_energy = float(spsa_info.get("optimizer_decision_energy", current_decision_energy))
            spsa_success = True
            success = True
            message = f"{message}_spsa_polish_success"
        else:
            message = f"{message}_spsa_polish_failed"

    return (
        theta,
        float(current_energy),
        {
            "nfev": int(nfev),
            "nit": int(nit),
            "success": bool(success),
            "message": message,
            "optimizer": optimizer_name if not spsa_success else "exact_bench_qngd:logical_shared_metric_backtracking+spsa_polish",
            "qngd_metric_rank_last": int(last_rank),
            "qngd_metric_condition_last": last_condition,
            "qngd_step_fs_norm_last": float(last_fs_norm),
            "qngd_step_l2_norm_last": float(last_l2_norm),
            "qngd_max_abs_step_last": float(last_max_abs_step),
            "qngd_line_search_backtracks_total": int(backtracks_total),
            "qngd_accepted_step_count": int(accepted_step_count),
            "qngd_energy_decrease_total": float(energy_decrease_total),
            "qngd_metric_eval_count": int(qngd_metric_eval_count),
            "qngd_metric_operator_probe_count_total": int(qngd_metric_probe_count_total),
            "qngd_gradient_operator_probe_count_total": int(qngd_gradient_probe_count_total),
            "optimizer_decision_energy": float(current_decision_energy),
            "optimizer_reported_energy": float(current_decision_energy),
            "optimizer_exact_energy": float(current_energy),
            "optimizer_decision_surface": "adapt_refit_qngd_objective"
            if decision_noise_recorder is not None and bool(getattr(decision_noise_recorder.config, "enabled", False))
            else None,
            "qngd_fallback_optimizer": "spsa" if spsa_attempted else None,
            "qngd_spsa_polish_attempted": bool(spsa_attempted),
            "qngd_spsa_polish_success": bool(spsa_success),
            "qngd_spsa_polish_message": spsa_message,
            "qngd_spsa_polish_seed": int(spsa_seed) if spsa_attempted else None,
            "qngd_spsa_polish_nfev": int(spsa_nfev),
            "qngd_spsa_polish_nit": int(spsa_nit),
            "qngd_spsa_polish_accepted_step_count": int(spsa_accepted_step_count),
            "qngd_spsa_polish_energy_before": spsa_energy_before,
            "qngd_spsa_polish_energy_after": spsa_energy_after,
            "qngd_spsa_polish_energy_decrease_total": float(spsa_energy_decrease_total),
            "qngd_bfgs_polish_attempted": False,
            "qngd_bfgs_polish_success": False,
            "qngd_bfgs_polish_message": None,
            "qngd_bfgs_polish_nfev": 0,
            "qngd_bfgs_polish_nit": None,
            "qngd_bfgs_polish_energy_before": None,
            "qngd_bfgs_polish_energy_after": None,
        },
    )


def _project_tangent(psi: np.ndarray, gpsi: np.ndarray) -> np.ndarray:
    psi_arr = np.asarray(psi, dtype=complex).reshape(-1)
    tangent = -1j * np.asarray(gpsi, dtype=complex).reshape(-1)
    return tangent - psi_arr * complex(np.vdot(psi_arr, tangent))


def _geo_metric_diagnostics(tangents: Sequence[np.ndarray], *, metric_floor: float) -> dict[str, Any]:
    m = int(len(tangents))
    if m <= 0:
        return {
            "metric": np.zeros((0, 0), dtype=float),
            "regularization": float(metric_floor),
            "rank": 0,
            "condition": None,
            "offdiag_norm": 0.0,
            "scale": 1.0,
        }
    metric = np.empty((m, m), dtype=float)
    for i, ti in enumerate(tangents):
        for j, tj in enumerate(tangents[: i + 1]):
            val = float(np.real(np.vdot(ti, tj)))
            metric[i, j] = val
            metric[j, i] = val
    diag = np.diag(metric)
    scale = float(max(1.0, float(np.max(np.abs(diag))) if diag.size else 1.0))
    regularization = float(max(metric_floor, metric_floor * scale))
    try:
        rank = int(np.linalg.matrix_rank(metric, tol=max(1e-12, regularization)))
    except Exception:
        rank = 0
    try:
        condition = float(np.linalg.cond(metric + regularization * np.eye(m)))
        if not math.isfinite(condition):
            condition = None
    except Exception:
        condition = None
    offdiag = metric - np.diag(np.diag(metric))
    return {
        "metric": metric,
        "regularization": regularization,
        "rank": rank,
        "condition": condition,
        "offdiag_norm": float(np.linalg.norm(offdiag)),
        "scale": scale,
    }


def _score_candidates(
    *,
    config: _VariantConfig,
    psi: np.ndarray,
    hpsi: np.ndarray,
    compiled_pool: Sequence[_CompiledCandidate],
    selected_labels: set[str],
    previous_selected_label: str | None,
    metric_floor: float,
    decision_noise_recorder: BenchmarkDecisionNoiseRecorder | None = None,
    adapt_iteration: int | None = None,
) -> list[dict[str, Any]]:
    candidates: list[_PoolCandidate] = []
    gpsis: list[np.ndarray] = []
    gradients: list[float] = []
    blocked_labels = _blocked_labels_for_config(
        config,
        selected_labels=selected_labels,
        previous_selected_label=previous_selected_label,
    )
    for item in compiled_pool:
        candidate = item.candidate
        if candidate.label in blocked_labels:
            continue
        gpsi = apply_compiled_polynomial(psi, item.compiled)
        candidates.append(candidate)
        gpsis.append(np.asarray(gpsi, dtype=complex).reshape(-1))
        gradients.append(float(adapt_commutator_grad_from_hpsi(hpsi, gpsi)))

    if not candidates:
        return []

    geo_screen_diag: dict[str, Any] = {}
    if _is_geo_config(config):
        pre_screen_count = int(len(candidates))
        geo_screen_diag = {
            "geo_metric_candidate_count_before_screen": pre_screen_count,
            "geo_metric_candidate_count_after_screen": int(len(candidates)),
            "geo_metric_prescreen_mode": "full_candidate_set",
        }

    decision_enabled = bool(
        decision_noise_recorder is not None and getattr(decision_noise_recorder.config, "enabled", False)
    )
    selector_scores = [float(abs(grad)) for grad in gradients]
    gradient_decisions: list[float | None] = [None for _ in candidates]
    abs_gradient_decisions: list[float | None] = [None for _ in candidates]
    if decision_enabled and _uses_raw_gradient_stop(config):
        for idx, (candidate, grad) in enumerate(zip(candidates, gradients, strict=False)):
            grad_decision = _decision_value(
                decision_noise_recorder,
                float(grad),
                surface="adapt_selector_gradient",
                value_kind="gradient",
                phase="selector",
                extra_scope={
                    "adapt_iteration": adapt_iteration,
                    "candidate_index": int(idx),
                    "candidate_label": str(candidate.label),
                },
            )
            gradient_decisions[idx] = float(grad_decision)
            abs_gradient_decisions[idx] = float(abs(grad_decision))

    metric_variances: list[float | None] = [None for _ in candidates]
    geo_steps: list[float | None] = [None for _ in candidates]
    geo_forces: list[float | None] = [None for _ in candidates]
    geo_step_decisions: list[float | None] = [None for _ in candidates]
    selector_score_decisions: list[float | None] = [
        (float(value) if value is not None else None) for value in abs_gradient_decisions
    ]
    geo_diag: dict[str, Any] = {}
    if _is_geo_config(config):
        psi_arr = np.asarray(psi, dtype=complex).reshape(-1)
        hpsi_arr = np.asarray(hpsi, dtype=complex).reshape(-1)
        energy = float(np.real(np.vdot(psi_arr, hpsi_arr)))
        residual = hpsi_arr - energy * psi_arr
        tangents = [_project_tangent(psi_arr, gpsi) for gpsi in gpsis]
        diag = _geo_metric_diagnostics(tangents, metric_floor=float(metric_floor))
        metric = np.asarray(diag["metric"], dtype=float)
        regularization = float(diag["regularization"])
        force = np.asarray([-2.0 * float(np.real(np.vdot(tangent, residual))) for tangent in tangents], dtype=float)
        # The full_meta pool is intentionally redundant.  Geo selection is a
        # state-space natural-gradient rule on the tangent span, so use the
        # Moore--Penrose inverse rather than a Tikhonov solve that can bury
        # physically useful directions in the null-space regularizer.
        pinv_rcond = float(max(1e-10, float(metric_floor)))
        step = np.linalg.pinv(metric, rcond=pinv_rcond) @ force
        step_arr = np.asarray(step, dtype=float).reshape(-1)
        rank_deficient = int(diag["rank"]) < int(len(candidates))
        selector_scores = [float(abs(x)) for x in step_arr]
        metric_variances = [float(metric[i, i]) for i in range(len(candidates))]
        geo_steps = [float(x) for x in step_arr]
        geo_forces = [float(x) for x in force.reshape(-1)]
        geo_step_fs_norm = float(math.sqrt(max(0.0, float(step_arr @ metric @ step_arr)))) if step_arr.size else 0.0
        geo_step_l2_norm = float(np.linalg.norm(step_arr)) if step_arr.size else 0.0
        geo_max_abs_step = float(np.max(np.abs(step_arr))) if step_arr.size else 0.0
        geo_diag = {
            "geo_selector_mode": _geo_selector_mode_for_config(config),
            **geo_screen_diag,
            "geo_metric_rank": int(diag["rank"]),
            "geo_metric_condition": diag["condition"],
            "geo_metric_regularization": regularization,
            "geo_metric_regularization_used_in_selector_solve": False,
            "geo_metric_condition_source": "condition_of_metric_plus_diagnostic_floor",
            "geo_metric_pinv_rcond": pinv_rcond,
            "geo_metric_solve_kind": "moore_penrose_pseudoinverse",
            "geo_metric_offdiag_norm": float(diag["offdiag_norm"]),
            "geo_metric_rank_deficient": bool(rank_deficient),
            "geo_metric_diagonal_fallback": False,
            "geo_natural_step_fs_norm": geo_step_fs_norm,
            "geo_natural_step_l2_norm": geo_step_l2_norm,
            "geo_max_abs_natural_step": geo_max_abs_step,
        }
        if decision_enabled:
            geo_step_fs_norm_decision = _decision_value(
                decision_noise_recorder,
                geo_step_fs_norm,
                surface="adapt_selector_geo_natural_step_norm",
                value_kind="natural_step_norm",
                phase="selector_stop",
                extra_scope={"adapt_iteration": adapt_iteration, "candidate_count": int(len(candidates))},
            )
            geo_diag["geo_natural_step_fs_norm_decision"] = float(geo_step_fs_norm_decision)
            for idx, (candidate, step_value) in enumerate(zip(candidates, step_arr, strict=False)):
                step_decision = _decision_value(
                    decision_noise_recorder,
                    float(step_value),
                    surface="adapt_selector_geo_natural_step",
                    value_kind="natural_step",
                    phase="selector",
                    extra_scope={
                        "adapt_iteration": adapt_iteration,
                        "candidate_index": int(idx),
                        "candidate_label": str(candidate.label),
                    },
                )
                geo_step_decisions[idx] = float(step_decision)
                selector_score_decisions[idx] = float(abs(step_decision))

    scored: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        row = {
            "label": candidate.label,
            "support": list(candidate.support),
            "pauli_labels_exyz": list(candidate.pauli_labels_exyz),
            "pauli_terms": _candidate_pauli_terms_payload(candidate),
            "execution_mode": str(
                getattr(candidate, "execution_mode", "termwise_product") or "termwise_product"
            ),
            "gradient": float(gradients[idx]),
            "abs_gradient": float(abs(gradients[idx])),
            "metric_variance": metric_variances[idx],
            "selector_score": float(selector_scores[idx]),
        }
        runtime_split_meta = _pool_runtime_split_metadata(candidate)
        if runtime_split_meta is not None:
            row.update(
                {
                    "runtime_split_mode": runtime_split_meta["mode"],
                    "runtime_split_representation": runtime_split_meta["representation"],
                    "runtime_split_parent_label": runtime_split_meta["parent_label"],
                    "runtime_split_child_indices": runtime_split_meta["child_indices"],
                    "runtime_split_child_labels": runtime_split_meta["child_labels"],
                    "runtime_split_symmetry_gate": runtime_split_meta["symmetry_gate"],
                }
            )
        if gradient_decisions[idx] is not None:
            row.update(
                {
                    "gradient_decision": float(gradient_decisions[idx]),
                    "abs_gradient_decision": float(abs_gradient_decisions[idx] or 0.0),
                }
            )
        if selector_score_decisions[idx] is not None:
            row["selector_score_decision"] = float(selector_score_decisions[idx] or 0.0)
        if _is_geo_config(config):
            row.update(geo_diag)
            row.update(
                {
                    "geo_natural_step": geo_steps[idx],
                    "geo_projected_residual_force": geo_forces[idx],
                    "geo_projected_residual_score": float(selector_scores[idx]),
                }
            )
            if geo_step_decisions[idx] is not None:
                row.update(
                    {
                        "geo_natural_step_decision": float(geo_step_decisions[idx] or 0.0),
                        "geo_projected_residual_score_decision": float(selector_score_decisions[idx] or 0.0),
                    }
                )
        scored.append(row)
    scored.sort(
        key=lambda row: (
            -float(row.get("selector_score_decision", row.get("selector_score", 0.0))),
            str(row["label"]),
        )
    )
    return scored


def _batch_admission_gradient_threshold(config: _VariantConfig, gradient_threshold: float) -> float | None:
    """Return the gradient floor used to admit candidates into the selected batch.

    Fixed-horizon comparator rows deliberately pass ``gradient_threshold=0`` to
    disable gradient-based stopping.  That stop-policy choice should not make
    exact zero-gradient candidates eligible as TETRIS batch fillers, but it also
    must not reintroduce the conventional ADAPT convergence threshold as a stop
    condition; nonzero-gradient single-candidate batches remain valid TETRIS
    steps.
    Geo-natural-gradient rows intentionally score by the geometric rule instead
    and therefore do not use a raw-gradient batch-admission floor here.
    """

    if config.stop_rule == "geo_natural_gradient_norm":
        return None
    threshold = float(gradient_threshold)
    if config.variant == "tetris" and threshold <= 0.0:
        return float(_TETRIS_FIXED_HORIZON_NUMERICAL_ZERO_GRADIENT_FLOOR)
    return threshold


def _select_batch(
    *,
    config: _VariantConfig,
    scored: Sequence[Mapping[str, Any]],
    gradient_threshold: float,
    max_tetris_batch_size: int,
) -> list[Mapping[str, Any]]:
    admission_gradient_threshold = _batch_admission_gradient_threshold(config, float(gradient_threshold))
    if admission_gradient_threshold is None:
        eligible = list(scored)
    else:
        eligible = [
            row
            for row in scored
            if float(row.get("abs_gradient_decision", row.get("abs_gradient", 0.0))) >= admission_gradient_threshold
        ]
    if not eligible:
        return []
    if config.variant != "tetris":
        return [eligible[0]]
    batch: list[Mapping[str, Any]] = []
    occupied: set[int] = set()
    for row in eligible:
        support = {int(q) for q in row.get("support", [])}
        if support and support.isdisjoint(occupied):
            batch.append(row)
            occupied.update(support)
        if len(batch) >= int(max_tetris_batch_size):
            break
    return batch


def _candidate_proxy_cost(candidate: _PoolCandidate) -> dict[str, int]:
    runtime_rotation_count = int(len(candidate.pauli_labels_exyz))
    count_2q = 0
    depth_2q = 0
    oneq_basis_proxy = 0
    for label in candidate.pauli_labels_exyz:
        weight = sum(1 for ch in str(label) if ch != "e")
        twoq_chain_depth = max(0, 2 * (int(weight) - 1))
        count_2q += twoq_chain_depth
        depth_2q += twoq_chain_depth
        oneq_basis_proxy += int(weight)
    return {
        "pauli_rotation_proxy": runtime_rotation_count,
        "depth_proxy": int(3 * runtime_rotation_count),
        "count_2q": int(count_2q),
        "depth_2q": int(depth_2q),
        "basis_change_1q_proxy": int(oneq_basis_proxy),
    }


def _compiled_proxy_stats(
    selected: Sequence[_PoolCandidate],
    *,
    selected_batches: Sequence[Sequence[_PoolCandidate]] | None = None,
) -> dict[str, Any]:
    costs = [_candidate_proxy_cost(candidate) for candidate in selected]
    runtime_rotation_count = int(sum(cost["pauli_rotation_proxy"] for cost in costs))
    count_2q = int(sum(cost["count_2q"] for cost in costs))
    sequential_depth_2q = int(sum(cost["depth_2q"] for cost in costs))
    oneq_basis_proxy = int(sum(cost["basis_change_1q_proxy"] for cost in costs))
    sequential_depth = int(sum(cost["depth_proxy"] for cost in costs))

    batches = [list(batch) for batch in (selected_batches or ()) if list(batch)]
    if batches:
        depth = int(
            sum(max(_candidate_proxy_cost(candidate)["depth_proxy"] for candidate in batch) for batch in batches)
        )
        depth_2q = int(
            sum(max(_candidate_proxy_cost(candidate)["depth_2q"] for candidate in batch) for batch in batches)
        )
        depth_model = "batch_aware_max_parallel_disjoint_support_per_adapt_iteration"
    else:
        depth = sequential_depth
        depth_2q = sequential_depth_2q
        depth_model = "sequential_pauli_rotation_proxy"

    return {
        "depth_proxy": depth,
        "circuit_depth": depth,
        "count_2q": int(count_2q),
        "compiled_depth_total": depth,
        "compiled_depth_2q_total": int(depth_2q),
        "compiled_count_2q_total": int(count_2q),
        "compiled_op_counts": {
            "pauli_rotation_proxy": int(runtime_rotation_count),
            "cx": int(count_2q),
            "rz": int(runtime_rotation_count),
            "basis_change_1q_proxy": int(oneq_basis_proxy),
        },
        "compiled_circuit_stats_status": "deterministic_pauli_rotation_proxy",
        "compiled_circuit_stats_error": None,
        "compiled_depth_model": depth_model,
        "compiled_depth_2q_model": f"two_qubit_layers::{depth_model}",
        "compiled_depth_2q_semantics": "deterministic_pauli_rotation_cnot_chain_two_qubit_layer_depth",
        "compiled_depth_2q_status": "deterministic_decomposition",
        "compiled_depth_sequential_pauli_rotation_proxy": sequential_depth,
        "compiled_depth_2q_sequential_pauli_rotation_proxy": sequential_depth_2q,
        "compiled_circuit_stats_note": (
            "No Qiskit circuit is constructed by this benchmark-local statevector runner; "
            "compiled stats use the deterministic Pauli-rotation CNOT-chain decomposition. "
            "TETRIS rows use a batch-aware depth proxy while retaining additive two-qubit counts."
        ),
    }


def _diagnostic_proxy_stats(
    selected: Sequence[_PoolCandidate],
    *,
    selected_batches: Sequence[Sequence[_PoolCandidate]] | None = None,
) -> dict[str, Any]:
    return {"diagnostic_pauli_rotation_proxy_stats": _compiled_proxy_stats(selected, selected_batches=selected_batches)}


def _ansatz_terms_from_candidates(selected: Sequence[_PoolCandidate]) -> tuple[AnsatzTerm, ...]:
    return tuple(
        AnsatzTerm(
            label=str(candidate.label),
            polynomial=candidate.polynomial,
            execution_mode=str(getattr(candidate, "execution_mode", "termwise_product")),
        )
        for candidate in selected
    )


def _candidate_pauli_terms_payload(candidate: _PoolCandidate) -> list[dict[str, Any]]:
    combined: dict[str, complex] = {}
    for term in candidate.polynomial.return_polynomial():
        label = str(term.pw2strng()).strip().lower()
        coefficient = complex(term.p_coeff)
        if not label or abs(coefficient) <= 1.0e-12:
            continue
        combined[label] = combined.get(label, 0.0 + 0.0j) + coefficient
    return [
        {
            "pauli_exyz": label,
            "coeff_re": float(coefficient.real),
            "coeff_im": float(coefficient.imag),
        }
        for label, coefficient in sorted(combined.items())
        if abs(coefficient) > 1.0e-12
    ]


def _qiskit_compiled_stats_for_selected(
    *,
    selected: Sequence[_PoolCandidate],
    selected_batches: Sequence[Sequence[_PoolCandidate]],
    config: _VariantConfig,
    num_qubits: int | None,
    reference_state: np.ndarray | Sequence[complex] | None,
    source_kind: str,
) -> dict[str, Any]:
    diagnostic = _diagnostic_proxy_stats(
        selected,
        selected_batches=selected_batches if config.variant == "tetris" else None,
    )
    status_prefix = "qiskit_first_hit" if "first_hit" in str(source_kind) else "qiskit_final_ansatz"
    grouped_exact_labels = [
        str(candidate.label)
        for candidate in selected
        if str(getattr(candidate, "execution_mode", "termwise_product")) == "grouped_exact"
        and len(tuple(candidate.pauli_labels_exyz)) > 1
    ]
    if num_qubits is None:
        return {
            "compiled_circuit_stats_status": f"{status_prefix}_compile_unavailable",
            "compiled_circuit_stats_error": "num_qubits_missing",
            "first_hit_cost_source_kind": str(source_kind),
            "compiled_resource_source_kind": str(source_kind),
            "compiled_resource_qiskit_validated": False,
            "qiskit_first_hit_cost_validated": False,
            **diagnostic,
        }
    if reference_state is None:
        return {
            "compiled_circuit_stats_status": f"{status_prefix}_compile_unavailable",
            "compiled_circuit_stats_error": "reference_state_missing",
            "first_hit_cost_source_kind": str(source_kind),
            "compiled_resource_source_kind": str(source_kind),
            "compiled_resource_qiskit_validated": False,
            "qiskit_first_hit_cost_validated": False,
            **diagnostic,
        }
    try:
        compiled = compile_table_i_ansatz_terms(
            ops=_ansatz_terms_from_candidates(selected),
            num_qubits=int(num_qubits),
            reference_state=reference_state,
            source_kind=str(source_kind),
        )
    except TableICompileUnavailable as exc:
        return {
            "compiled_circuit_stats_status": f"{status_prefix}_compile_unavailable",
            "compiled_circuit_stats_error": f"{exc.status}: {exc.reason}",
            "first_hit_cost_source_kind": str(source_kind),
            "compiled_resource_source_kind": str(source_kind),
            "compiled_resource_qiskit_validated": False,
            "qiskit_first_hit_cost_validated": False,
            **diagnostic,
        }
    except Exception as exc:  # pragma: no cover - defensive Qiskit/reporting guard
        return {
            "compiled_circuit_stats_status": f"{status_prefix}_compile_failed",
            "compiled_circuit_stats_error": f"{type(exc).__name__}: {exc}",
            "first_hit_cost_source_kind": str(source_kind),
            "compiled_resource_source_kind": str(source_kind),
            "compiled_resource_qiskit_validated": False,
            "qiskit_first_hit_cost_validated": False,
            **diagnostic,
        }
    return {
        **compiled,
        "compiled_grouped_exact_operator_labels": grouped_exact_labels,
        **diagnostic,
    }


def _qiskit_first_hit_stats_for_selected(
    *,
    selected: Sequence[_PoolCandidate],
    selected_batches: Sequence[Sequence[_PoolCandidate]],
    config: _VariantConfig,
    num_qubits: int | None,
    reference_state: np.ndarray | Sequence[complex] | None,
) -> dict[str, Any]:
    return _qiskit_compiled_stats_for_selected(
        selected=selected,
        selected_batches=selected_batches,
        config=config,
        num_qubits=num_qubits,
        reference_state=reference_state,
        source_kind="qiskit_compiled_first_hit_ansatz_circuit",
    )


def _qiskit_final_ansatz_stats_for_selected(
    selected: Sequence[_PoolCandidate],
    *,
    selected_batches: Sequence[Sequence[_PoolCandidate]],
    config: _VariantConfig,
    num_qubits: int | None,
    reference_state: np.ndarray | Sequence[complex] | None,
) -> dict[str, Any]:
    return _qiskit_compiled_stats_for_selected(
        selected=selected,
        selected_batches=selected_batches,
        config=config,
        num_qubits=num_qubits,
        reference_state=reference_state,
        source_kind="qiskit_compiled_final_ansatz_circuit",
    )


def _shot_proxy_fields(
    *,
    hamiltonian_pauli_term_count: int,
    pool_term_count: int,
    energy_eval_count: int | None,
    gradient_scan_count: int,
    gradient_operator_probe_count: int,
    metric_operator_probe_count: int,
    shots_per_pauli_term_proxy: int = _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
) -> dict[str, Any]:
    return build_deterministic_shot_proxy_fields(
        hamiltonian_pauli_term_count=hamiltonian_pauli_term_count,
        pool_term_count=pool_term_count,
        energy_eval_count=energy_eval_count,
        gradient_scan_count=gradient_scan_count,
        gradient_operator_probe_count=gradient_operator_probe_count,
        metric_operator_probe_count=metric_operator_probe_count,
        shots_per_pauli_term_proxy=shots_per_pauli_term_proxy,
        comparator_legacy_coercion=False,
    )


def _measurement_component_fields(
    *,
    selected_operator_count: int,
    energy_eval_count: int | None,
    outer_hamiltonian_eval_count: int | None = None,
    gradient_operator_probe_count: int,
    metric_operator_probe_count: int,
    metric_selector_probe_count: int = 0,
    metric_qngd_refit_probe_count: int = 0,
    metric_position_trial_probe_count: int = 0,
    gradient_selector_probe_count: int | None = None,
    gradient_qngd_refit_probe_count: int = 0,
    gradient_position_trial_probe_count: int = 0,
) -> dict[str, Any]:
    """Emit disjoint Table-I measurement-work components.

    These fields are estimator/probe event counts, not hardware shots.  They
    exist so downstream Table-I summaries can use component/event-ledger cost
    instead of inferring a paper cost from raw ``shots_total``.
    """

    selected_count = max(0, int(selected_operator_count))
    energy_count = max(1, int(energy_eval_count or 0))
    grad_count = max(0, int(gradient_operator_probe_count))
    selector_gradient = (
        grad_count
        if gradient_selector_probe_count is None
        else max(0, int(gradient_selector_probe_count))
    )
    qngd_refit_gradient = max(0, int(gradient_qngd_refit_probe_count))
    position_gradient = max(0, int(gradient_position_trial_probe_count))
    gradient_split_total = selector_gradient + qngd_refit_gradient + position_gradient
    residual_gradient = max(0, grad_count - gradient_split_total)
    gradient_split_status = (
        "explicit_disjoint" if residual_gradient == 0 else "partial_with_residual_gradient"
    )
    metric_count = max(0, int(metric_operator_probe_count))
    selector_metric = max(0, int(metric_selector_probe_count))
    qngd_refit_metric = max(0, int(metric_qngd_refit_probe_count))
    position_metric = max(0, int(metric_position_trial_probe_count))
    split_total = selector_metric + qngd_refit_metric + position_metric
    residual_metric = max(0, metric_count - split_total)
    split_status = "explicit_disjoint" if residual_metric == 0 else "partial_with_residual_metric"
    if outer_hamiltonian_eval_count is not None:
        n_h_outer = max(0, int(outer_hamiltonian_eval_count))
        n_h_refit = max(0, int(energy_eval_count or 0))
        h_outer_source = "explicit_outer_selector_hamiltonian_apply_count"
        h_refit_source = "optimizer_energy_eval_count"
    elif selected_count > 0:
        n_h_outer = 0
        n_h_refit = energy_count
        h_outer_source = "legacy_adaptive_refit_partition"
        h_refit_source = "legacy_energy_eval_count_proxy"
    else:
        n_h_outer = energy_count
        n_h_refit = 0
        h_outer_source = "energy_eval_count_proxy"
        h_refit_source = "no_selected_operators"
    s_alg = float(n_h_outer + grad_count + metric_count + n_h_refit)
    components = {
        "N_H_outer_eval": float(n_h_outer),
        "N_grad_probe": float(grad_count),
        "N_metric_probe": float(metric_count),
        "N_H_refit_eval": float(n_h_refit),
        "N_other_quantum": 0.0,
    }
    component_sources = {
        "N_H_outer_eval": h_outer_source,
        "N_grad_probe": "gradient_operator_probe_count_proxy",
        "N_metric_probe": "metric_operator_probe_count_proxy",
        "N_H_refit_eval": h_refit_source,
        "N_other_quantum": "method_zero",
    }
    return {
        "S_alg": s_alg,
        "S_alg_N_H_outer_eval": float(n_h_outer),
        "S_alg_N_grad_probe": float(grad_count),
        "S_alg_N_metric_probe": float(metric_count),
        "S_alg_N_H_refit_eval": float(n_h_refit),
        "S_alg_N_other_quantum": 0.0,
        "algorithmic_measurement_work_N_H_outer_eval": float(n_h_outer),
        "algorithmic_measurement_work_N_grad_probe": float(grad_count),
        "algorithmic_measurement_work_N_metric_probe": float(metric_count),
        "algorithmic_measurement_work_N_H_refit_eval": float(n_h_refit),
        "algorithmic_measurement_work_N_other_quantum": 0.0,
        "S_norm": s_alg,
        "S_norm_N_H_outer_eval": float(n_h_outer),
        "S_norm_N_grad": float(grad_count),
        "S_norm_N_metric": float(metric_count),
        "S_norm_N_H_refit_eval": float(n_h_refit),
        "S_norm_N_H_eval": float(n_h_outer),
        "S_norm_N_refit_eval": float(n_h_refit),
        "S_norm_N_other_quantum": 0.0,
        "N_H_outer_eval": float(n_h_outer),
        "N_grad": float(grad_count),
        "N_metric": float(metric_count),
        "N_H_refit_eval": float(n_h_refit),
        "N_other_quantum": 0.0,
        "N_metric_selector_probe": float(selector_metric),
        "N_metric_qngd_refit_probe": float(qngd_refit_metric),
        "N_metric_position_trial_probe": float(position_metric),
        "N_metric_residual_probe": float(residual_metric),
        "N_metric_split_status": split_status,
        "N_grad_selector_probe": float(selector_gradient),
        "N_grad_qngd_refit_probe": float(qngd_refit_gradient),
        "N_grad_position_trial_probe": float(position_gradient),
        "N_grad_residual_probe": float(residual_gradient),
        "N_grad_split_status": gradient_split_status,
        "metric_fraction": float(metric_count / s_alg) if s_alg > 0.0 else 0.0,
        "table_i_measurement_event_ledger": {
            "schema": TABLE_I_EVENT_LEDGER_SCHEMA,
            "status": "ok",
            "source_kind": "exact_bench_native_component_fields_v1",
            "component_totals": components,
            "component_sources": component_sources,
            "metric_split_totals": {
                "N_metric_selector_probe": float(selector_metric),
                "N_metric_qngd_refit_probe": float(qngd_refit_metric),
                "N_metric_position_trial_probe": float(position_metric),
                "N_metric_residual_probe": float(residual_metric),
                "status": split_status,
            },
            "gradient_split_totals": {
                "N_grad_selector_probe": float(selector_gradient),
                "N_grad_qngd_refit_probe": float(qngd_refit_gradient),
                "N_grad_position_trial_probe": float(position_gradient),
                "N_grad_residual_probe": float(residual_gradient),
                "status": gradient_split_status,
            },
            "event_count_convention": "fresh_measurement_bearing_estimator_or_probe_events",
            "cache_policy": "no_cache_reuse_in_current_exact_bench_replay",
            "measurement_model_id": "noiseless_estimator_schedule_count_v1",
            "N_other_quantum": 0.0,
        },
    }



def _finite_float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _float_or_none(value: Any) -> float | None:
    out = _finite_float_or_none(value)
    if out is None or out <= 0.0:
        return None
    return float(out)


def _env_text_or_none(name: str) -> str | None:
    raw = os.environ.get(str(name), "")
    text = str(raw).strip()
    return text or None


def _env_finite_float_or_none(name: str) -> float | None:
    return _finite_float_or_none(os.environ.get(str(name), ""))


def _normalize_reference_metric_fields(
    *,
    same_cutoff_exact_gs_energy: float | str | None = None,
    exact_reference_energy: float | str | None = None,
    exact_reference_n_ph_max: int | str | None = None,
    primary_energy_metric: str | None = None,
    same_cutoff_error_role: str | None = None,
    fallback_same_cutoff_energy: float | None = None,
) -> dict[str, Any]:
    same = _finite_float_or_none(same_cutoff_exact_gs_energy)
    if same is None:
        same = _env_finite_float_or_none(_SAME_CUTOFF_EXACT_GS_ENERGY_ENV)
    if same is None:
        same = _finite_float_or_none(fallback_same_cutoff_energy)
    ref = _finite_float_or_none(exact_reference_energy)
    if ref is None:
        ref = _env_finite_float_or_none(_EXACT_REFERENCE_ENERGY_ENV)
    metric = str(primary_energy_metric or _env_text_or_none(_PRIMARY_ENERGY_METRIC_ENV) or "same_cutoff_abs_delta_e").strip()
    role = str(same_cutoff_error_role or _env_text_or_none(_SAME_CUTOFF_ERROR_ROLE_ENV) or "primary").strip()
    raw_ref_nph = exact_reference_n_ph_max if exact_reference_n_ph_max not in {None, ""} else _env_text_or_none(_EXACT_REFERENCE_N_PH_MAX_ENV)
    ref_nph: int | None = None
    if raw_ref_nph not in {None, ""}:
        try:
            ref_nph = int(float(str(raw_ref_nph)))
        except Exception:
            ref_nph = None
    use_reference = metric == "higher_cutoff_reference_abs_delta_e" and ref is not None
    primary_reference = ref if use_reference else same
    if primary_reference is None:
        primary_reference = ref
    return {
        "same_cutoff_exact_gs_energy": same,
        "exact_reference_energy": ref,
        "exact_reference_n_ph_max": ref_nph,
        "primary_energy_metric": metric,
        "same_cutoff_error_role": role,
        "primary_reference_energy": primary_reference,
        "primary_reference_source": "exact_reference_energy" if use_reference else "same_cutoff_exact_gs_energy",
    }


def _energy_error_fields(energy: float, reference: Mapping[str, Any]) -> dict[str, Any]:
    same = _finite_float_or_none(reference.get("same_cutoff_exact_gs_energy"))
    ref = _finite_float_or_none(reference.get("exact_reference_energy"))
    primary = _finite_float_or_none(reference.get("primary_reference_energy"))
    out: dict[str, Any] = {}
    if same is not None:
        out["abs_delta_e_same_cutoff"] = float(abs(float(energy) - same))
    if ref is not None:
        out["abs_delta_e_reference"] = float(abs(float(energy) - ref))
    if primary is not None:
        out["delta_E_abs"] = float(abs(float(energy) - primary))
        out["abs_delta_e"] = float(abs(float(energy) - primary))
    return out


def _normalize_first_hit_thresholds(values: Sequence[float] | str | None) -> tuple[float, ...]:
    if values is None or values == "":
        raw = os.environ.get(_FIRST_HIT_THRESHOLDS_ENV, "")
        if not raw:
            return ()
        parts: Sequence[Any] = tuple(part.strip() for part in raw.split(",") if part.strip())
    elif isinstance(values, str):
        parts = tuple(part.strip() for part in values.split(",") if part.strip())
    else:
        parts = tuple(values)
    thresholds = sorted({float(x) for x in parts if _float_or_none(x) is not None}, reverse=True)
    return tuple(thresholds)


def _normalize_energy_stop_target(value: float | str | None) -> float | None:
    explicit = _float_or_none(value)
    if explicit is not None:
        return explicit
    return _float_or_none(os.environ.get(_ENERGY_STOP_TARGET_ENV))


def _normalize_powell_maxiter_cap_policy(value: str | None) -> str:
    policy = str(value or _POWELL_MAXITER_CAP_POLICY_STRICT).strip().lower()
    if policy not in _POWELL_MAXITER_CAP_POLICY_CHOICES:
        allowed = ", ".join(sorted(_POWELL_MAXITER_CAP_POLICY_CHOICES))
        raise ValueError(
            f"unsupported Powell maxiter cap policy {value!r}; allowed values: {allowed}"
        )
    return policy


def _classify_powell_maxiter_cap(
    *,
    config: _VariantConfig,
    policy: str,
    theta: np.ndarray,
    energy_before: float,
    energy_after: float,
    optimizer_maxiter: int,
    opt_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Return effective optimizer status plus auditable cap telemetry.

    The opt-in repair policy is intentionally narrower than generic
    best-so-far optimizer recovery.  It recognizes only SciPy Powell's exact
    max-iteration termination and accepts that capped point only when the
    returned objective, exact refit energy, and parameters are finite and the
    exact refit energy is non-increasing relative to the pre-refit state within
    the repository's normal ``1e-10`` numerical tolerance.
    """

    out = dict(opt_info)
    raw_success = bool(out.get("success", False))
    message = str(out.get("message", ""))
    normalized_message = " ".join(message.strip().lower().rstrip(".").split())
    status_raw = out.get("status")
    nit_raw = out.get("nit")

    def _int_or_none(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    status_value = _int_or_none(status_raw)
    nit_value = _int_or_none(nit_raw)
    decision_energy = _finite_float_or_none(out.get("optimizer_decision_energy"))
    theta_array = np.asarray(theta, dtype=float).reshape(-1)
    parameters_finite = bool(np.all(np.isfinite(theta_array)))
    energy_before_finite = bool(math.isfinite(float(energy_before)))
    energy_after_finite = bool(math.isfinite(float(energy_after)))
    objective_finite = decision_energy is not None
    energy_nonincreasing = bool(
        energy_before_finite
        and energy_after_finite
        and (
            float(energy_after) <= float(energy_before)
            or math.isclose(
                float(energy_after),
                float(energy_before),
                rel_tol=_POWELL_CAP_ENERGY_REL_TOL,
                abs_tol=_POWELL_CAP_ENERGY_ABS_TOL,
            )
        )
    )
    is_powell = bool(
        config.optimizer_kind == "powell"
        and str(out.get("optimizer") or "") == "scipy.optimize.minimize:Powell"
    )
    is_maxiter_only = bool(
        (not raw_success)
        and is_powell
        and status_value == _POWELL_MAXITER_STATUS
        and nit_value == int(optimizer_maxiter)
        and normalized_message == _POWELL_MAXITER_MESSAGE
    )
    policy_enabled = bool(
        policy == _POWELL_MAXITER_CAP_POLICY_ACCEPT_FINITE_NONINCREASING
    )
    optimizer_capped = bool(is_maxiter_only)
    optimizer_capped_accepted = bool(
        policy_enabled
        and optimizer_capped
        and parameters_finite
        and objective_finite
        and energy_after_finite
        and energy_nonincreasing
    )
    effective_success = bool(raw_success or optimizer_capped_accepted)

    if raw_success:
        acceptance_reason = "optimizer_success"
    elif not optimizer_capped:
        acceptance_reason = "not_powell_maxiter_only"
    elif not policy_enabled:
        acceptance_reason = "strict_failure_policy"
    elif not parameters_finite:
        acceptance_reason = "nonfinite_parameters"
    elif not objective_finite:
        acceptance_reason = "nonfinite_objective"
    elif not energy_after_finite:
        acceptance_reason = "nonfinite_exact_energy"
    elif not energy_nonincreasing:
        acceptance_reason = "energy_increase_exceeds_tolerance"
    else:
        acceptance_reason = "finite_nonincreasing_powell_maxiter_accepted"

    out.update(
        {
            "raw_success": bool(raw_success),
            "success": bool(effective_success),
            "optimizer_capped": bool(optimizer_capped),
            "optimizer_capped_accepted": bool(optimizer_capped_accepted),
            "optimizer_cap_policy": str(policy),
            "optimizer_cap_acceptance_reason": str(acceptance_reason),
            "optimizer_cap_status": status_value,
            "optimizer_cap_expected_status": int(_POWELL_MAXITER_STATUS),
            "optimizer_cap_nit": nit_value,
            "optimizer_cap_maxiter": int(optimizer_maxiter),
            "optimizer_cap_message_match": bool(
                normalized_message == _POWELL_MAXITER_MESSAGE
            ),
            "optimizer_cap_parameters_finite": bool(parameters_finite),
            "optimizer_cap_objective_finite": bool(objective_finite),
            "optimizer_cap_energy_before_finite": bool(energy_before_finite),
            "optimizer_cap_energy_after_finite": bool(energy_after_finite),
            "optimizer_cap_energy_nonincreasing": bool(energy_nonincreasing),
            "optimizer_cap_energy_before": (
                float(energy_before) if energy_before_finite else None
            ),
            "optimizer_cap_energy_after": (
                float(energy_after) if energy_after_finite else None
            ),
            "optimizer_cap_energy_delta": (
                float(energy_after) - float(energy_before)
                if energy_before_finite and energy_after_finite
                else None
            ),
            "optimizer_cap_energy_rel_tol": float(_POWELL_CAP_ENERGY_REL_TOL),
            "optimizer_cap_energy_abs_tol": float(_POWELL_CAP_ENERGY_ABS_TOL),
        }
    )
    return out


def _first_hit_key(threshold: float) -> str:
    # 1e-06 -> 1e_6, 1e-08 -> 1e_8 for stable JSON field names.
    text = f"{float(threshold):.0e}".replace("-0", "-").replace("-", "_")
    return f"first_hit_{text}"


def _first_hit_record(
    *,
    threshold: float,
    iteration: int,
    energy: float,
    reference: Mapping[str, Any],
    selected: Sequence[_PoolCandidate],
    selected_batches: Sequence[Sequence[_PoolCandidate]],
    config: _VariantConfig,
    hamiltonian_pauli_term_count: int,
    pool_term_count: int,
    nfev_total: int,
    gradient_scan_count: int,
    gradient_probe_count: int,
    metric_probe_count: int,
    metric_selector_probe_count: int = 0,
    metric_qngd_refit_probe_count: int = 0,
    metric_position_trial_probe_count: int = 0,
    gradient_selector_probe_count: int | None = None,
    gradient_qngd_refit_probe_count: int = 0,
    gradient_position_trial_probe_count: int = 0,
    shots_per_pauli_term_proxy: int,
    outer_hamiltonian_eval_count: int = 0,
    num_qubits: int | None = None,
    reference_state: np.ndarray | Sequence[complex] | None = None,
) -> dict[str, Any]:
    compiled_stats = _qiskit_first_hit_stats_for_selected(
        selected=selected,
        selected_batches=selected_batches,
        config=config,
        num_qubits=num_qubits,
        reference_state=reference_state,
    )
    shot_proxy = _shot_proxy_fields(
        hamiltonian_pauli_term_count=hamiltonian_pauli_term_count,
        pool_term_count=pool_term_count,
        energy_eval_count=max(1, int(nfev_total) + int(outer_hamiltonian_eval_count)),
        gradient_scan_count=gradient_scan_count,
        gradient_operator_probe_count=gradient_probe_count,
        metric_operator_probe_count=metric_probe_count,
        shots_per_pauli_term_proxy=int(shots_per_pauli_term_proxy),
    )
    component_fields = _measurement_component_fields(
        selected_operator_count=len(selected),
        energy_eval_count=max(0, int(nfev_total)),
        outer_hamiltonian_eval_count=max(0, int(outer_hamiltonian_eval_count)),
        gradient_operator_probe_count=gradient_probe_count,
        metric_operator_probe_count=metric_probe_count,
        metric_selector_probe_count=metric_selector_probe_count,
        metric_qngd_refit_probe_count=metric_qngd_refit_probe_count,
        metric_position_trial_probe_count=metric_position_trial_probe_count,
        gradient_selector_probe_count=gradient_selector_probe_count,
        gradient_qngd_refit_probe_count=gradient_qngd_refit_probe_count,
        gradient_position_trial_probe_count=gradient_position_trial_probe_count,
    )
    error_fields = _energy_error_fields(float(energy), reference)
    primary_ref = _finite_float_or_none(reference.get("primary_reference_energy"))
    out = {
        "threshold_abs_delta_e": float(threshold),
        "source": "native_adaptive_iteration",
        "first_hit_semantics": "native_first_crossing_after_adapt_iteration",
        "iteration": int(iteration),
        "energy": float(energy),
        "exact_energy": primary_ref,
        "same_cutoff_exact_gs_energy": reference.get("same_cutoff_exact_gs_energy"),
        "exact_reference_energy": reference.get("exact_reference_energy"),
        "exact_reference_n_ph_max": reference.get("exact_reference_n_ph_max"),
        "primary_energy_metric": reference.get("primary_energy_metric"),
        "primary_reference_source": reference.get("primary_reference_source"),
        "same_cutoff_error_role": reference.get("same_cutoff_error_role"),
        **error_fields,
        "num_parameters": int(len(selected)),
        "selected_operator_count": int(len(selected)),
        "selected_operators": [str(candidate.label) for candidate in selected],
        "selected_operator_supports": [list(candidate.support) for candidate in selected],
        "selected_operator_pauli_labels_exyz": [list(candidate.pauli_labels_exyz) for candidate in selected],
        "selected_operator_batches": [[str(candidate.label) for candidate in batch] for batch in selected_batches],
        "first_hit_theta_status": "not_required_for_structural_gate_count_compile",
        **compiled_stats,
        **shot_proxy,
        **component_fields,
    }
    if "delta_E_abs" not in out:
        out["delta_E_abs"] = None
        out["abs_delta_e"] = None
    return out

def _sector_or_unavailable(context: ResolvedProblemContext, psi: np.ndarray | None) -> dict[str, Any]:
    if psi is None:
        return {
            "sector_probability": None,
            "sector_leak_probability": None,
            "sector_leak_flag": None,
            "sector_leak_threshold": None,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": None,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
            "policy": "not_available_final_state_not_reconstructed",
        }
    try:
        return sector_probability(context, psi)
    except Exception as exc:
        return {
            "sector_probability": None,
            "sector_leak_probability": None,
            "sector_leak_flag": None,
            "sector_leak_threshold": None,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": None,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
            "policy": "failed_sector_diagnostic",
            "sector_diagnostic_error": str(exc),
        }


def _run_impl(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    output_dir: Path,
    max_adapt_iterations: int = _DEFAULT_MAX_ADAPT_ITERATIONS,
    optimizer_maxiter: int = _DEFAULT_OPTIMIZER_MAXITER,
    gradient_threshold: float = _DEFAULT_GRADIENT_THRESHOLD,
    metric_floor: float = _DEFAULT_METRIC_FLOOR,
    max_tetris_batch_size: int = _DEFAULT_MAX_TETRIS_BATCH_SIZE,
    seed: int = 42,
    shots_per_pauli_term_proxy: int = _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
    energy_stop_target: float | None = None,
    first_hit_thresholds: Sequence[float] | str | None = None,
    same_cutoff_exact_gs_energy: float | str | None = None,
    exact_reference_energy: float | str | None = None,
    exact_reference_n_ph_max: int | str | None = None,
    primary_energy_metric: str | None = None,
    same_cutoff_error_role: str | None = None,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | Mapping[str, Any] | None = None,
    selected_logical_route: str | None = None,
    selected_logical_source_json: str | Path | None = None,
    selected_logical_transfer_mode: str = "exact_match_v1",
    allow_repeats: bool | None = None,
    progress_jsonl_path: str | Path | None = None,
    generic_adapt_stop_policy: str | None = None,
    powell_maxiter_cap_policy: str | None = None,
    generic_adapt_runtime_split_mode: str | None = _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF,
    generic_adapt_runtime_split_symmetry_policy: str | None = _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_OFF,
    generic_adapt_runtime_split_max_subset_size: int | str | None = 3,
    shared_pauli_pool_mode: str | None = SHARED_PAULI_POOL_MODE_OFF,
    shared_pauli_pool_symmetry_policy: str | None = SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF,
    shared_pauli_pool_max_subset_size: int | str | None = 3,
    initial_selected_operator_labels: Sequence[str] | None = None,
    initial_selected_operator_batches: Sequence[Sequence[str]] | None = None,
    initial_theta: Sequence[float] | None = None,
    initial_adapt_history: Sequence[Mapping[str, Any]] | None = None,
    optimizer_profile: str | None = None,
    optimizer_profile_source: str | None = None,
    adapt_optimizer_kind: str | None = None,
    adapt_spsa_maxiter: int | None = None,
    adapt_spsa_seed: int | None = None,
    adapt_spsa_a: float | str | None = None,
    adapt_spsa_c: float | str | None = None,
    adapt_spsa_alpha: float | str | None = None,
    adapt_spsa_gamma: float | str | None = None,
    adapt_spsa_big_a: float | str | None = None,
    optimizer_overlay_source: str | None = None,
    hh_adaptive_pool_profile: str | None = None,
    hh_full_meta_class_filter_json: str | Path | None = None,
) -> dict[str, Any]:
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    config = _apply_environment_position_policy_override(_get_config(algorithm_id), family=family_key)
    config, optimizer_settings = _effective_optimizer_settings_for_config(
        config,
        optimizer_profile=optimizer_profile,
        optimizer_profile_source=optimizer_profile_source,
        adapt_optimizer_kind=adapt_optimizer_kind,
        adapt_spsa_maxiter=adapt_spsa_maxiter,
        adapt_spsa_seed=adapt_spsa_seed,
        adapt_spsa_a=adapt_spsa_a,
        adapt_spsa_c=adapt_spsa_c,
        adapt_spsa_alpha=adapt_spsa_alpha,
        adapt_spsa_gamma=adapt_spsa_gamma,
        adapt_spsa_big_a=adapt_spsa_big_a,
        optimizer_maxiter=int(optimizer_maxiter),
        seed=int(seed),
        optimizer_overlay_source=optimizer_overlay_source,
    )
    optimizer_maxiter = int(optimizer_settings["optimizer_maxiter"])
    spsa_seed_base = int(optimizer_settings["spsa_seed_base"])
    spsa_schedule = optimizer_settings.get("spsa_schedule")
    if allow_repeats is not None:
        config = replace(
            config,
            repeat_policy=(
                _repeat_enabled_policy_for_config(config)
                if bool(allow_repeats)
                else "exclude_selected_labels"
            ),
        )
    decision_noise_config = coerce_benchmark_decision_noise_config(
        benchmark_decision_noise_config,
        family=family_key,
        case_id=case_key,
        algorithm_id=config.algorithm_id,
    )
    decision_noise_recorder = BenchmarkDecisionNoiseRecorder(
        decision_noise_config,
        base_scope={"family": family_key, "case_id": case_key, "algorithm_id": config.algorithm_id},
    )
    output = Path(output_dir)
    started_utc = _utc_now()
    t0 = time.perf_counter()
    progress_path = Path(progress_jsonl_path) if progress_jsonl_path not in {None, ""} else None
    if progress_path is not None:
        try:
            progress_path.unlink()
        except FileNotFoundError:
            pass
    progress_stdout = str(os.environ.get("GENERIC_STATIC_TABLE_PROGRESS_STDOUT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    def _progress(event: str, **fields: Any) -> None:
        if progress_path is None and not progress_stdout:
            return
        payload = {
            "schema": "generic_static_adapt_iteration_progress_v1",
            "event": str(event),
            "utc": _utc_now(),
            "elapsed_s": float(time.perf_counter() - t0),
            "family": family_key,
            "case_id": case_key,
            "algorithm_id": config.algorithm_id,
            **fields,
        }
        if progress_path is not None:
            _append_jsonl(progress_path, payload)
        if progress_stdout and str(event) in {"iteration_complete", "stop", "terminal"}:
            print("GENERIC_STATIC_PROGRESS " + json.dumps(payload, sort_keys=True), flush=True)

    benchmark_energy_stop_target = _normalize_energy_stop_target(energy_stop_target)
    benchmark_first_hit_thresholds = _normalize_first_hit_thresholds(first_hit_thresholds)
    generic_adapt_stop_policy_label = str(generic_adapt_stop_policy or "").strip() or None
    powell_cap_policy = _normalize_powell_maxiter_cap_policy(powell_maxiter_cap_policy)
    if powell_cap_policy == _POWELL_MAXITER_CAP_POLICY_ACCEPT_FINITE_NONINCREASING:
        if config.algorithm_id != STATIC_FULL_META_APPEND_ADAPT_VQE:
            raise ValueError(
                "accept_finite_nonincreasing_v1 is restricted to append-only ADAPT repair rows"
            )
        if config.optimizer_kind != "powell":
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires the Powell inner optimizer"
            )
        if generic_adapt_stop_policy_label != _FIXED_HORIZON_NO_TARGET_STOP_POLICY:
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires fixed_horizon_no_target_v1"
            )
        if benchmark_energy_stop_target is not None:
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires energy_stop_target to be absent; "
                "the repaired row must continue to the fixed outer horizon"
            )
        if float(gradient_threshold) != 0.0:
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires gradient_threshold=0 so gradient "
                "rules cannot terminate the repaired fixed-horizon row early"
            )
    runtime_split_mode = _normalize_generic_adapt_runtime_split_mode(generic_adapt_runtime_split_mode)
    runtime_split_symmetry_policy = _normalize_generic_adapt_runtime_split_symmetry_policy(
        generic_adapt_runtime_split_symmetry_policy
    )
    runtime_split_max_subset_size_raw = (
        3
        if generic_adapt_runtime_split_max_subset_size in {None, ""}
        else generic_adapt_runtime_split_max_subset_size
    )
    runtime_split_max_subset_size = _positive_int(
        runtime_split_max_subset_size_raw,
        field="generic_adapt_runtime_split_max_subset_size",
    )
    shared_pool_mode = normalize_shared_pauli_pool_mode(shared_pauli_pool_mode)
    shared_pool_symmetry_policy = normalize_shared_pauli_pool_symmetry_policy(
        shared_pauli_pool_symmetry_policy
    )
    shared_pool_max_subset_size_raw = 3 if shared_pauli_pool_max_subset_size in {None, ""} else shared_pauli_pool_max_subset_size
    shared_pool_max_subset_size = _positive_int(
        shared_pool_max_subset_size_raw,
        field="shared_pauli_pool_max_subset_size",
    )
    if (
        shared_pool_mode != SHARED_PAULI_POOL_MODE_OFF
        and runtime_split_mode != _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF
    ):
        raise ValueError("shared_pauli_pool_mode cannot be combined with generic_adapt_runtime_split_mode.")
    first_hits: dict[float, dict[str, Any]] = {}
    first_hit_candidates: list[dict[str, Any]] = []

    spec = _spec_by_case_id(family_key, case_key, config.algorithm_id)
    minimize_fn = None
    if _requires_scipy_minimize(config):
        if not has_scipy_minimize_support():
            return _skip_payload(
                family=family_key,
                case_id=case_key,
                config=config,
                output_dir=output,
                reason="optional scipy.optimize.minimize dependency is not importable",
                started_utc=started_utc,
            )
        minimize_fn = _import_scipy_minimize()

    try:
        np.random.seed(int(seed))
    except Exception:
        pass

    context = _resolve_context_from_spec(spec)
    num_qubits = int(context.layout.total_qubits)
    qubit_cap = _resource_cap_from_env(_RESOURCE_QUBIT_CAP_ENV, _QUBIT_CAP)
    pool_term_cap = _resource_cap_from_env(_RESOURCE_POOL_TERM_CAP_ENV, _POOL_TERM_CAP)
    selected_logical_mode = _selected_logical_mode_from_route(selected_logical_route)
    selected_logical_requested = selected_logical_mode != "off" or selected_logical_source_json not in {None, ""}
    if selected_logical_requested and config.pool_kind != "full_meta":
        raise ValueError(
            "selected-logical reduced-pool overlay is only supported for full_meta generic static ADAPT variants; "
            f"algorithm_id={config.algorithm_id!r} uses pool_kind={config.pool_kind!r}."
        )
    if qubit_cap is not None and num_qubits > int(qubit_cap):
        guard = {
            "resource_guard": True,
            "resource_guard_kind": "generic_adapt_variant_qubit_cap",
            "reason": "Generic statevector ADAPT variant canonical case qubit count exceeds cap",
            "num_qubits": num_qubits,
            "qubit_cap": int(qubit_cap),
            "pool_term_count": 0,
            "pool_term_cap": None if pool_term_cap is None else int(pool_term_cap),
        }
        return _resource_guard_payload(
            family=family_key,
            case_id=case_key,
            config=config,
            output_dir=output,
            spec=spec,
            started_utc=started_utc,
            reason=str(guard["reason"]),
            guard=guard,
        )
    pool_name = _pool_name_for_config(config)
    selected_logical_filter_meta: dict[str, Any] | None = None
    full_meta_class_filter_meta: dict[str, Any] | None = None
    full_meta_label_filter_meta: dict[str, Any] | None = None
    pool_legal_subspace_filter_meta: dict[str, Any] | None = None
    pool_cache_events: tuple[dict[str, Any], ...] = ()
    effective_hh_adaptive_pool_profile, hh_full_meta_class_filter_json = _resolve_hh_full_meta_pool_profile(
        config=config,
        context=context,
        hh_adaptive_pool_profile=hh_adaptive_pool_profile,
        hh_full_meta_class_filter_json=hh_full_meta_class_filter_json,
    )
    try:
        if config.pool_kind == "full_meta":
            if selected_logical_requested:
                pool_result = _build_reduced_full_meta_candidate_pool_with_meta(
                    context,
                    max_terms=pool_term_cap,
                    selected_logical_source_json=selected_logical_source_json,
                    selected_logical_mode=selected_logical_mode,
                    selected_logical_transfer_mode=selected_logical_transfer_mode,
                    hh_full_meta_class_filter_json=hh_full_meta_class_filter_json,
                )
            else:
                if hh_full_meta_class_filter_json is None:
                    pool = build_full_meta_candidate_pool(context, max_terms=pool_term_cap)
                    pool_result = None
                else:
                    pool_result = _build_full_meta_candidate_pool_with_meta(
                        context,
                        max_terms=pool_term_cap,
                        hh_full_meta_class_filter_json=hh_full_meta_class_filter_json,
                    )
                    pool = pool_result.candidates
            if pool_result is not None:
                pool = pool_result.candidates
                selected_logical_filter_meta = pool_result.selected_logical_filter_meta
                full_meta_class_filter_meta = pool_result.full_meta_class_filter_meta
                full_meta_label_filter_meta = pool_result.full_meta_label_filter_meta
                pool_legal_subspace_filter_meta = pool_result.pool_legal_subspace_filter_meta
                pool_cache_events = pool_result.pool_cache_events
        else:
            pool = build_pairwise_qubit_excitation_pool(num_qubits, max_terms=pool_term_cap)
    except ValueError as exc:
        if "full_meta pool exceeds cap" not in str(exc) and "QEB singles+doubles pool exceeds cap" not in str(exc):
            raise
        guard = {
            "resource_guard": True,
            "resource_guard_kind": "generic_adapt_variant_pool_term_cap",
            "reason": f"{pool_name} exceeds cap",
            "num_qubits": num_qubits,
            "qubit_cap": None if qubit_cap is None else int(qubit_cap),
            "pool_term_count": None if pool_term_cap is None else int(pool_term_cap + 1),
            "pool_term_cap": None if pool_term_cap is None else int(pool_term_cap),
        }
        return _resource_guard_payload(
            family=family_key,
            case_id=case_key,
            config=config,
            output_dir=output,
            spec=spec,
            started_utc=started_utc,
            reason=str(guard["reason"]),
            guard=guard,
        )
    try:
        if shared_pool_mode != SHARED_PAULI_POOL_MODE_OFF:
            pool, runtime_split_pool_meta = _expand_pool_with_runtime_split_children(
                pool=pool,
                context=context,
                config=config,
                split_mode=_GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF,
                symmetry_policy=_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_OFF,
                max_subset_size=int(runtime_split_max_subset_size),
                max_terms=pool_term_cap,
            )
            pool, shared_pauli_pool_meta = _expand_pool_with_shared_pauli_children(
                pool=pool,
                context=context,
                config=config,
                mode=shared_pool_mode,
                symmetry_policy=shared_pool_symmetry_policy,
                max_subset_size=int(shared_pool_max_subset_size),
                max_terms=pool_term_cap,
            )
        else:
            pool, runtime_split_pool_meta = _expand_pool_with_runtime_split_children(
                pool=pool,
                context=context,
                config=config,
                split_mode=runtime_split_mode,
                symmetry_policy=runtime_split_symmetry_policy,
                max_subset_size=int(runtime_split_max_subset_size),
                max_terms=pool_term_cap,
            )
            _, shared_pauli_pool_meta = _expand_pool_with_shared_pauli_children(
                pool=pool,
                context=context,
                config=config,
                mode=SHARED_PAULI_POOL_MODE_OFF,
                symmetry_policy=SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF,
                max_subset_size=int(shared_pool_max_subset_size),
                max_terms=pool_term_cap,
            )
    except ValueError as exc:
        if "runtime-split full_meta pool exceeds cap" not in str(exc) and "shared Pauli-child pool exceeds cap" not in str(exc):
            raise
        guard = {
            "resource_guard": True,
            "resource_guard_kind": "generic_adapt_variant_expanded_pool_term_cap",
            "reason": f"{pool_name} expanded pool exceeds cap",
            "num_qubits": num_qubits,
            "qubit_cap": None if qubit_cap is None else int(qubit_cap),
            "pool_term_count": None if pool_term_cap is None else int(pool_term_cap + 1),
            "pool_term_cap": None if pool_term_cap is None else int(pool_term_cap),
            "generic_adapt_runtime_split_mode": runtime_split_mode,
            "generic_adapt_runtime_split_symmetry_policy": runtime_split_symmetry_policy,
            "generic_adapt_runtime_split_max_subset_size": int(runtime_split_max_subset_size),
            "shared_pauli_pool_mode": shared_pool_mode,
            "shared_pauli_pool_symmetry_policy": shared_pool_symmetry_policy,
            "shared_pauli_pool_max_subset_size": int(shared_pool_max_subset_size),
        }
        return _resource_guard_payload(
            family=family_key,
            case_id=case_key,
            config=config,
            output_dir=output,
            spec=spec,
            started_utc=started_utc,
            reason=str(guard["reason"]),
            guard=guard,
        )
    guard = _resource_guard_for_context(
        context,
        pool,
        pool_cap=pool_term_cap,
        qubit_cap=qubit_cap,
        pool_name=pool_name,
    )
    if guard is not None:
        return _resource_guard_payload(
            family=family_key,
            case_id=case_key,
            config=config,
            output_dir=output,
            spec=spec,
            started_utc=started_utc,
            reason=str(guard["reason"]),
            guard=guard,
        )

    parameterization_kind = str(config.optimizer_kind).strip().lower()
    if parameterization_kind in {"rotosolve", "geo_qngd"}:
        incompatible = [
            str(candidate.label)
            for candidate in pool
            if str(getattr(candidate, "execution_mode", "termwise_product")) == "grouped_exact"
            and len(tuple(candidate.pauli_labels_exyz)) > 1
        ]
        if incompatible:
            preview = ", ".join(incompatible[:5])
            raise ValueError(
                f"{parameterization_kind} per-Pauli coordinates are incompatible with grouped_exact multi-Pauli "
                f"generators; refusing to change ansatz semantics. First labels: {preview}"
            )

    psi_ref = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi_ref))
    if norm <= 0.0:
        raise ValueError("reference state has zero norm")
    psi_ref = psi_ref / norm

    pauli_action_cache: dict[str, Any] = {}
    h_compiled = compile_polynomial_action(context.hamiltonian, tol=1e-12, pauli_action_cache=pauli_action_cache)
    compiled_pool = _compile_pool(pool, pauli_action_cache=pauli_action_cache)
    by_label = {item.candidate.label: item.candidate for item in compiled_pool}
    hamiltonian_pauli_term_count = _hamiltonian_pauli_term_count(context.hamiltonian)
    reference_metrics = _normalize_reference_metric_fields(
        same_cutoff_exact_gs_energy=same_cutoff_exact_gs_energy,
        exact_reference_energy=exact_reference_energy,
        exact_reference_n_ph_max=exact_reference_n_ph_max,
        primary_energy_metric=primary_energy_metric,
        same_cutoff_error_role=same_cutoff_error_role,
        fallback_same_cutoff_energy=None,
    )
    target_stop_enabled = (
        benchmark_energy_stop_target is not None
        and _finite_float_or_none(reference_metrics.get("primary_reference_energy")) is not None
    )

    selected: list[_PoolCandidate] = []
    selected_batches: list[list[_PoolCandidate]] = []
    selected_labels: set[str] = set()
    theta = np.zeros(0, dtype=float)
    rotosolve_runtime_coordinates = str(config.optimizer_kind).strip().lower() == "rotosolve"

    def _selected_parameterization_layout(
        selected_now: Sequence[_PoolCandidate],
    ) -> AnsatzParameterLayout:
        return build_parameter_layout(
            list(selected_now),
            ignore_identity=True,
            coefficient_tolerance=1e-12,
            sort_terms=True,
        )

    def _runtime_theta_for_selected(
        selected_now: Sequence[_PoolCandidate],
        theta_now: np.ndarray,
    ) -> tuple[np.ndarray, AnsatzParameterLayout]:
        layout_now = _selected_parameterization_layout(selected_now)
        theta_arr = np.asarray(theta_now, dtype=float).reshape(-1)
        if int(theta_arr.size) == int(layout_now.runtime_parameter_count):
            return np.asarray(theta_arr, dtype=float), layout_now
        if int(theta_arr.size) == int(layout_now.logical_parameter_count):
            return np.asarray(expand_legacy_logical_theta(theta_arr, layout_now), dtype=float), layout_now
        raise ValueError(
            "ROTOSOLVE runtime theta/layout mismatch: "
            f"got {int(theta_arr.size)} theta values for {int(layout_now.logical_parameter_count)} "
            f"logical operators and {int(layout_now.runtime_parameter_count)} runtime Pauli coordinates."
        )

    def _rotosolve_parameterization_kwargs(
        selected_now: Sequence[_PoolCandidate],
        layout_now: AnsatzParameterLayout | None = None,
    ) -> dict[str, Any]:
        if not rotosolve_runtime_coordinates:
            return {}
        layout_use = layout_now if layout_now is not None else _selected_parameterization_layout(selected_now)
        return {"parameterization_mode": "per_pauli_term", "parameterization_layout": layout_use}

    def _theta_with_inserted_candidate(
        *,
        selected_before: Sequence[_PoolCandidate],
        theta_before: np.ndarray,
        candidate: _PoolCandidate,
        position: int,
    ) -> tuple[np.ndarray, AnsatzParameterLayout | None]:
        if not rotosolve_runtime_coordinates:
            return (
                np.insert(np.asarray(theta_before, dtype=float).reshape(-1), int(position), 0.0),
                None,
            )
        old_theta, old_layout = _runtime_theta_for_selected(selected_before, theta_before)
        pos = max(0, min(int(position), int(old_layout.logical_parameter_count)))
        positioned_selected = list(selected_before[:pos]) + [candidate] + list(selected_before[pos:])
        new_layout = _selected_parameterization_layout(positioned_selected)
        out = np.zeros(int(new_layout.runtime_parameter_count), dtype=float)
        old_logical_idx = 0
        for new_logical_idx, new_block in enumerate(new_layout.blocks):
            if int(new_logical_idx) == int(pos):
                continue
            old_block = old_layout.blocks[int(old_logical_idx)]
            out[int(new_block.runtime_start) : int(new_block.runtime_stop)] = old_theta[
                int(old_block.runtime_start) : int(old_block.runtime_stop)
            ]
            old_logical_idx += 1
        return np.asarray(out, dtype=float), new_layout

    def _theta_with_appended_candidates(
        *,
        selected_before: Sequence[_PoolCandidate],
        theta_before: np.ndarray,
        appended: Sequence[_PoolCandidate],
    ) -> tuple[np.ndarray, AnsatzParameterLayout | None]:
        if not rotosolve_runtime_coordinates:
            return (
                np.concatenate(
                    [
                        np.asarray(theta_before, dtype=float).reshape(-1),
                        np.zeros(len(appended), dtype=float),
                    ]
                ),
                None,
            )
        old_theta, old_layout = _runtime_theta_for_selected(selected_before, theta_before)
        selected_after = list(selected_before) + list(appended)
        new_layout = _selected_parameterization_layout(selected_after)
        added_runtime_count = int(new_layout.runtime_parameter_count) - int(old_layout.runtime_parameter_count)
        if added_runtime_count < 0:
            raise ValueError("ROTOSOLVE append theta layout unexpectedly shrank.")
        return (
            np.concatenate([old_theta, np.zeros(int(added_runtime_count), dtype=float)]),
            new_layout,
        )

    adapt_history: list[dict[str, Any]] = []
    warm_start_continuation_enabled = any(
        value is not None
        for value in (
            initial_selected_operator_labels,
            initial_selected_operator_batches,
            initial_theta,
            initial_adapt_history,
        )
    )
    warm_start_source_iterations = 0
    warm_start_source_depth = 0
    if warm_start_continuation_enabled:
        if initial_selected_operator_labels is None or initial_theta is None:
            raise ValueError("warm-start continuation requires selected operator labels and theta")
        warm_labels = [str(label) for label in initial_selected_operator_labels]
        missing_labels = [label for label in warm_labels if label not in by_label]
        if missing_labels:
            preview = ", ".join(missing_labels[:5])
            raise ValueError(f"warm-start selected operators are absent from the candidate pool: {preview}")
        theta = np.asarray(list(initial_theta), dtype=float).reshape(-1)
        selected = [by_label[label] for label in warm_labels]
        if int(theta.size) != int(len(warm_labels)):
            if not rotosolve_runtime_coordinates:
                raise ValueError(
                    "warm-start selected/theta length mismatch: "
                    f"{len(warm_labels)} operators vs {int(theta.size)} theta values"
                )
            warm_layout = _selected_parameterization_layout(selected)
            if int(theta.size) != int(warm_layout.runtime_parameter_count):
                raise ValueError(
                    "warm-start selected/theta length mismatch: "
                    f"{len(warm_labels)} operators vs {int(theta.size)} theta values "
                    f"and runtime parameter count {int(warm_layout.runtime_parameter_count)}"
                )
        selected_labels = {str(candidate.label) for candidate in selected}
        if initial_selected_operator_batches is None:
            selected_batches = [[candidate] for candidate in selected]
        else:
            selected_batches = []
            flattened_labels: list[str] = []
            for raw_batch in initial_selected_operator_batches:
                batch_labels = [str(label) for label in raw_batch]
                batch_missing = [label for label in batch_labels if label not in by_label]
                if batch_missing:
                    preview = ", ".join(batch_missing[:5])
                    raise ValueError(f"warm-start batch operators are absent from the candidate pool: {preview}")
                selected_batches.append([by_label[label] for label in batch_labels])
                flattened_labels.extend(batch_labels)
            if flattened_labels != warm_labels:
                raise ValueError("warm-start selected_operator_batches do not flatten to selected_operator_labels")
        adapt_history = [dict(item) for item in (initial_adapt_history or ())]
        warm_start_source_iterations = int(len(adapt_history))
        warm_start_source_depth = int(len(selected))
    if rotosolve_runtime_coordinates:
        theta, _initial_runtime_layout = _runtime_theta_for_selected(selected, theta)
    _progress(
        "run_start",
        continuation_mode=("warm_start_selected_theta_v1" if warm_start_continuation_enabled else "fresh"),
        initial_history_len=int(len(adapt_history)),
        initial_depth=int(len(selected)),
        theta_size=int(theta.size),
        max_adapt_iterations=int(max_adapt_iterations),
        gradient_threshold=float(gradient_threshold),
        target_stop_enabled=bool(target_stop_enabled),
        benchmark_energy_stop_target=benchmark_energy_stop_target,
        generic_adapt_runtime_split_mode=runtime_split_mode,
        generic_adapt_runtime_split_symmetry_policy=runtime_split_symmetry_policy,
        generic_adapt_runtime_split_base_pool_term_count=runtime_split_pool_meta.get("base_pool_term_count"),
        generic_adapt_runtime_split_expanded_pool_term_count=runtime_split_pool_meta.get("expanded_pool_term_count"),
        shared_pauli_pool_mode=shared_pool_mode,
        shared_pauli_pool_ordered_pool_hash=shared_pauli_pool_meta.get("ordered_pool_hash"),
        shared_pauli_pool_base_pool_term_count=shared_pauli_pool_meta.get("base_pool_term_count"),
        shared_pauli_pool_expanded_pool_term_count=shared_pauli_pool_meta.get("expanded_pool_term_count"),
    )
    gradient_scan_count = 0
    gradient_probe_count = 0
    gradient_selector_probe_count = 0
    gradient_qngd_refit_probe_count = 0
    gradient_position_trial_probe_count = 0
    metric_probe_count = 0
    metric_selector_probe_count = 0
    metric_qngd_refit_probe_count = 0
    metric_position_trial_probe_count = 0
    outer_hamiltonian_eval_count = 0
    nfev_total = 0
    nit_total = 0
    optimizer_messages: list[str] = []
    optimizer_info_last: dict[str, Any] = {}
    optimizer_success_all = True
    optimizer_raw_success_all = True
    optimizer_capped_iterations: list[int] = []
    optimizer_capped_accepted_iterations: list[int] = []
    last_scored_top: Mapping[str, Any] | None = None
    stop_reason = "max_adapt_iterations"
    progress_best_abs_delta_e: float | None = None
    # The first selector scan computes the first algorithmic Hamiltonian
    # application.  Avoid a redundant uncounted energy call here.
    energy = float("nan")

    for iteration in range(int(len(adapt_history)), int(max_adapt_iterations)):
        history_position = int(len(adapt_history))
        depth_before_iteration = int(len(selected))
        _progress(
            "iteration_start",
            iteration=int(iteration),
            history_len=int(len(adapt_history)),
            depth_before=int(len(selected)),
            theta_size=int(theta.size),
        )
        psi_current = _prepare_selected_state(
            selected=selected,
            theta=theta,
            psi_ref=psi_ref,
            pauli_action_cache=pauli_action_cache,
            **_rotosolve_parameterization_kwargs(selected),
        )
        energy_current, hpsi = energy_via_one_apply(psi_current, h_compiled)
        outer_hamiltonian_eval_count += 1
        scored = _score_candidates(
            config=config,
            psi=psi_current,
            hpsi=hpsi,
            compiled_pool=compiled_pool,
            selected_labels=selected_labels,
            previous_selected_label=(selected[-1].label if selected else None),
            metric_floor=float(metric_floor),
            decision_noise_recorder=decision_noise_recorder,
            adapt_iteration=int(iteration),
        )
        gradient_scan_count += 1
        gradient_probe_count += len(scored)
        gradient_selector_probe_count += len(scored)
        selector_metric_probe_count = (
            int(len(scored) * (len(scored) + 1) // 2) if _is_geo_config(config) else 0
        )
        if selector_metric_probe_count:
            metric_probe_count += selector_metric_probe_count
            metric_selector_probe_count += selector_metric_probe_count
        if scored:
            last_scored_top = scored[0]
        max_abs_gradient = max((float(row["abs_gradient"]) for row in scored), default=0.0)
        max_abs_gradient_decision = max(
            (float(row.get("abs_gradient_decision", row.get("abs_gradient", 0.0))) for row in scored),
            default=0.0,
        )
        gradient_l2_norm = float(
            math.sqrt(sum(float(row.get("gradient", 0.0)) ** 2 for row in scored))
        )
        gradient_l2_norm_decision = float(
            math.sqrt(
                sum(
                    float(row.get("gradient_decision", row.get("gradient", 0.0))) ** 2
                    for row in scored
                )
            )
        )
        best_score = float(scored[0]["selector_score"]) if scored else 0.0
        best_score_decision = float(scored[0].get("selector_score_decision", best_score)) if scored else 0.0
        geo_natural_step_fs_norm = (
            None if not scored else scored[0].get("geo_natural_step_fs_norm")
        )
        geo_natural_step_fs_norm_decision = (
            None if not scored else scored[0].get("geo_natural_step_fs_norm_decision", geo_natural_step_fs_norm)
        )
        selection_gradient_threshold = 0.0 if target_stop_enabled else float(gradient_threshold)
        batch_admission_gradient_threshold = _batch_admission_gradient_threshold(
            config,
            float(selection_gradient_threshold),
        )
        batch = _select_batch(
            config=config,
            scored=scored,
            gradient_threshold=float(selection_gradient_threshold),
            max_tetris_batch_size=int(max_tetris_batch_size),
        )
        previous_selected_label = selected[-1].label if selected else None
        geo_immediate_repeat_skipped = bool(
            config.repeat_policy == "with_replacement_except_immediate_repeat"
            and previous_selected_label is not None
            and len(batch) == 1
            and str(batch[0].get("label")) == str(previous_selected_label)
        )
        appended_batch = [] if geo_immediate_repeat_skipped else list(batch)
        _progress(
            "iteration_scored",
            iteration=int(iteration),
            energy_before=float(energy_current),
            depth_before=int(len(selected)),
            candidate_count_scored=int(len(scored)),
            geo_metric_candidate_count_before_screen=(
                None if not scored or not _is_geo_config(config) else scored[0].get("geo_metric_candidate_count_before_screen")
            ),
            geo_metric_candidate_count_after_screen=(
                None if not scored or not _is_geo_config(config) else scored[0].get("geo_metric_candidate_count_after_screen")
            ),
            max_abs_gradient=float(max_abs_gradient),
            max_abs_gradient_decision=float(max_abs_gradient_decision),
            gradient_l2_norm=float(gradient_l2_norm),
            gradient_l2_norm_decision=float(gradient_l2_norm_decision),
            best_selector_score=float(best_score),
            selection_gradient_threshold=float(selection_gradient_threshold),
            batch_admission_gradient_threshold=batch_admission_gradient_threshold,
            selected_candidate_count=int(len(batch)),
            selected_candidate_labels=[str(row["label"]) for row in batch],
            appended_batch_size=int(len(appended_batch)),
            appended_batch_labels=[str(row["label"]) for row in appended_batch],
            batch_size=int(len(appended_batch)),
            batch_labels=[str(row["label"]) for row in appended_batch],
            geo_immediate_repeat_skipped=bool(geo_immediate_repeat_skipped),
            top_label=(None if not scored else str(scored[0].get("label"))),
        )
        if not scored:
            stop_reason = "pool_exhausted"
            energy = float(energy_current)
            _progress("stop", iteration=int(iteration), reason=str(stop_reason), energy=float(energy))
            break
        if config.stop_rule == "geo_natural_gradient_norm":
            if (
                not target_stop_enabled
                and float(geo_natural_step_fs_norm_decision or 0.0) < float(gradient_threshold)
            ):
                stop_reason = "geo_natural_gradient_norm_threshold"
                energy = float(energy_current)
                _progress("stop", iteration=int(iteration), reason=str(stop_reason), energy=float(energy))
                break
        elif (
            config.stop_rule == "raw_gradient"
            and not target_stop_enabled
            and max_abs_gradient_decision < float(gradient_threshold)
        ):
            stop_reason = "gradient_threshold"
            energy = float(energy_current)
            _progress("stop", iteration=int(iteration), reason=str(stop_reason), energy=float(energy))
            break
        if not batch:
            stop_reason = "tetris_no_compatible_candidate" if config.variant == "tetris" else "no_candidate_selected"
            energy = float(energy_current)
            _progress("stop", iteration=int(iteration), reason=str(stop_reason), energy=float(energy))
            break

        new_candidates = [by_label[str(row["label"])] for row in appended_batch]
        _progress(
            "iteration_selected",
            iteration=int(iteration),
            selected_candidate_labels=[str(row["label"]) for row in batch],
            appended_batch_labels=[str(candidate.label) for candidate in new_candidates],
            appended_batch_size=int(len(new_candidates)),
            selected_batch_labels=[str(candidate.label) for candidate in new_candidates],
            batch_size=int(len(new_candidates)),
            geo_immediate_repeat_skipped=bool(geo_immediate_repeat_skipped),
            depth_before=int(len(selected)),
            optimizer_kind=str(config.optimizer_kind),
        )
        position_policy = str(config.position_policy)
        position_trials: list[dict[str, Any]] = []
        qngd_metric_event_blocks: list[dict[str, Any]] = []
        inserted_position: int | None = None

        if position_policy == "best_insert_refit" and len(new_candidates) == 1:
            candidate = new_candidates[0]
            best_payload: tuple[float, float, int, list[_PoolCandidate], np.ndarray, dict[str, Any]] | None = None
            aggregate_nfev = 0
            aggregate_nit = 0
            aggregate_metric_probes = 0
            aggregate_gradient_probes = 0
            aggregate_success_all = True
            # Pos-Geo insertion: choose the operator from the natural-gradient rule,
            # then test every insertion position by refitting the full ansatz.
            for pos in range(len(selected) + 1):
                positioned_selected = list(selected[:pos]) + [candidate] + list(selected[pos:])
                x0, positioned_layout = _theta_with_inserted_candidate(
                    selected_before=selected,
                    theta_before=theta,
                    candidate=candidate,
                    position=int(pos),
                )
                if config.optimizer_kind == "geo_qngd":
                    trial_theta, trial_energy, trial_info = _optimize_selected_qngd(
                        selected=positioned_selected,
                        x0=x0,
                        psi_ref=psi_ref,
                        h_compiled=h_compiled,
                        pauli_action_cache=pauli_action_cache,
                        optimizer_maxiter=int(optimizer_maxiter),
                        metric_floor=float(metric_floor),
                        spsa_seed=int(spsa_seed_base) + 104729 * (int(iteration) + 1) + 1009 * (int(pos) + 1),
                        decision_noise_recorder=decision_noise_recorder,
                        decision_scope={
                            "adapt_iteration": int(iteration),
                            "position_trial": int(pos),
                            "selected_operator_count": int(len(positioned_selected)),
                            "selected_labels": tuple(str(item.label) for item in positioned_selected),
                        },
                    )
                elif config.optimizer_kind == "spsa":
                    trial_theta, trial_energy, trial_info = _optimize_selected_spsa(
                        selected=positioned_selected,
                        x0=x0,
                        psi_ref=psi_ref,
                        h_compiled=h_compiled,
                        pauli_action_cache=pauli_action_cache,
                        optimizer_maxiter=int(optimizer_maxiter),
                        spsa_seed=int(spsa_seed_base) + 104729 * (int(iteration) + 1) + 1009 * (int(pos) + 1),
                        spsa_schedule=spsa_schedule,
                        decision_noise_recorder=decision_noise_recorder,
                        decision_scope={
                            "adapt_iteration": int(iteration),
                            "position_trial": int(pos),
                            "selected_operator_count": int(len(positioned_selected)),
                            "selected_labels": tuple(str(item.label) for item in positioned_selected),
                        },
                    )
                else:
                    trial_theta, trial_energy, trial_info = _optimize_selected(
                        minimize_fn=minimize_fn,
                        selected=positioned_selected,
                        x0=x0,
                        psi_ref=psi_ref,
                        h_compiled=h_compiled,
                        pauli_action_cache=pauli_action_cache,
                        optimizer_maxiter=int(optimizer_maxiter),
                        optimizer_method=_scipy_or_coordinate_optimizer_method(config),
                        **_rotosolve_parameterization_kwargs(positioned_selected, positioned_layout),
                        decision_noise_recorder=decision_noise_recorder,
                        decision_scope={
                            "adapt_iteration": int(iteration),
                            "position_trial": int(pos),
                            "selected_operator_count": int(len(positioned_selected)),
                            "selected_labels": tuple(str(item.label) for item in positioned_selected),
                        },
                    )
                aggregate_nfev += int(trial_info.get("nfev") or 0)
                aggregate_success_all = bool(
                    aggregate_success_all and bool(trial_info.get("success", False))
                )
                if trial_info.get("nit") is not None:
                    aggregate_nit += int(trial_info.get("nit") or 0)
                aggregate_metric_probes += int(trial_info.get("qngd_metric_operator_probe_count_total") or 0)
                aggregate_gradient_probes += int(
                    trial_info.get("qngd_gradient_operator_probe_count_total") or 0
                )
                trial_metric_eval_count = int(trial_info.get("qngd_metric_eval_count") or 0)
                trial_metric_probe_count = int(trial_info.get("qngd_metric_operator_probe_count_total") or 0)
                trial_gradient_probe_count = int(
                    trial_info.get("qngd_gradient_operator_probe_count_total") or 0
                )
                trial_decision_energy = _decision_value(
                    decision_noise_recorder,
                    float(trial_energy),
                    surface="adapt_pos_geo_position_trial_energy",
                    value_kind="energy",
                    phase="position_trial_selection",
                    extra_scope={
                        "adapt_iteration": int(iteration),
                        "position": int(pos),
                        "candidate_label": str(candidate.label),
                        "selected_operator_count": int(len(positioned_selected)),
                        "selected_labels": tuple(str(item.label) for item in positioned_selected),
                    },
                )
                trial_record = {
                    "position": int(pos),
                    "energy": float(trial_energy),
                    "energy_exact": float(trial_energy),
                    "success": bool(trial_info.get("success", False)),
                    "message": str(trial_info.get("message", "")),
                    "nfev": int(trial_info.get("nfev") or 0),
                    "nit": trial_info.get("nit"),
                    "optimizer": str(trial_info.get("optimizer", "")),
                    "qngd_metric_eval_count": trial_metric_eval_count,
                    "qngd_metric_operator_probe_count_total": trial_metric_probe_count,
                    "qngd_gradient_operator_probe_count_total": trial_gradient_probe_count,
                    "qngd_fallback_optimizer": trial_info.get("qngd_fallback_optimizer"),
                    "qngd_spsa_polish_attempted": trial_info.get("qngd_spsa_polish_attempted"),
                    "qngd_spsa_polish_success": trial_info.get("qngd_spsa_polish_success"),
                    "qngd_bfgs_polish_attempted": trial_info.get("qngd_bfgs_polish_attempted"),
                }
                if bool(decision_noise_config.enabled):
                    trial_record["energy_decision"] = float(trial_decision_energy)
                    trial_record["position_trial_decision_surface"] = "adapt_pos_geo_position_trial_energy"
                position_trials.append(trial_record)
                if config.optimizer_kind == "geo_qngd":
                    qngd_metric_event_blocks.append(
                        {
                            "block_kind": "pos_geo_position_trial_qngd_metric",
                            "position": int(pos),
                            "selected_labels": [str(item.label) for item in positioned_selected],
                            "parameter_count": int(len(positioned_selected)),
                            "metric_eval_count": trial_metric_eval_count,
                            "metric_operator_probe_count": trial_metric_probe_count,
                            "gradient_operator_probe_count": trial_gradient_probe_count,
                            "metric_pair_count_per_eval": int(
                                len(positioned_selected) * (len(positioned_selected) + 1) // 2
                            ),
                        }
                    )
                if best_payload is None or float(trial_decision_energy) < best_payload[0]:
                    best_payload = (
                        float(trial_decision_energy),
                        float(trial_energy),
                        int(pos),
                        positioned_selected,
                        np.asarray(trial_theta, dtype=float),
                        dict(trial_info),
                    )
            if best_payload is None:  # pragma: no cover - defensive guard
                stop_reason = "pos_geo_no_position_trial"
                energy = float(energy_current)
                break
            selected_position_decision_energy, energy, inserted_position, selected, theta, opt_info = best_payload
            opt_info = dict(opt_info)
            opt_info["nfev"] = int(aggregate_nfev)
            opt_info["nit"] = int(aggregate_nit)
            opt_info["success"] = bool(aggregate_success_all)
            if not aggregate_success_all:
                opt_info["message"] = "one_or_more_position_trial_optimizers_failed"
            if config.optimizer_kind == "geo_qngd":
                opt_info["qngd_metric_operator_probe_count_total"] = int(aggregate_metric_probes)
                opt_info["qngd_gradient_operator_probe_count_total"] = int(aggregate_gradient_probes)
                metric_position_trial_probe_count += int(aggregate_metric_probes)
                gradient_position_trial_probe_count += int(aggregate_gradient_probes)
            opt_info["position_trial_count"] = int(len(position_trials))
            opt_info["selected_insertion_position"] = int(inserted_position)
            opt_info["selected_position_exact_energy"] = float(energy)
            opt_info["selected_position_decision_energy"] = float(selected_position_decision_energy)
            opt_info["pos_geo_position_trial_spsa_attempt_count"] = sum(
                1 for trial in position_trials if bool(trial.get("qngd_spsa_polish_attempted"))
            )
            opt_info["pos_geo_position_trial_spsa_success_count"] = sum(
                1 for trial in position_trials if bool(trial.get("qngd_spsa_polish_success"))
            )
            opt_info["pos_geo_position_trial_bfgs_attempt_count"] = sum(
                1 for trial in position_trials if bool(trial.get("qngd_bfgs_polish_attempted"))
            )
            selected_labels.add(candidate.label)
            selected_batches.append([candidate])
        else:
            selected_before_refit = list(selected)
            theta_before_refit = np.asarray(theta, dtype=float).reshape(-1)
            for candidate in new_candidates:
                selected.append(candidate)
                selected_labels.add(candidate.label)
            if new_candidates:
                selected_batches.append(list(new_candidates))
            x0, selected_layout_for_refit = _theta_with_appended_candidates(
                selected_before=selected_before_refit,
                theta_before=theta_before_refit,
                appended=new_candidates,
            )
            if config.optimizer_kind == "geo_qngd":
                theta, energy, opt_info = _optimize_selected_qngd(
                    selected=selected,
                    x0=x0,
                    psi_ref=psi_ref,
                    h_compiled=h_compiled,
                    pauli_action_cache=pauli_action_cache,
                    optimizer_maxiter=int(optimizer_maxiter),
                    metric_floor=float(metric_floor),
                    spsa_seed=int(spsa_seed_base) + 104729 * (int(iteration) + 1),
                    decision_noise_recorder=decision_noise_recorder,
                    decision_scope={
                        "adapt_iteration": int(iteration),
                        "selected_operator_count": int(len(selected)),
                        "selected_labels": tuple(str(item.label) for item in selected),
                    },
                )
            elif config.optimizer_kind == "spsa":
                theta, energy, opt_info = _optimize_selected_spsa(
                    selected=selected,
                    x0=x0,
                    psi_ref=psi_ref,
                    h_compiled=h_compiled,
                    pauli_action_cache=pauli_action_cache,
                    optimizer_maxiter=int(optimizer_maxiter),
                    spsa_seed=int(spsa_seed_base) + 104729 * (int(iteration) + 1),
                    spsa_schedule=spsa_schedule,
                    decision_noise_recorder=decision_noise_recorder,
                    decision_scope={
                        "adapt_iteration": int(iteration),
                        "selected_operator_count": int(len(selected)),
                        "selected_labels": tuple(str(item.label) for item in selected),
                    },
                )
            else:
                theta, energy, opt_info = _optimize_selected(
                    minimize_fn=minimize_fn,
                    selected=selected,
                    x0=x0,
                    psi_ref=psi_ref,
                    h_compiled=h_compiled,
                    pauli_action_cache=pauli_action_cache,
                    optimizer_maxiter=int(optimizer_maxiter),
                    optimizer_method=_scipy_or_coordinate_optimizer_method(config),
                    **_rotosolve_parameterization_kwargs(selected, selected_layout_for_refit),
                    decision_noise_recorder=decision_noise_recorder,
                    decision_scope={
                        "adapt_iteration": int(iteration),
                        "selected_operator_count": int(len(selected)),
                        "selected_labels": tuple(str(item.label) for item in selected),
                    },
                )
            if config.optimizer_kind == "geo_qngd":
                qngd_metric_event_blocks.append(
                    {
                        "block_kind": "geo_qngd_refit_metric",
                        "selected_labels": [str(item.label) for item in selected],
                        "parameter_count": int(len(selected)),
                        "metric_eval_count": int(opt_info.get("qngd_metric_eval_count") or 0),
                        "metric_operator_probe_count": int(
                            opt_info.get("qngd_metric_operator_probe_count_total") or 0
                        ),
                        "gradient_operator_probe_count": int(
                            opt_info.get("qngd_gradient_operator_probe_count_total") or 0
                        ),
                        "metric_pair_count_per_eval": int(len(selected) * (len(selected) + 1) // 2),
                    }
                )
                metric_qngd_refit_probe_count += int(
                    opt_info.get("qngd_metric_operator_probe_count_total") or 0
                )
                gradient_qngd_refit_probe_count += int(
                    opt_info.get("qngd_gradient_operator_probe_count_total") or 0
                )

        opt_info = _classify_powell_maxiter_cap(
            config=config,
            policy=powell_cap_policy,
            theta=np.asarray(theta, dtype=float),
            energy_before=float(energy_current),
            energy_after=float(energy),
            optimizer_maxiter=int(optimizer_maxiter),
            opt_info=opt_info,
        )
        nfev_total += int(opt_info.get("nfev") or 0)
        if opt_info.get("nit") is not None:
            nit_total += int(opt_info["nit"])
        if config.optimizer_kind == "geo_qngd":
            metric_probe_count += int(opt_info.get("qngd_metric_operator_probe_count_total") or 0)
            gradient_probe_count += int(opt_info.get("qngd_gradient_operator_probe_count_total") or 0)
        optimizer_raw_success_all = bool(
            optimizer_raw_success_all and bool(opt_info.get("raw_success", False))
        )
        optimizer_success_all = bool(optimizer_success_all and bool(opt_info.get("success", False)))
        if bool(opt_info.get("optimizer_capped", False)):
            optimizer_capped_iterations.append(int(iteration))
        if bool(opt_info.get("optimizer_capped_accepted", False)):
            optimizer_capped_accepted_iterations.append(int(iteration))
        optimizer_messages.append(str(opt_info.get("message", "")))
        optimizer_info_last = dict(opt_info)
        optimizer_label = str(opt_info.get("optimizer") or "scipy.optimize.minimize:BFGS")
        selector_metric_candidate_labels = [str(row["label"]) for row in scored] if _is_geo_config(config) else []
        adapt_history.append(
            {
                "iteration": int(iteration),
                "history_position": history_position,
                "depth_before": depth_before_iteration,
                "depth_after": int(len(selected)),
                "appended_operator_count": int(len(new_candidates)),
                "energy_before": float(energy_current),
                "energy_after": float(energy),
                "max_abs_gradient": float(max_abs_gradient),
                "gradient_l2_norm": float(gradient_l2_norm),
                "best_selector_score": float(best_score),
                "selected_candidate_labels": [str(row["label"]) for row in batch],
                "selected_candidate_supports": [list(row.get("support", [])) for row in batch],
                "selected_candidate_execution_modes": [
                    str(row.get("execution_mode", "termwise_product")) for row in batch
                ],
                "batch_size": int(len(new_candidates)),
                "selected_batch_labels": [str(candidate.label) for candidate in new_candidates],
                "selected_batch_supports": [list(candidate.support) for candidate in new_candidates],
                "selected_batch_execution_modes": [
                    str(
                        getattr(candidate, "execution_mode", "termwise_product")
                        or "termwise_product"
                    )
                    for candidate in new_candidates
                ],
                "geo_immediate_repeat_skipped": bool(geo_immediate_repeat_skipped),
                "geo_immediate_repeat_label": (
                    str(previous_selected_label) if geo_immediate_repeat_skipped else None
                ),
                "candidate_count_scored": int(len(scored)),
                "selector_metric_candidate_labels": selector_metric_candidate_labels,
                "selector_metric_probe_count": selector_metric_probe_count,
                "selector_gradient_probe_count": int(len(scored)),
                "outer_hamiltonian_eval_count": 1,
                "geo_metric_candidate_count_before_screen": (
                    None
                    if not scored or not _is_geo_config(config)
                    else scored[0].get("geo_metric_candidate_count_before_screen")
                ),
                "geo_metric_candidate_count_after_screen": (
                    None
                    if not scored or not _is_geo_config(config)
                    else scored[0].get("geo_metric_candidate_count_after_screen")
                ),
                "position_policy": position_policy,
                "selected_insertion_position": inserted_position,
                "position_trial_count": int(len(position_trials)),
                "position_trials": position_trials[: min(10, len(position_trials))],
                "qngd_metric_event_blocks": qngd_metric_event_blocks,
                "optimizer": optimizer_label,
                "geo_stop_rule": _GEO_STOP_RULE if config.stop_rule == "geo_natural_gradient_norm" else "raw_gradient",
                "raw_gradient_used_for_stop": bool(_uses_raw_gradient_stop(config)),
                "raw_gradient_stop_rule": (
                    str(config.stop_rule) if _uses_raw_gradient_stop(config) else None
                ),
                "geo_natural_step_fs_norm": geo_natural_step_fs_norm,
                "optimizer_success": bool(opt_info.get("success", False)),
                "optimizer_raw_success": bool(opt_info.get("raw_success", False)),
                "optimizer_capped": bool(opt_info.get("optimizer_capped", False)),
                "optimizer_capped_accepted": bool(
                    opt_info.get("optimizer_capped_accepted", False)
                ),
                "optimizer_cap_policy": str(opt_info.get("optimizer_cap_policy", "")),
                "optimizer_cap_acceptance_reason": str(
                    opt_info.get("optimizer_cap_acceptance_reason", "")
                ),
                "optimizer_cap_status": opt_info.get("optimizer_cap_status"),
                "optimizer_cap_nit": opt_info.get("optimizer_cap_nit"),
                "optimizer_cap_maxiter": opt_info.get("optimizer_cap_maxiter"),
                "optimizer_cap_message_match": bool(
                    opt_info.get("optimizer_cap_message_match", False)
                ),
                "optimizer_cap_parameters_finite": bool(
                    opt_info.get("optimizer_cap_parameters_finite", False)
                ),
                "optimizer_cap_objective_finite": bool(
                    opt_info.get("optimizer_cap_objective_finite", False)
                ),
                "optimizer_cap_energy_nonincreasing": bool(
                    opt_info.get("optimizer_cap_energy_nonincreasing", False)
                ),
                "optimizer_cap_energy_before": opt_info.get("optimizer_cap_energy_before"),
                "optimizer_cap_energy_after": opt_info.get("optimizer_cap_energy_after"),
                "optimizer_cap_energy_delta": opt_info.get("optimizer_cap_energy_delta"),
                "optimizer_cap_energy_rel_tol": opt_info.get("optimizer_cap_energy_rel_tol"),
                "optimizer_cap_energy_abs_tol": opt_info.get("optimizer_cap_energy_abs_tol"),
                "optimizer_message": str(opt_info.get("message", "")),
                "optimizer_nfev": int(opt_info.get("nfev") or 0),
                "optimizer_nit": opt_info.get("nit"),
                "prefix_state_cache_enabled": bool(opt_info.get("prefix_state_cache_enabled", False)),
                "prefix_state_cache_id": opt_info.get("prefix_state_cache_id"),
                "grouped_exact_plan_cache_id": opt_info.get("grouped_exact_plan_cache_id"),
                "prefix_state_cache_evaluation_count": int(
                    opt_info.get("prefix_state_cache_evaluation_count") or 0
                ),
                "prefix_state_cache_hit_count": int(opt_info.get("prefix_state_cache_hit_count") or 0),
                "prefix_state_cache_reused_operator_count": int(
                    opt_info.get("prefix_state_cache_reused_operator_count") or 0
                ),
                "rotosolve_accepted_steps": opt_info.get("accepted_steps"),
                "rotosolve_stencil_source": opt_info.get("rotosolve_stencil_source"),
                "rotosolve_period": opt_info.get("rotosolve_period"),
                "rotosolve_shift": opt_info.get("rotosolve_shift"),
                "qngd_accepted_step_count": opt_info.get("qngd_accepted_step_count"),
                "qngd_energy_decrease_total": opt_info.get("qngd_energy_decrease_total"),
                "qngd_metric_eval_count": opt_info.get("qngd_metric_eval_count"),
                "qngd_metric_operator_probe_count_total": opt_info.get("qngd_metric_operator_probe_count_total"),
                "qngd_gradient_operator_probe_count_total": opt_info.get(
                    "qngd_gradient_operator_probe_count_total"
                ),
                "qngd_fallback_optimizer": opt_info.get("qngd_fallback_optimizer"),
                "qngd_spsa_polish_attempted": opt_info.get("qngd_spsa_polish_attempted"),
                "qngd_spsa_polish_success": opt_info.get("qngd_spsa_polish_success"),
                "qngd_spsa_polish_nfev": opt_info.get("qngd_spsa_polish_nfev"),
                "qngd_spsa_polish_nit": opt_info.get("qngd_spsa_polish_nit"),
                "qngd_bfgs_polish_attempted": opt_info.get("qngd_bfgs_polish_attempted"),
                "spsa_seed": opt_info.get("spsa_seed"),
                "spsa_accepted_step_count": opt_info.get("spsa_accepted_step_count"),
                "spsa_energy_decrease_total": opt_info.get("spsa_energy_decrease_total"),
                "pos_geo_position_trial_spsa_attempt_count": opt_info.get("pos_geo_position_trial_spsa_attempt_count"),
                "pos_geo_position_trial_spsa_success_count": opt_info.get("pos_geo_position_trial_spsa_success_count"),
                "pos_geo_position_trial_bfgs_attempt_count": opt_info.get("pos_geo_position_trial_bfgs_attempt_count"),
                "top_candidates": list(scored[: min(5, len(scored))]),
            }
        )
        if bool(decision_noise_config.enabled) and adapt_history:
            decision_history_fields = {
                "max_abs_gradient_decision": float(max_abs_gradient_decision),
                "gradient_l2_norm_decision": float(gradient_l2_norm_decision),
                "best_selector_score_decision": float(best_score_decision),
                "geo_natural_step_fs_norm_decision": geo_natural_step_fs_norm_decision,
                "optimizer_decision_energy": opt_info.get("optimizer_decision_energy"),
                "optimizer_exact_energy": opt_info.get("optimizer_exact_energy"),
                "optimizer_decision_surface": opt_info.get("optimizer_decision_surface"),
                "selected_position_exact_energy": opt_info.get("selected_position_exact_energy"),
                "selected_position_decision_energy": opt_info.get("selected_position_decision_energy"),
            }
            adapt_history[-1].update(
                {key: value for key, value in decision_history_fields.items() if value is not None}
            )
        progress_error_fields = _energy_error_fields(float(energy), reference_metrics)
        progress_abs_delta_e = _finite_float_or_none(progress_error_fields.get("abs_delta_e"))
        if progress_abs_delta_e is not None:
            progress_best_abs_delta_e = (
                progress_abs_delta_e
                if progress_best_abs_delta_e is None
                else min(progress_best_abs_delta_e, progress_abs_delta_e)
            )
        _progress(
            "iteration_complete",
            iteration=int(iteration),
            energy_after=float(energy),
            abs_delta_e=progress_abs_delta_e,
            current_best_abs_delta_e=progress_best_abs_delta_e,
            depth_after=int(len(selected)),
            history_len=int(len(adapt_history)),
            theta_size=int(theta.size),
            appended_batch_size=int(len(new_candidates)),
            appended_batch_labels=[str(candidate.label) for candidate in new_candidates],
            batch_size=int(len(new_candidates)),
            selected_batch_labels=[str(candidate.label) for candidate in new_candidates],
            geo_immediate_repeat_skipped=bool(geo_immediate_repeat_skipped),
            optimizer=str(optimizer_label),
            optimizer_success=bool(opt_info.get("success", False)),
            optimizer_raw_success=bool(opt_info.get("raw_success", False)),
            optimizer_capped=bool(opt_info.get("optimizer_capped", False)),
            optimizer_capped_accepted=bool(opt_info.get("optimizer_capped_accepted", False)),
            optimizer_cap_acceptance_reason=str(
                opt_info.get("optimizer_cap_acceptance_reason", "")
            ),
            optimizer_nfev=int(opt_info.get("nfev") or 0),
            optimizer_nit=opt_info.get("nit"),
            nfev_total=int(nfev_total),
            nit_total=int(nit_total),
        )
        if not bool(opt_info.get("success", False)):
            stop_reason = _optimizer_failure_reason(config)
            adapt_history[-1]["eligible_for_first_hit"] = False
            _progress("stop", iteration=int(iteration), reason=str(stop_reason), energy=float(energy))
            break
        adapt_history[-1]["eligible_for_first_hit"] = True
        first_hit_candidates.append(
            {
                "iteration": int(iteration),
                "energy": float(energy),
                "selected": list(selected),
                "selected_batches": [list(batch_items) for batch_items in selected_batches],
                "nfev_total": int(nfev_total),
                "outer_hamiltonian_eval_count": int(outer_hamiltonian_eval_count),
                "gradient_scan_count": int(gradient_scan_count),
                "gradient_probe_count": int(gradient_probe_count),
                "gradient_selector_probe_count": int(gradient_selector_probe_count),
                "gradient_qngd_refit_probe_count": int(gradient_qngd_refit_probe_count),
                "gradient_position_trial_probe_count": int(gradient_position_trial_probe_count),
                "metric_probe_count": int(metric_probe_count),
                "metric_selector_probe_count": int(metric_selector_probe_count),
                "metric_qngd_refit_probe_count": int(metric_qngd_refit_probe_count),
                "metric_position_trial_probe_count": int(metric_position_trial_probe_count),
            }
        )
        if target_stop_enabled:
            after_error_fields = _energy_error_fields(float(energy), reference_metrics)
            abs_delta_after = after_error_fields.get("abs_delta_e")
            if abs_delta_after is not None and float(abs_delta_after) <= float(benchmark_energy_stop_target):
                stop_reason = "benchmark_abs_delta_e_target"
                _progress(
                    "target_hit",
                    iteration=int(iteration),
                    reason=str(stop_reason),
                    energy=float(energy),
                    abs_delta_e=float(abs_delta_after),
                    benchmark_energy_stop_target=float(benchmark_energy_stop_target),
                )
                break

    else:
        if int(max_adapt_iterations) <= 0:
            stop_reason = "max_adapt_iterations_zero"
        else:
            stop_reason = "max_adapt_iterations"

    psi_final = _prepare_selected_state(
        selected=selected,
        theta=theta,
        psi_ref=psi_ref,
        pauli_action_cache=pauli_action_cache,
        **_rotosolve_parameterization_kwargs(selected),
    )
    energy, hpsi_final = energy_via_one_apply(psi_final, h_compiled)
    terminal_scored = _score_candidates(
        config=config,
        psi=psi_final,
        hpsi=hpsi_final,
        compiled_pool=compiled_pool,
        selected_labels=selected_labels,
        previous_selected_label=(selected[-1].label if selected else None),
        metric_floor=float(metric_floor),
        decision_noise_recorder=None,
        adapt_iteration=int(len(adapt_history)),
    )
    terminal_max_abs_gradient = max(
        (float(row.get("abs_gradient", 0.0)) for row in terminal_scored),
        default=0.0,
    )
    terminal_gradient_l2_norm = float(
        math.sqrt(sum(float(row.get("gradient", 0.0)) ** 2 for row in terminal_scored))
    )
    terminal_best_selector_score = (
        float(terminal_scored[0].get("selector_score", 0.0)) if terminal_scored else 0.0
    )
    terminal_geo_natural_step_fs_norm = (
        None
        if not terminal_scored or not _is_geo_config(config)
        else terminal_scored[0].get("geo_natural_step_fs_norm")
    )
    terminal_selection_gradient_threshold = 0.0 if target_stop_enabled else float(gradient_threshold)
    terminal_batch_admission_gradient_threshold = _batch_admission_gradient_threshold(
        config,
        float(terminal_selection_gradient_threshold),
    )
    terminal_batch = _select_batch(
        config=config,
        scored=terminal_scored,
        gradient_threshold=float(terminal_selection_gradient_threshold),
        max_tetris_batch_size=int(max_tetris_batch_size),
    )
    if str(stop_reason) == "benchmark_abs_delta_e_target":
        terminal_stop_condition = "target_energy_reached"
    elif str(stop_reason) == "max_adapt_iterations":
        terminal_stop_condition = "max_adapt_iterations"
    elif not terminal_scored:
        terminal_stop_condition = "pool_exhausted_after_repeat_filter"
    elif config.stop_rule == "geo_natural_gradient_norm" and float(terminal_geo_natural_step_fs_norm or 0.0) < float(gradient_threshold):
        terminal_stop_condition = "geo_natural_gradient_norm_threshold"
    elif config.stop_rule == "raw_gradient" and float(terminal_max_abs_gradient) < float(gradient_threshold):
        terminal_stop_condition = "gradient_threshold"
    elif not terminal_batch:
        terminal_stop_condition = "no_admissible_candidate_after_filtering"
    else:
        terminal_stop_condition = "nonterminal_candidate_remains"
    if nfev_total <= 0:
        nfev_total = 1
    exact_energy = _finite_float_or_none(reference_metrics.get("same_cutoff_exact_gs_energy"))
    if exact_energy is None:
        exact_energy = _finite_float_or_none(reference_metrics.get("exact_reference_energy"))
    if _finite_float_or_none(reference_metrics.get("primary_reference_energy")) is None:
        exact_energy = _safe_exact_energy(context)
        reference_metrics = _normalize_reference_metric_fields(
            same_cutoff_exact_gs_energy=same_cutoff_exact_gs_energy,
            exact_reference_energy=exact_reference_energy,
            exact_reference_n_ph_max=exact_reference_n_ph_max,
            primary_energy_metric=primary_energy_metric,
            same_cutoff_error_role=same_cutoff_error_role,
            fallback_same_cutoff_energy=exact_energy,
        )
    terminal_error_fields = _energy_error_fields(float(energy), reference_metrics)
    abs_delta = terminal_error_fields.get("abs_delta_e")
    terminal_abs_delta = _finite_float_or_none(abs_delta)
    if terminal_abs_delta is not None:
        progress_best_abs_delta_e = (
            terminal_abs_delta
            if progress_best_abs_delta_e is None
            else min(progress_best_abs_delta_e, terminal_abs_delta)
        )
    _progress(
        "terminal",
        reason=str(stop_reason),
        terminal_stop_condition=str(terminal_stop_condition),
        energy=float(energy),
        abs_delta_e=terminal_abs_delta,
        current_best_abs_delta_e=progress_best_abs_delta_e,
        history_len=int(len(adapt_history)),
        depth=int(len(selected)),
        terminal_candidate_count=int(len(terminal_scored)),
        terminal_max_abs_gradient=float(terminal_max_abs_gradient),
        terminal_gradient_l2_norm=float(terminal_gradient_l2_norm),
        terminal_admissible_batch_size=int(len(terminal_batch)),
    )
    primary_reference_energy = _finite_float_or_none(reference_metrics.get("primary_reference_energy"))
    if primary_reference_energy is not None:
        for candidate in first_hit_candidates:
            iteration_index = int(candidate["iteration"])
            candidate_error_fields = _energy_error_fields(float(candidate["energy"]), reference_metrics)
            abs_delta_after = candidate_error_fields.get("abs_delta_e")
            if 0 <= iteration_index < len(adapt_history):
                adapt_history[iteration_index]["delta_E_abs_after"] = abs_delta_after
                adapt_history[iteration_index]["abs_delta_e_after"] = abs_delta_after
                if "abs_delta_e_same_cutoff" in candidate_error_fields:
                    adapt_history[iteration_index]["abs_delta_e_same_cutoff_after"] = candidate_error_fields["abs_delta_e_same_cutoff"]
                if "abs_delta_e_reference" in candidate_error_fields:
                    adapt_history[iteration_index]["abs_delta_e_reference_after"] = candidate_error_fields["abs_delta_e_reference"]
                adapt_history[iteration_index]["primary_energy_metric_after"] = reference_metrics.get("primary_energy_metric")
            if abs_delta_after is None:
                continue
            for threshold in benchmark_first_hit_thresholds:
                threshold_value = float(threshold)
                if threshold_value not in first_hits and float(abs_delta_after) <= threshold_value:
                    first_hits[threshold_value] = _first_hit_record(
                        threshold=threshold_value,
                        iteration=iteration_index,
                        energy=float(candidate["energy"]),
                        reference=reference_metrics,
                        selected=candidate["selected"],
                        selected_batches=candidate["selected_batches"],
                        config=config,
                        hamiltonian_pauli_term_count=hamiltonian_pauli_term_count,
                        pool_term_count=len(pool),
                        nfev_total=int(candidate["nfev_total"]),
                        outer_hamiltonian_eval_count=int(
                            candidate["outer_hamiltonian_eval_count"]
                        ),
                        gradient_scan_count=int(candidate["gradient_scan_count"]),
                        gradient_probe_count=int(candidate["gradient_probe_count"]),
                        metric_probe_count=int(candidate["metric_probe_count"]),
                        metric_selector_probe_count=int(candidate["metric_selector_probe_count"]),
                        metric_qngd_refit_probe_count=int(candidate["metric_qngd_refit_probe_count"]),
                        metric_position_trial_probe_count=int(candidate["metric_position_trial_probe_count"]),
                        gradient_selector_probe_count=int(candidate["gradient_selector_probe_count"]),
                        gradient_qngd_refit_probe_count=int(
                            candidate["gradient_qngd_refit_probe_count"]
                        ),
                        gradient_position_trial_probe_count=int(
                            candidate["gradient_position_trial_probe_count"]
                        ),
                        shots_per_pauli_term_proxy=int(shots_per_pauli_term_proxy),
                        num_qubits=int(num_qubits),
                        reference_state=psi_ref,
                    )
    sector = _sector_or_unavailable(context, psi_final)
    shot_proxy = _shot_proxy_fields(
        hamiltonian_pauli_term_count=hamiltonian_pauli_term_count,
        pool_term_count=len(pool),
        energy_eval_count=int(nfev_total) + int(outer_hamiltonian_eval_count),
        gradient_scan_count=gradient_scan_count,
        gradient_operator_probe_count=gradient_probe_count,
        metric_operator_probe_count=metric_probe_count,
        shots_per_pauli_term_proxy=int(shots_per_pauli_term_proxy),
    )
    component_fields = _measurement_component_fields(
        selected_operator_count=len(selected),
        energy_eval_count=nfev_total,
        outer_hamiltonian_eval_count=outer_hamiltonian_eval_count,
        gradient_operator_probe_count=gradient_probe_count,
        metric_operator_probe_count=metric_probe_count,
        metric_selector_probe_count=metric_selector_probe_count,
        metric_qngd_refit_probe_count=metric_qngd_refit_probe_count,
        metric_position_trial_probe_count=metric_position_trial_probe_count,
        gradient_selector_probe_count=gradient_selector_probe_count,
        gradient_qngd_refit_probe_count=gradient_qngd_refit_probe_count,
        gradient_position_trial_probe_count=gradient_position_trial_probe_count,
    )
    compiled_stats = _qiskit_final_ansatz_stats_for_selected(
        selected,
        selected_batches=selected_batches,
        config=config,
        num_qubits=int(num_qubits),
        reference_state=psi_ref,
    )
    selected_label_counts = Counter(candidate.label for candidate in selected)
    selected_execution_modes = [
        str(getattr(candidate, "execution_mode", "termwise_product") or "termwise_product")
        for candidate in selected
    ]
    selected_pauli_terms = [_candidate_pauli_terms_payload(candidate) for candidate in selected]
    selected_generator_semantics_sha256 = hashlib.sha256(
        json.dumps(
            [
                {
                    "label": str(candidate.label),
                    "execution_mode": mode,
                    "pauli_labels_exyz": list(candidate.pauli_labels_exyz),
                    "pauli_terms": pauli_terms,
                }
                for candidate, mode, pauli_terms in zip(
                    selected,
                    selected_execution_modes,
                    selected_pauli_terms,
                    strict=False,
                )
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    last_geo_top: Mapping[str, Any] | None = last_scored_top if _is_geo_config(config) else None
    walltime = float(time.perf_counter() - t0)
    finished_utc = _utc_now()
    decision_noise_metadata = None
    if bool(decision_noise_config.enabled):
        decision_noise_metadata = copy_decision_noise_metadata(
            decision_noise_recorder.summary(
                status="ok",
                supported=True,
                extra={"runner": "generic_static_adapt_variants"},
            )
        )

    final_parameterization_layout = _selected_parameterization_layout(selected)
    if rotosolve_runtime_coordinates:
        theta_runtime_final, final_parameterization_layout = _runtime_theta_for_selected(selected, theta)
        theta = np.asarray(theta_runtime_final, dtype=float).reshape(-1)
        theta_logical_final = np.asarray(
            project_runtime_theta_block_mean(theta, final_parameterization_layout),
            dtype=float,
        ).reshape(-1)
        parameterization_mode_result = "per_pauli_term"
    else:
        theta_logical_final = np.asarray(theta, dtype=float).reshape(-1)
        parameterization_mode_result = "logical_shared"
    fidelity_fields = _dense_exact_state_fidelity_for_selected(
        selected=selected,
        theta=theta,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        pauli_action_cache=pauli_action_cache,
        exact_energy=exact_energy,
        parameterization_mode=parameterization_mode_result,
        parameterization_layout=final_parameterization_layout,
        max_qubits=_exact_fidelity_max_qubits_from_env(),
    )

    row = _base_row(
        family=family_key,
        case_id=case_key,
        config=config,
        status="ok",
        started_utc=started_utc,
        finished_utc=finished_utc,
    )
    row.update(
        {
            "L": int(context.request.num_sites),
            "energy": float(energy),
            "exact_energy": exact_energy,
            "exact_gs_energy": exact_energy,
            "same_cutoff_exact_gs_energy": reference_metrics.get("same_cutoff_exact_gs_energy"),
            "exact_reference_energy": reference_metrics.get("exact_reference_energy"),
            "exact_reference_n_ph_max": reference_metrics.get("exact_reference_n_ph_max"),
            "primary_energy_metric": reference_metrics.get("primary_energy_metric"),
            "primary_reference_source": reference_metrics.get("primary_reference_source"),
            "same_cutoff_error_role": reference_metrics.get("same_cutoff_error_role"),
            "delta_E_abs": abs_delta,
            "abs_delta_e": abs_delta,
            "abs_delta_e_same_cutoff": terminal_error_fields.get("abs_delta_e_same_cutoff"),
            "abs_delta_e_reference": terminal_error_fields.get("abs_delta_e_reference"),
            "infidelity_exact": fidelity_fields.get("infidelity_exact"),
            "exact_state_fidelity": fidelity_fields.get("exact_state_fidelity"),
            "infidelity_status": fidelity_fields.get("infidelity_status"),
            "exact_state_fidelity_source": fidelity_fields.get("exact_state_fidelity_source"),
            "exact_state_fidelity_energy": fidelity_fields.get("exact_state_fidelity_energy"),
            "exact_state_fidelity_target_energy": fidelity_fields.get("exact_state_fidelity_target_energy"),
            "exact_state_fidelity_energy_abs_delta": fidelity_fields.get("exact_state_fidelity_energy_abs_delta"),
            "observable_error_status": "not_implemented_static_train_suite",
            "num_qubits": num_qubits,
            "num_parameters": int(theta.size),
            "logical_parameter_count": int(theta_logical_final.size),
            "runtime_parameter_count": int(theta.size),
            "theta_coordinate_mode": str(parameterization_mode_result),
            "parameterization_mode": str(parameterization_mode_result),
            "parameterization": serialize_layout(final_parameterization_layout),
            "optimal_point": [float(x) for x in np.asarray(theta, dtype=float).reshape(-1).tolist()],
            "logical_optimal_point": [
                float(x) for x in np.asarray(theta_logical_final, dtype=float).reshape(-1).tolist()
            ],
            "selected_operator_count": int(len(selected)),
            "selected_unique_operator_count": int(len(selected_label_counts)),
            "selected_label_counts": {str(label): int(count) for label, count in sorted(selected_label_counts.items())},
            "pool_term_count": int(len(pool)),
            "generic_adapt_runtime_split_mode": runtime_split_mode,
            "generic_adapt_runtime_split_symmetry_policy": runtime_split_symmetry_policy,
            "generic_adapt_runtime_split_max_subset_size": int(runtime_split_max_subset_size),
            "generic_adapt_runtime_split_enabled": bool(
                runtime_split_mode != _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF
            ),
            "generic_adapt_runtime_split_pool_expansion_meta": dict(runtime_split_pool_meta),
            "generic_adapt_runtime_split_base_pool_term_count": int(
                runtime_split_pool_meta.get("base_pool_term_count", len(pool))
            ),
            "generic_adapt_runtime_split_expanded_pool_term_count": int(
                runtime_split_pool_meta.get("expanded_pool_term_count", len(pool))
            ),
            "shared_pauli_pool_contract_id": shared_pauli_pool_meta.get("contract_id"),
            "shared_pauli_pool_requested": {
                "mode": shared_pool_mode,
                "symmetry_policy": shared_pool_symmetry_policy,
                "max_subset_size": int(shared_pool_max_subset_size),
            },
            "shared_pauli_pool_mode": shared_pool_mode,
            "shared_pauli_pool_symmetry_policy": shared_pool_symmetry_policy,
            "shared_pauli_pool_max_subset_size": int(shared_pool_max_subset_size),
            "shared_pauli_pool_enabled": bool(shared_pool_mode != SHARED_PAULI_POOL_MODE_OFF),
            "shared_pauli_pool_symmetry_gate_enforced": shared_pauli_pool_meta.get("symmetry_gate_enforced"),
            "shared_pauli_pool_explicit_no_guard": shared_pauli_pool_meta.get("explicit_no_guard"),
            "shared_pauli_pool_base_pool_term_count": int(
                shared_pauli_pool_meta.get("base_pool_term_count", len(pool))
            ),
            "shared_pauli_pool_expanded_pool_term_count": int(
                shared_pauli_pool_meta.get("expanded_pool_term_count", len(pool))
            ),
            "shared_pauli_pool_ordered_label_hash": shared_pauli_pool_meta.get("ordered_label_hash"),
            "shared_pauli_pool_ordered_pool_hash": shared_pauli_pool_meta.get("ordered_pool_hash"),
            "shared_pauli_pool_contract": dict(shared_pauli_pool_meta),
            "pool_runtime_split_metadata": {
                str(candidate.label): meta
                for candidate in pool
                for meta in [_pool_runtime_split_metadata(candidate)]
                if meta is not None
            },
            "base_pool_name": _pool_name_for_config(config),
            "hh_adaptive_pool_profile": effective_hh_adaptive_pool_profile,
            "hh_full_meta_minus_hva_active": bool(hh_full_meta_class_filter_json is not None),
            "hh_full_meta_class_filter_json": _repo_relative_or_abs(hh_full_meta_class_filter_json),
            "hh_full_meta_class_filter_classifier_version": (
                full_meta_class_filter_meta.get("classifier_version")
                if isinstance(full_meta_class_filter_meta, Mapping)
                else None
            ),
            "hh_full_meta_class_filter_keep_classes": (
                list(full_meta_class_filter_meta.get("keep_classes", []))
                if isinstance(full_meta_class_filter_meta, Mapping)
                and isinstance(full_meta_class_filter_meta.get("keep_classes", []), Sequence)
                and not isinstance(full_meta_class_filter_meta.get("keep_classes", []), (str, bytes, bytearray))
                else []
            ),
            "hh_full_meta_class_filter_class_counts_before": (
                dict(full_meta_class_filter_meta.get("class_counts_before", {}))
                if isinstance(full_meta_class_filter_meta, Mapping)
                and isinstance(full_meta_class_filter_meta.get("class_counts_before", {}), Mapping)
                else None
            ),
            "hh_full_meta_class_filter_class_counts_after": (
                dict(full_meta_class_filter_meta.get("class_counts_after", {}))
                if isinstance(full_meta_class_filter_meta, Mapping)
                and isinstance(full_meta_class_filter_meta.get("class_counts_after", {}), Mapping)
                else None
            ),
            "hh_full_meta_class_filter_dropped_classes": (
                list(full_meta_class_filter_meta.get("dropped_classes", []))
                if isinstance(full_meta_class_filter_meta, Mapping)
                and isinstance(full_meta_class_filter_meta.get("dropped_classes", []), Sequence)
                and not isinstance(full_meta_class_filter_meta.get("dropped_classes", []), (str, bytes, bytearray))
                else []
            ),
            "hh_full_meta_class_filter_prebuild_skipped_classes": (
                list(full_meta_class_filter_meta.get("prebuild_skipped_classes", []))
                if isinstance(full_meta_class_filter_meta, Mapping)
                and isinstance(full_meta_class_filter_meta.get("prebuild_skipped_classes", []), Sequence)
                and not isinstance(full_meta_class_filter_meta.get("prebuild_skipped_classes", []), (str, bytes, bytearray))
                else []
            ),
            "hh_full_meta_class_filter_meta": (
                dict(full_meta_class_filter_meta)
                if isinstance(full_meta_class_filter_meta, Mapping)
                else None
            ),
            "hh_full_meta_label_filter_meta": (
                dict(full_meta_label_filter_meta)
                if isinstance(full_meta_label_filter_meta, Mapping)
                else None
            ),
            "hh_pool_legal_subspace_filter_meta": (
                dict(pool_legal_subspace_filter_meta)
                if isinstance(pool_legal_subspace_filter_meta, Mapping)
                else None
            ),
            "hh_pool_cache_mode": (
                os.environ.get(_HH_POOL_CACHE_ENV)
                if hh_full_meta_class_filter_json is not None
                else None
            ),
            "hh_pool_cache_scope": (
                os.environ.get(_HH_POOL_CACHE_SCOPE_ENV)
                if hh_full_meta_class_filter_json is not None
                else None
            ),
            "hh_pool_cache_dir": (
                _repo_relative_or_abs(os.environ.get(_HH_POOL_CACHE_DIR_ENV))
                if hh_full_meta_class_filter_json is not None and os.environ.get(_HH_POOL_CACHE_DIR_ENV)
                else None
            ),
            "hh_pool_cache_events": [dict(event) for event in pool_cache_events],
            "hh_pool_cache_event_names": [str(event.get("event")) for event in pool_cache_events],
            "selected_logical_route": str(selected_logical_route or "standard").strip().lower().replace("-", "_"),
            "selected_logical_mode": str(selected_logical_mode),
            "selected_logical_source_json": (
                None if selected_logical_source_json in {None, ""} else str(selected_logical_source_json)
            ),
            "selected_logical_transfer_mode": str(selected_logical_transfer_mode or "exact_match_v1"),
            "selected_logical_filter_applied": (
                bool(selected_logical_filter_meta.get("applied", False))
                if isinstance(selected_logical_filter_meta, Mapping)
                else False
            ),
            "selected_logical_filter_fallback_to_full_pool": (
                bool(selected_logical_filter_meta.get("fallback_to_full_pool", False))
                if isinstance(selected_logical_filter_meta, Mapping)
                else False
            ),
            "selected_logical_filter_fallback_reason": (
                selected_logical_filter_meta.get("fallback_reason")
                if isinstance(selected_logical_filter_meta, Mapping)
                else None
            ),
            "selected_logical_pool_size_before": (
                selected_logical_filter_meta.get("pool_size_before")
                if isinstance(selected_logical_filter_meta, Mapping)
                else None
            ),
            "selected_logical_pool_size_after": (
                selected_logical_filter_meta.get("pool_size_after")
                if isinstance(selected_logical_filter_meta, Mapping)
                else None
            ),
            "selected_logical_matched_count": (
                selected_logical_filter_meta.get("matched_count")
                if isinstance(selected_logical_filter_meta, Mapping)
                else None
            ),
            "selected_logical_match_method_counts": (
                dict(selected_logical_filter_meta.get("match_method_counts", {}))
                if isinstance(selected_logical_filter_meta, Mapping)
                and isinstance(selected_logical_filter_meta.get("match_method_counts", {}), Mapping)
                else {}
            ),
            "selected_logical_operator_family_ids": (
                list(selected_logical_filter_meta.get("operator_family_ids", []))
                if isinstance(selected_logical_filter_meta, Mapping)
                and isinstance(selected_logical_filter_meta.get("operator_family_ids", []), Sequence)
                and not isinstance(selected_logical_filter_meta.get("operator_family_ids", []), (str, bytes, bytearray))
                else []
            ),
            "selected_logical_filter_meta": (
                dict(selected_logical_filter_meta)
                if isinstance(selected_logical_filter_meta, Mapping)
                else None
            ),
            "hamiltonian_pauli_term_count": hamiltonian_pauli_term_count,
            "pool_labels": [candidate.label for candidate in pool],
            "pool_qubit_supports": {candidate.label: list(candidate.support) for candidate in pool},
            "pool_pauli_labels_exyz": {candidate.label: list(candidate.pauli_labels_exyz) for candidate in pool},
            "pool_execution_modes": {
                candidate.label: str(
                    getattr(candidate, "execution_mode", "termwise_product") or "termwise_product"
                )
                for candidate in pool
            },
            "selected_operators": [candidate.label for candidate in selected],
            "selected_operator_supports": [list(candidate.support) for candidate in selected],
            "selected_operator_pauli_labels_exyz": [list(candidate.pauli_labels_exyz) for candidate in selected],
            "selected_operator_execution_modes": selected_execution_modes,
            "selected_operator_pauli_terms": selected_pauli_terms,
            "selected_generator_semantics_sha256": selected_generator_semantics_sha256,
            "selected_operator_runtime_split_metadata": [
                _pool_runtime_split_metadata(candidate) for candidate in selected
            ],
            "selected_operator_batches": [[candidate.label for candidate in batch] for batch in selected_batches],
            "selected_operator_batch_supports": [[list(candidate.support) for candidate in batch] for batch in selected_batches],
            "adapt_depth_reached": int(len(selected)),
            "adapt_num_iterations": int(len(adapt_history)),
            "adapt_max_iterations": int(max_adapt_iterations),
            "adapt_stop_reason": str(stop_reason),
            "adapt_target_stop_policy": (
                "fixed_iteration_horizon"
                if generic_adapt_stop_policy_label == _FIXED_HORIZON_NO_TARGET_STOP_POLICY
                else
                "first_hit_or_max_depth"
                if target_stop_enabled
                else "gradient_threshold_or_pool_exhaustion"
            ),
            "adapt_allow_repeats_override": None if allow_repeats is None else bool(allow_repeats),
            "adapt_continuation_mode": (
                "warm_start_selected_theta_v1" if warm_start_continuation_enabled else "fresh"
            ),
            "adapt_warm_start_source_iterations": int(warm_start_source_iterations),
            "adapt_warm_start_source_depth": int(warm_start_source_depth),
            "adapt_history": adapt_history,
            "adapt_last_selected_max_gradient": (
                None if not adapt_history else float(adapt_history[-1].get("max_abs_gradient", 0.0))
            ),
            "adapt_last_selected_gradient_l2_norm": (
                None if not adapt_history else float(adapt_history[-1].get("gradient_l2_norm", 0.0))
            ),
            "adapt_final_max_gradient": float(terminal_max_abs_gradient),
            "adapt_terminal_max_abs_gradient": float(terminal_max_abs_gradient),
            "adapt_terminal_gradient_l2_norm": float(terminal_gradient_l2_norm),
            "adapt_terminal_best_selector_score": float(terminal_best_selector_score),
            "adapt_terminal_geo_natural_step_fs_norm": terminal_geo_natural_step_fs_norm,
            "adapt_terminal_candidate_count": int(len(terminal_scored)),
            "adapt_terminal_diagnostic_hamiltonian_eval_count": 1,
            "adapt_terminal_diagnostic_gradient_probe_count": int(len(terminal_scored)),
            "adapt_terminal_diagnostic_metric_probe_count": (
                int(len(terminal_scored) * (len(terminal_scored) + 1) // 2)
                if _is_geo_config(config)
                else 0
            ),
            "adapt_terminal_diagnostic_queries_in_S_alg": False,
            "adapt_terminal_admissible_batch_size": int(len(terminal_batch)),
            "adapt_terminal_selection_gradient_threshold": float(terminal_selection_gradient_threshold),
            "adapt_terminal_batch_admission_gradient_threshold": terminal_batch_admission_gradient_threshold,
            "adapt_terminal_stop_condition": str(terminal_stop_condition),
            "adapt_terminal_top_candidates": list(terminal_scored[: min(5, len(terminal_scored))]),
            "benchmark_energy_stop_target": benchmark_energy_stop_target,
            "uses_exact_for_decision": bool(target_stop_enabled),
            "uses_reference_for_decision": bool(target_stop_enabled),
            "exact_reference_usage": (
                "post_iteration_adaptive_stop_decision"
                if target_stop_enabled
                else "reporting_only_after_optimization"
            ),
            "adaptive_trajectory_reference_independent": bool(not target_stop_enabled),
            "generic_adapt_stop_policy": generic_adapt_stop_policy_label,
            "benchmark_energy_stop_target_decision_usage": (
                "used_for_post_iteration_exact_reference_stop"
                if target_stop_enabled
                else "requested_but_reference_unavailable"
                if benchmark_energy_stop_target is not None
                else None
            ),
            "benchmark_first_hit_thresholds": [float(x) for x in benchmark_first_hit_thresholds],
            "benchmark_first_hits": {f"{float(k):.0e}": v for k, v in sorted(first_hits.items())},
            "first_hit_1e6": first_hits.get(1e-6),
            "first_hit_1e8": first_hits.get(1e-8),
            "first_hit_abs_delta_e_le_1e_6": first_hits.get(1e-6),
            "first_hit_abs_delta_e_le_1e_8": first_hits.get(1e-8),
            "optimizer": (
                str(optimizer_info_last.get("optimizer"))
                if optimizer_info_last.get("optimizer")
                else (
                    "exact_bench_qngd:logical_shared_metric_backtracking"
                    if config.optimizer_kind == "geo_qngd"
                    else _NATIVE_SPSA_OPTIMIZER_LABEL
                    if config.optimizer_kind == "spsa"
                    else "repo_coordinate_descent:rotosolve_coordinate_descent"
                    if config.optimizer_kind == "rotosolve"
                    else "scipy.optimize.minimize:BFGS"
                )
            ),
            "optimizer_kind": str(config.optimizer_kind),
            "optimizer_source": str(optimizer_settings["optimizer_source"]),
            "optimizer_profile": optimizer_settings["optimizer_profile"],
            "optimizer_profile_source": optimizer_settings["optimizer_profile_source"],
            "optimizer_overlay_source": optimizer_settings["optimizer_overlay_source"],
            "adapt_optimizer_kind": str(config.optimizer_kind),
            "optimizer_maxiter": int(optimizer_maxiter),
            "adapt_spsa_maxiter": optimizer_settings["adapt_spsa_maxiter"],
            "adapt_spsa_seed": optimizer_settings["adapt_spsa_seed"],
            "adapt_spsa_a": optimizer_settings["adapt_spsa_a"],
            "adapt_spsa_c": optimizer_settings["adapt_spsa_c"],
            "adapt_spsa_alpha": optimizer_settings["adapt_spsa_alpha"],
            "adapt_spsa_gamma": optimizer_settings["adapt_spsa_gamma"],
            "adapt_spsa_big_a": optimizer_settings["adapt_spsa_big_a"],
            "optimizer_success_all": bool(optimizer_success_all),
            "optimizer_raw_success_all": bool(optimizer_raw_success_all),
            "powell_maxiter_cap_policy": str(powell_cap_policy),
            "optimizer_capped": bool(optimizer_capped_iterations),
            "optimizer_capped_count": int(len(optimizer_capped_iterations)),
            "optimizer_capped_iterations": [
                int(index) for index in optimizer_capped_iterations
            ],
            "optimizer_capped_accepted_count": int(
                len(optimizer_capped_accepted_iterations)
            ),
            "optimizer_capped_accepted_iterations": [
                int(index) for index in optimizer_capped_accepted_iterations
            ],
            "optimizer_cap_acceptance_semantics": (
                "accept only scipy Powell status=2 maxiter exhaustion with finite objective, "
                "finite parameters, finite exact refit energy, and non-increasing energy within "
                "rel_tol=abs_tol=1e-10"
                if powell_cap_policy
                == _POWELL_MAXITER_CAP_POLICY_ACCEPT_FINITE_NONINCREASING
                else "strict optimizer success required"
            ),
            "optimizer_messages": optimizer_messages,
            "optimizer_decision_energy": optimizer_info_last.get("optimizer_decision_energy"),
            "optimizer_reported_energy": optimizer_info_last.get("optimizer_reported_energy"),
            "optimizer_exact_energy": optimizer_info_last.get("optimizer_exact_energy"),
            "optimizer_decision_surface": optimizer_info_last.get("optimizer_decision_surface"),
            "rotosolve_accepted_steps_last": optimizer_info_last.get("accepted_steps"),
            "rotosolve_stencil_source": optimizer_info_last.get("rotosolve_stencil_source"),
            "rotosolve_period": optimizer_info_last.get("rotosolve_period"),
            "rotosolve_shift": optimizer_info_last.get("rotosolve_shift"),
            "selected_position_exact_energy": optimizer_info_last.get("selected_position_exact_energy"),
            "selected_position_decision_energy": optimizer_info_last.get("selected_position_decision_energy"),
            "gradient_threshold": float(gradient_threshold),
            "metric_floor": float(metric_floor) if _is_geo_config(config) else None,
            "geo_metric_floor": float(metric_floor) if _is_geo_config(config) else None,
            "geo_score_formula": _GEO_SCORE_FORMULA if _is_geo_config(config) else None,
            "geo_metric_definition": _geo_metric_definition_for_config(config),
            "geo_selector_mode": _geo_selector_mode_for_config(config),
            "geo_natural_step_norm_threshold": (
                float(gradient_threshold) if config.stop_rule == "geo_natural_gradient_norm" else None
            ),
            "geo_metric_rank_last": (None if last_geo_top is None else last_geo_top.get("geo_metric_rank")),
            "geo_metric_condition_last": (None if last_geo_top is None else last_geo_top.get("geo_metric_condition")),
            "geo_metric_regularization_last": (None if last_geo_top is None else last_geo_top.get("geo_metric_regularization")),
            "geo_metric_offdiag_norm_last": (None if last_geo_top is None else last_geo_top.get("geo_metric_offdiag_norm")),
            "geo_metric_candidate_count_before_screen_last": (
                None if last_geo_top is None else last_geo_top.get("geo_metric_candidate_count_before_screen")
            ),
            "geo_metric_candidate_count_after_screen_last": (
                None if last_geo_top is None else last_geo_top.get("geo_metric_candidate_count_after_screen")
            ),
            "geo_metric_prescreen_mode_last": (
                None if last_geo_top is None else last_geo_top.get("geo_metric_prescreen_mode")
            ),
            "geo_projected_residual_force_last": (
                None if last_geo_top is None else last_geo_top.get("geo_projected_residual_force")
            ),
            "geo_natural_step_last": (None if last_geo_top is None else last_geo_top.get("geo_natural_step")),
            "geo_natural_step_fs_norm_last": (
                None if last_geo_top is None else last_geo_top.get("geo_natural_step_fs_norm")
            ),
            "geo_natural_step_l2_norm_last": (
                None if last_geo_top is None else last_geo_top.get("geo_natural_step_l2_norm")
            ),
            "geo_max_abs_natural_step_last": (
                None if last_geo_top is None else last_geo_top.get("geo_max_abs_natural_step")
            ),
            "qngd_metric_rank_last": optimizer_info_last.get("qngd_metric_rank_last"),
            "qngd_metric_condition_last": optimizer_info_last.get("qngd_metric_condition_last"),
            "qngd_step_fs_norm_last": optimizer_info_last.get("qngd_step_fs_norm_last"),
            "qngd_step_l2_norm_last": optimizer_info_last.get("qngd_step_l2_norm_last"),
            "qngd_max_abs_step_last": optimizer_info_last.get("qngd_max_abs_step_last"),
            "qngd_line_search_backtracks_total": optimizer_info_last.get("qngd_line_search_backtracks_total"),
            "qngd_accepted_step_count": optimizer_info_last.get("qngd_accepted_step_count"),
            "qngd_energy_decrease_total": optimizer_info_last.get("qngd_energy_decrease_total"),
            "qngd_metric_eval_count": optimizer_info_last.get("qngd_metric_eval_count"),
            "qngd_metric_operator_probe_count_total": optimizer_info_last.get("qngd_metric_operator_probe_count_total"),
            "qngd_gradient_operator_probe_count_total": optimizer_info_last.get(
                "qngd_gradient_operator_probe_count_total"
            ),
            "outer_hamiltonian_eval_count": int(outer_hamiltonian_eval_count),
            "qngd_fallback_optimizer": optimizer_info_last.get("qngd_fallback_optimizer"),
            "qngd_spsa_polish_attempted": optimizer_info_last.get("qngd_spsa_polish_attempted"),
            "qngd_spsa_polish_success": optimizer_info_last.get("qngd_spsa_polish_success"),
            "qngd_spsa_polish_message": optimizer_info_last.get("qngd_spsa_polish_message"),
            "qngd_spsa_polish_seed": optimizer_info_last.get("qngd_spsa_polish_seed"),
            "qngd_spsa_polish_nfev": optimizer_info_last.get("qngd_spsa_polish_nfev"),
            "qngd_spsa_polish_nit": optimizer_info_last.get("qngd_spsa_polish_nit"),
            "qngd_spsa_polish_accepted_step_count": optimizer_info_last.get("qngd_spsa_polish_accepted_step_count"),
            "qngd_spsa_polish_energy_before": optimizer_info_last.get("qngd_spsa_polish_energy_before"),
            "qngd_spsa_polish_energy_after": optimizer_info_last.get("qngd_spsa_polish_energy_after"),
            "qngd_spsa_polish_energy_decrease_total": optimizer_info_last.get("qngd_spsa_polish_energy_decrease_total"),
            "qngd_bfgs_polish_attempted": optimizer_info_last.get("qngd_bfgs_polish_attempted"),
            "qngd_bfgs_polish_success": optimizer_info_last.get("qngd_bfgs_polish_success"),
            "qngd_bfgs_polish_message": optimizer_info_last.get("qngd_bfgs_polish_message"),
            "qngd_bfgs_polish_nfev": optimizer_info_last.get("qngd_bfgs_polish_nfev"),
            "qngd_bfgs_polish_nit": optimizer_info_last.get("qngd_bfgs_polish_nit"),
            "qngd_bfgs_polish_energy_before": optimizer_info_last.get("qngd_bfgs_polish_energy_before"),
            "qngd_bfgs_polish_energy_after": optimizer_info_last.get("qngd_bfgs_polish_energy_after"),
            "spsa_seed": optimizer_info_last.get("spsa_seed"),
            "spsa_a": optimizer_settings["adapt_spsa_a"],
            "spsa_c": optimizer_settings["adapt_spsa_c"],
            "spsa_alpha": optimizer_settings["adapt_spsa_alpha"],
            "spsa_gamma": optimizer_settings["adapt_spsa_gamma"],
            "spsa_A": optimizer_settings["adapt_spsa_big_a"],
            "spsa_big_a": optimizer_settings["adapt_spsa_big_a"],
            "spsa_refit_engine": optimizer_info_last.get("spsa_refit_engine"),
            "spsa_return_policy": optimizer_info_last.get("spsa_return_policy"),
            "spsa_optimizer_memory": optimizer_info_last.get("spsa_optimizer_memory"),
            "spsa_history_tail": optimizer_info_last.get("spsa_history_tail"),
            "spsa_accepted_step_count": optimizer_info_last.get("spsa_accepted_step_count"),
            "spsa_energy_before": optimizer_info_last.get("spsa_energy_before"),
            "spsa_energy_after": optimizer_info_last.get("spsa_energy_after"),
            "spsa_energy_decrease_total": optimizer_info_last.get("spsa_energy_decrease_total"),
            "pos_geo_position_trial_spsa_attempt_count": optimizer_info_last.get("pos_geo_position_trial_spsa_attempt_count"),
            "pos_geo_position_trial_spsa_success_count": optimizer_info_last.get("pos_geo_position_trial_spsa_success_count"),
            "pos_geo_position_trial_bfgs_attempt_count": optimizer_info_last.get("pos_geo_position_trial_bfgs_attempt_count"),
            "tetris_batch_sizes": [int(entry["batch_size"]) for entry in adapt_history] if config.variant == "tetris" else [],
            "tetris_compatibility_rule": _TETRIS_COMPATIBILITY_RULE if config.variant == "tetris" else None,
            "max_tetris_batch_size": int(max_tetris_batch_size) if config.variant == "tetris" else None,
            "seed": int(seed),
            "nfev": int(nfev_total),
            "nit": int(nit_total),
            "runtime_s": walltime,
            **compiled_stats,
            **shot_proxy,
            **component_fields,
            "sector_probability": sector.get("sector_probability"),
            "sector_leak_probability": sector.get("sector_leak_probability"),
            "sector_leak_flag": sector.get("sector_leak_flag"),
            "sector_leak_threshold": sector.get("sector_leak_threshold"),
            "boson_legal_probability_min": sector.get("boson_legal_probability_min"),
            "boson_illegal_probability_max": sector.get("boson_illegal_probability_max"),
            "boson_truncation_leak_flag": sector.get("boson_truncation_leak_flag"),
            "boson_subspace_diagnostics": sector.get("boson_subspace_diagnostics"),
            "truncation_diagnostics": sector.get("truncation_constraints_evaluated"),
            "sector_diagnostics": sector,
            "theta": theta.tolist(),
        }
    )
    if decision_noise_metadata is not None:
        row.update(
            {
                "benchmark_decision_noise_status": "ok",
                "benchmark_decision_noise": copy_decision_noise_metadata(decision_noise_metadata),
            }
        )
    payload_status = "completed"
    if not bool(optimizer_success_all):
        row["status"] = "quality_nonpassing"
        row["quality_gate_reason"] = _optimizer_failure_reason(config)
        payload_status = "completed_quality_nonpassing"
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family_key,
        "case_id": case_key,
        "algorithm_id": config.algorithm_id,
        "method_id": config.algorithm_id,
        "status": payload_status,
        "runner": "pipelines.exact_bench.generic_static_adapt_variants.run_generic_static_adapt_variant_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(
            config,
            exact_reference_usage=(
                "post_iteration_adaptive_stop_decision"
                if target_stop_enabled
                else "reporting_only_after_optimization"
            ),
            uses_reference_for_decision=bool(target_stop_enabled),
        ),
        "generic_adapt_runtime_split": dict(runtime_split_pool_meta),
        "shared_pauli_pool_contract": dict(shared_pauli_pool_meta),
        "spec": _spec_metadata(spec),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }
    if generic_adapt_stop_policy_label is not None:
        payload["generic_adapt_stop_policy"] = generic_adapt_stop_policy_label
        payload["metadata"] = {"generic_adapt_stop_policy": generic_adapt_stop_policy_label}
    payload["powell_maxiter_cap_policy"] = str(powell_cap_policy)
    metadata = payload.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["powell_maxiter_cap_policy"] = str(powell_cap_policy)
    if decision_noise_metadata is not None:
        payload.update(
            {
                "benchmark_decision_noise_status": "ok",
                "benchmark_decision_noise": copy_decision_noise_metadata(decision_noise_metadata),
            }
        )
    runtime_seed_payload = None
    if config.algorithm_id in {STATIC_GEO_ADAPT_VQE, STATIC_FULL_META_APPEND_ADAPT_VQE}:
        runtime_seed_payload = _build_runtime_seed_payload(
            context=context,
            family=family_key,
            case_id=case_key,
            config=config,
            selected=selected,
            selected_batches=selected_batches,
            theta=theta,
            psi_ref=psi_ref,
            psi_final=psi_final,
            row=row,
            spec=spec,
            generated_utc=finished_utc,
        )
    return _write_artifacts(output, payload, [row], runtime_seed_payload=runtime_seed_payload)


def run_generic_static_adapt_variant_single(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    output_dir: Path,
    max_adapt_iterations: int = _DEFAULT_MAX_ADAPT_ITERATIONS,
    optimizer_maxiter: int = _DEFAULT_OPTIMIZER_MAXITER,
    gradient_threshold: float = _DEFAULT_GRADIENT_THRESHOLD,
    metric_floor: float = _DEFAULT_METRIC_FLOOR,
    max_tetris_batch_size: int = _DEFAULT_MAX_TETRIS_BATCH_SIZE,
    seed: int = 42,
    shots_per_pauli_term_proxy: int = _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
    energy_stop_target: float | None = None,
    first_hit_thresholds: Sequence[float] | str | None = None,
    same_cutoff_exact_gs_energy: float | str | None = None,
    exact_reference_energy: float | str | None = None,
    exact_reference_n_ph_max: int | str | None = None,
    primary_energy_metric: str | None = None,
    same_cutoff_error_role: str | None = None,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | Mapping[str, Any] | None = None,
    selected_logical_route: str | None = None,
    selected_logical_source_json: str | Path | None = None,
    selected_logical_transfer_mode: str = "exact_match_v1",
    allow_repeats: bool | None = None,
    progress_jsonl_path: str | Path | None = None,
    generic_adapt_stop_policy: str | None = None,
    powell_maxiter_cap_policy: str | None = None,
    generic_adapt_runtime_split_mode: str | None = _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF,
    generic_adapt_runtime_split_symmetry_policy: str | None = _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_OFF,
    generic_adapt_runtime_split_max_subset_size: int | str | None = 3,
    shared_pauli_pool_mode: str | None = SHARED_PAULI_POOL_MODE_OFF,
    shared_pauli_pool_symmetry_policy: str | None = SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF,
    shared_pauli_pool_max_subset_size: int | str | None = 3,
    initial_selected_operator_labels: Sequence[str] | None = None,
    initial_selected_operator_batches: Sequence[Sequence[str]] | None = None,
    initial_theta: Sequence[float] | None = None,
    initial_adapt_history: Sequence[Mapping[str, Any]] | None = None,
    optimizer_profile: str | None = None,
    optimizer_profile_source: str | None = None,
    adapt_optimizer_kind: str | None = None,
    adapt_spsa_maxiter: int | None = None,
    adapt_spsa_seed: int | None = None,
    adapt_spsa_a: float | str | None = None,
    adapt_spsa_c: float | str | None = None,
    adapt_spsa_alpha: float | str | None = None,
    adapt_spsa_gamma: float | str | None = None,
    adapt_spsa_big_a: float | str | None = None,
    optimizer_overlay_source: str | None = None,
    hh_adaptive_pool_profile: str | None = None,
    hh_full_meta_class_filter_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run one generic statevector ADAPT variant row and always emit artifacts."""
    started_utc = _utc_now()
    config = _get_config(algorithm_id)
    try:
        return _run_impl(
            family=family,
            case_id=case_id,
            algorithm_id=algorithm_id,
            output_dir=output_dir,
            max_adapt_iterations=max_adapt_iterations,
            optimizer_maxiter=optimizer_maxiter,
            gradient_threshold=gradient_threshold,
            metric_floor=metric_floor,
            max_tetris_batch_size=max_tetris_batch_size,
            seed=seed,
            shots_per_pauli_term_proxy=shots_per_pauli_term_proxy,
            energy_stop_target=energy_stop_target,
            first_hit_thresholds=first_hit_thresholds,
            same_cutoff_exact_gs_energy=same_cutoff_exact_gs_energy,
            exact_reference_energy=exact_reference_energy,
            exact_reference_n_ph_max=exact_reference_n_ph_max,
            primary_energy_metric=primary_energy_metric,
            same_cutoff_error_role=same_cutoff_error_role,
            benchmark_decision_noise_config=benchmark_decision_noise_config,
            selected_logical_route=selected_logical_route,
            selected_logical_source_json=selected_logical_source_json,
            selected_logical_transfer_mode=selected_logical_transfer_mode,
            allow_repeats=allow_repeats,
            progress_jsonl_path=progress_jsonl_path,
            generic_adapt_stop_policy=generic_adapt_stop_policy,
            powell_maxiter_cap_policy=powell_maxiter_cap_policy,
            generic_adapt_runtime_split_mode=generic_adapt_runtime_split_mode,
            generic_adapt_runtime_split_symmetry_policy=generic_adapt_runtime_split_symmetry_policy,
            generic_adapt_runtime_split_max_subset_size=generic_adapt_runtime_split_max_subset_size,
            shared_pauli_pool_mode=shared_pauli_pool_mode,
            shared_pauli_pool_symmetry_policy=shared_pauli_pool_symmetry_policy,
            shared_pauli_pool_max_subset_size=shared_pauli_pool_max_subset_size,
            initial_selected_operator_labels=initial_selected_operator_labels,
            initial_selected_operator_batches=initial_selected_operator_batches,
            initial_theta=initial_theta,
            initial_adapt_history=initial_adapt_history,
            optimizer_profile=optimizer_profile,
            optimizer_profile_source=optimizer_profile_source,
            adapt_optimizer_kind=adapt_optimizer_kind,
            adapt_spsa_maxiter=adapt_spsa_maxiter,
            adapt_spsa_seed=adapt_spsa_seed,
            adapt_spsa_a=adapt_spsa_a,
            adapt_spsa_c=adapt_spsa_c,
            adapt_spsa_alpha=adapt_spsa_alpha,
            adapt_spsa_gamma=adapt_spsa_gamma,
            adapt_spsa_big_a=adapt_spsa_big_a,
            optimizer_overlay_source=optimizer_overlay_source,
            hh_adaptive_pool_profile=hh_adaptive_pool_profile,
            hh_full_meta_class_filter_json=hh_full_meta_class_filter_json,
        )
    except ImportError as exc:
        return _skip_payload(
            family=str(family).strip(),
            case_id=str(case_id).strip(),
            config=config,
            output_dir=Path(output_dir),
            reason=str(exc),
            started_utc=started_utc,
        )
    except Exception as exc:
        return _failure_payload(
            family=str(family).strip(),
            case_id=str(case_id).strip(),
            config=config,
            output_dir=Path(output_dir),
            reason=str(exc),
            exception_type=type(exc).__name__,
            started_utc=started_utc,
        )


_FIXED_SCAFFOLD_AUDIT_SCHEMA = "fixed_scaffold_expressivity_audit_v1"
_FIXED_SCAFFOLD_AUDIT_OPTIMIZERS = ("powell", "rotosolve", "qnspsa", "geo_qngd", "spsa", "bfgs")


def _normalize_fixed_scaffold_optimizer_kinds(optimizer_kinds: Sequence[str] | str | None) -> tuple[str, ...]:
    if optimizer_kinds in {None, ""}:
        raw = ("powell", "qnspsa")
    elif isinstance(optimizer_kinds, str):
        raw = tuple(item.strip() for item in optimizer_kinds.split(",") if item.strip())
    else:
        raw = tuple(str(item).strip() for item in optimizer_kinds if str(item).strip())
    normalized: list[str] = []
    for item in raw:
        key = str(item).strip().lower()
        if key not in _FIXED_SCAFFOLD_AUDIT_OPTIMIZERS:
            allowed = ", ".join(_FIXED_SCAFFOLD_AUDIT_OPTIMIZERS)
            raise ValueError(f"fixed-scaffold optimizer must be one of {{{allowed}}}; got {item!r}")
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("at least one fixed-scaffold optimizer must be requested")
    return tuple(normalized)


def _adapt_vqe_block_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload.get("adapt_vqe"), Mapping):
        return payload["adapt_vqe"]  # type: ignore[index,return-value]
    result = payload.get("result")
    if isinstance(result, Mapping) and isinstance(result.get("adapt_vqe"), Mapping):
        return result["adapt_vqe"]  # type: ignore[index,return-value]
    runtime_seed = payload.get("runtime_seed_payload")
    if isinstance(runtime_seed, Mapping) and isinstance(runtime_seed.get("adapt_vqe"), Mapping):
        return runtime_seed["adapt_vqe"]  # type: ignore[index,return-value]
    raise ValueError("source JSON does not contain an adapt_vqe block with scaffold operators")


def _extract_fixed_scaffold_source(
    source_json: str | Path | None,
) -> tuple[list[str], np.ndarray, list[list[str]], dict[str, Any]]:
    if source_json in {None, ""}:
        return [], np.zeros(0, dtype=float), [], {"source_json": None, "source_kind": "full_meta_prefix_only"}
    source_path = Path(source_json)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"source JSON {source_path} did not parse to an object")
    adapt = _adapt_vqe_block_from_payload(payload)
    raw_labels = (
        adapt.get("operators")
        or adapt.get("operator_labels")
        or adapt.get("selected_operator_labels")
        or adapt.get("selected_labels")
    )
    if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes, bytearray)):
        raise ValueError(f"source JSON {source_path} adapt_vqe block is missing operator labels")
    labels = [str(label) for label in raw_labels]
    parameterization_payload = adapt.get("parameterization")
    parameterization_mode = str(adapt.get("parameterization_mode") or "").strip()
    if not parameterization_mode and isinstance(parameterization_payload, Mapping):
        parameterization_mode = str(parameterization_payload.get("mode") or "").strip()
    runtime_theta_raw = adapt.get("optimal_point")
    logical_theta_raw = adapt.get("logical_optimal_point")
    fallback_theta_raw = adapt.get("theta") or adapt.get("parameters")
    raw_theta = None
    theta_coordinate_mode = "logical_shared"
    if runtime_theta_raw is not None:
        runtime_theta = np.asarray(list(runtime_theta_raw), dtype=float).reshape(-1)
        if int(runtime_theta.size) != int(len(labels)):
            raw_theta = runtime_theta
            theta_coordinate_mode = "per_pauli_term"
    if raw_theta is None and logical_theta_raw is not None:
        raw_theta = logical_theta_raw
        theta_coordinate_mode = "logical_shared"
    if raw_theta is None:
        raw_theta = fallback_theta_raw
    if raw_theta is None:
        theta = np.zeros(len(labels), dtype=float)
    else:
        theta = np.asarray(list(raw_theta), dtype=float).reshape(-1)
    if theta_coordinate_mode == "logical_shared" and int(theta.size) != int(len(labels)):
        raise ValueError(
            f"source JSON selected/theta length mismatch: {len(labels)} operators vs {int(theta.size)} theta values"
        )
    raw_batches = adapt.get("selected_operator_batches")
    batches: list[list[str]] = []
    if isinstance(raw_batches, Sequence) and not isinstance(raw_batches, (str, bytes, bytearray)):
        for raw_batch in raw_batches:
            if isinstance(raw_batch, Sequence) and not isinstance(raw_batch, (str, bytes, bytearray)):
                batches.append([str(label) for label in raw_batch])
    if not batches:
        batches = [[label] for label in labels]
    return labels, theta, batches, {
        "source_json": str(source_path),
        "source_kind": "adapt_vqe_scaffold_json",
        "source_operator_count": int(len(labels)),
        "source_theta_count": int(theta.size),
        "source_theta_coordinate_mode": theta_coordinate_mode,
        "source_parameterization_mode": parameterization_mode or None,
        "_source_parameterization_blocks": (
            list(parameterization_payload.get("blocks", [])) if isinstance(parameterization_payload, Mapping) else None
        ),
        "_source_parameterization_layout": (
            deserialize_layout(parameterization_payload) if isinstance(parameterization_payload, Mapping) else None
        ),
        "source_algorithm_id": adapt.get("algorithm_id") or payload.get("algorithm_id") or payload.get("method_id"),
        "source_energy": adapt.get("energy") or payload.get("energy"),
        "source_abs_delta_e": adapt.get("abs_delta_e") or payload.get("abs_delta_e"),
    }


def _pool_candidate_from_parameterization_block(block: Mapping[str, Any]) -> _PoolCandidate:
    label = str(block.get("candidate_label", "")).strip()
    if label == "":
        raise ValueError("source parameterization block missing candidate_label")
    raw_terms = block.get("runtime_terms_exyz", [])
    if not isinstance(raw_terms, Sequence) or isinstance(raw_terms, (str, bytes, bytearray)):
        raise ValueError(f"source parameterization block {label} has invalid runtime_terms_exyz")
    terms: list[PauliTerm] = []
    pauli_labels: list[str] = []
    support: set[int] = set()
    for raw in raw_terms:
        if not isinstance(raw, Mapping):
            raise ValueError(f"source parameterization block {label} contains a non-object runtime term")
        pauli = str(raw.get("pauli_exyz", "")).strip().lower()
        if not pauli:
            raise ValueError(f"source parameterization block {label} has runtime term missing pauli_exyz")
        nq = int(raw.get("nq", len(pauli)))
        coeff = complex(float(raw.get("coeff_re", 0.0)), float(raw.get("coeff_im", 0.0)))
        if len(pauli) != nq:
            raise ValueError(f"source parameterization block {label} term {pauli} length mismatch vs nq={nq}")
        if abs(coeff.imag) > 1e-12:
            raise ValueError(f"source parameterization block {label} term {pauli} has non-real coefficient {coeff}")
        terms.append(PauliTerm(nq, ps=pauli, pc=float(coeff.real)))
        pauli_labels.append(pauli)
        support.update(idx for idx, char in enumerate(pauli) if char != "e")
    poly = PauliPolynomial("JW", terms)
    return _PoolCandidate(
        label=label,
        polynomial=poly,
        support=tuple(sorted(support)),
        pauli_labels_exyz=tuple(pauli_labels),
        construction="source_serialized_parameterization_block_v1",
    )


def _normalize_fixed_scaffold_pool_indices(
    pool_indices: Sequence[int] | str | None,
    *,
    pool_size: int,
) -> list[int]:
    if pool_indices in {None, ""}:
        return []
    raw_items: list[str | int] = []
    if isinstance(pool_indices, str):
        raw_items = [item.strip() for item in pool_indices.split(",") if item.strip()]
    else:
        raw_items = list(pool_indices)
    out: list[int] = []
    seen: set[int] = set()
    for item in raw_items:
        if isinstance(item, int):
            candidates = [int(item)]
        else:
            token = str(item).strip()
            if "-" in token:
                left, right = token.split("-", 1)
                start = int(left.strip())
                stop = int(right.strip())
                step = 1 if stop >= start else -1
                candidates = list(range(start, stop + step, step))
            else:
                candidates = [int(token)]
        for idx in candidates:
            if idx < 0 or idx >= int(pool_size):
                raise ValueError(f"fixed scaffold pool index {idx} outside pool size {pool_size}")
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def _assemble_fixed_full_meta_scaffold(
    *,
    pool: Sequence[_PoolCandidate],
    source_labels: Sequence[str],
    source_theta: np.ndarray,
    theta_coordinate_mode: str,
    source_parameterization_blocks: Sequence[Mapping[str, Any]] | None = None,
    source_parameterization_layout: AnsatzParameterLayout | None = None,
    max_scaffold_terms: int | None,
    missing_source_policy: str,
    pool_indices: Sequence[int] | str | None = None,
) -> tuple[list[_PoolCandidate], np.ndarray, dict[str, Any]]:
    by_label = {str(candidate.label): candidate for candidate in pool}
    source_block_candidates: list[_PoolCandidate] = []
    if source_parameterization_blocks is not None:
        source_block_candidates = [
            _pool_candidate_from_parameterization_block(block)
            for block in source_parameterization_blocks
            if isinstance(block, Mapping)
        ]
        if source_block_candidates and int(len(source_block_candidates)) != int(len(source_labels)):
            raise ValueError(
                "source parameterization block count does not match source labels: "
                f"{len(source_block_candidates)} blocks vs {len(source_labels)} labels"
            )
    selected: list[_PoolCandidate] = []
    source_selected_count = 0
    missing_labels: list[str] = []
    duplicate_source_labels: list[str] = []
    seen_for_padding: set[str] = set()
    duplicate_counter: Counter[str] = Counter()
    source_theta_arr = np.asarray(source_theta, dtype=float).reshape(-1)
    for source_idx, label in enumerate(str(item) for item in source_labels):
        duplicate_counter[label] += 1
        if source_block_candidates:
            candidate = source_block_candidates[int(source_idx)]
            if str(candidate.label) != label:
                raise ValueError(
                    "source parameterization block label mismatch: "
                    f"block={candidate.label!r}, operator={label!r}"
                )
            selected.append(candidate)
            seen_for_padding.add(label)
            source_selected_count += 1
            if duplicate_counter[label] > 1:
                duplicate_source_labels.append(label)
            continue
        if label not in by_label:
            missing_labels.append(label)
            continue
        selected.append(by_label[label])
        seen_for_padding.add(label)
        source_selected_count += 1
        if duplicate_counter[label] > 1:
            duplicate_source_labels.append(label)
    policy = str(missing_source_policy or "fail").strip().lower()
    if missing_labels and policy not in {"skip", "ignore"}:
        preview = ", ".join(missing_labels[:8])
        raise ValueError(f"source scaffold labels absent from full_meta pool: {preview}")
    explicit_pool_indices = _normalize_fixed_scaffold_pool_indices(pool_indices, pool_size=len(pool))
    explicit_added_labels: list[str] = []
    explicit_skipped_duplicate_labels: list[str] = []
    for pool_idx in explicit_pool_indices:
        candidate = pool[int(pool_idx)]
        label = str(candidate.label)
        if label in seen_for_padding:
            explicit_skipped_duplicate_labels.append(label)
            continue
        selected.append(candidate)
        seen_for_padding.add(label)
        explicit_added_labels.append(label)
    cap = None if max_scaffold_terms is None else max(int(max_scaffold_terms), int(len(selected)))
    target_count = int(len(pool)) if cap is None else int(min(cap, len(pool)))
    for candidate in pool:
        if int(len(selected)) >= target_count:
            break
        label = str(candidate.label)
        if label in seen_for_padding:
            continue
        seen_for_padding.add(label)
        selected.append(candidate)
    coordinate_mode = "per_pauli_term" if str(theta_coordinate_mode).startswith("per_pauli") else "logical_shared"
    if coordinate_mode == "per_pauli_term":
        if source_parameterization_layout is not None:
            source_layout = source_parameterization_layout
            if int(source_layout.logical_parameter_count) != int(source_selected_count):
                raise ValueError(
                    "source parameterization layout logical count mismatch: "
                    f"{source_layout.logical_parameter_count} vs {source_selected_count}"
                )
            padding = selected[source_selected_count:]
            padding_layout = build_parameter_layout(
                padding,
                ignore_identity=True,
                coefficient_tolerance=1e-12,
                sort_terms=True,
            )
            blocks = list(source_layout.blocks)
            runtime_start = int(source_layout.runtime_parameter_count)
            for block in padding_layout.blocks:
                new_block = GeneratorParameterBlock(
                    candidate_label=str(block.candidate_label),
                    logical_index=int(len(blocks)),
                    runtime_start=int(runtime_start),
                    terms=tuple(block.terms),
                )
                blocks.append(new_block)
                runtime_start = int(new_block.runtime_stop)
            layout = AnsatzParameterLayout(
                mode=str(source_layout.mode),
                term_order=str(source_layout.term_order),
                ignore_identity=bool(source_layout.ignore_identity),
                coefficient_tolerance=float(source_layout.coefficient_tolerance),
                blocks=tuple(blocks),
            )
        else:
            layout = build_parameter_layout(
                selected,
                ignore_identity=True,
                coefficient_tolerance=1e-12,
                sort_terms=True,
            )
            source_layout = build_parameter_layout(
                selected[:source_selected_count],
                ignore_identity=True,
                coefficient_tolerance=1e-12,
                sort_terms=True,
            )
        expected_source_runtime = int(source_layout.runtime_parameter_count)
        if int(source_theta_arr.size) != expected_source_runtime:
            raise ValueError(
                "source runtime theta length mismatch for reconstructed scaffold: "
                f"got {int(source_theta_arr.size)}, expected {expected_source_runtime}"
            )
        theta_out = np.zeros(int(layout.runtime_parameter_count), dtype=float)
        theta_out[: int(source_theta_arr.size)] = source_theta_arr
    else:
        if int(source_theta_arr.size) != int(len(source_labels)):
            raise ValueError(
                f"source logical theta length mismatch: got {int(source_theta_arr.size)}, expected {int(len(source_labels))}"
            )
        selected_theta: list[float] = []
        source_idx = 0
        for label in (str(item) for item in source_labels):
            if label in by_label:
                selected_theta.append(float(source_theta_arr[source_idx]))
            source_idx += 1
        selected_theta.extend([0.0] * max(0, int(len(selected)) - int(len(selected_theta))))
        theta_out = np.asarray(selected_theta, dtype=float).reshape(-1)
    meta = {
        "pool_term_count": int(len(pool)),
        "max_scaffold_terms": None if max_scaffold_terms is None else int(max_scaffold_terms),
        "selected_operator_count": int(len(selected)),
        "theta_coordinate_mode": coordinate_mode,
        "theta_parameter_count": int(theta_out.size),
        "_parameterization_layout": layout if coordinate_mode == "per_pauli_term" else None,
        "parameterization_runtime_parameter_count": (
            int(layout.runtime_parameter_count) if coordinate_mode == "per_pauli_term" else None
        ),
        "parameterization_logical_parameter_count": (
            int(layout.logical_parameter_count) if coordinate_mode == "per_pauli_term" else None
        ),
        "source_operator_count_loaded": int(source_selected_count),
        "full_meta_padding_count": int(max(0, len(selected) - source_selected_count)),
        "missing_source_operator_count": int(len(missing_labels)),
        "missing_source_operator_labels_preview": missing_labels[:16],
        "duplicate_source_operator_count": int(len(duplicate_source_labels)),
        "duplicate_source_operator_labels_preview": duplicate_source_labels[:16],
        "missing_source_policy": policy,
        "explicit_pool_indices": [int(idx) for idx in explicit_pool_indices],
        "explicit_pool_index_count": int(len(explicit_pool_indices)),
        "explicit_pool_added_count": int(len(explicit_added_labels)),
        "explicit_pool_added_labels_preview": explicit_added_labels[:24],
        "explicit_pool_skipped_duplicate_count": int(len(explicit_skipped_duplicate_labels)),
        "explicit_pool_skipped_duplicate_labels_preview": explicit_skipped_duplicate_labels[:24],
    }
    return selected, np.asarray(theta_out, dtype=float).reshape(-1), meta


def _extract_fixed_scaffold_warm_start(
    warm_start_json: str | Path | None,
    *,
    target_selected: Sequence[_PoolCandidate],
    target_theta0: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_theta = np.asarray(target_theta0, dtype=float).reshape(-1).copy()
    if warm_start_json in {None, ""}:
        return target_theta, {
            "warm_start_json": None,
            "warm_start_kind": None,
            "warm_start_applied": False,
            "warm_start_note": "No fixed-scaffold warm start requested.",
        }
    warm_path = Path(warm_start_json)
    payload = json.loads(warm_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"warm-start JSON {warm_path} did not parse to an object")
    warm_result = payload.get("best_result")
    if not isinstance(warm_result, Mapping):
        results = payload.get("results")
        if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)) and results:
            result_rows = [row for row in results if isinstance(row, Mapping) and row.get("energy") is not None]
            if result_rows:
                warm_result = min(result_rows, key=lambda row: float(row["energy"]))
    if not isinstance(warm_result, Mapping):
        raise ValueError(f"warm-start JSON {warm_path} has no best_result/results row")
    warm_theta_raw = warm_result.get("theta")
    if not isinstance(warm_theta_raw, Sequence) or isinstance(warm_theta_raw, (str, bytes, bytearray)):
        raise ValueError(f"warm-start JSON {warm_path} best result has no theta array")
    warm_theta = np.asarray(list(warm_theta_raw), dtype=float).reshape(-1)
    if int(warm_theta.size) > int(target_theta.size):
        raise ValueError(
            "warm-start theta is longer than target fixed scaffold theta: "
            f"{int(warm_theta.size)} > {int(target_theta.size)}"
        )

    warm_scaffold = payload.get("scaffold")
    warm_labels_raw = warm_scaffold.get("operator_labels") if isinstance(warm_scaffold, Mapping) else None
    if not isinstance(warm_labels_raw, Sequence) or isinstance(warm_labels_raw, (str, bytes, bytearray)):
        raise ValueError(f"warm-start JSON {warm_path} has no scaffold.operator_labels array")
    warm_labels = [str(item) for item in warm_labels_raw]
    target_labels = [str(candidate.label) for candidate in target_selected]
    if int(len(warm_labels)) > int(len(target_labels)):
        raise ValueError(
            "warm-start scaffold has more operators than target scaffold: "
            f"{len(warm_labels)} > {len(target_labels)}"
        )
    prefix = target_labels[: int(len(warm_labels))]
    if prefix != warm_labels:
        mismatch_index = next(
            (idx for idx, (left, right) in enumerate(zip(prefix, warm_labels, strict=False)) if left != right),
            None,
        )
        raise ValueError(
            "warm-start scaffold labels must match the target scaffold prefix; "
            f"first mismatch index={mismatch_index}, warm_start={warm_labels[mismatch_index] if mismatch_index is not None else None!r}, "
            f"target={prefix[mismatch_index] if mismatch_index is not None else None!r}"
        )
    target_theta[: int(warm_theta.size)] = warm_theta
    return target_theta, {
        "warm_start_json": str(warm_path),
        "warm_start_kind": "fixed_scaffold_result_prefix_theta",
        "warm_start_applied": True,
        "warm_start_result_optimizer_kind": warm_result.get("optimizer_kind"),
        "warm_start_result_energy": _finite_float_or_none(warm_result.get("energy")),
        "warm_start_result_same_cutoff_abs_delta_e": _finite_float_or_none(warm_result.get("same_cutoff_abs_delta_e")),
        "warm_start_operator_count": int(len(warm_labels)),
        "warm_start_theta_count": int(warm_theta.size),
        "warm_start_target_operator_count": int(len(target_labels)),
        "warm_start_target_theta_count": int(target_theta.size),
        "warm_start_zero_padded_theta_count": int(max(0, int(target_theta.size) - int(warm_theta.size))),
        "warm_start_note": (
            "Warm-start theta was copied into the target fixed scaffold prefix after exact operator-label "
            "prefix validation; any additional target coordinates remain at zero."
        ),
    }


def _run_fixed_scaffold_optimizer(
    *,
    optimizer_kind: str,
    selected: Sequence[_PoolCandidate],
    theta0: np.ndarray,
    psi_ref: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any],
    optimizer_maxiter: int,
    metric_floor: float,
    seed: int,
    minimize_fn: Any | None,
    parameterization_mode: str = "logical_shared",
    parameterization_layout: AnsatzParameterLayout | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    kind = str(optimizer_kind).strip().lower()
    if kind in {"bfgs", "powell", "rotosolve"}:
        if kind in {"bfgs", "powell"} and minimize_fn is None:
            raise ImportError("scipy.optimize.minimize is required for BFGS/Powell fixed-scaffold audits")
        return _optimize_selected(
            minimize_fn=minimize_fn,
            selected=selected,
            x0=theta0,
            psi_ref=psi_ref,
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            optimizer_maxiter=int(optimizer_maxiter),
            optimizer_method=("Powell" if kind == "powell" else "ROTOSOLVE" if kind == "rotosolve" else "BFGS"),
            parameterization_mode=parameterization_mode,
            parameterization_layout=parameterization_layout,
            decision_scope={"fixed_scaffold_audit": True, "optimizer_kind": kind},
        )
    if kind == "spsa":
        return _optimize_selected_spsa(
            selected=selected,
            x0=theta0,
            psi_ref=psi_ref,
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            optimizer_maxiter=int(optimizer_maxiter),
            spsa_seed=int(seed),
            parameterization_mode=parameterization_mode,
            parameterization_layout=parameterization_layout,
            decision_scope={"fixed_scaffold_audit": True, "optimizer_kind": kind},
        )
    if kind == "qnspsa":
        return _optimize_selected_qnspsa(
            selected=selected,
            x0=theta0,
            psi_ref=psi_ref,
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            optimizer_maxiter=int(optimizer_maxiter),
            qnspsa_seed=int(seed),
            parameterization_mode=parameterization_mode,
            parameterization_layout=parameterization_layout,
            decision_scope={"fixed_scaffold_audit": True, "optimizer_kind": kind},
        )
    if kind == "geo_qngd":
        if str(parameterization_mode).startswith("per_pauli"):
            raise ValueError("geo_qngd fixed-scaffold audit currently requires logical_shared theta; use powell/qnspsa/spsa")
        return _optimize_selected_qngd(
            selected=selected,
            x0=theta0,
            psi_ref=psi_ref,
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            optimizer_maxiter=int(optimizer_maxiter),
            metric_floor=float(metric_floor),
            spsa_seed=int(seed),
            decision_scope={"fixed_scaffold_audit": True, "optimizer_kind": kind},
        )
    raise ValueError(f"unsupported fixed-scaffold optimizer {optimizer_kind!r}")


def run_fixed_scaffold_expressivity_audit_single(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    table_i_suite_profile: str | None = None,
    source_json: str | Path | None = None,
    warm_start_json: str | Path | None = None,
    theta_coordinate_mode: str | None = None,
    pool_indices: Sequence[int] | str | None = None,
    max_scaffold_terms: int | None = 64,
    optimizer_kinds: Sequence[str] | str | None = None,
    optimizer_maxiter: int = _DEFAULT_OPTIMIZER_MAXITER,
    metric_floor: float = _DEFAULT_METRIC_FLOOR,
    seed: int = 42,
    pool_term_cap: int | None = 512,
    missing_source_policy: str = "fail",
    same_cutoff_exact_gs_energy: float | str | None = None,
    exact_reference_energy: float | str | None = None,
    exact_reference_n_ph_max: int | str | None = None,
) -> dict[str, Any]:
    """Diagnostic-only oracle scaffold refit over a large full_meta ansatz.

    This is an expressivity/optimizer audit, not an adaptive selector row.  It
    fixes a scaffold, optionally seeded from a Route-C/ADAPT JSON, pads it from
    the problem-local full_meta pool, and refits identical initial parameters
    with requested strong optimizers.
    """

    family_key = str(family).strip()
    case_key = str(case_id).strip()
    output = Path(output_dir)
    started_utc = _utc_now()
    t0 = time.perf_counter()
    config = _get_config(STATIC_FULL_META_APPEND_ADAPT_VQE)
    optimizers = _normalize_fixed_scaffold_optimizer_kinds(optimizer_kinds)
    source_labels, source_theta, _source_batches, source_meta = _extract_fixed_scaffold_source(source_json)
    requested_theta_coordinate_mode = str(theta_coordinate_mode or "auto").strip().lower()
    if requested_theta_coordinate_mode in {"", "auto", "source"}:
        scaffold_theta_coordinate_mode = str(source_meta.get("source_theta_coordinate_mode") or "logical_shared")
    elif requested_theta_coordinate_mode in {"logical", "logical_shared"}:
        scaffold_theta_coordinate_mode = "logical_shared"
    elif requested_theta_coordinate_mode in {"per_pauli", "per_pauli_term", "runtime", "runtime_per_pauli"}:
        scaffold_theta_coordinate_mode = "per_pauli_term"
    else:
        raise ValueError(
            "theta_coordinate_mode must be one of auto, logical_shared, per_pauli_term; "
            f"got {theta_coordinate_mode!r}"
        )
    spec = with_molecular_vibronic_h2_fixture_override(
        table_i_canonical_spec_by_case_id(family_key, case_key, table_i_suite_profile),
        family=family_key,
    )
    context = _resolve_context_from_spec(spec)
    psi_ref = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi_ref))
    if norm <= 0.0:
        raise ValueError("reference state has zero norm")
    psi_ref = psi_ref / norm
    pool = build_full_meta_candidate_pool(context, max_terms=pool_term_cap)
    selected, theta0, scaffold_meta = _assemble_fixed_full_meta_scaffold(
        pool=pool,
        source_labels=source_labels,
        source_theta=source_theta,
        theta_coordinate_mode=scaffold_theta_coordinate_mode,
        source_parameterization_blocks=source_meta.get("_source_parameterization_blocks"),
        source_parameterization_layout=source_meta.get("_source_parameterization_layout"),
        max_scaffold_terms=max_scaffold_terms,
        missing_source_policy=missing_source_policy,
        pool_indices=pool_indices,
    )
    theta0, warm_start_meta = _extract_fixed_scaffold_warm_start(
        warm_start_json,
        target_selected=selected,
        target_theta0=theta0,
    )
    parameterization_mode = str(scaffold_meta.get("theta_coordinate_mode") or "logical_shared")
    parameterization_layout = scaffold_meta.get("_parameterization_layout")
    pauli_action_cache: dict[str, Any] = {}
    h_compiled = compile_polynomial_action(context.hamiltonian, tol=1e-12, pauli_action_cache=pauli_action_cache)
    initial_state = _prepare_selected_state(
        selected=selected,
        theta=theta0,
        psi_ref=psi_ref,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode=parameterization_mode,
        parameterization_layout=parameterization_layout if isinstance(parameterization_layout, AnsatzParameterLayout) else None,
    )
    initial_energy, _ = energy_via_one_apply(initial_state, h_compiled)
    reference_metrics = _normalize_reference_metric_fields(
        same_cutoff_exact_gs_energy=same_cutoff_exact_gs_energy,
        exact_reference_energy=exact_reference_energy,
        exact_reference_n_ph_max=exact_reference_n_ph_max,
        primary_energy_metric="same_cutoff_abs_delta_e",
        same_cutoff_error_role="primary",
        fallback_same_cutoff_energy=_safe_exact_energy(context),
    )
    same_cutoff_ref = _finite_float_or_none(reference_metrics.get("same_cutoff_exact_gs_energy"))
    external_ref = _finite_float_or_none(reference_metrics.get("exact_reference_energy"))
    primary_ref = _finite_float_or_none(reference_metrics.get("primary_reference_energy"))
    minimize_fn = None
    if any(kind in {"bfgs", "powell"} for kind in optimizers):
        if not has_scipy_minimize_support():
            raise ImportError("scipy.optimize.minimize is required for requested fixed-scaffold optimizers")
        minimize_fn = _import_scipy_minimize()

    source_reported_energy = _finite_float_or_none(source_meta.get("source_energy"))
    source_reconstruction_delta_e = (
        None if source_reported_energy is None else float(float(initial_energy) - float(source_reported_energy))
    )
    source_reconstruction_abs_delta_e = (
        None if source_reconstruction_delta_e is None else float(abs(float(source_reconstruction_delta_e)))
    )
    reconstruction_check = {
        "source_reported_energy": source_reported_energy,
        "reconstructed_initial_energy": float(initial_energy),
        "source_reconstruction_delta_e": source_reconstruction_delta_e,
        "source_reconstruction_abs_delta_e": source_reconstruction_abs_delta_e,
        "source_reconstruction_warning": (
            False if source_reconstruction_abs_delta_e is None else bool(source_reconstruction_abs_delta_e > 1e-10)
        ),
        "note": (
            "Compares the source JSON reported energy to the fixed-scaffold energy reconstructed "
            "from serialized scaffold/theta. Padding terms are initialized at zero."
        ),
    }
    source_scaffold_payload = {key: value for key, value in source_meta.items() if not str(key).startswith("_")}
    source_scaffold_payload.update(reconstruction_check)
    scaffold_payload = {
        **{key: value for key, value in scaffold_meta.items() if not str(key).startswith("_")},
        "operator_labels": [str(candidate.label) for candidate in selected],
        "operator_supports": [list(candidate.support) for candidate in selected],
        "initial_theta": [float(x) for x in np.asarray(theta0, dtype=float).reshape(-1).tolist()],
        "requested_theta_coordinate_mode": requested_theta_coordinate_mode,
        "resolved_theta_coordinate_mode": scaffold_theta_coordinate_mode,
    }

    def _best_fixed_result(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        if not rows:
            return None
        return min(rows, key=lambda row: float(row["energy"]))

    def _audit_payload(
        *,
        status: str,
        results: Sequence[Mapping[str, Any]],
        last_event: str,
        current_optimizer_kind: str | None = None,
        current_optimizer_index: int | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        progress = {
            "last_event": str(last_event),
            "completed_optimizer_count": int(len(results)),
            "total_optimizer_count": int(len(optimizers)),
            "current_optimizer_kind": current_optimizer_kind,
            "current_optimizer_index": current_optimizer_index,
            "pending_optimizer_kinds": [str(item) for item in optimizers[int(len(results)) :]],
        }
        payload = {
            "schema": _FIXED_SCAFFOLD_AUDIT_SCHEMA,
            "status": str(status),
            "runner": f"{_RUNNER_MODULE}.run_fixed_scaffold_expressivity_audit_single",
            "generated_utc": _utc_now(),
            "started_utc": started_utc,
            "runtime_s": float(time.perf_counter() - t0),
            "family": family_key,
            "case_id": case_key,
            "table_i_suite_profile": table_i_suite_profile,
            "algorithm_id": "fixed_full_meta_scaffold_expressivity_audit",
            "method_id": "fixed_full_meta_scaffold_expressivity_audit",
            "run_class": "diagnostic",
            "diagnostic_scope": "fixed_scaffold_expressivity_optimizer_audit",
            "guardrails": {
                "uses_exact_for_decision": False,
                "uses_reference_for_decision": False,
                "exact_reference_usage": "reporting_only_after_fixed_scaffold_optimization",
                "phase3_controller_called": False,
                "static_adapt_controller_boundary": "not_called",
                "paper_promotion_decision": "user_only",
            },
            "progress": progress,
            "spec": _spec_metadata(spec),
            "reference_metrics": reference_metrics,
            "initial_energy": float(initial_energy),
            "initial_same_cutoff_abs_delta_e": (
                None if same_cutoff_ref is None else float(abs(float(initial_energy) - float(same_cutoff_ref)))
            ),
            "source_scaffold": source_scaffold_payload,
            "source_reconstruction_check": reconstruction_check,
            "warm_start": warm_start_meta,
            "scaffold": scaffold_payload,
            "optimizer_kinds": list(optimizers),
            "optimizer_maxiter": int(optimizer_maxiter),
            "metric_floor": float(metric_floor),
            "seed": int(seed),
            "results": [dict(row) for row in results],
            "best_result": None if _best_fixed_result(results) is None else dict(_best_fixed_result(results) or {}),
        }
        if error is not None:
            payload["error"] = dict(error)
        return payload

    def _write_progress(
        *,
        status: str,
        results: Sequence[Mapping[str, Any]],
        last_event: str,
        current_optimizer_kind: str | None = None,
        current_optimizer_index: int | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = _audit_payload(
            status=status,
            results=results,
            last_event=last_event,
            current_optimizer_kind=current_optimizer_kind,
            current_optimizer_index=current_optimizer_index,
            error=error,
        )
        _write_json(output / "fixed_scaffold_expressivity_audit.progress.json", payload)
        _write_json(output / "partial_result.json", payload)
        return payload

    optimizer_results: list[dict[str, Any]] = []
    _write_progress(status="running", results=optimizer_results, last_event="setup_completed")
    for opt_index, kind in enumerate(optimizers):
        optimizer_started_utc = _utc_now()
        optimizer_t0 = time.perf_counter()
        _write_progress(
            status="running",
            results=optimizer_results,
            last_event="optimizer_started",
            current_optimizer_kind=str(kind),
            current_optimizer_index=int(opt_index),
        )
        try:
            theta_opt, energy_opt, info = _run_fixed_scaffold_optimizer(
                optimizer_kind=kind,
                selected=selected,
                theta0=np.asarray(theta0, dtype=float).reshape(-1).copy(),
                psi_ref=psi_ref,
                h_compiled=h_compiled,
                pauli_action_cache=pauli_action_cache,
                optimizer_maxiter=int(optimizer_maxiter),
                metric_floor=float(metric_floor),
                seed=int(seed) + 1009 * (int(opt_index) + 1),
                minimize_fn=minimize_fn,
                parameterization_mode=parameterization_mode,
                parameterization_layout=parameterization_layout if isinstance(parameterization_layout, AnsatzParameterLayout) else None,
            )
        except Exception as exc:
            _write_progress(
                status="failed",
                results=optimizer_results,
                last_event="optimizer_failed",
                current_optimizer_kind=str(kind),
                current_optimizer_index=int(opt_index),
                error={
                    "optimizer_kind": str(kind),
                    "optimizer_index": int(opt_index),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "optimizer_runtime_s": float(time.perf_counter() - optimizer_t0),
                },
            )
            raise
        row = {
            "optimizer_kind": str(kind),
            "optimizer": str(info.get("optimizer", kind)),
            "optimizer_started_utc": optimizer_started_utc,
            "optimizer_completed_utc": _utc_now(),
            "optimizer_runtime_s": float(time.perf_counter() - optimizer_t0),
            "energy": float(energy_opt),
            "energy_initial": float(initial_energy),
            "absolute_delta_e_initial_to_final": float(float(initial_energy) - float(energy_opt)),
            "same_cutoff_abs_delta_e": (
                None if same_cutoff_ref is None else float(abs(float(energy_opt) - float(same_cutoff_ref)))
            ),
            "external_reference_abs_delta_e": (
                None if external_ref is None else float(abs(float(energy_opt) - float(external_ref)))
            ),
            "primary_abs_delta_e": None if primary_ref is None else float(abs(float(energy_opt) - float(primary_ref))),
            "theta": [float(x) for x in np.asarray(theta_opt, dtype=float).reshape(-1).tolist()],
            "theta_l2_norm": float(np.linalg.norm(np.asarray(theta_opt, dtype=float).reshape(-1))),
            "nonzero_theta_count_1e_minus_10": int(np.count_nonzero(np.abs(theta_opt) > 1e-10)),
            "info": dict(info),
        }
        optimizer_results.append(row)
        _write_json(output / f"optimizer_{int(opt_index):02d}_{str(kind)}.json", row)
        _write_progress(
            status="running",
            results=optimizer_results,
            last_event="optimizer_completed",
            current_optimizer_kind=str(kind),
            current_optimizer_index=int(opt_index),
        )

    payload = _audit_payload(status="completed", results=optimizer_results, last_event="completed")
    _write_json(output / "fixed_scaffold_expressivity_audit.json", payload)
    _write_json(output / "result.json", payload)
    return payload


__all__ = [
    "GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS",
    "GENERIC_STATIC_ADAPT_VARIANT_FAMILIES",
    "SCHEMA_VERSION",
    "STATIC_GEO_ADAPT_VQE",
    "STATIC_GEO_QEB_ADAPT_VQE",
    "STATIC_GEO_QUBIT_ADAPT_VQE",
    "STATIC_FULL_META_APPEND_ADAPT_VQE",
    "STATIC_POS_GEO_ADAPT_VQE",
    "STATIC_QUBIT_QEB_ADAPT_VQE",
    "STATIC_TETRIS_QUBIT_ADAPT_VQE",
    "build_full_meta_candidate_pool",
    "build_pairwise_qubit_excitation_pool",
    "default_static_adapt_variant_case_ids",
    "has_scipy_minimize_support",
    "run_fixed_scaffold_expressivity_audit_single",
    "run_generic_static_adapt_variant_single",
]
