#!/usr/bin/env python3
"""Generic static benchmark manifest/dispatch surface.

The first responsibility of this module is CHTC safety: produce one row per
(family, case, algorithm) with unsupported combinations explicitly skipped.
The manifest is conservative: unsupported benchmark rows remain explicit skips.
Only the project controller row and the append-only ADAPT limit dispatch through
the generic static ADAPT runner; external CEO/TETRIS/Overlap rows must not be
emulated by toggling Phase3 internals.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from dataclasses import asdict, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.exact_bench.benchmark_algorithm_registry import (
    compatibility_matrix,
    default_benchmark_algorithms,
    evaluate_algorithm_for_family,
)
from pipelines.exact_bench.external_adapt.provenance import external_algorithm_manifest_metadata
from pipelines.exact_bench.comparator_provenance import (
    comparator_source_fields,
    maybe_comparator_source_profile,
)
from pipelines.exact_bench.benchmark_metrics_proxy import write_proxy_sidecars
from pipelines.exact_bench.molecular_vibronic_h2_fixture_override import (
    with_molecular_vibronic_h2_fixture_override,
)
from pipelines.exact_bench.benchmark_decision_noise import (
    BENCHMARK_DECISION_NOISE_SEMANTIC,
    BenchmarkDecisionNoiseConfig,
    config_from_env as benchmark_decision_noise_config_from_env,
    copy_decision_noise_metadata,
    unsupported_metadata as benchmark_decision_noise_unsupported_metadata,
)
from pipelines.exact_bench.snake_table_i_measurement_work import (
    snake_algorithmic_work_from_payload,
    snake_deterministic_shot_proxy_from_payload,
)
from pipelines.exact_bench.table_i_canonical_cases import (
    table_i_deferred_case_ids,
    table_i_deferred_case_reason,
    table_i_executable_specs,
    table_i_suite_profile,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (
    PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS,
    PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS,
    PAPER_I_MAIN_TABLES_SPSA_DISPLAYED_ALGORITHM_IDS,
    PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES,
    PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    PAPER_I_MAIN_TABLES_SPSA_SNAKE_ALGORITHM_ID,
    normalize_paper_i_main_tables_spsa_profile,
    paper_i_main_tables_spsa_contains_case,
)
from pipelines.reporting.benchmark_manifest import BenchmarkJob, write_manifest_bundle
from pipelines.static_adapt.builders.problem_registry import available_problem_keys
_HH_ALGORITHM_MAP: dict[str, tuple[str, ...]] = {
    "static_hva_vqe": ("hh_hva_termwise_vqe", "hh_hva_layerwise_vqe"),
    "static_uccsd_vqe": ("hh_uccsd_lifted_vqe",),
    "static_avqite_uccsd": ("hh_avqite_uccsd_lifted",),
    "static_qeb_sq_lf_adapt": ("hh_adapt_qeb_sq_lf_std_legacy",),
    "static_lang_firsov_vqe": ("hh_lang_firsov_sq_lf_vqe",),
    "static_qsci_sqd_sq_lf": ("hh_qsci_sq_lf_std", "hh_sqd_sq_lf_std"),
}

_PHASE3_ADAPT_ALGORITHM_IDS = frozenset(
    {
        "static_family_native_adapt_phase3",
        "static_append_only_adapt_phase3",
    }
)
_GENERIC_QISKIT_HEA_ALGORITHM_IDS = frozenset({"static_hea_qiskit_vqe"})
_GENERIC_FAMILY_INFORMED_VQE_ALGORITHM_IDS = frozenset({"static_family_informed_vqe"})
_GENERIC_QISKIT_ADAPTVQE_ALGORITHM_IDS = frozenset({"static_qiskit_adapt_vqe"})
_GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS = frozenset(
    {
        "static_full_meta_append_adapt_vqe",
        "static_qubit_qeb_adapt_vqe",
        "static_geo_qubit_adapt_vqe",
        "static_geo_qeb_adapt_vqe",
        "static_geo_adapt_vqe",
        "static_pos_geo_adapt_vqe",
    }
)
_GENERIC_STATIC_ED_REFERENCE_ALGORITHM_IDS = frozenset({"static_ed_reference"})
_EXTERNAL_STATIC_ADAPT_ALGORITHM_IDS = frozenset({"static_ceo_adapt_phase3", "static_tetris_adapt_phase3"})
_EXTERNAL_STATIC_ADAPT_DISPATCHES = frozenset(
    {"external_static_adapt_ceo_public_code", "external_static_adapt_tetris_public_code"}
)
_BENCHMARK_DECISION_NOISE_SUPPORTED_DISPATCHES = frozenset(
    {
        "generic_static_hea_qiskit_vqe",
        "generic_static_family_informed_vqe",
        "generic_static_adapt_variants",
    }
)
_BENCHMARK_DECISION_NOISE_SUPPORTED_HH_ALGORITHM_IDS = frozenset(
    {
        "static_hva_vqe",
        "static_uccsd_vqe",
        "static_lang_firsov_vqe",
    }
)
_ENERGY_STOP_TARGET_ENV = "GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET"
_FIRST_HIT_THRESHOLDS_ENV = "GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS"
_GENERIC_STATIC_ADAPT_SEED_ENV = "GENERIC_STATIC_TABLE_ADAPT_SEED"
_GENERIC_STATIC_ADAPT_STOP_POLICY_ENV = "GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY"
_GENERIC_STATIC_ADAPT_STOP_POLICY_DEFAULT = "default"
_GENERIC_STATIC_ADAPT_STOP_POLICY_FIXED_HORIZON_NO_TARGET = "fixed_horizon_no_target_v1"
_GENERIC_STATIC_ADAPT_STOP_POLICY_CHOICES = frozenset(
    {
        _GENERIC_STATIC_ADAPT_STOP_POLICY_DEFAULT,
        _GENERIC_STATIC_ADAPT_STOP_POLICY_FIXED_HORIZON_NO_TARGET,
    }
)
_POWELL_MAXITER_CAP_POLICY_ENV = "GENERIC_STATIC_TABLE_POWELL_MAXITER_CAP_POLICY"
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
_HH_ADAPTIVE_POOL_PROFILE_ENV = "GENERIC_STATIC_TABLE_HH_ADAPTIVE_POOL_PROFILE"
_HH_FULL_META_CLASS_FILTER_JSON_ENV = "GENERIC_STATIC_TABLE_HH_FULL_META_CLASS_FILTER_JSON"
_GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF = "off"
_GENERIC_ADAPT_RUNTIME_SPLIT_MODE_PAULI_CHILDREN = "shortlist_pauli_children_v1"
_GENERIC_ADAPT_RUNTIME_SPLIT_MODE_CHOICES = frozenset(
    {
        _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF,
        _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_PAULI_CHILDREN,
    }
)
_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_CHOICES = frozenset({"off", "hard_guard"})
_SHARED_PAULI_POOL_MODE_OFF = "off"
_SHARED_PAULI_POOL_MODE_CHILD_SETS_V1 = "shared_pauli_child_sets_v1"
_SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1 = (
    "projected_singleton_children_only_v1"
)
_SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1 = (
    "guarded_singleton_children_only_v1"
)
_SHARED_PAULI_POOL_MODE_CHOICES = frozenset(
    {
        _SHARED_PAULI_POOL_MODE_OFF,
        _SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
        _SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
        _SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1,
        "pauli_child_sets_v1",
        "global_pauli_child_sets_v1",
    }
)
_SHARED_PAULI_POOL_SYMMETRY_POLICY_CHOICES = frozenset({"off", "hard_guard"})
_SAME_CUTOFF_EXACT_GS_ENERGY_ENV = "GENERIC_STATIC_TABLE_SAME_CUTOFF_EXACT_GS_ENERGY"
_EXACT_REFERENCE_ENERGY_ENV = "GENERIC_STATIC_TABLE_EXACT_REFERENCE_ENERGY"
_EXACT_REFERENCE_N_PH_MAX_ENV = "GENERIC_STATIC_TABLE_EXACT_REFERENCE_N_PH_MAX"
_PRIMARY_ENERGY_METRIC_ENV = "GENERIC_STATIC_TABLE_PRIMARY_ENERGY_METRIC"
_SAME_CUTOFF_ERROR_ROLE_ENV = "GENERIC_STATIC_TABLE_SAME_CUTOFF_ERROR_ROLE"
_SHOTS_PER_PAULI_TERM_PROXY_ENV = "GENERIC_STATIC_TABLE_SHOTS_PER_PAULI_TERM_PROXY"
_GENERIC_STATIC_ADAPT_PROGRESS_JSONL_ENV = "GENERIC_STATIC_TABLE_PROGRESS_JSONL_PATH"
_GENERIC_ADAPT_RUNTIME_SPLIT_TSV_FIELDS = (
    "generic_adapt_runtime_split_mode",
    "generic_adapt_runtime_split_symmetry_policy",
    "generic_adapt_runtime_split_max_subset_size",
)
_SHARED_PAULI_POOL_TSV_FIELDS = (
    "shared_pauli_pool_mode",
    "shared_pauli_pool_symmetry_policy",
    "shared_pauli_pool_max_subset_size",
)
_PHASE3_POLICY_JSON_ENV = "GENERIC_STATIC_TABLE_PHASE3_POLICY_JSON"
_HEA_REPS_ENV = "GENERIC_STATIC_TABLE_HEA_REPS"
_HEA_MAXITER_ENV = "GENERIC_STATIC_TABLE_HEA_MAXITER"
_PHASE3_ORACLE_TSV_FIELDS = (
    "phase3_oracle_gradient_mode",
    "phase3_oracle_backend_name",
    "phase3_oracle_use_fake_backend",
    "phase3_oracle_shots",
    "phase3_oracle_repeats",
    "phase3_oracle_aggregate",
    "phase3_oracle_seed",
    "phase3_oracle_execution_surface",
    "phase3_oracle_inner_objective_mode",
    "phase3_oracle_value_noise_model",
    "phase3_oracle_value_noise_std",
    "phase3_oracle_value_noise_seed",
)
_PHASE3_BUDGET_TSV_FIELDS = (
    "phase3_adapt_max_depth",
    "phase3_adapt_maxiter",
    "phase3_refit_maxiter",
    "phase3_final_maxiter",
    "phase3_adapt_spsa_a",
    "phase3_adapt_spsa_c",
    "phase3_adapt_spsa_big_a",
    "phase3_adapt_spsa_alpha",
    "phase3_adapt_spsa_gamma",
    "phase3_adapt_spsa_eval_repeats",
    "phase3_adapt_spsa_avg_last",
    "phase3_adapt_allow_repeats",
)
_PHASE3_RUNTIME_TSV_FIELDS = (
    "phase3_adapt_parallel_gradient_workers",
    "phase3_adapt_beam_parent_workers",
)
_HARDWARE_RESOLUTION_PROFILE_TSV_FIELDS = (
    "hardware_resolution_mode",
    "hardware_resolution_profile_json",
    "hardware_resolution_profile_name",
)
_RETIRED_STATIC_ROUTE_ENV_NAMES = (
    "GENERIC_STATIC_TABLE_STATIC_ROUTE_ID",
    "STATIC_ROUTE_ID",
)
_SELECTED_LOGICAL_TSV_FIELDS = (
    "selected_logical_route",
    "selected_logical_source_json",
    "selected_logical_transfer_mode",
)
_PHASE3_BUDGET_POLICY_FIELDS = {
    "phase3_adapt_max_depth": "adapt_max_depth",
    "phase3_adapt_maxiter": "adapt_maxiter",
    "phase3_refit_maxiter": "refit_maxiter",
    "phase3_final_maxiter": "final_maxiter",
    "phase3_adapt_spsa_a": "spsa_a",
    "phase3_adapt_spsa_c": "spsa_c",
    "phase3_adapt_spsa_big_a": "spsa_A",
    "phase3_adapt_spsa_alpha": "spsa_alpha",
    "phase3_adapt_spsa_gamma": "spsa_gamma",
    "phase3_adapt_spsa_eval_repeats": "adapt_spsa_eval_repeats",
    "phase3_adapt_spsa_avg_last": "adapt_spsa_avg_last",
    "phase3_adapt_allow_repeats": "adapt_allow_repeats",
}
_PHASE3_BUDGET_STATIC_POLICY_FIELDS = {"adapt_max_depth", "adapt_maxiter", "adapt_allow_repeats"}
_PHASE3_BUDGET_INNER_POLICY_FIELDS = {
    "refit_maxiter",
    "final_maxiter",
    "spsa_a",
    "spsa_c",
    "spsa_A",
    "spsa_alpha",
    "spsa_gamma",
}
_PHASE3_BUDGET_BASE_ARG_FLAGS = {
    "adapt_spsa_eval_repeats": "--adapt-spsa-eval-repeats",
    "adapt_spsa_avg_last": "--adapt-spsa-avg-last",
}
_PHASE3_RUNTIME_POLICY_FIELDS = {
    "phase3_adapt_parallel_gradient_workers": "adapt_parallel_gradient_workers",
    "phase3_adapt_beam_parent_workers": "adapt_beam_parent_workers",
}
_PHASE3_RUNTIME_BASE_ARG_FLAGS = {
    "adapt_parallel_gradient_workers": "--adapt-parallel-gradient-workers",
    "adapt_beam_parent_workers": "--adapt-beam-parent-workers",
}
_PHASE3_ORACLE_GRADIENT_MODE_CHOICES = {
    "off",
    "ideal",
    "shots",
    "aer_noise",
    "aer_density_matrix",
    "backend_scheduled",
    "runtime",
}
_PHASE3_ORACLE_EXECUTION_SURFACE_CHOICES = {"auto", "expectation_v1", "raw_measurement_v1"}
_PHASE3_ORACLE_VALUE_NOISE_MODEL_CHOICES = {"off", "gaussian_iid_v1"}
_BENCHMARK_VALUE_NOISE_TSV_FIELDS = (
    "benchmark_value_noise_model",
    "benchmark_value_noise_std",
    "benchmark_value_noise_seed",
)
_BENCHMARK_VALUE_NOISE_MODEL_CHOICES = {"off", "gaussian_iid_v1"}
_BENCHMARK_VALUE_NOISE_SEMANTIC = "post_static_result_value_noise_not_physical_shots"
_BENCHMARK_VALUE_NOISE_EXACT_ENERGY_KEYS = (
    "exact_energy",
    "exact_gs_energy",
    "exact_reference_energy",
    "same_cutoff_exact_gs_energy",
    "target_exact_energy",
    "exact_energy_total",
)
_BENCHMARK_VALUE_NOISE_DELTA_FIELDS = ("delta_E_abs", "abs_delta_e")
_BENCHMARK_VALUE_NOISE_JSON_ARTIFACTS = (
    "generic_static_single.json",
    "result.json",
    "manifest.json",
    "rows.json",
    "hh_static_benchmark_result.json",
    "hh_static_benchmark_manifest.json",
    "hh_static_benchmark_rows.json",
)


def _env_float_or_none(name: str) -> float | None:
    raw = os.environ.get(name, "")
    if not raw:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value if value > 0.0 else None


def _env_int_or_none(name: str, *, min_value: int = 1) -> int | None:
    raw = os.environ.get(name, "")
    if not raw:
        return None
    try:
        value = int(str(raw).strip())
    except Exception:
        return None
    return int(value) if value >= int(min_value) else None


def _generic_static_adapt_seed_from_env() -> int | None:
    raw = _env_text_or_none(_GENERIC_STATIC_ADAPT_SEED_ENV)
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except Exception as exc:
        raise ValueError(
            f"{_GENERIC_STATIC_ADAPT_SEED_ENV} must be an integer >= 0; got {raw!r}."
        ) from exc
    if value < 0:
        raise ValueError(
            f"{_GENERIC_STATIC_ADAPT_SEED_ENV} must be an integer >= 0; got {raw!r}."
        )
    return int(value)


def _shots_per_pauli_term_proxy_from_env() -> int | None:
    raw = _env_text_or_none(_SHOTS_PER_PAULI_TERM_PROXY_ENV)
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except Exception as exc:
        raise ValueError(
            f"{_SHOTS_PER_PAULI_TERM_PROXY_ENV} must be a positive integer-like value when provided; got {raw!r}."
        ) from exc
    if not math.isfinite(value) or not value.is_integer() or value < 1:
        raise ValueError(
            f"{_SHOTS_PER_PAULI_TERM_PROXY_ENV} must be a positive integer-like value when provided; got {raw!r}."
        )
    return int(value)


def _env_first_hit_thresholds() -> tuple[float, ...]:
    raw = os.environ.get(_FIRST_HIT_THRESHOLDS_ENV, "")
    if not raw:
        return ()
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except Exception:
            continue
        if value > 0.0:
            out.append(value)
    return tuple(out)


def _env_text_or_none(name: str) -> str | None:
    raw = os.environ.get(str(name), "")
    text = str(raw).strip()
    return text or None


def _generic_static_adapt_stop_policy_from_env() -> str:
    raw = _env_text_or_none(_GENERIC_STATIC_ADAPT_STOP_POLICY_ENV)
    if raw is None:
        return _GENERIC_STATIC_ADAPT_STOP_POLICY_DEFAULT
    policy = str(raw).strip()
    if policy not in _GENERIC_STATIC_ADAPT_STOP_POLICY_CHOICES:
        allowed = ", ".join(sorted(_GENERIC_STATIC_ADAPT_STOP_POLICY_CHOICES))
        raise ValueError(
            f"Unsupported generic static ADAPT stop policy {policy!r}; allowed values: {allowed}."
        )
    return policy


def _powell_maxiter_cap_policy_from_env() -> str:
    raw = _env_text_or_none(_POWELL_MAXITER_CAP_POLICY_ENV)
    if raw is None:
        return _POWELL_MAXITER_CAP_POLICY_STRICT
    policy = str(raw).strip().lower()
    if policy not in _POWELL_MAXITER_CAP_POLICY_CHOICES:
        allowed = ", ".join(sorted(_POWELL_MAXITER_CAP_POLICY_CHOICES))
        raise ValueError(
            f"Unsupported Powell maxiter cap policy {policy!r}; allowed values: {allowed}."
        )
    return policy


def _env_reference_kwargs() -> dict[str, str]:
    mapping = {
        "same_cutoff_exact_gs_energy": _SAME_CUTOFF_EXACT_GS_ENERGY_ENV,
        "exact_reference_energy": _EXACT_REFERENCE_ENERGY_ENV,
        "exact_reference_n_ph_max": _EXACT_REFERENCE_N_PH_MAX_ENV,
        "primary_energy_metric": _PRIMARY_ENERGY_METRIC_ENV,
        "same_cutoff_error_role": _SAME_CUTOFF_ERROR_ROLE_ENV,
    }
    out: dict[str, str] = {}
    for key, env_name in mapping.items():
        value = _env_text_or_none(env_name)
        if value is not None:
            out[key] = value
    return out


def _env_hh_pool_kwargs() -> dict[str, str]:
    out: dict[str, str] = {}
    profile = _env_text_or_none(_HH_ADAPTIVE_POOL_PROFILE_ENV)
    if profile is not None:
        out["hh_adaptive_pool_profile"] = profile
    class_filter = _env_text_or_none(_HH_FULL_META_CLASS_FILTER_JSON_ENV)
    if class_filter is not None:
        out["hh_full_meta_class_filter_json"] = class_filter
    return out


_OPTIMIZER_PROFILE_FIELD = "optimizer_profile"
_HEA_OPTIMIZER_SCHEDULE_FIELDS = frozenset(PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS)
_FAMILY_INFORMED_OPTIMIZER_SCHEDULE_FIELDS = frozenset(
    PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS
)
_ADAPT_OPTIMIZER_SCHEDULE_FIELDS = frozenset(PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS)
_HEA_OPTIMIZER_FIELDS = frozenset({"hea_optimizer", "hea_spsa_maxiter", "hea_spsa_seed"}) | _HEA_OPTIMIZER_SCHEDULE_FIELDS
_FAMILY_INFORMED_OPTIMIZER_FIELDS = (
    frozenset({"family_informed_optimizer", "family_informed_spsa_maxiter", "family_informed_spsa_seed"})
    | _FAMILY_INFORMED_OPTIMIZER_SCHEDULE_FIELDS
)
_ADAPT_OPTIMIZER_FIELDS = (
    frozenset({"adapt_optimizer_kind", "adapt_spsa_maxiter", "adapt_spsa_seed"})
    | _ADAPT_OPTIMIZER_SCHEDULE_FIELDS
)
_OPTIMIZER_INT_FIELDS = frozenset(
    {
        "hea_spsa_maxiter",
        "hea_spsa_seed",
        "family_informed_spsa_maxiter",
        "family_informed_spsa_seed",
        "family_informed_spsa_eval_repeats",
        "family_informed_spsa_avg_last",
        "adapt_spsa_maxiter",
        "adapt_spsa_seed",
    }
)
_OPTIMIZER_FLOAT_FIELDS = (
    _HEA_OPTIMIZER_SCHEDULE_FIELDS
    | (_FAMILY_INFORMED_OPTIMIZER_SCHEDULE_FIELDS - {"family_informed_spsa_eval_repeats", "family_informed_spsa_avg_last"})
    | _ADAPT_OPTIMIZER_SCHEDULE_FIELDS
)
_OPTIMIZER_SUPPORTED_DISPATCHES = frozenset(
    {
        "generic_static_hea_qiskit_vqe",
        "generic_static_family_informed_vqe",
        "generic_static_adapt_variants",
    }
)


def _optimizer_env_name(field: str) -> str:
    return PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES.get(
        str(field),
        "GENERIC_STATIC_TABLE_" + str(field).upper(),
    )


def _optimizer_env_value(field: str) -> str | None:
    # Optimizer profile fields are TSV/record-runner controlled.  Unlike older
    # overlays, do not read unprefixed names such as HEA_OPTIMIZER: a blank
    # GENERIC_STATIC_TABLE_* field must preserve legacy defaults even if a stale
    # shell variable exists in the parent environment.
    raw = os.environ.get(_optimizer_env_name(field))
    if raw not in {None, ""}:
        return str(raw).strip()
    return None


def _parse_optimizer_env_int(raw: str, *, field: str) -> int:
    try:
        value = int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"{field} must be a positive integer when provided; got {raw!r}.") from exc
    if value < 1:
        raise ValueError(f"{field} must be a positive integer when provided; got {raw!r}.")
    return int(value)


def _parse_optimizer_env_float(raw: str, *, field: str) -> float:
    try:
        value = float(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"{field} must be a positive finite float when provided; got {raw!r}.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be a positive finite float when provided; got {raw!r}.")
    return float(value)


def _optimizer_env_values_from_env() -> dict[str, object]:
    values: dict[str, object] = {}
    for field in PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS:
        raw = _optimizer_env_value(field)
        if raw in {None, ""}:
            continue
        if field == _OPTIMIZER_PROFILE_FIELD:
            profile = normalize_paper_i_main_tables_spsa_profile(str(raw))
            if profile is not None:
                values[field] = profile
        elif field in _OPTIMIZER_INT_FIELDS:
            values[field] = _parse_optimizer_env_int(str(raw), field=field)
        elif field in _OPTIMIZER_FLOAT_FIELDS:
            values[field] = _parse_optimizer_env_float(str(raw), field=field)
        else:
            values[field] = str(raw).strip().lower()
    return values


def _optimizer_fields_requested(values: Mapping[str, object], fields: frozenset[str]) -> bool:
    return any(field in values for field in fields)


def _validate_optimizer_choice(value: object, *, field: str, allowed: set[str]) -> str:
    key = str(value).strip().lower()
    if key not in allowed:
        known = ", ".join(sorted(allowed))
        raise ValueError(f"{field} must be one of {{{known}}} when provided; got {value!r}.")
    return key


def _profile_requires_spsa(value: str, *, field: str, profile: str) -> None:
    if str(value).strip().lower() != "spsa":
        raise ValueError(f"optimizer_profile={profile} requires {field}=spsa; got {value!r}.")


def _schedule_fields_requested(values: Mapping[str, object], fields: frozenset[str]) -> bool:
    return any(field in values for field in fields)


def _schedule_requires_spsa(
    values: Mapping[str, object],
    fields: frozenset[str],
    *,
    optimizer_value: str,
    optimizer_field: str,
) -> None:
    if _schedule_fields_requested(values, fields) and str(optimizer_value).strip().lower() != "spsa":
        names = ", ".join(sorted(field for field in fields if field in values))
        raise ValueError(f"SPSA schedule fields {{{names}}} require {optimizer_field}=spsa; got {optimizer_value!r}.")


def _copy_requested_fields(out: dict[str, Any], values: Mapping[str, object], fields: frozenset[str]) -> None:
    for field in sorted(fields):
        if field in values:
            out[field] = values[field]


def _optimizer_dispatch_overrides_from_env(
    *,
    dispatch: str | None,
    family: str,
    case_id: str,
    algorithm_id: str,
) -> dict[str, Any]:
    """Parse optimizer profile/env overlays and validate dispatch ownership.

    This is plumbing only.  Method runners must explicitly accept the returned
    keyword names before the overlay can execute; otherwise ``run_single`` fails
    closed with a clear runner-interface error.
    """

    values = _optimizer_env_values_from_env()
    if not values:
        return {}

    profile = values.get(_OPTIMIZER_PROFILE_FIELD)
    if _optimizer_fields_requested(values, _HEA_OPTIMIZER_FIELDS) and dispatch != "generic_static_hea_qiskit_vqe":
        raise ValueError(
            "HEA optimizer env overlay is only valid for generic_static_hea_qiskit_vqe records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, algorithm_id={algorithm_id!r}."
        )
    if (
        _optimizer_fields_requested(values, _FAMILY_INFORMED_OPTIMIZER_FIELDS)
        and dispatch != "generic_static_family_informed_vqe"
    ):
        raise ValueError(
            "family-informed optimizer env overlay is only valid for generic_static_family_informed_vqe records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, algorithm_id={algorithm_id!r}."
        )
    if _optimizer_fields_requested(values, _ADAPT_OPTIMIZER_FIELDS) and dispatch != "generic_static_adapt_variants":
        raise ValueError(
            "generic-ADAPT optimizer env overlay is only valid for generic_static_adapt_variants records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, algorithm_id={algorithm_id!r}."
        )

    if profile == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID:
        if not paper_i_main_tables_spsa_contains_case(family, case_id):
            raise ValueError(
                f"optimizer_profile={profile} is only valid for the exact visible Paper-I main-table case set; "
                f"got family={family!r}, case_id={case_id!r}."
            )
        if str(algorithm_id) not in PAPER_I_MAIN_TABLES_SPSA_DISPLAYED_ALGORITHM_IDS:
            raise ValueError(
                f"optimizer_profile={profile} is only valid for displayed Paper-I methods; "
                f"got algorithm_id={algorithm_id!r}."
            )
        if str(algorithm_id) == PAPER_I_MAIN_TABLES_SPSA_SNAKE_ALGORITHM_ID:
            raise ValueError(
                f"optimizer_profile={profile} SNAKE/Route-A propagation is outside generic comparator dispatch; "
                "Route-A/SNAKE profile records are handled by the dedicated Phase3 generator."
            )
        if str(algorithm_id) not in PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS or dispatch not in _OPTIMIZER_SUPPORTED_DISPATCHES:
            raise ValueError(
                f"optimizer_profile={profile} has no generic comparator optimizer dispatch for "
                f"dispatch={dispatch!r}, algorithm_id={algorithm_id!r}."
            )
    elif profile is not None:
        # ``normalize_paper_i_main_tables_spsa_profile`` currently raises before
        # this point for unknown nonblank profiles.  Keep this guard explicit in
        # case another profile is added without dispatch plumbing.
        raise ValueError(f"Unsupported optimizer_profile {profile!r} for generic static dispatch.")

    if dispatch == "generic_static_hea_qiskit_vqe" and (
        profile is not None or _optimizer_fields_requested(values, _HEA_OPTIMIZER_FIELDS)
    ):
        defaults = PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["hea"]
        optimizer = _validate_optimizer_choice(
            values.get("hea_optimizer", defaults["optimizer"] if profile else "spsa"),
            field="hea_optimizer",
            allowed={"cobyla", "spsa"},
        )
        if profile:
            _profile_requires_spsa(optimizer, field="hea_optimizer", profile=str(profile))
        _schedule_requires_spsa(
            values,
            _HEA_OPTIMIZER_SCHEDULE_FIELDS,
            optimizer_value=optimizer,
            optimizer_field="hea_optimizer",
        )
        hea_has_learning_rate = "hea_spsa_learning_rate" in values
        hea_has_perturbation = "hea_spsa_perturbation" in values
        if hea_has_learning_rate != hea_has_perturbation:
            raise ValueError(
                "HEA Qiskit SPSA schedule requires hea_spsa_learning_rate and "
                "hea_spsa_perturbation to be provided together."
            )
        out: dict[str, Any] = {
            "optimizer_profile": profile,
            "optimizer_profile_source": "env" if profile else None,
            "hea_optimizer": optimizer,
            "optimizer_overlay_source": "generic_static_benchmark_env",
        }
        if optimizer == "spsa":
            out["hea_spsa_maxiter"] = int(values.get("hea_spsa_maxiter", defaults["spsa_maxiter"]))
            out["hea_spsa_seed"] = int(values.get("hea_spsa_seed", defaults["spsa_seed"]))
            _copy_requested_fields(out, values, _HEA_OPTIMIZER_SCHEDULE_FIELDS)
        return out

    if dispatch == "generic_static_family_informed_vqe" and (
        profile is not None or _optimizer_fields_requested(values, _FAMILY_INFORMED_OPTIMIZER_FIELDS)
    ):
        defaults = PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["family_informed"]
        optimizer = _validate_optimizer_choice(
            values.get("family_informed_optimizer", defaults["optimizer"] if profile else "spsa"),
            field="family_informed_optimizer",
            allowed={"bfgs", "spsa"},
        )
        if profile:
            _profile_requires_spsa(optimizer, field="family_informed_optimizer", profile=str(profile))
        _schedule_requires_spsa(
            values,
            _FAMILY_INFORMED_OPTIMIZER_SCHEDULE_FIELDS,
            optimizer_value=optimizer,
            optimizer_field="family_informed_optimizer",
        )
        out = {
            "optimizer_profile": profile,
            "optimizer_profile_source": "env" if profile else None,
            "family_informed_optimizer": optimizer,
            "optimizer_overlay_source": "generic_static_benchmark_env",
        }
        if optimizer == "spsa":
            out["family_informed_spsa_maxiter"] = int(
                values.get("family_informed_spsa_maxiter", defaults["spsa_maxiter"])
            )
            out["family_informed_spsa_seed"] = int(values.get("family_informed_spsa_seed", defaults["spsa_seed"]))
            _copy_requested_fields(out, values, _FAMILY_INFORMED_OPTIMIZER_SCHEDULE_FIELDS)
        return out

    if dispatch == "generic_static_adapt_variants" and (
        profile is not None or _optimizer_fields_requested(values, _ADAPT_OPTIMIZER_FIELDS)
    ):
        defaults = PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["adapt"]
        optimizer_kind = _validate_optimizer_choice(
            values.get("adapt_optimizer_kind", defaults["optimizer_kind"] if profile else "spsa"),
            field="adapt_optimizer_kind",
            allowed={"bfgs", "geo_qngd", "powell", "rotosolve", "spsa"},
        )
        if profile:
            _profile_requires_spsa(optimizer_kind, field="adapt_optimizer_kind", profile=str(profile))
        _schedule_requires_spsa(
            values,
            _ADAPT_OPTIMIZER_SCHEDULE_FIELDS,
            optimizer_value=optimizer_kind,
            optimizer_field="adapt_optimizer_kind",
        )
        out = {
            "optimizer_profile": profile,
            "optimizer_profile_source": "env" if profile else None,
            "adapt_optimizer_kind": optimizer_kind,
            "optimizer_overlay_source": "generic_static_benchmark_env",
        }
        if optimizer_kind == "spsa":
            out["adapt_spsa_maxiter"] = int(values.get("adapt_spsa_maxiter", defaults["spsa_maxiter"]))
            out["adapt_spsa_seed"] = int(values.get("adapt_spsa_seed", defaults["spsa_seed"]))
            _copy_requested_fields(out, values, _ADAPT_OPTIMIZER_SCHEDULE_FIELDS)
        return out

    if profile is not None:
        raise ValueError(
            f"optimizer_profile={profile} is not valid for dispatch={dispatch!r}, algorithm_id={algorithm_id!r}."
        )
    return {}


def _call_with_optimizer_overrides(
    runner: Any,
    kwargs: dict[str, Any],
    optimizer_overrides: Mapping[str, Any],
    *,
    dispatch: str,
) -> dict[str, Any]:
    if not optimizer_overrides:
        return runner(**kwargs)
    signature = inspect.signature(runner)
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    missing = [key for key in optimizer_overrides if key not in signature.parameters]
    if missing and not accepts_var_kwargs:
        runner_name = getattr(runner, "__qualname__", getattr(runner, "__name__", str(runner)))
        raise ValueError(
            "Optimizer profile/env overlay was parsed for "
            f"{dispatch}, but runner {runner_name} does not accept optimizer plumbing kwargs {sorted(missing)}. "
            "Method-specific optimizer internals are intentionally not implemented in this plumbing subset."
        )
    return runner(**kwargs, **dict(optimizer_overrides))


def _phase3_oracle_env_name(field: str) -> str:
    return "GENERIC_STATIC_TABLE_" + str(field).upper()


def _phase3_budget_env_name(field: str) -> str:
    return "GENERIC_STATIC_TABLE_" + str(field).upper()


def _phase3_runtime_env_name(field: str) -> str:
    return "GENERIC_STATIC_TABLE_" + str(field).upper()


def _hardware_resolution_profile_env_name(field: str) -> str:
    return "GENERIC_STATIC_TABLE_" + str(field).upper()


def _benchmark_value_noise_env_name(field: str) -> str:
    return "GENERIC_STATIC_TABLE_" + str(field).upper()


def _phase3_oracle_env_value(field: str) -> str | None:
    for name in (_phase3_oracle_env_name(field), str(field).upper()):
        raw = os.environ.get(name)
        if raw not in {None, ""}:
            return str(raw).strip()
    return None


def _benchmark_value_noise_env_value(field: str) -> str | None:
    for name in (_benchmark_value_noise_env_name(field), str(field).upper()):
        raw = os.environ.get(name)
        if raw not in {None, ""}:
            return str(raw).strip()
    return None


def _parse_phase3_oracle_bool(raw: str, *, field: str) -> bool:
    key = str(raw).strip().lower()
    if key in {"1", "true", "yes", "on"}:
        return True
    if key in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{field} must be boolean-like when provided; got {raw!r}.")


def _parse_phase3_oracle_int(raw: str, *, field: str) -> int:
    try:
        value = int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"{field} must be a positive integer when provided; got {raw!r}.") from exc
    if value < 1:
        raise ValueError(f"{field} must be a positive integer when provided; got {raw!r}.")
    return int(value)


def _phase3_budget_env_value(field: str) -> str | None:
    for name in (_phase3_budget_env_name(field), str(field).upper()):
        raw = os.environ.get(name)
        if raw not in {None, ""}:
            return str(raw).strip()
    return None


def _phase3_runtime_env_value(field: str) -> str | None:
    for name in (_phase3_runtime_env_name(field), str(field).upper()):
        raw = os.environ.get(name)
        if raw not in {None, ""}:
            return str(raw).strip()
    return None


def _hardware_resolution_profile_env_value(field: str) -> str | None:
    for name in (_hardware_resolution_profile_env_name(field), str(field).upper()):
        raw = os.environ.get(name)
        if raw not in {None, ""}:
            return str(raw).strip()
    return None


def _selected_logical_env_name(field: str) -> str:
    return "GENERIC_STATIC_TABLE_" + str(field).upper()


def _generic_adapt_runtime_split_env_name(field: str) -> str:
    return "GENERIC_STATIC_TABLE_" + str(field).upper()


def _shared_pauli_pool_env_name(field: str) -> str:
    return "GENERIC_STATIC_TABLE_" + str(field).upper()


def _selected_logical_env_value(field: str) -> str | None:
    for name in (_selected_logical_env_name(field), str(field).upper()):
        raw = os.environ.get(name)
        if raw not in {None, ""}:
            return str(raw).strip()
    return None


def _generic_adapt_runtime_split_env_value(field: str) -> str | None:
    for name in (_generic_adapt_runtime_split_env_name(field), str(field).upper()):
        raw = os.environ.get(name)
        if raw not in {None, ""}:
            return str(raw).strip()
    return None


def _shared_pauli_pool_env_value(field: str) -> str | None:
    for name in (_shared_pauli_pool_env_name(field), str(field).upper()):
        raw = os.environ.get(name)
        if raw not in {None, ""}:
            return str(raw).strip()
    return None


def _parse_phase3_budget_int(raw: str, *, field: str, min_value: int = 1) -> int:
    try:
        value = int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"{field} must be an integer >= {int(min_value)} when provided; got {raw!r}.") from exc
    if value < int(min_value):
        raise ValueError(f"{field} must be an integer >= {int(min_value)} when provided; got {raw!r}.")
    return int(value)


def _parse_phase3_oracle_float(raw: str, *, field: str) -> float:
    try:
        value = float(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"{field} must be finite numeric when provided; got {raw!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric when provided; got {raw!r}.")
    return float(value)


def _parse_phase3_budget_bool(raw: str, *, field: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{field} must be boolean when provided; got {raw!r}.")


def _phase3_budget_policy_overrides_from_env() -> dict[str, object]:
    overrides: dict[str, object] = {}
    for field in _PHASE3_BUDGET_TSV_FIELDS:
        raw = _phase3_budget_env_value(field)
        if raw in {None, ""}:
            continue
        policy_field = _PHASE3_BUDGET_POLICY_FIELDS[field]
        if field == "phase3_adapt_allow_repeats":
            overrides[policy_field] = _parse_phase3_budget_bool(str(raw), field=field)
        elif policy_field in {"spsa_a", "spsa_c", "spsa_A", "spsa_alpha", "spsa_gamma"}:
            overrides[policy_field] = _parse_phase3_oracle_float(str(raw), field=field)
        else:
            min_value = 0 if field == "phase3_adapt_spsa_avg_last" else 1
            overrides[policy_field] = _parse_phase3_budget_int(str(raw), field=field, min_value=min_value)
    return overrides


def _phase3_budget_policy_overrides_requested(overrides: dict[str, object]) -> bool:
    return bool(overrides)


def _phase3_runtime_policy_overrides_from_env() -> dict[str, int]:
    overrides: dict[str, int] = {}
    for field in _PHASE3_RUNTIME_TSV_FIELDS:
        raw = _phase3_runtime_env_value(field)
        if raw in {None, ""}:
            continue
        policy_field = _PHASE3_RUNTIME_POLICY_FIELDS[field]
        overrides[policy_field] = _parse_phase3_budget_int(str(raw), field=field, min_value=0)
    return overrides


def _phase3_runtime_policy_overrides_requested(overrides: dict[str, int]) -> bool:
    return bool(overrides)


def _hardware_resolution_profile_policy_overrides_from_env() -> dict[str, object]:
    raw = {field: _hardware_resolution_profile_env_value(field) for field in _HARDWARE_RESOLUTION_PROFILE_TSV_FIELDS}
    if not any(value not in {None, ""} for value in raw.values()):
        return {}
    mode = str(raw.get("hardware_resolution_mode") or "").strip().lower()
    profile_json = str(raw.get("hardware_resolution_profile_json") or "").strip()
    profile_name = str(raw.get("hardware_resolution_profile_name") or "").strip()
    if mode != "profile":
        raise ValueError("hardware_resolution_mode must be 'profile' for hardware-resolution profile env overlay.")
    if not profile_json or not profile_name:
        raise ValueError(
            "hardware-resolution profile env overlay requires hardware_resolution_profile_json "
            "and hardware_resolution_profile_name together."
        )
    return {
        "hardware_resolution_mode": "profile",
        "hardware_resolution_profile_json": profile_json,
        "hardware_resolution_profile_name": profile_name,
    }


def _hardware_resolution_profile_policy_overrides_requested(overrides: dict[str, object]) -> bool:
    return bool(overrides)


def _static_route_policy_overrides_from_env() -> dict[str, object]:
    configured = {
        name: str(os.environ[name]).strip()
        for name in _RETIRED_STATIC_ROUTE_ENV_NAMES
        if str(os.environ.get(name, "")).strip()
    }
    if configured:
        names = ", ".join(sorted(configured))
        raise ValueError(
            "Historical static-route execution controls are retired from the "
            f"generic benchmark launcher ({names}). Canonical RA-ADAPT and "
            "Append-ADAPT execution must use their typed source-locked bundle "
            "protocols."
        )
    return {}


def _selected_logical_overrides_from_env() -> dict[str, str]:
    raw = {field: _selected_logical_env_value(field) for field in _SELECTED_LOGICAL_TSV_FIELDS}
    if not any(value not in {None, ""} for value in raw.values()):
        return {}
    route = str(raw.get("selected_logical_route") or "standard").strip().lower().replace("-", "_")
    if route not in {"standard", "historical_selected"}:
        raise ValueError("selected_logical_route must be one of {'standard','historical_selected'}.")
    source_json = str(raw.get("selected_logical_source_json") or "").strip()
    transfer_mode = str(raw.get("selected_logical_transfer_mode") or "exact_match_v1").strip().lower()
    if transfer_mode not in {"exact_match_v1", "boundary_v1"}:
        raise ValueError("selected_logical_transfer_mode must be one of {'exact_match_v1','boundary_v1'}.")
    if route == "historical_selected" and not source_json:
        raise ValueError("selected_logical_source_json is required when selected_logical_route=historical_selected.")
    return {
        "selected_logical_route": route,
        "selected_logical_source_json": source_json,
        "selected_logical_transfer_mode": transfer_mode,
    }


def _selected_logical_overrides_requested(overrides: dict[str, str]) -> bool:
    return bool(overrides)


def _generic_adapt_runtime_split_overrides_from_env() -> dict[str, object]:
    raw = {
        field: _generic_adapt_runtime_split_env_value(field)
        for field in _GENERIC_ADAPT_RUNTIME_SPLIT_TSV_FIELDS
    }
    if not any(value not in {None, ""} for value in raw.values()):
        return {}
    mode = str(raw.get("generic_adapt_runtime_split_mode") or _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF).strip().lower()
    if mode in {"", "none", "false", "0", "disabled"}:
        mode = _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_OFF
    if mode not in _GENERIC_ADAPT_RUNTIME_SPLIT_MODE_CHOICES:
        allowed = ", ".join(sorted(_GENERIC_ADAPT_RUNTIME_SPLIT_MODE_CHOICES))
        raise ValueError(f"generic_adapt_runtime_split_mode must be one of {{{allowed}}}; got {mode!r}.")
    policy = str(raw.get("generic_adapt_runtime_split_symmetry_policy") or "off").strip().lower().replace("-", "_")
    if policy in {"", "none", "false", "0", "disabled"}:
        policy = "off"
    if policy not in _GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_CHOICES:
        allowed = ", ".join(sorted(_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY_CHOICES))
        raise ValueError(f"generic_adapt_runtime_split_symmetry_policy must be one of {{{allowed}}}; got {policy!r}.")
    raw_subset_size = raw.get("generic_adapt_runtime_split_max_subset_size")
    subset_size = (
        3
        if raw_subset_size in {None, ""}
        else _parse_phase3_budget_int(
            str(raw_subset_size),
            field="generic_adapt_runtime_split_max_subset_size",
            min_value=1,
        )
    )
    return {
        "generic_adapt_runtime_split_mode": mode,
        "generic_adapt_runtime_split_symmetry_policy": policy,
        "generic_adapt_runtime_split_max_subset_size": int(subset_size),
    }


def _generic_adapt_runtime_split_overrides_requested(overrides: Mapping[str, object]) -> bool:
    return bool(overrides)


def _shared_pauli_pool_overrides_from_env() -> dict[str, object]:
    raw = {field: _shared_pauli_pool_env_value(field) for field in _SHARED_PAULI_POOL_TSV_FIELDS}
    if not any(value not in {None, ""} for value in raw.values()):
        return {}
    mode = str(raw.get("shared_pauli_pool_mode") or _SHARED_PAULI_POOL_MODE_OFF).strip().lower().replace("-", "_")
    if mode in {"", "none", "false", "0", "disabled"}:
        mode = _SHARED_PAULI_POOL_MODE_OFF
    if mode == "pauli_child_sets_v1" or mode == "global_pauli_child_sets_v1":
        mode = _SHARED_PAULI_POOL_MODE_CHILD_SETS_V1
    if mode not in _SHARED_PAULI_POOL_MODE_CHOICES:
        allowed = ", ".join(sorted(_SHARED_PAULI_POOL_MODE_CHOICES))
        raise ValueError(f"shared_pauli_pool_mode must be one of {{{allowed}}}; got {mode!r}.")
    policy = str(raw.get("shared_pauli_pool_symmetry_policy") or "off").strip().lower().replace("-", "_")
    if policy in {"", "none", "false", "0", "disabled"}:
        policy = "off"
    if policy not in _SHARED_PAULI_POOL_SYMMETRY_POLICY_CHOICES:
        allowed = ", ".join(sorted(_SHARED_PAULI_POOL_SYMMETRY_POLICY_CHOICES))
        raise ValueError(f"shared_pauli_pool_symmetry_policy must be one of {{{allowed}}}; got {policy!r}.")
    raw_subset_size = raw.get("shared_pauli_pool_max_subset_size")
    subset_size = (
        3
        if raw_subset_size in {None, ""}
        else _parse_phase3_budget_int(str(raw_subset_size), field="shared_pauli_pool_max_subset_size", min_value=1)
    )
    if mode in {
        _SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
        _SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1,
    }:
        if policy != "hard_guard":
            raise ValueError(
                f"{mode} requires "
                "shared_pauli_pool_symmetry_policy=hard_guard."
            )
        if int(subset_size) != 1:
            raise ValueError(
                f"{mode} requires "
                "shared_pauli_pool_max_subset_size=1."
            )
    return {
        "shared_pauli_pool_mode": mode,
        "shared_pauli_pool_symmetry_policy": policy,
        "shared_pauli_pool_max_subset_size": int(subset_size),
    }


def _shared_pauli_pool_overrides_requested(overrides: Mapping[str, object]) -> bool:
    return bool(overrides) and str(overrides.get("shared_pauli_pool_mode") or "off") != "off"


def _set_cli_option(args: Sequence[str], flag: str, value: object) -> tuple[str, ...]:
    out: list[str] = []
    idx = 0
    while idx < len(args):
        if str(args[idx]) == str(flag):
            idx += 2
            continue
        out.append(str(args[idx]))
        idx += 1
    out.extend([str(flag), str(value)])
    return tuple(out)


def _phase3_spec_with_budget_base_args(spec, budget_overrides: dict[str, int]):
    args = tuple(str(x) for x in getattr(spec, "base_pipeline_args"))
    for policy_field, flag in _PHASE3_BUDGET_BASE_ARG_FLAGS.items():
        if policy_field in budget_overrides:
            args = _set_cli_option(args, flag, budget_overrides[policy_field])
    if args == tuple(getattr(spec, "base_pipeline_args")):
        return spec
    return replace(spec, base_pipeline_args=args)


def _phase3_spec_with_runtime_base_args(spec, runtime_overrides: dict[str, int]):
    args = tuple(str(x) for x in getattr(spec, "base_pipeline_args"))
    for policy_field, flag in _PHASE3_RUNTIME_BASE_ARG_FLAGS.items():
        if policy_field in runtime_overrides:
            args = _set_cli_option(args, flag, runtime_overrides[policy_field])
    if args == tuple(getattr(spec, "base_pipeline_args")):
        return spec
    return replace(spec, base_pipeline_args=args)


def _finite_float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _parse_benchmark_value_noise_float(raw: str, *, field: str) -> float:
    try:
        value = float(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"{field} must be finite numeric when provided; got {raw!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric when provided; got {raw!r}.")
    return float(value)


def _parse_benchmark_value_noise_seed(raw: str, *, field: str) -> int:
    try:
        return int(str(raw).strip(), 10)
    except Exception as exc:
        raise ValueError(f"{field} must be an integer seed when provided; got {raw!r}.") from exc


def _stable_hash_int(*parts: Any, bits: int = 63) -> int:
    blob = json.dumps([str(part) for part in parts], sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    value = int.from_bytes(digest, "big")
    return int(value & ((1 << int(bits)) - 1))


def _benchmark_value_noise_config_from_env(*, family: str, case_id: str, algorithm_id: str) -> dict[str, Any]:
    raw = {field: _benchmark_value_noise_env_value(field) for field in _BENCHMARK_VALUE_NOISE_TSV_FIELDS}
    model = str(raw["benchmark_value_noise_model"] or "off").strip().lower() or "off"
    if model not in _BENCHMARK_VALUE_NOISE_MODEL_CHOICES:
        raise ValueError(
            f"benchmark_value_noise_model must be one of {sorted(_BENCHMARK_VALUE_NOISE_MODEL_CHOICES)}."
        )
    std = 0.0
    if raw["benchmark_value_noise_std"] not in {None, ""}:
        std = _parse_benchmark_value_noise_float(
            str(raw["benchmark_value_noise_std"]),
            field="benchmark_value_noise_std",
        )
    seed: int | None = None
    seed_source = "omitted"
    if raw["benchmark_value_noise_seed"] not in {None, ""}:
        seed = _parse_benchmark_value_noise_seed(
            str(raw["benchmark_value_noise_seed"]),
            field="benchmark_value_noise_seed",
        )
        seed_source = "env"
    if model == "off":
        if seed is not None:
            raise ValueError(
                "benchmark_value_noise_seed requires benchmark_value_noise_model='gaussian_iid_v1'."
            )
        if std != 0.0:
            raise ValueError("benchmark_value_noise_model='off' requires benchmark_value_noise_std == 0.")
        return {
            "enabled": False,
            "model": "off",
            "std": 0.0,
            "seed": None,
            "seed_source": seed_source,
            "semantic": _BENCHMARK_VALUE_NOISE_SEMANTIC,
        }
    if model == "gaussian_iid_v1":
        if (not math.isfinite(std)) or std <= 0.0:
            raise ValueError(
                "benchmark_value_noise_model='gaussian_iid_v1' requires finite benchmark_value_noise_std > 0."
            )
        if seed is None:
            seed = _stable_hash_int(
                _BENCHMARK_VALUE_NOISE_SEMANTIC,
                family,
                case_id,
                algorithm_id,
                model,
                repr(float(std)),
            )
            seed_source = "derived_stable_hash_v1"
        return {
            "enabled": True,
            "model": model,
            "std": float(std),
            "seed": int(seed),
            "seed_source": seed_source,
            "semantic": _BENCHMARK_VALUE_NOISE_SEMANTIC,
        }
    raise ValueError(f"Unsupported benchmark_value_noise_model {model!r}.")


def _benchmark_value_noise_scope(
    row: Mapping[str, Any],
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
) -> dict[str, Any]:
    row_family = str(row.get("family") or row.get("problem") or family)
    row_case_id = str(row.get("case_id") or row.get("hamiltonian_id") or case_id)
    row_algorithm_id = str(row.get("algorithm_id") or row.get("method_id") or algorithm_id)
    scope = {
        "kind": "static_benchmark_row",
        "family": row_family,
        "case_id": row_case_id,
        "algorithm_id": row_algorithm_id,
    }
    if row.get("run_id") not in {None, ""}:
        scope["run_id"] = str(row.get("run_id"))
    return scope


def _benchmark_value_noise_standard_normal(*, seed: int, scope: Mapping[str, Any]) -> float:
    material = json.dumps(
        {
            "semantic": _BENCHMARK_VALUE_NOISE_SEMANTIC,
            "model": "gaussian_iid_v1",
            "seed": int(seed),
            "scope": dict(scope),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    def _uniform(label: bytes) -> float:
        digest = hashlib.sha256(material + b"|" + label).digest()
        mantissa = int.from_bytes(digest[:8], "big") >> 11
        return (float(mantissa) + 0.5) / float(1 << 53)

    u1 = _uniform(b"u1")
    u2 = _uniform(b"u2")
    return float(math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))


def _benchmark_value_noise_reference_energy(row: Mapping[str, Any]) -> tuple[str | None, float | None]:
    for key in _BENCHMARK_VALUE_NOISE_EXACT_ENERGY_KEYS:
        value = _finite_float_or_none(row.get(key))
        if value is not None:
            return key, float(value)
    return None, None


def _benchmark_value_noise_json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


def _benchmark_value_noise_matches_config(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    actual_std = _finite_float_or_none(metadata.get("std"))
    expected_std = _finite_float_or_none(config.get("std"))
    return (
        bool(metadata.get("enabled", False)) is True
        and str(metadata.get("model") or "").strip().lower() == str(config.get("model") or "").strip().lower()
        and actual_std is not None
        and expected_std is not None
        and math.isclose(actual_std, expected_std, rel_tol=1e-12, abs_tol=1e-18)
        and metadata.get("seed") == config.get("seed")
        and str(metadata.get("semantic") or "") == _BENCHMARK_VALUE_NOISE_SEMANTIC
    )


def _apply_benchmark_value_noise_to_row(
    row: dict[str, Any],
    config: Mapping[str, Any],
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
) -> str:
    if not bool(config.get("enabled", False)):
        return "not_requested"
    scope = _benchmark_value_noise_scope(row, family=family, case_id=case_id, algorithm_id=algorithm_id)
    existing = row.get("benchmark_value_noise")
    existing_scope = existing.get("scope") if isinstance(existing, Mapping) else None
    if (
        str(row.get("benchmark_value_noise_status") or "") == "ok"
        and isinstance(existing, Mapping)
        and isinstance(existing_scope, Mapping)
        and _benchmark_value_noise_matches_config(existing, config)
        and dict(existing_scope) == dict(scope)
    ):
        return "ok"

    seed = int(config["seed"])
    std = float(config["std"])
    base_metadata = {
        "enabled": True,
        "model": str(config["model"]),
        "std": std,
        "seed": seed,
        "seed_source": str(config.get("seed_source") or ""),
        "semantic": _BENCHMARK_VALUE_NOISE_SEMANTIC,
        "physical_shots_unchanged": True,
        "scope": scope,
    }

    if isinstance(existing, Mapping):
        energy_pre_benchmark_value_noise = None
        for candidate in (
            row.get("energy_pre_benchmark_value_noise"),
            row.get("benchmark_value_noise_energy_ideal"),
            existing.get("energy_pre_benchmark_value_noise"),
            existing.get("benchmark_value_noise_energy_ideal"),
        ):
            parsed = _finite_float_or_none(candidate)
            if parsed is not None:
                energy_pre_benchmark_value_noise = parsed
                break
        missing_energy_status = "missing_pre_value_noise_energy"
        missing_energy_reason = (
            "existing benchmark value-noise metadata is present with a different config/scope, "
            "but no finite pre-noise energy baseline is available for safe reapplication"
        )
    else:
        energy_pre_benchmark_value_noise = _finite_float_or_none(row.get("energy"))
        missing_energy_status = "missing_energy"
        missing_energy_reason = "finite energy is required for post-static-result value-noise overlay"
    if energy_pre_benchmark_value_noise is None:
        row["benchmark_value_noise_status"] = missing_energy_status
        row["benchmark_value_noise"] = {
            **base_metadata,
            "applied": False,
            "status": missing_energy_status,
            "reason": missing_energy_reason,
        }
        return missing_energy_status

    row["energy_pre_benchmark_value_noise"] = float(energy_pre_benchmark_value_noise)
    row["benchmark_value_noise_energy_ideal"] = float(energy_pre_benchmark_value_noise)
    if row.get("energy_ideal") in {None, ""}:
        row["energy_ideal"] = float(energy_pre_benchmark_value_noise)
    ideal_delta_by_key: dict[str, float] = {}
    for key in _BENCHMARK_VALUE_NOISE_DELTA_FIELDS:
        ideal_key = f"{key}_ideal"
        ideal_delta = _finite_float_or_none(row.get(ideal_key))
        if ideal_delta is None:
            ideal_delta = _finite_float_or_none(row.get(key))
            if ideal_delta is not None and row.get(ideal_key) in {None, ""}:
                row[ideal_key] = float(ideal_delta)
        if ideal_delta is not None:
            ideal_delta_by_key[key] = float(ideal_delta)

    noise_draw = float(std * _benchmark_value_noise_standard_normal(seed=seed, scope=scope))
    energy_noisy = float(energy_pre_benchmark_value_noise + noise_draw)
    row["energy"] = energy_noisy
    applied_fields = ["energy"]
    reference_key, reference_energy = _benchmark_value_noise_reference_energy(row)
    delta_recompute_status = "missing_reference_energy"
    delta_noisy: float | None = None
    if reference_energy is not None:
        delta_noisy = float(abs(energy_noisy - float(reference_energy)))
        for key in _BENCHMARK_VALUE_NOISE_DELTA_FIELDS:
            row[key] = delta_noisy
            applied_fields.append(key)
        delta_recompute_status = "ok"
        reference_delta = _finite_float_or_none(row.get("reference_abs_delta_e"))
        if reference_delta is not None and "improvement_over_reference_abs_delta_e" in row:
            ideal_improvement_key = "improvement_over_reference_abs_delta_e_ideal"
            if row.get(ideal_improvement_key) in {None, ""}:
                ideal_improvement = _finite_float_or_none(row.get("improvement_over_reference_abs_delta_e"))
                if ideal_improvement is not None:
                    row[ideal_improvement_key] = float(ideal_improvement)
            row["improvement_over_reference_abs_delta_e"] = float(reference_delta - delta_noisy)
            applied_fields.append("improvement_over_reference_abs_delta_e")
        if reference_delta is not None and "beats_reference_state" in row:
            row["beats_reference_state"] = bool(float(reference_delta - delta_noisy) > 0.0)
            applied_fields.append("beats_reference_state")
    else:
        for key in _BENCHMARK_VALUE_NOISE_DELTA_FIELDS:
            if key in row:
                row[key] = None
                applied_fields.append(key)

    metadata = {
        **base_metadata,
        "applied": True,
        "status": "ok",
        "noise_draw": noise_draw,
        "noise_draw_units": "energy",
        "energy_pre_benchmark_value_noise": float(energy_pre_benchmark_value_noise),
        "benchmark_value_noise_energy_ideal": float(energy_pre_benchmark_value_noise),
        "energy_noisy": energy_noisy,
        "delta_ideal_by_field": ideal_delta_by_key,
        "delta_noisy": delta_noisy,
        "delta_reference_key": reference_key,
        "delta_recompute_status": delta_recompute_status,
        "applied_fields": sorted(dict.fromkeys(applied_fields)),
    }
    row["benchmark_value_noise_status"] = "ok"
    row["benchmark_value_noise"] = metadata
    return "ok"


def _benchmark_value_noise_row_targets(payload: Any) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, dict):
            targets.append(result)
        rows = payload.get("rows")
        if isinstance(rows, list):
            targets.extend(row for row in rows if isinstance(row, dict))
        if not targets and any(key in payload for key in ("energy", "energy_ideal", "delta_E_abs", "abs_delta_e")):
            targets.append(payload)
    elif isinstance(payload, list):
        targets.extend(row for row in payload if isinstance(row, dict))
    return targets


def _primary_reference_config_from_env() -> dict[str, Any]:
    return _env_reference_kwargs()


def _apply_primary_reference_metrics_to_row(row: dict[str, Any], config: Mapping[str, Any]) -> str:
    if not config:
        return "not_requested"
    energy = _finite_float_or_none(row.get("energy"))
    if energy is None:
        return "missing_energy"
    same = _finite_float_or_none(config.get("same_cutoff_exact_gs_energy"))
    if same is None:
        same = _finite_float_or_none(row.get("same_cutoff_exact_gs_energy"))
    if same is None:
        same = _finite_float_or_none(row.get("exact_energy"))
    ref = _finite_float_or_none(config.get("exact_reference_energy"))
    if ref is None:
        ref = _finite_float_or_none(row.get("exact_reference_energy"))
    metric = str(config.get("primary_energy_metric") or row.get("primary_energy_metric") or "same_cutoff_abs_delta_e").strip()
    role = str(config.get("same_cutoff_error_role") or row.get("same_cutoff_error_role") or "primary").strip()
    raw_ref_nph = config.get("exact_reference_n_ph_max") or row.get("exact_reference_n_ph_max")
    ref_nph = None
    if raw_ref_nph not in {None, ""}:
        try:
            ref_nph = int(float(str(raw_ref_nph)))
        except Exception:
            ref_nph = None
    primary_ref = ref if metric == "higher_cutoff_reference_abs_delta_e" and ref is not None else same
    if primary_ref is None:
        primary_ref = ref
    if same is not None:
        row["same_cutoff_exact_gs_energy"] = float(same)
        row["abs_delta_e_same_cutoff"] = float(abs(float(energy) - float(same)))
    if ref is not None:
        row["exact_reference_energy"] = float(ref)
        row["abs_delta_e_reference"] = float(abs(float(energy) - float(ref)))
    if ref_nph is not None:
        row["exact_reference_n_ph_max"] = int(ref_nph)
    row["primary_energy_metric"] = metric
    row["same_cutoff_error_role"] = role
    row["primary_reference_source"] = "exact_reference_energy" if primary_ref is not None and ref is not None and metric == "higher_cutoff_reference_abs_delta_e" else "same_cutoff_exact_gs_energy"
    if primary_ref is not None:
        primary_delta = float(abs(float(energy) - float(primary_ref)))
        row["delta_E_abs"] = primary_delta
        row["abs_delta_e"] = primary_delta
    hits = row.get("benchmark_first_hits")
    if isinstance(hits, dict):
        filtered: dict[str, Any] = {}
        for key, value in hits.items():
            if not isinstance(value, dict):
                filtered[str(key)] = value
                continue
            hit_energy = _finite_float_or_none(value.get("energy"))
            try:
                threshold = float(str(key).replace("e_", "e-"))
            except Exception:
                threshold = _finite_float_or_none(value.get("threshold_abs_delta_e")) or math.inf
            if hit_energy is None:
                filtered[str(key)] = value
                continue
            if same is not None:
                value["same_cutoff_exact_gs_energy"] = float(same)
                value["abs_delta_e_same_cutoff"] = float(abs(hit_energy - same))
            if ref is not None:
                value["exact_reference_energy"] = float(ref)
                value["abs_delta_e_reference"] = float(abs(hit_energy - ref))
            if ref_nph is not None:
                value["exact_reference_n_ph_max"] = int(ref_nph)
            value["primary_energy_metric"] = metric
            value["same_cutoff_error_role"] = role
            if primary_ref is not None:
                hit_delta = float(abs(hit_energy - float(primary_ref)))
                value["exact_energy"] = float(primary_ref)
                value["delta_E_abs"] = hit_delta
                value["abs_delta_e"] = hit_delta
                if hit_delta > threshold:
                    continue
            filtered[str(key)] = value
        row["benchmark_first_hits"] = filtered
        row["first_hit_1e6"] = filtered.get("1e-06") or filtered.get("1e-6")
        row["first_hit_1e8"] = filtered.get("1e-08") or filtered.get("1e-8")
        row["first_hit_abs_delta_e_le_1e_6"] = row.get("first_hit_1e6")
        row["first_hit_abs_delta_e_le_1e_8"] = row.get("first_hit_1e8")
    return "ok" if primary_ref is not None else "missing_primary_reference"


def _apply_primary_reference_metrics_to_payload(payload: Any, config: Mapping[str, Any]) -> Any:
    if not config:
        return payload
    statuses: list[str] = []
    seen_ids: set[int] = set()
    for row in _benchmark_value_noise_row_targets(payload):
        if id(row) in seen_ids:
            continue
        seen_ids.add(id(row))
        statuses.append(_apply_primary_reference_metrics_to_row(row, config))
    if isinstance(payload, dict):
        payload["primary_reference_metric_status"] = "ok" if statuses and all(s == "ok" for s in statuses) else (statuses[0] if len(set(statuses)) == 1 and statuses else "not_requested")
        payload["primary_reference_metric_status_counts"] = {status: statuses.count(status) for status in sorted(set(statuses))}
    return payload


def _rewrite_primary_reference_metric_artifacts(*, output_dir: Path, config: Mapping[str, Any]) -> None:
    if not config:
        return
    for name in _BENCHMARK_VALUE_NOISE_JSON_ARTIFACTS:
        path = Path(output_dir) / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        _apply_primary_reference_metrics_to_payload(payload, config)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_benchmark_value_noise_json_default) + "\n", encoding="utf-8")


def _apply_benchmark_value_noise_to_payload(
    payload: Any,
    config: Mapping[str, Any],
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
) -> Any:
    if not bool(config.get("enabled", False)):
        return payload
    statuses: list[str] = []
    seen_ids: set[int] = set()
    for row in _benchmark_value_noise_row_targets(payload):
        if id(row) in seen_ids:
            continue
        seen_ids.add(id(row))
        statuses.append(
            _apply_benchmark_value_noise_to_row(
                row,
                config,
                family=family,
                case_id=case_id,
                algorithm_id=algorithm_id,
            )
        )
    status_counts = {status: statuses.count(status) for status in sorted(set(statuses))}
    payload_status = "ok" if statuses and all(status == "ok" for status in statuses) else (statuses[0] if len(set(statuses)) == 1 and statuses else "partial")
    top_level_metadata = {
        "enabled": True,
        "model": str(config["model"]),
        "std": float(config["std"]),
        "seed": int(config["seed"]),
        "seed_source": str(config.get("seed_source") or ""),
        "semantic": _BENCHMARK_VALUE_NOISE_SEMANTIC,
        "physical_shots_unchanged": True,
        "scope": "generic_static_benchmark_dispatch_payload",
        "status": payload_status,
        "status_counts": status_counts,
        "row_target_count": len(statuses),
        "applied_row_count": int(status_counts.get("ok", 0)),
    }
    if isinstance(payload, dict):
        payload["benchmark_value_noise_status"] = payload_status
        payload["benchmark_value_noise"] = top_level_metadata
    return payload


def _rewrite_benchmark_value_noise_artifacts(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    family: str,
    case_id: str,
    algorithm_id: str,
) -> None:
    if not bool(config.get("enabled", False)):
        return
    for name in _BENCHMARK_VALUE_NOISE_JSON_ARTIFACTS:
        path = Path(output_dir) / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        _apply_benchmark_value_noise_to_payload(
            payload,
            config,
            family=family,
            case_id=case_id,
            algorithm_id=algorithm_id,
        )
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_benchmark_value_noise_json_default) + "\n",
            encoding="utf-8",
        )


def _rewrite_benchmark_value_noise_proxy_sidecars(*, payload: Mapping[str, Any], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str, str, str, str]] = set()
    for row in _benchmark_value_noise_row_targets(payload):
        if not isinstance(row, Mapping):
            continue
        row_key = (
            str(row.get("run_id") or ""),
            str(row.get("family") or row.get("problem") or ""),
            str(row.get("case_id") or row.get("hamiltonian_id") or ""),
            str(row.get("algorithm_id") or ""),
            str(row.get("method_id") or ""),
        )
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        rows.append(dict(row))
    if not rows:
        return
    sidecar_dirs: list[Path] = []
    output_path = Path(output_dir)
    if any((output_path / name).exists() for name in ("metrics_proxy_runs.csv", "metrics_proxy_runs.jsonl", "metrics_proxy_summary.json")):
        sidecar_dirs.append(output_path)
    sidecars = payload.get("sidecars") if isinstance(payload, Mapping) else None
    if isinstance(sidecars, Mapping):
        for raw_path in sidecars.values():
            if raw_path in {None, ""}:
                continue
            sidecar_dirs.append(Path(str(raw_path)).parent)
    seen: set[str] = set()
    for sidecar_dir in sidecar_dirs:
        key = str(sidecar_dir)
        if key in seen:
            continue
        seen.add(key)
        write_proxy_sidecars(
            rows,
            sidecar_dir,
            summary_extras={
                "schema_source": "generic_static_benchmark_value_noise_overlay_v1",
                "benchmark_value_noise_semantic": _BENCHMARK_VALUE_NOISE_SEMANTIC,
                "benchmark_value_noise_physical_shots_unchanged": True,
            },
        )


def _benchmark_decision_noise_unsupported_reason(*, dispatch: str | None, algorithm_id: str) -> str:
    dispatch_key = str(dispatch or "no_dispatch")
    algorithm_key = str(algorithm_id)
    if dispatch_key == "generic_static_ed_reference":
        return "static_ed_reference has no optimizer/selector decision surface for benchmark_decision_noise."
    if dispatch_key == "generic_static_qiskit_adapt_vqe":
        return "Qiskit AdaptVQE decisions are inside external public APIs; true benchmark_decision_noise support is not wired."
    if dispatch_key in _EXTERNAL_STATIC_ADAPT_DISPATCHES:
        return "external/public-code ADAPT decisions are outside local exact-bench control for benchmark_decision_noise."
    if dispatch_key == "hh_static_ground_state":
        return (
            "HH static benchmark_decision_noise is wired only for local VQE-style "
            "static_hva_vqe/static_uccsd_vqe/static_lang_firsov_vqe rows in this slice."
        )
    return (
        "benchmark_decision_noise true runner support is not wired for "
        f"dispatch={dispatch_key!r}, algorithm_id={algorithm_key!r}; refusing to run exact decisions silently."
    )


def _benchmark_decision_noise_supported_for_dispatch(
    *,
    dispatch: str | None,
    algorithm_id: str,
) -> bool:
    dispatch_key = str(dispatch or "")
    algorithm_key = str(algorithm_id)
    if dispatch_key in _BENCHMARK_DECISION_NOISE_SUPPORTED_DISPATCHES:
        return True
    return (
        dispatch_key == "hh_static_ground_state"
        and algorithm_key in _BENCHMARK_DECISION_NOISE_SUPPORTED_HH_ALGORITHM_IDS
    )


def _write_benchmark_decision_noise_unsupported_artifacts(
    *,
    output_dir: Path,
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in _BENCHMARK_VALUE_NOISE_JSON_ARTIFACTS:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    json_payload = json.dumps(payload, indent=2, sort_keys=True, default=_benchmark_value_noise_json_default) + "\n"
    for name in ("generic_static_single.json", "result.json"):
        (output_dir / name).write_text(json_payload, encoding="utf-8")
    rows_payload = {
        "schema": "generic_static_benchmark_decision_noise_unsupported_rows_v1",
        "benchmark_decision_noise_status": payload.get("benchmark_decision_noise_status"),
        "benchmark_decision_noise": payload.get("benchmark_decision_noise"),
        "rows": [dict(row)],
    }
    (output_dir / "rows.json").write_text(
        json.dumps(rows_payload, indent=2, sort_keys=True, default=_benchmark_value_noise_json_default) + "\n",
        encoding="utf-8",
    )
    manifest_payload = {
        "schema": "generic_static_benchmark_decision_noise_unsupported_manifest_v1",
        **{key: value for key, value in dict(payload).items() if key != "schema"},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True, default=_benchmark_value_noise_json_default) + "\n",
        encoding="utf-8",
    )
    write_proxy_sidecars(
        [dict(row)],
        output_dir,
        summary_extras={
            "schema_source": "generic_static_benchmark_decision_noise_unsupported_v1",
            "benchmark_decision_noise_semantic": BENCHMARK_DECISION_NOISE_SEMANTIC,
            "benchmark_decision_noise_status": "unsupported",
            "benchmark_decision_noise_physical_shots_unchanged": True,
        },
    )


def _unsupported_benchmark_decision_noise_payload(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    dispatch: str | None,
    output_dir: Path,
    config: BenchmarkDecisionNoiseConfig,
) -> dict[str, Any]:
    reason = _benchmark_decision_noise_unsupported_reason(dispatch=dispatch, algorithm_id=algorithm_id)
    metadata = benchmark_decision_noise_unsupported_metadata(
        config,
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
        dispatch=dispatch,
        reason=reason,
    )
    row_metadata = copy_decision_noise_metadata(metadata)
    row = {
        "schema": "generic_static_benchmark_decision_noise_unsupported_row_v1",
        "family": str(family),
        "case_id": str(case_id),
        "algorithm_id": str(algorithm_id),
        "method_id": str(algorithm_id),
        "run_id": f"{family}__{case_id}__{algorithm_id}",
        "status": "skipped_unsupported_decision_noise",
        "quality_gate_reason": "benchmark_decision_noise_unsupported",
        "failure_reason": "benchmark_decision_noise_unsupported",
        "reason": reason,
        "dispatch": None if dispatch is None else str(dispatch),
        "energy": None,
        "energy_ideal": None,
        "exact_energy": None,
        "delta_E_abs": None,
        "abs_delta_e": None,
        "phase3_controller_called": False,
        "uses_exact_for_decision": False,
        "phase3_emulation": False,
        "shots_total": 0,
        "compiled_depth_total": 0,
        "compiled_count_2q_total": 0,
        "static_shot_estimate_status": "not_applicable_decision_noise_unsupported",
        "compiled_circuit_stats_status": "not_applicable_decision_noise_unsupported",
        "benchmark_decision_noise_status": "unsupported",
        "benchmark_decision_noise": row_metadata,
    }
    payload_metadata = copy_decision_noise_metadata(
        {
            **metadata,
            "status_counts": {"unsupported": 1},
            "row_target_count": 1,
            "handled_row_count": 1,
            "unsupported_row_count": 1,
            "applied_row_count": 0,
        }
    )
    payload = {
        "schema": "generic_static_benchmark_decision_noise_unsupported_v1",
        "family": str(family),
        "case_id": str(case_id),
        "algorithm_id": str(algorithm_id),
        "status": "skipped_unsupported_decision_noise",
        "reason": reason,
        "dispatch": None if dispatch is None else str(dispatch),
        "benchmark_decision_noise_status": "unsupported",
        "benchmark_decision_noise": payload_metadata,
        "result": row,
        "rows": [dict(row)],
    }
    _write_benchmark_decision_noise_unsupported_artifacts(output_dir=output_dir, payload=payload, row=row)
    return payload


def _finalize_benchmark_payload(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    value_noise_config: Mapping[str, Any],
    reference_metric_config: Mapping[str, Any],
    family: str,
    case_id: str,
    algorithm_id: str,
) -> dict[str, Any]:
    if reference_metric_config:
        _apply_primary_reference_metrics_to_payload(payload, reference_metric_config)
        _rewrite_primary_reference_metric_artifacts(output_dir=output_dir, config=reference_metric_config)
    if bool(value_noise_config.get("enabled", False)):
        _apply_benchmark_value_noise_to_payload(
            payload,
            value_noise_config,
            family=family,
            case_id=case_id,
            algorithm_id=algorithm_id,
        )
        _rewrite_benchmark_value_noise_artifacts(
            output_dir=output_dir,
            config=value_noise_config,
            family=family,
            case_id=case_id,
            algorithm_id=algorithm_id,
        )
        _rewrite_benchmark_value_noise_proxy_sidecars(payload=payload, output_dir=output_dir)
    return payload


def _phase3_static_table_contract_fields_from_result(result_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Expose honest generic-table contract fields for native Phase3 rows."""

    out: dict[str, Any] = {"phase3_controller_called": True}
    depth = _finite_float_or_none(result_payload.get("circuit_depth"))
    count_2q = _finite_float_or_none(result_payload.get("count_2q"))
    if depth is not None:
        out["compiled_depth_total"] = int(depth) if float(depth).is_integer() else float(depth)
    if count_2q is not None:
        out["compiled_count_2q_total"] = int(count_2q) if float(count_2q).is_integer() else float(count_2q)
    if depth is not None or count_2q is not None:
        out["compiled_circuit_stats_status"] = "phase3_compile_json_metrics_v1"
    legacy_proxy: dict[str, Any] = {
        "schema": "paper_i_legacy_work_proxies_v1",
        "display_policy": "diagnostic_only_not_cross_method_S_alg_or_shots_total",
    }
    for key in (
        "measurement_groups_proxy",
        "measurement_shots_proxy",
        "shot_cost_proxy",
        "measurement_proxy_validated",
        "measurement_proxy_validation",
    ):
        value = result_payload.get(key)
        if value is None or value == "":
            continue
        legacy_proxy[key] = value
    if len(legacy_proxy) > 2:
        out["legacy_work_proxies"] = legacy_proxy
        out["legacy_work_proxy_status"] = "controller_proxy_diagnostic_only"
    return {key: value for key, value in out.items() if result_payload.get(key) in {None, ""}}


def _phase3_static_algorithmic_work_fields_from_result(
    result_payload: Mapping[str, Any],
    *,
    shots_per_pauli_term_proxy: int | None = None,
) -> dict[str, Any]:
    """Build strict S_alg fields for native Phase3/SNAKE rows.

    Use the same SNAKE normalization helper as reporting and sidecar code.  Raw
    controller shot proxies remain diagnostic fields and are not promoted into
    ``S_alg``.
    """

    result_json = result_payload.get("result_json")
    if result_json in {None, ""}:
        return {
            "S_alg_status": "source_json_missing",
            "algorithmic_measurement_work_source": "snake_canonical_runtime_reconstruction_blocked",
            "snake_deterministic_shot_proxy": {
                "schema": "snake_deterministic_shot_proxy_v1",
                "status": "source_json_missing",
                "display_policy": "blocked_no_comparable_shots_total",
            },
        }
    try:
        payload = json.loads(Path(str(result_json)).read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "S_alg_status": "source_json_unreadable",
            "algorithmic_measurement_work_source": "snake_canonical_runtime_reconstruction_blocked",
            "algorithmic_measurement_work_error": str(exc),
            "snake_deterministic_shot_proxy": {
                "schema": "snake_deterministic_shot_proxy_v1",
                "status": "source_json_unreadable",
                "display_policy": "blocked_no_comparable_shots_total",
                "source_label": str(result_json),
                "error": str(exc),
            },
        }
    if not isinstance(payload, Mapping):
        return {
            "S_alg_status": "source_json_not_mapping",
            "algorithmic_measurement_work_source": "snake_canonical_runtime_reconstruction_blocked",
            "snake_deterministic_shot_proxy": {
                "schema": "snake_deterministic_shot_proxy_v1",
                "status": "source_json_not_mapping",
                "display_policy": "blocked_no_comparable_shots_total",
                "source_label": str(result_json),
            },
        }
    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="terminal",
        source_label=str(result_json),
    )
    status = str(work.get("S_alg_status") or audit.get("status") or "missing_event_ledger_component_breakdown")
    deterministic_fields, deterministic_audit = snake_deterministic_shot_proxy_from_payload(
        payload,
        scope="terminal",
        source_label=str(result_json),
        shots_per_pauli_term_proxy=shots_per_pauli_term_proxy,
    )
    if status != "ok" or work.get("S_alg") is None:
        return {
            "S_alg_status": status,
            "algorithmic_measurement_work": work.get("algorithmic_measurement_work"),
            "algorithmic_measurement_work_source": "snake_canonical_runtime_reconstruction_blocked",
            "snake_deterministic_shot_proxy": deterministic_audit,
        }
    out = {
        "S_alg": float(work["S_alg"]),
        "S_alg_status": "ok",
        "algorithmic_measurement_work_source": "snake_canonical_runtime_reconstruction_v1",
        "algorithmic_measurement_work": work.get("algorithmic_measurement_work"),
        "table_i_measurement_event_ledger": work.get("table_i_measurement_event_ledger"),
    }
    for key in (
        "S_alg_N_H_outer_eval",
        "S_alg_N_grad_probe",
        "S_alg_N_metric_probe",
        "S_alg_N_H_refit_eval",
        "S_alg_N_other_quantum",
    ):
        if key in work:
            out[key] = work[key]
    out["snake_deterministic_shot_proxy"] = deterministic_audit
    if deterministic_audit.get("status") == "ok":
        out.update(deterministic_fields)
    return out


def _phase3_oracle_policy_overrides_from_env() -> dict[str, object]:
    raw = {field: _phase3_oracle_env_value(field) for field in _PHASE3_ORACLE_TSV_FIELDS}
    overrides: dict[str, object] = {}
    for field in (
        "phase3_oracle_gradient_mode",
        "phase3_oracle_backend_name",
        "phase3_oracle_aggregate",
        "phase3_oracle_execution_surface",
        "phase3_oracle_inner_objective_mode",
        "phase3_oracle_value_noise_model",
    ):
        if raw[field] not in {None, ""}:
            value = str(raw[field]).strip()
            overrides[field] = value if field == "phase3_oracle_backend_name" else value.lower()
    if raw["phase3_oracle_use_fake_backend"] not in {None, ""}:
        overrides["phase3_oracle_use_fake_backend"] = _parse_phase3_oracle_bool(
            str(raw["phase3_oracle_use_fake_backend"]),
            field="phase3_oracle_use_fake_backend",
        )
    for field in (
        "phase3_oracle_shots",
        "phase3_oracle_repeats",
        "phase3_oracle_seed",
        "phase3_oracle_value_noise_seed",
    ):
        if raw[field] not in {None, ""}:
            overrides[field] = _parse_phase3_oracle_int(str(raw[field]), field=field)
    if raw["phase3_oracle_value_noise_std"] not in {None, ""}:
        overrides["phase3_oracle_value_noise_std"] = _parse_phase3_oracle_float(
            str(raw["phase3_oracle_value_noise_std"]),
            field="phase3_oracle_value_noise_std",
        )
    _validate_phase3_oracle_policy_overrides(overrides)
    return overrides


def _phase3_oracle_policy_overrides_requested(overrides: dict[str, object]) -> bool:
    if not overrides:
        return False
    if str(overrides.get("phase3_oracle_gradient_mode", "off")).strip().lower() != "off":
        return True
    if str(overrides.get("phase3_oracle_inner_objective_mode", "exact")).strip().lower() != "exact":
        return True
    if str(overrides.get("phase3_oracle_value_noise_model", "off")).strip().lower() != "off":
        return True
    if float(overrides.get("phase3_oracle_value_noise_std", 0.0) or 0.0) != 0.0:
        return True
    for field in (
        "phase3_oracle_backend_name",
        "phase3_oracle_use_fake_backend",
        "phase3_oracle_shots",
        "phase3_oracle_repeats",
        "phase3_oracle_aggregate",
        "phase3_oracle_seed",
        "phase3_oracle_execution_surface",
        "phase3_oracle_value_noise_seed",
    ):
        if field in overrides:
            return True
    return False


def _validate_phase3_oracle_policy_overrides(overrides: dict[str, object]) -> None:
    mode = str(overrides.get("phase3_oracle_gradient_mode", "off")).strip().lower()
    if mode not in _PHASE3_ORACLE_GRADIENT_MODE_CHOICES:
        raise ValueError(f"phase3_oracle_gradient_mode must be one of {sorted(_PHASE3_ORACLE_GRADIENT_MODE_CHOICES)}.")
    aggregate = str(overrides.get("phase3_oracle_aggregate", "mean")).strip().lower()
    if aggregate != "mean":
        raise ValueError("phase3_oracle_aggregate currently supports only 'mean'.")
    execution_surface = str(overrides.get("phase3_oracle_execution_surface", "auto")).strip().lower()
    if execution_surface not in _PHASE3_ORACLE_EXECUTION_SURFACE_CHOICES:
        raise ValueError(
            f"phase3_oracle_execution_surface must be one of {sorted(_PHASE3_ORACLE_EXECUTION_SURFACE_CHOICES)}."
        )
    inner_mode = str(overrides.get("phase3_oracle_inner_objective_mode", "exact")).strip().lower()
    if inner_mode not in {"exact", "noisy_v1"}:
        raise ValueError("phase3_oracle_inner_objective_mode must be one of {'exact','noisy_v1'}.")
    if inner_mode == "noisy_v1" and mode == "off":
        raise ValueError("phase3_oracle_inner_objective_mode='noisy_v1' requires phase3_oracle_gradient_mode != 'off'.")
    value_noise_model = str(overrides.get("phase3_oracle_value_noise_model", "off")).strip().lower()
    if value_noise_model not in _PHASE3_ORACLE_VALUE_NOISE_MODEL_CHOICES:
        raise ValueError(
            f"phase3_oracle_value_noise_model must be one of {sorted(_PHASE3_ORACLE_VALUE_NOISE_MODEL_CHOICES)}."
        )
    value_noise_std = float(overrides.get("phase3_oracle_value_noise_std", 0.0) or 0.0)
    if value_noise_model == "off":
        if "phase3_oracle_value_noise_seed" in overrides:
            raise ValueError(
                "phase3_oracle_value_noise_seed requires phase3_oracle_value_noise_model='gaussian_iid_v1'."
            )
        if value_noise_std != 0.0:
            raise ValueError("phase3_oracle_value_noise_model='off' requires phase3_oracle_value_noise_std == 0.")
        return
    if value_noise_model == "gaussian_iid_v1":
        if (not math.isfinite(value_noise_std)) or value_noise_std <= 0.0:
            raise ValueError(
                "phase3_oracle_value_noise_model='gaussian_iid_v1' requires finite phase3_oracle_value_noise_std > 0."
            )
        if mode == "off":
            raise ValueError("phase3_oracle_value_noise_model='gaussian_iid_v1' requires phase3_oracle_gradient_mode != 'off'.")
        if execution_surface == "raw_measurement_v1":
            raise ValueError("phase3 oracle value noise is post-expectation metadata and cannot use raw_measurement_v1.")


def _families_from_args(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(values) if values else available_problem_keys()


def _static_algorithms_from_args(values: Sequence[str] | None):
    algs = default_benchmark_algorithms(domain="static")
    if not values:
        return algs
    wanted = set(str(v) for v in values)
    known = {alg.algorithm_id for alg in algs}
    unknown = sorted(wanted - known)
    if unknown:
        known_msg = ", ".join(sorted(known))
        raise ValueError(f"Unknown static benchmark algorithm id(s): {unknown}. Known static algorithms: {known_msg}")
    return tuple(alg for alg in algs if alg.algorithm_id in wanted)


@lru_cache(maxsize=None)
def _phase3_specs_for_family_profile(family_key: str, profile_key: str) -> tuple[object, ...]:
    return tuple(spec for spec in table_i_executable_specs(profile_key) if spec.family == family_key)


def _phase3_specs_for_family(family: str, profile: str | None = None) -> tuple[object, ...]:
    family_key = str(family).strip()
    if family_key not in available_problem_keys():
        return ()
    profile_key = table_i_suite_profile(profile)
    return _phase3_specs_for_family_profile(family_key, profile_key)


def _phase3_static_spec_for_case(*, family: str, case_id: str, algorithm_id: str):
    if algorithm_id not in _PHASE3_ADAPT_ALGORITHM_IDS:
        return None
    for spec in _phase3_specs_for_family(family, table_i_suite_profile()):
        if str(getattr(spec, "benchmark_id")) == str(case_id):
            return with_molecular_vibronic_h2_fixture_override(spec, family=family)
    return None


def _case_ids_for_family(family: str, algorithm_id: str) -> tuple[str, ...]:
    if family == "hh" and algorithm_id in _HH_ALGORITHM_MAP:
        from pipelines.exact_bench.hh_static_ground_state_benchmark import canonical_hh_benchmark_cases

        return tuple(case.case_id for case in canonical_hh_benchmark_cases())
    if algorithm_id in _GENERIC_QISKIT_HEA_ALGORITHM_IDS:
        from pipelines.exact_bench.generic_static_hea_qiskit_vqe import default_static_hea_case_ids

        ids = tuple(default_static_hea_case_ids(family))
        if ids:
            return ids
    if algorithm_id in _GENERIC_FAMILY_INFORMED_VQE_ALGORITHM_IDS:
        from pipelines.exact_bench.generic_static_family_informed_vqe import (
            default_static_family_informed_vqe_case_ids,
        )

        ids = tuple(default_static_family_informed_vqe_case_ids(family))
        if ids:
            return ids
    if algorithm_id in _GENERIC_QISKIT_ADAPTVQE_ALGORITHM_IDS:
        from pipelines.exact_bench.generic_static_qiskit_adapt_vqe import default_static_qiskit_adapt_vqe_case_ids

        ids = tuple(default_static_qiskit_adapt_vqe_case_ids(family))
        if ids:
            return ids
    if algorithm_id in _GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS:
        from pipelines.exact_bench.generic_static_adapt_variants import default_static_adapt_variant_case_ids

        ids = tuple(default_static_adapt_variant_case_ids(family, algorithm_id))
        if ids:
            return ids
    if algorithm_id in _GENERIC_STATIC_ED_REFERENCE_ALGORITHM_IDS:
        from pipelines.exact_bench.generic_static_ed_reference import default_static_ed_reference_case_ids

        ids = tuple(default_static_ed_reference_case_ids(family))
        if ids:
            return ids
    if algorithm_id in _EXTERNAL_STATIC_ADAPT_ALGORITHM_IDS:
        from pipelines.exact_bench.external_adapt.external_static_adapt_benchmark import (
            default_external_static_adapt_case_ids,
        )

        ids = tuple(default_external_static_adapt_case_ids(family, algorithm_id))
        if ids:
            return ids
    if algorithm_id in _PHASE3_ADAPT_ALGORITHM_IDS:
        ids = tuple(str(getattr(spec, "benchmark_id")) for spec in _phase3_specs_for_family(family))
        if ids:
            return ids
    deferred_ids = table_i_deferred_case_ids(family)
    if deferred_ids:
        return deferred_ids
    return (f"{family}_L2_default",)


def _command_for_job(*, family: str, case_id: str, algorithm_id: str, output_dir: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pipelines.exact_bench.generic_static_benchmark",
        "--run-single",
        "--family",
        family,
        "--case-id",
        case_id,
        "--algorithm-id",
        algorithm_id,
        "--output-dir",
        str(output_dir),
    )


def _dispatch_kind(*, family: str, case_id: str, algorithm_id: str) -> str | None:
    if family == "hh" and algorithm_id in _HH_ALGORITHM_MAP:
        return "hh_static_ground_state"
    if algorithm_id in _GENERIC_QISKIT_HEA_ALGORITHM_IDS:
        from pipelines.exact_bench.generic_static_hea_qiskit_vqe import default_static_hea_case_ids

        if case_id in default_static_hea_case_ids(family):
            return "generic_static_hea_qiskit_vqe"
    if algorithm_id in _GENERIC_FAMILY_INFORMED_VQE_ALGORITHM_IDS:
        from pipelines.exact_bench.generic_static_family_informed_vqe import (
            default_static_family_informed_vqe_case_ids,
        )

        if case_id in default_static_family_informed_vqe_case_ids(family):
            return "generic_static_family_informed_vqe"
    if algorithm_id in _GENERIC_QISKIT_ADAPTVQE_ALGORITHM_IDS:
        from pipelines.exact_bench.generic_static_qiskit_adapt_vqe import default_static_qiskit_adapt_vqe_case_ids

        if case_id in default_static_qiskit_adapt_vqe_case_ids(family):
            return "generic_static_qiskit_adapt_vqe"
    if algorithm_id in _GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS:
        from pipelines.exact_bench.generic_static_adapt_variants import default_static_adapt_variant_case_ids

        if case_id in default_static_adapt_variant_case_ids(family, algorithm_id):
            return "generic_static_adapt_variants"
    if algorithm_id in _GENERIC_STATIC_ED_REFERENCE_ALGORITHM_IDS:
        from pipelines.exact_bench.generic_static_ed_reference import default_static_ed_reference_case_ids

        if case_id in default_static_ed_reference_case_ids(family):
            return "generic_static_ed_reference"
    if algorithm_id in _PHASE3_ADAPT_ALGORITHM_IDS:
        if _phase3_static_spec_for_case(family=family, case_id=case_id, algorithm_id=algorithm_id) is not None:
            return "phase3_static_adapt"
    if algorithm_id in _EXTERNAL_STATIC_ADAPT_ALGORITHM_IDS:
        from pipelines.exact_bench.external_adapt.external_static_adapt_benchmark import (
            default_external_static_adapt_case_ids,
            external_static_adapt_dispatch_for_algorithm,
        )

        if case_id in default_external_static_adapt_case_ids(family, algorithm_id):
            return external_static_adapt_dispatch_for_algorithm(algorithm_id)
    return None


def _resources_for_dispatch(dispatch: str | None) -> dict[str, str | int]:
    if dispatch == "phase3_static_adapt":
        return {"request_cpus": 1, "request_memory": "16GB", "request_disk": "20GB"}
    if dispatch == "generic_static_hea_qiskit_vqe":
        return {"request_cpus": 1, "request_memory": "8GB", "request_disk": "8GB"}
    if dispatch == "generic_static_family_informed_vqe":
        return {"request_cpus": 1, "request_memory": "16GB", "request_disk": "20GB"}
    if dispatch == "generic_static_qiskit_adapt_vqe":
        return {"request_cpus": 1, "request_memory": "16GB", "request_disk": "20GB"}
    if dispatch == "generic_static_adapt_variants":
        return {"request_cpus": 1, "request_memory": "16GB", "request_disk": "20GB"}
    if dispatch == "generic_static_ed_reference":
        return {"request_cpus": 1, "request_memory": "16GB", "request_disk": "20GB"}
    if dispatch in _EXTERNAL_STATIC_ADAPT_DISPATCHES:
        return {"request_cpus": 1, "request_memory": "8GB", "request_disk": "8GB"}
    return {"request_cpus": 1, "request_memory": "4GB", "request_disk": "4GB"}


def build_static_jobs(
    *,
    output_root: Path,
    families: Sequence[str] | None = None,
    algorithm_ids: Sequence[str] | None = None,
    include_skipped: bool = True,
) -> list[BenchmarkJob]:
    fams = _families_from_args(families)
    algs = _static_algorithms_from_args(algorithm_ids)
    jobs: list[BenchmarkJob] = []
    for family in fams:
        for alg in algs:
            app = evaluate_algorithm_for_family(alg, family)
            for case_id in _case_ids_for_family(family, alg.algorithm_id):
                status = app.status
                reason = app.reason
                command: tuple[str, ...] = ()
                deferred_reason = table_i_deferred_case_reason(family, case_id)
                dispatch = _dispatch_kind(family=family, case_id=case_id, algorithm_id=alg.algorithm_id) if status == "runnable" else None
                job_output = output_root / "static" / family / case_id / alg.algorithm_id
                if status == "runnable" and dispatch is None:
                    if deferred_reason is not None:
                        status = "skipped_not_implemented"
                        reason = deferred_reason
                    else:
                        status = "skipped_no_runner"
                        reason = "no concrete static dispatch mapping for this family/algorithm/case"
                if status == "runnable":
                    command = _command_for_job(
                        family=family,
                        case_id=case_id,
                        algorithm_id=alg.algorithm_id,
                        output_dir=job_output,
                    )
                if status != "runnable" and not include_skipped:
                    continue
                metadata = {
                    "dispatch": dispatch,
                    "required_pool_key": app.required_pool_key,
                    "resolved_pool_key": app.resolved_pool_key,
                    "external_algorithm": bool(
                        dispatch in {
                            "generic_static_hea_qiskit_vqe",
                            "generic_static_qiskit_adapt_vqe",
                        }
                        or dispatch in _EXTERNAL_STATIC_ADAPT_DISPATCHES
                    ),
                    "benchmark_local_competitor": bool(dispatch == "generic_static_adapt_variants"),
                    "optional_dependencies": (
                        ["qiskit"]
                        if dispatch == "generic_static_hea_qiskit_vqe"
                        else ["qiskit", "qiskit_algorithms"]
                        if dispatch == "generic_static_qiskit_adapt_vqe"
                        else ["scipy"]
                        if dispatch == "generic_static_family_informed_vqe"
                        else (
                            []
                            if alg.algorithm_id
                            in {"static_geo_qeb_adapt_vqe", "static_geo_adapt_vqe", "static_pos_geo_adapt_vqe"}
                            else ["scipy"]
                        )
                        if dispatch == "generic_static_adapt_variants"
                        else ["adaptvqe", "openfermion", "qiskit", "quimb", "scipy"]
                        if dispatch in _EXTERNAL_STATIC_ADAPT_DISPATCHES
                        else []
                    ),
                    "phase3_controller_called": bool(dispatch == "phase3_static_adapt"),
                    "uses_existing_exact_target": bool(dispatch == "generic_static_ed_reference"),
                    "resource_guarded_execution": bool(
                        dispatch
                        in {
                            "generic_static_ed_reference",
                            "generic_static_qiskit_adapt_vqe",
                            "generic_static_family_informed_vqe",
                            "generic_static_adapt_variants",
                        }
                    ),
                    "table_i_deferred_case": bool(deferred_reason is not None),
                }
                if deferred_reason is not None:
                    metadata["table_i_deferred_reason"] = deferred_reason
                if maybe_comparator_source_profile(alg.algorithm_id) is not None:
                    metadata["comparator_source"] = comparator_source_fields(
                        alg.algorithm_id,
                        runner_module=app.runner_module,
                    )
                metadata.update(
                    external_algorithm_manifest_metadata(
                        alg.algorithm_id,
                        status=status,
                        dispatch=dispatch,
                    )
                )
                jobs.append(
                    BenchmarkJob(
                        job_id=f"static__{family}__{case_id}__{alg.algorithm_id}",
                        domain="static",
                        family=family,
                        case_id=case_id,
                        algorithm_id=alg.algorithm_id,
                        status=status,
                        reason=reason,
                        command=command,
                        output_dir=str(job_output),
                        runner_module=app.runner_module,
                        qpu_faithful=app.qpu_faithful,
                        exact_assisted=app.exact_assisted,
                        diagnostic=app.diagnostic,
                        hamiltonian_generic=app.hamiltonian_generic,
                        resources=_resources_for_dispatch(dispatch),
                        metadata=metadata,
                    )
                )
    return jobs


def _require_unspecified_static_route(
    *,
    algorithm_id: str,
    static_route_id: object,
) -> str:
    route = (
        "unspecified"
        if static_route_id is None or static_route_id == ""
        else str(static_route_id).strip().lower().replace("-", "_")
    )
    if route != "unspecified":
        raise ValueError(
            f"{algorithm_id} cannot execute historical "
            f"static_route_id={static_route_id!r} from the generic benchmark "
            "launcher; static_route_id must be 'unspecified'. Canonical "
            "RA-ADAPT and Append-ADAPT execution must use their typed "
            "source-locked bundle protocols."
        )
    return "unspecified"


def _enforce_phase3_policy_algorithm_route_contract(policy, *, algorithm_id: str):
    _require_unspecified_static_route(
        algorithm_id=str(algorithm_id),
        static_route_id=getattr(policy.static, "static_route_id", None),
    )
    return policy


def _phase3_policy_for_algorithm(
    *,
    algorithm_id: str,
    pool_key: str,
    phase3_oracle_overrides: dict[str, object] | None = None,
    phase3_budget_overrides: dict[str, object] | None = None,
    phase3_runtime_overrides: dict[str, int] | None = None,
    hardware_resolution_overrides: dict[str, object] | None = None,
    shared_pauli_pool_overrides: dict[str, object] | None = None,
):
    from pipelines.exact_bench.static_benchmark_runtime import (
        AlgorithmPolicy,
        InnerOptimizerPolicy,
        PoolPolicy,
        SizeScaledBudget,
        StaticScaffoldPolicy,
    )

    pool = PoolPolicy(pool_key=str(pool_key))
    oracle_kwargs = dict(phase3_oracle_overrides or {})
    hardware_kwargs = dict(hardware_resolution_overrides or {})
    shared_pool_kwargs = dict(shared_pauli_pool_overrides or {})
    budget_overrides = dict(phase3_budget_overrides or {})
    runtime_kwargs = dict(phase3_runtime_overrides or {})
    static_budget_kwargs = {
        field: value
        for field, value in budget_overrides.items()
        if field in _PHASE3_BUDGET_STATIC_POLICY_FIELDS
    }
    inner_budget_kwargs = {
        field: value
        for field, value in budget_overrides.items()
        if field in _PHASE3_BUDGET_INNER_POLICY_FIELDS
    }
    static_kwargs = {
        **oracle_kwargs,
        **hardware_kwargs,
        **shared_pool_kwargs,
        **static_budget_kwargs,
        **runtime_kwargs,
    }
    inner = InnerOptimizerPolicy(**inner_budget_kwargs)
    if algorithm_id == "static_append_only_adapt_phase3":
        append_static_kwargs = {**static_kwargs, "static_route_id": "unspecified"}
        policy = AlgorithmPolicy(
            pool=pool,
            static=StaticScaffoldPolicy(
                phase2_enable_batching=False,
                phase1_prune_enabled=False,
                adapt_beam_live_branches=1,
                adapt_beam_children_per_parent=1,
                adapt_beam_terminated_keep=1,
                adapt_reopt_policy="append_only",
                adapt_window_size=3,
                adapt_window_topk=0,
                adapt_full_refit_every=0,
                adapt_final_full_refit=False,
                adapt_insertion_mode="append_only",
                adapt_allow_repeats=True,
                phase1_probe_max_positions=1,
                **append_static_kwargs,
            ),
            inner_optimizer=inner,
        )
        return _enforce_phase3_policy_algorithm_route_contract(
            policy,
            algorithm_id=algorithm_id,
        )
    policy = AlgorithmPolicy(
        pool=pool,
        static=StaticScaffoldPolicy(**static_kwargs),
        inner_optimizer=inner,
    )
    return _enforce_phase3_policy_algorithm_route_contract(
        policy,
        algorithm_id=algorithm_id,
    )


def _phase3_policy_json_env_value() -> str | None:
    raw = os.environ.get(_PHASE3_POLICY_JSON_ENV, "")
    value = str(raw).strip()
    return value or None


def _dataclass_kwargs_for(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {str(key): value for key, value in dict(data).items() if str(key) in allowed}


def _phase3_algorithm_policy_from_json(path: str | Path):
    from pipelines.exact_bench.static_benchmark_runtime import (
        AlgorithmPolicy,
        InnerOptimizerPolicy,
        PoolPolicy,
        SizeScaledBudget,
        StaticScaffoldPolicy,
    )

    policy_path = Path(path)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_payload = payload.get("policy", payload)
    if not isinstance(policy_payload, Mapping):
        raise ValueError(f"{_PHASE3_POLICY_JSON_ENV}={str(path)!r} does not contain a policy mapping.")
    pool_payload = policy_payload.get("pool", {})
    static_payload = policy_payload.get("static", {})
    inner_payload = policy_payload.get("inner_optimizer", {})
    if not isinstance(pool_payload, Mapping) or not isinstance(static_payload, Mapping) or not isinstance(inner_payload, Mapping):
        raise ValueError(f"{_PHASE3_POLICY_JSON_ENV}={str(path)!r} has malformed pool/static/inner_optimizer sections.")
    pool_kwargs = _dataclass_kwargs_for(PoolPolicy, pool_payload)
    for budget_key in ("phase1_budget", "phase2_budget"):
        budget_payload = pool_kwargs.get(budget_key)
        if isinstance(budget_payload, Mapping):
            pool_kwargs[budget_key] = SizeScaledBudget(**_dataclass_kwargs_for(SizeScaledBudget, budget_payload))
    return AlgorithmPolicy(
        pool=PoolPolicy(**pool_kwargs),
        static=StaticScaffoldPolicy(**_dataclass_kwargs_for(StaticScaffoldPolicy, static_payload)),
        inner_optimizer=InnerOptimizerPolicy(**_dataclass_kwargs_for(InnerOptimizerPolicy, inner_payload)),
    )


def _phase3_policy_with_env_overrides(
    policy,
    *,
    phase3_oracle_overrides: Mapping[str, object] | None = None,
    phase3_budget_overrides: Mapping[str, object] | None = None,
    phase3_runtime_overrides: Mapping[str, int] | None = None,
    hardware_resolution_overrides: Mapping[str, object] | None = None,
    shared_pauli_pool_overrides: Mapping[str, object] | None = None,
):
    static_updates: dict[str, object] = {}
    static_field_names = {field.name for field in fields(type(policy.static))}
    inner_field_names = {field.name for field in fields(type(policy.inner_optimizer))}
    for source in (
        phase3_oracle_overrides or {},
        phase3_runtime_overrides or {},
        hardware_resolution_overrides or {},
        shared_pauli_pool_overrides or {},
    ):
        for key, value in dict(source).items():
            if key in static_field_names:
                static_updates[str(key)] = value
    inner_updates: dict[str, object] = {}
    for key, value in dict(phase3_budget_overrides or {}).items():
        if key in static_field_names:
            static_updates[str(key)] = value
        if key in inner_field_names:
            inner_updates[str(key)] = value
    if static_updates:
        policy = replace(policy, static=replace(policy.static, **static_updates))
    if inner_updates:
        policy = replace(policy, inner_optimizer=replace(policy.inner_optimizer, **inner_updates))
    return policy


def _run_phase3_static_single(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    output_dir: Path,
    resolved_pool_key: str | None,
    phase3_oracle_overrides: dict[str, object] | None = None,
    phase3_budget_overrides: dict[str, object] | None = None,
    phase3_runtime_overrides: dict[str, int] | None = None,
    hardware_resolution_overrides: dict[str, object] | None = None,
    shared_pauli_pool_overrides: dict[str, object] | None = None,
    selected_logical_overrides: dict[str, str] | None = None,
    shots_per_pauli_term_proxy: int | None = None,
) -> dict:
    from pipelines.exact_bench.static_benchmark_runtime import run_static_benchmark

    spec = _phase3_static_spec_for_case(family=family, case_id=case_id, algorithm_id=algorithm_id)
    if spec is None:
        raise ValueError(f"No Phase3 static benchmark spec for family={family!r}, case_id={case_id!r}")
    budget_overrides = dict(phase3_budget_overrides or {})
    runtime_overrides = dict(phase3_runtime_overrides or {})
    if budget_overrides:
        spec = _phase3_spec_with_budget_base_args(spec, budget_overrides)
    if runtime_overrides:
        spec = _phase3_spec_with_runtime_base_args(spec, runtime_overrides)
    selected_overrides = dict(selected_logical_overrides or {})
    if selected_overrides:
        spec = replace(
            spec,
            selected_logical_route=str(selected_overrides.get("selected_logical_route") or "standard"),
            selected_logical_source_json=(
                str(selected_overrides.get("selected_logical_source_json") or "")
                or None
            ),
            selected_logical_transfer_mode=str(
                selected_overrides.get("selected_logical_transfer_mode") or "exact_match_v1"
            ),
        )
    policy_json_env = _phase3_policy_json_env_value()
    if policy_json_env is not None:
        policy = _phase3_algorithm_policy_from_json(policy_json_env)
        policy = _phase3_policy_with_env_overrides(
            policy,
            phase3_oracle_overrides=phase3_oracle_overrides,
            phase3_budget_overrides=budget_overrides,
            phase3_runtime_overrides=runtime_overrides,
            hardware_resolution_overrides=hardware_resolution_overrides,
            shared_pauli_pool_overrides=shared_pauli_pool_overrides,
        )
        policy = _enforce_phase3_policy_algorithm_route_contract(policy, algorithm_id=algorithm_id)
        pool_key = str(getattr(policy.pool, "pool_key", resolved_pool_key or "full_meta"))
    else:
        pool_key = str(resolved_pool_key or "full_meta")
        policy = _phase3_policy_for_algorithm(
            algorithm_id=algorithm_id,
            pool_key=pool_key,
            phase3_oracle_overrides=phase3_oracle_overrides,
            phase3_budget_overrides=budget_overrides,
            phase3_runtime_overrides=runtime_overrides,
            hardware_resolution_overrides=hardware_resolution_overrides,
            shared_pauli_pool_overrides=shared_pauli_pool_overrides,
        )
    result = run_static_benchmark(
        spec,
        policy,
        output_dir=Path(output_dir),
        benchmark_target_abs_delta_e=_env_float_or_none(_ENERGY_STOP_TARGET_ENV),
    )
    result_payload = asdict(result)
    result_payload.update(_phase3_static_table_contract_fields_from_result(result_payload))
    result_payload.update(
        _phase3_static_algorithmic_work_fields_from_result(
            result_payload,
            shots_per_pauli_term_proxy=shots_per_pauli_term_proxy,
        )
    )
    sidecar_to_write: dict[str, Any] | None = None
    if algorithm_id == "static_family_native_adapt_phase3":
        try:
            from pipelines.exact_bench.table_i_first_hit_sidecars import (
                SIDECAR_KEY,
                build_snake_first_hit_sidecar_for_payload,
            )

            sidecar_threshold = _env_float_or_none(_ENERGY_STOP_TARGET_ENV) or 2e-4
            sidecar, inventory = build_snake_first_hit_sidecar_for_payload(
                payload=result_payload,
                payload_path=result_payload.get("result_json"),
                threshold=float(sidecar_threshold),
            )
            result_payload["paper_i_first_hit_cost_replay"] = inventory
            if sidecar is not None:
                result_payload[SIDECAR_KEY] = sidecar
                sidecar_to_write = dict(sidecar)
        except Exception as exc:  # pragma: no cover - fail-closed reporting guard
            result_payload["paper_i_first_hit_cost_replay"] = {
                "status": "rerun_needed",
                "compiled_resource_status": "failed",
                "work_resource_status": "missing",
                "missing_fields": ["qiskit_first_hit_sidecar_generation"],
                "rerun_needed_reason": f"{type(exc).__name__}: {exc}",
            }
    payload = {
        "schema": "generic_static_benchmark_phase3_single_v1",
        "family": family,
        "case_id": case_id,
        "algorithm_id": algorithm_id,
        "status": "completed" if bool(result.success) else "failed",
        "runner": "pipelines.exact_bench.static_benchmark_runtime.run_static_benchmark",
        "resolved_pool_key": pool_key,
        "phase3_oracle_env_overlay": dict(phase3_oracle_overrides or {}),
        "phase3_budget_env_overlay": dict(budget_overrides),
        "phase3_runtime_env_overlay": dict(runtime_overrides),
        "hardware_resolution_profile_env_overlay": dict(hardware_resolution_overrides or {}),
        "static_route_env_overlay": {},
        "selected_logical_env_overlay": dict(selected_overrides),
        "shared_pauli_pool_env_overlay": dict(shared_pauli_pool_overrides or {}),
        "phase3_policy_json_env": policy_json_env,
        "result": result_payload,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if sidecar_to_write is not None:
        (output_dir / "paper_i_first_crossing_compiled_cost.json").write_text(
            json.dumps(sidecar_to_write, indent=2, sort_keys=True) + "\n"
        )
    (output_dir / "generic_static_single.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run_single(*, family: str, case_id: str, algorithm_id: str, output_dir: Path) -> dict:
    app = evaluate_algorithm_for_family(algorithm_id, family)
    hardware_resolution_overrides = _hardware_resolution_profile_policy_overrides_from_env()
    hardware_resolution_requested = _hardware_resolution_profile_policy_overrides_requested(hardware_resolution_overrides)
    _static_route_policy_overrides_from_env()
    dispatch = _dispatch_kind(family=family, case_id=case_id, algorithm_id=algorithm_id) if app.status == "runnable" else None
    phase3_oracle_overrides = _phase3_oracle_policy_overrides_from_env()
    phase3_oracle_requested = _phase3_oracle_policy_overrides_requested(phase3_oracle_overrides)
    phase3_budget_overrides = _phase3_budget_policy_overrides_from_env()
    phase3_budget_requested = _phase3_budget_policy_overrides_requested(phase3_budget_overrides)
    phase3_runtime_overrides = _phase3_runtime_policy_overrides_from_env()
    phase3_runtime_requested = _phase3_runtime_policy_overrides_requested(phase3_runtime_overrides)
    selected_logical_overrides = _selected_logical_overrides_from_env()
    selected_logical_requested = _selected_logical_overrides_requested(selected_logical_overrides)
    generic_adapt_runtime_split_overrides = _generic_adapt_runtime_split_overrides_from_env()
    generic_adapt_runtime_split_requested = _generic_adapt_runtime_split_overrides_requested(
        generic_adapt_runtime_split_overrides
    )
    shared_pauli_pool_overrides = _shared_pauli_pool_overrides_from_env()
    shared_pauli_pool_requested = _shared_pauli_pool_overrides_requested(shared_pauli_pool_overrides)
    generic_static_adapt_seed = _generic_static_adapt_seed_from_env()
    shots_per_pauli_term_proxy = _shots_per_pauli_term_proxy_from_env()
    shots_per_pauli_term_proxy_requested = shots_per_pauli_term_proxy is not None
    benchmark_decision_noise_config = benchmark_decision_noise_config_from_env(
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
    )
    benchmark_value_noise_config = _benchmark_value_noise_config_from_env(
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
    )

    reference_metric_config = _primary_reference_config_from_env()
    generic_static_adapt_stop_policy = _generic_static_adapt_stop_policy_from_env()
    generic_static_adapt_stop_policy_requested = (
        generic_static_adapt_stop_policy != _GENERIC_STATIC_ADAPT_STOP_POLICY_DEFAULT
    )
    powell_maxiter_cap_policy = _powell_maxiter_cap_policy_from_env()
    powell_maxiter_cap_policy_requested = bool(
        powell_maxiter_cap_policy
        != _POWELL_MAXITER_CAP_POLICY_STRICT
    )
    optimizer_overrides = _optimizer_dispatch_overrides_from_env(
        dispatch=dispatch,
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
    )

    def _finalize(payload: dict[str, Any]) -> dict[str, Any]:
        return _finalize_benchmark_payload(
            payload,
            output_dir=output_dir,
            value_noise_config=benchmark_value_noise_config,
            reference_metric_config=reference_metric_config,
            family=family,
            case_id=case_id,
            algorithm_id=algorithm_id,
        )

    if (
        phase3_oracle_requested
        or phase3_runtime_requested
        or hardware_resolution_requested
    ) and dispatch != "phase3_static_adapt":
        raise ValueError(
            "Phase3 oracle/value-noise or runtime CHTC env overlay is only valid for "
            "phase3_static_adapt records; hardware-resolution profile CHTC env overlay is only "
            "valid for phase3_static_adapt records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, algorithm_id={algorithm_id!r}."
        )
    if phase3_budget_requested and dispatch not in {"phase3_static_adapt", "generic_static_adapt_variants"}:
        raise ValueError(
            "Phase3/budget CHTC env overlay is only valid for phase3_static_adapt or "
            "generic_static_adapt_variants records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, algorithm_id={algorithm_id!r}."
        )
    if selected_logical_requested and dispatch not in {"generic_static_adapt_variants", "phase3_static_adapt"}:
        raise ValueError(
            "selected-logical CHTC env overlay is only valid for generic_static_adapt_variants or phase3_static_adapt records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, algorithm_id={algorithm_id!r}."
        )
    if shots_per_pauli_term_proxy_requested and dispatch != "phase3_static_adapt":
        raise ValueError(
            "shots_per_pauli_term_proxy CHTC env overlay is only valid for phase3_static_adapt records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, algorithm_id={algorithm_id!r}."
        )
    if generic_static_adapt_stop_policy_requested and dispatch != "generic_static_adapt_variants":
        raise ValueError(
            "generic_adapt_stop_policy env overlay is only valid for generic_static_adapt_variants records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, algorithm_id={algorithm_id!r}."
        )
    if powell_maxiter_cap_policy_requested and dispatch != "generic_static_adapt_variants":
        raise ValueError(
            "powell_maxiter_cap_policy env overlay is only valid for "
            "generic_static_adapt_variants records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, "
            f"algorithm_id={algorithm_id!r}."
        )
    if generic_adapt_runtime_split_requested and dispatch != "generic_static_adapt_variants":
        raise ValueError(
            "generic_adapt_runtime_split env overlay is only valid for generic_static_adapt_variants records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, algorithm_id={algorithm_id!r}."
        )
    if shared_pauli_pool_requested and dispatch not in {"generic_static_adapt_variants", "phase3_static_adapt"}:
        raise ValueError(
            "shared_pauli_pool env overlay is only valid for generic_static_adapt_variants or phase3_static_adapt records; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, algorithm_id={algorithm_id!r}."
        )
    if shared_pauli_pool_requested and generic_adapt_runtime_split_requested:
        raise ValueError("shared_pauli_pool env overlay cannot be combined with generic_adapt_runtime_split env overlay.")
    if generic_static_adapt_seed is not None and dispatch != "generic_static_adapt_variants":
        raise ValueError(
            f"{_GENERIC_STATIC_ADAPT_SEED_ENV} is only valid for generic_static_adapt_variants; "
            f"record dispatch={dispatch!r}, family={family!r}, case_id={case_id!r}, "
            f"algorithm_id={algorithm_id!r}."
        )
    if app.status != "runnable" or dispatch is None:
        deferred_reason = table_i_deferred_case_reason(family, case_id)
        status = app.status if app.status != "runnable" else "skipped_no_runner"
        reason = app.reason if app.status != "runnable" else "no concrete static dispatch mapping for this family/algorithm/case"
        if deferred_reason is not None:
            status = "skipped_not_implemented"
            reason = deferred_reason
        metadata = external_algorithm_manifest_metadata(
            algorithm_id,
            status=status,
            dispatch=dispatch,
        )
        metadata["table_i_deferred_case"] = bool(deferred_reason is not None)
        if deferred_reason is not None:
            metadata["table_i_deferred_reason"] = deferred_reason
        payload = {
            "schema": "generic_static_benchmark_single_v1",
            "family": family,
            "case_id": case_id,
            "algorithm_id": algorithm_id,
            "status": status,
            "reason": reason,
            "metadata": metadata,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "skip.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return _finalize(payload)

    if bool(benchmark_decision_noise_config.enabled):
        if dispatch == "phase3_static_adapt":
            raise ValueError(
                "benchmark_decision_noise_* is not valid for Phase3/SNAKE static rows; "
                "use phase3_oracle_value_noise_* for native Phase3 controller value noise."
            )
        if not _benchmark_decision_noise_supported_for_dispatch(
            dispatch=dispatch,
            algorithm_id=algorithm_id,
        ):
            payload = _unsupported_benchmark_decision_noise_payload(
                family=family,
                case_id=case_id,
                algorithm_id=algorithm_id,
                dispatch=dispatch,
                output_dir=output_dir,
                config=benchmark_decision_noise_config,
            )
            return _finalize(payload)

    if dispatch == "phase3_static_adapt":
        payload = _run_phase3_static_single(
            family=family,
            case_id=case_id,
            algorithm_id=algorithm_id,
            output_dir=output_dir,
            resolved_pool_key=app.resolved_pool_key,
            phase3_oracle_overrides=phase3_oracle_overrides if phase3_oracle_requested else None,
            phase3_budget_overrides=phase3_budget_overrides if phase3_budget_requested else None,
            phase3_runtime_overrides=phase3_runtime_overrides if phase3_runtime_requested else None,
            hardware_resolution_overrides=hardware_resolution_overrides if hardware_resolution_requested else None,
            shared_pauli_pool_overrides=shared_pauli_pool_overrides if shared_pauli_pool_requested else None,
            selected_logical_overrides=selected_logical_overrides if selected_logical_requested else None,
            shots_per_pauli_term_proxy=shots_per_pauli_term_proxy,
        )
        return _finalize(payload)
    if dispatch == "generic_static_hea_qiskit_vqe":
        from pipelines.exact_bench.generic_static_hea_qiskit_vqe import run_static_hea_qiskit_vqe_single

        kwargs: dict[str, Any] = {
            "family": family,
            "case_id": case_id,
            "output_dir": output_dir,
        }
        hea_reps = _env_int_or_none(_HEA_REPS_ENV)
        hea_maxiter = _env_int_or_none(_HEA_MAXITER_ENV)
        if hea_reps is not None:
            kwargs["reps"] = int(hea_reps)
        if hea_maxiter is not None:
            kwargs["maxiter"] = int(hea_maxiter)
        if bool(benchmark_decision_noise_config.enabled):
            kwargs["benchmark_decision_noise_config"] = benchmark_decision_noise_config
        payload = _call_with_optimizer_overrides(
            run_static_hea_qiskit_vqe_single,
            kwargs,
            optimizer_overrides,
            dispatch="generic_static_hea_qiskit_vqe",
        )
        return _finalize(payload)
    if dispatch == "generic_static_qiskit_adapt_vqe":
        from pipelines.exact_bench.generic_static_qiskit_adapt_vqe import run_static_qiskit_adapt_vqe_single

        payload = run_static_qiskit_adapt_vqe_single(
            family=family,
            case_id=case_id,
            output_dir=output_dir,
        )
        return _finalize(payload)
    if dispatch == "generic_static_family_informed_vqe":
        from pipelines.exact_bench.generic_static_family_informed_vqe import (
            run_static_family_informed_vqe_single,
        )

        kwargs = {
            "family": family,
            "case_id": case_id,
            "output_dir": output_dir,
        }
        if bool(benchmark_decision_noise_config.enabled):
            kwargs["benchmark_decision_noise_config"] = benchmark_decision_noise_config
        payload = _call_with_optimizer_overrides(
            run_static_family_informed_vqe_single,
            kwargs,
            optimizer_overrides,
            dispatch="generic_static_family_informed_vqe",
        )
        return _finalize(payload)
    if dispatch == "generic_static_adapt_variants":
        from pipelines.exact_bench.generic_static_adapt_variants import run_generic_static_adapt_variant_single

        energy_stop_target = _env_float_or_none(_ENERGY_STOP_TARGET_ENV)
        kwargs: dict[str, Any] = {
            "family": family,
            "case_id": case_id,
            "algorithm_id": algorithm_id,
            "output_dir": output_dir,
            "energy_stop_target": energy_stop_target,
            "first_hit_thresholds": _env_first_hit_thresholds(),
            **_env_reference_kwargs(),
            **_env_hh_pool_kwargs(),
        }
        if generic_static_adapt_seed is not None:
            kwargs["seed"] = int(generic_static_adapt_seed)
        progress_jsonl_path = _env_text_or_none(_GENERIC_STATIC_ADAPT_PROGRESS_JSONL_ENV)
        if progress_jsonl_path is not None:
            kwargs["progress_jsonl_path"] = progress_jsonl_path
        if bool(benchmark_decision_noise_config.enabled):
            kwargs["benchmark_decision_noise_config"] = benchmark_decision_noise_config
        if selected_logical_requested:
            kwargs.update(selected_logical_overrides)
        if generic_adapt_runtime_split_requested:
            kwargs.update(generic_adapt_runtime_split_overrides)
        if shared_pauli_pool_requested:
            kwargs.update(shared_pauli_pool_overrides)
        if phase3_budget_requested:
            if "adapt_max_depth" in phase3_budget_overrides:
                kwargs["max_adapt_iterations"] = int(phase3_budget_overrides["adapt_max_depth"])
            if "adapt_maxiter" in phase3_budget_overrides:
                kwargs["optimizer_maxiter"] = int(phase3_budget_overrides["adapt_maxiter"])
            if "adapt_allow_repeats" in phase3_budget_overrides:
                kwargs["allow_repeats"] = bool(phase3_budget_overrides["adapt_allow_repeats"])
        if generic_static_adapt_stop_policy == _GENERIC_STATIC_ADAPT_STOP_POLICY_FIXED_HORIZON_NO_TARGET:
            if energy_stop_target is not None:
                raise ValueError(
                    "fixed_horizon_no_target_v1 requires energy_stop_target to be absent/blank; "
                    "target-hit stopping is intentionally disabled for matched-horizon comparator runs."
                )
            if "adapt_max_depth" not in phase3_budget_overrides:
                raise ValueError(
                    "fixed_horizon_no_target_v1 requires GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH "
                    "so the fixed horizon is explicit in the record."
                )
            kwargs["gradient_threshold"] = 0.0
        if generic_static_adapt_stop_policy_requested:
            kwargs["generic_adapt_stop_policy"] = generic_static_adapt_stop_policy
        if powell_maxiter_cap_policy_requested:
            kwargs["powell_maxiter_cap_policy"] = powell_maxiter_cap_policy
        payload = _call_with_optimizer_overrides(
            run_generic_static_adapt_variant_single,
            kwargs,
            optimizer_overrides,
            dispatch="generic_static_adapt_variants",
        )
        if shared_pauli_pool_requested and isinstance(payload, dict):
            payload["shared_pauli_pool_env_overlay"] = dict(shared_pauli_pool_overrides)
        if generic_static_adapt_stop_policy_requested:
            metadata = payload.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["generic_adapt_stop_policy"] = generic_static_adapt_stop_policy
            payload["generic_adapt_stop_policy"] = generic_static_adapt_stop_policy
        if powell_maxiter_cap_policy_requested:
            metadata = payload.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["powell_maxiter_cap_policy"] = powell_maxiter_cap_policy
            payload["powell_maxiter_cap_policy"] = powell_maxiter_cap_policy
        return _finalize(payload)
    if dispatch == "generic_static_ed_reference":
        from pipelines.exact_bench.generic_static_ed_reference import run_static_ed_reference_single

        payload = run_static_ed_reference_single(
            family=family,
            case_id=case_id,
            output_dir=output_dir,
        )
        return _finalize(payload)
    if dispatch in _EXTERNAL_STATIC_ADAPT_DISPATCHES:
        from pipelines.exact_bench.external_adapt.external_static_adapt_benchmark import (
            run_external_static_adapt_single,
        )

        payload = run_external_static_adapt_single(
            family=family,
            case_id=case_id,
            algorithm_id=algorithm_id,
            output_dir=output_dir,
        )
        return _finalize(payload)

    from pipelines.exact_bench import hh_static_ground_state_benchmark as hhbench

    cases_by_id = {case.case_id: case for case in hhbench.canonical_hh_benchmark_cases()}
    if case_id not in cases_by_id:
        raise ValueError(f"Unknown HH static benchmark case_id={case_id!r}")
    wanted_hh_ids = set(_HH_ALGORITHM_MAP[algorithm_id])
    selected_algs = tuple(
        alg for alg in hhbench.default_hh_benchmark_algorithms() if alg.algorithm_id in wanted_hh_ids
    )
    missing = wanted_hh_ids - {alg.algorithm_id for alg in selected_algs}
    if missing:
        raise ValueError(f"HH static benchmark algorithm mapping missing implementations: {sorted(missing)}")
    payload = hhbench.run_hh_static_ground_state_benchmark(
        output_dir=output_dir,
        cases=(cases_by_id[case_id],),
        algorithms=selected_algs,
        benchmark_decision_noise_config=(
            benchmark_decision_noise_config if bool(benchmark_decision_noise_config.enabled) else None
        ),
    )
    return _finalize(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or run generic static benchmark jobs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", action="append", dest="families", default=None)
    parser.add_argument("--algorithm-id", action="append", dest="algorithm_ids", default=None)
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--include-skipped", action="store_true", default=False)
    parser.add_argument("--run-single", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_single:
        if not args.families or len(args.families) != 1:
            raise SystemExit("--run-single requires exactly one --family")
        if not args.algorithm_ids or len(args.algorithm_ids) != 1:
            raise SystemExit("--run-single requires exactly one --algorithm-id")
        if not args.case_id:
            raise SystemExit("--run-single requires --case-id")
        result = run_single(
            family=str(args.families[0]),
            case_id=str(args.case_id),
            algorithm_id=str(args.algorithm_ids[0]),
            output_dir=Path(args.output_dir),
        )
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, sort_keys=True))
        status = str(result.get("status") or "").strip().lower()
        nested_result = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        nested_status = str(nested_result.get("status") or "").strip().lower()
        failed_statuses = {
            "failed",
            "quality_nonpassing",
            "completed_quality_nonpassing",
            "skipped_optional_dependency",
            "resource_guard",
            "blocked",
        }
        if status in failed_statuses or nested_status in failed_statuses:
            return 3
        if not status or not (status in {"ok", "success", "completed"} or status.startswith("completed_")):
            return 2
        return 0

    jobs = build_static_jobs(
        output_root=Path(args.output_dir),
        families=args.families,
        algorithm_ids=args.algorithm_ids,
        include_skipped=bool(args.include_skipped),
    )
    summary = write_manifest_bundle(output_dir=args.output_dir, jobs=jobs, label="generic_static_benchmark")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
