#!/usr/bin/env python3
"""Paper-I HH shared SPSA calibration contract.

This is a diagnostic/settings-search lane for the active Hubbard-Holstein
Table-III method surface.  Each Optuna trial samples one SPSA gain profile for
one method, evaluates that same profile over the six displayed HH regimes, and
returns one shared objective value.  The artifacts produced here are settings
calibration artifacts, not manuscript table cells.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (
    PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    PAPER_I_MAIN_TABLES_SPSA_TARGET,
)


PAPER_I_HH_SHARED_SPSA_CALIBRATION_PROFILE_ID = "paper_i_hh_shared_spsa_optuna_calibration_v1"
PAPER_I_HH_SHARED_SPSA_CALIBRATION_CONFIG_VERSION = "paper_i_hh_shared_spsa_calibration_config_v1"
PAPER_I_HH_SHARED_SPSA_CALIBRATION_PLAN_PATH = (
    "docs/plans/paper-i-hh-shared-spsa-calibration-2026-06-20.md"
)
PAPER_I_HH_SHARED_SPSA_REFIT_ENGINE = "src.quantum.spsa_optimizer:spsa_minimize"
PAPER_I_HH_SHARED_SPSA_OPTIMIZER_FAIRNESS_POLICY = "shared_spsa_equal_adapt_and_final_refit_maxiter_v1"
PAPER_I_HH_SHARED_SPSA_DEFAULT_MAXITER = 180
PAPER_I_HH_SHARED_SPSA_DEFAULT_MAX_DEPTH = 30
PAPER_I_HH_SHARED_SPSA_DEFAULT_SEED = 42
PAPER_I_HH_SHARED_SPSA_DEFAULT_EVAL_REPEATS = 1
PAPER_I_HH_SHARED_SPSA_DEFAULT_EVAL_AGG = "mean"
PAPER_I_HH_SHARED_SPSA_DEFAULT_AVG_LAST = 0
PAPER_I_HH_SHARED_SPSA_TARGET_LABEL = "2e-4"
PAPER_I_HH_SHARED_SPSA_NATIVE_FORCED_ENGINE = "src.quantum.spsa_optimizer:spsa_minimize"
PAPER_I_HH_SHARED_SPSA_LEGACY_MONOTONE_ENGINE = "exact_bench_spsa:energy_only_descent"
PAPER_I_HH_SHARED_SPSA_REFIT_ENGINE_BY_KEY = {
    "native_forced": PAPER_I_HH_SHARED_SPSA_NATIVE_FORCED_ENGINE,
    "legacy_monotone": PAPER_I_HH_SHARED_SPSA_LEGACY_MONOTONE_ENGINE,
}
PAPER_I_HH_SHARED_SPSA_ENGINE_LABEL_BY_KEY = {
    "native_forced": "native forced full budget",
    "legacy_monotone": "legacy monotone cap",
}
PAPER_I_HH_SHARED_SPSA_SNAKE_RUNTIME_WORKER_FLAGS = (
    "--adapt-parallel-gradient-workers",
    "--adapt-spsa-parallel-evaluations",
)

DEFAULT_APPEND_GEO_SOURCE_RECORDS = Path(
    "chtc/phase3_optuna/input/paper_i_hh_native200_depth30_20260619_v1/"
    "paper_i_hh_spsa_budget_ladder_records.tsv"
)
DEFAULT_SNAKE_SOURCE_RECORDS = Path(
    "chtc/phase3_optuna/input/paper_i_hh_native200_forced_depth30_noearlystop_20260619_v2/"
    "paper_i_hh_spsa_budget_ladder_records.tsv"
)


@dataclass(frozen=True)
class HHSharedSPSAMethod:
    method_key: str
    method_label: str
    algorithm_id: str
    runner_kind: str


@dataclass(frozen=True)
class HHSharedSPSARegime:
    display_regime: str
    case_id: str
    suite_profile: str
    n_ph_work: int
    n_ph_ref: int
    u_over_t: float
    lambda_value: float


PAPER_I_HH_SHARED_SPSA_METHODS: tuple[HHSharedSPSAMethod, ...] = (
    HHSharedSPSAMethod(
        method_key="append",
        method_label="append-only ADAPT",
        algorithm_id="static_full_meta_append_adapt_vqe",
        runner_kind="generic_static",
    ),
    HHSharedSPSAMethod(
        method_key="geo",
        method_label="Geo-ADAPT",
        algorithm_id="static_geo_adapt_vqe",
        runner_kind="generic_static",
    ),
    HHSharedSPSAMethod(
        method_key="snake",
        method_label="SNAKE",
        algorithm_id="static_family_native_adapt_phase3",
        runner_kind="snake_source_locked",
    ),
)

PAPER_I_HH_SHARED_SPSA_REGIMES: tuple[HHSharedSPSARegime, ...] = (
    HHSharedSPSARegime(
        display_regime="weak-weak",
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
        suite_profile=PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
        n_ph_work=2,
        n_ph_ref=2,
        u_over_t=0.25,
        lambda_value=0.25,
    ),
    HHSharedSPSARegime(
        display_regime="intermediate-weak",
        case_id="hh_L2_nph2_three_model_sym_strong_weak",
        suite_profile=PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
        n_ph_work=2,
        n_ph_ref=2,
        u_over_t=1.25,
        lambda_value=0.25,
    ),
    HHSharedSPSARegime(
        display_regime="strong-weak",
        case_id="hh_L2_nph2_three_model_sym_u8_strong_weak",
        suite_profile="hh_symmetric_u8",
        n_ph_work=2,
        n_ph_ref=2,
        u_over_t=8.0,
        lambda_value=0.25,
    ),
    HHSharedSPSARegime(
        display_regime="weak-strong",
        case_id="hh_L2_nph4_three_model_sym_weak_strong",
        suite_profile=PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
        n_ph_work=4,
        n_ph_ref=4,
        u_over_t=0.25,
        lambda_value=1.25,
    ),
    HHSharedSPSARegime(
        display_regime="intermediate-strong",
        case_id="hh_L2_nph4_three_model_sym_strong_strong",
        suite_profile=PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
        n_ph_work=4,
        n_ph_ref=4,
        u_over_t=1.25,
        lambda_value=1.25,
    ),
    HHSharedSPSARegime(
        display_regime="strong-strong",
        case_id="hh_L2_nph4_three_model_sym_u8_strong_strong",
        suite_profile="hh_symmetric_u8",
        n_ph_work=4,
        n_ph_ref=4,
        u_over_t=8.0,
        lambda_value=1.25,
    ),
)

PAPER_I_HH_SHARED_SPSA_METHOD_KEYS = tuple(method.method_key for method in PAPER_I_HH_SHARED_SPSA_METHODS)
PAPER_I_HH_SHARED_SPSA_ALGORITHM_IDS = tuple(method.algorithm_id for method in PAPER_I_HH_SHARED_SPSA_METHODS)
PAPER_I_HH_SHARED_SPSA_REGIME_LABELS = tuple(regime.display_regime for regime in PAPER_I_HH_SHARED_SPSA_REGIMES)
PAPER_I_HH_SHARED_SPSA_SCHEDULE_FIELDS = PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS

_CONFIG_REQUIRED_FIELDS = frozenset(
    {
        "profile_id",
        "config_version",
        "mode",
        "approved_for_full_generation",
        "approved_by",
        "approved_at",
        "plan_path",
        "n_trials",
        "sampler_seed",
        "maxiter",
        "max_depth",
        "case_parallelism",
        "per_case_cpus",
        "spsa_seed",
        "spsa_eval_repeats",
        "spsa_eval_agg",
        "spsa_avg_last",
        "failure_penalty",
        "clipping_log10_error_ratio",
        "resource_tiebreak_weight",
        "target_abs_delta_e",
        "methods",
        "regimes",
        "per_method_search_spaces",
    }
)
_SEARCH_SPACE_KINDS = frozenset({"float", "int", "log_uniform", "log-uniform", "choice"})
_SCALAR_FINAL_REFIT_FIELDS = frozenset(
    {
        "final_refit_maxiter",
        "adapt_final_refit_maxiter",
        "shared_final_refit_maxiter",
    }
)
_METHOD_BUDGET_OVERRIDE_FIELDS = frozenset(
    {
        "method_maxiter_budgets",
        "method_final_refit_maxiter",
        "method_final_refit_maxiters",
        "final_refit_maxiter_by_method",
        "adapt_final_refit_maxiter_by_method",
    }
)


def hh_shared_spsa_methods() -> tuple[HHSharedSPSAMethod, ...]:
    return PAPER_I_HH_SHARED_SPSA_METHODS


def hh_shared_spsa_regimes() -> tuple[HHSharedSPSARegime, ...]:
    return PAPER_I_HH_SHARED_SPSA_REGIMES


def method_by_key_or_id(value: str) -> HHSharedSPSAMethod:
    key = str(value).strip()
    for method in PAPER_I_HH_SHARED_SPSA_METHODS:
        if key in {method.method_key, method.algorithm_id, method.method_label}:
            return method
    known = ", ".join(PAPER_I_HH_SHARED_SPSA_METHOD_KEYS)
    raise ValueError(f"Unknown Paper-I HH shared-SPSA method {value!r}; known method keys: {known}")


def regime_by_label(value: str) -> HHSharedSPSARegime:
    key = str(value).strip()
    for regime in PAPER_I_HH_SHARED_SPSA_REGIMES:
        if key in {regime.display_regime, regime.case_id}:
            return regime
    known = ", ".join(PAPER_I_HH_SHARED_SPSA_REGIME_LABELS)
    raise ValueError(f"Unknown Paper-I HH shared-SPSA regime {value!r}; known regimes: {known}")


def config_sha256_for_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive_int(value: object, *, field: str) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{field} must be a positive integer; got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive; got {value!r}")
    return int(parsed)


def _nonnegative_int(value: object, *, field: str) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{field} must be a nonnegative integer; got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be nonnegative; got {value!r}")
    return int(parsed)


def _finite_float(value: object, *, field: str, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{field} must be finite numeric; got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite; got {value!r}")
    if positive and parsed <= 0.0:
        raise ValueError(f"{field} must be positive; got {value!r}")
    if nonnegative and parsed < 0.0:
        raise ValueError(f"{field} must be nonnegative; got {value!r}")
    return float(parsed)


def _validate_search_space(config: Mapping[str, Any]) -> None:
    raw_spaces = config.get("per_method_search_spaces")
    if not isinstance(raw_spaces, Mapping):
        raise ValueError("per_method_search_spaces must be an object")
    methods = tuple(str(item) for item in config["methods"])
    for method_key in methods:
        method = method_by_key_or_id(method_key)
        space = raw_spaces.get(method.method_key) or raw_spaces.get(method.algorithm_id)
        if not isinstance(space, Mapping):
            raise ValueError(f"per_method_search_spaces missing object for {method.method_key!r}")
        fields = set(str(name) for name in space)
        expected = set(PAPER_I_HH_SHARED_SPSA_SCHEDULE_FIELDS)
        if fields != expected:
            raise ValueError(
                f"Search-space fields for {method.method_key} must be {sorted(expected)}; got {sorted(fields)}"
            )
        for name, spec in space.items():
            if not isinstance(spec, Mapping):
                raise ValueError(f"Search-space spec for {method.method_key}.{name} must be an object")
            kind = str(spec.get("type", spec.get("kind", ""))).strip().lower().replace("-", "_")
            if kind not in {k.replace("-", "_") for k in _SEARCH_SPACE_KINDS}:
                raise ValueError(f"Unsupported search-space type for {method.method_key}.{name}: {kind!r}")
            if kind == "choice":
                choices = spec.get("choices")
                if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
                    raise ValueError(f"Choice search-space {method.method_key}.{name} must have non-empty choices")
                for choice in choices:
                    _finite_float(choice, field=f"{method.method_key}.{name}.choice", positive=True)
            else:
                low_key = "low" if "low" in spec else "min"
                high_key = "high" if "high" in spec else "max"
                if low_key not in spec or high_key not in spec:
                    raise ValueError(f"Search-space {method.method_key}.{name} must define low/high or min/max")
                low = _finite_float(spec[low_key], field=f"{method.method_key}.{name}.low", positive=True)
                high = _finite_float(spec[high_key], field=f"{method.method_key}.{name}.high", positive=True)
                if low > high:
                    raise ValueError(f"Search-space {method.method_key}.{name} has low > high")


def _optimizer_budget_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer budget; got {value!r}")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{field} must be an integer budget; got {value!r}") from exc
    if str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise ValueError(f"{field} must be an integer budget; got {value!r}")
    if parsed < 0:
        raise ValueError(f"{field} must be nonnegative; got {value!r}")
    return int(parsed)


def _shared_spsa_budget_violation(field: str, method_key: str | None, value: int, expected: int) -> ValueError:
    scope = "" if method_key is None else f".{method_key}"
    return ValueError(
        "optimizer_fairness_violation: "
        f"{field}{scope}={value} but active Paper-I HH shared-SPSA runs require "
        f"adapt maxiter and final-refit maxiter to equal shared maxiter={expected}"
    )


def validate_shared_spsa_optimizer_fairness(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return optimizer-fairness metadata or raise on unequal budgets.

    A value of 0 for a final-refit budget means "reuse the ADAPT maxiter" in
    the lower-level SNAKE runner, so it is normalized to the shared maxiter.
    """

    expected = _optimizer_budget_int(config.get("maxiter"), field="maxiter")
    if expected <= 0:
        raise ValueError(f"maxiter must be positive; got {expected!r}")
    method_keys = tuple(method_by_key_or_id(str(item)).method_key for item in config.get("methods", ()))

    for field in sorted(_SCALAR_FINAL_REFIT_FIELDS):
        if field not in config:
            continue
        raw = _optimizer_budget_int(config[field], field=field)
        normalized = expected if raw == 0 else raw
        if normalized != expected:
            raise _shared_spsa_budget_violation(field, None, raw, expected)

    for field in sorted(_METHOD_BUDGET_OVERRIDE_FIELDS):
        if field not in config:
            continue
        raw_map = config[field]
        if not isinstance(raw_map, Mapping):
            raise ValueError(f"{field} must be a method-keyed object when present")
        for raw_method, raw_value in raw_map.items():
            method = method_by_key_or_id(str(raw_method))
            if method.method_key not in method_keys:
                continue
            parsed = _optimizer_budget_int(raw_value, field=f"{field}.{method.method_key}")
            normalized = expected if parsed == 0 and "final_refit" in field else parsed
            if normalized != expected:
                raise _shared_spsa_budget_violation(field, method.method_key, parsed, expected)

    return {
        "optimizer_fairness_policy": PAPER_I_HH_SHARED_SPSA_OPTIMIZER_FAIRNESS_POLICY,
        "shared_optimizer_maxiter": expected,
        "shared_final_refit_maxiter": expected,
        "methods": list(method_keys),
    }


def normalize_spsa_refit_engine_key(value: object) -> str:
    key = str(value or "native_forced").strip()
    if key in PAPER_I_HH_SHARED_SPSA_REFIT_ENGINE_BY_KEY:
        return key
    known = ", ".join(sorted(PAPER_I_HH_SHARED_SPSA_REFIT_ENGINE_BY_KEY))
    raise ValueError(f"Unsupported spsa_refit_engine key {value!r}; known keys: {known}")


def spsa_refit_engine_for_key(value: object) -> str:
    return PAPER_I_HH_SHARED_SPSA_REFIT_ENGINE_BY_KEY[normalize_spsa_refit_engine_key(value)]


def spsa_refit_engine_label_for_key(value: object) -> str:
    return PAPER_I_HH_SHARED_SPSA_ENGINE_LABEL_BY_KEY[normalize_spsa_refit_engine_key(value)]


def normalize_spsa_refit_engine_keys(value: object) -> list[str]:
    raw = ["native_forced"] if value is None or value == "" else value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("spsa_refit_engines must be a non-empty sequence when provided")
    keys = [normalize_spsa_refit_engine_key(item) for item in raw]
    if len(set(keys)) != len(keys):
        raise ValueError(f"spsa_refit_engines contain duplicates after normalization: {keys}")
    return keys


def normalize_snake_runtime_worker_overrides(value: object) -> dict[str, int]:
    if value is None or value == "":
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("snake_runtime_worker_overrides must be an object when provided")
    out: dict[str, int] = {}
    for flag, raw_count in value.items():
        key = str(flag)
        if key not in PAPER_I_HH_SHARED_SPSA_SNAKE_RUNTIME_WORKER_FLAGS:
            known = ", ".join(PAPER_I_HH_SHARED_SPSA_SNAKE_RUNTIME_WORKER_FLAGS)
            raise ValueError(f"Unsupported snake runtime worker override flag {key!r}; known flags: {known}")
        out[key] = _positive_int(raw_count, field=f"snake_runtime_worker_overrides.{key}")
    return out


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config must be a JSON object: {path}")
    missing = sorted(_CONFIG_REQUIRED_FIELDS - set(str(key) for key in payload))
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")
    if payload["profile_id"] != PAPER_I_HH_SHARED_SPSA_CALIBRATION_PROFILE_ID:
        raise ValueError(f"Unsupported profile_id={payload['profile_id']!r}")
    if payload["config_version"] != PAPER_I_HH_SHARED_SPSA_CALIBRATION_CONFIG_VERSION:
        raise ValueError(f"Unsupported config_version={payload['config_version']!r}")
    methods = payload["methods"]
    regimes = payload["regimes"]
    if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)) or not methods:
        raise ValueError("methods must be a non-empty sequence")
    if not isinstance(regimes, Sequence) or isinstance(regimes, (str, bytes)) or not regimes:
        raise ValueError("regimes must be a non-empty sequence")
    method_keys = tuple(method_by_key_or_id(str(item)).method_key for item in methods)
    regime_labels = tuple(regime_by_label(str(item)).display_regime for item in regimes)
    if len(set(method_keys)) != len(method_keys):
        raise ValueError(f"methods contain duplicates after normalization: {method_keys}")
    if len(set(regime_labels)) != len(regime_labels):
        raise ValueError(f"regimes contain duplicates after normalization: {regime_labels}")
    config = dict(payload)
    config["methods"] = list(method_keys)
    config["regimes"] = list(regime_labels)
    config["n_trials"] = _positive_int(config["n_trials"], field="n_trials")
    config["sampler_seed"] = _nonnegative_int(config["sampler_seed"], field="sampler_seed")
    config["maxiter"] = _positive_int(config["maxiter"], field="maxiter")
    config["max_depth"] = _positive_int(config["max_depth"], field="max_depth")
    config["case_parallelism"] = _positive_int(config["case_parallelism"], field="case_parallelism")
    config["per_case_cpus"] = _positive_int(config["per_case_cpus"], field="per_case_cpus")
    config["spsa_seed"] = _nonnegative_int(config["spsa_seed"], field="spsa_seed")
    config["spsa_eval_repeats"] = _positive_int(config["spsa_eval_repeats"], field="spsa_eval_repeats")
    config["spsa_avg_last"] = _nonnegative_int(config["spsa_avg_last"], field="spsa_avg_last")
    config["spsa_refit_engines"] = normalize_spsa_refit_engine_keys(config.get("spsa_refit_engines"))
    config["snake_runtime_worker_overrides"] = normalize_snake_runtime_worker_overrides(
        config.get("snake_runtime_worker_overrides")
    )
    if str(config["spsa_eval_agg"]) not in {"mean", "median"}:
        raise ValueError("spsa_eval_agg must be 'mean' or 'median'")
    config["failure_penalty"] = _finite_float(config["failure_penalty"], field="failure_penalty", positive=True)
    config["resource_tiebreak_weight"] = _finite_float(
        config["resource_tiebreak_weight"],
        field="resource_tiebreak_weight",
        nonnegative=True,
    )
    config["target_abs_delta_e"] = _finite_float(config["target_abs_delta_e"], field="target_abs_delta_e", positive=True)
    clipping = config["clipping_log10_error_ratio"]
    if not isinstance(clipping, Sequence) or isinstance(clipping, (str, bytes)) or len(clipping) != 2:
        raise ValueError("clipping_log10_error_ratio must be a two-element sequence")
    clip_min = _finite_float(clipping[0], field="clipping_log10_error_ratio[0]")
    clip_max = _finite_float(clipping[1], field="clipping_log10_error_ratio[1]")
    if clip_min > clip_max:
        raise ValueError("clipping_log10_error_ratio min must be <= max")
    config["clipping_log10_error_ratio"] = [clip_min, clip_max]
    _validate_search_space(config)
    config.update(validate_shared_spsa_optimizer_fairness(config))
    if config["target_abs_delta_e"] != PAPER_I_MAIN_TABLES_SPSA_TARGET:
        # A nonstandard target is allowed for smoke/debug configs but must be explicit.
        config["target_label"] = f"{config['target_abs_delta_e']:.6g}"
    else:
        config["target_label"] = PAPER_I_HH_SHARED_SPSA_TARGET_LABEL
    return config


def search_space_for_method(config: Mapping[str, Any], method: HHSharedSPSAMethod) -> Mapping[str, Any]:
    spaces = config["per_method_search_spaces"]
    return spaces.get(method.method_key) or spaces[method.algorithm_id]


__all__ = [
    "DEFAULT_APPEND_GEO_SOURCE_RECORDS",
    "DEFAULT_SNAKE_SOURCE_RECORDS",
    "HHSharedSPSAMethod",
    "HHSharedSPSARegime",
    "PAPER_I_HH_SHARED_SPSA_ALGORITHM_IDS",
    "PAPER_I_HH_SHARED_SPSA_CALIBRATION_CONFIG_VERSION",
    "PAPER_I_HH_SHARED_SPSA_CALIBRATION_PLAN_PATH",
    "PAPER_I_HH_SHARED_SPSA_CALIBRATION_PROFILE_ID",
    "PAPER_I_HH_SHARED_SPSA_DEFAULT_AVG_LAST",
    "PAPER_I_HH_SHARED_SPSA_DEFAULT_EVAL_AGG",
    "PAPER_I_HH_SHARED_SPSA_DEFAULT_EVAL_REPEATS",
    "PAPER_I_HH_SHARED_SPSA_DEFAULT_MAXITER",
    "PAPER_I_HH_SHARED_SPSA_DEFAULT_MAX_DEPTH",
    "PAPER_I_HH_SHARED_SPSA_DEFAULT_SEED",
    "PAPER_I_HH_SHARED_SPSA_ENGINE_LABEL_BY_KEY",
    "PAPER_I_HH_SHARED_SPSA_LEGACY_MONOTONE_ENGINE",
    "PAPER_I_HH_SHARED_SPSA_NATIVE_FORCED_ENGINE",
    "PAPER_I_HH_SHARED_SPSA_METHOD_KEYS",
    "PAPER_I_HH_SHARED_SPSA_OPTIMIZER_FAIRNESS_POLICY",
    "PAPER_I_HH_SHARED_SPSA_REFIT_ENGINE",
    "PAPER_I_HH_SHARED_SPSA_REFIT_ENGINE_BY_KEY",
    "PAPER_I_HH_SHARED_SPSA_REGIME_LABELS",
    "PAPER_I_HH_SHARED_SPSA_SCHEDULE_FIELDS",
    "PAPER_I_HH_SHARED_SPSA_SNAKE_RUNTIME_WORKER_FLAGS",
    "config_sha256_for_path",
    "hh_shared_spsa_methods",
    "hh_shared_spsa_regimes",
    "load_and_validate_config",
    "method_by_key_or_id",
    "normalize_snake_runtime_worker_overrides",
    "normalize_spsa_refit_engine_key",
    "normalize_spsa_refit_engine_keys",
    "regime_by_label",
    "search_space_for_method",
    "spsa_refit_engine_for_key",
    "spsa_refit_engine_label_for_key",
    "validate_shared_spsa_optimizer_fairness",
]
