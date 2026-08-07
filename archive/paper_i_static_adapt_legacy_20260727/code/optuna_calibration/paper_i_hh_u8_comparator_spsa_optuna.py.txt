#!/usr/bin/env python3
"""Paper-I HH U/t=8 comparator SPSA Optuna calibration contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.exact_bench.paper_i_comparator_spsa_calibration import (
    PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_EXCLUDED_METHOD_REASONS,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (
    PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS,
)
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE,
    table_i_canonical_spec_by_case_id,
)

PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID = "paper_i_hh_u8_comparator_spsa_optuna_v1"
PAPER_I_HH_U8_COMPARATOR_SPSA_CONFIG_VERSION = "paper_i_hh_u8_comparator_spsa_config_v1"
PAPER_I_HH_U8_COMPARATOR_SPSA_PLAN_PATH = "docs/plans/paper-i-hh-u8-comparator-spsa-2026-06-11.md"
PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE = TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE
PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_ABS_DELTA_E = 2e-4
PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_LABEL = "2e-4"


@dataclass(frozen=True)
class HHU8ComparatorSPSATarget:
    target_id: str
    family: str
    case_ids: tuple[str, ...]
    hh_regime: str
    n_ph_work: int
    n_ph_ref: int
    u_over_t: float
    lambda_ep: float
    g_ep: float


PAPER_I_HH_U8_COMPARATOR_SPSA_TARGETS: tuple[HHU8ComparatorSPSATarget, ...] = (
    HHU8ComparatorSPSATarget(
        target_id="hh_u8_strong_weak",
        family="hh",
        case_ids=("hh_L2_nph2_three_model_sym_u8_strong_weak",),
        hh_regime="strong_weak",
        n_ph_work=2,
        n_ph_ref=5,
        u_over_t=8.0,
        lambda_ep=0.25,
        g_ep=0.3535533905932738,
    ),
    HHU8ComparatorSPSATarget(
        target_id="hh_u8_strong_strong",
        family="hh",
        case_ids=("hh_L2_nph4_three_model_sym_u8_strong_strong",),
        hh_regime="strong_strong",
        n_ph_work=4,
        n_ph_ref=7,
        u_over_t=8.0,
        lambda_ep=1.25,
        g_ep=0.7905694150420949,
    ),
)
PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_IDS: tuple[str, ...] = tuple(
    target.target_id for target in PAPER_I_HH_U8_COMPARATOR_SPSA_TARGETS
)
PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS: tuple[str, ...] = PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS
PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD = PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD

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
        "method_maxiter_budgets",
        "failure_penalty",
        "clipping_log10_error_ratio",
        "resource_tiebreak_weight",
        "per_method_search_spaces",
    }
)
_SEARCH_SPACE_KINDS = frozenset({"float", "int", "log_uniform", "log-uniform", "choice"})
_INT_SCHEDULE_FIELDS = frozenset({"family_informed_spsa_eval_repeats", "family_informed_spsa_avg_last"})


def calibration_targets() -> tuple[HHU8ComparatorSPSATarget, ...]:
    return PAPER_I_HH_U8_COMPARATOR_SPSA_TARGETS


def allowed_method_ids() -> tuple[str, ...]:
    return PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS


def target_by_id(target_id: str) -> HHU8ComparatorSPSATarget:
    key = str(target_id).strip()
    for target in PAPER_I_HH_U8_COMPARATOR_SPSA_TARGETS:
        if target.target_id == key:
            return target
    known = ", ".join(PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_IDS)
    raise ValueError(f"Unknown Paper-I HH U/t=8 comparator SPSA target_id={target_id!r}; known: {known}")


def validate_method_id(method_id: str) -> str:
    key = str(method_id).strip()
    if key in PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS:
        return key
    if key in PAPER_I_COMPARATOR_SPSA_CALIBRATION_EXCLUDED_METHOD_REASONS:
        raise ValueError(
            f"Method {method_id!r} is explicitly excluded from {PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID}: "
            f"{PAPER_I_COMPARATOR_SPSA_CALIBRATION_EXCLUDED_METHOD_REASONS[key]}."
        )
    known = ", ".join(PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS)
    raise ValueError(
        f"Method {method_id!r} is not one of the six visible non-SNAKE comparator methods for "
        f"{PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID}; allowed: {known}."
    )


def validate_target_cases() -> None:
    for target in PAPER_I_HH_U8_COMPARATOR_SPSA_TARGETS:
        if target.family != "hh" or len(target.case_ids) != 1:
            raise ValueError(f"U8 comparator target {target.target_id!r} must be one HH case.")
        spec = table_i_canonical_spec_by_case_id(
            target.family,
            target.case_ids[0],
            PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE,
        )
        if str(spec.benchmark_id) != target.case_ids[0]:
            raise ValueError(f"U8 comparator target {target.target_id!r} resolved to {spec.benchmark_id!r}.")
        if int(spec.exact_reference_n_ph_max) != int(target.n_ph_ref):
            raise ValueError(
                f"U8 comparator target {target.target_id!r} expected n_ph_ref={target.n_ph_ref}, "
                f"spec has {spec.exact_reference_n_ph_max}."
            )


def full_method_target_records(
    *,
    method_ids: Sequence[str] | None = None,
    target_ids: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:
    validate_target_cases()
    methods = tuple(validate_method_id(method_id) for method_id in (method_ids or allowed_method_ids()))
    targets = tuple(target_by_id(target_id) for target_id in (target_ids or PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_IDS))
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for method_id in methods:
        for target in targets:
            key = (method_id, target.target_id)
            if key in seen:
                raise ValueError(f"Duplicate U8 comparator SPSA method-target record {key!r}.")
            seen.add(key)
            records.append(
                {
                    "profile_id": PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID,
                    "method_id": method_id,
                    "target_id": target.target_id,
                    "family": target.family,
                    "case_ids": target.case_ids,
                    "case_ids_json": json.dumps(list(target.case_ids), separators=(",", ":")),
                }
            )
    return tuple(records)


def _text_or_empty(value: object) -> str:
    return "" if value is None else str(value).strip()


def _case_ids_from_record(record: Mapping[str, object], *, target: HHU8ComparatorSPSATarget) -> tuple[str, ...]:
    raw = _text_or_empty(record.get("case_ids_json"))
    if raw:
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"U8 comparator record target={target.target_id!r} has malformed case_ids_json.") from exc
        if not isinstance(payload, list) or not payload or not all(isinstance(item, str) and item.strip() for item in payload):
            raise ValueError(f"U8 comparator record target={target.target_id!r} case_ids_json must be a non-empty string array.")
        parsed_json = tuple(str(item).strip() for item in payload)
    else:
        parsed_json = None
    case_ids_value = record.get("case_ids")
    if case_ids_value is None or case_ids_value == "":
        parsed_plain = None
    elif isinstance(case_ids_value, str):
        parsed_plain = (case_ids_value,)
    else:
        try:
            parsed_plain = tuple(str(case_id) for case_id in case_ids_value)  # type: ignore[union-attr]
        except TypeError as exc:
            raise ValueError(f"U8 comparator record target={target.target_id!r} case_ids must be a sequence.") from exc
    if parsed_json is not None and parsed_plain is not None and parsed_json != parsed_plain:
        raise ValueError(
            f"U8 comparator record target={target.target_id!r} has mismatched case_ids_json={parsed_json!r} "
            f"and case_ids={parsed_plain!r}."
        )
    if parsed_json is not None:
        return parsed_json
    if parsed_plain is not None:
        return parsed_plain
    raise ValueError(f"U8 comparator record target={target.target_id!r} must include case_ids_json or case_ids.")


def validate_full_method_target_records(records: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    expected = {
        (method_id, target_id)
        for method_id in PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS
        for target_id in PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_IDS
    }
    for record in records:
        method_id = validate_method_id(str(record.get("method_id", record.get("algorithm_id", ""))))
        target = target_by_id(str(record.get("target_id", "")))
        if str(record.get("profile_id", "")) != PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID:
            raise ValueError(
                f"U8 comparator record method={method_id!r}, target={target.target_id!r} has wrong "
                f"profile_id={record.get('profile_id')!r}; expected {PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID!r}."
            )
        case_ids = _case_ids_from_record(record, target=target)
        if case_ids != target.case_ids:
            raise ValueError(
                f"U8 comparator record method={method_id!r}, target={target.target_id!r} has case_ids={case_ids!r}; "
                f"expected {target.case_ids!r}."
            )
        key = (method_id, target.target_id)
        if key in seen:
            raise ValueError(f"Duplicate U8 comparator SPSA method-target record {key!r}.")
        seen.add(key)
        normalized.append(dict(record))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"Expected exactly 12 full method-target records for {PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID}; "
            f"got {len(seen)} unique records, missing={missing}, extra={extra}."
        )
    return tuple(normalized)


def config_sha256_for_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def config_sha256_for_path(path: str | Path) -> str:
    return config_sha256_for_bytes(Path(path).read_bytes())


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    contents = Path(path).read_bytes()
    data = json.loads(contents.decode("utf-8"))
    return validate_calibration_config(data, config_sha256=config_sha256_for_bytes(contents))


def _as_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object/mapping.")
    return value


def _positive_int(value: object, *, field: str, min_value: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer >= {min_value}; got {value!r}.")
    if isinstance(value, int):
        parsed = int(value)
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or any(ch not in "+-0123456789" for ch in text) or text in {"+", "-"}:
            raise ValueError(f"{field} must be an integer >= {min_value}; got {value!r}.")
        parsed = int(text)
    else:
        raise ValueError(f"{field} must be an integer >= {min_value}; got {value!r}.")
    if parsed < int(min_value):
        raise ValueError(f"{field} must be an integer >= {min_value}; got {value!r}.")
    return parsed


def _finite_float(value: object, *, field: str, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite numeric; got {value!r}.")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{field} must be finite numeric; got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite numeric; got {value!r}.")
    if positive and parsed <= 0.0:
        raise ValueError(f"{field} must be > 0; got {value!r}.")
    if nonnegative and parsed < 0.0:
        raise ValueError(f"{field} must be >= 0; got {value!r}.")
    return parsed


def _validate_config_header(data: Mapping[str, Any]) -> None:
    missing = sorted(_CONFIG_REQUIRED_FIELDS - set(data))
    if missing:
        raise ValueError(f"U8 comparator SPSA config missing required fields: {missing}.")
    if data.get("profile_id") != PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID:
        raise ValueError(
            f"profile_id must be {PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID!r}; got {data.get('profile_id')!r}."
        )
    if data.get("config_version") != PAPER_I_HH_U8_COMPARATOR_SPSA_CONFIG_VERSION:
        raise ValueError(
            f"config_version must be {PAPER_I_HH_U8_COMPARATOR_SPSA_CONFIG_VERSION!r}; "
            f"got {data.get('config_version')!r}."
        )
    mode = str(data.get("mode", "")).strip().lower()
    if mode not in {"smoke", "full"}:
        raise ValueError(f"mode must be 'smoke' or 'full'; got {data.get('mode')!r}.")
    if _text_or_empty(data.get("plan_path")) != PAPER_I_HH_U8_COMPARATOR_SPSA_PLAN_PATH:
        raise ValueError(
            f"plan_path must be {PAPER_I_HH_U8_COMPARATOR_SPSA_PLAN_PATH!r}; got {data.get('plan_path')!r}."
        )
    if mode == "full":
        if data.get("approved_for_full_generation") is not True:
            raise ValueError("full U8 comparator SPSA config requires approved_for_full_generation=true.")
        if not _text_or_empty(data.get("approved_by")) or not _text_or_empty(data.get("approved_at")):
            raise ValueError("full U8 comparator SPSA config requires approval metadata: approved_by and approved_at.")


def _validate_exact_method_keys(mapping: Mapping[str, Any], *, field: str) -> None:
    keys = set(str(key) for key in mapping)
    expected = set(PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS)
    if keys != expected:
        raise ValueError(
            f"{field} must have exactly the six visible comparator method keys; "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}."
        )
    for key in keys:
        validate_method_id(key)


def _validate_search_space_entry(spec: Mapping[str, Any], *, field: str, param_name: str) -> None:
    kind = str(spec.get("type", spec.get("kind", ""))).strip().lower().replace("-", "_")
    if kind not in {value.replace("-", "_") for value in _SEARCH_SPACE_KINDS}:
        raise ValueError(f"{field} search-space type must be one of {sorted(_SEARCH_SPACE_KINDS)}; got {kind!r}.")
    expects_int = param_name in _INT_SCHEDULE_FIELDS
    if expects_int and kind not in {"int", "choice"}:
        raise ValueError(f"{field} must use int or choice search-space type for integer schedule field {param_name!r}.")
    if not expects_int and kind == "int":
        raise ValueError(f"{field} must not use int search-space type for float schedule field {param_name!r}.")
    if kind == "choice":
        choices = spec.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise ValueError(f"{field}.choices must be a non-empty array.")
        for index, choice in enumerate(choices):
            choice_field = f"{field}.choices[{index}]"
            if expects_int:
                _positive_int(choice, field=choice_field)
            else:
                _finite_float(choice, field=choice_field, positive=True)
        return
    low_key = "low" if "low" in spec else "min"
    high_key = "high" if "high" in spec else "max"
    if low_key not in spec or high_key not in spec:
        raise ValueError(f"{field} must define low/high or min/max bounds.")
    low = _finite_float(spec[low_key], field=f"{field}.{low_key}", positive=True)
    high = _finite_float(spec[high_key], field=f"{field}.{high_key}", positive=True)
    if not high > low:
        raise ValueError(f"{field} high bound must be greater than low bound; got low={low}, high={high}.")
    if kind == "int" and (not float(low).is_integer() or not float(high).is_integer()):
        raise ValueError(f"{field} integer search-space bounds must be integral.")


def _validate_per_method_search_spaces(data: Mapping[str, Any]) -> None:
    spaces = _as_mapping(data.get("per_method_search_spaces"), field="per_method_search_spaces")
    _validate_exact_method_keys(spaces, field="per_method_search_spaces")
    for method_id, method_space_raw in spaces.items():
        method = validate_method_id(str(method_id))
        method_space = _as_mapping(method_space_raw, field=f"per_method_search_spaces.{method}")
        if not method_space:
            raise ValueError(f"per_method_search_spaces.{method} must not be empty.")
        allowed = set(PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD[method])
        for param_name, spec_raw in method_space.items():
            param = str(param_name)
            if param not in allowed:
                raise ValueError(
                    f"per_method_search_spaces.{method}.{param} is not an allowed SPSA schedule field; "
                    f"allowed={sorted(allowed)}."
                )
            _validate_search_space_entry(
                _as_mapping(spec_raw, field=f"per_method_search_spaces.{method}.{param}"),
                field=f"per_method_search_spaces.{method}.{param}",
                param_name=param,
            )


def validate_calibration_config(data: Mapping[str, Any], *, config_sha256: str | None = None) -> dict[str, Any]:
    _validate_config_header(data)
    _positive_int(data.get("n_trials"), field="n_trials")
    _positive_int(data.get("sampler_seed"), field="sampler_seed", min_value=0)
    _finite_float(data.get("failure_penalty"), field="failure_penalty", positive=True)
    _finite_float(data.get("resource_tiebreak_weight"), field="resource_tiebreak_weight", nonnegative=True)
    clipping = data.get("clipping_log10_error_ratio")
    if not isinstance(clipping, Sequence) or isinstance(clipping, (str, bytes)) or len(clipping) != 2:
        raise ValueError("clipping_log10_error_ratio must be a two-value array [low, high].")
    low = _finite_float(clipping[0], field="clipping_log10_error_ratio[0]")
    high = _finite_float(clipping[1], field="clipping_log10_error_ratio[1]")
    if not high > low:
        raise ValueError("clipping_log10_error_ratio high value must be greater than low value.")
    budgets = _as_mapping(data.get("method_maxiter_budgets"), field="method_maxiter_budgets")
    _validate_exact_method_keys(budgets, field="method_maxiter_budgets")
    for method_id, value in budgets.items():
        _positive_int(value, field=f"method_maxiter_budgets.{method_id}")
    _validate_per_method_search_spaces(data)
    out = json.loads(json.dumps(dict(data), sort_keys=True))
    out["mode"] = str(data.get("mode", "")).strip().lower()
    if config_sha256 is not None:
        if (
            not isinstance(config_sha256, str)
            or len(config_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in config_sha256)
        ):
            raise ValueError("config_sha256 must be a 64-character lowercase SHA256 hex digest when provided.")
        out["config_sha256"] = config_sha256
    return out


validate_target_cases()
if len(PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS) != 6:
    raise ValueError("Paper-I HH U/t=8 comparator SPSA must expose exactly six allowed methods.")
if len(PAPER_I_HH_U8_COMPARATOR_SPSA_TARGETS) != 2:
    raise ValueError("Paper-I HH U/t=8 comparator SPSA must expose exactly two targets.")
if len(full_method_target_records()) != 12:
    raise ValueError("Paper-I HH U/t=8 comparator SPSA full matrix must contain exactly 12 records.")


__all__ = [
    "HHU8ComparatorSPSATarget",
    "PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS",
    "PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD",
    "PAPER_I_HH_U8_COMPARATOR_SPSA_CONFIG_VERSION",
    "PAPER_I_HH_U8_COMPARATOR_SPSA_PLAN_PATH",
    "PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID",
    "PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE",
    "PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_ABS_DELTA_E",
    "PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_IDS",
    "PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_LABEL",
    "PAPER_I_HH_U8_COMPARATOR_SPSA_TARGETS",
    "allowed_method_ids",
    "calibration_targets",
    "config_sha256_for_path",
    "full_method_target_records",
    "load_and_validate_config",
    "load_config",
    "target_by_id",
    "validate_calibration_config",
    "validate_full_method_target_records",
    "validate_method_id",
    "validate_target_cases",
]
