#!/usr/bin/env python3
"""Paper-I comparator SPSA Optuna calibration contract.

This module is intentionally pure: it names the isolated calibration profile,
visible non-SNAKE comparator methods, method-target matrix, method-specific SPSA
schedule fields, and JSON config validation helpers for the calibration lane.
Calibration artifacts produced under this profile are not manuscript/table
evidence; later runners must consume this contract explicitly before generating
records or evaluating schedules.
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
    PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS,
    PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    PAPER_I_MAIN_TABLES_SPSA_SNAKE_ALGORITHM_ID,
    paper_i_main_tables_spsa_contains_case,
)

PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID = "paper_i_comparator_spsa_optuna_calibration_v1"
PAPER_I_COMPARATOR_SPSA_CALIBRATION_CONFIG_VERSION = "paper_i_comparator_spsa_calibration_config_v1"
PAPER_I_COMPARATOR_SPSA_CALIBRATION_PLAN_PATH = (
    "docs/plans/paper-i-comparator-spsa-optuna-calibration-2026-05-31.md"
)


@dataclass(frozen=True)
class ComparatorCalibrationTarget:
    """A calibration objective target: one family and one or more visible cases."""

    target_id: str
    family: str
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class HHTableIIIRepairTarget(ComparatorCalibrationTarget):
    """Repair metadata for the visible Hubbard-Holstein Table III SPSA gap."""

    hh_tableiii_regime: str
    n_ph_work: int
    n_ph_ref: int
    adapt_max_depth: int


PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGETS: tuple[ComparatorCalibrationTarget, ...] = (
    ComparatorCalibrationTarget(
        target_id="hubbard_family",
        family="hubbard",
        case_ids=("hubbard_L2_three_model_weak", "hubbard_L2_three_model_strong"),
    ),
    ComparatorCalibrationTarget(
        target_id="spin_boson_family",
        family="spin_boson",
        case_ids=("spin_boson_L2_nph1_three_model_weak", "spin_boson_L2_nph2_three_model_strong"),
    ),
    ComparatorCalibrationTarget(
        target_id="hh_sym_weak_weak",
        family="hh",
        case_ids=("hh_L2_nph2_three_model_sym_weak_weak",),
    ),
    ComparatorCalibrationTarget(
        target_id="hh_sym_strong_weak",
        family="hh",
        case_ids=("hh_L2_nph2_three_model_sym_strong_weak",),
    ),
    ComparatorCalibrationTarget(
        target_id="hh_sym_weak_strong",
        family="hh",
        case_ids=("hh_L2_nph4_three_model_sym_weak_strong",),
    ),
    ComparatorCalibrationTarget(
        target_id="hh_sym_strong_strong",
        family="hh",
        case_ids=("hh_L2_nph4_three_model_sym_strong_strong",),
    ),
)
PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS: tuple[str, ...] = tuple(
    target.target_id for target in PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGETS
)
PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS: tuple[str, ...] = (
    PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS
)
HH_GEO_QEB_TABLEIII_REPAIR_SCOPE = "hh_geo_qeb_tableiii_v1"
HH_GEO_QEB_TABLEIII_REPAIR_RECORD_MODE = "hh_geo_qeb_tableiii_repair_v1"
HH_GEO_QEB_TABLEIII_REPAIR_SMOKE_RECORD_MODE = "hh_geo_qeb_tableiii_repair_smoke_v1"
HH_TABLEIII_REPAIR_TABLE_LABEL = "tab:fixed_accuracy_hh_cartesian"
HH_TABLEIII_REPAIR_PRIMARY_ENERGY_METRIC = "same_cutoff_abs_delta_e"
HH_TABLEIII_REPAIR_SAME_CUTOFF_ERROR_ROLE = "primary"
HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY = "finite_metrics_allow_quality_nonpassing_v1"
HH_TABLEIII_REPAIR_QUALITY_NONPASSING_PENALTY = 1.0
HH_TABLEIII_REPAIR_METHOD_IDS: tuple[str, ...] = (
    "static_geo_adapt_vqe",
    "static_qubit_qeb_adapt_vqe",
)
HH_TABLEIII_REPAIR_TARGETS: tuple[HHTableIIIRepairTarget, ...] = (
    HHTableIIIRepairTarget(
        target_id="hh_sym_weak_weak",
        family="hh",
        case_ids=("hh_L2_nph2_three_model_sym_weak_weak",),
        hh_tableiii_regime="weak_weak",
        n_ph_work=2,
        n_ph_ref=2,
        adapt_max_depth=30,
    ),
    HHTableIIIRepairTarget(
        target_id="hh_sym_strong_weak",
        family="hh",
        case_ids=("hh_L2_nph2_three_model_sym_strong_weak",),
        hh_tableiii_regime="strong_weak",
        n_ph_work=2,
        n_ph_ref=2,
        adapt_max_depth=50,
    ),
    HHTableIIIRepairTarget(
        target_id="hh_sym_weak_strong",
        family="hh",
        case_ids=("hh_L2_nph4_three_model_sym_weak_strong",),
        hh_tableiii_regime="weak_strong",
        n_ph_work=4,
        n_ph_ref=4,
        adapt_max_depth=50,
    ),
    HHTableIIIRepairTarget(
        target_id="hh_sym_strong_strong",
        family="hh",
        case_ids=("hh_L2_nph4_three_model_sym_strong_strong",),
        hh_tableiii_regime="strong_strong",
        n_ph_work=4,
        n_ph_ref=4,
        adapt_max_depth=60,
    ),
)
HH_TABLEIII_REPAIR_TARGET_IDS: tuple[str, ...] = tuple(target.target_id for target in HH_TABLEIII_REPAIR_TARGETS)

# Fail-closed spellings for methods explicitly out of scope for this calibration
# lane.  Some aliases are not current registry rows; they are included so record
# generators/checkers reject ambiguous requests before launch surfaces exist.
PAPER_I_COMPARATOR_SPSA_CALIBRATION_EXCLUDED_METHOD_REASONS: dict[str, str] = {
    PAPER_I_MAIN_TABLES_SPSA_SNAKE_ALGORITHM_ID: "SNAKE/Route-A is calibrated by the dedicated Phase3 lane",
    "route_a": "SNAKE/Route-A is calibrated by the dedicated Phase3 lane",
    "snake": "SNAKE/Route-A is calibrated by the dedicated Phase3 lane",
    "static_pos_geo_adapt_vqe": "PosGeo is diagnostic/appendix-only, not visible Geo-ADAPT",
    "static_qiskit_adapt_vqe": "Qiskit AdaptVQE is not a visible Paper-I comparator row",
    "static_uccsd_vqe": "UCCSD/lifted-UCCSD is not a visible Paper-I comparator row",
    "hh_uccsd_lifted_vqe": "UCCSD/lifted-UCCSD is not a visible Paper-I comparator row",
    "static_avqite_uccsd": "UCCSD/lifted-UCCSD is not a visible Paper-I comparator row",
    "static_qse": "QSE belongs to Paper III, not Paper-I comparator SPSA calibration",
    "static_qse_spectra": "QSE belongs to Paper III, not Paper-I comparator SPSA calibration",
    "qse": "QSE belongs to Paper III, not Paper-I comparator SPSA calibration",
    "geometry_selected_qse": "QSE belongs to Paper III, not Paper-I comparator SPSA calibration",
    "static_vqe": "extra plain VQE rows are out of scope for the visible comparator-only decision",
    "static_plain_vqe": "extra plain VQE rows are out of scope for the visible comparator-only decision",
    "plain_vqe": "extra plain VQE rows are out of scope for the visible comparator-only decision",
    "vqe": "extra plain VQE rows are out of scope for the visible comparator-only decision",
}

PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE = "native_spsa_adaptive_refit_rerun_v1"
PAPER_I_NATIVE_SPSA_REFIT_ENGINE = "src.quantum.spsa_optimizer:spsa_minimize"
_ADAPT_COMPARATOR_METHOD_IDS = frozenset(
    {
        "static_full_meta_append_adapt_vqe",
        "static_qubit_qeb_adapt_vqe",
        "static_geo_adapt_vqe",
    }
)
PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS: tuple[str, ...] = tuple(sorted(_ADAPT_COMPARATOR_METHOD_IDS))
PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD: dict[str, tuple[str, ...]] = {
    "static_hea_qiskit_vqe": PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS,
    "static_family_informed_vqe": PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS,
    **{
        method_id: PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS
        for method_id in sorted(_ADAPT_COMPARATOR_METHOD_IDS)
    },
}

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
_FLOAT_SCHEDULE_FIELDS = frozenset(
    field
    for fields in PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD.values()
    for field in fields
    if field not in _INT_SCHEDULE_FIELDS
)


def calibration_targets() -> tuple[ComparatorCalibrationTarget, ...]:
    """Return the six visible calibration targets in deterministic order."""

    return PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGETS


def allowed_method_ids() -> tuple[str, ...]:
    """Return the five retained visible non-SNAKE comparator method IDs."""

    return PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS


def target_by_id(target_id: str) -> ComparatorCalibrationTarget:
    key = str(target_id).strip()
    for target in PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGETS:
        if target.target_id == key:
            return target
    known = ", ".join(PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS)
    raise ValueError(f"Unknown Paper-I comparator SPSA calibration target_id={target_id!r}; known: {known}")


def hh_tableiii_repair_targets() -> tuple[HHTableIIIRepairTarget, ...]:
    """Return the visible Table III HH Geo/QEB repair targets in table order."""

    return HH_TABLEIII_REPAIR_TARGETS


def hh_tableiii_repair_target_by_id(target_id: str) -> HHTableIIIRepairTarget:
    key = str(target_id).strip()
    for target in HH_TABLEIII_REPAIR_TARGETS:
        if target.target_id == key:
            return target
    known = ", ".join(HH_TABLEIII_REPAIR_TARGET_IDS)
    raise ValueError(f"Unknown HH Table III repair target_id={target_id!r}; known: {known}")


def validate_method_id(method_id: str) -> str:
    key = str(method_id).strip()
    if key in PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS:
        return key
    if key in PAPER_I_COMPARATOR_SPSA_CALIBRATION_EXCLUDED_METHOD_REASONS:
        raise ValueError(
            f"Method {method_id!r} is explicitly excluded from "
            f"{PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID}: "
            f"{PAPER_I_COMPARATOR_SPSA_CALIBRATION_EXCLUDED_METHOD_REASONS[key]}."
        )
    known = ", ".join(PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS)
    raise ValueError(
        f"Method {method_id!r} is not one of the five retained visible non-SNAKE comparator methods for "
        f"{PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID}; allowed: {known}."
    )


def validate_target_cases() -> None:
    """Fail closed unless every target case belongs to the visible SPSA profile."""

    for target in PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGETS:
        if not target.case_ids:
            raise ValueError(f"Calibration target {target.target_id!r} has no case_ids.")
        for case_id in target.case_ids:
            if not paper_i_main_tables_spsa_contains_case(target.family, case_id):
                raise ValueError(
                    f"Calibration target {target.target_id!r} case {case_id!r} is not in "
                    f"{PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID}."
                )


def full_method_target_records(
    *,
    method_ids: Sequence[str] | None = None,
    target_ids: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Build deterministic method-target record stubs for the full 5x6 matrix."""

    validate_target_cases()
    methods = tuple(validate_method_id(method_id) for method_id in (method_ids or allowed_method_ids()))
    targets = tuple(target_by_id(target_id) for target_id in (target_ids or PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS))
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for method_id in methods:
        for target in targets:
            key = (method_id, target.target_id)
            if key in seen:
                raise ValueError(f"Duplicate method-target record {key!r}.")
            seen.add(key)
            records.append(
                {
                    "profile_id": PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
                    "method_id": method_id,
                    "target_id": target.target_id,
                    "family": target.family,
                    "case_ids": target.case_ids,
                    "case_ids_json": json.dumps(list(target.case_ids), separators=(",", ":")),
                }
            )
    return tuple(records)


def native_spsa_adaptive_method_target_records(
    *,
    method_ids: Sequence[str] | None = None,
    target_ids: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Build deterministic affected-adaptive rerun stubs for native SPSA refits."""

    methods = tuple(
        validate_method_id(method_id) for method_id in (method_ids or PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS)
    )
    unexpected = sorted(set(methods) - set(PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS))
    if unexpected:
        raise ValueError(
            "Native SPSA adaptive rerun scope only supports affected exact-bench adaptive methods; "
            f"got {unexpected!r}."
        )
    records = list(full_method_target_records(method_ids=methods, target_ids=target_ids))
    for record in records:
        record["rerun_scope"] = PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE
        record["adaptive_refit_engine"] = PAPER_I_NATIVE_SPSA_REFIT_ENGINE
    return tuple(records)


def hh_tableiii_repair_method_target_records(
    *,
    method_ids: Sequence[str] | None = None,
    target_ids: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Build deterministic Geo/QEB repair stubs for the visible Table III HH gap."""

    validate_target_cases()
    methods = tuple(validate_method_id(method_id) for method_id in (method_ids or HH_TABLEIII_REPAIR_METHOD_IDS))
    targets = tuple(hh_tableiii_repair_target_by_id(target_id) for target_id in (target_ids or HH_TABLEIII_REPAIR_TARGET_IDS))
    unknown_methods = sorted(set(methods) - set(HH_TABLEIII_REPAIR_METHOD_IDS))
    if unknown_methods:
        raise ValueError(f"HH Table III repair only supports {HH_TABLEIII_REPAIR_METHOD_IDS}; got {unknown_methods}.")
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for method_id in methods:
        for target in targets:
            key = (method_id, target.target_id)
            if key in seen:
                raise ValueError(f"Duplicate HH Table III repair method-target record {key!r}.")
            seen.add(key)
            records.append(
                {
                    "profile_id": PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
                    "repair_scope": HH_GEO_QEB_TABLEIII_REPAIR_SCOPE,
                    "table_label": HH_TABLEIII_REPAIR_TABLE_LABEL,
                    "method_id": method_id,
                    "target_id": target.target_id,
                    "hh_tableiii_regime": target.hh_tableiii_regime,
                    "family": target.family,
                    "case_ids": target.case_ids,
                    "case_ids_json": json.dumps(list(target.case_ids), separators=(",", ":")),
                    "n_ph_work": target.n_ph_work,
                    "n_ph_ref": target.n_ph_ref,
                    "phase3_adapt_max_depth": target.adapt_max_depth,
                    "primary_energy_metric": HH_TABLEIII_REPAIR_PRIMARY_ENERGY_METRIC,
                    "same_cutoff_error_role": HH_TABLEIII_REPAIR_SAME_CUTOFF_ERROR_ROLE,
                    "calibration_usable_status_policy": HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY,
                    "quality_nonpassing_penalty": HH_TABLEIII_REPAIR_QUALITY_NONPASSING_PENALTY,
                }
            )
    return tuple(records)


def _case_ids_from_record(record: Mapping[str, object], *, target: ComparatorCalibrationTarget) -> tuple[str, ...]:
    has_case_ids_json = "case_ids_json" in record and _text_or_empty(record.get("case_ids_json"))
    case_ids_value = record.get("case_ids")
    has_case_ids = "case_ids" in record and case_ids_value is not None and case_ids_value != ""
    parsed_json: tuple[str, ...] | None = None
    if has_case_ids_json:
        raw_json = str(record.get("case_ids_json"))
        try:
            payload = json.loads(raw_json)
        except Exception as exc:
            raise ValueError(f"Calibration record target={target.target_id!r} has malformed case_ids_json.") from exc
        if not isinstance(payload, list) or not payload or not all(isinstance(item, str) and item.strip() for item in payload):
            raise ValueError(f"Calibration record target={target.target_id!r} case_ids_json must be a non-empty JSON string array.")
        parsed_json = tuple(str(item).strip() for item in payload)
    parsed_plain: tuple[str, ...] | None = None
    if has_case_ids:
        case_ids_raw = record.get("case_ids")
        if isinstance(case_ids_raw, str):
            parsed_plain = (case_ids_raw,)
        else:
            try:
                parsed_plain = tuple(str(case_id) for case_id in case_ids_raw)  # type: ignore[union-attr]
            except TypeError as exc:
                raise ValueError(f"Calibration record target={target.target_id!r} case_ids must be a sequence.") from exc
        if not parsed_plain or any(not case_id.strip() for case_id in parsed_plain):
            raise ValueError(f"Calibration record target={target.target_id!r} case_ids must be non-empty strings.")
    if parsed_json is not None and parsed_plain is not None and parsed_json != parsed_plain:
        raise ValueError(
            f"Calibration record target={target.target_id!r} has mismatched case_ids_json={parsed_json!r} "
            f"and case_ids={parsed_plain!r}."
        )
    if parsed_json is not None:
        return parsed_json
    if parsed_plain is not None:
        return parsed_plain
    raise ValueError(f"Calibration record target={target.target_id!r} must include case_ids_json or case_ids.")


def validate_full_method_target_records(records: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Validate that ``records`` are exactly the full 30 method-target matrix."""

    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    expected = {
        (method_id, target_id)
        for method_id in PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS
        for target_id in PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS
    }
    for record in records:
        method_id = validate_method_id(str(record.get("method_id", record.get("algorithm_id", ""))))
        target = target_by_id(str(record.get("target_id", "")))
        profile_id = str(record.get("profile_id", ""))
        if profile_id != PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID:
            raise ValueError(
                f"Calibration record for method={method_id!r}, target={target.target_id!r} has wrong "
                f"profile_id={profile_id!r}; expected {PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID!r}."
            )
        case_ids = _case_ids_from_record(record, target=target)
        if case_ids != target.case_ids:
            raise ValueError(
                f"Calibration record method={method_id!r}, target={target.target_id!r} has case_ids={case_ids!r}; "
                f"expected {target.case_ids!r}."
            )
        key = (method_id, target.target_id)
        if key in seen:
            raise ValueError(f"Duplicate calibration method-target record {key!r}.")
        seen.add(key)
        normalized.append(dict(record))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"Expected exactly 30 full method-target records for {PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID}; "
            f"got {len(seen)} unique records, missing={missing}, extra={extra}."
        )
    return tuple(normalized)


def validate_native_spsa_adaptive_records(records: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Validate exactly the 3x6 affected-adaptive native SPSA rerun matrix."""

    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    expected = {
        (method_id, target_id)
        for method_id in PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS
        for target_id in PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS
    }
    for record in records:
        method_id = validate_method_id(str(record.get("method_id", record.get("algorithm_id", ""))))
        if method_id not in PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS:
            raise ValueError(f"Native SPSA adaptive rerun method_id={method_id!r} is not affected/in scope.")
        target = target_by_id(str(record.get("target_id", "")))
        profile_id = str(record.get("profile_id", ""))
        if profile_id != PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID:
            raise ValueError(
                f"Native SPSA adaptive rerun record for method={method_id!r}, target={target.target_id!r} has wrong "
                f"profile_id={profile_id!r}; expected {PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID!r}."
            )
        scope = _text_or_empty(record.get("rerun_scope"))
        if scope and scope != PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE:
            raise ValueError(
                f"Native SPSA adaptive rerun record has rerun_scope={scope!r}; "
                f"expected {PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE!r}."
            )
        engine = _text_or_empty(record.get("adaptive_refit_engine"))
        if engine and engine != PAPER_I_NATIVE_SPSA_REFIT_ENGINE:
            raise ValueError(
                f"Native SPSA adaptive rerun record has adaptive_refit_engine={engine!r}; "
                f"expected {PAPER_I_NATIVE_SPSA_REFIT_ENGINE!r}."
            )
        case_ids = _case_ids_from_record(record, target=target)
        if case_ids != target.case_ids:
            raise ValueError(
                f"Native SPSA adaptive rerun record method={method_id!r}, target={target.target_id!r} "
                f"has case_ids={case_ids!r}; expected {target.case_ids!r}."
            )
        key = (method_id, target.target_id)
        if key in seen:
            raise ValueError(f"Duplicate native SPSA adaptive rerun method-target record {key!r}.")
        seen.add(key)
        normalized.append(dict(record))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"Expected exactly 18 native SPSA adaptive rerun records; got {len(seen)} unique records, "
            f"missing={missing}, extra={extra}."
        )
    return tuple(normalized)


def validate_hh_tableiii_repair_records(records: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Validate that ``records`` are exactly the 2x4 HH Geo/QEB repair matrix."""

    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    expected = {(method_id, target_id) for method_id in HH_TABLEIII_REPAIR_METHOD_IDS for target_id in HH_TABLEIII_REPAIR_TARGET_IDS}
    for record in records:
        method_id = validate_method_id(str(record.get("method_id", record.get("algorithm_id", ""))))
        if method_id not in HH_TABLEIII_REPAIR_METHOD_IDS:
            raise ValueError(f"HH Table III repair method_id={method_id!r} is not in {HH_TABLEIII_REPAIR_METHOD_IDS!r}.")
        target = hh_tableiii_repair_target_by_id(str(record.get("target_id", "")))
        profile_id = str(record.get("profile_id", ""))
        if profile_id != PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID:
            raise ValueError(
                f"HH Table III repair record for method={method_id!r}, target={target.target_id!r} has wrong "
                f"profile_id={profile_id!r}; expected {PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID!r}."
            )
        scope = _text_or_empty(record.get("repair_scope"))
        if scope and scope != HH_GEO_QEB_TABLEIII_REPAIR_SCOPE:
            raise ValueError(f"HH Table III repair record has repair_scope={scope!r}; expected {HH_GEO_QEB_TABLEIII_REPAIR_SCOPE!r}.")
        if _text_or_empty(record.get("table_label")) and _text_or_empty(record.get("table_label")) != HH_TABLEIII_REPAIR_TABLE_LABEL:
            raise ValueError(f"HH Table III repair record has table_label={record.get('table_label')!r}.")
        case_ids = _case_ids_from_record(record, target=target)
        if case_ids != target.case_ids:
            raise ValueError(
                f"HH Table III repair record method={method_id!r}, target={target.target_id!r} has case_ids={case_ids!r}; "
                f"expected {target.case_ids!r}."
            )
        for field, expected_value in (
            ("hh_tableiii_regime", target.hh_tableiii_regime),
            ("n_ph_work", target.n_ph_work),
            ("n_ph_ref", target.n_ph_ref),
        ):
            raw = _text_or_empty(record.get(field))
            if raw and raw != str(expected_value):
                raise ValueError(
                    f"HH Table III repair record method={method_id!r}, target={target.target_id!r} "
                    f"has {field}={raw!r}; expected {expected_value!r}."
                )
        key = (method_id, target.target_id)
        if key in seen:
            raise ValueError(f"Duplicate HH Table III repair method-target record {key!r}.")
        seen.add(key)
        normalized.append(dict(record))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"Expected exactly 8 HH Table III Geo/QEB repair records; got {len(seen)} unique records, "
            f"missing={missing}, extra={extra}."
        )
    return tuple(normalized)


def config_sha256_for_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def config_sha256_for_path(path: str | Path) -> str:
    return config_sha256_for_bytes(Path(path).read_bytes())


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    contents = config_path.read_bytes()
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
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{field} must be an integer >= {min_value}; got {value!r}.")
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


def _text_or_empty(value: object) -> str:
    return "" if value is None else str(value).strip()


def _validate_config_header(data: Mapping[str, Any]) -> None:
    missing = sorted(_CONFIG_REQUIRED_FIELDS - set(data))
    if missing:
        raise ValueError(f"Calibration config missing required fields: {missing}.")
    if data.get("profile_id") != PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID:
        raise ValueError(
            f"profile_id must be {PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID!r}; got {data.get('profile_id')!r}."
        )
    if data.get("config_version") != PAPER_I_COMPARATOR_SPSA_CALIBRATION_CONFIG_VERSION:
        raise ValueError(
            f"config_version must be {PAPER_I_COMPARATOR_SPSA_CALIBRATION_CONFIG_VERSION!r}; "
            f"got {data.get('config_version')!r}."
        )
    mode = str(data.get("mode", "")).strip().lower()
    if mode not in {"smoke", "full"}:
        raise ValueError(f"mode must be 'smoke' or 'full'; got {data.get('mode')!r}.")
    if _text_or_empty(data.get("plan_path")) != PAPER_I_COMPARATOR_SPSA_CALIBRATION_PLAN_PATH:
        raise ValueError(
            f"plan_path must be {PAPER_I_COMPARATOR_SPSA_CALIBRATION_PLAN_PATH!r}; got {data.get('plan_path')!r}."
        )
    if mode == "full":
        if data.get("approved_for_full_generation") is not True:
            raise ValueError("full calibration config requires approved_for_full_generation=true.")
        if not _text_or_empty(data.get("approved_by")) or not _text_or_empty(data.get("approved_at")):
            raise ValueError("full calibration config requires approval metadata: approved_by and approved_at.")


def _validate_exact_method_keys(mapping: Mapping[str, Any], *, field: str) -> None:
    keys = set(str(key) for key in mapping)
    expected = set(PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS)
    if keys != expected:
        raise ValueError(
            f"{field} must have exactly the five retained visible comparator method keys; "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}."
        )
    for key in keys:
        validate_method_id(key)


def _validate_method_maxiter_budgets(data: Mapping[str, Any]) -> None:
    budgets = _as_mapping(data.get("method_maxiter_budgets"), field="method_maxiter_budgets")
    _validate_exact_method_keys(budgets, field="method_maxiter_budgets")
    for method_id, value in budgets.items():
        _positive_int(value, field=f"method_maxiter_budgets.{method_id}")


def _bound_pair(spec: Mapping[str, Any], *, field: str, positive: bool) -> tuple[float, float]:
    low_key = "low" if "low" in spec else "min"
    high_key = "high" if "high" in spec else "max"
    if low_key not in spec or high_key not in spec:
        raise ValueError(f"{field} must define low/high or min/max bounds.")
    low = _finite_float(spec[low_key], field=f"{field}.{low_key}", positive=positive)
    high = _finite_float(spec[high_key], field=f"{field}.{high_key}", positive=positive)
    if not high > low:
        raise ValueError(f"{field} high bound must be greater than low bound; got low={low}, high={high}.")
    return low, high


def _validate_search_space_entry(spec: Mapping[str, Any], *, field: str, param_name: str) -> None:
    kind = str(spec.get("type", spec.get("kind", ""))).strip().lower().replace("-", "_")
    if kind not in {value.replace("-", "_") for value in _SEARCH_SPACE_KINDS}:
        raise ValueError(f"{field} search-space type must be one of {sorted(_SEARCH_SPACE_KINDS)}; got {kind!r}.")
    expects_int = param_name in _INT_SCHEDULE_FIELDS
    if expects_int and kind not in {"int", "choice"}:
        raise ValueError(f"{field} must use int or choice search-space type for integer schedule field {param_name!r}.")
    if (not expects_int) and kind == "int":
        raise ValueError(f"{field} must not use int search-space type for float schedule field {param_name!r}.")
    if kind == "choice":
        choices = spec.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or len(choices) == 0:
            raise ValueError(f"{field}.choices must be a non-empty array.")
        for index, choice in enumerate(choices):
            choice_field = f"{field}.choices[{index}]"
            if expects_int:
                _positive_int(choice, field=choice_field)
            else:
                _finite_float(choice, field=choice_field, positive=True)
        return
    if kind == "int":
        low, high = _bound_pair(spec, field=field, positive=True)
        if not float(low).is_integer() or not float(high).is_integer():
            raise ValueError(f"{field} integer search-space bounds must be integral.")
        return
    _bound_pair(spec, field=field, positive=True)


def _validate_per_method_search_spaces(data: Mapping[str, Any]) -> None:
    spaces = _as_mapping(data.get("per_method_search_spaces"), field="per_method_search_spaces")
    _validate_exact_method_keys(spaces, field="per_method_search_spaces")
    for method_id, method_space_raw in spaces.items():
        method = validate_method_id(str(method_id))
        method_space = _as_mapping(method_space_raw, field=f"per_method_search_spaces.{method}")
        if not method_space:
            raise ValueError(f"per_method_search_spaces.{method} must not be empty.")
        allowed_fields = set(PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD[method])
        for param_name, spec_raw in method_space.items():
            param = str(param_name)
            if param not in allowed_fields:
                raise ValueError(
                    f"per_method_search_spaces.{method}.{param} is not an allowed SPSA schedule field for this method; "
                    f"allowed={sorted(allowed_fields)}."
                )
            spec = _as_mapping(spec_raw, field=f"per_method_search_spaces.{method}.{param}")
            _validate_search_space_entry(spec, field=f"per_method_search_spaces.{method}.{param}", param_name=param)


def _validate_objective_fields(data: Mapping[str, Any]) -> None:
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


def validate_calibration_config(
    data: Mapping[str, Any],
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a calibration JSON object and return a normalized shallow copy.

    Smoke configs may be checked in with ``approved_for_full_generation=false``.
    Full configs fail closed unless approval metadata is present.
    """

    _validate_config_header(data)
    _validate_objective_fields(data)
    _validate_method_maxiter_budgets(data)
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


# Validate import-time constants so record generators cannot import a drifted
# method/target contract silently.
validate_target_cases()
if len(PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS) != 5:
    raise ValueError("Paper-I comparator SPSA calibration must expose exactly five allowed methods.")
if len(PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGETS) != 6:
    raise ValueError("Paper-I comparator SPSA calibration must expose exactly six targets.")
if len(full_method_target_records()) != 30:
    raise ValueError("Paper-I comparator SPSA calibration full matrix must contain exactly 30 records.")
if len(hh_tableiii_repair_method_target_records()) != 8:
    raise ValueError("HH Table III Geo/QEB repair matrix must contain exactly 8 records.")
if len(native_spsa_adaptive_method_target_records()) != 18:
    raise ValueError("Native SPSA adaptive rerun matrix must contain exactly 18 records.")


__all__ = [
    "ComparatorCalibrationTarget",
    "HH_GEO_QEB_TABLEIII_REPAIR_RECORD_MODE",
    "HH_GEO_QEB_TABLEIII_REPAIR_SCOPE",
    "HH_GEO_QEB_TABLEIII_REPAIR_SMOKE_RECORD_MODE",
    "HH_TABLEIII_REPAIR_METHOD_IDS",
    "HH_TABLEIII_REPAIR_PRIMARY_ENERGY_METRIC",
    "HH_TABLEIII_REPAIR_QUALITY_NONPASSING_PENALTY",
    "HH_TABLEIII_REPAIR_SAME_CUTOFF_ERROR_ROLE",
    "HH_TABLEIII_REPAIR_TABLE_LABEL",
    "HH_TABLEIII_REPAIR_TARGET_IDS",
    "HH_TABLEIII_REPAIR_TARGETS",
    "HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY",
    "HHTableIIIRepairTarget",
    "PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD",
    "PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS",
    "PAPER_I_COMPARATOR_SPSA_CALIBRATION_CONFIG_VERSION",
    "PAPER_I_COMPARATOR_SPSA_CALIBRATION_EXCLUDED_METHOD_REASONS",
    "PAPER_I_COMPARATOR_SPSA_CALIBRATION_PLAN_PATH",
    "PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID",
    "PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS",
    "PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGETS",
    "PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE",
    "PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS",
    "PAPER_I_NATIVE_SPSA_REFIT_ENGINE",
    "allowed_method_ids",
    "calibration_targets",
    "config_sha256_for_bytes",
    "config_sha256_for_path",
    "full_method_target_records",
    "hh_tableiii_repair_method_target_records",
    "hh_tableiii_repair_target_by_id",
    "hh_tableiii_repair_targets",
    "load_and_validate_config",
    "load_config",
    "native_spsa_adaptive_method_target_records",
    "target_by_id",
    "validate_calibration_config",
    "validate_full_method_target_records",
    "validate_hh_tableiii_repair_records",
    "validate_method_id",
    "validate_native_spsa_adaptive_records",
    "validate_target_cases",
]
