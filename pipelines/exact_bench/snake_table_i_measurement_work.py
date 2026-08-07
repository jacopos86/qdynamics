#!/usr/bin/env python3
"""Normalize SNAKE Table-I support-artifact measurement-work metadata.

This module is reporting-only. It enriches the paper-facing SNAKE support JSON
with the same ``normalized_measurement_work_v1`` schema used by exact-bench
comparator rows, but it does not infer normalized components from raw scalar
shot proxies.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from pipelines.exact_bench.deterministic_shot_proxy import build_deterministic_shot_proxy_fields
from pipelines.exact_bench.generic_static_metric_enrichment import (
    ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
    GROUPED_MEASUREMENT_PROXY_SCHEMA,
    NORMALIZED_MEASUREMENT_WORK_SCHEMA,
    PHYSICAL_MEASUREMENT_WORK_SCHEMA,
    TABLE_I_EVENT_LEDGER_SCHEMA,
    _num,
    algorithmic_measurement_work_from_row,
    grouped_measurement_proxy_from_explicit_row,
    normalized_measurement_work_from_explicit_row,
    physical_measurement_work_from_row,
)

from pipelines.static_adapt.selector_measurement_proxy import controller_proxy_from_history_rows

SCHEMA_VERSION = "table_i_snake_measurement_work_normalization_v1"
SOURCE_MAP_SCHEMA_VERSION = "snake_table_i_source_payload_map_v1"
RAW_PROXY_PRIORITY = ("measurement_shots_proxy", "shot_proxy", "shot_cost_proxy", "shots_total")
_TOP_LEVEL_SUPPORT_KEYS = (
    "all_averaged_snake_current_table_support",
    "fermionic_snake_current_table_support",
)
_COMPONENT_KEYS = ("N_H_outer_eval", "N_grad", "N_metric", "N_H_refit_eval")
OPERATOR_PROBE_CHARGE_BASIS = "logical_estimator_request_pre_grouping_v1"
COMMON_EXPOSURE_STAGE = "post_common_eligibility_post_expansion_pre_method_filter"
COMMON_EXPOSURE_POLICY_ID = "trajectory_conditioned_full_child_common_exposure_v1"
FAIR_WORK_CURRENCY = "expanded_common_candidate_probe_event_count_v1"
BEAM_AGGREGATE_RUN_SCOPE = "all_expanded_scored_branches"
BEAM_WINNER_RUN_SCOPE = "winner_lineage_only"
BEAM_TERMINAL_ROW_POLICY = "beam_terminal_winner_history_v1"
BEAM_SEARCH_TOTAL_PROVENANCE_SCHEMA = "snake_beam_search_total_work_provenance_v1"
SNAKE_TERMINAL_WORK_SEMANTICS_VERSION = "snake_terminal_s_alg_winner_lineage_v1"
SNAKE_MECHANISM_RESOLVED_WORK_SCHEMA_VERSION = "paper_i_hh_snake_mechanism_resolved_work_v1"
_S_NORM_STRICT_COMPONENT_ALIASES = (
    "N_H_outer_eval",
    "N_H_eval",
    "S_norm_N_H_outer_eval",
    "S_norm_N_H_eval",
    "measurement_work_N_H_outer_eval",
    "measurement_work_N_H_eval",
    "N_grad",
    "S_norm_N_grad",
    "measurement_work_N_grad",
    "N_metric",
    "S_norm_N_metric",
    "measurement_work_N_metric",
    "N_H_refit_eval",
    "N_refit_eval",
    "S_norm_N_H_refit_eval",
    "S_norm_N_refit_eval",
    "measurement_work_N_H_refit_eval",
    "measurement_work_N_refit_eval",
)
_S_NORM_STRICT_OTHER_ALIASES = (
    "N_other_quantum",
    "S_norm_N_other_quantum",
    "measurement_work_N_other_quantum",
)


def _raw_proxy(row: Mapping[str, Any]) -> dict[str, float | None]:
    return {key: _num(row.get(key)) for key in RAW_PROXY_PRIORITY}


def _first_raw_proxy(raw_proxy: Mapping[str, float | None]) -> tuple[float | None, str | None]:
    for key in RAW_PROXY_PRIORITY:
        value = raw_proxy.get(key)
        if value is not None:
            return float(value), key
    return None, None


def _has_explicit_s_norm_component_fields(row: Mapping[str, Any]) -> bool:
    return any(key in row for key in (*_S_NORM_STRICT_COMPONENT_ALIASES, *_S_NORM_STRICT_OTHER_ALIASES))


def _strict_explicit_s_norm_rejection(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Reject malformed explicit S_norm aliases before any runtime fallback.

    The generic legacy helper treats nonfinite values as "missing". For SNAKE
    support promotion that is too weak: if a row explicitly emits invalid
    normalized-work fields, a valid source payload must not silently override
    them.
    """

    for key in _S_NORM_STRICT_COMPONENT_ALIASES:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        value = _num(raw)
        if value is None or not math.isfinite(float(value)):
            return "invalid_component_value", f"invalid_{key}"
        if float(value) < 0.0:
            return "invalid_component_value", f"negative_{key}"
    for key in _S_NORM_STRICT_OTHER_ALIASES:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        value = _num(raw)
        if value is None or not math.isfinite(float(value)):
            return "invalid_component_value", f"invalid_{key}"
        if float(value) < 0.0:
            return "invalid_component_value", f"negative_{key}"
        if float(value) > 0.0:
            return "unassigned_other_quantum_work", f"nonzero_{key}"
    return None, None


def _has_explicit_s_alg_component_fields(row: Mapping[str, Any]) -> bool:
    aliases = (
        "S_alg_N_H_outer_eval",
        "algorithmic_measurement_work_N_H_outer_eval",
        "S_alg_N_grad_probe",
        "algorithmic_measurement_work_N_grad_probe",
        "S_alg_N_metric_probe",
        "algorithmic_measurement_work_N_metric_probe",
        "S_alg_N_H_refit_eval",
        "algorithmic_measurement_work_N_H_refit_eval",
        "S_alg_N_other_quantum",
        "algorithmic_measurement_work_N_other_quantum",
        "N_other_algorithmic_quantum",
        "table_i_measurement_event_ledger",
        "measurement_event_ledger",
        "measurement_events",
    )
    return any(key in row for key in aliases)


def _has_explicit_physical_component_fields(row: Mapping[str, Any]) -> bool:
    aliases = (
        "S_phys_H_outer",
        "physical_measurement_work_S_H_outer",
        "S_phys_grad",
        "physical_measurement_work_S_grad",
        "S_phys_metric",
        "physical_measurement_work_S_metric",
        "S_phys_H_refit",
        "physical_measurement_work_S_H_refit",
        "S_l2_H_outer",
        "grouped_l2_measurement_work_S_H_outer",
        "S_l2_grad",
        "grouped_l2_measurement_work_S_grad",
        "S_l2_metric",
        "grouped_l2_measurement_work_S_metric",
        "S_l2_H_refit",
        "grouped_l2_measurement_work_S_H_refit",
    )
    return any(key in row for key in aliases)


def _has_explicit_s_grp_component_fields(row: Mapping[str, Any]) -> bool:
    aliases = (
        "S_grp_H_outer",
        "S_grp_H_outer_eval",
        "grouped_measurement_S_H_outer",
        "S_grp_grad",
        "grouped_measurement_S_grad",
        "S_grp_metric",
        "grouped_measurement_S_metric",
        "S_grp_H_refit",
        "S_grp_H_refit_eval",
        "grouped_measurement_S_H_refit",
    )
    return any(key in row for key in aliases)


def _path_key(path: Sequence[str]) -> str:
    return ".".join(str(part) for part in path)


def _finite_nonnegative(value: Any) -> float | None:
    parsed = _num(value)
    if parsed is None or not math.isfinite(float(parsed)) or float(parsed) < 0.0:
        return None
    return float(parsed)


def _adapt_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("adapt_vqe")
    if isinstance(nested, Mapping):
        return nested
    return payload


def _controller_summary(adapt_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = adapt_payload.get("controller_measurement_work_summary")
    if isinstance(direct, Mapping):
        return direct
    continuation = adapt_payload.get("continuation")
    if isinstance(continuation, Mapping):
        nested = continuation.get("controller_measurement_work_summary")
        if isinstance(nested, Mapping):
            return nested
    return None


def _controller_phase_map(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    by_phase = summary.get("by_phase")
    if isinstance(by_phase, Mapping):
        return by_phase
    per_phase = summary.get("per_phase")
    if isinstance(per_phase, Mapping):
        return per_phase
    return None


def _controller_numeric_validation_blocker(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen: set[int] = set()

    def collect(location: str, node: Mapping[str, Any]) -> None:
        ident = id(node)
        if ident in seen:
            return
        seen.add(ident)
        validation = node.get("controller_numeric_validation")
        if not isinstance(validation, Mapping):
            validation = node.get("numeric_validation")
        if isinstance(validation, Mapping):
            for source_key, target in (
                ("missing_required_fields", missing),
                ("invalid_fields", invalid),
            ):
                entries = validation.get(source_key)
                if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
                    for entry in entries:
                        if not isinstance(entry, Mapping) or not bool(entry.get("paper_i_blocking", False)):
                            continue
                        item = dict(entry)
                        item.setdefault("location", location)
                        target.append(item)
        for nested_key in ("by_phase", "per_phase", "by_scope"):
            nested = node.get(nested_key)
            if not isinstance(nested, Mapping):
                continue
            for key, value in nested.items():
                if isinstance(value, Mapping):
                    collect(f"{location}.{nested_key}.{key}", value)

    collect("controller_measurement_work_summary", summary)
    if not missing and not invalid:
        return None
    status = "invalid_controller_numeric_fields" if invalid else "missing_controller_numeric_fields"
    return {
        "status": status,
        "controller_numeric_validation": {
            "schema": "controller_measurement_work_numeric_validation_v1",
            "status": "invalid",
            "missing_required_fields": missing,
            "invalid_fields": invalid,
        },
    }


def _controller_candidate_work_ledger_audit(summary: Mapping[str, Any]) -> dict[str, Any]:
    status = str(summary.get("candidate_work_ledger_status") or "")
    schema = str(summary.get("candidate_work_ledger_schema") or "")
    event_count = _finite_nonnegative_integer(summary.get("candidate_work_event_count"))
    missing_event_count = _finite_nonnegative_integer(summary.get("candidate_work_missing_event_count"))
    candidate_count_total = _finite_nonnegative_integer(summary.get("candidate_count_total"))
    if status == "explicit_candidate_work_ledger_v1" and schema == "controller_candidate_work_ledger_v1":
        if missing_event_count in {None, 0} and (event_count is None or event_count > 0):
            return {
                "schema": "snake_candidate_work_ledger_audit_v1",
                "status": "ok",
                "candidate_work_ledger_status": status,
                "candidate_work_ledger_schema": schema,
                "candidate_work_event_count": event_count,
                "candidate_work_missing_event_count": missing_event_count,
                "candidate_count_total": candidate_count_total,
                "evaluated_count_total": _finite_nonnegative_integer(summary.get("evaluated_count_total")),
                "pre_shortlist_count_total": _finite_nonnegative_integer(summary.get("pre_shortlist_count_total")),
                "shortlist_size_total": _finite_nonnegative_integer(summary.get("shortlist_size_total")),
                "retained_count_total": _finite_nonnegative_integer(summary.get("retained_count_total")),
                "rejected_count_total": _finite_nonnegative_integer(summary.get("rejected_count_total")),
                "candidate_work_ledger_scope": summary.get("candidate_work_ledger_scope"),
                "candidate_work_ledger_scopes": copy.deepcopy(summary.get("candidate_work_ledger_scopes")),
            }
    return {
        "schema": "snake_candidate_work_ledger_audit_v1",
        "status": "missing_explicit_candidate_work_ledger",
        "candidate_work_ledger_status": status or None,
        "candidate_work_ledger_schema": schema or None,
        "candidate_work_event_count": event_count,
        "candidate_work_missing_event_count": missing_event_count,
        "candidate_count_total": candidate_count_total,
        "reason": "runtime_reconstruction_does_not_prove_comparable_candidate_screening_work",
    }


def _phase_records_with_group_keys(
    phase_map: Mapping[str, Any],
    phase: str,
) -> tuple[float | None, dict[str, Any]]:
    entry = phase_map.get(phase)
    if entry is None:
        return 0.0, {"phase": phase, "status": "absent_zero"}
    if not isinstance(entry, Mapping):
        return None, {"phase": phase, "status": "invalid_phase_payload"}
    value = _finite_nonnegative(entry.get("records_with_group_keys"))
    if value is None:
        return None, {"phase": phase, "status": "invalid_records_with_group_keys"}
    return value, {"phase": phase, "status": "ok", "records_with_group_keys": float(value)}


def _phase_has_positive_diagnostic_work(entry: Mapping[str, Any]) -> bool:
    for key in (
        "records_evaluated",
        "records_with_group_keys",
        "groups_total",
        "group_key_count",
        "expanded_measurement_group_probe_count",
        "expanded_measurement_group_probe_count_total",
        "shots_total",
        "shots_new",
        "total_shots_new",
    ):
        value = _finite_nonnegative(entry.get(key))
        if value is not None and value > 0.0:
            return True
    return False


def _phase_actual_operator_probe_count(
    phase_map: Mapping[str, Any],
    phase: str,
) -> tuple[float | None, dict[str, Any]]:
    """Return typed actual operator-probe work for a controller phase.

    Paper-I comparable ``S_alg`` is a pre-grouping estimator/probe event
    currency. Measurement-basis cache fields such as ``records_with_group_keys``
    are diagnostics only and never substitute for this typed count.
    """

    entry = phase_map.get(phase)
    if entry is None:
        return 0.0, {"phase": phase, "status": "absent_zero"}
    if not isinstance(entry, Mapping):
        return None, {"phase": phase, "status": "invalid_phase_payload"}
    for key in ("actual_operator_probe_count", "actual_operator_probe_count_total"):
        if key not in entry:
            continue
        value = _finite_nonnegative_integer(entry.get(key))
        if value is None:
            return None, {"phase": phase, "status": "nonintegral_or_negative_count", "source": key}
        charge_basis = str(entry.get("operator_probe_charge_basis") or "")
        if charge_basis != OPERATOR_PROBE_CHARGE_BASIS:
            return None, {
                "phase": phase,
                "status": "policy_mismatch",
                "source": key,
                "operator_probe_charge_basis": charge_basis or None,
                "expected_operator_probe_charge_basis": OPERATOR_PROBE_CHARGE_BASIS,
            }
        return float(value), {
            "phase": phase,
            "status": "ok",
            "operator_probe_count": int(value),
            "source": key,
            "operator_probe_charge_basis": OPERATOR_PROBE_CHARGE_BASIS,
        }
    if not _phase_has_positive_diagnostic_work(entry):
        return 0.0, {
            "phase": phase,
            "status": "explicit_zero_without_typed_operator_probe_count",
            "source": "diagnostic_zero_fields",
        }
    legacy_value, legacy_detail = _phase_records_with_group_keys(phase_map, phase)
    detail = {
        "phase": phase,
        "status": "missing_actual_operator_probe_count",
        "source": "actual_operator_probe_count",
        "reason": "records_with_group_keys_is_cache_diagnostic_not_comparable_S_alg",
        "records_with_group_keys_detail": legacy_detail,
    }
    if legacy_value is not None:
        detail["records_with_group_keys_diagnostic"] = float(legacy_value)
    return None, detail


def _phase_common_exposure_operator_probe_count(
    phase_map: Mapping[str, Any],
    phase: str,
) -> tuple[float | None, dict[str, Any]]:
    """Return typed full-common-exposure operator probes for a controller phase.

    Measurement-group counts, shortlist sizes, and generic candidate counts are
    diagnostics here.  The fair SNAKE column is a pre-grouping operator-probe
    currency, so the canonical path accepts only explicit common-exposure
    operator-probe fields with stage/policy provenance.
    """

    entry = phase_map.get(phase)
    if entry is None:
        return 0.0, {"phase": phase, "status": "absent_zero"}
    if not isinstance(entry, Mapping):
        return None, {"phase": phase, "status": "invalid_phase_payload"}
    aliases = (
        "common_exposure_operator_probe_count",
        "common_exposure_operator_probe_count_total",
        "common_operator_probe_count",
        "common_operator_probe_count_total",
    )
    source_key: str | None = None
    parsed_count: int | None = None
    for key in aliases:
        if key not in entry:
            continue
        value = _finite_nonnegative_integer(entry.get(key))
        if value is None:
            return None, {"phase": phase, "status": "nonintegral_or_negative_count", "source": key}
        source_key = key
        parsed_count = int(value)
        break
    if parsed_count is None:
        forbidden_present = [
            key
            for key in (
                "expanded_algorithmic_probe_count",
                "expanded_algorithmic_probe_count_total",
                "expanded_candidate_probe_count",
                "expanded_candidate_probe_count_total",
                "expanded_measurement_group_probe_count",
                "expanded_measurement_group_probe_count_total",
                "algorithmic_group_probe_count",
                "algorithmic_group_probe_count_total",
                "groups_total",
                "group_key_count",
                "group_key_count_total",
                "candidate_count_total",
                "pre_shortlist_count_total",
                "shortlist_size_total",
                "retained_count_total",
                "rejected_count_total",
            )
            if key in entry and entry.get(key) not in {None, ""}
        ]
        detail = {"phase": phase, "status": "missing_common_exposure_ledger"}
        if forbidden_present:
            detail["forbidden_operator_probe_aliases_present"] = forbidden_present
            detail["forbidden_alias_reason"] = "legacy_group_or_candidate_fields_are_not_operator_probe_counts"
        grouped = _finite_nonnegative(entry.get("records_with_group_keys"))
        if grouped is not None:
            detail["records_with_group_keys_diagnostic"] = float(grouped)
        return None, detail
    charge_basis = str(entry.get("operator_probe_charge_basis") or "")
    if charge_basis != OPERATOR_PROBE_CHARGE_BASIS:
        return None, {
            "phase": phase,
            "status": "policy_mismatch",
            "source": source_key,
            "operator_probe_charge_basis": charge_basis or None,
            "expected_operator_probe_charge_basis": OPERATOR_PROBE_CHARGE_BASIS,
        }
    stage = str(entry.get("common_exposure_stage") or "")
    if stage != COMMON_EXPOSURE_STAGE:
        return None, {
            "phase": phase,
            "status": "policy_mismatch",
            "source": source_key,
            "common_exposure_stage": stage or None,
            "expected_common_exposure_stage": COMMON_EXPOSURE_STAGE,
        }
    policy_id = str(entry.get("common_exposure_policy_id") or "")
    if policy_id != COMMON_EXPOSURE_POLICY_ID:
        return None, {
            "phase": phase,
            "status": "policy_mismatch",
            "source": source_key,
            "common_exposure_policy_id": policy_id or None,
            "expected_common_exposure_policy_id": COMMON_EXPOSURE_POLICY_ID,
        }
    missing_policy_fields = [
        key
        for key in (
            "expansion_policy_id",
            "eligibility_policy_id",
            "deduplication_policy_id",
            "probe_enumerator_id",
        )
        if not str(entry.get(key) or "")
    ]
    if missing_policy_fields:
        return None, {
            "phase": phase,
            "status": "policy_mismatch",
            "source": source_key,
            "missing_policy_fields": missing_policy_fields,
        }
    return float(parsed_count), {
        "phase": phase,
        "status": "ok",
        "operator_probe_count": int(parsed_count),
        "source": source_key,
        "operator_probe_charge_basis": OPERATOR_PROBE_CHARGE_BASIS,
        "common_exposure_stage": COMMON_EXPOSURE_STAGE,
        "common_exposure_policy_id": COMMON_EXPOSURE_POLICY_ID,
        "expansion_policy_id": str(entry.get("expansion_policy_id")),
        "eligibility_policy_id": str(entry.get("eligibility_policy_id")),
        "deduplication_policy_id": str(entry.get("deduplication_policy_id")),
        "probe_enumerator_id": str(entry.get("probe_enumerator_id")),
    }


def _positive_unassigned_controller_phase_work(phase_map: Mapping[str, Any]) -> dict[str, float]:
    allowed = {"phase0", "phase_0", "phase1", "phase2", "phase3"}
    out: dict[str, float] = {}
    for phase, entry in phase_map.items():
        if str(phase) in allowed:
            continue
        if not isinstance(entry, Mapping):
            continue
        value = _finite_nonnegative(entry.get("groups_total"))
        if value is None:
            value = _finite_nonnegative(entry.get("records_with_group_keys"))
        if value is not None and value > 0.0:
            out[str(phase)] = float(value)
    return out


def _as_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _finite_nonnegative_integer(value: Any) -> int | None:
    parsed = _finite_nonnegative(value)
    if parsed is None:
        return None
    rounded = int(round(float(parsed)))
    if abs(float(parsed) - float(rounded)) > 1e-9:
        return None
    return rounded


def _trusted_integer_from_mapping(value: Any) -> tuple[int | None, str | None]:
    parsed = _finite_nonnegative_integer(value)
    if parsed is not None:
        return int(parsed), None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value), "sequence_length"
    if isinstance(value, Mapping):
        return len(value), "mapping_length"
    return None, None


def _trusted_payload_integer_at_paths(
    source_payload: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> tuple[int | None, str | None, str | None]:
    for path in paths:
        current: Any = source_payload
        missing = False
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                missing = True
                break
            current = current.get(part)
        if missing or current is None or current == "":
            continue
        value, source_kind = _trusted_integer_from_mapping(current)
        if value is None:
            return None, _path_key(path), "invalid"
        return int(value), _path_key(path), source_kind or "explicit_integer"
    return None, None, None


def _snake_legacy_work_proxies_from_payload(source_payload: Mapping[str, Any]) -> dict[str, Any]:
    adapt = _adapt_payload(source_payload)
    out: dict[str, Any] = {
        "schema": "paper_i_legacy_work_proxies_v1",
        "display_policy": "diagnostic_only_not_cross_method_S_alg_or_shots_total",
    }
    for scope_name, scope in (("root", source_payload), ("adapt_vqe", adapt)):
        if not isinstance(scope, Mapping):
            continue
        for key in (
            "controller_shot_proxy",
            "measurement_shots_proxy",
            "shot_cost_proxy",
            "S_norm",
            "S_norm_provenance",
            "measurement_work_proxy",
        ):
            value = scope.get(key)
            if value is None or value == "":
                continue
            out[f"{scope_name}.{key}"] = copy.deepcopy(value)
    return out


def _mapping_at_path(root: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any] | None:
    current: Any = root
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current if isinstance(current, Mapping) else None


def _selected_logical_filter_meta_from_payload(
    source_payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    paths = (
        ("adapt_vqe", "adapt_selected_logical_filter"),
        ("adapt_vqe", "continuation", "selected_logical_pool_filter"),
        ("continuation", "selected_logical_pool_filter"),
        ("adapt_selected_logical_filter",),
        ("selected_logical_pool_filter",),
    )
    for path in paths:
        meta = _mapping_at_path(source_payload, path)
        if meta is not None:
            return meta, _path_key(path)
    return None, None


def _explicit_phase0_controller_count(phase_map: Mapping[str, Any]) -> tuple[float | None, dict[str, Any]]:
    present: list[tuple[str, float]] = []
    details: dict[str, Any] = {}
    for phase in ("phase0", "phase_0"):
        if phase not in phase_map:
            details[phase] = {"phase": phase, "status": "absent_zero"}
            continue
        value, detail = _phase_actual_operator_probe_count(phase_map, phase)
        details[phase] = detail
        if value is None:
            return None, {"status": "invalid_explicit_controller_phase0", "phases": details}
        if float(value) > 0.0:
            present.append((phase, float(value)))
    if len(present) > 1:
        return None, {
            "status": "ambiguous_duplicate_explicit_controller_phase0",
            "positive_phase0_aliases": {phase: value for phase, value in present},
            "phases": details,
        }
    if present:
        phase, value = present[0]
        return float(value), {
            "status": "explicit_controller_phase0",
            "phase": phase,
            "operator_probe_count": float(value),
            "phases": details,
        }
    return 0.0, {"status": "absent_zero", "phases": details}


def _phase_shots_new(phase_map: Mapping[str, Any], phase: str) -> tuple[float | None, dict[str, Any]]:
    entry = phase_map.get(phase)
    if entry is None:
        return 0.0, {"phase": phase, "status": "absent_zero"}
    if not isinstance(entry, Mapping):
        return None, {"phase": phase, "status": "invalid_phase_payload"}
    for key in ("total_shots_new", "shots_new", "records_with_group_keys"):
        value = _finite_nonnegative(entry.get(key))
        if value is not None:
            return float(value), {"phase": phase, "status": "ok", "shots_new": float(value), "source": key}
    return None, {"phase": phase, "status": "invalid_shots_new"}


def _selected_logical_phase0_gradient_work(
    source_payload: Mapping[str, Any],
    phase_map: Mapping[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    explicit_count, explicit_meta = _explicit_phase0_controller_count(phase_map)
    meta, source = _selected_logical_filter_meta_from_payload(source_payload)
    base: dict[str, Any] = {
        "schema": "selected_logical_phase0_gradient_accounting_v1",
        "source": source,
        "explicit_phase0_present": bool(explicit_count is not None and explicit_count > 0.0),
        "explicit_phase0": explicit_meta,
    }
    if explicit_count is None:
        base.update(status=str(explicit_meta.get("status", "invalid_explicit_controller_phase0")))
        return None, base
    if explicit_count > 0.0:
        if isinstance(meta, Mapping):
            base.update(
                applied=_as_optional_bool(meta.get("applied")),
                fallback_to_full_pool=_as_optional_bool(meta.get("fallback_to_full_pool")),
                pool_size_before=meta.get("pool_size_before"),
                pool_size_after=meta.get("pool_size_after"),
                metadata_not_added_due_to_explicit_phase0=True,
            )
        base.update(
            status="explicit_controller_phase0",
            gradient_probe_count=float(explicit_count),
        )
        return float(explicit_count), base
    if not isinstance(meta, Mapping):
        base.update(status="not_applicable", gradient_probe_count=0.0)
        return 0.0, base

    applied = _as_optional_bool(meta.get("applied"))
    fallback_to_full_pool = _as_optional_bool(meta.get("fallback_to_full_pool"))
    pool_size_before = _finite_nonnegative_integer(meta.get("pool_size_before"))
    raw_pool_size_after = meta.get("pool_size_after")
    pool_size_after = None if raw_pool_size_after is None else _finite_nonnegative_integer(raw_pool_size_after)
    base.update(
        applied=applied,
        fallback_to_full_pool=fallback_to_full_pool,
        pool_size_before=meta.get("pool_size_before"),
        pool_size_after=raw_pool_size_after,
        metadata_not_added_due_to_explicit_phase0=False,
    )
    if applied is not True:
        base.update(status="not_applicable", gradient_probe_count=0.0)
        return 0.0, base
    if fallback_to_full_pool is True:
        base.update(status="fallback_to_full_pool_no_inference", gradient_probe_count=0.0)
        return 0.0, base
    if fallback_to_full_pool is None:
        base.update(status="invalid_selected_logical_filter", reason="missing_or_invalid_fallback_to_full_pool")
        return None, base
    if pool_size_before is None:
        base.update(status="invalid_selected_logical_filter", reason="missing_or_invalid_pool_size_before")
        return None, base
    if pool_size_after is None and raw_pool_size_after is not None:
        base.update(status="invalid_selected_logical_filter", reason="invalid_pool_size_after")
        return None, base
    if pool_size_after is not None and pool_size_after > pool_size_before:
        base.update(status="invalid_selected_logical_filter", reason="pool_size_after_exceeds_pool_size_before")
        return None, base
    base.update(
        status="inferred_from_selected_logical_filter",
        gradient_probe_count=float(pool_size_before),
        pool_size_before=int(pool_size_before),
        pool_size_after=None if pool_size_after is None else int(pool_size_after),
        source_field=f"{source}.pool_size_before" if source else "selected_logical_filter.pool_size_before",
    )
    return float(pool_size_before), base


def _history_refit_nfev(history: Any) -> tuple[float | None, dict[str, Any]]:
    if not isinstance(history, list):
        return None, {"status": "missing_history"}
    total = 0.0
    missing_indices: list[int] = []
    for idx, row in enumerate(history):
        if not isinstance(row, Mapping):
            missing_indices.append(int(idx))
            continue
        value = _finite_nonnegative(row.get("nfev_opt"))
        if value is None:
            value = _finite_nonnegative(row.get("optimizer_nfev"))
        if value is None:
            missing_indices.append(int(idx))
            continue
        total += float(value)
    if missing_indices:
        return None, {"status": "missing_history_nfev", "indices": missing_indices[:20]}
    return float(total), {"status": "ok", "history_count": int(len(history)), "nfev": float(total)}


def _optional_refit_nfev(adapt_payload: Mapping[str, Any], key: str) -> tuple[float | None, dict[str, Any]]:
    info = adapt_payload.get(key)
    if info is None:
        return None, {"status": "missing_refit_payload", "field": key}
    if not isinstance(info, Mapping):
        return None, {"status": "invalid_refit_payload", "field": key}
    value = _finite_nonnegative(info.get("nfev"))
    if value is not None:
        return float(value), {"status": "ok", "field": key, "nfev": float(value)}
    # Some runs include the refit block only to report that it did not execute.
    executed = info.get("executed")
    attempted = info.get("attempted")
    if executed is False or attempted is False:
        return 0.0, {"status": "explicit_not_executed_zero", "field": key}
    return None, {"status": "missing_refit_nfev", "field": key}


def _runtime_s_norm_components_from_payload(
    source_payload: Mapping[str, Any],
    *,
    source_label: str | None = None,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Reconstruct SNAKE S_norm bins from detailed runtime payload telemetry.

    This is deliberately conservative. It accepts only native controller phase
    summaries and explicit Hamiltonian-objective nfev metadata. Scalar shot/work
    proxies are not used.
    """

    adapt = _adapt_payload(source_payload)
    summary = _controller_summary(adapt)
    meta: dict[str, Any] = {
        "status": "unknown",
        "source_label": source_label,
        "schema": "snake_runtime_s_norm_reconstruction_v1",
        "component_mapping": {
            "N_grad": [
                "selected_logical_phase0_accounting.gradient_probe_count",
                "controller_measurement_work_summary.by_phase.phase1.actual_operator_probe_count",
            ],
            "N_grad_common_exposure": [
                "controller_measurement_work_summary.by_phase.phase0.common_exposure_operator_probe_count",
                "controller_measurement_work_summary.by_phase.phase1.common_exposure_operator_probe_count",
            ],
            "N_metric": [
                "controller_measurement_work_summary.by_phase.phase2.actual_operator_probe_count",
                "controller_measurement_work_summary.by_phase.phase3.actual_operator_probe_count",
            ],
            "N_metric_common_exposure": [
                "controller_measurement_work_summary.by_phase.phase2.common_exposure_operator_probe_count",
                "controller_measurement_work_summary.by_phase.phase3.common_exposure_operator_probe_count",
            ],
            "N_H_refit_eval": [
                "sum(history[*].nfev_opt)",
                "resume_boundary_refit.nfev",
                "final_full_refit.nfev",
            ],
            "N_H_outer_eval": ["nfev_total - N_H_refit_eval"],
        },
    }
    if not isinstance(summary, Mapping):
        meta.update(status="missing_controller_measurement_work_summary")
        return None, meta
    if summary.get("legacy_fallback_used") is not False:
        meta.update(status="legacy_controller_summary_not_promotable")
        return None, meta
    source_kind = str(summary.get("source_kind", "") or "")
    source = str(summary.get("source", "") or "")
    allowed_sources = {"native_controller_live_decision_work_v1", "native_controller_work"}
    if source_kind != "native_controller_work" and source not in allowed_sources:
        meta.update(
            status="missing_native_controller_provenance",
            source_kind=source_kind or None,
            source=source or None,
        )
        return None, meta
    if source_kind and source_kind != "native_controller_work":
        meta.update(status="non_native_controller_summary_not_promotable", source_kind=source_kind)
        return None, meta
    if source and source not in allowed_sources:
        meta.update(status="non_native_controller_source_not_promotable", source=source)
        return None, meta
    numeric_blocker = _controller_numeric_validation_blocker(summary)
    if numeric_blocker is not None:
        meta.update(numeric_blocker)
        return None, meta
    candidate_work_ledger = _controller_candidate_work_ledger_audit(summary)
    phase_map = _controller_phase_map(summary)
    if not isinstance(phase_map, Mapping):
        meta.update(status="missing_controller_phase_breakdown")
        return None, meta

    unassigned = _positive_unassigned_controller_phase_work(phase_map)
    if unassigned:
        meta.update(status="unassigned_controller_phase_work", unassigned_phases=unassigned)
        return None, meta

    phase_details: dict[str, Any] = {}
    records_with_group_key_details: dict[str, Any] = {}
    records_with_group_key_values: dict[str, float | None] = {}
    for phase_name in ("phase0", "phase1", "phase2", "phase3"):
        value, detail = _phase_records_with_group_keys(phase_map, phase_name)
        records_with_group_key_details[phase_name] = detail
        records_with_group_key_values[phase_name] = None if value is None else float(value)
    phase0, phase0_accounting = _selected_logical_phase0_gradient_work(source_payload, phase_map)
    phase_details["phase0"] = phase0_accounting
    phase1, detail = _phase_actual_operator_probe_count(phase_map, "phase1")
    phase_details["phase1"] = detail
    phase2, detail = _phase_actual_operator_probe_count(phase_map, "phase2")
    phase_details["phase2"] = detail
    phase3, detail = _phase_actual_operator_probe_count(phase_map, "phase3")
    phase_details["phase3"] = detail
    common_phase_details: dict[str, Any] = {}
    common_phase0, common_detail = _phase_common_exposure_operator_probe_count(phase_map, "phase0")
    common_phase_details["phase0"] = common_detail
    if common_phase0 == 0.0 and "phase_0" in phase_map:
        common_phase0, common_detail = _phase_common_exposure_operator_probe_count(phase_map, "phase_0")
        common_phase_details["phase0"] = common_detail
    common_phase1, common_detail = _phase_common_exposure_operator_probe_count(phase_map, "phase1")
    common_phase_details["phase1"] = common_detail
    common_phase2, common_detail = _phase_common_exposure_operator_probe_count(phase_map, "phase2")
    common_phase_details["phase2"] = common_detail
    common_phase3, common_detail = _phase_common_exposure_operator_probe_count(phase_map, "phase3")
    common_phase_details["phase3"] = common_detail
    if phase0 is None:
        meta.update(status="invalid_selected_logical_phase0_accounting", phases=phase_details)
        return None, meta
    if phase1 is None or phase2 is None or phase3 is None:
        meta.update(status="invalid_controller_phase_counts", phases=phase_details)
        return None, meta

    history_refit, history_meta = _history_refit_nfev(adapt.get("history"))
    resume_refit, resume_meta = _optional_refit_nfev(adapt, "resume_boundary_refit")
    final_refit, final_meta = _optional_refit_nfev(adapt, "final_full_refit")
    refit_sources = {
        "history": history_meta,
        "resume_boundary_refit": resume_meta,
        "final_full_refit": final_meta,
    }
    if history_refit is None or resume_refit is None or final_refit is None:
        meta.update(status="invalid_refit_nfev_metadata", refit_sources=refit_sources)
        return None, meta
    nfev_total = _finite_nonnegative(adapt.get("nfev_total"))
    if nfev_total is None:
        meta.update(status="missing_or_invalid_nfev_total", refit_sources=refit_sources)
        return None, meta

    n_h_refit = float(history_refit + resume_refit + final_refit)
    n_h_outer = float(nfev_total - n_h_refit)
    if n_h_outer < -1e-9:
        meta.update(
            status="inconsistent_nfev_partition",
            nfev_total=float(nfev_total),
            N_H_refit_eval=float(n_h_refit),
            refit_sources=refit_sources,
        )
        return None, meta
    n_h_outer = max(0.0, n_h_outer)

    components = {
        "N_H_outer_eval": float(n_h_outer),
        "N_grad": float(phase0 + phase1),
        "N_metric": float(phase2 + phase3),
        "N_H_refit_eval": float(n_h_refit),
    }
    common_status = "ok"
    common_components: dict[str, float] | None = None
    common_missing_phases = [
        phase
        for phase, value in (
            ("phase0", common_phase0),
            ("phase1", common_phase1),
            ("phase2", common_phase2),
            ("phase3", common_phase3),
        )
        if value is None
    ]
    if (
        phase0_accounting.get("status") == "inferred_from_selected_logical_filter"
        and common_phase_details.get("phase0", {}).get("status") == "absent_zero"
    ):
        common_status = "missing_common_exposure_ledger"
    elif common_missing_phases:
        policy_mismatch = next(
            (
                str(common_phase_details.get(phase, {}).get("status"))
                for phase in common_missing_phases
                if str(common_phase_details.get(phase, {}).get("status") or "") == "policy_mismatch"
            ),
            None,
        )
        invalid_count = next(
            (
                str(common_phase_details.get(phase, {}).get("status"))
                for phase in common_missing_phases
                if str(common_phase_details.get(phase, {}).get("status") or "") == "nonintegral_or_negative_count"
            ),
            None,
        )
        common_status = policy_mismatch or invalid_count or "missing_common_exposure_ledger"
    else:
        common_components = {
            "N_H_outer_eval": float(n_h_outer),
            "N_grad": float((common_phase0 or 0.0) + (common_phase1 or 0.0)),
            "N_metric": float((common_phase2 or 0.0) + (common_phase3 or 0.0)),
            "N_H_refit_eval": float(n_h_refit),
        }
    meta.update(
        status="ok",
        components=components,
        common_algorithmic_component_status=common_status,
        common_algorithmic_components=common_components,
        common_exposure_policy_id=COMMON_EXPOSURE_POLICY_ID,
        operator_probe_charge_basis=OPERATOR_PROBE_CHARGE_BASIS,
        controller_phase_records_with_group_keys=records_with_group_key_values,
        controller_phase_records_with_group_key_details=records_with_group_key_details,
        controller_phase_actual_operator_probe_counts={
            "phase0": float(phase0),
            "phase1": float(phase1),
            "phase2": float(phase2),
            "phase3": float(phase3),
        },
        controller_phase_common_exposure_operator_probe_counts=common_phase_details,
        controller_phase_common_probe_counts=common_phase_details,
        controller_shot_proxy=float(phase0 + phase1 + phase2 + phase3),
        candidate_work_ledger=candidate_work_ledger,
        selected_logical_phase0_accounting=phase0_accounting,
        refit_sources=refit_sources,
        nfev_total=float(nfev_total),
        N_other_quantum=0.0,
        source_kind=source_kind or None,
        source=source or None,
    )
    return components, meta


def _is_beam_aggregate_controller_summary(summary: Any) -> bool:
    if not isinstance(summary, Mapping):
        return False
    return str(summary.get("beam_run_scope") or "") == BEAM_AGGREGATE_RUN_SCOPE


def _sum_components(components: Mapping[str, Any] | None) -> float | None:
    if not isinstance(components, Mapping):
        return None
    total = 0.0
    for key in _COMPONENT_KEYS:
        value = _finite_nonnegative(components.get(key))
        if value is None:
            return None
        total += float(value)
    return float(total)


def _beam_search_total_provenance(
    source_payload: Mapping[str, Any],
    *,
    source_label: str | None = None,
) -> dict[str, Any]:
    components, meta = _runtime_s_norm_components_from_payload(source_payload, source_label=source_label)
    total = _sum_components(components)
    status = str(meta.get("status") or "runtime_reconstruction_failed")
    return {
        "schema": BEAM_SEARCH_TOTAL_PROVENANCE_SCHEMA,
        "status": "ok" if components is not None and total is not None and status == "ok" else status,
        "scope": BEAM_AGGREGATE_RUN_SCOPE,
        "components": copy.deepcopy(components) if isinstance(components, Mapping) else None,
        "S_beam_search_total": total,
        "runtime_reconstruction": meta,
        "promoted_to_row_s_alg": False,
    }


def _beam_winner_terminal_s_norm_components_from_payload(
    source_payload: Mapping[str, Any],
    *,
    source_label: str | None = None,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    adapt = _adapt_payload(source_payload)
    history = adapt.get("history")
    aggregate_provenance = _beam_search_total_provenance(source_payload, source_label=source_label)
    base_meta: dict[str, Any] = {
        "schema": "snake_runtime_s_norm_reconstruction_v1",
        "status": "unknown",
        "scope": "terminal",
        "source_label": source_label,
        "beam_terminal_work_scope": "winner_lineage_terminal",
        "S_alg_work_scope": "winner_lineage_terminal",
        "S_alg_row_policy": BEAM_TERMINAL_ROW_POLICY,
        "beam_aggregate_summary_blocked_as_row_s_alg": True,
        "S_beam_search_total": aggregate_provenance.get("S_beam_search_total"),
        "S_beam_search_total_status": aggregate_provenance.get("status"),
        "S_beam_search_scope": BEAM_AGGREGATE_RUN_SCOPE,
        "S_beam_search_components": aggregate_provenance.get("components"),
        "beam_search_total_reconstruction": aggregate_provenance,
    }
    if not isinstance(history, list):
        base_meta.update(status="beam_winner_history_missing")
        return None, base_meta
    history_rows: list[Mapping[str, Any]] = []
    invalid_indices: list[int] = []
    for idx, row in enumerate(history, start=1):
        if isinstance(row, Mapping):
            history_rows.append(row)
        else:
            invalid_indices.append(idx)
    if invalid_indices:
        base_meta.update(status="invalid_beam_winner_history_row", invalid_indices=invalid_indices[:20])
        return None, base_meta

    outer_nfev, outer_meta = _explicit_prefix_outer_nfev(history_rows)
    if outer_nfev is None:
        base_meta.update(status=outer_meta.get("status", "invalid_beam_winner_outer_nfev"), winner_outer_nfev=outer_meta)
        return None, base_meta
    history_refit, history_meta = _history_refit_nfev(history_rows)
    resume_refit, resume_meta = _optional_refit_nfev(adapt, "resume_boundary_refit")
    final_refit, final_meta = _optional_refit_nfev(adapt, "final_full_refit")
    refit_sources = {
        "history": history_meta,
        "resume_boundary_refit": resume_meta,
        "final_full_refit": final_meta,
    }
    if history_refit is None or resume_refit is None or final_refit is None:
        base_meta.update(status="invalid_beam_winner_refit_nfev", refit_sources=refit_sources)
        return None, base_meta

    controller_summary = controller_proxy_from_history_rows(history_rows)
    if not isinstance(controller_summary, Mapping):
        base_meta.update(status="beam_winner_controller_summary_unavailable")
        return None, base_meta
    controller_summary = dict(controller_summary)
    controller_summary["beam_run_scope"] = BEAM_WINNER_RUN_SCOPE
    controller_summary["row_s_alg_policy"] = BEAM_TERMINAL_ROW_POLICY
    controller_summary["history_row_count"] = int(len(history_rows))

    winner_nfev_total = float(outer_nfev + history_refit + resume_refit + final_refit)
    scoped_adapt = dict(adapt)
    scoped_adapt["history"] = [dict(row) for row in history_rows]
    scoped_adapt["controller_measurement_work_summary"] = controller_summary
    scoped_adapt["nfev_total"] = winner_nfev_total
    scoped_payload = dict(source_payload)
    if "adapt_vqe" in scoped_payload or "adapt_vqe" in source_payload:
        scoped_payload["adapt_vqe"] = scoped_adapt
    else:
        scoped_payload.update(scoped_adapt)
    components, meta = _runtime_s_norm_components_from_payload(scoped_payload, source_label=source_label)
    runtime_status = str(meta.get("status") or "beam_winner_runtime_reconstruction_failed")
    meta = dict(meta)
    meta.update(base_meta)
    if components is None:
        meta["status"] = runtime_status
        return None, meta
    meta.update(
        status="ok",
        components=components,
        winner_history_position=int(len(history_rows)),
        winner_history_count=int(len(history_rows)),
        winner_outer_nfev=outer_meta,
        winner_refit_sources=refit_sources,
        winner_nfev_total=winner_nfev_total,
        controller_summary_source="controller_proxy_from_history_rows(winner_history)",
    )
    return components, meta


def _terminal_s_norm_components_from_payload(
    source_payload: Mapping[str, Any],
    *,
    source_label: str | None = None,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    adapt = _adapt_payload(source_payload)
    summary = _controller_summary(adapt)
    if _is_beam_aggregate_controller_summary(summary):
        return _beam_winner_terminal_s_norm_components_from_payload(source_payload, source_label=source_label)
    return _runtime_s_norm_components_from_payload(source_payload, source_label=source_label)


def snake_controller_shot_proxy_from_payload(
    source_payload: Mapping[str, Any],
    *,
    source_label: str | None = None,
) -> dict[str, Any]:
    """Return the corrected SNAKE controller shot proxy from phase telemetry.

    This is the reporting-facing proxy used by legacy three-model support rows:
    it counts controller gradient/metric work, including selected-logical
    Phase-0 full-pool screening when old payloads omitted explicit phase0
    telemetry. It deliberately excludes optimizer/refit nfev bins.
    """

    adapt = _adapt_payload(source_payload)
    summary = _controller_summary(adapt)
    meta: dict[str, Any] = {
        "schema": "snake_controller_shot_proxy_reconstruction_v1",
        "status": "unknown",
        "source_label": source_label,
    }
    if not isinstance(summary, Mapping):
        meta.update(status="missing_controller_measurement_work_summary")
        return meta
    if summary.get("legacy_fallback_used") is not False:
        meta.update(status="legacy_controller_summary_not_promotable")
        return meta
    source_kind = str(summary.get("source_kind", "") or "")
    source = str(summary.get("source", "") or "")
    allowed_sources = {"native_controller_live_decision_work_v1", "native_controller_work"}
    if source_kind != "native_controller_work" and source not in allowed_sources:
        meta.update(
            status="missing_native_controller_provenance",
            source_kind=source_kind or None,
            source=source or None,
        )
        return meta
    if source_kind and source_kind != "native_controller_work":
        meta.update(status="non_native_controller_summary_not_promotable", source_kind=source_kind)
        return meta
    if source and source not in allowed_sources:
        meta.update(status="non_native_controller_source_not_promotable", source=source)
        return meta
    phase_map = _controller_phase_map(summary)
    if not isinstance(phase_map, Mapping):
        meta.update(status="missing_controller_phase_breakdown")
        return meta
    unassigned = _positive_unassigned_controller_phase_work(phase_map)
    if unassigned:
        meta.update(status="unassigned_controller_phase_work", unassigned_phases=unassigned)
        return meta

    phase_details: dict[str, Any] = {}
    phase0, phase0_accounting = _selected_logical_phase0_gradient_work(source_payload, phase_map)
    phase_details["phase0"] = phase0_accounting
    phase1, detail = _phase_actual_operator_probe_count(phase_map, "phase1")
    phase_details["phase1"] = detail
    phase2, detail = _phase_actual_operator_probe_count(phase_map, "phase2")
    phase_details["phase2"] = detail
    phase3, detail = _phase_actual_operator_probe_count(phase_map, "phase3")
    phase_details["phase3"] = detail
    if phase0 is None:
        meta.update(status="invalid_selected_logical_phase0_accounting", phases=phase_details)
        return meta
    if phase1 is None or phase2 is None or phase3 is None:
        meta.update(status="invalid_controller_phase_counts", phases=phase_details)
        return meta

    shot_details: dict[str, Any] = {}
    phase1_shots, detail = _phase_shots_new(phase_map, "phase1")
    shot_details["phase1"] = detail
    phase2_shots, detail = _phase_shots_new(phase_map, "phase2")
    shot_details["phase2"] = detail
    phase3_shots, detail = _phase_shots_new(phase_map, "phase3")
    shot_details["phase3"] = detail
    if phase1_shots is None or phase2_shots is None or phase3_shots is None:
        meta.update(status="invalid_controller_phase_shot_counts", phase_shots=shot_details)
        return meta
    phase0_shots = 0.0
    explicit_phase0 = phase0_accounting.get("status") == "explicit_controller_phase0"
    if explicit_phase0:
        explicit_phase = str(phase0_accounting.get("explicit_phase0", {}).get("phase") or "phase0")
        phase0_shots_value, detail = _phase_shots_new(phase_map, explicit_phase)
        shot_details["phase0"] = detail
        if phase0_shots_value is None:
            meta.update(status="invalid_controller_phase_shot_counts", phase_shots=shot_details)
            return meta
        phase0_shots = float(phase0_shots_value)
    elif phase0_accounting.get("status") == "inferred_from_selected_logical_filter":
        phase0_shots = float(phase0)
        shot_details["phase0"] = {
            "phase": "phase0",
            "status": "inferred_from_selected_logical_filter",
            "shots_new": float(phase0_shots),
            "source": "selected_logical_phase0_accounting.gradient_probe_count",
        }
    else:
        shot_details["phase0"] = {"phase": "phase0", "status": str(phase0_accounting.get("status") or "not_applicable"), "shots_new": 0.0}

    n_grad_records = float(phase0 + phase1)
    n_metric_records = float(phase2 + phase3)
    n_grad_shots = float(phase0_shots + phase1_shots)
    n_metric_shots = float(phase2_shots + phase3_shots)
    total = float(n_grad_shots + n_metric_shots)
    meta.update(
        status="ok",
        controller_shot_proxy=total,
        N_grad_controller=float(n_grad_shots),
        N_metric_controller=float(n_metric_shots),
        N_grad_controller_records=float(n_grad_records),
        N_metric_controller_records=float(n_metric_records),
        controller_phase_records_with_group_keys={
            "phase0": float(phase0),
            "phase1": float(phase1),
            "phase2": float(phase2),
            "phase3": float(phase3),
        },
        controller_phase_actual_operator_probe_counts={
            "phase0": float(phase0),
            "phase1": float(phase1),
            "phase2": float(phase2),
            "phase3": float(phase3),
        },
        controller_phase_shots_new={
            "phase0": float(phase0_shots),
            "phase1": float(phase1_shots),
            "phase2": float(phase2_shots),
            "phase3": float(phase3_shots),
        },
        controller_phase_shot_details=shot_details,
        selected_logical_phase0_accounting=phase0_accounting,
        source_kind=source_kind or None,
        source=source or None,
    )
    return meta


def _explicit_component_normalization(
    *,
    row: Mapping[str, Any],
    raw: Mapping[str, Any],
    missing_reason: str = "missing_component_breakdown",
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    return normalized_measurement_work_from_explicit_row(
        row=row,
        raw_proxy=raw,
        missing_reason=missing_reason,
    )


def _runtime_component_normalization(
    *,
    row: Mapping[str, Any],
    raw: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    source_label: str | None,
) -> tuple[dict[str, Any] | None, dict[str, float], dict[str, str], dict[str, Any], dict[str, float] | None]:
    components, reconstruction = _runtime_s_norm_components_from_payload(
        source_payload,
        source_label=source_label,
    )
    if components is None:
        return None, {}, {"S_norm": str(reconstruction.get("status", "runtime_reconstruction_failed"))}, reconstruction, None
    component_row = {key: row[key] for key in RAW_PROXY_PRIORITY if key in row}
    component_row.update({key: float(value) for key, value in components.items()})
    component_row["N_other_quantum"] = 0.0
    measurement_work, updates, statuses = normalized_measurement_work_from_explicit_row(
        row=component_row,
        raw_proxy=raw,
        missing_reason="runtime_reconstruction_failed",
    )
    measurement_work = dict(measurement_work)
    measurement_work["runtime_reconstruction"] = reconstruction
    return measurement_work, updates, statuses, reconstruction, {key: float(value) for key, value in components.items()}



def _history_prefix_rows_for_scope(
    adapt_payload: Mapping[str, Any],
    *,
    history_position: int | None,
) -> tuple[list[Mapping[str, Any]] | None, dict[str, Any]]:
    if history_position is None:
        return None, {"status": "history_position_required"}
    if int(history_position) < 1:
        return None, {"status": "history_position_out_of_range", "history_position": history_position}
    history = adapt_payload.get("history")
    if not isinstance(history, list):
        return None, {"status": "missing_history"}
    if int(history_position) > len(history):
        return None, {
            "status": "history_position_out_of_range",
            "history_position": int(history_position),
            "history_row_count": len(history),
        }
    rows: list[Mapping[str, Any]] = []
    invalid_indices: list[int] = []
    for idx, row in enumerate(history[: int(history_position)], start=1):
        if isinstance(row, Mapping):
            rows.append(row)
        else:
            invalid_indices.append(idx)
    if invalid_indices:
        return None, {"status": "invalid_history_row", "indices": invalid_indices[:20]}
    return rows, {"status": "ok", "history_position": int(history_position), "history_count": len(rows)}


def _explicit_prefix_outer_nfev(history_rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, dict[str, Any]]:
    """Return explicitly scoped prefix outer/seed objective evaluations.

    Prefix work must not use terminal ``nfev_total`` residuals.  Count only
    fields that are already scoped to the accepted history rows.  Missing
    fields fail closed; otherwise an incomplete prefix can masquerade as a
    comparable ``S_alg`` value.
    """

    aliases = (
        "nfev_seed_probe",
        "seed_nfev",
        "outer_nfev",
        "nfev_outer",
        "objective_nfev",
        "nfev_initial_energy",
        "initial_energy_nfev",
        "nfev_schur_warm_start_guard",
        "nfev_schur_prune_warm_start_guard",
    )
    total = 0.0
    sources: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    missing_rows: list[int] = []
    for idx, row in enumerate(history_rows, start=1):
        row_total = 0.0
        present: list[str] = []
        for key in aliases:
            if key not in row or row.get(key) in {None, ""}:
                continue
            value = _finite_nonnegative(row.get(key))
            if value is None:
                invalid.append({"history_index": idx, "field": key, "value": row.get(key)})
                continue
            row_total += float(value)
            present.append(key)
        total += row_total
        if present:
            sources.append({"history_index": idx, "fields": present, "nfev": float(row_total)})
        else:
            missing_rows.append(idx)
    if invalid:
        return None, {"status": "invalid_prefix_outer_nfev", "invalid": invalid[:20]}
    if missing_rows:
        return None, {
            "status": "missing_prefix_outer_nfev",
            "missing_history_indices": missing_rows[:20],
            "aliases_checked": list(aliases),
            "sources": sources,
            "missing_policy": "block_without_explicit_prefix_outer_fields_per_history_row",
        }
    return float(total), {
        "status": "ok",
        "nfev": float(total),
        "sources": sources,
        "missing_policy": "explicit_prefix_outer_fields_only",
    }


def _scoped_runtime_s_norm_components_from_payload(
    source_payload: Mapping[str, Any],
    *,
    scope: str,
    history_position: int | None = None,
    source_label: str | None = None,
    allow_terminal_scope_equivalence: bool = True,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    normalized_scope = str(scope or "terminal").strip().lower().replace("-", "_")
    adapt = _adapt_payload(source_payload)
    if normalized_scope == "terminal":
        components, meta = _terminal_s_norm_components_from_payload(source_payload, source_label=source_label)
        meta = dict(meta)
        meta["scope"] = "terminal"
        return components, meta
    if normalized_scope not in {"display_prefix", "plateau_prefix", "fixed_prefix", "prefix"}:
        return None, {
            "schema": "snake_scoped_algorithmic_work_reconstruction_v1",
            "status": "unsupported_scope",
            "scope": normalized_scope,
            "source_label": source_label,
        }
    prefix_rows, prefix_meta = _history_prefix_rows_for_scope(adapt, history_position=history_position)
    if prefix_rows is None:
        return None, {
            "schema": "snake_scoped_algorithmic_work_reconstruction_v1",
            "status": prefix_meta.get("status", "prefix_history_unavailable"),
            "scope": normalized_scope,
            "source_label": source_label,
            "prefix": prefix_meta,
        }
    history = adapt.get("history")
    if (
        allow_terminal_scope_equivalence
        and isinstance(history, list)
        and int(history_position or 0) == len(history)
    ):
        terminal_components, terminal_meta = _terminal_s_norm_components_from_payload(
            source_payload,
            source_label=source_label,
        )
        if terminal_components is not None:
            meta = dict(terminal_meta)
            meta.update(
                scoped_schema="snake_scoped_algorithmic_work_reconstruction_v1",
                scope="display_prefix" if normalized_scope == "prefix" else normalized_scope,
                history_position=int(history_position or 0),
                prefix=prefix_meta,
                terminal_scope_equivalence=True,
                terminal_scope_equivalence_policy="final_prefix_uses_terminal_runtime_ledger_v1",
            )
            return terminal_components, meta
    outer_nfev, outer_meta = _explicit_prefix_outer_nfev(prefix_rows)
    if outer_nfev is None:
        return None, {
            "schema": "snake_scoped_algorithmic_work_reconstruction_v1",
            "status": outer_meta.get("status", "invalid_prefix_outer_nfev"),
            "scope": normalized_scope,
            "source_label": source_label,
            "prefix": prefix_meta,
            "outer_nfev": outer_meta,
        }
    history_refit, history_meta = _history_refit_nfev(prefix_rows)
    if history_refit is None:
        return None, {
            "schema": "snake_scoped_algorithmic_work_reconstruction_v1",
            "status": history_meta.get("status", "invalid_prefix_refit_nfev"),
            "scope": normalized_scope,
            "source_label": source_label,
            "prefix": prefix_meta,
            "refit_sources": {"history": history_meta},
        }
    controller_summary = controller_proxy_from_history_rows(prefix_rows)
    scoped_adapt = dict(adapt)
    scoped_adapt["history"] = [dict(row) for row in prefix_rows]
    scoped_adapt["controller_measurement_work_summary"] = controller_summary
    scoped_adapt["resume_boundary_refit"] = {"executed": False}
    scoped_adapt["final_full_refit"] = {"executed": False}
    scoped_adapt["nfev_total"] = float(history_refit + outer_nfev)
    scoped_payload = dict(source_payload)
    if "adapt_vqe" in scoped_payload or "adapt_vqe" in source_payload:
        scoped_payload["adapt_vqe"] = scoped_adapt
    else:
        scoped_payload.update(scoped_adapt)
    components, meta = _runtime_s_norm_components_from_payload(scoped_payload, source_label=source_label)
    meta = dict(meta)
    meta.update(
        scoped_schema="snake_scoped_algorithmic_work_reconstruction_v1",
        scope="display_prefix" if normalized_scope == "prefix" else normalized_scope,
        history_position=None if history_position is None else int(history_position),
        prefix=prefix_meta,
        prefix_outer_nfev=outer_meta,
        prefix_refit_sources={"history": history_meta},
        controller_summary_source="controller_proxy_from_history_rows(prefix)",
    )
    return components, meta


def snake_algorithmic_work_from_payload(
    source_payload: Mapping[str, Any],
    *,
    scope: str = "terminal",
    history_position: int | None = None,
    source_label: str | None = None,
    allow_terminal_scope_equivalence: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return canonical SNAKE ``S_alg`` work for a declared scope.

    This helper is reporting-only.  It promotes only component/event telemetry
    and keeps raw controller shot proxies as diagnostics.  Prefix scopes are
    reconstructed from history-row controller summaries rather than terminal
    aggregate summaries.
    """

    components, reconstruction = _scoped_runtime_s_norm_components_from_payload(
        source_payload,
        scope=scope,
        history_position=history_position,
        source_label=source_label,
        allow_terminal_scope_equivalence=allow_terminal_scope_equivalence,
    )
    scope_fields = {"work_semantics_version": SNAKE_TERMINAL_WORK_SEMANTICS_VERSION}
    scope_fields.update({
        key: reconstruction.get(key)
        for key in (
            "S_alg_work_scope",
            "S_alg_row_policy",
            "S_beam_search_total",
            "S_beam_search_total_status",
            "S_beam_search_scope",
            "S_beam_search_components",
        )
        if key in reconstruction
    })
    status = str(reconstruction.get("status") or "runtime_reconstruction_failed")
    if components is None:
        work = {
            **scope_fields,
            "S_alg": None,
            "S_actual": None,
            "S_actual_status": status,
            "S_actual_missing_reason": status,
            "S_actual_policy": "actual_chargeable_operator_probes_pre_grouping_v1",
            "S_norm": None,
            "S_alg_status": status,
            "S_alg_missing_reason": status,
            "algorithmic_measurement_work": {
                "schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
                "status": status,
                "reason": status,
                "S_alg": None,
                "components": None,
                "runtime_reconstruction": reconstruction,
                "unit": "algorithmic_estimator_or_probe_event_count_not_physical_shots",
            },
        }
        return work, reconstruction
    ledger = _runtime_reconstruction_event_ledger(
        components=components,
        reconstruction=reconstruction,
        source_label=source_label,
    )
    if ledger is None:
        status = "runtime_event_ledger_unavailable"
        work = {
            **scope_fields,
            "S_alg": None,
            "S_actual": None,
            "S_actual_status": status,
            "S_actual_missing_reason": status,
            "S_actual_policy": "actual_chargeable_operator_probes_pre_grouping_v1",
            "S_norm": None,
            "S_alg_status": status,
            "S_alg_missing_reason": status,
            "algorithmic_measurement_work": {
                "schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
                "status": status,
                "reason": status,
                "S_alg": None,
                "components": None,
                "runtime_reconstruction": reconstruction,
                "unit": "algorithmic_estimator_or_probe_event_count_not_physical_shots",
            },
        }
        audit = dict(reconstruction)
        audit["status"] = status
        return work, audit
    alg_work, alg_updates, alg_statuses = algorithmic_measurement_work_from_row(
        row={"table_i_measurement_event_ledger": ledger},
        raw_proxy={},
    )
    s_alg_status = alg_statuses.get("S_alg", "missing_event_ledger_component_breakdown")
    if s_alg_status != "ok":
        work = {
            **scope_fields,
            "S_alg": None,
            "S_actual": None,
            "S_actual_status": s_alg_status,
            "S_actual_missing_reason": s_alg_status,
            "S_actual_policy": "actual_chargeable_operator_probes_pre_grouping_v1",
            "S_norm": None,
            "S_alg_status": s_alg_status,
            "S_alg_missing_reason": s_alg_status,
            "algorithmic_measurement_work": alg_work,
            "table_i_measurement_event_ledger": ledger,
        }
        audit = dict(reconstruction)
        audit["status"] = s_alg_status
        return work, audit
    ledger_blocker = _runtime_event_ledger_blocker(ledger)
    if ledger_blocker is not None:
        status = str(ledger_blocker)
        blocked_alg_work = dict(alg_work)
        blocked_alg_work.update(
            status=status,
            reason="runtime_reconstruction_does_not_prove_comparable_candidate_screening_work",
            S_alg=None,
            runtime_reconstruction=dict(reconstruction),
            unit="algorithmic_estimator_or_probe_event_count_not_physical_shots",
        )
        work = {
            **scope_fields,
            "S_alg": None,
            "S_actual": None,
            "S_actual_status": status,
            "S_actual_missing_reason": status,
            "S_actual_policy": "actual_chargeable_operator_probes_pre_grouping_v1",
            "S_norm": None,
            "S_alg_lower_bound": None,
            "S_actual_lower_bound": None,
            "S_alg_status": status,
            "S_alg_missing_reason": status,
            "algorithmic_measurement_work": blocked_alg_work,
            "table_i_measurement_event_ledger": ledger,
            "work_scope": reconstruction.get("scope", scope),
            "history_position": reconstruction.get("history_position"),
        }
        audit = dict(reconstruction)
        audit.update(status=status, S_alg_lower_bound=None)
        return work, audit
    work = {
        **scope_fields,
        **alg_updates,
        "S_actual": float(alg_updates["S_alg"]),
        "S_actual_status": "ok",
        "S_actual_missing_reason": None,
        "S_actual_policy": "actual_chargeable_operator_probes_pre_grouping_v1",
        "S_actual_source": "validated_legacy_runtime_operator_record_reconstruction",
        "S_norm": float(alg_updates["S_alg"]),
        "S_alg_status": "ok",
        "S_alg_missing_reason": None,
        "algorithmic_measurement_work": alg_work,
        "table_i_measurement_event_ledger": ledger,
        "work_scope": reconstruction.get("scope", scope),
        "history_position": reconstruction.get("history_position"),
    }
    audit = dict(reconstruction)
    audit.update(status="ok", S_alg=float(alg_updates["S_alg"]), S_actual=float(alg_updates["S_alg"]))
    return work, audit


def _normalize_controller_phase_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text == "phase_0":
        return "phase0"
    return text


def _parse_controller_work_scope(scope_key: Any) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in str(scope_key or "").split("|"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            parsed[key] = value
    return parsed


def _mechanism_chargeable_operator_probe_count(entry: Mapping[str, Any]) -> tuple[float | None, dict[str, Any]]:
    for key in ("actual_operator_probe_count", "actual_operator_probe_count_total"):
        if key not in entry:
            continue
        value = _finite_nonnegative_integer(entry.get(key))
        if value is None:
            return None, {"status": "nonintegral_or_negative_count", "source": key}
        charge_basis = str(entry.get("operator_probe_charge_basis") or "")
        if charge_basis != OPERATOR_PROBE_CHARGE_BASIS:
            return None, {
                "status": "policy_mismatch",
                "source": key,
                "operator_probe_charge_basis": charge_basis or None,
                "expected_operator_probe_charge_basis": OPERATOR_PROBE_CHARGE_BASIS,
            }
        return float(value), {
            "status": "ok",
            "source": key,
            "operator_probe_count": int(value),
            "operator_probe_charge_basis": OPERATOR_PROBE_CHARGE_BASIS,
        }
    if not _phase_has_positive_diagnostic_work(entry):
        return 0.0, {"status": "explicit_zero_without_typed_operator_probe_count"}
    return None, {
        "status": "missing_actual_operator_probe_count",
        "source": "actual_operator_probe_count",
        "reason": "candidate_or_group_counters_are_exposure_diagnostics_not_S_alg",
    }


def _mechanism_event_records_from_controller_summary(
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_events = summary.get("events")
    if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes, bytearray)):
        events: list[dict[str, Any]] = []
        for idx, entry in enumerate(raw_events):
            if not isinstance(entry, Mapping):
                continue
            event = dict(entry)
            event.setdefault("event_index", idx)
            event.setdefault("source_kind", "controller_events")
            events.append(event)
        if events:
            return events, {"status": "ok", "source": "events", "event_count": len(events)}

    by_scope = summary.get("by_scope")
    if isinstance(by_scope, Mapping):
        events = []
        for scope_key, entry in by_scope.items():
            if not isinstance(entry, Mapping):
                continue
            parsed = _parse_controller_work_scope(scope_key)
            event = dict(entry)
            if "phase" not in event and parsed.get("phase"):
                event["phase"] = parsed["phase"]
            if "event_kind" not in event and parsed.get("event"):
                event["event_kind"] = parsed["event"]
            event["scope_key"] = str(scope_key)
            event["scope_fields"] = parsed
            event["source_kind"] = "controller_by_scope"
            events.append(event)
        if events:
            return events, {"status": "ok", "source": "by_scope", "event_count": len(events)}

    phase_map = _controller_phase_map(summary)
    if isinstance(phase_map, Mapping):
        events = []
        for phase, entry in phase_map.items():
            if not isinstance(entry, Mapping):
                continue
            event = dict(entry)
            event.setdefault("phase", str(phase))
            event.setdefault("event_kind", None)
            event["source_kind"] = "controller_by_phase"
            events.append(event)
        return events, {
            "status": "coarse_phase_only",
            "source": "by_phase",
            "event_count": len(events),
            "mechanism_limitation": "phase_totals_reconcile_S_alg_but_do_not_identify_mechanism_events",
        }

    return [], {"status": "missing_controller_event_or_phase_breakdown", "source": None, "event_count": 0}


def _mechanism_scoped_controller_summary(
    source_payload: Mapping[str, Any],
    *,
    scope: str,
    history_position: int | None,
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    normalized_scope = str(scope or "terminal").strip().lower().replace("-", "_")
    adapt = _adapt_payload(source_payload)
    if normalized_scope == "terminal":
        summary = _controller_summary(adapt)
        if _is_beam_aggregate_controller_summary(summary):
            history = adapt.get("history")
            if not isinstance(history, list):
                return None, {"status": "beam_winner_history_missing", "scope": "terminal"}
            history_rows = [row for row in history if isinstance(row, Mapping)]
            if len(history_rows) != len(history):
                return None, {"status": "invalid_beam_winner_history_row", "scope": "terminal"}
            summary = controller_proxy_from_history_rows(history_rows)
            if isinstance(summary, Mapping):
                scoped = dict(summary)
                scoped["beam_run_scope"] = BEAM_WINNER_RUN_SCOPE
                scoped["row_s_alg_policy"] = BEAM_TERMINAL_ROW_POLICY
                scoped["history_row_count"] = int(len(history_rows))
                return scoped, {
                    "status": "ok",
                    "scope": "terminal",
                    "summary_source": "controller_proxy_from_history_rows(winner_history)",
                    "beam_aggregate_summary_excluded": True,
                }
            return None, {"status": "beam_winner_controller_summary_unavailable", "scope": "terminal"}
        if isinstance(summary, Mapping):
            return summary, {"status": "ok", "scope": "terminal", "summary_source": "controller_measurement_work_summary"}
        return None, {"status": "missing_controller_measurement_work_summary", "scope": "terminal"}

    if normalized_scope not in {"display_prefix", "plateau_prefix", "fixed_prefix", "prefix"}:
        return None, {"status": "unsupported_scope", "scope": normalized_scope}
    prefix_rows, prefix_meta = _history_prefix_rows_for_scope(adapt, history_position=history_position)
    if prefix_rows is None:
        return None, {
            "status": prefix_meta.get("status", "prefix_history_unavailable"),
            "scope": normalized_scope,
            "prefix": prefix_meta,
        }
    summary = controller_proxy_from_history_rows(prefix_rows)
    if isinstance(summary, Mapping):
        return summary, {
            "status": "ok",
            "scope": "display_prefix" if normalized_scope == "prefix" else normalized_scope,
            "history_position": int(history_position or 0),
            "summary_source": "controller_proxy_from_history_rows(prefix)",
            "prefix": prefix_meta,
        }
    return None, {
        "status": "prefix_controller_summary_unavailable",
        "scope": normalized_scope,
        "history_position": int(history_position or 0),
        "prefix": prefix_meta,
    }


def _mechanism_scoped_history_rows(
    source_payload: Mapping[str, Any],
    *,
    scope: str,
    history_position: int | None,
) -> tuple[list[Mapping[str, Any]] | None, dict[str, Any]]:
    normalized_scope = str(scope or "terminal").strip().lower().replace("-", "_")
    adapt = _adapt_payload(source_payload)
    history = adapt.get("history")
    if not isinstance(history, list):
        return None, {"status": "missing_history", "scope": normalized_scope}
    history_rows = [row for row in history if isinstance(row, Mapping)]
    if len(history_rows) != len(history):
        return None, {"status": "invalid_history_row", "scope": normalized_scope}
    if normalized_scope == "terminal":
        return history_rows, {
            "status": "ok",
            "scope": "terminal",
            "history_row_count": int(len(history_rows)),
        }
    if normalized_scope not in {"display_prefix", "plateau_prefix", "fixed_prefix", "prefix"}:
        return None, {"status": "unsupported_scope", "scope": normalized_scope}
    prefix_rows, prefix_meta = _history_prefix_rows_for_scope(adapt, history_position=history_position)
    if prefix_rows is None:
        return None, {
            "status": prefix_meta.get("status", "prefix_history_unavailable"),
            "scope": normalized_scope,
            "prefix": prefix_meta,
        }
    return prefix_rows, {
        "status": "ok",
        "scope": "display_prefix" if normalized_scope == "prefix" else normalized_scope,
        "history_position": int(history_position or 0),
        "history_row_count": int(len(prefix_rows)),
        "prefix": prefix_meta,
    }


def _int_window_tuple(value: Any) -> tuple[int, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return ()
    try:
        return tuple(int(x) for x in value)
    except Exception:
        return ()


def _phase2_formula_records_from_history_row(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in (
        "scored_surface_records",
        "phase2_scored_surface_records",
        "phase2_last_shortlist_records",
        "phase2_last_shortlist_eval_records",
    ):
        records = row.get(key)
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
            return [record for record in records if isinstance(record, Mapping)]
    return []


def _phase2_novelty_enabled_for_payload(source_payload: Mapping[str, Any]) -> bool:
    settings = source_payload.get("settings")
    settings = settings if isinstance(settings, Mapping) else {}
    adapt = _adapt_payload(source_payload)
    try:
        gamma = float(settings.get("phase2_gamma_N", adapt.get("phase2_gamma_N", 1.0)))
    except Exception:
        gamma = 1.0
    if gamma <= 0.0:
        return False
    mode = str(
        adapt.get("phase3_novelty_ablation_mode")
        or settings.get("phase3_novelty_ablation_mode")
        or "off"
    ).strip().lower()
    return mode not in {"all", "both", "no_phase2", "phase2_off", "phase2", "raw_off"}


def _phase2_second_order_enabled_for_payload(source_payload: Mapping[str, Any]) -> bool:
    settings = source_payload.get("settings")
    settings = settings if isinstance(settings, Mapping) else {}
    mode = str(settings.get("phase2_selector_gain_mode") or "").strip().lower()
    return mode not in {"unit_gain_v1", "unit_gain", "novelty_only", "novelty-only"}


def _upper_triangle_count(width: int) -> int:
    n = int(max(0, width))
    return int(n * (n + 1) // 2)


def _phase2_window_formula_reconstruction(
    source_payload: Mapping[str, Any],
    *,
    scope: str,
    history_position: int | None,
) -> dict[str, Any]:
    rows, row_meta = _mechanism_scoped_history_rows(
        source_payload,
        scope=scope,
        history_position=history_position,
    )
    if rows is None:
        return {
            "schema": "snake_phase2_window_formula_work_v1",
            "status": str(row_meta.get("status") or "missing_history"),
            "publishable": False,
            "history": row_meta,
        }

    novelty_enabled = _phase2_novelty_enabled_for_payload(source_payload)
    second_order_enabled = _phase2_second_order_enabled_for_payload(source_payload)
    base_norm = 0
    novelty_candidate_window = 0
    novelty_window_gram = 0
    second_order_candidate_hessian = 0
    second_order_candidate_window = 0
    second_order_window_hessian = 0
    event_rows: list[dict[str, Any]] = []

    for step_index, row in enumerate(rows, start=1):
        records = _phase2_formula_records_from_history_row(row)
        if not records:
            continue
        phase2_windows = {_int_window_tuple(record.get("phase2_geometry_window_indices")) for record in records}
        schur_windows = {_int_window_tuple(record.get("schur_window_indices")) for record in records}
        step_base = int(len(records))
        step_novelty_candidate_window = 0
        step_second_order_candidate_window = 0
        for record in records:
            if novelty_enabled:
                step_novelty_candidate_window += int(len(_int_window_tuple(record.get("phase2_geometry_window_indices"))))
            if second_order_enabled:
                step_second_order_candidate_window += int(len(_int_window_tuple(record.get("schur_window_indices"))))
        step_novelty_window_gram = (
            sum(_upper_triangle_count(len(window)) for window in phase2_windows)
            if novelty_enabled
            else 0
        )
        step_second_order_window_hessian = (
            sum(_upper_triangle_count(len(window)) for window in schur_windows)
            if second_order_enabled
            else 0
        )
        step_second_order_candidate_hessian = step_base if second_order_enabled else 0

        base_norm += step_base
        novelty_candidate_window += step_novelty_candidate_window
        novelty_window_gram += step_novelty_window_gram
        second_order_candidate_hessian += step_second_order_candidate_hessian
        second_order_candidate_window += step_second_order_candidate_window
        second_order_window_hessian += step_second_order_window_hessian
        event_rows.append(
            {
                "step_index": int(step_index),
                "phase2_record_count": int(step_base),
                "phase2_distinct_window_count": int(len(phase2_windows)),
                "schur_distinct_window_count": int(len(schur_windows)),
                "base_candidate_norm": int(step_base),
                "novelty_candidate_window_gram": int(step_novelty_candidate_window),
                "novelty_window_gram": int(step_novelty_window_gram),
                "second_order_candidate_hessian": int(step_second_order_candidate_hessian),
                "second_order_candidate_window_hessian": int(step_second_order_candidate_window),
                "second_order_window_hessian": int(step_second_order_window_hessian),
            }
        )

    novelty_total = int(novelty_candidate_window + novelty_window_gram)
    second_order_total = int(
        second_order_candidate_hessian
        + second_order_candidate_window
        + second_order_window_hessian
    )
    total = int(base_norm + novelty_total + second_order_total)
    components = {
        "phase2_base_candidate_norm": int(base_norm),
        "phase2_novelty_candidate_window_gram": int(novelty_candidate_window),
        "phase2_novelty_window_gram": int(novelty_window_gram),
        "phase2_novelty_total": int(novelty_total),
        "phase2_second_order_candidate_hessian": int(second_order_candidate_hessian),
        "phase2_second_order_candidate_window_hessian": int(second_order_candidate_window),
        "phase2_second_order_window_hessian": int(second_order_window_hessian),
        "phase2_second_order_total": int(second_order_total),
        "phase2_formula_metric_total": int(total),
    }
    if base_norm <= 0:
        return {
            "schema": "snake_phase2_window_formula_work_v1",
            "status": "missing_phase2_formula_records",
            "publishable": False,
            "scope": row_meta.get("scope", scope),
            "history_position": row_meta.get("history_position"),
            "history_row_count": row_meta.get("history_row_count"),
            "novelty_enabled": bool(novelty_enabled),
            "second_order_enabled": bool(second_order_enabled),
            "components": components,
            "events": event_rows,
        }
    return {
        "schema": "snake_phase2_window_formula_work_v1",
        "status": "ok",
        "publishable": True,
        "scope": row_meta.get("scope", scope),
        "history_position": row_meta.get("history_position"),
        "history_row_count": row_meta.get("history_row_count"),
        "novelty_enabled": bool(novelty_enabled),
        "second_order_enabled": bool(second_order_enabled),
        "components": components,
        "events": event_rows,
    }


def _empty_mechanism_bins() -> dict[str, Any]:
    return {
        "gradient": {
            "phase0_pilot_screen": 0.0,
            "phase1_append_probe": 0.0,
            "phase1_insertion_probe": 0.0,
            "route_a_child_phase1_gradient": 0.0,
            "route_a_direct_child_phase3_gradient": 0.0,
            "unclassified_gradient": 0.0,
        },
        "metric": {
            "phase2_rerank_unclassified": 0.0,
            "phase3_reduced_geometry_scoring": 0.0,
            "route_a_child_phase2_metric": 0.0,
            "route_a_child_phase3_metric": 0.0,
            "route_a_direct_child_phase3_metric": 0.0,
            "phase3_batch_union_scoring": 0.0,
            "unclassified_metric": 0.0,
        },
        "H": {
            "H_outer": 0.0,
            "H_refit": 0.0,
            "H_total": 0.0,
        },
    }


def _mechanism_classify_event(
    *,
    phase: str,
    event_kind: str,
    count: float,
    bins: MutableMapping[str, Any],
    statuses: MutableMapping[str, Any],
) -> None:
    if count == 0.0:
        return
    if phase in {"phase0", "phase1"}:
        gradient = bins["gradient"]
        if phase == "phase0" and event_kind in {"phase0_pilot_screen", ""}:
            gradient["phase0_pilot_screen"] += float(count)
            statuses["gradient.phase0_pilot_screen"] = "ledger_exact"
        elif phase == "phase1" and event_kind == "phase1_append_probe":
            gradient["phase1_append_probe"] += float(count)
            statuses["gradient.phase1_append_probe"] = "ledger_exact"
        elif phase == "phase1" and event_kind == "phase1_insertion_probe":
            gradient["phase1_insertion_probe"] += float(count)
            statuses["gradient.phase1_insertion_probe"] = "ledger_exact"
        elif phase == "phase1" and event_kind in {
            "route_a_child_phase1_gradient",
            "route_a_direct_child_phase3_gradient",
        }:
            gradient[event_kind] += float(count)
            statuses[f"gradient.{event_kind}"] = "ledger_exact"
        else:
            gradient["unclassified_gradient"] += float(count)
            statuses["gradient.unclassified_gradient"] = "coarse_unclassified"
        return
    if phase in {"phase2", "phase3"}:
        metric = bins["metric"]
        if phase == "phase2" and event_kind == "phase2_rerank_records":
            metric["phase2_rerank_unclassified"] += float(count)
            statuses["metric.phase2_rerank_unclassified"] = "requires_formula_reconstruction"
        elif phase == "phase3" and event_kind == "phase3_reduced_geometry_rerank":
            metric["phase3_reduced_geometry_scoring"] += float(count)
            statuses["metric.phase3_reduced_geometry_scoring"] = "ledger_exact"
        elif event_kind in {
            "route_a_child_phase2_metric",
            "route_a_child_phase3_metric",
            "route_a_direct_child_phase3_metric",
        }:
            metric[event_kind] += float(count)
            statuses[f"metric.{event_kind}"] = "ledger_exact"
        elif event_kind == "batch_union_scoring":
            metric["phase3_batch_union_scoring"] += float(count)
            statuses["metric.phase3_batch_union_scoring"] = "ledger_exact"
        else:
            metric["unclassified_metric"] += float(count)
            statuses["metric.unclassified_metric"] = "coarse_unclassified"


def _sum_leaf_values(mapping: Mapping[str, Any]) -> float:
    total = 0.0
    for value in mapping.values():
        parsed = _finite_nonnegative(value)
        if parsed is not None:
            total += float(parsed)
    return float(total)


def _first_mapping_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def snake_mechanism_resolved_work_from_payload(
    source_payload: Mapping[str, Any],
    *,
    scope: str = "terminal",
    history_position: int | None = None,
    source_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return conservative mechanism-resolved SNAKE work for a declared scope.

    The existing four-bin ``snake_algorithmic_work_from_payload`` result remains
    authoritative.  This helper only partitions already-validated typed
    operator-probe/H-eval work into mechanism bins when event provenance supports
    it.  Candidate exposure counters are emitted as diagnostics/formula operands
    and are never promoted directly to ``S_alg``.
    """

    coarse_work, coarse_audit = snake_algorithmic_work_from_payload(
        source_payload,
        scope=scope,
        history_position=history_position,
        source_label=source_label,
    )
    base: dict[str, Any] = {
        "schema": SNAKE_MECHANISM_RESOLVED_WORK_SCHEMA_VERSION,
        "status": str(coarse_work.get("S_alg_status") or coarse_audit.get("status") or "unknown"),
        "scope": coarse_work.get("work_scope", coarse_audit.get("scope", scope)),
        "history_position": coarse_work.get("history_position", coarse_audit.get("history_position")),
        "source_label": source_label,
        "coarse_S_alg": {
            "S_alg": coarse_work.get("S_alg"),
            "S_alg_status": coarse_work.get("S_alg_status"),
            "S_alg_N_grad_probe": coarse_work.get("S_alg_N_grad_probe"),
            "S_alg_N_metric_probe": coarse_work.get("S_alg_N_metric_probe"),
            "S_alg_N_H_outer_eval": coarse_work.get("S_alg_N_H_outer_eval"),
            "S_alg_N_H_refit_eval": coarse_work.get("S_alg_N_H_refit_eval"),
            "work_semantics_version": coarse_work.get("work_semantics_version"),
        },
        "beam_search_total_provenance": {
            key: coarse_work.get(key)
            for key in (
                "S_beam_search_total",
                "S_beam_search_total_status",
                "S_beam_search_scope",
                "S_beam_search_components",
            )
            if key in coarse_work
        },
        "candidate_exposure": None,
        "formula_operands": {"event_records": []},
        "measurement_work": _empty_mechanism_bins(),
        "mechanism_bin_status": {},
        "mechanism_resolution_status": "blocked",
    }
    if coarse_work.get("S_alg_status") != "ok":
        base["mechanism_resolution_status"] = "blocked"
        base["status_detail"] = "coarse_S_alg_not_available"
        return base, {"schema": SNAKE_MECHANISM_RESOLVED_WORK_SCHEMA_VERSION, "status": base["status"], "coarse_audit": coarse_audit}

    summary, summary_meta = _mechanism_scoped_controller_summary(
        source_payload,
        scope=scope,
        history_position=history_position,
    )
    if not isinstance(summary, Mapping):
        base.update(status=str(summary_meta.get("status") or "controller_summary_unavailable"))
        base["mechanism_resolution_status"] = "blocked"
        return base, {"schema": SNAKE_MECHANISM_RESOLVED_WORK_SCHEMA_VERSION, "status": base["status"], "summary": summary_meta}

    candidate_exposure = _controller_candidate_work_ledger_audit(summary)
    events, event_meta = _mechanism_event_records_from_controller_summary(summary)
    bins = base["measurement_work"]
    statuses: dict[str, Any] = {}
    event_operands: list[dict[str, Any]] = []
    invalid_events: list[dict[str, Any]] = []
    for idx, event in enumerate(events):
        phase = _normalize_controller_phase_name(
            event.get("phase") or event.get("controller_phase") or event.get("scope_fields", {}).get("phase")
        )
        event_kind = str(
            event.get("event_kind") or event.get("event") or event.get("scope_fields", {}).get("event") or ""
        )
        count, count_detail = _mechanism_chargeable_operator_probe_count(event)
        event_operands.append(
            {
                "index": idx,
                "phase": phase or None,
                "event_kind": event_kind or None,
                "source_kind": event.get("source_kind"),
                "scope_key": event.get("scope_key"),
                "actual_operator_probe_count": count,
                "count_status": count_detail.get("status"),
                "candidate_count_total": _first_mapping_value(event, ("candidate_count_total", "candidate_count")),
                "evaluated_count_total": _first_mapping_value(event, ("evaluated_count_total", "evaluated_count")),
                "records_evaluated": event.get("records_evaluated"),
                "shortlist_size_total": _first_mapping_value(event, ("shortlist_size_total", "shortlist_size")),
                "retained_count_total": _first_mapping_value(event, ("retained_count_total", "retained_count")),
            }
        )
        if count is None:
            invalid_events.append({"index": idx, "phase": phase, "event_kind": event_kind or None, "detail": count_detail})
            continue
        _mechanism_classify_event(
            phase=phase,
            event_kind=event_kind,
            count=float(count),
            bins=bins,
            statuses=statuses,
        )

    h_outer = _finite_nonnegative(coarse_work.get("S_alg_N_H_outer_eval")) or 0.0
    h_refit = _finite_nonnegative(coarse_work.get("S_alg_N_H_refit_eval")) or 0.0
    bins["H"]["H_outer"] = float(h_outer)
    bins["H"]["H_refit"] = float(h_refit)
    bins["H"]["H_total"] = float(h_outer + h_refit)
    statuses["H.H_outer"] = "coarse_exact_unclassified"
    statuses["H.H_refit"] = "coarse_exact_unclassified"

    grad_total = _sum_leaf_values(bins["gradient"])
    metric_total = _sum_leaf_values(bins["metric"])
    expected_grad = _finite_nonnegative(coarse_work.get("S_alg_N_grad_probe")) or 0.0
    expected_metric = _finite_nonnegative(coarse_work.get("S_alg_N_metric_probe")) or 0.0
    expected_total = _finite_nonnegative(coarse_work.get("S_alg")) or 0.0
    reconstructed_total = float(grad_total + metric_total + h_outer + h_refit)
    mismatches: dict[str, dict[str, float]] = {}
    for key, observed, expected in (
        ("S_alg_N_grad_probe", grad_total, expected_grad),
        ("S_alg_N_metric_probe", metric_total, expected_metric),
        ("S_alg", reconstructed_total, expected_total),
    ):
        if abs(float(observed) - float(expected)) > 1e-9:
            mismatches[key] = {"observed": float(observed), "expected": float(expected)}

    has_unclassified = (
        bins["gradient"]["unclassified_gradient"] > 0.0
        or bins["metric"]["phase2_rerank_unclassified"] > 0.0
        or bins["metric"]["unclassified_metric"] > 0.0
    )
    requires_formula_reconstruction = bool(bins["metric"]["phase2_rerank_unclassified"] > 0.0)
    mechanism_resolution = "exact"
    if event_meta.get("source") == "by_phase" or has_unclassified:
        mechanism_resolution = "partial"
    if invalid_events or mismatches:
        mechanism_resolution = "blocked"
    status = "ok" if not invalid_events and not mismatches else "invalid_reconciliation"
    phase2_formula = _phase2_window_formula_reconstruction(
        source_payload,
        scope=scope,
        history_position=history_position,
    )
    phase2_formula_available = bool(
        expected_metric > 0.0
        and phase2_formula.get("publishable") is True
        and phase2_formula.get("status") == "ok"
    )
    phase2_formula_components = phase2_formula.get("components")
    phase2_formula_components = (
        phase2_formula_components if isinstance(phase2_formula_components, Mapping) else {}
    )
    formula_metric_total = _finite_nonnegative(
        phase2_formula_components.get("phase2_formula_metric_total")
    )
    formula_supersedes_coarse_metric_mismatch = bool(
        phase2_formula_available
        and not invalid_events
        and set(mismatches).issubset({"S_alg_N_metric_probe", "S_alg"})
    )
    phase2_coarse_metric = float(bins["metric"]["phase2_rerank_unclassified"])
    non_phase2_metric_total = (
        max(0.0, float(expected_metric) - phase2_coarse_metric)
        if phase2_coarse_metric > 0.0
        else 0.0
    )
    display_metric_total = (
        float(formula_metric_total) + non_phase2_metric_total
        if phase2_formula_available and formula_metric_total is not None
        else float(expected_metric)
    )
    if phase2_formula_available and isinstance(phase2_formula_components, MutableMapping):
        phase2_formula_components["phase2_replaced_coarse_metric"] = phase2_coarse_metric
        phase2_formula_components["non_phase2_metric_preserved"] = non_phase2_metric_total
        phase2_formula_components["display_metric_total"] = display_metric_total
    display_total = float(expected_grad + display_metric_total + h_outer + h_refit)
    mechanism_s_alg_publishable = bool(
        (
            status == "ok"
            and (
                (mechanism_resolution == "exact" and not requires_formula_reconstruction)
                or (phase2_formula_available and mechanism_resolution == "partial")
            )
        )
        or formula_supersedes_coarse_metric_mismatch
    )
    if mechanism_s_alg_publishable:
        mechanism_s_alg_status = (
            "ok_phase2_window_formula_v1" if phase2_formula_available else "ok"
        )
    elif requires_formula_reconstruction:
        mechanism_s_alg_status = "requires_phase2_formula_reconstruction"
    elif mechanism_resolution == "partial":
        mechanism_s_alg_status = "partial_mechanism_reconstruction"
    else:
        mechanism_s_alg_status = str(status or mechanism_resolution or "blocked")
    base.update(
        status=status,
        S_alg=float(expected_total),
        mechanism_algorithmic_work={
            "schema": "snake_mechanism_algorithmic_work_v1",
            "publishable": mechanism_s_alg_publishable,
            "status": mechanism_s_alg_status,
            "S_alg": float(display_total) if mechanism_s_alg_publishable else None,
            "S_alg_N_grad_probe": float(expected_grad) if mechanism_s_alg_publishable else None,
            "S_alg_N_metric_probe": float(display_metric_total) if mechanism_s_alg_publishable else None,
            "S_alg_N_H_outer_eval": float(h_outer) if mechanism_s_alg_publishable else None,
            "S_alg_N_H_refit_eval": float(h_refit) if mechanism_s_alg_publishable else None,
            "coarse_S_alg": float(expected_total),
            "coarse_S_alg_N_grad_probe": float(expected_grad),
            "coarse_S_alg_N_metric_probe": float(expected_metric),
            "coarse_S_alg_N_H_outer_eval": float(h_outer),
            "coarse_S_alg_N_H_refit_eval": float(h_refit),
            "requires_formula_reconstruction": requires_formula_reconstruction,
            "mechanism_resolution_status": mechanism_resolution,
            "phase2_formula_reconstruction": phase2_formula,
        },
        candidate_exposure=candidate_exposure,
        formula_operands={"event_records": event_operands},
        mechanism_event_source=event_meta,
        mechanism_scope_summary=summary_meta,
        mechanism_bin_status=statuses,
        mechanism_resolution_status=mechanism_resolution,
        partial_mechanism_reconstruction=bool(mechanism_resolution == "partial"),
        requires_formula_reconstruction=requires_formula_reconstruction,
        mechanism_resolution_detail=(
            "phase2_metric_subsplit_requires_formula_reconstruction"
            if mechanism_resolution == "partial" and requires_formula_reconstruction
            else None
        ),
        reconciliation={
            "status": "ok" if not mismatches else "mismatch",
            "mismatches": mismatches,
            "S_alg_N_grad_probe": {"observed": float(grad_total), "expected": float(expected_grad)},
            "S_alg_N_metric_probe": {"observed": float(metric_total), "expected": float(expected_metric)},
            "S_alg": {"observed": float(reconstructed_total), "expected": float(expected_total)},
        },
        invalid_event_records=invalid_events,
    )
    audit = {
        "schema": SNAKE_MECHANISM_RESOLVED_WORK_SCHEMA_VERSION,
        "status": status,
        "mechanism_resolution_status": mechanism_resolution,
        "coarse_audit": coarse_audit,
        "summary": summary_meta,
        "event_source": event_meta,
        "reconciliation": base["reconciliation"],
    }
    return base, audit


def snake_fair_expanded_work_from_payload(
    source_payload: Mapping[str, Any],
    *,
    scope: str = "terminal",
    history_position: int | None = None,
    source_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return SNAKE work in actual and common-exposure probe currencies.

    ``S_actual`` is the actual chargeable operator-probe work reconstructed for
    the implemented trajectory. ``S_fair`` is the Paper-I equal-access diagnostic
    for asymmetric child/polyfilter comparisons and is therefore sourced only
    from typed full-common-exposure operator-probe telemetry.  It is not the
    default visible Hubbard--Holstein three-method table ``S`` currency; those
    rows use scoped ``S_alg`` / winner-lineage actual work unless a report
    explicitly requests a fair/common-exposure diagnostic. Measurement group
    counts and legacy candidate/shortlist fields are never accepted as
    substitutes for that common-exposure currency.
    """

    work, audit = snake_algorithmic_work_from_payload(
        source_payload,
        scope=scope,
        history_position=history_position,
        source_label=source_label,
    )
    ledger = work.get("table_i_measurement_event_ledger") if isinstance(work, Mapping) else None
    ledger = ledger if isinstance(ledger, Mapping) else {}
    alg_work = work.get("algorithmic_measurement_work") if isinstance(work, Mapping) else None
    alg_work = alg_work if isinstance(alg_work, Mapping) else {}
    candidate = ledger.get("candidate_work_ledger")
    candidate = candidate if isinstance(candidate, Mapping) else {}

    status = str(work.get("S_alg_status") or audit.get("status") or "unknown")
    component_source_kind = str(ledger.get("component_source_kind") or "")
    common_status = str(ledger.get("common_algorithmic_component_status") or "")
    candidate_status = str(ledger.get("candidate_work_ledger_status") or candidate.get("status") or "")
    actual_value = _finite_nonnegative_integer(work.get("S_actual", work.get("S_alg")))
    actual_status = "ok" if status == "ok" and actual_value is not None else status
    common_components = ledger.get("common_algorithmic_components")
    common_components = common_components if isinstance(common_components, Mapping) else None
    common_component_fields: dict[str, int] = {}
    common_blocker: str | None = None
    if actual_status != "ok":
        common_blocker = str(actual_status)
    elif candidate_status != "ok":
        common_blocker = "missing_explicit_candidate_work_ledger"
    elif common_status != "ok" or common_components is None:
        common_blocker = common_status or "missing_common_exposure_ledger"
    elif ledger.get("common_exposure_policy_id") != COMMON_EXPOSURE_POLICY_ID:
        common_blocker = "policy_mismatch"
    elif ledger.get("operator_probe_charge_basis") != OPERATOR_PROBE_CHARGE_BASIS:
        common_blocker = "policy_mismatch"
    else:
        aliases = {
            "N_H_outer_eval": "N_H_outer_eval",
            "N_grad": "N_grad_probe_common_exposure",
            "N_metric": "N_metric_probe_common_exposure",
            "N_H_refit_eval": "N_H_refit_eval",
        }
        for source_key, dest_key in aliases.items():
            parsed = _finite_nonnegative_integer(common_components.get(source_key))
            if parsed is None:
                common_blocker = "nonintegral_or_negative_count"
                break
            common_component_fields[dest_key] = int(parsed)
    common_value = None if common_blocker is not None else int(sum(common_component_fields.values()))

    base = {
        "schema": "paper_i_hh_algorithmic_work_v2",
        "work_contract_id": "paper_i_hh_operator_probe_contract_v2",
        "S_actual": None if actual_value is None else int(actual_value),
        "S_actual_status": actual_status,
        "S_actual_reason": None if actual_status == "ok" else actual_status,
        "S_actual_policy": "actual_chargeable_operator_probes_pre_grouping_v1",
        "S_actual_provenance": "validated_legacy_reconstruction" if actual_status == "ok" else None,
        "S_common_exposure": common_value,
        "S_common_exposure_status": "ok" if common_blocker is None else str(common_blocker),
        "S_common_exposure_reason": None if common_blocker is None else str(common_blocker),
        "S_common_exposure_policy": COMMON_EXPOSURE_POLICY_ID,
        "S_common_exposure_provenance": "explicit_event_ledger" if common_blocker is None else None,
        "S_fair": common_value,
        "S_fair_status": "ok" if common_blocker is None else str(common_blocker),
        "S_fair_missing_reason": None if common_blocker is None else str(common_blocker),
        "S_fair_reason": None if common_blocker is None else str(common_blocker),
        "S_fair_policy": COMMON_EXPOSURE_POLICY_ID,
        "S_fair_source": "S_common_exposure",
        "fair_work_currency": FAIR_WORK_CURRENCY,
        "operator_probe_charge_basis": OPERATOR_PROBE_CHARGE_BASIS,
        "component_source_kind": component_source_kind or None,
        "common_algorithmic_component_status": common_status or None,
        "candidate_work_ledger_status": candidate_status or None,
        "S_alg": work.get("S_alg"),
        "S_alg_status": status,
        "algorithmic_measurement_work": alg_work or None,
        "table_i_measurement_event_ledger": ledger or None,
    }
    if actual_status == "ok":
        base["S_actual_components"] = {
            "N_H_outer_eval": _finite_nonnegative_integer(work.get("S_alg_N_H_outer_eval")),
            "N_grad_probe_actual": _finite_nonnegative_integer(work.get("S_alg_N_grad_probe")),
            "N_metric_probe_actual": _finite_nonnegative_integer(work.get("S_alg_N_metric_probe")),
            "N_H_refit_eval": _finite_nonnegative_integer(work.get("S_alg_N_H_refit_eval")),
            "N_other_quantum": _finite_nonnegative_integer(work.get("S_alg_N_other_quantum", 0)),
        }
    if common_component_fields:
        base["S_common_exposure_components"] = common_component_fields

    if common_blocker is not None:
        out = {
            **base,
            "S_fair_source_kind": "blocked_snake_expanded_common_probe_ledger",
        }
        blocked_audit = dict(audit)
        blocked_audit.update(
            status=str(common_blocker),
            S_actual=None if actual_value is None else int(actual_value),
            S_common_exposure=None,
            S_fair=None,
        )
        return out, blocked_audit

    out = {
        **base,
        "S_fair_source_kind": "snake_expanded_common_probe_ledger",
        "component_counts": common_component_fields,
        "component_sources": alg_work.get("component_sources"),
    }
    ok_audit = dict(audit)
    ok_audit.update(status="ok", S_actual=base["S_actual"], S_common_exposure=common_value, S_fair=common_value)
    return out, ok_audit


def snake_deterministic_shot_proxy_from_payload(
    source_payload: Mapping[str, Any],
    *,
    scope: str = "terminal",
    history_position: int | None = None,
    source_label: str | None = None,
    shots_per_pauli_term_proxy: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return comparator-compatible SNAKE ``shots_total`` only when safe.

    This helper deliberately does not use controller ``shots_new`` or legacy
    scalar work proxies.  It requires canonical scoped ``S_alg`` components plus
    explicit deterministic-shot scaling inputs.
    """

    work, work_audit = snake_algorithmic_work_from_payload(
        source_payload,
        scope=scope,
        history_position=history_position,
        source_label=source_label,
    )
    normalized_scope = str(work.get("work_scope") or work_audit.get("scope") or scope)
    audit: dict[str, Any] = {
        "schema": "snake_deterministic_shot_proxy_v1",
        "status": "unknown",
        "display_policy": "blocked_no_comparable_shots_total",
        "scope": normalized_scope,
        "history_position": work.get("history_position", work_audit.get("history_position")),
        "source_label": source_label,
        "S_alg_status": work.get("S_alg_status") or work_audit.get("status"),
        "S_alg": work.get("S_alg"),
        "algorithmic_work_audit": work_audit,
    }
    legacy = _snake_legacy_work_proxies_from_payload(source_payload)
    if len(legacy) > 2:
        audit["legacy_work_proxies"] = legacy

    if work.get("S_alg_status") != "ok" or work.get("S_alg") is None:
        audit.update(status="s_alg_blocked", blocker=str(work.get("S_alg_status") or work_audit.get("status") or "s_alg_blocked"))
        return {}, audit

    component_fields = {
        "N_H_outer_eval": work.get("S_alg_N_H_outer_eval"),
        "N_H_refit_eval": work.get("S_alg_N_H_refit_eval"),
        "N_grad_probe": work.get("S_alg_N_grad_probe"),
        "N_metric_probe": work.get("S_alg_N_metric_probe"),
    }
    parsed_components: dict[str, int] = {}
    invalid_components: list[str] = []
    for key, value in component_fields.items():
        parsed = _finite_nonnegative_integer(value)
        if parsed is None:
            invalid_components.append(key)
        else:
            parsed_components[key] = int(parsed)
    if invalid_components:
        audit.update(
            status="inconsistent_s_alg_components",
            blocker="invalid_or_noninteger_component_counts",
            invalid_components=invalid_components,
            component_counts=component_fields,
        )
        return {}, audit

    s_alg_total = sum(parsed_components.values())
    parsed_s_alg = _finite_nonnegative_integer(work.get("S_alg"))
    if parsed_s_alg is None or int(parsed_s_alg) != int(s_alg_total):
        audit.update(
            status="inconsistent_s_alg_components",
            blocker="S_alg_component_sum_mismatch",
            component_counts=parsed_components,
            S_alg_component_sum=int(s_alg_total),
        )
        return {}, audit

    h_count, h_source, h_source_kind = _trusted_payload_integer_at_paths(
        source_payload,
        (
            ("hamiltonian_pauli_term_count",),
            ("result", "hamiltonian_pauli_term_count"),
            ("adapt_vqe", "hamiltonian_pauli_term_count"),
            ("compiled_pauli_cache", "h_terms"),
            ("adapt_vqe", "compiled_pauli_cache", "h_terms"),
        ),
    )
    if h_count is None:
        status = "invalid_hamiltonian_pauli_term_count" if h_source else "missing_hamiltonian_pauli_term_count"
        audit.update(status=status, blocker=status, hamiltonian_pauli_term_count_source=h_source)
        return {}, audit

    if shots_per_pauli_term_proxy is not None:
        shots_per_term = _finite_nonnegative_integer(shots_per_pauli_term_proxy)
        shots_source = "argument.shots_per_pauli_term_proxy"
        shots_source_kind = "explicit_integer"
        if shots_per_term is None:
            audit.update(status="invalid_shots_per_pauli_term_proxy", blocker="invalid_argument_shots_per_pauli_term_proxy")
            return {}, audit
    else:
        shots_per_term, shots_source, shots_source_kind = _trusted_payload_integer_at_paths(
            source_payload,
            (
                ("shots_per_pauli_term_proxy",),
                ("result", "shots_per_pauli_term_proxy"),
                ("adapt_vqe", "shots_per_pauli_term_proxy"),
                ("paper_i_deterministic_shot_proxy_inputs", "shots_per_pauli_term_proxy"),
                ("adapt_vqe", "paper_i_deterministic_shot_proxy_inputs", "shots_per_pauli_term_proxy"),
            ),
        )
        if shots_per_term is None:
            status = "invalid_shots_per_pauli_term_proxy" if shots_source else "missing_shots_per_pauli_term_proxy"
            audit.update(status=status, blocker=status, shots_per_pauli_term_proxy_source=shots_source)
            return {}, audit

    energy_eval_count = parsed_components["N_H_outer_eval"] + parsed_components["N_H_refit_eval"]
    gradient_probe_count = parsed_components["N_grad_probe"]
    metric_probe_count = parsed_components["N_metric_probe"]
    fields = build_deterministic_shot_proxy_fields(
        hamiltonian_pauli_term_count=int(h_count),
        pool_term_count=0,
        energy_eval_count=int(energy_eval_count),
        gradient_scan_count=0,
        gradient_operator_probe_count=int(gradient_probe_count),
        metric_operator_probe_count=int(metric_probe_count),
        shots_per_pauli_term_proxy=int(shots_per_term),
        comparator_legacy_coercion=False,
    )
    fields["snake_deterministic_shot_proxy"] = {
        **audit,
        "status": "ok",
        "display_policy": "comparable_deterministic_total_shot_proxy",
        "component_counts": {
            **parsed_components,
            "energy_eval_count_proxy": int(energy_eval_count),
            "gradient_operator_probe_count_proxy": int(gradient_probe_count),
            "metric_operator_probe_count_proxy": int(metric_probe_count),
        },
        "hamiltonian_pauli_term_count": int(h_count),
        "hamiltonian_pauli_term_count_source": h_source,
        "hamiltonian_pauli_term_count_source_kind": h_source_kind,
        "shots_per_pauli_term_proxy": int(shots_per_term),
        "shots_per_pauli_term_proxy_source": shots_source,
        "shots_per_pauli_term_proxy_source_kind": shots_source_kind,
        "shots_total": fields["shots_total"],
    }
    return fields, fields["snake_deterministic_shot_proxy"]


def _runtime_reconstruction_event_ledger(
    *,
    components: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    source_label: str | None = None,
) -> dict[str, Any] | None:
    """Translate validated native SNAKE runtime bins into strict Table-I ledger form."""

    if str(reconstruction.get("status") or "") != "ok":
        return None
    if str(reconstruction.get("schema") or "") != "snake_runtime_s_norm_reconstruction_v1":
        return None
    allowed_sources = {"native_controller_live_decision_work_v1", "native_controller_work"}
    source_kind = str(reconstruction.get("source_kind", "") or "")
    source = str(reconstruction.get("source", "") or "")
    if source_kind and source_kind != "native_controller_work":
        return None
    if source and source not in allowed_sources:
        return None
    if not source_kind and source not in allowed_sources:
        return None
    other_quantum = _finite_nonnegative(reconstruction.get("N_other_quantum", 0.0))
    if other_quantum is None or float(other_quantum) != 0.0:
        return None
    common_status = str(reconstruction.get("common_algorithmic_component_status") or "")
    common_components_raw = reconstruction.get("common_algorithmic_components")
    component_source_payload = components
    component_source_kind = "actual_operator_probe_components"
    aliases = {
        "N_H_outer_eval": "N_H_outer_eval",
        "N_grad": "N_grad_probe",
        "N_metric": "N_metric_probe",
        "N_H_refit_eval": "N_H_refit_eval",
    }
    totals: dict[str, float] = {}
    sources: dict[str, str] = {}
    mapping = reconstruction.get("component_mapping")
    mapping = mapping if isinstance(mapping, Mapping) else {}
    reconstructed_components = reconstruction.get("components")
    if not isinstance(reconstructed_components, Mapping):
        return None
    for source_key, dest_key in aliases.items():
        value = _finite_nonnegative(component_source_payload.get(source_key))
        if value is None:
            return None
        reconstructed_value = _finite_nonnegative(reconstructed_components.get(source_key))
        if reconstructed_value is None or abs(float(reconstructed_value) - float(value)) > 1e-9:
            return None
        totals[dest_key] = float(value)
        source = mapping.get(source_key)
        if isinstance(source, list):
            sources[dest_key] = "; ".join(str(item) for item in source)
        elif source is not None:
            sources[dest_key] = str(source)
        else:
            sources[dest_key] = f"runtime_reconstruction.components.{source_key}"
    totals["N_other_quantum"] = 0.0
    sources["N_other_quantum"] = "runtime_reconstruction_validated_zero"
    candidate_work_ledger = reconstruction.get("candidate_work_ledger")
    candidate_work_ledger = candidate_work_ledger if isinstance(candidate_work_ledger, Mapping) else {}
    return {
        "schema": TABLE_I_EVENT_LEDGER_SCHEMA,
        "status": "ok",
        "source_kind": "snake_native_runtime_reconstruction_v1",
        "source_label": source_label or reconstruction.get("source_label"),
        "component_totals": totals,
        "component_sources": sources,
        "component_source_kind": component_source_kind,
        "runtime_reconstruction": dict(reconstruction),
        "common_algorithmic_component_status": common_status or "missing_common_exposure_ledger",
        "common_algorithmic_components": copy.deepcopy(common_components_raw) if isinstance(common_components_raw, Mapping) else None,
        "common_exposure_policy_id": reconstruction.get("common_exposure_policy_id"),
        "operator_probe_charge_basis": reconstruction.get("operator_probe_charge_basis"),
        "candidate_work_ledger_status": str(
            candidate_work_ledger.get("status") or "missing_explicit_candidate_work_ledger"
        ),
        "candidate_work_ledger": copy.deepcopy(candidate_work_ledger),
        "event_count_convention": "fresh_measurement_bearing_estimator_or_probe_events",
        "cache_policy": "no_cache_reuse_in_current_snake_runtime_reconstruction",
        "measurement_model_id": "noiseless_estimator_schedule_count_v1",
        "N_other_quantum": 0.0,
    }


def _runtime_event_ledger_candidate_work_ok(ledger: Mapping[str, Any] | None) -> bool:
    if not isinstance(ledger, Mapping):
        return False
    candidate = ledger.get("candidate_work_ledger")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    status = str(ledger.get("candidate_work_ledger_status") or candidate.get("status") or "")
    return status == "ok"


def _runtime_event_ledger_blocker(ledger: Mapping[str, Any] | None) -> str | None:
    if not isinstance(ledger, Mapping):
        return "runtime_event_ledger_unavailable"
    candidate = ledger.get("candidate_work_ledger")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    status = str(ledger.get("candidate_work_ledger_status") or candidate.get("status") or "")
    if status != "ok":
        return "missing_explicit_candidate_work_ledger"
    return None


def _embedded_runtime_reconstruction_event_ledger(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recover a ledger from an already enriched runtime row when no fresh source is supplied."""

    work = row.get("measurement_work")
    if not isinstance(work, Mapping):
        return None
    reconstruction = work.get("runtime_reconstruction")
    components = work.get("components")
    if not isinstance(reconstruction, Mapping) or not isinstance(components, Mapping):
        return None
    return _runtime_reconstruction_event_ledger(
        components=components,
        reconstruction=reconstruction,
        source_label=str(reconstruction.get("source_label") or "") or None,
    )


def normalize_snake_measurement_work_row(
    row: Mapping[str, Any],
    *,
    source_payload: Mapping[str, Any] | None = None,
    source_label: str | None = None,
    allow_runtime_reconstruction: bool = True,
) -> dict[str, Any]:
    """Return ``row`` with additive SNAKE measurement-work normalization fields.

    Explicit four-bin components are preferred. If those are absent and a
    detailed runtime payload is supplied, this function may reconstruct the same
    four bins from native controller phase telemetry and nfev metadata. Raw
    scalar proxies remain fallback/provenance only.
    """

    existing_work = row.get("measurement_work")
    existing_grouped = row.get("grouped_measurement_work")
    existing_algorithmic = row.get("algorithmic_measurement_work")
    existing_physical = row.get("physical_measurement_work")
    existing_s_norm_status = str(row.get("S_norm_status", ""))
    existing_s_grp_status = str(row.get("S_grp_status", ""))
    existing_s_alg_status = str(row.get("S_alg_status", ""))
    existing_s_phys_status = str(row.get("S_phys_status", ""))
    existing_s_l2_status = str(row.get("S_l2_status", ""))
    strict_s_norm_rejection, strict_s_norm_reason = _strict_explicit_s_norm_rejection(row)
    has_new_accounting = (
        isinstance(existing_algorithmic, Mapping)
        and existing_algorithmic.get("schema") == ALGORITHMIC_MEASUREMENT_WORK_SCHEMA
        and "S_alg_status" in row
        and isinstance(existing_physical, Mapping)
        and existing_physical.get("schema") == PHYSICAL_MEASUREMENT_WORK_SCHEMA
        and "S_phys_status" in row
        and "S_l2_status" in row
    )
    can_upgrade_s_alg_from_fresh_runtime = (
        allow_runtime_reconstruction
        and isinstance(source_payload, Mapping)
        and existing_s_alg_status != "ok"
        and not _has_explicit_s_alg_component_fields(row)
        and strict_s_norm_rejection is None
    )
    can_upgrade_s_alg_from_embedded_runtime = (
        allow_runtime_reconstruction
        and source_payload is None
        and existing_s_alg_status != "ok"
        and not _has_explicit_s_alg_component_fields(row)
        and strict_s_norm_rejection is None
        and _embedded_runtime_reconstruction_event_ledger(row) is not None
    )
    if (
        has_new_accounting
        and isinstance(existing_work, Mapping)
        and existing_work.get("schema") == NORMALIZED_MEASUREMENT_WORK_SCHEMA
        and "S_norm_status" in row
        and isinstance(existing_grouped, Mapping)
        and existing_grouped.get("schema") == GROUPED_MEASUREMENT_PROXY_SCHEMA
        and "S_grp_status" in row
        and (
            (
                existing_s_norm_status == "ok"
                and (existing_s_grp_status == "ok" or not _has_explicit_s_grp_component_fields(row))
                and (existing_s_alg_status == "ok" or not _has_explicit_s_alg_component_fields(row))
                and (
                    existing_s_phys_status == "ok"
                    or existing_s_l2_status == "ok"
                    or not _has_explicit_physical_component_fields(row)
                )
            )
            or (
                (source_payload is None or not allow_runtime_reconstruction)
                and not _has_explicit_s_norm_component_fields(row)
                and not _has_explicit_s_grp_component_fields(row)
                and not _has_explicit_s_alg_component_fields(row)
                and not _has_explicit_physical_component_fields(row)
            )
        )
        and not can_upgrade_s_alg_from_fresh_runtime
        and not can_upgrade_s_alg_from_embedded_runtime
    ):
        return dict(row)

    out = dict(row)
    raw = _raw_proxy(row)
    measurement_work, updates, statuses = _explicit_component_normalization(
        row=row,
        raw=raw,
        missing_reason="missing_component_breakdown",
    )
    s_norm_status = statuses.get("S_norm", "missing_component_breakdown")
    if strict_s_norm_rejection is not None:
        measurement_work = dict(measurement_work)
        measurement_work.update(
            status=strict_s_norm_rejection,
            reason=strict_s_norm_reason,
            S_norm=None,
            components=None,
        )
        updates = {}
        statuses = {"S_norm": strict_s_norm_rejection}
        s_norm_status = strict_s_norm_rejection
    runtime_reconstruction: dict[str, Any] | None = None
    runtime_components: dict[str, float] | None = None
    if (
        (
            s_norm_status == "missing_component_breakdown"
            or (s_norm_status == "ok" and can_upgrade_s_alg_from_fresh_runtime)
        )
        and allow_runtime_reconstruction
        and isinstance(source_payload, Mapping)
    ):
        runtime_work, runtime_updates, runtime_statuses, runtime_reconstruction, runtime_components = _runtime_component_normalization(
            row=row,
            raw=raw,
            source_payload=source_payload,
            source_label=source_label,
        )
        runtime_status = runtime_statuses.get("S_norm", "runtime_reconstruction_failed")
        if runtime_status == "ok" and runtime_work is not None:
            if s_norm_status == "missing_component_breakdown":
                measurement_work = runtime_work
                updates = runtime_updates
                statuses = runtime_statuses
                s_norm_status = "ok"
            else:
                measurement_work = dict(measurement_work)
                measurement_work["runtime_reconstruction"] = runtime_reconstruction
        else:
            if s_norm_status == "missing_component_breakdown":
                measurement_work = dict(measurement_work)
                measurement_work["runtime_reconstruction"] = runtime_reconstruction
                s_norm_status = statuses.get("S_norm", s_norm_status)

    fallback_value, fallback_source = _first_raw_proxy(raw)

    out["measurement_work"] = measurement_work
    out["S_norm_status"] = s_norm_status
    grouped_work, grouped_updates, grouped_statuses = grouped_measurement_proxy_from_explicit_row(row=row)
    s_grp_status = grouped_statuses.get("S_grp", "missing_grouped_measurement_breakdown")
    out["grouped_measurement_work"] = grouped_work
    out["S_grp_status"] = s_grp_status
    if s_grp_status == "ok":
        out.update(grouped_updates)
    else:
        out["S_grp_total"] = None

    alg_row: Mapping[str, Any] = row
    runtime_event_ledger: dict[str, Any] | None = None
    if strict_s_norm_rejection is None and not _has_explicit_s_alg_component_fields(row):
        if runtime_components is not None and isinstance(runtime_reconstruction, Mapping):
            runtime_event_ledger = _runtime_reconstruction_event_ledger(
                components=runtime_components,
                reconstruction=runtime_reconstruction,
                source_label=source_label,
            )
        elif source_payload is None:
            runtime_event_ledger = _embedded_runtime_reconstruction_event_ledger(row)
    if runtime_event_ledger is not None:
        alg_row = dict(row)
        alg_row["table_i_measurement_event_ledger"] = runtime_event_ledger
        out["table_i_measurement_event_ledger"] = runtime_event_ledger

    alg_work, alg_updates, alg_statuses = algorithmic_measurement_work_from_row(row=alg_row, raw_proxy=raw)
    s_alg_status = alg_statuses.get("S_alg", "missing_event_ledger_component_breakdown")
    ledger_blocker = _runtime_event_ledger_blocker(runtime_event_ledger) if runtime_event_ledger is not None else None
    if s_alg_status == "ok" and runtime_event_ledger is not None and ledger_blocker is not None:
        s_alg_status = str(ledger_blocker)
        alg_work = dict(alg_work)
        alg_work.update(
            status=s_alg_status,
            reason="runtime_reconstruction_does_not_prove_comparable_candidate_screening_work",
            S_alg=None,
            runtime_reconstruction=runtime_reconstruction or {},
        )
        out["S_alg_lower_bound"] = None
    out["algorithmic_measurement_work"] = alg_work
    out["S_alg_status"] = s_alg_status
    if s_alg_status == "ok":
        out.update(alg_updates)
        out["S_actual"] = float(alg_updates["S_alg"])
        out["S_actual_status"] = "ok"
        out["S_actual_missing_reason"] = None
        out["S_actual_policy"] = "actual_chargeable_operator_probes_pre_grouping_v1"
    else:
        out["S_alg"] = None
        out["S_actual"] = None
        out["S_actual_status"] = s_alg_status
        out["S_actual_missing_reason"] = s_alg_status
        out["S_actual_policy"] = "actual_chargeable_operator_probes_pre_grouping_v1"

    physical_work, physical_updates, physical_statuses = physical_measurement_work_from_row(row=row)
    s_phys_status = physical_statuses.get("S_phys", "missing_fresh_grouped_event_components")
    s_l2_status = physical_statuses.get("S_l2", "missing_fresh_grouped_event_components")
    out["physical_measurement_work"] = physical_work
    out["S_phys_status"] = s_phys_status
    out["S_l2_status"] = s_l2_status
    if s_phys_status == "ok" or s_l2_status == "ok":
        out.update(physical_updates)
    if s_phys_status != "ok":
        out["S_phys"] = None
    if s_l2_status != "ok":
        out["S_l2"] = None

    if s_norm_status == "ok":
        out.update(updates)
        out["legacy_measurement_work_proxy"] = float(updates["S_norm"])
        out["legacy_measurement_work_proxy_source"] = "S_norm"
        out["legacy_measurement_work_proxy_status"] = "legacy_normalized"
    else:
        out["S_norm"] = None
        out["legacy_measurement_work_proxy"] = None
        out["legacy_measurement_work_proxy_source"] = None
        out["legacy_measurement_work_proxy_status"] = f"unavailable:{s_norm_status}"
        out["raw_shot_proxy_fallback_forbidden"] = fallback_value is not None
        out["raw_shot_proxy_fallback_audit"] = {
            "value": fallback_value,
            "source": fallback_source,
            "S_norm_status": s_norm_status,
        }

    if s_alg_status == "ok":
        out["measurement_work_proxy"] = float(alg_updates["S_alg"])
        out["measurement_work_proxy_source"] = "S_alg"
        out["measurement_work_proxy_status"] = "ok"
    else:
        out["measurement_work_proxy"] = None
        out["measurement_work_proxy_source"] = None
        out["measurement_work_proxy_status"] = s_alg_status
    return out


def _source_for_path(
    source_payloads: Mapping[str, Mapping[str, Any]] | None,
    path: tuple[str, ...],
    consumed_source_paths: set[str] | None = None,
) -> Mapping[str, Any] | None:
    if not isinstance(source_payloads, Mapping):
        return None
    key = _path_key(path)
    source = source_payloads.get(key)
    if source is not None and consumed_source_paths is not None:
        consumed_source_paths.add(key)
    return source


def _record_enrichment_stats(enriched: Mapping[str, Any], stats: Counter[str]) -> None:
    status = str(enriched.get("S_norm_status", "unknown"))
    stats[f"status:{status}"] += 1
    stats["enriched_path_count"] += 1
    if enriched.get("legacy_measurement_work_proxy_source") == "S_norm":
        stats["s_norm_available_count"] += 1
    elif enriched.get("raw_shot_proxy_fallback_forbidden"):
        stats["raw_fallback_forbidden_count"] += 1
    s_alg_status = str(enriched.get("S_alg_status", "unknown"))
    stats[f"s_alg_status:{s_alg_status}"] += 1
    if enriched.get("measurement_work_proxy_source") == "S_alg":
        stats["s_alg_available_count"] += 1
    s_phys_status = str(enriched.get("S_phys_status", "unknown"))
    stats[f"s_phys_status:{s_phys_status}"] += 1
    if enriched.get("S_phys") is not None:
        stats["s_phys_available_count"] += 1
    s_l2_status = str(enriched.get("S_l2_status", "unknown"))
    stats[f"s_l2_status:{s_l2_status}"] += 1
    if enriched.get("S_l2") is not None:
        stats["s_l2_available_count"] += 1
    s_grp_status = str(enriched.get("S_grp_status", "unknown"))
    stats[f"s_grp_status:{s_grp_status}"] += 1
    if enriched.get("S_grp_total") is not None:
        stats["s_grp_available_count"] += 1
    runtime = enriched.get("measurement_work")
    if isinstance(runtime, Mapping):
        reconstruction = runtime.get("runtime_reconstruction")
        if isinstance(reconstruction, Mapping):
            stats[f"runtime_reconstruction:{reconstruction.get('status', 'unknown')}"] += 1


def _s_alg_components_from_enriched_row(row: Mapping[str, Any]) -> dict[str, float] | None:
    work = row.get("algorithmic_measurement_work")
    if not isinstance(work, Mapping) or str(work.get("status") or "") != "ok":
        return None
    components = work.get("components")
    if not isinstance(components, Mapping):
        return None
    out: dict[str, float] = {}
    for key in ("N_H_outer_eval", "N_grad_probe", "N_metric_probe", "N_H_refit_eval"):
        value = _finite_nonnegative(components.get(key))
        if value is None:
            return None
        out[key] = float(value)
    other = _finite_nonnegative(work.get("N_other_quantum", 0.0))
    if other is None or float(other) != 0.0:
        return None
    return out


def _aggregate_s_alg_event_ledger(
    *,
    child_rows: Sequence[Mapping[str, Any]],
    source_label: str,
) -> dict[str, Any] | None:
    component_rows: list[dict[str, float]] = []
    for row in child_rows:
        components = _s_alg_components_from_enriched_row(row)
        if components is None:
            return None
        component_rows.append(components)
    if not component_rows:
        return None
    totals = {
        key: float(sum(row[key] for row in component_rows) / len(component_rows))
        for key in ("N_H_outer_eval", "N_grad_probe", "N_metric_probe", "N_H_refit_eval")
    }
    totals["N_other_quantum"] = 0.0
    return {
        "schema": TABLE_I_EVENT_LEDGER_SCHEMA,
        "status": "ok",
        "source_kind": "aggregate_mean_of_child_event_ledgers_v1",
        "source_label": source_label,
        "component_totals": totals,
        "component_sources": {
            key: "mean_of_child_strict_s_alg_components"
            for key in ("N_H_outer_eval", "N_grad_probe", "N_metric_probe", "N_H_refit_eval")
        }
        | {"N_other_quantum": "child_rows_validated_zero"},
        "child_row_count": int(len(component_rows)),
        "event_count_convention": "class_average_of_fresh_measurement_bearing_estimator_or_probe_events",
        "cache_policy": "inherits_child_event_ledger_cache_policy",
        "measurement_model_id": "noiseless_estimator_schedule_count_v1",
        "N_other_quantum": 0.0,
    }


def _enrich_or_aggregate_row(
    row: Mapping[str, Any],
    *,
    path: tuple[str, ...],
    stats: Counter[str],
    child_rows: Sequence[Mapping[str, Any]] | None = None,
    source_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row_for_enrichment: Mapping[str, Any] = row
    allow_runtime_reconstruction = True
    normalizer_source_payload = source_payload
    if child_rows is not None and not _has_explicit_s_alg_component_fields(row):
        aggregate_ledger = _aggregate_s_alg_event_ledger(
            child_rows=child_rows,
            source_label=_path_key(path),
        )
        if aggregate_ledger is not None:
            row_for_enrichment = dict(row)
            row_for_enrichment["table_i_measurement_event_ledger"] = aggregate_ledger
        else:
            allow_runtime_reconstruction = False
            normalizer_source_payload = None
    enriched = normalize_snake_measurement_work_row(
        row_for_enrichment,
        source_payload=normalizer_source_payload,
        source_label=_path_key(path) if normalizer_source_payload is not None else None,
        allow_runtime_reconstruction=allow_runtime_reconstruction,
    )
    _record_enrichment_stats(enriched, stats)
    return enriched


def _enrich_mapping_at_path(
    payload: MutableMapping[str, Any],
    path: tuple[str, ...],
    stats: Counter[str],
    *,
    source_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    consumed_source_paths: set[str] | None = None,
) -> None:
    current: Any = payload
    for key in path[:-1]:
        if not isinstance(current, MutableMapping):
            return
        current = current.get(key)
    if not isinstance(current, MutableMapping):
        return
    key = path[-1]
    value = current.get(key)
    if not isinstance(value, Mapping):
        return
    source = _source_for_path(source_payloads, path, consumed_source_paths)
    enriched = normalize_snake_measurement_work_row(
        value,
        source_payload=source,
        source_label=_path_key(path) if source is not None else None,
    )
    current[key] = enriched
    _record_enrichment_stats(enriched, stats)


def _enrich_bosonic_per_benchmark(
    payload: MutableMapping[str, Any],
    stats: Counter[str],
    *,
    source_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    consumed_source_paths: set[str] | None = None,
) -> None:
    bosonic = payload.get("bosonic_snake_current_table_support")
    if not isinstance(bosonic, MutableMapping):
        return
    per_benchmark = bosonic.get("per_benchmark")
    enriched_children: list[Mapping[str, Any]] = []
    if isinstance(per_benchmark, MutableMapping):
        for benchmark_id, row in list(per_benchmark.items()):
            if not isinstance(row, Mapping):
                continue
            path = ("bosonic_snake_current_table_support", "per_benchmark", str(benchmark_id))
            source = _source_for_path(source_payloads, path, consumed_source_paths)
            enriched = _enrich_or_aggregate_row(
                row,
                path=path,
                stats=stats,
                source_payload=source,
            )
            per_benchmark[benchmark_id] = enriched
            enriched_children.append(enriched)
    path = ("bosonic_snake_current_table_support",)
    source = _source_for_path(source_payloads, path, consumed_source_paths)
    enriched = _enrich_or_aggregate_row(
        bosonic,
        path=path,
        stats=stats,
        child_rows=enriched_children if enriched_children else None,
        source_payload=source,
    )
    payload["bosonic_snake_current_table_support"] = enriched


def _enrich_mixed_inputs(
    payload: MutableMapping[str, Any],
    stats: Counter[str],
    *,
    source_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    consumed_source_paths: set[str] | None = None,
) -> None:
    mixed = payload.get("fermion_boson_snake_current_table_support")
    if not isinstance(mixed, MutableMapping):
        return
    inputs = mixed.get("inputs")
    enriched_children: list[Mapping[str, Any]] = []
    if isinstance(inputs, list):
        for idx, row in enumerate(list(inputs)):
            if not isinstance(row, Mapping):
                continue
            path = ("fermion_boson_snake_current_table_support", "inputs", str(idx))
            source = _source_for_path(source_payloads, path, consumed_source_paths)
            enriched = _enrich_or_aggregate_row(
                row,
                path=path,
                stats=stats,
                source_payload=source,
            )
            inputs[idx] = enriched
            enriched_children.append(enriched)
    aggregate = mixed.get("aggregate")
    if isinstance(aggregate, Mapping):
        path = ("fermion_boson_snake_current_table_support", "aggregate")
        source = _source_for_path(source_payloads, path, consumed_source_paths)
        enriched = _enrich_or_aggregate_row(
            aggregate,
            path=path,
            stats=stats,
            child_rows=enriched_children if enriched_children else None,
            source_payload=source,
        )
        mixed["aggregate"] = enriched


def enrich_snake_support_payload(
    payload: Mapping[str, Any],
    *,
    source_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Enrich known SNAKE Table-I support rows and return a new payload."""

    out = copy.deepcopy(dict(payload))
    stats: Counter[str] = Counter()
    consumed_source_paths: set[str] = set()
    for key in _TOP_LEVEL_SUPPORT_KEYS:
        _enrich_mapping_at_path(
            out,
            (key,),
            stats,
            source_payloads=source_payloads,
            consumed_source_paths=consumed_source_paths,
        )
    _enrich_bosonic_per_benchmark(
        out,
        stats,
        source_payloads=source_payloads,
        consumed_source_paths=consumed_source_paths,
    )
    _enrich_mixed_inputs(
        out,
        stats,
        source_payloads=source_payloads,
        consumed_source_paths=consumed_source_paths,
    )

    status_counts = {
        key.removeprefix("status:"): int(value)
        for key, value in sorted(stats.items())
        if key.startswith("status:")
    }
    s_alg_status_counts = {
        key.removeprefix("s_alg_status:"): int(value)
        for key, value in sorted(stats.items())
        if key.startswith("s_alg_status:")
    }
    s_phys_status_counts = {
        key.removeprefix("s_phys_status:"): int(value)
        for key, value in sorted(stats.items())
        if key.startswith("s_phys_status:")
    }
    s_l2_status_counts = {
        key.removeprefix("s_l2_status:"): int(value)
        for key, value in sorted(stats.items())
        if key.startswith("s_l2_status:")
    }
    s_grp_status_counts = {
        key.removeprefix("s_grp_status:"): int(value)
        for key, value in sorted(stats.items())
        if key.startswith("s_grp_status:")
    }
    runtime_reconstruction_status_counts = {
        key.removeprefix("runtime_reconstruction:"): int(value)
        for key, value in sorted(stats.items())
        if key.startswith("runtime_reconstruction:")
    }
    source_payload_keys = set(source_payloads or {})
    unmatched_source_payload_paths = sorted(source_payload_keys - consumed_source_paths)
    out["snake_measurement_work_normalization"] = {
        "schema": SCHEMA_VERSION,
        "normalized_measurement_work_schema": "normalized_measurement_work_v1",
        "algorithmic_measurement_work_schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
        "physical_measurement_work_schema": PHYSICAL_MEASUREMENT_WORK_SCHEMA,
        "grouped_measurement_proxy_schema": "grouped_measurement_proxy_v1",
        "source_map_schema": SOURCE_MAP_SCHEMA_VERSION,
        "policy": "legacy_s_norm_provenance_plus_strict_s_alg_event_telemetry_no_raw_scalar_promotion",
        "status_counts": status_counts,
        "s_alg_status_counts": s_alg_status_counts,
        "s_phys_status_counts": s_phys_status_counts,
        "s_l2_status_counts": s_l2_status_counts,
        "s_grp_status_counts": s_grp_status_counts,
        "runtime_reconstruction_status_counts": runtime_reconstruction_status_counts,
        "s_norm_available_count": int(stats.get("s_norm_available_count", 0)),
        "s_alg_available_count": int(stats.get("s_alg_available_count", 0)),
        "s_phys_available_count": int(stats.get("s_phys_available_count", 0)),
        "s_l2_available_count": int(stats.get("s_l2_available_count", 0)),
        "s_grp_available_count": int(stats.get("s_grp_available_count", 0)),
        "raw_fallback_count": int(stats.get("raw_fallback_count", 0)),
        "raw_fallback_forbidden_count": int(stats.get("raw_fallback_forbidden_count", 0)),
        "enriched_path_count": int(stats.get("enriched_path_count", 0)),
        "source_payload_count": int(len(source_payloads or {})),
        "source_payload_consumed_count": int(len(consumed_source_paths)),
        "unmatched_source_payload_paths": unmatched_source_payload_paths,
    }
    return out


def _load_source_payload_map(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    repo_root = Path(__file__).resolve().parents[2]
    if isinstance(payload, Mapping) and isinstance(payload.get("sources"), list):
        items = payload["sources"]
    elif isinstance(payload, Mapping):
        items = [
            {"support_path": str(key), "result_json": value}
            for key, value in payload.items()
            if key != "schema"
        ]
    else:
        raise SystemExit(f"invalid source map JSON: {path}")
    out: dict[str, Mapping[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise SystemExit(f"invalid source map item in {path}: {item!r}")
        support_path = str(item.get("support_path", "")).strip()
        result_json = item.get("result_json")
        if not support_path or result_json is None:
            raise SystemExit(f"source map item requires support_path and result_json: {item!r}")
        result_path = Path(str(result_json))
        if not result_path.is_absolute():
            map_relative = base / result_path
            repo_relative = repo_root / result_path
            if map_relative.exists():
                result_path = map_relative
            elif repo_relative.exists():
                result_path = repo_relative
            elif result_path.exists():
                # Last-resort compatibility for callers that intentionally pass
                # cwd-relative source maps from a scratch directory.
                result_path = result_path
            else:
                result_path = map_relative
        if not result_path.exists():
            raise SystemExit(f"source payload does not exist for {support_path}: {result_path}")
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise SystemExit(f"source payload is not a JSON object for {support_path}: {result_path}")
        out[support_path] = loaded
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--source-map-json",
        type=Path,
        help=(
            "Optional snake_table_i_source_payload_map_v1 JSON mapping support paths "
            "to detailed SNAKE result.json payloads for runtime S_norm reconstruction."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.in_place and args.output_json is not None:
        raise SystemExit("--in-place and --output-json are mutually exclusive")
    if not args.in_place and args.output_json is None:
        raise SystemExit("provide --output-json or --in-place")
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    source_payloads = _load_source_payload_map(args.source_map_json) if args.source_map_json else None
    enriched = enrich_snake_support_payload(payload, source_payloads=source_payloads)
    unmatched = enriched.get("snake_measurement_work_normalization", {}).get("unmatched_source_payload_paths", [])
    if source_payloads and unmatched:
        raise SystemExit(f"unmatched source payload paths: {', '.join(map(str, unmatched))}")
    output = args.input_json if args.in_place else args.output_json
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(enriched, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
