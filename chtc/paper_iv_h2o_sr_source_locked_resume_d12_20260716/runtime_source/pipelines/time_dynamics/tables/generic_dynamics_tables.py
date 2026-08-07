#!/usr/bin/env python3
"""Table-ready JSON aggregation for Hamiltonian-generic dynamics rows.

This module is intentionally data-only.  It consumes normalized
``dynamics_benchmark_row_v1`` rows and emits JSON fragments that reporting/TeX
layers can render later.  It does not run physics kernels, mutate controller
state, or write LaTeX.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DYNAMICS_TABLE_FIELD_KEYS,
    DYNAMICS_TABLE_BUNDLE_SCHEMA,
    DYNAMICS_TABLE_I_ALGORITHMS,
    DYNAMICS_CASE_METADATA_OVERRIDE_SOURCE,
    DYNAMICS_COARSE_TUNING_CLASSES,
    DYNAMICS_TUNING_GRANULARITY_CLASS,
    DynamicsBenchmarkRow,
    normalize_dynamics_tuning_class,
    dynamics_table_bundle_payload,
    json_safe,
    table_iv_prune_pilot_contract,
    validate_dynamics_metric_contract,
)
from pipelines.time_dynamics.tables.table_lock_contract import validate_same_seed_rows

DYNAMICS_TABLE_SUMMARY_SCHEMA = "generic_dynamics_table_summary_v1"
PAPER_II_TABLE_I_FINAL_CLOSEOUT_GATE_SCHEMA = "paper_ii_table_i_final_aggregation_promotion_gate_v1"
DYN_CLAIMS_TABLE_ID = "tab:dyn_claims"
DYN_ABLATION_MATRIX_TABLE_ID = "tab:dyn_ablation_matrix"
FULL_CONTROLLER_ABLATION_VARIANT = "full_controller"
FULL_CONTROLLER_ALGORITHM_ID = "dyn_controller_full"
_CONTROLLER_ABLATION_ALGORITHM_VARIANTS: dict[str, str] = {
    FULL_CONTROLLER_ALGORITHM_ID: FULL_CONTROLLER_ABLATION_VARIANT,
    "dyn_controller_fixed_scaffold": "fixed_scaffold",
    "dyn_controller_no_append": "no_append",
    "dyn_controller_no_pruning": "no_pruning",
    "dyn_controller_fixed_integrator": "fixed_integrator_policy",
    "dyn_controller_no_residual_split": "no_residual_split",
}

_PAPER_II_REQUIRED_ROW_CHECKS: dict[str, tuple[tuple[str, str], ...]] = {
    "dyn_controller_full": (
        ("strict_decision_contract_passed", "strict exact-free decision contract"),
        ("non_frozen_trajectory_check_passed", "non-frozen controller trajectory when exact diagnostics move"),
    ),
    "dyn_fixed_mclachlan": (
        ("qiskit_parity_passed", "fixed scaffold/state/observable Qiskit parity"),
        ("mclachlan_correctness_passed", "McLachlan metric/RHS/solve/integrator correctness, including non-frozen invariants"),
    ),
    "dyn_product_formula_envelope": (("qiskit_parity_passed", "product-formula Qiskit parity"),),
    "dyn_qdrift": (("qiskit_parity_passed", "qDRIFT realized-sample Qiskit parity"),),
    "dyn_fixed_pvqd": (("qiskit_parity_passed", "fixed-pVQD component Qiskit parity"),),
    "dyn_adaptive_pvqd": (("qiskit_parity_passed", "adaptive-pVQD component Qiskit parity"),),
    "dyn_avqds": (("avqds_correctness_passed", "AVQDS dense/component correctness"),),
    "dyn_avqds_t": (("avqds_t_correctness_passed", "AVQDS-T dense/component correctness"),),
}

_PAPER_II_REQUIRED_AGGREGATE_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("epsilon_spec", "epsilon_spec_mean"),
    ("shots_total", "shots_total_mean"),
    ("fidelity", "one_minus_min_fidelity_exact_mean"),
)
_PAPER_II_EXPECTED_SNAKE_SAME_SEED_GROUP_COUNT = 20
_PAPER_II_EXPECTED_FAMILY_COUNT = 10

_DYN_CLAIMS_COLUMNS: tuple[str, ...] = (
    "table_class",
    "source_table_classes",
    "algorithm_id",
    "method_label",
    "completed_case_count",
    "skipped_case_count",
    "mean_abs_energy_total_error_mean",
    "epsilon_obs_2_mean",
    "one_minus_min_fidelity_exact_mean",
    "epsilon_spec_mean",
    "compiled_count_2q_total_mean",
    "compiled_depth_2q_total_mean",
    "compiled_depth_total_mean",
    "shots_total_mean",
    "families",
    "case_ids",
    "aggregate_case_ids",
    "excluded_case_ids",
    "status_counts",
    "tuning_granularity",
    "tuning_class",
    "settings_source_values",
    "settings_id_values",
    "controller_settings_id_values",
    "comparator_settings_id_values",
    "static_scaffold_scope_values",
    "same_seed_comparator_group_ids",
    "same_seed_validation_status",
    "seed_artifact_hash_values",
    "class_settings_lock_manifest_values",
    "source_json_paths",
    "class_settings_selected_trial_numbers",
    "class_tuned_result_locked",
    "tuning_validation_status",
)

_DYN_ABLATION_COLUMNS: tuple[str, ...] = (
    "ablation_group_id",
    "table_class",
    "family",
    "case_id",
    "variant_id",
    "disabled_feature",
    "algorithm_id",
    "method_label",
    "status",
    "reason",
    "paired_full_status",
    "paired_with_full",
    "mean_abs_energy_total_error",
    "delta_mean_abs_energy_total_error_disabled_minus_full",
    "epsilon_obs_2",
    "delta_epsilon_obs_2_disabled_minus_full",
    "compiled_count_2q_total",
    "delta_compiled_count_2q_total_disabled_minus_full",
    "compiled_depth_2q_total",
    "delta_compiled_depth_2q_total_disabled_minus_full",
    "compiled_depth_total",
    "delta_compiled_depth_total_disabled_minus_full",
    "shots_total",
    "delta_shots_total_disabled_minus_full",
    "append_count",
    "delta_append_count_disabled_minus_full",
    "prune_count",
    "delta_prune_count_disabled_minus_full",
    "final_runtime_parameter_count",
    "delta_final_runtime_parameter_count_disabled_minus_full",
    "strict_decision_contract_passed",
    "tuning_granularity",
    "tuning_class",
    "settings_source_values",
    "settings_id_values",
    "controller_settings_id_values",
    "comparator_settings_id_values",
    "static_scaffold_scope_values",
    "same_seed_comparator_group_ids",
    "same_seed_validation_status",
    "seed_artifact_hash_values",
    "class_settings_lock_manifest_values",
    "source_json_paths",
    "class_settings_selected_trial_numbers",
    "class_tuned_result_locked",
    "tuning_validation_status",
)

_ABLATION_DELTA_KEYS: tuple[str, ...] = (
    "mean_abs_energy_total_error",
    "epsilon_obs_2",
    "compiled_count_2q_total",
    "compiled_depth_2q_total",
    "compiled_depth_total",
    "shots_total",
    "append_count",
    "prune_count",
    "final_runtime_parameter_count",
)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _row_dict(row: DynamicsBenchmarkRow | Mapping[str, Any]) -> dict[str, Any]:
    return row.to_dict() if isinstance(row, DynamicsBenchmarkRow) else dict(row)


def _row_dicts(rows: Sequence[DynamicsBenchmarkRow | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_row_dict(row) for row in rows]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_values(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        maybe = _float_or_none(value)
        if maybe is not None:
            out.append(float(maybe))
    return out


def _mean_or_none(values: Sequence[Any]) -> float | None:
    finite = _finite_values(values)
    return None if not finite else float(sum(finite) / len(finite))


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = int(counts.get(status, 0)) + 1
    return dict(sorted(counts.items()))


def _metric_contract_validation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validations = [validate_dynamics_metric_contract(row, strict=False) for row in rows]
    return json_safe(
        {
            "schema": "dynamics_metric_contract_validation_summary_v1",
            "row_count": int(len(validations)),
            "passed_count": int(sum(1 for item in validations if item.get("passed"))),
            "failed_count": int(sum(1 for item in validations if not item.get("passed"))),
            "failed_rows": [
                {
                    "index": int(index),
                    "algorithm_id": rows[index].get("algorithm_id"),
                    "case_id": rows[index].get("case_id"),
                    "violations": item.get("violations", []),
                }
                for index, item in enumerate(validations)
                if not item.get("passed")
            ],
        }
    )


def _qiskit_parity_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    modes: list[str] = []
    failed: list[dict[str, Any]] = []
    requested = 0
    passed = 0
    for row in rows:
        metrics = _metrics(row)
        provenance = _provenance(row)
        status = metrics.get("qiskit_parity_status", provenance.get("qiskit_parity_status"))
        if status in {None, ""}:
            continue
        requested += 1
        status_text = str(status)
        status_counts[status_text] = int(status_counts.get(status_text, 0)) + 1
        mode = metrics.get("qiskit_parity_mode", provenance.get("qiskit_parity_mode"))
        if mode not in {None, ""} and str(mode) not in modes:
            modes.append(str(mode))
        if metrics.get("qiskit_parity_passed", provenance.get("qiskit_parity_passed")) is True:
            passed += 1
        elif status_text not in {"not_applicable", "skipped_optional_dependency", "skipped_resource_guard"}:
            failed.append(
                {
                    "algorithm_id": row.get("algorithm_id"),
                    "case_id": row.get("case_id"),
                    "status": status_text,
                    "passed": metrics.get("qiskit_parity_passed", provenance.get("qiskit_parity_passed")),
                }
            )
    return json_safe(
        {
            "schema": "dynamics_qiskit_parity_summary_v1",
            "requested_row_count": int(requested),
            "passed_count": int(passed),
            "status_counts": dict(sorted(status_counts.items())),
            "modes": sorted(modes),
            "failed_rows": failed,
            "policy": "parity sidecars are additive and do not overwrite native table fields",
        }
    )


def _row_gate_value(row: Mapping[str, Any], key: str) -> Any:
    metrics = _metrics(row)
    provenance = _provenance(row)
    for source in (metrics, provenance, row):
        if key in source:
            return source.get(key)
    nested = provenance.get("tuning_provenance", {})
    if isinstance(nested, Mapping) and key in nested:
        return nested.get(key)
    return None


def _paper_ii_row_parity_correctness_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row_statuses: list[dict[str, Any]] = []
    failed_or_missing: list[dict[str, Any]] = []
    for row in rows:
        algorithm_id = str(row.get("algorithm_id", ""))
        checks = _PAPER_II_REQUIRED_ROW_CHECKS.get(algorithm_id, ())
        if not checks:
            continue
        check_statuses: list[dict[str, Any]] = []
        for key, description in checks:
            value = _row_gate_value(row, key)
            passed = value is True
            check = {
                "key": key,
                "description": description,
                "passed": bool(passed),
                "value": value,
            }
            check_statuses.append(check)
            if not passed:
                failed_or_missing.append(
                    {
                        "algorithm_id": algorithm_id,
                        "case_id": row.get("case_id"),
                        "status": row.get("status"),
                        "check_key": key,
                        "description": description,
                        "value": value,
                    }
                )
        row_statuses.append(
            {
                "algorithm_id": algorithm_id,
                "case_id": row.get("case_id"),
                "status": row.get("status"),
                "checks": check_statuses,
                "passed": all(item["passed"] for item in check_statuses),
            }
        )
    return json_safe(
        {
            "schema": "paper_ii_row_parity_correctness_summary_v1",
            "required_algorithm_ids": list(_PAPER_II_REQUIRED_ROW_CHECKS),
            "checked_row_count": int(len(row_statuses)),
            "failed_or_missing_count": int(len(failed_or_missing)),
            "passed": not failed_or_missing,
            "failed_or_missing": failed_or_missing,
            "rows": row_statuses,
            "policy": "required parity/correctness checks are additive provenance and must pass before final Table-I use",
        }
    )


def _paper_ii_algorithm_coverage_summary(claim_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = list(DYNAMICS_TABLE_I_ALGORITHMS)
    by_table_class: dict[str, set[str]] = {}
    for row in claim_rows:
        table_class = str(row.get("table_class", "")) or "unclassified"
        by_table_class.setdefault(table_class, set()).add(str(row.get("algorithm_id", "")))
    class_summaries: list[dict[str, Any]] = []
    missing_total = 0
    for table_class, present in sorted(by_table_class.items()):
        missing = [algorithm for algorithm in required if algorithm not in present]
        missing_total += len(missing)
        class_summaries.append(
            {
                "table_class": table_class,
                "present_algorithm_ids": sorted(present),
                "missing_algorithm_ids": missing,
                "complete": not missing,
            }
        )
    return json_safe(
        {
            "schema": "paper_ii_table_i_algorithm_coverage_summary_v1",
            "required_algorithm_ids": required,
            "table_class_count": int(len(class_summaries)),
            "missing_algorithm_entry_count": int(missing_total),
            "complete": bool(class_summaries) and missing_total == 0,
            "table_classes": class_summaries,
        }
    )


def _paper_ii_same_seed_group_coverage_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_same_seed_group_count: int,
) -> dict[str, Any]:
    required = list(DYNAMICS_TABLE_I_ALGORITHMS)
    groups: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        group_id = _seed_value(row, "same_seed_comparator_group_id")
        if group_id in {None, ""}:
            continue
        group = groups.setdefault(str(group_id), {"algorithms": set(), "families": set(), "tuning_classes": set()})
        group["algorithms"].add(str(row.get("algorithm_id", "")))
        group["families"].add(str(row.get("family", "")))
        tuning_class = _tuning_value(row, "tuning_class")
        if tuning_class not in {None, ""}:
            group["tuning_classes"].add(str(tuning_class))
    group_summaries: list[dict[str, Any]] = []
    missing_total = 0
    for group_id, details in sorted(groups.items()):
        present = details["algorithms"]
        missing = [algorithm for algorithm in required if algorithm not in present]
        missing_total += len(missing)
        group_summaries.append(
            {
                "same_seed_comparator_group_id": group_id,
                "present_algorithm_ids": sorted(present),
                "missing_algorithm_ids": missing,
                "families": sorted(details["families"]),
                "tuning_classes": sorted(details["tuning_classes"]),
                "complete": not missing,
            }
        )
    group_count = len(group_summaries)
    return json_safe(
        {
            "schema": "paper_ii_same_seed_group_algorithm_coverage_summary_v1",
            "required_algorithm_ids": required,
            "expected_same_seed_group_count": int(expected_same_seed_group_count),
            "same_seed_group_count": int(group_count),
            "missing_group_count": max(0, int(expected_same_seed_group_count) - int(group_count)),
            "missing_algorithm_entry_count": int(missing_total),
            "complete": group_count == int(expected_same_seed_group_count) and missing_total == 0,
            "groups": group_summaries,
        }
    )


def _paper_ii_missing_evidence_summary(
    claim_rows: Sequence[Mapping[str, Any]],
    *,
    source_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    for row in claim_rows:
        for field, aggregate_field in _PAPER_II_REQUIRED_AGGREGATE_EVIDENCE_FIELDS:
            if row.get(aggregate_field) is None:
                missing.append(
                    {
                        "scope": "aggregate",
                        "table_class": row.get("table_class"),
                        "algorithm_id": row.get("algorithm_id"),
                        "field": field,
                        "aggregate_field": aggregate_field,
                        "status": "marked_unavailable_json_null",
                    }
                )
    raw_field_by_name = {
        "epsilon_spec": "epsilon_spec",
        "shots_total": "shots_total",
        "fidelity": "one_minus_min_fidelity_exact",
    }
    for row in source_rows:
        for field, raw_field in raw_field_by_name.items():
            if _numeric_cell(row, raw_field) is None:
                missing.append(
                    {
                        "scope": "source_row",
                        "case_id": row.get("case_id"),
                        "family": row.get("family"),
                        "algorithm_id": row.get("algorithm_id"),
                        "field": field,
                        "raw_field": raw_field,
                        "status": "marked_unavailable_json_null",
                    }
                )
    return json_safe(
        {
            "schema": "paper_ii_missing_evidence_summary_v1",
            "required_fields": [field for field, _ in _PAPER_II_REQUIRED_AGGREGATE_EVIDENCE_FIELDS],
            "missing_count": int(len(missing)),
            "missing": missing,
            "status": "complete" if not missing else "marked_unavailable_requires_user_acceptance_before_promotion",
            "policy": "missing epsilon_spec, shots_total, or fidelity evidence is reported as unavailable; values are never invented",
        }
    )


def _valid_metric_contract_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if bool(validate_dynamics_metric_contract(row, strict=False).get("passed"))
    ]


def _table_fields(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("table_fields", {})
    return value if isinstance(value, Mapping) else {}


def _metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metrics", {})
    return value if isinstance(value, Mapping) else {}


def _resources(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("resources", {})
    return value if isinstance(value, Mapping) else {}


def _provenance(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("provenance", {})
    return value if isinstance(value, Mapping) else {}


def _tuning_value(row: Mapping[str, Any], key: str) -> Any:
    prov = _provenance(row)
    value: Any = None
    if key in prov:
        value = prov.get(key)
    else:
        nested = prov.get("tuning_provenance", {})
        if isinstance(nested, Mapping):
            value = nested.get(key)
    if key == "tuning_class":
        return normalize_dynamics_tuning_class(value)
    return value


def _unique_tuning_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    for row in rows:
        value = _tuning_value(row, key)
        if value in {None, ""}:
            continue
        if value not in values:
            values.append(value)
    return sorted(values, key=lambda item: str(item))


def _seed_value(row: Mapping[str, Any], key: str) -> Any:
    prov = _provenance(row)
    if key in prov:
        return prov.get(key)
    seed_lock = prov.get("seed_lock", {})
    if isinstance(seed_lock, Mapping):
        return seed_lock.get(key)
    return None


def _unique_seed_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    for row in rows:
        value = _seed_value(row, key)
        if value in {None, ""}:
            continue
        if value not in values:
            values.append(value)
    return sorted(values, key=lambda item: str(item))


def _source_json_paths(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        prov = _provenance(row)
        for value in (
            row.get("source_json_path"),
            prov.get("source_json_path"),
            prov.get("source_row_json"),
            prov.get("source_result_json"),
        ):
            if value in {None, ""}:
                continue
            text = str(value)
            if text not in values:
                values.append(text)
    return sorted(values)


def _class_settings_manifest_values(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        prov = _provenance(row)
        meta = prov.get("case_metadata", {}) if isinstance(prov.get("case_metadata", {}), Mapping) else {}
        for value in (
            _tuning_value(row, "class_settings_lock_manifest"),
            prov.get("class_settings_lock_manifest"),
            meta.get("class_settings_lock_manifest"),
        ):
            if value in {None, ""}:
                continue
            text = str(value)
            if text not in values:
                values.append(text)
    return sorted(values)


def _selected_trial_numbers(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    values: list[int] = []
    for row in rows:
        value = _tuning_value(row, "class_settings_selected_trial_number")
        if value in {None, ""}:
            continue
        try:
            number = int(value)
        except Exception:
            continue
        if number not in values:
            values.append(number)
    return sorted(values)


def _same_seed_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validation = validate_same_seed_rows(rows)
    if not rows:
        status = "no_rows"
    elif not _unique_seed_values(rows, "same_seed_comparator_group_id"):
        status = "missing_same_seed_group"
    elif not _unique_seed_values(rows, "static_seed_artifact_sha256"):
        status = "same_seed_group_without_hash"
    elif bool(validation.get("passed", False)):
        status = "same_seed_locked"
    else:
        status = "same_seed_mismatch"
    return {
        "same_seed_comparator_group_ids": _unique_seed_values(rows, "same_seed_comparator_group_id"),
        "same_seed_validation_status": status,
        "seed_artifact_hash_values": _unique_seed_values(rows, "static_seed_artifact_sha256"),
    }



def _tuning_validation_status(table_class: str, rows: Sequence[Mapping[str, Any]]) -> str:
    completed = [row for row in rows if str(row.get("status")) == "completed"]
    if not completed:
        return "no_completed_rows"
    if any(not _tuning_value(row, "settings_id") for row in completed):
        return "provisional_missing_tuning_provenance"
    granularities = set(str(value) for value in _unique_tuning_values(completed, "tuning_granularity"))
    if granularities and granularities != {DYNAMICS_TUNING_GRANULARITY_CLASS}:
        return "inconsistent_tuning_granularity"
    classes = set(str(value) for value in _unique_tuning_values(completed, "tuning_class"))
    if not classes:
        return "provisional_missing_tuning_provenance"
    if any(item not in DYNAMICS_COARSE_TUNING_CLASSES for item in classes):
        return "inconsistent_tuning_class"
    if str(table_class) != "all_classes" and classes != {str(table_class)}:
        return "inconsistent_tuning_class_within_group"
    if len(_unique_tuning_values(completed, "settings_id")) > 1:
        return "inconsistent_tuning_settings_within_class"
    sources = set(str(value) for value in _unique_tuning_values(completed, "settings_source"))
    statuses = set(str(value) for value in _unique_tuning_values(completed, "tuning_validation_status"))
    if DYNAMICS_CASE_METADATA_OVERRIDE_SOURCE in sources or "provisional_case_metadata_override" in statuses:
        return "provisional_case_metadata_override"
    if not completed or not all(bool(_tuning_value(row, "class_tuned_result_locked")) for row in completed):
        return "provisional_unlocked_coarse_class"
    return "locked_coarse_class_tuned" if len(classes) == 1 else "locked_multi_coarse_class_tuned"


def _tuning_summary(table_class: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if str(row.get("status")) == "completed"]
    summary = {
        "tuning_granularity": ",".join(str(value) for value in _unique_tuning_values(rows, "tuning_granularity")),
        "tuning_class": ",".join(str(value) for value in _unique_tuning_values(rows, "tuning_class")),
        "settings_source_values": _unique_tuning_values(rows, "settings_source"),
        "settings_id_values": _unique_tuning_values(rows, "settings_id"),
        "controller_settings_id_values": _unique_tuning_values(rows, "controller_settings_id"),
        "comparator_settings_id_values": _unique_tuning_values(rows, "comparator_settings_id"),
        "static_scaffold_scope_values": _unique_tuning_values(rows, "static_scaffold_scope"),
        "class_tuned_result_locked": bool(completed)
        and all(bool(_tuning_value(row, "class_tuned_result_locked")) for row in completed),
        "tuning_validation_status": _tuning_validation_status(table_class, rows),
        "class_settings_lock_manifest_values": _class_settings_manifest_values(rows),
        "source_json_paths": _source_json_paths(rows),
        "class_settings_selected_trial_numbers": _selected_trial_numbers(rows),
    }
    summary.update(_same_seed_summary(rows))
    return summary


def _case_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _provenance(row).get("case_metadata", {})
    return value if isinstance(value, Mapping) else {}


def _exclude_from_main_aggregate(row: Mapping[str, Any]) -> bool:
    meta = _case_metadata(row)
    value = meta.get("exclude_from_main_dynamics_aggregate", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "exclude"}
    return bool(value)


def _numeric_cell(row: Mapping[str, Any], key: str) -> float | int | None:
    table_fields = _table_fields(row)
    metrics = _metrics(row)
    resources = _resources(row)
    for source in (table_fields, metrics, resources):
        if key in source:
            if key in {
                "compiled_count_2q_total",
                "compiled_depth_2q_total",
                "compiled_depth_total",
                "shots_total",
                "append_count",
                "prune_count",
                "final_runtime_parameter_count",
            }:
                return _int_or_none(source.get(key))
            return _float_or_none(source.get(key))
    return None


def _group_key_for_claim(row: Mapping[str, Any]) -> tuple[str, str]:
    tuning_class = _tuning_value(row, "tuning_class")
    table_class = str(tuning_class or row.get("table_class", "unclassified"))
    return (table_class, str(row.get("algorithm_id", "")))


def build_dyn_claims_table(
    rows: Sequence[DynamicsBenchmarkRow | Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate generic dynamics rows into the manuscript claims table shape."""

    source_row_dicts = _row_dicts(rows)
    metric_contract_validation = _metric_contract_validation_summary(source_row_dicts)
    row_dicts = _valid_metric_contract_rows(source_row_dicts)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in row_dicts:
        if _is_disabled_ablation(row):
            continue
        grouped.setdefault(_group_key_for_claim(row), []).append(row)

    table_rows: list[dict[str, Any]] = []
    for (table_class, algorithm_id), group_rows in sorted(grouped.items()):
        completed = [row for row in group_rows if str(row.get("status")) == "completed"]
        main_completed = [row for row in completed if not _exclude_from_main_aggregate(row)]
        excluded_completed = [row for row in completed if _exclude_from_main_aggregate(row)]
        aggregate_rows = main_completed if main_completed else completed
        first = aggregate_rows[0] if aggregate_rows else group_rows[0]
        status_counts = _status_counts(group_rows)
        output_row = {
            "table_class": str(table_class),
            "source_table_classes": sorted({str(row.get("table_class", "")) for row in group_rows}),
            "algorithm_id": str(algorithm_id),
            "method_label": str(first.get("method_label", algorithm_id)),
            "completed_case_count": int(len(main_completed)),
            "excluded_completed_case_count": int(len(excluded_completed)),
            "skipped_case_count": int(len(group_rows) - len(completed)),
            "families": sorted({str(row.get("family", "")) for row in group_rows}),
            "case_ids": sorted({str(row.get("case_id", "")) for row in group_rows}),
            "aggregate_case_ids": sorted({str(row.get("case_id", "")) for row in aggregate_rows}),
            "excluded_case_ids": sorted({str(row.get("case_id", "")) for row in excluded_completed}),
            "aggregate_exclusion_reasons": sorted(
                {
                    str(_case_metadata(row).get("aggregate_exclusion_reason", ""))
                    for row in excluded_completed
                    if _case_metadata(row).get("aggregate_exclusion_reason", "") not in {None, ""}
                }
            ),
            "status_counts": status_counts,
        }
        for key in DYNAMICS_TABLE_FIELD_KEYS:
            if key == "table_status_label":
                continue
            output_row[f"{key}_mean"] = _mean_or_none([_numeric_cell(row, key) for row in aggregate_rows])
        output_row.update(_tuning_summary(str(table_class), aggregate_rows))
        table_rows.append(output_row)

    return json_safe(
        {
            "schema": DYNAMICS_TABLE_SUMMARY_SCHEMA,
            "table_id": DYN_CLAIMS_TABLE_ID,
            "caption_label": DYN_CLAIMS_TABLE_ID,
            "source_row_schema": "dynamics_benchmark_row_v1",
            "columns": list(_DYN_CLAIMS_COLUMNS),
            "row_count": int(len(table_rows)),
            "source_row_count": int(len(source_row_dicts)),
            "validated_source_row_count": int(len(row_dicts)),
            "status_counts": _status_counts(row_dicts),
            "metric_contract_validation": metric_contract_validation,
            "qiskit_parity_summary": _qiskit_parity_summary(row_dicts),
            "aggregation": (
                "mean_over_completed_non_excluded_rows_grouped_by_table_class_and_algorithm; "
                "completed rows with provenance.case_metadata.exclude_from_main_dynamics_aggregate "
                "are preserved as stress rows but excluded from main class aggregates; "
                "rows failing metric_contract_validation are quarantined from aggregates"
            ),
            "rows": table_rows,
        }
    )


def paper_ii_table_i_final_closeout_gate(
    rows: Sequence[DynamicsBenchmarkRow | Mapping[str, Any]],
    *,
    claims_table: Mapping[str, Any] | None = None,
    post_fetch_audit_passed: bool = False,
    family_repair_tranche_pdfs_user_approved: bool = False,
    final_aggregation_user_approved: bool = False,
    table_promotion_user_approved: bool = False,
    missing_evidence_user_accepted: bool = False,
    expected_same_seed_group_count: int = _PAPER_II_EXPECTED_SNAKE_SAME_SEED_GROUP_COUNT,
    expected_family_count: int = _PAPER_II_EXPECTED_FAMILY_COUNT,
    expected_tuning_classes: Sequence[str] = DYNAMICS_COARSE_TUNING_CLASSES,
) -> dict[str, Any]:
    """Return the fail-closed final Table-I aggregation/promotion gate.

    This is deliberately data-only.  It summarizes whether already-normalized
    rows are eligible for a later user-approved final aggregation/promotion
    workflow; it never promotes settings, rewrites aggregates, or edits LaTeX.
    """

    row_dicts = _row_dicts(rows)
    claims = dict(claims_table) if claims_table is not None else build_dyn_claims_table(row_dicts)
    claim_rows = [dict(row) for row in claims.get("rows", []) if isinstance(row, Mapping)]
    blockers: list[dict[str, Any]] = []
    promotion_blockers: list[dict[str, Any]] = []

    def _block(gate_id: str, reason: str, *, scope: str = "final_aggregation") -> None:
        target = promotion_blockers if scope == "table_promotion" else blockers
        target.append({"gate_id": gate_id, "reason": reason, "severity": "ERROR"})

    if not post_fetch_audit_passed:
        _block("post_fetch_audit_no_blocking_errors", "post-fetch audit has not been recorded as passed")
    if not family_repair_tranche_pdfs_user_approved:
        _block(
            "family_repair_tranche_pdfs_user_visual_approval",
            "family repair tranche PDFs have not been recorded as user-approved",
        )
    if not final_aggregation_user_approved:
        _block("final_aggregation_user_approval", "final aggregation still requires explicit user approval")
    if not table_promotion_user_approved:
        _block(
            "separate_table_promotion_user_approval",
            "manuscript/table promotion requires a separate explicit user-approved workflow",
            scope="table_promotion",
        )

    metric_contract = claims.get("metric_contract_validation", {})
    if isinstance(metric_contract, Mapping) and int(metric_contract.get("failed_count", 0) or 0) > 0:
        _block("metric_contract_validation", "one or more source rows failed the dynamics metric contract")

    status_counts = _status_counts(row_dicts)
    if int(status_counts.get("failed", 0)) > 0:
        _block("failed_rows_present", "failed rows remain in the source row set")
    non_completed = [
        {"algorithm_id": row.get("algorithm_id"), "case_id": row.get("case_id"), "status": row.get("status")}
        for row in row_dicts
        if str(row.get("status")) != "completed"
    ]
    if non_completed:
        _block("non_completed_rows_present", "all rows must be completed before final Table-I aggregation")

    same_seed_validation = validate_same_seed_rows(row_dicts)
    same_seed_groups = _unique_seed_values(row_dicts, "same_seed_comparator_group_id")
    if not same_seed_groups:
        _block("same_seed_validation", "source rows lack same_seed_comparator_group_id")
    elif not bool(same_seed_validation.get("passed", False)):
        _block("same_seed_validation", "same-seed hash validation did not pass for all groups")

    coverage = _paper_ii_algorithm_coverage_summary(claim_rows)
    if not bool(coverage.get("complete", False)):
        _block("all_table_i_algorithms_present", "one or more Table-I algorithms are missing from a class aggregate")
    group_coverage = _paper_ii_same_seed_group_coverage_summary(
        row_dicts,
        expected_same_seed_group_count=expected_same_seed_group_count,
    )
    if not bool(group_coverage.get("complete", False)):
        _block("all_same_seed_groups_have_all_algorithms", "the full SNAKE same-seed group surface is incomplete")
    families = sorted({str(row.get("family", "")) for row in row_dicts if row.get("family") not in {None, ""}})
    if len(families) != int(expected_family_count):
        _block("all_snake_families_present", "the final gate does not see the expected SNAKE family count")
    tuning_classes = sorted({str(value) for value in _unique_tuning_values(row_dicts, "tuning_class")})
    expected_classes = sorted(str(value) for value in expected_tuning_classes)
    if tuning_classes != expected_classes:
        _block("all_tuning_classes_present", "the final gate does not see all expected coarse tuning classes")

    row_checks = _paper_ii_row_parity_correctness_summary(row_dicts)
    if not bool(row_checks.get("passed", False)):
        _block("required_parity_correctness_sidecars_passing", "required parity/correctness checks are missing or failed")

    bad_tuning_rows = [
        {
            "table_class": row.get("table_class"),
            "algorithm_id": row.get("algorithm_id"),
            "tuning_validation_status": row.get("tuning_validation_status"),
            "class_tuned_result_locked": row.get("class_tuned_result_locked"),
        }
        for row in claim_rows
        if str(row.get("tuning_validation_status", "")) not in {
            "locked_coarse_class_tuned",
            "locked_multi_coarse_class_tuned",
        }
        or row.get("class_tuned_result_locked") is not True
    ]
    candidate_tuning_rows = []
    for row in row_dicts:
        candidate_only = _tuning_value(row, "class_settings_candidate_only_not_promoted")
        entry_lock_status = _tuning_value(row, "class_settings_entry_lock_status")
        entry_promotion_status = _tuning_value(row, "class_settings_entry_promotion_status")
        if (
            candidate_only is True
            or "candidate" in str(entry_lock_status or "").lower()
            or "not_promoted" in str(entry_promotion_status or "").lower()
        ):
            candidate_tuning_rows.append(
                {
                    "algorithm_id": row.get("algorithm_id"),
                    "case_id": row.get("case_id"),
                    "class_settings_candidate_only_not_promoted": candidate_only,
                    "class_settings_entry_lock_status": entry_lock_status,
                    "class_settings_entry_promotion_status": entry_promotion_status,
                }
            )
    if bad_tuning_rows:
        _block("approved_class_level_settings", "one or more class aggregates lack approved locked class-level settings")
    if candidate_tuning_rows:
        _block("candidate_class_settings_not_promoted", "candidate/not-promoted class settings cannot pass the final gate")

    missing_evidence = _paper_ii_missing_evidence_summary(claim_rows, source_rows=row_dicts)
    if int(missing_evidence.get("missing_count", 0) or 0) > 0 and not missing_evidence_user_accepted:
        _block(
            "missing_evidence_user_acceptance",
            "missing epsilon_spec, shots_total, or fidelity evidence requires explicit user acceptance before promotion",
            scope="table_promotion",
        )
    final_aggregation_allowed = not blockers
    table_promotion_allowed = final_aggregation_allowed and not promotion_blockers
    status = "blocked"
    if final_aggregation_allowed and table_promotion_allowed:
        status = "table_promotion_allowed"
    elif final_aggregation_allowed:
        status = "final_aggregation_allowed_table_promotion_blocked"

    return json_safe(
        {
            "schema": PAPER_II_TABLE_I_FINAL_CLOSEOUT_GATE_SCHEMA,
            "status": status,
            "final_aggregation_allowed": bool(final_aggregation_allowed),
            "table_promotion_allowed": bool(table_promotion_allowed),
            "blockers": blockers,
            "promotion_blockers": promotion_blockers,
            "post_fetch_audit_passed": bool(post_fetch_audit_passed),
            "family_repair_tranche_pdfs_user_approved": bool(family_repair_tranche_pdfs_user_approved),
            "final_aggregation_user_approved": bool(final_aggregation_user_approved),
            "table_promotion_user_approved": bool(table_promotion_user_approved),
            "missing_evidence_user_accepted": bool(missing_evidence_user_accepted),
            "expected_same_seed_group_count": int(expected_same_seed_group_count),
            "expected_family_count": int(expected_family_count),
            "expected_tuning_classes": expected_classes,
            "same_seed_validation": same_seed_validation,
            "algorithm_coverage": coverage,
            "same_seed_group_algorithm_coverage": group_coverage,
            "families": families,
            "tuning_classes": tuning_classes,
            "parity_correctness": row_checks,
            "bad_tuning_rows": bad_tuning_rows,
            "candidate_tuning_rows": candidate_tuning_rows,
            "non_completed_rows": non_completed,
            "missing_evidence": missing_evidence,
            "aggregate_separation_policy": {
                "physical_errors": "exact diagnostic/reference error fields stay separate from parity diagnostics",
                "qiskit_parity_deltas": "implementation parity only, never physical accuracy",
                "dense_component_correctness": "native correctness sidecars remain additive provenance",
                "resource_cost_fields": "compiled cost and shot/work proxies stay distinct from physical errors",
                "missing_evidence": "epsilon_spec, shots_total, and fidelity gaps are JSON null/unavailable, not invented",
            },
            "separate_explicit_workflow_required_for": [
                "settings_promotion",
                "final_table_value_promotion",
                "manuscript_tex_edits",
                "visible_table_edits",
            ],
            "policy": "fail closed: this summary is a gate for later review and does not perform aggregation promotion",
        }
    )



def _ablation_variant(row: Mapping[str, Any]) -> str | None:
    prov = _provenance(row)
    metrics = _metrics(row)
    for source in (prov, metrics, row):
        value = source.get("ablation_variant", source.get("variant_id", None))
        if value not in {None, ""}:
            return str(value)
    algorithm_id = str(row.get("algorithm_id", ""))
    return _CONTROLLER_ABLATION_ALGORITHM_VARIANTS.get(algorithm_id)


def _ablation_group_id(row: Mapping[str, Any]) -> str:
    prov = _provenance(row)
    value = prov.get("ablation_group_id", None)
    if value not in {None, ""}:
        return str(value)
    return f"{row.get('table_class', 'unclassified')}::{row.get('family', '')}::{row.get('case_id', '')}"


def _is_full_ablation(row: Mapping[str, Any]) -> bool:
    variant = _ablation_variant(row)
    return bool(
        variant == FULL_CONTROLLER_ABLATION_VARIANT
        or str(row.get("algorithm_id", "")) == FULL_CONTROLLER_ALGORITHM_ID
    )


def _is_disabled_ablation(row: Mapping[str, Any]) -> bool:
    variant = _ablation_variant(row)
    return bool(variant is not None and variant != FULL_CONTROLLER_ABLATION_VARIANT)


def _paired_delta(
    row: Mapping[str, Any],
    full: Mapping[str, Any] | None,
    key: str,
) -> float | int | None:
    if full is None:
        return None
    lhs = _numeric_cell(row, key)
    rhs = _numeric_cell(full, key)
    if lhs is None or rhs is None:
        return None
    delta = float(lhs) - float(rhs)
    if key in {
        "compiled_count_2q_total",
        "compiled_depth_2q_total",
        "compiled_depth_total",
        "shots_total",
        "append_count",
        "prune_count",
        "final_runtime_parameter_count",
    }:
        return int(round(delta))
    return float(delta)


def build_dyn_ablation_matrix_table(
    rows: Sequence[DynamicsBenchmarkRow | Mapping[str, Any]],
    *,
    include_prune_pilot: bool = True,
) -> dict[str, Any]:
    """Build table-ready paired controller ablation rows.

    Deltas are reported as disabled-minus-full within each
    ``ablation_group_id``.  Skipped/failed rows are preserved with null deltas so
    table generation cannot silently drop non-runnable variants.
    """

    row_dicts = [row for row in _row_dicts(rows) if _ablation_variant(row) is not None]
    full_by_group: dict[str, dict[str, Any]] = {}
    for row in row_dicts:
        if _is_full_ablation(row):
            full_by_group[_ablation_group_id(row)] = row

    table_rows: list[dict[str, Any]] = []
    for row in sorted(
        row_dicts,
        key=lambda item: (
            _ablation_group_id(item),
            0 if _is_full_ablation(item) else 1,
            str(_ablation_variant(item)),
            str(item.get("algorithm_id", "")),
        ),
    ):
        group_id = _ablation_group_id(row)
        full = full_by_group.get(group_id)
        variant = _ablation_variant(row) or "unknown"
        prov = _provenance(row)
        metrics = _metrics(row)
        full_status = None if full is None else str(full.get("status", "unknown"))
        output_row: dict[str, Any] = {
            "ablation_group_id": str(group_id),
            "table_class": str(row.get("table_class", "unclassified")),
            "family": str(row.get("family", "")),
            "case_id": str(row.get("case_id", "")),
            "variant_id": str(variant),
            "disabled_feature": prov.get("disabled_feature", metrics.get("disabled_feature")),
            "algorithm_id": str(row.get("algorithm_id", "")),
            "method_label": str(row.get("method_label", "")),
            "status": str(row.get("status", "unknown")),
            "reason": str(row.get("reason", "")),
            "paired_full_status": full_status,
            "paired_with_full": bool(full is not None),
            "strict_decision_contract_passed": prov.get(
                "strict_decision_contract_passed",
                metrics.get("strict_decision_contract_passed"),
            ),
        }
        for key in _ABLATION_DELTA_KEYS:
            output_row[key] = _numeric_cell(row, key)
            output_row[f"delta_{key}_disabled_minus_full"] = (
                0 if _is_full_ablation(row) and _numeric_cell(row, key) is not None else _paired_delta(row, full, key)
            )
        output_row.update(_tuning_summary(str(row.get("table_class", "unclassified")), [row]))
        table_rows.append(output_row)

    completed_rows = [row for row in table_rows if str(row.get("status")) == "completed"]
    full_completed = [
        row for row in completed_rows if str(row.get("variant_id")) == FULL_CONTROLLER_ABLATION_VARIANT
    ]
    append_cases = sorted(
        {
            str(row.get("case_id", ""))
            for row in full_completed
            if (_int_or_none(row.get("append_count")) or 0) > 0
        }
    )
    prune_cases = sorted(
        {
            str(row.get("case_id", ""))
            for row in full_completed
            if (_int_or_none(row.get("prune_count")) or 0) > 0
        }
    )
    if append_cases and prune_cases:
        bidirectional_status = "sufficient"
    elif append_cases:
        bidirectional_status = "missing_prune_evidence"
    elif prune_cases:
        bidirectional_status = "missing_append_evidence"
    else:
        bidirectional_status = "missing_append_and_prune_evidence"
    append_prune_evidence = {
        "full_controller_completed_case_count": int(len(full_completed)),
        "cases_with_append": append_cases,
        "cases_with_prune": prune_cases,
        "bidirectional_evidence_status": bidirectional_status,
        "policy": "append_count/prune_count counted only on completed full-controller rows",
    }
    prune_pilot = table_iv_prune_pilot_contract() if include_prune_pilot else None
    return json_safe(
        {
            "schema": DYNAMICS_TABLE_SUMMARY_SCHEMA,
            "table_id": DYN_ABLATION_MATRIX_TABLE_ID,
            "caption_label": DYN_ABLATION_MATRIX_TABLE_ID,
            "source_row_schema": "dynamics_benchmark_row_v1",
            "columns": list(_DYN_ABLATION_COLUMNS),
            "row_count": int(len(table_rows)),
            "source_row_count": int(len(row_dicts)),
            "status_counts": _status_counts(row_dicts),
            "pairing_policy": "disabled_minus_full_by_ablation_group_id",
            "delta_keys": list(_ABLATION_DELTA_KEYS),
            "prune_pilot_label": (None if prune_pilot is None else prune_pilot["status"]),
            "prune_pilot": prune_pilot,
            "append_prune_evidence": append_prune_evidence,
            "aggregation_notes": [
                "Skipped and failed ablation rows are retained with null deltas.",
                "The prune pilot is a single explicit paired pilot and is not class-wide evidence.",
            ],
            "rows": table_rows,
        }
    )


def build_generic_dynamics_table_summaries(
    rows: Sequence[DynamicsBenchmarkRow | Mapping[str, Any]],
    *,
    label: str = "generic_dynamics_tables",
    post_fetch_audit_passed: bool = False,
    family_repair_tranche_pdfs_user_approved: bool = False,
    final_aggregation_user_approved: bool = False,
    table_promotion_user_approved: bool = False,
    missing_evidence_user_accepted: bool = False,
    expected_same_seed_group_count: int = _PAPER_II_EXPECTED_SNAKE_SAME_SEED_GROUP_COUNT,
    expected_family_count: int = _PAPER_II_EXPECTED_FAMILY_COUNT,
    expected_tuning_classes: Sequence[str] = DYNAMICS_COARSE_TUNING_CLASSES,
) -> dict[str, Any]:
    row_dicts = _row_dicts(rows)
    bundle = dynamics_table_bundle_payload(rows=row_dicts, label=label)
    claims = build_dyn_claims_table(row_dicts)
    ablation = build_dyn_ablation_matrix_table(row_dicts)
    final_closeout_gate = paper_ii_table_i_final_closeout_gate(
        row_dicts,
        claims_table=claims,
        post_fetch_audit_passed=post_fetch_audit_passed,
        family_repair_tranche_pdfs_user_approved=family_repair_tranche_pdfs_user_approved,
        final_aggregation_user_approved=final_aggregation_user_approved,
        table_promotion_user_approved=table_promotion_user_approved,
        missing_evidence_user_accepted=missing_evidence_user_accepted,
        expected_same_seed_group_count=expected_same_seed_group_count,
        expected_family_count=expected_family_count,
        expected_tuning_classes=expected_tuning_classes,
    )
    return json_safe(
        {
            "schema": DYNAMICS_TABLE_SUMMARY_SCHEMA,
            "label": str(label),
            "source_bundle_schema": DYNAMICS_TABLE_BUNDLE_SCHEMA,
            "source_bundle": bundle,
            "qiskit_parity_summary": _qiskit_parity_summary(row_dicts),
            "paper_ii_table_i_final_closeout_gate": final_closeout_gate,
            "tables": {
                DYN_CLAIMS_TABLE_ID: claims,
                DYN_ABLATION_MATRIX_TABLE_ID: ablation,
            },
        }
    )


def load_dynamics_rows(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load normalized rows from rows/result/summary JSON files."""

    loaded: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_path = str(path)

        def _annotated(item: Mapping[str, Any]) -> dict[str, Any]:
            row = dict(item)
            row.setdefault("source_json_path", source_path)
            provenance = dict(row.get("provenance", {}) or {}) if isinstance(row.get("provenance", {}), Mapping) else {}
            provenance.setdefault("source_json_path", source_path)
            row["provenance"] = provenance
            return row

        if isinstance(payload, list):
            loaded.extend(_annotated(item) for item in payload if isinstance(item, Mapping))
        elif isinstance(payload, Mapping):
            if isinstance(payload.get("rows"), list):
                loaded.extend(
                    _annotated(item) for item in payload["rows"] if isinstance(item, Mapping)
                )
            elif payload.get("schema") == "dynamics_benchmark_row_v1":
                loaded.append(_annotated(payload))
            elif isinstance(payload.get("row"), Mapping):
                loaded.append(_annotated(payload["row"]))
    return loaded


def write_generic_dynamics_table_summaries(
    *,
    rows: Sequence[DynamicsBenchmarkRow | Mapping[str, Any]],
    output_dir: str | Path,
    label: str = "generic_dynamics_tables",
    post_fetch_audit_passed: bool = False,
    family_repair_tranche_pdfs_user_approved: bool = False,
    final_aggregation_user_approved: bool = False,
    table_promotion_user_approved: bool = False,
    missing_evidence_user_accepted: bool = False,
    expected_same_seed_group_count: int = _PAPER_II_EXPECTED_SNAKE_SAME_SEED_GROUP_COUNT,
    expected_family_count: int = _PAPER_II_EXPECTED_FAMILY_COUNT,
    expected_tuning_classes: Sequence[str] = DYNAMICS_COARSE_TUNING_CLASSES,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    payload = build_generic_dynamics_table_summaries(
        rows,
        label=label,
        post_fetch_audit_passed=post_fetch_audit_passed,
        family_repair_tranche_pdfs_user_approved=family_repair_tranche_pdfs_user_approved,
        final_aggregation_user_approved=final_aggregation_user_approved,
        table_promotion_user_approved=table_promotion_user_approved,
        missing_evidence_user_accepted=missing_evidence_user_accepted,
        expected_same_seed_group_count=expected_same_seed_group_count,
        expected_family_count=expected_family_count,
        expected_tuning_classes=expected_tuning_classes,
    )
    claims_path = root / "tab_dyn_claims.json"
    ablation_path = root / "tab_dyn_ablation_matrix.json"
    summary_path = root / "tables_summary.json"
    _write_json(claims_path, payload["tables"][DYN_CLAIMS_TABLE_ID])
    _write_json(ablation_path, payload["tables"][DYN_ABLATION_MATRIX_TABLE_ID])
    payload["paths"] = {
        "tab_dyn_claims_json": str(claims_path),
        "tab_dyn_ablation_matrix_json": str(ablation_path),
        "tables_summary_json": str(summary_path),
    }
    _write_json(summary_path, payload)
    return payload


__all__ = [
    "DYN_ABLATION_MATRIX_TABLE_ID",
    "DYN_CLAIMS_TABLE_ID",
    "DYNAMICS_TABLE_SUMMARY_SCHEMA",
    "FULL_CONTROLLER_ABLATION_VARIANT",
    "FULL_CONTROLLER_ALGORITHM_ID",
    "PAPER_II_TABLE_I_FINAL_CLOSEOUT_GATE_SCHEMA",
    "build_dyn_ablation_matrix_table",
    "build_dyn_claims_table",
    "build_generic_dynamics_table_summaries",
    "paper_ii_table_i_final_closeout_gate",
    "load_dynamics_rows",
    "write_generic_dynamics_table_summaries",
]
